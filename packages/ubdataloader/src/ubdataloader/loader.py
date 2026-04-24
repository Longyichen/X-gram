import os
import time
import torch
import threading
from queue import Queue
from typing import Optional, List

from .text_token_cache import TextTokenChunkCache


class TokenStreamDataLoader:
    text_it: int
    token_it: int

    def __init__(
        self,
        local_path: List[Optional[str]],
        remote_path: List[Optional[str]],
        proportion: List[float],
        seq_len: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        tokenizer_path: str,
        ckpt_path: str,
        text_chunk_queue_size: int,
        text_chunk_size: int,
        prefetch_queue_size: int,
        use_token_column: Optional[str],
        pack_method: str = "native:truncate",
    ):
        """
        NOTE: each iteration, we will load the seq_len * micro_batch_size * data_parallel_size tokens
        and each data parallel rank will load the seq_len * micro_batch_size tokens
        1. use the consumed_samples (due it's not consistent with the iteration) to resume the training from the last checkpoint
        2. pack_method in include:
            * native: pack the tokens and labels into a single tensor
            * truncate: drop the last sample in the document
        """
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.consumed_samples = consumed_samples

        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size

        self.is_master = int(os.environ["RANK"]) == 0
        self._get_global_rank_to_dp_rank_map()

        self.text_token_chunk_cache = TextTokenChunkCache(
            local_path=local_path,
            remote_path=remote_path,
            proportion=proportion,
            seq_len_each_sample=seq_len,
            consumed_samples=consumed_samples,
            global_batch_size=micro_batch_size * data_parallel_size,
            tokenizer_path=tokenizer_path,
            ckpt_path=ckpt_path,
            text_chunk_queue_size=text_chunk_queue_size,
            text_chunk_size=text_chunk_size,
            use_token_column=use_token_column,
            pack_method=pack_method,
        )

        self.cache_queue = Queue(maxsize=prefetch_queue_size)
        self.prefetch_thread = threading.Thread(target=self.fetch_next_item_thread)
        self.prefetch_thread.start()

    def _get_global_rank_to_dp_rank_map(self):
        # NOTE: To ensure compatibility with different frameworks
        # we need to map the global rank to the data parallel (dp) rank.
        assert torch.distributed.is_initialized()
        self.world_size = torch.distributed.get_world_size()
        self.dataloader_group = torch.distributed.new_group(
            ranks=list(range(self.world_size)),
            backend="gloo",
        )
        rank_list = [
            torch.zeros(1, dtype=torch.int32, device=torch.device("cpu"))
            for _ in range(self.world_size)
        ]
        rank_tensor = torch.tensor(
            [self.data_parallel_rank], dtype=torch.int32, device=torch.device("cpu")
        )
        torch.distributed.all_gather(
            rank_list,
            rank_tensor,
            group=self.dataloader_group,
        )
        self.global_rank_to_dp_rank_map = [int(rank.item()) for rank in rank_list]
        if self.is_master:
            print(
                f"!!!NOTE: the global rank to dp rank map is: {self.global_rank_to_dp_rank_map}"
            )

    def _scatter_data_to_different_data_parallel_ranks(
        self, full_tensor: Optional[torch.Tensor]
    ):
        receiver_tensor = torch.zeros(
            self.micro_batch_size,
            self.seq_len,
            dtype=torch.long,
            device=torch.device("cpu"),
        )

        scatter_tensor_list = None
        if self.is_master:
            assert full_tensor is not None
            full_tensor_chunk = full_tensor.chunk(self.data_parallel_size, dim=0)
            scatter_tensor_list = [
                full_tensor_chunk[self.global_rank_to_dp_rank_map[i]]
                for i in range(self.world_size)
            ]

        torch.distributed.scatter(
            receiver_tensor,
            scatter_tensor_list,
            src=0,
            group=self.dataloader_group,
        )

        return receiver_tensor

    def fetch_next_item_thread(self):
        while True:
            data = next(self.text_token_chunk_cache)

            all_tokens = None
            all_labels = None
            if self.is_master:
                # scatter the data to the different data parallel ranks
                all_tokens: torch.Tensor = data["tokens"]
                all_labels: torch.Tensor = data["labels"]
                assert (
                    self.micro_batch_size * self.data_parallel_size
                    == all_tokens.shape[0]
                )
                assert all_tokens.shape[1] == self.seq_len
                assert all_labels.shape == all_tokens.shape

            tokens = self._scatter_data_to_different_data_parallel_ranks(all_tokens)
            labels = self._scatter_data_to_different_data_parallel_ranks(all_labels)

            item = {
                "tokens": tokens,
                "labels": labels,
            }

            self.cache_queue.put(item)

    def get_checksum(self, consumed_samples: int):
        return self.text_token_chunk_cache.get_checksum(
            consumed_samples=consumed_samples
        )

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        item = self.cache_queue.get(block=True)
        end_time = time.time()

        fetch_item_time = end_time - start_time

        if fetch_item_time > 0.1 and self.data_parallel_rank == 0:
            print(
                f"!!!Warning: fetch the item time is too long, cost {fetch_item_time} seconds."
            )

        # NOTE: the comsumed_samples is the global consumed samples
        self.consumed_samples += self.micro_batch_size * self.data_parallel_size
        return item
