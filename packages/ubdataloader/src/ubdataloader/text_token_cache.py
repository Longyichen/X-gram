import os
import time
import torch
import multiprocessing
from typing import List, Optional

from .ckpt_db import CkptDB, StateDict, DatasetConfig
from .text_dataset import TextChunkState
from .tokenizer_worker import text_token_worker


class TextTokenChunkCache:
    def __init__(
        self,
        local_path: List[Optional[str]],
        remote_path: List[Optional[str]],
        proportion: List[float],
        seq_len_each_sample: int,
        consumed_samples: int,
        global_batch_size: int,
        tokenizer_path: str,
        ckpt_path: str,
        text_chunk_queue_size: int,
        text_chunk_size: int,
        use_token_column: Optional[str],
        pack_method: str,
    ):
        # dataset path configuration
        self.local_path = local_path
        self.remote_path = remote_path
        self.proportion = proportion

        # sequence length configuration
        self.seq_len_each_sample = seq_len_each_sample
        self.consumed_samples = consumed_samples
        self.global_batch_size = global_batch_size

        # tokenizer configuration
        self.tokenizer_path = tokenizer_path

        # ckpt configuration
        self.ckpt_path = ckpt_path

        # cache configuration
        self.text_chunk_queue_size = text_chunk_queue_size
        self.text_chunk_size = text_chunk_size

        # tokenizer and packing configuration
        self.use_token_column = use_token_column
        self.pack_method = pack_method

        # part token chunk queue to recv part token chunk
        # and token chunk queue to gather all token chunk
        mp_context = multiprocessing.get_context("spawn")
        self.token_chunk_queue = mp_context.Queue(maxsize=text_chunk_queue_size)

        self.process_tokens: Optional[List[int]] = None
        self.process_labels: Optional[List[int]] = None
        self.chunk_state = None

        self.is_master = int(os.environ["RANK"]) == 0

        # the state of the cache
        self.token_it = 0
        self.ckpt_db = CkptDB(
            is_master=self.is_master,
            ckpt_path=ckpt_path,
        )

        # use the gloo backend to gather the part token chunk
        # the data is in cpu device
        assert torch.distributed.is_initialized(), (
            "torch distributed is not initialized"
        )
        self.worker_rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.text_token_worker_group = torch.distributed.new_group(
            ranks=list(range(self.world_size)),
            backend="gloo",
        )

        if self.is_master:
            print(
                "text token cache rank(%s): global batch size: %s, consumed samples: %s"
                % (self.worker_rank, global_batch_size, consumed_samples)
            )

        text_chunk_state = self.load_text_chunk_state()

        self.token_worker = mp_context.Process(
            target=text_token_worker,
            args=(
                local_path,
                remote_path,
                proportion,
                seq_len_each_sample,
                text_chunk_size,
                tokenizer_path,
                self.worker_rank,
                self.world_size,
                self.token_chunk_queue,
                text_chunk_state,
                self.use_token_column,
                pack_method,
            ),
        )
        self.token_worker.start()

    def get_checksum(self, consumed_samples: int):
        assert self.is_master

        return self.ckpt_db.load_checksum(consumed_samples=consumed_samples)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_master:
            self.token_chunk_queue.get(block=True)
            return None, None

        ret_token_ids = []
        ret_label_ids = []

        seq_len_each_batch = self.seq_len_each_sample * self.global_batch_size
        left_need_tokens = seq_len_each_batch

        is_saved: bool = False

        fill_list_start_time = time.time()
        while len(ret_token_ids) < seq_len_each_batch:
            if self.process_tokens is None:
                assert self.chunk_state is None
                assert self.process_labels is None

                self.chunk_state, self.process_tokens, self.process_labels = (
                    self.token_chunk_queue.get(block=True)
                )

                assert len(self.process_tokens) % self.seq_len_each_sample == 0
                assert len(self.process_labels) == len(self.process_tokens)
                is_saved = False

            if not is_saved:
                self.save_text_chunk_state()
                is_saved = True

            tokens = self.process_tokens[
                self.token_it : self.token_it + left_need_tokens
            ]
            labels = self.process_labels[
                self.token_it : self.token_it + left_need_tokens
            ]

            tokens_cnt = len(tokens)
            ret_token_ids.extend(tokens)
            ret_label_ids.extend(labels)

            assert tokens_cnt % self.seq_len_each_sample == 0
            assert self.token_it % self.seq_len_each_sample == 0

            self.token_it += tokens_cnt
            left_need_tokens -= tokens_cnt

            if self.token_it >= len(self.process_tokens):
                self.process_tokens = None
                self.process_labels = None
                self.chunk_state = None
                self.token_it = 0
        fill_list_end_time = time.time()
        fill_list_time = fill_list_end_time - fill_list_start_time

        assert len(ret_token_ids) == seq_len_each_batch, (
            f"len(ret_token_ids): {len(ret_token_ids)}, seq_len_each_batch: {seq_len_each_batch}"
        )
        assert len(ret_label_ids) == len(ret_token_ids)

        fill_tensor_start_time = time.time()
        # prepare the mask and others
        text = torch.tensor(
            ret_token_ids, dtype=torch.long, device=torch.device("cpu")
        ).view(self.global_batch_size, -1)
        labels = torch.tensor(
            ret_label_ids, dtype=torch.long, device=torch.device("cpu")
        ).view(self.global_batch_size, -1)

        ret_tokens = text.clone().contiguous()
        ret_labels = labels.clone().contiguous()

        del text, labels, ret_token_ids, ret_label_ids

        fill_tensor_end_time = time.time()
        fill_tensor_time = fill_tensor_end_time - fill_tensor_start_time

        total_time = fill_list_time + fill_tensor_time

        if total_time > 2:
            print(
                f"!!!Notice: fetch the item time is too long, cost {total_time} = {fill_list_time} + {fill_tensor_time} seconds."
            )

        tokens_checksum = ret_tokens.sum().item() % 1000000007
        labels_checksum = ret_labels.sum().item() % 1000000007
        self.ckpt_db.save_checksum(
            consumed_samples=self.consumed_samples,
            tokens_checksum=tokens_checksum,
            labels_checksum=labels_checksum,
        )

        self.consumed_samples += self.global_batch_size

        return {
            "tokens": ret_tokens,
            "labels": ret_labels,
        }

    def save_text_chunk_state(
        self,
    ):
        # only the master process will save the state dict
        if not self.is_master:
            return

        assert self.chunk_state is not None

        self.ckpt_db.save(
            state_dict=StateDict(
                consumed_samples=self.consumed_samples,
                token_it=self.token_it,
                epoch=self.chunk_state.epoch,
                sample_in_epoch=self.chunk_state.sample_in_epoch,
                streaming_dict=self.chunk_state.streaming_dict,
            ),
            dataset_config=DatasetConfig(
                remote_path=self.remote_path,
                proportion=self.proportion,
                seq_len_each_sample=self.seq_len_each_sample,
                global_batch_size=self.global_batch_size,
                text_chunk_size=self.text_chunk_size,
                pack_method=self.pack_method,
            ),
        )

    def load_text_chunk_state(self) -> Optional[TextChunkState]:
        # if the consumed_samples is 0, it means the first iteration
        if self.consumed_samples == 0:
            return None

        dataset_config, state_dict = self.ckpt_db.load(
            consumed_samples=self.consumed_samples
        )

        assert dataset_config.remote_path == self.remote_path
        assert dataset_config.proportion == self.proportion
        assert dataset_config.seq_len_each_sample == self.seq_len_each_sample
        assert dataset_config.global_batch_size == self.global_batch_size
        assert dataset_config.text_chunk_size == self.text_chunk_size
        assert dataset_config.pack_method == self.pack_method

        self.token_it = int(state_dict.token_it)

        print(
            "Resume from token it: %s, epoch: %s, sample in epoch: %s, streaming dict: %s"
            % (
                self.token_it,
                int(state_dict.epoch),
                int(state_dict.sample_in_epoch),
                state_dict.streaming_dict,
            )
        )

        return TextChunkState(
            epoch=int(state_dict.epoch),
            sample_in_epoch=int(state_dict.sample_in_epoch),
            streaming_dict=state_dict.streaming_dict,
        )
