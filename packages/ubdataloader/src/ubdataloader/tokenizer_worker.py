import os
import time
import torch
import multiprocessing
from typing import List, Optional, Tuple
from transformers import AutoTokenizer

from .text_dataset import TextDataset, TextChunkState, TextTokenChunk, TextTokenItem


def bestfit_pack_tokens(
    token_list: List[List[int]],
    each_sample_seq_len: int,
    tokenizer: AutoTokenizer,
) -> Tuple[List[int], List[int]]:
    """
    Pack the tokens into a single tensor, the tokens are packed into a single tensor
    Use the best-fit algorithm to pack the tokens into a single tensor
    The algorithm is as follows:
    1. Sort the tokens by length, in descending order
    2. For each token, find the best bin to pack it into
    3. If the token is too long for any bin, create a new bin
    4. If the token is too short for any bin, create a new bin
    5. Return the packed tokens and labels
    """
    sorted_sequences = sorted(token_list, key=len, reverse=True)

    bins: List[List[List[int]]] = []
    bin_remaining_space: List[int] = []

    for seq in sorted_sequences:
        seq_len = len(seq)

        assert seq_len <= each_sample_seq_len and seq_len > 0, (
            f"the sequence length({seq_len}) is greater than the each sample sequence length: ({each_sample_seq_len})"
        )

        best_bin_idx = -1
        min_remaining = float("inf")

        for i, remaining in enumerate(bin_remaining_space):
            if remaining >= seq_len and remaining < min_remaining:
                best_bin_idx = i
                min_remaining = remaining

        if best_bin_idx == -1:
            bins.append([seq])
            bin_remaining_space.append(each_sample_seq_len - seq_len)
        else:
            bins[best_bin_idx].append(seq)
            bin_remaining_space[best_bin_idx] -= seq_len

    ret_tokens = []
    ret_labels = []

    pad_token_id = tokenizer.pad_token_id
    ignore_label_id = -100

    for i, bin_sequences in enumerate(bins):
        current_sequence = [token for seq in bin_sequences for token in seq]
        ret_labels.extend(current_sequence[1:])
        ret_labels.append(ignore_label_id)
        ret_tokens.extend(current_sequence)

        padding_len = each_sample_seq_len - len(current_sequence)
        ret_tokens.extend([pad_token_id] * padding_len)
        ret_labels.extend([ignore_label_id] * padding_len)

    return ret_tokens, ret_labels


def native_pack_tokens(
    token_list: List[List[int]],
    each_sample_seq_len: int,
    tokenizer: AutoTokenizer,
) -> Tuple[List[int], List[int]]:
    ret_tokens = []
    ret_labels = []

    remaining_space = each_sample_seq_len
    pad_token_id = tokenizer.pad_token_id
    ignore_label_id = -100

    for token in token_list:
        token_len = len(token)
        assert token_len <= each_sample_seq_len and token_len > 0, (
            f"the token length({token_len}) is greater than the each sample sequence length: ({each_sample_seq_len})"
        )

        if token_len > remaining_space:
            ret_tokens.extend([pad_token_id] * remaining_space)
            ret_labels.extend([ignore_label_id] * remaining_space)
            remaining_space = each_sample_seq_len

        ret_tokens.extend(token)
        ret_labels.extend(token[1:])
        ret_labels.append(ignore_label_id)
        remaining_space -= token_len

    if remaining_space > 0:
        ret_tokens.extend([pad_token_id] * remaining_space)
        ret_labels.extend([ignore_label_id] * remaining_space)

    return ret_tokens, ret_labels


def create_tokens_buffer(
    token_list: List[List[int]],
    each_sample_seq_len: int,
    tokenizer: AutoTokenizer,
    pack_method: str,
) -> Tuple[List[int], List[int]]:
    if "native" in pack_method:
        return native_pack_tokens(token_list, each_sample_seq_len, tokenizer)
    elif "bestfit" in pack_method:
        return bestfit_pack_tokens(token_list, each_sample_seq_len, tokenizer)
    else:
        raise NotImplementedError(f"Invalid pack method: {pack_method}")


def scatter_item(
    is_master_node: bool, text_token_chunk: TextTokenChunk, world_size: int
) -> List[TextTokenItem]:
    # scatter the text chunk to the world size
    # each worker will tokenize the text chunk
    local_rank_item_list = [None]

    if is_master_node:
        all_item_list_in_master = text_token_chunk.text_token_payloads
        # split the text chunk to the world size
        assert len(all_item_list_in_master) % world_size == 0, (
            "the text chunk size is not divisible by the world size"
        )

        rank_chunk_size = len(all_item_list_in_master) // world_size

        item_list_for_scatter = []
        for i in range(world_size):
            rank_start_idx = i * rank_chunk_size
            rank_end_idx = rank_start_idx + rank_chunk_size
            item_list_for_scatter.append(
                all_item_list_in_master[rank_start_idx:rank_end_idx]
            )
        torch.distributed.scatter_object_list(
            local_rank_item_list,
            item_list_for_scatter,
            src=0,
        )

    else:
        torch.distributed.scatter_object_list(
            local_rank_item_list,
            [None] * world_size,
            src=0,
        )

    assert isinstance(local_rank_item_list, list), "text_chunk_list is not a list"

    return local_rank_item_list[0]


def gather_item(
    is_master_node: bool, part_item_list: Optional[List[TextTokenItem]], world_size: int
) -> List[TextTokenItem]:
    # allgather the part token chunk
    all_part_item_list = [None for _ in range(world_size)]

    torch.distributed.gather_object(
        part_item_list if is_master_node else [],
        all_part_item_list if is_master_node else [],
        dst=0,
    )

    if not is_master_node:
        return None

    # flatten the part token chunk
    ret_item_list: List[TextTokenItem] = []
    for part_item_list in all_part_item_list:
        for item in part_item_list:
            ret_item_list.append(item)

    return ret_item_list


def text_token_worker(
    local_path: List[Optional[str]],
    remote_path: List[Optional[str]],
    proportion: List[float],
    each_sample_seq_len: int,
    text_chunk_size: int,
    tokenizer_path: str,
    worker_rank: int,
    world_size: int,
    token_chunk_queue: multiprocessing.Queue,
    text_chunk_state: Optional[TextChunkState],
    use_token_column: Optional[str],
    pack_method: str,
):
    # don't use the tokenizers parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # must ensure the torch distributed is not initialized
    assert not torch.distributed.is_initialized(), (
        "torch distributed is not initialized"
    )

    is_master_node = worker_rank == 0
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ.get("UBDATALOADER_MASTER_PORT", 23410))

    # the streaming dataset only use the cpu device, and do not use distributed backend
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES "] = ""

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        add_eos_token=True,
    )

    # only master node will read the text data
    if is_master_node:
        extra_args = {}
        if text_chunk_state is not None:
            extra_args["epoch"] = text_chunk_state.epoch
            extra_args["sample_in_epoch"] = text_chunk_state.sample_in_epoch
            extra_args["streaming_dict"] = text_chunk_state.streaming_dict

        dataset = TextDataset(
            local_path=local_path,
            remote_path=remote_path,
            proportion=proportion,
            world_size=world_size,
            chunk_size=text_chunk_size,
            use_token_column=use_token_column,
            **extra_args,
        )

        dataset_iter = iter(dataset)

    start_time = time.time()
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=worker_rank,
        store=torch.distributed.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=is_master_node,
        ),
    )
    end_time = time.time()
    if is_master_node:
        print(
            "Waiting for the dataset distributed process group barrier done cost %s seconds"
            % (end_time - start_time)
        )

    while True:
        get_chunk_start_time = time.time()
        if is_master_node:
            text_token_chunk: TextTokenChunk = next(dataset_iter)
            chunk_state = text_token_chunk.text_chunk_state
        else:
            text_token_chunk = None
            chunk_state = None
        get_chunk_end_time = time.time()
        get_chunk_time = get_chunk_end_time - get_chunk_start_time

        # the master scatter the text chunk or token chunk to the world size
        scatter_start_time = time.time()
        if use_token_column is None:
            part_item_list: List[TextTokenItem] = scatter_item(
                is_master_node, text_token_chunk, world_size
            )

            # each rank process the part of the text chunk
            part_text_list = [item.text for item in part_item_list]
            part_token_list = tokenizer(part_text_list)["input_ids"]

            for item, token in zip(part_item_list, part_token_list):
                item.token = token
        else:
            part_item_list = (
                text_token_chunk.text_token_payloads if is_master_node else None
            )
        scatter_end_time = time.time()
        scatter_time = scatter_end_time - scatter_start_time

        # the master gather the part token chunk to the all token chunk
        gather_start_time = time.time()
        if use_token_column is None:
            all_item_list: List[TextTokenItem] = gather_item(
                is_master_node, part_item_list, world_size
            )
        else:
            all_item_list = (
                text_token_chunk.text_token_payloads if is_master_node else None
            )
        gather_end_time = time.time()
        gather_time = gather_end_time - gather_start_time

        # only master have all the token items
        if is_master_node:
            all_token_list = []
            for item in all_item_list:
                token = item.token
                # drop the last token of the chunk
                if (
                    "truncate" in pack_method
                    and item.ck_tot > 1
                    and item.ck_idx == item.ck_tot - 1
                ):
                    continue

                all_token_list.append(token)

            ret_tokens, ret_labels = create_tokens_buffer(
                all_token_list, each_sample_seq_len, tokenizer, pack_method
            )
        else:
            ret_tokens = None
            ret_labels = None

        total_time = get_chunk_time + scatter_time + gather_time
        if total_time > 2 and is_master_node:
            print(
                "!!!Notice: master get the text/token chunk time (get chunk + scatter + gather) is too long,"
                f" cost {total_time:.4f} = {get_chunk_time:.4f} + {scatter_time:.4f} + {gather_time:.4f} seconds"
            )

        token_chunk_queue.put((chunk_state, ret_tokens, ret_labels))
