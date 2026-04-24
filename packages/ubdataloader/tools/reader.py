import os
import torch
import duckdb
import argparse
from transformers import AutoTokenizer

from ubdataloader.loader import TokenStreamDataLoader


def argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mbs", type=int, required=True)
    parser.add_argument("--dp-world-size", type=int, default=1)
    parser.add_argument("--dp-rank", type=int, default=0)
    parser.add_argument("--data-path", nargs="*", required=True)
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--text-chunk-queue-size", type=int, default=8)
    parser.add_argument("--text-chunk-size", type=int, default=4096)
    parser.add_argument("--prefetch-queue-size", type=int, default=8)
    parser.add_argument("--use-token-column", type=str, default=None)
    parser.add_argument(
        "--truncate-last", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--pack", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-tokens-in-sample", type=int, default=8192)
    return parser.parse_args()


def init_data_loader(args, consumed_samples: int):
    print("Initializing data loader for UBDataLoader")

    local_path = []
    remote_path = []
    proportion = []
    for weight, local, remote in zip(
        args.data_path[::3],
        args.data_path[1::3],
        args.data_path[2::3],
    ):
        proportion.append(float(weight))
        local_path.append(local if local != "None" else None)
        remote_path.append(remote if remote != "None" else None)

    torch.distributed.init_process_group(
        backend="gloo",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    # NOTE: ONLY SUPPORT THE DP PARALLEL MODE
    return TokenStreamDataLoader(
        local_path=local_path,
        remote_path=remote_path,
        proportion=proportion,
        seq_len=args.seq_len,
        ckpt_path=args.ckpt_path,
        consumed_samples=consumed_samples,
        micro_batch_size=args.mbs,
        data_parallel_rank=args.dp_rank,
        data_parallel_size=args.dp_world_size,
        tokenizer_path=args.tokenizer_model,
        need_dataset_on_this_rank=True,
        text_chunk_queue_size=args.text_chunk_queue_size,
        text_chunk_size=args.text_chunk_size,
        prefetch_queue_size=args.prefetch_queue_size,
        use_token_column=args.use_token_column,
        truncate_last=args.truncate_last,
        pack=args.pack,
        max_tokens_in_sample=args.max_tokens_in_sample,
    )


if __name__ == "__main__":
    args = argparse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, trust_remote_code=True
    )
    step = args.step
    samples = (step - 1) * args.mbs * args.dp_world_size
    loader = init_data_loader(args, samples)
    loader_it = iter(loader)
    pad_token_id = tokenizer.pad_token_id
    while True:
        data = next(loader_it)
        batch_size = data["tokens"].shape[0]

        tokens_checksum = data["tokens"].sum().item() % 1000000007
        labels_checksum = data["labels"].sum().item() % 1000000007

        tokens_checksum, labels_checksum = loader.get_checksum(samples)

        assert tokens_checksum == tokens_checksum
        assert labels_checksum == labels_checksum

        tokens_num_pad_tokens = (data["tokens"] == pad_token_id).sum().item()
        tokens_num_eos_tokens = (data["tokens"] == tokenizer.eos_token_id).sum().item()
        labels_num_pad_tokens = (data["labels"] == pad_token_id).sum().item()
        labels_num_eos_tokens = (data["labels"] == tokenizer.eos_token_id).sum().item()
        with open(f"token_text_{step}.txt", "a+") as f:
            f.write(
                f"TOKEN TOTAL PAD TOKENS: {tokens_num_pad_tokens}, RATIO: {tokens_num_pad_tokens / int(data['tokens'].numel()) * 100:.2f}%\n"
                f"TOKEN TOTAL EOS TOKENS: {tokens_num_eos_tokens}, RATIO: {tokens_num_eos_tokens / int(data['tokens'].numel()) * 100:.2f}%\n"
                f"LABEL TOTAL PAD TOKENS: {labels_num_pad_tokens}, RATIO: {labels_num_pad_tokens / int(data['labels'].numel()) * 100:.2f}%\n"
                f"LABEL TOTAL EOS TOKENS: {labels_num_eos_tokens}, RATIO: {labels_num_eos_tokens / int(data['labels'].numel()) * 100:.2f}%\n"
            )
        for i in range(batch_size):
            token_text = tokenizer.decode(data["tokens"][i])
            with open(f"token_text_{step}.txt", "a+") as f:
                f.write(
                    ">" * 50
                    + " ID: "
                    + str(i)
                    + " "
                    + "<" * 50
                    + "\n"
                    + token_text
                    + "\n"
                    + ">" * 50
                    + " END "
                    + "<" * 50
                    + "\n"
                    + "\n"
                )
        step += 1
        samples += args.mbs * args.dp_world_size
        is_continue = input("Continue? (y/n)")
        if is_continue != "y":
            exit()
