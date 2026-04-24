#!/usr/bin/env python3
"""Build a hash token map via alpha-weighted buckets with gap filling."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "token_maps"

DEFAULT_BUCKET_COUNT = 32
DEFAULT_ALPHA = 0.5
DEFAULT_TOP_K = 200
DEFAULT_TOTAL_CAPACITY = 75968
DEFAULT_MAX_COPIES = 3
DEFAULT_MULTI_WEIGHT_DECAY = 0.8
UNMAPPED_GROUP_ID = -999
VIP_GROUP_ID = -1


def _format_label(value: float) -> str:
    """Format numeric values for filenames with safe characters."""
    return f"{value:.3g}".replace(".", "_").replace("-", "m")


def _metadata_path(path: Path | None) -> str:
    """Avoid embedding machine-specific absolute paths in generated artifacts."""
    if path is None:
        return ""
    return path.name if path.is_absolute() else str(path)


def _compute_top_tokens(freq_map: Dict[int, float], top_k: int) -> List[int]:
    top_tokens = sorted(
        freq_map.items(), key=lambda pair: (-pair[1], pair[0])
    )[:top_k]
    return [token_id for token_id, _ in top_tokens]


def load_freq_map(freq_path: Path) -> Dict[int, float]:
    """Load frequency statistics from JSON or NPY payloads."""
    freq_path = freq_path.expanduser()
    if not freq_path.exists():
        raise FileNotFoundError(f"{freq_path} does not exist")

    if freq_path.suffix.lower() == ".json":
        with freq_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        freq_map: Dict[int, float] = {}
        for token_str, payload in raw.items():
            token_id = int(token_str)
            if isinstance(payload, dict):
                freq_value = payload.get("freq")
                if freq_value is None:
                    raise ValueError(f"missing 'freq' for token {token_id}")
                freq_map[token_id] = float(freq_value)
            else:
                freq_map[token_id] = float(payload)
        return freq_map

    if freq_path.suffix.lower() == ".npy":
        arr = np.load(freq_path)
        if arr.ndim != 1:
            raise ValueError("frequency array must be 1-dimensional")
        return {int(idx): float(val) for idx, val in enumerate(arr)}

    raise ValueError(f"unsupported frequency file suffix {freq_path.suffix}")


def load_top_tokens(
    freq_map: Dict[int, float], top_k: int, top_token_path: Path | None = None
) -> Sequence[int]:
    """Load an optional precomputed top-k token list or derive one from frequencies."""
    if top_token_path is not None:
        top_token_path = top_token_path.expanduser()

    if top_token_path is not None and top_token_path.exists():
        with np.load(top_token_path) as data:
            names = data.files
            if not names:
                raise ValueError("top token archive contains no arrays")
            arr = data[names[0]]
        tokens = [int(v) for v in np.asarray(arr).ravel().tolist()]
        logging.info("loaded %d top tokens from %s", len(tokens), top_token_path)
        return tokens[:top_k]

    if top_token_path is not None:
        logging.warning(
            "top token file %s missing, computing top-%d tokens from freq map",
            top_token_path,
            top_k,
        )

    token_ids = _compute_top_tokens(freq_map, top_k)
    logging.info("computed top-%d tokens from frequency map", len(token_ids))

    if top_token_path is not None:
        top_token_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(top_token_path, indices=np.array(token_ids, dtype=np.int64))
        logging.info(
            "saved computed top tokens (%d entries) to %s",
            len(token_ids),
            top_token_path,
        )
    return token_ids


def build_lookup_table(
    freq_map: Dict[int, float],
    top_tokens: Sequence[int],
    alpha: float,
    bucket_count: int,
    vip_top_k: int,
    total_capacity: int,
    max_copies: int,
    multi_weight_decay: float,
):
    real_vip_tokens = top_tokens[:vip_top_k]
    vip_set = set(real_vip_tokens)

    max_token_id = max(freq_map.keys(), default=-1)
    if real_vip_tokens:
        max_token_id = max(max_token_id, max(real_vip_tokens))
    vocab_size = max_token_id + 1

    token_to_group_id = np.full(vocab_size, UNMAPPED_GROUP_ID, dtype=np.int32)
    token_intra_rank = np.full(vocab_size, -1, dtype=np.int32)

    for rank, token_id in enumerate(real_vip_tokens):
        token_to_group_id[token_id] = VIP_GROUP_ID
        token_intra_rank[token_id] = rank

    eligible_tokens = [
        (tid, freq) for tid, freq in freq_map.items()
        if tid not in vip_set and freq >= 0
    ]

    eligible_tokens.sort(key=lambda x: (-x[1], x[0]))
    total_freq = sum(freq for _, freq in eligible_tokens) if eligible_tokens else 1.0

    total_weight = sum((freq / total_freq) ** alpha for _, freq in eligible_tokens)
    bucket_step = total_weight / bucket_count if bucket_count > 0 else 0

    bucket_counts = np.zeros(bucket_count, dtype=np.int32)
    bucket_members = [[] for _ in range(bucket_count)]

    current_bucket = 0
    cumulative_weight = 0.0

    for token_id, freq in eligible_tokens:
        weight = (freq / total_freq) ** alpha

        while (
            current_bucket < bucket_count - 1
            and cumulative_weight >= (current_bucket + 1) * bucket_step
        ):
            current_bucket += 1

        token_to_group_id[token_id] = current_bucket
        token_intra_rank[token_id] = bucket_counts[current_bucket]

        bucket_members[current_bucket].append(token_id)
        bucket_counts[current_bucket] += 1
        cumulative_weight += weight

    logging.info(
        "Bucketing complete. %d tokens distributed across %d buckets.",
        len(eligible_tokens),
        bucket_count,
    )

    vip_reserved_total = vip_top_k * (1 + max_copies)
    remaining_for_buckets = total_capacity - vip_reserved_total

    if remaining_for_buckets < bucket_count:
        raise ValueError(
            f"Total capacity {total_capacity} too small. VIP needs {vip_reserved_total}, "
            f"leaving {remaining_for_buckets} for {bucket_count} buckets."
        )

    bucket_physical_size = remaining_for_buckets // bucket_count

    logging.info(
        f"Layout: VIP Reserved={vip_reserved_total} (Base {vip_top_k} + Copies {vip_top_k * max_copies}), "
        f"Buckets={bucket_count}x{bucket_physical_size}. "
        f"Total Used={vip_reserved_total + bucket_count * bucket_physical_size}/{total_capacity}"
    )

    alias_records: List[Tuple[int, int, int, float]] = []

    if max_copies > 0:
        for rank, token_id in enumerate(real_vip_tokens):
            for c_idx in range(max_copies):
                abs_rank = vip_top_k + rank * max_copies + c_idx
                weight = multi_weight_decay ** (c_idx + 1)
                alias_records.append((token_id, VIP_GROUP_ID, abs_rank, weight))

    for b_idx in range(bucket_count):
        current_count = bucket_counts[b_idx]
        limit = bucket_physical_size

        if current_count >= limit:
            continue

        slots_available = limit - current_count
        members = bucket_members[b_idx]
        if not members:
            continue

        token_copy_counts = {t: 0 for t in members}
        member_count = len(members)

        cursor = current_count
        iter_idx = 0

        while slots_available > 0:
            candidate = members[iter_idx % member_count]

            if token_copy_counts[candidate] < max_copies:
                weight = multi_weight_decay ** (token_copy_counts[candidate] + 1)
                alias_records.append((candidate, b_idx, cursor, weight))

                cursor += 1
                token_copy_counts[candidate] += 1
                slots_available -= 1
                bucket_counts[b_idx] += 1

            iter_idx += 1

            if iter_idx > member_count * (max_copies + 2):
                break

    alias_offsets = None
    alias_group_ids = None
    alias_intra_ranks = None
    alias_weights = None

    if alias_records:
        alias_records.sort(key=lambda x: x[0])

        alias_offsets = np.zeros(vocab_size + 1, dtype=np.int64)
        g_list, r_list, w_list = [], [], []

        curr_ptr = 0
        rec_idx = 0
        num_recs = len(alias_records)

        for t_id in range(vocab_size):
            alias_offsets[t_id] = curr_ptr
            while rec_idx < num_recs and alias_records[rec_idx][0] == t_id:
                _, g, r, w = alias_records[rec_idx]
                g_list.append(g)
                r_list.append(r)
                w_list.append(w)
                curr_ptr += 1
                rec_idx += 1
        alias_offsets[vocab_size] = curr_ptr
        
        alias_group_ids = np.array(g_list, dtype=np.int64)
        alias_intra_ranks = np.array(r_list, dtype=np.int64)
        alias_weights = np.array(w_list, dtype=np.float32)

    return (
        token_to_group_id,
        token_intra_rank,
        bucket_counts,
        alias_offsets,
        alias_group_ids,
        alias_intra_ranks,
        alias_weights,
    )


def write_lookup_table(
    output_dir: Path,
    alpha: float,
    bucket_count: int,
    total_capacity: int,
    max_copies: int,
    multi_weight_decay: float,
    top_k: int,
    freq_path: Path,
    top_token_path: Path | None,
    token_to_group_id: np.ndarray,
    token_intra_rank: np.ndarray,
    bucket_token_counts: np.ndarray,
    alias_offsets: np.ndarray | None,
    alias_group_ids: np.ndarray | None,
    alias_intra_ranks: np.ndarray | None,
    alias_weights: np.ndarray | None,
) -> Path:
    """Persist the lookup arrays into a descriptive NPZ archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_label = _format_label(alpha)
    wd_label = _format_label(multi_weight_decay)

    filename = (
        "injection_token_map_"
        f"alpha{alpha_label}_M{bucket_count}"
        f"_Cap{total_capacity}_mc{max_copies}_mwd{wd_label}.npz"
    )

    target_path = output_dir / filename
    payload = {
        "token_to_group_id": token_to_group_id,
        "token_intra_rank": token_intra_rank,
        "bucket_token_counts": bucket_token_counts,
        "format_version": np.array(1, dtype=np.int32),
        "vocab_size": np.array(token_to_group_id.shape[0], dtype=np.int64),
        "total_capacity": np.array(total_capacity, dtype=np.int64),
        "max_copies": np.array(max_copies, dtype=np.int64),
        "bucket_count": np.array(bucket_count, dtype=np.int64),
        "top_k": np.array(top_k, dtype=np.int64),
        "alpha": np.array(alpha, dtype=np.float32),
        "multi_weight_decay": np.array(multi_weight_decay, dtype=np.float32),
        "freq_path": np.array(_metadata_path(freq_path)),
        "top_token_path": np.array(_metadata_path(top_token_path)),
    }
    if alias_offsets is not None:
        payload["alias_offsets"] = alias_offsets
        payload["alias_group_ids"] = alias_group_ids
        payload["alias_intra_ranks"] = alias_intra_ranks
        payload["alias_weights"] = alias_weights

    np.savez(target_path, **payload)
    logging.info("saved lookup table to %s", target_path)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a hash token map with reserved VIP rows and bucket gap filling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--freq-path",
        type=Path,
        required=True,
        help="Token frequency JSON/NPY file",
    )

    parser.add_argument(
        "--top-token-path",
        type=Path,
        default=None,
        help="Optional path to a precomputed top-k token NPZ",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the injection lookup table",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Power exponent (p^alpha) used for bucketing",
    )
    parser.add_argument(
        "--bucket-count",
        type=int,
        default=DEFAULT_BUCKET_COUNT,
        help="Number of buckets used for the remaining tokens",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top tokens to treat as VIP",
    )
    parser.add_argument(
        "--total-capacity",
        type=int,
        default=DEFAULT_TOTAL_CAPACITY,
        help="Total physical size of embedding matrix",
    )
    parser.add_argument(
        "--max-copies",
        type=int,
        default=DEFAULT_MAX_COPIES,
        help="Max copies per token (applies to VIP and Gap Filling)",
    )
    parser.add_argument(
        "--multi-weight-decay",
        type=float,
        default=DEFAULT_MULTI_WEIGHT_DECAY,
        help="Geometric decay factor for alias weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.alpha <= 0:
        raise ValueError("alpha must be positive")
    if args.bucket_count <= 0:
        raise ValueError("bucket count must be positive")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.total_capacity <= 0:
        raise ValueError("total capacity must be positive")
    if args.max_copies < 0:
        raise ValueError("max copies must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    freq_map = load_freq_map(args.freq_path)

    top_tokens = load_top_tokens(freq_map, args.top_k, args.top_token_path)

    (
        token_to_group_id,
        token_intra_rank,
        bucket_token_counts,
        alias_offsets,
        alias_group_ids,
        alias_intra_ranks,
        alias_weights,
    ) = build_lookup_table(
        freq_map,
        top_tokens,
        args.alpha,
        args.bucket_count,
        args.top_k,
        args.total_capacity,
        args.max_copies,
        args.multi_weight_decay,
    )

    write_lookup_table(
        args.output_dir,
        args.alpha,
        args.bucket_count,
        args.total_capacity,
        args.max_copies,
        args.multi_weight_decay,
        args.top_k,
        args.freq_path,
        args.top_token_path,
        token_to_group_id,
        token_intra_rank,
        bucket_token_counts,
        alias_offsets,
        alias_group_ids,
        alias_intra_ranks,
        alias_weights,
    )


if __name__ == "__main__":
    main()
