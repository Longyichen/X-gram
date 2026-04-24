# Copyright 2022-2024 MosaicML Streaming authors
# SPDX-License-Identifier: Apache-2.0

"""Shuffling algorithm that naively shuffles all-to-all.

Useful for single-node training on small data, where you want the most random shuffle possible.

Statistically, this algorithm will result in all nodes downloading all shards, with those downloads
all happening at the start of the epoch, bringing training to a crawl.
"""

import os
import hashlib
import numpy as np
from numpy.typing import NDArray


def _calculate_hash(
    shard_sizes: NDArray[np.int64],
    num_canonical_nodes: int,
    seed: int,
    epoch: int,
    block_size: int,
) -> str:
    """Calculate hash based on input parameters.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        num_canonical_nodes (int): Number of canonical nodes.
        seed (int): Base random seed.
        epoch (int): Current epoch.
        block_size (int): Unit of shuffle.

    Returns:
        str: Hash string based on input parameters.
    """
    # Create a string representation of all parameters
    param_str = (
        f"{shard_sizes.tobytes()}_{num_canonical_nodes}_{seed}_{epoch}_{block_size}"
    )

    # Calculate SHA256 hash
    hash_obj = hashlib.sha256(param_str.encode("utf-8"))
    return hash_obj.hexdigest()


def get_shuffle_naive(
    shard_sizes: NDArray[np.int64],
    num_canonical_nodes: int,
    seed: int,
    epoch: int,
    block_size: int = 1 << 18,
) -> NDArray[np.int64]:
    """Get the shuffled global ordering of samples for an epoch.

    The assignment of shards to nodes is fixed across epochs, but each grouping
    of shards is processed concurrently in a different order by each node's
    workers each epoch.

    Args:
        shard_sizes (NDArray[np.int64]): Number of samples contained in each shard, in order.
        num_canonical_nodes (int): Number of canonical nodes.
        seed (int): Base random seed, which is held constant over an entire training run.
        epoch (int): Current epoch, which is added to the seed to get a different deterministic
            shuffle each epoch.
        block_size (int): Unit of shuffle (ignored because we shuffle all samples together).
            Defaults to ``1 << 18``.

    Returns:
        NDArray[np.int64]: 1:1 mapping of sample ID to shuffled sample ID.
    """
    paramter_hash = _calculate_hash(
        shard_sizes, num_canonical_nodes, seed, epoch, block_size
    )
    cache_path = None

    if os.environ.get("STREAMING_CACHE", None) is not None:
        cache_dir = os.environ.get("STREAMING_CACHE")
        assert cache_dir is not None
        cache_path = os.path.join(cache_dir, f"shuffle_py1e_{paramter_hash}.npy")
        if os.path.exists(cache_path):
            print(f"Load the shuffle_py1e_{paramter_hash}.npy from the cache")
            return np.load(cache_path)

    rng = np.random.default_rng(seed + epoch)
    ids = rng.permutation(sum(shard_sizes))

    if cache_path is not None:
        np.save(cache_path, ids)
        print(f"Save the shuffle_naive_{paramter_hash}.npy to the cache")

    return ids
