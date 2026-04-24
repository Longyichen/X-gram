import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)


class HashTokenMapInjection(nn.Module):


    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        dtype: torch.dtype,
        init_device: str,
        token_map_path: Path,
        num_buckets: Optional[int],
        top_k_count: Optional[int],
        hash_multipliers: List[int] = [11400714819323198485, 14313749767032793493] 
    ):
        super().__init__()
        if not token_map_path or not token_map_path.exists():
            raise OLMoConfigurationError(
                f"token map '{token_map_path}' does not exist"
            )

        with np.load(token_map_path, allow_pickle=True) as data:
            def _read_required_int_field(key: str) -> int:
                if key not in data:
                    raise OLMoConfigurationError(
                        f"token map missing required field '{key}'"
                    )
                value = np.asarray(data[key]).reshape(()).item()
                return int(value)

            try:
                token_to_group_id = np.asarray(data["token_to_group_id"]).ravel().astype(np.int64)
                token_intra_rank = np.asarray(data["token_intra_rank"]).ravel().astype(np.int64)
                bucket_token_counts = np.asarray(data["bucket_token_counts"]).ravel().astype(np.int64)
                token_map_total_capacity = _read_required_int_field("total_capacity")
                token_map_max_copies = _read_required_int_field("max_copies")
                alias_offsets = (
                    np.asarray(data["alias_offsets"]).ravel().astype(np.int64)
                    if "alias_offsets" in data else None
                )
                alias_group_ids = (
                    np.asarray(data["alias_group_ids"]).ravel().astype(np.int64)
                    if "alias_group_ids" in data else None
                )
                alias_intra_ranks = (
                    np.asarray(data["alias_intra_ranks"]).ravel().astype(np.int64)
                    if "alias_intra_ranks" in data else None
                )
                alias_weights = (
                    np.asarray(data["alias_weights"]).ravel().astype(np.float32)
                    if "alias_weights" in data else None
                )
            except KeyError as exc:
                raise OLMoConfigurationError(
                    f"token map missing key {exc.args[0]}"
                ) from exc

        if token_to_group_id.shape[0] != vocab_size:
            raise OLMoConfigurationError(
                "token map lengths must match vocabulary size"
            )

        derived_num_buckets = bucket_token_counts.shape[0]
        if num_buckets is None:
            num_buckets = derived_num_buckets
        elif num_buckets != derived_num_buckets:
            raise OLMoConfigurationError(
                "num_buckets must match bucket_token_counts length"
            )
            
        vip_count = int(np.count_nonzero(token_to_group_id == -1))
        if top_k_count is None:
            top_k_count = vip_count
        elif top_k_count != vip_count:
            raise OLMoConfigurationError(
                "top_k_count must match number of VIP tokens in token map"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_buckets = num_buckets

        total_capacity = token_map_total_capacity
        if total_capacity <= 0:
            raise OLMoConfigurationError("hash token map total_capacity must be positive")

        max_copies = token_map_max_copies
        if max_copies < 0:
            raise OLMoConfigurationError("max_copies must be non-negative")

        self.total_capacity = int(total_capacity)
        self.max_copies = max_copies

        # Reserved Zone Size
        self.vip_reserved_total = top_k_count * (1 + max_copies)

        converted_mults = []
        for val in hash_multipliers:
            if val >= (1 << 63):
                val -= (1 << 64)
            converted_mults.append(val)
        
        self.hash_multipliers_val = converted_mults
        self.num_heads = len(converted_mults)
        if self.num_heads <= 0:
            raise OLMoConfigurationError("hash_multipliers must be non-empty")
        self.register_buffer("_hash_multipliers_tensor", torch.tensor(self.hash_multipliers_val, dtype=torch.long))
        
        self._token_to_group_id_array = token_to_group_id
        self._token_intra_rank_array = token_intra_rank
        self._alias_offsets_array = alias_offsets
        self._alias_group_ids_array = alias_group_ids
        self._alias_intra_ranks_array = alias_intra_ranks
        self._alias_weights_array = alias_weights
        
        # Keep dense tables as non-persistent buffers so FSDP can move them with
        # to_empty() without writing them into the state dict.
        self.register_buffer(
            "_dense_indices",
            torch.zeros(1, dtype=torch.long, device=init_device),
            persistent=False,
        )
        self.register_buffer(
            "_dense_weights",
            torch.zeros(1, dtype=torch.float32, device=init_device),
            persistent=False,
        )

        self._scalar_weight_embeddings = nn.ModuleList()

        remaining_capacity = total_capacity - self.vip_reserved_total
        if remaining_capacity < num_buckets:
            raise OLMoConfigurationError(
                f"Capacity too small: {total_capacity} < {self.vip_reserved_total} + buckets"
            )

        self.bucket_physical_size = remaining_capacity // num_buckets
        self._bucket_sparse_array = bucket_token_counts <= self.bucket_physical_size
        self._bucket_embedding = nn.Embedding(
            self.total_capacity,
            d_model,
            dtype=dtype,
            device=init_device,
        )

        # Initialize scalar gate embeddings to bias gates toward 1.0 at startup.
        w_emb = nn.Embedding(
            self.total_capacity,
            1,
            _weight=torch.full((self.total_capacity, 1), 4.0),
        )
        self._scalar_weight_embeddings.append(w_emb.to(init_device))

        self._reset_injection_buffers(device=init_device)

    def _reset_injection_buffers(self, *, device: Union[str, torch.device]) -> None:
        device_str = str(device)
        buffer_device: Union[str, torch.device] = device if device_str != "meta" else "cpu"

        token_to_group_id = torch.from_numpy(self._token_to_group_id_array).to(
            dtype=torch.long, device=buffer_device
        )
        token_intra_rank = torch.from_numpy(self._token_intra_rank_array).to(
            dtype=torch.long, device=buffer_device
        )
        bucket_sparse = torch.from_numpy(self._bucket_sparse_array).to(
            dtype=torch.bool, device=buffer_device
        )

        if "token_to_group_id" in self._buffers:
            self.token_to_group_id = token_to_group_id
        else:
            self.register_buffer("token_to_group_id", token_to_group_id, persistent=True)
        if "token_intra_rank" in self._buffers:
            self.token_intra_rank = token_intra_rank
        else:
            self.register_buffer("token_intra_rank", token_intra_rank, persistent=True)
        if "bucket_sparse" in self._buffers:
            self.bucket_sparse = bucket_sparse
        else:
            self.register_buffer("bucket_sparse", bucket_sparse, persistent=True)

        if self._alias_offsets_array is not None:
            alias_offsets = torch.from_numpy(np.asarray(self._alias_offsets_array)).to(
                dtype=torch.long, device=buffer_device
            )
            alias_counts = alias_offsets[1:] - alias_offsets[:-1]
            if "alias_offsets" in self._buffers:
                self.alias_offsets = alias_offsets
                self.alias_counts = alias_counts
            else:
                self.register_buffer("alias_offsets", alias_offsets, persistent=True)
                self.register_buffer("alias_counts", alias_counts, persistent=True)
        if self._alias_group_ids_array is not None:
            alias_group_ids = torch.from_numpy(np.asarray(self._alias_group_ids_array)).to(
                dtype=torch.long, device=buffer_device
            )
            if "alias_group_ids" in self._buffers:
                self.alias_group_ids = alias_group_ids
            else:
                self.register_buffer("alias_group_ids", alias_group_ids, persistent=True)
        if self._alias_intra_ranks_array is not None:
            alias_intra_ranks = torch.from_numpy(np.asarray(self._alias_intra_ranks_array)).to(
                dtype=torch.long, device=buffer_device
            )
            if "alias_intra_ranks" in self._buffers:
                self.alias_intra_ranks = alias_intra_ranks
            else:
                self.register_buffer("alias_intra_ranks", alias_intra_ranks, persistent=True)
        if self._alias_weights_array is not None:
            alias_weights = torch.from_numpy(np.asarray(self._alias_weights_array)).to(
                dtype=torch.float32, device=buffer_device
            )
            if "alias_weights" in self._buffers:
                self.alias_weights = alias_weights
            else:
                self.register_buffer("alias_weights", alias_weights, persistent=True)

        # Rebuild the hash multiplier buffer from the stored Python values.
        if hasattr(self, "hash_multipliers_val"):
            multipliers_t = torch.tensor(self.hash_multipliers_val, dtype=torch.long, device=buffer_device)
            if "_hash_multipliers_tensor" in self._buffers:
                self._hash_multipliers_tensor = multipliers_t
            else:
                self.register_buffer("_hash_multipliers_tensor", multipliers_t, persistent=True)


        # Skip dense-table precomputation on meta devices to avoid unnecessary
        # CPU work during initialization.
        if str(device) != "meta":
            dense_indices, dense_weights = self._precompute_dense_tables(device=device)
            self._dense_indices = dense_indices
            self._dense_weights = dense_weights

    def _precompute_dense_tables(
        self, *, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_str = str(device)
        buffer_device: Union[str, torch.device] = device if device_str != "meta" else "cpu"

        token_ids = torch.arange(self.vocab_size, device=buffer_device)
        group_ids = torch.from_numpy(self._token_to_group_id_array).to(buffer_device)
        intra_ranks = torch.from_numpy(self._token_intra_rank_array).to(buffer_device)

        alias_counts = None
        max_copies = 0
        if self._alias_offsets_array is not None:
            alias_offsets_arr = self._alias_offsets_array
            alias_counts_arr = alias_offsets_arr[1:] - alias_offsets_arr[:-1]
            if alias_counts_arr.size > 0:
                max_copies = int(alias_counts_arr.max())

        K = 1 + max_copies
        H = self.num_heads
        
        log.info(
            "shortconv dense precompute (H=%d heads): vocab=%d K=%d MaxCopies=%d device=%s",
            H, self.vocab_size, K, max_copies, buffer_device
        )

        # The last dimension is expanded to K * H so each alias slot stores one
        # physical index per hash head.
        # Layout: [Head1_Base, Head2_Base... HeadH_Base, Head1_Alias1, Head2_Alias1... HeadH_Alias1, ...]
        dense_indices = torch.zeros(
            (self.vocab_size, K * H), dtype=torch.long, device=buffer_device
        )
        dense_weights = torch.zeros(
            (self.vocab_size, K), dtype=torch.float32, device=buffer_device
        )

        group_ids_clamped = group_ids.clamp_min(0)
        is_bucketed = group_ids >= 0
        is_vip = group_ids == -1

        dense_weights[:, 0] = 1.0
        
        # --- Handle Aliases Setup ---
        alias_token_indices = None
        alias_col_indices = None
        alias_groups = None
        alias_intras = None
        
        if max_copies > 0 and self._alias_offsets_array is not None:
            counts_t = self.alias_counts.to(buffer_device)
            total_alias = int(counts_t.sum())
            if total_alias > 0:
                alias_token_indices = torch.repeat_interleave(
                    torch.arange(self.vocab_size, device=buffer_device), counts_t
                )
                alias_prefix = torch.cumsum(counts_t, dim=0) - counts_t
                alias_within = torch.arange(total_alias, device=buffer_device) - torch.repeat_interleave(
                    alias_prefix, counts_t
                )
                alias_col_indices = alias_within + 1
                alias_groups = self.alias_group_ids.to(buffer_device)
                alias_intras = self.alias_intra_ranks.to(buffer_device)
                if self._alias_weights_array is not None:
                    dense_weights[alias_token_indices, alias_col_indices] = self.alias_weights.to(buffer_device)
                else:
                    dense_weights[alias_token_indices, alias_col_indices] = 1.0

        # Materialize the per-head hash multipliers.
        multipliers = self._hash_multipliers_tensor.to(buffer_device)
        if multipliers.numel() < H:
            raise OLMoConfigurationError(
                f"Hash injection expects at least {H} hash multipliers, got {multipliers.numel()}"
            )

        phys_size = self.bucket_physical_size
        bucket_sparse_flags = self.bucket_sparse[group_ids_clamped]
        bucket_starts = self.vip_reserved_total + group_ids_clamped * phys_size

        # Generate base indices for each hash head.
        for h_idx in range(H):
            m = multipliers[h_idx]

            if phys_size > 0:
                hash_vals = (token_ids * m) & ((1 << 63) - 1)
                hash_idx = (hash_vals % phys_size).long()
            else:
                hash_idx = torch.zeros_like(token_ids)

            local_offsets = torch.where(bucket_sparse_flags, intra_ranks, hash_idx)

            table_indices = torch.zeros(self.vocab_size, dtype=torch.long, device=buffer_device)
            table_indices = torch.where(is_bucketed, bucket_starts + local_offsets, table_indices)
            table_indices = torch.where(is_vip, intra_ranks.clamp(min=0), table_indices)

            dense_indices[:, h_idx] = table_indices

        # --- Handle Alias Indices ---
        if max_copies > 0 and alias_token_indices is not None:
            is_alias_vip = (alias_groups == -1)
            is_alias_bucket = (alias_groups >= 0)
            ag_clamped = alias_groups.clamp_min(0)
            a_bucket_sparse = self.bucket_sparse[ag_clamped]
            a_bucket_starts = self.vip_reserved_total + ag_clamped * phys_size

            for h_idx in range(H):
                m = multipliers[h_idx]
                flat_phys = torch.zeros_like(alias_groups)
                flat_phys = torch.where(is_alias_vip, alias_intras, flat_phys)

                if is_alias_bucket.any():
                    if phys_size > 0:
                        a_hash_vals = (alias_token_indices * m) & ((1 << 63) - 1)
                        a_hash_idx = (a_hash_vals % phys_size).long()
                    else:
                        a_hash_idx = torch.zeros_like(alias_groups, dtype=torch.long)

                    a_local = torch.where(a_bucket_sparse, alias_intras, a_hash_idx)
                    bucket_indices = a_bucket_starts + a_local
                    flat_phys = torch.where(is_alias_bucket, bucket_indices, flat_phys)

                # Map logic indices (col_idx) to physical columns
                target_cols = h_idx + H * alias_col_indices
                dense_indices[alias_token_indices, target_cols] = flat_phys


        return dense_indices, dense_weights

    def _log_dense_snapshot(
        self, dense_indices: torch.Tensor, *, stage: str, device: Union[str, torch.device]
    ) -> None:
        try:
            cap = self._bucket_embedding.num_embeddings
            t_min = int(dense_indices.min().item())
            t_max = int(dense_indices.max().item())

        except Exception: 
            pass

    def _ensure_dense(self, device: torch.device) -> None:
        # Dense tables are precomputed in _reset_injection_buffers(); keep this
        # method for backward-compatible external calls.
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat_ids = token_ids.view(-1)
        device = self._bucket_embedding.weight.device
        if flat_ids.device != device:
            flat_ids = flat_ids.to(device=device, non_blocking=True)
        dtype = self._bucket_embedding.weight.dtype

        dense_indices = self._dense_indices   # [V, K*H]
        dense_weights = self._dense_weights   # [V, K]

        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        U = unique_ids.shape[0]

        H = self.num_heads

        u_idx = dense_indices[unique_ids, :]                   # [U, K*H]
        u_vecs = self._bucket_embedding(u_idx)                 # [U, K*H, D]
        u_paired = u_vecs.view(U, -1, H, self.d_model)        # [U, K, H, D]

        w_embedding = self._scalar_weight_embeddings[0]
        u_gates_raw = w_embedding(u_idx)                  # [U, K*H, 1]
        u_gates_sig = torch.sigmoid(u_gates_raw)
        u_gates_paired = u_gates_sig.view(U, -1, H, 1)    # [U, K, H, 1]
        u_vectors = (u_paired * u_gates_paired).sum(dim=2)  # [U, K, D]

        u_weights = dense_weights[unique_ids]                 # [U, K]
        u_weighted = (u_vectors * u_weights.unsqueeze(-1)).sum(dim=1)  # [U, D]
        u_denom = u_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)   # [U, 1]
        u_output = (u_weighted / u_denom).to(dtype)           # [U, D]


        combined = F.embedding(inverse, u_output)             # [N, D]

        return combined.view(*token_ids.shape, self.d_model)
        

