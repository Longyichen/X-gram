import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.distributed.utils import distribute_like
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from .runtime import InjectionBlockContext, resolve_configured_layers

log = logging.getLogger(__name__)


def _build_compressed_lookup_table(
    *, tokenizer_id: str, vocab_size: int, cache_path: Optional[Union[str, Path]] = None
) -> torch.Tensor:
    """
    Build a lookup tensor that maps original tokenizer IDs to normalized/merged IDs.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
        from tokenizers import normalizers, Regex  # type: ignore
    except Exception as exc:
        log.warning("Compressed tokenizer unavailable (%s); using identity mapping", exc)
        return torch.arange(vocab_size, dtype=torch.long)

    cache_tensor: Optional[torch.Tensor] = None
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            try:
                cache_np = np.load(cache_file)["lookup_table"]
                cache_tensor = torch.from_numpy(cache_np.astype(np.int64))
                log.info("Loaded compressed token lookup from %s", cache_file)
            except Exception as exc:
                log.warning("Failed to load compressed lookup cache %s (%s); recomputing", cache_file, exc)

    if cache_tensor is not None:
        return cache_tensor

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    SENTINEL = "\uE000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ]
    )

    raw_vocab_size = len(tokenizer)
    table_size = max(vocab_size, raw_vocab_size)
    lookup_np = np.arange(table_size, dtype=np.int64)
    key2new: Dict[str, int] = {}
    new_id = 0
    for tid in range(raw_vocab_size):
        try:
            text = tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            text = tokenizer.convert_ids_to_tokens(tid)

        if "\ufffd" in text:
            key = tokenizer.convert_ids_to_tokens(tid)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text

        nid = key2new.get(key)
        if nid is None:
            nid = new_id
            key2new[key] = nid
            new_id += 1

        lookup_np[tid] = nid

    lookup_tensor = torch.from_numpy(lookup_np)
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, lookup_table=lookup_np)
        log.info("Saved compressed token lookup to %s (unique tokens: %d)", cache_file, new_id)
    else:
        log.info("Built compressed token lookup on-the-fly (unique tokens: %d)", new_id)

    return lookup_tensor


def _map_ids_with_lookup(input_ids: torch.Tensor, lookup_table: torch.Tensor) -> torch.Tensor:
    """Apply a lookup table to token IDs safely (clamp range, keep dtype/device)."""
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()
    if lookup_table.device != input_ids.device:
        lookup_table = lookup_table.to(device=input_ids.device)
    max_idx = lookup_table.shape[0] - 1
    clamped = input_ids.clamp(min=0, max=max_idx)
    return lookup_table[clamped]


def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: Set[int]) -> int:
    candidate = max(2, start)
    while True:
        if _is_prime(candidate) and candidate not in seen:
            return candidate
        candidate += 1


class EngramNgramHash(nn.Module):
    """
    Multi-head n-gram hash router matching engram.py:
    mix = XOR(token_{t-k} * multipliers[k] for k in 0..ngram_size-1), then prime modulo per head.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_heads: int,
        target_total_capacity: int,
        ngram_size: int = 2,
        pad_id: int = 0,
        seed: int = 137,
        seen_primes: Optional[Set[int]] = None,
    ):
        super().__init__()
        if ngram_size < 2:
            raise OLMoConfigurationError("ngram_size must be >= 2")
        if num_heads < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_HEADS must be >= 1")
        if target_total_capacity < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_TARGET_BUCKETS must be > 0")
        target_total_capacity = max(num_heads, int(target_total_capacity))
        per_head_capacity = max(2, target_total_capacity // num_heads)

        seen: Set[int] = seen_primes if seen_primes is not None else set()
        primes: List[int] = []
        current = per_head_capacity
        for _ in range(num_heads):
            prime = _next_prime(current, seen)
            primes.append(prime)
            seen.add(prime)
            current = prime + 1

        offsets: List[int] = [0]
        for prime in primes[:-1]:
            offsets.append(offsets[-1] + prime)

        self.num_heads = num_heads
        self.ngram_size = ngram_size
        self._prime_values = tuple(primes)
        self._offset_values = tuple(offsets)
        raw_total = int(sum(primes))
        align = 16
        self.total_embeddings = ((raw_total + align - 1) // align) * align
        self._pad_id = pad_id
        gen = torch.Generator()
        gen.manual_seed(seed)
        max_long = (1 << 63) - 1
        half_bound = max(1, max_long // vocab_size // 2)
        r = torch.randint(0, half_bound, (ngram_size,), generator=gen, dtype=torch.long)
        multipliers = r.mul(2).add(1)  # ensure odd
        self._multiplier_values = tuple(int(v) for v in multipliers.tolist())

        self.register_buffer("primes", torch.tensor(primes, dtype=torch.long), persistent=True)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=True)
        self.register_buffer("multipliers", multipliers, persistent=True)

        log.info(
            "Engram %d-gram hash initialized: heads=%d total_embeddings=%d target=%d multipliers=%s pad_id=%d",
            ngram_size, num_heads, self.total_embeddings, target_total_capacity,
            self._multiplier_values, pad_id,
        )

    def reset_hash_buffers(self, *, device: Optional[torch.device] = None) -> None:
        target_device = device or get_default_device()
        items = {
            "primes": torch.tensor(self._prime_values, dtype=torch.long, device=target_device),
            "offsets": torch.tensor(self._offset_values, dtype=torch.long, device=target_device),
            "multipliers": torch.tensor(self._multiplier_values, dtype=torch.long, device=target_device),
        }
        for name, tensor in items.items():
            existing = getattr(self, name, None)
            if existing is None or name not in self._buffers or existing.is_meta or existing.shape != tensor.shape:
                setattr(self, name, tensor)
                continue
            if existing.device != tensor.device or existing.dtype != tensor.dtype:
                setattr(self, name, tensor)
            else:
                existing.copy_(tensor)

    def forward(self, compressed_ids: torch.Tensor) -> torch.Tensor:
        ids = compressed_ids.long()
        B, T = ids.shape
        primes = self.primes
        offsets = self.offsets
        multipliers = self.multipliers
        if primes.device != ids.device:
            primes = primes.to(device=ids.device)
            offsets = offsets.to(device=ids.device)
            multipliers = multipliers.to(device=ids.device)

        padded = F.pad(ids, (self.ngram_size - 1, 0), value=self._pad_id)

        mix = padded[:, (self.ngram_size - 1):] * multipliers[0]
        for k in range(1, self.ngram_size):
            shifted = padded[:, (self.ngram_size - 1 - k):(T + self.ngram_size - 1 - k)]
            mix = torch.bitwise_xor(mix, shifted * multipliers[k])

        hashes: List[torch.Tensor] = []
        for i in range(self.num_heads):
            head_hash = torch.remainder(mix, primes[i]) + offsets[i]
            hashes.append(head_hash)
        return torch.stack(hashes, dim=2)

class EngramShortConv(nn.Module):
    """
    Depthwise causal short convolution operating on ``(B, L, HC_MULT, D)`` tensors.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 1,
        activation: bool = True,
        *,
        init_device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            device=init_device,
            dtype=dtype,
        )
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps, device=init_device, dtype=dtype)
            for _ in range(hc_mult)
        ])
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, G, C = x.shape
        normed_chunks = []
        for i in range(G):
            normed_chunks.append(self.norms[i](x[:, :, i, :]))
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]  # causal trim
        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        return y


class EngramModule(nn.Module):


    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        tokenizer_id: str,
        cache_path: Optional[str],
        init_device: str,
        dtype: torch.dtype,
        hc_mult: int = 1,
        shortconv_enabled: bool = True,
        shortconv_kernel_size: int = 4,
        shortconv_dilation: int = 1,
        shortconv_activation: bool = True,
        ngram_levels: Optional[List[int]] = None,
        ngram_dim_per_level: int = 0,
        ngram_heads_per_level: int = 4,
        ngram_target_capacity: int = 0,
        ngram_base_seed: int = 0,
        ngram_pad_id: int = 0,
        seen_primes: Optional[Set[int]] = None,
    ):
        super().__init__()
        if ngram_levels is None or len(ngram_levels) == 0:
            raise OLMoConfigurationError("Engram mode requires at least one n-gram level")
        if any(level < 2 for level in ngram_levels):
            raise OLMoConfigurationError("Engram mode only supports n-gram levels >= 2")
        if ngram_dim_per_level < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_DIM must be >= 1")
        if ngram_heads_per_level < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_HEADS must be >= 1")
        if ngram_dim_per_level % ngram_heads_per_level != 0:
            raise OLMoConfigurationError(
                f"ENGRAM_NGRAM_DIM ({ngram_dim_per_level}) must be divisible by heads ({ngram_heads_per_level})"
            )
        if ngram_target_capacity < 1:
            raise OLMoConfigurationError("ENGRAM_NGRAM_TARGET_BUCKETS must be > 0")

        self._tokenizer_id = tokenizer_id
        self._cache_path = cache_path
        self._vocab_size = vocab_size
        self.d_model = d_model
        self.hc_mult = hc_mult
        self.shortconv_enabled = shortconv_enabled
        self.ngram_levels = sorted(ngram_levels)

        lookup = _build_compressed_lookup_table(
            tokenizer_id=tokenizer_id,
            vocab_size=vocab_size,
            cache_path=cache_path,
        )
        self.register_buffer("lookup_table", lookup, persistent=False)
        compressed_vocab_size = int(lookup.numel())

        self.ngram_hashers = nn.ModuleList()
        self.ngram_embeddings = nn.ModuleList()
        ngram_total_hidden = 0
        _ngram_dim = ngram_dim_per_level
        _ngram_pad = int(lookup[ngram_pad_id].item()) if ngram_pad_id < lookup.numel() else 0
        _ngram_base_seed = ngram_base_seed

        for level in self.ngram_levels:
            level_seed = _ngram_base_seed + 7919 * (level - 1)
            hasher = EngramNgramHash(
                vocab_size=compressed_vocab_size,
                num_heads=ngram_heads_per_level,
                target_total_capacity=ngram_target_capacity,
                ngram_size=level,
                pad_id=_ngram_pad,
                seed=level_seed,
                seen_primes=seen_primes,
            )
            self.ngram_hashers.append(hasher)
            level_head_dim = _ngram_dim // ngram_heads_per_level
            emb = nn.Embedding(
                hasher.total_embeddings,
                level_head_dim,
                dtype=dtype,
                device=init_device,
            )
            self.ngram_embeddings.append(emb)
            ngram_total_hidden += _ngram_dim

        engram_hidden_size = ngram_total_hidden

        # value_proj: project engram embedding to d_model
        self.value_proj = nn.Linear(engram_hidden_size, d_model, bias=False, dtype=dtype, device=init_device)

        # key_projs: one per hc channel, for learned gating
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, d_model, bias=False, dtype=dtype, device=init_device)
            for _ in range(hc_mult)
        ])

        # per-hc RMSNorm for key (norm1) and query (norm2)
        self.norm1 = nn.ModuleList([
            nn.RMSNorm(d_model, device=init_device, dtype=dtype)
            for _ in range(hc_mult)
        ])
        self.norm2 = nn.ModuleList([
            nn.RMSNorm(d_model, device=init_device, dtype=dtype)
            for _ in range(hc_mult)
        ])

        # ShortConv (optional, default on per engram.py)
        self.short_conv: Optional[EngramShortConv] = None
        if shortconv_enabled:
            self.short_conv = EngramShortConv(
                hidden_size=d_model,
                kernel_size=shortconv_kernel_size,
                dilation=shortconv_dilation,
                hc_mult=hc_mult,
                activation=shortconv_activation,
                init_device=init_device,
                dtype=dtype,
            )

    def reset_parameters(self) -> None:
        for emb in self.ngram_embeddings:
            emb.reset_parameters()

    def reset_buffers(self, *, device: Optional[torch.device] = None) -> None:
        target_device = device or get_default_device()
        lookup = _build_compressed_lookup_table(
            tokenizer_id=self._tokenizer_id,
            vocab_size=self._vocab_size,
            cache_path=self._cache_path,
        ).to(device=target_device, dtype=torch.long)
        existing = getattr(self, "lookup_table", None)
        if existing is None or "lookup_table" not in self._buffers:
            self.register_buffer("lookup_table", lookup, persistent=False)
        elif existing.is_meta or existing.shape != lookup.shape:
            setattr(self, "lookup_table", lookup)
        elif existing.device != lookup.device or existing.dtype != lookup.dtype:
            setattr(self, "lookup_table", lookup)
        else:
            existing.copy_(lookup)
        for hasher in self.ngram_hashers:
            hasher.reset_hash_buffers(device=target_device)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        return_gate_mean: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        compressed_ids = _map_ids_with_lookup(input_ids, self.lookup_table)

        parts: List[torch.Tensor] = []
        for hasher, emb in zip(self.ngram_hashers, self.ngram_embeddings):
            ngram_hash_ids = hasher(compressed_ids)
            parts.append(emb(ngram_hash_ids).flatten(start_dim=2))

        embeddings = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        # 2. per-hc learned gate: key-query dot -> abs -> sqrt -> sigmoid
        value = self.value_proj(embeddings)

        if hidden_states is None:
            if self.hc_mult > 1:
                out = value.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
            else:
                out = value
            if return_gate_mean:
                return out, torch.ones(1, device=out.device, dtype=out.dtype)
            return out

        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            if hidden_states.dim() == 4:
                query = hidden_states[:, :, hc_idx, :]
            else:
                query = hidden_states
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.d_model)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        gates_t = torch.stack(gates, dim=2)
        gated_value = gates_t * value.unsqueeze(2)

        # 3. short conv residual
        if self.short_conv is not None:
            output = gated_value + self.short_conv(gated_value)
        else:
            output = gated_value

        # 4. squeeze if no hyper-connection
        if self.hc_mult == 1:
            output = output.squeeze(2)

        if return_gate_mean:
            gate_mean = gates_t.detach().mean()
            return output, gate_mean
        return output


def _warmup_scale_to_python_float(warmup_scale: Union[float, torch.Tensor]) -> float:
    if torch.is_tensor(warmup_scale):
        return float(warmup_scale.detach().float().cpu().item())
    return float(warmup_scale)


def build_engram_modules(
    transformer: Any,
    embedding_injection: Any,
    *,
    vocab_size: int,
    d_model: int,
    dtype: torch.dtype,
    init_device: str,
) -> None:
    tokenizer_id = getattr(embedding_injection, "engram_tokenizer_id", None)
    if not tokenizer_id:
        raise OLMoConfigurationError(
            "Engram mode requires engram_tokenizer_id in TransformerEmbeddingInjectionConfig"
        )

    transformer._engram_enabled = True

    engram_mode = getattr(embedding_injection, "engram_mode", "2gram+3gram")
    engram_mode_parts = {s.strip() for s in engram_mode.split("+") if s.strip()}
    engram_ngram_levels: List[int] = sorted(int(p.replace("gram", "")) for p in engram_mode_parts)
    if len(engram_ngram_levels) == 0:
        raise OLMoConfigurationError("Engram mode requires at least one n-gram level")
    if any(level < 2 for level in engram_ngram_levels):
        raise OLMoConfigurationError("Engram mode only supports n-gram levels >= 2")
    num_active_ngrams = len(engram_ngram_levels)
    engram_dim_per_ngram_cfg = getattr(embedding_injection, "engram_dim_per_ngram", None)
    engram_dim_per_ngram = (
        int(engram_dim_per_ngram_cfg)
        if engram_dim_per_ngram_cfg is not None
        else d_model // num_active_ngrams
    )

    transformer._engram_config = {
        "tokenizer_id": tokenizer_id,
        "cache_path": getattr(embedding_injection, "engram_cache_path", None),
        "base_seed": int(getattr(embedding_injection, "engram_base_seed", 42)),
        "hc_mult": int(getattr(embedding_injection, "engram_hc_mult", 1)),
        "shortconv_enabled": bool(getattr(embedding_injection, "engram_shortconv_enabled", True)),
        "shortconv_kernel": int(getattr(embedding_injection, "engram_shortconv_kernel", 4)),
        "shortconv_dilation": int(getattr(embedding_injection, "engram_shortconv_dilation", 1)),
        "shortconv_activation": bool(getattr(embedding_injection, "engram_shortconv_activation", True)),
        "ngram_levels": engram_ngram_levels,
        "ngram_heads": int(getattr(embedding_injection, "engram_ngram_heads", 4)),
        "ngram_target_capacity": getattr(embedding_injection, "engram_ngram_target_buckets", 75968),
        "ngram_dim": engram_dim_per_ngram,
        "ngram_seed": int(getattr(embedding_injection, "engram_ngram_seed", 137)),
    }

    h_injection_layers = resolve_configured_layers(
        getattr(embedding_injection, "h_layers", None),
        default_layers=list(embedding_injection.layers),
    )
    seen_primes: Set[int] = set()
    seen_layers: Set[int] = set()
    ecfg = transformer._engram_config
    if ecfg is None:
        return

    for layer_idx in h_injection_layers:
        if layer_idx in seen_layers:
            raise OLMoConfigurationError(
                f"Engram supports at most one module per layer; duplicate layer index {layer_idx} found"
            )
        seen_layers.add(layer_idx)
        block_key = str(layer_idx)
        if block_key not in transformer._injection_h_embeddings:
            transformer._injection_h_embeddings[block_key] = torch.nn.ModuleList()
        elif len(transformer._injection_h_embeddings[block_key]) > 0:
            raise OLMoConfigurationError(
                f"Engram supports at most one module per layer; layer {layer_idx} is already initialized"
            )
        module_index = 0
        instance_seed = ecfg["base_seed"] + 10007 * layer_idx + module_index
        engram_module = EngramModule(
            d_model=d_model,
            vocab_size=vocab_size,
            tokenizer_id=ecfg["tokenizer_id"],
            cache_path=ecfg["cache_path"],
            init_device=init_device,
            dtype=dtype,
            hc_mult=ecfg["hc_mult"],
            shortconv_enabled=ecfg["shortconv_enabled"],
            shortconv_kernel_size=ecfg["shortconv_kernel"],
            shortconv_dilation=ecfg["shortconv_dilation"],
            shortconv_activation=ecfg["shortconv_activation"],
            ngram_levels=ecfg.get("ngram_levels"),
            ngram_dim_per_level=ecfg.get("ngram_dim"),
            ngram_heads_per_level=ecfg.get("ngram_heads", 4),
            ngram_target_capacity=ecfg.get("ngram_target_capacity"),
            ngram_base_seed=instance_seed,
            ngram_pad_id=2,
            seen_primes=seen_primes,
        )
        transformer._injection_h_embeddings[block_key].append(engram_module)


def apply_engram_pre_block(
    transformer: Any,
    context: InjectionBlockContext,
) -> torch.Tensor:
    if (
        context.block_key not in transformer._injection_h_embeddings
        or len(transformer._injection_h_embeddings[context.block_key]) == 0
    ):
        return context.hidden_states

    h = context.hidden_states
    engram_modules = transformer._injection_h_embeddings[context.block_key]
    if len(engram_modules) != 1:
        raise OLMoConfigurationError(
            f"Engram expects exactly one module per layer after initialization; "
            f"found {len(engram_modules)} for layer {context.block_idx}"
        )

    engram_delta = engram_modules[0](
        context.input_ids,
        hidden_states=h,
        return_gate_mean=False,
    )
    engram_delta = engram_delta.to(dtype=h.dtype, device=h.device)
    if isinstance(h, DTensor):
        engram_delta = distribute_like(engram_delta, h)
    return h + engram_delta
