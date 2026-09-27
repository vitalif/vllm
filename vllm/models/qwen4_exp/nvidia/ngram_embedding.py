# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp n-gram embeddings with device and pinned-host storage."""

from collections.abc import Iterable
from typing import cast

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.utils.torch_utils import get_dtype_size

from ..common.ngram_embedding import (
    Qwen4ExpPLEDeviceEmbedding,
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)
from . import ple_mmap
from .ops.ple import ple_ngram_ids

logger = init_logger(__name__)

__all__ = [
    "Qwen4ExpPLEDeviceEmbedding",
    "Qwen4ExpPLEEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "Qwen4ExpPLEFp8EmbeddingMethod",
    "Qwen4ExpPLEPinnedHostEmbedding",
    "Qwen4ExpPLEUnquantizedEmbeddingMethod",
    "Qwen4ExpNGramEmbedding",
]


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        """Mix an integer into a deterministic unsigned 64-bit value."""
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        """Return whether a 64-bit integer is prime."""
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        """Return the ``count``-th prime strictly greater than ``start``."""
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list[int]:
        """Build deterministic hash multipliers for one PLE layer."""
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> tuple[list[int], list[int], int]:
        """Build per-head vocabulary sizes, offsets, and total row count."""
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        *,
        data_parallel_rank: int,
        prefix: str,
        layer_name: str | None = None,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = prefix if layer_name is None else layer_name
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        self.ngram_embedding: Qwen4ExpPLEEmbedding | ple_mmap.MmapNgramEmbedding
        # Non-persistent, module-owned staging buffer for V2 mmap external
        # staging (see initialize_mmap_staging). None until V2 model state
        # allocates it; stays None for the non-mmap embedding.
        self._mmap_staging: torch.Tensor | None = None
        if ple_mmap.enabled():
            vllm_config = get_current_vllm_config()
            ple_mmap.check_cudagraph_safety(vllm_config)
            discovered_dtype = ple_mmap.validate_shards_for(
                vllm_config.model_config, self.layer_name, self.head_dim
            )
            self.ngram_embedding = ple_mmap.MmapNgramEmbedding(
                padded_vocab_size, self.head_dim
            )
            if discovered_dtype is not None:
                # Seed the placeholder's fallback dtype from validated shard
                # headers now, before any weights stream, so a dummy load's
                # V2 staging buffer (initialize_mmap_staging) allocates at
                # the checkpoint's real dtype instead of the FP8 default.
                self.ngram_embedding.torch_dtype = discovered_dtype
            return
        embedding_quant_method = Qwen4ExpPLEEmbeddingMethod.from_quant_config(
            quant_config,
            embedding_prefix,
            getattr(config, "ple_embedding_dtype", None),
        )
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        engram_config = get_current_vllm_config().engram_config
        embedding_cls = (
            Qwen4ExpPLEPinnedHostEmbedding
            if engram_config is not None and engram_config.cpu_offload
            else Qwen4ExpPLEDeviceEmbedding
        )
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.ngram_embedding.supports_prefetch:
            # The side-stream lookup outlives eager-break args, whose
            # graph-pool storage later segments may reuse.
            self._prefetch_ids = torch.empty(
                max_total_tokens, self.ngram_heads, dtype=torch.long
            )
        weight = self.ngram_embedding.weight
        logger.info(
            "Initialized PLE embedding %s: quantization_method=%s, "
            "weight_dtype=%s, weight_device=%s, pinned=%s",
            embedding_prefix,
            type(embedding_quant_method).__name__,
            weight.dtype,
            weight.device,
            weight.is_pinned(),
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute n-gram embedding indices for the current request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if input_ids.is_cuda:
            return ple_ngram_ids(
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
                layer_multipliers=self.layer_multipliers,
                ngram_heads_vocab_sizes=self.ngram_heads_vocab_sizes,
                ngram_heads_offsets=self.ngram_heads_offsets,
                eos_token_id=self.eos_token_id,
                heads_per_ngram=self.heads_per_ngram,
                output=output,
            )
        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()
        positions = torch.arange(num_tokens, device=input_ids.device, dtype=torch.int64)
        packed = torch.full(
            (num_reqs, num_tokens),
            self.eos_token_id,
            device=input_ids.device,
            dtype=torch.int64,
        )
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def _require_mmap_embedding(self) -> ple_mmap.MmapNgramEmbedding:
        if not isinstance(self.ngram_embedding, ple_mmap.MmapNgramEmbedding):
            raise RuntimeError(
                f"PLE mmap: {self.layer_name!r} is not using mmap staging"
            )
        return self.ngram_embedding

    def _resolve_mmap_dtype(self) -> torch.dtype:
        """Resolve this layer's PLE row dtype without allocating a buffer.

        Prefers the attached table's dtype — the authoritative source once a
        real load has streamed weights. Falls back to the placeholder's own
        ``torch_dtype`` (derived from validated shard headers at
        construction, or its FP8 default when construction never resolved a
        model path) for a dummy load that has not attached a table.

        Raises:
            RuntimeError: a real (non-dummy) load already streamed weights
                but ``build_tables`` never attached a table — fail closed
                rather than silently stage zeros as if they were real rows.

        """
        embedding = self._require_mmap_embedding()
        table = embedding.table
        if table is not None:
            return table.torch_dtype
        if embedding.weights_streamed:
            raise RuntimeError(
                f"PLE mmap: {self.layer_name!r} streamed weights but never "
                "attached a table before mmap staging was initialized"
            )
        return embedding.torch_dtype

    def mmap_staging_nbytes(self, max_num_tokens: int) -> int:
        """Bytes this layer's staging buffer would occupy at ``max_num_tokens``.

        Used by V2 model state to compute the aggregate allocation preflight
        BEFORE any layer's buffer is actually allocated.
        """
        dtype = self._resolve_mmap_dtype()
        return max_num_tokens * self.ngram_heads * self.head_dim * get_dtype_size(dtype)

    def initialize_mmap_staging(
        self, max_num_tokens: int, device: torch.device
    ) -> None:
        """Allocate this layer's stable, non-persistent staged-row buffer.

        Called once by V2 model state, after the aggregate allocation
        preflight has already cleared every layer for allocation. The
        buffer's address, dtype, and shape never change afterward; only its
        contents are overwritten in place by ``prepare_mmap_rows`` /
        ``prepare_dummy_mmap_rows``.
        """
        dtype = self._resolve_mmap_dtype()
        self._mmap_staging = torch.zeros(
            (max_num_tokens, self.ngram_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )

    def prepare_mmap_rows(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        actual_tokens: int,
        padded_tokens: int,
    ) -> None:
        """Gather this layer's staged rows for the current real step.

        V2 model state calls this from ``prepare_inputs``, BEFORE the
        compiled/captured forward runs. ``input_ids``/``query_start_loc``/
        ``ngram_context`` must already be sliced to actual (unpadded)
        extents by the caller — this never gathers graph padding. Zeros
        ``[actual_tokens:padded_tokens]`` every call so a smaller batch
        reusing a larger graph's buffer never replays stale rows.
        """
        if self._mmap_staging is None:
            raise RuntimeError(
                f"PLE mmap: {self.layer_name!r} staging was never initialized"
            )
        embedding = self._require_mmap_embedding()
        if actual_tokens > 0:
            ngram_ids = self.compute_ngram_ids(
                input_ids, query_start_loc, ngram_context
            )
            embedding.gather_into(ngram_ids, self._mmap_staging[:actual_tokens])
        if padded_tokens > actual_tokens:
            self._mmap_staging[actual_tokens:padded_tokens].zero_()

    def prepare_dummy_mmap_rows(self, padded_tokens: int) -> None:
        """Zero this layer's staged rows for a dummy/capture step.

        No hashing, mmap file access, pinned allocation, or H2D copy —
        dummy preparation performs no table access at all.
        """
        if self._mmap_staging is None:
            raise RuntimeError(
                f"PLE mmap: {self.layer_name!r} staging was never initialized"
            )
        self._mmap_staging[:padded_tokens].zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(self.ngram_embedding, ple_mmap.MmapNgramEmbedding):
            if self._mmap_staging is None:
                raise RuntimeError(
                    f"PLE mmap: input preparation did not initialize "
                    f"{self.layer_name!r}; Model Runner V2 is required"
                )
            # Keep this symbolic under torch.compile: no int() and no
            # .numel()-derived slicing, which is what specialized vLLM's
            # dynamic dims into a ConstraintViolationError on
            # query_start_loc.size()[0] under the old whole-forward custom
            # op. A plain shape[0] read stays a SymInt when traced.
            num_tokens = input_ids.reshape(-1).shape[0]
            return self._mmap_staging[:num_tokens].flatten(-2)
        if query_start_loc is None or ngram_context is None:
            raise RuntimeError("PLE inputs were not prepared")
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def start_prefetch(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        """Start the pinned lookup while the preceding decoder layer runs."""
        embedding = self.ngram_embedding
        if not embedding.supports_prefetch:
            return
        ngram_ids = self.compute_ngram_ids(
            input_ids,
            query_start_loc,
            ngram_context,
            output=self._prefetch_ids[: input_ids.numel()],
        )
        embedding.start_prefetch(hidden_states, ngram_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""
        embedding = self.ngram_embedding
        if (
            isinstance(embedding, ple_mmap.MmapNgramEmbedding)
            and embedding.table is not None
        ):
            # Fail closed BEFORE touching `weights` at all: a same-path
            # reload (build_tables' own model_path check never fires, since
            # nothing about the path changed) would otherwise mutate
            # weight_scale and discard this call's shards onto a module
            # whose table+scale still belong to the load that already
            # attached — silently pairing the new checkpoint's scale with
            # the previous checkpoint's mmap rows. Rejecting here, before
            # the iterator is ever advanced, leaves the existing table,
            # scale, weights_streamed, and iterator exactly as they were.
            raise RuntimeError(
                f"PLE mmap: {self.layer_name!r} already has a table "
                "attached from a previous load; calling load_weights again "
                "on the same live module is unsupported — it would mix "
                "this reload's rows with the already-attached checkpoint's "
                "scale. Restart the seat to load different weights."
            )
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if (
                isinstance(embedding, ple_mmap.MmapNgramEmbedding)
                and name == "ngram_embedding.weight_scale"
            ):
                # The placeholder has no registered weight_scale Parameter for
                # AutoWeightsLoader to find generically; register it directly,
                # on whatever device the module's other buffers already live.
                ple_mmap.set_weight_scale(
                    embedding,
                    loaded_weight,
                    cast(torch.Tensor, self.layer_multipliers).device,
                )
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if isinstance(embedding, ple_mmap.MmapNgramEmbedding):
                    # Served from disk via mmap; the loader still streams
                    # this shard transiently, but it is never retained.
                    # weights_streamed distinguishes this real (non-dummy)
                    # load from a --load-format dummy probe that never
                    # calls load_weights at all — build_tables must attach
                    # a real table before any forward, or the placeholder
                    # raises instead of silently serving fp8 zeros.
                    embedding.weights_streamed = True
                    loaded.add("ngram_embedding.weight")
                    continue
                embedding.weight.weight_loader(
                    embedding.weight,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


__all__ = [
    "Qwen4ExpNGramEmbedding",
]
