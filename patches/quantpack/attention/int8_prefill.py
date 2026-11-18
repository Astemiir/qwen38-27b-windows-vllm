"""Long-context prefill over the int8_per_token_head KV cache with FlashAttention.

int8 KV is served only by vLLM's Triton unified-attention kernel, which unpacks the cache codes inside the
attention loop: on a 33k context that prefill runs at 832 t/s against 1 178 for the int4 path, where a segment is
unpacked once to bf16 and handed to FlashAttention 2. This module gives int8 the same second path: a request whose
chunk and context are long enough is walked in SEGMENT-token pieces, each unpacked by
``int4_prefill.unpack_segment`` (the int8 branch: ``code * scale``, no zero point) and attended by FlashAttention;
the pieces are folded together by log-sum-exp in fp32.

Decode steps, short chunks, short contexts and the features this path does not cover (sliding window, ALiBi, sinks,
softcap, FP8 query/descales, R-SWA, chunk lookback, tensor descriptors) stay on the Triton kernel.
VLLM_QUANTPACK_INT8_FA_PREFILL=0 disables the patch.
"""
import itertools
import os

import torch

from . import int4_prefill
from .int4_kv import _step_lengths

MIN_QUERY = int4_prefill.MIN_QUERY       # a chunk shorter than this is cheap on the Triton kernel
MIN_CONTEXT = int4_prefill.MIN_CONTEXT   # below this the Triton kernel is already fast

# Per-token-head caches this hook can unpack: one code per channel, so the same segment kernel serves both. vLLM
# stores fp8_per_token_head in uint8 and views it as float8 before attention, and both dtypes are one byte wide.
PTH_KV_DTYPES = tuple(d for d in (torch.int8, getattr(torch, "float8_e4m3fn", None),
                                  getattr(torch, "float8_e5m2", None)) if d is not None)


def _eligible(kw, modes):
    """True when the call is a plain long causal prefix over an int8/fp8 per-token-head cache."""
    if not modes or kw.get("kv_quant_mode") not in modes:
        return False
    if kw.get("k") is None or kw["k"].dtype not in PTH_KV_DTYPES:
        return False
    if kw.get("causal") is not True:                 # per-sequence causal tensors stay on Triton
        return False
    if tuple(kw.get("window_size") or (-1, -1)) != (-1, -1):
        return False
    if kw.get("max_seqlen_q", 0) < MIN_QUERY or kw.get("max_seqlen_k", 0) < MIN_CONTEXT:
        return False
    if kw.get("k_scale_cache") is None or kw.get("v_scale_cache") is None:
        return False
    return not (
        kw.get("softcap")
        or kw.get("sinks") is not None
        or kw.get("alibi_slopes") is not None
        or kw.get("qq_bias") is not None
        or kw.get("output_scale") is not None
        or kw.get("mm_prefix_range") is not None
        or kw.get("rswa_window") is not None
        or kw.get("chunk_lookback", -1) >= 0
        or kw.get("use_td", False)
        or kw.get("q_descale") is not None
        or kw.get("k_descale") is not None
        or kw.get("v_descale") is not None
    )


def patch_int8_prefill(logger):
    if os.environ.get("VLLM_QUANTPACK_INT8_FA_PREFILL", "1") == "0":
        return False
    try:
        from vllm.v1.attention.ops import triton_unified_attention as stock
        from vllm.v1.kv_cache_interface import KVQuantMode
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except Exception as e:      # no FlashAttention build for this GPU, or a vLLM without the int8 KV mode
        logger.info("vllm-quantpack: int8 long prefills stay on Triton (%s)", e)
        return False
    orig = stock.unified_attention
    if getattr(orig, "_quantpack_int8", False):
        return True
    modes = {m for m in (getattr(KVQuantMode, "INT8_PER_TOKEN_HEAD", None),
                         getattr(KVQuantMode, "FP8_PER_TOKEN_HEAD", None)) if m is not None}

    def unified_attention(*args, **kw):
        # triton_attn.py calls this with keyword arguments only; anything else goes straight through.
        if args or not _eligible(kw, modes):
            return orig(*args, **kw)
        q, out = kw["q"], kw["out"]
        cu, used = _step_lengths(kw["cu_seqlens_q"], kw["seqused_k"])
        long = [i for i in range(len(used)) if cu[i + 1] - cu[i] >= MIN_QUERY and used[i] >= MIN_CONTEXT]
        if not long:
            return orig(**kw)
        rest = [i for i in range(len(used)) if i not in long and cu[i + 1] > cu[i]]
        if rest:                   # decode tokens and short prefills of the same step: Triton on a sub-batch
            rows = torch.cat([torch.arange(cu[i], cu[i + 1], device=q.device) for i in rest])
            lens = [cu[i + 1] - cu[i] for i in rest]
            sub_cu = torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device=q.device)
            sel = torch.tensor(rest, device=q.device)
            sub_out = torch.empty((len(rows),) + tuple(out.shape[1:]), dtype=out.dtype, device=out.device)
            orig(**{**kw, "q": q[rows], "out": sub_out, "cu_seqlens_q": sub_cu,
                    "max_seqlen_q": max(lens), "seqused_k": kw["seqused_k"][sel],
                    "block_table": kw["block_table"][sel], "max_seqlen_k": max(used[i] for i in rest)})
            out[rows] = sub_out
        for i in long:
            out[cu[i]:cu[i + 1]] = int4_prefill.prefill_sequence(
                q[cu[i]:cu[i + 1]], kw["k"], kw["v"], kw["k_scale_cache"], kw["v_scale_cache"],
                kw["block_table"][i], used[i], kw["softmax_scale"], flash_attn_varlen_func)
        return None

    unified_attention._quantpack_int8 = True
    stock.unified_attention = unified_attention
    try:
        from vllm.v1.attention.backends import triton_attn
        if getattr(triton_attn, "unified_attention", None) is orig:
            triton_attn.unified_attention = unified_attention   # bound at import time in the backend module
    except Exception as e:
        logger.warning("vllm-quantpack: int8 prefill hook not visible to the backend (%s)", e)
    logger.info("vllm-quantpack: int8/fp8 per-token-head long prefills run FlashAttention over unpacked segments")
    return True


PTH_DTYPES = ("int8_per_token_head", "fp8_per_token_head")


def _segments_from_env(logger):
    """Segments per sequence for the int8 split-KV decode: the env value, or auto from the SM count."""
    env = (os.environ.get("VLLM_QUANTPACK_INT8_SEGMENTS") or "auto").strip()
    try:
        from .int4_kv import segments_for
        if not torch.cuda.is_available():
            return 0
        segments = (segments_for(torch.cuda.get_device_properties(0).multi_processor_count)
                    if env.lower() == "auto" else int(env))
    except Exception as e:
        logger.warning("vllm-quantpack: int8 split-KV segments left stock (%s)", e)
        return 0
    if segments < 1 or segments & (segments - 1):
        logger.warning("vllm-quantpack: int8 split-KV segments must be a power of two, got %d", segments)
        return 0
    return segments


def patch_int8_segments(logger):
    """Size the int8 split-KV scratch buffers for more segments so the decode grid fills the SMs.

    vLLM keeps NUM_PAR_SOFTMAX_SEGMENTS = 16 for every dtype, and the metadata builder reads that global once in
    __init__ to size the per-segment scratch (rows x heads x segments x head size). Raising it gives the decode
    grid num_q_blocks x num_kv_heads x segments programs instead of x16; on the 27B shape (4 KV heads, 24 query
    heads, head 256) 16 segments leave most of the SMs idle. VLLM_QUANTPACK_INT8_SEGMENTS sets the number
    directly ("auto" takes it from the SM count, the same rule the int4 decode uses).
    """
    segments = _segments_from_env(logger)
    if not segments:
        return False
    try:
        from vllm.v1.attention.backends import triton_attn
        builder = triton_attn.TritonAttentionMetadataBuilder
        stock_init = builder.__init__
        triton_attn.NUM_PAR_SOFTMAX_SEGMENTS     # the anchor this relies on
    except Exception as e:
        logger.info("vllm-quantpack: int8 split-KV segments left stock (%s)", e)
        return False
    if getattr(stock_init, "_quantpack_int8_segments", False):
        return True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.cache_config.cache_dtype not in PTH_DTYPES:
            return stock_init(self, kv_cache_spec, layer_names, vllm_config, device)
        saved = triton_attn.NUM_PAR_SOFTMAX_SEGMENTS
        triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = segments   # read once, in __init__, to size the buffers
        try:
            stock_init(self, kv_cache_spec, layer_names, vllm_config, device)
        finally:
            triton_attn.NUM_PAR_SOFTMAX_SEGMENTS = saved

    __init__._quantpack_int8_segments = True
    builder.__init__ = __init__
    logger.info("vllm-quantpack: int8_per_token_head split-KV decode uses %d segments", segments)
    return True
