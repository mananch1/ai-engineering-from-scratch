# --quantization and --kv-cache-dtype

> Quantization buys memory first and speed second, and SGLang keeps weights, GEMM kernels, and the KV cache as three separate choices so that you can buy only what you need.

Cloudflare served Kimi K2.6 with a BF16 KV cache and raised concurrency step by step. At 32 concurrent requests the server produced 1,558 tok/s. At 64 it ran out of memory. With an FP8 KV cache the same service reached 2,192 tok/s at 64 requests, about 41% above the best BF16 result {{field Cloudflare 2026-08}}. The FP8 cache was slower per token at every concurrency both versions could run. It won because twice as many requests fit.

When you finish this section, you can tell weight quantization from KV cache quantization, and load a quantized checkpoint without breaking detection. You can also size the memory gain before you measure speed, and decide where a lower precision belongs.

## Three choices, three flags

**Weight quantization:** storing the model weights in fewer bits, such as FP8, FP4, or INT4. It shrinks the weights and the memory traffic of each GEMM. `--quantization` selects the method.

**GEMM backend:** the kernel library that multiplies quantized weights. `--fp8-gemm-backend` and `--fp4-gemm-backend` select it, and both default to `auto`, which picks by GPU generation.

**KV cache dtype:** the format of the cached keys and values. `--kv-cache-dtype` selects it, separately from the weights. Its default `auto` uses the model dtype.

A model can run FP8 weights with a BF16 KV cache, or BF16 weights with an FP8 KV cache. Each choice moves a different budget, so treat them as three decisions.

## Weights: offline first

At v0.5.22, `--quantization` accepts 31 method names. Most of them describe a checkpoint that someone already quantized: AWQ, GPTQ, ModelOpt FP8 and FP4, compressed-tensors, GGUF, Quark, and others. A few quantize BF16 weights while the model loads: `fp8`, `mxfp8`, `nvfp4_online`, and `mlx_q4` or `mlx_q8` on Apple silicon.

**Offline quantization:** the checkpoint already stores low-bit weights and their scales. **Online quantization:** SGLang converts full-precision weights at load time.

When you load a quantized checkpoint, do not pass `--quantization`. The model config check reads `quantization_config.quant_method` from the checkpoint and uses it. If you pass a method that differs, the check accepts only known compatible pairs, such as `modelopt_fp4` on a `modelopt` checkpoint. For a requantization it logs that you should validate accuracy again. Every other mismatch raises at startup.

```figure
id: fig-4-4
kind: flow
title: how SGLang decides the weight format
claim: The checkpoint names its own quantization method, and a --quantization flag only helps when it matches that method or a known upgrade.
caption: Read each row left to right. The rose arrow ends in a startup error. The last row is the CUDA path at v0.5.22. From _verify_quantization and the GPTQ Marlin override at v0.5.22.
```

The flag can also block an upgrade. A GPTQ checkpoint loaded with no flag is detected, converted at runtime, and served by the Marlin kernel. The upgrade code accepts that conversion only when you passed no method or `gptq_marlin`.

PR #32114 deleted the plain GPTQ CUDA kernel on 2026-09-10, and on a CUDA build the method registry has no `gptq` entry. The flag parser still accepts `--quantization gptq`. The model config check then skips the Marlin upgrade and raises `Unknown quantization method: gptq`. We read this path in the code and did not run it on a GPU.

## The KV cache: capacity, with a scale

`--kv-cache-dtype` accepts `auto`, `bf16`, `fp8_e4m3`, `fp8_e5m2`, `mxfp8` (FA4 only), `nvfp4`, and `fp4_mx_block16`. FP8 halves the bytes per token and FP4 quarters them, so the same pool holds two or four times as many tokens. [Section 3.1](#s-kv-bytes-per-token-and-the-kv-pool) shows how to compute bytes per token for your model.

FP8 needs a scale for each tensor. The help text of `--quantization-param-path` says the file "should generally be supplied, when KV cache dtype is FP8" and that without it the scales "default to 1.0, which may cause accuracy issues" {{code}}. Run an accuracy check whenever you change the KV cache dtype.

Cloudflare published its check. On Kimi K2.6, BF16 against FP8 KV scored 94.24 and 94.09 on GSM8K, 89.11 and 89.04 on MMLU, and 80.29 and 79.29 on MMLU-Pro {{field Cloudflare 2026-08}}. Those are their numbers on their model, and your model can differ.

## Quantization is a capacity decision

```figure
id: fig-4-5
kind: comparison
title: BF16 against FP8 KV cache on Kimi K2.6
claim: The FP8 KV cache is a few percent slower at every load both caches can run, and keeps scaling after the BF16 cache runs out of memory.
caption: Output tokens per second by concurrency, as Cloudflare reported them on 2026-08-03. Dashed bars are BF16 and solid bars are FP8. Rose marks the out-of-memory run. No SGLang version was published.
```

Cloudflare also tested INT4 weights on GLM 5.2 {{field Cloudflare 2026-08}}. Decode got faster, 60 to 92 tok/s at concurrency 1 and 1,672 to 1,933 tok/s at 64. Prefill got slower, from about 10,160 tok/s with FP8 to about 8,660 with INT4.

Decode is limited by memory traffic, so smaller weights help. Prefill is limited by compute, and dequantizing INT4 costs compute. Cloudflare split the two phases across servers and runs INT4 for decode and FP8 for prefill. [Section 6.3](#s-prefill-decode-disaggregation) explains the split.

## Our position

Use an offline-quantized checkpoint and pass no `--quantization` flag. The docs say the same: "offline quantization is recommended over online quantization" {{docs}}. Online quantization adds load time and gives you a checkpoint that nobody evaluated.

Choose a KV cache dtype from memory, not from speed. If your server queues requests because the pool is full, an FP8 cache with a scale file is the cheapest capacity you can buy. If the pool has room, keep the model dtype.

Quantize per phase when you run prefill and decode on separate servers. Smaller weights help decode, and they can slow prefill.

## Incidents

**The BF16 cache ran out at 64 requests.** Cloudflare {{field Cloudflare 2026-08}} reported that Kimi K2.6 with a BF16 KV cache ran out of memory at concurrency 64. The FP8 cache held about 1.37 million tokens against about 686,000 and served the same load.

**A container could not run NVFP4.** A field report from joshua8.ai {{field joshua8 2026-01}} used an RTX PRO 6000 Blackwell. NVFP4 did not run in its SGLang container, so it compared SGLang on GPTQ-INT4 against vLLM on NVFP4. Compare engines on the same weight format, or the result measures the format.

**FP4 needs one GPU generation.** A user summary on Pivot News {{field Pivot News 2026-05}} reported that SGLang crashed when RTX 4090 cards joined a Blackwell pipeline. Its FP4 path required compute capability 10.0 and had no software path for older cards. This report reached us through a secondary summary of a Reddit post.

```takeaways
- Load quantized checkpoints without --quantization and let SGLang read the method.
- Replace --quantization gptq with no flag or gptq_marlin on CUDA builds.
- Supply a KV scale file and run an accuracy check before serving an FP8 KV cache.
- Choose KV cache dtype by queue length and pool usage, then measure speed.
```

Sources: python/sglang/srt/arg_groups/choices.py (QUANTIZATION_CHOICES), arg_groups/fields/model.py (kv_cache_dtype, quantization_param_path), configs/model_config.py (_verify_quantization), layers/quantization/__init__.py (get_quantization_config), layers/quantization/gptq/gptq.py (override_quantization_method), layers/quantization/fp8.py, and docs/docs/advanced_features/quantization.mdx at v0.5.22; PR #32114; field reports by Cloudflare (2026-08-03), joshua8.ai (2026-01-10), and Pivot News (2026-05-19).
