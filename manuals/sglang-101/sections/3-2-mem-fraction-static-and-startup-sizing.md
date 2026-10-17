# --mem-fraction-static and startup sizing

> SGLang sizes the KV pool once at startup from a snapshot of free memory, so a machine that changes after startup can kill the whole server.

Our first SGLang server on the capture Mac ran for four minutes. Then a benchmark at 32 concurrent requests crashed it with a Metal out-of-memory error, and every process of the server exited. The KV pool at that moment was 95 percent empty. The memory had gone to the pool, and the next prefill had nowhere to run.

When you finish this section, you can read how SGLang divides memory at startup and predict when that division fails. You can also set `--mem-fraction-static` on purpose.

## The crash we recorded

We started the server twice before the capture kit existed, with the same command and no `--mem-fraction-static`. The two startups sized the pool very differently.

```listing
title: two startups, two pool sizes
source: capture/out/oom-exploratory.log
lang: text
note: Lines from two exploratory runs on the same Mac, before the capture kit. The comment lines are ours.
---
# run 1 startup
[2026-10-16 13:30:31] Auto-sized attention KV pool: sys_available=27.81 GB, mlx_limit=51.8 GB, mlx_used=0.31 GB, kv_budget=24.25 GB, bytes_per_slot=114688, pool_size=227022
# run 2 startup
'mem_fraction_static': 0.872
[2026-10-16 13:38:58] Auto-sized attention KV pool: sys_available=38.29 GB, mlx_limit=51.8 GB, mlx_used=0.31 GB, kv_budget=33.39 GB, bytes_per_slot=114688, pool_size=312581
```

Nothing changed in SGLang between the runs. Other programs on the Mac used less memory at the second start, so `sys_available` grew from 27.81 GB to 38.29 GB. The pool grew with it, from 24.25 GB to 33.39 GB. Both budgets are 87.2 percent of the free memory, because the automatic `mem_fraction_static` resolved to 0.872.

```listing
title: the crash at concurrency 32
source: capture/out/oom-exploratory.log
lang: text
note: Lines from run 2. The traceback above the RuntimeError line is cut.
---
[2026-10-16 13:42:53] Decode batch, #running-req: 32, #token: 16095, token usage: 0.05, cuda graph: False, gen throughput (token/s): 564.55, #queue-req: 0
[2026-10-16 13:42:55] Prefill batch, #new-seq: 1, #new-token: 511, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 9, #pending-token: 0, cuda graph: False, input throughput (token/s): 66.85
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).

[2026-10-16 13:42:57] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
…
[2026-10-16 13:43:02] kill_process_tree called: parent_pid=81559, include_parent=True, pid=81559
```

Read `token usage: 0.05` first. Thirty-two requests held 16,095 tokens, which is 5 percent of a 312,581-token pool. The pool reserved 34 GB and used under 2 GB of it. The activations of the next forward pass needed memory outside the pool, and the operating system had none left. One failed command buffer took down the scheduler, and the server killed its whole process tree.

```figure
id: fig-3-2
kind: comparison
title: the same Mac, two memory splits
claim: A pool sized at 87 percent of free memory left too little for activations, and a fraction of 0.4 kept the server alive.
caption: Read each bar left to right. The green part is the KV pool, and the grey part is everything else. From capture/out/oom-exploratory.log and capture/out/server-default.log.
```

## How SGLang decides the split

**--mem-fraction-static:** the share of memory that SGLang gives to the model weights plus the KV pool. Everything outside that share is for activations, CUDA graphs, and other programs. The flag defaults to `None`, and SGLang computes a value at startup {{code}}.

On a GPU, the automatic value comes from a reserve. SGLang starts with 512 MB and adds 1.5 MB for each token of the prefill chunk {{code}}. Then it adds more for tensor and pipeline parallelism and for CUDA graphs. On a GPU with more than 60 GB, the reserve is at least 10 GiB. The fraction is then the GPU memory minus the reserve, divided by the GPU memory. When SGLang cannot read the device capacity, it uses 0.95.

After the weights load, SGLang measures free memory again. The KV budget is the free memory minus the `1 - fraction` share of the memory that was free before loading {{code}}. The pool holds that budget divided by the bytes per token from [section 3.1](#s-kv-bytes-per-token-and-the-kv-pool). If the budget is not positive, startup stops and prints the smallest fraction that would work.

The MLX backend on Apple silicon uses a simpler rule {{code}}. It takes the smaller of two numbers. The first is the fraction of the Metal working set, minus memory already in use. The second is the fraction of system memory available right now. The second term is the snapshot that changed between our two runs.

One public tuning write-up describes the fraction the other way round, as weights plus buffers, with the KV cache in the remainder. The code says the opposite: the static share holds the weights and the KV pool.

## The default assumes the machine belongs to SGLang

The automatic rule is reasonable on a dedicated GPU. Nothing else allocates memory there, and the reserve covers the activations of one prefill chunk and the CUDA graphs. The rule fails in three common cases.

- **A shared machine.** A laptop, a workstation, or a node with other jobs. The snapshot at startup does not hold for the life of the server.
- **Unknown capacity.** On a device where SGLang cannot read the memory size, the fraction becomes 0.95 and leaves 5 percent for activations. Issue #39675 reports exactly that failure on Apple MPS, an out-of-memory error on the first prefill {{issue #39675}}.
- **Extra memory the estimate does not know about.** Issue #42510 shows draft-model copies that stay in memory while the target pool is sized, so the draft pool allocation fails {{issue #42510}}. Issue #6027 shows a 64K-token DeepSeek-V3 prompt that ran out of memory in attention, although the log reported room for 274,934 KV tokens {{issue #6027}}.

## Our position: set the fraction yourself on any shared machine

On a machine that runs anything besides SGLang, pass `--mem-fraction-static` yourself. The capture kit uses 0.4 on the Mac. With that value, the same benchmark that crashed run 2 finished all 128 requests at concurrency 32.

```listing
title: the kit's server, with an explicit fraction
source: capture/out/server-default.log
lang: text
note: One startup line from the default kit configuration.
---
[2026-10-16 13:58:03] Auto-sized attention KV pool: sys_available=48.49 GB, mlx_limit=51.8 GB, mlx_used=0.31 GB, kv_budget=19.39 GB, bytes_per_slot=114688, pool_size=181574
```

The budget is now 19.39 GB, which is 0.4 of the 48.49 GB snapshot. The pool still holds 181,574 tokens, far more than 32 requests of 640 tokens need. A smaller pool cost nothing in this workload, because the workload never filled the pool.

On a dedicated GPU, trust the automatic value first. Raise the fraction in steps of 0.02 only when the scheduler log shows a full pool and retractions. Find the upper limit in staging with your longest prompts, then run production one or two steps below it. Lower it when long prompts fail in attention, as in issue #6027.

The fraction also changes comparisons. One independent benchmark ran SGLang with its automatic fraction and vLLM with its default of 0.9 of GPU memory {{field Rawlinson 2025-12}}. The two engines had KV pools of very different sizes. Read [section 5.3](#s-reading-other-peoples-benchmarks) before you trust a comparison that does not state both values.

## What to measure

- The `max_total_num_tokens` line at startup. It is the result of the whole calculation.
- `token usage` in decode lines. A value near 1.0 means the pool is full.
- The free memory of the machine during the heaviest prefill, from the operating system.

One more caution for Apple silicon. The `kv_cache_memory_usage_gb` metric read 0.0 in our capture, because the MLX backend allocates its own pool outside the accounting that the metric reads. Use the `MlxAttentionKVPool` log line, which reported about 19,860 MB.

```takeaways
- Pass --mem-fraction-static yourself on any machine that runs other programs.
- Read kv_budget and max_total_num_tokens in the startup log after every change.
- Raise the fraction in small steps only when the pool is full and requests retract.
- Lower the fraction when long prompts fail with out-of-memory errors in attention.
- Compare engines only when both report their KV pool sizes.
```

Sources: python/sglang/srt/arg_groups/memory_hook.py (handle_gpu_memory_settings), python/sglang/srt/mem_cache/kv_cache_configurator.py (_profile_available_bytes), python/sglang/srt/hardware_backend/mlx/model_runner.py (_compute_pool_size), python/sglang/srt/arg_groups/fields/schedule.py (mem_fraction_static) at v0.5.22, capture/out/oom-exploratory.log, capture/out/server-default.log, capture/out/metrics.txt, issues #39675, #42510, and #6027, the Rawlinson benchmark of 2025-12-05.
