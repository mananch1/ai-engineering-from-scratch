# sglang.benchmark.serving

> One benchmark run measures one concurrency, so you sweep concurrency until extra load stops adding throughput and only adds waiting.

A teammate posts "SGLang does 255 tokens per second on my laptop" in the team channel. The number is true. It came from our capture. It is also useless on its own. The same server gave that throughput with a median first token of 0.8 s at one load and 3.9 s at another. A throughput number without the latency that came with it describes nothing a user feels.

When you finish this section, you can run `sglang.benchmark.serving` with settings you can defend and read every line of its report. You can find the knee where more concurrency stops paying. You can also say which of your own numbers will transfer to a GPU and which will not.

## The command we ran

The server is one process with the model `mlx-community/Qwen3-0.6B-4bit` on the capture Mac. Every run in this section talks to this one server.

```listing
title: the server under test
source: capture/out/server-default.command.txt
lang: text
note: The full command, as the capture kit started it.
---
SGLANG_USE_MLX=1 sglang serve --model-path mlx-community/Qwen3-0.6B-4bit --disable-cuda-graph --host 127.0.0.1 --port 30000 --enable-cache-report --enable-metrics --mem-fraction-static 0.4
```

Two flags in that command need a note. SGLang's Apple silicon guide still passes `--disable-cuda-graph`, and the v0.5.22 help marks it deprecated in favor of `--cuda-graph-backend-{decode,prefill}=disabled`. On the MLX backend, decode is greedy unless you add `--mlx-enable-sampling`, so our runs measured the greedy path. The benchmark sends temperature 0 by default, so the client and the server agree here.

The client is the benchmark module that comes with SGLang. We ran it three times, with a concurrency cap of 1, 8, and 32. The run at 8 looks like this:

```bash
python -m sglang.benchmark.serving --backend sglang --port 30000 \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --random-range-ratio 1.0 --num-prompts 32 --max-concurrency 8 --seed 7
```

**The path changed.** The benchmark tools moved into the `sglang.benchmark` package in 2026-06. `python -m sglang.bench_serving` still runs as a deprecated alias that imports the new module. `python -m sglang.bench_one_batch` and `python -m sglang.bench_offline_throughput` do not run at v0.5.22. [PR #40795](https://github.com/sgl-project/sglang/pull/40795) deleted both aliases on 2026-09-22 {{code}}. If a tutorial uses those two paths, it was written for an older SGLang, and its other flags are suspect too.

## Four defaults that change the result

The client makes choices before it measures anything. Each one changes the number you report, and most published benchmarks do not say which ones they kept.

| Setting | Default at v0.5.22 | What it does to the result |
|---|---|---|
| `--random-range-ratio` | 0.0 | Each length is drawn from 1 to the length you asked for, so the average prompt is about half of `--random-input-len` |
| `ignore_eos` | on | The server ignores the end token, so every request generates exactly `--random-output-len` tokens |
| `--request-rate` | `inf` | All requests are sent at time 0, and only `--max-concurrency` holds them back |
| `--warmup-requests` | 1 | One request runs before the clock starts |

The first row is the trap. With the default ratio, `--random-input-len 512` gives an average prompt near 256 tokens. Your report then says 512 and describes half the work. We set the ratio to 1.0, and the report confirms it. 32 prompts of 512 tokens give `Total input tokens: 16384`. With `ignore_eos`, the run gives `Total generated tokens: 4096`, exactly 32 times 128.

**Our position:** pass `--random-range-ratio 1.0` every time, or write the length distribution next to the result. A benchmark that hides its lengths cannot be compared with anything.

## Read the report, block by block

```listing
title: the report at a concurrency cap of 8
source: capture/out/bench-random-c8.txt
lang: text
note: Cut from the full output. The argument dump, the progress bar, and the P90 and P95 rows are cut.
---
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 8
Successful requests:                     32
Benchmark duration (s):                  16.02
Total input tokens:                      16384
Total generated tokens:                  4096
Request throughput (req/s):              2.00
Output token throughput (tok/s):         255.63
Peak output token throughput (tok/s):    440.00
Peak concurrent requests:                16
Total token throughput (tok/s):          1278.13
Concurrency:                             7.96
…
Median TTFT (ms):                        785.79
P99 TTFT (ms):                           2180.87
…
Mean TPOT (ms):                          23.02
Median TPOT (ms):                        22.52
…
Median ITL (ms):                         20.35
P99 ITL (ms):                            78.79
Max ITL (ms):                            1948.37
```

The function `calculate_metrics` in the benchmark module computes every line. Five of them are easy to misread.

**Concurrency:** the sum of every request's end-to-end latency divided by the run duration. It is the average number of requests in progress, measured after the fact. Here it is 7.96 against a cap of 8. It tells you the cap was reached, and nothing more.

**Peak concurrent requests:** the run is cut into one-second buckets, and each request counts in every bucket that its interval touches. The line reports the largest bucket count. Two requests that run one after the other inside the same second both count, so the number can exceed the cap. That is why the report says 16 at a cap of 8, and 64 at a cap of 32.

**TTFT:** the time from sending the request to receiving the first streamed chunk. It contains queueing, prefill, and the network.

**TPOT:** for each request, the time after the first token divided by the remaining tokens, `(latency - ttft) / (output_len - 1)`. The report gives the mean and percentiles over requests. **ITL:** every gap between two streamed chunks, pooled across all requests. A chunk with several tokens counts as one gap.

**Max ITL:** the single longest gap, here 1,948 ms. One request waited almost two seconds between two tokens while new prompts ran their prefill. The median ITL of 20 ms hides that stall. The P99 of 79 ms shows part of it.

```figure
id: fig-5-1b
kind: structure
title: what each line of the report is computed from
claim: Concurrency and peak concurrent requests come from request timestamps after the run, so neither line repeats the cap you set.
caption: Read each row left to right, from the report line to the formula in calculate_metrics to our value at a cap of 8. Olive boxes are client measurements. From capture/out/bench-random-c8.txt and the benchmark module at v0.5.22.
```

## The sweep and the knee

One run tells you one point. Three runs at caps of 1, 8, and 32 tell you the shape.

| Cap | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | Concurrency |
|---|---|---|---|---|---|
| 1 | 175.15 | 162.98 ms | 270.18 ms | 4.38 ms | 0.99 |
| 8 | 255.63 | 785.79 ms | 2180.87 ms | 22.52 ms | 7.96 |
| 32 | 252.74 | 3917.54 ms | 8262.86 ms | 93.61 ms | 31.94 |

**We measured** this on the capture Mac. From 1 to 8, output throughput rises 46 percent and the median first token gets 4.8 times slower. From 8 to 32, throughput falls about 1 percent. The median first token gets 5 times slower, and the median time per token gets 4.2 times slower.

```figure
id: fig-5-1
kind: comparison
title: throughput stops at a cap of 8, latency does not
claim: Raising the cap from 8 to 32 added no throughput and made the median first token five times slower.
caption: Left, output tokens per second at each cap. Right, median TTFT at the same caps on a shared scale. Olive bars are client measurements, and rose marks the run past the knee. From capture/out/bench-random-c1.txt, c8.txt, and c32.txt.
```

The knee is the cap after which throughput stops rising. On this machine it is at or below 8. Past the knee, the server has no idle compute left to give. New requests only wait for a share of the same decode steps. The scheduler in [section 2.2](#s-the-scheduler-loop-and-schedule-policy) accepts them anyway, because its admission limit on this Mac was `max_running_requests=4096`.

**What transfers to a GPU.** The absolute numbers do not. A 0.6B model on Apple silicon says nothing about an H100. The shape does transfer. Every server has a knee, the latency past it grows faster than the throughput, and the default admission limit sits far above it.

## Our position

Report the highest concurrency that still meets your latency target, and report its throughput together with p50 and p99 TTFT and TPOT. "255 tok/s at a cap of 8, with a median first token of 0.79 s" is a result. "255 tok/s" is a slogan.

Then enforce the knee outside the engine. The engine's own limit is set from memory, and memory runs out long after latency is unacceptable. Put the cap in your client, your gateway, or `--max-running-requests`, and size it from the sweep.

## The incident: a hand-written script

[Issue #3196](https://github.com/sgl-project/sglang/issues/3196) reported 13 tok/s for DeepSeek V3 on eight H200 GPUs, against 60 tok/s from the DeepSeek API {{issue #3196}}. Another user reached 33 tok/s on two H100 nodes. The measuring script counted words split on spaces and included the first-token time in the rate. A word is not a token, and TTFT is not decode time, so the script reported less than the server did. The fix was to measure with the benchmark client, which counts tokens and keeps TTFT and TPOT apart.

```takeaways
- Run `python -m sglang.benchmark.serving`, never a hand-written timing loop.
- Pass `--random-range-ratio 1.0` or state the length distribution you used.
- Sweep at least three concurrency caps and plot throughput against median TTFT.
- Report throughput only together with p50 and p99 TTFT and TPOT at the same cap.
- Read Concurrency as an achieved average and Peak concurrent requests as a per-second count.
- Enforce the knee in your client or gateway, because the engine admits far more.
```

Sources: python/sglang/benchmark/serving.py (calculate_metrics, argument defaults), python/sglang/benchmark/datasets/common.py (compute_random_lens), python/sglang/bench_serving.py, PR #40795, all at v0.5.22 (baa9befa); capture/out/serve-help.txt, capture/out/server-default.command.txt, capture/out/server-default.log, capture/out/bench-random-c1.txt, capture/out/bench-random-c8.txt, capture/out/bench-random-c32.txt; issue #3196.
