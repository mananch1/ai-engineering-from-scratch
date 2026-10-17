# One request, end to end

> One `POST /generate` with a five-token prompt crossed three processes, reused two cached tokens, and returned twelve tokens in 950 ms.

The fastest way to understand a serving engine is to send it one small request and read everything it writes about it. Most people skip this. They start the server, see "ready", and point a load test at it. Then the first surprise in production is a log line they never read.

When you finish this section, you can start SGLang on a laptop and send one request through each API. You can explain every timing and token count in the response.

## Starting the server

We ran SGLang 0.5.22 on Apple silicon through its MLX backend, with a 4-bit Qwen3-0.6B. This is the command every capture in this manual starts from.

```listing
title: the server command
source: capture/out/server-default.command.txt
lang: text
---
SGLANG_USE_MLX=1 sglang serve --model-path mlx-community/Qwen3-0.6B-4bit --disable-cuda-graph --host 127.0.0.1 --port 30000 --enable-cache-report --enable-metrics --mem-fraction-static 0.4
```

`sglang serve` is the entry point that SGLang recommends at v0.5.22. The older `python -m sglang.launch_server` still works and prints a warning. `--disable-cuda-graph` is in the Apple silicon instructions and does nothing useful on a Mac, and the server says it is deprecated. `--mem-fraction-static 0.4` is our choice, and [section 3.2](#s-mem-fraction-static-and-startup-sizing) shows the crash that made us set it.

Before the first request, the server sizes its memory and builds its cache. Read these four lines on every start.

```listing
title: the startup lines that matter
source: capture/out/server-default.log
lang: text
note: Four lines cut from the startup log. Download, import, and warning lines between them are cut.
---
[2026-10-16 13:58:03] Auto-sized attention KV pool: sys_available=48.49 GB, mlx_limit=51.8 GB, mlx_used=0.31 GB, kv_budget=19.39 GB, bytes_per_slot=114688, pool_size=181574
…
[2026-10-16 13:58:05] max_total_num_tokens=181574, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=4096, context_len=40960, available_gpu_mem=47.96 GB
…
[2026-10-16 13:58:06] Tree cache initialized: source=default impl=UnifiedRadixCache hybrid_swa=False hybrid_ssm=False hicache_attached=False
…
[2026-10-16 13:58:08] The server is fired up and ready to roll!
```

**KV pool:** the memory that holds the keys and values of every token the server keeps. It has room for 181,574 tokens here, at 114,688 bytes each. [Section 3.1](#s-kv-bytes-per-token-and-the-kv-pool) derives that number from the model's shape. **max_running_requests:** the most requests the scheduler runs at once. The server computed 4096 from the pool size. Nobody set it.

The last line is printed after the server sends itself a request. SGLang posts "The capital city of France is" to its own `/generate` endpoint, then calls `/freeze_gc`. That freezes the Python garbage collector in all three processes, so the objects created at startup are never scanned again. Keep that detail in mind. It returns in the field report at the end of this section.

## The request

We sent the native API a five-token prompt and asked for twelve tokens at temperature zero.

```listing
title: the response metadata
source: capture/out/generate.txt
lang: json
note: Cut from the /generate response. The output ids, the weight version blocks, and the storage and timestamp fields are cut.
---
"meta_info": {
  …
  "finish_reason": {
    "type": "length",
    "length": 12
  },
  "prompt_tokens": 5,
  …
  "queue_time": 0.0002437499351799488,
  …
  "completion_tokens": 12,
  "cached_tokens": 2,
  …
  "e2e_latency": 0.9496850408613682,
  …
  "first_token_latency": 0.8885334208607674,
  …
  "decode_throughput": 179.8807619469758,
```

Read it field by field. The request waited 0.24 ms in the queue, because nothing else was running. Two of its five prompt tokens were already in the cache. Our prompt "The capital of France is" shares its first two tokens with the warm-up prompt "The capital city of France is". The radix cache found them and skipped them. The text "Paris, and the capital of Germany is Berlin. The capital" stopped at 12 tokens because we asked for 12, and `finish_reason` says `length`.

The scheduler logged the same request from its side.

```listing
title: the scheduler's view of the request
source: capture/out/server-default.log
lang: text
---
[2026-10-16 13:58:10] Prefill batch, #new-seq: 1, #new-token: 3, #cached-token: 2, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1.55
[2026-10-16 13:58:10] INFO:     127.0.0.1:63795 - "POST /generate HTTP/1.1" 200 OK
```

`#new-token: 3` and `#cached-token: 2` add up to the five prompt tokens. That pair is the most useful line SGLang prints. [Section 3.4](#s-prefix-friendly-prompts) uses nothing else to find a broken prompt.

```figure
id: fig-1-3
kind: sequence
title: one /generate request through the three processes
claim: The tokenizer manager, the scheduler, and the detokenizer each touched the request once, and the prefill computed 3 of 5 tokens.
caption: Read top to bottom. Numbers are from one request. From capture/out/generate.txt and capture/out/server-default.log.
```

## The first-token time is not what you think

The first token took 889 ms. The whole request took 950 ms, so the eleven tokens after the first took about 61 ms. After the warm-up of the benchmark tool, a 512-token prompt reached its first token in a median of 163 ms on the same server. A prompt that is 100 times longer started five times faster.

The prompt length did not cause the 889 ms. The first request of a new shape pays one-time costs. On MLX, the first request that reuses a cached prefix also ran slow in [section 3.4](#s-prefix-friendly-prompts). We did not find the exact cause in the log. The lesson holds on any backend. The first request after a start measures the start, and you should throw it away.

## The same question through the OpenAI API

Most clients speak the OpenAI chat API, and SGLang serves it on the same port. We asked the chat endpoint to name one planet in one word, with thinking turned off.

```listing
title: the chat completion
source: capture/out/generate.txt
lang: json
note: Cut from the /v1/chat/completions response. The id, timestamps, and metadata block are cut.
---
"message": {
  "role": "assistant",
  "content": "Ardens",
  "reasoning_content": null
},
"logprobs": null,
"finish_reason": "stop",
"matched_stop": 151645
…
"usage": {
  "prompt_tokens": 21,
  "total_tokens": 25,
  "completion_tokens": 4,
  "prompt_tokens_details": {
    "cached_tokens": 0
  },
```

A seven-word question became 21 prompt tokens, because the chat template wraps the message in role markers. That template cost is real. Count it when you budget context. The model said "Ardens", which is no planet. SGLang did its job: it ran the model the operator chose and returned its answer in 4 tokens.

`cached_tokens` appears here only because we started the server with `--enable-cache-report`. Without that flag, the OpenAI response reports no cached tokens at all, even when the scheduler reused thousands. Turn it on. It costs nothing, and it is the only way a client can see the cache.

## Our position

Read the startup log before you send traffic, and warm the server before you measure it. The startup lines tell you the pool size, the request limit, and the cache type the server chose for you. All three were computed, and [section 3.2](#s-mem-fraction-static-and-startup-sizing) shows a case where the computed value crashed the server.

A field report from LinkedIn shows why the request path deserves the same attention as the kernels {{field LinkedIn 2026-02}}. Their ranking service ran a 0.6B model on 300-token inputs at 100 queries per second. The bottlenecks they found were tokenization, ZMQ messages between the processes, Python garbage collection, the GIL, and the scheduler. A scoring path that skips decode cut P99 latency from 6,220 ms to 454 ms. Garbage collection paused the server for 100 to 300 ms every few seconds. Their fix was a warm-up followed by a frozen collector, merged as pull request 9241.

That is the freeze you saw at startup.

```takeaways
- Read the KV pool, max_running_requests, and tree cache lines on every server start.
- Send warm-up requests before you measure anything.
- Start the server with --enable-cache-report so clients can see cached_tokens.
- Check #new-token and #cached-token in the scheduler log for each request you debug.
```

Sources: capture/out/server-default.command.txt, capture/out/server-default.log, capture/out/generate.txt, capture/out/bench-random-c1.txt; python/sglang/srt/entrypoints/http_server.py (_execute_server_warmup, _freeze_gc_after_server_warmup) and python/sglang/srt/hardware_backend/mlx/model_runner.py (_compute_pool_size) at v0.5.22; LinkedIn Engineering, "Scaling LLM-based ranking systems with SGLang at LinkedIn" (https://www.linkedin.com/blog/engineering/ai/scaling-llm-based-ranking-systems-with-sglang-at-linkedin, 2026-02-20); pull request 9241 (https://github.com/sgl-project/sglang/pull/9241).
