# What a serving engine owns

> SGLang owns the time between a request and its tokens: it decides which prompt tokens to compute, which to reuse, and when each request runs.

You downloaded an open model, wrote a loop that calls `generate`, and put it behind an HTTP endpoint. One user gets answers fast. Ten users wait in a line, the GPU sits idle between requests, and every request recomputes the same system prompt. A serving engine exists to fix exactly that loop.

When you finish this section, you can name the two phases of every request and the four numbers a server is judged by. You can also say what SGLang controls and what it leaves to others.

## Two phases with two different bottlenecks

**Prefill:** the forward pass over the whole prompt. It computes the keys and values of every prompt token in one batch of matrix multiplications, and it produces the first output token. **Decode:** the loop that produces each next token, one forward pass per token per request.

The two phases stress different parts of the GPU. Prefill has many tokens per pass, so it keeps the arithmetic units busy. Decode has one new token per request per pass. Each step reads all the weights and cached keys and values to do little arithmetic. Decode speed is set by memory bandwidth, and prefill speed is set by compute.

We measured the difference on one Mac with a 0.6B model. A 512-token prompt reached its first token in a median of 163 ms. After that, each new token took 4.38 ms on average.

```listing
title: one request at a time, 512 tokens in and 128 tokens out
source: capture/out/bench-random-c1.txt
lang: text
note: Cut from the benchmark summary. The latency percentiles other than the median and the inter-token block are cut.
---
Max request concurrency:                 1         
Successful requests:                     16        
…
Median TTFT (ms):                        162.98    
…
Mean TPOT (ms):                          4.38      
```

That is about 0.32 ms per prompt token against 4.38 ms per output token, a factor of about 14. Every design choice in this manual follows from that gap. Prefill work can be shared and skipped. Decode work can only be batched, so that one read of the weights serves many requests.

```figure
id: fig-1-1
kind: comparison
title: prefill against decode on the capture Mac
claim: One prompt token costs about 0.32 ms in prefill, and one output token costs 4.38 ms in decode.
caption: Read left to right. Both columns come from one benchmark run with one request at a time. From capture/out/bench-random-c1.txt.
```

## The four numbers

A server has to answer four questions, and a report that answers one of them answers nothing.

- **Time to first token (TTFT):** how long a user waits before the first word. Prefill and queueing set it.
- **Time per output token (TPOT):** the average gap between tokens after the first. **Inter-token latency (ITL)** is the same gap measured one token at a time, so its tail shows stalls. Decode sets both.
- **Throughput:** output tokens per second across all users. Batching sets it.
- **Goodput:** requests per second that meet a TTFT target and a TPOT target at the same time. It is the number your users feel.

The first three trade against each other. A larger batch raises throughput and makes each user's tokens slower. A large prefill chunk finishes a long prompt sooner and stalls every decode in the batch while it runs. [Section 5.1](#s-sglang-benchmark-serving) shows the trade on a real sweep, where more users bought no extra throughput and five times the wait.

SGLang's benchmark tool prints TTFT, TPOT, ITL, and throughput. It does not print goodput at v0.5.22, so you compute it from the per-request records yourself. The [inference metrics lesson](phases/17-infrastructure-and-production/08-inference-metrics-goodput) builds that calculation.

## What SGLang owns, and what it does not

SGLang is three processes and a cache around a model. The tokenizer manager turns text into token ids. The scheduler decides which requests run in each forward pass, keeps the KV cache, and finds prompt prefixes it already computed. The detokenizer turns token ids back into text and streams it out. [Section 2.1](#s-sglang-serve-and-the-three-processes) follows a request through all three.

```figure
id: fig-1-2
kind: layers
title: what SGLang owns
claim: SGLang owns tokenizing, scheduling, the KV cache, and the forward pass, while the model and the operator own quality, memory, and routing.
caption: Read top to bottom, from the client to the weights. Grey layers are outside SGLang. From the v0.5.22 source and capture/out/server-default.log.
```

Three things stay outside. The model decides what the answer says. In our capture, a 0.6B model asked to name a planet answered "Ardens", which is no planet. SGLang served that answer correctly and fast, and no serving engine can make a model right.

The operator decides how much memory SGLang gets and which machine a request goes to. SGLang provides a separate router, the model gateway, for that second job, and [section 6.4](#s-sgl-model-gateway) covers it.

## Where SGLang wins and where it loses

Every engine posts a benchmark where it wins. We read the reports by people outside the SGLang team, and they agree on a shape that follows from the mechanisms above.

**SGLang wins when prompts repeat.** A field report ran SGLang 0.5.6 and an 8B model on two H100 GPUs {{field Rawlinson 2025-12}}. With one user, TTFT was 583 ms against 2,141 ms for vLLM 0.11. That is the radix cache from [section 3.3](#s-radixcache) skipping a prefill it already did.

**SGLang ties on unique prompts and dense models.** A field report ran SGLang 0.5.9 and Llama 3.3 70B in FP8 on one H100 {{field Spheron 2026}}. At 50 users it found 1,920 output tokens per second against 1,850. Both engines now run the same kernel families, so a 4% gap is noise for most teams.

**SGLang loses TTFT on long prompts with its defaults.** A field report on one H100 with 16K-token prompts measured a mean TTFT of 10.8 s against 1.9 s at 60 users {{field Jarvislabs 2026-05}}. The same run gave SGLang the best TPOT, 63 to 81 ms against 97 to 120 ms. SGLang splits a long prompt into chunks so that decodes keep moving, and it pays for that in first-token time.

**SGLang is the reference engine for large mixture-of-experts models.** SemiAnalysis made it the default engine for DeepSeek models in InferenceMAX {{field SemiAnalysis 2025-10}}. The first audited MLPerf result with SGLang ran DeepSeek-R1 on 512 MI355X GPUs {{field Crusoe 2026-09}}. [Part 6](#s-tp-size-dp-size-and-attn-dp-size) explains the parallelism that makes that work.

## Our position

Choose SGLang for the shape of your traffic, never for a headline number. Agents, chat with a fixed system prompt, and retrieval with a fixed preamble all share long prefixes. For that traffic, SGLang's cache pays you back on every request. If every prompt is unique and long, measure both engines on your own prompts before you commit.

Distrust any SGLang benchmark that does not name its version. The most repeated number on the web, "SGLang 29% faster than vLLM", comes from one offline test whose page lists SGLang v0.2.3 next to vLLM 0.11 {{field AIMultiple 2026-01}}. At least five other posts copy it under other names. [Section 5.3](#s-reading-other-peoples-benchmarks) shows how to read a benchmark before you believe it.

```takeaways
- Report TTFT, TPOT, throughput, and goodput together, at a stated concurrency.
- Expect prefill to cost a fraction of a millisecond per token and decode several milliseconds.
- Pick SGLang when your prompts share prefixes, and measure it when they do not.
- Check the SGLang version before you trust a benchmark number.
```

Sources: capture/out/bench-random-c1.txt, capture/out/generate.txt, capture/out/server-default.log; python/sglang/benchmark/serving.py and python/sglang/srt/entrypoints/http_server.py at v0.5.22; Rawlinson, "vLLM vs SGLang: Performance Benchmark on Dual H100 GPUs" (https://rawlinson.ca/articles/vllm-vs-sglang-performance-benchmark-h100, 2025-12-05); Spheron, "vLLM vs TensorRT-LLM vs SGLang" (https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/); Jarvislabs, "SGLang vs vLLM: H100 Benchmarks" (https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison, 2026-05-18); SemiAnalysis, "InferenceMAX" (https://newsletter.semianalysis.com/p/inferencemax-open-source-inference, 2025-10-09); Crusoe, MLPerf Inference v6.1 (https://www.crusoe.ai/resources/blog/serving-5-75-million-tokens-per-second-crusoes-mlperf-inference-v6-1-results-on-amd-mi355x, 2026-09-16); AIMultiple, "LLM Inference Engines" (https://aimultiple.com/inference-engines, 2026-01-25).
