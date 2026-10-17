# --speculative-algorithm

> Speculative decoding spends idle GPU compute to commit several tokens per step, so it cuts latency at low load and costs throughput when the GPU is already full.

A user turned on EAGLE-3 for Llama-3.1-8B on an A6000 and ran two tests. At batch 1, output went from 42.0 to 92.5 tok/s, more than twice as fast. At concurrency 20, output fell from 217.8 to 164.4 tok/s, a quarter slower. Both numbers are in issue #5274, and both are correct. A maintainer answered that speculative decoding "is for small batch sizes rather than stress tests" {{issue #5274}}.

When you finish this section, you can explain draft, verify, and acceptance length, and read the defaults SGLang picks. You can also find the request limit that speculation sets and decide where speculation belongs.

## The cost: decode is one token at a time

Each decode step commits one token per request, and the next step needs that token as input. At batch 1 a step reads every weight of the model to compute one token. The GPU has compute to spare and memory traffic is the limit. A larger batch fills that spare compute with other requests.

Speculative decoding fills it with guesses. **Draft model:** a small model, or a small head trained on the target model, that proposes the next few tokens cheaply. **Verify pass:** one forward of the target model over all proposed tokens at once. The target accepts the longest prefix that matches what it would have produced, and adds one token of its own.

**Acceptance length:** the number of tokens committed per verify pass, accepted drafts plus that one bonus token. An acceptance length of 3 means one target forward commits three tokens.

## One step in SGLang

```figure
id: fig-4-6
kind: sequence
title: one EAGLE step with steps 3, topk 1, and 4 draft tokens
claim: One EAGLE step drafts a chain of tokens, verifies them in one target forward, and commits the accepted prefix plus one bonus token.
caption: Read top to bottom. The numbers follow the default (3, 1, 4) for most models at v0.5.22. Green tokens are accepted and the rose token is rejected. From EAGLEWorkerV2.forward_batch_generation at v0.5.22.
```

1. **Draft.** The draft model runs `--speculative-num-steps` times. With `--speculative-eagle-topk` above 1, each step keeps several candidates per node and grows a tree. A tree builder keeps the best `--speculative-num-draft-tokens` nodes and builds a tree mask.
2. **Verify.** The target runs once in `TARGET_VERIFY` mode over every draft token with the tree mask. For greedy requests it keeps the longest matching path. For sampled requests it compares probabilities with temperature, top-k, and top-p applied.
3. **Draft extend.** The draft model reads the accepted tokens in `DRAFT_EXTEND_V2` mode, so its cache is ready for the next step.

The accept thresholds `--speculative-accept-threshold-single` and `--speculative-accept-threshold-acc` default to 1.0. With those values, a greedy request produces the same tokens it would produce without speculation.

## The algorithms and the defaults

`--speculative-algorithm` accepts `EAGLE`, `EAGLE3`, `NEXTN`, `STANDALONE`, `NGRAM`, `DFLASH`, `DSPARK`, and `UNO`. `NEXTN` is another name for `EAGLE`. For DeepSeek V3 and other models that include multi-token prediction layers in the checkpoint, SGLang sets the draft model path to the model path. `NGRAM` needs no draft model at all. It looks up repeated token sequences in the text it has already seen.

When you leave the shape unset, SGLang chooses `(num_steps, topk, num_draft_tokens)` from the model architecture.

| Model | Default shape |
|---|---|
| `LlamaForCausalLM`, Grok | (5, 4, 8): a tree of up to 8 drafts |
| DeepSeek, GLM, GPT-OSS, MiMo, `STANDALONE`, and every other model | (3, 1, 4): a chain of 3 drafts |

With topk 1, the code forces the draft token count to the step count plus one.

## The limit nobody asked for

The EAGLE, NGRAM, DFLASH, and DSPARK handlers at v0.5.22 each do the same thing when you did not set `--max-running-requests`. It sets the value to 48 and logs one warning: "Max running requests is reset to 48 for speculative decoding" {{code}}. Speculation needs memory for draft tokens and works best at small batch, so the cap protects the server. It also caps your benchmark.

A throughput test with speculation on and no explicit limit measures a server that admits 48 requests. If your traffic runs 200 concurrent requests, the other 152 wait in the queue, and time to first token grows. The warning prints once at startup, and it scrolls away before the first request.

## What to measure

The scheduler logs `accept len` and `accept rate` on each decode log line, and `/metrics` exports `sglang:spec_accept_length` and `sglang:spec_accept_rate`. Measure them on your own prompts, because acceptance depends on the text and on the draft. A field report by David Noel Ng {{field David Noel Ng 2026-08}} on two GH200 measured 3.5 to 5.05 tokens on DeepSeek-V4-Flash after a loader fix.

Then sweep concurrency with speculation on and off, and find where the two lines cross. Ant Group {{field Ant Group 2025-09}} measured this on DeepSeek-R1 on H20. Longer drafts gained 21% at batch 1 and lost about 18% at batch 32.

```figure
id: fig-4-7
kind: comparison
title: the same feature at two batch sizes
claim: Speculation doubled single-request speed and cut throughput at concurrency 20 on the same GPU and model.
caption: Output tok/s for Llama-3.1-8B with and without EAGLE-3 on one A6000, as reported in issue 5274 on 2025-04-11. The bars use one scale.
```

## Our position

Treat speculative decoding as a latency tool for low and middle concurrency. Turn it on for interactive traffic with a few users per replica. Leave it off for batch jobs and for servers that run at high concurrency.

Always set `--max-running-requests` yourself when speculation is on. Choose it from the concurrency where your sweep shows speculation still wins.

Use the draft layers that come with your model first. A model with multi-token prediction layers needs no extra download. For EAGLE-3, check that the draft model supports your prompt length, and train a draft on your own traffic when acceptance is low. Google Cloud {{team Google Cloud 2025-11}} and Modal with Decagon {{team Modal 2025-11}} both trained their own drafts. Modal reported 38% longer acceptance than open-source drafts. Both are vendor reports.

## Incidents

**Speculation made a stress test slower.** Issue #5274 (2025-04-11): EAGLE-3 on an A6000 lowered throughput at concurrency 20 from 217.8 to 164.4 tok/s. It raised batch-1 speed from 42.0 to 92.5 tok/s. The fix was to run speculation only at small batch.

**The draft could not read long prompts.** Issue #6783 (2025-05-30): Llama-3.3-70B FP8 with EAGLE-3 accepted 3.2 tokens at 1k input, 1.6 at 3k, and 1.2 at 10k. At 10k it produced 63.9 tok/s against 102.7 without speculation. The draft config had 2,048 positions and no rope scaling. Adding Llama 3 rope scaling and 131,072 positions to the draft config fixed it.

**A loader bug hid the expert weights.** David Noel Ng {{field David Noel Ng 2026-08}} found SGLang slower than vLLM on DeepSeek-V4-Flash. Acceptance was 1.3 to 1.8 tokens, and only a log warning pointed at the cause. The draft loader never mapped the shared expert weights into the fused slot. After a local fix, decode went from 115.7 to 176.0 tok/s, and PR #33276 fixed it upstream.

```takeaways
- Turn on speculative decoding for low-concurrency interactive traffic, and measure where it stops winning.
- Set --max-running-requests explicitly whenever --speculative-algorithm is set.
- Read accept len in the decode log on your own prompts before you trust a speedup.
- Check the draft model context length against your longest prompts.
```

Sources: python/sglang/srt/speculative/eagle_worker_v2.py (forward_batch_generation), speculative/eagle_utils.py (build_tree_kernel_efficient, eagle_sample), arg_groups/speculative_hook.py (_auto_choose_speculative_params, _handle_eagle_family, _handle_ngram, _resolve_speculative_algorithm_alias), arg_groups/fields/spec.py, managers/scheduler_components/metrics_reporter.py, and observability/metrics_collector.py at v0.5.22; issues #5274 and #6783; PR #33276; field reports by Ant Group on the LMSYS blog (2025-09-26) and David Noel Ng (2026-08-04); vendor reports by Google Cloud (2025-11-25) and Modal with Decagon (2025-11-13).
