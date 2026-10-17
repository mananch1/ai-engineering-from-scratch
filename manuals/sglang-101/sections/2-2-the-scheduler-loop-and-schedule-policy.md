# The scheduler loop and --schedule-policy

> The scheduler answers one question every few milliseconds, which tokens the GPU computes next, and at v0.5.22 its default answer is first come, first served.

Eight users send prompts in the same second. One of them pasted a 4,000-token document, and the others typed a sentence each. Should the long prompt go first, or the short ones? Should the request that shares a system prompt with yesterday's traffic jump the queue? Somebody has to decide, on every step, and in SGLang that somebody is the scheduler loop.

When you finish this section, you can read the scheduler's `Prefill batch` and `Decode batch` log lines. You can name the policy that orders the waiting queue and decide whether to change it.

## The cost of a bad order

A GPU step costs about the same whether it computes tokens for 1 request or for 8. So the scheduler wants big batches. A user who waits in the queue sees nothing at all. So the scheduler also wants short queues. Every policy is a trade between those two costs.

The obvious fix is a smart order, for example the request with the most cached prefix first. The SGLang paper proposed exactly that, longest prefix match, and proved it optimal for an offline batch {{paper}}. Online traffic is not an offline batch, and the smart order costs CPU time on every step.

## One turn of the loop

**Scheduler loop:** the function that runs forever in the scheduler process. Each turn takes new requests, picks the next batch, runs it on the GPU, and processes the results.

One turn does four things in order {{code}}:

1. **Ingest.** Drain new requests from the ZMQ socket and add them to the waiting queue.
2. **Pick.** If the last batch was a prefill, merge it into the running decode batch. Then try to build a new prefill batch from the waiting queue. If one exists, run it. If not, run one decode step for every running request.
3. **Run.** Build a `ForwardBatch` from the `ScheduleBatch` and run the model.
4. **Process.** Append the sampled tokens, check stop conditions, free or cache the KV of finished requests, and send results to the detokenizer.

A request rides one prefill batch, possibly over several chunks, and then joins the long-lived decode batch until it stops. Prefill wins when the scheduler can admit something. That rule keeps time to first token low, and [section 2.3](#s-admission-and-retraction) shows the budget check that decides "can admit".

```figure
id: fig-2-2
kind: flow
title: one turn of the scheduler loop
claim: Each turn ingests requests, orders the queue by policy, admits what fits the budget, and runs either one prefill batch or one decode step.
caption: Read top to bottom. Amber boxes are scheduler phases, the green box is the KV budget check, the plum boxes are GPU work. Function names are from the scheduler at v0.5.22.
```

## The two log lines

The scheduler prints one line per prefill batch and one decode line every 40 decode steps, the default of `--decode-log-interval` {{code}}. These lines are the most useful text the server writes. Here are three consecutive lines from our benchmark at concurrency 8.

```listing
title: eight requests, two prefill batches, one decode batch
source: capture/out/server-default.log
lang: text
note: Three consecutive lines from the concurrency 8 benchmark.
---
[2026-10-16 13:59:07] Prefill batch, #new-seq: 1, #new-token: 511, #cached-token: 1, token usage: 0.00, #running-req: 0, #queue-req: 7, #pending-token: 0, cuda graph: False, input throughput (token/s): 155.03
[2026-10-16 13:59:09] Prefill batch, #new-seq: 7, #new-token: 3553, #cached-token: 2, token usage: 0.02, #running-req: 1, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1914.13
[2026-10-16 13:59:09] Decode batch, #running-req: 8, #token: 4075, token usage: 0.02, cuda graph: False, gen throughput (token/s): 98.37, #queue-req: 0
```

Read the fields:

- `#new-seq` is how many requests this prefill admits. `#new-token` is how many prompt tokens it computes, and `#cached-token` is how many it reused from the radix cache.
- `token usage` is the share of the KV pool in use. Our pool is large, so usage stays near zero here.
- `#running-req` counts requests that are decoding. `#queue-req` counts requests that are waiting.
- `cuda graph: False` is expected on a Mac. On CUDA, `True` means the decode step replayed a captured graph, and [section 4.1](#s-forwardmode-and-cuda-graphs) explains why that matters.

The first line admits one request while seven wait. The next prefill admits the other seven together, 3,553 new tokens in one step at 1,914 tokens per second. Then all eight decode as one batch. Batching prefill is where the throughput comes from, and the log shows it happen.

## The policies

**Schedule policy:** the rule that orders the waiting queue before admission. Set it with `--schedule-policy`.

The default at v0.5.22 is `fcfs` {{code}}. The choices split into two groups.

| Group | Policies | What they use |
|---|---|---|
| Cache-aware | `lpm`, `dfs-weight`, `hrrn`, `shortest-prefill-first` | the radix tree: how much of each prompt is already cached |
| Cache-agnostic | `fcfs`, `lof`, `random`, `routing-key`, `priority` | arrival order, output length, a routing key, or a priority |

Many readers expect `lpm`, because the paper describes it. It was the default in v0.4.3 and became `fcfs` by v0.4.6. The research for this manual did not find the pull request that changed it.

Two rules in the code limit the cache-aware policies {{code}}:

- When the waiting queue holds more than 128 requests, `lpm` and `hrrn` change to FCFS for that turn. The comment says why: "Turn off the expensive prefix matching and sorting when the #queue is large."
- When the radix cache is off, every cache-aware policy becomes FCFS.

So `lpm` helps only in a narrow band. The queue must be long enough to reorder, short enough to stay under 128, and full of shared prefixes.

## Our position

Keep `fcfs`. It is fair, it costs no prefix matching on the hot path, and the radix cache still gives you the cache hits. A shared prefix that arrived first is in the tree when the next request matches it, whatever the order.

Switch to `lpm` only when three things are true. Your traffic shares long prefixes. Your queue is often between a few and 128 requests deep. You measured the cache hit rate before and after with `/metrics`. [Section 5.2](#s-the-metrics-endpoint) shows the `cache_hit_rate` series to compare.

If you need tiers of users, use `--enable-priority-scheduling` instead of a cache policy. A new request can then preempt running ones when its priority beats theirs by `--priority-scheduling-preemption-threshold`, 10 by default {{code}}.

## An incident: the limit you set is not the limit you get

A user served a hybrid Mamba model and set `--max-running-requests 8` {{issue #36889}}. The server accepted the flag and reported 8. Throughput flattened at 2 streams anyway. Only an info line in the log said that the Mamba state cache capped `max_running_requests` to 2.

The scheduler loop admits against every limit at once, and the smallest one wins. The fix the reporter used was more Mamba cache, `--max-mamba-cache-size 40` with bf16 state and `--mem-fraction-static 0.90`, for 75 percent more throughput at concurrency 12. At v0.5.22 the cap is a warning. Read the log for the limit that the loop really uses, not the one you typed.

```takeaways
- Read Prefill batch and Decode batch lines to see admission, caching, and queueing.
- Keep the fcfs default unless you measured a cache gain from lpm.
- Expect lpm to change to FCFS above 128 waiting requests.
- Check the log for the effective limit after you set a limit.
```

Sources: python/sglang/srt/managers/scheduler.py (run_event_loop, event_loop_normal, get_next_batch_to_run, run_batch, process_batch_result), python/sglang/srt/managers/schedule_policy.py (CacheAwarePolicy, CacheAgnosticPolicy, WAITING_QUEUE_PREFIX_MATCH_MAX, _determine_active_policy, _validate_and_adjust_policy), python/sglang/srt/arg_groups/fields/schedule.py at v0.5.22; SGLang paper, arXiv 2312.07104; capture/out/server-default.log; issue https://github.com/sgl-project/sglang/issues/36889.
