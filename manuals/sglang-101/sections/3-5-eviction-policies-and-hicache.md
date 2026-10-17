# Eviction policies and HiCache

> When the GPU pool is full, SGLang evicts the least recently used prefix by default, and HiCache keeps evicted prefixes in host memory and in optional storage.

A coding agent holds a session of eight turns, and its history grows past 25,000 tokens. The GPU pool holds a few dozen such sessions. The ninety-first user arrives, and SGLang must evict a cached prefix to make room. When the first user returns, their history is gone, and the server computes all 25,000 tokens again.

When you finish this section, you can choose an eviction policy from a measured hit rate. You can also decide when hierarchical caching pays for its cost and which of its settings to change first.

## What eviction removes

Eviction removes only cached tokens that no running request locks, the leaves of [the radix tree](#s-radixcache). It never removes the KV of a running request. That case is retraction, and [section 2.3](#s-admission-and-retraction) covers it. Eviction starts from the leaves with the lowest score and moves toward the root.

`--radix-eviction-policy` decides the score {{code}}. The default is `lru`.

| Policy | Evicts first | When it helps |
|---|---|---|
| `lru` | the prefix with the oldest last access | most workloads, the default |
| `lfu` | the prefix with the fewest hits, then the oldest | a hot set of prompts that stays stable for hours |
| `slru` | prefixes with fewer than 2 hits, then the oldest | a stable hot set with a stream of one-time prompts |
| `priority` | prefixes of lower-priority requests | mixed traffic with request priorities |
| `tlru` | history that cannot change the next turn's prefill time | multi-turn agents with a tail-latency target |

**Tail-Optimized LRU (tlru):** a policy that keeps only the part of a conversation that the next turn's latency target needs, and evicts the rest first. It comes from a paper that the code names, arXiv 2510.15152. The SGLang team reports 43.9 percent lower p99 inter-token latency on DeepSeek-V4-Pro at concurrency 32 {{team 2026-10}}. That number comes from the release notes of v0.5.21, and we found no independent test of it.

## Our position: keep lru until you have measured a reason

The SGLang docs give the same advice: start with `lru` and change it only against a measured hit rate {{docs}}. Frequency-based policies hurt when popularity shifts, because yesterday's popular prompt stays in memory and today's prompt pays for it. Change the policy only when you can name the workload that the new policy serves, and measure the hit rate before and after.

## HiCache: three tiers for one tree

**HiCache:** SGLang's hierarchical cache, which extends the radix tree from GPU memory to host memory and to a storage backend. `--enable-hierarchical-cache` turns it on, and it defaults to off {{code}}. The tiers have names in the code and the docs.

- **L1:** the GPU pool from [section 3.1](#s-kv-bytes-per-token-and-the-kv-pool).
- **L2:** a pool in pinned host memory, private to one server instance.
- **L3:** an optional storage backend shared across instances, set with `--hicache-storage-backend`. The choices include `file`, `mooncake`, `hf3fs`, and `nixl`.

The tree records where the KV of each node lives. A match returns a GPU hit and a host hit length. SGLang then loads the host part back into fresh GPU slots, one layer ahead of the computation that needs it {{docs}}.

`--hicache-write-policy` decides when a node gets a copy in host memory. The default is `write_through`. The code sets a hit threshold of 1 for `write_through` and 2 for `write_through_selective` {{code}}. So `write_through` copies a prefix on its first hit, and `write_through_selective` waits for a second hit. `write_back` copies a node only when the GPU evicts it.

```figure
id: fig-3-5
kind: flow
title: one prefix through the three HiCache tiers
claim: Under write_through, a prefix gets a host copy on its first hit, so GPU eviction no longer forces a full prefill.
caption: Read left to right for writes and right to left for load-back. Thresholds and defaults from UnifiedRadixCache at v0.5.22.
```

## The settings that decide whether it helps

L3 prefetch has its own rules. SGLang skips a storage prefetch when the storage hit is shorter than 256 tokens {{code}}. The default prefetch policy is `timeout`, which stops waiting for storage after a time that grows with the prompt.

The docs and the code disagree on that time. The docs give a base of 2 seconds, 0.1 seconds per 1,024 tokens, and a maximum of 30 seconds {{docs}}. The code at v0.5.22 uses a base of 1.0 second and 0.25 seconds per 1,024 tokens {{code}}. We found no maximum in the cache code. The code wins. A 100,000-token prompt can wait about 25 seconds for storage before SGLang computes it again.

The host pool size comes from `--hicache-ratio` or `--hicache-size`. Without either, SGLang takes up to `--hicache-host-memory-fraction`, which defaults to 0.8 of available host memory {{code}}.

## What other teams measured

We could not run HiCache on the capture Mac, and the field evidence is thin. The numbers we found come from the SGLang team and its partners.

- The SGLang RBG team and the Mooncake team ran a multi-turn benchmark on Kubernetes {{field Alibaba Cloud 2026-03}}. The average time to first token was 5.91 seconds with GPU memory only. The L2 tier lowered it to 3.77 seconds, and a Mooncake L3 lowered it to 2.58 seconds.
- The SGLang HiCache post quotes Novita {{team 2025-09}}. Novita reported a 56 percent lower average time to first token, and a hit rate that rose from 40 to 80 percent.

The failure reports are better documented than the successes. In issue #35129, load-back that found no room evicted other sessions' GPU prefixes, with 250,000 evict calls against 376 completed load-backs {{issue #35129}}. Issue #30760 shows one tensor-parallel rank waiting forever in a prefetch check while the other ranks moved on {{issue #30760}}. Issue #42465 shows bursts of 540,000-token prompts that hang every rank with HiCache on and run without it {{issue #42465}}.

## Our position: add HiCache for long sessions, and add it in steps

HiCache pays for itself when sessions outlive the GPU pool: multi-turn chat, coding agents, and long documents asked several times. For short one-time prompts it adds copies and load-backs with nothing to reuse. Add it in this order, and measure the hit rate and the p99 time to first token at each step.

1. Make prompts prefix-stable first, as in [section 3.4](#s-prefix-friendly-prompts).
2. Turn on L2 only, with the default `write_through`.
3. Switch to `write_through_selective` if host copies use too much PCIe bandwidth.
4. Add an L3 backend when several instances should share prefixes.
5. Change the eviction policy last.

```takeaways
- Keep lru until a measured hit rate shows a reason to change it.
- Turn on HiCache for long multi-turn sessions, not for one-time prompts.
- Start HiCache with host memory only, then add storage after you measure.
- Trust the code's prefetch timeouts over the docs at v0.5.22.
- Watch for evict storms and rank hangs in the issues before you deploy HiCache.
```

Sources: python/sglang/srt/mem_cache/evict_policy.py, python/sglang/srt/mem_cache/unified_radix_cache.py (write_through_threshold, prefetch_timeout_base), python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py (prefetch_threshold), python/sglang/srt/arg_groups/fields/memory.py at v0.5.22, docs/docs/advanced_features/hicache_design.mdx and radix_eviction_policy.mdx at v0.5.22, the v0.5.21 release notes, the SGLang HiCache post of 2025-09-10, the RBG and Mooncake post on the Alibaba Cloud blog of 2026-03-10, issues #35129, #30760, and #42465, arXiv 2510.15152.
