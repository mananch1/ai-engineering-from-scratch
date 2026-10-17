# SGLang 101 capture kit

This kit starts the real SGLang 0.5.22 server on Apple silicon through its MLX backend, sends requests to it, runs the official benchmark, and writes what it sees to `out/`. Every listing and every measured number in the manual comes from these files.

## Requirements

- An Apple silicon Mac with macOS 14 or newer.
- SGLang 0.5.22 installed from the tag with the `srt_mps` extra (see the front matter of the manual).
- The `sglang` command on `PATH`, and network access the first time, to download `mlx-community/Qwen3-0.6B-4bit` into the Hugging Face cache.

## Run

```bash
python3 capture/run.py           # record everything into capture/out/
python3 capture/run.py --check   # record into a temporary directory and compare
python3 capture/run.py --only small-pool   # record one server configuration
```

Run the commands with the Python of the SGLang environment. A full run starts four servers and takes 20 to 25 minutes on the recording Mac. `--check` takes the same time. Without SGLang 0.5.22 and MLX on Apple silicon, `--check` prints `skipped` with the reason and exits 0.

## What it records

| Server | Flags added to the common flags | Files |
|---|---|---|
| default | none | `generate.txt`, `cache-experiment.txt`, `bench-random-c1.txt`, `bench-random-c8.txt`, `bench-random-c32.txt`, `bench-shared-prefix.txt`, `metrics.txt` |
| no-radix-cache | `--disable-radix-cache` | `no-radix-cache-bench-shared-prefix.txt`, `no-radix-cache-metrics.txt` |
| no-overlap | `--disable-overlap-schedule` | `no-overlap-bench-random-c8.txt` (the server crashes, see the log) |
| small-pool | `--max-total-tokens 4096` | `small-pool-bench-long-output-c16.txt`, `small-pool-metrics.txt` |

The common flags are `--model-path mlx-community/Qwen3-0.6B-4bit --disable-cuda-graph --enable-cache-report --enable-metrics --mem-fraction-static 0.4`, with `SGLANG_USE_MLX=1`. Each server writes `server-<name>.log`, and `server-<name>.command.txt` holds its exact command. `serve-help.txt` is the output of `sglang serve --help`, and `environment.json` records the versions. The benchmark JSONL files beside each `.txt` hold the per-run records the benchmark writes.

`one_request.py` sends one native `/generate` request and one OpenAI chat request. `cache_experiment.py` sends the same 3,000-token system prompt five times in three forms: unchanged, with a random request id at the start, and with the same id at the end.

## What `--check` compares

Timings change on every run, so the check compares the facts that must stay the same:

- the generated text and the token counts at temperature 0;
- the median cached tokens of each cache case, and whether the two id cases stay under 50 and over 2,900;
- the successful requests and the input and output token totals of each benchmark;
- `bytes_per_slot`, the tree cache line, and any `RuntimeError` line in each server log;
- the set of metric names and the set of `sglang serve` options;
- the server commands.

Absolute home paths in benchmark output are replaced with `~`.

`oom-exploratory.log` was not recorded by this kit. It comes from an exploratory run before the kit existed, with the default memory fraction, and section 3.2 says so.
