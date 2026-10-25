---
title: "Why a cold vLLM boot gets 256 MiB less KV cache"
date: 2026-10-24
summary: "Same flags, two boots, two KV cache sizes. Tracing the missing 256 MiB to a 296 MiB example input that PyTorch's standalone_compile allocates inside vLLM's memory-profiling window, and comparing the two upstream fixes on the same machine."
linkedin:
---

Boot vLLM twice with exactly the same flags, and the first boot gets 0.25 GiB less KV cache than the second. This is how I traced that gap to one line of PyTorch configuration.

At a glance:

- Symptom: a cold boot (no compile cache) gets 256 MiB less KV cache than a warm one, with identical flags.
- Cause: torch.compile runs inside vLLM's memory-profiling window, and `standalone_compile`'s forced autotuning allocates a 296 MiB example input that gets counted as activation memory.
- Impact: the KV cache size depends on whether a compile cache exists. Conservative, no OOM.
- Workaround: set `--kv-cache-memory` explicitly.
- Upstream: two open fixes, [#54479](https://github.com/vllm-project/vllm/pull/54479) and [#54209](https://github.com/vllm-project/vllm/pull/54209), compared below.

This is the first of two posts. The second chases something that looked like a bug and was not, and explains the one number this post leaves open.

Setup:

- GPU: RTX 5060 Ti 16 GB (sm_120)
- vLLM: v0.19.1-cu130, then v0.32.0 (torch 2.13.0+cu130, Model Runner V2)
- Model: Qwen3-Embedding-0.6B as a pooling model, fp8 weights and KV cache, prefix caching, chunked prefill, `--gpu-memory-utilization 0.5`
- Attention backend: FlashInfer

## From a leak check to a KV cache that moved

I started with a simpler question: does an embedding server's VRAM keep growing with traffic? A one-hour soak with 128 workers and 16,349 requests, deliberately mixing shapes the startup profile never ran, said no. Memory stepped up once at startup and stayed flat for the rest of the hour.

A follow-up test needed a container restart, and the startup log changed:

| | first boot | after restart |
|---|---:|---:|
| `torch.compile` | 15.99 s | 3.13 s (`Directly load`) |
| KV cache | 6.63 GiB / 124,224 tok | 6.88 GiB / 128,880 tok |
| process memory, idle | 8,048 MiB | 8,272 MiB |

Faster compilation after a restart is expected, because the artifacts were cached. The KV cache size should not move: it comes from the budget, and the flags were identical. Three warm boots gave 6.88 GiB every time, so it was reproducible and worth chasing.

Producing a cold boot reliably took one detour. vLLM, Inductor and Triton each keep their own compile cache, so deleting vLLM's alone still gave warm-boot numbers. Every cold run below uses a fresh container, confirmed in the log by a long compile and no `Directly load`.

## Same gap on v0.32.0, all in peak activation

The latest release still does it:

| | cold boot | warm boot |
|---|---:|---:|
| `torch.compile` | 25.40 s | 0.19 s (AOT load) |
| KV cache | 5.99 GiB / 112,096 tok | 6.24 GiB / 116,784 tok |
| process memory after first large request | 8,226 MiB | 8,482 MiB |

With an fp8 KV cache, each token of this model takes:

```
28 layers × 8 KV heads × 128 head_dim × 2 (K and V) × 1 byte = 56 KiB
```

The cold boot is 4,688 tokens short: 256 MiB. On v0.19.1 the gap was 4,656 tokens, about 255 MiB. A number that stable across a dozen releases looks like one fixed-size allocation, and it became the fingerprint for the search.

The budget breakdown v0.32.0 prints at startup shows where it goes:

| | cold | warm |
|---|---:|---:|
| weights + non-torch | 0.84 GiB | 0.84 GiB |
| **peak activation** | **0.92 GiB** | **0.67 GiB** |
| CUDA graph | 0.03 GiB | 0.03 GiB |
| KV cache | 5.99 GiB | 6.24 GiB |

Only the activation peak measured during profiling differs, and that 0.25 GiB comes straight out of the KV cache. The runtime non-KV usage (final memory minus KV cache) is 2,092 MiB on both boots, so the running server never needs the extra 0.25 GiB. **The cold boot overestimates.**

## Three wrong guesses

The measurement window in `gpu_worker.py` wraps a single call, `profile_run()`. Peak activation is how far `allocated_bytes.all.peak` rises after a reset, and a peak counter works like a high-water mark: it never moves down, even if the memory is freed a moment later. So the culprit was some PyTorch allocation made and freed during `profile_run()`. With 256 MiB as the fingerprint, I suspected three things:

- FlashInfer's workspace: 394 MiB according to the source. Wrong size.
- Inductor's combo-kernel benchmarking: turning it off made compilation 5 s faster, so the change applied, but peak stayed at 0.92 GiB.
- Triton's L2-flush buffer: hard-coded as 256 × 1024 × 1024 bytes, an exact match. Shrinking it to 64 MiB through a read-only bind mount left the peak at 0.92 GiB.

Each test changed one variable and confirmed the change took effect before ruling anything out. After the third miss I noticed I had been hunting for a hard-coded 256 MiB constant, and the culprit did not have to be one. **Two wrong guesses in a row is the signal to stop guessing and start measuring.**

## Recording every allocation

PyTorch's `torch.cuda.memory._record_memory_history` logs every allocation and free with its Python call stack. I turned it on at the start of vLLM's measurement window and dumped it at the end, so the recording covers exactly the span vLLM uses for the peak:

| | cold | warm |
|---|---|---|
| high-water mark in the window | 306.3 MiB | 52.0 MiB |
| large allocations at the peak | one 296.2 MiB tensor, freed before the window ends | none |

The 254 MiB difference matches the missing KV cache. The allocation came from `torch._dynamo.testing.rand_strided`, which builds a random tensor of a given shape, and its size gave it away:

```
vocab 151,669 × hidden 1,024 × 2 bytes (bf16) = 296.23 MiB
```

A random tensor exactly the size of the embedding table. The full call stack, outermost first:

1. `gpu_worker.py`, `determine_available_memory`: the measurement window opens.
2. `profile_run`, then `_dummy_run`: the dummy forward runs inside the window.
3. `qwen3.py` `forward`, then vLLM's `aot_compile`: with no artifacts on a cold boot, the first forward triggers compilation.
4. PyTorch's `standalone_compile`, then `compile_fx`: Inductor compiles the graph.
5. `codegen/wrapper.py`, `generate_and_run_autotune_block`: Inductor generates an autotuning script and runs it with `exec`.
6. `select_algorithm.py`, `generate_example_value`: example inputs are built for each kernel argument at its real shape.
7. `rand_strided`: 296.2 MiB allocated.

Steps 4 to 6 almost escaped me. My first analysis script filtered out every frame under `/torch/` to keep the output readable, and the answer was in exactly the frames it removed.

## The line that turns it on

Inductor only runs that autotuning block when `config.triton.autotune_at_compile_time` is true. Setting `TORCHINDUCTOR_AUTOTUNE_AT_COMPILE_TIME=0` changed nothing: the container read it as `False`, yet the 296 MiB allocation was still there. Something had to be setting it back during compilation. It is PyTorch's `standalone_compile.py`:

```python
@contextlib.contextmanager
def _standalone_context(gm, dynamic_shapes, aot, fake_mode=None):
    ...
    with (
        torch._guards.tracing(tracing_context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
        ...
    ):
        yield
```

`standalone_compile` is how vLLM produces artifacts it can save, and it turns compile-time autotuning on unconditionally. That is reasonable: an artifact meant to be loaded later has to finish autotuning when it is compiled.

To prove cause rather than infer it, I flipped only that `True` to `False` and did a cold boot in a fresh container:

| condition | peak activation | recorder peak | 296 MiB alloc |
|---|---:|---:|---|
| cold (stock) | 0.92 GiB | 306.3 MiB | yes |
| warm (stock) | 0.67 GiB | 52.0 MiB | no |
| cold, env var set to 0 | 0.92 GiB | 306.3 MiB | yes (overridden) |
| cold, that line set to `False` | 0.70 GiB | 82.0 MiB | no |

One line of code, and the cold boot looks almost like a warm one. The remaining ~30 MiB is smaller compile-time scratch. This was a diagnostic, not a fix: the setting exists for a reason, and the bug is elsewhere.

## Root cause

On a cold boot, the first forward pass lands inside `memory_profiling`, triggers `standalone_compile`, and its forced autotuning allocates an embedding-sized example input that the peak counter records. A warm boot loads the artifacts and never compiles. Three conditions have to hold at once:

- A. Compilation is deferred until it falls inside the measurement window (vLLM).
- B. `standalone_compile` forces compile-time autotuning (PyTorch, needed for saveable artifacts).
- C. Example inputs are allocated at real shapes (PyTorch, memory-hungry but correct).

The experiment flipped B because it was the easiest switch. **The bug is A, and that is where the fix belongs:** move compilation out of the window and nothing it allocates can be mistaken for activation. The error is conservative and never causes OOM, but it makes the KV cache size depend on whether a compile cache happens to exist, which is a silent variable for anyone planning capacity.

## Upstream: already reported, with two fixes

Searching before filing turned up two reports, [#54383](https://github.com/vllm-project/vllm/issues/54383) and [#54122](https://github.com/vllm-project/vllm/issues/54122), and two competing fixes, none of them linked to each other:

- [#54479](https://github.com/vllm-project/vllm/pull/54479) runs an unprofiled warm-up forward before measuring, so compilation finishes first.
- [#54209](https://github.com/vllm-project/vllm/pull/54209) resets the peak counter when compilation finishes, keeping any earlier peak such as a multimodal encoder pass.

Neither had been reviewed, and nobody had instrumented which allocation was being counted. I applied each PR to v0.32.0 and ran two cold boots each, with the recorder attached:

| v0.32.0, cold boot | KV tokens (2 runs) | vs warm | 296 MiB in window |
|---|---:|---:|---|
| unpatched | 112,096 | −4,688 | yes, counted |
| [#54479](https://github.com/vllm-project/vllm/pull/54479) | 116,784 / 116,784 | 0 | absent (moved out) |
| [#54209](https://github.com/vllm-project/vllm/pull/54209) | 116,672 / 116,672 | −112 | present, not counted |
| warm boot | 116,784 | – | absent |

Both close nearly all of the gap. The recorder shows how they differ: one moves the allocation out of the window, the other leaves it there and stops counting it. The #54209 author measured exact parity on an H100, so my 112-token residual may come from applying the patch to v0.32.0 rather than to its base. I posted the stack, the causal experiment and this comparison in [a comment on #54383](https://github.com/vllm-project/vllm/issues/54383#issuecomment-6382246294).

## Until a fix lands

Raising `--gpu-memory-utilization` does not help: you cannot know in advance whether a boot will be cold, and extra headroom for a cold boot becomes an overshoot on a warm one.

- Set `--kv-cache-memory` explicitly. Cold and warm boots then match; you compute the value yourself.
- Keep the compile cache. Every boot after the first is warm, until a new node, a new version or a cleared cache.
- `VLLM_ENABLE_STARTUP_PLAN=1` reuses the first boot's KV size, but what it saves depends on whether that first boot was cold, and its fingerprint ignores the compile cache.

## One number left over

The recorder's high-water mark and vLLM's reported peak activation differ by the same constant on both boots:

| | recorder peak | vLLM peak activation | difference |
|---|---:|---:|---:|
| cold | 306 MiB | 0.92 GiB (about 942 MiB) | about 635 MiB |
| warm | 52 MiB | 0.67 GiB (about 686 MiB) | about 635 MiB |

Because the offset is identical, it does not change anything above, and I called it unexplained upstream rather than invent a reason. The answer is in another line of the startup log, the one claiming the CUDA graph memory estimate is 2150% too high. That is [the next post](../2026-10-vllm_cudagraph_estimate_log/).

## Takeaways

- Exact numbers make the best fingerprints. 256 MiB and 296.2 MiB turned "looks related" into "matches exactly".
- After two wrong guesses, measure instead. When the trail goes cold, check what your own tools are hiding.
- Search the repo before filing. Adding the exact allocation and a side-by-side comparison to the existing threads was worth more than a third duplicate issue.
