---
title: "Why a cold vLLM boot gets 256 MiB less KV cache"
date: 2026-10-24
summary: "Same flags, two boots, two KV cache sizes. Tracing the missing 256 MiB to a 296 MiB example input that PyTorch's standalone_compile allocates inside vLLM's memory-profiling window, and comparing the two upstream fixes on the same machine."
linkedin:
---

Boot vLLM twice with exactly the same flags, and the first boot gets 0.25 GiB less KV cache than the second. This post follows that 0.25 GiB down to a single line of PyTorch configuration: the guesses that were wrong, the measurement that found it, and the one-line experiment that proved it.

**The short version.** On a cold start (no compile cache), torch.compile runs inside the window where vLLM measures its activation memory. PyTorch's `standalone_compile` forces kernel autotuning at compile time, and the autotuning code allocates an example input the size of the embedding table: 296.2 MiB. It is freed right away, but the peak counter has already recorded it, so vLLM charges it as activation memory and hands 0.25 GiB less to the KV cache. A warm start loads the compiled artifacts directly and never makes that allocation.

This is the first of two posts. This one chases a real bug. The second chases something that looked like a bug and was not, and it answers the one number this post leaves unexplained.

Setup:

- GPU: RTX 5060 Ti 16 GB (sm_120)
- vLLM: v0.19.1-cu130, then v0.32.0 (torch 2.13.0+cu130, Model Runner V2)
- Model: Qwen3-Embedding-0.6B as a pooling model, fp8 weights and KV cache, prefix caching, chunked prefill, `--gpu-memory-utilization 0.5`
- Attention backend: FlashInfer

## Is VRAM still growing?

The original question was simple: once an embedding server is up, does its VRAM keep growing with traffic? On a GPU shared by several services, anything past the reservation lands on someone else.

vLLM runs one forward pass with dummy inputs at startup, measures the activation peak, and gives everything left in the `--gpu-memory-utilization` budget to the KV cache in one allocation. In theory nothing grows after that.

To tell one-time initialization apart from a leak, I wrote a small load generator that sends requests continuously and samples the process memory from `nvidia-smi` once a second. Initialization shows up as a step that then stays flat; a leak keeps climbing. The traffic deliberately mixed shapes the startup profile never ran: single sequences that fill a whole step, very long requests that span chunks, and requests that hit the prefix cache.

One hour, 128 concurrent workers, 16,349 requests. Memory reached 8,540 MiB at second 81 and did not move for the rest of the hour; the last five minutes averaged exactly the same as the first five. **No leak, only one-time initialization.**

If the story ended here, it would be a small "confirmed it is fine" experiment. The real problem showed up after a container restart.

## A restart changed the KV cache

A follow-up experiment needed a restart, and afterwards several numbers in the startup log had changed:

```
                       first boot              after restart
torch.compile          15.99 s                 3.13 s (Directly load)
KV cache               6.63 GiB / 124,224 tok  6.88 GiB / 128,880 tok
process memory, idle   8,048 MiB               8,272 MiB
```

Faster compilation after a restart makes sense: the compiled artifacts were cached. But **the KV cache size should not move**. It is set by the budget, and the flags were identical.

My first move with an unexpected difference was not to explain it but to check it was not noise. Three warm boots gave 6.88 GiB and an 8,272 MiB baseline every time. Reproducible, so worth chasing.

### A cold start is harder to produce than it looks

Comparing cold and warm boots requires producing a cold boot reliably. My first attempt deleted vLLM's own compile cache (`/root/.cache/vllm/torch_compile_cache`). Compilation did rerun, but the KV cache was still 6.88 GiB, exactly like a warm boot.

The container also holds PyTorch Inductor's cache (`/tmp/torchinductor_root`) and Triton's (`/root/.triton`). Hunting caches one by one is unreliable, so I settled on **a fresh container from the image every time**, and checked the log on each run: compilation took over ten seconds and there was no `Directly load`. Only then did it count as cold.

That habit paid off for the rest of the investigation: **verify that your control condition actually holds, every time, with evidence.**

## Same gap on v0.32.0, and only in one line item

Before reporting anything upstream, the first question is always whether the latest release still does it. The same procedure on v0.32.0:

```
                        cold boot               warm boot
torch.compile           25.40 s                 0.19 s (AOT load)
KV cache                5.99 GiB / 112,096 tok  6.24 GiB / 116,784 tok
process memory after
first large request     8,226 MiB               8,482 MiB
```

Still there. More interesting is the size of the gap. Qwen3-0.6B has 28 layers, 8 KV heads and a head dimension of 128, so with an fp8 KV cache each token takes:

```
28 layers × 8 KV heads × 128 head_dim × 2 (K and V) × 1 byte = 56 KiB
```

The cold boot is 4,688 tokens short, which is **256 MiB**. On v0.19.1 the gap was 4,656 tokens, about 255 MiB. A dozen releases apart, almost the same number. A figure that tidy looks like one fixed-size allocation, and it became the fingerprint for everything that followed.

### The gap is all in peak activation

At the end of startup, v0.32.0 prints a full breakdown of the budget. Side by side:

```
                          cold       warm
weights + non-torch       0.84 GiB   0.84 GiB
peak activation           0.92 GiB   0.67 GiB
CUDA graph                0.03 GiB   0.03 GiB
KV cache                  5.99 GiB   6.24 GiB
```

Everything matches except the activation peak measured during profiling, and that 0.25 GiB comes straight out of the KV cache. The search space shrank from "the whole startup" to "that one peak".

Which side is wrong: does the cold boot overestimate, or does the warm boot underestimate? The runtime non-KV usage settles it. Final usage minus KV cache is 8,226 − 6,134 = 2,092 MiB cold and 8,482 − 6,390 = 2,092 MiB warm. **Identical.** The running server never needs the extra 0.25 GiB, so the cold boot overestimates.

## Three suspects that turned out innocent

First, the scope of the measurement. The `with memory_profiling(...)` block in `gpu_worker.py` contains one call, `profile_run()`. On entry, `memory_profiling` calls `empty_cache()` and then `reset_peak_memory_stats()`, and peak activation is the growth of `allocated_bytes.all.peak` after that. So the culprit had to be some PyTorch memory allocated during `profile_run()` and freed again.

A peak counter works like a high-water mark on a tank. It only records the highest level and never moves down. Pour in a bucket and drain it straight away, and the mark is still there.

With 256 MiB as the fingerprint, I suspected three things in turn:

- **FlashInfer's workspace.** The attention backend allocates a large scratch buffer. Its size from the source is max(394 MiB, 2,048 × 16 × 128 × 16 B) = 394 MiB. Wrong size, ruled out.
- **Inductor's combo-kernel benchmarking.** The config has `benchmark_combo_kernel: True`, and only a cold boot benchmarks. Turning just that flag off made compilation 5 s faster, which proved the change took effect, but peak stayed at 0.92 GiB. Ruled out.
- **Triton's L2-flush buffer.** Triton's `driver.py` hard-codes `cache_size` as 256 × 1024 × 1024 bytes, a perfect size match. I copied the file, changed 256 to 64, bind-mounted it read-only into a fresh container, and confirmed the container read the patched file. Peak stayed at 0.92 GiB. Ruled out.

The third one hurt: exact size, plausible mechanism. But a causal experiment does not lie. Shrinking the buffer to 64 MiB did not move the peak at all.

Two rules held throughout. **Change one variable at a time**, spelling out every other setting explicitly. And **confirm the change actually took effect** before using the result to rule anything out; otherwise "nothing changed" may only mean the setting never applied.

After two misses I noticed I had been hunting for a hard-coded 256 MiB constant, and the culprit did not have to be a constant. **Two wrong guesses in a row is the signal to stop guessing and start measuring.**

## Stop guessing, record every allocation

PyTorch ships a memory recorder, `torch.cuda.memory._record_memory_history`, which logs the size of every allocation and free together with the Python call stack. I turned it on at the start of the measurement window in vLLM's `mem_utils.py` and dumped it at the end, so the recording covers exactly the span vLLM uses to compute the peak. As before, the patch was a read-only bind mount into a throwaway container; the image itself never changed.

A small script then replays the recording and lists what was alive at the high-water mark:

```
                                 cold          warm
high-water mark in the window    306.3 MiB     52.0 MiB
large allocations at the peak    one, 296.2    none
                                 MiB, freed
                                 before the
                                 window ends
```

The gap is 254 MiB, matching the 0.25 GiB missing from the KV cache. That one allocation came from `torch._dynamo.testing.rand_strided`, which builds a random tensor of a given shape.

### What 296.2 MiB is

That size made me think of the embedding table. From the model's `config.json`: vocab_size 151,669, hidden_size 1,024, dtype bfloat16.

```
151,669 × 1,024 × 2 bytes = 310,618,112 bytes = 296.23 MiB
```

An exact match. A random tensor the size of the embedding table, allocated and freed during cold-start profiling.

### The answer my own tool was hiding

The first version of my analysis script filtered out every frame under `/torch/` to keep the output readable, leaving only vLLM's frames. The stack appeared to jump from vLLM's `compiler_interface.py` straight to something called `<string>:426`. The missing link was in exactly the frames I had filtered out. With a full-stack option added, the chain was complete (outermost first):

1. `gpu_worker.py`, `determine_available_memory`: the measurement window opens.
2. `profile_run`, then `_dummy_run`: the dummy forward runs inside the window.
3. `qwen3.py` `forward`, then vLLM's `aot_compile`: with no artifacts on a cold boot, **the first forward triggers compilation**.
4. `compiler_interface.py`, then PyTorch's `standalone_compile` and `compile_fx`: Inductor compiles the graph.
5. `codegen/wrapper.py`, `generate_and_run_autotune_block`: Inductor generates an autotuning script and runs it with `exec`, which is where `<string>:426` comes from.
6. `select_algorithm.py`, `generate_example_value`: example inputs are generated for each kernel argument at its real shape.
7. `rand_strided`: 296.2 MiB allocated.

This is the lesson that stuck with me most: **a filter you added for readability can hide exactly the answer you need.** When the trail goes cold, check your own tools.

## Who turned on compile-time autotuning

The autotuning block in `wrapper.py` has a single trigger:

```python
if config.triton.autotune_at_compile_time:
    self.generate_and_run_autotune_block()
```

Its default comes from the environment variable `TORCHINDUCTOR_AUTOTUNE_AT_COMPILE_TIME` and is `None` when unset. The obvious first try was setting it to `0`. The container read it as `False`, yet the cold peak was still 0.92 GiB, the 296 MiB allocation was still there, and even the recorder's event count was identical.

With a single trigger and the variable verifiably applied, only one possibility remained: **something sets it back to `True` while compilation runs.** Searching vLLM and PyTorch for that setting turned up nothing in vLLM. The answer is in PyTorch's `standalone_compile.py`:

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

`standalone_compile` is how vLLM produces compiled artifacts it can save, and it turns compile-time autotuning on **unconditionally**, overriding the environment variable. There is a good reason for that: an artifact meant to be saved and loaded later has to finish autotuning every kernel at compile time.

### The causal experiment

So far this was inference from reading code. To establish cause, change it directly: flip only that `True` to `False` (with `diff` confirming it is the only change in the file) and do a cold boot in a fresh container.

```
condition                       peak act.   recorder peak   296 MiB alloc
cold (stock)                    0.92 GiB    306.3 MiB       yes
warm (stock)                    0.67 GiB     52.0 MiB       no
cold, env var set to 0          0.92 GiB    306.3 MiB       yes (overridden)
cold, that line set to False    0.70 GiB     82.0 MiB       no
```

The last row differs from the first by one line of code, and the cold boot now looks almost like a warm one. The remaining ~30 MiB is other, smaller compile-time scratch.

This was a diagnostic, not a fix. Turning that setting off could break the integrity of the saved artifacts, and as the next section explains, it is not where the fix belongs anyway.

## Root cause

Putting the evidence together:

1. On a cold vLLM boot, the first forward pass runs inside the `memory_profiling` window.
2. That forward triggers compilation through `standalone_compile`.
3. `standalone_compile` forces compile-time autotuning.
4. The autotuning script builds example inputs at real shapes, one of them as large as the embedding table (296.2 MiB).
5. It is freed immediately, but the peak counter has recorded it, so vLLM counts it as activation and gives 0.25 GiB less to the KV cache.
6. A warm boot loads the compiled artifacts and skips steps 2 to 4.

It takes three conditions at once, and separating them matters because it decides where the fix goes:

- **A. Compilation is deferred until it lands inside the measurement window** (vLLM). This is the bug: the window is meant to capture runtime needs, and it captures compile-time scratch as well.
- **B. `standalone_compile` forces compile-time autotuning** (PyTorch). Not a bug: a saveable artifact needs it.
- **C. Example inputs are allocated at real shapes** (PyTorch). Not a bug, just memory-hungry.

The causal experiment flipped B because it was the easiest switch. The fix belongs in A: move compilation out of the window, and nothing compilation allocates can be mistaken for activation.

The error is conservative. It under-allocates KV and does not cause OOM. The real cost is that **the same configuration gets a different KV cache size depending on whether a compile cache happens to exist**. For anyone planning capacity precisely, that is a silent variable.

## Upstream: already reported, with two fixes

Searching before filing turned up two reports of the same problem:

- [#54383](https://github.com/vllm-project/vllm/issues/54383): the first boot of a new batch shape gets about 5% less KV cache than later boots (2 × RTX 4090, Qwen3-32B-AWQ).
- [#54122](https://github.com/vllm-project/vllm/issues/54122): the same mechanism, reported for a 176B model at TP4 with a much larger compile-time spike.
- [#54479](https://github.com/vllm-project/vllm/pull/54479): fix one, run an unprofiled warm-up forward before the measurement so compilation finishes first.
- [#54209](https://github.com/vllm-project/vllm/pull/54209): fix two, reset the peak counter when compilation finishes, keeping any peak recorded before it (for example a multimodal encoder pass).

Two reports and two fixes, none of them linking to each other, and neither PR reviewed or merged yet. The reporter's own caveats noted that the exact allocation being counted had never been instrumented. That was exactly what I had.

### Both fixes on the same machine

Using the same bind-mount approach, I applied each PR to v0.32.0 and ran two cold boots each, with the recorder attached:

```
v0.32.0, cold boot   KV tokens (2 runs)     vs warm   296 MiB in window
unpatched            112,096                -4,688    yes, counted
#54479               116,784 / 116,784          0     absent (moved out)
#54209               116,672 / 116,672       -112     present, not counted
warm boot            116,784                    -     absent
```

Both close nearly all of the gap, and the recorder shows how they differ: one **moves** the allocation out of the window, the other **leaves it in the window but stops counting it**. The #54209 author measured exact parity on an H100; my 112-token residual (about 6 MiB) is reproducible, but it may come from applying the patch to v0.32.0 rather than to the PR's base.

I posted the stack, the causal experiment and this comparison as [a comment on #54383](https://github.com/vllm-project/vllm/issues/54383#issuecomment-6382246294), linking the two reports and the two PRs together.

## What to do until a fix lands

Raising `--gpu-memory-utilization` for cold boots does not work. It is a startup flag, you do not know in advance whether a boot will be cold or warm, and whatever you add for a cold boot becomes an overshoot on a warm one.

- **Set `--kv-cache-memory` explicitly.** The KV size no longer depends on profiling, so cold and warm boots match. The cost is computing the right value yourself.
- **Keep the compile cache.** Every boot after the first is warm. New nodes, new versions and cleared caches still bring you back to a cold boot.
- **`VLLM_ENABLE_STARTUP_PLAN=1`** saves the KV size from the first boot and reuses it. What it saves depends on whether that first boot was cold or warm, and its fingerprint does not include the state of the compile cache.

## One number I could not explain

One number still did not add up. The recorder's high-water mark and vLLM's reported peak activation differ by the same constant on both cold and warm boots:

```
         recorder peak   vLLM peak activation      difference
cold     306 MiB         0.92 GiB (about 942 MiB)  about 635 MiB
warm      52 MiB         0.67 GiB (about 686 MiB)  about 635 MiB
```

Because the offset is identical on both sides, it does not affect anything above, and I said "unexplained" in the upstream comment rather than invent a reason. But a stable 635 MiB does not come from nowhere.

The answer is in another line of the vLLM startup log, the one claiming the CUDA graph memory estimate is 2150% too high. That is the next post.

## What I would tell myself

None of this came from a clever guess. I guessed wrong several times, but each step narrowed the search: from "is VRAM growing", to "only peak activation differs", to "one 296.2 MiB allocation", to "one line of PyTorch configuration".

- Confirm it reproduces before explaining it.
- Find where the difference is before looking for why.
- Use exact numbers as fingerprints: 256 MiB and 296.2 MiB turned "looks related" into "matches exactly".
- Verify your control condition holds; deleting one cache is not a cold boot.
- Change one variable at a time, and confirm the change took effect.
- After two wrong guesses, switch from guessing to measuring.
- When the trail goes cold, check your own tools.
- End with a causal experiment.
- Search the repo before filing. I would have opened a third duplicate issue; adding the exact allocation and a side-by-side comparison to the existing threads was far more useful.