---
title: "vLLM's CUDA graph estimate isn't 2150% off. Tracing the log."
date: 2026-10-25
summary: "A startup log says vLLM reserves twenty times more memory for CUDA graphs than it uses. Turning the estimate off seemed to prove it. Taking the 630 MiB apart showed the estimate was within 3%, and that fixing it would have pushed the engine over its budget."
linkedin:
---

vLLM's startup log told me its CUDA graph memory estimate was 2150% too high, and an experiment seemed to prove it. That conclusion was wrong. This post is mostly about how I found out.

At a glance:

- Symptom: the log reports `0.03 GiB (actual), 0.62 GiB (estimated), difference: 2150%` for CUDA graph memory.
- What it really is: only 28 MiB of the 0.62 GiB is CUDA graph memory. Most of the rest is memory allocated for the first time during the profiling capture that stays resident, including a 394 MiB FlashInfer workspace.
- The estimate: about 20 MiB too high, roughly 3%. The total is right; the label is wrong.
- The risk: treating it as an overestimate and turning it off pushes the engine about 0.6 GiB over its budget.
- Upstream: [#59368](https://github.com/vllm-project/vllm/pull/59368) fixes the accounting and the log. I validated it on my machine.

This is the second of two posts. [The first one](../2026-10-vllm_cold_start_kv_cache/) traced a cold-start KV cache gap and ended with one number it could not explain: a constant 635 MiB between my memory recorder and vLLM's reported peak activation. That number is where this one starts.

Same setup as before: RTX 5060 Ti 16 GB (sm_120), vLLM v0.32.0, Qwen3-Embedding-0.6B as a pooling model, FlashInfer, `--gpu-memory-utilization 0.5`. A pooling model only captures PIECEWISE CUDA graphs.

## The log line

Before allocating the KV cache, vLLM does a trial CUDA graph capture against a tiny throwaway KV cache, measures how much memory it took, and reserves that much in the budget. It has to estimate: the real graphs can only be captured after the KV cache exists, and the KV cache size depends on how much room the graphs need. At the end of startup it compares the two:

```
CUDA graph pool memory: 0.03 GiB (actual), 0.62 GiB (estimated), difference: 0.59 GiB (2150.0%).
```

The obvious reading: the estimate is twenty times too high, and 0.59 GiB of KV cache, about 10% on this card, is reserved for nothing. But only the two printed numbers are facts. That the estimate is subtracted from the budget, and that 0.03 GiB is what the graphs really need, are assumptions, and one of them turned out to be wrong.

## Solving the first post's leftover number

Is the estimate subtracted at all? The cold-boot budget line says 7.74 GiB budget, 0.84 GiB weights and non-torch, 0.92 GiB peak activation, and 5.99 GiB of KV cache. 7.74 − 0.84 − 0.92 = 5.98, so the arithmetic closes without the estimate anywhere in it. Yet the same log says utilization 0.5 is "equivalent to 0.4603 without CUDA graph memory profiling", which is 0.0397 × 15.48 GiB ≈ 0.61 GiB reserved for graphs.

The two clues contradict each other. `gpu_worker.py` resolves it:

```python
self.peak_activation_memory = (
    profile_result.transient_peak_headroom + cudagraph_memory_estimate_applied
)
...
self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)
```

**The peak activation in the log is the real activation peak plus the CUDA graph estimate.** The estimate is subtracted; it was hiding inside peak activation. That also solves the first post's mystery:

| | log's peak activation | minus the 0.62 GiB estimate | memory recorder's peak |
|---|---:|---:|---:|
| cold | 0.92 GiB | 0.30 GiB (about 307 MiB) | 306.3 MiB |
| warm | 0.67 GiB | 0.05 GiB (about 51 MiB) | 52.0 MiB |

The constant 635 MiB was the CUDA graph estimate all along. Writing "unexplained" instead of inventing a reason paid off: the unexplained number became the next lead.

## The experiment that misled me

If the estimate is subtracted, is the extra 0.59 GiB ever used? `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` turns the reservation off; the trial capture still runs, its result just is not subtracted. I wrote down predictions first:

| | estimate on (default) | predicted, estimate off | measured |
|---|---:|---:|---:|
| trial capture | 0.62 GiB | still 0.62 | 0.62 GiB |
| peak activation | 0.92 GiB | about 0.30 | 0.30 GiB |
| KV tokens | 112,096 | about 123,700 | 123,616 |
| real capture | 0.03 GiB | about 0.03 | 0.03 GiB |
| process memory after first full step | 8,226 MiB | about 8,860 | 8,856 MiB |
| errors | 0 | 0 | 0 |

Every prediction held: 11,520 extra tokens, about 630 MiB, matching the process growth. My conclusion was that the 0.59 GiB had been sitting idle and turning the estimate off gives almost 10% more KV cache for free. The numbers were clean. The conclusion was wrong.

## Same function, different moment

To fix an overestimate I still needed to know why the trial measured 0.62 when the real capture measured 0.03. Both numbers come from the same function, `capture_model()`:

```python
torch.accelerator.empty_cache()
start_free_gpu_memory = torch.accelerator.get_memory_info()[0]
...  # capture every CUDA graph
end_free_gpu_memory = torch.accelerator.get_memory_info()[0]
cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
```

It measures the drop in free memory on the whole device, not the size of the graph pool, so anything else allocated during the capture is counted too. It also calls `empty_cache()` before the capture but not after, so blocks freed during the capture but still held by the caching allocator are counted as well.

The difference between the two calls is timing. The trial runs before the KV cache is allocated and before the warm-up; it is the first time these graphs and their kernels ever run. The real capture comes after both.

## Swapping the attention backend

One suspect stood out. FlashInfer allocates a large workspace, 394 MiB by default, enough to explain much of the 0.62 GiB. vLLM also has a Triton attention backend that does not need it, so the cleanest test was to change only `--attention-backend`:

| | FlashInfer | TRITON_ATTN | difference |
|---|---:|---:|---:|
| trial capture (estimate) | 0.62 GiB | 0.21 GiB | −420 MiB |
| real capture (actual) | 0.03 GiB | 0.03 GiB | 0 |
| KV tokens | 112,096 | 119,712 | +7,616 |

The estimate dropped by about 420 MiB, close to 394, and the actual did not move. (My grep for the backend line first printed nothing, because a manually chosen backend logs in a different format; I only trusted these numbers once the log showed `Using AttentionBackendEnum.TRITON_ATTN backend`.) Two thirds of the gap was tied to FlashInfer, but that did not yet prove it was the workspace, or explain the remaining 0.18 GiB.

## Taking the 630 MiB apart

Inside the `capture_model()` window I recorded two more numbers next to the device-level one, and turned on the same allocation recorder as in the first post, for both the trial and the real capture:

| | device | reserved | allocated |
|---|---:|---:|---:|
| trial capture | 630.0 MiB | 444.0 MiB | 404.6 MiB |
| real capture | 28.0 MiB | 28.0 MiB | 2.4 MiB |

Device is the free-memory drop vLLM uses; reserved is what PyTorch's allocator holds, including freed blocks; allocated is what live tensors use. Splitting the trial's 630 MiB:

| Part | Computed as | Size |
|---|---|---:|
| PyTorch allocations still alive | allocated | 404.6 MiB |
| freed blocks the allocator kept | reserved − allocated | 39.4 MiB |
| memory outside PyTorch | device − reserved | 186.0 MiB |

The recorder pointed straight at the largest piece:

```
394.0 MiB  [still alive at the end]
  vllm/v1/attention/backends/flashinfer.py:142  _get_trtllm_workspace_buffer
  vllm/v1/attention/backends/flashinfer.py:2474  forward
  ...
  vllm/v1/worker/gpu/cudagraph_utils.py:462  capture
```

A global FlashInfer workspace, allocated the first time attention runs a forward, which happens to be the trial capture. The 0.18 GiB left under Triton matched as well: the 186 MiB outside PyTorch plus 28 MiB of graphs. I was still reading all of it as "should not count as graph memory", until "still alive at the end" made me doubt the first conclusion.

## Does it stay?

A global workspace stays allocated for as long as the server runs. If so, it is not memory to leave out of the budget but memory that must be in it. I measured it: around the trial capture call in `gpu_worker.py`, with an `empty_cache()` on both sides, I recorded what was still on the device after the trial and its own teardown:

| left after the trial and its teardown | if transient | if it stays | measured |
|---|---:|---:|---:|
| device | about 394 MiB | about 580 MiB | 582 MiB |
| reserved | about 394 MiB | about 394 MiB | 396 MiB |
| outside PyTorch | about 0 | about 186 MiB | 186 MiB |

It all stays. The full account of the 0.62 GiB:

| The trial's 630 MiB | Size | Needed while serving? |
|---|---:|---|
| trtllm workspace and small allocations | 396 MiB | yes, stays resident |
| memory outside PyTorch | 186 MiB | yes, stays resident |
| the CUDA graphs themselves | about 28 MiB | yes, recaptured for real later |
| transient blocks freed during the capture | about 20 MiB | no |

The server needs 582 + 28 = 610 MiB. The estimate was 630. **The estimate is about 20 MiB high, roughly 3%.** The "actual" 0.03 GiB is small only because the workspace and the 186 MiB were already allocated by the trial, so the real capture never sees them.

### Where my first conclusion went wrong

"Turning off the estimate grew the KV cache and nothing broke" cannot tell these two cases apart:

| If the 0.59 GiB was... | after turning the estimate off |
|---|---|
| idle | KV grows, total stays within budget, server runs |
| really needed | KV grows, total goes about 0.6 GiB over budget, server runs |

Both raise process memory by about 630 MiB, and both serve without errors. Nothing ran out of memory only because utilization was 0.5 and half the card was free to absorb the overshoot. On a shared GPU or at a high utilization, the same setting would crowd out other services or OOM. **Not failing is not the same as being correct.** Memory changes have to be checked against the budget, not against whether the server stays up.

## So what is actually wrong

Not an overestimate. Three smaller and subtler things:

- The log compares two numbers that cover different things: the estimate includes first-use allocations, the "actual" does not.
- The accounting is fragile: the workspace lands in the budget only because it happens to be allocated for the first time during the trial. Move that allocation and nothing accounts for it.
- The real overcount is the roughly 20 MiB of transient blocks, not 0.59 GiB.

This decides the fix. Treat it as an overestimate and make the estimate measure only the graph pool, and the resident workspace drops out of the budget: the engine goes over its limit. Same symptom, opposite fix, and the log invites exactly the wrong reading.

## Upstream: someone walked the same path

Before opening a PR to fix the log, I checked its history and searched for duplicates. `git log -S "CUDA graph pool memory"` points to [#30515](https://github.com/vllm-project/vllm/pull/30515), which introduced the estimate and already treats the first capture as a fixed cost; only the comparison line ignores that. A general web search found nothing, but `gh search` in the repo with the exact log text found:

- [#59318](https://github.com/vllm-project/vllm/issues/59318): the same gap on GLM-5.3 across 4 × H200. The author first read it as an overestimate, then instrumented it and publicly corrected himself: most of it is retained memory the server really uses, mislabeled as graph memory.
- [#59368](https://github.com/vllm-project/vllm/pull/59368): the fix. It measures the graph pool directly, reserves the retained memory as its own line item, and makes the log compare pool with pool.

Same conclusion on different hardware, down to getting it wrong first. Without searching the repo, I would have opened a duplicate PR.

### Validating #59368 here

My setup captures only PIECEWISE graphs, so it isolates the part of #59318 that does not involve FULL-graph extrapolation. The PR applied to v0.32.0 with line offsets only:

| | v0.32.0 | with #59368 |
|---|---:|---:|
| graph pool, estimated / actual | 0.62 / 0.03 GiB (2150%) | 0.03 / 0.03 GiB (0.0%) |
| retained by the profiling pass | not shown | 0.57 GiB |
| total reserved for graphs | 0.62 GiB | 0.59 GiB |
| KV tokens | 112,096 | 112,496 |
| process memory after first full step | 8,226 MiB | 8,248 MiB |

The PR's 0.57 GiB retained matches the 582 MiB I measured by hand. The extra 400 tokens (about 22 MiB) are the transient blocks it no longer charges, and the process grows by the same amount, so nothing resident is dropped. I sent this, with the workspace stack and the Triton comparison, as a review on #59368.

One caveat for newer builds: v0.32.0 does not include [#61044](https://github.com/vllm-project/vllm/pull/61044), which creates cuBLAS handles before memory profiling, so part of the 186 MiB outside PyTorch may move into the profile run's consumed memory there.

## Takeaways

- Not failing is not the same as being correct. Check memory against the budget, not against whether the server stays up.
- Measure what stays, not just what was allocated. The post-teardown measurement turned "overestimate" into "accurate within 3%".
- The same symptom can call for opposite fixes. Find the cause before deciding what to change.
