---
title: "Table rendering test"
date: 2026-10-24
summary: "A draft that exercises every table case the build supports. Not for publishing."
draft: true
---

A test page for the GFM pipe-table support in `scripts/build.mjs`. It stays `draft: true`, so only `node scripts/build.mjs --drafts` renders it. The numbers below are made-up sample data.

## Numbers in several columns

Numeric columns are marked `---:` in the delimiter row, which right-aligns them with tabular figures. Nothing is detected automatically.

| Model | Params | TP | KV cache | Throughput (tok/s) | p99 latency (ms) |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 8.2B | 1 | 24.5 GB | 1,284 | 412 |
| Qwen3-32B | 32.8B | 2 | 38.0 GB | 611 | 958 |
| Llama-3.1-70B | 70.6B | 4 | 52.25 GB | 307.5 | 1,742 |
| Embedding-8B | 7.6B | 1 | 2.1 GB | 9,860 | 37 |

## Code and links inside cells

| Setting | Default | Where it is documented |
|---|---|---|
| `--gpu-memory-utilization` | `0.9` | [vLLM engine args](https://docs.vllm.ai/en/latest/serving/engine_args.html) |
| `--max-num-seqs` | `256` | **Scheduler** — see the [PagedAttention notes](../2026-07-vllm-paged-attention/) |
| `NCCL_P2P_DISABLE` | unset | A pipe inside a cell is escaped: `a \| b` |

## Three alignments

| Left | Center | Right |
|:---|:---:|---:|
| short | short | 1 |
| a longer left cell | a longer centred cell | 1,000 |
| *italic* | **bold** | 1,000,000 |

## A very wide table

On a phone this table scrolls inside its own box; the page itself does not scroll sideways.

| Node | GPU | Count | Driver | CUDA | NCCL | Interconnect | NIC | Storage | Kernel | Container runtime | Notes |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| worker-01 | RTX PRO 6000 Blackwell | 4 | 580.65.06 | 13.0 | 2.27.7 | PCIe Gen5 x16 | ConnectX-7 400GbE | local NVMe 7.68 TB | 6.8.0-60-generic | containerd 2.1.4 | sample row |
| worker-02 | B300 SXM | 8 | 580.65.06 | 13.0 | 2.27.7 | NVLink 5 / NVSwitch | ConnectX-8 800GbE | NFS over RDMA | 6.8.0-60-generic | containerd 2.1.4 | sample row |

> Blockquotes and h2 markers on this page also show the orange accent.
