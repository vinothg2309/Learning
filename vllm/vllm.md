vLLM.md
# vLLM: A Beginner's Guide to Fast and Efficient LLM Inference

## Table of Contents
- [What is vLLM?](#what-is-vllm)
- [Why vLLM Was Created](#why-vllm-was-created)
- [Understanding LLM Inference Basics](#understanding-llm-inference-basics)
- [Key Features and Innovations](#key-features-and-innovations)
  - [Automatic Prefix Caching (APC)](#automatic-prefix-caching-apc)
- [How vLLM Reduces Inference Time](#how-vllm-reduces-inference-time)
- [Practical Examples](#practical-examples)
- [Comparison with Alternatives](#comparison-with-alternatives)
- [Getting Started](#getting-started)
- [Resources and Next Steps](#resources-and-next-steps)
- [Production Deployment of vLLM](#production-deployment-of-vllm)
- [Disaggregated Prefill/Decode: Architecture, Scheduling & Kubernetes Configuration](#disaggregated-prefilldecode-architecture-scheduling--kubernetes-configuration)
- [vLLM Sleep Mode: Multi-Model Serving on a Single GPU](#vllm-sleep-mode-multi-model-serving-on-a-single-gpu)
  - [Overview](#overview)
  - [The Problem: Multi-Model Deployment Challenges](#the-problem-multi-model-deployment-challenges)
  - [How vLLM Sleep Mode Works](#how-vllm-sleep-mode-works)
  - [Benchmark: Model Switching Performance](#benchmark-model-switching-performance)
  - [Sleep Mode Implementation](#sleep-mode-implementation)
  - [Deployment Patterns](#deployment-patterns)
  - [Advantages & Limitations](#advantages--limitations)
  - [When to Use vLLM Sleep Mode](#when-to-use-vllm-sleep-mode)
  - [Comparison: Sleep Mode vs Alternatives](#comparison-sleep-mode-vs-alternatives)
  - [Production Considerations](#production-considerations)
  - [Future Work & Platform Integration](#future-work--platform-integration)
  - [Conclusion](#conclusion-1)
- [LLM Inferencing GPU Optimization: Document Processing Case Study](#llm-inferencing-gpu-optimization-document-processing-case-study)
  - [Overview](#overview-1)
  - [The GPU Utilization Paradox](#the-gpu-utilization-paradox)
  - [Understanding the Workload](#understanding-the-workload)
  - [Understanding the Bottleneck: Why LLM Inference is Memory-Bandwidth-Bound](#understanding-the-bottleneck-why-llm-inference-is-memory-bandwidth-bound)
  - [The Three Root Problems](#the-three-root-problems)
  - [Performance Baseline](#performance-baseline)
  - [Optimization Changes: Systematic Tuning](#optimization-changes-systematic-tuning)
  - [Optimization Results](#optimization-results)
  - [GPU Requirements: From Impossible to Achievable](#gpu-requirements-from-impossible-to-achievable)
  - [Cost Savings](#cost-savings-1)
  - [Optimization Playbook for Document Processing Workloads](#optimization-playbook-for-document-processing-workloads)
  - [Key Learnings](#key-learnings)
  - [Production Deployment Checklist](#production-deployment-checklist)
  - [Conclusion](#conclusion-2)
- [vLLM Production Stack: Enterprise-Grade Serving](#vllm-production-stack-enterprise-grade-serving)
  - [Overview](#overview-2)
  - [What Problems Does It Solve?](#what-problems-does-it-solve)
  - [Core Components](#core-components)
  - [Production Stack Architecture](#production-stack-architecture)
  - [Deployment Options](#deployment-options)
  - [Key Features Summary](#key-features-summary)
  - [Getting Started with Production Stack](#getting-started-with-production-stack)
  - [Common Use Cases](#common-use-cases)
  - [Production Readiness Checklist](#production-readiness-checklist)
  - [Conclusion](#conclusion-3)

---

## What is vLLM?

**vLLM** (Virtual Large Language Model) is a fast and easy-to-use library for LLM inference and serving. Developed by researchers at UC Berkeley, it's designed to maximize the throughput and efficiency of serving large language models like GPT, LLaMA, Mistral, and others.

### Key Benefits at a Glance:
- **2-4x faster** throughput compared to traditional inference methods
- **Significantly reduced memory usage** (up to 50-75% less)
- **Easy integration** with existing LLM models from HuggingFace
- **Production-ready** with OpenAI-compatible API server
- **State-of-the-art performance** for both single and batched requests

---

## Why vLLM Was Created

When deploying large language models in production, developers face several critical challenges:

### Problem 1: Memory Inefficiency
Traditional LLM inference requires storing "KV cache" (key-value pairs from the attention mechanism) for each token. This cache grows with sequence length and quickly consumes GPU memory, limiting how many requests you can handle simultaneously.

**Analogy:** Imagine a library where each book (request) must reserve a full shelf from start to finish, even if it only uses a few books. Most of the shelf space sits empty, wasted.

### Problem 2: Low GPU Utilization
Most inference frameworks process requests in fixed batches, meaning the GPU sits idle waiting for all requests in a batch to complete before starting new ones.

**Analogy:** It's like a bus that must wait at the station until all passengers finish their trips before picking up new passengers—very inefficient!

### Problem 3: Slow Throughput
These memory and utilization issues combine to create a bottleneck: you can't serve many users simultaneously, and each request takes longer than necessary.

**vLLM was created to solve these problems with innovative algorithms and optimizations.**

---

## Understanding LLM Inference Basics

Before diving into vLLM's innovations, let's understand how LLM inference works:

### The Token Generation Process

```
Input: "The capital of France is"
     ↓
[LLM processes input]
     ↓
Output: "The capital of France is Paris"
```

LLMs generate text one token (word piece) at a time:
1. Process the input prompt
2. Predict the next token
3. Add that token to the sequence
4. Repeat until completion

### The KV Cache Challenge

For each token generated, the model creates **key-value pairs** used in the attention mechanism. These must be stored and reused for subsequent tokens.

**Memory requirement:** `num_layers × num_tokens × hidden_size × 2 (key + value)`

For a 13B parameter model with 2048 tokens, this can consume **several gigabytes of GPU memory per request!**

---

## Memory Requirements

### Basic Formula

```
Total GPU Memory = Model Weights + KV Cache + Overhead (20%)
```

### Model Weight Memory

```
Memory = Number of Parameters × Bytes per Parameter

Data Types:
- FP16 (standard): 2 bytes/param → 7B model = 14 GB
- INT8 (quantized): 1 byte/param → 7B model = 7 GB
- INT4 (quantized): 0.5 bytes/param → 7B model = 3.5 GB
```

### KV Cache Memory

Traditional approach: **~1 GB per request** (for 7B model, 512 tokens)
vLLM with PagedAttention: **~0.25 GB per request** (4x reduction!)

### Quick Reference

| Model | FP16 Weights | Recommended GPU | Max Concurrent Users (vLLM) |
|-------|--------------|-----------------|----------------------------|
| 7B | 14 GB | RTX 3090 (24GB) | 30-40 |
| 13B | 26 GB | A100 (40GB) | 40-60 |
| 70B | 140 GB | 2× A100 (80GB) | 50-80 |

### Example: LLaMA-7B on RTX 3090 (24 GB)

```
Model weights: 14 GB
KV cache (32 users): 8 GB
Overhead: 2 GB
Total: 24 GB ✓ Perfect fit!
```

### If You Don't Have Enough Memory

**Option 1: Quantization** (reduce model size)
```python
llm = LLM(model="llama-7b", quantization="awq")  # 14GB → 4GB
```

**Option 2: Multi-GPU** (split across GPUs)
```python
llm = LLM(model="llama-70b", tensor_parallel_size=4)  # Use 4 GPUs
```

**Option 3: Reduce limits**
```python
llm = LLM(
    model="llama-7b",
    max_model_len=2048,      # Shorter sequences
    max_num_seqs=16          # Fewer concurrent requests
)
```

### Monitor Memory

```bash
nvidia-smi  # Check GPU memory usage
```

**Key takeaway:** vLLM's PagedAttention uses 4x less KV cache memory, letting you serve 4x more users on the same hardware!

---

## Key Features and Innovations

### 1. PagedAttention: The Core Innovation

**PagedAttention** is vLLM's breakthrough algorithm, inspired by virtual memory and paging in operating systems.

#### Traditional Approach:
```
Request 1: [████████████░░░░░░░░] ← Contiguous memory, wasted space
Request 2: [██████░░░░░░░░░░░░░░] ← More wasted space
Request 3: [████████████████░░░░] ← Even more waste
```
Problems: Memory fragmentation, pre-allocation, inflexibility

#### PagedAttention Approach:
```
Request 1: [████][████][████]    ← Non-contiguous blocks
Request 2: [███][███]            ← Exactly what's needed
Request 3: [████][████][████][█] ← Grows dynamically
```
Benefits: No waste, flexible, efficient sharing

#### How It Works:

1. **Divide KV cache into blocks** (pages) instead of requiring contiguous memory
2. **Allocate blocks on-demand** as the sequence grows
3. **Share blocks** between requests (useful for parallel sampling/beam search)
4. **Eliminate fragmentation** by using non-contiguous memory

**Memory Savings:** Up to 4x reduction in memory usage!

**Analogy:** Instead of reserving an entire parking lot for each car (traditional), PagedAttention is like a valet service that parks cars efficiently in available spots, even if they're not next to each other.

### 2. Continuous Batching

Traditional inference uses **static batching**—wait for a batch to fill, process all requests, then start over.

vLLM uses **continuous batching** (also called iteration-level batching):

```
Traditional Static Batching:
Time: [Batch 1 processing........] [Batch 2 processing........]
      └─ GPU idle while waiting ─┘

Continuous Batching:
Time: [Req1█Req2█Req3█Req4█Req5█Req6█Req7█...]
      └─ New requests added immediately as others complete ─┘
```

**Benefits:**
- No waiting for batches to fill
- GPU stays busy continuously
- Lower latency for individual requests
- Higher overall throughput

### 3. Optimized CUDA Kernels

vLLM implements custom, highly optimized CUDA kernels specifically designed for:
- PagedAttention operations
- Efficient memory access patterns
- Reduced kernel launch overhead
- Fused operations to minimize data movement

**Result:** Faster computation at the hardware level.

### 4. Tensor Parallelism

For very large models that don't fit on a single GPU:

```
Single GPU: [13B model - doesn't fit!]

Tensor Parallelism:
GPU 1: [Layers 1-20]
GPU 2: [Layers 21-40]
GPU 3: [Layers 41-60]
GPU 4: [Layers 61-80]
```

vLLM automatically distributes the model across multiple GPUs for seamless scaling.

### 5. Automatic Prefix Caching (APC)

**APC** reuses the KV cache of a previously computed prompt prefix across requests — so if two requests share the same prefix (e.g., same system prompt), vLLM computes it **once** and reuses it for all subsequent requests.

#### How It Works

```
Request 1:  [System Prompt (500 tokens)] + [User Question A]
            └── KV cache computed and stored ──┘

Request 2:  [System Prompt (500 tokens)] + [User Question B]
            └── KV cache REUSED ──┘         └── only this computed
                (0 computation cost)
```

The prefix is hashed and stored in GPU memory. On every new request, vLLM checks if the prefix hash exists — if yes, it skips recomputing it entirely.

#### When APC Helps Most

```
Scenario                            Cache Hit?
─────────────────────────────────   ──────────
RAG with same document context      ✅ High — same docs, different questions
Multi-turn chat (same system prompt)✅ High — system prompt reused every turn
Few-shot prompts (fixed examples)   ✅ High — examples never change
Unique prompts per request          ❌ None  — no shared prefix
```

#### Enable APC in vLLM

```python
from vllm import LLM

# Enable at server startup
llm = LLM(
    model="meta-llama/Llama-3-8b",
    enable_prefix_caching=True      # ← one flag to enable APC
)
```

```bash
# Or via CLI server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b \
    --enable-prefix-caching
```

#### Does APC Reduce Cost?

**Yes — for self-hosted vLLM**, APC reduces GPU compute directly:

```
Without APC:
  100 requests × 500-token system prompt = 50,000 tokens computed
  GPU time: ~5 seconds per batch

With APC:
  First request  → 500 tokens computed, cached
  Next 99 requests → 0 tokens computed for prefix
  GPU time: <1 second per batch
  Savings: ~80-90% less compute for the prefix portion
```

**For managed APIs (OpenAI, Anthropic)**, they run their own prefix caching server-side — you pay less per cached token automatically. vLLM's APC is the **self-hosted equivalent** of that.

| Setup | Cost Impact |
|-------|-------------|
| Self-hosted vLLM + APC | Fewer GPU hours → lower infrastructure cost |
| OpenAI / Anthropic APIs | Cached input tokens billed at 50–90% discount |
| vLLM without APC | Full KV recomputed every request — wasted GPU |

#### APC vs Prompt Caching (API-level)

```
vLLM APC                            API Prompt Caching (OpenAI/Anthropic)
────────────────────────────────    ──────────────────────────────────────
Self-hosted GPU memory cache        Provider-side server cache
Automatic — no code change          Requires cache_control flag (Anthropic)
Saves GPU compute                   Saves API token cost (50-90% discount)
Works on any repeated prefix        Works on marked stable context blocks
```

> **One-liner:** APC is "don't recompute what you've already computed." Same system prompt used 1000 times? Compute once, cache, reuse — saving up to 90% of GPU work for the prefix.

---

### Managed API Prompt Caching — TTL & Cache Key

#### Can You Change the TTL?

**No** — TTL is fixed by the provider. You cannot configure it.

```
Anthropic:  5 minutes    (resets on every cache hit)
OpenAI:     5–10 minutes (non-deterministic, resets on cache hit)
Gemini:     1 hour       (fixed)
```

The only way to extend effective TTL is to **keep hitting the cache** before it expires:

```python
# Warm the cache proactively — hit every 4 min for a 5-min TTL
async def keep_cache_warm(prompt, interval_seconds=240):
    while True:
        client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1,                   # minimal output — just to reset TTL
            system=[{"type": "text", "text": prompt,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "ping"}]
        )
        await asyncio.sleep(interval_seconds)
```

> Only worth it for very high-traffic systems with a shared fixed context (e.g., a common RAG document).

---

#### How Does Managed API Identify the Same Request?

Both providers use **exact prefix hashing** — not semantic similarity.

```
Cache key = hash(model + prompt_prefix_text)

✅ Identical model
✅ Identical text, character-for-character, up to the cached block
✅ Same message order
❌ NOT semantic — "Hello" ≠ "Hi" even if meaning is the same
❌ NOT parameter-based — temperature/max_tokens do NOT affect cache key
```

**Anthropic** — caches the exact content of the block tagged `cache_control`:

```python
# WRITE — first request
system=[
    {"type": "text", "text": "You are a helpful assistant."},
    {"type": "text", "text": "<10K RAG document>",
     "cache_control": {"type": "ephemeral"}}   # this block is hashed & cached
]

# HIT — identical block
system=[
    {"type": "text", "text": "You are a helpful assistant."},
    {"type": "text", "text": "<10K RAG document>",           # exact same text ✅
     "cache_control": {"type": "ephemeral"}}
]

# MISS — one character different
system=[
    {"type": "text", "text": "You are a helpful assistant."},
    {"type": "text", "text": "<10K RAG document> ",          # trailing space ❌
     "cache_control": {"type": "ephemeral"}}
]
```

**OpenAI** — automatically caches the longest matching prefix (≥1024 tokens), no marking needed:

```python
# HIT — same prefix, different user message
messages = [
    {"role": "system", "content": "<5000 token context>"},  # prefix cached ✅
    {"role": "user",   "content": "Different question"}     # ignored for cache key
]

# MISS — prefix changed
messages = [
    {"role": "system", "content": "<5000 token context v2>"},  # prefix changed ❌
    {"role": "user",   "content": "Same question"}
]

# Check cache hit in response
print(response.usage.prompt_tokens_details.cached_tokens)
```

**What breaks the cache (both providers):**
```
❌ Any character change in the cached prefix
❌ Extra whitespace or newline added
❌ Message order changed before the cached block
❌ Different model used
✅ Content AFTER the cached prefix changed → still a HIT
✅ temperature / max_tokens changed → still a HIT
```

---

#### Caching Cost vs Normal API Cost

| Token Type | Anthropic (Claude 3.5 Sonnet) | OpenAI (GPT-4o) |
|------------|-------------------------------|-----------------|
| Normal input | $3.00 / 1M | $2.50 / 1M |
| Cache write | $3.75 / 1M (+25%) | $2.50 / 1M (same) |
| **Cache read** | **$0.30 / 1M (−90%)** | **$1.25 / 1M (−50%)** |
| Output | $15.00 / 1M | $10.00 / 1M |

```
Example: 10,000-token system prompt × 100 requests

Anthropic without caching: 100 × 10K × $3.00/M   = $3.00
Anthropic with caching:    1 write + 99 reads      = $0.04 + $0.30 = $0.34  (89% savings)

OpenAI without caching:   100 × 10K × $2.50/M    = $2.50
OpenAI with caching:      1 normal + 99 cached    = $0.025 + $0.62 = $0.645 (74% savings)
```

---

#### Provider Comparison Summary

| | **Anthropic** | **OpenAI** |
|---|---|---|
| Enable | `cache_control` flag required | Automatic (≥1024 tokens) |
| TTL | 5 min, resets on hit | 5–10 min, resets on hit |
| Cache write cost | +25% vs normal | Same as normal |
| Cache read discount | **−90%** | **−50%** |
| Cache key | Hash of marked block | Hash of longest prefix |
| Breaks on | Any char change in marked block | Any change in prefix tokens |
| Safe to change | Content after marked block | Content after cached prefix |

> **Rule:** Managed API caches are **exact-match, prefix-based**. Treat the cached portion like an immutable constant — any edit, even a space, is a full cache miss and full recompute charge.

---

### 6. Streaming Output

Generate and send tokens as they're produced, rather than waiting for the entire response:

```
Traditional: [Wait............] "The complete response appears"

Streaming:   "The" "complete" "response" "appears" "word" "by" "word"
             ↑ Each token sent immediately
```

**Benefit:** Better user experience with lower perceived latency.

---

## How vLLM Reduces Inference Time

### 1. Memory Efficiency = More Parallelism

By reducing memory usage by 4x, you can:
- **Run 4x more requests simultaneously**
- **Use larger models** on the same hardware
- **Reduce costs** by needing fewer GPUs

**Example:**
- Traditional approach: 8 concurrent requests on an A100 GPU
- vLLM: 32 concurrent requests on the same GPU
- **Result: 4x throughput improvement**

### 2. Continuous Batching Eliminates Idle Time

```
Efficiency Comparison:

Static Batching:
Utilization: ▓▓▓░░░▓▓▓░░░▓▓▓░░░ (60% GPU utilization)

Continuous Batching:
Utilization: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (95% GPU utilization)
```

**Result:**
- 50-60% improvement in GPU utilization
- Faster processing for all requests

### 3. Optimized Memory Access

PagedAttention's block-based design improves cache locality and memory access patterns:

- **Fewer cache misses**
- **More efficient memory bandwidth usage**
- **Reduced memory transfer overhead**

### 4. Efficient Request Scheduling

vLLM's scheduler intelligently:
- Prioritizes requests based on resource availability
- Preempts and swaps requests when needed
- Maximizes batch sizes dynamically

### Performance Numbers

Based on real-world benchmarks:

| Metric | HuggingFace Transformers | vLLM | Improvement |
|--------|-------------------------|------|-------------|
| Throughput (requests/sec) | 0.5 | 2.0 | **4x faster** |
| Memory per request | 4 GB | 1 GB | **4x less** |
| GPU utilization | 40-60% | 85-95% | **~1.5x better** |
| Latency (per token) | 50ms | 30ms | **1.7x faster** |

*Example: LLaMA-13B on A100 GPU, average sequence length 256 tokens*

---

## Practical Examples

### Example 1: Basic Usage

```python
from vllm import LLM, SamplingParams

# Initialize the model
llm = LLM(model="facebook/opt-125m")

# Define prompts
prompts = [
    "The capital of France is",
    "The meaning of life is",
]

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Generate responses
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}\n")
```

### Example 2: Serving with OpenAI-Compatible API

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-6.7b \
    --tensor-parallel-size 2

# Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

response = client.completions.create(
    model="facebook/opt-6.7b",
    prompt="San Francisco is a",
    max_tokens=50,
)
print(response.choices[0].text)
```

### When to Use vLLM

**Use vLLM when:**
- You need to serve LLMs in production with high throughput
- You want to maximize GPU utilization and reduce costs
- You're handling many concurrent requests
- You need low-latency inference
- You want an easy-to-use, production-ready solution

**Consider alternatives when:**
- You're doing research and need maximum flexibility
- You need very specific custom modifications to model internals
- You're running on CPU-only environments (vLLM is GPU-optimized)
- You have very specialized quantization requirements

---

## Comparison with Alternatives

### vLLM vs HuggingFace Transformers

| Feature | HuggingFace Transformers | vLLM |
|---------|------------------------|------|
| **Ease of use** | ⭐⭐⭐⭐⭐ Very simple | ⭐⭐⭐⭐ Simple for inference |
| **Throughput** | ⭐⭐ Baseline | ⭐⭐⭐⭐⭐ 4x faster |
| **Memory efficiency** | ⭐⭐ Standard | ⭐⭐⭐⭐⭐ 4x better |
| **Production ready** | ⭐⭐⭐ Requires work | ⭐⭐⭐⭐⭐ Built for it |
| **Flexibility** | ⭐⭐⭐⭐⭐ Full control | ⭐⭐⭐⭐ Optimized path |

**Best for:** HF for research/experimentation, vLLM for production serving

### vLLM vs TensorRT-LLM

| Feature | TensorRT-LLM | vLLM |
|---------|-------------|------|
| **Setup complexity** | ⭐⭐ Complex (compilation) | ⭐⭐⭐⭐⭐ Simple |
| **Performance** | ⭐⭐⭐⭐⭐ Fastest | ⭐⭐⭐⭐ Very fast |
| **Model support** | ⭐⭐⭐ Limited | ⭐⭐⭐⭐⭐ Extensive |
| **Flexibility** | ⭐⭐ Compilation needed | ⭐⭐⭐⭐ Dynamic |
| **Hardware** | ⭐⭐⭐ NVIDIA only | ⭐⭐⭐⭐ Broader support |

**Best for:** TensorRT-LLM for maximum performance (NVIDIA), vLLM for ease of use and flexibility

### vLLM vs Text Generation Inference (TGI)

| Feature | TGI (HuggingFace) | vLLM |
|---------|-------------------|------|
| **Performance** | ⭐⭐⭐⭐ Very good | ⭐⭐⭐⭐⭐ Excellent |
| **Memory efficiency** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Better (PagedAttention) |
| **Ease of deployment** | ⭐⭐⭐⭐⭐ Docker-first | ⭐⭐⭐⭐ Python-first |
| **API compatibility** | ⭐⭐⭐⭐ Custom API | ⭐⭐⭐⭐⭐ OpenAI compatible |
| **Batching** | ⭐⭐⭐⭐ Continuous | ⭐⭐⭐⭐⭐ Continuous |

**Best for:** Both are excellent for production; vLLM has edge in memory efficiency

---

## Getting Started

### Installation

```bash
# Install vLLM
pip install vllm

# For the latest features (from source)
pip install git+https://github.com/vllm-project/vllm.git
```

### Quick Start Checklist

1. **Choose your model** from HuggingFace (e.g., LLaMA, Mistral, GPT-J)
2. **Estimate GPU memory** needs (model size + KV cache)
3. **Install vLLM** via pip
4. **Run basic inference** with sample code
5. **Deploy API server** for production use
6. **Monitor performance** and adjust parameters

### Common Configuration Options

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,      # Use 2 GPUs
    max_num_seqs=128,            # Max concurrent sequences
    max_model_len=4096,          # Max sequence length
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
)
```

---

## Resources and Next Steps

### Official Resources
- **Documentation:** [https://docs.vllm.ai](https://docs.vllm.ai)
- **GitHub Repository:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Paper:** "Efficient Memory Management for Large Language Model Serving with PagedAttention"

### Community and Support
- **Discord:** Join the vLLM community Discord for support
- **GitHub Issues:** Report bugs or request features
- **Examples:** Browse the examples/ directory in the GitHub repo

### Further Learning

1. **Understand attention mechanisms** in transformers (foundational knowledge)
2. **Study the PagedAttention paper** for deep technical details
3. **Experiment with different models** to see performance differences
4. **Read the vLLM blog** for updates and optimization tips
5. **Join the community** to learn from others' experiences

### Advanced Topics to Explore
- Custom sampling methods
- Speculative decoding
- Quantization with vLLM (INT8, INT4)
- Multi-node distributed serving
- Custom request scheduling policies

---

## Summary

**vLLM is a game-changer for LLM inference** because it:

1. **Solves memory problems** with PagedAttention (4x more efficient)
2. **Maximizes GPU usage** with continuous batching (95% utilization)
3. **Delivers fast throughput** (2-4x faster than alternatives)
4. **Easy to use** with simple Python API and OpenAI compatibility
5. **Production-ready** from day one

Whether you're building a chatbot, API service, or research platform, vLLM provides the performance and efficiency needed to serve large language models at scale.

**Start building today and experience the difference!**

---

## Production Deployment of vLLM

### Do You Need Docker to Run vLLM in Production?

> **Beginner context:**
> Think of Docker as a "shipping container" for your software. Just like a shipping container can carry any goods and be loaded on any ship, Docker packages your app + all its dependencies into a single box that runs the same way everywhere — your laptop, a cloud server, Kubernetes. Without Docker, your app might work on your machine but break on the server because the CUDA version is different, or Python packages conflict.

**Short answer: Yes — Docker is the standard for production. Not strictly required, but strongly recommended.**

```
Without Docker (bare metal):
  pip install vllm
  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B
  → Works, but: no isolation, no reproducibility, hard to scale, risky in prod

With Docker:
  docker run vllm/vllm-openai --model meta-llama/Llama-3-8B
  → Isolated, reproducible, K8s-compatible, scalable ✓
```

**Why Docker in production:**

| Reason | Explanation |
|---|---|
| **Isolation** | vLLM + CUDA deps don't conflict with other services |
| **Reproducibility** | Same image = same behaviour on any node |
| **K8s compatibility** | K8s only runs containers — Docker is mandatory |
| **GPU driver isolation** | Container includes CUDA runtime, only kernel driver needed on host |
| **Rollback** | Tag images by version — instant rollback on failure |

---

### One Model = One Container (Always)

> **Beginner context:**
> Imagine each LLM as a large restaurant kitchen. One kitchen can only cook one type of cuisine at a time — the equipment is fully occupied. If you want to serve Italian AND Chinese food simultaneously, you need two separate kitchens. Same with vLLM: one model fills the entire GPU memory, so each model needs its own dedicated container with its own dedicated GPUs.

**Yes — each model runs as a separate container.** This is intentional, not a limitation.

```
WRONG mental model:
  One vLLM container → serves multiple models
  ✗ vLLM loads one model into GPU memory at startup
  ✗ Loading a second model would require evicting the first

CORRECT model:
  Container A: vLLM + Llama-3-70B  → GPU 0,1,2,3  (tensor parallel across 4 GPUs)
  Container B: vLLM + Mistral-7B   → GPU 4        (single GPU)
  Container C: vLLM + CodeLlama    → GPU 5        (single GPU)

Each container owns its GPUs exclusively.
A load balancer routes requests to the right container by model name.
```

```
                  ┌─────────────────────────────────┐
                  │      Load Balancer / Gateway      │
                  └───────┬─────────────┬────────────┘
                          │             │
              ┌───────────▼──┐    ┌─────▼──────────┐
              │  Container A  │    │  Container B    │
              │  Llama-3-70B  │    │  Mistral-7B     │
              │  GPU: 0,1,2,3 │    │  GPU: 4         │
              │  Port: 8000   │    │  Port: 8001     │
              └───────────────┘    └────────────────┘
```

---

### Sample Docker Setup for Kubernetes

> **Beginner context — What is Kubernetes (K8s)?**
> Kubernetes is a system that manages containers at scale. Instead of manually starting Docker containers on servers, you describe what you want ("I want 3 copies of this vLLM container always running, each with 1 GPU") and K8s makes it happen — restarting crashed containers, spreading load, scheduling on GPU nodes. Key objects:
> - **Deployment** — "run N copies of this container" — K8s keeps them alive
> - **Service** — a stable network address that routes traffic to your pods (since pod IPs change on restart)
> - **Secret** — stores sensitive values (API keys, tokens) safely — never hardcode in YAML
> - **PVC** (PersistentVolumeClaim) — a shared disk that multiple pods can read from (used to cache model weights)

#### Dockerfile

```dockerfile
# vLLM provides official images — no need to build from scratch
# Use official image as base, only add custom configs if needed
FROM vllm/vllm-openai:latest

# Optional: copy custom tokenizer configs or prompt templates
COPY custom_chat_template.jinja /app/

EXPOSE 8000

# Entrypoint is already set in official image to start the API server
# Override via K8s args
```

#### Kubernetes Deployment

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
spec:
  replicas: 2                              # 2 pods = 2 model replicas
  selector:
    matchLabels:
      app: vllm-llama3-8b
  template:
    metadata:
      labels:
        app: vllm-llama3-8b
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Meta-Llama-3-8B-Instruct"
        - "--tensor-parallel-size"
        - "1"                              # 1 GPU per pod
        - "--gpu-memory-utilization"
        - "0.90"
        - "--max-num-seqs"
        - "256"                            # max concurrent requests
        - "--port"
        - "8000"
        ports:
        - containerPort: 8000
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
        resources:
          limits:
            nvidia.com/gpu: 1              # 1 GPU per pod
            memory: "24Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "20Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60          # model loading takes time
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
      nodeSelector:
        nvidia.com/gpu: "true"             # only schedule on GPU nodes

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-8b-svc
spec:
  selector:
    app: vllm-llama3-8b
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
# HuggingFace token secret
apiVersion: v1
kind: Secret
metadata:
  name: hf-secret
type: Opaque
stringData:
  token: "hf_your_token_here"
```

#### Model Weights as Persistent Volume (avoid re-downloading on every pod start)

```yaml
# PersistentVolumeClaim — shared model cache across pods
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
  - ReadWriteMany                          # multiple pods read same cache
  resources:
    requests:
      storage: 100Gi                       # Llama-3-70B = ~140GB, 8B = ~16GB
  storageClassName: standard-rwx          # GKE Filestore or AWS EFS

# Add to pod spec:
volumes:
- name: model-cache
  persistentVolumeClaim:
    claimName: model-cache-pvc

volumeMounts:
- name: model-cache
  mountPath: /root/.cache/huggingface     # HF cache dir — model downloaded once, shared
```

---

### vLLM Metrics: num_requests_waiting and num_requests_running

> **Beginner context:**
> Think of vLLM as a busy restaurant. The GPU is the kitchen. `num_requests_running` = number of orders the kitchen is actively cooking right now. `num_requests_waiting` = number of orders sitting on the ticket rail waiting for a free chef. If the ticket rail is empty, the kitchen is humming smoothly. If tickets are piling up, you need more kitchens (more pods).
>
> These metrics come from **Prometheus** — a time-series database that periodically scrapes (pulls) numbers from your services. **Grafana** then visualizes them as dashboards. **Alertmanager** fires alerts when thresholds are breached.

vLLM exposes Prometheus metrics at `/metrics`. Two of the most critical:

```
GET http://vllm-service:8000/metrics

# HELP vllm:num_requests_running Number of requests currently being processed
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 12

# HELP vllm:num_requests_waiting Number of requests waiting in queue
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 47
```

**`vllm:num_requests_running`** — requests actively using GPU right now:

```
Running = request has been scheduled → tokens being generated → GPU is working

max_num_seqs = 256   (your configured limit)
num_requests_running = 200

→ 200 requests sharing GPU memory via PagedAttention
→ GPU is busy — good utilization
→ 56 slots still available before hitting the limit
```

**`vllm:num_requests_waiting`** — requests sitting in queue, not yet scheduled:

```
Waiting = request arrived but no KV cache slots available yet
         → held in CPU memory queue until a running request finishes

num_requests_waiting = 47

→ 47 requests stuck waiting for GPU slots
→ users experiencing latency
→ time to scale up pods or increase max_num_seqs
```

**Reading them together:**

```
HEALTHY:
  running = 180  (high — GPU busy)
  waiting = 0    (no queue — requests served immediately)

SATURATED (scale up):
  running = 256  (at max_num_seqs limit)
  waiting = 50+  (queue growing — requests waiting)

UNDERUTILIZED (scale down):
  running = 10   (low — GPU mostly idle)
  waiting = 0    (no pressure)
```

**Grafana alert rule:**
```yaml
# Alert when queue builds up — time to scale
- alert: VllmQueueBuildup
  expr: vllm:num_requests_waiting > 20
  for: 2m
  annotations:
    summary: "vLLM queue building up — scale pods or increase max_num_seqs"
```

---

### KEDA and KServe for vLLM Autoscaling

> **Beginner context — What is autoscaling?**
> Autoscaling = automatically adding or removing pods based on traffic. Instead of you manually running `kubectl scale deployment vllm --replicas=5`, the system watches metrics and scales itself.
>
> K8s has a built-in autoscaler called **HPA** (Horizontal Pod Autoscaler). But HPA only watches CPU and memory — useless for LLMs where the GPU queue is what actually matters.
>
> That's where **KEDA** and **KServe** come in.

#### What is KEDA?

> **Beginner context:**
> KEDA (Kubernetes Event-Driven Autoscaler) is a plugin that extends K8s HPA to scale on **any metric** — Prometheus, Kafka queue length, Redis list size, RabbitMQ message count, etc. You install KEDA in your cluster once, then create a `ScaledObject` that says "watch this Prometheus metric, scale this Deployment when it crosses a threshold."

**KEDA** = Kubernetes Event-Driven Autoscaler. It scales pods based on **custom metrics** — not just CPU/memory like standard HPA.

```
Standard HPA:                      KEDA:
  Scale on CPU > 70%                Scale on vllm:num_requests_waiting > 10
  → CPU is lagging indicator        → Queue length is direct pressure signal
  → by the time CPU is high,        → Scale before users feel latency
    users already waiting
```

```
┌──────────────┐     scrapes metrics     ┌────────────────┐
│  Prometheus  │ ◄─────────────────────  │  vLLM /metrics │
└──────┬───────┘                         └────────────────┘
       │  vllm:num_requests_waiting
       ▼
┌──────────────┐     scale trigger       ┌────────────────┐
│    KEDA      │ ──────────────────────► │  K8s Deployment│
│  ScaledObject│  "waiting > 10 → +1 pod"│  (vLLM pods)   │
└──────────────┘                         └────────────────┘
```

**KEDA ScaledObject for vLLM:**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-llama3-8b                  # deployment to scale
  minReplicaCount: 1                      # always keep 1 warm pod
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_requests_waiting
      query: vllm:num_requests_waiting{deployment="vllm-llama3-8b"}
      threshold: "10"                     # add pod when queue > 10 requests
```

**Scale-down problem with LLMs:**
```
LLM pods take 60–120s to start (model loading).
Scale-down must be gradual — don't kill a pod mid-generation.

Solution:
  cooldownPeriod: 300   # wait 5min before scaling down
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
    - type: Pods
      value: 1          # remove at most 1 pod at a time
      periodSeconds: 120
```

---

#### What is KServe?

> **Beginner context:**
> KServe is to model serving what Heroku is to web apps — it hides the infrastructure complexity. Without KServe, deploying a model requires writing ~10 YAML files (Deployment, Service, Ingress, HPA, KEDA, PVC, Secret, etc.) and wiring them together. With KServe, you write **one YAML** called an `InferenceService` and KServe creates all the rest automatically.
>
> KServe is built on top of Knative (for scale-to-zero serverless) and Istio (for traffic routing). It's the standard for ML model serving on Kubernetes.

**KServe** is a Kubernetes-native model serving platform — it handles the full lifecycle: deploy, scale, monitor, version models. Think of it as "managed vLLM on K8s."

```
WITHOUT KServe (manual):
  You write: Deployment + Service + Ingress + HPA + KEDA + PVC + Secret
  You manage: rollouts, canary, A/B testing, monitoring, scaling
  → ~10 YAML files per model, all maintained by your team

WITH KServe:
  You write: one InferenceService YAML
  KServe manages: everything above automatically
```

**KServe InferenceService for vLLM:**

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-8b
spec:
  predictor:
    model:
      modelFormat:
        name: vllm                        # KServe knows vLLM natively
      storageUri: "hf://meta-llama/Meta-Llama-3-8B-Instruct"
      resources:
        limits:
          nvidia.com/gpu: "1"
          memory: 24Gi
      args:
      - "--max-num-seqs=256"
      - "--gpu-memory-utilization=0.90"
    scaleTarget: 10                       # target: 10 concurrent requests per pod
    scaleMetric: "concurrency"            # scale on request concurrency
    minReplicas: 1
    maxReplicas: 8
```

**What KServe gives you automatically:**

| Feature | What it does |
|---|---|
| **Canary rollout** | Route 10% traffic to new model version, 90% to old |
| **Autoscaling** | Scale-to-zero when idle (cost saving), scale up on traffic |
| **Multi-model serving** | Route by model name to right container |
| **gRPC + REST** | Both protocols out of the box |
| **Metrics + tracing** | Prometheus + Jaeger integrated |
| **Storage initializer** | Downloads model from HF/S3/GCS before pod starts |

---

### Productionizing vLLM with Autoscaling — Full Architecture

> **Beginner context — The big picture:**
> Production autoscaling works in 3 layers. Think of a restaurant chain:
> - **Layer 1 (KEDA — Pod scaling):** The shift manager sees too many orders → calls in more chefs (adds pods). Queue drops → sends chefs home (removes pods). Happens in minutes.
> - **Layer 2 (Cluster Autoscaler — Node scaling):** The restaurant is too small for more chefs → opens a new location (provisions a new GPU server). Slower — takes 3–5 min.
> - **Layer 3 (PVC — Model cache):** New chefs don't need to re-train from scratch — they use the shared recipe book (model weights on shared disk). Saves 10+ minutes of re-download time.
>
> All 3 layers work together automatically. You configure them once and the system self-manages.

```
Internet
    │
    ▼
Ingress / API Gateway (Kong / NGINX)
    │  rate limiting, auth, routing by model name
    ▼
Kubernetes Services (one per model)
    │
    ├──► vllm-llama3-8b   (pods: 1–10, GPU: 1 each)
    ├──► vllm-mistral-7b  (pods: 1–5,  GPU: 1 each)
    └──► vllm-codellama   (pods: 1–3,  GPU: 1 each)
              │
              │  /metrics → Prometheus
              ▼
         KEDA ScaledObject
         (watches vllm:num_requests_waiting)
         adds/removes pods based on queue depth
              │
              ▼
         GPU Node Pool (GKE Autopilot / EKS Karpenter)
         auto-provisions GPU nodes when pods can't be scheduled
```

**Full autoscaling stack:**

```
Layer 1 — Pod scaling (KEDA):
  Metric: vllm:num_requests_waiting > 10 → add pod
  Cooldown: 5min before scale-down
  New pod startup: 60–120s (model loading from PVC cache)

Layer 2 — Node scaling (Cluster Autoscaler / Karpenter):
  Pod can't schedule (no GPU node available) → provision new GPU node
  Node provision time: 3–5min (cloud instance + driver init)
  → Pre-warm nodes to avoid cold start: keep min 1 node always running

Layer 3 — Model cache (PVC):
  Model weights on shared PersistentVolume (ReadWriteMany)
  New pod mounts PVC → no re-download from HuggingFace
  Startup: 60s (load from disk) vs 600s (download from internet)
```

**Complete Prometheus + KEDA + Grafana setup:**

```yaml
# prometheus-scrape config for vLLM
scrape_configs:
- job_name: vllm
  static_configs:
  - targets: ['vllm-llama3-8b-svc:8000']
  metrics_path: /metrics

# Key metrics to dashboard in Grafana:
# vllm:num_requests_running     — GPU utilization proxy
# vllm:num_requests_waiting     — queue pressure → scale trigger
# vllm:gpu_cache_usage_perc     — KV cache fill % (>95% = OOM risk)
# vllm:time_to_first_token_ms   — TTFT p50/p95/p99
# vllm:time_per_output_token_ms — TBT (inter-token latency)
# vllm:request_success_total    — throughput
```

**Key production configuration checklist:**

```yaml
# vLLM startup args for production
--model meta-llama/Meta-Llama-3-8B-Instruct
--tensor-parallel-size 1         # match to GPU count per pod
--gpu-memory-utilization 0.90    # leave 10% headroom for spikes
--max-num-seqs 256               # tune based on avg sequence length
--max-model-len 8192             # cap to control KV cache size
--enable-chunked-prefill         # prevents long prefill blocking decode
--disable-log-requests           # reduce log volume in prod
--uvicorn-log-level warning
--served-model-name llama3-8b   # alias for API routing

# Environment variables
VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=0           # explicit GPU assignment per container
```

**Scale-to-zero pattern (cost saving for low-traffic models):**

```yaml
# KServe handles scale-to-zero natively
spec:
  predictor:
    minReplicas: 0               # scale to zero when idle
    scaleTarget: 1               # scale up on first request
    scaleMetric: "concurrency"

# First request after idle: cold start = 60–120s (model loading)
# Acceptable for batch/async jobs — not for real-time chat
# Use minReplicas: 1 for real-time, minReplicas: 0 for batch
```

---

### Production Readiness Checklist

```
INFRA:
  ✓ Docker image pinned to specific vLLM version (not :latest)
  ✓ Model weights on shared PVC (not downloaded per pod)
  ✓ HuggingFace token in K8s Secret (not env var in YAML)
  ✓ GPU node pool with autoscaling enabled
  ✓ readinessProbe initialDelaySeconds ≥ 60 (model load time)

SCALING:
  ✓ KEDA ScaledObject watching vllm:num_requests_waiting
  ✓ Scale-down cooldown ≥ 300s (avoid killing mid-generation)
  ✓ minReplicas ≥ 1 for real-time endpoints
  ✓ Pre-warmed GPU nodes (avoid 5min node provision delay)

OBSERVABILITY:
  ✓ Prometheus scraping /metrics
  ✓ Grafana dashboards: TTFT p99, queue depth, GPU cache %
  ✓ Alert: num_requests_waiting > 20 for 2min → scale trigger
  ✓ Alert: gpu_cache_usage_perc > 90% → OOM risk
  ✓ Structured request logs with request_id for tracing

RELIABILITY:
  ✓ --enable-chunked-prefill (prevents head-of-line blocking)
  ✓ livenessProbe restarts hung pods automatically
  ✓ Pod disruption budget: minAvailable: 1 during upgrades
  ✓ Resource limits set (prevent one pod consuming all GPU mem)
```

---

## Disaggregated Prefill/Decode: Architecture, Scheduling & Kubernetes Configuration

### Why Separate Prefill and Decode?

**The Core Problem:** Prefill and decode have fundamentally different compute profiles, yet standard vLLM runs both on the same GPU.

| Phase | Workload | Bottleneck | GPU Utilization (H100) | Best Hardware |
|---|---|---|---|---|
| **Prefill** | Process entire prompt (e.g., 10K token RAG context) in **one parallel forward pass** | Compute (FLOPS) | **92%** | High-compute GPUs: H100, A100, H200 |
| **Decode** | Generate **one token at a time** autoregressively, reusing KV cache | Memory bandwidth (reading KV cache repeatedly) | **28%** | High-bandwidth GPUs: L4, A10G, A100 |

**Cost Impact:** You provision 64 H100s to hit 92% compute utilization during prefill. Same cluster averages 28% utilization overall due to decode. That's ~36 H100s worth of waste.

**Disaggregated serving** fixes this by splitting into two independent vLLM instances:
1. **Prefill instance** → runs on A100/H100 (expensive but high TFLOPS)
2. **Decode instance** → runs on L4/A10G (cheaper, sufficient bandwidth)

They communicate via KV cache transfer: prefill computes the cache, sends it to decode, decode streams tokens back.

---

### The Token Generation Pipeline (Autoregressive Decoding)

**What is a token?** A token is roughly a word or sub-word. "Hello world" ≈ 2 tokens. LLMs process tokens, not characters.

**The three-phase flow:**

```
Phase 1: PREFILL (once per request)
┌─────────────────────────────────────────────┐
│  Input prompt: "What is the capital of     │
│   France?" [7 tokens]                      │
│                                             │
│  → Process ALL 7 tokens in parallel        │
│  → Compute KEY-VALUE attention cache        │
│  → Predict first output token → "Paris"    │
│  → Transfer KV cache to decode node        │
│                                             │
│  Execution time: 50–200ms                  │
│  GPU usage: 92% (compute-saturated)        │
└─────────────────────────────────────────────┘
                    ↓
Phase 2: DECODE (loop, 1+ times per output token)
┌─────────────────────────────────────────────┐
│  Context: [prompt (7 tokens)] + [Paris]    │
│                                             │
│  Step 1: Read KV cache → predict " is"    │
│  Step 2: Read KV cache → predict " the"   │
│  Step 3: Read KV cache → predict " capital" │
│  Step 4: ...                               │
│  Step N: Read KV cache → predict <EOS>    │
│          STOP — generation complete       │
│                                             │
│  Each step: ~10–20ms (memory-bound)        │
│  GPU usage: 28% (memory-bandwidth bound)   │
│  Total: N × 10ms for N output tokens       │
└─────────────────────────────────────────────┘
```

**The KV Cache Mystery Explained:**

During prefill, vLLM computes **attention keys and values** for every input token and stores them (the "cache"). During decode, each new token only needs to compute its own attention against the *cached* keys/values from the prefill phase — it doesn't reprocess the entire prompt.

```
Naive approach (no cache):
  Gen token 1: Process [prompt + new_token_1]           → O(N+1)² compute
  Gen token 2: Process [prompt + new_token_1 + token_2] → O(N+2)² compute
  Gen token 3: ...                                      → SLOW

With KV cache:
  Prefill: Process [prompt]                 → O(N)² compute, cache stored
  Gen token 1: Compute token 1 attention with cached KVs → O(1) compute
  Gen token 2: Compute token 2 attention with cached KVs → O(1) compute
  ...                                       → FAST
```

**This is why decode is memory-bandwidth-bound:** Each step reloads gigabytes of KV cache from GPU HBM. With a 70B model and 10K token context, the KV cache alone is 10–20 GB. Reading this repeatedly becomes the bottleneck, not compute.

---

### How Prefill and Decode Are Not Re-run

**Common misconception:** "When a new token is generated, don't we re-run prefill with the prompt + new tokens?"

**Answer: No. Absolutely not.** That would be incredibly wasteful and defeats the entire purpose of the KV cache.

The KV cache is **computed once during prefill and reused for the entire decode phase**. Each decode step only computes the new token's attention, leveraging the cached KVs.

---

### Kubernetes Scheduling: Taints, Tolerations & Node Affinity

**The Problem:** Without scheduling constraints, prefill and decode pods can land on any node — potentially on the wrong GPU type.

- Prefill pod lands on L4 (low compute) → 10× slower, GPU underutilized
- Decode pod lands on H100 (high cost, overkill) → money wasted

**Solution:** Use Kubernetes **taints** and **tolerations** to enforce GPU placement.

#### Concept: Taints & Tolerations

```
TAINT (on node):     "This node is for prefill workloads only"
TOLERATION (on pod): "I can run on prefill-tainted nodes"

Result: Only pods with matching toleration can land on tainted nodes
```

**The Three Taint Effects:**

```
NoSchedule:         ████████████ (HARD RULE)
                    New pods without toleration: BLOCKED
                    Existing pods: unaffected

PreferNoSchedule:   ░░░░░░░░░░░░ (SOFT RULE)
                    New pods: scheduler avoids, but falls back if needed
                    Existing pods: unaffected

NoExecute:          ████████████ + EVICTION
                    New pods: BLOCKED
                    Existing pods without toleration: EVICTED after tolerationSeconds
```

#### Step 1: Create Tainted Node Pools

```bash
# Create prefill pool with H100 GPUs
gcloud container node-pools create prefill-pool \
  --cluster=my-llm-cluster \
  --machine-type=a3-highgpu-8g \
  --accelerator=type=nvidia-h100,count=8 \
  --num-nodes=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=5 \
  --node-taints=inference-role=prefill:NoSchedule

# Create decode pool with L4 GPUs
gcloud container node-pools create decode-pool \
  --cluster=my-llm-cluster \
  --machine-type=g2-standard-24 \
  --accelerator=type=nvidia-l4,count=2 \
  --num-nodes=2 \
  --enable-autoscaling --min-nodes=0 --max-nodes=10 \
  --node-taints=inference-role=decode:NoSchedule
```

#### Step 2: Label Nodes for Fine-Grained Control

```bash
# Label prefill nodes (multiple nodes at once)
kubectl label nodes <prefill-node-1> <prefill-node-2> \
  gpu-type=h100 \
  inference-role=prefill

# Label decode nodes
kubectl label nodes <decode-node-1> <decode-node-2> <decode-node-3> \
  gpu-type=l4 \
  inference-role=decode
```

> **Why both taint AND label?**
> - **Taint + Toleration:** Permissioning layer (which pods can land here)
> - **Label + Node Affinity:** Navigation layer (which nodes should a pod prefer)
> Both work together for redundant control.

#### Step 3: Deploy Prefill with Toleration + Node Affinity

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill-worker
  namespace: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-prefill
  template:
    metadata:
      labels:
        app: vllm-prefill
        inference-role: prefill
    spec:
      # ── Toleration: "I can tolerate the prefill taint" ──
      tolerations:
        - key: "inference-role"
          operator: "Equal"
          value: "prefill"
          effect: "NoSchedule"

      # ── Node Affinity: "I MUST run on H100 nodes" ──
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: gpu-type
                    operator: In
                    values: ["h100"]

      containers:
        - name: vllm-prefill
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "meta-llama/Llama-3.1-70B-Instruct"
            - "--kv-transfer-config"
            - '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer"}'
            - "--tensor-parallel-size"
            - "8"
            - "--enable-chunked-prefill"  # Prevent long prefill blocking decode
          resources:
            limits:
              nvidia.com/gpu: "8"
              memory: "320Gi"
            requests:
              nvidia.com/gpu: "8"
              memory: "320Gi"
          env:
            - name: VLLM_ROLE
              value: "prefill"
          ports:
            - containerPort: 8100
              name: prefill-api
```

#### Step 4: Deploy Decode with Toleration + Node Affinity

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-decode-worker
  namespace: inference
spec:
  replicas: 4                    # More decode replicas — token generation is sequential
  selector:
    matchLabels:
      app: vllm-decode
  template:
    metadata:
      labels:
        app: vllm-decode
        inference-role: decode
    spec:
      # ── Toleration: "I can tolerate the decode taint" ──
      tolerations:
        - key: "inference-role"
          operator: "Equal"
          value: "decode"
          effect: "NoSchedule"

      # ── Node Affinity: "I MUST run on L4 nodes" ──
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: gpu-type
                    operator: In
                    values: ["l4"]

      containers:
        - name: vllm-decode
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "meta-llama/Llama-3.1-70B-Instruct"
            - "--kv-transfer-config"
            - '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer"}'
            - "--tensor-parallel-size"
            - "2"
            - "--max-num-seqs"
            - "512"  # More sequences for decode (token generation)
          resources:
            limits:
              nvidia.com/gpu: "2"
              memory: "100Gi"
            requests:
              nvidia.com/gpu: "2"
              memory: "100Gi"
          env:
            - name: VLLM_ROLE
              value: "decode"
          ports:
            - containerPort: 8200
              name: decode-api
```

#### Step 5: Deploy Request Router (CPU node, no GPU)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-router
  namespace: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: router
  template:
    metadata:
      labels:
        app: router
    spec:
      # No toleration → automatically lands on untainted (CPU) nodes
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: node-type
                    operator: In
                    values: ["cpu-compute"]
      containers:
        - name: router
          image: nvidia/dynamo-router:latest  # or custom Python FastAPI router
          env:
            - name: PREFILL_ENDPOINT
              value: "http://vllm-prefill-svc:8100"
            - name: DECODE_ENDPOINT
              value: "http://vllm-decode-svc:8200"
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
```

---

### Understanding requiredDuringSchedulingIgnoredDuringExecution

This long name breaks down as:

```
required during scheduling   +   ignored during execution
├─ HARD RULE during scheduling    └─ SOFT during execution
├─ Pod won't start if no match
├─ Pod stays PENDING forever
│  if no node matches
└─ Already-running pod NOT evicted
   if labels later removed
```

**Alternative modes:**

| Mode | Scheduling | Execution | Use Case |
|---|---|---|---|
| `requiredDuringSchedulingIgnoredDuringExecution` | HARD | SOFT | Standard: pin at start, tolerate changes later |
| `preferredDuringSchedulingIgnoredDuringExecution` | SOFT | SOFT | Best-effort: try to match, fallback if needed |
| `requiredDuringSchedulingRequiredDuringExecution` | HARD | HARD | Not yet implemented in K8s |

---

### GPU Resource Allocation: Limits vs Requests

```yaml
resources:
  limits:
    nvidia.com/gpu: "8"      # This pod exclusively owns 8 GPUs
    memory: "320Gi"          # Host RAM limit (not GPU VRAM)
  requests:
    nvidia.com/gpu: "8"      # Scheduler needs 8 free GPUs to place pod
    memory: "320Gi"
```

**Key rules:**

| Resource | requests < limits? | Why |
|---|---|---|
| CPU | ✅ Yes (can be overbooked) | CPU is virtualizable |
| Memory | ✅ Yes (can swap) | Memory is virtualizable |
| GPU | ❌ **NO** (requests MUST = limits) | GPUs are NOT virtualizable in standard K8s |

GPU resources in Kubernetes are always 1:1 assigned — no sharing, no overbooking. A GPU is either fully allocated to one pod or unused.

---

### Verifying Scheduling

```bash
# See node taints
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

# Check which nodes have which labels
kubectl get nodes --show-labels

# Verify pod scheduled on correct node
kubectl describe pod vllm-prefill-worker-xxxxx -n inference | grep -A 10 "Node-Selectors"

# Check GPU allocation
kubectl get pods -n inference -o custom-columns=\
  NAME:.metadata.name,\
  NODE:.spec.nodeName,\
  GPUS:.spec.containers[*].resources.limits['nvidia.com/gpu']
```

---

### MIG: Multi-Instance GPU for Maximum Capacity

If you want to pack multiple smaller models onto one expensive H100, partition it with **MIG** (Multi-Instance GPU) — splits one physical GPU into multiple isolated instances.

```bash
# Enable MIG at node pool creation (H100 example)
gcloud container node-pools create mig-prefill-pool \
  --cluster=my-llm-cluster \
  --machine-type=a3-highgpu-8g \
  --accelerator=type=nvidia-h100,count=8,gpu-partition-size=3g.40gb \
  --node-taints=inference-role=prefill:NoSchedule
```

This creates **2 MIG instances per H100** (each 3g.40gb = 40GB memory + 42 SMs).

**MIG Profiles (H100 80GB):**

| Profile | Memory | Compute | Max instances |
|---|---|---|---|
| 1g.10gb | 10GB | 1/7 | 7 per GPU |
| 2g.20gb | 20GB | 2/7 | 3 per GPU |
| 3g.40gb | 40GB | 3/7 | 2 per GPU |
| 7g.80gb | 80GB | 7/7 | 1 (full GPU) |

**Request a MIG slice in the pod:**

```yaml
resources:
  limits:
    nvidia.com/mig-3g.40gb: "1"    # Request one 3g.40gb MIG slice
```

**Maximum utilization:** all-1g.10gb = 7 pods per H100, each with 10GB isolated memory.

---

### KV Cache Transfer Between Prefill and Decode

After prefill computes the KV cache, it must transfer to the decode pod's reserved GPU memory.

**Transfer mechanisms:**

| Method | Speed | Where Used |
|---|---|---|
| **TCP/gRPC over network** | ~100 MB/s | GKE (pod-to-pod on different nodes) |
| **NVLink (GPU-to-GPU direct)** | ~900 GB/s (9× faster) | NVIDIA DGX, on-prem clusters |

GKE typically uses TCP because prefill and decode pods are on different node types. NVIDIA Dynamo can use NVLink on dedicated clusters for lower transfer latency.

**Transfer size example:**
- Llama 70B model
- 10K token context
- KV cache: ~10–20 GB per request
- Transfer time over TCP: ~100–200ms

The transfer blocks the decode pod from starting token generation, so minimize cache size via KV cache quantization or reduction techniques.


---

## vLLM Sleep Mode: Multi-Model Serving on a Single GPU

### Overview

**vLLM Sleep Mode** is an optimization technique that enables efficient multi-model serving on a single GPU by allowing models to "sleep" (offload weights while preserving execution context) and wake up quickly without full reloading overhead. This is particularly valuable for **sequential multi-model pipelines** where only one model is active at a time.

> **Use Case:** Enterprise NLP pipelines often run a sequence of models (e.g., LLaMA-3.1 8B → Qwen-14B → Qwen-32B) for multi-stage tasks like entity extraction, relationship inference, and decision-making. Each model handles a different stage in the pipeline, and execution is strictly sequential. Sleep Mode enables fast switching between models without provisioning multiple GPUs or maintaining separate endpoints for each model.

---

### The Problem: Multi-Model Deployment Challenges

#### Multi-Model Pipeline Requirements

Running multiple large language models in a single pipeline—each handling a different task—creates a critical constraint: **LLMs are not great at sharing GPUs.**

**Multi-Stage Pipeline Example:**
- **LLaMA-3.1 8B** → Stage 1 (feature extraction)
- **Qwen-14B** → Stage 2 (analysis and reasoning)
- **Qwen-32B** → Stage 3 (final decision/output generation)

Because the pipeline is strictly sequential, only one model is active at any given time. The challenge: **switch between models quickly and cheaply without provisioning multiple GPUs.**

#### Traditional Deployment Trade-offs

| Approach | Architecture | Limitations |
|---|---|---|
| **Dedicated GPU per Model** | One endpoint/GPU per model, each model tied to dedicated hardware | For 3 models: requires 3+ GPUs; extremely expensive and operationally complex |
| **Dynamic Container Loading** | Single container with custom model loading/unloading logic | No standardized API for model lifecycle management; requires custom implementation |
| **Multi-Model in Memory** | Keep all models in GPU memory simultaneously | Infeasible for large models (70B+); requires 2x+ memory capacity |

#### The Memory Problem

Without Sleep Mode, switching models requires one of:

| Approach | Cost | Tradeoff |
|---|---|---|
| **Keep all models in GPU** | Memory cost | Needs 2x+ memory (models don't fit together) |
| **Reload every time** | Latency cost | 30–100+ seconds lost per switch |

---

### How vLLM Sleep Mode Works

**vLLM Sleep Mode** introduces a lightweight mechanism for managing multi-model serving by **preserving key execution state while offloading model weights** from GPU.

#### The Core Insight

During a **cold start** (full model reload), the system must:
1. Set up CUDA allocator
2. Compile kernels
3. Capture execution graphs
4. Warm caches

**With Sleep Mode,** all of these components remain intact:
- Only model weights are offloaded (to CPU or disk)
- CUDA context, execution graphs, and kernels stay warm
- Waking a model becomes nearly instantaneous

#### Cost Breakdown: Cold Load vs Sleep → Wake

| Cost Component | Cold Load | Sleep Mode |
|---|---|---|
| **VRAM Weight Load** | ✅ | ✅ Preserved |
| **CUDA Allocator Setup** | ❌ Every time | ✅ Preserved |
| **CUDA Graph Capture** | ❌ Every time | ✅ Preserved |
| **Kernel JIT Compile** | ❌ Every time | ✅ Preserved |
| **Cache Warm-Up** | ❌ Every time | ⚡ Quick re-warm |

---

### Benchmark: Model Switching Performance

**Model Switching Performance on A100/H100 GPUs:**

| Model | Transition | Load Time (s) | Inference (s) | Total (s) | **Speedup** |
|---|---|---|---|---|---|
| **LLaMA-8B** | Cold Start | 36.23 | 0.1606 | 36.39 | – |
| **LLaMA-8B** | Sleep → Wake | 0.85 | 0.1318 | 0.98 | **37× faster** |
| **Qwen-14B** | Cold Start | 78.87 | 0.2228 | 79.09 | – |
| **Qwen-14B** | Sleep → Wake | 2.44 | 0.2106 | 2.65 | **30× faster** |
| **Qwen-32B** | Cold Start | 125.46 | 0.4339 | 125.89 | – |
| **Qwen-32B** | Sleep → Wake | 3.73 | 0.4221 | 4.15 | **30× faster** |

**Key Observations:**
- Model load time drops from **30–125 seconds → 0.85–3.73 seconds**
- Inference time remains essentially unchanged (Sleep Mode doesn't affect compute)
- **Total latency reduction: ~30–37×**

**Real-World Impact:**
- ✅ Lower pipeline latency (model switching becomes negligible)
- ✅ Lower GPU cost (one GPU serves multiple models)
- ✅ Higher scalability (no need for GPUs per model)
- ✅ Smoother orchestration (fast transitions between stages)

---

### Sleep Mode Implementation

#### Basic Usage Example

```python
from vllm import LLM
from transformers import AutoTokenizer

# Load LLaMA-8B with Sleep Mode enabled
llm_a = LLM(
    "/models/Meta-Llama-3.1-8B-Instruct",
    enable_sleep_mode=True
)
tokenizer = AutoTokenizer.from_pretrained(llm_a.model, trust_remote_code=True)

# Run inference on LLaMA
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Extract entities from this document"}],
    tokenize=False,
    add_generation_prompt=True,
)
output_a = llm_a.generate([prompt])
print("LLaMA Output:", output_a[0].outputs[0].text)

# ── SLEEP LLaMA AND SWITCH TO QWEN ──
llm_a.sleep(level=1)  # level=1: CPU offloading, level=2: disk offloading

# Load Qwen-14B (weights load from CPU/disk, context preserved)
llm_b = LLM(
    "/models/Qwen/Qwen2.5-14B-Instruct",
    enable_sleep_mode=True
)

# Run inference on Qwen (fast, as CUDA context is warm)
prompt_b = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Infer relationships from entities"}],
    tokenize=False,
    add_generation_prompt=True,
)
output_b = llm_b.generate([prompt_b])
print("Qwen Output:", output_b[0].outputs[0].text)

# ── SWITCH TO QWEN-32B ──
llm_b.sleep(level=1)

llm_c = LLM(
    "/models/Qwen/Qwen2.5-32B-Instruct",
    enable_sleep_mode=True
)

prompt_c = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Make final compliance decision"}],
    tokenize=False,
    add_generation_prompt=True,
)
output_c = llm_c.generate([prompt_c])
print("Qwen-32B Output:", output_c[0].outputs[0].text)
```

#### Configuration Options

```python
# Sleep Mode levels
llm.sleep(level=1)  # CPU offloading (fast wake, uses CPU RAM)
llm.sleep(level=2)  # Disk offloading (slower wake, uses disk)

# Enable at initialization
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    enable_sleep_mode=True,      # Enable Sleep Mode
    sleep_offload_type="cpu",    # or "disk"
)
```

---

### Deployment Patterns

#### Sequential Pipeline (Multi-Stage Processing)

```
                    Request Input
                         │
                    ┌────▼────┐
                    │ LLaMA 8B │  ◄─ Wake (0.98s)
                    │ (Active) │
                    └────┬────┘
                         │
                    [Sleep LLaMA]
                         │
                    ┌────▼─────┐
                    │ Qwen-14B  │  ◄─ Wake (2.65s)
                    │ (Active)  │
                    └────┬─────┘
                         │
                    [Sleep Qwen-14B]
                         │
                    ┌────▼─────┐
                    │ Qwen-32B  │  ◄─ Wake (4.15s)
                    │ (Active)  │
                    └────┬─────┘
                         │
                  Final Decision Output

Total Model Switching Latency: ~7.78s (vs ~241s cold starts)
GPU Utilization: Single A100/H100
```

#### Deployment Options

**Option 1: Custom Model Manager with Sleep Mode**
- More control over model lifecycle and scheduling
- Requires custom implementation of sleep/wake orchestration logic
- Suitable for organizations with specialized infrastructure teams

**Option 2: Platform-Level Sleep Mode (Emerging)**
- Sleep Mode integration at the inference serving platform level
- Platform handles sleep/wake lifecycle automatically
- Engineers focus on business logic, with infrastructure abstracted away

---

### Advantages & Limitations

#### ✅ Advantages

| Benefit | Impact |
|---|---|
| **Fast model switching** | 30–37× speedup vs cold loads |
| **Single GPU serving** | Reduce infrastructure cost significantly |
| **Preserved CUDA context** | No kernel recompilation on wake |
| **Transparent to application** | Simple API (just call `.sleep()`) |
| **Sequential pipelines** | Perfect fit for stage-by-stage workflows |

#### ⚠️ Limitations & Trade-offs

| Limitation | Mitigation |
|---|---|
| **Offload overhead** | Choose CPU (fast) vs disk (slower) based on available memory |
| **Not for concurrent models** | Sleep Mode assumes sequential execution (only one model active) |
| **Per-request state** | Context is per-LLM instance; requests to different models need separate instances |
| **Infrastructure support** | Currently experimental; needs integration into managed platforms |

---

### When to Use vLLM Sleep Mode

#### ✅ Use Sleep Mode When:
- Multiple models in **sequential pipeline** (one at a time)
- Models don't fit together in GPU memory
- Model switching latency is a bottleneck
- Cost per GPU is high (want to maximize utilization)
- Inference is batch/async (can tolerate 1–4s wake time)

#### ❌ Don't Use Sleep Mode When:
- Models need **concurrent execution** (same request on multiple models)
- All models fit in GPU memory simultaneously
- Real-time latency is critical (<100ms is required)
- Models are rarely switched (cold loads acceptable)

---

### Comparison: Sleep Mode vs Alternatives

| Approach | Load Time | GPU Cost | Complexity | Use Case |
|---|---|---|---|---|
| **Cold Load (separate GPUs)** | 30–125s | High (3+ GPUs) | Low | Unlimited budget, simplicity |
| **vLLM Sleep Mode** | 0.85–3.73s | Low (1 GPU) | Medium | Sequential pipelines, cost-sensitive |
| **Model Quantization + Cold Load** | 10–50s | Medium (1–2 GPUs) | High | Memory-constrained, can afford latency |
| **MIG (Multi-Instance GPU)** | 30–125s | Low (1 GPU) | High | Multiple small models, complex setup |

---

### Production Considerations

#### Memory Planning

```
GPU Memory Allocation:

Cold Model (Weights + KV Cache):  80–140 GB (large models)
CPU Offload Buffer:               80–140 GB
Disk Offload Buffer:              80–140 GB (optional)

Example (Single H100 80GB):
├─ Model weights in GPU:  80 GB (when active)
└─ During sleep:  Weights moved to CPU RAM (need 80+ GB on host)
```

#### Monitoring Sleep Mode

```python
# Log model lifecycle
logger.info(f"Sleeping {model_name}...")
start_time = time.time()
llm.sleep(level=1)
sleep_time = time.time() - start_time
logger.info(f"Model asleep in {sleep_time:.2f}s")

# Monitor wake time
logger.info(f"Waking {model_name}...")
start_time = time.time()
output = llm.generate([prompt])
wake_time = time.time() - start_time
logger.info(f"Model awake and generated output in {wake_time:.2f}s")
```

#### Error Handling

```python
try:
    llm_a.sleep(level=1)
except RuntimeError as e:
    logger.error(f"Failed to sleep model: {e}")
    # Fallback: unload model or restart
    
try:
    output = llm_b.generate([prompt])
except RuntimeError as e:
    logger.error(f"Wake/generate failed: {e}")
    # Fallback: reload from disk
```

---

### Future Work & Platform Integration

**Current Status:**
- vLLM Sleep Mode proven effective in production multi-model pipelines
- Validated across sequential inference workloads at scale

**Next Steps:**
- Standardize Sleep Mode API across vLLM releases
- Integrate with Kubernetes orchestration platforms
- Develop platform-native support for automatic sleep/wake lifecycle management
- Goal: Simplify multi-model serving so teams focus on business logic, not infrastructure

---

### Conclusion

vLLM Sleep Mode is **not just a micro-optimization**—it introduces a new way to manage multi-model serving in production:

1. **Closes the gap** between specialized multi-model workflows and GPU constraints
2. **Delivers 30–37× faster switching** vs cold loads
3. **Improves cost efficiency** by serving multiple models on one GPU
4. **Maintains simplicity** with a clean API

For sequential multi-model pipelines across domains—compliance, content moderation, anomaly detection, decision automation—Sleep Mode enables **fast, efficient, scalable inference without architectural complexity.**

---

## LLM Inferencing GPU Optimization: Document Processing Case Study

### Overview

A production document AI service processing tens of thousands of documents daily faced a critical GPU bottleneck: **2 A100 GPUs with only 12% utilization while missing performance SLAs.** Through systematic vLLM parameter tuning, the team achieved **6× throughput improvement and 4.4× faster latency** without additional hardware—reducing required GPUs from 10 to 4.

### The GPU Utilization Paradox

**The Problem:**
- Current capacity: 2 A100 GPUs supporting ~650 req/hr
- Production requirement: 6,000 req/hr (10× current load)
- GPUs needed: 10 A100s minimum
- GPUs available: 0 (procurement blocked)

**The Paradox:**
During peak loads with poor latency (31–38 seconds), GPU utilization showed only **12–15%**—two of the world's most powerful inference GPUs sitting mostly idle while failing SLAs.

**The Insight:**
The problem wasn't capacity or budget; it was configuration. The service didn't need more GPUs—it needed to use the existing GPUs effectively.

### Understanding the Workload

#### Workload Characteristics

| Dimension | Value | Impact |
|---|---|---|
| **Input tokens** | ~2,000–3,000 per request | Long prefill phase |
| **Output tokens** | ~50–100 per request (200 max) | Short decode phase |
| **Input/Output ratio** | 20:1 to 60:1 | 87% of time in memory-bound decode |
| **Model** | LLaMA-3 8B (FP16, ~16GB) | Moderate memory footprint |
| **Task** | Structured extraction | High precision required |

#### Typical Request Timeline

```
OCR:                2.3s  (13% of total)
LLM Prefill:        4.6s  (25% of total)
LLM Decode:        11.3s  (62% of total) - generating 75 tokens at 6.64 tok/s
Batching Overhead:  4.8s  (26% of total)
────────────────────────
Total:             18.2s
```

**Why This Matters:**
The extreme input/output ratio (20:1 to 60:1) meant:
- ✅ Batching strategy needed to prioritize decode throughput
- ✅ Memory optimization more critical than compute
- ✅ KV cache management would be the biggest lever
- ❌ Prefill chunking would actually hurt performance

### Understanding the Bottleneck: Why LLM Inference is Memory-Bandwidth-Bound

#### The Two Phases of LLM Inference

**Phase 1: Prefill (Compute-Bound)**
```
Input: Process all 2,500 input tokens at once
Operation: Massive parallel matrix-matrix multiplication
GPU Behavior: Compute cores SATURATED (70–85% utilization)
Time: ~3.6s for the workload
```

**Phase 2: Decode (Memory-Bandwidth-Bound)**
```
For each output token (up to 50–100 typically):
  1. Load 16GB model weights from GPU VRAM → compute units + 0.5GB KV cache
  2. Compute attention (tiny matrix-vector multiply)
  3. Generate 1 token
  4. Store new KV values back to VRAM
  5. Repeat for next token

GPU behavior: Compute idle, waiting for memory (8–12% utilization)
Time: ~11.3s for 75 tokens at 6.64 tokens/s
```

**The Bottleneck:**
- 62% of time in memory-bandwidth-bound decode
- Even with vLLM's continuous batching, only 3–4 concurrent requests (should be 20+)
- GPU utilization: 12% (should be 70–80%)
- Memory bandwidth utilization: 7.3% (should be 60–80%)

#### Three Types of GPU Bounds

| Type | Meaning | Example | Solution |
|---|---|---|---|
| **Compute-Bound** | Limited by compute throughput (FLOPS) | Training, prefill phase | Faster GPU, more FLOPS, tensor parallelism |
| **Memory-Bandwidth-Bound** | Limited by memory data transfer rate | LLM decode (our case) | Faster bandwidth, better batching, less data movement |
| **Memory-Capacity-Bound** | Limited by total GPU memory available | Model too large, OOM | Bigger GPU, quantization, tensor parallelism |

This service's decode phase is **Type 2: memory-bandwidth-bound.**

### The Three Root Problems

#### Problem 1: Low Concurrency (Memory Fragmentation)

```
Available for KV cache: 19.5 GB
Theoretical capacity: ~19 concurrent requests
Actual concurrent requests: 3–4 (with default config)

Why? max_num_seqs=16 combined with conservative gpu_memory_utilization
     meant vLLM couldn't fully utilize available memory
```

#### Problem 2: Prefill Chunking

```
Input documents: ~2,500 tokens average
Default max_num_batched_tokens: 2,048
Result: Prefills split across 2 iterations
  ├─ Iteration 1: Process 2,048 tokens (full GPU utilization)
  └─ Iteration 2: Process 452 tokens (GPU only 22% full!)
  └─ Overhead: Extra scheduler call, context switching
```

#### Problem 3: Decode Inefficiency

```
Decode throughput: Only 6.64 tokens/s
Memory bandwidth utilization: 7.3%
Why? Low concurrency (3–4 requests) meant GPU mostly idle
     between token generations
```

### Performance Baseline

| Metric | Value |
|---|---|
| **Throughput** | 612 req/hr |
| **Average Latency** | 18.2s |
| **P99 Latency** | 31–38s (Target: <7s) |
| **GPU Utilization** | 12% |
| **Decode Throughput** | 6.64 tokens/s |
| **KV Cache Usage** | 0.3–0.4% |

**Problems:**
- Prefills were chunked (2,500 tokens with 2,048 limit)
- Only 3–4 effective concurrent requests
- Suboptimal memory and dtype settings
- Missing decode optimizations

### Optimization Changes: Systematic Tuning

We followed a disciplined approach: establish baseline, change **one parameter at a time**, test under realistic load, measure everything.

#### Change 1: dtype – Float16 Instead of BFloat16

**Why It Matters:**
Both FP16 and BF16 use 2 bytes, but different precision/range trade-offs:
- **FP16:** 10-bit mantissa (better precision for small values)
- **BF16:** 7-bit mantissa, 8-bit exponent (better range)

For structured extraction with attention scores (small, precise values):
- ✅ FP16 provides better precision
- ❌ BF16's wider range unnecessary

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    dtype="float16"  # Changed from "bfloat16"
)
```

#### Change 2: max_num_seqs – Concurrent Requests

**Why It Matters:**
Controls how many requests can be processed concurrently in vLLM's continuous batching. More concurrent requests = better GPU utilization and memory bandwidth usage.

**Testing Journey:**
| Trial | max_num_seqs | Throughput | P99 Latency | GPU Util | Status |
|---|---|---|---|---|---|
| 1 | 16 | 612 req/hr | 31s | 12% | Underutilized |
| 2 | 32 | 890 req/hr | 42s | 42% | OOM errors, unstable |
| 3 | **24** | **3,600 req/hr** | **6.4s** | **78%** | ✅ Optimal |

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    max_num_seqs=24  # Changed from default 16
)
```

#### Change 3: max_num_batched_tokens – 2048 to 12288

**Why It Matters:**
Controls how many tokens can be processed in a single forward pass, primarily affecting prefill operations.

**The Problem:**
```
Typical document: 2,500 input tokens
vLLM default: 2,048 max tokens per batch

Result: Prefill gets CHUNKED
  ├─ Iteration 1: Process 2,048 tokens (full GPU)
  ├─ Iteration 2: Process 452 tokens (GPU underutilized—22% full!)
  └─ Overhead: Extra scheduler call, context switching

For 3,000-token documents:
  ├─ Iteration 1: Process 2,048 tokens
  ├─ Iteration 2: Process 952 tokens (47% utilized)
  └─ Even worse inefficiency
```

**Reasoning for 12,288:**
```python
# Workload analysis:
avg_input = 2500
p95_input = 3500

# Goals:
# 1. No chunking for 95% of requests
# 2. Room to batch multiple prefills together
# 3. Follow vLLM recommendation (>8192)

# Choice: 12,288
# = 6x the default
# = 3x our average input
# = Can fit 4 concurrent 3K prefills (4 × 3K = 12K)
```

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    max_num_batched_tokens=12288  # Changed from default 2048
)
```

**Why It Worked:**
- ✅ Eliminated prefill chunking for 95% of requests
- ✅ Enabled batching of multiple prefills together
- ✅ GPU compute fully saturated during prefill phase
- ✅ Reduced scheduler overhead (fewer iterations)

#### Change 4: gpu_memory_utilization – 0.80 to 0.60

**Why We DECREASED It:**
The OCR model shares the same GPU (2.5 GB overhead).

```python
At 0.80:
  24 GB × 0.80 = 19.2 GB allocated for KV cache
  Risk: Tight with OCR sharing, occasional memory pressure

At 0.60:
  24 GB × 0.60 = 14.4 GB allocated for KV cache
  Benefit: More stable, buffer for OCR sharing

Observation:
  KV cache peaked at 5.9% (850 MB of 14.4 GB)
  Conclusion: Plenty of headroom, 0.60 was sufficient
```

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    gpu_memory_utilization=0.60  # Changed from 0.80
)
```

**Lesson Learned:**
| Scenario | Recommendation |
|---|---|
| Shared GPU (multiple services) | Be conservative (0.60–0.75) |
| Dedicated GPU for LLM | Can push higher (0.90–0.95) |
| Highly variable workload | Leave buffer (0.70–0.80) |
| Stable, predictable workload | Can be more aggressive (0.85–0.95) |

#### Change 5: swap_space – 16 GB to 4 GB

**Why 4 GB?**
Monitored actual swap usage:
- Normal load: 0–200 MB
- Stress load: 800 MB–1.2 GB
- Peak bursts: ~3 GB

**Decision:**
4 GB covers 99% of scenarios; 16 GB was wasteful.

**Configuration:**
```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    swap_space=4  # Changed from 16 GB
)
```

**Impact:**
- ✅ Saved 12 GB RAM
- ✅ Prevented re-computation during stress (<2% requests swapped)

#### Change 6: Sampling Parameter Optimizations

**1. Early Stopping (stop_sequences):**
```python
# Outputs average 50–75 tokens, but max_tokens=200
# Without stop: Would generate to 200 tokens
# With stop tokens: Stops at ~75 tokens

llm.generate(
    prompt,
    stop=["[END]"],  # Stop generation at specific token
    max_tokens=200
)

# Saves ~125 tokens per request = massive time savings
```

**2. Disable Sampling Overhead (temperature=0.0):**
```python
# With temperature=0.0, doing greedy decoding anyway
# Explicit disable removes unnecessary logits processing

sampling_params = SamplingParams(
    temperature=0.0,      # Greedy decoding
    top_p=1.0,           # Disable nucleus sampling
    top_k=-1             # Disable top-k sampling
)

# Saves ~5–10ms per token
```

#### Change 7: CUDA Graphs (enforce_eager)

**Why Enable CUDA Graphs?**
Pre-compiled computation sequences reduce kernel launch overhead.

```python
llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enforce_eager=False  # Enable CUDA graphs (from True)
)
```

**Trade-off:**
- +28s startup time (one-time for long-running service)
- 5% decode speedup on every token generation
- **Worth it for production long-running services**

### Optimization Results

| Metric | Before | After | Improvement |
|---|---|---|---|
| **Throughput** | 612 req/hr | 3,600 req/hr | **5.9×** |
| **P99 Latency** | 31–38s | 6.4–8.7s | **4.4–4.8× faster** |
| **GPU Utilization** | 12% | 78% | **6.5× better** |
| **Decode Throughput** | 6.64 tok/s | ~40 tok/s | **6× better** |

**Key Achievement:**
SLA target of <7s P99 latency now achieved with existing hardware.

### GPU Requirements: From Impossible to Achievable

**Before Optimization:**
- Need: 10 GPUs minimum
- Have: 2 GPUs
- Gap: 8 GPUs (impossible to procure)
- Status: Project blocked

**After Optimization:**
- Need: 2 GPUs minimum (4 recommended for production headroom)
- Have: 2 GPUs
- Gap: 0–2 GPUs (achievable in 6–8 weeks)
- Status: Production deployed

**GPU Reduction: 66% (12 → 4 GPUs for full production capacity)**

### Cost Savings

Annual infrastructure cost (3-year reserved pricing):

| Scenario | GPUs | Annual Cost | Savings |
|---|---|---|---|
| Unoptimized | 10 | $144,000 | – |
| Optimized | 4 | $57,600 | **$86,400 annually** |

### Optimization Playbook for Document Processing Workloads

#### Step 1: Diagnose Your Bottleneck (15 minutes)

```bash
# Check GPU utilization (may be misleading for LLMs)
nvidia-smi

# Check memory bandwidth (the real bottleneck)
nvidia-smi dmon -s um

# If Mem-Util > 80% while GPU-Util < 30%:
# → You're memory-bandwidth-bound (like this case)
```

#### Step 2: Profile Your Workload

- Input token length (average and P95)
- Output token length (average and max)
- Input/output ratio
- Current throughput and latency
- Whether model shares GPU with other services

#### Step 3: Configure Parameters by Workload Type

| Workload Type | max_num_batched_tokens | max_num_seqs | gpu_memory_util |
|---|---|---|---|
| **Long input, short output** | 8,192–16,384 | 16–32 | 0.80–0.90 |
| (Document extraction, this case) | _(12,288 optimal)_ | _(24 optimal)_ | _(0.60 with shared GPU)_ |
| **Short input, long output** | 1,024–4,096 | 12–48 | 0.85–0.95 |
| (Chatbot, code completion) | | | |
| **Balanced** | 2,048–8,192 | 12–32 | 0.85–0.95 |
| (Summarization, translation) | | | |

#### Step 4: Monitor These Metrics

- **P50, P95, P99 latency** (not just average)
- **Error rate under stress** (OOM, timeouts)
- **GPU utilization AND memory bandwidth** (not just GPU%)
- **KV cache usage at peak**
- **Decode throughput (tokens/s)**

### Key Learnings

1. **Workload characteristics drive everything.** Your high input/output ratio meant large max_num_batched_tokens were critical. A chatbot (opposite ratio) would make opposite choices.

2. **Default vLLM config is optimized for chat.** For document processing, code completion, or summarization, you must tune for your workload.

3. **Memory-bandwidth-bound decode is the norm.** If you see low GPU utilization despite poor latency, you're likely memory-bandwidth-bound. Configuration, not hardware, is your lever.

4. **Test methodically.** Change one parameter at a time. Measure under realistic load. Document your choices.

5. **Monitor the right metrics.** GPU utilization % is misleading for LLMs. Focus on memory bandwidth, decode throughput, and latency percentiles.

### Production Deployment Checklist

```yaml
Configuration:
  ✓ dtype: float16 (or bfloat16 depending on precision needs)
  ✓ max_num_seqs: Tuned to workload (12–32 typical)
  ✓ max_num_batched_tokens: Tuned to input length (8K–12K for documents)
  ✓ gpu_memory_utilization: Conservative if shared (0.60–0.75)
  ✓ swap_space: Just enough for stress spikes (2–4 GB typical)
  ✓ enforce_eager: False (enable CUDA graphs)

Sampling:
  ✓ stop sequences: Defined (avoid max_tokens generation)
  ✓ temperature: 0.0 for deterministic output
  ✓ top_p: 1.0, top_k: -1 (disable if using temperature)

Monitoring:
  ✓ Track P50/P95/P99 latency (not average)
  ✓ Monitor decode throughput (tokens/s)
  ✓ Check memory bandwidth utilization
  ✓ Alert on error rate spikes (OOM, timeouts)

Load Testing:
  ✓ Test at normal load (production SLA target)
  ✓ Test at stress load (peak traffic, +20%)
  ✓ Run 4+ hour soak tests (catch stability issues)
  ✓ Measure error rates, tail latency under stress
```

### Conclusion

This case demonstrates that **GPU optimization is primarily about configuration, not hardware.** When you see:
- ✅ Low GPU utilization (<30%) despite poor latency
- ✅ Memory bandwidth utilization higher than GPU utilization
- ✅ Decode phase dominating request time (>60%)

You're facing a **memory-bandwidth-bound decode problem**—solvable through:
1. Increasing concurrency (max_num_seqs)
2. Reducing chunking (max_num_batched_tokens)
3. Optimizing batch composition
4. Fine-tuning memory allocation
5. Removing unnecessary compute (sampling, early stopping)

The results: **6× throughput, 4.4× faster latency, 66% fewer GPUs needed**—without throwing hardware at the problem.

---

## vLLM Production Stack: Enterprise-Grade Serving

### Overview

The **vLLM Production Stack** is a comprehensive, Kubernetes-native framework for deploying LLMs at scale in production environments. It provides battle-tested tools, patterns, and best practices for managing inference workloads across multiple GPUs, multiple nodes, and cloud environments.

> **Think of it as:** A complete toolkit that takes vLLM from development to production—handling scaling, routing, monitoring, and resource optimization automatically.

### What Problems Does It Solve?

| Problem | Solution |
|---|---|
| **Single point of failure** | Distributed deployment across multiple nodes |
| **Inefficient request routing** | Smart routing strategies (KV cache aware, semantic) |
| **Unpredictable traffic spikes** | Automatic scaling with KEDA |
| **Difficult performance debugging** | Built-in distributed tracing (Jaeger, OpenTelemetry) |
| **GPU resource waste** | Intelligent batching and cache sharing |
| **Complex deployments** | Kubernetes-native architecture |

### Core Components

#### 1. **Smart Request Routing**

**KV Cache Aware Routing**
Routes requests to instances with relevant cached KV data, avoiding recomputation.
```
Request for Document A
  ↓
Router checks: Which instance has cached KV for Document A?
  ↓
Sends to that instance (reuses cache)
  ↓
Result: 10–50% faster, reduced GPU load
```

**Semantic Aware Routing**
Routes similar queries to the same instance for better cache utilization.
```
Query 1: "What is machine learning?"
Query 2: "Define machine learning"
  ↓
Both routed to same instance (semantic similarity detected)
  ↓
Result: Better prefix cache hits
```

#### 2. **Disaggregated Prefill/Decode**

Separates compute-heavy prefill from memory-bandwidth-heavy decode onto different GPU types.
- **Prefill:** H100 (high compute)
- **Decode:** L4 (high bandwidth, cheaper)
- **Result:** 40–60% cost reduction vs mixed serving

See [Disaggregated Prefill/Decode section](#disaggregated-prefilldecode-architecture-scheduling--kubernetes-configuration) for detailed architecture.

#### 3. **Intelligent Scaling**

**KEDA-Based Autoscaling**
Monitors queue depth and scales pods based on actual LLM workload (not just CPU).

```yaml
Metric: vllm:num_requests_waiting
Rule: waiting > 10 requests → add pod
Rule: waiting = 0 for 5 min → remove pod
Result: Right-sized infrastructure, cost optimized
```

**Dynamic Batching**
Continuously batches incoming requests instead of waiting for fixed batch sizes.
- Benefit: Lower latency, higher throughput
- Automatically enabled in vLLM

#### 4. **Distributed Tracing & Observability**

Built-in integration with **Jaeger** and **OpenTelemetry** for end-to-end visibility.

```
User Request
  ↓ [Trace: 0.2ms waiting]
API Gateway
  ↓ [Trace: 1.3ms routing decision]
Load Balancer
  ↓ [Trace: 0.5ms network]
vLLM Instance (prefill)
  ↓ [Trace: 45ms prefill]
vLLM Instance (decode)
  ↓ [Trace: 3500ms generation]
Response
  └─ Total: 3550ms
  └─ Bottleneck: Decode phase (98% of time)
```

**What You Monitor:**
- Request routing decisions
- Prefill vs decode latency breakdown
- Queue depth at each stage
- Cache hit rates
- GPU utilization per instance
- Network latency between instances

#### 5. **KV Cache Sharing Across Instances**

Share KV cache across multiple vLLM instances for distributed inference.

```
Instance A (prefill)
  ├─ Processes prompt
  └─ Computes KV cache
       ↓
Instance B (decode 1)
       ↓
Instance C (decode 2)
       ↓
Instance D (decode 3)
  └─ All share same cached context
  └─ Result: 3× decode parallelism on same cached KV
```

Useful for:
- ✅ Long prompts (20K+ tokens)
- ✅ High concurrency (1000+ requests/min)
- ✅ Cost optimization (fewer GPUs needed)

#### 6. **Pipeline Parallelism with KubeRay**

For very large models (>70B), distribute model layers across multiple GPUs using KubeRay.

```
Model: Llama-70B on 4 GPUs

GPU 0: Layers 1–20
  ↓ (pass intermediate activations)
GPU 1: Layers 21–40
  ↓
GPU 2: Layers 41–60
  ↓
GPU 3: Layers 61–80
  ↓
Output

Benefit: Fit models that don't fit on single GPU
Cost: Network latency between GPUs (1–5ms per layer)
```

#### 7. **Sleep/Wakeup Mode for Multi-Model**

Allows models to "sleep" (offload weights) while preserving CUDA context.

See [vLLM Sleep Mode section](#vllm-sleep-mode-multi-model-serving-on-a-single-gpu) for full details.

**Quick Summary:**
- 30–37× faster model switching
- Single GPU serves 3+ models sequentially
- Perfect for multi-stage pipelines

### Production Stack Architecture

```
┌─────────────────────────────────────────────────┐
│              API Gateway / Ingress               │
│  (Rate limiting, auth, request validation)      │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  Router Service  │
        │                  │
        ├─ KV Cache Aware  │
        ├─ Semantic Aware  │
        └────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌───▼───┐   ┌───▼───┐
│ vLLM  │   │ vLLM  │   │ vLLM  │
│Pod 1  │   │Pod 2  │   │Pod 3  │
│H100   │   │H100   │   │H100   │
│8 GPUs │   │8 GPUs │   │8 GPUs │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    └───────────┼───────────┘
                │
        ┌───────▼────────┐
        │  Observability │
        │  ├─ Jaeger     │
        │  ├─ Prometheus │
        │  └─ Grafana    │
        └────────────────┘
```

### Deployment Options

#### Option 1: Single-Node Deployment (Dev/Small Scale)
```bash
# Simple Docker setup
docker run vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b \
  --tensor-parallel-size 1
```
- Best for: Testing, small scale (<1,000 req/min)
- Cost: 1 GPU
- Latency: P99 <2s

#### Option 2: Multi-Node Kubernetes (Production)
```yaml
# Production stack with autoscaling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prod-stack
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: "8"
---
# Autoscaling based on queue depth
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef:
    name: vllm-prod-stack
  triggers:
  - type: prometheus
    metadata:
      query: vllm:num_requests_waiting
      threshold: "10"
```
- Best for: Production, high scale (>10,000 req/min)
- Cost: 3–10 GPUs (auto-scales)
- Latency: P99 <5s with proper tuning

#### Option 3: Disaggregated Prefill/Decode (Cost Optimized)
```yaml
# Prefill pool: H100s (high compute)
prefill-pool:
  machine-type: a3-highgpu-8g (H100)
  replicas: 2
  
# Decode pool: L4s (high bandwidth, cheaper)
decode-pool:
  machine-type: g2-standard-24 (L4)
  replicas: 8
```
- Best for: High throughput, cost-sensitive (>50,000 req/min)
- Cost: 50–60% reduction vs unified serving
- Latency: P99 <8s

### Key Features Summary

| Feature | Benefit | When to Use |
|---|---|---|
| **KV Cache Aware Routing** | 10–50% latency reduction | Long documents, RAG |
| **Semantic Routing** | Better cache hits | Similar queries |
| **Disaggregated Prefill/Decode** | 40–60% cost reduction | High throughput |
| **KEDA Autoscaling** | Right-sized infrastructure | Variable traffic |
| **Distributed Tracing** | Find bottlenecks | Production debugging |
| **KV Cache Sharing** | 3–5× decode parallelism | Large batch jobs |
| **Pipeline Parallelism** | Fit 70B+ models | Very large models |
| **Sleep/Wakeup Mode** | Multi-model on 1 GPU | Sequential pipelines |

### Getting Started with Production Stack

**Step 1: Check Prerequisites**
- Kubernetes cluster (GKE, EKS, on-prem)
- 2+ GPUs (NVIDIA A100/H100/L4)
- Docker and kubectl installed
- Jaeger/OpenTelemetry (for tracing)

**Step 2: Deploy vLLM Pod**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f keda-scaler.yaml
```

**Step 3: Configure Routing**
Choose routing strategy:
```python
# Option A: KV Cache Aware Routing
router.enable_kv_cache_routing = True

# Option B: Semantic Aware Routing
router.enable_semantic_routing = True
```

**Step 4: Enable Tracing**
```yaml
env:
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: "http://jaeger:4317"
- name: OTEL_SDK_DISABLED
  value: "false"
```

**Step 5: Monitor & Optimize**
```bash
# Check queue depth
kubectl logs <pod-name> | grep "num_requests_waiting"

# View traces in Jaeger
curl http://jaeger:16686

# Monitor GPU utilization
nvidia-smi dmon
```

### Common Use Cases

#### Use Case 1: RAG at Scale
**Requirements:** Long documents (10K+ tokens), real-time serving
**Stack Configuration:**
- ✅ KV Cache Aware Routing (cache same document queries)
- ✅ KEDA autoscaling (handle traffic spikes)
- ✅ Disaggregated Prefill/Decode (cost optimization)
- ✅ Distributed tracing (debugging latency)

**Expected Performance:**
- Throughput: 500–2,000 req/min
- P99 Latency: <5s
- Cost: $50K–100K/month for 10 GPU nodes

#### Use Case 2: Multi-Model Serving
**Requirements:** Multiple specialized models (entity extraction, summarization, etc.)
**Stack Configuration:**
- ✅ Sleep/Wakeup Mode (single GPU, multiple models)
- ✅ Semantic Routing (route by query type)
- ✅ KEDA autoscaling (scale by queue depth)
- ✅ Pipeline Parallelism (if models are large)

**Expected Performance:**
- Throughput: 1,000–5,000 req/min
- P99 Latency: <8s
- Cost: $10K–30K/month for 2–4 GPU nodes

#### Use Case 3: Batch Inference
**Requirements:** High throughput, not latency-sensitive
**Stack Configuration:**
- ✅ KV Cache Sharing (maximize parallelism)
- ✅ Large batch sizes (8–16)
- ✅ Disaggregated Prefill/Decode (throughput focus)
- ✅ Long timeout windows (3–10 seconds)

**Expected Performance:**
- Throughput: 10,000–50,000 req/day
- P99 Latency: <30s
- Cost: $5K–15K/month for 2–3 GPU nodes

### Production Readiness Checklist

```yaml
DEPLOYMENT:
  ✓ Multi-node Kubernetes setup
  ✓ At least 2 replicas per service
  ✓ Resource limits and requests set
  ✓ Health checks (readiness + liveness probes)

SCALING:
  ✓ KEDA configured and tested
  ✓ Scale limits (min/max replicas) set
  ✓ Cooldown periods configured
  ✓ Load tested at 2× expected peak

OBSERVABILITY:
  ✓ Distributed tracing enabled
  ✓ Prometheus metrics scraping
  ✓ Grafana dashboards created
  ✓ Alert rules configured
  ✓ Log aggregation (ELK, Loki, etc.)

ROUTING:
  ✓ Smart routing strategy chosen
  ✓ Router failover configured
  ✓ Request timeout set (10–60s)
  ✓ Circuit breaker implemented

PERFORMANCE:
  ✓ vLLM parameters tuned (max_num_seqs, etc.)
  ✓ Quantization applied if needed
  ✓ Batching optimized for workload
  ✓ Cache settings configured

RELIABILITY:
  ✓ Backup/rollback strategy
  ✓ Graceful shutdown configured
  ✓ Pod disruption budgets set
  ✓ Regular disaster recovery drills
```

### Conclusion

The vLLM Production Stack abstracts away infrastructure complexity, letting you focus on:
- ✅ Model selection and fine-tuning
- ✅ Application logic
- ✅ Business metrics

Instead of managing:
- ❌ Kubernetes YAML files
- ❌ Network configuration
- ❌ Scaling policies
- ❌ Tracing setup
- ❌ Routing logic

It's a **batteries-included solution** for enterprise LLM serving—proven patterns, battle-tested components, and extensive tooling all working together seamlessly.