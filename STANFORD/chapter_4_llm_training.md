chapter_4_llm_training.md
# Resources

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |

---

# Additional Resources

- https://cme295.stanford.edu/syllabus/
- https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf

---

# Table of Contents

- [Resources](#resources)
- [Additional Resources](#additional-resources)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [FLOPs(Floating-point OPerations)](#flopsfloating-point-operations)
- [FLOPS or FLOP/s(Floating-point OPerations per Second)](#flops-or-flopsfloating-point-operations-per-second)
- [Scaling Law](#scaling-law)
  - [What are Scaling Laws?](#what-are-scaling-laws)
  - [Problem Solved](#problem-solved)
  - [How It Works: Small → Large Prediction](#how-it-works-small--large-prediction)
  - [Real-World Allocation Example](#real-world-allocation-example)
  - [How to Compute Scaling Rules for Your Model Training](#how-to-compute-scaling-rules-for-your-model-training)
    - [What is "Loss"?](#what-is-loss)
    - [How is Loss Computed?](#how-is-loss-computed)
    - [Step-by-Step: Fitting Your Scaling Law](#step-by-step-fitting-your-scaling-law)
    - [Practical Tips](#practical-tips)
  - [Sampling Efficiency](#sampling-efficiency)
    - [Key Concepts:](#key-concepts)
    - [Why It Matters:](#why-it-matters)
    - [Practical Example:](#practical-example)
    - [The Golden Rule:](#the-golden-rule)
  - [Chinchilla Law](#chinchilla-law)
    - [The Big Discovery:](#the-big-discovery)
    - [Real Example - The Chinchilla Model:](#real-example---the-chinchilla-model)
    - [Key Formula:](#key-formula)
    - [Why This Matters:](#why-this-matters)
    - [Practical Impact:](#practical-impact)
    - [Bottom Line:](#bottom-line)
- [Data Parallelism](#data-parallelism)
  - [What is Data Parallelism?](#what-is-data-parallelism)
    - [Simple Analogy:](#simple-analogy)
    - [How It Works:](#how-it-works)
    - [Libraries That Handle Data Parallelism:](#libraries-that-handle-data-parallelism)
    - [Key Concepts: What Do These Terms Mean?](#key-concepts-what-do-these-terms-mean)
      - [1. **Model Parameters** (Weights)](#1-model-parameters-weights)
      - [2. **Gradients**](#2-gradients)
      - [3. **Optimizer States** (Adam/AdamW)](#3-optimizer-states-adamadamw)
  - [ZeRO (Zero Redundancy Optimizer)](#zero-zero-redundancy-optimizer)
    - [ZeRO-1: Partition Optimizer States Only](#zero-1-partition-optimizer-states-only)
    - [ZeRO-2: Partition Optimizer States + Gradients](#zero-2-partition-optimizer-states--gradients)
    - [ZeRO-3: Partition Everything (Model + Gradients + Optimizer)](#zero-3-partition-everything-model--gradients--optimizer)
    - [ZeRO Summary: Which One to Use?](#zero-summary-which-one-to-use)
    - [Why ZeRO is Slower](#why-zero-is-slower)
  - [Model Parallelism](#model-parallelism)
    - [ZeRO vs Model Parallelism](#zero-vs-model-parallelism)
    - [Techniques in Model Parallelism](#techniques-in-model-parallelism)
      - [0. **Data Parallelism**](#0-data-parallelism)
      - [1. **Pipeline Parallelism** (Vertical Split)](#1-pipeline-parallelism-vertical-split)
      - [2. **Tensor Parallelism** (Horizontal Split)](#2-tensor-parallelism-horizontal-split)
      - [3. **3D Parallelism** (Combined Approach)](#3-3d-parallelism-combined-approach)
    - [Quick Comparison](#quick-comparison)
    - [Real-World Examples](#real-world-examples)
  - [Deep Dive: Parallelism Strategies in LLM Training](#deep-dive-parallelism-strategies-in-llm-training)
  - [Data Parallelism](#data-parallelism-1)
    - [What is Data Parallelism?](#what-is-data-parallelism-1)
    - [How It Works (Step-by-Step)](#how-it-works-step-by-step)
    - [Visual Example](#visual-example)
    - [Types of Data Parallelism](#types-of-data-parallelism)
      - [1. **Standard Data Parallelism (DDP - Distributed Data Parallel)**](#1-standard-data-parallelism-ddp---distributed-data-parallel)
      - [2. **Sharded Data Parallelism (ZeRO - Zero Redundancy Optimizer)**](#2-sharded-data-parallelism-zero---zero-redundancy-optimizer)
    - [Gradient Synchronization Methods](#gradient-synchronization-methods)
      - [**AllReduce Communication Pattern**](#allreduce-communication-pattern)
    - [Real-World Example](#real-world-example)
    - [When to Use Data Parallelism](#when-to-use-data-parallelism)
  - [Tensor Parallelism (Intra-Layer Model Parallelism)](#tensor-parallelism-intra-layer-model-parallelism)
    - [How It Works](#how-it-works-1)
    - [Megatron-LM: Column vs Row Parallel](#megatron-lm-column-vs-row-parallel)
    - [Memory Savings (GPT-3 layer, TP=8)](#memory-savings-gpt-3-layer-tp8)
    - [Real-World: LLaMA-65B with TP=8](#real-world-llama-65b-with-tp8)
  - [Pipeline Parallelism (Inter-Layer Model Parallelism)](#pipeline-parallelism-inter-layer-model-parallelism)
    - [The Bubble Problem](#the-bubble-problem)
    - [Fix: Micro-batching (GPipe)](#fix-micro-batching-gpipe)
    - [1F1B Schedule — Less Memory](#1f1b-schedule--less-memory)
    - [Communication Pattern](#communication-pattern)
    - [Real-World: GPT-3 175B with PP=16](#real-world-gpt-3-175b-with-pp16)
  - [Comparison: Data vs Tensor vs Pipeline Parallelism](#comparison-data-vs-tensor-vs-pipeline-parallelism)
    - [Quick Reference Table](#quick-reference-table)
    - [Visual Comparison](#visual-comparison)
    - [Which Parallelism to Choose?](#which-parallelism-to-choose)
      - [Decision Tree](#decision-tree)
      - [Practical Examples](#practical-examples)
  - [Data Parallelism vs Flash Attention: Key Differences](#data-parallelism-vs-flash-attention-key-differences)
    - [Quick Summary](#quick-summary)
    - [Key Differences Explained](#key-differences-explained)
      - [1. **Problem Domain**](#1-problem-domain)
      - [2. **Memory Behavior**](#2-memory-behavior)
      - [3. **When to Use**](#3-when-to-use)
      - [4. **Combination Effect**](#4-combination-effect)
    - [Real-World Impact](#real-world-impact)
    - [Summary](#summary)
- [Flash Attention](#flash-attention)
  - [What is Flash Attention?](#what-is-flash-attention)
    - [The Problem It Solves](#the-problem-it-solves)
    - [Key Components](#key-components)
  - [Traditional Self-Attention Computation](#traditional-self-attention-computation)
    - [Memory Traffic Example](#memory-traffic-example)
  - [Flash Attention Solution](#flash-attention-solution)
    - [Core Innovation](#core-innovation)
    - [How It Works](#how-it-works-2)
    - [Block-wise Processing Visual](#block-wise-processing-visual)
  - [Comparison: Standard vs Flash Attention](#comparison-standard-vs-flash-attention)
    - [Memory Access Patterns](#memory-access-patterns)
  - [Flash Attention Implementation in Code](#flash-attention-implementation-in-code)
    - [1. Using the Official Flash Attention Library (Most Common)](#1-using-the-official-flash-attention-library-most-common)
    - [Key Benefits in Training](#key-benefits-in-training)
    - [Important Notes](#important-notes)
- [Quantization](#quantization)
- [Mixed Precision Training](#mixed-precision-training)
  - [What is Mixed Precision Training?](#what-is-mixed-precision-training)
  - [Purpose](#purpose)
  - [How It Works (Simple Explanation)](#how-it-works-simple-explanation)
    - [Why This Split?](#why-this-split)
  - [Real Example for Beginners](#real-example-for-beginners)
  - [Concrete Benefits Example](#concrete-benefits-example)
  - [Key Insight](#key-insight)
  - [When to Use](#when-to-use)
- [Supervised Fine-tuning](#supervised-fine-tuning)
  - [What is Supervised Fine-tuning?](#what-is-supervised-fine-tuning)
  - [The Process](#the-process)
  - [How It Works](#how-it-works-3)
  - [Real-World Example](#real-world-example-1)
  - [Common Fine-tuning Tasks](#common-fine-tuning-tasks)
  - [Why Fine-tune?](#why-fine-tune)
  - [Simple Code Example](#simple-code-example)
  - [Key Takeaway](#key-takeaway)
  - [Instruction Tuning](#instruction-tuning)
    - [What Makes It Special?](#what-makes-it-special)
    - [Key Difference from Regular Fine-tuning](#key-difference-from-regular-fine-tuning)
    - [Simple Example](#simple-example)
    - [Why It Matters](#why-it-matters-1)
    - [Challenges](#challenges)
  - [SFT vs CFT: What's the Difference?](#sft-vs-cft-whats-the-difference)
    - [Quick Summary](#quick-summary-1)
    - [Detailed Comparison](#detailed-comparison)
    - [Real-World Scenario: Building a Medical Chatbot](#real-world-scenario-building-a-medical-chatbot)
    - [Why Both Matter](#why-both-matter)
    - [Simple Analogy](#simple-analogy-1)
    - [The Pipeline](#the-pipeline)
    - [Real Examples](#real-examples)
    - [When Do You Need Which?](#when-do-you-need-which)
    - [Key Takeaway](#key-takeaway-1)
  - [What is Alignment in LLMs?](#what-is-alignment-in-llms)
    - [The Problem](#the-problem)
    - [What Alignment Teaches](#what-alignment-teaches)
    - [How It's Done](#how-its-done)
    - [Why It Matters](#why-it-matters-2)
    - [Key Takeaway](#key-takeaway-2)
  - [Temperature in LLMs](#temperature-in-llms)
    - [What is Temperature?](#what-is-temperature)
    - [How It Works](#how-it-works-4)
    - [The Math: How Probabilities Are Actually Adjusted](#the-math-how-probabilities-are-actually-adjusted)
    - [Visual Summary](#visual-summary)
    - [Key Insight](#key-insight-1)
    - [Simple Examples](#simple-examples)
    - [Real-World Impact](#real-world-impact-1)
    - [Important Clarification: Temperature Does NOT Change Self-Attention or FFN](#important-clarification-temperature-does-not-change-self-attention-or-ffn)
    - [Practical Example](#practical-example-1)
    - [Key Takeaway](#key-takeaway-3)
- [Benchmarking (Evaluation)](#benchmarking-evaluation)
  - [What is Benchmarking?](#what-is-benchmarking)
    - [Why We Need It](#why-we-need-it)
    - [How It Works](#how-it-works-5)
    - [What is MMLU?](#what-is-mmlu)
    - [Simple Example](#simple-example-1)
    - [Common Benchmarks](#common-benchmarks)
    - [Why Benchmarks Matter](#why-benchmarks-matter)
    - [Important Note](#important-note)
    - [Key Takeaway](#key-takeaway-4)
- [Parameter-Efficient Finetuning(PEFT) with LoRA](#parameter-efficient-finetuningpeft-with-lora)
    - [Where Are LoRA Weights Updated?](#where-are-lora-weights-updated)
  - [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
    - [The Problem](#the-problem-1)
    - [LoRA's Solution](#loras-solution)
    - [How LoRA Works - Visual Explanation](#how-lora-works---visual-explanation)
    - [The Math Behind LoRA](#the-math-behind-lora)
    - [Concrete Example](#concrete-example)
    - [Visual: Parameter Comparison](#visual-parameter-comparison)
    - [Simple Analogy](#simple-analogy-2)
    - [Key Benefits Visualization](#key-benefits-visualization)
    - [Key Takeaway](#key-takeaway-5)
    - [LoRA Deep Dive: Common Questions](#lora-deep-dive-common-questions)
      - [Q1: Are adapter matrices added to pretrained weights or run in parallel?](#q1-are-adapter-matrices-added-to-pretrained-weights-or-run-in-parallel)
      - [Q2: Why does it matter which target\_modules we update?](#q2-why-does-it-matter-which-target_modules-we-update)
      - [Q3: How are adapter weights saved in the final fine-tuned model?](#q3-how-are-adapter-weights-saved-in-the-final-fine-tuned-model)
      - [Q4: What is Alpha (α) and why does it matter?](#q4-what-is-alpha-α-and-why-does-it-matter)
  - [QLoRA (Quantized LoRA)](#qlora-quantized-lora)
    - [What is QLoRA?](#what-is-qlora)
    - [Memory Comparison](#memory-comparison)
    - [Real Example: Llama-2 65B Fine-tuning](#real-example-llama-2-65b-fine-tuning)
    - [Trade-offs](#trade-offs)
    - [When to Use](#when-to-use-1)
    - [Key Takeaway](#key-takeaway-6)
- [Model Pruning — Beginner's Guide](#model-pruning--beginners-guide)
  - [What Is Pruning?](#what-is-pruning)
  - [Core Idea](#core-idea)
  - [Types of Pruning](#types-of-pruning)
    - [1. Weight Pruning (Unstructured)](#1-weight-pruning-unstructured)
    - [2. Structured Pruning (More Practical)](#2-structured-pruning-more-practical)
  - [Which Layers Get Pruned in LLMs?](#which-layers-get-pruned-in-llms)
  - [How Importance Is Measured](#how-importance-is-measured)
  - [Pruning + Fine-tuning (Standard Pipeline)](#pruning--fine-tuning-standard-pipeline)
  - [Outcome of Pruning](#outcome-of-pruning)
  - [Pruning vs Quantization — Key Difference](#pruning-vs-quantization--key-difference)
  - [Summary](#summary-1)

---

# Overview

**Overview of LLM training process showing key stages: data collection → preprocessing → model training → final trained model. Understanding the big picture of how powerful AI systems are created.**

<img src="images/overview_training_process.png" alt="LLM Training Process Overview" width="700">

**Data pipeline for LLM training: raw text flows through various preprocessing steps before being used to train the model. Massive amounts of internet text are transformed into usable training data.**

<img src="images/overview_data_pipeline.png" alt="Data Pipeline" width="700">

`Common crawl (https://commoncrawl.org/) has repository of web crawl data. Over 300 billion pages spanning 18 years. 3–5 billion new pages added each month.`

**Common Crawl dataset: one of the **largest publicly available datasets** with **hundreds of billions of web pages** collected over many years. Understanding the **massive scale of data** used to train modern language models.**

<img src="images/overview_common_crawl.png" alt="Common Crawl Dataset" width="700">

**Data filtering and quality control process: Not all web pages are suitable for LLM training. **Filtering out spam, duplicates, and low-quality content** to keep only **high-quality text** that helps the model learn effectively.**

<img src="images/overview_data_filtering.png" alt="Data Filtering Process" width="700">


# FLOPs(Floating-point OPerations)

**FLOPs = counting **total math operations** (additions, multiplications) needed to train a model. For LLMs, this number is **incredibly large - trillions or more**. Key metric for understanding **computational cost** of training.**

<img src="images/flops_operations.png" alt="FLOPs Operations" width="650">

# FLOPS or FLOP/s(Floating-point OPerations per Second)

**FLOPS = **speed** of computation. While **FLOPs = total operations needed**, **FLOPS = how fast** hardware (GPUs) executes them. **Higher FLOPS = faster training**. Compares different hardware options showing why **powerful GPUs or specialized AI chips** are necessary for efficient LLM training.**

<img src="images/flops_speed.png" alt="FLOPs and Speed Comparison" width="650">

# Scaling Law

`https://arxiv.org/abs/2001.08361`

## What are Scaling Laws?

Mathematical relationships that **predict LLM performance** before training, based on three variables:

| Variable | What it means |
|---|---|
| **Model size (N)** | Number of parameters |
| **Dataset size (D)** | Training tokens |
| **Compute (C)** | FLOPs = N × D × 6 (approx) |

**Key insight:** Performance (loss) improves **smoothly and predictably** as you scale these up — doubling compute gives consistent, measurable gains.

---

## Problem Solved

**Before:** Spend millions training a large model → find it didn't improve → wasteful guesswork.

**After:** Run cheap small experiments → fit a curve → **predict large model loss before spending anything**.

---

## How It Works: Small → Large Prediction

Train cheap small models, observe the loss curve, extrapolate:

```
1M params,  10M tokens  → Loss: 3.5  ┐
10M params, 100M tokens → Loss: 2.8  ├─ fit power-law curve
100M params, 1B tokens  → Loss: 2.3  ┘
                                ↓
Prediction: 1B params, 10B tokens → Loss ≈ 1.9
            (without training it!)
```

If your target is loss < 2.0 → you now know the minimum scale needed → budget accordingly.

---

## Real-World Allocation Example

**$1M budget, which option?**

| Option | Params | Tokens | Predicted Loss |
|---|---|---|---|
| A | 10B | 100B | 2.1 |
| B | 3B | 500B | 1.95 |
| C | 7B | 250B | **1.85** ✅ |

Scaling laws show Option C wins — **before spending a dollar on full training**.

<img src="images/scaling_law_overview.png" alt="Scaling Law Overview" width="700">

## How to Compute Scaling Rules for Your Model Training

### What is "Loss"?

**Loss** = how wrong the model's predictions are. Lower = better.

$$\text{Loss} = -\log(P_{\text{correct}})$$

`P_correct` = probability assigned to the correct next token.

```
"The cat sat on the ___"  correct = "mat"

Model confident (P=0.80) → Loss = -log(0.80) = 0.22  ✅
Model uncertain (P=0.20) → Loss = -log(0.20) = 1.61
Model wrong     (P=0.02) → Loss = -log(0.02) = 3.91  ❌
```

**Typical benchmarks:**

| Model state | Loss |
|---|---|
| Untrained (random) | 10–12 |
| Decent | 2–3 |
| Good | 1.5–2.0 |
| State-of-the-art | 1.0–1.5 |

**Training vs Test Loss:**
- **Training loss** → measures memorization (used to update weights)
- **Test loss** → measures generalization on unseen data ← scaling laws use this

### How is Loss Computed?

Each sentence yields multiple training signals — one per token:

```
"The cat sat on the mat"
  → predict "cat"  from "The"
  → predict "sat"  from "The cat"
  → predict "on"   from "The cat sat"
  → predict "the"  from "The cat sat on"
  → predict "mat"  from "The cat sat on the"
  5 predictions from 1 sentence
```

Per step: forward pass → compute loss → backprop → update weights.

### Step-by-Step: Fitting Your Scaling Law

**Step 1: Train 3–5 small models**

```
Model A:  10M params,  50M tokens  → Test Loss: 3.2
Model B:  50M params, 250M tokens  → Test Loss: 2.6
Model C: 100M params, 500M tokens  → Test Loss: 2.3
Model D: 200M params,   1B tokens  → Test Loss: 2.0
```

**Step 2: Fit the power-law formula**

$$L(N, D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + L_{\infty}$$

| Term | Meaning |
|---|---|
| N, D | Parameters, tokens |
| α, β | How much each factor reduces loss |
| A, B | Fitted magnitude constants |
| L∞ | Irreducible minimum loss |

Plot log(Loss) vs log(N) and log(D) — straight lines confirm a power law. Slope = −α or −β.

```
Log-Log plot (Loss vs Parameters):

Log(Loss)
  1.2 ●
  1.0    ●
  0.9       ●
  0.7          ●
               ╲← slope = −α = −0.36
      ──────────────→ Log(Parameters)

Real values (Chinchilla):  α ≈ 0.076,  β ≈ 0.095
→ Data matters slightly more than model size
```

**Step 3: Predict large model loss**

$$L_{\text{new}} = L_{\text{current}} \times \left(\frac{N_c}{N_{\text{new}}}\right)^{\alpha} \times \left(\frac{D_c}{D_{\text{new}}}\right)^{\beta}$$

```
Current: 200M params, 1B tokens → Loss 2.0
Target:  1B params,  5B tokens  → Loss?

= 2.0 × (200M/1B)^0.076 × (1B/5B)^0.095
= 2.0 × 0.88 × 0.87 = 1.53  ← predicted without training!
```

**Step 4: Optimal compute allocation (Chinchilla ratio)**

$$\text{FLOPs} \approx 6 \times N \times D, \quad N : D = 1 : 20$$

| Compute Budget | Optimal Params | Optimal Tokens |
|---|---|---|
| 1e18 | 91M | 1.8B |
| 1e20 | 910M | 18B |
| 1e21 | 2.9B | 58B |
| 1e22 | 9.1B | 180B |
| 1e23 | 29B | 580B |

### Practical Tips

1. **Start tiny** — 3–4 small runs are enough to fit the curve
2. **Always use test loss** — training loss overfits, misleads predictions
3. **Validate mid-scale** — test prediction on a medium model before committing to full scale
4. **Domain matters** — α and β differ for code vs text vs multilingual data

## Sampling Efficiency

**Sampling Efficiency** measures how well a model learns from each piece of training data - essentially, **how many times you need to see the same data** for the model to learn effectively.

### Key Concepts:

- **High Sampling Efficiency** = Model learns quickly from fewer data exposures (fewer epochs needed)
- **Low Sampling Efficiency** = Model needs to see data multiple times to learn (more epochs needed)

### Why It Matters:

Training on the **same data multiple times (multiple epochs)** has **diminishing returns**:
- **1st pass**: Model learns a lot
- **2nd pass**: Model learns less
- **3rd+ pass**: Minimal learning, risk of **overfitting** (memorizing instead of understanding)

### Practical Example:

**Scenario A - Low Efficiency:**
- Train a 10B parameter model on 100B tokens
- Need to loop through data **10 times** (10 epochs)
- Model sees the same Wikipedia article 10 times before learning it

**Scenario B - High Efficiency:**  
- Train a 5B parameter model on 500B tokens
- Need only **2 passes** (2 epochs)
- Model sees fresh, diverse data - learns better patterns

**Result:** Scenario B typically performs better because the model learns from **diverse experiences** rather than memorizing the same limited dataset.

### The Golden Rule:

**More unique data is better than repeatedly training on the same data.** Sampling efficiency tells you when you've reached the point of diminishing returns.

<img src="images/scaling_law_sampling_efficiency.png" alt="Sampling Efficiency" width="700">

## Chinchilla Law

**Chinchilla Law** is a breakthrough scaling law discovery from DeepMind (2022) that revealed: **most large language models are severely undertrained** - they're too big for the amount of data they're trained on.

### The Big Discovery:

**Previous approach (like GPT-3):**
- Build **massive models** (175B parameters)
- Train on **relatively less data** (300B tokens)
- Assumption: Bigger model = Better performance

**Chinchilla's finding:**
- For optimal performance, **model size and training data should scale equally**
- **Rule: For every <mark>1 parameter</mark>, you need about <mark>20 tokens</mark> of training data**
- Many models were "data-starved" - too big for their training data

### Real Example - The Chinchilla Model:

**Gopher (Previous model):**
- 280B parameters
- 300B tokens
- Result: Good but inefficient

**Chinchilla (Optimized):**
- 70B parameters (**4x smaller!**)
- 1.4T tokens (**4.6x more data**)
- Result: **Better performance** despite being much smaller!

**Original Understanding (2020):**

```
"More data + bigger model = better performance"
Scaling: Just add more raw data
```

**Updated Understanding (2022+):**

```
"Data quality × model size = optimal performance"
Chinchilla findings:
- Optimal ratio: 20 tokens per parameter
- But quality matters MORE than hitting the ratio
```

**Real Example:**

```
Model A: 70B params, 1.4T tokens (raw) = Poor performance
Model B: 70B params, 400B tokens (curated) = Better performance!

Model B has 3.5× LESS data but higher quality
→ Outperforms Model A by 8-12% on benchmarks
```

### Key Formula:

**Optimal ratio: N (parameters) : D (tokens) = 1 : 20**

For a model with:
- **10B parameters** → Train on **200B tokens**
- **100B parameters** → Train on **2T (trillion) tokens**

### Why This Matters:

**Before Chinchilla Law:**
- Companies built huge 500B+ parameter models
- Training cost: Extremely high
- Inference cost: Very expensive (bigger model = more compute per query)

**After Chinchilla Law:**
- Build **smaller, data-rich models**
- **Same or better performance**
- **Much cheaper** to train and run
- More efficient use of resources

### Practical Impact:

**Old way:** "Let's make GPT-5 with 1 trillion parameters!"
- Training cost: $100M+
- Running cost: $1 per 1000 queries

**Chinchilla way:** "Let's make a 50B parameter model with 1T tokens of data"
- Training cost: $20M
- Running cost: $0.10 per 1000 queries
- Performance: Equal or better ✅

### Bottom Line:

**Chinchilla Law proves: It's better to train a smaller model on more data than a giant model on less data.** This revolutionized how companies approach LLM development, leading to more efficient models like Llama 2, Mistral, and others.

<img src="images/scaling_law_chinchilla_comparison.png" alt="Chinchilla Scaling Comparison" width="700">

<img src="images/scaling_law_chinchilla_details.png" alt="Chinchilla Details" width="700">

---

# Data Parallelism

## What is Data Parallelism?

**Data Parallelism** is a training strategy where you split your training data across multiple GPUs, but each GPU has a **complete copy** of the model.

**Used for:** Pre-training AND fine-tuning large models. Same memory requirements for components being trained.

### Simple Analogy:
Imagine 4 students (GPUs) learning the same textbook (model):
- Each student has their own copy of the textbook
- They practice different problems (different data batches)
- They share their answers to learn together

### How It Works:
```
GPU 0: [Full Model Copy] + processes batch 1
GPU 1: [Full Model Copy] + processes batch 2  
GPU 2: [Full Model Copy] + processes batch 3
GPU 3: [Full Model Copy] + processes batch 4

After each step: GPUs sync gradients to update all copies
```

**Problem:** Each GPU stores the SAME model → wasteful memory usage! 🔴

This is where **ZeRO** helps by eliminating redundant copies.

### Libraries That Handle Data Parallelism:

**For Training (Fine-tuning):**
- **DeepSpeed** - Most popular, implements ZeRO stages 1/2/3
- **PyTorch FSDP** - Native PyTorch, similar to ZeRO-3
- **Hugging Face Accelerate** - Simplifies multi-GPU, uses DeepSpeed/FSDP
- **Megatron-LM** - Tensor + Pipeline Parallelism for massive models

**For Inference (Serving):**
- **vLLM** - Uses Tensor Parallelism (NOT Data Parallelism), inference only
- **TensorRT-LLM** - Optimized inference with Tensor Parallelism

**Key Distinction**:
- Training libraries (DeepSpeed, FSDP) use **Data Parallelism** (replicate model, split data)
- Inference libraries (vLLM) use **Tensor Parallelism** (split model, same input)

**Use Data Parallelism for fine-tuning when:**
- Full fine-tuning large models (7B+) → Required
- LoRA/QLoRA fine-tuning → Optional (for speed)
- Small models (< 1B) → Optional (for speed)

---

### Key Concepts: What Do These Terms Mean?

Before understanding ZeRO, let's clarify three important components stored in GPU memory during training:

#### 1. **Model Parameters** (Weights)
- The actual numbers that make up your neural network
- **Example:** In a layer with 1000 neurons, each neuron has weights
- **Size for 7B model:** ~14 GB (in fp16)
- **Analogy:** The knowledge stored in a student's brain

#### 2. **Gradients**
- `∂Loss/∂W` for every weight — "how much did this weight cause the mistake?"
- Computed **layer by layer backward** (chain rule) — each layer needs the layer above's gradient to compute its own
- Used to update weights: `W_new = W_old - lr × gradient`, then **discarded**
- **Size:** Same as model parameters (~14GB for 7B) — doubles memory at minimum
- Not "weighted loss" — they are partial derivatives telling each weight which direction to move and by how much

#### 3. **Optimizer States** (Adam/AdamW)
- Extra memory used by the optimizer to train smarter
- For Adam: stores **momentum (m)** and **variance (v)** for each parameter

**What is Momentum (m)?**
- **Simple idea**: Remembers which direction the training has been going
- **Formula**: `m_t = 0.9 × m_{t-1} + 0.1 × gradient_t` (90% old direction + 10% new)
- **Why?**: Smooths out noisy gradients, builds speed in consistent directions
- **Analogy**: A ball rolling downhill
  - If you keep pushing it the same way → it speeds up (gains momentum)
  - If the direction keeps changing → momentum prevents zigzagging

**Example:**
```
Step 1: gradient = -0.5, momentum = -0.05
Step 2: gradient = -0.4, momentum = -0.085  (building up!)
Step 3: gradient = -0.6, momentum = -0.14   (more momentum!)
→ Training accelerates in the consistent downward direction
```

**What is Variance (v)?**
- **Simple idea**: Remembers how much the gradients have been jumping around
- **Formula**: `v_t = 0.999 × v_{t-1} + 0.001 × gradient_t²` (tracks gradient size history)
- **Why?**: Adapts learning rate for each parameter individually
  - Stable gradients (low variance) → take bigger steps
  - Jumpy gradients (high variance) → take smaller, careful steps
- **Analogy**: Walking on different terrain
  - Smooth flat road → walk faster (large steps)
  - Rocky unstable cliff → walk slower (small steps)

**Example:**
```
Parameter A: gradients = [0.5, 0.4, 0.5, 0.4] → stable, low variance
  → Adam takes LARGER steps (confident)

Parameter B: gradients = [0.1, 5.0, 0.2, 4.8] → jumpy, high variance
  → Adam takes SMALLER steps (cautious)
```

**Why Both Together?**
- **Momentum**: Knows WHERE to go (direction)
- **Variance**: Knows HOW FAST to go (step size)
- **Result**: Each parameter gets a customized update speed and direction

**Adam Update Formula (How They're Used):**
```
Step 1: Compute momentum (m)
  m_t = β₁ × m_{t-1} + (1 - β₁) × gradient_t
  (β₁ = 0.9, keeps 90% old momentum + 10% new gradient)

Step 2: Compute variance (v)
  v_t = β₂ × v_{t-1} + (1 - β₂) × gradient_t²
  (β₂ = 0.999, tracks gradient magnitude history)

Step 3: Bias correction (compensate for initial zero values)
  m̂_t = m_t / (1 - β₁ᵗ)
  v̂_t = v_t / (1 - β₂ᵗ)

Step 4: Update parameter (THIS IS WHERE MAGIC HAPPENS!)
  weight_new = weight_old - learning_rate × m̂_t / (√v̂_t + ε)
                                            ↑       ↑
                                      momentum  variance
                                      (direction) (adaptive step)
```

**Breaking Down the Update:**
```
weight_new = weight_old - α × (m̂_t / √v̂_t)
                          ↑    ↑      ↑
                    learning  |   variance adapts
                    rate      |   the step size
                             |
                       momentum gives
                       the direction
```

- **Momentum (m̂_t)**: In numerator → controls direction of update
- **Variance (√v̂_t)**: In denominator → controls size of update
  - Large variance → divide by bigger number → smaller step
  - Small variance → divide by smaller number → larger step

**Memory:**
- **Size:** 2× model parameters (~28 GB for 7B model) - stores both m and v
- **Why 2×?**: One copy for momentum values, one copy for variance values

**Complete Analogy:**
- **Parameters**: Your knowledge (what you know)
- **Gradients**: What you got wrong (corrections needed)
- **Momentum**: Your learning velocity (building understanding in a direction)
- **Variance**: Your adaptive pace (slow down on hard topics, speed up on easy ones)

**Total Memory Per GPU (7B model):**
```
Model Parameters:    14 GB
Gradients:           14 GB
Optimizer States:    28 GB (2× parameters for Adam)
─────────────────────────
Total:              ~56 GB per GPU
```

---

<img src="images/data_parallelism_overview.png" alt="Data Parallelism Overview" width="700">

<img src="images/data_parallelism_detailed.png" alt="Data Parallelism Detailed" width="700">

---

## ZeRO (Zero Redundancy Optimizer)

**Problem with Standard Data Parallelism:** Every GPU stores a full copy of everything (parameters, gradients, optimizer states). For 4 GPUs, you're storing 4× redundant data! 

**ZeRO Solution:** Partition (split) the data across GPUs instead of replicating. Each GPU stores only a portion.

**Key Idea:** 
- Standard: Each GPU has 100% of everything
- ZeRO: Each GPU has 25% (for 4 GPUs), shares when needed

---

### ZeRO-1: Partition Optimizer States Only

**What Gets Split:**
- ✅ **Optimizer States** → Divided across GPUs (each GPU stores 1/4)
- ❌ **Gradients** → Full copy on each GPU
- ❌ **Model Parameters** → Full copy on each GPU

**Simple Explanation:**
```
4 GPUs training 7B model:

GPU 0: [Full Model 14GB] + [Full Gradients 14GB] + [Optimizer 1/4 = 7GB]
GPU 1: [Full Model 14GB] + [Full Gradients 14GB] + [Optimizer 1/4 = 7GB]
GPU 2: [Full Model 14GB] + [Full Gradients 14GB] + [Optimizer 1/4 = 7GB]
GPU 3: [Full Model 14GB] + [Full Gradients 14GB] + [Optimizer 1/4 = 7GB]

Memory per GPU: 35 GB (was 56 GB)
Savings: 21 GB per GPU (37% reduction) ✓
```

**When to Use:** Model + gradients fit in memory, but optimizer states don't.

**Analogy:** 4 students each have the full textbook and their notes, but split the study strategy tracker.

<img src="images/zero1_optimizer_partition.png" alt="ZeRO-1: Optimizer State Partitioning" width="700">

---

### ZeRO-2: Partition Optimizer States + Gradients

**What Gets Split:**
- ✅ **Optimizer States** → Divided across GPUs (each GPU stores 1/4)
- ✅ **Gradients** → Divided across GPUs (each GPU stores 1/4)
- ❌ **Model Parameters** → Full copy on each GPU

**Simple Explanation:**
```
4 GPUs training 7B model:

GPU 0: [Full Model 14GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]
GPU 1: [Full Model 14GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]
GPU 2: [Full Model 14GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]
GPU 3: [Full Model 14GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]

Memory per GPU: 24.5 GB (was 56 GB)
Savings: 31.5 GB per GPU (56% reduction) ✓
```

**When to Use:** Model fits barely, need to save gradient memory too.

**Analogy:** 4 students share textbook copies, but split both notes and study trackers.

<img src="images/zero2_optimizer_gradient_partition.png" alt="ZeRO-2: Optimizer + Gradient Partitioning" width="700">

---

### ZeRO-3: Partition Everything (Model + Gradients + Optimizer)

**What Gets Split:**
- ✅ **Optimizer States** → Divided across GPUs (each GPU stores 1/4)
- ✅ **Gradients** → Divided across GPUs (each GPU stores 1/4)
- ✅ **Model Parameters** → Divided across GPUs (each GPU stores 1/4)

**Simple Explanation:**
```
4 GPUs training 7B model:

GPU 0: [Model 1/4 = 3.5GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]
GPU 1: [Model 1/4 = 3.5GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]
GPU 2: [Model 1/4 = 3.5GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]
GPU 3: [Model 1/4 = 3.5GB] + [Gradients 1/4 = 3.5GB] + [Optimizer 1/4 = 7GB]

Memory per GPU: 14 GB (was 56 GB)
Savings: 42 GB per GPU (75% reduction) ✓✓✓
```

**When to Use:** Training MASSIVE models (65B+) that don't fit on single GPU even for the model alone.

**How It Works:** GPUs gather (share) model pieces when needed for computation, then discard after use.

**Trade-off:** More communication between GPUs (slower ~20-30%), but can train 4× larger models!

**Analogy:** 4 students split the textbook into chapters, split notes, split trackers. They share pages when needed.

<img src="images/zero3_full_partition.png" alt="ZeRO-3: Full Partitioning (Model + Optimizer + Gradients)" width="700">

---

### ZeRO Summary: Which One to Use?

| ZeRO Stage | What's Split? | Memory Savings | Speed Impact | Use Case |
|------------|---------------|----------------|--------------|----------|
| **ZeRO-1** | Optimizer only | 37% | ~5% slower | Model+gradients fit, optimizer doesn't |
| **ZeRO-2** | Optimizer + Gradients | 56% | ~10% slower | Medium-large models (13B-30B) |
| **ZeRO-3** | Everything! | 75% | ~20-30% slower | Massive models (65B-175B+) |

**Real Example: LLaMA-2 70B**
- Without ZeRO: Needs ~140 GB per GPU ❌ Doesn't fit on A100 80GB!
- With ZeRO-3: Needs ~45 GB per GPU ✓ Fits comfortably!

**Key Insight:** ZeRO lets you train models that would otherwise be impossible! You trade speed for the ability to train at all.

### Why ZeRO is Slower

ZeRO splits data across GPUs to save memory — GPUs must **communicate to reassemble** that data whenever needed. That communication is the cost.

```
Standard (no ZeRO):  each GPU holds everything → no communication → fast
                     but: 140GB per GPU for 70B model → doesn't fit ❌

ZeRO:                each GPU holds a slice → must fetch other slices → slower
                     but: 45GB per GPU → fits ✅
```

**What causes slowdown per stage:**

| Stage | What's fetched each step | Overhead |
|---|---|---|
| ZeRO-1 | Optimizer states (once, at update only) | ~5% — minimal |
| ZeRO-2 | Gradients (every layer, during backward) | ~10% — layer-by-layer sync |
| ZeRO-3 | Weights (every layer, forward + backward) | ~20-30% — double fetch per layer |

**ZeRO-3 in detail:**
```
Forward pass:  GPU needs layer weights → fetch from other GPUs
Backward pass: GPU needs layer weights again → fetch again
               + gradient shard sync across GPUs
= weights fetched twice per layer, every step → heavy communication
```

**The network is the bottleneck:**
```
GPU compute:        ~312 TFLOPS   (A100)
GPU↔GPU NVLink:     ~600 GB/s     (fast, but finite)
Inter-node (IB):    ~200 GB/s     (slower across servers)

ZeRO-3 on 70B: ~280GB transferred per step → GPU sits idle ~0.5s waiting
```

> ZeRO doesn't slow down compute — it adds **communication overhead**. The GPU sits idle waiting for weight/gradient shards from other GPUs. That idle time is the slowdown.

---

## Model Parallelism

**What is Model Parallelism?**

When your model is **too large to fit on a single GPU**, you split the model itself across multiple GPUs. Unlike Data Parallelism (which replicates the full model), Model Parallelism divides the model's layers or operations.

**Why Use It?** To train models that are physically too big for one GPU's memory (e.g., GPT-3 175B parameters ≈ 350GB).

### ZeRO vs Model Parallelism

**ZeRO — Splits Training State**

```
All GPUs hold the SAME data (same batch)
All GPUs run the SAME layers

But training state is partitioned:
  GPU 0: owns optimizer shard 0, gradient shard 0, weight shard 0
  GPU 1: owns optimizer shard 1, gradient shard 1, weight shard 1
  GPU 2: owns optimizer shard 2, gradient shard 2, weight shard 2

When a GPU needs a weight it doesn't own → fetch → use → discard

```
**Purpose**: Reduce memory per GPU. Still uses data parallelism — each GPU processes a different mini-batch.

---

**Model Parallelism — Splits the Model**
```
GPU 0: owns Layer 1–20    → processes ALL data through layers 1–20
GPU 1: owns Layer 21–40   → receives output from GPU 0, processes layers 21–40
GPU 2: owns Layer 41–60   → receives output from GPU 1, processes layers 41–60
GPU 3: owns Layer 61–80   → final output

Each GPU never sees layers it doesn't own
```
Purpose: Fit a model that physically cannot fit on one GPU — even with ZeRO.

Both split across GPUs — but split **different things** for different reasons:

```
ZeRO:              splits training STATE   (optimizer / gradients / weights)
Model Parallelism: splits the MODEL itself (layers or tensors)
```

| | ZeRO | Model Parallelism |
|---|---|---|
| **What's split** | Optimizer, gradients, weights | Layers or matrix operations |
| **Each GPU sees all layers?** | Yes — fetches shards when needed | No — owns only its assigned layers |
| **Data flow** | Each GPU processes different mini-batch | Data flows GPU→GPU→GPU sequentially |
| **Communication** | All-reduce (broadcast shards) | Point-to-point (pass activations forward) |
| **Why use it** | Training state too large for one GPU | Model too large to fit on one GPU at all |

```
Model fits on 1 GPU but training state doesn't → ZeRO only
Model doesn't fit on 1 GPU at all             → Model Parallelism
Both problems (e.g. GPT-3 175B)               → ZeRO + Model Parallelism = 3D Parallelism
```

> **Analogy:**
> ZeRO — same recipe, ingredients split across 4 fridges. Each GPU cooks the full dish, fetching ingredients as needed.
> Model Parallelism — recipe split into 4 stages. GPU 0 does prep → GPU 1 cooks → GPU 2 plates. Each GPU only knows its stage.

---

### Techniques in Model Parallelism

#### 0. **Data Parallelism**

Each GPU holds the **full model** but processes a **different mini-batch**. Gradients are averaged across GPUs after each step.

```
GPU 0: full model → batch A → gradient_A
GPU 1: full model → batch B → gradient_B
GPU 2: full model → batch C → gradient_C
                ↓
        All-reduce: average gradients
                ↓
        All GPUs update weights identically
```

- ✅ Simple, fast — no sequential dependency between GPUs
- ✅ Linear speedup with more GPUs
- ❌ Requires full model to fit on each GPU → fails for very large models
- **ZeRO is Data Parallelism + partitioned training state** — same idea, less memory

---

#### 1. **Pipeline Parallelism** (Vertical Split)

Split model **by layers** across GPUs - like stages in a pipeline.

```
GPU 0: [Layers 1-8]   → Forward pass → Send activations to GPU 1
GPU 1: [Layers 9-16]  → Forward pass → Send activations to GPU 2  
GPU 2: [Layers 17-24] → Forward pass → Send activations to GPU 3
GPU 3: [Layers 25-32] → Forward pass → Output

Data flows: GPU 0 → GPU 1 → GPU 2 → GPU 3 (like an assembly line)
```

**Analogy:** Assembly line with 4 stations - each station handles specific layers.

**Trade-off:**
- ✅ Simple to implement, each GPU stores fewer layers
- ❌ GPUs sit idle (GPU 1 waits for GPU 0, GPU 2 waits for GPU 1...)
- ❌ Poor GPU utilization (~25% with 4 GPUs if naive)

**Solution:** Micro-batching - split batch into smaller chunks, pipeline multiple micro-batches.

---

#### 2. **Tensor Parallelism** (Horizontal Split)

Split **individual operations** (like matrix multiplication) across GPUs within a layer.

```
Single Attention Layer:

Input (X)
   ↓
GPU 0: Q₁, K₁, V₁ (1st half of attention heads)
GPU 1: Q₂, K₂, V₂ (2nd half of attention heads)
   ↓
Compute attention in parallel
   ↓
Combine results → Output

All GPUs work on the SAME layer simultaneously!
```

**Example:** 16 attention heads split across 2 GPUs:
- GPU 0: Computes heads 1-8
- GPU 1: Computes heads 9-16
- Both GPUs communicate to combine results

**Trade-off:**
- ✅ All GPUs active simultaneously (better utilization)
- ✅ Works within each layer (fine-grained parallelism)
- ❌ High communication overhead (GPUs must sync frequently)
- ❌ Only works for operations that can be split (matrix multiplication, attention)

---

#### 3. **3D Parallelism** (Combined Approach)

Combines **TP + PP + DP** — each solves a different problem:

```
TP = layer too wide for 1 GPU     → split one layer across GPUs (parallel within layer)
PP = too many layers for 1 GPU    → split layers into sequential stage groups
DP = need more throughput         → replicate full TP+PP setup on different batches
```

**Example: 60-layer model, TP=2, PP=4, DP=2 → 16 GPUs total**

```
              ← PP: 4 pipeline stages (sequential) →
           Stage 0    Stage 1    Stage 2    Stage 3
          (L 1-15)  (L 16-30)  (L 31-45)  (L 46-60)
Replica 1  GPU 0,1 → GPU 2,3 → GPU 4,5 → GPU 6,7    ← batch A
Replica 2  GPU 8,9 → GPU10,11→ GPU12,13→ GPU14,15   ← batch B
              ↑TP↑
         2 GPUs share each layer's tensor operations

Total: 2(TP) × 4(PP) × 2(DP) = 16 GPUs
Each GPU stores: 1/2 layer × 1/4 layers = 1/8 of model
```

**How mini-batch flows through PP (important!):**

PP stages are **sequential, not independent** — every batch passes through ALL stages:
```
Batch A:  Stage 0 → Stage 1 → Stage 2 → Stage 3 → Loss
          (all 60 layers see batch A, just on different GPU groups)
```

Problem: GPUs idle while waiting for previous stage → **pipeline bubble**:
```
Stage 0: [batch A]  idle     idle     idle
Stage 1:  wait    [batch A]  idle     idle      ← 75% idle time!
Stage 2:  wait     wait    [batch A]  idle
Stage 3:  wait     wait     wait    [batch A]
```

Fix — **micro-batching** (split mini-batch into 4 micro-batches):
```
Stage 0: [m1][m2][m3][m4] idle  idle  idle  idle
Stage 1:  -  [m1][m2][m3][m4]  idle  idle  idle   ← stages overlap!
Stage 2:  -   -  [m1][m2][m3] [m4]  idle  idle
Stage 3:  -   -   -  [m1][m2] [m3] [m4]  idle
```

**DP vs PP — key distinction:**
```
PP: same batch flows through all stages sequentially (assembly line)
DP: different batches run on separate full replicas independently
    → after both finish, gradients averaged → weights synced
```

**TP=2 vs TP=4 (same PP=4):**
```
TP=2 → 2 GPUs split each layer (holds 1/2 layer each) →  8 GPUs
TP=4 → 4 GPUs split each layer (holds 1/4 layer each) → 16 GPUs
       use TP=4 when one layer is too big even for 2 GPUs
```

**Communication per dimension:**
```
TP:  all-reduce within stage GPUs        ← fast  (NVLink, same node)
PP:  pass activations to next stage      ← medium (point-to-point)
DP:  all-reduce gradients across replicas← slow  (InfiniBand, across nodes)
```

- ✅ Scales to 1000+ GPUs, trains 100B+ models
- ❌ Complex to tune — wrong TP/PP/DP ratio wastes GPU idle time

---

### Quick Comparison

| Technique | What's Split? | GPU Utilization | Communication | Use Case |
|-----------|---------------|-----------------|---------------|----------|
| **Pipeline (PP)** | Layers (vertical) | Low (sequential) | Low | Simple large models |
| **Tensor (TP)** | Operations (horizontal) | High (parallel) | High | Within-layer splitting |
| **3D (TP+PP+DP)** | Everything! | Very High | Optimized | Massive models (100B+) |

---

### Real-World Examples

**GPT-3 (175B parameters):**
- Pipeline Parallelism: 4 stages (96 layers ÷ 4 = 24 layers per stage)
- Tensor Parallelism: 8-way (split attention heads)
- Data Parallelism: Multiple replicas
- Total: ~300+ GPUs

**LLaMA-2 70B:**
- Tensor Parallelism: 2-4 way
- Pipeline Parallelism: 2-4 stages  
- Data Parallelism: 8-16 replicas
- Total: 16-64 A100 GPUs

---

<img src="images/model_parallelism_overview.png" alt="Model Parallelism Overview" width="700">


---

## Deep Dive: Parallelism Strategies in LLM Training

When training large language models, we need to distribute the workload across multiple GPUs. There are three main strategies, each solving different problems. Let's understand them in detail.

---

## Data Parallelism

### What is Data Parallelism?

**Core Idea:** Keep a complete copy of the model on each GPU, but give each GPU different batches of training data.

Think of it like having multiple students (GPUs) studying from the same textbook (model), but each solving different practice problems (data batches).

### How It Works (Step-by-Step)

```
Step 1: Replicate Model
GPU 0: [Full Model Copy]
GPU 1: [Full Model Copy]
GPU 2: [Full Model Copy]
GPU 3: [Full Model Copy]

Step 2: Split Data Batch
Batch of 32 samples split into:
- GPU 0: Samples 1-8
- GPU 1: Samples 9-16
- GPU 2: Samples 17-24
- GPU 3: Samples 25-32

Step 3: Forward Pass (Parallel)
Each GPU computes loss independently on its mini-batch

Step 4: Backward Pass (Parallel)
Each GPU computes gradients independently

Step 5: Gradient Synchronization
All GPUs share and average their gradients
GPU 0: gradient_A  ┐
GPU 1: gradient_B  ├─→ Average → Final Gradient
GPU 2: gradient_C  │
GPU 3: gradient_D  ┘

Step 6: Update Model
All GPUs update their model copy with the averaged gradient
(Now all GPUs have identical models again)

Step 7: Repeat
Go back to Step 2 with next batch
```

### Visual Example

```
Training Loop Iteration:

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │
│ Model Copy  │  │ Model Copy  │  │ Model Copy  │  │ Model Copy  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       ↓                ↓                ↓                ↓
   Batch 1-8        Batch 9-16      Batch 17-24      Batch 25-32
       ↓                ↓                ↓                ↓
  Forward Pass     Forward Pass     Forward Pass     Forward Pass
       ↓                ↓                ↓                ↓
  Backward Pass    Backward Pass    Backward Pass    Backward Pass
       ↓                ↓                ↓                ↓
   Gradient A       Gradient B       Gradient C       Gradient D
       └────────────────┴────────────────┴────────────────┘
                              ↓
                    All-Reduce (Average)
                              ↓
                      Final Gradient
                              ↓
       ┌────────────────┬────────────────┬────────────────┐
       ↓                ↓                ↓                ↓
  Update Model     Update Model     Update Model     Update Model
```

### Types of Data Parallelism

#### 1. **Standard Data Parallelism (DDP - Distributed Data Parallel)**

**How it works:**
- Each GPU stores complete model, optimizer state, and gradients
- Uses `AllReduce` to synchronize gradients after backward pass

**Memory Breakdown (per GPU):**
```
For a 7B parameter model with Adam optimizer:

Parameters:           7B × 4 bytes = 28 GB
Gradients:            7B × 4 bytes = 28 GB
Optimizer State:      7B × 8 bytes = 56 GB (Adam: momentum + variance)
Activations (batch):  ~10 GB
────────────────────────────────────────
Total per GPU:        ~122 GB

Problem: Each GPU needs 122 GB!
```

**Pros:**
- ✅ Simple to implement
- ✅ Perfect linear scaling with more GPUs (4 GPUs = 4× faster)
- ✅ No model sharding complexity
- ✅ Low communication overhead (only gradients)

**Cons:**
- ❌ Memory inefficient (duplicates everything)
- ❌ Can't train models larger than single GPU memory
- ❌ Wastes memory on large models

**Best For:**
- Models that fit on single GPU (< 40GB for A100)
- You want maximum training speed
- Simple deployment

#### 2. **Sharded Data Parallelism (ZeRO - Zero Redundancy Optimizer)**

Already covered in detail in the ZeRO section above! This optimizes Data Parallelism by sharding optimizer states, gradients, and parameters across GPUs.

**Quick Recap:**
- **ZeRO-1**: Shard optimizer states → Save 4× memory
- **ZeRO-2**: Shard optimizer + gradients → Save 8× memory
- **ZeRO-3**: Shard optimizer + gradients + parameters → Save 16× memory

### Gradient Synchronization Methods

#### **AllReduce Communication Pattern**

```
Naive Approach (Ring AllReduce):

GPU 0: [grad_A] ─┐
GPU 1: [grad_B] ─┤
GPU 2: [grad_C] ─┼→ Each GPU broadcasts to all others
GPU 3: [grad_D] ─┘   (N² communication)

Optimized Ring AllReduce:

Step 1: GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0 (ring)
Step 2: Each GPU sends portion of gradient
Step 3: After N-1 steps, all GPUs have averaged gradient

Communication Cost: O(N) instead of O(N²)
```

### Real-World Example

**Training LLaMA-2 7B with Data Parallelism:**

**Setup:** 8× A100 GPUs (80GB each)

```
Configuration:
- Model size: 7B parameters
- Batch size: 256 total (32 per GPU)
- Sequence length: 4096 tokens

Memory per GPU:
- Model:          28 GB
- Gradients:      28 GB
- Optimizer:      56 GB
- Activations:    12 GB
- Total:          124 GB ❌ Exceeds 80GB!

Solution: Use ZeRO-2
- Model:          28 GB
- Gradients:      3.5 GB (sharded)
- Optimizer:      7 GB (sharded)
- Activations:    12 GB
- Total:          50.5 GB ✓ Fits!

Training Speed:
- Single GPU:     8 hours/epoch
- 8 GPUs (DDP):   1 hour/epoch (8× speedup)
- Communication:  ~5% overhead
- Effective:      7.6× speedup
```

### When to Use Data Parallelism

✅ **Use Data Parallelism when:**
- Model fits in GPU memory (with optimizer)
- You have multiple GPUs available
- You want simple, fast training
- Your dataset is large

❌ **Don't use when:**
- Model is too large for single GPU → Use Model Parallelism
- Very large batch sizes cause convergence issues
- Communication bandwidth is limited

---

## Tensor Parallelism (Intra-Layer Model Parallelism)

**Core idea:** Split one layer's tensor operations across multiple GPUs — all GPUs work on the **same layer simultaneously**.

### How It Works

Each layer has weight matrices (W_q, W_k, W_v, W_ffn). TP splits these **horizontally** across GPUs:

```
Attention layer (16 heads, TP=2):

Input X (same on both GPUs)
         ↓              ↓
GPU 0: heads 1-8     GPU 1: heads 9-16
  Q₀K₀V₀ → Attn₀      Q₁K₁V₁ → Attn₁
         │              │
         └──── AllGather/Concat ────┘
                    ↓
             Full Attention output
                    ↓
         ┌──────────┴──────────┐
GPU 0: FFN left half      GPU 1: FFN right half
         │                     │
         └────── AllReduce ─────┘
                    ↓
               Layer output
```

**Per layer: 4 communication ops** (2× AllGather + 2× AllReduce) — that's why TP needs fast NVLink.

### Megatron-LM: Column vs Row Parallel

Two ways to split a weight matrix `Y = X × W`:

```
Column Parallel (split W by columns → concat outputs):
  GPU 0: Y₀ = X × W[:, :half]   GPU 1: Y₁ = X × W[:, half:]
  Result: Y = concat(Y₀, Y₁)

Row Parallel (split X and W by rows → sum outputs):
  GPU 0: Y₀ = X[:half] × W[:half, :]   GPU 1: Y₁ = X[half:] × W[half:, :]
  Result: Y = Y₀ + Y₁  (AllReduce)
```

Used in sequence: Column Parallel (Q,K,V proj) → Row Parallel (output proj) = only 1 AllReduce needed.

### Memory Savings (GPT-3 layer, TP=8)

```
Single GPU per layer:   ~7.1 GB
With TP=8 per GPU:      7.1 ÷ 8 = ~900 MB/layer
For 96 layers:          ~86 GB total → ~11 GB/GPU ✓
```

### Real-World: LLaMA-65B with TP=8

```
Without TP:  130 GB weights → doesn't fit on A100 80GB ❌
With TP=8:   130 ÷ 8 = 16.25 GB weights/GPU
             + 16.25 GB gradients + 32.5 GB optimizer + 10 GB activations
             = ~75 GB/GPU ✓ (fits in 80 GB)
             Communication overhead: ~20% → effective speedup: 6.4× (not 8×)
```

| | TP |
|---|---|
| **Splits** | Weight matrices within each layer |
| **All GPUs active?** | Yes — same layer, simultaneously |
| **Communication** | 4 ops/layer — needs NVLink (fast) |
| **Best range** | 2–8 GPUs (beyond 8, communication dominates) |
| **Use when** | Single layer too wide for 1 GPU (hidden_dim > 8192) |

---

## Pipeline Parallelism (Inter-Layer Model Parallelism)

**Core idea:** Split layers into sequential stage groups across GPUs — data flows GPU→GPU→GPU like an assembly line.

### The Bubble Problem

With 1 mini-batch and 4 GPU stages (8 layers each):

```
Time  │ GPU 0    │ GPU 1    │ GPU 2    │ GPU 3
──────┼──────────┼──────────┼──────────┼──────────
  1   │ Fwd      │ idle     │ idle     │ idle
  2   │ idle     │ Fwd      │ idle     │ idle
  3   │ idle     │ idle     │ Fwd      │ idle
  4   │ idle     │ idle     │ idle     │ Fwd→Loss
  5   │ idle     │ idle     │ idle     │ Bwd
  6   │ idle     │ idle     │ Bwd      │ idle
  7   │ idle     │ Bwd      │ idle     │ idle
  8   │ Bwd      │ idle     │ idle     │ idle

GPU utilization = 25% ❌  ← pipeline bubble
```

### Fix: Micro-batching (GPipe)

Split mini-batch into 4 micro-batches → stages overlap:

```
Time  │ GPU 0 │ GPU 1 │ GPU 2 │ GPU 3
──────┼───────┼───────┼───────┼───────
  1   │ F1    │       │       │
  2   │ F2    │ F1    │       │
  3   │ F3    │ F2    │ F1    │
  4   │ F4    │ F3    │ F2    │ F1
  5   │ B4    │ F4    │ F3    │ F2
  6   │ B3    │ B4    │ F4    │ F3
  7   │ B2    │ B3    │ B4    │ F4
  8   │ B1    │ B2    │ B3    │ B4

GPU utilization = 75% ✓
```

**Bubble formula:**
```
Bubble % = (p-1) / (m+p-1)    p=stages, m=micro-batches

4 GPUs, 4 micro-batches:  3/7  = 43% ❌
4 GPUs, 8 micro-batches:  3/11 = 27% ⚠️
4 GPUs, 16 micro-batches: 3/19 = 16% ✓

Rule: m ≥ 4p for bubble < 20%
```

### 1F1B Schedule — Less Memory

GPipe stores ALL micro-batch activations simultaneously → high memory.
1F1B interleaves forward + backward → stores only 1 micro-batch at a time:

```
Memory comparison (GPT-3, 4 GPUs, 8 micro-batches):
  GPipe:  87.5 GB model + 40 GB activations = 127.5 GB ❌
  1F1B:   87.5 GB model +  5 GB activations =  92.5 GB ✓
  Saves: 35 GB/GPU
```

### Communication Pattern

```
Forward:  GPU0 → GPU1 → GPU2 → GPU3  (pass activations)
Backward: GPU3 → GPU2 → GPU1 → GPU0  (pass gradients)

Point-to-point only — no all-reduce needed → works on slower interconnects (PCIe)
```

### Real-World: GPT-3 175B with PP=16

```
96 layers ÷ 16 stages = 6 layers/GPU
64 micro-batches, 64 GPUs

Bubble: (16-1)/(64+16-1) = 15/79 = 19% → 81% utilization ✓

Memory/GPU: 22 GB model + 22 GB gradients + 44 GB optimizer + 8 GB activations = ~96 GB
→ Add ZeRO-1 to shard optimizer: 22 + 22 + 5.5 + 8 = 57.5 GB ✓
```

| | PP |
|---|---|
| **Splits** | Layers into sequential stage groups |
| **All GPUs active?** | No — pipeline bubble (idle time) |
| **Communication** | Point-to-point — works on PCIe |
| **Best range** | 16+ GPUs |
| **Use when** | Model too deep, need to scale beyond 8 GPUs |

---

## Comparison: Data vs Tensor vs Pipeline Parallelism

### Quick Reference Table

| Aspect | Data Parallelism | Tensor Parallelism | Pipeline Parallelism |
|--------|------------------|--------------------|--------------------|
| **What's Split** | Training data | Operations within layers | Layers across GPUs |
| **Model Copies** | Full copy per GPU | Partial (sharded horizontally) | Partial (sharded vertically) |
| **Communication** | After each backward | Multiple times per layer | Between adjacent stages |
| **Bandwidth Need** | Medium (gradients) | Very High (activations) | Low (point-to-point) |
| **GPU Utilization** | ~100% | ~100% | 75-85% (due to bubbles) |
| **Scaling Limit** | Data size | 2-8 GPUs | 16+ GPUs |
| **Best For** | Model fits on GPU | Wide layers | Deep models |
| **Complexity** | Low | Medium | Medium-High |
| **Memory Savings** | None (duplicates) | P× (P=GPUs) | P× (P=stages) |

### Visual Comparison

```
Data Parallelism (4 GPUs):
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Layer 1 │  │ Layer 1 │  │ Layer 1 │  │ Layer 1 │
│ Layer 2 │  │ Layer 2 │  │ Layer 2 │  │ Layer 2 │
│ Layer 3 │  │ Layer 3 │  │ Layer 3 │  │ Layer 3 │
│ Layer 4 │  │ Layer 4 │  │ Layer 4 │  │ Layer 4 │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
   GPU 0        GPU 1        GPU 2        GPU 3
Data: A       Data: B      Data: C      Data: D

Tensor Parallelism (4 GPUs):
┌─────────────────────────────────────────────┐
│         Layer 1 (split 4 ways)              │
│  ┌────┐    ┌────┐    ┌────┐    ┌────┐     │
│  │1/4 │    │2/4 │    │3/4 │    │4/4 │     │
│  └────┘    └────┘    └────┘    └────┘     │
│  GPU 0     GPU 1     GPU 2     GPU 3      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         Layer 2 (split 4 ways)              │
│  ┌────┐    ┌────┐    ┌────┐    ┌────┐     │
│  │1/4 │    │2/4 │    │3/4 │    │4/4 │     │
│  └────┘    └────┘    └────┘    └────┘     │
│  GPU 0     GPU 1     GPU 2     GPU 3      │
└─────────────────────────────────────────────┘
(Same data, split computation)

Pipeline Parallelism (4 GPUs):
┌─────────┐
│ Layer 1 │  GPU 0
│ Layer 2 │
└─────────┘
     ↓
┌─────────┐
│ Layer 3 │  GPU 1
│ Layer 4 │
└─────────┘
     ↓
┌─────────┐
│ Layer 5 │  GPU 2
│ Layer 6 │
└─────────┘
     ↓
┌─────────┐
│ Layer 7 │  GPU 3
│ Layer 8 │
└─────────┘
(Data flows through pipeline)
```

### Which Parallelism to Choose?

#### Decision Tree

```
START
  │
  ├─ Does model fit on 1 GPU? ────Yes──→ Use Data Parallelism
  │                                       (simplest, fastest)
  No
  │
  ├─ Are layers very wide? ────Yes──→ Use Tensor Parallelism
  │  (hidden_dim > 8192)               (2-8 GPUs, needs NVLink)
  │
  No (model is very deep)
  │
  └─→ Use Pipeline Parallelism
      (scales to many GPUs)

For models > 100B parameters:
  └─→ Use 3D Parallelism (all three combined!)
```

#### Practical Examples

**LLaMA-7B:**
```
Model: 7B parameters, 32 layers, 4096 hidden
Recommendation: Data Parallelism + ZeRO-2
- Fits on single A100 with optimizer
- Use 8× DDP for 8× speedup
- Communication: ~5% overhead
```

**LLaMA-70B:**
```
Model: 70B parameters, 80 layers, 8192 hidden
Recommendation: Tensor Parallelism (2-4 way) + Data Parallelism
- TP=2: Split wide layers
- DP=8: Replicate TP pairs
- Total: 16 GPUs
- Requires NVLink for TP
```

**GPT-3 175B:**
```
Model: 175B parameters, 96 layers, 12,288 hidden
Recommendation: 3D Parallelism
- TP=8: Split each layer 8 ways
- PP=4: Pipeline 4 stages (24 layers each)
- DP=4: 4 data replicas
- Total: 8 × 4 × 4 = 128 GPUs
- Mixed communication (NVLink for TP, IB for PP/DP)
```

---

## Data Parallelism vs Flash Attention: Key Differences

**Important:** These are solving **completely different problems** - don't confuse them!

### Quick Summary

| Aspect | Data Parallelism | Flash Attention |
|--------|------------------|-----------------|
| **Problem Solved** | How to train faster with multiple GPUs | How to compute attention more efficiently |
| **What It Optimizes** | Training throughput (speed) | Memory usage and computation speed |
| **Scope** | Entire training process across GPUs | Single operation (attention layer) |
| **Hardware Requirement** | Multiple GPUs | Single GPU (but helps all GPUs) |
| **When Applied** | Training strategy (architectural level) | Algorithm optimization (layer level) |
| **Memory Impact** | No reduction (duplicates model) | 3-4× memory reduction |

---

### Key Differences Explained

#### 1. **Problem Domain**

**Data Parallelism:**
```
Question: "How do I train faster?"
Answer: "Use more GPUs to process more data"
Level: Training infrastructure
```

**Flash Attention:**
```
Question: "How do I make attention computation efficient?"
Answer: "Use smarter algorithm with less memory"
Level: Single operation optimization
```

#### 2. **Memory Behavior**

**Data Parallelism:**
```
1 GPU:  14 GB (model) + 16 GB (attention) = 30 GB
4 GPUs: 30 GB × 4 = 120 GB total

Increases total memory usage (more copies)!
```

**Flash Attention:**
```
1 GPU:  14 GB (model) + 4 GB (attention) = 18 GB
4 GPUs: 18 GB × 4 = 72 GB total

Reduces memory per GPU (smarter computation)!
```

#### 3. **When to Use**

**Data Parallelism:**
- ✅ You have multiple GPUs
- ✅ Want to speed up training
- ✅ Model fits on single GPU
- ❌ Don't use if model doesn't fit (use Model Parallelism)

**Flash Attention:**
- ✅ ALWAYS use it (free performance!)
- ✅ Want longer sequences
- ✅ Want lower memory usage
- ✅ Works with or without multiple GPUs
- ❌ No reason NOT to use it

#### 4. **Combination Effect**

They **complement each other perfectly!**

```
Data Parallelism: More GPUs → More throughput
Flash Attention: Less memory → Bigger batches/sequences

Together: More GPUs + Efficient memory = Maximum training speed!
```

---

### Real-World Impact

**Training GPT-3 (175B parameters):**

**Without Flash Attention:**
- Needs 350+ GPUs minimum
- Sequence length: 1024
- Training time: ~1 month on 1000 GPUs
- Cost: ~$12M

**With Flash Attention:**
- Can use 200-250 GPUs
- Sequence length: 2048 (2× longer)
- Training time: ~2 weeks on 1000 GPUs
- Cost: ~$6M
- Savings: $6M + Better model quality! ✓

---

### Summary

| Feature | Data Parallelism | Flash Attention |
|---------|------------------|-----------------|
| **What** | Training strategy | Algorithm optimization |
| **Why** | Speed up training | Save memory, speed up attention |
| **Where** | Across multiple GPUs | Within each GPU |
| **When** | Have multiple GPUs | Always! |
| **How** | Replicate model, split data | Tile-based fused computation |
| **Benefit** | N× faster (N GPUs) | 2-4× memory savings, 2-3× faster |
| **Use together?** | ✅ YES! Best results | ✅ YES! Best results |

**Bottom Line:** 
- **Data Parallelism** = "Let's use more workers (GPUs)"
- **Flash Attention** = "Let's make each worker more efficient"
- **Together** = Fastest possible training! 🚀

---
# Flash Attention

## What is Flash Attention?

**Flash Attention** is an optimized algorithm that makes attention computation **faster** and **memory-efficient**.

### The Problem It Solves

Standard attention has a major bottleneck:
- **Memory hungry**: Requires O(N²) memory for attention matrix
- **Too many data transfers**: Constantly moving data between slow and fast memory
- **Inefficient**: Wastes time reading/writing intermediate results

### Key Components

**GPU Memory Hierarchy:**
- **SRAM (fast cache)**: 19 TB/s speed, but only 20 MB size
- **HBM (main memory)**: 1.5 TB/s speed (~12x slower), but 40-80 GB size
- **Bottleneck**: Moving data between SRAM ↔ HBM is slow!

<img src="images/flash_attention_overview.png" alt="Flash Attention Overview" width="700">

---

## Traditional Self-Attention Computation

**The Problem: Too Many Slow Memory Operations**

Standard attention performs these steps:

1. **LOAD** Q, K from HBM → SRAM
2. **COMPUTE** S = Q × K^T (in SRAM ✅)
3. **WRITE** S to HBM ⬅️ Slow! (stores intermediate result)
4. **READ** S from HBM → SRAM ⬅️ Slow!
5. **COMPUTE** P = softmax(S) (in SRAM ✅)
6. **WRITE** P to HBM ⬅️ Slow! (stores intermediate result)
7. **READ** P from HBM → SRAM ⬅️ Slow!
8. **LOAD** V from HBM → SRAM
9. **COMPUTE** O = P × V (in SRAM ✅)
10. **WRITE** O to HBM

**The Inefficiency:** Computation happens in fast SRAM, BUT intermediate results (S, P) are constantly written to and read from slow HBM. This back-and-forth data movement is the bottleneck!

**Result: ~14 MB of HBM traffic per attention operation × every layer = Very slow!**

### Memory Traffic Example

For sequence length N=1024, hidden dim d=64:
- S matrix size: 1024 × 1024 = 1M elements = **2 MB** (FP16)
- P matrix size: **2 MB**
- Total HBM traffic: Load Q,K (4MB) + Write S (2MB) + Read S (2MB) + Write P (2MB) + Read P (2MB) + Load V (2MB) = **14 MB**

<img src="images/flash_attention_memory_hierarchy.png" alt="Flash Attention Memory Hierarchy" width="650">

<img src="images/flash_attention_standard_flow.png" alt="Standard Attention Flow" width="700">

---

## Flash Attention Solution

### Core Innovation

Instead of computing the full attention matrix:
- **Divide Q, K, V into small blocks** (tiles that fit in SRAM)
- **Process blocks one at a time** in fast SRAM
- **Never materialize** the full N×N attention matrix in slow HBM
- **Reduce memory**: O(N²) → O(N)

### How It Works

**Key Technique: Block-wise Processing + Online Softmax**

1. **Outer loop**: Iterate over Q blocks (size B_r)
2. **Inner loop**: For each Q block, iterate over K,V blocks (size B_c)
3. **Compute partial attention** entirely in SRAM
4. **Incrementally update** output using running statistics
5. **No intermediate writes** to HBM!

**Benefits:**
- All computation in fast SRAM
- Only load Q,K,V once and write output once
- Dramatically reduces HBM traffic

<img src="images/flash_attention_algorithm.png" alt="Flash Attention Algorithm" width="700">

<img src="images/flash_attention_detailed_algorithm.png" alt="Flash Attention Detailed Algorithm" width="700">

### Block-wise Processing Visual

**How blocks are processed:**
- Load Q block (e.g., rows 0-63) into SRAM
- Iterate through K,V blocks (cols 0-63, 64-127, etc.)
- Compute partial attention **in SRAM** (no HBM writes!)
- Accumulate results block-by-block
- Block size: Typically 64-128 tokens

<img src="images/flash_attention_blockwise.png" alt="Flash Attention Blockwise Computation" width="700">

**Key**: `Q, K, V blocks (green) stay in SRAM during computation. Only final output O written to HBM.`

---

## Comparison: Standard vs Flash Attention

### Memory Access Patterns

**Standard Attention:**
- Multiple HBM reads/writes (red arrows)
- Load Q,K → Write S → Read S → Write P → Read P → Load V → Write O
- **O(N²) HBM accesses** for attention matrix

**Flash Attention:**
- Q,K,V loaded **once** in blocks
- All computation in SRAM (green)
- Only final output O written to HBM
- **O(N) HBM accesses** only for input/output

**Result: For 2048 tokens, ~2000x reduction in HBM accesses!**

<img src="images/flash_attention_memory_pattern.png" alt="Flash Attention Memory Pattern" width="700">

<img src="images/flash_attention_comparison_diagram.png" alt="Flash Attention Comparison" width="700">




## Flash Attention Implementation in Code

### 1. Using the Official Flash Attention Library (Most Common)

```python
# Installation
# pip install flash-attn --no-build-isolation

import torch
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# Method 1: Packed QKV format (most efficient)
def flash_attention_packed(qkv, dropout_p=0.0, causal=True):
    """
    qkv: (batch, seqlen, 3, num_heads, head_dim)
    Returns: (batch, seqlen, num_heads, head_dim)
    """
    output = flash_attn_qkvpacked_func(
        qkv, 
        dropout_p=dropout_p,
        causal=causal,  # For autoregressive models
        softmax_scale=None  # defaults to 1/sqrt(head_dim)
    )
    return output

# Method 2: Separate Q, K, V tensors
def flash_attention_separate(q, k, v, dropout_p=0.0, causal=True):
    """
    q: (batch, seqlen_q, num_heads, head_dim)
    k: (batch, seqlen_k, num_heads, head_dim)
    v: (batch, seqlen_v, num_heads, head_dim)
    """
    output = flash_attn_func(
        q, k, v,
        dropout_p=dropout_p,
        causal=causal
    )
    return output
```


### Key Benefits in Training

1. **Memory Efficiency**: O(N) memory instead of O(N²)
2. **Speed**: 2-4x faster for long sequences
3. **Longer Sequences**: Can train with 4K-8K tokens instead of 512-1024
4. **Same Accuracy**: Mathematically equivalent to standard attention

### Important Notes

- Flash Attention requires **CUDA** (doesn't work on CPU)
- Works best with **A100, H100 GPUs** (Ampere/Hopper architecture)
- Use **float16 or bfloat16** precision for best performance
- Automatically handles **causal masking** for autoregressive models


# Quantization

**Quantization is a technique to make AI models smaller and faster by using less precise numbers. Instead of storing model weights as 32-bit floating-point numbers (FP32), we use smaller formats like 16-bit (FP16), 8-bit (INT8), or even 4-bit (INT4). This reduces memory usage and speeds up computation, making it possible to run large models on smaller GPUs or even CPUs.**

<img src="images/quantization_overview.png" alt="Quantization Overview" width="700">

**Comparison of different number formats: FP32 (full precision) uses 32 bits to store each number with high accuracy. FP16 uses 16 bits (half the memory), INT8 uses 8 bits (quarter the memory), and INT4 uses only 4 bits. Lower precision means less memory but slightly less accurate numbers. For example, a 7B parameter model in FP32 takes ~28GB memory, but in INT8 only ~7GB, and in INT4 just ~3.5GB - making it possible to run on consumer hardware!**

<img src="images/quantization_formats.png" alt="Quantization Formats" width="700">

**Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT): PTQ is the simpler approach - you take an already-trained model and convert its weights to lower precision (like converting FP32 → INT8). It's fast and easy but may lose some accuracy. QAT is more advanced - you train the model while simulating quantization effects, so the model learns to work well with lower precision from the start. QAT gives better accuracy but takes longer since you need to retrain.**

<img src="images/quantization_ptq_vs_qat.png" alt="PTQ vs QAT" width="700">

**Practical quantization techniques and their trade-offs: Shows popular methods like GPTQ, GGUF, and bitsandbytes. Each method balances memory savings vs accuracy loss differently. For example, 8-bit quantization typically loses <1% accuracy while cutting memory in half, while 4-bit quantization can reduce memory by 4x but may lose 2-5% accuracy. The key is choosing the right precision for your use case - if you need maximum accuracy, use FP16; if you need to run on limited hardware, INT4 works surprisingly well!**

<img src="images/quantization_techniques.png" alt="Quantization Techniques" width="700">

# Mixed Precision Training

## What is Mixed Precision Training?

**Mixed Precision Training** is a smart technique that uses **different number precisions for different parts of training** to make it faster and use less memory, while still maintaining model accuracy.

Instead of using high-precision numbers (FP32) everywhere, it strategically uses:
- **FP16 (half precision)** for most calculations → Faster & uses less memory
- **FP32 (full precision)** for critical operations → Maintains accuracy

## Purpose

**Speed up training by 2-3x** and **reduce memory usage by ~50%** without sacrificing model quality. This means you can:
- Train larger models on the same hardware
- Train faster (finish in hours instead of days)
- Fit bigger batch sizes in GPU memory

## How It Works (Simple Explanation)

Think of it like cooking:
- **Measuring ingredients** (critical) → Use precise scale (FP32)
- **Mixing and cooking** (most operations) → Rough measurements work fine (FP16)

**In training:**
1. **Store model weights in FP32** (master copy for accuracy)
2. **Do forward/backward passes in FP16** (fast computation)
3. **Update weights in FP32** (precise updates)

### Why This Split?

**Forward/Backward in FP16:**
- Activations & gradients computed in FP16 for each neuron at each layer
- These are **temporary** (discarded after weight update)
- Gradients can be large values (e.g., 2.3, 5.1) → FP16 handles fine
- **Result:** 2-3x speedup, 50% less memory

**Weights in FP32:**
- **After backpropagation**, weight updates become tiny: `update = learning_rate × gradient`
- Example: `0.001 × 2.3 = 0.0023` ← FP16 can't represent this accurately
- FP16 might round 0.0023 → 0.0000 → training fails
- **Solution:** Keep master weights in FP32, apply precise updates there

**Key Point**: Gradients (FP16) ≠ Weight updates (FP32)
- Gradients are computed values during backward pass (can be large)
- Updates are tiny after multiplying by learning rate (need FP32 precision)

## Real Example for Beginners

**Without Mixed Precision:**
```python
# Everything in FP32 (32-bit)
model = Model()  # ~7GB memory for 7B params
optimizer = AdamW(model.parameters())

# Training one batch
loss = model(inputs)  # Forward: slow
loss.backward()       # Backward: slow
optimizer.step()      # Takes ~10 seconds per batch
```

**With Mixed Precision:**
```python
# Using PyTorch's automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

model = Model()  # Same model
optimizer = AdamW(model.parameters())
scaler = GradScaler()  # Handles precision automatically

# Training one batch
with autocast():  # Magic happens here! Uses FP16 automatically
    loss = model(inputs)  # Forward: 2x faster, uses ~3.5GB

scaler.scale(loss).backward()  # Backward in FP16
scaler.step(optimizer)         # Update weights in FP32
scaler.update()                # Takes ~4 seconds per batch (2.5x speedup!)
```

## Concrete Benefits Example

**Training a 7B parameter model for 1 day:**

| Metric | FP32 (Full) | Mixed Precision | Improvement |
|--------|-------------|-----------------|-------------|
| Memory | 28 GB | 14 GB | **50% less** |
| Speed | 100 tokens/sec | 250 tokens/sec | **2.5x faster** |
| Training time | 24 hours | 10 hours | **Save 14 hours!** |
| Cost (cloud GPU) | $240 | $100 | **Save $140** |
| Final accuracy | 68.5% | 68.4% | **Same quality!** |

## Key Insight

**Mixed Precision is like using a faster car for most of your journey, but switching to a precise measurement tool only when you really need accuracy.** You get 90% of the speed benefits with 99.9% of the accuracy!

## When to Use

✅ **Always use it when:**
- Training on modern GPUs (V100, A100, H100)
- Working with large models (>1B parameters)
- Limited GPU memory
- Want faster training

❌ **Avoid when:**
- Using older GPUs without Tensor Cores
- Training tiny models (overhead not worth it)
- Need guaranteed bit-exact reproducibility

**Visualization showing mixed precision training workflow: The diagram illustrates how data flows through the model - inputs are converted to FP16 for fast computation during forward and backward passes (green path), while master weights are maintained in FP32 (blue) for precise gradient updates. Loss scaling prevents gradient underflow when using FP16. This hybrid approach gives you the speed of FP16 with the stability of FP32!**

<img src="images/mixed_precision_workflow.png" alt="Mixed Precision Workflow" width="700">


# Supervised Fine-tuning

<img src="images/sft_overview_1.png" alt="SFT Overview 1" width="700">

<img src="images/sft_overview_2.png" alt="SFT Overview 2" width="700">

<img src="images/sft_overview_3.png" alt="SFT Overview 3" width="700">

<img src="images/sft_overview_4.png" alt="SFT Overview 4" width="700">

## What is Supervised Fine-tuning?

**Supervised Fine-tuning (SFT)** is the process of taking a pre-trained language model and teaching it to perform specific tasks by training it on labeled examples. Think of it as specializing a general-purpose tool for a particular job.

**Simple analogy:** A medical student has general knowledge (pre-training), then does a residency in cardiology (fine-tuning) to become a heart specialist.

## The Process

**Before Fine-tuning (Base Model):**
- Model knows language, grammar, and general knowledge
- Can complete sentences but not great at following instructions
- Example: "Write a poem about dogs" → might just continue the text randomly

**After Fine-tuning (Specialized Model):**
- Model learns to follow specific formats and instructions
- Understands task structure and expected outputs
- Example: "Write a poem about dogs" → generates a proper, well-structured poem

## How It Works

**Step 1: Collect labeled data (Input → Output pairs)**
```
Input: "Translate to French: Hello, how are you?"
Output: "Bonjour, comment allez-vous ?"

Input: "Summarize: [long article]"
Output: "The article discusses..."

Input: "Answer: What is 2+2?"
Output: "4"
```

**Step 2: Train the model on these examples**
- Model learns to map inputs to desired outputs
- Adjusts weights to match the expected behavior
- Typically uses 1,000 to 100,000 examples

**Step 3: Model now performs the task well**
- Can generalize to new, similar inputs
- Follows the learned format and style

## Real-World Example

**Turning GPT (base) into ChatGPT:**

**Base GPT (no fine-tuning):**
```
User: "How do I bake a cake?"
Model: "How do I bake a cake? Well, first you need flour..."
```
*Just continues the text, doesn't answer properly*

**Fine-tuned ChatGPT:**
```
User: "How do I bake a cake?"
Model: "Here's a simple recipe to bake a cake:

1. Preheat oven to 350°F
2. Mix flour, sugar, eggs...
3. Pour into pan...
4. Bake for 30 minutes..."
```
*Actually answers as a helpful assistant!*

## Common Fine-tuning Tasks

| Task | Input | Output |
|------|-------|--------|
| **Translation** | "Translate: Hello" | "Hola" |
| **Summarization** | "Summarize: [article]" | "Brief summary..." |
| **Q&A** | "What causes rain?" | "Rain is caused by..." |
| **Code generation** | "Write Python function to add numbers" | `def add(a, b): return a+b` |
| **Sentiment analysis** | "This movie was great!" | "Positive" |

## Why Fine-tune?

✅ **Specialization**: Makes model expert at specific tasks  
✅ **Control**: Teaches desired output format and style  
✅ **Efficiency**: Much cheaper than training from scratch (hours vs months)  
✅ **Quality**: Better performance on target task than general model

## Simple Code Example

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Your fine-tuning dataset
train_data = [
    {"input": "Translate to Spanish: Hello", "output": "Hola"},
    {"input": "Translate to Spanish: Goodbye", "output": "Adiós"},
    # ... more examples
]

# Fine-tune (simplified)
training_args = TrainingArguments(
    output_dir="./finetuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()  # Fine-tuning happens here!
```

## Key Takeaway

**Pre-training teaches the model language. Fine-tuning teaches it to be useful.** It's like the difference between knowing English and knowing how to write professional emails in English.


---

## Instruction Tuning

**Instruction Tuning** is a specific type of supervised fine-tuning that teaches models to follow human instructions. It's the key technique that transforms a base language model into a helpful assistant.

<img src="images/instruction_tuning_overview.png" alt="Instruction Tuning Overview" width="700">

<img src="images/instruction_tuning_examples_1.png" alt="Instruction Tuning Examples 1" width="700">

<img src="images/instruction_tuning_examples_2.png" alt="Instruction Tuning Examples 2" width="700">

<img src="images/instruction_tuning_examples_3.png" alt="Instruction Tuning Examples 3" width="700">

<img src="images/instruction_tuning_examples_4.png" alt="Instruction Tuning Examples 4" width="700">

### What Makes It Special?

Instead of training on random text, the model learns from:
```
Instruction: "Summarize this article in 3 sentences"
Input: [article text]
Output: "The article discusses... Key points are... In conclusion..."
```

### Key Difference from Regular Fine-tuning

| Regular Fine-tuning | Instruction Tuning |
|---------------------|-------------------|
| Learns one specific task | Learns to follow ANY instruction |
| "Translate this" → Translation | "Translate/Summarize/Explain this" → Does it all |

### Simple Example

**Before Instruction Tuning:**
```
Input: "Explain gravity"
Model: "Explain gravity to a child. Gravity is the force..."
```
*Just continues text, doesn't explain*

**After Instruction Tuning:**
```
Input: "Explain gravity"
Model: "Gravity is a force that pulls objects toward each other. 
       The Earth's gravity keeps us on the ground..."
```
*Actually follows the instruction!*

### Why It Matters

This is what made **ChatGPT** possible - teaching models to understand and execute diverse instructions like "write a poem," "debug this code," or "explain like I'm 5."

**Key Takeaway:** Instruction Tuning teaches models to be obedient assistants, not just text predictors.


### Challenges

<img src="images/instruction_tuning_challenges.png" alt="Instruction Tuning Challenges" width="700">

---

## SFT vs CFT: What's the Difference?

### Quick Summary

**SFT (Supervised Fine-Tuning)** = Teaching the model **HOW to behave**  
**CFT (Continued Fine-Tuning)** = Teaching the model **WHAT to know**

### Detailed Comparison

| Aspect | **SFT** (Supervised Fine-Tuning) | **CFT** (Continued Fine-Tuning) |
|--------|----------------------------------|----------------------------------|
| **Purpose** | Teach how to respond | Add domain knowledge |
| **Data** | Instruction → Response pairs | Raw text documents |
| **Example** | "Translate: Hello" → "Hola" | Medical textbooks (plain text) |
| **Learns** | Task behavior & format | New facts & terminology |
| **Use Case** | Make model conversational | Specialize in a domain |

### Real-World Scenario: Building a Medical Chatbot

**Step 1 - CFT (Add Knowledge):**
```
Feed the model: 
"The heart has four chambers. The left ventricle pumps blood..."
"Diabetes mellitus is characterized by high blood sugar..."
"Common antibiotics include penicillin and amoxicillin..."

Result: Model now knows medical facts
Problem: Doesn't know how to answer questions
```

**Step 2 - SFT (Teach Conversation):**
```
Feed the model:
Q: "What is diabetes?"
A: "Diabetes is a condition where your blood sugar levels are too high..."

Q: "How does the heart work?"
A: "The heart pumps blood through four chambers..."

Result: Model can answer questions like a doctor!
```

### Why Both Matter

**Without CFT (only SFT):**
- Model can chat nicely but gives wrong medical info
- Like a friendly person who doesn't know medicine

**Without SFT (only CFT):**
- Model knows medicine but can't have a conversation
- Like reading a textbook that doesn't answer your questions

**With Both:**
- Model knows medicine AND can explain it conversationally
- Like an actual helpful medical assistant!

### Simple Analogy
- **CFT** = Going to medical school (learning facts)
- **SFT** = Bedside manner training (learning to talk to patients)

### The Pipeline
```
1. Pre-training (learns language from internet)
   ↓
2. CFT - OPTIONAL (learns specialty: medicine, law, code)
   ↓
3. SFT (learns to follow instructions & chat)
   ↓
4. RLHF (learns human preferences)
   ↓
5. Final Model (ChatGPT, Claude, etc.)
```

### Real Examples

**CFT Models:**
- **Code Llama**: Llama 2 + trained on GitHub code → knows programming
- **Med-PaLM**: GPT + trained on medical papers → knows medicine
- **BloombergGPT**: GPT + trained on financial news → knows finance

**SFT Models:**
- **ChatGPT**: GPT-4 + trained on conversations → can chat naturally
- **InstructGPT**: GPT-3 + trained on instructions → follows commands
- **Vicuna**: Llama + trained on user chats → conversational

### When Do You Need Which?

**Use CFT when:**
- Model says "I don't know about [domain topic]"
- Need specialized knowledge (legal, medical, coding)
- Want to update model with new information (2024 events)
- Adapting to a new language

**Use SFT when:**
- Model has knowledge but doesn't respond properly
- Want assistant-like behavior
- Need specific output format
- Making model follow instructions

### Key Takeaway

**CFT fills the brain with knowledge. SFT teaches how to communicate that knowledge.** You usually need both for a useful AI assistant!

---

## What is Alignment in LLMs?

**Alignment** = Making AI models behave in ways that are **helpful, harmless, and honest** - matching what humans actually want.

### The Problem

**Without Alignment:**
```
User: "How do I bake a cake?"
Model: "I'll tell you, but first let me ramble about the history of baking 
       for 10 paragraphs, use offensive language, and give you a recipe 
       that might poison you."
```

**With Alignment:**
```
User: "How do I bake a cake?"
Model: "Here's a simple, safe recipe:
       1. Preheat oven to 350°F
       2. Mix 2 cups flour, 1 cup sugar..."
```

### What Alignment Teaches

1. **Helpful**: Answer the actual question, don't ramble
2. **Harmless**: Don't give dangerous/illegal/offensive advice
3. **Honest**: Say "I don't know" instead of making things up

### How It's Done

**Main Technique: RLHF (Reinforcement Learning from Human Feedback)**

1. Model generates multiple responses
2. Humans rank which response is best
3. Model learns to prefer highly-ranked responses
4. Repeat thousands of times

**Simple Example:**
```
Question: "What's 2+2?"

Response A: "2+2 equals 4."
Response B: "2+2 is a mathematical expression representing..."
Response C: "The answer is 5."

Human ranks: A (best) > B (wordy) > C (wrong)
Model learns: Be like A, not B or C
```

### Why It Matters

**Before Alignment (Base GPT-3):**
- Completes text but doesn't follow instructions
- May generate harmful content
- Often goes off-topic

**After Alignment (ChatGPT):**
- Follows instructions precisely
- Refuses harmful requests politely
- Stays helpful and on-topic

### Key Takeaway

**Alignment is teaching the AI to be a good assistant, not just a smart parrot.** It's the difference between a knowledgeable person and a helpful friend!

---

## Temperature in LLMs

### What is Temperature?

**Temperature** is a setting that controls how **random** or **creative** the model's responses are. It's like adjusting a "creativity dial" from safe and predictable to wild and creative.

**Range:** 0.0 to 2.0 (typically 0.0 to 1.0)

### How It Works

**The model predicts next word with probabilities:**
```
Next word predictions:
"cat" → 60%
"dog" → 30%
"bird" → 10%
```

**Temperature modifies these probabilities:**

**Low Temperature (0.1):** Makes probabilities sharper
```
"cat" → 95%   ← Almost always picks this
"dog" → 4%
"bird" → 1%
```

**High Temperature (1.5):** Makes probabilities flatter
```
"cat" → 40%   ← More variety in choices
"dog" → 35%
"bird" → 25%
```

### The Math: How Probabilities Are Actually Adjusted

**Step-by-step process:**

**Step 1: Model produces raw scores (logits)**
```
"cat"  → logit: 2.0
"dog"  → logit: 1.0
"bird" → logit: 0.5
```
These are the raw outputs from the model's final layer (NOT probabilities yet).

**Step 2: Divide logits by temperature**
```
Temperature = T

Adjusted logits:
"cat"  → 2.0 / T
"dog"  → 1.0 / T
"bird" → 0.5 / T
```

**Step 3: Apply softmax to get probabilities**
$$P(w_i) = \frac{e^{logit_i / T}}{\sum_{j} e^{logit_j / T}}$$

**Concrete Example with Numbers:**

**Original Logits:**
- "cat": 2.0
- "dog": 1.0  
- "bird": 0.5

**Temperature = 1.0 (Default):**

`Adjusted logits(AL) = Output Probability/Temperature(t)`



```
Adjusted logits: 2.0, 1.0, 0.5 (unchanged)
After softmax:
P(cat)  = e^2.0 / (e^2.0 + e^1.0 + e^0.5) = 7.39 / 11.61 = 63.6%
P(dog)  = e^1.0 / (e^2.0 + e^1.0 + e^0.5) = 2.72 / 11.61 = 23.4%
P(bird) = e^0.5 / (e^2.0 + e^1.0 + e^0.5) = 1.65 / 11.61 = 14.2%
```

**Temperature = 0.5 (Low - More Confident):**
```
Adjusted logits: 4.0, 2.0, 1.0 (doubled)
After softmax:
P(cat)  = e^4.0 / (e^4.0 + e^2.0 + e^1.0) = 54.60 / 62.93 = 86.8% ← Much higher!
P(dog)  = e^2.0 / (e^4.0 + e^2.0 + e^1.0) = 7.39 / 62.93 = 11.7%
P(bird) = e^1.0 / (e^4.0 + e^2.0 + e^1.0) = 2.72 / 62.93 = 4.3%
```
**Effect:** Winner "cat" becomes MORE dominant (63.6% → 86.8%)

**Temperature = 2.0 (High - More Random):**
```
Adjusted logits: 1.0, 0.5, 0.25 (halved)
After softmax:
P(cat)  = e^1.0 / (e^1.0 + e^0.5 + e^0.25) = 2.72 / 5.78 = 47.1% ← Lower!
P(dog)  = e^0.5 / (e^1.0 + e^0.5 + e^0.25) = 1.65 / 5.78 = 28.5%
P(bird) = e^0.25 / (e^1.0 + e^0.5 + e^0.25) = 1.28 / 5.78 = 22.1%
```
**Effect:** Probabilities become more even/flat (63.6% → 47.1%)

### Visual Summary

**Original probabilities at T=1.0:**
```
cat  ████████████████████ 63.6%
dog  ████████ 23.4%
bird ████ 14.2%
```

**Low temperature T=0.5 (sharper):**
```
cat  ████████████████████████████ 86.8%  ← Winner dominates
dog  ████ 11.7%
bird ██ 4.3%
```

**High temperature T=2.0 (flatter):**
```
cat  ███████████████ 47.1%  ← More balanced
dog  █████████ 28.5%
bird ███████ 22.1%
```

### Key Insight

**Temperature doesn't change the model's intelligence** - it only adjusts how you sample from its predictions:
- **T < 1.0:** Amplifies differences (confident, deterministic)
- **T = 1.0:** Use model's raw probabilities
- **T > 1.0:** Smooths differences (creative, random)

### Simple Examples

**Question: "The capital of France is ___"**

**Temperature = 0.0 (Deterministic):**
```
Output: "Paris" (always the same, most likely answer)
```

**Temperature = 0.7 (Balanced):**
```
Output: "Paris" (usually), sometimes "Paris, which is..."
```

**Temperature = 1.5 (Creative/Random):**
```
Output: "Paris", or "a beautiful city called Paris", 
        or even random: "Lyon" (less likely but possible)
```

### Real-World Impact

| Temperature | Behavior | Best For | Example |
|------------|----------|----------|---------|
| **0.0 - 0.3** | Focused, consistent, repetitive | Factual Q&A, code, math | "2+2=4" |
| **0.7 - 0.9** | Balanced creativity | Chatbots, general writing | ChatGPT default |
| **1.0 - 2.0** | Very creative, unpredictable | Story writing, brainstorming | Poetry, fiction |

### Important Clarification: Temperature Does NOT Change Self-Attention or FFN

**Common Misconception:** Temperature modifies how the model processes information internally.

**Reality:** Temperature is applied **ONLY at the output stage** after all computations are done.

**The Process:**
```
1. Input: "Write a poem about"
   ↓
2. Self-Attention: [Processes relationships - UNCHANGED by temperature]
   ↓
3. FFN: [Transforms features - UNCHANGED by temperature]
   ↓
4. Output Logits: [cat: 2.5, dog: 1.8, bird: 0.9]
   ↓
5. Apply Temperature: [Adjust probabilities HERE]
   ↓
6. Sample: Pick next word based on adjusted probabilities
```

**Temperature Formula:**
```
Probability = softmax(logits / temperature)
```

**Effect:**
- **Self-Attention weights:** NOT affected (always the same)
- **FFN computations:** NOT affected (always the same)
- **Only the final sampling:** Affected (more/less random)

### Practical Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "Once upon a time"

# Temperature 0.1 - Predictable
output_low = model.generate(
    tokenizer.encode(prompt, return_tensors="pt"),
    max_length=20,
    temperature=0.1,  # <-- Only affects final sampling
    do_sample=True
)
# Result: "Once upon a time, there was a little girl who lived in a small town"

# Temperature 1.5 - Creative
output_high = model.generate(
    tokenizer.encode(prompt, return_tensors="pt"),
    max_length=20,
    temperature=1.5,  # <-- Only affects final sampling
    do_sample=True
)
# Result: "Once upon a time in ancient Mesopotamia, dragons soared through purple skies"
```

### Key Takeaway

**Temperature is a post-processing knob that controls randomness in word selection, NOT how the model thinks.** The model's internal computations (self-attention, FFN) remain exactly the same regardless of temperature. It's like a chef (model) cooking the same way every time, but you (temperature) decide how adventurous you want to be in picking from their menu!

**Low temp = Pick the safest option**  
**High temp = Take more chances**

---


# Benchmarking (Evaluation)

**MMLU - Massive Multi-tasking Language Understanding**

<img src="images/benchmarking_mmlu_1.png" alt="MMLU Benchmark 1" width="700">

<img src="images/benchmarking_mmlu_2.png" alt="MMLU Benchmark 2" width="700">

<img src="images/benchmarking_mmlu_3.png" alt="MMLU Benchmark 3" width="700">

---

## What is Benchmarking?

**Benchmarking** is how we measure and compare how good different AI models are - like giving them standardized tests to see which one performs best.

### Why We Need It

**Problem:** How do you know if GPT-4 is better than Claude or Llama?
**Solution:** Give them all the same tests and compare scores!

### How It Works

Think of it like school exams:
- **Question bank:** Thousands of questions across different subjects
- **All models take the same test:** Fair comparison
- **Score = % correct answers:** Higher score = better model

### What is MMLU?

**MMLU (Massive Multitask Language Understanding)** is one of the most popular LLM benchmarks - it's like the SAT for AI models.

**What it tests:**
- **57 different subjects**: From elementary math to professional law
- **15,908 questions**: Multiple choice format (A, B, C, D)
- **4 difficulty levels**: High school, college, professional, expert

**Subject examples:**
- Mathematics, Physics, Chemistry, Biology
- History, Geography, Law, Medicine
- Computer Science, Economics, Philosophy
- And many more!

### Simple Example

**Question from MMLU (High School Biology):**
```
What is the powerhouse of the cell?
A) Nucleus
B) Ribosome
C) Mitochondria ✓
D) Golgi apparatus

Model answer: C
Score: +1 point (correct!)
```

### Common Benchmarks

| Benchmark | What It Tests | Example Score |
|-----------|---------------|---------------|
| **MMLU** | General knowledge (57 subjects) | GPT-4: 86.4% |
| **HumanEval** | Code generation ability | GPT-4: 67% |
| **GSM8K** | Math word problems | GPT-4: 92% |
| **HellaSwag** | Common sense reasoning | GPT-4: 95.3% |
| **TruthfulQA** | Truthfulness & accuracy | GPT-4: 59% |

### Why Benchmarks Matter

**For Researchers:**
- Track if new model is actually better
- Know where model is weak (e.g., good at math, bad at history)
- Justify spending millions on training

**For Users:**
- Choose the right model for your task
- Know what to expect (e.g., "90% accuracy on coding tasks")

**Example comparison:**
```
Task: Writing Code
GPT-4: 67% on HumanEval → Best for coding
Llama 2: 29% on HumanEval → Not ideal for coding
```

### Important Note

**Benchmarks aren't perfect!** A model can:
- Score high but still make mistakes in real use
- Memorize test questions (like cheating in school)
- Be great at tests but bad at conversation

**Think of benchmarks as one tool, not the whole story.**

### Key Takeaway

**Benchmarks are standardized tests that help us objectively compare AI models across different skills.** MMLU is the most comprehensive, testing 57 subjects from basic to expert level - it's the "gold standard" for measuring general intelligence in LLMs!


# Parameter-Efficient Finetuning(PEFT) with LoRA

**LoRA - Low Ranking Adaptation**

**Overview of **Parameter-Efficient Fine-Tuning (PEFT)** methods: Instead of updating **all model parameters** (expensive), PEFT techniques like **LoRA**, **Adapter Layers**, and **Prefix Tuning** train only a **small subset** of parameters while keeping the original model **frozen**. This dramatically reduces **memory**, **training time**, and **cost** while maintaining comparable performance.**

<img src="images/lora_peft_overview.png" alt="LoRA PEFT Overview" width="700">

**LoRA's core innovation: **Decompose weight updates** into two **low-rank matrices** (A and B). Instead of updating the full weight matrix W [d×d], we train A [d×r] and B [r×d] where **r << d** (rank is much smaller). For example: Full matrix = 4096×4096 = **16M params**, LoRA with r=8 = 4096×8 + 8×4096 = **65K params** (**256x reduction!**). The output is W + A×B, combining frozen original weights with learned updates.**

<img src="images/lora_decomposition.png" alt="LoRA Matrix Decomposition" width="700">

**Comparison: **Full Fine-tuning vs LoRA**: Full fine-tuning updates **every parameter** requiring massive GPU memory and long training. LoRA keeps the **base model frozen** and only trains tiny **adapter matrices**, enabling training on **consumer GPUs** (RTX 3090) instead of expensive data center GPUs (A100). Memory savings: **84GB → 12GB**, training cost: **$1200 → $50**, while maintaining **99% quality**.**

<img src="images/lora_comparison.png" alt="LoRA vs Full Fine-tuning Comparison" width="700">

### Where Are LoRA Weights Updated?

**LoRA adapters are applied to specific weight matrices in the transformer architecture:**

**1. Attention Layers (Primary Target):**
- **Query (Q)** projection: W_q + A_q × B_q
- **Key (K)** projection: W_k + A_k × B_k  
- **Value (V)** projection: W_v + A_v × B_v
- **Output (O)** projection: W_o + A_o × B_o

**2. Feed-Forward Network (FFN) Layers:**
- **Up projection** (first layer): W_up + A_up × B_up
- **Down projection** (second layer): W_down + A_down × B_down

**Visual Example - Single Transformer Layer:**
```
Input
  ↓
┌─────────────────────────────────────┐
│  Self-Attention                     │
│  ┌──────────────────────────────┐   │
│  │ Q: W_q + (A_q × B_q) ← LoRA  │   │
│  │ K: W_k + (A_k × B_k) ← LoRA  │   │
│  │ V: W_v + (A_v × B_v) ← LoRA  │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │ O: W_o + (A_o × B_o) ← LoRA  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│  Feed-Forward Network (FFN)         │
│  ┌──────────────────────────────┐   │
│  │ Up:   W_up + (A × B) ← LoRA  │   │
│  │ Down: W_down + (A × B) ← LoRA│   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
  ↓
Output
```

**Key Points:**
- **Most common:** Apply LoRA only to Q, K, V, O matrices (attention weights)
- **More aggressive:** Also include FFN layers for better adaptation
- **Trade-off:** More LoRA layers = better quality but more parameters to train

**Typical Configuration (r=16):**
```
Attention-only LoRA:
- 4 matrices per layer × 32 layers = 128 adapter matrices
- ~8M trainable params

Attention + FFN LoRA:
- 6 matrices per layer × 32 layers = 192 adapter matrices  
- ~12M trainable params
```

**Multiple LoRA adapters on a **single frozen base model**: Train **different adapters** for **different tasks** (e.g., medical, legal, coding) and **swap adapters** instantly without reloading the base model. One 7B base model can serve **hundreds of specialized tasks** by switching lightweight adapters. This is **875x more efficient** than training separate full models for each task!**

<img src="images/lora_multiple_adapters.png" alt="LoRA Multiple Adapters" width="700">

## LoRA (Low-Rank Adaptation)


**LoRA** is a smart way to fine-tune large models by only training a tiny fraction of the parameters, making it much cheaper and faster.

### The Problem
Fine-tuning a 7B model normally updates **all 7 billion parameters** → Expensive (needs huge GPU memory).

### LoRA's Solution
Instead of updating all weights, LoRA adds small "adapter" matrices that are trained while keeping the original model frozen.

**Result:** Train only **0.1-1% of parameters** (7M instead of 7B!) → Same quality, **10x cheaper**, **3x faster**.

### How LoRA Works - Visual Explanation

**Full Fine-tuning (Traditional):**
```
Input (x)
   ↓
┌─────────────────────────────┐
│   Pre-trained Weight (W)    │  ← ALL parameters trained
│      [4096 × 4096]          │     (16M params updated!)
│      ❌ Updated             │
└─────────────────────────────┘
   ↓
Output
```
**Memory needed:** ~64 GB for 7B model  
**Training:** All 7 billion parameters updated

---

**LoRA Fine-tuning (Efficient):**
```
Input (x)
   ↓
   ├─────────────────────┬──────────────────────┐
   │                     │                      │
   │  Pre-trained W      │    LoRA Adapters     │
   │   [4096 × 4096]     │                      │
   │   ✅ Frozen          │    ┌──────────┐     │
   │   (not trained)     │    │ A Matrix │     │
   │                     │    │ [4096×8] │     │
   │                     │    └─────┬────┘     │
   │                     │          │          │
   │                     │    ┌─────▼────┐     │
   │                     │    │ B Matrix │     │
   │                     │    │  [8×4096]│     │
   │                     │    └─────┬────┘     │
   │                     │          │          │
   └──────────┬──────────┴──────────┘          │
              │         (A × B)                 │
              ↓            ↓                    │
         Original    +   ΔW (update)           │
              └────────────┴────────────────────┘
                          ↓
                       Output
```
**Memory needed:** ~8 GB for 7B model  
**Training:** Only A and B matrices (65K params instead of 16M!)

### The Math Behind LoRA

**Full Fine-tuning:**
```
Output = W × Input
where W is updated: W_new = W_old + ΔW
→ Need to store and update ALL of W (huge!)
```

**LoRA:**
```
Output = (W + A×B) × Input

where:
- W = Original weights [4096 × 4096] → FROZEN ❄️
- A = Small matrix [4096 × r] → TRAINED 🔥
- B = Small matrix [r × 4096] → TRAINED 🔥
- r = rank (typically 8, 16, or 32)

Number of trainable params = 4096×r + r×4096
With r=8: 4096×8 + 8×4096 = 65,536 params
vs original: 4096×4096 = 16,777,216 params
Reduction: 256x smaller! 🎉
```

### Concrete Example

**Scenario:** Fine-tune Llama-2 7B for medical chatbot

**Traditional Fine-tuning:**
```
Parameters to train: 7,000,000,000
Memory needed: 84 GB (won't fit on single GPU!)
Training time: 48 hours on A100
Cost: $1,200
```

**LoRA Fine-tuning (r=16):**
```
Parameters to train: 8,000,000 (0.11% of model!)
Memory needed: 12 GB (fits on RTX 3090!)
Training time: 6 hours on RTX 3090
Cost: $50
Result: 99% of full fine-tuning quality ✅
```

### Visual: Parameter Comparison

**What is Rank (r)?**

Rank (r) controls how many parameters are in the LoRA adapter matrices:

A matrix: shape (d, r) where d is the original dimension
B matrix: shape (r, d)
Total LoRA parameters: d × r + r × d = 2 × d × r

**Example with Llama-2 7B:**
Let's say the weight matrix W has dimensions 4096 × 4096 (common for attention):

LoRA (r=8):
A: 4096 × 8 = 32,768 parameters
B: 8 × 4096 = 32,768 parameters
Total: 65,536 parameters per layer

LoRA (r=16):
A: 4096 × 16 = 65,536 parameters
B: 16 × 4096 = 65,536 parameters
Total: 131,072 parameters per layer

LoRA (r=32):
A: 4096 × 32 = 131,072 parameters
B: 32 × 4096 = 131,072 parameters
Total: 262,144 parameters per layer

Trade-offs:


|Rank|	Parameters|	Quality|	Speed|	Memory|
|--|--|--|--|--|
|r=8|	Fewest|	Good|	Fastest|	Least|
|r=16|	Medium|	Better|	Medium|	Medium|
|r=32|	Most|	Best|	Slower|	Most|

```
Full Fine-tuning:
████████████████████████████████  7B params (100%)

LoRA (r=8):
█                                  8M params (0.11%)

LoRA (r=16):
██                                 16M params (0.22%)

LoRA (r=32):
████                               32M params (0.45%)
```

**The Trade-off:** Higher rank (r) = better quality but more parameters to train.

### Simple Analogy

**Traditional Fine-tuning:**  
Remodeling your entire house - tear down walls, rewire everything.  
💰 Expensive | ⏰ Slow | 🏠 Disruptive

**LoRA:**  
Adding smart furniture and decorations - house stays the same, just add new stuff.  
💵 Cheap | ⚡ Fast | 🪑 Non-disruptive

### Key Benefits Visualization

```
                    Full Fine-tuning  │  LoRA
──────────────────────────────────────┼───────────────
GPU Memory           84 GB            │   12 GB  ✅
Training Speed       48 hours         │   6 hours ✅
Training Cost        $1,200           │   $50    ✅
Quality              100%             │   99%    ✅
Multiple Adapters    No               │   Yes!   ✅
```

**Bonus:** With LoRA, you can train multiple adapters (medical, legal, coding) and swap them on-the-fly without retraining the whole model!

### Key Takeaway

**LoRA decomposes weight updates into two small matrices (A×B), training only them while freezing the original model.** This reduces trainable parameters by 100-1000x, making fine-tuning affordable on consumer GPUs while maintaining 99% of full fine-tuning quality!

---

### LoRA Deep Dive: Common Questions

#### Q1: Are adapter matrices added to pretrained weights or run in parallel?

**Parallel path — not added to weights.**

The pretrained weights stay frozen. LoRA adds a second path that runs alongside:

```
output = (x × W_q)  +  (x × A × B)
          ↑ frozen       ↑ trainable adapter
```

- `W_q` is never modified — it stays untouched during training
- `A × B` is a low-rank update (~0.1% parameters) computed in parallel
- Final output = frozen result + adapter correction

**Example (W_q is 4096×4096 = 16M params):**
```
Without LoRA:  output = x × W_q              (16M frozen)
With LoRA:     output = x × W_q + x × A × B  (16M frozen + 65K trainable)
```
Only A and B are updated during backprop. Weights never change.

---

#### Q2: Why does it matter which target_modules we update?

Each module captures a different type of knowledge. Skipping one = missing that capability upgrade.

| Module | What it learns | Impact if skipped |
|--------|---------------|-------------------|
| `W_q` (Query) | What the model searches for | Can't shift attention focus |
| `W_k` (Key) | What tokens are "findable" | Retrieval stays old-style |
| `W_v` (Value) | What content is returned | Output stays generic |
| `W_o` (Output) | How heads are combined | Multi-head fusion unchanged |
| `FFN` (up/down) | Factual knowledge, patterns | No new facts learned |

**Example — Fine-tuning for medical Q&A:**
- `W_q/W_k`: Model learns to search "symptoms → diagnosis" style
- `W_v`: Returns medical terminology instead of generic text
- `FFN`: Learns new drug names, dosage facts

```python
target_modules=["q_proj", "v_proj"]         # minimal (attention only)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # full attention
target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"]  # + FFN knowledge
```

Rule of thumb: For **style/tone** → attention modules. For **new facts** → include FFN.

---

#### Q3: How are adapter weights saved in the final fine-tuned model?

**By default: saved separately** (small file, ~40MB). Base model stays untouched.

```
fine_tuned_model/
├── adapter_model.safetensors   ← only A & B matrices (~40MB)
├── adapter_config.json         ← rank, target_modules, alpha
```

To use it, load base model + adapter:
```python
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
model = PeftModel.from_pretrained(model, "./fine_tuned_model")
```

**Optional: Merge adapters into base model** (for deployment, removes PEFT dependency):
```python
model = model.merge_and_unload()
model.save_pretrained("./merged_model")  # full model, ~16GB
```

| Mode | Size | Use case |
|------|------|----------|
| Separate adapter | ~40MB | Research, swapping adapters |
| Merged model | ~16GB | Production deployment |

---

#### Q4: What is Alpha (α) and why does it matter?

**Alpha is a volume knob — controls how strongly the adapter influences the output.**

```
output = (x × W_q)  +  (α/r) × (x × A × B)
          ↑ frozen       ↑ scale   ↑ adapter
```

- `r` = rank (adapter size)
- `α` = alpha (scaling constant)
- `α/r` = actual multiplier applied to adapter output

**Why not just use a larger learning rate?**
Without alpha, changing rank `r` accidentally changes adapter strength too (larger matrices → larger outputs). Alpha decouples them — you can tune capacity (rank) and strength (alpha) independently.

```
rank=8,  alpha=16  →  scale = 2.0  (strong)
rank=16, alpha=16  →  scale = 1.0  (neutral)
rank=32, alpha=16  →  scale = 0.5  (conservative)
```

**Common configs:**
```python
LoraConfig(r=16, lora_alpha=16)   # alpha = rank → neutral (safe default)
LoraConfig(r=16, lora_alpha=32)   # alpha = 2×rank → aggressive update
LoraConfig(r=16, lora_alpha=8)    # alpha < rank → conservative
```

| α/r | Effect |
|-----|--------|
| < 1 | Adapter whispers (conservative) |
| = 1 | Neutral — start here |
| > 1 | Adapter shouts (aggressive) |

**Rule of thumb:** Start with `alpha = rank`. Increase if fine-tuning feels weak; decrease if unstable.

---

## QLoRA (Quantized LoRA)

### What is QLoRA?

**QLoRA = LoRA + 4-bit Quantization** - Maximum memory efficiency for training huge models on consumer GPUs!

**Key Difference:**
- **LoRA:** Base model in 16-bit (7B = 14GB memory)
- **QLoRA:** Base model in 4-bit (7B = 3.5GB memory) → **4x less memory!**

Both train the same small adapters, but QLoRA compresses the frozen base model dramatically.

### Memory Comparison

```
7B Model Fine-tuning:
Full: ████████████████████████████████████ 84 GB
LoRA: ██████████████ 14 GB
QLoRA: ███ 3.5 GB ✅

65B Model Fine-tuning:
Full: Too large (520 GB - needs 8× A100s)
LoRA: 130 GB (needs 2× A100s)
QLoRA: 33 GB (fits on 1× RTX 4090!) ✅
```

**QLoRA's innovation: Store base model in **4-bit**, decompress to **16-bit** only during computation, train adapters in **16-bit**. Gets you **4x memory savings** with minimal quality loss:**

<img src="images/qlora_innovation.png" alt="QLoRA Innovation" width="700">

**QLoRA architecture: Base weights compressed to **4-bit** (stored), dequantized on-the-fly to **16-bit** (computed), while LoRA adapters stay **16-bit** throughout:**

<img src="images/qlora_architecture.png" alt="QLoRA Architecture" width="700">

### Real Example: Llama-2 65B Fine-tuning

| Method | GPU Needed | Cost | Memory |
|--------|-----------|------|--------|
| **Full Fine-tuning** | 8× A100 | $1,536 | 520 GB |
| **LoRA** | 2× A100 | $96 | 130 GB |
| **QLoRA** | 1× RTX 4090 | $24 | 33 GB ✅ |

**Memory reduction enables consumer GPU training - QLoRA makes 65B models accessible on single GPU:**

<img src="images/qlora_memory_efficiency.png" alt="QLoRA Memory Efficiency" width="700">

### Trade-offs

| Feature | LoRA | QLoRA |
|---------|------|-------|
| **Memory (7B)** | 14 GB | 3.5 GB ✅ |
| **Speed** | Fast | ~30% slower |
| **Quality** | 99% | 97-98% |
| **Consumer GPUs** | Partial | YES! ✅ |

**QLoRA maintains 97-98% quality with **4-bit quantization** - minimal accuracy loss for massive memory savings:**

<img src="images/qlora_results.png" alt="QLoRA Results" width="700">

### When to Use

**Use LoRA:** Data center GPUs available, need best quality, smaller models (<10B)  
**Use QLoRA:** Consumer GPUs only, huge models (30B-65B+), tight budget

### Key Takeaway

**QLoRA compresses the frozen base model to 4-bit, enabling 65B model fine-tuning on a single consumer GPU (vs 8 data center GPUs). Trade-off: 30% slower, 1-2% quality loss - usually worth it for the massive cost savings!**

---

# Model Pruning — Beginner's Guide

---

## What Is Pruning?

```
Real world analogy:

A tree has many branches.
Some branches are healthy and productive.
Some branches are weak and contribute little.

Pruning = cut the weak branches
          tree stays healthy, grows better
          with less wasted energy

Neural Network Pruning = same idea
  Remove weights/neurons/layers that contribute
  the least to the model's output
```

---

## Core Idea

```
NOT all 70 Billion parameters in LLaMA-3 70B
are equally important.

Some weights:  change output significantly   ← KEEP
Some weights:  barely affect output at all   ← PRUNE (set to 0)

Before pruning:          After pruning:
  W = [0.82,               W = [0.82,
       0.0003,                   0,        ← pruned (tiny)
       0.71,                    0.71,
       0.0001,                   0,        ← pruned (tiny)
       0.65]                    0.65]

Model still works almost as well
but now has fewer active parameters → smaller + faster
```

---

## Types of Pruning

### 1. Weight Pruning (Unstructured)

```
Set individual tiny weights to ZERO

Before:                    After:
[0.82, 0.003, 0.71]   →   [0.82, 0,     0.71]
[0.54, 0.001, 0.33]   →   [0.54, 0,     0.33]
[0.12, 0.91,  0.002]  →   [0.12, 0.91,  0   ]

Sparsity = % of weights set to zero
  50% sparse = half the weights are zero

✅ High accuracy retention
❌ Hard to speed up (zeros still occupy memory)
❌ Needs sparse hardware support
```

### 2. Structured Pruning (More Practical)

```
Remove ENTIRE structures — heads, neurons, layers

Attention Head Pruning:
  LLaMA has 64 attention heads per layer
  Some heads learn redundant patterns
  → Remove entire heads

  Before: 64 heads    After: 48 heads
  Model still attends well, just fewer heads

Neuron Pruning (FFN):
  FFN hidden dim = 16384 neurons
  Some neurons rarely activate
  → Remove entire neurons

  Before: 16384 neurons    After: 12000 neurons

Layer Pruning (Most Aggressive):
  Remove entire transformer layers
  LLaMA-3 70B: 80 layers → 60 layers

✅ Real memory + speed savings
⚠️ More accuracy loss than weight pruning
```

---

## Which Layers Get Pruned in LLMs?

```
Research shows layers are NOT equally important:

Layer Importance in a 80-layer LLM:

Importance
    ▲
    │██                                      ██
    │███                                    ███
    │████                                  ████
    │█████                                █████
    │███████                            ███████
    │██████████                    ████████████
    │████████████████████████████████████████
    └────────────────────────────────────────► Layer
     1  5  10  20  30  40  50  60  70  80

First layers  → CRITICAL (learn basic language structure) → KEEP
Last layers   → CRITICAL (learn output predictions)       → KEEP
Middle layers → Less critical, more redundancy            → PRUNE


Specifically pruned most often:

Component              Why Pruned               Typical Reduction
───────────────────    ──────────────────────   ─────────────────
Middle layers          High redundancy          10–30% of layers
Attention heads        Many are redundant       20–40% of heads
FFN neurons            Many rarely activate     20–50% of neurons
Entire FFN blocks      Some layers barely used  10–20% of layers
```

---

## How Importance Is Measured

```
Method 1: Magnitude-based (simplest)
  Small weight = unimportant → prune it
  importance = |weight value|

Method 2: Gradient-based
  If removing weight barely changes loss → unimportant
  importance = |weight × gradient|

Method 3: Activation-based
  Run sample data through model
  Neurons that rarely activate → prune
  importance = average activation magnitude

Method 4: Taylor Expansion (most accurate)
  Estimate exactly how much loss increases
  if this weight is removed
  importance = |weight × gradient|²
```

---

## Pruning + Fine-tuning (Standard Pipeline)

```
Step 1: Start with pretrained LLM
        LLaMA-3 70B — 80 layers, 70B params

Step 2: Measure importance of every component
        Run calibration data through model
        Score each layer/head/neuron

Step 3: Prune least important components
        Remove bottom 20% by importance score

Step 4: Fine-tune (critical step!)
        Model accuracy drops after pruning
        Fine-tune on small dataset to recover
        LLM relearns to work without pruned parts

Step 5: Evaluate
        Compare pruned vs original on benchmarks
        If quality acceptable → done ✅
        If not → prune less aggressively

Full pipeline:
  70B → prune → 35B → fine-tune → 35B (near 70B quality)
```

---

## Outcome of Pruning

```
What You Gain:                   What You Lose:
──────────────                   ──────────────
Smaller model size               Some accuracy
  70B → 35–50B                   Typically 1–3% benchmark drop
                                 (acceptable for most use cases)
Faster inference
  Less compute per token         Requires calibration data
                                 + fine-tuning to recover quality
Less memory needed
  Fit on fewer GPUs              One-time engineering effort
                                 to find pruning recipe
Lower serving cost
  Fewer GPUs = cheaper API


Real world example:
  LLaMA-3 70B (original):
    Memory:  140GB (fp16)
    Speed:   100 tokens/sec
    Quality: MMLU 82%

  LLaMA-3 70B (30% pruned + fine-tuned):
    Memory:  95GB  ← 32% less
    Speed:   135 tokens/sec  ← 35% faster
    Quality: MMLU 80%  ← only 2% drop ✅
```

---

## Pruning vs Quantization — Key Difference

```
Quantization:               Pruning:
─────────────               ────────
Keep ALL weights            REMOVE some weights
Make each weight smaller    Fewer weights total
0.8731 → 0.8 (int8)        0.0003 → gone (deleted)

Complementary — use BOTH:
  Prune first → smaller model
  Then quantize → even smaller + faster
  Result: 70B model running quality on tiny hardware
```

---

## Summary

```
Pruning      = Remove least important parts of neural network
What pruned  = Middle layers, redundant attention heads,
               inactive FFN neurons
Why it works = LLMs are massively over-parameterized —
               not all parameters are equally needed
Outcome      = Smaller + faster model, minimal quality loss
Key step     = Always fine-tune after pruning to recover accuracy

Importance order (most → least prunable):
  KEEP → First layers (basic language understanding)
  KEEP → Last layers  (output prediction)
  PRUNE → Middle layers (most redundancy here)
```

> **Key insight for beginners:** LLMs are intentionally over-built — they have far more parameters than strictly necessary. Pruning exploits this redundancy. Think of it as finding that 30% of your team is doing duplicate work — reorganize, let them go, and the team still delivers the same output with less cost. The **middle layers** of an LLM are where the most redundancy lives, making them the primary pruning target.

---


