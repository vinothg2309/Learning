transformer_complete.md
# Resources

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |

---

# Transformers: A Beginner's Guide

## Table of Contents

- [Resources](#resources)
- [Transformers: A Beginner's Guide](#transformers-a-beginners-guide)
  - [Table of Contents](#table-of-contents)
- [Resources](#resources-1)
  - [What are Transformers?](#what-are-transformers)
  - [Transformer Architecture](#transformer-architecture)
    - [Detailed Explanation](#detailed-explanation)
  - [**Q \& K^T**](#q--kt)
  - [**Q × K^T**](#q--kt-1)
    - [Encoder](#encoder)
      - [What Does the Encoder Do?](#what-does-the-encoder-do)
      - [How It Works:](#how-it-works)
      - [Key Components:](#key-components)
      - [Use Cases:](#use-cases)
    - [Decoder](#decoder)
      - [What Does the Decoder Do?](#what-does-the-decoder-do)
      - [How It Works:](#how-it-works-1)
      - [Key Components:](#key-components-1)
      - [Two Types of Decoder Usage:](#two-types-of-decoder-usage)
      - [Use Cases:](#use-cases-1)
    - [Encoder-Only vs Decoder-Only: BERT vs GPT](#encoder-only-vs-decoder-only-bert-vs-gpt)
      - [BERT: Encoder-Only (Bidirectional Understanding)](#bert-encoder-only-bidirectional-understanding)
      - [GPT: Decoder-Only (Autoregressive Generation)](#gpt-decoder-only-autoregressive-generation)
      - [Visual Comparison](#visual-comparison)
      - [Why Not Use Both?](#why-not-use-both)
      - [Quick Summary](#quick-summary)
  - [The Attention Mechanism](#the-attention-mechanism)
    - [Three Components (Library Analogy):](#three-components-library-analogy)
  - [The Attention Formula](#the-attention-formula)
  - [Example: "The cat sat on the mat"](#example-the-cat-sat-on-the-mat)
    - [Step 1: Q × K^T (Similarity Scores)](#step-1-q--kt-similarity-scores)
    - [Step 2: Softmax (Normalize to Percentages)](#step-2-softmax-normalize-to-percentages)
      - [Purpose of Dividing by √dk in Scaled Dot-Product Attention](#purpose-of-dividing-by-dk-in-scaled-dot-product-attention)
    - [Step 3: Multiply by V (Mix Information)](#step-3-multiply-by-v-mix-information)
  - [Complete Flow](#complete-flow)
  - [Communication between Transformer Layer](#communication-between-transformer-layer)
  - [Why This Works](#why-this-works)
  - [Positional Encoding: Teaching Position to Transformers](#positional-encoding-teaching-position-to-transformers)
  - [Absolute Positional Encoding (Sinusoidal)](#absolute-positional-encoding-sinusoidal)
    - [Core Concept](#core-concept)
    - [The Formula](#the-formula)
      - [Understanding "i" = Dimension Index 🎯](#understanding-i--dimension-index-)
    - [Why Sine/Cosine Instead of Simple Numbers (0, 1, 2...)?](#why-sinecosine-instead-of-simple-numbers-0-1-2)
    - [How It Creates Unique Position Patterns](#how-it-creates-unique-position-patterns)
      - [Understanding Dimensions and Frequencies](#understanding-dimensions-and-frequencies)
      - [Real Example with Actual Numbers 📊](#real-example-with-actual-numbers-)
      - [Why This Multi-Speed Design Works 🎯](#why-this-multi-speed-design-works-)
      - [Clock Analogy - Revisited with Clarity ⏰](#clock-analogy---revisited-with-clarity-)
      - [Combined Effect - Unique Fingerprints](#combined-effect---unique-fingerprints)
      - [Summary Table 📋](#summary-table-)
    - [Key Terms Clarified](#key-terms-clarified)
    - [Training vs Inference](#training-vs-inference)
    - [Can We Change These After Training?](#can-we-change-these-after-training)
    - [Visual Representation](#visual-representation)
  - [](#)
    - [Visual Example (Sinusoidal PE)](#visual-example-sinusoidal-pe)
    - [Why Sine/Cosine?](#why-sinecosine)
    - [Complete Picture (Sinusoidal PE)](#complete-picture-sinusoidal-pe)
  - [RoPE (Rotary Position Embedding)](#rope-rotary-position-embedding)
    - [The Problem with Sinusoidal PE (Absolute Encoding)](#the-problem-with-sinusoidal-pe-absolute-encoding)
    - [What is RoPE?](#what-is-rope)
    - [Why Rotation Instead of Addition?](#why-rotation-instead-of-addition)
    - [Why RoPE?](#why-rope)
    - [How RoPE Works](#how-rope-works)
    - [Visual Comparison](#visual-comparison-1)
    - [Concrete Example: Sinusoidal vs RoPE](#concrete-example-sinusoidal-vs-rope)
      - [Sinusoidal PE (Absolute):](#sinusoidal-pe-absolute)
      - [RoPE (Relative):](#rope-relative)
    - [Visual Comparison](#visual-comparison-2)
    - [Why Modern LLMs Use RoPE](#why-modern-llms-use-rope)
    - [Real-World Benefits](#real-world-benefits)
    - [Key Takeaways (RoPE)](#key-takeaways-rope)
    - [Analogy](#analogy)
    - [RoPE Scaling: Extending Context Length](#rope-scaling-extending-context-length)
  - [T5 Relative Position Bias](#t5-relative-position-bias)
    - [What is T5 Bias?](#what-is-t5-bias)
    - [How T5 Bias Works](#how-t5-bias-works)
    - [Example](#example)
    - [Why T5 Bias?](#why-t5-bias)
  - [ALiBi (Attention with Linear Biases)](#alibi-attention-with-linear-biases)
    - [What is ALiBi?](#what-is-alibi)
    - [How ALiBi Works](#how-alibi-works)
    - [Visual Example](#visual-example)
    - [Key Clarification: ALiBi vs RoPE](#key-clarification-alibi-vs-rope)
    - [Bias Matrix: What It Looks Like](#bias-matrix-what-it-looks-like)
    - [Simple Analogy](#simple-analogy)
    - [Why ALiBi?](#why-alibi)
  - [Comparison: All Positional Encoding Methods](#comparison-all-positional-encoding-methods)
    - [Quick Guide](#quick-guide)
  - [Multi-Head Attention: Looking at Words from Different Perspectives](#multi-head-attention-looking-at-words-from-different-perspectives)
    - [How It Works](#how-it-works-2)
    - [Visual Example: "The cat sat on the mat"](#visual-example-the-cat-sat-on-the-mat)
    - [Architecture Diagram](#architecture-diagram)
    - [Splitting Dimensions](#splitting-dimensions)
    - [Why Each Head Has Its Own Q, K, V](#why-each-head-has-its-own-q-k-v)
    - [What Each Head Learns](#what-each-head-learns)
    - [Complete Flow](#complete-flow-1)
    - [Why Multi-Head Works](#why-multi-head-works)
    - [Analogy](#analogy-1)
  - [How Multi-Head Attention Learns: Backpropagation](#how-multi-head-attention-learns-backpropagation)
    - [The Learning Process (Training)](#the-learning-process-training)
    - [Detailed Backpropagation Flow](#detailed-backpropagation-flow)
    - [Simple Example: Learning Step by Step](#simple-example-learning-step-by-step)
    - [Weight Update Formula](#weight-update-formula)
    - [How Heads Specialize During Training](#how-heads-specialize-during-training)
    - [Why Different Heads Learn Different Patterns](#why-different-heads-learn-different-patterns)
    - [Gradient Flow Diagram](#gradient-flow-diagram)
    - [Analogy: Learning Like Students](#analogy-learning-like-students)
    - [Key Points](#key-points)
  - [Transformer Block Internals: MHA, FFN, and Loss Computation](#transformer-block-internals-mha-ffn-and-loss-computation)
    - [Component Overview](#component-overview)
    - [1. Attention Scores: How Words Find Related Words](#1-attention-scores-how-words-find-related-words)
    - [2. Shape Flow Through Transformer](#2-shape-flow-through-transformer)
    - [3. FFN Purpose \& Loss Computation](#3-ffn-purpose--loss-computation)
      - [FFN Explained for Beginners](#ffn-explained-for-beginners)
    - [4. Where Loss is Computed: End-to-End Flow](#4-where-loss-is-computed-end-to-end-flow)
    - [5. Backpropagation: How ALL Layers Learn](#5-backpropagation-how-all-layers-learn)
    - [Key Takeaways](#key-takeaways)
  - [Key Takeaways](#key-takeaways-1)
  - [Real-World Analogy](#real-world-analogy)
  - [Layer Normalization](#layer-normalization)
    - [The Problem \& Solution](#the-problem--solution)
    - [The Formula](#the-formula-1)
    - [Example Calculation](#example-calculation)
    - [Where Used in Transformers](#where-used-in-transformers)
    - [Why It Helps](#why-it-helps)
    - [Layer Norm vs Batch Norm](#layer-norm-vs-batch-norm)
    - [Visual Example: Full Flow](#visual-example-full-flow)
    - [Analogy](#analogy-2)
    - [Key Takeaways](#key-takeaways-2)
  - [Pre-Norm vs Post-Norm: LayerNorm Placement](#pre-norm-vs-post-norm-layernorm-placement)
    - [Post-Norm (Original Transformer, 2017)](#post-norm-original-transformer-2017)
    - [Pre-Norm (Modern: GPT-3, GPT-4, LLaMA)](#pre-norm-modern-gpt-3-gpt-4-llama)
    - [Post-Norm vs Pre-Norm: Representation Collapse \& Stability](#post-norm-vs-pre-norm-representation-collapse--stability)
    - [Pre-Norm vs Post-Norm vs ResiDual](#pre-norm-vs-post-norm-vs-residual)
  - [](#-1)
    - [Backpropagation: Why Pre-Norm is Better](#backpropagation-why-pre-norm-is-better)
      - [The Key Difference](#the-key-difference)
      - [Gradient Flow Comparison](#gradient-flow-comparison)
      - [Simple Math Explanation](#simple-math-explanation)
      - [Concrete Example: 10-Layer Model](#concrete-example-10-layer-model)
    - [Comparison Table](#comparison-table)
    - [Why Pre-Norm Enables Deep Models](#why-pre-norm-enables-deep-models)
    - [When to Use Each](#when-to-use-each)
    - [Code Example](#code-example)
    - [Key Takeaways](#key-takeaways-3)
- [Sparse Attention](#sparse-attention)
  - [Common Sparse Attention Patterns](#common-sparse-attention-patterns)
  - [Example: Longformer (Combined Patterns)](#example-longformer-combined-patterns)
    - [Three Types of Attention in Longformer](#three-types-of-attention-in-longformer)
      - [1. Local Window Attention (w=512)](#1-local-window-attention-w512)
      - [2. Global Attention (g=8 special tokens)](#2-global-attention-g8-special-tokens)
    - [Understanding Special Tokens in Transformers and LLMs](#understanding-special-tokens-in-transformers-and-llms)
      - [Common Special Tokens Reference Table](#common-special-tokens-reference-table)
      - [Model-Specific Formats](#model-specific-formats)
      - [Training Usage](#training-usage)
      - [3. Strided Attention (every 64th token)](#3-strided-attention-every-64th-token)
  - [Key Models Using Sparse Attention](#key-models-using-sparse-attention)
  - [Benefits \& Trade-offs](#benefits--trade-offs)
  - [Key Takeaways](#key-takeaways-4)
  - [Is Sparse Attention Widely Used in Modern LLMs?](#is-sparse-attention-widely-used-in-modern-llms)
- [Sharing Attention Heads](#sharing-attention-heads)
  - [Multi-Head Attention (MHA) vs Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA)](#multi-head-attention-mha-vs-multi-query-attention-mqa-vs-grouped-query-attention-gqa)
    - [Overview](#overview)
    - [1. Multi-Head Attention (MHA) - Standard](#1-multi-head-attention-mha---standard)
    - [2. Multi-Query Attention (MQA) - Maximum Sharing](#2-multi-query-attention-mqa---maximum-sharing)
    - [3. Grouped-Query Attention (GQA) - Balanced](#3-grouped-query-attention-gqa---balanced)
    - [Comparison Table](#comparison-table-1)
    - [Visual Comparison](#visual-comparison-3)
    - [Why This Matters](#why-this-matters)
    - [Example: Llama-2 Uses GQA](#example-llama-2-uses-gqa)
    - [When to Use Each](#when-to-use-each-1)
    - [Why Sharing KV Causes Quality Drop](#why-sharing-kv-causes-quality-drop)
    - [Key Takeaway](#key-takeaway)
  - [How It Works](#how-it-works-3)
  - [Sharing Strategies](#sharing-strategies)
  - [Example: ALBERT (A Lite BERT)](#example-albert-a-lite-bert)
  - [Benefits \& Trade-offs](#benefits--trade-offs-1)
  - [Visual Comparison](#visual-comparison-4)
  - [Key Models Using Shared Heads](#key-models-using-shared-heads)
  - [Key Takeaways](#key-takeaways-5)
- [Transformer-Based Models](#transformer-based-models)
  - [Quick Comparison](#quick-comparison)
  - [1. Encoder-Only (BERT-style)](#1-encoder-only-bert-style)
  - [2. Encoder-Decoder (T5/BART-style)](#2-encoder-decoder-t5bart-style)
  - [3. Decoder-Only (GPT/LLaMA-style)](#3-decoder-only-gptllama-style)
  - [Why Architecture Matters: Example Task](#why-architecture-matters-example-task)
    - [Encoder-Only (BERT):](#encoder-only-bert)
    - [Decoder-Only (GPT):](#decoder-only-gpt)
  - [Modern Trend: Decoder-Only Dominance](#modern-trend-decoder-only-dominance)
  - [Summary for Beginners](#summary-for-beginners)

---

# Resources
- https://cme295.stanford.edu/syllabus/

## What are Transformers?

Neural network architecture from 2017 ("Attention is All You Need") - foundation for GPT, BERT, and modern AI models.

**Key Idea**: Process entire sentences at once (not word-by-word) using "attention" to understand word relationships.

```
Old RNNs:        "The" → "cat" → "sat" (slow, forgets earlier words)
Transformers:    All words at once! (fast, remembers everything)
```

---

## Transformer Architecture

<img src="images/transformer-architecture.png" alt="Transformer Architecture Overview" width="600">

---

### Detailed Explanation

<img src="images/attention-detailed-explanation.png" alt="Attention Detailed Explanation" width="650">

**Q & K^T**
---
<img src="images/q-and-k-transpose.png" alt="Q and K Transpose" width="500">


**Q × K^T**
---
<img src="images/q-times-k-transpose.png" alt="Q times K Transpose" width="550">


### Encoder

**Purpose**: The encoder's job is to **understand and encode the input** into rich contextual representations.

#### What Does the Encoder Do?

1. **Bidirectional Understanding**: Processes all input tokens simultaneously, allowing each token to attend to every other token (both left and right context)
2. **Context Extraction**: Creates deep contextual embeddings that capture the meaning and relationships of input tokens
3. **Feature Extraction**: Transforms raw input into meaningful representations that capture semantic and syntactic information

#### How It Works:

```
Input: "The cat sat on the mat"
   ↓
Encoder processes ALL tokens at once:
   ↕     ↕   ↕   ↕   ↕    ↕
Each word can see ALL other words
   ↕     ↕   ↕   ↕   ↕    ↕
   ↓
Output: Rich contextual embeddings for each token
```

#### Key Components:

- **Self-Attention**: Each token attends to all other tokens in the input
- **Feed-Forward Networks**: Further processes the attended representations
- **Layer Normalization & Residual Connections**: Stabilizes training and preserves information
- **Stacked Layers** (typically 6-12): Each layer refines the understanding

#### Use Cases:

- **BERT-style models**: Classification, sentiment analysis, named entity recognition
- **Translation (Encoder part)**: Understanding the source language
- **Question Answering**: Understanding both the question and context
- **Semantic Search**: Creating meaningful embeddings for similarity matching

<img src="images/encoder-architecture.png" alt="Encoder Architecture" width="600">

---

### Decoder

**Purpose**: The decoder's job is to **generate output sequences** one token at a time, using previously generated tokens and optionally encoder outputs.

#### What Does the Decoder Do?

1. **Sequential Generation**: Produces output tokens one at a time in an autoregressive manner
2. **Causal/Masked Attention**: Each token can only attend to previously generated tokens (left-to-right, no future peeking)
3. **Cross-Attention** (in Encoder-Decoder models): Attends to encoder outputs to incorporate source information
4. **Conditional Generation**: Generates outputs conditioned on the input context

#### How It Works:

```
Generation Process (e.g., GPT):
Input: "The cat sat on the"
   ↓
Decoder generates sequentially:
Token 1: "The"      → can see: [The]
Token 2: "cat"      → can see: [The, cat]
Token 3: "sat"      → can see: [The, cat, sat]
Token 4: "on"       → can see: [The, cat, sat, on]
Token 5: "the"      → can see: [The, cat, sat, on, the]
Token 6: "mat" ✨   → prediction!
   ↓
Output: "The cat sat on the mat"
```

#### Key Components:

- **Masked Self-Attention**: Prevents tokens from attending to future positions (maintains causality)
- **Cross-Attention** (in full Transformers): Attends to encoder outputs to use source context
- **Feed-Forward Networks**: Processes the attended representations
- **Output Projection**: Maps decoder outputs to vocabulary for next token prediction

#### Two Types of Decoder Usage:

**1. Decoder-Only (GPT, LLaMA, etc.)**
```
Input → Decoder Stack → Output
(No encoder, self-contained generation)
```

**2. Encoder-Decoder (Original Transformer, T5, BART)**
```
Input → Encoder → Rich Representations
                        ↓
          Previous Outputs → Decoder → Next Token
          (uses both masked self-attention and cross-attention)
```

#### Use Cases:

- **Text Generation**: GPT models for creative writing, completion
- **Translation (Decoder part)**: Generating target language from encoded source
- **Summarization**: Generating concise summaries from encoded documents
- **Code Generation**: Producing code from natural language descriptions
- **Conversational AI**: Generating responses in chatbots

**Important Note**: Send your already decoded words along with encoded embeddings from Encoder (in encoder-decoder models) to generate next words.

**self-attention layer** in the decoder represents **Masked self-attention** (causal attention).

<img src="images/decoder-masked-attention.png" alt="Decoder Masked Attention" width="650">


<img src="images/decoder-detailed.png" alt="Decoder Detailed" width="600">

---

### Encoder-Only vs Decoder-Only: BERT vs GPT

**Why different architectures?** Different tasks need different designs.

| Model Type | Architecture | Use Case | Example |
|------------|--------------|----------|---------|
| **BERT** | Encoder Only | Understanding/Reading | "Is this email spam?" |
| **GPT** | Decoder Only | Generation/Writing | "Write a story about cats" |

#### BERT: Encoder-Only (Bidirectional Understanding)

**Why encoder only?** Needs to understand context from BOTH directions.

```
Task: "Is this email spam?"

Email: "Congratulations! You won a prize. Click here."
         ↕         ↕     ↕     ↕      ↕      ↕
    BERT looks BOTH ways (left + right context)
         ↕         ↕     ↕     ↕      ↕      ↕

Understanding: "won" + "prize" + "click" → Pattern = SPAM

BERT sees full context → Better understanding
```

**Key Features:**
- Reads entire sentence at once
- Sees future and past words simultaneously
- Perfect for: classification, question answering, sentiment analysis

#### GPT: Decoder-Only (Autoregressive Generation)

**Why decoder only?** Generates text one word at a time, left-to-right.

```
Task: "Write a story about cats"

Generated: "The cat sat on the ___"
            ↓   ↓   ↓   ↓   ↓    ?
           Can only see LEFT context (already generated)
            ↓   ↓   ↓   ↓   ↓
          Predicts next: "mat"

GPT generates sequentially → Natural text generation
```

**Key Features:**
- Masked attention (can't see future words)
- Generates one token at a time
- Perfect for: text generation, completion, creative writing

**Important Clarification:**

```
❌ MISCONCEPTION:
"GPT uses a pretrained encoder"

✓ TRUTH:
GPT has NO encoder at all!

GPT architecture:
┌─────────────────────────────────┐
│  Input Embedding                │
│         ↓                       │
│  Positional Encoding            │
│         ↓                       │
│  Decoder Blocks (ONLY)          │
│  - Masked Self-Attention        │
│  - Feed Forward                 │
│  (No encoder, no cross-attention) │
│         ↓                       │
│  Output Prediction              │
└─────────────────────────────────┘
```

**Why GPT doesn't need an encoder:**

1. **Self-Sufficient**: The decoder blocks process input directly
2. **Causal Attention**: Each position can only attend to previous positions
3. **Simple Architecture**: Input → Decoder Stack → Output
4. **No Separate Understanding Phase**: Understanding happens within decoder layers

**What GPT actually does:**

```
Input: "The cat sat on the"
  ↓
Embedding + Positional Encoding
  ↓
Decoder Layer 1 (masked self-attention)
  "The" sees: [The]
  "cat" sees: [The, cat]
  "sat" sees: [The, cat, sat]
  "on"  sees: [The, cat, sat, on]
  "the" sees: [The, cat, sat, on, the]
  ↓
Decoder Layer 2, 3, 4... (stacked)
  Each layer refines understanding
  ↓
Output: Predict next word → "mat"
```

**Contrast with Original Transformer (Encoder + Decoder):**

```
Original Transformer (Translation):
┌──────────────┐         ┌──────────────┐
│   ENCODER    │────────→│   DECODER    │
│              │         │              │
│ Input: "Cat" │         │ Output: "Chat"│
│ (English)    │         │ (French)     │
└──────────────┘         └──────────────┘
      ↓                        ↓
Understands input    Generates output using
bidirectionally      encoder's understanding

GPT (Generation only):
┌──────────────┐
│   DECODER    │
│    (ONLY)    │
│              │
│ Input: "Cat" │
│ Output: "sat"│
└──────────────┘
      ↓
Everything happens
in decoder layers
```

#### Visual Comparison

```
BERT (Encoder):
┌─────────────────────────────────────┐
│ "The cat sat on the mat"            │
│   ↕    ↕   ↕   ↕   ↕    ↕           │
│  All words see ALL other words      │
│  (bidirectional attention)          │
└─────────────────────────────────────┘
Output: Understanding/Classification

GPT (Decoder):
┌─────────────────────────────────────┐
│ "The cat sat on the ___"            │
│   ↓    ↓   ↓   ↓   ↓                │
│  Each word only sees previous words │
│  (masked/causal attention)          │
└─────────────────────────────────────┘
Output: Next word prediction "mat"
```

#### Why Not Use Both?

**Original Transformer** (2017) used both:
- Encoder: Understands input (e.g., English sentence)
- Decoder: Generates output (e.g., French translation)

**Modern Trend** - Specialization:
- **Understanding tasks** → Encoder-only (BERT) - simpler, faster
- **Generation tasks** → Decoder-only (GPT) - simpler, scalable
- **Translation tasks** → Encoder-Decoder (T5, BART)

#### Quick Summary

```
Need to understand? → BERT (Encoder-only)
  - Classification
  - Question answering
  - Named entity recognition

Need to generate? → GPT (Decoder-only)
  - Text completion
  - Story writing
  - Code generation

Need both? → Full Transformer (Encoder + Decoder)
  - Translation
  - Summarization
```

---

## The Attention Mechanism

When you read **"The cat sat on the mat"**, your brain links "cat" with "sat" and "mat", ignoring "the" and "on".

**Attention does exactly this!**

### Three Components (Library Analogy):

| Component | Purpose | Analogy |
|-----------|---------|---------|
| **Query (Q)** | What am I looking for? | Your question |
| **Key (K)** | What's available? | Library catalog |
| **Value (V)** | Actual information | The books |

**V (Value) contains the actual information we want to extract. The attention weights from softmax tell us "how much of each value to use," and multiplying creates a weighted sum of values based on relevance.**

**The Intuitive Explanation**
Think of it like a weighted average:

```
Question: "What should I focus on?"
Keys: "I can help with this topic"
Values: "Here's the actual information"

Attention = "How much to focus" × "What to focus on"
          = softmax weights     × Values
```

**1. Each Component's Role**
```
Q (Query):  "What am I looking for?"
            - Represents what the current token needs

K (Key):    "What do I contain?"
            - Represents what information each token offers

V (Value):  "Here's my actual content"
            - Contains the actual information to be passed forward
```

**2. The Attention Process**
```
Step 1: Q·K^T → "How relevant is each position?"
        Produces attention SCORES (relevance measures)

Step 2: softmax(Q·K^T/√dk) → "What percentage for each?"
        Converts scores to WEIGHTS that sum to 1

Step 3: weights × V → "Extract weighted information"
        Creates final output by mixing Values
```

---

## The Attention Formula

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

**Step-by-step:**
1. **Q × K^T** → Find similarity scores between all word pairs
2. **÷ √d_k** → Scale down (prevents large numbers)
3. **softmax** → Convert to percentages (sum = 100%)
4. **× V** → Mix information based on percentages

---

## Example: "The cat sat on the mat"

### Step 1: Q × K^T (Similarity Scores)

**Why?** Calculate how related each word is to every other word.
**What?** Dot product gives higher scores for related words, lower for unrelated.

```
Q(cat) · K(The) = 1.25  (low)
Q(cat) · K(cat) = 5.40  (HIGH)
Q(cat) · K(sat) = 4.80  (HIGH)
Q(cat) · K(on)  = 1.20  (low)
Q(cat) · K(the) = 1.25  (low)
Q(cat) · K(mat) = 4.10  (high)

Visual:
The   cat   sat   on    the   mat
▁     ████  ███   ▁     ▁     ███
```

**Why K^T?** Creates attention matrix comparing every word with every word.

### Step 2: Softmax (Normalize to Percentages)

**Why?** Raw scores are hard to interpret - need probabilities that sum to 100%.
**What?** Converts scores to attention weights showing "how much" to focus on each word.

```
Before: [1.25, 5.40, 4.80, 1.20, 1.25, 4.10]
After:  [4%,   41%,  29%,  4%,   4%,   19%]  ← Sums to 100%

Attention Map for "cat":
  The     cat      sat      on      the     mat
  ▁▁▁     ████████ ██████   ▁▁      ▁▁      ████
  4%      41%      29%      4%      4%      19%
```

#### Purpose of Dividing by √dk in Scaled Dot-Product Attention

1. **√dk scaling prevents softmax saturation** by keeping dot products in a reasonable range
2. **Normalizes variance** back to 1, regardless of dimension size
3. **Enables stable training** by preventing vanishing gradients
4. **Works for any dimension** - automatically adjusts as dk changes
5. **Simple but crucial** - without it, transformers wouldn't train effectively!

**The Detailed Explanation:**

**1. The Attention Formula**
Where:

Q (Query): shape (seq_len, dk)
K (Key): shape (seq_len, dk)
dk: dimension of key/query vectors

**2. Why Division is Needed**

**Problem**: Dot products grow with dimension

When computing QK^T, each element is a dot product of two dk-dimensional vectors:

If Q and K have mean 0 and variance 1 (typical after initialization):

Variance of dot product = dk
Standard deviation = √dk

Example with Numbers:
**Small dimension (dk=4):**

```
q = [0.5, -0.3, 0.8, -0.2]
k = [0.4, 0.6, -0.5, 0.7]

q·k = 0.5×0.4 + (-0.3)×0.6 + 0.8×(-0.5) + (-0.2)×0.7
    = 0.2 - 0.18 - 0.4 - 0.14 = -0.52
```

**Large dimension (dk=64):**

```
q·k could be around ±8 to ±16  (grows with √64 = 8)
```

**3. The Softmax Problem**

Without scaling(√dk division), large dot products cause softmax to saturate:

```
# Without scaling (dk=64) - Not divided by √dk
scores = [15, 12, -18, 2]  # Large values!

softmax([15, 12, -18, 2])
= [0.952, 0.047, 0.000, 0.001]  # One value dominates!
```
Problems:

❌ Softmax becomes almost one-hot (one value ≈ 1, others ≈ 0)
❌ Gradients become extremely small (vanishing gradients)
❌ Model can't learn effectively

**With scaling (divide by √64 = 8):**

scores = [15, 12, -18, 2]
scaled = [1.875, 1.5, -2.25, 0.25]  # Smaller values!

softmax([1.875, 1.5, -2.25, 0.25])
= [0.465, 0.319, 0.007, 0.091]  # More balanced!

✅ Gradients remain healthy
✅ Model can learn from multiple positions



### Step 3: Multiply by V (Mix Information)

**Why?** Need to actually extract and combine information from relevant words.
**What?** Weighted average - take more info from high-attention words, less from low-attention words.

```
Output for "cat" = 41% × V(cat)
                 + 29% × V(sat)
                 + 19% × V(mat)
                 + 11% × others

Result: "cat" now has context from related words!
```

---

## Complete Flow

```
INPUT: "The cat sat on the mat"
   ↓
Create Q, K, V (learned transformations)
   ↓
Q × K^T → Similarity scores
   ↓
÷ √d_k → Scale
   ↓
softmax → Percentages
   ↓
× V → Context-aware output
   ↓
OUTPUT: Each word understands its context!
```

---

## Communication between Transformer Layer

**Initialize K,V,Q Matrix**

```
"The"           "cat"            "sat"
   ↓               ↓                ↓
[Token ID: 464] [Token ID: 3797] [Token ID: 3332]
   ↓               ↓                ↓
┌────────────────────────────────────────────────┐
│  Embedding Matrix E  (50,257 rows × 768 cols)  │
│  Fetch row for each token ID  → 768-dim vector │
└────────────────────────────────────────────────┘
   ↓               ↓                ↓
emb("The")     emb("cat")       emb("sat")    ← static lookup, same always
   ↓               ↓                ↓
+ pos_enc(0)   + pos_enc(1)    + pos_enc(2)   ← add position info
   ↓               ↓                ↓
Stack into matrix X  [3 × 768]
   │
   ├── × W_Q [768×64] → Q [3×64]   "What am I looking for?"      ← NOT cached
   ├── × W_K [768×64] → K [3×64]   "What do I offer for matching?" ← CACHED
   └── × W_V [768×64] → V [3×64]   "What do I contribute?"        ← CACHED
                            ↓
               Attention(Q, K, V) = softmax(QKᵀ/√d) × V
                            ↓
               Each token enriched with context from all others
```

**Layer 1 → Layer 2: Step by Step**
Let's trace "The cat sat" through two full layers:

**Layer 1 Input — Raw Embeddings**

```
H_in (Layer 1):
  "The" → [0.30,  0.70,  0.20,  0.10]   ← raw embedding + position
  "cat" → [0.21, -0.53,  0.87,  0.12]   ← raw embedding + position
  "sat" → [0.60,  0.10, -0.40,  0.50]   ← raw embedding + position

```

**Layer 1 — Self Attention**

```
Compute Q, K, V from H_in using Layer 1's OWN weight matrices:
  Q_L1 = H_in × W_Q_L1
  K_L1 = H_in × W_K_L1
  V_L1 = H_in × W_V_L1


Attention runs:
  "cat" attends to "The" (low), "cat" (medium), "sat" (high)
  → learns: "cat is the one doing the sitting"

  "sat" attends to "cat" (high), "The" (low)
  → learns: "sitting was done by cat"

attn_output:
  "The" → [0.11, 0.33, 0.05, 0.22]   ← slightly updated
  "cat" → [0.55, 0.41, 0.73, 0.38]   ← now knows it's the subject
  "sat" → [0.43, 0.62, 0.29, 0.71]   ← now knows who did it

```

**Layer 1 — Residual Connection**

```
H_mid = LayerNorm(H_in + attn_output)

"cat":
  H_in      = [0.21, -0.53,  0.87,  0.12]   ← original
  attn_out  = [0.55,  0.41,  0.73,  0.38]   ← what attention added
  sum       = [0.76, -0.12,  1.60,  0.50]   ← added together
  LayerNorm → [0.44,  0.33,  0.91,  0.61]   ← normalized
```
The residual connection is crucial — it ensures the original embedding is never lost. Attention adds ON TOP of what was already there.

**Layer 1 — Feed Forward Network (FFN)**

```
FFN processes each token independently:
  ffn_out("cat") = FFN([0.44, 0.33, 0.91, 0.61])

H_out = LayerNorm(H_mid + ffn_out)


Final Layer 1 output for "cat":
  H_out_L1("cat") = [0.67, 0.22, 0.85, 0.74]
                     ← "cat" enriched with Layer 1 understanding
```

**This H_out Becomes Layer 2's Input**

```
Layer 1 H_out:                         Layer 2 H_in:
  "The" → [0.31, 0.45, 0.18, 0.29]  →  "The" → [0.31, 0.45, 0.18, 0.29]
  "cat" → [0.67, 0.22, 0.85, 0.74]  →  "cat" → [0.67, 0.22, 0.85, 0.74]
  "sat" → [0.52, 0.78, 0.33, 0.61]  →  "sat" → [0.52, 0.78, 0.33, 0.61]

```
Same values, just relabeled as the new layer's input.

**Layer 2 — Freshly Computes Its Own Q, K, V**

```
Layer 2 has its OWN weight matrices: W_Q_L2, W_K_L2, W_V_L2
  (completely different from W_Q_L1, W_K_L1, W_V_L1)

Q_L2 = H_in_L2 × W_Q_L2   ← different projection than Layer 1
K_L2 = H_in_L2 × W_K_L2   ← different projection than Layer 1
V_L2 = H_in_L2 × W_V_L2   ← different projection than Layer 1

```
The Q, K, V in Layer 2 are completely new vectors — but they are computed from a richer hidden state. So they implicitly carry Layer 1's knowledge.

```
Layer 1 K("cat"):  "I am a noun, I can be a subject"
                    (computed from raw embedding)

Layer 2 K("cat"):  "I am the specific agent who performed 'sat'"
                    (computed from Layer 1's enriched hidden state)

```
Layer 2's K, Q, V are more meaningful because the hidden state they're built from already contains Layer 1's insights.

---

## Why This Works

**Q × K^T**: Dot product measures similarity
- Similar vectors → High score → Related words
- Different vectors → Low score → Unrelated

**Softmax**: Converts scores to probabilities (0-100%)

**× V**: Weighted sum - blend information from important words

---

## Positional Encoding: Teaching Position to Transformers

**The Problem**: Transformers process all words simultaneously - they don't know word order!
- "cat chased dog" vs "dog chased cat" would look identical without position info

**The Solution**: Add positional information to word embeddings.

There are multiple approaches:
1. **Absolute Positional Encoding** (Original Transformer, BERT) - Add fixed position patterns
2. **Relative Positional Encoding (RoPE)** (LLaMA, GPT-NeoX) - Rotate Q/K by position
3. **T5 Relative Position Bias** (T5, FLAN-T5) - Add learned bias to attention scores
4. **ALiBi** (BLOOM, MPT) - Add linear bias based on distance

---

## Absolute Positional Encoding (Sinusoidal)

### Core Concept

**Why?** Transformers process all words simultaneously - need to tell them word order.
**What?** Add unique sine/cosine wave patterns to each word's embedding.

```
Word Embedding + Positional Encoding = Final Input

"The cat sat" → [E_The+PE_0] [E_cat+PE_1] [E_sat+PE_2]
```

### The Formula

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

pos = position (0, 1, 2, ...)
i   = dimension index (which slot in the vector)
d   = embedding dimension (e.g., 512, 768)
```

#### Understanding "i" = Dimension Index 🎯

**Think of each word as a vector (list of numbers):**

```
Word "cat" at position 1 has a positional encoding vector:
PE_cat = [value_0, value_1, value_2, value_3, ..., value_511]
          ↑        ↑        ↑        ↑              ↑
          dim 0    dim 1    dim 2    dim 3        dim 511

"i" tells us WHICH SLOT in this vector we're calculating!
```

**Step-by-step calculation:**

If embedding dimension d = 512, we calculate:

```
For dimension 0 (i=0):
  PE(pos, 0) = sin(pos / 10000^(0/512))   ← Even dimension uses sin
  PE(pos, 1) = cos(pos / 10000^(0/512))   ← Odd dimension uses cos

For dimension 2 (i=1):
  PE(pos, 2) = sin(pos / 10000^(2/512))   ← Even dimension uses sin
  PE(pos, 3) = cos(pos / 10000^(2/512))   ← Odd dimension uses cos

For dimension 4 (i=2):
  PE(pos, 4) = sin(pos / 10000^(4/512))   ← Even dimension uses sin
  PE(pos, 5) = cos(pos / 10000^(4/512))   ← Odd dimension uses cos

...continues until dimension 511
```

**Formula breakdown:**
- **2i** = even dimensions (0, 2, 4, 6...) → use **sin**
- **2i+1** = odd dimensions (1, 3, 5, 7...) → use **cos**
- **i** starts from 0 and goes to (d/2 - 1)

**Example for position 1, dimension 0 (i=0):**
```
PE(1, 0) = sin(1 / 10000^(0/512))
         = sin(1 / 10000^0)
         = sin(1 / 1)
         = sin(1)
         ≈ 0.84
```

**Example for position 1, dimension 1 (i=0):**
```
PE(1, 1) = cos(1 / 10000^(0/512))
         = cos(1 / 10000^0)
         = cos(1 / 1)
         = cos(1)
         ≈ 0.54
```

### Why Sine/Cosine Instead of Simple Numbers (0, 1, 2...)?

| Problem | Simple Indexing | Sine/Cosine |
|---------|----------------|-------------|
| **Bounded** | ✗ Grows to infinity | ✓ Always [-1, +1] |
| **Normalized** | ✗ Hard to learn | ✓ Consistent range |
| **Extrapolation** | ✗ Fails on unseen positions | ✓ Continues pattern |
| **Relative distance** | ✗ No pattern | ✓ Encodes distance |

**Example:**
```
Simple: Position 1000 = [1000, 1000, ...] ← Never seen in training!
Sine/Cosine: Position 1000 = [sin(1000/...), cos(1000/...)] ← Still [-1,+1]
```

### How It Creates Unique Position Patterns

**Different frequencies = Different change speeds** 🎵

#### Understanding Dimensions and Frequencies

**Key Insight:** Each dimension pair uses a different frequency (how fast the wave oscillates).

```
Lower dimension index (i) → HIGHER frequency → Changes FAST
Higher dimension index (i) → LOWER frequency → Changes SLOW
```

**Why?** Look at the formula exponent: `10000^(2i/d)`
- When i=0 (dim 0,1): `10000^(0/512) = 10000^0 = 1` (small divisor → high freq)
- When i=1 (dim 2,3): `10000^(2/512) = 10000^0.0039 ≈ 1.018` (bigger divisor → lower freq)
- When i=255 (dim 510,511): `10000^(510/512) ≈ 9772` (huge divisor → very low freq)

#### Real Example with Actual Numbers 📊

Let's see dimensions 0, 2, and 510 for positions 0-5 in a 512-dimensional encoding:

**Dimension 0 & 1 (i=0) - HIGH FREQUENCY = Changes FAST:**
```
Position:    0      1      2       3       4       5
dim 0 (sin): 0.00   0.84   0.91    0.14   -0.76   -0.96  ← Changes rapidly!
dim 1 (cos): 1.00   0.54  -0.42   -0.99   -0.65    0.28  ← Every position differs

Wave oscillates quickly → Good for detecting NEARBY positions
"Is 'cat' right next to 'The'?" → YES! dim 0 changed from 0.00→0.84
```

**Dimension 2 & 3 (i=1) - MEDIUM FREQUENCY = Changes MEDIUM:**
```
Position:    0      1      2       3       4       5
dim 2 (sin): 0.00   0.001  0.002   0.003   0.004   0.005  ← Changes slowly
dim 3 (cos): 1.00   1.00   1.00    1.00    1.00    1.00   ← Almost constant

Wave oscillates moderately → Good for detecting MEDIUM-RANGE positions
"Is 'mat' within 10 words of 'The'?" → Check these dimensions
```

**Dimension 510 & 511 (i=255) - LOW FREQUENCY = Changes VERY SLOW:**
```
Position:    0        1        2        3        4        5
dim 510:     0.00000  0.00010  0.00020  0.00030  0.00040  0.00050  ← Barely changes!
dim 511:     1.00000  0.99999  0.99999  0.99999  0.99999  0.99999  ← Almost flat

Wave oscillates very slowly → Good for GLOBAL position in long sequences
"Is this word in the first half or second half of a 512-word document?"
```

#### Why This Multi-Speed Design Works 🎯

**Think of it like measuring distances:**

```
🏃 Fast Dimensions (dim 0, 1, 2, 3):
   Like a millimeter ruler - precise for nearby words
   Position 5 vs 6: BIG difference in these dimensions
   Use case: "The cat" (adjacent words)

🚶 Medium Dimensions (dim 100, 101, ...):
   Like a meter stick - measures within a paragraph
   Position 5 vs 6: Small difference, but 5 vs 50: BIG difference
   Use case: Words in the same sentence/paragraph

🚗 Slow Dimensions (dim 510, 511):
   Like a kilometer marker - tracks global position
   Position 5 vs 6: Tiny difference, but 5 vs 500: BIG difference
   Use case: Beginning vs end of a long document
```

#### Clock Analogy - Revisited with Clarity ⏰

```
Imagine a position as a time:

Dimension 0-1 (i=0):    🕐 SECOND HAND (fast)
   Position 0: 0 seconds
   Position 1: 1 second     ← Noticeable change!
   Position 2: 2 seconds    ← Every step is visible

Dimension 100-101:      🕐 MINUTE HAND (medium)
   Position 0: 0 minutes
   Position 1: Still ~0     ← Barely moved
   Position 60: 1 minute    ← Now we see a difference!

Dimension 510-511:      🕐 HOUR HAND (slow)
   Position 0: 12 o'clock
   Position 1: Still 12     ← Can't see movement
   Position 100: Still 12
   Position 300: Now ~1 PM  ← Only changes over long sequences
```

#### Combined Effect - Unique Fingerprints

**Each position gets a UNIQUE combination across ALL dimensions:**

```
Position 0:  [0.00, 1.00, 0.00, 1.00, ..., 0.00000, 1.00000] ← Unique fingerprint
Position 1:  [0.84, 0.54, 0.001, 1.00, ..., 0.00010, 0.99999] ← Different!
Position 2:  [0.91,-0.42, 0.002, 1.00, ..., 0.00020, 0.99999] ← Also unique!
Position 50: [0.26,-0.97, 0.05, 0.999, ..., 0.00512, 0.99997] ← Completely different!

Think of it like:
- Fast dims: Distinguish position 1 from 2
- Slow dims: Distinguish position 1 from 100
- Together: Every position has a unique pattern!
```

#### Summary Table 📋

| Dimension Index (i) | Dimensions | Frequency | Change Speed | Good For |
|---------------------|------------|-----------|--------------|----------|
| **i = 0** | dim 0, 1 | Highest | Very Fast | Adjacent words (1-2 positions apart) |
| **i = 1** | dim 2, 3 | High | Fast | Nearby words (3-10 positions) |
| **i = 50** | dim 100, 101 | Medium | Medium | Same paragraph (10-50 positions) |
| **i = 100** | dim 200, 201 | Low | Slow | Same section (50-200 positions) |
| **i = 255** | dim 510, 511 | Lowest | Very Slow | Document-level (200-512 positions) |

**Bottom Line:**
- **Low dimension indices (i=0,1,2)** → High frequency → Fast changes → Detect NEARBY positions
- **High dimension indices (i=254,255)** → Low frequency → Slow changes → Detect FAR-APART positions
- **Together:** Every position gets a unique encoding that captures both local and global position!

### Key Terms Clarified

```
❌ CONFUSION: "512 dimensions"
✓ TWO separate numbers:

1. Sequence Length (context window): 512, 2048, 4096 tokens
   → How many words the model can process

2. Embedding Dimensions (vector size): 512, 768, 12288 dims
   → Size of each word's representation

Example - BERT:
  - Context: 512 tokens (trained on sequences ≤ 512 words)
  - Embedding: 768 dims (each word = 768-dim vector)
  - PE: 768-dim vector for each of 512 positions
```

### Training vs Inference

**During Training:**
```
Model trained on max 512 tokens:
- Sees positions 0-511 only
- NEVER sees position 513, 1000, etc.
```

**During Inference:**
```
Can still handle longer sequences:
✓ Position 513: sin(513/10000^x) ← Continues the wave pattern
✓ Position 1000: Works (quality may degrade)

Why? Sine/cosine formulas work for ANY position!
```

### Can We Change These After Training?

**Both parameters are FIXED at training:**

| Parameter | Examples | Can Change? |
|-----------|----------|-------------|
| **Sequence length** | BERT: 512, GPT-3: 2048, LLaMA-2: 4096 | ✗ Fixed |
| **Embedding dims** | BERT: 768, GPT-3: 12288 | ✗ Fixed |

**Extending context (extrapolation):**
- Sinusoidal PE: Works but degrades
- RoPE: Better (2K → 32K with scaling)
- ALiBi: Best (2K → 10K+ easily)

### Visual Representation

**Frequency Formula (ω_i)**

This shows the frequency parameter used in positional encoding. The value 10000 controls how quickly the sine/cosine waves oscillate. Different dimension indices (i) create different frequencies, allowing each position to have a unique pattern.

<img src="images/positional-encoding-frequency-formula.png" alt="Frequency Formula" width="400">

**Position Encoding for Different Positions**

This diagram shows how the same sine/cosine formulas create different encoding vectors for different word positions (m and n). Each position gets its own unique combination of sine and cosine values across all dimensions (2i, 2i+1, etc.).

<img src="images/positional-encoding-visualization.png" alt="Position Encoding Visualization" width="500">

**Relative Position Property**

This explains a key advantage of sine/cosine encoding: the dot product between two position encodings (PE_m, PE_n) depends only on their relative distance (m - n), not their absolute positions. This helps the model understand word relationships regardless of where they appear in the sentence.

<img src="images/positional-encoding-relative-property.png" alt="Relative Position Property" width="450">
---

### Visual Example (Sinusoidal PE)

**What does PE_dim0, PE_dim1 mean?**
- Each word gets a positional encoding **vector** with multiple dimensions (e.g., 512 dimensions)
- **PE_dim0** = value at dimension 0 of the vector
- **PE_dim1** = value at dimension 1 of the vector
- Different dimensions use different frequency sine/cosine waves
- **Lower dimensions** (dim0, dim1) = slow-changing patterns (low frequency)
- **Higher dimensions** (dim2, dim3) = fast-changing patterns (high frequency)

```
Sentence: "The cat sat on mat"

Position:  0     1     2    3    4
          ─────────────────────────
PE_dim0:  0.0   0.84  0.91 0.14 -0.76  (sin wave - low freq)
PE_dim1:  1.0   0.54 -0.42 -0.99 -0.65  (cos wave - low freq)
PE_dim2:  0.0   0.01  0.02  0.03  0.05  (sin wave - high freq)
PE_dim3:  1.0   1.0   1.0   1.0   1.0   (cos wave - high freq)

Pattern visualization:
Pos 0: ████████████████  (unique pattern)
Pos 1: ██████  ████      (different pattern)
Pos 2:   ████████  ██    (another unique pattern)
Pos 3: ██  ████    ████  (each position = unique fingerprint)
```

**Example for position 1 ("cat")**:
```
PE vector for "cat" (position 1) = [0.84, 0.54, 0.01, 1.0, ...]
                                     ↑     ↑     ↑     ↑
                                   dim0  dim1  dim2  dim3  (continues to dim511)
```

### Why Sine/Cosine?

1. **Unique patterns**: Each position gets a distinct encoding
2. **Relative positions**: `Model can learn "distance" between words`
3. **Generalization**: Works for sequences longer than training data
4. **Smooth**: Nearby positions have similar encodings

### Complete Picture (Sinusoidal PE)

```
INPUT: "The cat sat"
   ↓
Word Embeddings:     [E_The] [E_cat] [E_sat]
                        +       +       +
Positional Encodings: [PE_0]  [PE_1]  [PE_2]
   ↓
Final Input:         [E_The+PE_0] [E_cat+PE_1] [E_sat+PE_2]
   ↓
Now transformer knows WHAT each word is AND WHERE it appears!
   ↓
Feed to Attention Mechanism...
```

---

## RoPE (Rotary Position Embedding)

**Used in**: LLaMA, GPT-NeoX, PaLM, and most modern LLMs

### The Problem with Sinusoidal PE (Absolute Encoding)

**Issue 1: Position added BEFORE attention**
```
Sinusoidal PE:
Word "cat" at position 1:
  Embedding[cat] + PE[1] → [combined vector]
                    ↓
            Goes to attention
                    ↓
Problem: Position info mixed with word meaning from the start
```

**Issue 2: Can't directly measure relative distance**
```
When "sat" (pos 2) looks at "cat" (pos 1):
  ❌ Model doesn't naturally know they are 1 position apart
  ❌ Has to LEARN this from the added position values
  ❌ Harder to extrapolate to longer sequences
```

### What is RoPE?

**Purpose**: Encode **relative** positions directly into the attention mechanism.

**Key Idea**: Instead of adding position to embeddings, **rotate** the Q and K vectors based on their positions.

```
Sinusoidal PE:
  Q = Embedding + Position  (addition - mixes position with meaning)

RoPE:
  Q = Rotate(Embedding, position)  (rotation - preserves both separately)
```

### Why Rotation Instead of Addition?

**Mathematical Magic**: When you rotate two vectors by different angles and compute their dot product, the result **automatically** depends on the angle difference!

```
Simple analogy - Clock hands:

Sinusoidal PE (Addition):
  Hour hand at 3 o'clock: Add +3 to the hand
  Hour hand at 5 o'clock: Add +5 to the hand
  Distance between them? Have to calculate: 5 - 3 = 2

RoPE (Rotation):
  Hour hand at 3 o'clock: Rotate by 90°
  Hour hand at 5 o'clock: Rotate by 150°
  Distance between them? Automatically in the angle: 150° - 90° = 60°

The angle DIFFERENCE is built into the rotation!
```

### Why RoPE?

**What it solves:**
1. **Relative distances**: Attention scores naturally depend on distance between tokens
2. **Better extrapolation**: Can handle sequences longer than training length
3. **Decaying attention**: Farther tokens get naturally lower attention
4. **No position limit**: No fixed maximum sequence length

### How RoPE Works

**Core Concept**: Rotate Q and K vectors by angles proportional to their positions.

```
For tokens at position m and n:

Q_m = Rotate(Q, angle = m × θ)
K_n = Rotate(K, angle = n × θ)

When computing attention:
Q_m · K_n depends on (m - n)  ← Relative distance!
```

### Visual Comparison

```
Sinusoidal PE (Absolute):
┌─────────────────────────────┐
│ Position 0: Add [0.0, 1.0]  │
│ Position 1: Add [0.84, 0.54]│
│ Position 2: Add [0.91,-0.42]│
└─────────────────────────────┘
Problem: Same PE regardless of context

RoPE (Relative):
┌─────────────────────────────┐
│ Token at pos 3 looking at:  │
│   pos 1: distance = -2      │
│   pos 2: distance = -1      │
│   pos 4: distance = +1      │
└─────────────────────────────┘
Benefit: Attention knows relative distances
```

### Concrete Example: Sinusoidal vs RoPE

**Scenario**: "The cat sat on mat" - when "sat" (pos 2) attends to other words

#### Sinusoidal PE (Absolute):

```
Step 1: Add position to embeddings BEFORE attention
  "The" (pos 0): E_The + PE_0 = [combined_0]
  "cat" (pos 1): E_cat + PE_1 = [combined_1]
  "sat" (pos 2): E_sat + PE_2 = [combined_2]

Step 2: Compute attention
  Q_sat · K_The = ?  ← Distance not directly encoded
  Q_sat · K_cat = ?  ← Model has to LEARN distance from PE values

Problem:
  - Position 0 and Position 1 have DIFFERENT PE vectors
  - But their DISTANCE (1 apart) is not directly visible
  - Model needs many layers to learn "PE_1 - PE_0 = distance 1"
```

#### RoPE (Relative):

```
Step 1: Rotate Q and K by their positions
  Q_sat rotated by: 2×θ (angle = 2θ)
  K_The rotated by: 0×θ (angle = 0θ)
  K_cat rotated by: 1×θ (angle = 1θ)
  K_on  rotated by: 3×θ (angle = 3θ)

Step 2: Compute attention (dot product)
  Q_sat · K_The → Depends on (2θ - 0θ) = 2θ  ← Distance = 2!
  Q_sat · K_cat → Depends on (2θ - 1θ) = 1θ  ← Distance = 1!
  Q_sat · K_on  → Depends on (2θ - 3θ) = -1θ ← Distance = -1!

✓ Advantage:
  - Relative distance AUTOMATICALLY encoded in dot product
  - No need to learn distance from position values
  - Works better for unseen sequence lengths
```

### Visual Comparison

```
Sinusoidal PE:
┌────────────────────────────────────────┐
│ "The cat sat"                          │
│  pos0 pos1 pos2                        │
│   ↓    ↓    ↓                          │
│  +PE0 +PE1 +PE2  (addition first)      │
│   ↓    ↓    ↓                          │
│  [mix][mix][mix] (position + meaning)  │
│   ↓    ↓    ↓                          │
│  Attention computes Q·K                │
│  Distance? Not directly visible        │
└────────────────────────────────────────┘

RoPE:
┌────────────────────────────────────────┐
│ "The cat sat"                          │
│  pos0 pos1 pos2                        │
│   ↓    ↓    ↓                          │
│  ∠0°  ∠θ   ∠2θ  (rotation by position) │
│   ↓    ↓    ↓                          │
│  [E0] [E1] [E2] (meaning preserved)    │
│   ↓    ↓    ↓                          │
│  Q·K automatically knows distance      │
│  pos2 - pos0 = 2θ angle difference ✓   │
└────────────────────────────────────────┘
```

### Why Modern LLMs Use RoPE

| Feature | Sinusoidal PE | RoPE |
|---------|---------------|------|
| **Position info** | Absolute | Relative |
| **Extrapolation** | Poor | Excellent |
| **Long sequences** | Struggles | Handles well |
| **Attention scores** | Indirect | Direct relative distance |
| **Used in** | Original Transformer, BERT | LLaMA, GPT-NeoX, PaLM |

### Real-World Benefits

**Example: Trained on 2K tokens, used on 8K tokens**

```
Sinusoidal PE:
  Training: Saw positions 0-2047
  Inference: Position 5000 uses sin(5000/10000^x)
  ❌ Problem: Position 5000 NEVER seen during training
  ❌ Result: Model confused, quality drops significantly

RoPE:
  Training: Learned "distance 1", "distance 2", "distance 100"
  Inference: Position 5000 looking at position 4999
  ✓ Distance = 1 (SAME as training!)
  ✓ Distance = 100 (SAME as training!)
  ✓ Result: Works well because relative distances are familiar
```

**Why this works:**

```
Sinusoidal learns: "Position 0 looks like [0.0, 1.0, ...]"
                   "Position 512 looks like [0.5, 0.8, ...]"
                   "Position 5000 looks like ???? ← UNKNOWN"

RoPE learns:       "Distance 1 = rotate by 1θ"
                   "Distance 10 = rotate by 10θ"
                   "Distance 1 at position 5000? Same as distance 1 at position 0!"
                   ✓ KNOWN pattern!
```

### Key Takeaways (RoPE)

1. **Rotation, not addition**: Positions encoded as rotations in Q/K space
2. **Relative by design**: Attention naturally sees token distances
3. **Better for long contexts**: Works beyond training sequence lengths
4. **Standard in modern LLMs**: Default choice for new models
5. **Mathematical elegance**: Angle difference automatically encodes distance

### Analogy

**Sinusoidal PE**: Like giving everyone a name tag with their seat number (absolute)
- `"I'm seat 5", "I'm seat 6"`

**RoPE**: Like everyone knowing their relative position to others (relative)
- `"You're 2 seats to my left", "You're 1 seat to my right"`

More flexible and natural for understanding relationships!

---

### RoPE Scaling: Extending Context Length

**Problem**: Pre-trained model with 4K max length → Want to use 32K tokens

**The Issue**:
```
Model trained on positions 0-4095
Using position 20000 at inference → Out-of-distribution!
Result: Degraded performance, hallucinations ❌
```

**Solution: RoPE Scaling**

Scale position values to compress longer sequences into the training range:

```python
# Linear scaling (simplest)
scaled_position = actual_position / scale_factor

# Example: Extend 4K → 32K (scale_factor = 8)
position 32000 / 8 = position 4000 ✓ (within training range)
```

**Why Fine-tuning is Required**:

```
Just RoPE scaling alone:
├─ Positions compressed mathematically
├─ BUT: Model weights never trained on scaled positions
└─ Result: Poor quality ❌

RoPE scaling + Fine-tuning:
├─ Scale positions (factor = 8 for 4K→32K)
├─ Fine-tune on 3-5K long documents (8K-32K tokens)
├─ Model learns to work with compressed positions
└─ Result: Good quality ✅
```

**Common Scaling Methods**:

| Method | How it works | Quality |
|--------|--------------|---------|
| **Linear** | Divide all positions by factor | Good |
| **NTK-aware** | Scale different frequencies differently | Better |
| **YaRN** | Sophisticated multi-frequency scaling | Best |

**Example: Llama-2 Extension**:
```python
# Extend from 4K → 32K context
config = LlamaConfig(
    max_position_embeddings=32768,
    rope_scaling={"type": "linear", "factor": 8.0}
)
# Then fine-tune on long documents for 1K-3K steps
```

**Benefits**:
- 100x cheaper than pre-training from scratch
- Extends context 4x-8x efficiently
- Preserves base model capabilities

**Use Cases**: Long document QA, book analysis, large codebase understanding

---

## T5 Relative Position Bias

**Used in**: T5, FLAN-T5, UL2

### What is T5 Bias?

**Purpose**: Add learnable biases directly to attention scores based on relative distance.

**Key Idea**: Instead of modifying embeddings, add a bias term to the attention matrix.

```
Standard Attention:
  Attention = softmax(Q × K^T / √d)

T5 Attention:
  Attention = softmax(Q × K^T / √d + Bias)
              where Bias depends on distance between tokens
```

### How T5 Bias Works

```
Token distances are bucketed and learned:

Distance 0 (self):     Bias = learned_bias[0]
Distance 1:            Bias = learned_bias[1]
Distance 2-3:          Bias = learned_bias[2]  (bucketed)
Distance 4-7:          Bias = learned_bias[3]
...
Very far (>128):       Bias = learned_bias[max_bucket]

These biases are LEARNED during training!
```

### Example

```
Sentence: "The cat sat"

Attention from "cat" to:
  "The" (distance -1) → Add bias_bucket_1
  "cat" (distance 0)  → Add bias_bucket_0
  "sat" (distance +1) → Add bias_bucket_1

Model learns optimal bias values through training!
```

### Why T5 Bias?

**Advantages**:
- Simple and effective
- No positional embeddings needed
- Learns task-specific position importance
- Works well for various sequence lengths

---

## ALiBi (Attention with Linear Biases)

**Used in**: BLOOM, MPT, Falcon (some variants)

### What is ALiBi?

**Purpose**: Add simple linear penalties to attention scores based on distance.

**Key Idea**: Farther tokens get linearly lower attention scores - no learning required!

```
Standard Attention:
  Attention = softmax(Q × K^T)

ALiBi Attention:
  Attention = softmax(Q × K^T - distance × slope)
              slope is fixed per attention head
```

### How ALiBi Works

**Core concept**: Penalize distant tokens with a linear slope.

```
For each attention head, use a fixed slope:

Head 1 (slope = 0.5):
  Distance 0: penalty = 0.0
  Distance 1: penalty = -0.5
  Distance 2: penalty = -1.0
  Distance 3: penalty = -1.5

Head 2 (slope = 0.25):
  Distance 0: penalty = 0.0
  Distance 1: penalty = -0.25
  Distance 2: penalty = -0.50
  Distance 3: penalty = -0.75

Different heads use different slopes!
```

### Visual Example

```
Sentence: "The cat sat on mat"

When "sat" attends to others (using slope = 1.0):

Token    Distance   Penalty   Attention Score
─────────────────────────────────────────────
"The"      -2       -2.0      Q·K - 2.0
"cat"      -1       -1.0      Q·K - 1.0
"sat"       0        0.0      Q·K - 0.0  ← No penalty
"on"       +1       -1.0      Q·K - 1.0
"mat"      +2       -2.0      Q·K - 2.0

Farther words get lower scores automatically!
```

### Key Clarification: ALiBi vs RoPE

ALiBi is often confused with RoPE. They touch **completely different parts** of the pipeline:

```
Sinusoidal:  add position to input embeddings   → modifies X
RoPE:        rotate Q and K vectors             → modifies Q, K before dot product
ALiBi:       subtract penalty from score matrix → modifies scores AFTER Q·Kᵀ
```

In ALiBi, **Q and K are never touched** — the raw attention score matrix gets a distance penalty applied on top.

### Bias Matrix: What It Looks Like

For a 5-token sequence with slope = 1.0:

```
         tok0  tok1  tok2  tok3  tok4
tok0  [   0    -1    -2    -3    -4  ]
tok1  [   0     0    -1    -2    -3  ]
tok2  [   0     0     0    -1    -2  ]
tok3  [   0     0     0     0    -1  ]
tok4  [   0     0     0     0     0  ]

formula: bias[i][j] = -slope × |i - j|
```

This matrix is **added to** `Q·Kᵀ / √d` before softmax.
Farther apart → more negative → softmax gives less attention weight automatically.

### Simple Analogy

> **Sinusoidal** — stamp each word's name tag before the meeting starts
>
> **RoPE** — rotate how each person faces when talking (affects who they address)
>
> **ALiBi** — after everyone speaks, penalize conversations between people sitting far apart

All three teach the model about position — but at completely different stages.

---

### Why ALiBi?

**Advantages**:
1. **No learned parameters**: Just fixed slopes
2. **Extreme extrapolation**: Trained on 512 tokens, works on 10K+ tokens
3. **Simple**: One line of code to implement
4. **Efficient**: No extra computation beyond a subtraction

**How slopes are chosen**:
```
For 8 heads: slopes = [1/2^1, 1/2^2, 1/2^3, ..., 1/2^8]
           = [0.5, 0.25, 0.125, 0.0625, ...]

Different heads have different distance sensitivities!
```

---

## Comparison: All Positional Encoding Methods

| Method | Type | Where Applied | Parameters | Extrapolation | Used In |
|--------|------|---------------|------------|---------------|---------|
| **Sinusoidal** | Absolute | Add to embeddings | None (fixed) | Poor | BERT, Original Transformer |
| **RoPE** | Relative | Rotate Q/K | None (fixed) | Good | LLaMA, GPT-NeoX, PaLM |
| **T5 Bias** | Relative | Add to attention | Learned | Good | T5, FLAN-T5 |
| **ALiBi** | Relative | Subtract from attention | None (fixed slopes) | Excellent | BLOOM, MPT, Falcon |

### Quick Guide

```
Choose Sinusoidal if:
  - Using original Transformer/BERT architecture
  - Fixed sequence lengths

Choose RoPE if:
  - Building modern LLM (most popular choice)
  - Need good long-context performance

Choose T5 Bias if:
  - Want model to learn position importance
  - Encoder-decoder architecture

Choose ALiBi if:
  - Need extreme length extrapolation
  - Want simplest implementation
  - Don't want any position parameters
```

---

## Multi-Head Attention: Looking at Words from Different Perspectives

**The Problem**: Single attention focuses on one type of relationship at a time.
- Can't simultaneously capture "what is the subject doing?" AND "where is it happening?"

**The Solution**: Run multiple attention mechanisms in parallel (multiple "heads").

### How It Works

**Why?** Different heads learn different types of relationships (syntax, semantics, position).
**What?** Split into multiple attention operations, then combine results.

```
Instead of 1 attention:
   Input → Attention → Output

Use 8 heads (typical):
  Input → Head 1 (syntax relationships)    ─┐
        → Head 2 (semantic meaning)         │
        → Head 3 (positional context)       ├→ Concat → Linear → Output
        → Head 4 (subject-verb links)       │
        → ...                               │
        → Head 8 (long-range dependencies) ─┘
```

### Visual Example: "The cat sat on the mat"

```
Single-Head Attention:
  cat → sat, mat  (captures one relationship pattern)

Multi-Head Attention (8 heads):

Head 1 (Subject-Verb):     Head 2 (Object-Location):
  cat → sat (90%)            mat → on (80%)

Head 3 (Adjective-Noun):   Head 4 (Long-range):
  The → cat (70%)            The → mat (60%)

Head 5-8: Other patterns...

Combined: Rich understanding from all perspectives!
```

### Architecture Diagram

**What does Q₁, K₁, V₁ mean?**
- The subscript (1, 2, 3...) indicates **which head** it belongs to
- **NOT** which word it processes
- All heads receive the SAME input, but use DIFFERENT learned weights

```
Notation:
Q₁ = Query for Head 1  (learned transformation W_Q₁)
K₁ = Key for Head 1    (learned transformation W_K₁)
V₁ = Value for Head 1  (learned transformation W_V₁)

Q₂ = Query for Head 2  (DIFFERENT learned transformation W_Q₂)
K₂ = Key for Head 2    (DIFFERENT learned transformation W_K₂)
V₂ = Value for Head 2  (DIFFERENT learned transformation W_V₂)

...and so on for all 8 heads
```

```
INPUT: "The cat sat on the mat" (with positional encoding)
   │
   │ SAME input goes to ALL heads
   │
   ├──────────────────────────────────────────────────-┐
   │                                                   │
   ▼                 ▼                 ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Head 1   │    │ Head 2   │    │ Head 3   │ ...│ Head 8   │
│          │    │          │    │          │    │          │
│ Q₁ K₁ V₁ │    │ Q₂ K₂ V₂ │    │ Q₃ K₃ V₃ │    │ Q₈ K₈ V₈ │
│  (W_Q₁)  │    │  (W_Q₂)  │    │  (W_Q₃)  │    │  (W_Q₈)  │
│    ↓     │    │    ↓     │    │    ↓     │    │    ↓     │
│ Attn₁    │    │ Attn₂    │    │ Attn₃    │    │ Attn₈    │
└────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │               │
     └───────────────┴───────────────┴───────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Concatenate │  [64 + 64 + 64 + ... + 64 = 512]
                   └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Linear    │  (combine information)
                   └─────────────┘
                          │
                          ▼
                     OUTPUT
```

**Key Insight:**
```
V₁ doesn't mean "Value of word 1"
V₁ means "Value transformation by Head 1"

For word "cat":
  Head 1 creates: Q₁(cat), K₁(cat), V₁(cat) using W_Q₁, W_K₁, W_V₁
  Head 2 creates: Q₂(cat), K₂(cat), V₂(cat) using W_Q₂, W_K₂, W_V₂
  ...
  Head 8 creates: Q₈(cat), K₈(cat), V₈(cat) using W_Q₈, W_K₈, W_V₈

Same word, 8 different transformations!
```

### Splitting Dimensions

**How are heads created?** Each head gets its own learned Q, K, V transformations.

**Why 512 dimensions?** It's a design choice (hyperparameter):
- GPT-2 small: 768 dims
- BERT base: 768 dims
- Original Transformer: 512 dims
- Larger = more capacity, but slower and needs more data

```
Original embedding: 512 dimensions (design choice)
8 heads: 512 ÷ 8 = 64 dimensions per head

Example for word "cat":
┌─────────────────────────────────────┐
│ Full vector: [512 dimensions]       │
└─────────────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │  Each head has    │
    │  its OWN learned  │
    │  W_Q, W_K, W_V    │
    └───────────────────┘
              ↓
         Transform to 64 dims each
              ↓
┌───────┐ ┌───────┐ ┌───────┐   ┌───────┐
│ 64dim │ │ 64dim │ │ 64dim │...│ 64dim │
│Q₁K₁V₁ │ │Q₂K₂V₂ │ │Q₃K₃V₃ │   │Q₈K₈V₈ │
│Head 1 │ │Head 2 │ │Head 3 │   │Head 8 │
└───────┘ └───────┘ └───────┘   └───────┘

Each head processes independently with different weights
```

### Why Each Head Has Its Own Q, K, V

**Key Point**: H1 doesn't "focus on V1" by position - each head learns **different transformation weights**.

```
Input (512 dim) goes to ALL heads:

Head 1:                          Head 2:
Input [512] → W_Q₁ → Q₁ [64]     Input [512] → W_Q₂ → Q₂ [64]
Input [512] → W_K₁ → K₁ [64]     Input [512] → W_K₂ → K₂ [64]
Input [512] → W_V₁ → V₁ [64]     Input [512] → W_V₂ → V₂ [64]

Different W matrices = Different patterns learned!
```

**Why this matters:**

```
Same input "cat" [512 dims] processed by different heads:

Head 1 (W_Q₁, W_K₁, W_V₁):
  Learns to focus on: SYNTAX (subject-verb relationships)
  "cat" → pays attention to "sat" (verb)

Head 2 (W_Q₂, W_K₂, W_V₂):
  Learns to focus on: LOCATION (spatial relationships)
  "cat" → pays attention to "mat" (where)

Head 3 (W_Q₃, W_K₃, W_V₃):
  Learns to focus on: ARTICLES (determiners)
  "cat" → pays attention to "The"

Each head sees the SAME input but through DIFFERENT "lenses"!
```

**Concrete Example:**

```
Word "cat" [512 dimensions]
   ↓
Head 1 transforms:
  Q₁ = [512] × W_Q₁[512×64] = [64]  ← Learns "syntax questions"
  K₁ = [512] × W_K₁[512×64] = [64]  ← Learns "syntax keys"
  V₁ = [512] × W_V₁[512×64] = [64]  ← Learns "syntax values"

Head 2 transforms:
  Q₂ = [512] × W_Q₂[512×64] = [64]  ← Learns "semantic questions"
  K₂ = [512] × W_K₂[512×64] = [64]  ← Learns "semantic keys"
  V₂ = [512] × W_V₂[512×64] = [64]  ← Learns "semantic values"

Different W matrices → Different learned behaviors!
```

### What Each Head Learns

```
Example attention patterns in 8 heads:

Head 1: Subject → Verb
  The [cat] → [sat] ████████

Head 2: Verb → Object
  [sat] → [mat] ██████

Head 3: Preposition → Object
  [on] → [mat] █████

Head 4: Article → Noun
  [The] → [cat], [the] → [mat] ███

Head 5: Position-based (nearby words)
  Each word → neighbors ████

Head 6-8: Abstract patterns learned from data
```

### Complete Flow

```
INPUT: [The, cat, sat, on, the, mat]
   ↓
Multi-Head Attention:
   ├─ Head 1 learns: syntax structure
   ├─ Head 2 learns: semantic relationships
   ├─ Head 3 learns: positional patterns
   └─ ... (parallel processing)
   ↓
Concatenate all head outputs
   ↓
Linear transformation (mix information)
   ↓
OUTPUT: Each word has context from MULTIPLE perspectives
```

### Why Multi-Head Works

1. **Diverse perspectives**: Each head specializes in different patterns
2. **Redundancy**: If one head fails, others compensate
3. **Richer representations**: Captures multiple relationship types simultaneously
4. **Parallelizable**: All heads run at the same time (fast!)

### Analogy

Reading a book with multiple experts:
- **Single attention**: One expert reads and explains
- **Multi-head**: 8 experts read simultaneously
  - Expert 1: Grammar analysis
  - Expert 2: Character relationships
  - Expert 3: Plot structure
  - Expert 4: Themes and motifs
  - Experts 5-8: Other aspects

You get a richer, more complete understanding!

---

## How Multi-Head Attention Learns: Backpropagation

**The Question**: How do W_Q₁, W_K₁, W_V₁ (and all other W matrices) learn to specialize?

**The Answer**: Backpropagation - gradients flow back from the loss to update each head's weights.

### The Learning Process (Training)

**Why?** Need to adjust weights so predictions match actual outputs.
**What?** Compute error, send it backwards, update weights to reduce error.

```
FORWARD PASS (Prediction):
Input → Multi-Head Attention → Output → Loss (error)
  ↓            ↓                 ↓         ↓
"cat"    8 heads process      "sat"?   How wrong?

BACKWARD PASS (Learning):
Loss → Gradients → Update Weights ← Each head gets feedback
  ↓         ↓            ↓
Error  Flow back   Adjust W matrices
```

### Detailed Backpropagation Flow

```
FORWARD PASS:
────────────────────────────────────────────────
Input: "The cat sat"
   ↓
   ├────────┬────────┬────────┐
   ▼        ▼        ▼        ▼
 Head1    Head2    Head3 ... Head8
 W_Q₁     W_Q₂     W_Q₃      W_Q₈  (current weights)
 W_K₁     W_K₂     W_K₃      W_K₈
 W_V₁     W_V₂     W_V₃      W_V₈
   ↓        ↓        ↓        ↓
  Out₁    Out₂    Out₃     Out₈
   └────────┴────────┴────────┘
              ↓
         Concatenate
              ↓
            Linear
              ↓
         Prediction: "mat"
         Actual:     "sat"
              ↓
         Loss = 0.8 (high error!)


BACKWARD PASS:
────────────────────────────────────────────────
         Loss = 0.8
              ↓
       ∂Loss/∂Output = gradient
              ↓
         Linear (backprop)
              ↓
         Split gradients
              ↓
   ├────────┬────────┬────────┐
   ▼        ▼        ▼        ▼
∂Loss/∂W_Q₁ ∂L/∂W_Q₂ ∂L/∂W_Q₃  ∂L/∂W_Q₈
∂Loss/∂W_K₁ ∂L/∂W_K₂ ∂L/∂W_K₃  ∂L/∂W_K₈
∂Loss/∂W_V₁ ∂L/∂W_V₂ ∂L/∂W_V₃  ∂L/∂W_V₈
   ↓        ↓        ↓        ↓
UPDATE WEIGHTS:
W_Q₁ = W_Q₁ - learning_rate × ∂Loss/∂W_Q₁
W_K₁ = W_K₁ - learning_rate × ∂Loss/∂W_K₁
W_V₁ = W_V₁ - learning_rate × ∂Loss/∂W_V₁
(Same for all heads)
```

### Simple Example: Learning Step by Step

```
Example: Model predicts wrong word
─────────────────────────────────

Sentence: "The cat ___ on the mat"
Correct:  "sat"
Predicted: "mat" (WRONG!)

Step 1: Calculate Loss
  Loss = |Predicted - Actual|² = high error

Step 2: Which head contributed to the error?
  Head 1 output: 0.3 → small contribution
  Head 2 output: 0.1 → small contribution
  Head 3 output: 0.8 → LARGE contribution (main problem!)
  ...

Step 3: Compute gradients for each head
  ∂Loss/∂W_Q₃ = large (Head 3 needs big update)
  ∂Loss/∂W_Q₁ = small (Head 1 is mostly fine)

Step 4: Update weights
  W_Q₃ = W_Q₃ - 0.01 × (large gradient)  ← Big change
  W_Q₁ = W_Q₁ - 0.01 × (small gradient) ← Small change
```

### Weight Update Formula

```
For each weight matrix in each head:

W_new = W_old - learning_rate × gradient

Example for Head 1's Query matrix:
W_Q₁[new] = W_Q₁[old] - α × ∂Loss/∂W_Q₁

Where:
- α (alpha) = learning_rate (e.g., 0.001)
- ∂Loss/∂W_Q₁ = how much W_Q₁ contributed to error
```

### How Heads Specialize During Training

```
Initially (random weights):
─────────────────────────────
Head 1: random patterns
Head 2: random patterns
Head 3: random patterns
...all heads look similar

After 1000 updates:
─────────────────────────────
Head 1: Starting to focus on nearby words
Head 2: Starting to focus on verbs
Head 3: Still random
...gradual specialization

After 100,000 updates:
─────────────────────────────
Head 1: SPECIALIZED in syntax (subject-verb)
Head 2: SPECIALIZED in semantics (word meaning)
Head 3: SPECIALIZED in position (word order)
...each head found its "role"
```

### Why Different Heads Learn Different Patterns

**Key Insight**: Random initialization + different gradients = specialization

```
Iteration 1:
  W_Q₁ = [random values A]  ← Slightly different
  W_Q₂ = [random values B]  ← Slightly different
         ↓
     Process data
         ↓
  Head 1 gets gradient for syntax
  Head 2 gets gradient for semantics
         ↓
     Update weights
         ↓
  W_Q₁ adjusted toward syntax
  W_Q₂ adjusted toward semantics
         ↓
  Difference AMPLIFIES over time!

Iteration 100,000:
  W_Q₁ = [syntax specialist]
  W_Q₂ = [semantic specialist]
```

### Gradient Flow Diagram

```
                 LOSS
                  │
                  ▼
         ┌────────────────┐
         │   ∂Loss/∂Out   │ (gradient from output)
         └────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Linear Layer  │
         │   (backprop)   │
         └────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   ┌─────────┐         ┌─────────┐
   │ Head 1  │         │ Head 2  │
   │Gradient │         │Gradient │
   └─────────┘         └─────────┘
        │                   │
        ▼                   ▼
   Update W_Q₁         Update W_Q₂
   Update W_K₁         Update W_K₂
   Update W_V₁         Update W_V₂

Each head receives its OWN gradient and updates independently!
```

### Analogy: Learning Like Students

```
8 Students (Heads) learning to analyze sentences:

Day 1 (Random):
  All students guess randomly
  Teacher gives feedback (gradients) to each

Day 100:
  Student 1: Getting better at grammar
  Student 2: Getting better at meaning
  Student 3: Still confused
  (Each gets different feedback based on their mistakes)

Day 10,000:
  Student 1: Expert in grammar (W_Q₁ specialized)
  Student 2: Expert in meaning (W_Q₂ specialized)
  Student 3: Expert in word order (W_Q₃ specialized)

WHY? Each started randomly different, received different
feedback, and specialized over time!
```

### Key Points

1. **Each head has separate weights** (W_Q, W_K, W_V) that update independently
2. **Gradients flow back** from loss through each head differently
3. **Random initialization** + different gradients → specialization
4. **Thousands of updates** → heads become experts in different patterns
5. **No manual assignment** - heads discover their roles automatically!

---

## Transformer Block Internals: MHA, FFN, and Loss Computation

### Component Overview

| **Component** | **Purpose** | **Input Shape** | **Output Shape** | **Computes Loss?** |
|---------------|-------------|-----------------|------------------|--------------------|
| **MHA** | Gathers context from related tokens | `[n × d]` | `[n × d]` | ❌ No |
| **FFN** | Transforms each token individually | `[n × d]` | `[n × d]` | ❌ No |
| **Final Linear** | Projects to vocabulary space | `[n × d]` | `[n × vocab_size]` | ❌ No |
| **Loss Layer** | Compares prediction with true label | `[vocab_size]` | Scalar | ✅ **YES** |

---

### 1. Attention Scores: How Words Find Related Words

**Question**: Do related words like "cat" and "sat" have higher dot product values?

**Answer**: ✅ **Yes!** The model learns to give higher scores to related words.

**Visual Example: "The cat sat on the mat"**

```
Attention scores when "cat" looks at other words:

        The    cat    sat     on    the    mat
        ▁▁    ████   █████    ▁     ▁     ▁
        10%    30%    40%    10%    5%    5%
              (self) (HIGH!)
                      ↑
              Strong subject-verb relationship
```

**How it works:**

```
Step 1: QK^T                Step 2: Softmax           Step 3: Mix with V
─────────────────           ────────────────          ─────────────────
"cat" · "sat"              Raw → Probabilities        Weighted average
= HIGH score               40% to "sat"               = 40% × V(sat)
                                                        + 30% × V(cat)
"cat" · "on"               10% to "on"                + 10% × V(on)
= LOW score                                           + ...
```

---

### 2. Shape Flow Through Transformer

**Question**: Does MHA output `n×1` (one value per token)?

**Answer**: ❌ **No!** MHA outputs `n × d` (rich vector per token).

**Why `d` dimensions instead of 1?**

```
If output was [n × 1]:              With [n × d]:
─────────────────────               ──────────────
"cat" → 0.5  (single number)        "cat" → [0.2, 0.8, -0.3, ...] (768 values)
           ↓                                    ↓
    ❌ Lost all meaning!                   ✅ Rich representation:
                                              - Semantics
                                              - Syntax
                                              - Context
```

**Complete Shape Flow (3 tokens, 768 dims):**

```
Input:      [3 × 768]  "The cat sat"
    ↓
MHA:        [3 × 768]  (context-aware)
    ↓
Add&Norm:   [3 × 768]  (residual + normalize)
    ↓
FFN:        [3 × 768]  (individual transformation)
    ↓
Add&Norm:   [3 × 768]  (residual + normalize)
    ↓
Next Layer or Output
```

---

### 3. FFN Purpose & Loss Computation

**Question**: Does FFN compute loss?

**Answer**: ❌ **No!** FFN is just an intermediate transformation. Loss computed **once** at the end.

**What FFN Does (Expand → Compress):**

```
FFN Architecture:

[n × d=768]  Input
     ↓
  Linear + ReLU
     ↓
[n × 4d=3072]  ← Expansion (more capacity)
     ↓
  Linear
     ↓
[n × d=768]  Output (back to original size)
```

**Formula**: `FFN(x) = Linear₂(ReLU(Linear₁(x)))`

---

#### FFN Explained for Beginners

**One-line summary:**
> MHA lets tokens *talk to each other*. FFN lets each token *think about what it heard*.

**What happens inside FFN:**

```
MHA output (token "bank" now knows river is nearby)
      ↓
FFN Hidden Layer (2048 neurons / perceptrons)
      ↓  each neuron asks one yes/no question:
         Neuron #89:   "Is water nearby?"        → 2.3  fires!
         Neuron #4:    "Is this financial?"       → 0.0  silent
         Neuron #1203: "Is there a landscape?"   → 3.1  fires!
      ↓
Only ~2% of neurons fire (ReLU kills negatives)
      ↓
Compress back → "river bank" features confirmed
```

**Is the hidden layer (2048) just neurons with weights?**
Yes. Each neuron holds a row of pre-trained weights `W1` (512 numbers).
It computes: `score = dot(W1_row, input) + bias`
- High score → ReLU fires → pattern matched
- Low/negative → ReLU outputs 0 → ignored

**How did neurons learn "Is this financial?"?**
Nobody labeled them. During pretraining on millions of sentences, backpropagation
adjusted weights so that neurons which recognized useful patterns reduced the loss.
The pattern detectors *emerged* automatically.

**FFN vs MHA — simple analogy:**

| | MHA | FFN |
|---|---|---|
| Analogy | Group discussion — tokens talk | Individual thinking — token reflects |
| Operates on | All tokens together | Each token independently |
| Role | Route & blend information | Transform & store knowledge |

**Key insight:** FFN weights are the model's *long-term memory*.
Lower layers store grammar facts. Upper layers store world knowledge (capitals, people, events).

---

### 4. Where Loss is Computed: End-to-End Flow

**Single Loss Computation at the Very End:**

```
┌─────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "The cat sat on" + Target: "the"                   │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Layer 1: MHA → FFN                  │  [n × d]         │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Layer 2: MHA → FFN                  │  [n × d]         │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│         ... (78 more layers)                                │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Layer 80: MHA → FFN                 │  [n × d]         │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Final Linear Projection             │                  │
│  │  [4 × 768] → [4 × 50000]            │  (vocab size)    │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  Last token "on" → [50000] logits                          │
│     ↓                                                       │
│  Softmax: P("the")=0.35, P("a")=0.25...                    │
│     ↓                                                       │
│  ╔══════════════════════════════════════╗                  │
│  ║  LOSS = -log(0.35) = 1.05           ║  ← ONLY HERE!    │
│  ╚══════════════════════════════════════╝                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   BACKWARD PASS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Loss = 1.05                                                │
│     ↓ (gradients flow backward)                             │
│  ┌──────────────────────────────────────┐                  │
│  │  Final Linear (update weights)       │  ∂Loss/∂W       │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Layer 80: FFN₈₀ + MHA₈₀            │  ∂Loss/∂W₈₀     │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│         ... (backprop through all 80 layers)                │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Layer 2: FFN₂ + MHA₂               │  ∂Loss/∂W₂      │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  ┌──────────────────────────────────────┐                  │
│  │  Layer 1: FFN₁ + MHA₁               │  ∂Loss/∂W₁      │
│  └──────────────────────────────────────┘                  │
│     ↓                                                       │
│  All weights updated to minimize loss                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. Backpropagation: How ALL Layers Learn

**Question**: Is backpropagation performed in all 80 layers' FFNs?

**Answer**: ✅ **YES!** Gradients flow through **EVERY** layer (all 80 FFNs and all 80 MHAs).

**How it Works:**

```
Loss (computed once) → Gradients flow backward to ALL components:

Layer 80:  FFN₈₀ receives gradients → Updates W₈₀
Layer 79:  FFN₇₉ receives gradients → Updates W₇₉
Layer 78:  FFN₇₈ receives gradients → Updates W₇₈
...
Layer 2:   FFN₂  receives gradients → Updates W₂
Layer 1:   FFN₁  receives gradients → Updates W₁

ALL layers learn from the SAME loss!
```

**Gradient Flow Visualization:**

```
                    SINGLE LOSS VALUE
                         (1.05)
                           ↓
            ╔══════════════════════════╗
            ║    Compute ∂Loss/∂Output ║
            ╚══════════════════════════╝
                           ↓
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
    ┌─────────┐                         ┌─────────┐
    │ Layer 80│                         │ Layer 80│
    │  FFN₈₀  │ ← ∂L/∂W₈₀              │  MHA₈₀  │ ← ∂L/∂W₈₀
    └─────────┘                         └─────────┘
         ↓                                   ↓
    ┌─────────┐                         ┌─────────┐
    │ Layer 79│                         │ Layer 79│
    │  FFN₇₉  │ ← ∂L/∂W₇₉              │  MHA₇₉  │ ← ∂L/∂W₇₉
    └─────────┘                         └─────────┘
         ↓                                   ↓
       ...                                  ...
         ↓                                   ↓
    ┌─────────┐                         ┌─────────┐
    │ Layer 1 │                         │ Layer 1 │
    │  FFN₁   │ ← ∂L/∂W₁               │  MHA₁   │ ← ∂L/∂W₁
    └─────────┘                         └─────────┘
         ↓                                   ↓
    Update ALL weights simultaneously
```

**Real Example with Numbers:**

```
Loss = 1.05 (high error - prediction was wrong)
    ↓
Compute gradients for EVERY layer:

Layer 80 FFN: gradient = 0.02  → W₈₀ = W₈₀ - 0.001 × 0.02
Layer 79 FFN: gradient = 0.03  → W₇₉ = W₇₉ - 0.001 × 0.03
Layer 78 FFN: gradient = 0.01  → W₇₈ = W₇₈ - 0.001 × 0.01
...
Layer 2 FFN:  gradient = 0.04  → W₂  = W₂  - 0.001 × 0.04
Layer 1 FFN:  gradient = 0.05  → W₁  = W₁  - 0.001 × 0.05

ALL 80 layers updated based on the SINGLE loss value!
```

**Why This Works:**

```
Chain Rule of Calculus:

∂Loss/∂W₁ = ∂Loss/∂Output × ∂Output/∂Layer₈₀ × ... × ∂Layer₂/∂W₁
            └─────────────────────────────────────────────────┘
              Gradient flows backward through ALL layers
```

---

### Key Takeaways

| Concept | Truth |
|---------|-------|
| **Loss computed in FFN?** | ❌ No - FFN is intermediate transformation |
| **Loss computed where?** | ✅ Once at final output (after all 80 layers) |
| **Backprop in all FFNs?** | ✅ Yes - gradients flow through all 80 FFNs |
| **Backprop in all MHAs?** | ✅ Yes - gradients flow through all 80 MHAs |
| **MHA output shape?** | `[n × d]` - rich vectors, NOT `[n × 1]` |
| **Related words score?** | ✅ Higher attention scores (learned automatically) |

**Mental Model:**
- **Loss**: Computed once at the end (like a final exam score)
- **Backpropagation**: Feedback distributed to ALL layers (like reviewing all your mistakes from the exam)
- **FFN/MHA**: Workers that ALL receive feedback from the final score and improve

---

## Key Takeaways

1. Transformers use **attention** to find which words relate to each other
2. **Q × K^T** computes all pairwise similarities
3. **Softmax** normalizes to percentages
4. **Multiply by V** mixes information based on attention weights
5. **Parallel processing** - all words simultaneously (unlike RNNs)
6. **Positional encoding** injects word order using sine/cosine patterns
7. **Multi-head attention** captures multiple relationship types (syntax, semantics, position) simultaneously
8. **Everything is learned** - model discovers optimal Q, K, V during training

---

## Real-World Analogy

Conference networking:
- **Q**: Your interests
- **K**: Others' name tags (expertise)
- **Q × K^T**: Scan room for relevant people
- **Softmax**: Decide time allocation (%)
- **V**: Their actual knowledge
- **Output**: Weighted knowledge from conversations

This is transformer attention with words!

---

## Layer Normalization

**Purpose**: Stabilize training by normalizing layer outputs to consistent scale (mean=0, std=1).

### The Problem & Solution

```
Without LayerNorm:
  Layer outputs: [100, 0.5, -50, 200]  ← Wild variation, unstable training

With LayerNorm:
  Normalized: [-0.5, 0.2, -1.0, 1.3]  ← Consistent scale, stable training
```

---

### The Formula

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β

Steps:
1. μ (mean) = average of values
2. σ (std) = standard deviation
3. Normalize: (x - μ) / σ  → mean=0, std=1
4. Scale & shift: γ and β (learned parameters)
```

---

### Example Calculation

```
Input: x = [10, 20, 5, 15]

Step 1 - Mean:
  μ = (10+20+5+15)/4 = 12.5

Step 2 - Standard deviation:
  σ = √[((10-12.5)² + (20-12.5)² + (5-12.5)² + (15-12.5)²)/4]
    = √31.25 ≈ 5.59

Step 3 - Normalize:
  x_norm = (x - μ) / σ
         = [(10-12.5)/5.59, (20-12.5)/5.59, (5-12.5)/5.59, (15-12.5)/5.59]
         = [-0.45, 1.34, -1.34, 0.45]  ← mean≈0, std≈1

Step 4 - Scale & Shift (γ=1, β=0 initially):
  output = γ × x_norm + β = [-0.45, 1.34, -1.34, 0.45]
```

---

### Where Used in Transformers

**Two locations per transformer block:**

```
INPUT
  ↓
Multi-Head Attention
  ↓
Add & Normalize  ← LayerNorm 1 (residual + normalize)
  ↓
Feed Forward Network
  ↓
Add & Normalize  ← LayerNorm 2 (residual + normalize)
  ↓
OUTPUT
```

**"Add & Normalize" breakdown:**

```
input = [1, 2, 3]
layer_output = [0.5, 1.0, 1.5]

Step 1 - Residual (Add):
  combined = input + layer_output = [1.5, 3.0, 4.5]

Step 2 - LayerNorm:
  normalized = LayerNorm([1.5, 3.0, 4.5]) → mean=0, std=1
```

---

### Why It Helps

| Problem | Without LayerNorm | With LayerNorm |
|---------|-------------------|----------------|
| **Scale variation** | [0.1] → [100] → [0.001] | [0.1] → [0.5] → [0.3] |
| **Training** | Unstable, slow | Stable, fast |
| **Gradients** | Exploding/vanishing | Consistent flow |
| **Learning rate** | Must be small | Can be larger |

---

### Layer Norm vs Batch Norm

```
Batch Norm (CNNs):
  Sentence 1: [10, 20, 5, 15]
  Sentence 2: [8,  18, 3, 13]
  Sentence 3: [12, 22, 7, 17]
               ↓   ↓   ↓   ↓
  Normalize each column (across batch)

Layer Norm (Transformers):
  Sentence: [10, 20, 5, 15]
            ←──────────────→
  Normalize across features (within example)
```

**Why Transformers use Layer Norm:**
- Works with variable sequence lengths
- No batch size dependency
- Better for sequential data

---

### Visual Example: Full Flow

```
INPUT: [The, cat, sat]
  ↓
Multi-Head Attention → [10, 20, 5]
  ↓
Residual: [10, 20, 5] + input
  ↓
LayerNorm 1 → [-0.8, 1.2, -0.4]  ← Normalized!
  ↓
Feed Forward → [100, 50, 75]
  ↓
Residual: [100, 50, 75] + previous
  ↓
LayerNorm 2 → [0.5, -1.0, 0.5]  ← Normalized!
  ↓
Next Block
```

---

### Analogy

**Standardized test scoring:**
```
Without normalization:
  Math: 0-100, English: 0-50, Science: 0-200  ❌ Different scales

With normalization:
  All tests: mean=0, std=1  ✓ Comparable scores

Layer Norm does the same for neuron outputs!
```

---

### Key Takeaways

1. **Formula**: `γ × (x - mean) / std + β`
2. **Purpose**: Normalize to mean=0, std=1 for stable training
3. **Location**: After attention & feed-forward (2× per block)
4. **Benefits**: Faster training, stable gradients, higher learning rates
5. **Combined with**: Residual connections ("Add & Norm")

---

## Pre-Norm vs Post-Norm: LayerNorm Placement

**The Question**: Should LayerNorm be applied BEFORE (Pre-Norm) or AFTER (Post-Norm) the sublayer?

1st is `Post-Norm` and next is `Pre-Norm`

![alt text](image-2.png)

---

### Post-Norm (Original Transformer, 2017)

**Structure**: LayerNorm applied AFTER adding residual connection.

![alt text](image.png)

```
┌─────────────────────────────────────────────────────────┐
│  POST-NORM ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input (x)                                              │
│    ↓                                                    │
│  Attention(x)                                           │
│    ↓                                                    │
│  Add: x + Attention(x)                                  │
│    ↓                                                    │
│  LayerNorm  ← Applied AFTER Add                         │
│    ↓                                                    │
│  FFN(x)                                                 │
│    ↓                                                    │
│  Add: x + FFN(x)                                        │
│    ↓                                                    │
│  LayerNorm  ← Applied AFTER Add                         │
│    ↓                                                    │
│  Output                                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Formula:**
```
# Post-Norm
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

> **Representation Collapse risk** — Post-Norm normalizes after every residual, compressing token differences at each layer. Combined with vanishing gradients in early layers, deep Post-Norm networks are prone to all tokens converging to similar vectors.

![alt text](image-3.png)

**Structure**: LayerNorm applied BEFORE the sublayer.
```
┌─────────────────────────────────────────────────────────┐
│  PRE-NORM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input (x)                                              │
│    ↓                                                    │
│  LayerNorm  ← Applied BEFORE Attention                  │
│    ↓                                                    │
│  Attention(LayerNorm(x))                                │
│    ↓                                                    │
│  Add: x + Attention(...)  ← Residual outside            │
│    ↓                                                    │
│  LayerNorm  ← Applied BEFORE FFN                        │
│    ↓                                                    │
│  FFN(LayerNorm(x))                                      │
│    ↓                                                    │
│  Add: x + FFN(...)  ← Residual outside                  │
│    ↓                                                    │
│  Output                                                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### Pre-Norm (Modern: GPT-3, GPT-4, LLaMA)

![alt text](image-1.png)


**Formula:**
```
# Pre-Norm
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Post-Norm vs Pre-Norm: Representation Collapse & Stability

| | **Post-Norm** | **Pre-Norm** |
|---|---|---|
| **Representation Collapse** | Higher risk | Lower risk |
| **Why collapse happens** | Normalizes after residual → compresses token differences every layer | Residual stream never normalized → token identity preserved |
| **Gradient flow** | Vanishes in early layers → poor early representations | Clean gradient highway → all layers update well |
| **Training stability** | Unstable in deep nets, needs LR warmup | Stable, no warmup needed |
| **Deep networks (100+ layers)** | Prone to collapse without careful tuning | Preferred — used by GPT-3, LLaMA, PaLM |

> **Root cause of Post-Norm collapse:** Two compounding problems — LayerNorm squashes token differences at every layer + early layers get near-zero gradients → tokens converge to similar vectors across layers.

---
### Pre-Norm vs Post-Norm vs ResiDual

`ResiDual` - Combination of Post-Norm with addition of separate Gradient

ResiDual is a hybrid architecture that combines the benefits of both Post-Norm and Pre-Norm:


**The Key Innovation**
ResiDual adds a THIRD gradient path:

1. Direct path (like Pre-Norm) → Gradient = 1
2. Through LayerNorm (like Pre-Norm) → Secondary gradients
3. Separate gradient branch (NEW!) → Additional learning signal with learnable α

![alt text](image-4.png)
---

### Backpropagation: Why Pre-Norm is Better

**The Core Problem**: During training, gradients (learning signals) must flow backward through all layers. If they get too weak, early layers can't learn.

---

#### The Key Difference

**Post-Norm**: Residual connection is INSIDE LayerNorm
```
y = LayerNorm(x + Attention(x))
    ↑ Add happens first, then LayerNorm
```

**Pre-Norm**: Residual connection is OUTSIDE
```
y = x + Attention(LayerNorm(x))
    ↑ Direct addition, no LayerNorm blocking
```

---

#### Gradient Flow Comparison

> **What is a gradient?**
> During training, the model measures its mistake (loss) and sends a correction signal **backward** through every layer — this signal is the gradient. If it gets too weak, early layers receive almost no correction and stop learning.

---

**POST-NORM — Gradient blocked by LayerNorm:**

```
FORWARD PASS (top → bottom)         BACKWARD PASS (bottom → top)
─────────────────────────────────   ─────────────────────────────────
Input                                         Input
  ↓                                             ↑
Attention(x)                        gradient = 0.05 ❌ too weak!
  ↓                                             ↑
[x + Attention(x)]  ← residual add  gradient shrinks at every LayerNorm
  ↓                                             ↑
LayerNorm           ← bottleneck!   gradient = 0.4  (shrunk here)
  ↓                                             ↑
Output                              gradient = 1.0  (started strong)

Problem: Every layer's gradient MUST pass through LayerNorm.
         LayerNorm re-scales values → shrinks gradient each time.
         After 50 layers, gradient reaching Layer 1 ≈ 0.05 → learns nothing.
```

---

**PRE-NORM — Gradient has a free highway:**

```
FORWARD PASS (top → bottom)
──────────────────────────────────────────────────────
                    Input (x)
                   ↙          ↘
        (Path 2)               (Path 1 — residual copy)
            ↓                          ↓
        LayerNorm                      │  ← x passed directly, no LayerNorm
            ↓                          │
        Attention                      │
            ↓                          ↓
            └──────────→  Add  ←───────┘
                           ↓
                         Output


BACKWARD PASS (bottom → top) — gradients flow via BOTH paths
──────────────────────────────────────────────────────
                    Input (x)
                   ↗          ↖
        (Path 2)               (Path 1 — residual)
            ↑                          ↑
        LayerNorm              gradient = 1.0 always
        (shrinks gradient)     DIRECT — no LayerNorm!
            ↑                          ↑
        Attention                      │
            ↑                          │
            └──────────  Add  ──────────┘
                          ↑
                   ∂L/∂x = 1 (Path 1) + shrunk value (Path 2)
                            ↑
                     "1" guarantees gradient never vanishes!




Result in a 50-layer model:
  Layer 50: gradient = 1.0 + 0.0x = 1.0
  Layer 25: gradient = 1.0 + 0.0x = ~0.8   (Path 1 always contributes 1)
  Layer 1:  gradient = 1.0 + 0.0x = ~0.6 ✅ still strong enough to learn!
```

![alt text](image-1.png)

---

**Result in a 50-layer model:**

```
              Post-Norm    Pre-Norm
              ─────────    ────────
Layer 50:       1.0          1.0
Layer 25:       0.4          0.8
Layer 1:        0.05 ❌      0.6  ✅
```

---

#### Simple Math Explanation

**Post-Norm:**
```
y = LayerNorm(x + f(x))

Gradient calculation:
∂y/∂x = ∂LayerNorm/∂(x+f(x)) × (1 + ∂f/∂x)
        ↑ This term can shrink the gradient!

→ Gradients get weaker through LayerNorm
```

**Pre-Norm:**
```
y = x + f(LayerNorm(x))

Gradient calculation:
∂y/∂x = 1 + ∂f/∂LayerNorm × ∂LayerNorm/∂x
        ↑ Direct "1" term!

→ The "1" guarantees gradient flows through
```

---

#### Concrete Example: 10-Layer Model

**Training step with loss = 2.0**

**Post-Norm:**
```
Layer 10 → gradient = 1.0
  ↓ passes through LayerNorm (reduces by 10%)
Layer 9  → gradient = 0.9
  ↓ passes through LayerNorm
Layer 8  → gradient = 0.81
  ↓
Layer 1  → gradient = 0.35 ❌

Early layers update slowly → Poor training
```

**Pre-Norm:**
```
Layer 10 → gradient = 1.0
  ↓ direct path via residual
Layer 9  → gradient = 0.98 (only 2% reduction)
  ↓ direct path
Layer 8  → gradient = 0.96
  ↓
Layer 1  → gradient = 0.80 ✅

All layers learn well → Good training
```

---

### Comparison Table

| **Aspect** | **Post-Norm** | **Pre-Norm** |
|------------|---------------|--------------|
| **LayerNorm position** | After Add | Before sublayer |
| **Residual position** | Inside LayerNorm | Outside (direct path) |
| **Gradient flow** | Passes through LayerNorm | Direct path via residual |
| **Deep model stability** | Poor (gradients vanish) | Excellent |
| **Training difficulty** | Harder, needs warmup | Easier |
| **Learning rate** | Requires careful tuning | More robust |
| **Used in** | Original Transformer (2017) | GPT-3, GPT-4, LLaMA, modern models |
| **Max stable depth** | ~12-24 layers | 96+ layers |

---

### Why Pre-Norm Enables Deep Models

**The Key Insight**: Residual connection placement

```
Post-Norm problem:
  y = LayerNorm(x + f(x))
      ↑ Residual INSIDE LayerNorm
  → Gradient must pass through LayerNorm derivative
  → Gradient magnitude reduced

Pre-Norm solution:
  y = x + f(LayerNorm(x))
      ↑ Residual OUTSIDE LayerNorm
  → Gradient has direct identity path (∂y/∂x = 1)
  → Gradient magnitude preserved
```

**Analogy:**

```
Post-Norm = Highway with toll booths
  → Each LayerNorm is a toll booth
  → 192 toll booths slow you down ❌

Pre-Norm = Highway with express lane
  → Residual connection is the express lane
  → Direct path to destination ✅
```

---

### When to Use Each

| **Use Post-Norm if:** | **Use Pre-Norm if:** |
|----------------------|---------------------|
| Small models (≤12 layers) | Deep models (24+ layers) |
| Research/experimentation | Production deployment |
| You need exact Transformer replication | You want modern best practices |
| Smaller learning rates acceptable | You want training stability |

**Modern recommendation**: **Always use Pre-Norm** unless you have a specific reason not to.

---

### Code Example

```python
# Post-Norm
class PostNormBlock(nn.Module):
    def forward(self, x):
        # Attention sublayer
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)  # Norm AFTER add

        # FFN sublayer
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)   # Norm AFTER add
        return x

# Pre-Norm
class PreNormBlock(nn.Module):
    def forward(self, x):
        # Attention sublayer
        x = x + self.attention(self.norm1(x))  # Norm BEFORE, add AFTER

        # FFN sublayer
        x = x + self.ffn(self.norm2(x))         # Norm BEFORE, add AFTER
        return x
```

---

### Key Takeaways

1. **Post-Norm**: LayerNorm AFTER Add → Gradients pass through LayerNorm → Weak gradients
2. **Pre-Norm**: LayerNorm BEFORE sublayer → Direct gradient path → Strong gradients
3. **Gradient formula**:
   - Post-Norm: Must pass through LayerNorm derivative
   - Pre-Norm: `∂y/∂x = 1 + ...` (identity term preserved)
4. **Modern practice**: Pre-Norm is standard (GPT-3, GPT-4, LLaMA, etc.)
5. **Why it matters**: Enables training 96+ layer models stably

---

# Sparse Attention

**The Problem**: Standard attention computes relationships between ALL token pairs → O(n²) complexity.
- 512 tokens: 262,144 comparisons
- 4096 tokens: 16,777,216 comparisons (too expensive!)

**The Solution**: Compute attention only for selected token pairs (sparse patterns) → O(n√n) or O(n log n).

<img src="images/sparse-attention-patterns.png" alt="Sparse Attention Patterns" width="600">

---

## Common Sparse Attention Patterns

| Pattern | How It Works | Complexity | Used In |
|---------|--------------|------------|---------|
| **Local (Sliding Window)** | Each token attends to nearby tokens only | O(n×w) | Longformer, BigBird |
| **Global** | Special tokens attend to all, all attend to special tokens | O(n×g) | Longformer, BigBird |
| **Strided** | Attend every k-th token | O(n×n/k) | Sparse Transformer |
| **Random** | Random subset of tokens | O(n×r) | BigBird, Sparse Transformer |
| **Block-Sparse** | Divide into blocks, attend within/between blocks | O(n√n) | Sparse Transformer |

---

## Example: Longformer (Combined Patterns)

Longformer is a transformer variant designed to handle long sequences (4096+ tokens) efficiently by replacing the standard O(n²) self-attention with **sparse attention patterns** that reduce complexity to O(n).

**The Problem with Standard Attention**

```
Standard Transformer Attention:
- Every token attends to EVERY other token
- Complexity: O(n²)
- Memory: O(n²)

Example with 4096 tokens:
- Attention matrix: 4096 × 4096 = 16.7 million computations
- Memory: ~67 MB per attention head
- With 12 heads: ~800 MB just for attention!
- Too expensive for long documents! ❌
```

**Longformer's Solution**

```
Longformer Sparse Attention:
- Each token only attends to SELECTED tokens
- Complexity: O(n × w) where w is window size
- Memory: Much less!

Example with 4096 tokens, window=512:
- Each token attends to ~512 tokens (not 4096!)
- Memory: ~50 MB total (16× less!)
- Can process documents 8× longer! ✓
```

**Task**: Process 4096-token document

```
Standard Attention (Full):
  Every token attends to all 4096 tokens
  Complexity: 4096² = 16.7M comparisons ❌ Too slow!

Longformer (Sparse):
  1. Local Window (w=512):    Each token → 512 nearby tokens
  2. Global Tokens (g=8):     8 special tokens → all 4096 tokens
  3. Strided (every 64th):    Long-range patterns

  Complexity: 4096×512 + 4096×8 = 2.1M comparisons ✓ 8× faster!
```

**Visual Pattern**:
```
Full Attention:           Sparse Attention (Longformer):
████████████████          ████░░░░░░░░░░░░  (local + global + strided)
████████████████    vs    ░███░░░░░░░░░░░░
████████████████          ░░██░█░░░░░░░░░░
████████████████          ░░░█░░█░░░░░░░░░
(all pairs)               (selected pairs only)
```

### Three Types of Attention in Longformer

#### 1. Local Window Attention (w=512)
Each token attends to nearby tokens within a fixed window.

```
Document: 4096 tokens

Token at position 1000:
  Attends to: positions 744 to 1256 (512 tokens centered around 1000)
  
Visual:
... [744] [745] ... [999] [1000] [1001] ... [1255] [1256] ...
           ←──────────────────────────────────→
                    Window size = 512
                    
Token 1000 can "see" these 512 neighbors
```

**Why?** Most relevant context is nearby (e.g., same sentence/paragraph)


#### 2. Global Attention (g=8 special tokens)

A few special tokens attend to ALL tokens (and all tokens attend to them).

```
Special tokens: [CLS], [SEP], or task-specific tokens

Example: Document with 4096 tokens + 8 global tokens

Global token [CLS] at position 0:
  Attends to: ALL 4096 tokens (full document view)
  
Regular token at position 1000:
  Attends to: [CLS] + 512 local neighbors
  
Visual:
[CLS]─────────────────────────────────────→ ALL tokens
  ↑                                            ↓
  └──────────────────────────────────────← ALL tokens attend back
```

**Why?** These tokens aggregate global information for classification/summarization

---

### Understanding Special Tokens in Transformers and LLMs

Special tokens are non-vocabulary tokens with specific functions in transformer models. They mark boundaries, indicate structure, and enable special behaviors during training and inference.

#### Common Special Tokens Reference Table

| Token | Purpose | Used In | Example |
|-------|---------|---------|---------|
| `[PAD]` | Padding shorter sequences to match batch length | BERT, All models | `["Hello", "world", "[PAD]", "[PAD]"]` |
| `[UNK]` | Unknown/out-of-vocabulary words | BERT, Legacy models | Input: "supercalifragilisticexpialidocious" → `[UNK]` |
| `[CLS]` | Classification token (sentence representation) | BERT | `[CLS] I love NLP [SEP]` |
| `[SEP]` | Separator between segments/sentences | BERT | `[CLS] Question [SEP] Answer [SEP]` |
| `[MASK]` | Masked token for MLM pre-training | BERT, RoBERTa | `I [MASK] coding` (predict "love") |
| `<s>` | Beginning of sequence | GPT-2, LLaMA | `<s> Once upon a time` |
| `</s>` | End of sequence | GPT-2, LLaMA, T5 | `The end. </s>` |
| `<unk>` | Unknown token | GPT-2, LLaMA | Rare word → `<unk>` |
| `<pad>` | Padding token | GPT-2, T5 | `["Hi", "<pad>", "<pad>"]` |
| `<|endoftext|>` | Document boundary/EOS | GPT-2/3 | `Doc1 text <|endoftext|> Doc2 text` |
| `<|im_start|>` | Start of chat turn | ChatGPT/GPT-4 | `<|im_start|>user\nHello<|im_end|>` |
| `<|im_end|>` | End of chat turn | ChatGPT/GPT-4 | `<|im_start|>assistant\nHi!<|im_end|>` |
| `<BOS>` | Beginning of sequence | LLaMA | `<BOS> The capital of France` |
| `<EOS>` | End of sequence | LLaMA | `is Paris. <EOS>` |
| `<SYS>` | System instruction marker | LLaMA 2 Chat | `<SYS> You are helpful </SYS>` |
| `<INST>` | User instruction marker | LLaMA 2 Chat | `<INST> Explain AI </INST>` |
| `<<SYS>>` | System message start | LLaMA 2 | `<<SYS>> Be concise <</SYS>>` |

#### Model-Specific Formats

**BERT (Encoder-only)**
```
Single sentence: [CLS] I love transformers [SEP]
Sentence pair:   [CLS] What is AI? [SEP] AI is... [SEP]
```

**GPT-2/3 (Decoder-only)**
```
Training:   <|endoftext|> Article text... <|endoftext|> Next article... <|endoftext|>
Generation: The capital of France is → Paris
```

**LLaMA Base (Decoder-only)**
```
Format: <BOS> Prompt text <EOS>
```

**LLaMA 2 Chat (Instruction-tuned)**
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is machine learning? [/INST] Machine learning is a subset of AI... </s>
```

**ChatGPT API Format**
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Explain quantum computing.
<|im_end|>
<|im_start|>assistant
Quantum computing uses quantum mechanics...
<|im_end|>
```

#### Training Usage

**Pre-training**: Models learn to ignore/predict special tokens
```python
# MLM (BERT)
input:  "I [MASK] coding"
target: [MASK] → "love"

# CLM (GPT)
input:  "The cat sat on the"
target: "mat <|endoftext|>"
```

**Fine-tuning**: Special tokens structure task-specific inputs
```python
# Classification
"[CLS] This movie is amazing! [SEP]" → Sentiment: Positive

# QA
"[CLS] What is AI? [SEP] AI stands for... [SEP]" → Extract answer span

# Chat
"<|im_start|>user\nHello<|im_end|><|im_start|>assistant" → Generate: "Hi! How can I help?"
```

---



**Example: Document Classification**

```
Task: Classify a research paper (4096 tokens) as "Computer Science" or "Biology"

Paper structure:
  [CLS] [Abstract tokens...] [SEP] [Introduction tokens...] [SEP] 
  [Methods tokens...] [SEP] [Results tokens...] [SEP] [Conclusion tokens...]

Global tokens: [CLS], [SEP] (8 total)

[CLS] token:
  - Attends to entire paper (all 4096 tokens)
  - Sees: "neural networks" (position 50), "GPU" (position 500), 
          "backpropagation" (position 1500), "accuracy" (position 3000)
  - Aggregates: This is Computer Science! ✓
  
Regular token "networks" at position 50:
  - Attends to: [CLS] (global context) + local window (nearby words)
  - Benefits from both local and global information
```

#### 3. Strided Attention (every 64th token)

Dilated/strided pattern: attend to tokens at fixed intervals for long-range patterns.

```
Token at position 1000:
  Attends to: 1000 ± 64, 1000 ± 128, 1000 ± 192, ...
  
Positions: 1000, 936, 872, 808, ..., 1064, 1128, 1192, ...
           ↓     ↓     ↓     ↓           ↓     ↓     ↓
         Every 64 tokens (stride = 64)

Visual (simplified, showing every 64th position):
Pos:  0    64   128  192  256  ...  936  1000  1064  1128  ...  4096
      ●────●────●────●────●────────●────●────●────●─────────────●
                                        ↑
                                   Token 1000
      ←─────────────────────────────────────────────────────→
                Attends to these positions
```

**Why?** Captures long-range dependencies (e.g., document structure, recurring themes)

**Example: Legal Document Analysis**

```
Contract (4096 tokens):
Position 0:    "This Agreement..."
Position 512:  "Party A shall..."
Position 1024: "Payment terms..."
Position 1536: "Party A must..."
Position 2048: "Termination..."
Position 2560: "Party A agrees..."
Position 3072: "Dispute resolution..."
Position 3584: "Party A acknowledges..."

Token "Party A" at position 1536 with stride=64:
  Attends to positions: 0, 64, 128, ..., 1472, 1536, 1600, ..., 4032
  
Sees other mentions of "Party A" at:
  - Position 512  (strided attention catches this)
  - Position 2560 (strided attention catches this)
  - Position 3584 (strided attention catches this)
  
Result: Understands "Party A" appears throughout and has consistent obligations ✓
```

**Attention Loss in Strided Attention**

You miss 63 out of every 64 tokens (98.4% of tokens are ignored). This is an intentional trade-off for efficiency.

**What Gets Lost**
**Concrete Example: 4096-token document with stride=64**

```
Total tokens: 4096
Strided attends to: 4096 ÷ 64 = 64 tokens (only 1.6% of document!)
Tokens ignored: 4096 - 64 = 4032 tokens (98.4% lost!)

Document tokens:
0, 1, 2, 3, ..., 62, 63, 64, 65, 66, ..., 126, 127, 128, 129, ...
█  ░  ░  ░       ░   ░   █   ░   ░        ░    ░    █    ░
↑  ←────Lost─────→       ↑   ←──Lost──→            ↑
Seen                   Seen                      Seen

Only positions: 0, 64, 128, 192, ... are attended
Everything in between: LOST!

```

---

## Key Models Using Sparse Attention

| Model | Patterns | Max Length | Speedup |
|-------|----------|------------|---------|
| **Longformer** | Local + Global | 4096 tokens | 8× faster |
| **BigBird** | Local + Global + Random | 4096 tokens | 8× faster |
| **Sparse Transformer** | Strided + Block | 16,384 tokens | 100× faster |
| **Reformer** | LSH (Locality-Sensitive Hashing) | 64,000 tokens | 1000× faster |

---

## Benefits & Trade-offs

| Aspect | Full Attention | Sparse Attention |
|--------|----------------|------------------|
| **Complexity** | O(n²) | O(n√n) or O(n log n) |
| **Max sequence** | ~512-2048 | 4K-64K tokens |
| **Speed** | Slow for long texts | Fast |
| **Information** | Sees all relationships | Misses some relationships |
| **Use case** | Short texts | Long documents, books |

---

## Key Takeaways

1. **Sparse Attention** reduces O(n²) → O(n√n) by computing attention for selected token pairs only
2. **Common patterns**: Local windows, global tokens, strided, random
3. **Enables long contexts**: 4K-64K tokens vs 512-2K in standard transformers
4. **Trade-off**: Speed/memory vs complete information
5. **Used in**: Longformer (documents), BigBird (long sequences), Sparse Transformer (very long texts)

---

## Is Sparse Attention Widely Used in Modern LLMs?

**Short answer: No — largely replaced by better alternatives.**

| Period | Status |
|---|---|
| 2020–2022 | Popular research direction (Longformer, BigBird, Reformer) |
| 2022 | FlashAttention made full attention fast enough → sparse attention lost its edge |
| 2023–2025 | Mostly abandoned; replaced by FlashAttention + MoE |

**Why it faded:**
- **FlashAttention** (2022) made full attention IO-efficient — the main problem sparse attention solved
- **Sparse patterns skip token pairs** — hurts reasoning quality
- **Hard to parallelize** on GPUs efficiently

**What replaced it:**

| Technique | Used by |
|---|---|
| Full attention + FlashAttention | GPT-4, LLaMA 3, Gemini, Claude |
| Sliding window (local attention) | Mistral 7B, Gemma |
| Sparse MoE (experts, not attention) | Mixtral, Grok, GPT-4 (rumored) |

> **Key insight:** Industry moved from sparse *attention* to sparse *MoE* — activating only 2 of 8 FFN experts per token gives efficiency gains without sacrificing attention quality.

---

# Sharing Attention Heads

**The Problem**: Standard transformers have separate attention weights for each layer → many parameters.
- BERT-base: 12 layers × 12 heads = 144 attention heads
- Each head has its own W_Q, W_K, W_V matrices → high memory cost

**The Solution**: Share attention head weights across multiple layers → reduce parameters while maintaining performance.

**Context**: `In decoder masked attention, transformers must attend to all previous words to generate new tokens. KV (Key-Value) pairs of all previously generated words are cached for efficiency.`

<img src="images/kv-cache-mechanism.png" alt="KV Cache Mechanism" width="650">

**Grouped Query Attention (GQA)**: `h = number of heads in attention, G = number of groups`

<img src="images/attention_heads.png" alt="Grouped Query Attention" width="700">

---

## Multi-Head Attention (MHA) vs Multi-Query Attention (MQA) vs Grouped-Query Attention (GQA)

### Overview

These are different strategies for organizing Query, Key, and Value matrices in attention to improve efficiency.

---

### 1. Multi-Head Attention (MHA) - Standard

**What it is**: Each head has its **own** Q, K, and V matrices.

**Structure** (8 heads):
```
Head 1: Q₁, K₁, V₁  (unique)
Head 2: Q₂, K₂, V₂  (unique)
...
Head 8: Q₈, K₈, V₈  (unique)

Total: 8 × 3 = 24 matrices
```

**Characteristics**:
- ✅ Best quality (most expressive)
- ❌ Highest memory (caches 8 K,V pairs)
- ❌ Slower inference

**KV Cache per token**: `8 heads × 2 (K,V) = 16 matrices`

---

### 2. Multi-Query Attention (MQA) - Maximum Sharing

**What it is**: All heads share **one** K and V, but each has its own Q.

**Structure** (8 heads):
```
Head 1: Q₁ ─┐
Head 2: Q₂ ─┤
Head 3: Q₃ ─┼─→ K_shared, V_shared (ONE pair)
...         │
Head 8: Q₈ ─┘

Total: 8 Q + 1 K + 1 V = 10 matrices
```

**Characteristics**:
- ✅ Minimal memory (only 1 K,V pair)
- ✅ Fastest inference (8× less memory bandwidth)
- ⚠️ Small quality drop (~1-2%)
- 🔧 Used in: PaLM, Falcon

**KV Cache per token**: `1 × 2 (K,V) = 2 matrices` (8× reduction!)

---

### 3. Grouped-Query Attention (GQA) - Balanced

**What it is**: Heads divided into **groups**, each group shares one K,V pair.

**Structure** (8 heads, 4 groups):
```
Group 1: Head 1 (Q₁), Head 2 (Q₂) → share K₁, V₁
Group 2: Head 3 (Q₃), Head 4 (Q₄) → share K₂, V₂
Group 3: Head 5 (Q₅), Head 6 (Q₆) → share K₃, V₃
Group 4: Head 7 (Q₇), Head 8 (Q₈) → share K₄, V₄

Total: 8 Q + 4 K + 4 V = 16 matrices
```

**Characteristics**:
- ✅ Balanced memory (4× less than MHA)
- ✅ Near-MHA quality (~99%)
- ✅ Flexible (tune number of groups)
- 🔧 Used in: Llama-2, Mistral, Gemma

**KV Cache per token**: `4 groups × 2 (K,V) = 8 matrices` (2× reduction)

---

### Comparison Table

| Aspect | MHA | GQA (4 groups) | MQA |
|--------|-----|----------------|-----|
| **Q matrices** | 8 | 8 | 8 |
| **K,V matrices** | 8 each | 4 each | 1 each |
| **Total matrices** | 24 | 16 | 10 |
| **KV Cache size** | 100% | 50% | 12.5% |
| **Inference speed** | 1× | 1.5-2× | 3-5× |
| **Quality** | 100% | 99% | 97-98% |
| **Used in** | GPT-3, BERT | Llama-2, Mistral | PaLM, Falcon |

---

### Visual Comparison

```
MHA (No sharing):
H1:[Q₁][K₁][V₁]  H2:[Q₂][K₂][V₂]  ... H8:[Q₈][K₈][V₈]
KV Cache: 8 pairs → High memory

GQA (Group sharing):
Group1: H1[Q₁], H2[Q₂] → [K₁][V₁]
Group2: H3[Q₃], H4[Q₄] → [K₂][V₂]
Group3: H5[Q₅], H6[Q₆] → [K₃][V₃]
Group4: H7[Q₇], H8[Q₈] → [K₄][V₄]
KV Cache: 4 pairs → Medium memory

MQA (All share):
H1[Q₁], H2[Q₂], ... H8[Q₈] → [K_shared][V_shared]
KV Cache: 1 pair → Low memory
```

---

### Why This Matters

**During text generation** (1000 tokens):
```
MHA:  1000 × 8 heads × 2 = 16,000 cached values
GQA:  1000 × 4 groups × 2 = 8,000 cached values  (2× faster)
MQA:  1000 × 1 group × 2 = 2,000 cached values   (8× faster)
```

**Memory bandwidth** is the bottleneck - fewer K,V pairs = faster inference!

---

### Example: Llama-2 Uses GQA

```
Configuration:
- 32 attention heads
- 8 KV groups
- Each group: 4 heads share K,V

Result:
- 32 Q matrices
- 8 K matrices (4× reduction)
- 8 V matrices (4× reduction)
- KV cache 4× smaller than MHA
- Quality: ~99% of MHA
```

---

### When to Use Each

| Use Case | Best Choice | Why |
|----------|-------------|-----|
| Quality priority | MHA | Most expressive |
| Balanced needs | GQA | Best quality/efficiency trade-off |
| Speed critical | MQA | Fastest inference |
| Long contexts | GQA | Good quality + manageable memory |
| Large scale | GQA | Most practical for production |

---

### Why Sharing KV Causes Quality Drop

**The Core Issue**: Loss of representational diversity

**Standard MHA** (no sharing):
```
Head 1: K₁ = X · W_K1 → learns syntax patterns (grammatical roles)
Head 2: K₂ = X · W_K2 → learns semantic relationships (word meanings)
Head 3: K₃ = X · W_K3 → learns positional patterns (word order)
Head 4: K₄ = X · W_K4 → learns co-reference (pronouns, entities)
...each head has unique W_K weight matrix → specializes independently
```

**MQA** (all share K, V):
```
All Heads: K_shared = X · W_K_shared
Head 1: Q₁·K_shared → constrained by same K
Head 2: Q₂·K_shared → constrained by same K
Head 3: Q₃·K_shared → constrained by same K
Head 4: Q₄·K_shared → constrained by same K
...only ONE W_K weight matrix → K must compromise for all patterns
```

**How K Works**:
- **"Different K"** = Different weight matrices (W_K), NOT frozen values
- **Training**: W_K updated every step, K recomputed each forward pass
- **Inference**: W_K frozen, K computed fresh for each new token
- **MHA**: Each head's W_K specializes → diverse K representations
- **MQA**: Single W_K serves all heads → compromised K representation

**Example: Word "bank"** (input embedding = [0.5, 0.3, 0.8, ...]):
```
MHA (different W_K per head):
  Head 1 (Financial): K₁ = [0.5, 0.3, 0.8] · W_K1 = [0.9, 0.1, 0.2]  ← emphasizes money context
  Head 2 (Physical):  K₂ = [0.5, 0.3, 0.8] · W_K2 = [0.1, 0.8, 0.3]  ← emphasizes location context
  → Different W_K → Different K representations

MQA (shared W_K):
  All Heads: K_shared = [0.5, 0.3, 0.8] · W_K_shared = [0.5, 0.5, 0.3]  ← compromise
  → Same W_K → Same K for all heads (must serve both financial AND physical contexts)
```

**Analogy**:
- **MHA**: 8 cameras with different lenses (wide, zoom, infrared) → rich perspectives
- **MQA**: 8 people viewing 1 photo → limited perspective, only ask different questions

**Quality Impact**:
- **MHA**: 100% - heads learn diverse attention patterns (syntax, semantics, position)
- **GQA**: 99% - some diversity preserved through groups
- **MQA**: 97-98% - single K/V bottlenecks information

**Trade-off**: Speed vs diversity. MQA is 8× faster but heads can't specialize as effectively.

---

### Key Takeaway

- **MHA**: Every head unique → Best quality, most memory
- **GQA**: Heads grouped → Balanced (⭐ most popular in modern LLMs)
- **MQA**: All share one K,V → Fastest, some quality loss

**Trend**: Modern LLMs prefer **GQA** (Llama-2, Mistral) for optimal quality/efficiency balance!

---

## How It Works

**Standard Transformer** (No Sharing):
```
Layer 1: Head 1 (W_Q₁, W_K₁, W_V₁) ← Unique weights
Layer 2: Head 1 (W_Q₂, W_K₂, W_V₂) ← Different weights
Layer 3: Head 1 (W_Q₃, W_K₃, W_V₃) ← Different weights
...
All 144 heads have separate weights
```

**With Sharing** (Shared Heads):
```
Layer 1: Head 1 (W_Q, W_K, W_V) ← Shared weights
Layer 2: Head 1 (W_Q, W_K, W_V) ← SAME weights reused
Layer 3: Head 1 (W_Q, W_K, W_V) ← SAME weights reused
...
One set of weights used across multiple layers
```

---

## Sharing Strategies

| Strategy | Description | Parameter Reduction | Used In |
|----------|-------------|---------------------|---------|
| **Cross-Layer Sharing** | Same head weights across all layers | ~70% reduction | ALBERT, Universal Transformer |
| **Group Sharing** | Share within groups of layers | ~30-50% reduction | Some efficient transformers |
| **Factorized Sharing** | Share Q/K/V separately | ~40% reduction | ALBERT |

---

## Example: ALBERT (A Lite BERT)

**BERT-base** (No Sharing):
```
Parameters: 110M
  - 12 layers × 12 heads × 3 matrices (Q,K,V)
  - Each layer learns independently
Memory: High
```

**ALBERT-base** (With Sharing):
```
Parameters: 12M  ← 90% reduction!
  - 12 layers × 12 heads
  - All layers share SAME attention weights
  - Only embedding/output layers differ
Memory: Low
```

**Performance Comparison**:
```
Task: SQuAD (Question Answering)

BERT-base:     110M params → 88.5% accuracy
ALBERT-base:    12M params → 89.3% accuracy  ← Better with fewer params!

Why? Cross-layer sharing acts as regularization
```

---

## Benefits & Trade-offs

| Aspect | Without Sharing | With Sharing |
|--------|----------------|--------------|
| **Parameters** | 100M-1B | 10M-100M (10× fewer) |
| **Memory** | High | Low |
| **Training speed** | Slower | Faster (fewer params to update) |
| **Inference speed** | Same | Same (computation unchanged) |
| **Performance** | Baseline | Similar or better (regularization effect) |
| **Flexibility** | Each layer specializes | Layers more similar |

---

## Visual Comparison

```
Standard Transformer:
┌─────────────┐
│  Layer 12   │  W_Q₁₂, W_K₁₂, W_V₁₂  ← Unique
├─────────────┤
│  Layer 11   │  W_Q₁₁, W_K₁₁, W_V₁₁  ← Unique
├─────────────┤
│  Layer 10   │  W_Q₁₀, W_K₁₀, W_V₁₀  ← Unique
├─────────────┤
│    ...      │  ...
└─────────────┘
Total: 12 sets of weights

Shared Heads (ALBERT):
┌─────────────┐
│  Layer 12   │ ─┐
├─────────────┤  │
│  Layer 11   │ ─┤
├─────────────┤  ├─→ W_Q, W_K, W_V (SHARED)
│  Layer 10   │ ─┤
├─────────────┤  │
│    ...      │ ─┘
└─────────────┘
Total: 1 set of weights (reused 12 times)
```

---

## Key Models Using Shared Heads

| Model | Sharing Type | Params (vs BERT) | Performance |
|-------|--------------|------------------|-------------|
| **ALBERT** | Full cross-layer | 12M (vs 110M) | Same or better |
| **Universal Transformer** | Full sharing + recurrence | Variable | Better for structured tasks |
| **DeBERTa-v3** | Partial sharing | 86M (vs 110M) | Better |

---

## Key Takeaways

1. **Sharing Attention Heads** = Reuse same attention weights across multiple layers
2. **Reduces parameters** by 70-90% without hurting performance
3. **Acts as regularization** → prevents overfitting, can improve accuracy
4. **Memory efficient** → enables larger models on limited hardware
5. **Used in**: ALBERT (90% reduction), Universal Transformer, efficient BERT variants

---

# Transformer-Based Models

Transformers come in three architectural variants, each designed for different tasks:

## Quick Comparison

| Architecture | Attention Type | Primary Use | Example Models | Example Task |
|--------------|----------------|-------------|----------------|--------------|
| **Encoder-Only** | Bidirectional (sees all tokens) | Understanding & Classification | BERT, RoBERTa | "Is this email spam?" |
| **Encoder-Decoder** | Encoder: bidirectional<br>Decoder: causal | Transformation (seq-to-seq) | T5, BART | Translate English → French |
| **Decoder-Only** | Causal (left-to-right) | Text Generation | GPT-3, LLaMA, ChatGPT | Complete: "Once upon a time..." |

<img src="images/transformer_based_model.png" alt="Transformer-Based Models" width="650">

---

## 1. Encoder-Only (BERT-style)

**Purpose:** Deep understanding by seeing the entire sentence at once.

**How it works:** Each word sees ALL other words (both past and future).

```
Input: "The cat sat on the mat"

Processing:
  The ↔ cat ↔ sat ↔ on ↔ the ↔ mat
  (All tokens attend to ALL others - bidirectional)

Result: Rich understanding of each word in context
```

**Use Cases:**
- **Sentiment Analysis:** `[CLS] This movie is amazing! [SEP]` → Positive ✓
- **Named Entity Recognition:** "Apple announced new iPhone" → Apple=ORG, iPhone=PRODUCT
- **Question Answering:** Extract answer span from context

**Key Point:** Cannot generate new text, only understands/classifies existing text.

---

## 2. Encoder-Decoder (T5/BART-style)

**Purpose:** Transform one sequence into another (different language, length, or format).

**How it works:** 
- **Encoder** understands input (bidirectional)
- **Decoder** generates output (causal, one token at a time)

```
ENCODER (bidirectional):
  "Hello world" → [Rich understanding of English]
  
DECODER (causal + attends to encoder):
  [BOS] → "Bonjour" → "monde" → [EOS]
```

**Use Cases:**
- **Translation:** English → French
- **Summarization:** Long article → Short summary
- **Text-to-SQL:** "Show customers from CA" → `SELECT * FROM customers WHERE state='CA'`

**Key Point:** Best when input and output are different (language, length, format).

---

## 3. Decoder-Only (GPT/LLaMA-style)

**Purpose:** Generate text by predicting the next token, one at a time.

**How it works:** Each token ONLY sees past tokens (left-to-right).

```
Input:  "The cat sat on the"
        ↓ (only sees past) ↓
Output: "mat"

Next step:
Input:  "The cat sat on the mat"
        ↓ (only sees past) ↓
Output: "and"
```

**Use Cases:**
- **Text Completion:** "Write a story: Once upon a time" → [generates full story]
- **Conversational AI:** ChatGPT processes entire conversation history → generates response
- **Code Generation:** `# Function to calculate factorial` → [generates Python function]
- **In-Context Learning:** Learn from examples in the prompt (few-shot)

**Key Point:** Most flexible - can do almost any task via prompting (modern LLMs like GPT-4, Claude).

---

## Why Architecture Matters: Example Task

**Task: Sentiment Analysis**

### Encoder-Only (BERT):
```
Input:  [CLS] This movie is terrible but the acting is good [SEP]
        ↓ Sees "terrible" AND "but" AND "good" simultaneously
Output: Mixed sentiment ✓
Best: Sees full context for nuanced understanding
```

### Decoder-Only (GPT):
```
Input:  "Sentiment: This movie is terrible but the acting is good\nAnswer:"
        ↓ Processes left-to-right
Output: "Mixed"
Works: But relies on prompt engineering
```

---

## Modern Trend: Decoder-Only Dominance

**Evolution:**
- **2018:** BERT (Encoder-only) - Best for classification
- **2019:** T5 (Encoder-decoder) - Best for translation
- **2023-2025:** GPT-4/LLaMA (Decoder-only) - Best for EVERYTHING

**Why Decoder-Only Won:**
- ✅ Single unified architecture (simpler)
- ✅ Scales to massive sizes (100B+ parameters)
- ✅ In-context learning (no fine-tuning needed)
- ✅ Flexible via prompting (one model, all tasks)

**Trade-off:**
- Slightly less optimal for pure encoding tasks (vs BERT)
- But gains universal capability via prompting

---

## Summary for Beginners

| When you need to... | Use |
|---------------------|-----|
| **Classify or understand** text (spam detection, sentiment) | Encoder-Only (BERT) |
| **Transform** one sequence to another (translation, summarization) | Encoder-Decoder (T5) or Decoder-Only with prompting |
| **Generate** new text (stories, conversations, code) | Decoder-Only (GPT, LLaMA) |
| **Do everything** with one model | Decoder-Only (modern LLMs) |

**Key Insight:** Modern LLMs (GPT-4, Claude, LLaMA) are all decoder-only because they can handle any task through clever prompting, making them the most practical choice today! 🎯