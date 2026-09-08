Deep_learning.md
# Deep Learning Interview Guide — Senior AI/ML Engineer (14 Years Experience)

> Covers fundamentals through cutting-edge techniques. Each section includes concept explanation, working PyTorch code, and interview insights.

---

## Table of Contents

### Beginner's Foundation Guide
> **New to Deep Learning?** Start here — plain language, analogies, and diagrams before the math.

- [B1. Activation Functions](#b1-activation-functions)
    - [Sigmoid](#1-sigmoid)
    - [Tanh](#2-tanh-hyperbolic-tangent)
    - [ReLU](#3-relu-rectified-linear-unit)
    - [Leaky ReLU](#4-leaky-relu)
    - [GELU](#5-gelu-gaussian-error-linear-unit)
    - [PReLU](#8-parameterised-relu-prelu)
    - [ELU](#9-elu-exponential-linear-unit)
    - [Swish](#10-swish)
    - [Softplus](#11-softplus)
    - [Softmax](#12-softmax-output-layer-for-multi-class)
  - [Leaky ReLU vs GELU](#leaky-relu-vs-gelu)
  - [All Activation Functions — Master Summary Table](#all-activation-functions--master-summary)
  - [Does activation run during backprop?](#q2-is-the-activation-function-applied-again-during-backpropagation)
- [B2. Chain Rule & Backpropagation](#b2-chain-rule--backpropagation)
  - [Does optimizer update ALL layers or just final?](#q1-does-the-optimizer-update-weights-in-all-layers-or-just-the-final-layer)
- [B3. Vanishing vs Exploding Gradients](#b3-vanishing-vs-exploding-gradients)
  - [Batch Normalization fix](#batch-normalization-fix-for-unnormalized-layer-outputs)
  - [Residual connections fix](#residual-connection-the-key-fix-for-very-deep-nets)
- [B4. Optimizers](#b4-optimizers)
  - [SGD](#1-sgd--stochastic-gradient-descent)
    - [SGD+Momentum](#2-sgd--momentum)
    - [AdaGrad](#3-adagrad--adaptive-gradient)
    - [Adadelta](#4b-adadelta--adaptive-learning-rate-no-manual-lr)
    - [RMSProp](#4-rmsprop--root-mean-square-propagation)
    - [Adam](#5-adam--adaptive-moment-estimation--most-popular)
    - [AdamW](#6-adamw--adam-with-weight-decay-modern-default)
  - [β₁=0.9, β₂=0.98 — weight spike prevention](#β₁09-β₂098--what-it-means-for-weight-updates)
  - [Adam L2 problem & AdamW fix](#6-adamw--adam-with-weight-decay-modern-default)
- [B5. Batch GD vs SGD vs Mini-Batch SGD](#b5-batch-gd-vs-sgd-vs-mini-batch-sgd)
- [B6. Global vs Local Minima](#b6-global-vs-local-minima)
  - [Convex vs Non-Convex](#convex-vs-non-convex-functions)
- [Saddle Points](#saddle-points)
- [B7. CNN for Beginners](#b7-cnn-for-beginners)
  - [Pixels & Images](#what-is-an-image-to-a-computer)
- [Convolution](#what-is-convolution)
- [Padding](#padding--keeping-image-size)
- [Max Pooling](#max-pooling--shrinking-smartly)
- [Full Architecture](#full-cnn-architecture-flow)
- [Data Augmentation](#data-augmentation)
- [B8. Transfer Learning](#b8-transfer-learning--data-augmentation)
- [B9. Object Detection: R-CNN & YOLO](#b9-object-detection-r-cnn--yolo)
  - [R-CNN](#r-cnn-region-based-cnn)
- [Fast R-CNN](#fast-r-cnn)
- [Faster R-CNN](#faster-r-cnn)
- [YOLO](#yolo--you-only-look-once-)
- [B10. Word Embeddings & Word2Vec](#b10-word-embeddings--word2vec)
- [B11. Sequence-to-Sequence & Bidirectional LSTM](#b11-sequence-to-sequence--bidirectional-lstm)
  - [Bidirectional LSTM](#bidirectional-lstm)
- [Seq2Seq](#sequence-to-sequence-seq2seq)
- [LSTM vs Seq2Seq](#lstm-vs-seq2seq--when-to-use-which)

### Advanced Reference Guide
1. [Fundamentals](#1-fundamentals)
   - [1.1 Perceptron & MLP](#11-perceptron--mlp)
   - [1.2 Activation Functions](#12-activation-functions)
   - [1.3 Backpropagation & Chain Rule](#13-backpropagation--chain-rule)
   - [1.4 Loss Functions](#14-loss-functions)
   - [1.5 Optimizers](#15-optimizers)
   - [1.6 Learning Rate Schedulers](#16-learning-rate-schedulers)
2. [Regularization & Optimization](#2-regularization--optimization)
   - [2.1 Dropout, DropPath, Normalization](#21-dropout-droppath-normalization)
   - [2.2 Weight Initialization](#22-weight-initialization)
   - [2.3 Gradient Clipping & Accumulation](#23-gradient-clipping--accumulation)
   - [2.4 Mixed Precision Training](#24-mixed-precision-training)
3. [CNNs](#3-cnns)
   - [3.1 Convolution Types](#31-convolution-types)
   - [3.2 ResNet & Residual Blocks](#32-resnet--residual-blocks)
4. [RNNs & Sequence Models](#4-rnns--sequence-models)
   - [4.1 LSTM & GRU](#41-lstm--gru)
5. [Attention & Transformers](#5-attention--transformers)
   - [5.1 Scaled Dot-Product Attention](#51-scaled-dot-product-attention)
   - [5.2 Multi-Head Attention](#52-multi-head-attention)
   - [5.3 Positional Encoding](#53-positional-encoding)
   - [5.4 Full Transformer Block](#54-full-transformer-block)
     - [Forward vs Backprop through Transformer](#forward-pass-vs-backpropagation-through-a-transformer-block)
     - [Self-Attention during backprop](#what-happens-inside-self-attention-during-backpropagation)
     - [768-dim → Self-Attention → FFN dimension flow](#dimension-flow-768-dim-input-through-self-attention--ffn)
6. [Advanced Training Techniques](#6-advanced-training-techniques)
   - [6.1 Knowledge Distillation](#61-knowledge-distillation)
   - [6.2 PEFT — LoRA, QLoRA, Adapters](#62-peft--lora-qlora-adapters)
     - [Observing weight updates during Fine-Tuning / DPO](#how-to-observe-weight-updates-during-fine-tuning--dpo)
   - [6.3 Contrastive Learning — SimCLR & CLIP](#63-contrastive-learning--simclr--clip)
7. [Generative Models](#7-generative-models)
   - [7.1 Variational Autoencoder (VAE)](#71-variational-autoencoder-vae)
   - [7.2 GANs](#72-gans)
   - [7.3 Diffusion Models (DDPM)](#73-diffusion-models-ddpm)
8. [Scaling & Distributed Training](#8-scaling--distributed-training)
   - [8.1 Data, Model, and Tensor Parallelism](#81-data-model-and-tensor-parallelism)
   - [8.2 ZeRO Optimization & Gradient Checkpointing](#82-zero-optimization--gradient-checkpointing)
   - [8.3 Pipeline Parallelism](#83-pipeline-parallelism)
9. [Quantization & Efficiency](#9-quantization--efficiency)
   - [9.1 Post-Training Quantization](#91-post-training-quantization)
   - [9.2 KV Cache & Speculative Decoding](#92-kv-cache--speculative-decoding)
   - [9.3 Pruning](#93-pruning)
10. [Advanced Topics](#10-advanced-topics)
    - [10.1 Mixture of Experts (MoE)](#101-mixture-of-experts-moe)
    - [10.2 Continual Learning & Catastrophic Forgetting](#102-continual-learning--catastrophic-forgetting)
    - [10.3 Interpretability — GradCAM & SHAP](#103-interpretability--gradcam--shap)
    - [10.4 Neural Architecture Search (NAS)](#104-neural-architecture-search-nas)
    - [10.5 Hyperparameter Optimization with Optuna](#105-hyperparameter-optimization-with-optuna)
11. [Key Interview Tradeoffs Summary](#key-interview-tradeoffs-summary)
12. [Common Interview Questions & Answers](#common-interview-questions--answers)

---

## B1. Activation Functions

### What problem do they solve?

Without an activation function, no matter how many layers you stack, your network is just doing **one big linear transformation** — it can only learn straight-line relationships.

#### What Does "One Big Linear Transformation" Mean?

**Linear transformation = multiplication and addition only.**

```
A linear transformation can be written as:
  y = W·x + b

Where:
  W = weights (matrix)
  x = input (vector)
  b = bias (vector)
  · = matrix multiplication

Properties:
  • Straight lines → straight lines
  • Can't represent curves
  • Can't learn XOR, circles, or any non-linear pattern
```

**Why stacking linear layers doesn't help:**

```
WITHOUT activation (useless stacking):

Layer1:  h1 = W1·x + b1
         
Layer2:  h2 = W2·h1 + b2
            = W2·(W1·x + b1) + b2
            = (W2·W1)·x + (W2·b1 + b2)
            = W_combined·x + b_combined
            ↑ still just ONE linear equation!

No matter how many layers you add, you're just composing linear transformations:
  L3: h3 = W3·h2 + b3  =  (W3·W2·W1)·x + (...)
  L4: h4 = W4·h3 + b4  =  (W4·W3·W2·W1)·x + (...)
  
All collapse into: y = W_final·x + b_final (ONE big linear equation)
```

**Visual example: Can't learn XOR without activation**

```
XOR problem (needs curves):
     y
     1 │  ●        ●
       │
     0 │  ●        ●    (● = class 1, ○ = class 0)
       │
       └─────────────── x
         0         1

Without activation: Straight line can't separate.
  Best effort:      _______
                   /
  Accuracy: 50% (worst possible for 4-point XOR)

With activation (e.g., ReLU):
  Layer1: ReLU turns straight line into a "bend"
  Layer2: Can now separate the XOR pattern
  Accuracy: 100% ✓
```

**WITH activation (powerful):**

```
Layer1:  h1 = ReLU(W1·x + b1)       ← introduces a "bend" (non-linearity)
         
Layer2:  h2 = ReLU(W2·h1 + b2)      ← another bend
         
Output:  y = W3·h2 + b3            ← final transformation

This is NO LONGER a linear transformation!
ReLU "breaks" the linearity by zeroing negative values.
Each layer can now learn different representations.
```

**Mathematical proof:**

```
Composition of linear functions = linear:
  f(x) = ax + b
  g(f(x)) = c·(ax + b) + d = (ca)x + (cb + d) = linear

Composition with non-linear function = non-linear:
  f(x) = ax + b (linear)
  g(x) = max(0, x) (ReLU, non-linear)
  g(f(x)) = max(0, ax + b) ≠ linear!
  
Example:
  If ax + b < 0 → g(f(x)) = 0
  If ax + b ≥ 0 → g(f(x)) = ax + b
  
  This creates a "kink" — no longer a straight line!
```

**Why this matters for neural networks:**

```
Linear model capabilities:
  • Binary classification (straight line separator)
  • Linear regression (y = mx + c)
  • Handwritten digit recognition? ✗ (too complex for straight line)

Non-linear model capabilities (with activation):
  • Image classification (curves separate different classes)
  • Language models (transformers use ReLU/GELU non-linearities)
  • Complex decision boundaries (any shape possible)

Universal Approximation Theorem:
  A 2-layer network with activation functions can approximate
  ANY continuous function to ANY accuracy.
  
  But without activation? Only linear functions. Useless.
```

---

`Note: Activation is NOT re-applied during backpropagation. Only its derivative is computed.`

---

### Neuron anatomy

```
                        Activation
         Inputs          Function        Output
  x1 ──┐                    │
  x2 ──┼──► [Σ w·x + b] ──►[f(z)]──────► ŷ
  x3 ──┘

  z = weighted sum   f(z) = activation applied to z
```

---

### Activation Functions at a Glance

```
OUTPUT RANGE COMPARISON

  Sigmoid  ──────────────────────────────
  f(z)     1 │              .........
           0.5│        .....
           0 │....___                    z
              ──────────────────────────►
             -4   -2    0    2    4

  Tanh     ──────────────────────────────
  f(z)     1 │            ..........
           0 │      ......
          -1 │......                     z
              ──────────────────────────►
             -4   -2    0    2    4

  ReLU     ──────────────────────────────
  f(z)     4 │                  /
           2 │              /
           0 │___________/               z
              ──────────────────────────►
             -4   -2    0    2    4

  GELU     ──────────────────────────────
  f(z)     4 │                  /
           0 │     .___........          z
          -0.2│..../
              ──────────────────────────►
             -4   -2    0    2    4
```

---

### Q1: Does the optimizer update weights in ALL layers or just the final layer?

**ALL layers — every single weight in every layer gets updated.**

```
  3-layer network example:

  Input → [Layer 1: W1] → [Layer 2: W2] → [Layer 3: W3] → Loss
                                                              │
  Backprop ◄────────────────────────────────────────────────┘

  optimizer.step() updates:
  ┌──────────────────────────────────────────────────────────────┐
  │  W3  ← updated  (closest to loss, gets largest gradient)     │
  │  W2  ← updated  (gradient flows back through W3)             │
  │  W1  ← updated  (gradient flows back through W3 and W2)      │
  └──────────────────────────────────────────────────────────────┘
  ALL weights updated in one optimizer.step() call.
```

Each weight gets its own gradient `dL/dW` via the chain rule — the optimizer just applies `W = W - lr
- grad` to every one of them simultaneously.

**Exception — intentionally frozen layers:**

```
  Fine-tuning (LoRA / Transfer Learning):

  [Layer 1: W1] frozen  ← requires_grad=False → optimizer SKIPS it
  [Layer 2: W2] frozen  ← requires_grad=False → optimizer SKIPS it
  [Layer 3: W3] trainable ← only this updates

  Gradient still flows THROUGH frozen layers during backprop
  (needed to compute gradients for earlier trainable layers)
  but the frozen weights themselves are NOT updated.
```

---

### 1. Sigmoid

```
         1
f(z) = ──────        range: (0, 1)
        1 + e⁻ᶻ
```

**Analogy:** Like a light dimmer switch — smoothly goes from OFF (0) to ON (1).

```
   Input z    │  Output f(z)   │  Meaning
  ────────────┼────────────────┼──────────────────────
    very neg  │  ≈ 0.0         │  definitely NOT
       0      │  0.5           │  50/50 uncertain
    very pos  │  ≈ 1.0         │  definitely YES
```

| Pros | Cons |
|------|------|
| Output is a probability (0–1) | **Vanishing gradient** — gradient ≈ 0 at extremes |
| Smooth, differentiable | Slow training on deep nets |
| Historically important | Output not zero-centered (causes zig-zag updates) |

**Best used for:** Binary classification output layer only.

---

### 2. Tanh (Hyperbolic Tangent)

```
       eᶻ - e⁻ᶻ
f(z) = ──────────       range: (-1, 1)
       eᶻ + e⁻ᶻ
```

**Analogy:** Sigmoid's sibling — same shape but centered at 0 (like a balanced seesaw).

| Pros | Cons |
|------|------|
| Zero-centered (better gradient flow) | Still has **vanishing gradient** at extremes |
| Stronger gradients than sigmoid | Computationally expensive |

**Best used for:** RNNs, hidden layers when zero-centering matters.

---

### 3. ReLU (Rectified Linear Unit)

```
f(z) = max(0, z)        range: [0, ∞)
```

**Analogy:** A one-way valve — lets positive signals through, blocks negatives.

```
  z = -5  →  f(z) = 0    (blocked)
  z =  3  →  f(z) = 3    (passed through)
```

| Pros | Cons |
|------|------|
| **No vanishing gradient** for positive z | **Dying ReLU** — neurons stuck at 0 forever if z always negative |
| Extremely fast to compute | Not zero-centered |
| Sparse activations (efficient) | Unbounded output |

**Best used for:** Default choice for hidden layers in CNNs and MLPs.

---

### 4. Leaky ReLU

```
f(z) = max(0.01·z, z)   range: (-∞, ∞)
```

**Analogy:** ReLU but with a tiny leak — negative inputs get a small gradient (0.01) instead of zero.

```
  z = -10 →  f(z) = -0.1   (tiny signal, not dead!)
  z =   3 →  f(z) = 3      (same as ReLU)
```

| Pros | Cons |
|------|------|
| Fixes Dying ReLU problem | α (0.01) is a hyperparameter to tune |
| Simple, fast | Not always better than ReLU in practice |

---

### 5. GELU (Gaussian Error Linear Unit)

#### Step 1 — Start with ReLU's problem

ReLU makes a hard decision at zero:

```
  ReLU:  z = -0.01  →  output = 0   (completely blocked)
         z = +0.01  →  output = 0.01 (fully passed through)

  The jump at exactly z=0 is abrupt — like a light switch.
```

GELU asks: *what if instead of a hard switch, we use a soft probabilistic one?*

---

#### Step 2 — The core idea (no math needed)

> **"How confident are we that this input is positive? Pass it through exactly that much."**

Think of a **bouncer at a club**:
- ReLU bouncer: "Negative number? You're OUT. Positive? You're IN." *(harsh, binary)*
- GELU bouncer: "Barely negative? You mostly stay out, but a tiny bit gets through. Very positive? You're fully in." *(smooth, probabilistic)*

```
  Input z   │  ReLU output  │  GELU output   │  What GELU says
  ──────────┼───────────────┼────────────────┼─────────────────────────
    -3.0    │  0.00         │  -0.004        │  "1% chance positive → pass 1%"
    -1.0    │  0.00         │  -0.159        │  "16% chance positive → pass 16%"
     0.0    │  0.00         │   0.00         │  "50/50 → pass half of 0 = 0"
    +1.0    │  1.00         │   0.841        │  "84% chance positive → pass 84%"
    +3.0    │  3.00         │   2.996        │  "99% chance positive → pass ~all"
```

Notice: for large positive values, GELU ≈ ReLU. The difference is only near zero.

---

#### Step 3 — Shape comparison

```
  Output
    │
  3 │                              GELU ─ ─ ─
    │                          ReLU /  GELU /
  2 │                            / /       /
    │                           / /       /
  1 │                          / /       /
    │                         / /       /
  0 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─/ /─ ─ ─ /
    │                   ReLU  /  ↑    /
 -0.2                        /  small dip
    │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     (GELU goes slightly negative near -1)
    └─────────────────────────────────────── z
       -3    -2    -1     0     1     2     3

  Key difference: GELU has a tiny negative bump around z ≈ -0.5 to -1
                  ReLU is flat zero for all negatives
```

---

#### Step 4 — Why does the tiny negative dip matter?

With ReLU, once z < 0, the gradient is exactly **0** — the neuron gives no signal at all.

With GELU, even slightly negative inputs produce a **small non-zero output and gradient** — the neuron is still "alive" and contributing a tiny bit.

```
  Gradient flow:

  ReLU:   z = -0.5  →  gradient = 0    (dead, no learning signal)
  GELU:   z = -0.5  →  gradient ≈ 0.13 (small but alive!)
```

This gives the network richer, more nuanced gradients to learn from.

---

#### Step 5 — The formula (demystified)

```
  f(z) = z
- Φ(z)
          ↑    ↑
          │    └── Φ(z) = probability that a random normal value < z
          │              = a number between 0 and 1  (the "gate")
          └── the original input (the "signal")

  So GELU = signal × gate
           = z     × (how confident we are z is positive)
```

`Φ(z)` is just a lookup on the bell curve (normal distribution):

```
  Bell curve (normal distribution):

        │        ████
        │      ████████
        │    ████████████
        │  ████████████████
        │████████████████████
        └────────────────────── z
         -3  -2  -1   0   1  2  3

  Φ(-3) ≈ 0.001   (almost nothing to the left of -3)
  Φ( 0) = 0.5     (50% of the curve is left of 0)
  Φ(+3) ≈ 0.999   (almost everything is left of +3)
```

So `Φ(z)` smoothly goes from 0 → 1 as z goes from very negative → very positive — exactly the soft gate we want.

---

#### Summary

```
  ┌─────────────────┬──────────────────────────────────────────┐
  │                 │                                          │
  │  ReLU           │  Hard switch: off below 0, on above 0   │
  │  GELU           │  Soft gate: probability-weighted pass    │
  │                 │                                          │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Pros            │  Smoother gradients near zero            │
  │                 │  Slightly negative outputs = richer info │
  │                 │  Default in BERT, GPT, most modern LLMs  │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Cons            │  Slower than ReLU (more math)            │
  │                 │  Harder to explain intuitively           │
  └─────────────────┴──────────────────────────────────────────┘
```

**Best used for:** Transformers and modern LLMs (GPT, BERT, LLaMA all use GELU or a variant).

---

#### Leaky ReLU vs GELU

```
  Input x = -1.0:
  Leaky ReLU → 0.01 × (-1) = -0.010   (fixed mechanical slope)
  GELU       → -1  × Φ(-1) = -0.159   (smooth, stronger signal)
```

| | Leaky ReLU | GELU |
|---|---|---|
| Negative slope | Fixed (0.01) | Smooth, input-dependent |
| Gradient | 0.01 or 1.0 (hard step) | Continuously varying |
| Speed | Faster (no exp/erf) | Slower |
| Best for | CNNs, edge devices | Transformers, deep nets |

> **Rule:** Use Leaky ReLU for speed/simplicity. Use GELU when training Transformers or networks >40 layers.

---

### 6. Binary Step Function

```
  f(x) = 1, if x >= 0
         0, if x < 0
```

**Analogy:** A light switch — fully ON or fully OFF. No in-between.

```
  Output
    1 │──────────────────────────────
      │                   ↑ threshold = 0
    0 │──────────────────
      └──────────────────────────────► x
        -3   -2   -1    0    1    2

  x = -5  →  output = 0  (off)
  x =  0  →  output = 1  (on)
  x =  3  →  output = 1  (on)
```

| Pros | Cons |
|------|------|
| Simple, binary output | **No gradient** — derivative is 0 everywhere |
| Good for simple binary classifiers | Cannot use backpropagation |
| | Only yes/no — can't handle multi-class |

**Best used for:** Simple threshold-based binary output. Almost never used in modern deep learning.

---

### 7. Linear (Identity) Function

```
  f(x) = ax           f'(x) = a  (constant derivative)
```

**Analogy:** A straight pipe — output is directly proportional to input. No bending.

```
  Output
    4 │                    /
    2 │               /
    0 │          /
   -2 │     /
   -4 │/
      └──────────────────────────────► x
        -4   -2    0    2    4
```

**The fatal problem:**

```
  Layer1 output: h₁ = a·x
  Layer2 output: h₂ = a·h₁ = a·(a·x) = a²·x
  Layer3 output: h₃ = a·h₂ = a³·x
  ...
  Any number of layers = still just c·x  (one line!)
```

No matter how many layers you stack, the whole network collapses into a single linear function.

| Pros | Cons |
|------|------|
| Simple | No non-linearity — can't learn complex patterns |
| Useful at output for regression | Derivative is constant → same weight update regardless of input |
| | Stacking layers is pointless |

**Best used for:** Regression output layer only (predicting a continuous number).

---

### 8. Parameterised ReLU (PReLU)

```
  f(x) = x,   if x >= 0
         a·x, if x < 0        ← 'a' is LEARNED by the network
```

**Difference from Leaky ReLU:** In Leaky ReLU, `a = 0.01` is fixed. In PReLU, `a` is a **trainable parameter** — the network learns the best value during training.

```
  PReLU (a learned, e.g. a=0.05):       vs       ReLU:

  f(x)                                   f(x)
    4│            /                         4│            /
    2│        /                             2│        /
    0│───────/                              0│───────/
  -0.2│ \  /                              0 │───────
      └──────────────► x                    └──────────────► x

   Small slope on left = learned             No slope on left = zero
```

| Pros | Cons |
|------|------|
| Adapts the negative slope to the data | Extra parameter to learn |
| Fixes Dying ReLU | Slight overfitting risk |

---

### 9. ELU (Exponential Linear Unit)

```
  f(x) = x,          if x >= 0
         α(eˣ - 1),  if x < 0       α ≈ 1.0

  f'(x) = 1,          if x >= 0
          f(x) + α,   if x < 0      ← derivative never exactly 0
```

**Key difference from ReLU:** Negative inputs produce a **smooth exponential** output instead of hard zero, so there's always a gradient.

```
  ELU vs ReLU negative side:

  x = -2:   ReLU → 0         (zero gradient, dead neuron risk)
  x = -2:   ELU  → -0.865    (non-zero, gradient flows)

  Output
    2│             /
    0│─────────/
  -1 │....../           ← smooth curve, approaches -α = -1
      └──────────────────► x
       -4   -2   0   2
```

| Pros | Cons |
|------|------|
| Solves Dying ReLU | Computationally expensive (exponential) |
| Smooth negative outputs | Requires tuning α |
| No abrupt zero — richer gradients | Slower than ReLU |

---

### 10. Swish

```
  f(x) = x
- sigmoid(x) = x / (1 + e⁻ˣ)
```

**Analogy:** Like GELU — a smooth self-gated function. "Let input through proportional to how activated it is."

**Recommended:** Only when your network has **more than 40 layers**. For smaller networks, ReLU or GELU are better.

```
  Swish vs ReLU:

  f(x)
    3│                    / (Swish ≈ ReLU for large x)
    1│              /
    0│──────────/
  -0.3│    \../          ← small negative dip (like GELU)
      └──────────────────────────► x
       -4   -2    0    2    4

  Swish is smoother than ReLU and has no hard zero cutoff
```

| Pros | Cons |
|------|------|
| Outperforms ReLU on deep nets (40+ layers) | More compute than ReLU |
| Smooth non-monotonic curve | Overkill for shallow networks |
| Self-gated (no extra parameters) | |

---

### 11. Softplus

```
  f(x) = log(1 + eˣ)       f'(x) = sigmoid(x)
```

**Analogy:** A smooth, rounded version of ReLU — like ReLU with its sharp corner sanded down.

```
  Softplus vs ReLU:

  f(x)
    4│                  / ReLU (sharp corner)
    2│             / /
    0│──────────/ /   Softplus (smooth curve)
      └──────────────────────────► x
       -4   -2    0    2    4

  At x=0: ReLU = 0 (sharp), Softplus ≈ 0.693 (smooth)
  As x→+∞: both ≈ x
  As x→-∞: ReLU = 0, Softplus → 0 (but never exactly 0)
```

**Key property:** Derivative of Softplus = Sigmoid. So if you differentiate Softplus, you get Sigmoid.

| Pros | Cons |
|------|------|
| Always positive output | Computationally expensive |
| Smooth (differentiable everywhere) | Rarely used in practice |
| No dead neurons | |

---

### 12. Softmax (output layer for multi-class)

```
         e^zᵢ
f(zᵢ) = ──────       range: (0,1), all outputs sum to 1
          Σ e^zⱼ
```

**Analogy:** Converts raw scores into a probability distribution — like turning exam marks into percentage scores that add to 100%.

```
  Raw scores z = [2.0, 1.0, 0.1]
  After softmax  = [0.66, 0.24, 0.10]   ← sum = 1.0
```

---

### Quick Decision Guide

```
  What am I building?
        │
        ├── OUTPUT LAYER:
        │       ├── Binary classification?           → Sigmoid
        │       ├── Multi-class classification?      → Softmax
        │       └── Regression (predict a number)?   → Linear (no activation)
        │
        └── HIDDEN LAYERS:
                │
                ├── Default (CNN/MLP)?               → ReLU
                ├── Transformer / LLM?               → GELU
                ├── RNN / LSTM?                      → Tanh
                ├── Dying ReLU problem?              → Leaky ReLU or PReLU
                ├── Deep net (40+ layers)?           → Swish
                ├── Always need gradient?            → ELU
                ├── Simple threshold only?           → Binary Step (rare)
                └── Smooth continuous output?        → Softplus (rare)
```

### All Activation Functions — Master Summary

| Function | Range | Derivative | Problem | Best For |
|----------|-------|------------|---------|----------|
| Binary Step | {0, 1} | 0 always | No backprop | Simple threshold only |
| Linear | (-∞, +∞) | constant | No stacking | Regression output |
| Sigmoid | (0, 1) | 0–0.25 | Vanishing gradient | Binary output layer |
| Tanh | (-1, 1) | 0–1 | Vanishing gradient | RNN hidden layers |
| ReLU | [0, +∞) | 0 or 1 | Dying ReLU | CNN/MLP hidden layers |
| Leaky ReLU | (-∞, +∞) | 0.01 or 1 | Fixed α | When ReLU dies |
| PReLU | (-∞, +∞) | a or 1 | Extra param | Learned negative slope |
| ELU | (-α, +∞) | smooth | Slow compute | Always-alive neurons |
| GELU | (-0.2, +∞) | smooth | Slow compute | Transformers / LLMs |
| Swish | (-0.3, +∞) | smooth | Deep nets only | 40+ layer networks |
| Softplus | (0, +∞) | sigmoid | Slow compute | Smooth positive output |
| Softmax | (0, 1)×n | complex | — | Multi-class output layer |

---
### Is the activation function applied again during backpropagation?

**No — it is NOT re-applied. Its derivative is computed using the value already stored from the forward pass.**

```
  FORWARD PASS:
  ─────────────────────────────────────────────────────
  z = W·x + b              ← linear step
  a = ReLU(z)              ← activation applied → a stored in memory ✓
  output = W2·a + b2


  BACKPROPAGATION:
  ─────────────────────────────────────────────────────
  Gradient arrives at the activation node.

  Chain rule needs:  dL/dz = dL/da
- da/dz
                                        ↑
                                 derivative of ReLU
                                 computed from stored z (NOT re-run)

  For ReLU:   da/dz = 1  if z > 0   (stored z is checked)
                    = 0  if z ≤ 0

  ┌──────────────────────────────────────────────────────────────┐
  │  Forward:  ReLU(z) → produces output a                       │
  │  Backprop: ReLU'(z) → produces gradient multiplier (0 or 1)  │
  │                                                              │
  │  Two different operations. Activation runs once (forward).   │
  │  Its derivative runs once (backward). Never re-run.          │
  └──────────────────────────────────────────────────────────────┘
```

---

## B2. Chain Rule & Backpropagation

### The Big Picture

Training a neural network means answering: **"By how much should I adjust each weight to reduce the error?"**

Backpropagation answers this using the **chain rule** from calculus — systematically working backwards through the network.

---

### Chain Rule (Plain English)

> If `y` depends on `u`, and `u` depends on `x`, then the rate of change of `y` with respect to `x` = (rate `y` changes w.r.t. `u`) × (rate `u` changes w.r.t. `x`).

**Analogy:** Speed of a train through gears.
```
  Engine RPM → Gearbox → Wheel speed

  d(wheel_speed)/d(rpm) = d(wheel_speed)/d(gearbox) × d(gearbox)/d(rpm)
```

**Mathematically:**
```
  dy   dy   du
  ── = ──
- ──
  dx   du   dx
```

For a chain of functions `L = f(g(h(x)))`:
```
  dL   dL   df   dg
  ── = ──
- ──
- ──
  dx   df   dg   dx
```

---

### Forward Pass → Compute Loss

```
  Input x
     │
     ▼
  [Layer 1]  z₁ = W₁·x + b₁
     │
     ▼
  [ReLU]     a₁ = max(0, z₁)
     │
     ▼
  [Layer 2]  z₂ = W₂·a₁ + b₂
     │
     ▼
  [Sigmoid]  ŷ = σ(z₂)
     │
     ▼
  [Loss]     L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

---

### Backward Pass → Compute Gradients

```
  ◄──────────────────────────────────────────

  dL/dW₁ ◄── dL/da₁ ◄── dL/dz₂ ◄── dL/dŷ ◄── dL/dL (=1)

  Each arrow = multiply by local derivative (chain rule)
```

Step by step:

```
  Step 1: dL/dŷ    = -(y/ŷ) + (1-y)/(1-ŷ)          ← derivative of loss
  Step 2: dŷ/dz₂  = ŷ(1-ŷ)                           ← derivative of sigmoid
  Step 3: dz₂/dW₂ = a₁                               ← derivative of linear layer
  Step 4: dz₂/da₁ = W₂
  Step 5: da₁/dz₁ = 1 if z₁ > 0, else 0              ← derivative of ReLU
  Step 6: dz₁/dW₁ = x

  So: dL/dW₂ = dL/dŷ
- dŷ/dz₂
- dz₂/dW₂   (chain rule!)
      dL/dW₁ = dL/dŷ
- dŷ/dz₂
- dz₂/da₁
- da₁/dz₁
- dz₁/dW₁
```

---

### Weight Update

```
  W_new = W_old - learning_rate × dL/dW
            ↑              ↑          ↑
        keep most    step size    direction to
        of old W     (α ≈ 0.001)  reduce loss
```

**Intuition:** The gradient tells you the slope of the loss hill. Subtract it to walk downhill.

```
  Loss
   │
   │    *
   │   / \
   │  /   \        * ← you are here (W, high loss)
   │ /     \      /
   │/       \    /
   └──────────────── W
              ↑
              gradient says "go left" → subtract → reduce loss
```

---

### ⚡ Computing & Updating BIAS During Backpropagation

---

#### **What is BIAS? (Beginner Explanation)**

**Simple analogy: Weights are "amplifiers", bias is a "shift knob".**

```
Think of a line: y = 2x + 3
                   ↑ weight (slope)  ↑ bias (y-intercept)
                   
Weight (2x):    Makes line steeper/flatter
Bias (+3):      Shifts line up/down
```

**In neural networks:**

```
Formula: z = W·x + b
         ↑ weight × input  ↑ bias (added constant)

Purpose of weights: Learn "how much" each input matters
Purpose of bias:    Shift the decision boundary to fit the data

Example:
  Without bias: Network can only learn y = 2x (must pass through origin)
  With bias:    Network can learn y = 2x + 3 (can shift anywhere!)
```

**Why bias matters:**

```
Imagine you need to classify:
  "If temperature > 20°C → turn on AC"
  
Without bias: z = W·temp → always passes through 0°C
  Hard to set the right threshold!

With bias: z = W·temp + b → can shift threshold to exactly 20°C
  Easy! Bias moves the boundary to the right place.
```

**Bottom line for beginners:**
- **Weights**: Learn the pattern (slope, direction)
- **Bias**: Learn the offset (shift, where to place the boundary)
- **Both are updated during backpropagation** — the network learns both

---

**Key concept:** Bias is also a learnable parameter — it needs gradients too!

#### Why Compute Bias Gradients?

**Without bias**, decision boundaries must pass through origin (no flexibility):
```
z = W·x  (no bias)

Network can only rotate/scale:
  ├─ z = 2x
  ├─ z = 0.5x
  ├─ z = -x
  └─ ALL pass through (0, 0)  ✗ Can't shift!
```

**With bias**, boundaries can shift anywhere (full flexibility):
```
z = W·x + b

Network can shift:
  ├─ z = 2x + 1     ↗ shifted up
  ├─ z = 2x - 3     ↗ shifted down
  ├─ z = -x + 5     ↖ shifted up
  └─ Can be placed ANYWHERE  ✓ Flexible!
```

**Real example:**
```
Task: Classify (4,4) as Class A, (5,5) as Class B

WITHOUT bias: z = x₁ + x₂
  (4,4) → z = 8, (5,5) → z = 10
  Can't find threshold to separate them!  ✗

WITH bias: z = x₁ + x₂ - 8.5
  (4,4) → z = -0.5 (Class A)  ✓
  (5,5) → z = +1.5 (Class B)  ✓
  Bias shifts boundary to perfect location!
```

#### How Bias Gradient is Computed

```
Forward:   z = W·x + b
                    ↓
Backward:  dz/db = 1  (always constant!)
           ↓
Chain rule: dL/db = dL/dz × dz/db
                  = dL/dz × 1
                  = dL/dz

For a batch:
  dL/db = sum(dL/dz over all samples)
```

#### Bias Update (Like Weight Update)

```
b_new = b_old - learning_rate × dL/db
         ↑          ↑           ↑
      old bias   step size   gradient (tells us how much bias affects loss)
```

#### Key Difference: Weight vs Bias Gradients

| Parameter | Gradient | Depends On | Example |
|-----------|----------|-----------|---------|
| **Weight** | dL/dW = dL/dz × x | Input value | If x=0, then dL/dW=0 (no update!) |
| **Bias** | dL/db = dL/dz × 1 | Nothing | Always updates (if dL/dz ≠ 0) |

**Why different?**
```
Weight: z = W·x + b
            ↑
            depends on x, so gradient involves x

Bias:   z = W·x + b
                ↑
                constant (1), independent of x
```

**Intuition:**
- Weights learn "gain" (amplification) — proportional to input strength
- Bias learns "DC offset" (shift) — same for all inputs

---

### ❓ Three Common Questions About Bias & Weights

#### Q1: Is Bias Applied in ALL Layers During Backpropagation?

**YES — bias exists in EVERY dense/fully-connected layer (if enabled).**

```
Network architecture:
  Input (4 features)
        ↓
  [Dense Layer 1]  z₁ = W₁·x + b₁          ← bias b₁ here!
        ↓                                     ↓ (backprop computes dL/db₁)
  [Dense Layer 2]  z₂ = W₂·a₁ + b₂         ← bias b₂ here!
        ↓                                     ↓ (backprop computes dL/db₂)
  [Dense Layer 3]  z₃ = W₃·a₂ + b₃         ← bias b₃ here!
        ↓                                     ↓ (backprop computes dL/db₃)
  Output (predictions)

During BACKPROPAGATION:
  ∂L/∂b₃ computed  ← Layer 3 bias
  ∂L/∂b₂ computed  ← Layer 2 bias
  ∂L/∂b₁ computed  ← Layer 1 bias
  
  ALL biases get updated!
```

**Exception: Convolutional layers**
```
Conv layer: out = Conv(x) + b  (one bias per filter)
            ↓
Backprop: ∂L/∂b computed
         ↓
Optimizer: b = b - lr * ∂L/∂b

(Convolutional bias is different — shared across spatial dimensions)
```

**Note:** Some layers don't have bias:
```
❌ No bias in:
  ├─ Batch Normalization (uses γ, β instead)
  ├─ Activation functions (ReLU, Sigmoid, etc.)
  └─ Pooling layers (Max/Avg pool)

✓ Has bias in:
  ├─ Dense/Linear layers
  ├─ Convolutional layers
  └─ Recurrent layers (LSTM, GRU)
```

#### Q2: Do Weight and Bias Apply Together?

**YES — they work as a team in the SAME layer.**

```
FORWARD PASS (both applied together):
  Input: x
    ↓
  Linear transformation: z = W·x + b
                         ↑   ↑   ↑
                    weights apply here together
                    bias applies here
    ↓
  Output: z (goes to activation)

BACKWARD PASS (both get updated together):
  Loss gradient: dL/dz
    ↓
  [THROUGH WEIGHTS]              [THROUGH BIAS]
  dL/dW = dL/dz · dz/dW        dL/db = dL/dz · dz/db
        = dL/dz · x                   = dL/dz · 1
    ↓                                 ↓
  W = W - lr · dL/dW            b = b - lr · dL/db
    ↑ updated!                   ↑ updated!

Both move together to minimize loss!
```

**Concrete example:**
```
Single neuron in a layer:

Forward:  z = 0.5·x₁ + 0.3·x₂ - 0.2
              ↑            ↑       ↑
            w₁·x₁        w₂·x₂     b (bias)
          (weight)     (weight)   (bias)

If actual output y = 1, predicted ŷ = 0.3 (error!)
  Loss = (y - ŷ)² = (1 - 0.3)² = 0.49

Backward:
  dL/dz = -1.4  (how much z affects loss)
  
  dL/dw₁ = dL/dz · dz/dw₁ = -1.4 · x₁
  dL/dw₂ = dL/dz · dz/dw₂ = -1.4 · x₂
  dL/db = dL/dz · dz/db = -1.4 · 1 = -1.4
  
Update:
  w₁_new = w₁ - lr · dL/dw₁  ← weight moves
  w₂_new = w₂ - lr · dL/dw₂  ← weight moves
  b_new = b - lr · dL/db     ← bias moves
  
All together to reduce loss!
```

#### Q3: Where is "z = W·x + b" Applied?

**In EVERY dense/linear layer of the network.**

```
Network diagram showing z = W·x + b:

Input ─────────────────────────────────────────
        │
        ├─→ Dense Layer 1:  z₁ = W₁·input + b₁
                                   ├─ W₁: (input_size, hidden_size)
                                   └─ b₁: (hidden_size,)
        ↓
      ReLU(z₁)
        │
        ├─→ Dense Layer 2:  z₂ = W₂·a₁ + b₂
                                   ├─ W₂: (hidden_size, hidden_size)
                                   └─ b₂: (hidden_size,)
        ↓
      ReLU(z₂)
        │
        ├─→ Dense Layer 3:  z₃ = W₃·a₂ + b₃
                                   ├─ W₃: (hidden_size, num_classes)
                                   └─ b₃: (num_classes,)
        ↓
      Softmax(z₃)
        │
        └─→ Output (predictions)
```

**Step by step for a single sample:**

```
Input: x = [1.0, 2.0, 3.0] (3 features)

Layer 1 (3 → 4 neurons):
  W₁ shape: (3, 4)      b₁ shape: (4,)
  z₁ = W₁·x + b₁
     = [0.1·1 + 0.2·2 + 0.3·3,  ...] + [0.5, 0.3, 0.1, 0.2]
     = [1.4,  ...]  + [0.5, 0.3, 0.1, 0.2]
     = [1.9,  ...]  ← z with bias added!
  
  a₁ = ReLU(z₁)

Layer 2 (4 → 2 neurons):
  W₂ shape: (4, 2)      b₂ shape: (2,)
  z₂ = W₂·a₁ + b₂      ← "z = W·x + b" applied AGAIN!
     = [...] + [0.1, 0.2]
  
  a₂ = ReLU(z₂)

Layer 3 (2 → 1 neuron):
  W₃ shape: (2, 1)      b₃ shape: (1,)
  z₃ = W₃·a₂ + b₃      ← "z = W·x + b" applied AGAIN!
     = [...] + [0.05]
  
  output = Sigmoid(z₃)
```

**Key insight:**
```
"z = W·x + b" is the FUNDAMENTAL operation in neural networks.

Every Dense/Linear layer applies this:
  ├─ Input comes from previous layer (x or a_prev)
  ├─ Multiply by weight matrix (W·x)
  ├─ Add bias vector (+b)
  ├─ Pass through activation function
  └─ Output becomes input to next layer

This happens in EVERY layer!
```

**In code:**

```python
# What happens inside a Dense layer:

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = initialize_weights(input_size, output_size)
        self.b = initialize_bias(output_size)
    
    def forward(self, x):
        # THIS IS "z = W·x + b" !!!
        z = x @ self.W + self.b        ← Linear transformation
        
        # Then apply activation
        a = activation(z)
        return a
    
    def backward(self, grad_output):
        # Compute gradients for W and b
        grad_W = grad_output @ input.T  ← dL/dW
        grad_b = sum(grad_output)        ← dL/db (sum over batch)
        
        # Update W and b
        self.W = self.W - lr * grad_W
        self.b = self.b - lr * grad_b
        
        # Propagate gradient to input
        grad_input = grad_output @ self.W.T
        return grad_input
```

---

### Summary: Bias & Weight Interaction

```
┌─────────────────────────────────────────────────────┐
│              FORWARD PASS (All Layers)              │
├─────────────────────────────────────────────────────┤
│  Layer i:                                           │
│  ├─ Input: a_{i-1}                                 │
│  ├─ Linear: z_i = W_i · a_{i-1} + b_i  ← BOTH!    │
│  ├─ Activation: a_i = f(z_i)                       │
│  └─ Output: a_i → next layer                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│             BACKWARD PASS (All Layers)              │
├─────────────────────────────────────────────────────┤
│  Layer i:                                           │
│  ├─ Input: dL/da_i (from next layer)               │
│  ├─ Through activation: dL/dz_i                     │
│  ├─ Gradient for W: dL/dW_i = dL/dz_i · a_{i-1}   │
│  ├─ Gradient for b: dL/db_i = dL/dz_i · 1  ← BOTH!│
│  ├─ Update W: W_i = W_i - lr · dL/dW_i            │
│  ├─ Update b: b_i = b_i - lr · dL/db_i            │
│  └─ Output: dL/da_{i-1} (to previous layer)       │
└─────────────────────────────────────────────────────┘

ALL LAYERS:  W and b applied together, updated together
```

---

### Backprop in Code (Conceptual)

```python
# PyTorch does all this automatically with autograd!

loss = criterion(output, target)   # forward pass computes loss
loss.backward()                    # backward pass: chain rule applied automatically
optimizer.step()                   # update weights: W = W - lr * grad
optimizer.zero_grad()              # clear gradients for next iteration
```

---



**Why stored values matter — the Dying ReLU example:**

```
  Forward:   z = -2.5  →  ReLU(-2.5) = 0    ← stored: z was negative

  Backprop:  da/dz = 0  (because stored z = -2.5 < 0)
             → gradient = 0
             → W gets zero update
             → neuron is "dead" — confirmed by the stored z value,
               not by re-running the activation
```

**Activation derivatives at a glance:**

```
  Activation  │ Forward f(z)      │ Backprop f'(z) (uses stored z)
  ────────────┼───────────────────┼──────────────────────────────
  ReLU        │ max(0, z)         │ 1 if z>0, else 0
  Sigmoid     │ 1/(1+e⁻ᶻ)        │ f(z)·(1−f(z))   ← reuses stored f(z)
  Tanh        │ (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)│ 1 − f(z)²       ← reuses stored f(z)
  GELU        │ z·Φ(z)           │ Φ(z) + z·φ(z)   ← reuses stored z
```

Notice: Sigmoid and Tanh reuse their own **forward output** `f(z)` to compute the derivative — another reason forward values must be stored.

---

## B3. Vanishing vs Exploding Gradients

### Why Gradients Go Wrong

In deep networks (many layers), the chain rule **multiplies** many gradients together.

```
  dL/dW₁ = δ_L × δ_(L-1) × δ_(L-2) × ... × δ_1

  If each δ < 1  →  product → 0        (VANISHING)
  If each δ > 1  →  product → ∞        (EXPLODING)
```

---

### Vanishing Gradient

**What happens:** Gradients shrink to near-zero as they travel back through layers. Early layers receive almost no signal and stop learning.

```
  10-layer network with sigmoid activations:

  Layer 10  gradient = 1.0
  Layer  9  gradient = 0.25   (sigmoid max gradient is 0.25)
  Layer  8  gradient = 0.063
  Layer  7  gradient = 0.016
  Layer  6  gradient = 0.004
  Layer  5  gradient = 0.001
  Layer  4  gradient = 0.00025  ← almost zero
  Layer  3  gradient ≈ 0.00006  ← dead
  Layer  2  gradient ≈ 0.00001  ← dead
  Layer  1  gradient ≈ 0.000003 ← completely dead!
```

**Symptoms:**
- Training loss barely decreases
- Early layers have weights that never change
- Network behaves like a shallow model

**Root cause:** Sigmoid and tanh **saturate** — their gradients are < 0.25 and approach 0 at extremes.

```
  Sigmoid gradient (derivative):

  f'(z)  0.25│     ▲ max gradient
             │    /|\
             │   / | \
          0  │../  |  \..  z
              ─────────────►
             -5   0    5

  At z = ±5, gradient ≈ 0.006  → almost no signal!
```

**Solutions:**

```
  ┌──────────────────────────────────────────────────────┐
  │ Problem          │ Solution                           │
  ├──────────────────┼────────────────────────────────────┤
  │ Sigmoid/Tanh     │ Use ReLU / GELU instead            │
  │ Very deep nets   │ Residual connections (skip layers) │
  │ Poor init        │ Xavier / He initialization         │
  │ No normalization │ Batch Normalization                 │
  └──────────────────┴────────────────────────────────────┘
```

### Batch Normalization (Fix for Unnormalized Layer Outputs)

#### **What it does: One-sentence definition**

Computes mean/variance **per batch** during training, normalizes layer outputs, learns scale (γ) and shift (β) parameters, and maintains running statistics for inference.

---

#### **Question 1: Are weights updated once per batch?**

**Yes.** Process 32 samples together → compute average loss → one gradient → one weight update. Then next batch with different loss → different gradient → different update.

```
Dataset: 1000 records, Batch size: 32

Batch 1 (32 records):  Loss₁ → Gradient₁ → Weights updated once
Batch 2 (32 records):  Loss₂ (different!) → Gradient₂ (different!) → Weights updated once (2nd time)
Batch 3 (32 records):  Loss₃ → Gradient₃ → Weights updated once (3rd time)
...
Batch 31 (8 records):  → Weights updated once (31st time per epoch)

Result: ~31 weight updates per epoch (1000 ÷ 32)
```

Each batch teaches the network something slightly different — this diversity helps escape local minima.

---

#### **Question 2: How are statistics persisted while processing a batch?**

**Statistics are computed once per batch, stored in GPU/CPU memory, used for normalization, then discarded. Running statistics persist separately.**

```
For Batch of 1000 records:
  1. Compute: mean = avg(outputs), var = variance(outputs)
  2. Store: mean/var in RAM (temporary, only this batch)
  3. Use: normalized = (output - mean) / √(var + ε)
  4. Learn: y = γ * normalized + β  (γ and β updated via backprop)
  5. Persist: running_mean = 0.9 × old + 0.1 × batch_mean
  6. Discard: batch statistics thrown away after backward pass
```

| Statistic | Computed | Persists? | Used When |
|-----------|----------|-----------|-----------|
| **Batch mean/var** | Per batch from 1000 samples | No (discarded) | Training only |
| **Running mean/var** | Exponential moving average | Yes (in model) | Inference |

**Code:**
```python
import torch.nn as nn
bn = nn.BatchNorm2d(64)
x = torch.randn(32, 64, 28, 28)  # 32 samples
y = bn(x)

# Inside: mean/var computed from 32 samples
# After backward: running_mean/var updated; batch stats discarded
```

---

#### **Question 3: Do weights update 10 times with 10 batches?**

**Yes, exactly 10 times per epoch.** Dataset 320 records ÷ 32 batch size = 10 batches = 10 weight updates.

```
Epoch 1:
  Batch 1 → Loss₁ → Update 1
  Batch 2 → Loss₂ → Update 2
  ...
  Batch 10 → Loss₁₀ → Update 10 ✓ (10 updates)

Epoch 2:
  Batch 1 → Update 11
  ...
  Batch 10 → Update 20 ✓ (20 updates total)
```

**Batch size tradeoff:**

| Batch Size | Batches/Epoch | Updates/Epoch | Gradient | Training |
|-----------|---------------|--------------|----------|----------|
| 1 | 1000 | 1000 | Noisy, escapes minima | Slow but good |
| 32 | 31 | 31 | Balanced | Standard |
| 100 | 10 | 10 | Stable, smooth | Fast but may get stuck |
| 1000 | 1 | 1 | Very stable | Fewest updates |

---

#### **Why it works: The problem & solution**

**Problem:** Outputs at each layer can explode/shrink as data flows through the network:
```
Layer 1: [0.1, 0.9, 0.05]
Layer 2: [0.001, 8.3, 0.0002]      ← activation shrinks/explodes
Layer 3: [0.00001, 94, 0.000001]   ← sigmoid/tanh pushed to flat zones
         ↓ gradients ≈ 0 → learning stops
```

**Solution:** Normalize outputs to mean=0, var=1, then let network learn scale/shift:
```
Layer outputs: [0.001, 8.3, 0.0002]
After norm:    [−1.2, 0.8, −1.5]     ← healthy range
After scale/shift: [−0.6, 1.2, −0.8]  (γ and β learned via backprop)
Result: Gradients flow ✓
```

---

#### **Key concepts table**

| Concept | Training | Inference |
|---------|----------|-----------|
| **Statistics used** | Batch mean/var (computed fresh) | Running mean/var (accumulated) |
| **Learnable params** | γ (scale), β (shift) | Same γ, β used |
| **Regularization** | Different stats per batch → randomness helps learning | N/A |
| **Why different?** | Each batch has different distribution | Single sample has no distribution |

**Formula:**
```
x_normalized = (x - μ_batch) / √(σ²_batch + ε)
y = γ × x_normalized + β
running_μ ← 0.9 × old_μ + 0.1 × μ_batch  (exponential moving average)
```

---

**Residual connection (the key fix for very deep nets):**

**The problem:** In a 100-layer network, gradients must travel backwards through all 100 layers. Each layer multiplies the gradient by a small number — by the time it reaches layer 1, it has nearly vanished to zero. Early layers stop learning.

**The analogy:** Imagine passing a whisper through 100 people. By person 100, the message is completely lost. Now add a phone line that lets the original speaker talk directly to any person — message preserved.

```
  WITHOUT residual (deep net suffers):

  x → [Layer 1] → [Layer 2] → [Layer 3] → ... → [Layer 100] → output
                                                                   │
  Backprop: gradient must pass through ALL 100 layers ←───────────┘
            shrinks at every step → early layers get ~0 gradient


  WITH residual connection (skip/shortcut):

  Step 1: Input x enters. It takes TWO simultaneous paths.

                     ┌─── Path A (skip): x travels unchanged ────────────────┐
                     │                                                        │
          x ─────────┤                                                        ▼
                     │                                                      [ + ] ──► output
                     └─── Path B (learn): x → [Layer 1] → [Layer 2] ──────► [ + ]
                                                   (learns what to ADD)

  Step 2: At the [ + ] node, both paths are ADDED together:

          output = F(x)   +   x
                    ↑          ↑
             what layers    original
             learned to     input
             change         (unchanged)

  Concrete example:

          x = [0.5, 0.3, 0.8]          ← input going into the block

          Path B (layers transform it):
          [Layer 1] → [Layer 2] → F(x) = [0.1, -0.1, 0.05]   ← small tweak

          Path A (skip, untouched):
          x = [0.5, 0.3, 0.8]

          Final output = F(x) + x = [0.6, 0.2, 0.85]  ← refined, not rewritten
```

**What the network learns:** Instead of learning the full transformation `y = F(x)`, it only needs to learn the *residual* (the change): `F(x) = y - x`. If a layer is unhelpful, it can learn `F(x) = 0`, making output `= x` (identity — do nothing).

```
  Gradient flow with skip connection:

  Backprop reaches the + node → gradient splits into TWO paths:
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Path 1: through [Layer] → may shrink (vanishing risk)      │
  │                                                             │
  │  Path 2: through skip ──────────────────────────────────►   │
  │          gradient = 1.0 (no multiplication, no shrinking!)  │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  Combined gradient is always at least as strong as Path 2.
  Early layers are GUARANTEED to receive a usable gradient.
```

**Real impact:** Without residuals, networks deeper than ~20 layers got *worse* accuracy. ResNet introduced skip connections in 2015 and trained a **152-layer** network successfully — enabling modern deep learning.

---

### Exploding Gradient

**What happens:** Gradients grow exponentially. Weights get massive updates, causing training to diverge (loss spikes to NaN or infinity).

```
  RNN with 100 time steps, each multiplying by W (eigenvalue=1.5):

  Step   1: gradient =  1.5
  Step   5: gradient =  7.6
  Step  10: gradient =  57
  Step  20: gradient =  3,325
  Step  50: gradient =  637,621,500
  Step 100: gradient =  406,561,177,535,215  ← NaN territory!
```

**Symptoms:**
- Loss goes to NaN suddenly
- Weights become extremely large
- Model produces garbage predictions

**Solutions:**

```
  ┌──────────────────────────────────────────────────────┐
  │ Solution            │ How it works                   │
  ├─────────────────────┼────────────────────────────────┤
  │ Gradient Clipping   │ Cap gradient norm to max value  │
  │ Weight Init         │ Keep weights small at start     │
  │ Smaller learn rate  │ Smaller updates each step       │
  │ Gradient Norm Mon.  │ Watch and alert if spike        │
  └─────────────────────┴────────────────────────────────┘
```

**Gradient Clipping (most common fix):**

```python
  # After loss.backward(), before optimizer.step():
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  # If ‖gradient‖ > 1.0, scale it down to exactly 1.0
```

---

### Side-by-Side Comparison

```
  ┌─────────────────┬───────────────────────┬───────────────────────┐
  │                 │   VANISHING           │   EXPLODING           │
  ├─────────────────┼───────────────────────┼───────────────────────┤
  │ Gradient size   │ → 0 (tiny)            │ → ∞ (huge)            │
  │ Weight change   │ Near zero (frozen)    │ Massive jumps         │
  │ Training loss   │ Stalls / barely moves │ Spikes / becomes NaN  │
  │ Common in       │ Deep nets w/ sigmoid  │ RNNs, very deep nets  │
  │ Main fix        │ ReLU + residuals      │ Gradient clipping     │
  └─────────────────┴───────────────────────┴───────────────────────┘
```

---

## B4. Optimizers

### What is an Optimizer?

**The optimizer decides HOW to update the weights** after gradients are computed by backprop.

```
  TRAINING LOOP:

  for each batch:
    1. Forward pass  → compute predictions
    2. Loss          → measure error
    3. Backward pass → compute gradients (chain rule)
    4. OPTIMIZER     → update weights   ← optimizer's job
```

**Analogy:** Imagine descending a foggy mountain to reach the lowest valley (minimum loss). The gradient tells you the slope direction. The optimizer decides *how big a step to take* and *which direction exactly*.

```
  Loss Landscape (simplified):

  Loss │
       │  *           *
       │   \         /
       │    \   *   /
       │     \_/ \_/     ← local minima
       │         ↑
       │      GOAL: reach lowest point
       └──────────────────── Weights

  Optimizer = your strategy for walking down this landscape
```

---

### 1. SGD — Stochastic Gradient Descent

```
  W = W - α
- ∇L(W)

  α = learning rate (step size)
  ∇L = gradient of loss w.r.t. weight W
```

**Analogy:** Walking downhill — always take a fixed-size step in the steepest descent direction.

```
  Iteration 1:  W = 5.0 - 0.1 × 2.0  = 4.8
  Iteration 2:  W = 4.8 - 0.1 × 1.8  = 4.62
  Iteration 3:  W = 4.62 - 0.1 × 1.6 = 4.46
  ...slowly converging...
```

| Pros | Cons |
|------|------|
| Simple, well-understood | Slow convergence |
| Low memory | Oscillates in ravines |
| Good generalization | Sensitive to learning rate |
| | Gets stuck in local minima |

---

### 2. SGD + Momentum

```
  v  = β
- v  +  ∇L(W)      ← velocity accumulates past gradients
  W  = W - α
- v

  β = momentum factor (typically 0.9)
```

**Analogy:** A ball rolling downhill — it picks up speed in consistent directions and slows down when direction changes (oscillations dampen out).

```
  Without momentum (zig-zag):        With momentum (smooth):

  Loss │                             Loss │
       │   ↗↘↗↘↗↘                        │   ──────→
       │      ↘↗↘↗↘                       │         ───→
       │          ↘↗↘                     │            ──→
       └──────────────── W               └──────────────── W
```

---

### 3. AdaGrad — Adaptive Gradient

```
  G  = G + (∇L)²                     ← accumulate squared gradients
  W  = W - (α / √(G + ε))
- ∇L      ← divide learning rate by √G
```

**Key idea:** Each parameter gets its own learning rate. **Parameters that update frequently → small lr. Rare parameters → large lr.**

**Analogy:** A student who reviews common topics quickly and spends more time on rarely-seen topics.

| Pros | Cons |
|------|------|
| Great for sparse features (NLP) | G only grows → lr shrinks to 0 |
| Automatic per-parameter lr | Training eventually stops |

---

### 4. RMSProp — Root Mean Square Propagation

```
  G  = β
- G  +  (1-β)
- (∇L)²      ← exponential moving average (forgets old)
  W  = W - (α / √(G + ε))
- ∇L

  β = decay factor (typically 0.9)
```

**Fixes AdaGrad's problem:** Uses a *moving average* of squared gradients instead of accumulating forever, so the learning rate doesn't go to zero.

**Best used for:** RNNs (originally developed by Geoff Hinton for RNNs).

---

### 4b. Adadelta — Adaptive Learning Rate (No Manual LR)

```
  E[g²]_t = ρ
- E[g²]_{t-1}  +  (1-ρ)
- g²_t    ← running avg of squared grads
  Δx_t     = - √(E[Δx²]_{t-1} + ε) / √(E[g²]_t + ε)
- g_t
  E[Δx²]_t = ρ
- E[Δx²]_{t-1} + (1-ρ)
- Δx_t²
```

**Adadelta's big idea:** Unlike AdaGrad (which accumulates ALL past gradients), Adadelta uses only a **window** of recent gradients — so the learning rate never collapses to zero.

```
  AdaGrad problem:
  G keeps growing → lr = α/√G → lr → 0 → training STOPS

  Adadelta fix:
  E[g²] = sliding window average → stays bounded → lr stays alive!
```

**Special feature:** Adadelta requires **no manual learning rate** (α). It estimates a natural unit of scale from the gradient history.

| Pros | Cons |
|------|------|
| No learning rate hyperparameter | More complex to implement |
| Doesn't decay to zero (unlike AdaGrad) | Less popular than Adam in practice |
| Robust to noisy gradients | Slower than Adam on most tasks |

**Use when:** You want adaptive learning but don't want to tune learning rate manually.

---

### 5. Adam — Adaptive Moment Estimation ⭐ (Most Popular)

`**Momentum** - In which Direction weight needs to updated(negative/positive)`

`**Variance((∇L)²)** - Represents the steps needed to reduce the loss. Its squared since we need only positive value`

```
  m  = β₁m  +  (1-β₁)∇L       ← 1st moment: mean of gradients(**Momentum**)
  v  = β₂v  +  (1-β₂)(∇L)²    ← 2nd moment: variance of gradients (**Variance**)

  m̂  = m / (1 - β₁ᵗ)                  ← bias correction
  v̂  = v / (1 - β₂ᵗ)

  W  = W - αm̂ / (√v̂ + ε)

  Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001
```

**Why Bias Correction?**

`m` and `v` are initialized to **0**. At early steps they're pulled toward 0 — making updates too small.

```
  t=1, β₁=0.9, true gradient g=0.5:

  m  = 0.9×0 + 0.1×0.5 = 0.05     ← 10× too small (should be ~0.5)
  v  = 0.999×0 + 0.001×0.25 = 0.00025  ← 1000× too small

  Bias correction divides by (1 - βᵗ):
  m̂  = 0.05   / (1 - 0.9¹)   = 0.05 / 0.1    = 0.5    ✓ corrected
  v̂  = 0.00025 / (1 - 0.999¹) = 0.00025 / 0.001 = 0.25  ✓ corrected

  t=1:   strong correction  (denominator small → scales m̂ up)
  t=100: correction fades   (1 - 0.9¹⁰⁰ ≈ 1.0  → m̂ ≈ m)
```

Only matters in first few hundred steps — once m and v are warmed up, `(1-βᵗ) → 1` and correction has no effect.

---
**beta1 vs beta2 Side by Side**

```
adam_beta1 = 0.9   (gradient EWA — smooths the DIRECTION of updates)
│
├── New gradient:  g  = [+5, -3, +1]    (current step says: go this way)
├── Old average:   m  = [+2, -1, +0.5]  (history says: go THIS way)
└── New average:   m' = 0.9*[2,-1,0.5] + 0.1*[5,-3,1]
                      = [2.3, -1.2, 0.55]  ← smoothed direction


adam_beta2 = 0.98  (gradient² EWA — controls HOW FAST each weight updates)
│
├── If gradient² is large → this parameter changes a lot → slow it down
└── If gradient² is small → this parameter rarely changes → speed it up
```

Result: Adam adapts learning rate INDIVIDUALLY per parameter
        Parameters that oscillate a lot → get smaller effective LR
        Parameters that are stable      → get larger effective LR

---

**β₁=0.9, β₂=0.98 — What it means for weight updates:**

The β values are **retention rates** — how much of the *past* to keep vs the *present* to accept:

```
  β₁ = 0.9  means:
       90% weight on old gradient history  (direction memory)
       10% weight on current gradient

  β₂ = 0.98 means:
       98% weight on old gradient² history (variance memory)
        2% weight on current gradient²
```

**Why adam_beta2 = 0.98 for Fine-Tuning?**

Pretraining (beta2=0.999):
  Very long memory → slow to adapt → stable for millions of steps

Fine-tuning (beta2=0.98):
  Shorter memory → adapts faster to new task-specific gradients
  Model needs to quickly adjust to your domain (shopping recommendations)
  without forgetting what it learned in pretraining

**Why high β₂=0.98 prevents weight spikes:**

**intuition:**
> β₂=0.98 → more importance to previous variance history → current spike gradient contributes only 2% → variance estimate stays stable → effective learning rate stays smooth → **no sudden weight spike**. ✓


```
  LOW β₂ = 0.9 (forgets fast):             HIGH β₂ = 0.98 (remembers long):
  ──────────────────────────────            ─────────────────────────────────
  Step 1: v = 0.9×0  + 0.1×(4²) = 1.6      Step 1: v = 0.98×0  + 0.02×(4²) = 0.32
  Step 2: v = 0.9×1.6 + 0.1×(0.1²) = 1.44  Step 2: v = 0.98×0.32 + 0.02×(0.1²) = 0.314

  effective lr = α / (√v + ε)               effective lr = α / (√v + ε)
               = 0.001 / √1.44 = 0.00083                 = 0.001 / √0.32 = 0.00177

  Spike gradient (g=4) at step 1:
  Low β₂  → v jumps to 1.6 quickly  → lr drops hard → weight barely moves
  High β₂ → v rises slowly to 0.32  → lr stays controlled → smooth update ✓
```

**Intuition — β₂=0.98 as a "slow memory":**

```
  Imagine a noisy stock price — one big spike shouldn't change your
  long-term average much.

  β₂ = 0.9  (short memory):   spike changes average a LOT  → lr unstable
  β₂ = 0.98 (long memory):    spike is diluted over time   → lr stays smooth

  Weight update with spike gradient:

  β₂=0.90:  ──────────/\──────────────  (lr dips sharply at spike)
  β₂=0.98:  ────────────~───────────── (lr barely reacts to spike) ✓
```

**Where β₂=0.98 is commonly used:**

| Setting | β₁ | β₂ | Why |
|---|---|---|---|
| Default Adam | 0.9 | 0.999 | Most stable general use |
| Transformer (Vaswani 2017 paper) | 0.9 | **0.98** | Sparse attention gradients need slow variance memory |
| LLM fine-tuning (DPO/SFT) | 0.9 | 0.95–0.999 | Prevents reward spikes destabilizing early training |
| LoRA fine-tuning | 0.9 | 0.999 | Small adapter — defaults work fine |

---

**Adam Optimizer — Full Picture**

Adam uses 3 separate values to update weights:

```
learning_rate  →  how big each step is          (alpha, α)
adam_beta1     →  smoothing of gradient          (momentum)
adam_beta2     →  smoothing of gradient²         (variance / speed control)
Think of it like driving a car:


learning_rate  =  how hard you press the accelerator
adam_beta1     =  how smoothly you steer (averages direction)
adam_beta2     =  how aware you are of road conditions (adapts speed per parameter)
```

---

### 6. AdamW — Adam with Weight Decay (Modern Default)

#### Step 1 — Cross-Entropy Loss

After LLM predicts next token probabilities via softmax, cross-entropy measures the error.

**When is loss computed?** Per batch — NOT after full sentence, NOT only at `<EOS>`.

**With 12 sentences (batch_size=4):**

| Batch | Sentences | Forward Pass | Weight Update |
|---|---|---|---|
| Batch 1 | s1, s2, s3, s4 | 1 (all tokens in parallel) | 1 |
| Batch 2 | s5, s6, s7, s8 | 1 | 1 |
| Batch 3 | s9, s10, s11, s12 | 1 | 1 |
| **1 Epoch** | **all 12 seen** | **3 total** | **3 total** |

---
**Weights updated per layer** (50-layer LLM, d_model=4096):

> **d_model** = the dimension of every token's vector representation throughout the model.
> Every token (e.g. "cat") is represented as a vector of 4096 numbers at every layer.
> All weight matrices use d_model as their input/output size — it's the model's "width".

| Weight | Shape | Size | Purpose |
|---|---|---|---|
| Wq | 4096 × 4096 | ~16M | Query projection |
| Wk | 4096 × 512 | ~2M | Key projection (GQA) |
| Wv | 4096 × 512 | ~2M | Value projection (GQA) |
| Wo | 4096 × 4096 | ~16M | Output projection |
| FFN W1 | 4096 → 14336 | ~58M | FFN expand |
| FFN W2 | 14336 → 4096 | ~58M | FFN contract |
| LayerNorm | 4096 | ~4K | Normalization scale |
| **Per layer** | | **~152M weights** | |
| **50 layers total** | | **~7.6B weights** | + embeddings ~500M |

---

**Memory per batch during training** (BF16 weights, FP32 optimizer states):

| batch_size | Forward passes | Weight updates | GPU Memory (approx) | Notes |
|---|---|---|---|---|
| batch_size=1 (SGD) | 12 | 12 | ~60 GB | Noisy, slow |
| batch_size=4 (Mini-batch) | 3 | 3 | ~90 GB | Balanced |
| batch_size=12 (Full batch) | 1 | 1 | ~180 GB | Stable, needs more memory |

> Memory = weights(~15GB) + gradients(~15GB) + Adam m/v(~30GB) + activations(scales with batch_size)

> **log(<1) is negative number and log(1) is 0**

```
  L_task = -log( P(correct token) )

  P = 1.0  → loss = 0.0   ✓ perfect
  P = 0.40 → loss = 0.916  moderate
  P = 0.01 → loss = 4.6    wrong → big gradient → large weight update

  Training = Open Book Exam (Teacher Forcing)
  Full answer key known upfront → all positions graded in ONE forward pass:

  Q1: <BOS>             → predict "The"  → loss₁   ┐
  Q2: <BOS> The         → predict "cat"  → loss₂   │ causal mask hides
  Q3: <BOS> The cat     → predict "sat"  → loss₃   │ future answers
  Q4: <BOS> The cat sat → predict "on"   → loss₄   │ per question
  ...all questions answered SIMULTANEOUSLY...       ┘

  <EOS> = just the last question, NOT a trigger for scoring

  L_task = [1.20 + 0.69 + 0.51 + 0.36 + 0.92 + 0.22] / 6 = 0.65
                                                        ↑
                                    6 = number of tokens with loss computed
                                    (<BOS> is input only — never predicted, no loss)

  Why average not sum?
  Short (6 tokens):  sum=3.90  avg=0.65
  Long  (60 tokens): sum=39.0  avg=0.65 ← same avg if equally confident
  Averaging makes loss comparable across different sentence lengths.

  Mini-batch (batch_size=4, each sentence 6 tokens):
  L_task = (sum of all losses across 4×6=24 tokens) / 24
         = average loss per token across the entire batch

  Inference = Closed Book Exam → one token at a time, sequential, no loss
```

---

#### Step 2 — L2 Regularization

Large weights → overfitting. L2 adds a size penalty to the loss:

```
  L_total = L_task  +  λ Σ(W²)
            (0.65)     (penalty strength)

  Σ(W²) = sum of squares of ALL weights in the model:
          Wq, Wk, Wv, Wo (attention), FFN W1/W2, LayerNorm, embeddings — all layers
          NOT selective — includes every single learned number

  Example — 3 weights [2.5, 0.8, 1.2]:
  Σ(W²)   = 2.5² + 0.8² + 1.2² = 8.33
  penalty = λ × 8.33 = 0.01 × 8.33 = 0.083
  L_total = 0.65 + 0.083 = 0.733

  For 70B weights (80 layers):
  Σ(W²) = one giant sum of 70 billion squared numbers
  Most weights small → tiny squares. Large weights rare → contribute more → model learns small weights ✓

  Gradient:  ∇L_total = ∇L_task + 2λ·W   ← d/dW(λW²) = 2λW (calculus)
```

> **LLaMA-3 70B:** 80 layers × ~618M weights/layer ≈ 70B weights total
> Training memory: ~570GB (weights 140GB + gradients 140GB + Adam m/v 280GB + activations ~50GB)
> Σ(W²) computed as a fused GPU kernel — negligible cost vs 80-layer forward+backward

---

#### Step 3 — Problem with Adam + L2

**1. L2 penalty gets mixed into the gradient:**
```
  ∇L_total = ∇L_task + 2λ·W
               ↑           ↑
         prediction     L2 penalty   ← both merged into ONE number
           error
```
> L2 goal: shrink every weight by a fixed amount (λ) equally.
> But mixing it into the gradient means Adam will touch it — and Adam is NOT fair to all weights.

**2. Adam scales updates differently per weight (that's its feature — now a bug here):**
```
  Adam tracks how often each weight gets updated:
  → Active weight (used often, large gradient):  v̂ is large  → 1/√v̂ is small
  → Rare   weight (used rarely, small gradient): v̂ is small  → 1/√v̂ is large
```

**3. The penalty gets distorted unevenly:**

`We are include λ in momentum and in variance which is in numberator and denominator hence weight decay(λ) is ignored`

```
  Recap — Adam momentum & variance (∇L_total already contains L2 penalty):
  m  = β₁·m + (1-β₁)·∇L_total   ← smoothed gradient — penalty mixed in
  v  = β₂·v + (1-β₂)·∇L_total²  ← smoothed squared gradient — penalty mixed in
  m̂  = m / (1 - β₁ᵗ)            ← bias-corrected momentum
  v̂  = v / (1 - β₂ᵗ)            ← bias-corrected variance

┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️ WHERE IS ∂Task/∂W IN THIS FORMULA?                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  The answer: ∂Task/∂W is HIDDEN inside ∇L_total!                  │
│                                                                     │
│  ∇L_total = ∂Task/∂W + λ·2W                                        │
│            ↑           ↑                                            │
│       TASK PART    PENALTY PART                                    │
│       (what you    (L2 penalty)                                    │
│        want)       (mixed in!)                                     │
│                                                                     │
│  So when the formula shows:                                        │
│    m = β₁·m + (1-β₁)·∇L_total                                     │
│         ↑              ↑                                            │
│       old value   THIS contains BOTH task + penalty!               │
│                                                                     │
│  Breaking it down:                                                 │
│    ∇L_total = ∂Task/∂W + λ·2W                                      │
│                                                                     │
│    m = β₁·m + (1-β₁)·[∂Task/∂W + λ·2W]                            │
│         ↑         ↑    ↑            ↑                              │
│       old     weight  TASK part   PENALTY part                     │
│       momentum  on past   (what we want)  (corrupts it!)           │
│                                                                     │
│  Example with numbers:                                             │
│  ────────────────────────────────────────────────                 │
│                                                                     │
│  Weight W = 0.5                                                    │
│  ∂Task/∂W = 0.8  (this is what we actually want to use)           │
│  λ·2W = 0.1 × 2 × 0.5 = 0.1  (penalty term)                       │
│                                                                     │
│  ∇L_total = 0.8 + 0.1 = 0.9  ← This is what gets used!            │
│             ↑    ↑                                                  │
│         HIDDEN inside ∇L_total                                     │
│                                                                     │
│  m = 0.9 × m + 0.1 × 0.9                                           │
│           ↑      ↑    ↑                                             │
│         old    weight [0.8 task + 0.1 penalty]                     │
│       momentum            ↑ MIXED!                                 │
│                                                                     │
│  Problem:                                                          │
│  ────────                                                          │
│  We wanted m to reflect ∂Task/∂W = 0.8 ONLY                        │
│  Instead, m got 0.9 (contaminated with penalty!)                   │
│                                                                     │
│  This causes momentum to learn WRONG direction when penalty is     │
│  strong, leading to unequal regularization across weights.         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

  W = W - α m̂ / (√v̂ + ε)     ← 1/√v̂ scales EVERYTHING including the L2 penalty

  Active weight (Wq — large ∇, updated every token): v̂=100  → 1/√v̂=0.10 → penalty shrunk to 10%  ✗
  Rare   weight (obscure token — small ∇):            v̂=0.01 → 1/√v̂=10   → penalty blown up 10×   ✗
```

**4. Result — regularization is backwards:**
```
  Active weights (Wq, FFN — most impactful) → LEAST regularized → can overfit  ✗
  Rare   weights (obscure tokens)           → MOST  regularized → over-shrunk  ✗
```
> L2 wanted equal treatment. Adam gave unequal treatment. That's the bug.

---

#### 🔍 Understanding Weight Decay (Regularization) in AdamW — For Beginners

**What is regularization?**

Regularization = preventing the model from learning patterns that are TOO specific to the training data.

**Analogy: Student studying**
```
WITHOUT regularization:
  Student memorizes every exam question + answer word-for-word
  ├─ Perfect score on training exam ✓
  └─ Fails when asked different questions ✗

WITH regularization:
  Student understands concepts, not just memorizes
  ├─ Good score on training exam ✓
  └─ Can answer different questions ✓
```

**In neural networks:**
```
WITHOUT regularization:
  Weights become HUGE (very specific to training data)
  ├─ Fits training data perfectly ✓
  └─ Fails on new data (overfitting) ✗

WITH regularization (weight decay):
  Weights stay SMALL (learns general patterns)
  ├─ Still fits training data well ✓
  └─ Works better on new data ✓
```

---

#### How Weight Decay Works (Simple Explanation)

**The idea:** Penalize large weights. Keep them small.

```
Loss = Task Loss + λ × sum(weights²)
       ↑          ↑     ↑
   how wrong   penalty  how big
   predictions   strength  are weights
   
Example:
  Original Loss = 2.5 (prediction error)
  Weight penalty = 0.1 × (0.5² + 0.3² + 0.8² + ...) = 0.1 × 0.98 = 0.098
  Total Loss = 2.5 + 0.098 = 2.598
  
If weights get bigger: penalty grows
  Original Loss = 2.5
  Weight penalty = 0.1 × (1.5² + 2.3² + 3.8² + ...) = 0.1 × 18.2 = 1.82
  Total Loss = 2.5 + 1.82 = 4.32  ← MUCH WORSE!
  
Optimizer says: "Getting worse! Shrink weights!" → updates make weights smaller
```

**Visual:**
```
Training loss (prediction error)
           ↗ gets worse if weights too big
          /
    ────────  ← sweet spot (small weights, good fit)
   /
 ─
(weights too small → underfitting)
```

---

#### Why AdamW Uses weight_decay Instead of L2 Penalty

**Problem with adding L2 penalty directly:**

```
Loss = Task Loss + λ × sum(weights²)
       
Then gradient:
∂Loss/∂W = ∂Task/∂W + λ × 2W
           ↑
        gets mixed into Adam's adaptive learning rate
        
In Adam:
  v = 0.999 × v + 0.001 × (gradient)²
      
If gradient is large (from task) + λ × 2W (from penalty):
  The penalty gets SCALED DOWN by Adam's adaptive rate ✗
  
Result:
  - Frequently updated weights: penalty reduced  (bad — most prone to overfitting!)
  - Rarely updated weights: penalty amplified    (bad — over-shrunk!)
```

**Solution — AdamW separates them:**

```
Step 1: Adam updates weights (using ONLY task loss gradient)
  m = 0.9 × m + 0.1 × ∇(Task Loss)
  v = 0.999 × v + 0.001 × ∇²(Task Loss)
  W = W - lr × m / √v

Step 2: Apply weight decay INDEPENDENTLY
  W = W - λ × W    ← same penalty for all weights!
  
Result:
  - All weights shrink equally by factor λ
  - Not affected by Adam's adaptive rates
  - More predictable regularization ✓
```

---

#### Simple Numerical Example

```
Suppose we have 2 weights:
  W₁ = 0.5  (frequently updated, large gradient)
  W₂ = 0.3  (rarely updated, small gradient)

Learning rate (lr) = 0.01
Weight decay (λ) = 0.1
```

**With Adam + L2 (OLD, BUGGY WAY):**

```
Step 1: Compute gradients
  ∇L_task for W₁ = 0.8 (large, frequently updated)
  ∇L_task for W₂ = 0.05 (small, rarely updated)

Step 2: Add L2 penalty to gradient
  ∂L/∂W₁ = 0.8 + 0.1 × 2 × 0.5 = 0.8 + 0.1 = 0.9    ← mixed!
  ∂L/∂W₂ = 0.05 + 0.1 × 2 × 0.3 = 0.05 + 0.06 = 0.11 ← mixed!

Step 3: Adam applies adaptive scaling (simplified)
  For W₁: v₁ ≈ 0.64 (large squared gradient)
          → 1/√v₁ ≈ 0.39 (shrinks gradient!)
          → effective update = 0.9 × 0.39 ≈ 0.35
          → W₁_new = 0.5 - 0.01 × 0.35 ≈ 0.496
          
  For W₂: v₂ ≈ 0.0121 (small squared gradient)
          → 1/√v₂ ≈ 9.09 (amplifies gradient!)
          → effective update = 0.11 × 9.09 ≈ 1.0
          → W₂_new = 0.3 - 0.01 × 1.0 ≈ 0.29

Result:
  W₁ (frequent): shrunk by 0.004  (very little regularization) ✗
  W₂ (rare):     shrunk by 0.01   (too much regularization!)  ✗
  
Opposite of what we want!
```

**With AdamW (NEW, CORRECT WAY):**

```
Step 1: Adam uses ONLY task loss (no penalty mixed in)
  m₁ = 0.1 × 0.8 = 0.08
  m₂ = 0.1 × 0.05 = 0.005
  (adaptive scaling same as above)
  
Step 2: Weight decay applied SEPARATELY
  W₁_after_adam ≈ 0.496
  W₁_after_decay = 0.496 - 0.1 × 0.496 = 0.4464  ← shrunk by 0.05 (equal rate)
  
  W₂_after_adam ≈ 0.29
  W₂_after_decay = 0.29 - 0.1 × 0.29 = 0.261   ← shrunk by 0.029 (equal rate)

Result:
  Both weights shrunk by same PROPORTION (10% each)
  Regular weights: good regularization ✓
  Rare weights: not over-shrunk ✓
  
Much better!
```

---

#### When to Use weight_decay in AdamW

```python
# Typical settings:

# For transformers / LLMs (GPT, BERT, LLaMA):
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=1e-4, 
                              weight_decay=0.01)  # strong regularization

# For CNNs (ResNet, ViT):
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-3,
                              weight_decay=1e-2)  # moderate

# For small models / small data:
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-3,
                              weight_decay=0.05)  # strong (need it more!)

# For large models / large data:
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=3e-4,
                              weight_decay=0.01)  # mild (less needed)
```

**How to choose weight_decay:**
```
Too small (0.001):     Weights grow large → overfitting
Just right (0.01):     Weights stay reasonable → good generalization
Too large (0.1):       Weights too small → underfitting

Rule of thumb:
  ├─ Start with 0.01
  ├─ If overfitting: increase to 0.05
  └─ If underfitting: decrease to 0.001
```

---

#### Key Insight: Why AdamW is Better

```
Adam + L2:      "Frequently updated weights barely get regularized!"
                → Overfitting on important parameters ✗

AdamW:          "All weights shrink equally by weight_decay!"
                → Fair regularization across all parameters ✓
```

This is why **AdamW is the standard for modern deep learning** (BERT, GPT, LLaMA, all use it).

---

#### ⚡ WHERE is Regularization Applied in Adam? (Momentum vs Variance)

**Critical insight:** In Adam+L2, the L2 penalty gets mixed into the GRADIENT, which then affects BOTH momentum AND variance!

**The flow:**

```
┌─────────────────────────────────────────────────────────┐
│                    ADAM + L2 (BUGGY)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Compute gradient (PENALTY MIXED IN!)          │
│  ─────────────────────────────────────────────         │
│  ∂Loss/∂W = ∂Task/∂W + λ × 2W                          │
│             ↑            ↑                              │
│          task loss    L2 penalty                        │
│          gets mixed!  gets mixed!                       │
│                                                         │
│  Step 2: Momentum (uses this mixed gradient)           │
│  ─────────────────────────────────────────            │
│  m = β₁·m + (1-β₁)·∂Loss/∂W                            │
│            ↑                      ↑                     │
│         accumulate    contains BOTH task + penalty      │
│                                                         │
│  Step 3: Variance (uses squared mixed gradient)        │
│  ─────────────────────────────────────────            │
│  v = β₂·v + (1-β₂)·(∂Loss/∂W)²                         │
│            ↑                      ↑                     │
│         accumulate    squared gradient (both mixed!)    │
│                                                         │
│  Step 4: Update (uses both m and v)                    │
│  ─────────────────────────────────────────            │
│  W = W - α·m / (√v + ε)                                │
│           ↑   ↑      ↑                                  │
│        momentum   variance  (both corrupted by penalty!)│
│                                                         │
└─────────────────────────────────────────────────────────┘

RESULT: Penalty affects BOTH momentum AND variance
        → Distorts the adaptive learning rate
        → Creates unequal regularization
```

**Visual of the problem:**

```
Gradient without penalty:       ∂Task/∂W = [0.8, 0.05]
                                          ↑   ↑
                                     large  small

Add L2 penalty:                 ∂Loss/∂W = [0.9, 0.11]
                                          ↑    ↑
                                    both get        (penalty obscures actual task!)
                                    larger

Momentum (accumulates):         m = 0.1 × [0.9, 0.11]
                                        = [0.09, 0.011]

Variance (accumulates squared): v = 0.001 × [0.9², 0.11²]
                                 = 0.001 × [0.81, 0.0121]
                                 = [0.00081, 0.0000121]

Adaptive rate (1/√v):          [1/√0.00081, 1/√0.0000121]
                              = [35, 287]
                                 ↑     ↑
                              HUGE difference! (ratio: 8×)

Weight updates:
  W₁ update = 0.09 × 35 = 3.15
  W₂ update = 0.011 × 287 = 3.16
  
  Same magnitude! But W₂ was supposed to update LESS (small task gradient)
  Instead, it got amplified by variance scaling ✗
```

---

#### ✅ WHERE AdamW Applies Regularization (Completely Separate)

```
┌─────────────────────────────────────────────────────────┐
│                    ADAMW (CORRECT)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: Compute gradient (ONLY task, NO penalty!)     │
│  ─────────────────────────────────────────────        │
│  ∂Loss/∂W = ∂Task/∂W  (CLEAN! No penalty mixed!)       │
│             ↑                                           │
│          task loss only                                 │
│                                                         │
│  Step 2: Momentum (pure task gradient)                 │
│  ──────────────────────────────────────                │
│  m = β₁·m + (1-β₁)·∂Task/∂W                            │
│            ↑                      ↑                     │
│         accumulate    task gradient ONLY (clean!)       │
│                                                         │
│  Step 3: Variance (pure task gradient squared)         │
│  ──────────────────────────────────────                │
│  v = β₂·v + (1-β₂)·(∂Task/∂W)²                         │
│            ↑                      ↑                     │
│         accumulate    task gradient squared (clean!)    │
│                                                         │
│  Step 4: Adam Update (with clean m and v)             │
│  ──────────────────────────────────────                │
│  W_temp = W - α·m / (√v + ε)                           │
│           ↑   ↑   ↑      ↑                             │
│        weights  momentum  variance  (both clean!)       │
│                                                         │
│  Step 5: Weight Decay (SEPARATE, INDEPENDENT!)         │
│  ──────────────────────────────────────                │
│  W = W_temp - λ·W                                       │
│             ↑   ↑                                       │
│    PENALTY applied independently                        │
│    NOT mixed with gradient!                             │
│    Same λ for all weights!                              │
│                                                         │
└─────────────────────────────────────────────────────────┘

KEY: Penalty is applied AFTER Adam step
     Does NOT affect momentum or variance
     All weights shrunk by same amount
```

**Comparison table:**

```
┌──────────────┬─────────────────────────┬──────────────────────┐
│              │   Adam + L2 (Buggy)      │  AdamW (Correct)     │
├──────────────┼─────────────────────────┼──────────────────────┤
│ Gradient     │ ∂Task + λ·2W (mixed!)   │ ∂Task (clean!)       │
│ Momentum     │ Contains penalty         │ Task only            │
│ Variance     │ Contains penalty         │ Task only            │
│ Weight decay │ Distorted by 1/√v       │ Fixed λ (separate)   │
│ Result       │ Unequal regularization  │ Equal regularization │
└──────────────┴─────────────────────────┴──────────────────────┘
```

---

#### Example: Seeing the Difference in Code

```python
# ─────────────────────────────────────────────────────────
# ADAM + L2 (What happens inside)
# ─────────────────────────────────────────────────────────

gradient = compute_gradient(loss_task)                # ∂Task/∂W
gradient += weight_decay * 2 * weights              # Add penalty HERE

# Now gradient is mixed!
# momentum calculation uses MIXED gradient
m = beta1 * m + (1 - beta1) * gradient              # ← contains penalty!

# variance calculation uses MIXED gradient squared  
v = beta2 * v + (1 - beta2) * (gradient ** 2)      # ← contains penalty!

# Update uses both (both corrupted!)
weights = weights - lr * m / (sqrt(v) + eps)       # ← affected by penalty pollution


# ─────────────────────────────────────────────────────────
# ADAMW (What happens inside)
# ─────────────────────────────────────────────────────────

gradient = compute_gradient(loss_task)               # ∂Task/∂W only!

# momentum calculation uses CLEAN gradient
m = beta1 * m + (1 - beta1) * gradient              # ← pure task gradient

# variance calculation uses CLEAN gradient squared
v = beta2 * v + (1 - beta2) * (gradient ** 2)      # ← pure task gradient

# Adam step (clean!)
weights = weights - lr * m / (sqrt(v) + eps)       # ← unaffected by penalty

# Weight decay step (separate, independent!)
weights = weights - weight_decay * weights          # ← penalty applied AFTER
                                                    # All weights shrunk equally!
```

**Key difference:**
```
Adam+L2:   Penalty → Gradient → Momentum & Variance → Weight Update
                      ↑
                   CONTAMINATED

AdamW:     Task → Momentum & Variance → Adam Update → Penalty (separate)
           ↑                                           ↑
        CLEAN                                    INDEPENDENT
```

---

#### Summary: Where Regularization is Applied

| Optimizer | Penalty Applied At | Affects | Result |
|-----------|-------------------|---------|--------|
| **SGD+L2** | Mixed in gradient | Direct weight update | Simple but unequal |
| **Adam+L2** | Mixed in gradient | **BOTH** momentum AND variance | **Distorts adaptive rates** ✗ |
| **AdamW** | AFTER Adam step | Weight values directly | **Fair and equal** ✓ |

**Answer to your question:**
```
In Adam+L2:
  Regularization is MIXED into the gradient
  ↓
  Affects BOTH momentum (m) and variance (v)
  ↓
  Causes unequal regularization (bad!)

In AdamW:
  Regularization is applied AFTER momentum and variance
  ↓
  Does NOT affect momentum or variance
  ↓
  Fair regularization for all weights (good!)
```

This is why **AdamW separates concerns** — task learning and regularization are kept independent!

---

#### Step 4 — AdamW Fix

> **∇L_task** = gradient of the cross-entropy loss only (how wrong the prediction was).
> It tells each weight: "change by this much to predict the correct token better."
> It does NOT include L2 penalty — that's kept separate in AdamW.

AdamW decouples the penalty — applies it directly to weight AFTER the Adam step:

```
  Adam step (learns from prediction error only):
  m = β₁·m + (1-β₁)·∇L_task        ← only prediction error enters Adam
  v = β₂·v + (1-β₂)·(∇L_task)²
  W_temp = W - α·m̂/(√v̂+ε)         ← adaptive weight update from task loss

  Weight decay step (shrink weight independently):
  W = W_temp - λ·W                  ← just multiply: shrink every weight by λ
              ↑
    fixed λ applies equally to ALL weights — bypasses 1/√v̂ entirely ✓
```

| | Adam + L2 | AdamW |
|---|---|---|
| Penalty enters | Inside gradient | After Adam step |
| Active weights | Weakened by `1/√v̂` | Full strength λ ✓ |
| Rare weights | Amplified by `1/√v̂` | Full strength λ ✓ |

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
# weight_decay fused into optimizer step — zero extra compute cost
```

**Used by:** BERT, GPT, LLaMA — de-facto standard for Transformers.

---

### Optimizer Comparison Visual

```
  CONVERGENCE ON A LOSS LANDSCAPE:

  ●=start, ★=optimum

                        SGD          SGD+Mom        Adam
  ●                     ●             ●              ●
  │ (zig-zag)           ↓ ↗           ↘              ↘
  │                     ↘ ↗           ↘              ↘
  ↓ (slowly)            ↓ ↗            ↘             ↘
  ★ (eventually)        ★             ★              ★
                     (slow,zig)   (faster,smooth)  (fastest)

  CONVERGENCE SPEED (typical):

  Adam / AdamW  ████████████████  fastest, most automatic
  RMSProp       ████████████      good for RNNs
  SGD+Momentum  ██████████        good generalization, needs tuning
  AdaGrad       ████████          good for sparse data
  SGD           ██████            slow but often best final model
```

---

### Memory Usage Comparison

```
  ┌─────────────┬──────────────────────────────────────────────┐
  │ Optimizer   │ Extra memory per parameter                   │
  ├─────────────┼──────────────────────────────────────────────┤
  │ SGD         │ 0  (just the gradient)                       │
  │ SGD+Mom     │ 1× (velocity v)                              │
  │ AdaGrad     │ 1× (accumulated G)                           │
  │ RMSProp     │ 1× (moving avg G)                            │
  │ Adam/AdamW  │ 2× (m = momentum, v = variance)              │
  └─────────────┴──────────────────────────────────────────────┘
  For a 7B parameter model: Adam uses ~56GB extra RAM for m & v!
```

---

### Which Optimizer Should I Use?

```
  START HERE
      │
      ├── Transformer / LLM?          → AdamW  (always)
      │
      ├── CNN / MLP (research)?       → SGD + Momentum
      │       (better generalization with right lr schedule)
      │
      ├── RNN / LSTM?                 → RMSProp or Adam
      │
      ├── NLP with sparse features?   → Adam or AdaGrad
      │
      ├── Fine-tuning a pretrained?   → AdamW (with small lr)
      │
      └── Memory constrained?        → SGD or SGD+Momentum
```

---

## B5. Batch GD vs SGD vs Mini-Batch SGD

### The Core Problem

During training you have thousands (or millions) of data samples. **How often do you update the weights?**

```
  TRAINING DATA: 10,000 rows
  ┌─────────────────────────────────────────────────────────┐
  │ row1  row2  row3  ...  row5000  ...  row9999  row10000  │
  └─────────────────────────────────────────────────────────┘

  Three strategies for how many rows to process before updating:
```

---

### Strategy 1 — Batch Gradient Descent (Full GD)

> Process ALL data → compute one loss → update weights once per epoch.

```
  Epoch 1:
  ┌──────────────────────────────────────────────┐
  │ Process all 10,000 rows                      │
  │          ↓                                   │
  │    Compute total loss                        │
  │          ↓                                   │
  │    Update weights (1 update)                 │
  └──────────────────────────────────────────────┘

  10 epochs = 10 weight updates total
```

| Pros | Cons |
|------|------|
| Stable loss curve | Extremely slow if data is huge |
| Guaranteed descent direction | Requires all data in RAM |
| No noise in gradient | 1 update per epoch = very slow learning |

---

### Strategy 2 — Stochastic Gradient Descent (SGD)

> Process ONE row → update weights → repeat.

```
  Epoch 1:
  row1 → loss → update    (update #1)
  row2 → loss → update    (update #2)
  row3 → loss → update    (update #3)
  ...
  row10000 → loss → update (update #10,000)

  10 epochs = 100,000 weight updates total
```

```
  Loss Landscape:
                                              ★ optimum
  Batch GD:   ──────────────────────────────►★    (smooth, slow)
  SGD:        ↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘►★    (noisy, zig-zag)
```

| Pros | Cons |
|------|------|
| Very fast first updates | Very noisy — zig-zags a lot |
| Low RAM usage | May never settle at minimum |
| Can escape local minima | Slow to actually converge |

---

### Strategy 3 — Mini-Batch SGD ⭐ (Most Widely Used)

> Process a BATCH of rows (e.g., 32 or 128) → update weights.

```
  10,000 rows ÷ batch_size 1,000 = 10 batches per epoch

  Epoch 1:
  Batch 1 (rows 1-1000)    → loss → update  (#1)
  Batch 2 (rows 1001-2000) → loss → update  (#2)
  ...
  Batch 10 (rows 9001-10000) → loss → update (#10)

  10 epochs = 100 weight updates total (10 per epoch)
```

```
  Loss Landscape Comparison:
                                              ★ optimum
  Batch GD:      ──────────────────────────►★   (smooth, very slow)
  Mini-Batch:    ──~──~──~──~──~──~──~──►★      (slight noise, fast)
  SGD:           ↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘↗↘►★      (very noisy)

  Mini-Batch is the sweet spot: fast AND stable!
```

| Aspect | Batch GD | SGD | Mini-Batch SGD |
|--------|----------|-----|----------------|
| Updates/epoch | 1 | N (all rows) | N/batch_size |
| RAM needed | All data | 1 row | batch_size rows |
| Noise | None | Very high | Low |
| Speed | Slowest | Fast updates, slow converge | Best of both |
| GPU efficiency | Poor | Poor | Excellent (parallelism) |

**Rule of thumb:** batch_size = 32 or 64 for most tasks. Powers of 2 for GPU efficiency.

---

## B6. Global vs Local Minima

### What is the Loss Landscape?

Imagine drawing a graph with **Weights** on the x-axis and **Loss** on the y-axis:

```
  Loss
   │
   │    *               *
   │     \             /\
   │      \           /  \
   │       \    *    /    \
   │        \__/ \_/      \___
   │          ↑    ↑          ↑
   │       Local Local     Global
   │       Min   Min        Min
   └────────────────────────────► Weights
```

The optimizer's goal: find the **global minimum** (lowest point overall).

---

### Convex vs Non-Convex Functions

```
  CONVEX (simple):                NON-CONVEX (deep learning):

  Loss                           Loss
   │                              │
   │      *                       │  *       *     *
   │     / \                      │   \   * / \ * / \
   │    /   \                     │    \_/   \_/   \_/
   │   /     \                    │     ↑               ↑
   │  /       \                   │  Local            Global
   └────────── W                  └──────────────────── W

  Only 1 global minima.          Many local minima possible.
  Linear & Logistic Regression.  Most deep learning problems.
```

**How to check if a function is convex:** Draw any two points on the curve and connect them with a line. If the line always stays *above* the curve, it's convex.

```
  Convex test:
  Loss
   │    .─────────.
   │   / (line)    \      ← Line is above curve → CONVEX ✓
   │  /             \
   │ /_______________\
   └─────────────────── W

  Non-Convex test:
  Loss
   │           .
   │    .─────────.      ← Line dips below curve → NOT CONVEX ✗
   │   / (line)  / \
   │  /  ___    /   \
   │ /  /   \  /     \
   └─────────────────── W
```

---

### Saddle Points

A **saddle point** is flat (gradient = 0) but it's NOT a minimum:

```
  Loss
   │
   │───────────────────   ← slope = 0 here (saddle point)
   │                  \
   │                   \
   │                    ↘ continues down
   └──────────────────────── W

  Danger: optimizer thinks it's at minimum (gradient ≈ 0)
          but there's a better solution further along!
```

**Solutions to local minima & saddle points:**
- Use **Momentum** (carries you past flat regions)
- Use **Adam** (adapts step size to escape)
- **Random initialization** (start from different points)
- **Larger batch sizes** (smoother loss landscape)

---

## B7. CNN for Beginners

### What is an Image to a Computer?

```
  A 6×6 GRAYSCALE IMAGE (black & white):

  Each cell = 1 pixel (0 = black, 255 = white)

  ┌────┬────┬────┬────┬────┬────┐
  │ 10 │ 20 │ 30 │ 30 │ 20 │ 10 │
  ├────┼────┼────┼────┼────┼────┤
  │ 20 │120 │200 │200 │120 │ 20 │
  ├────┼────┼────┼────┼────┼────┤
  │ 30 │200 │255 │255 │200 │ 30 │
  ├────┼────┼────┼────┼────┼────┤
  │ 30 │200 │255 │255 │200 │ 30 │
  ├────┼────┼────┼────┼────┼────┤
  │ 20 │120 │200 │200 │120 │ 20 │
  ├────┼────┼────┼────┼────┼────┤
  │ 10 │ 20 │ 30 │ 30 │ 20 │ 10 │
  └────┴────┴────┴────┴────┴────┘

  COLOR IMAGE (RGB): 3 separate 6×6 grids (Red, Green, Blue channels)
  Shape: 6 × 6 × 3
```

---

### What is Convolution?

A **filter (kernel)** slides over the image and extracts features (edges, curves, textures).

```
  INPUT IMAGE (6×6)          FILTER (3×3)          OUTPUT (4×4)
  ┌───┬───┬───┬───┬───┬───┐  ┌───┬───┬───┐
  │ 1 │ 0 │ 1 │ 0 │ 0 │ 0 │  │ 1 │ 0 │-1 │
  ├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┤  = Feature Map
  │ 0 │ 1 │ 1 │ 1 │ 0 │ 0 │  │ 1 │ 0 │-1 │
  ├───┼───┼───┼───┼───┼───┤  ├───┼───┼───┤
  │ 1 │ 0 │ 1 │ 0 │ 0 │ 0 │  │ 1 │ 0 │-1 │
  ├───┼───┼───┼───┼───┼───┤  └───┴───┴───┘
  │ ...                   │
  └───────────────────────┘

  Step 1: Place 3×3 filter on top-left of image
  Step 2: Multiply each overlapping value, then sum
  Step 3: Write result into output grid
  Step 4: Slide filter 1 step right, repeat

  Output size formula: (n - f + 1) × (n - f + 1)
                        = (6 - 3 + 1) × (6 - 3 + 1)
                        = 4 × 4
```

**Different filters detect different features:**
```
  Vertical edge filter:     Horizontal edge filter:
  ┌────┬───┬────┐           ┌────┬────┬────┐
  │  1 │ 0 │ -1 │           │  1 │  1 │  1 │
  ├────┼───┼────┤           ├────┼────┼────┤
  │  1 │ 0 │ -1 │           │  0 │  0 │  0 │
  ├────┼───┼────┤           ├────┼────┼────┤
  │  1 │ 0 │ -1 │           │ -1 │ -1 │ -1 │
  └────┴───┴────┘           └────┴────┴────┘

  You don't design these — CNN LEARNS the best filters during training!
```

---

### ⚡ Are Filters Applied Sequentially or in Parallel?

**Answer: LOGICALLY sequential, but HARDWARE parallel**

```
SEQUENTIAL (Logical):
  Filter 1 → 4×4 output (channel 1)
  Filter 2 → 4×4 output (channel 2)  
  Filter 3 → 4×4 output (channel 3)
  Stack all → 4×4×3

PARALLEL (GPU Hardware):
  GPU Core 1: applies Filter 1
  GPU Core 2: applies Filter 2
  GPU Core 3: applies Filter 3
  All at the SAME TIME!

Time with GPU: ~same as 1 filter (not 3× slower)
Speedup: 3× to 512× depending on filter count!
```

**Simple example:**
```
6×6 image + 3 filters (3×3 each)

Filter 1 (vertical edges):    outputs 4×4×1
Filter 2 (horizontal edges):  outputs 4×4×1
Filter 3 (diagonals):         outputs 4×4×1
                              ↓
                      Final: 4×4×3 (stacked)

Logically: one filter processes image → next filter processes same image
Hardware: all filters run simultaneously on GPU cores
```

**Why it matters:**
```
Without GPU (sequential): 8 filters = 8× slower
With GPU (parallel):      8 filters = ~same speed ✓

This is why deep learning needs GPUs!
```

---

### Padding — Keeping Image Size

Without padding, images shrink after each convolution layer:

```
  NO PADDING:                  WITH PADDING (zero-padding):

  6×6 ──[conv]──► 4×4          Add border of zeros:
  4×4 ──[conv]──► 2×2          ┌───┬───┬───┬───┬───┬───┬───┬───┐
  2×2 ──[conv]──► TOO SMALL!   │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                               ├───┼───┼───┼───┼───┼───┼───┼───┤
                               │ 0 │   original 6×6 image   │ 0 │
                               ├───┤                         ├───┤
                               │ 0 │                         │ 0 │
                               ├───┤                         ├───┤
                               │ 0 │   (padded to 8×8)       │ 0 │
                               ├───┼───┼───┼───┼───┼───┼───┼───┤
                               │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │
                               └───┴───┴───┴───┴───┴───┴───┴───┘

  Padded output: (8 - 3 + 1) × (8 - 3 + 1) = 6 × 6 ✓ (same size!)
  Formula with padding p: output = n - f + 2p + 1
```

---

### Max Pooling — Shrinking Smartly

After convolution, we **downsample** to reduce size while keeping important features.

```
  FEATURE MAP (4×4)       AFTER 2×2 MAX POOL (stride 2):

  ┌────┬────┬────┬────┐    ┌────┬────┐
  │  1 │  3 │  2 │  4 │    │  3 │  4 │   ← max of top-left 2×2 = 3
  ├────┼────┼────┼────┤    │    │    │      max of top-right 2×2 = 4
  │  5 │  6 │  1 │  2 │    ├────┼────┤
  ├────┼────┼────┼────┤    │  9 │  8 │   ← max of bottom-left 2×2 = 9
  │  3 │  9 │  8 │  7 │    │    │    │      max of bottom-right 2×2 = 8
  ├────┼────┼────┼────┤    └────┴────┘
  │  4 │  2 │  6 │  5 │
  └────┴────┴────┴────┘

  4×4 → 2×2 (reduced by 75%!)

  Why MAX? It keeps the strongest signal (most activated feature).
  Why not MEAN? Mean dilutes strong signals with weak ones.

  Other types: Average Pooling (takes mean), Min Pooling (takes min)
```

---

### Full CNN Architecture Flow

```
  INPUT IMAGE
       │
       ▼
  ┌─────────────────┐
  │  CONV Layer 1   │  ← Learns simple features (edges, colors)
  │  (e.g., 32 filters) │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  ReLU           │  ← Activation (non-linearity)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  MAX POOL       │  ← Shrink spatial dimensions
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  CONV Layer 2   │  ← Learns complex features (faces, wheels, text)
  │  (e.g., 64 filters) │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  ReLU + POOL    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  FLATTEN        │  ← Turn 2D feature maps into 1D vector
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Fully Connected│  ← Dense layers for classification
  │  + Softmax      │
  └────────┬────────┘
           │
           ▼
      CLASS LABEL
  (cat / dog / car)
```

---

### Data Augmentation

**Problem:** Deep learning needs lots of data. You may only have 1,000 cat photos.

**Solution:** Artificially create more by transforming existing images:

```
  ORIGINAL IMAGE → Multiple Augmented Versions:

  Original:        Flipped H:       Rotated:         Zoomed:
  ┌────────┐       ┌────────┐       ┌────────┐       ┌──────┐
  │  🐱    │  ──►  │    🐱  │       │  🐱↗   │       │ 🐱 big│
  └────────┘       └────────┘       └────────┘       └──────┘

  Also: brightness change, adding noise, color jitter, random crop

  1,000 photos × 8 augmentations = 8,000 effective training samples!
```

Augmentation is applied **only during training**, never on test data.

This teaches the model that a cat is still a cat whether it's flipped, rotated, bright or dim — called **Location/Transformation Invariance**.

---

## B8. Transfer Learning & Data Augmentation

### The Problem Transfer Learning Solves

Training a CNN from scratch on images needs **millions of images** and weeks of GPU time. Most teams don't have that.

**Transfer Learning:** Take a model already trained on millions of images → fine-tune it for your task.

```
  TRAINING FROM SCRATCH (hard):
  ┌──────────────────────────────────────────────┐
  │ Your 1,000 cat/dog photos                    │
  │         ↓                                    │
  │  Train entire network from random weights    │  ← Needs months + huge data
  │         ↓                                    │
  │  Hopefully learns "what a cat looks like"    │
  └──────────────────────────────────────────────┘

  TRANSFER LEARNING (smart):
  ┌──────────────────────────────────────────────┐
  │ VGG16 (trained on 14 million ImageNet images)│
  │  Already knows: edges, textures, shapes, objects │
  │         ↓                                    │
  │  FREEZE these layers (keep learned features) │
  │         ↓                                    │
  │  Replace ONLY the last output layer          │  ← Your 1,000 photos
  │         ↓                                    │
  │  Fine-tune → Cat vs Dog classifier           │
  └──────────────────────────────────────────────┘
```

---

### How Transfer Learning Works

```
  PRETRAINED MODEL (VGG16):
  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  [Conv1]──[Conv2]──[Conv3]──...──[Conv13]──[FC1]──[FC2]──[FC3: 1000 classes] │
  │  ←────────── FROZEN ─────────────────────────────►  ←REPLACE→ │
  │  (edges)  (shapes)  (objects)                       (your task)│
  │                                                            │
  └────────────────────────────────────────────────────────────┘

  YOUR MODEL:
  [Conv1..Conv13: FROZEN]──[FC1: trainable]──[FC2: 2 classes: cat/dog]

  Training only updates FC layers → Fast! Works with small data!
```

**Popular pretrained models:**
| Model | Params | Best For |
|-------|--------|----------|
| VGG16/19 | 138M | Simple tasks, teaching |
| ResNet-50 | 25M | General purpose |
| EfficientNet | 5-65M | Mobile/edge |
| ViT | 86M+ | State-of-art image tasks |

**ImageNet:** Dataset with 14M+ labeled images across 20,000 categories. Models trained on it learn universal visual features.

---

## B9. Object Detection: R-CNN & YOLO

### What is Object Detection?

```
  IMAGE CLASSIFICATION:         OBJECT DETECTION:
  ┌──────────────┐              ┌──────────────────────────┐
  │              │              │  ┌─────┐                 │
  │   🐱         │  → "cat"     │  │ cat │  🐱    ┌──────┐ │
  │              │              │  └─────┘        │ dog  │ │
  └──────────────┘              │                 │  🐶  │ │
                                │                 └──────┘ │
  One label for whole image.    └──────────────────────────┘
                                Bounding box + label for each object.
```

---

### R-CNN (Region-based CNN)

```
  INPUT IMAGE
       │
       ▼
  ┌──────────────────┐
  │ Selective Search │  ← Proposes ~2000 region candidates
  └────────┬─────────┘
           │  (2000 region proposals)
           ▼
  ┌──────────────────┐
  │  CNN Feature     │  ← Extract features from EACH region separately
  │  Extraction      │     (2000 × CNN forward passes = SLOW!)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  SVM Classifier  │  ← Classify each region
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Bounding Box    │  ← Refine box coordinates
  │  Regression      │
  └──────────────────┘

  Problem: ~47 seconds per image!
```

### Fast R-CNN

```
  INPUT IMAGE
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
  ┌──────────┐                         ┌─────────────┐
  │   CNN    │  ← ONE forward pass     │  Selective  │
  │ (whole   │    for whole image      │  Search     │
  │  image)  │                         │ (proposals) │
  └────┬─────┘                         └──────┬──────┘
       │ Feature map                          │ ROIs
       └──────────────┬───────────────────────┘
                      ▼
              ┌───────────────┐
              │  ROI Pooling  │  ← Extract features per region from map
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │  FC Layers    │  → Class + Bounding Box
              └───────────────┘

  Improvement: ~2 seconds per image (20× faster than R-CNN)
  Still slow: proposal step is the bottleneck
```

### Faster R-CNN

```
  INPUT IMAGE
       │
       ▼
  ┌──────────────┐
  │  CNN Backbone│  ← One forward pass
  └──────┬───────┘
         │ Feature Map
         ├──────────────────────────────┐
         ▼                              ▼
  ┌──────────────┐              ┌───────────────┐
  │  RPN         │              │  ROI Pooling  │
  │  (Region     │  ← proposals │  + Classifier │
  │  Proposal    │──────────────►               │
  │  Network)    │              └───────────────┘
  └──────────────┘
  Learned proposals    End-to-end trainable!  ~0.2 sec/image
```

### YOLO — You Only Look Once ⭐

```
  YOLO's big idea: Don't propose regions. Just predict everything at once!

  INPUT IMAGE (416×416)
       │
       ▼
  ┌────────────────────────────────────────┐
  │  Divide image into S×S grid (e.g. 7×7)│
  │                                        │
  │  ┌──┬──┬──┬──┬──┬──┬──┐               │
  │  │  │  │  │  │  │  │  │               │
  │  ├──┼──┼──┼──┼──┼──┼──┤               │
  │  │  │🐱│  │  │🐶│  │  │  ← objects    │
  │  ├──┼──┼──┼──┼──┼──┼──┤               │
  │  │  │  │  │  │  │  │  │               │
  │  └──┴──┴──┴──┴──┴──┴──┘               │
  └────────────────────────────────────────┘
       │
       ▼  (Single CNN forward pass)
       │
       ▼
  Each grid cell predicts:
  - B bounding boxes (x, y, w, h, confidence)
  - C class probabilities
  → All at once in one shot!

  Speed: ~45 fps (real-time video!)
```

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| R-CNN | 47 sec/img | Good | Offline batch |
| Fast R-CNN | 2 sec/img | Better | Near-real-time |
| Faster R-CNN | 0.2 sec/img | Best in class | Production |
| YOLO | Real-time | Good-Great | Video, edge devices |

---

## B10. Word Embeddings & Word2Vec

### The Problem: Computers Can't Read Words

```
  NAIVE APPROACH — One-Hot Encoding:

  Vocabulary: ["King", "Queen", "Man", "Woman", "Cat"]
                  0        1       2       3       4

  "King"  = [1, 0, 0, 0, 0]
  "Queen" = [0, 1, 0, 0, 0]
  "Man"   = [0, 0, 1, 0, 0]
  "Cat"   = [0, 0, 0, 0, 1]

  Problem 1: Vocabulary of 10,000 words → 10,000-dim vector (huge!)
  Problem 2: King and Queen look equally different as King and Cat
             cos_similarity("King","Queen") = 0
             cos_similarity("King","Cat")   = 0
             → No semantic relationships captured!
```

---

### Word Embeddings — Dense Vectors

```
  WORD EMBEDDING — Map each word to a dense vector of features:

  Feature:       Gender  Royal   Age    Food
  "King"    =  [ -0.95,  0.93,   0.70,  0.02 ]
  "Queen"   =  [  0.97,  0.92,   0.68,  0.03 ]
  "Man"     =  [ -0.96,  0.01,   0.50,  0.01 ]
  "Woman"   =  [  0.95,  0.02,   0.48,  0.02 ]
  "Cat"     =  [  0.00, -0.02,   0.40,  0.01 ]

  King - Man + Woman ≈ Queen   ← Famous word analogy!

  Why it works:
  King  - Man  = [ 0.01, 0.92, 0.20, 0.01 ]  (the "royal" part)
  Woman        = [ 0.95, 0.02, 0.48, 0.02 ]
  Sum          ≈ [ 0.96, 0.94, 0.68, 0.03 ]  ≈ Queen ✓
```

---

### Word2Vec — Learning Embeddings Automatically

```
  TRAINING SENTENCE: "The cat sat on the mat"

  SKIP-GRAM approach: predict context words from center word
  Center: "sat"  →  predict: ["cat", "on"]

  ┌───────────────────────────────────────────────────┐
  │                                                   │
  │  "sat" ──► [Embedding Layer] ──► dense vector     │
  │                 (300-dim)                         │
  │                     │                             │
  │                     ▼                             │
  │             [Hidden Layer]                        │
  │                     │                             │
  │                     ▼                             │
  │          [Softmax over vocab]                     │
  │                     │                             │
  │    Predict: P("cat"|"sat"),  P("on"|"sat"), ...   │
  │                     │                             │
  │    Loss = -log P(actual context words)            │
  └───────────────────────────────────────────────────┘

  After training on billions of sentences:
  Nearby words in vector space → similar meanings!
```

---

### Visualizing Word Embeddings (2D)

```
  Reduce 300-dim → 2D using t-SNE:

  Meaning Axis 2
       │
       │   Queen ●  King ●
       │   Woman ●  Man  ●
       │
       │           Paris ●  France ●
       │           Rome  ●  Italy  ●
       │
       └──────────────────────────── Meaning Axis 1

  Observation:
  King - Man + Woman ≈ Queen  (gender direction preserved)
  Paris - France + Italy ≈ Rome (capital-country direction preserved)
```

---

### How to Use Embeddings in Keras

```
  Vocabulary: 10,000 words → Embed to 300-dim vector

  Sentence: "Boy is Good"
  Step 1: "Boy"=2000, "is"=500, "Good"=8000  (word indices)
  Step 2: Embedding layer maps each index → 300-dim vector
  Step 3: Feed sequence of vectors to LSTM/Transformer

  Result: Each word is represented as a meaningful dense vector
          that the model can actually reason about.
```

---

## B11. Sequence-to-Sequence & Bidirectional LSTM

### When Standard RNNs/LSTMs Fall Short

A normal LSTM reads a sequence **left to right** — it knows the past but not the future:

```
  Standard (Unidirectional) LSTM:

  "I love ___"

  x1="I" → [LSTM] → h1
                      ↓
  x2="love" → [LSTM] → h2
                          ↓
  x3="___" → [LSTM] → h3 → predict "Paris"

  Only uses LEFT context (past words).
```

---

### Bidirectional LSTM

```
  BIDIRECTIONAL LSTM reads sequence in BOTH directions:

  Forward:   "I"→"love"→"Paris"→"it"→"very"→"much"
             →h1  →h2    →h3    →h4   →h5    →h6

  Backward:  "much"→"very"→"it"→"Paris"→"love"→"I"
              ←h6'  ←h5'   ←h4' ←h3'    ←h2'   ←h1'

  Final output at each position = [forward_h, backward_h] concatenated

  Example — Named Entity Recognition:
  "I visited Paris last summer"
                 ↑
  Forward sees: "I visited ___"
  Backward sees: "___ last summer"
  Together: "Paris" is clearly a location!

  Use when: full sequence available (classification, NER, sentiment)
  Cannot use for: text generation (future words not yet available)
```

---

### Sequence-to-Sequence (Seq2Seq)

```
  PROBLEM: Input and output sequences have DIFFERENT lengths.
  Example: "Hello" (English, 1 word) → "Bonjour" (French, 1 word)
           "How are you?" (3 words)  → "Comment allez-vous?" (3 words)
           "Good morning" (2 words)  → "Bonjour" (1 word)

  ARCHITECTURE: Encoder + Decoder

  ┌─────────────────────────────────────────────────────────────────┐
  │                        ENCODER                                  │
  │                                                                 │
  │  "Hello" "How" "are" "you"                                      │
  │     │      │    │     │                                         │
  │  [LSTM]→[LSTM]→[LSTM]→[LSTM]──► Context Vector W               │
  │                                  (summary of input)             │
  └─────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                        DECODER                                  │
  │                                                                 │
  │  <START>    "Comment" "allez"  "vous"  <END>                    │
  │     │           │        │       │       │                      │
  │  [LSTM]──►  [LSTM]──► [LSTM]──►[LSTM]──►[LSTM]                 │
  │    ↑           ↑        ↑        ↑                              │
  │    W        "Comment" "allez"  "vous"  (previous output)        │
  └─────────────────────────────────────────────────────────────────┘

  Key insight:
  - Encoder sees ALL input, produces ONE context vector
  - Decoder generates output ONE word at a time
  - Each decoded word is fed back as input to next decoder step
```

---

### Problem with Seq2Seq: Long Sentences

```
  SHORT SENTENCE (works great):
  "Hello" → [Encoder] → W → [Decoder] → "Bonjour"

  LONG SENTENCE (degrades badly):
  "The quick brown fox jumps over the lazy dog
   and then runs away into the forest at sunset"
         ↓
  [Encoder packs 20+ words into ONE vector W]
         ↓
  W = single vector tries to remember EVERYTHING
         ↓
  [Decoder] → translation quality drops significantly!

  BLEU Score (translation quality):
  Sentence length:  5    10    15    20    25    30
  BLEU score:      80%  75%   65%   50%   35%   20%  ← degrades!
```

**Solution:** Attention Mechanism (see Section 5 — Attention & Transformers)

---

### LSTM vs Seq2Seq — When to Use Which

```
  USE LSTM (simple):
  ┌─────────────────────────────────────────────────┐
  │ Input and output have SAME length                │
  │ Examples:                                        │
  │  - Sentiment: "I love this" → Positive           │
  │  - POS tagging: word → tag (1-to-1)              │
  │  - Time series: predict next value               │
  └─────────────────────────────────────────────────┘

  USE SEQ2SEQ:
  ┌─────────────────────────────────────────────────┐
  │ Input and output have DIFFERENT lengths          │
  │ Examples:                                        │
  │  - Translation: English → French                 │
  │  - Summarization: 500 words → 50 words           │
  │  - Chatbot: question → answer                    │
  │  - Text-to-SQL: English → SQL query              │
  └─────────────────────────────────────────────────┘
```

---

## 1. Fundamentals

### 1.1 Perceptron & MLP

**Perceptron** — the simplest unit: multiplies each input by a weight, sums them, adds bias, applies activation:

```
  x1 ──(w1)──┐
  x2 ──(w2)──┼──► [Σ w·x + b] ──► [activation f] ──► output
  x3 ──(w3)──┘

  e.g. x=[1,0,1], w=[0.5,0.3,0.2], b=0.1
       z = 0.5×1 + 0.3×0 + 0.2×1 + 0.1 = 0.8
       output = ReLU(0.8) = 0.8
```

**MLP (Multi-Layer Perceptron)** — stacks multiple layers of perceptrons:

```
  Input      Hidden 1     Hidden 2     Output
  [784] ──► [512, GELU] ──► [256, GELU] ──► [10, Softmax]

  Each layer: Linear(W·x + b) → Activation
  Each layer learns increasingly abstract features:
    Layer 1: raw pixel patterns
    Layer 2: edges and shapes
    Output:  digit class (0–9)
```

**Key properties:**
- **Universal approximation**: 1 hidden layer with enough neurons can approximate any function — but depth is more efficient than width
- **Hierarchical features**: deeper layers learn more abstract concepts

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Usage
model = MLP(input_dim=784, hidden_dims=[512, 256], output_dim=10)
x = torch.randn(32, 784)
out = model(x)  # (32, 10)
print(f"Output shape: {out.shape}")
```

**Interview Insight:**
- Universal approximation theorem: a single hidden layer MLP with enough neurons can approximate any continuous function — but depth is more parameter-efficient than width.
- Depth allows learning hierarchical representations; width provides expressiveness at a single level.

---

### 1.2 Activation Functions

Without activation functions, stacking layers is useless — the whole network collapses to one linear equation. Activations add non-linearity so the network can learn curves and complex patterns. *(Full explanation with diagrams in [B1. Activation Functions](#b1-activation-functions))*

**Quick reference:**
```
  ReLU    → max(0,x)        Fast, sparse. Default for CNNs. Risk: dying neurons.
  GELU    → x·Φ(x)          Smooth gate. Default for Transformers (BERT, GPT).
  Sigmoid → 1/(1+e⁻ˣ)      Output (0,1). Binary classification output only.
  Swish   → x·sigmoid(x)    Self-gated. Better than ReLU in very deep nets.
  Softmax → eˣⁱ/Σeˣʲ        Converts logits → probabilities. Multi-class output.
```

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)

activations = {
    "ReLU":    F.relu(x),
    "GELU":    F.gelu(x),
    "Sigmoid": torch.sigmoid(x),
    "Swish":   x * torch.sigmoid(x),          # SiLU
    "Softmax": F.softmax(x.unsqueeze(0), dim=-1).squeeze(),
}

# ReLU — fast, sparse, dead neuron problem
# GELU — smooth ReLU approx, used in BERT/GPT
# Sigmoid — saturates, vanishing gradient; use only at output for binary
# Swish/SiLU — self-gated, outperforms ReLU in deep nets (EfficientNet)
# Softmax — converts logits to probability distribution (multi-class output)

class SwishActivation(nn.Module):
    """SiLU / Swish: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class GELUApprox(nn.Module):
    """Gaussian Error Linear Unit — tanh approximation"""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))

# Softmax numerical stability example
def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values  # subtract max for stability
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

logits = torch.tensor([[1.0, 2.0, 3.0]])
print(stable_softmax(logits))
```

**Interview Insight:**
- **Dead ReLU problem**: neurons with negative pre-activations always output 0 and have zero gradient — use LeakyReLU or careful initialization to mitigate.
- **GELU vs ReLU**: GELU is preferred in transformers because it is smooth and probabilistically gates inputs. ReLU is preferred in CNNs for speed.
- **Softmax temperature**: `softmax(x/T)` — low T sharpens distribution, high T flattens it. Used in knowledge distillation and sampling.

---

### 1.3 Backpropagation & Chain Rule

Backprop answers: *"by how much should each weight change to reduce the loss?"* It uses the chain rule to propagate the gradient of loss backwards through every layer. *(Full explanation with diagrams in [B2. Chain Rule & Backpropagation](#b2-chain-rule--backpropagation))*

```
  FORWARD:   x → [W1,b1] → ReLU → [W2,b2] → loss
  BACKWARD:  loss → dL/dW2 → dL/da1 → dL/dW1   (chain rule, reverse order)

  Each weight gets:  dL/dW = "how much does loss change if I nudge W slightly?"
  Optimizer then:    W = W - lr × dL/dW
```

**Chain rule in one line:**
```
  dL/dW1 = dL/dz2
- dz2/da1
- da1/dz1
- dz1/dW1
             ↑          ↑          ↑           ↑
          loss grad   W2 grad   ReLU grad    x (input)
```

```python
import torch

# Manual backprop through a simple 2-layer network
# Forward: z1 = W1*x + b1, a1 = relu(z1), z2 = W2*a1 + b2, loss = MSE(z2, y)

torch.manual_seed(42)
x = torch.randn(4, 3)
y = torch.randn(4, 1)

W1 = torch.randn(3, 5, requires_grad=True)
b1 = torch.zeros(5, requires_grad=True)
W2 = torch.randn(5, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# Forward pass
z1 = x @ W1 + b1          # (4, 5)
a1 = torch.relu(z1)        # (4, 5)
z2 = a1 @ W2 + b2          # (4, 1)
loss = ((z2 - y)**2).mean()

# Backward pass — autograd does this automatically
loss.backward()

print(f"dL/dW1 shape: {W1.grad.shape}")  # (3, 5)
print(f"dL/dW2 shape: {W2.grad.shape}")  # (5, 1)

# Manual chain rule for educational purposes
with torch.no_grad():
    dL_dz2 = 2 * (z2 - y) / y.numel()          # dL/dz2
    dL_dW2 = a1.T @ dL_dz2                      # dL/dW2
    dL_da1 = dL_dz2 @ W2.T                      # dL/da1
    dL_dz1 = dL_da1 * (z1 > 0).float()          # ReLU gate
    dL_dW1 = x.T @ dL_dz1                       # dL/dW1

print(f"Manual dL/dW2 matches autograd: {torch.allclose(dL_dW2, W2.grad, atol=1e-5)}")
print(f"Manual dL/dW1 matches autograd: {torch.allclose(dL_dW1, W1.grad, atol=1e-5)}")
```

**Interview Insight:**
- Backprop is just dynamic programming on the computation graph using the chain rule.
- **Vanishing gradient**: gradients shrink exponentially through sigmoid/tanh — solved by ReLU, residual connections, normalization.
- **Exploding gradient**: gradients grow exponentially — solved by gradient clipping.
- PyTorch builds a dynamic computation graph (eager mode); TF1 used static graphs.

---

### 1.4 Loss Functions

The loss function measures **how wrong the predictions are**. Backprop uses its gradient to update weights. Choosing the wrong loss function is a common bug.

#### Joint & Conditional Probability in LLM

**Conditional probability** — what LLM computes at every step:
```
  P("cat" | "The")                 = 0.30  ← given "The", how likely is "cat"?
  P("sat" | "The cat")             = 0.50
  P("mat" | "The cat sat on the")  = 0.40
```

**Joint probability** — likelihood of the entire sentence:
```
  P("The cat sat on the mat")
    = P("The")
    × P("cat" | "The")
    × P("sat" | "The cat")
    × P("on"  | "The cat sat")
    × P("the" | "The cat sat on")
    × P("mat" | "The cat sat on the")

  Chain rule of probability: Joint = product of all conditionals
```

**How they connect to training:**
```
  Maximise joint P(sentence) = minimise -log P(joint)
                              = -[log P("The") + log P("cat"|"The") + ...]
                              = sum of cross-entropy losses per token  ✓
```

| | Joint P | Conditional P |
|---|---|---|
| What | Whole sentence likelihood | Next token given context |
| Formula | `P(w₁,w₂,...,wₙ)` | `P(wₙ\|w₁...wₙ₋₁)` |
| LLM computes | Product of all conditionals | One per token step |
| Used for | Training objective | Inference (token sampling) |

---

```
  Task                        Loss Function         Formula
  ────────────────────────────────────────────────────────────────
  Binary classification       BCE                   -[y·log(p) + (1-y)·log(1-p)]
  Multi-class classification  Cross-Entropy         -log P(correct class)
  Regression                  MSE                   (ŷ - y)²
  Class imbalance             Focal Loss            -(1-p)^γ
- log(p)
  Contrastive / similarity    NT-Xent, Triplet      push same-class together

  LLM next-token prediction   Cross-Entropy         -log P(correct next token)
  DPO alignment               Log-ratio reward      -log σ(β·log π_θ(y_w)/π_ref - ...)
```

**Cross-Entropy with example** *(explained fully in [Part 1 of AdamW section](#part-1--cross-entropy-loss-how-llm-measures-error))*:
```
  Predicted: P("mat")=0.40    Actual: "mat"
  Loss = -log(0.40) = 0.916
  High confidence → low loss. Low confidence → high loss.
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Cross-Entropy Loss ---
ce = nn.CrossEntropyLoss()  # expects raw logits, applies log_softmax internally
logits = torch.randn(8, 10)
targets = torch.randint(0, 10, (8,))
loss_ce = ce(logits, targets)

# Label smoothing variant
ce_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_smooth = ce_smooth(logits, targets)

# --- Mean Squared Error ---
mse = nn.MSELoss()
preds = torch.randn(8, 1)
targets_r = torch.randn(8, 1)
loss_mse = mse(preds, targets_r)

# --- Focal Loss (for class imbalance) ---
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Downweights easy examples, focuses on hard ones.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)           # p_t = probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# --- Contrastive Loss (Siamese networks) ---
class ContrastiveLoss(nn.Module):
    """
    L = (1-y) * 0.5 * D^2 + y * 0.5 * max(0, margin - D)^2
    y=0: similar pair, y=1: dissimilar pair
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = F.pairwise_distance(emb1, emb2)
        loss = (1 - label) * 0.5 * dist**2 + \
               label * 0.5 * torch.clamp(self.margin - dist, min=0)**2
        return loss.mean()

# --- NT-Xent Loss (SimCLR / Contrastive) ---
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy Loss"""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                # (2B, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature                    # (2B, 2B)
    # Mask out self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    # Positive pairs: (i, i+B) and (i+B, i)
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)

focal = FocalLoss(alpha=0.25, gamma=2.0)
loss_focal = focal(logits, targets)
print(f"CE: {loss_ce:.4f}, Focal: {loss_focal:.4f}, MSE: {loss_mse:.4f}")
```

**Interview Insight:**
- **Focal Loss**: originally from RetinaNet. Crucial when positives are rare (e.g., object detection with many background anchors). `gamma=2` is a common default.
- **Label smoothing**: prevents overconfident predictions, acts as regularization. Used in ViT, T5.
- **NT-Xent**: temperature is critical — too low = training instability, too high = poor representations. Typical range: 0.05–0.2.
- CrossEntropyLoss in PyTorch accepts raw logits — do NOT apply softmax before passing.

---

### 1.5 Optimizers

After backprop computes gradients, the optimizer decides **how to update weights**. Different optimizers vary in speed, memory, and generalization. *(Full explanation with diagrams in [B4. Optimizers](#b4-optimizers))*

```
  Optimizer        Update rule (simplified)               Best for
  ─────────────────────────────────────────────────────────────────────
  SGD              W = W - lr·∇L                          CNNs, fine-tuning
  SGD+Momentum     W = W - lr·(β·v + ∇L)                 ImageNet training
  AdaGrad          lr shrinks per param by Σ(∇²)          Sparse NLP features
  RMSProp          lr adapts via EWA of ∇²                RNNs
  Adam             adaptive lr + momentum                 General default
  AdamW            Adam + decoupled weight decay           Transformers, LLMs ⭐
  Lion             sign(gradient) only, less memory       Vision + LLMs
```

**Key rule:** Adam ≠ AdamW. Always use **AdamW** for Transformers (see [AdamW section](#6-adamw--adam-with-weight-decay-modern-default) for why Adam + L2 is broken).

```python
import torch
import torch.nn as nn

model = nn.Linear(128, 10)
params = model.parameters()

# SGD with momentum
sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam — adaptive learning rates per parameter
adam = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

# AdamW — Adam + decoupled weight decay (preferred for transformers)
adamw = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999),
                           weight_decay=0.01, eps=1e-8)

# --- Custom: Lion Optimizer (Google Brain, 2023) ---
class Lion(torch.optim.Optimizer):
    """
    Lion: EvoLved Sign Momentum
    Uses sign of gradient — memory efficient, outperforms AdamW on vision/language tasks.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                m = state['m']
                # Update: sign(beta1*m + (1-beta1)*g)
                update = (beta1 * m + (1 - beta1) * g).sign_()
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
                m.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss

lion = Lion(model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1)

print("Optimizers initialized successfully")
```

**Interview Insight:**
- **Adam vs AdamW**: Adam applies weight decay incorrectly (L2 regularization mixed with adaptive updates). AdamW decouples them — always prefer AdamW for transformers.
- **Lion**: uses only sign of gradient, memory = 1 momentum buffer vs Adam's 2. Reported to match/beat AdamW with ~3x smaller LR.
- **SGD with momentum**: still state-of-the-art for CNNs (ResNet training on ImageNet). Generalizes better than Adam in some CV tasks.
- **Sophia**: second-order optimizer estimating diagonal Hessian — 2x faster than Adam for LLM pre-training.

---

### 1.6 Learning Rate Schedulers

A fixed learning rate is rarely optimal — too high early causes instability, too high late prevents convergence. Schedulers change lr dynamically during training.

**AdamW vs Scheduler — different jobs, work together:**

```
  AdamW controls:      HOW MUCH each weight moves (adaptive per weight via m̂/√v̂)
  Scheduler controls:  WHEN and HOW the global lr changes over training time

  AdamW's lr = starting value → Scheduler decides how that value changes over time
```

| | AdamW alone | Scheduler alone | Both together |
|---|---|---|---|
| Per-weight adaptation | ✓ | ✗ | ✓ |
| Warmup stability | ✗ | ✓ | ✓ |
| Fine convergence at end | ✗ | ✓ | ✓ |
| Spike prevention | ✓ (β₂) | ✗ | ✓ |

```
  Analogy:
  AdamW     = car's steering  (adapts direction per wheel independently)
  Scheduler = accelerator pedal (controls overall speed over the journey)
```

**Why AdamW alone is not enough:**

**1. Early instability — no warmup:**
```
  Step 1: weights random → gradients large & chaotic
          m and v near 0 → m̂/v̂ unreliable
          full lr=3e-4 fires immediately → weights jump wildly before learning stabilises
```

**2. Can't converge precisely at the end:**
```
  Step 5000: model is close to minima — loss landscape is flat and narrow:
                    ___
  loss             /   \___/‾‾‾\___
                  /              ↑
                            narrow optimal valley

  AdamW still fires lr=3e-4 → overshoots valley → bounces back and forth → never settles
```

**3. AdamW adapts per-weight, not globally over time:**
```
  v̂ shrinks lr for frequent weights, grows it for rare weights — per weight
  BUT all weights still scale with global lr=3e-4 at every step
  AdamW has no mechanism to say "slow everything down at step 5000"
```

**Numeric example — near minima (step 5000):**
```
  Without scheduler:  update = 3e-4  × m̂/√v̂ = 3e-4  × 0.89 = 2.67e-4  ← overshoots ✗
  With cosine decay:  update = 3e-5  × m̂/√v̂ = 3e-5  × 0.89 = 2.67e-5  ← 10× smaller, settles ✓
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

optimizer.step()   # adaptive per-weight update
scheduler.step()   # reduce global lr on schedule → fine convergence
```

---

```
  WARMUP PHASE:            DECAY PHASE:
  lr                       lr
  │      /‾‾‾\             │‾‾\
  │     /     \            │   \
  │    /       \──────     │    \________
  └────────────────         └────────────
  steps →                  steps →

  Warmup: start small, ramp up   Cosine decay: smoothly reduce to min_lr
  (avoids destabilising           (smoother than step decay)
   attention at step 1)

  Standard LLM schedule = Warmup (1–5% of steps) + Cosine decay
```

**When to use which:**
```
  Cosine Annealing      → General purpose, smooth decay
  Cosine Warm Restarts  → Escape local minima by periodic restarts
  OneCycleLR            → Super-convergence, large batch training
  Linear Warmup+Cosine  → LLM pre-training and fine-tuning standard ⭐
  CyclicLR              → When you want to explore loss landscape
```

```python
import torch
import torch.nn as nn
import math

model = nn.Linear(128, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Cosine Annealing
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# Cosine with Warm Restarts
cosine_warm = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# OneCycleLR (combines warmup + cosine annealing)
one_cycle = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, steps_per_epoch=100, epochs=10
)

# --- Custom: Linear Warmup + Cosine Decay (standard for LLMs) ---
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int,
                                     num_training_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=10000)

# Cyclic LR
cyclic = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=1e-4, max_lr=3e-4, step_size_up=2000, mode='triangular2'
)

print("Schedulers initialized")

# Simulate a few steps
for step in range(5):
    optimizer.zero_grad()
    loss = model(torch.randn(4, 128)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Step {step}: LR = {scheduler.get_last_lr()[0]:.6f}")
```

**Interview Insight:**
- **Warmup**: critical for large models — large LR at the start can destabilize attention layers. Typical: 1–5% of total steps.
- **Cosine decay**: smoother than step decay, widely used. `min_lr` around 10% of peak LR.
- **OneCycleLR**: super-convergence — finds better minima faster. Especially effective with large batch sizes.
- LR and batch size scale together: doubling batch size → double LR (linear scaling rule, Goyal et al.).


---

## 2. Regularization & Optimization

### 2.1 Dropout, DropPath, Normalization

**What is it?** Regularization (Dropout/DropPath) and normalization (BatchNorm, LayerNorm, RMSNorm) techniques that stabilize training and reduce overfitting.

**Why it matters:** Without regularization, deep networks memorize training data. Without normalization, gradient magnitudes explode or vanish across layers, making training unstable.

**How it works:**
- **Dropout**: randomly zeroes neuron outputs with probability `p` during training; at eval time, all neurons are active (scaled by `1-p` during training to maintain expected value).
- **DropPath (Stochastic Depth)**: drops entire residual paths (entire samples in a batch) rather than individual neurons — stronger regularization for deep networks like ViT and Swin.
- **BatchNorm**: normalizes over the batch dimension per channel — fast convergence for CNNs but requires sufficient batch size and is problematic with distributed training.
- **LayerNorm**: normalizes over the feature dimension per sample — batch-independent, standard for Transformers.
- **RMSNorm**: like LayerNorm but without mean subtraction; simpler and ~10% faster, used in LLaMA and Mistral.

```python
import torch
import torch.nn as nn

# --- Standard Dropout ---
dropout = nn.Dropout(p=0.5)
x = torch.randn(4, 512)
# During training: randomly zeroes elements with prob p, scales remainder by 1/(1-p)
# During eval: identity (no dropout applied)

# --- DropPath (Stochastic Depth) — used in ViT, Swin ---
class DropPath(nn.Module):
    """
    Drop entire residual paths (samples in batch) during training.
    More aggressive than dropout, improves regularization in deep networks.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob

# --- Batch Normalization ---
# Normalizes over (N, H, W) for each channel C: mean/var computed per batch
bn = nn.BatchNorm2d(num_features=64)  # for conv features (N, C, H, W)
bn_1d = nn.BatchNorm1d(num_features=512)  # for linear features (N, C)

# --- Layer Normalization ---
# Normalizes over last D dims for each sample — batch-independent
ln = nn.LayerNorm(normalized_shape=512)  # normalizes over last dim
# Used in: Transformers, RNNs — works well with variable-length sequences

# --- RMSNorm (Root Mean Square Layer Norm) ---
class RMSNorm(nn.Module):
    """
    RMSNorm: simpler than LayerNorm — no mean subtraction, only RMS scaling.
    Used in LLaMA, T5, Mistral. ~10% faster than LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight

# --- Group Normalization (GN) ---
# Splits channels into groups, normalizes within each group
# Batch-size independent — great for small-batch detection/segmentation
gn = nn.GroupNorm(num_groups=32, num_channels=256)

x_conv = torch.randn(2, 64, 32, 32)
x_lin = torch.randn(4, 512)

print(f"BatchNorm2d output: {bn(x_conv).shape}")
print(f"LayerNorm output:   {ln(x_lin).shape}")
rms = RMSNorm(512)
print(f"RMSNorm output:     {rms(x_lin).shape}")
```

**Interview Insight:**
- **BatchNorm**: excellent for CNNs but depends on batch statistics — problematic with small batches (< 8) and during inference in distributed settings.
- **LayerNorm**: batch-independent, standard for transformers. `normalized_shape` can be multi-dim (e.g., `(H, W)` for spatial).
- **RMSNorm**: removes mean centering — empirically works as well as LN, is faster. LLaMA series uses it.
- **DropPath rate**: typically increases linearly with depth — layer `i` gets `drop_prob * i / total_layers`.

---

### 2.2 Weight Initialization

**What is it?** The strategy for setting initial values of neural network weights before training begins.

**Why it matters:** Poor initialization causes vanishing or exploding gradients from the very first forward pass, making the network untrainable before any learning occurs.

**How it works:**
- **Xavier/Glorot**: targets activations with symmetric distributions (tanh, sigmoid) — sets variance to `2 / (fan_in + fan_out)` to keep signal scale consistent across layers.
- **Kaiming/He**: designed for ReLU, which zeroes half the outputs — sets variance to `2 / fan_in` to compensate for that 50% loss.
- **GPT-style**: uses `Normal(0, 0.02)` for all linear/embedding weights; residual projection weights are scaled down by `1/sqrt(2 * n_layers)` to prevent cumulative growth through deep residual stacks.
- **Orthogonal init**: preserves gradient norms through matrix operations — often used for RNNs where repeated matrix multiplication over time can otherwise cause blow-up.

```python
import torch
import torch.nn as nn
import math

def init_weights(module: nn.Module):
    """
    Kaiming (He) initialization for ReLU networks.
    Xavier (Glorot) for tanh/sigmoid networks.
    """
    if isinstance(module, nn.Linear):
        # Kaiming uniform: fan_in mode for ReLU
        nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def init_transformer_weights(module: nn.Module, n_layers: int = 12):
    """
    GPT-style initialization:
    - Normal(0, 0.02) for embeddings and linear layers
    - Scale residual projections by 1/sqrt(2 * n_layers)
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Orthogonal init — preserves gradient norms, great for RNNs
def init_orthogonal(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=1.0)

# --- Xavier Uniform (Glorot) ---
W = torch.empty(512, 256)
nn.init.xavier_uniform_(W)       # U[-sqrt(6/(fan_in+fan_out)), ...]
nn.init.xavier_normal_(W)        # N(0, sqrt(2/(fan_in+fan_out)))

# --- Kaiming (He) ---
nn.init.kaiming_uniform_(W, a=0, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(W, mode='fan_out', nonlinearity='relu')

model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))
model.apply(init_weights)
print("Weights initialized")
```

**Interview Insight:**
- **Xavier**: designed for linear/tanh activations — maintains variance across layers with `var = 2 / (fan_in + fan_out)`.
- **Kaiming/He**: designed for ReLU — accounts for the ~50% zeroing with `var = 2 / fan_in`.
- **GPT init**: `std=0.02` empirically chosen; residual path scaling by `1/sqrt(2L)` prevents gradient explosion in deep transformers.
- Bad init → vanishing/exploding gradients even before training starts. Always verify activation variance in early layers.

---

### 2.3 Gradient Clipping & Accumulation

Two separate techniques used together in LLM training:
- **Gradient Clipping** — caps gradient size to prevent training crashes from loss spikes
- **Gradient Accumulation** — simulates large batch sizes on limited GPU memory

**Where in training loop do they happen?**
```
  1. Forward pass        → model predicts tokens
  2. Loss computed       → cross-entropy
  3. loss.backward()     → gradients computed via backprop

  ── Gradient Accumulation HERE ──
  4. (if not last micro-batch) → skip optimizer.step()
     accumulate gradients across N micro-batches

  ── Gradient Clipping HERE ──
  5. clip_grad_norm_()   → cap gradient magnitude (after all gradients summed)

  6. optimizer.step()    → update weights (AdamW)
  7. scheduler.step()    → adjust global lr
  8. optimizer.zero_grad() → clear gradients for next step

  Why this order?
  Accumulation before clipping → need all micro-batch gradients summed first
  Clipping before optimizer   → clip first, then AdamW uses the safe gradient
```

**Gradient Clipping:**
```
  1 step = 1 mini-batch processed = 1 weight update

  Step 1000 = 1000th mini-batch since training started:
  Step 1000: mini-batch [s3997..s4000] → gradient norm = 0.8   → normal → update ✓
  Step 1001: mini-batch [s4001..s4004] → gradient norm = 150.0 → spike! → blow up ✗

  Fix: if norm > max_norm(1.0), scale ALL gradients proportionally
       norm=150 → multiply every gradient by (1.0/150) → norm=1.0 → safe ✓
       Direction of update preserved — only magnitude is capped
```

**Gradient Accumulation:**

`Backpropogation is performed for all mini-btach but optimizer won't be applied, optimizer is applied at the last mini-batch`

```
  Problem: batch_size=128 doesn't fit in GPU (80GB H100 holds ~8 sequences)
  Solution: run 16 micro-batches of 8, accumulate gradients, ONE update at the end

  Micro-batch 1:  forward → loss/16 → backward() → .grad=g₁           → optimizer SKIP
  Micro-batch 2:  forward → loss/16 → backward() → .grad=g₁+g₂        → optimizer SKIP
  ...
  Micro-batch 16: forward → loss/16 → backward() → .grad=g₁+...+g₁₆  → optimizer.step() ✓ → zero_grad()

  Effective batch = micro_batch(8) × accumulation_steps(16) = 128
```

**Why does backprop run without optimizer?**

Backprop and optimizer are TWO separate operations — backprop does NOT change weights, it only computes and stores gradients.

```
  .grad bucket = in-memory storage (GPU/CPU RAM) that ACCUMULATES gradients

  Backprop  = computes gradient, stores in weight.grad in RAM
  Optimizer = reads weight.grad from RAM, updates weights

  Micro-batch 1:  forward → loss → backward() → w.grad = g₁                (RAM)
  Micro-batch 2:  forward → loss → backward() → w.grad = g₁ + g₂           (added, not replaced)
  ...
  Micro-batch 16: forward → loss → backward() → w.grad = g₁ + ... + g₁₆   (full bucket)
                                                  ↓
                                         optimizer.step()   (reads from RAM)
                                         optimizer.zero_grad() (empties bucket)

  Why accumulate before updating?
  Single batch gradient  = noisy  → zigzag path to minima        ✗
  Accumulated (16 batch) = stable → one smooth, well-directed update ✓

  Key: .grad exists ONLY in RAM during training — it's temporary, not saved to disk
       (saved model weights do NOT include gradients)
```

**Where each is used:**

| Technique | Problem it solves | Used in |
|---|---|---|
| Gradient Clipping | Loss spikes → exploding gradients | All LLM training, RNNs |
| Gradient Accumulation | Large batch doesn't fit in GPU | LLM fine-tuning, low-memory setups |
| Both together | Standard LLM training recipe | GPT, LLaMA, BERT pre-training |

```python
import torch
import torch.nn as nn

model = nn.Linear(256, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# --- Gradient Clipping ---
def training_step_with_clipping(model, optimizer, x, y, max_norm=1.0):
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    # Clip by global L2 norm across all parameters
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    return loss.item(), total_norm.item()

# --- Gradient Accumulation (simulate large batch with small GPU memory) ---
def training_with_accumulation(model, optimizer, dataloader,
                                accumulation_steps: int = 4,
                                max_norm: float = 1.0):
    model.train()
    optimizer.zero_grad()

    for step, (x, y) in enumerate(dataloader):
        # Scale loss by accumulation steps so effective loss magnitude is correct
        loss = nn.CrossEntropyLoss()(model(x), y) / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()
            print(f"Step {(step+1)//accumulation_steps}: loss={loss.item()*accumulation_steps:.4f}")

# Simulated dataloader
class FakeLoader:
    def __iter__(self):
        for _ in range(8):
            yield torch.randn(8, 256), torch.randint(0, 10, (8,))

training_with_accumulation(model, optimizer, FakeLoader(), accumulation_steps=4)
```

**Interview Insight:**
- **Gradient clipping**: standard `max_norm=1.0` for transformers. Prevents catastrophic parameter updates from loss spikes.
- **Accumulation**: effective batch size = `micro_batch * accumulation_steps * num_gpus`. Critical for LLM training where global batch is 256–4096 sequences.
- BatchNorm statistics are computed per micro-batch during accumulation — this creates a mismatch. Use SyncBN or LayerNorm to avoid this.
- With `torch.cuda.amp` (mixed precision), scale gradients before clipping using `scaler.unscale_(optimizer)`.

---

### 2.4 Mixed Precision Training

**What is it?** Training neural networks using lower-precision floating point formats (FP16 or BF16) for most operations while keeping critical computations in FP32.

**Why it matters:** Halves memory usage for activations and parameters, and speeds up matrix multiplications on modern GPUs (Tensor Cores), enabling larger models or batch sizes.

**How it works:**
- **FP16**: 5-bit exponent, 10-bit mantissa — narrow range risks overflow/underflow. `GradScaler` multiplies the loss by a large scale factor before backward, pushing gradients into the representable FP16 range, then divides back before the optimizer step.
- **BF16**: 8-bit exponent, 7-bit mantissa — same dynamic range as FP32, so no gradient scaling is needed. Preferred on A100/H100 hardware.
- **autocast**: PyTorch automatically selects the right dtype per operation — matmul and conv run in FP16/BF16; numerically sensitive ops like softmax and LayerNorm stay in FP32.
- Parameters and optimizer states (momentum, variance) are kept in FP32 — only activations and forward compute use reduced precision.

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Linear(d_model*4, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

model = TransformerBlock(512, 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# GradScaler prevents underflow of FP16 gradients
scaler = GradScaler()

def train_step_amp(model, optimizer, scaler, x, y):
    optimizer.zero_grad()

    # autocast: ops run in FP16/BF16 where safe (matmul, conv)
    # Some ops (LayerNorm, softmax) stay in FP32 automatically
    with autocast(dtype=torch.float16):
        out = model(x)
        loss = out.mean()  # dummy loss

    # Scale loss to avoid FP16 gradient underflow
    scaler.scale(loss).backward()

    # Unscale before clipping so clip threshold is in FP32 scale
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # If gradients contain inf/nan, skip update; otherwise step
    scaler.step(optimizer)
    scaler.update()  # adjust scale factor for next iteration

    return loss.item()

# BF16 — available on A100/H100, no gradient scaling needed
def train_step_bf16(model, optimizer, x):
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):
        loss = model(x).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

x = torch.randn(2, 16, 512)
y = torch.randn(2, 16, 512)
# loss = train_step_amp(model, optimizer, scaler, x, y)  # needs CUDA
print("AMP setup complete (CPU demo)")
```

**Interview Insight:**
- **FP16**: range `[6e-5, 65504]` — overflow/underflow risk. GradScaler multiplies loss by a large constant before backward to push gradients into representable range.
- **BF16**: same exponent range as FP32, 8-bit mantissa — no scaling needed. Preferred on A100/H100. LLaMA/Mistral use BF16.
- **AMP autocast**: PyTorch maintains a whitelist of ops that are safe in FP16 (matmul, conv) and keeps others (softmax, normalization) in FP32.
- Memory savings: FP16/BF16 = 2x reduction in activations + parameters. Critical for large models.


---

## 3. CNNs

### 3.1 Convolution Types

**What is it?** Variants of the convolution operation that trade off between receptive field size, parameter count, computation, and spatial resolution.

**Why it matters:** Choosing the right convolution type determines model efficiency, receptive field coverage, and suitability for tasks like segmentation, detection, and mobile deployment.

**How it works:**
- **Standard conv**: applies a `k×k` kernel across all input channels simultaneously — expensive: `k²·C_in·C_out` parameters per layer.
- **Depthwise separable conv (MobileNet)**: factorizes into a depthwise conv (one filter per input channel) followed by a 1×1 pointwise conv for channel mixing — ~8–9x fewer FLOPs for `k=3`.
- **Dilated (atrous) conv**: inserts gaps between kernel elements, expanding receptive field without adding parameters or reducing resolution — dilation rate `d` gives effective kernel size `d(k-1)+1`.
- **Transposed conv**: learned upsampling — used in decoders and generators to increase spatial resolution.

```python
import torch
import torch.nn as nn

# --- Standard Convolution ---
conv_std = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

# --- Depthwise Separable Convolution (MobileNet) ---
class DepthwiseSeparableConv(nn.Module):
    """
    Factorizes standard conv into depthwise (spatial) + pointwise (channel mixing).
    ~8-9x fewer FLOPs than standard conv with kernel_size=3.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                                   padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6()  # ReLU capped at 6 — more quantization-friendly

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

# --- Dilated (Atrous) Convolution ---
# dilation=2: kernel elements spaced 2 apart → larger receptive field without extra params
dilated_conv = nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2)

# --- Transposed Convolution (Upsampling / Decoder) ---
trans_conv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# Output size = (H-1)*stride - 2*padding + kernel_size

# --- Receptive Field Calculation ---
def compute_receptive_field(layers: list) -> int:
    """
    layers: list of (kernel_size, stride, dilation) tuples
    RF grows multiplicatively with strides and additively with kernels.
    """
    rf = 1
    total_stride = 1
    for k, s, d in layers:
        effective_k = d * (k - 1) + 1
        rf += (effective_k - 1) * total_stride
        total_stride *= s
    return rf

# ResNet-like: 3 conv layers with k=3, s=1, d=1
layers = [(3, 1, 1)] * 5
print(f"Receptive field after 5 conv3x3 layers: {compute_receptive_field(layers)}px")

# With dilation
layers_dilated = [(3, 1, 1), (3, 1, 2), (3, 1, 4)]
print(f"Receptive field with dilation [1,2,4]: {compute_receptive_field(layers_dilated)}px")
```

---

### 3.2 ResNet & Residual Blocks

**What is it?** ResNet introduces skip connections that add the input of a block directly to its output — `F(x) + x` — enabling training of very deep networks (50–152+ layers).

**Why it matters:** Before ResNets, networks deeper than ~20 layers suffered from degradation (not just vanishing gradients) where adding more layers hurt accuracy. Skip connections solved this, enabling the modern deep learning era.

**How it works:**
- **BasicBlock** (ResNet-18/34): two 3×3 convolutions with a skip connection; a projection shortcut (1×1 conv) is used when dimensions change.
- **Bottleneck** (ResNet-50/101/152): 1×1 → 3×3 → 1×1 convolutions, where the 1×1 convs reduce then restore channel dimensions — 4x expansion allows more channels with fewer FLOPs.
- **Gradient flow**: the identity path `+x` guarantees gradients can flow unchanged all the way to early layers, effectively giving each block a gradient highway bypassing the learned transform `F(x)`.
- **Pre-activation ResNet**: applies BN → ReLU → Conv (rather than Conv → BN → ReLU → Add) for better gradient flow and slightly improved accuracy.

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet BasicBlock (ResNet-18/34) — two 3x3 convs with skip connection"""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # residual connection
        return self.relu(out)

class Bottleneck(nn.Module):
    """ResNet Bottleneck (ResNet-50/101/152) — 1x1 -> 3x3 -> 1x1"""
    expansion = 4

    def __init__(self, in_ch: int, mid_ch: int, stride: int = 1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

# Test
basic = BasicBlock(64, 64)
x = torch.randn(2, 64, 56, 56)
print(f"BasicBlock output: {basic(x).shape}")

bottle = Bottleneck(64, 64)
print(f"Bottleneck output: {bottle(x).shape}")
```

**Interview Insight:**
- **Why residual connections work**: they provide gradient highways — gradients flow directly through skip paths, bypassing potential saturation.
- **Pre-activation ResNet** (He et al., 2016): BN → ReLU → Conv (vs post-activation). Better gradient flow, slightly better accuracy.
- **EfficientNet**: compound scaling (depth × width × resolution) via neural architecture search. Uses SE (Squeeze-and-Excitation) blocks for channel attention.
- **DenseNet**: each layer connects to all subsequent layers — extreme feature reuse, very parameter-efficient, but memory-intensive (all feature maps kept).


---

## 4. RNNs & Sequence Models

### 4.1 LSTM & GRU

**What is it?** Gated recurrent neural networks that process sequences step-by-step while maintaining a hidden state — designed to capture long-range dependencies that vanilla RNNs cannot.

**Why it matters:** Vanilla RNNs suffer from vanishing gradients over long sequences because gradients are multiplied by the same weight matrix at every step. LSTM and GRU use gates to control which information is retained or discarded, enabling gradients to flow across hundreds of steps.

**How it works:**
- **LSTM**: maintains two states — hidden `h_t` and cell `c_t`. Four gates (input, forget, cell, output) control the flow: the forget gate `f_t * c_{t-1}` creates a gradient highway through the cell state.
- **GRU**: simpler — merges cell and hidden state; uses two gates (reset, update). Fewer parameters (~25% less) and competitive performance with LSTM on most tasks.
- **Bidirectional**: runs two RNNs in opposite directions and concatenates outputs — doubles hidden size and captures both past and future context, but requires the full sequence upfront (not usable for generation).
- **Seq2Seq**: encoder RNN compresses sequence into a context vector; decoder RNN generates output token by token, attending to encoder states via attention.

```python
import torch
import torch.nn as nn

# --- Vanilla RNN (for educational comparison) ---
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=2, batch_first=True,
             dropout=0.1, bidirectional=False)

# --- LSTM ---
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               batch_first=True, dropout=0.1, bidirectional=True)

x = torch.randn(32, 50, 128)  # (batch, seq_len, input_size)
out, (h_n, c_n) = lstm(x)
print(f"LSTM output: {out.shape}")   # (32, 50, 512) — bidirectional doubles hidden
print(f"Hidden:      {h_n.shape}")   # (4, 32, 256) — 2 layers * 2 directions

# --- GRU (simpler than LSTM, fewer params) ---
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2,
             batch_first=True, dropout=0.1, bidirectional=True)
out_gru, h_gru = gru(x)
print(f"GRU output: {out_gru.shape}")

# --- Manual LSTM cell to understand internals ---
class LSTMCell(nn.Module):
    """
    LSTM equations:
      i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   # input gate
      f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   # forget gate
      g_t = tanh(W_g * [h_{t-1}, x_t] + b_g)       # cell gate
      o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)   # output gate
      c_t = f_t * c_{t-1} + i_t * g_t              # cell state
      h_t = o_t * tanh(c_t)                         # hidden state
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Combine all 4 gate matrices for efficiency
        self.xh = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h, c = state
        gates = self.xh(torch.cat([x, h], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

cell = LSTMCell(128, 256)
h = torch.zeros(32, 256)
c = torch.zeros(32, 256)
x_t = torch.randn(32, 128)
h_new, c_new = cell(x_t, (h, c))
print(f"LSTMCell h_new: {h_new.shape}")

# --- Seq2Seq with Attention ---
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, n_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        emb = self.embed(src)
        outputs, (hidden, cell) = self.rnn(emb)
        # Combine bidirectional hidden states
        hidden = torch.tanh(self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        return outputs, hidden
```

**Interview Insight:**
- **LSTM vs GRU**: LSTM has 4x hidden_size params (4 gates), GRU has 3x (reset + update + candidate). GRU is faster and competitive on most tasks.
- **Vanishing gradient in vanilla RNN**: gradient decays as `partial h_t / partial h_0 ≈ W_hh^t` — eigenvalues < 1 cause vanishing, > 1 cause explosion. LSTM's forget gate controls gradient flow through the cell state highway.
- **Bidirectional**: doubles model size and computation. Only usable when full sequence is available (classification, NER) — not for generation.
- RNNs are largely replaced by transformers, but still relevant for streaming/low-latency applications.


---

## 5. Attention & Transformers

### 5.1 Scaled Dot-Product Attention

**What is it?** The core operation of the Transformer: computes a weighted sum of Values where weights are determined by the similarity between Queries and Keys.

**Why it matters:** Unlike RNNs (sequential) or CNNs (local), attention directly connects any two positions in one operation — global context with content-dependent weights.

**What is d_k?**
```
  d_k = dimension of BOTH Q and K vectors (must match to compute dot product)
  d_k varies per model — it is always: d_k = d_model / n_heads

  Model          d_model   n_heads   d_k
  ─────────────────────────────────────────
  BERT-base        768       12       64
  GPT-2 small      768       12       64
  GPT-3 175B      12288      96      128
  LLaMA-3 70B      8192      64      128

  Q shape: (seq_len, d_k)   ← one query vector per token
  K shape: (seq_len, d_k)   ← one key   vector per token
  Both Q and K have same d_k — required for dot product
```

**How Q @ Kᵀ dimension grows — concrete example:**
```
  Sentence: "The cat sat mat" → 4 tokens, d_k=3 (simplified)

  Q = [[1, 0, 1],    ← "The"  query vector (3 numbers)
       [0, 1, 0],    ← "cat"  query vector
       [1, 1, 0],    ← "sat"  query vector
       [0, 0, 1]]    ← "mat"  query vector
  Shape: (4, 3)

  Kᵀ (transposed K):          shape (3, 4)
  [[1, 0, 1, 0],
   [0, 1, 1, 0],
   [1, 0, 0, 1]]

  Q @ Kᵀ = (4,3) @ (3,4) = (4,4)  ← one score per token pair

        "The" "cat" "sat" "mat"
  "The" [ 2    0    1    1  ]   ← how much "The" attends to each token
  "cat" [ 0    1    1    0  ]
  "sat" [ 1    1    2    0  ]
  "mat" [ 1    0    0    1  ]
```

**Why do values get large as d_k grows?**
```
  Each score = dot product = sum of d_k multiplications

  d_k=3  (above): score = 1×1 + 0×0 + 1×1 = 2          ← small, manageable
  d_k=64 (BERT):  score = sum of 64 terms
                        = 0.5×0.4 + 0.3×0.7 + ... (64 pairs)
                        ≈ 64 × avg(0.25) ≈ 16            ← grows with d_k

  If each weight ~N(0,1):  std of dot product = √d_k
  d_k=64  → std ≈ 8   → scores in range [-24, +24]
  d_k=128 → std ≈ 11  → scores in range [-33, +33]   ← larger model, larger scores
```

**Why divide by √d_k?**
```
  Large scores → softmax collapses:
  scores = [24, 1, -1]  → softmax = [~1.0, ~0.0, ~0.0] → gradient ≈ 0 → vanishing ✗

  Divide by √d_k normalises variance back to 1:
  scores / √64 = [3, 0.1, -0.1] → softmax balanced → healthy gradients ✓

  Works for any model: BERT (÷8), GPT-3 (÷√128≈11), LLaMA-70B (÷√128≈11)
```

**How it works:**
- `scores = Q @ Kᵀ / √d_k` → (seq_len × seq_len) similarity matrix
- Apply causal mask (set future positions to `-inf`) before softmax
- `softmax(scores)` → attention weights (each row sums to 1)
- Output = `attn_weights @ V` → weighted mix of Value vectors

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0
) -> tuple:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        mask: (batch, 1, seq_q, seq_k) boolean mask (True = mask out)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, Sq, Sk)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0 and torch.is_grad_enabled():
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    out = attn_weights @ V  # (B, H, Sq, d_v)
    return out, attn_weights
```

---

### 5.2 Multi-Head Attention

**What is it?** Multi-Head Attention runs several attention operations in parallel, each with its own learned projections, then concatenates and projects the results.

**Why it matters:** A single attention head can only attend to one type of relationship at a time. Multiple heads allow the model to simultaneously capture syntactic structure, semantic similarity, positional relations, and other patterns — each head specializes in different aspects.

**How it works:**
- Project input into Q, K, V using separate linear layers (often fused into one `3*d_model` projection for efficiency).
- Split the `d_model` dimension into `num_heads` chunks of size `d_k = d_model / num_heads` — each chunk is one head.
- Run scaled dot-product attention independently in each head.
- Concatenate all head outputs and apply a final output projection to mix information across heads.

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: runs h attention heads in parallel, concatenates results.
    Each head learns different aspects of the relationships.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)                              # (B, T, 3*d_model)
        Q, K, V = qkv.chunk(3, dim=-1)                      # each (B, T, d_model)

        # Reshape for multi-head: (B, H, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Attention
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask,
                                                    dropout_p=self.dropout if self.training else 0.0)

        # Merge heads: (B, H, T, d_k) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)

# Test
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 64, 512)
out = mha(x)
print(f"MHA output: {out.shape}")  # (2, 64, 512)
```

---

### 5.3 Positional Encoding

**What is it?** A mechanism for injecting position information into Transformer token embeddings, since self-attention is permutation-invariant and has no inherent notion of order.

**Why it matters:** Without positional encoding, "The cat sat on the mat" and "The mat sat on the cat" produce identical attention patterns — the model cannot distinguish token order at all.

**How it works:**
- **Sinusoidal PE** (original Transformer): adds fixed sine/cosine signals of varying frequencies to token embeddings; no learned parameters, can extrapolate to unseen lengths (poorly).
- **Rotary PE (RoPE)**: encodes position by rotating Q and K vectors in complex space — the dot product `Q_i
- K_j` naturally depends only on relative position `i-j`, not absolute positions. Used in LLaMA, Mistral, GPT-NeoX.
- **ALiBi**: adds a learned per-head linear bias to attention scores proportional to distance `|i-j|` — no positional parameters, strong length generalization. Used in MPT and BLOOM.
- **Learned PE** (BERT, GPT-2): simple trainable embedding lookup by position index — effective within training length but fails to generalize to longer sequences.

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Original Transformer PE: PE(pos, 2i) = sin(pos/10000^(2i/d_model))"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

# --- Rotary Position Embedding (RoPE) — LLaMA, GPT-NeoX ---
class RotaryEmbedding(nn.Module):
    """
    RoPE: encodes position by rotating Q and K vectors.
    Key property: relative position information is preserved in attention dot products.
    Generalizes to unseen context lengths better than sinusoidal.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        seq_len = q.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot

# ALiBi (Attention with Linear Biases) — MPT, BLOOM
def get_alibi_mask(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    ALiBi: adds linear position bias to attention scores instead of PE.
    bias(i, j) = -m * |i - j| where m is head-specific slope.
    No learned params. Excellent length generalization.
    """
    slopes = torch.tensor([2 ** (-8 * i / num_heads) for i in range(1, num_heads + 1)])
    positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    bias = -slopes.view(-1, 1, 1) * positions.abs().unsqueeze(0)
    return bias  # (num_heads, seq_len, seq_len)

pe = SinusoidalPositionalEncoding(512)
rope = RotaryEmbedding(64)  # per-head dim
alibi = get_alibi_mask(8, 128)
print(f"ALiBi mask shape: {alibi.shape}")
```

**Interview Insight:**
- **Sinusoidal PE**: no learned params, extrapolates to unseen lengths (poorly). Added to embeddings.
- **RoPE**: rotates Q/K in complex space by position — attention score only depends on relative distance. Strong length generalization with YaRN/LongRoPE extensions.
- **ALiBi**: best OOD length generalization — penalizes distant tokens with linear bias. No positional params at all.
- **Learned PE** (BERT, GPT-2): simple, effective within training length — fails at longer sequences.

---

### 5.4 Full Transformer Block

```python
class TransformerEncoderBlock(nn.Module):
    """Standard Pre-LN Transformer block (used in modern LLMs)"""
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN: normalize before sublayer (more stable than post-LN)
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class GPTDecoderBlock(nn.Module):
    """GPT-style decoder block with causal (autoregressive) mask"""
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Upper triangular mask: True = masked (future tokens)
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = self._causal_mask(T, x.device).unsqueeze(0).unsqueeze(0)
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# Flash Attention concept (algorithmic, not full impl)
def flash_attention_concept():
    """
    Flash Attention (Dao et al., 2022):
    - Standard attention: O(N^2) memory (stores full N×N attention matrix)
    - Flash Attention: O(N) memory by computing attention in tiles
    
    Key insight: recompute attention during backward instead of storing it.
    Tiles of Q, K, V fit in SRAM (fast) — avoid HBM (slow) round trips.
    
    Flash Attention 2: better parallelism, 2x speedup over FA1
    Flash Attention 3: FP8 support, H100 optimizations, ~75% utilization
    
    Result: 2-4x wall-clock speedup, enables much longer context lengths.
    """
    pass

block = TransformerEncoderBlock(d_model=512, num_heads=8, ffn_dim=2048)
x = torch.randn(2, 32, 512)
print(f"Encoder block output: {block(x).shape}")

gpt_block = GPTDecoderBlock(d_model=512, num_heads=8, ffn_dim=2048)
print(f"GPT block output: {gpt_block(x).shape}")
```

**Forward Pass vs Backpropagation through a Transformer Block:**

```
  FORWARD PASS (left to right):
  ─────────────────────────────────────────────────────────────────
  Input x
    │
    ├──────────────────────────────────┐  ← skip connection saved
    ▼                                  │
  [Self-Attention]                     │
    │                                  │
    ▼                                  │
  [Add & Norm]  ←── x + Attention(x) ─┘
    │
    ├──────────────────────────────────┐  ← skip connection saved
    ▼                                  │
  [MLP / FFN]                          │
    │                                  │
    ▼                                  │
  [Add & Norm]  ←── x + FFN(x) ───────┘
    │
    ▼
  Output


  BACKPROPAGATION (exact reverse — gradients flow right to left):
  ─────────────────────────────────────────────────────────────────
  Gradient from next layer
    │
    ▼
  [Add & Norm]        ← gradient split: one to MLP, one through skip
    │
    ▼
  [MLP / FFN]         ← gradients update MLP weights (W2, W1, biases)
    │
    ▼
  [Add & Norm]        ← gradient split: one to Attention, one through skip
    │
    ▼
  [Self-Attention]    ← gradients update Wq, Wk, Wv, Wo matrices
    │
    ▼
  Gradient passed to previous block
```

**Key point about the skip connections during backprop:**

```
  At each Add & Norm, gradient splits into TWO streams:

  incoming gradient
        │
        ▼
      [ + ]  ←── Add & Norm node
      /   \
     /     \
    ▼       ▼
  to MLP   through skip (gradient = 1.0, no shrinking)
  (may     ──────────────────────────────────────────►
  shrink)  directly to the layer BEFORE this block

  This is WHY transformers can be stacked 96+ layers deep (GPT-3)
  without vanishing gradients — the skip always delivers gradient intact.
```

**What happens inside Self-Attention during Backpropagation?**

Short answer: **weight matrices are updated AND attention scores are revisited** — but for completely different reasons.

```
  FORWARD PASS (everything stored in memory for backprop):
  ──────────────────────────────────────────────────────
  Tokens: ["The", "cat", "sat"]  →  x1, x2, x3

  Q = x
- Wq   ← Wq is LEARNABLE
  K = x
- Wk   ← Wk is LEARNABLE
  V = x
- Wv   ← Wv is LEARNABLE

  scores  = Q
- Kᵀ / √d_k        ← stored ✓
  weights = softmax(scores)        ← stored ✓
  output  = weights
- V
- Wo      ← Wo is LEARNABLE


  BACKPROP — chain rule reverses every forward step:
  ──────────────────────────────────────────────────
  Gradient arrives from Add & Norm above
       │
       ▼
  ① Through Wo projection
       → computes dL/dWo  →  Wo updated ✓
       → passes gradient back to (weights
- V)
       │
       ▼
  ② Through softmax + scores
       → uses stored attention weights to compute softmax gradient
       → attention scores ARE revisited — but only to do gradient math
         NOT to re-decide which tokens attend to which
       │
       ▼
  ③ Through Q
- Kᵀ
       → computes dL/dQ and dL/dK
       │
       ▼
  ④ Through Q, K, V projections
       → dL/dWq  →  Wq updated ✓
       → dL/dWk  →  Wk updated ✓
       → dL/dWv  →  Wv updated ✓
       → gradient passed back to input x → flows to previous block
```

**Clearing the confusion:**

```
  FORWARD:  scores = softmax(Q·Kᵀ)
            → DECIDES which tokens attend to which (e.g. "cat"→"sat")

  BACKPROP: gradient of softmax computed using STORED scores
            → ADJUSTS Wq,Wk,Wv so future attention will be better
            → NOT re-running attention, just computing its gradient

  Analogy:
  Forward  = student reads exam, writes answers, decisions made
  Backprop = teacher grades it; student learns what to focus on
             NEXT TIME — not re-sitting the same exam
```

| What happens in Backprop | Updated? |
|---|---|
| Gradient flows through Wo | Wo weights updated ✓ |
| Gradient flows through softmax | Uses stored scores — no new attention run |
| Gradient flows through Q·Kᵀ | Computes dL/dWq, dL/dWk |
| Gradient flows through V projection | Wv weights updated ✓ |
| **Attention re-performed on tokens?** | **No — forward scores are reused, not recomputed** |

**Dimension flow: 768-dim input through Self-Attention → FFN:**

```
  Input token embedding: [768-dim vector per token]

  ┌─────────────────────────────────────────────────────────────────────┐
  │                  SELF-ATTENTION  (12 heads, d_model=768)            │
  │                                                                     │
  │  head_dim = 768 / 12 heads = 64  ← each head gets a 64-dim slice   │
  │  Wq,Wk,Wv are LEARNED during pre-training — define what each       │
  │  head specialises in (syntax / coreference / position / etc.)      │
  │                                                                     │
  │  Per head (×12 in parallel):                                        │
  │  768 ──► Wq (768×64) ──► Q [64-dim]  ┐  Wq,Wk,Wv = learned weights│
  │  768 ──► Wk (768×64) ──► K [64-dim]  ├─► scores ──► weighted sum   │
  │  768 ──► Wv (768×64) ──► V [64-dim]  ┘      │                      │
  │          ↑ projects 768-dim into the            ▼                   │
  │            head's specialised 64-dim   head output [64-dim]         │
  │            subspace (emergent, not     (what this head noticed)     │
  │            manually assigned)                                       │
  │                                                                     │
  │  concat all 12 heads: 12 × 64 = 768-dim ──► Wo (768×768) ──► 768   │
  │                        ↑                                  ↑         │
  │              restores full dim                 output STILL 768 ✓   │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                         Add & Norm
                              │ still 768-dim
                              ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       FFN (Feed-Forward Network)                    │
  │                                                                     │
  │  Input:  [768-dim]  — one vector per token                          │
  │             │                                                       │
  │             ▼                                                       │
  │  Layer 1:  Linear(768 → 3072)   ← 768 × 3072 = 2,359,296 weights   │
  │            ↑                                                        │
  │   NOT 768 neurons — EXPANDS to 3072 (4 × 768)                      │
  │   Each of the 768 input dims connects to ALL 3072 neurons           │
  │             │                                                       │
  │             ▼                                                       │
  │  Activation: GELU applied to all 3072 values                        │
  │             │                                                       │
  │             ▼                                                       │
  │  Layer 2:  Linear(3072 → 768)   ← compresses back                  │
  │             │                                                       │
  │             ▼                                                       │
  │  Output: [768-dim]  ← same shape as input, ready for next block     │
  └─────────────────────────────────────────────────────────────────────┘
                              │
                         Add & Norm
                              │ still 768-dim
                              ▼
                      Next Transformer Block
```

**Your question clarified — what the FFN layers actually are:**

```
  Your assumption:   768 neurons in Layer 1 (one per input dimension)
  Actual:           3072 neurons in Layer 1 (4× expansion)

  Why expand?
  ┌────────────────────────────────────────────────────────────────┐
  │ Self-Attention finds WHICH tokens relate to each other.        │
  │ FFN processes WHAT each token means — needs more room          │
  │ to reason about the content.                                   │
  │                                                                │
  │ 768 → 3072:  expand (learn rich intermediate features)         │
  │ GELU:        non-linearity (like any MLP hidden layer)         │
  │ 3072 → 768:  compress back (distil into final representation)  │
  └────────────────────────────────────────────────────────────────┘

  Each of the 768 input values IS connected to all 3072 neurons —
  so your instinct was right that every dimension participates,
  but the layer width is 3072, not 768.
```

**Weight matrix sizes for BERT-base (d_model=768):**

```
  Layer               Shape            Parameters
  ─────────────────────────────────────────────────
  FFN Layer 1         768  × 3072      2,359,296
  FFN bias 1          3072             3,072
  FFN Layer 2         3072 × 768       2,359,296
  FFN bias 2          768              768
  ─────────────────────────────────────────────────
  Total per block                      4,722,432
  × 12 blocks (BERT-base)              56,669,184  ← just the FFN weights!
```

**Interview Insight:**
- **Pre-LN vs Post-LN**: Pre-LN (normalize input before sublayer) is more stable for deep networks and requires less warmup. GPT-3, LLaMA use Pre-LN.
- **FFN dimension**: typically 4× d_model for standard transformers. SwiGLU (LLaMA) uses `8/3 × d_model` split into gated projections.
- **BERT**: encoder-only, bidirectional, masked language modeling (MLM). Good for understanding tasks.
- **GPT**: decoder-only, autoregressive, next-token prediction (CLM). Good for generation.
- **Attention complexity**: `O(N^2 * d)` — quadratic in sequence length. Flash Attention reduces memory to O(N) but compute is still O(N^2).


---

## 6. Advanced Training Techniques

### 6.1 Knowledge Distillation

**What is it?** A training technique where a smaller "student" model is trained to mimic the outputs of a larger "teacher" model, transferring the teacher's learned knowledge into a compact form.

**Why it matters:** The teacher's soft probability outputs (e.g., `[0.7, 0.2, 0.1]` rather than `[1, 0, 0]`) encode rich inter-class similarity structure ("dark knowledge") that hard labels discard — the student learns a richer signal than supervised learning alone provides.

**How it works:**
- **Response distillation**: student matches the teacher's softened output logits using KL divergence. Temperature `T > 1` softens distributions, revealing relationships between classes.
- **Feature distillation**: student matches teacher's intermediate layer activations (via an adapter projection if dimensions differ).
- **Loss**: `alpha * CrossEntropy(student, labels) + (1-alpha) * T^2 * KL(teacher_soft || student_soft)` — the `T^2` factor compensates for the smaller gradient magnitudes at high temperature.
- Common applications: TinyBERT, DistilBERT use multi-stage distillation matching embedding + attention + hidden + output layers.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationLoss(nn.Module):
    """
    Hinton et al. (2015): Train student to match teacher's soft probability distribution.
    
    L = alpha * CE(student_hard, labels) + (1-alpha) * T^2 * KL(teacher_soft || student_soft)
    
    Temperature T > 1 softens distributions, revealing inter-class relationships.
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Hard loss: student vs ground truth
        hard_loss = F.cross_entropy(student_logits, targets)

        # Soft loss: student soft probs vs teacher soft probs
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
        # KL divergence: sum(teacher * (log_teacher - log_student))
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.T ** 2)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Feature-based distillation (intermediate layers)
class FeatureDistillationLoss(nn.Module):
    """Match intermediate feature maps via an adapter projection."""
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.adapter = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        projected = self.adapter(student_feat)
        return F.mse_loss(projected, teacher_feat.detach())

# Demo
teacher_logits = torch.randn(8, 1000)
student_logits = torch.randn(8, 1000)
targets = torch.randint(0, 1000, (8,))

kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
loss = kd_loss(student_logits, teacher_logits, targets)
print(f"KD Loss: {loss.item():.4f}")
```

**Interview Insight:**
- **Temperature intuition**: at T=1 teacher might output [0.99, 0.005, 0.005], at T=4 [0.6, 0.2, 0.2] — the "dark knowledge" in soft labels encodes similarity structure.
- **Response distillation**: match output logits. **Feature distillation**: match intermediate activations. **Relation distillation**: match similarity matrices between samples.
- TinyBERT, DistilBERT use multi-stage distillation (embedding + attention + hidden + output layers).

---

### 6.2 PEFT — LoRA, QLoRA, Adapters

**What is it?** Parameter-Efficient Fine-Tuning (PEFT) methods that adapt large pretrained models to new tasks by training only a small number of additional parameters while keeping the base model frozen.

**Why it matters:** Full fine-tuning of a 70B model requires ~280 GB of optimizer state memory. PEFT reduces trainable parameters to 0.1–5% of the original, making fine-tuning feasible on a single GPU while preserving pretrained knowledge.

**How it works:**
- **LoRA**: decomposes the weight update `ΔW` into two low-rank matrices `B @ A` (rank `r << d`). Only `A` and `B` are trained. At inference, merge `W + B@A*scaling` for zero latency overhead.
- **QLoRA**: quantize the frozen base model to 4-bit NF4 format, then train LoRA adapters in BF16 — fits a 65B model on a single 48 GB GPU.
- **Adapters**: insert small bottleneck MLP modules (down-project → activate → up-project) after attention and FFN sublayers. Add inference latency unlike LoRA.
- **Prefix Tuning**: prepend trainable "soft tokens" to the K and V sequences in every attention layer — no changes to model weights at all.

```python
import torch
import torch.nn as nn
import math

# --- LoRA: Low-Rank Adaptation ---
class LoRALinear(nn.Module):
    """
    LoRA: W' = W + BA, where B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d,k)
    
    Only A and B are trained; W is frozen.
    Adds ~0.1-1% extra parameters. Zero overhead at inference (merge B@A into W).
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.linear = linear
        self.linear.weight.requires_grad_(False)  # Freeze original weights

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))  # B init to 0 → starts as W
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora

    def merge_weights(self):
        """Merge LoRA into base weights for zero-overhead inference."""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.requires_grad_(False)

# --- Apply LoRA to a transformer ---
def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16,
               target_modules: list = None) -> nn.Module:
    """Replace target Linear layers with LoRALinear."""
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']  # typical for LLaMA

    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr_name, LoRALinear(module, rank=rank, alpha=alpha))
    return model

# --- Prefix Tuning ---
class PrefixTuning(nn.Module):
    """
    Prepend trainable soft tokens to K and V in every attention layer.
    Model parameters frozen; only prefix (p tokens × d_model × n_layers) trained.
    """
    def __init__(self, num_prefix: int, d_model: int, num_heads: int, n_layers: int):
        super().__init__()
        self.num_prefix = num_prefix
        self.prefix_keys   = nn.Parameter(torch.randn(n_layers, num_heads, num_prefix, d_model // num_heads))
        self.prefix_values = nn.Parameter(torch.randn(n_layers, num_heads, num_prefix, d_model // num_heads))

# --- Adapter Layer ---
class AdapterLayer(nn.Module):
    """
    Bottleneck adapter: down-project → activation → up-project + residual.
    Inserted after FFN and attention sublayers. ~0.5-5% of model params.
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act  = nn.GELU()
        self.up   = nn.Linear(bottleneck_dim, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)  # Near-identity init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))

# Parameter count comparison
base_linear = nn.Linear(4096, 4096)
lora_linear = LoRALinear(base_linear, rank=16, alpha=32)

base_params = sum(p.numel() for p in base_linear.parameters())
lora_trainable = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
print(f"Base params: {base_params:,} | LoRA trainable: {lora_trainable:,} ({lora_trainable/base_params*100:.2f}%)")
```

**Interview Insight:**
- **LoRA rank**: r=4-64 typically. Higher rank = more expressiveness but more params. `alpha/rank` is the effective LR scaling — common: `alpha = 2*rank`.
- **QLoRA** (Dettmers et al.): quantize base model to 4-bit NF4, train LoRA adapters in BF16. Fits 65B model on single 48GB GPU.
- **LoRA vs Adapters**: LoRA adds no inference latency (merge weights). Adapters add inference latency (extra forward passes).
- **Target modules**: Q/V projections most impactful for language tasks. Adding K and FFN further improves performance.

---

### How to Observe Weight Updates During Fine-Tuning / DPO

**What actually changes during fine-tuning vs DPO:**

```
  FULL FINE-TUNING:                    LoRA FINE-TUNING:
  ─────────────────                    ────────────────
  All weights W updated                Only A, B matrices updated
  W_new = W_old - lr
- dL/dW          W_new = W_frozen + (B·A) updated

  DPO (alignment):
  ─────────────────────────────────────────────────────
  Same weight update mechanism (gradient descent)
  BUT the loss function changes:

  SFT loss:  L = -log P(y | x)                   ← maximize chosen response
  DPO loss:  L = -log σ( β
- log(P_θ(y_w)/P_ref(y_w))
                        - β
- log(P_θ(y_l)/P_ref(y_l)) )
                              ↑ chosen              ↑ rejected
             → pushes weights to prefer chosen over rejected
```

**4 ways to observe weight updates in code:**

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# ── Method 1: gradient magnitude per layer ──────────────────────────
# Run after loss.backward(), before optimizer.step()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name:50s} | grad norm: {param.grad.norm():.6f}")

# Output example:
# transformer.h.0.attn.c_attn.weight  | grad norm: 0.000312
# transformer.h.11.attn.c_attn.weight | grad norm: 0.002841  ← later layers larger


# ── Method 2: weight delta (how much weights changed) ───────────────
# Snapshot before training
before = {n: p.data.clone() for n, p in model.named_parameters()}

# ... run one training step ...

# Compare after
for name, param in model.named_parameters():
    delta = (param.data - before[name]).abs().mean().item()
    print(f"{name:50s} | mean weight change: {delta:.8f}")


# ── Method 3: gradient norm per layer (health check) ────────────────
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm().item() ** 2
total_norm = total_norm ** 0.5
print(f"Global gradient norm: {total_norm:.4f}")
# > 10   → exploding gradients → lower lr or add grad clipping
# < 1e-4 → vanishing gradients → check frozen layers or lr


# ── Method 4: LoRA-specific — only A and B change ───────────────────
for name, param in model.named_parameters():
    status = "TRAINABLE" if param.requires_grad else "frozen"
    print(f"{name:50s} | {status}")

# Output for LoRA:
# base_model.model.q_proj.weight        | frozen      ← base untouched
# base_model.model.q_proj.lora_A.weight | TRAINABLE   ← only these update
# base_model.model.q_proj.lora_B.weight | TRAINABLE
```

**What healthy weight updates look like:**

```
  Layer            | Grad Norm  | Interpretation
  ─────────────────┼────────────┼──────────────────────────────────
  Early layers     | 0.0001     | Small — learning slowly (normal)
  Middle layers    | 0.001      | Healthy range ✓
  Last layers      | 0.01       | Larger — closest to loss signal ✓
  Any layer        | > 10       | Exploding → add gradient clipping
  Any layer        | < 1e-6     | Vanishing → check if frozen by mistake
  LoRA A/B only    | non-zero   | Correct LoRA setup ✓
  Base weights     | 0.0        | Correctly frozen in LoRA/DPO ✓


  DPO-specific signals to watch:
  ─────────────────────────────────────────────────────────────────
  chosen_rewards   → should INCREASE over training steps
  rejected_rewards → should DECREASE over training steps
  reward margin    → (chosen - rejected) should GROW
                     if margin shrinks → model forgetting preference
```

**Visualizing with TensorBoard / wandb:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for step, batch in enumerate(dataloader):
    loss.backward()

    # Log gradient norms
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, step)
            writer.add_scalar(f"grad_norm/{name}", param.grad.norm(), step)

    # Log weight norms (track drift from base model)
    for name, param in model.named_parameters():
        writer.add_scalar(f"weight_norm/{name}", param.data.norm(), step)

    optimizer.step()
    optimizer.zero_grad()
```

---

### 6.3 Contrastive Learning — SimCLR & CLIP

**What is it?** A self-supervised learning paradigm that trains representations by pulling together "positive" pairs (semantically similar) and pushing apart "negative" pairs (dissimilar) in embedding space.

**Why it matters:** Contrastive learning produces powerful general-purpose representations without manual labels — enabling models like CLIP to align vision and language at scale, powering zero-shot classification and image-text retrieval.

**How it works:**
- **SimCLR**: two random augmentations of the same image form a positive pair; all other images in the batch are negatives. A projection head maps encoder outputs to a contrastive space where NT-Xent loss is applied. The projection head is discarded after training.
- **CLIP**: pairs images with their natural language captions. A symmetric cross-entropy loss on the `(B, B)` similarity matrix aligns matching image-text pairs and separates mismatched ones. Trained on 400M+ image-text pairs.
- **Temperature**: controls how peaked the similarity distribution is — low temperature (0.07) makes the model more discriminative but sensitive to hard negatives.
- Large batch size is critical for SimCLR (more negatives = stronger signal); BYOL and SimSiam eliminate this requirement via momentum encoders or stop-gradient tricks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """
    SimCLR: self-supervised contrastive learning.
    Two augmented views of same image = positive pair.
    All other images in batch = negative pairs.
    Learns representations without labels.
    """
    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        # Projection head: maps encoder output to contrastive space
        d_model = encoder.output_dim  # placeholder
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
        # Note: only encoder is kept after training; projector is discarded

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(self.projector(self.encoder(x1)), dim=-1)
        z2 = F.normalize(self.projector(self.encoder(x2)), dim=-1)
        return nt_xent_loss(z1, z2, temperature=0.07)

# --- CLIP-style contrastive loss ---
class CLIPLoss(nn.Module):
    """
    CLIP: align image and text embeddings.
    Image[i] should be similar to Text[i] and dissimilar to all others.
    Symmetric cross-entropy over similarity matrix.
    """
    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        else:
            self.register_buffer('log_temp', torch.tensor(math.log(1 / temperature)))

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb  = F.normalize(text_emb, dim=-1)

        # Similarity matrix (B, B)
        logit_scale = self.log_temp.exp().clamp(max=100)
        logits = image_emb @ text_emb.T * logit_scale

        # Symmetric loss: image→text and text→image
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0

def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)

import math
clip_loss = CLIPLoss(temperature=0.07, learnable_temp=True)
img_emb = torch.randn(16, 512)
txt_emb = torch.randn(16, 512)
loss = clip_loss(img_emb, txt_emb)
print(f"CLIP loss: {loss.item():.4f}")
```

**Interview Insight:**
- **Large batch is critical for SimCLR**: more negatives → stronger signal. Google used batch size 8192 with 128 TPU cores. BYOL/SimSiam eliminate the need for large batches.
- **CLIP temperature**: `logit_scale` is learned, initialized to `1/0.07 ≈ 14.3`. Clamp prevents collapse to 0 temperature.
- **Projection head**: discard after training for SimCLR. Representations in encoder space generalize better than projection space.
- **Hard negatives**: sampling similar-but-different negatives accelerates convergence significantly.


---

## 7. Generative Models

### 7.1 Variational Autoencoder (VAE)

**What is it?** A generative model that learns a probabilistic latent space — the encoder maps input to a distribution `q(z|x)` rather than a single point, enabling smooth interpolation and generation of new samples by sampling from the prior `p(z) = N(0, I)`.

**Why it matters:** VAEs learn structured, continuous latent representations that can be sampled from and interpolated. They are the backbone of latent diffusion models (Stable Diffusion uses a VAE to compress images into an 8x smaller latent space).

**How it works:**
- **Encoder**: maps input `x` to parameters `(mu, log_var)` of a Gaussian distribution `q(z|x)`.
- **Reparameterization trick**: sample `z = mu + eps * std` where `eps ~ N(0,1)` — makes sampling differentiable so gradients flow back to the encoder.
- **Decoder**: maps latent `z` back to reconstructed `x_hat`.
- **ELBO loss**: `Reconstruction loss + KL(q(z|x) || p(z))` — the KL term regularizes the latent space toward the standard normal prior, preventing the encoder from ignoring the prior and making the space well-structured.
- **beta-VAE**: weighting KL by `beta > 1` encourages disentangled latent factors at the cost of reconstruction quality.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    VAE: encoder → (mu, log_var) → sample z via reparameterization → decoder
    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20):
        super().__init__()
        # Encoder: x → (mu, log_var)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std, eps ~ N(0,1)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, log_var = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             log_var: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    ELBO loss with beta weighting (beta-VAE for disentanglement).
    Reconstruction: binary cross entropy
    KL divergence: closed form for Gaussian prior
    """
    recon_loss = F.binary_cross_entropy(x_hat, x.view(x.size(0), -1), reduction='sum')
    # KL(N(mu, var) || N(0,1)) = -0.5 * sum(1 + log_var - mu^2 - var)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss

vae = VAE()
x = torch.randn(16, 784)
x_hat, mu, logvar = vae(x)
loss = vae_loss(x_hat, x, mu, logvar, beta=1.0)
print(f"VAE loss: {loss.item():.2f}, x_hat: {x_hat.shape}")
```

---

### 7.2 GANs

**What is it?** Generative Adversarial Networks train a Generator (G) and Discriminator (D) in a minimax game: G generates fake samples to fool D; D learns to distinguish real from fake.

**Why it matters:** GANs produce the sharpest, most photorealistic images of any generative model family — they directly optimize for perceptual realism rather than pixel-level reconstruction loss. StyleGAN and BigGAN set the standard for high-resolution image synthesis.

**How it works:**
- **Generator**: maps random noise `z ~ N(0, I)` to realistic-looking samples in the data space.
- **Discriminator**: classifies inputs as real (from dataset) or fake (from G); outputs a probability.
- **Training**: alternating updates — D is trained to output 1 for real, 0 for fake; G is trained to make D output 1 for its fake samples (i.e., fool D).
- **Mode collapse**: G learns to produce a limited variety of outputs that reliably fool D. Solutions: Wasserstein loss (WGAN), minibatch discrimination, spectral normalization.
- **WGAN**: replaces BCE with Earth Mover (Wasserstein) distance — more informative gradient even when D is very strong, giving much more stable training.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
            nn.Linear(256, 512),        nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
            nn.Linear(512, img_dim),    nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),     nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),       nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_gan_step(G, D, opt_G, opt_D, real_imgs, latent_dim=100):
    B = real_imgs.size(0)
    real_label = torch.ones(B, 1)
    fake_label = torch.zeros(B, 1)

    # --- Train Discriminator ---
    opt_D.zero_grad()
    # Real images
    d_real = D(real_imgs.view(B, -1))
    loss_real = F.binary_cross_entropy(d_real, real_label)
    # Fake images
    z = torch.randn(B, latent_dim)
    fake_imgs = G(z).detach()  # detach: don't update G here
    d_fake = D(fake_imgs)
    loss_fake = F.binary_cross_entropy(d_fake, fake_label)
    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    opt_D.step()

    # --- Train Generator ---
    opt_G.zero_grad()
    z = torch.randn(B, latent_dim)
    fake_imgs = G(z)
    # G wants D to classify fake as real
    loss_G = F.binary_cross_entropy(D(fake_imgs), real_label)
    loss_G.backward()
    opt_G.step()

    return loss_D.item(), loss_G.item()
```

**Interview Insight:**
- **Mode collapse**: generator maps many z to same output. Solutions: Wasserstein loss (WGAN), minibatch discrimination, unrolled GANs, training tricks (TTUR: different LR for G and D).
- **WGAN**: uses Earth Mover distance, removes log/sigmoid, clips weights or uses gradient penalty (WGAN-GP). More stable training.
- **Training balance**: D too strong → vanishing gradients for G. D too weak → G produces garbage. 1 G step per 5 D steps is common.
- **StyleGAN**: style-based generator with AdaIN (adaptive instance normalization), mapping network W. State of the art for image synthesis.

---

### 7.3 Diffusion Models (DDPM)

**What is it?** A generative model that learns to reverse a gradual noising process — the forward process slowly destroys data by adding Gaussian noise over `T` steps; a neural network learns the reverse denoising process to generate new samples.

**Why it matters:** Diffusion models surpassed GANs on image quality benchmarks and are more stable to train (no adversarial dynamics). They power Stable Diffusion, DALL-E 2, and Imagen.

**How it works:**
- **Forward process**: `q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)` — can sample any timestep `x_t` directly from `x_0` without stepping through intermediate states.
- **Reverse process**: a U-Net (conditioned on timestep `t`) predicts the noise `eps` that was added to `x_t` — minimizing `||eps - eps_theta(x_t, t)||^2`.
- **Inference**: starting from pure noise `x_T ~ N(0, I)`, iteratively apply the denoising step `T` times to recover a clean sample — slow (1000 steps for DDPM, but DDIM reduces to 10–50 steps deterministically).
- **Classifier-Free Guidance (CFG)**: `output = uncond + scale * (cond - uncond)` — scale > 1 sharpens quality/specificity at the cost of diversity.
- **Latent diffusion**: run the diffusion process in a compressed VAE latent space (~8x smaller than pixel space), making it practical for high-resolution generation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPMScheduler:
    """
    DDPM (Ho et al., 2020):
    Forward process: q(x_t | x_{t-1}) = N(sqrt(1-beta_t)*x_{t-1}, beta_t*I)
    Reverse process: p_theta(x_{t-1} | x_t) — learned denoising
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)    # \bar{alpha}_t
        self.sqrt_alpha_cumprod = self.alpha_cumprod.sqrt()
        self.sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod).sqrt()

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        """
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x0, (1 - alpha_bar_t) * I)
        Direct sampling at any timestep — key efficiency trick.
        """
        noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alpha_cumprod[t].view(-1, *([1]*(x0.ndim-1)))
        sqrt_1a = self.sqrt_one_minus_alpha_cumprod[t].view(-1, *([1]*(x0.ndim-1)))
        x_t = sqrt_a * x0 + sqrt_1a * noise
        return x_t, noise

    def ddpm_step(self, model_output: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        """Single reverse step: p(x_{t-1} | x_t)"""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_cumprod[t]

        # Predicted x0
        x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * model_output) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Compute mean for p(x_{t-1} | x_t, x0_pred)
        coef1 = beta_t * alpha_bar_t.sqrt() / (1 - alpha_bar_t)
        coef2 = (1 - alpha_bar_t / alpha_t).sqrt() * alpha_t.sqrt() / (1 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x_t

        if t > 0:
            noise = torch.randn_like(x_t)
            var = beta_t * (1 - alpha_bar_t / alpha_t) / (1 - alpha_bar_t)
            return mean + var.sqrt() * noise
        return mean

class SimpleUNet(nn.Module):
    """Minimal U-Net denoiser for DDPM (educational)"""
    def __init__(self, in_ch: int = 1, time_emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.conv_in  = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / 1000)
        h = self.conv_in(x)
        # In full U-Net: inject t_emb via AdaGN or addition to hidden features
        return self.conv_out(h)

def ddpm_training_step(model, scheduler, x0, optimizer):
    """Simple DDPM training loop"""
    B = x0.size(0)
    t = torch.randint(0, scheduler.T, (B,))     # random timesteps
    x_t, noise = scheduler.add_noise(x0, t)
    # Model predicts the noise that was added (epsilon parameterization)
    pred_noise = model(x_t, t)
    loss = F.mse_loss(pred_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

scheduler = DDPMScheduler(num_timesteps=1000)
x0 = torch.randn(4, 1, 28, 28)
t = torch.randint(0, 1000, (4,))
x_t, noise = scheduler.add_noise(x0, t)
print(f"Noisy sample: {x_t.shape}, noise: {noise.shape}")
```

**Interview Insight:**
- **DDPM**: 1000 denoising steps, slow inference. **DDIM** (Song et al., 2020): non-Markovian process, 10-50 steps deterministically, same quality.
- **Epsilon vs x0 vs v parameterization**: predict noise (epsilon), clean image (x0), or velocity (v). v-parameterization (used in Stable Diffusion 2) is more stable at low noise levels.
- **Score matching**: diffusion models learn the score `nabla_x log p(x)` — equivalent to predicting noise. Connects to Langevin dynamics.
- **CFG (Classifier-Free Guidance)**: `output = uncond + scale * (cond - uncond)`. Scale > 1 sharpens quality but reduces diversity. Standard in Stable Diffusion.
- **Latent diffusion** (Stable Diffusion): run diffusion in VAE latent space (~8x smaller) — makes it practical.


---

## 8. Scaling & Distributed Training

### 8.1 Data, Model, and Tensor Parallelism

**What is it?** Strategies for distributing neural network training across multiple GPUs when either the data volume or model size exceeds what a single device can handle.

**Why it matters:** A 70B parameter model requires ~140 GB in BF16 — impossible on a single 80 GB GPU. Distributed training strategies are required to train frontier LLMs and to achieve sufficient throughput at scale.

**How it works:**
- **Data Parallelism (DDP)**: each GPU holds a complete model copy and processes a different data shard; gradients are averaged via `all_reduce` after each backward pass. Scales easily but requires the whole model to fit on one GPU.
- **Tensor Parallelism (Megatron-LM)**: splits individual weight matrices across GPUs — column-parallel splits output features, row-parallel splits input features. Requires one `all_reduce` per layer. Best used within a single node (fast NVLink).
- **Pipeline Parallelism**: assigns different layers to different GPUs; micro-batching fills the pipeline to reduce idle "bubble" time. The 1F1B schedule (PipeDream) keeps GPUs busy by interleaving forward and backward passes.
- **3D Parallelism**: combines all three strategies — tensor parallel within a node, pipeline parallel across nodes, data parallel across replicas. Used by Megatron-DeepSpeed for GPT-3-scale training.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Data Parallelism (DDP) ---
def setup_ddp(rank: int, world_size: int):
    """Each GPU has full model copy; gradients averaged across GPUs."""
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def train_ddp(rank, world_size, model_cls, dataset):
    setup_ddp(rank, world_size)
    model = model_cls().to(rank)
    # DDP wraps model: backward all-reduces gradients automatically
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for batch in loader:
        optimizer.zero_grad()
        loss = model(batch).mean()
        loss.backward()  # triggers all-reduce under the hood
        optimizer.step()
    dist.destroy_process_group()

# --- Tensor Parallelism (column/row split) ---
class ColumnParallelLinear(nn.Module):
    """
    Splits output dimension across GPUs.
    Input X is broadcast; each GPU computes a shard of Y = X @ W^T.
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0
        shard_out = out_features // world_size
        self.linear = nn.Linear(in_features, shard_out)
        self.rank = rank

    def forward(self, x):
        local_out = self.linear(x)
        # In real TP: all-gather across GPUs to get full output
        return local_out

class RowParallelLinear(nn.Module):
    """
    Splits input dimension across GPUs.
    Each GPU has shard of X and W; partial results are all-reduced.
    """
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0
        shard_in = in_features // world_size
        self.linear = nn.Linear(shard_in, out_features)

    def forward(self, x_shard):
        local_out = self.linear(x_shard)
        # In real TP: all-reduce partial sums
        # dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out

print("Parallelism classes defined")
```

---

### 8.2 ZeRO Optimization & Gradient Checkpointing

**What is it?** ZeRO (Zero Redundancy Optimizer) eliminates the memory redundancy in data parallelism by sharding optimizer states, gradients, and/or parameters across GPUs. Gradient checkpointing trades computation for activation memory.

**Why it matters:** In standard DDP, each GPU stores full copies of optimizer states (Adam requires 2 fp32 buffers per parameter) — for a 13B model this is ~200 GB per GPU. ZeRO-3 reduces this to `200 GB / N` where N is the number of GPUs.

**How it works:**
- **ZeRO-1**: shard optimizer states (momentum, variance) — ~4x memory reduction per GPU.
- **ZeRO-2**: shard optimizer states + gradients — ~8x memory reduction.
- **ZeRO-3**: shard optimizer states + gradients + parameters — each GPU holds `1/N` of everything. During forward, parameters are gathered on demand (`all_gather`); during backward, gradients are `reduce_scatter`ed back to owners.
- **CPU offload** (ZeRO-Infinity): move optimizer states and parameters to CPU RAM, enabling models that exceed total GPU memory combined.
- **Gradient checkpointing**: during forward, do not store intermediate activations; recompute them during backward. Saves `O(L)` activation memory at the cost of one extra forward pass (~33% more compute). Standard for LLM fine-tuning.

```python
# ZeRO (Zero Redundancy Optimizer) — DeepSpeed
# ZeRO-1: Shard optimizer states across GPUs → 4x memory reduction
# ZeRO-2: Shard optimizer states + gradients → 8x memory reduction
# ZeRO-3: Shard optimizer states + gradients + parameters → unlimited model size

# DeepSpeed ZeRO config example
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "cpu"},  # CPU offload for stage 3
        "offload_param":     {"device": "cpu"},
    },
    "bf16": {"enabled": True},
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
}

# --- Gradient Checkpointing ---
import torch.utils.checkpoint as checkpoint

class CheckpointedTransformerBlock(nn.Module):
    """
    Gradient checkpointing: don't store activations during forward.
    Recompute them during backward. Saves memory at cost of ~33% more compute.
    """
    def __init__(self, d_model=512, num_heads=8, ffn_dim=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))

    def _forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # checkpoint: recompute activations during backward
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

model = CheckpointedTransformerBlock()
x = torch.randn(2, 32, 512, requires_grad=True)
out = model(x)
out.sum().backward()
print(f"Gradient checkpointing output: {out.shape}")
```

**Interview Insight:**
- **Data Parallelism**: easiest, scales to many GPUs. Bottleneck: all-reduce of gradients grows with model size. Uses collective `all_reduce`.
- **Tensor Parallelism (Megatron-LM)**: split individual matrices. Requires synchronization within each layer (all-reduce on every forward pass). Typically within a node.
- **Pipeline Parallelism**: split layers across GPUs. Bubble overhead from idle GPUs. Micro-batching (GPipe) or 1F1B schedule (PipeDream) mitigates bubble.
- **ZeRO-3**: each GPU stores `1/N` of parameters, gradients, and optimizer states. During forward: all-gather parameters. During backward: reduce-scatter gradients.
- **Gradient checkpointing**: standard for LLM fine-tuning. Rule of thumb: saves `sqrt(L)` memory for L layers at cost of 1 extra forward pass.

---

### 8.3 Pipeline Parallelism

**What is it?** A parallelism strategy that partitions model layers into sequential stages, each assigned to a different GPU, with data flowing through the stages like an assembly line.

**Why it matters:** When a model is too large for tensor parallelism alone (which requires fast NVLink), pipeline parallelism distributes layers across nodes connected by slower inter-node links — it is the key strategy for scaling to hundreds of GPUs.

**How it works:**
- Split the model's layers into `p` pipeline stages; assign each stage to one GPU (or a group of GPUs).
- Divide each mini-batch into `m` micro-batches; inject micro-batches into the pipeline so multiple stages are active simultaneously.
- **GPipe**: all `m` forward passes, then all `m` backward passes — simple but creates a "bubble" of `O(p)` idle time.
- **1F1B schedule (PipeDream)**: interleaves one forward with one backward for each micro-batch in steady state — reduces bubble to `O(1)` relative to batch size, keeping pipeline utilization near 100%.
- **Communication**: each stage sends its output activations (and receives gradients) only at stage boundaries — much less communication than tensor parallelism.

```python
class PipelineStage(nn.Module):
    """Simplified pipeline stage — runs subset of model layers."""
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def pipeline_parallel_training(stages: list, micro_batches: list, schedule: str = '1f1b'):
    """
    1F1B (one forward, one backward) schedule:
    - Fills pipeline with F microbatches in steady state
    - Immediately does backward after each forward
    - Reduces pipeline bubble from O(p) to O(1) relative to batch size
    
    GPipe: all-forward then all-backward. Simpler but larger bubble.
    """
    # Pseudocode for educational purposes
    # In practice: use torch.distributed.pipeline.sync.Pipe or DeepSpeed
    pass

# PyTorch native pipeline
# from torch.distributed.pipeline.sync import Pipe
# model = nn.Sequential(*layers)  # layers split across GPUs
# model = Pipe(model, chunks=8)   # chunks = num micro-batches
```

**Interview Insight:**
- **3D Parallelism**: combine Data + Tensor + Pipeline parallelism. Used by Megatron-DeepSpeed for GPT-3 scale training.
- **Communication bottleneck**: TP requires fast NVLink (intra-node). PP uses slower inter-node links. DP all-reduce is gradient-size proportional.
- **Activation memory**: major memory consumer, especially with long sequences. Gradient checkpointing + rematerialization reduces this.


---

## 9. Quantization & Efficiency

### 9.1 Post-Training Quantization

**What is it?** Reducing the numerical precision of a trained model's weights (and optionally activations) from FP32/BF16 to INT8 or INT4 — without retraining from scratch.

**Why it matters:** A 70B BF16 model requires ~140 GB of GPU memory. INT4 quantization cuts this to ~35 GB, enabling it to run on consumer GPUs. Quantization also reduces memory bandwidth pressure, which is the primary bottleneck during LLM inference.

**How it works:**
- **Dynamic quantization**: weights are quantized to INT8 at inference time; activations are dynamically computed per-batch. Simplest approach, CPU-only, no calibration needed.
- **Static quantization**: uses a calibration dataset to record activation ranges, then converts both weights and activations to INT8. Requires `QuantStub`/`DeQuantStub` markers and calibration before conversion.
- **INT4 weight quantization (GPTQ/bitsandbytes)**: groups weights into blocks (e.g., 128 elements), computes a per-group scale (`max(|W|) / 7`), rounds weights to 4-bit integers — dequantizes on-the-fly during inference (no speedup from INT4 ops, but 4x memory reduction).
- **NF4 (QLoRA)**: uses a non-uniform 4-bit quantization grid optimized for normally distributed weights — better preserves model accuracy than linear INT4.

```python
import torch
import torch.nn as nn

# --- Dynamic Quantization (easiest, CPU-only) ---
model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={nn.Linear},   # only quantize Linear layers
    dtype=torch.qint8            # INT8 weights
)

# --- Static Quantization (better for CNNs, needs calibration) ---
class QuantizableResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
        # QuantStub/DeQuantStub: mark quantization boundaries
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.skip_add = torch.nn.quantized.FloatFunctional()  # quantize-safe add

    def forward(self, x):
        x = self.quant(x)
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, residual)  # quantize-safe residual add
        return self.dequant(out)

def prepare_and_calibrate(model, calibration_loader):
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86
    torch.quantization.prepare(model, inplace=True)

    model.eval()
    with torch.no_grad():
        for batch, _ in calibration_loader:
            model(batch)  # collect activation statistics

    torch.quantization.convert(model, inplace=True)
    return model

# --- INT4 Weight Quantization (bitsandbytes style) ---
class INT4Linear(nn.Module):
    """
    4-bit weight quantization with absmax scaling.
    Quantize: w_q = round(w / scale * 7), scale = max(|w|) / 7
    Dequantize: w = w_q * scale / 7
    """
    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.group_size   = group_size

        W = linear.weight.data  # (out, in)
        W_q, scales = self.quantize_int4(W, group_size)
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.bias = linear.bias

    def quantize_int4(self, W, group_size):
        out_f, in_f = W.shape
        # Reshape into groups
        W_groups = W.view(out_f, -1, group_size)
        scales = W_groups.abs().max(dim=-1, keepdim=True).values / 7.0
        W_q = (W_groups / scales).round().clamp(-8, 7).to(torch.int8)
        return W_q.view(out_f, in_f), scales.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        W_dq = (self.W_q.float().view(self.out_features, -1, self.group_size) *
                self.scales.unsqueeze(-1)).view(self.out_features, self.in_features)
        return torch.nn.functional.linear(x, W_dq, self.bias)

print("Quantization classes defined")
```

---

### 9.2 KV Cache & Speculative Decoding

**What is it?** KV Cache avoids redundant recomputation during autoregressive generation by storing past Key and Value tensors. Speculative Decoding uses a fast draft model to propose multiple tokens that a large model verifies in parallel.

**Why it matters:** Without KV cache, generating each new token requires recomputing attention over the entire context — `O(N^2)` total work. With KV cache, each step is `O(N)`. Speculative decoding can give 2–3x wall-clock speedup for large models at no cost to output quality.

**How it works:**
- **KV Cache**: at each decoding step, append the new token's Key and Value vectors to the cache; only the new token's Query attends over the full cached K, V — one `O(N)` attention call instead of recomputing from scratch.
- **Memory cost**: `2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_bytes`. For LLaMA-70B at seq=4096 BF16: ~160 GB. MQA/GQA reduce this by sharing K,V heads.
- **Speculative Decoding**: (1) a small draft model greedily generates `gamma` candidate tokens; (2) the target model runs one forward pass over all `gamma` tokens simultaneously; (3) tokens are accepted where target agrees with draft (rejection sampling); at the first rejected token, resample from a corrected distribution. Expected tokens per step: `gamma × acceptance_rate + 1`.
- Acceptance rate is typically 70–90% when draft model is 7–10x smaller than the target.

```python
import torch
import torch.nn as nn

class KVCacheAttention(nn.Module):
    """
    KV Cache: store past K and V tensors to avoid recomputation during generation.
    Without cache: every new token recomputes K,V for all past tokens — O(N^2) total.
    With cache: O(N) total — each step only computes 1 new K,V pair.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, past_kv: tuple = None) -> tuple:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        # Reshape
        def to_heads(t):
            return t.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)

        # Append to cache
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)  # grow along seq_len dim
            V = torch.cat([past_V, V], dim=2)

        new_kv = (K, V)

        # Attention over full K, V but only new Q
        scale = self.d_k ** -0.5
        scores = Q @ K.transpose(-2, -1) * scale
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), new_kv

def speculative_decoding(draft_model, target_model, prompt, gamma=4, max_new_tokens=50):
    """
    Speculative Decoding (Chen et al., 2023):
    1. Draft model generates gamma tokens autoregressively (fast, small model)
    2. Target model verifies all gamma tokens in ONE forward pass (parallel)
    3. Accept tokens where target agrees; reject and resample at first mismatch
    
    Speedup: 2-3x for large models when draft model is 7-10x smaller.
    Same output distribution as target model alone (provably correct).
    """
    # Pseudocode
    generated = list(prompt)
    while len(generated) - len(prompt) < max_new_tokens:
        # Step 1: Draft generates gamma tokens
        draft_tokens = []
        draft_probs  = []
        for _ in range(gamma):
            logits = draft_model(torch.tensor([generated]))
            probs  = torch.softmax(logits[:, -1], dim=-1)
            token  = torch.multinomial(probs, 1).item()
            draft_tokens.append(token)
            draft_probs.append(probs[0, token].item())

        # Step 2: Target verifies all gamma+1 tokens in parallel
        full_seq  = generated + draft_tokens
        all_logits = target_model(torch.tensor([full_seq]))
        # Verify each draft token
        accepted = 0
        for i, (tok, dp) in enumerate(zip(draft_tokens, draft_probs)):
            tp = torch.softmax(all_logits[:, len(generated)+i-1], dim=-1)[0, tok].item()
            if torch.rand(1).item() < min(1.0, tp / dp):
                generated.append(tok)
                accepted += 1
            else:
                # Resample from corrected distribution
                corrected = torch.clamp(
                    torch.softmax(all_logits[:, len(generated)+i-1], dim=-1) -
                    torch.softmax(torch.tensor(draft_probs), dim=-1),
                    min=0
                )
                token = torch.multinomial(corrected, 1).item()
                generated.append(token)
                break
        else:
            # All accepted — sample one more from target
            bonus = torch.multinomial(torch.softmax(all_logits[:, -1], dim=-1), 1).item()
            generated.append(bonus)

    return generated

kv_attn = KVCacheAttention(d_model=512, num_heads=8)
x = torch.randn(1, 10, 512)
out1, kv = kv_attn(x, past_kv=None)

x_new = torch.randn(1, 1, 512)   # single new token
out2, kv = kv_attn(x_new, past_kv=kv)  # uses cached K, V
print(f"KV cache output: {out2.shape}, K cache shape: {kv[0].shape}")
```

---

### 9.3 Pruning

**What is it?** Removing weights, neurons, attention heads, or entire layers from a trained model to reduce its size and computational cost, typically followed by fine-tuning to recover accuracy.

**Why it matters:** Dense neural networks contain significant redundancy — empirically, 70–90% of weights in large models can be set to zero with minimal accuracy loss. Pruning reduces deployment cost, though the practical speedup depends on whether the hardware can exploit sparsity.

**How it works:**
- **Unstructured (magnitude) pruning**: removes individual weights with smallest absolute value. High compression ratio but no real speedup on GPUs/CPUs without sparse kernel support.
- **Structured pruning**: removes entire output neurons, filters, or attention heads — produces a genuinely smaller dense model that runs faster on standard hardware without sparse kernels.
- **Iterative Magnitude Pruning (IMP)**: prune a small fraction, fine-tune to recover accuracy, repeat — gradually reaches high sparsity (90%+) while maintaining performance.
- **Lottery Ticket Hypothesis**: iterative pruning + rewinding weights to their original initialization values reveals sparse "winning ticket" subnetworks that can train to full accuracy from scratch.

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10))

# --- Unstructured Magnitude Pruning (removes individual weights) ---
prune.l1_unstructured(model[0], name='weight', amount=0.3)  # prune 30% of weights
print(f"Sparsity: {100 * float(torch.sum(model[0].weight == 0)) / model[0].weight.numel():.1f}%")

# --- Structured Pruning (removes entire neurons/filters) ---
prune.ln_structured(model[0], name='weight', amount=0.3, n=2, dim=0)  # prune 30% of output neurons

# Make pruning permanent (remove mask buffers)
prune.remove(model[0], 'weight')

# --- Iterative Magnitude Pruning ---
def iterative_pruning(model, target_sparsity: float = 0.9, n_rounds: int = 10,
                       finetune_steps: int = 1000):
    """
    Lottery Ticket Hypothesis (Frankle & Carlin, 2019):
    Iterative pruning + rewinding to original init finds sparse subnetworks
    that train to full accuracy. Round-by-round increases sparsity.
    """
    amount_per_round = 1 - (1 - target_sparsity) ** (1 / n_rounds)  # geometric

    for round_i in range(n_rounds):
        # Prune
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=amount_per_round)

        # Fine-tune (placeholder)
        # train(model, steps=finetune_steps)
        print(f"Round {round_i+1}: target sparsity {1-(1-amount_per_round)**(round_i+1):.1%}")

iterative_pruning(model, target_sparsity=0.9, n_rounds=5)
```

**Interview Insight:**
- **Unstructured pruning**: high compression but no practical speedup on dense hardware (GPU/CPU ignore zeros). Need sparse CUDA kernels (cuSPARSE).
- **Structured pruning**: removes entire channels/heads/layers — gives real speedup on standard hardware.
- **KV Cache memory**: `2 * num_layers * num_heads * head_dim * seq_len * batch_size * dtype_size`. For LLaMA-70B at seq=4096, BF16: ~160 GB. Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) reduce this dramatically.
- **Speculative decoding**: token acceptance rate typically 70-90%, giving 2-3x wall-clock speedup. Draft model quality is critical.


---

## 10. Advanced Topics

### 10.1 Mixture of Experts (MoE)

**What is it?** A model architecture where the FFN layer of each Transformer block is replaced by multiple independent "expert" FFNs, with a learned router selecting which `k` experts process each token.

**Why it matters:** MoE scales model capacity (number of parameters) without proportionally increasing computation — Mixtral-8x7B has 47B parameters but only activates ~13B per token, giving the expressiveness of a large model at the compute cost of a small one.

**How it works:**
- **Router**: a linear layer maps each token's representation to a probability distribution over all `E` experts; the top-`k` experts (typically k=2) are selected.
- **Sparse computation**: only the selected experts' FFNs execute for each token — the rest are skipped. FLOPs scale as `k/E` of the total expert capacity.
- **Load balancing loss**: prevents "expert collapse" where all tokens route to the same 1–2 experts. The auxiliary loss penalizes uneven token-to-expert assignments.
- **Communication in distributed MoE**: tokens may be routed to experts on different GPUs, requiring expensive `all_to_all` collective operations — the primary scaling challenge.
- **Shared experts (DeepSeek-MoE)**: a subset of experts is always active for every token; the rest are selectively routed — improves knowledge sharing across tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """
    MoE Router: selects top-k experts for each token.
    With load balancing loss to prevent expert collapse.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.gate        = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (B, T, d_model)
        Returns: routing weights (B, T, k), expert indices (B, T, k), aux_loss
        """
        logits = self.gate(x)                                  # (B, T, E)
        probs  = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = probs.topk(self.top_k, dim=-1)  # (B, T, k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize

        # Load balancing loss (Switch Transformer): encourages uniform expert usage
        # L_aux = E * sum_i(f_i * P_i), f_i = fraction of tokens to expert i
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            tokens_per_expert += F.one_hot(topk_indices[..., k], self.num_experts).float().sum((0, 1))
        f = tokens_per_expert / (x.size(0) * x.size(1) * self.top_k)
        P = probs.mean((0, 1))  # mean routing probability per expert
        aux_loss = self.num_experts * (f * P).sum()

        return topk_weights, topk_indices, aux_loss

class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN layer.
    Each expert is an independent FFN. Only top-k are active per token.
    Sparse MoE: only k/E fraction of params active → scales model size without FLOPs increase.
    """
    def __init__(self, d_model: int, ffn_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = TopKRouter(d_model, num_experts, top_k)
        # Each expert: independent FFN
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple:
        B, T, D = x.shape
        weights, indices, aux_loss = self.router(x)  # (B, T, k), (B, T, k)

        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[..., k]    # (B, T)
            expert_w   = weights[..., k:k+1]  # (B, T, 1)
            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)  # (B, T) bool
                if mask.any():
                    tokens = x[mask]                       # (N, D)
                    expert_out = self.experts[e_id](tokens)
                    output[mask] += expert_w[mask] * expert_out

        return output, aux_loss

moe = MoEFFN(d_model=512, ffn_dim=2048, num_experts=8, top_k=2)
x = torch.randn(2, 16, 512)
out, aux = moe(x)
print(f"MoE output: {out.shape}, aux_loss: {aux.item():.4f}")
```

**Interview Insight:**
- **MoE scaling**: Mixtral-8x7B has 8 experts, activates 2 per token. Parameter count: 47B, active params: 13B. ~6x more expressive than dense 13B model with same compute.
- **Expert collapse**: all tokens route to 1-2 experts — rest are unused. Load balancing loss prevents this.
- **Communication in distributed MoE**: tokens may be routed to experts on different GPUs — requires all-to-all communication (expensive).
- **Shared experts** (DeepSeek-MoE): some experts always active (shared), others selectively routed. Improves knowledge sharing.

---

### 10.2 Continual Learning & Catastrophic Forgetting

**What is it?** Training a model sequentially on multiple tasks without forgetting previously learned knowledge — a fundamental challenge since standard gradient descent overwrites old task parameters when learning new ones.

**Why it matters:** Real-world AI systems need to learn from new data continuously without retraining from scratch. Catastrophic forgetting means a model fine-tuned on task B will lose most of its task A performance — a critical barrier to lifelong learning.

**How it works:**
- **Elastic Weight Consolidation (EWC)**: identifies which weights are most important for past tasks using the Fisher information matrix (squared gradients), then adds a quadratic penalty to prevent those weights from drifting during new task training.
- **Progressive Neural Networks**: add a new column of layers for each new task; old columns are frozen; lateral connections allow the new task to leverage old features without overwriting them. Never forgets, but grows linearly with tasks.
- **Replay methods**: store a small buffer of old task examples (or use a generative model to replay them) and mix them into new task training — the simplest and often most effective approach.
- **LoRA for continual learning**: train separate LoRA adapters per task; at inference, merge the relevant adapter — no forgetting, scales cleanly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# --- Elastic Weight Consolidation (EWC) ---
class EWC:
    """
    EWC (Kirkpatrick et al., 2017):
    Protect important weights for old tasks using Fisher information.
    L = L_new + lambda * sum_i F_i * (theta_i - theta_i*)^2
    
    F_i ≈ E[(d log p(y|x, theta) / d theta_i)^2] — diagonal Fisher
    High Fisher = parameter important for old task = penalize deviation heavily.
    """
    def __init__(self, model: nn.Module, dataloader, device='cpu'):
        self.model = model
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader, device)

    def _compute_fisher(self, dataloader, device):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            output = self.model(x)
            loss = F.nll_loss(F.log_softmax(output, -1), y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()
        # Normalize
        N = len(dataloader)
        for n in fisher:
            fisher[n] /= N
        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC regularization term."""
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return loss

# --- Progressive Neural Networks (column-based) ---
class ProgressiveColumn(nn.Module):
    """
    New task gets a new column. Old columns are frozen.
    Lateral connections from old columns to new column via adapters.
    Never forgets — but grows linearly with tasks.
    """
    def __init__(self, old_columns: list, d_model: int):
        super().__init__()
        self.old_columns = nn.ModuleList(old_columns)
        for col in self.old_columns:
            col.requires_grad_(False)  # Freeze all old columns
        # New column + lateral adapters
        self.new_layer = nn.Linear(d_model, d_model)
        self.adapters  = nn.ModuleList([nn.Linear(d_model, d_model) for _ in old_columns])

    def forward(self, x: torch.Tensor, old_activations: list) -> torch.Tensor:
        new_out = self.new_layer(x)
        for adapter, h_old in zip(self.adapters, old_activations):
            new_out = new_out + adapter(h_old)  # lateral connection
        return F.relu(new_out)

print("Continual learning classes defined")
```

---

### 10.3 Interpretability — GradCAM & SHAP

**What is it?** Techniques for explaining which parts of an input are responsible for a model's prediction — making black-box neural networks more transparent and trustworthy.

**Why it matters:** Without interpretability, it is impossible to debug model failures, detect spurious correlations, or satisfy regulatory requirements (e.g., "right to explanation" in GDPR). Interpretability is essential for deploying models in high-stakes domains.

**How it works:**
- **GradCAM**: registers backward hooks on the target convolutional layer; computes `alpha_k = global_average_pool(d_score / d_A_k)` — the average gradient for each feature map channel; the class activation map is `ReLU(sum_k alpha_k * A_k)`, highlighting spatial regions the model used.
- **Attention visualization**: extracts the `(B, num_heads, T, T)` attention weight matrices from Transformer layers via forward hooks — shows which tokens each position attends to most strongly.
- **SHAP (Shapley values)**: assigns each input feature a contribution score based on the game-theoretic Shapley value — measures the average marginal contribution of a feature across all possible subsets. Model-agnostic but computationally expensive for large inputs.
- **Probing classifiers**: train a simple linear classifier on intermediate layer representations to test whether a specific linguistic or visual concept is encoded at that layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- GradCAM ---
class GradCAM:
    """
    GradCAM: visualize which spatial regions are important for a CNN prediction.
    Uses gradient of class score w.r.t. feature map activations.
    CAM = ReLU(sum_k alpha_k * A_k), alpha_k = global_avg_pool(d score / d A_k)
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> torch.Tensor:
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # alpha_k: average gradient over spatial dimensions
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize [0,1]
        return cam.squeeze()

# --- Attention Visualization ---
class AttentionVisualizer:
    """Extract and visualize attention weights from transformer heads."""

    def __init__(self, model: nn.Module):
        self.attention_maps = {}
        self._register_hooks(model)

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(
                    lambda m, inp, out, n=name: self.attention_maps.update({n: out[1]})
                )

    def get_attention(self, layer_name: str) -> torch.Tensor:
        return self.attention_maps.get(layer_name)  # (B, num_heads, T, T)

# Demo
cnn = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10)
)

# In practice: target_layer = cnn[2] (last conv)
# gradcam = GradCAM(cnn, target_layer=cnn[2])
# x = torch.randn(1, 3, 224, 224)
# heatmap = gradcam.generate(x, class_idx=3)

print("Interpretability classes defined")
```

---

### 10.4 Neural Architecture Search (NAS)

**What is it?** Automated methods for searching over the space of possible neural network architectures to find designs that optimize a target metric (accuracy, latency, parameter count) for a given task and hardware.

**Why it matters:** Hand-designed architectures (ResNet, VGG) require years of expert iteration. NAS discovered architectures like EfficientNet and MobileNet that match or exceed hand-designed models with dramatically fewer FLOPs — and NAS is now applied to find optimal attention patterns, layer configurations, and activation functions in LLMs.

**How it works:**
- **Reinforcement Learning NAS** (early): a controller RNN generates architecture descriptions; trained models are evaluated on a validation set; accuracy is the reward. Expensive — requires training thousands of candidate networks.
- **DARTS (Differentiable Architecture Search)**: replaces discrete architecture choices with a continuous relaxation — each edge in the computation graph is a weighted sum of all candidate operations. Architecture weights `alpha` and model weights are jointly optimized via gradient descent on a bilevel objective.
- **One-shot NAS**: train a single "supernet" containing all candidate architectures as subgraphs; sample subnets for evaluation without additional training — weight sharing makes this feasible.
- **Hardware-aware NAS**: jointly optimizes accuracy and latency on target hardware (mobile CPU, edge TPU) — EfficientNet was found this way.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DARTS: Differentiable Architecture Search ---
OPERATIONS = {
    'skip':     lambda C: nn.Identity(),
    'conv3x3':  lambda C: nn.Conv2d(C, C, 3, padding=1, bias=False),
    'conv5x5':  lambda C: nn.Conv2d(C, C, 5, padding=2, bias=False),
    'maxpool':  lambda C: nn.MaxPool2d(3, 1, 1),
    'avgpool':  lambda C: nn.AvgPool2d(3, 1, 1),
}

class MixedOp(nn.Module):
    """
    DARTS mixed operation: weighted sum of all candidate operations.
    Architecture parameters alpha are learned jointly with weights.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.ops = nn.ModuleList([op(channels) for op in OPERATIONS.values()])

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # Soft mixture during search; discrete (argmax) during evaluation
        weights = F.softmax(alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class DARTSCell(nn.Module):
    """DARTS cell with learnable architecture parameters per edge."""
    def __init__(self, channels: int, n_nodes: int = 4):
        super().__init__()
        self.n_nodes = n_nodes
        # Each node has edges from all previous nodes
        self.edges = nn.ModuleList()
        self.arch_params = nn.ParameterList()

        for node in range(n_nodes):
            for _ in range(node + 2):  # connections from node + 2 inputs
                self.edges.append(MixedOp(channels))
                self.arch_params.append(nn.Parameter(torch.randn(len(OPERATIONS))))

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        states = [s0, s1]
        edge_idx = 0
        for node in range(self.n_nodes):
            node_out = sum(
                self.edges[edge_idx + i](states[i], self.arch_params[edge_idx + i])
                for i in range(len(states))
            )
            states.append(node_out)
            edge_idx += len(states) - 1
        return torch.cat(states[2:], dim=1)

print("NAS (DARTS) classes defined")
```

---

### 10.5 Hyperparameter Optimization with Optuna

**What is it?** Automated search for the best combination of hyperparameters (learning rate, batch size, architecture dimensions, dropout, etc.) using principled algorithms that learn from past trials.

**Why it matters:** Hyperparameters can change model accuracy by 10–30%. Manual tuning is slow and brittle — automated HPO consistently finds better configurations while using fewer total compute hours than grid search.

**How it works:**
- **TPE (Tree-structured Parzen Estimator)**: Optuna's default sampler — fits two density models, `p(x | good)` and `p(x | bad)`, using past trial results; new hyperparameters are sampled from the ratio `p(x | good) / p(x | bad)`, directing search toward promising regions.
- **Pruning**: intermediate metric values are reported during each trial; trials performing worse than the median of completed trials at the same step are stopped early — dramatically reduces wasted compute (Hyperband, Successive Halving).
- **Bayesian optimization vs random search**: Bayesian methods (TPE, Gaussian processes) are ~10x more efficient than random search for expensive objectives, but random search parallelizes better.
- **Population Based Training (PBT)**: evolves hyperparameters during training — poor-performing runs copy the hyperparameters and weights of better-performing runs, enabling online adaptation rather than fixed pre-training choices.

```python
import optuna
import torch
import torch.nn as nn

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective: define hyperparameter search space and return metric.
    Optuna uses TPE (Tree-structured Parzen Estimator) by default — Bayesian optimization.
    """
    # Define search space
    lr          = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size  = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    n_layers    = trial.suggest_int('n_layers', 2, 8)
    hidden_dim  = trial.suggest_int('hidden_dim', 64, 512, step=64)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    # Build model
    dims   = [784] + [hidden_dim] * n_layers + [10]
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1])]
        if i < len(dims) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    model = nn.Sequential(*layers)

    # Optimizer
    opts = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW, 'sgd': torch.optim.SGD}
    optimizer = opts[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay)

    # Fake training loop (replace with real)
    model.train()
    for step in range(20):
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Report intermediate values for pruning
        trial.report(loss.item(), step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss.item()

# Run optimization
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
)
study.optimize(objective, n_trials=5, timeout=60)

print(f"Best trial: {study.best_trial.params}")
print(f"Best value: {study.best_trial.value:.4f}")
```

**Interview Insight:**
- **Optuna vs Ray Tune**: Optuna is simpler for single-machine HPO; Ray Tune scales to distributed search across clusters.
- **TPE**: models p(hyperparams | good) and p(hyperparams | bad) as KDE; samples from ratio. More efficient than random/grid search.
- **Pruning**: early stopping of poor trials based on intermediate metrics — Hyperband and Successive Halving are common strategies.
- **Population Based Training (PBT)**: evolve hyperparameters during training, not just before. Used by DeepMind for AlphaStar.

---

## Key Interview Tradeoffs Summary

| Topic | Option A | Option B | When to Choose |
|-------|----------|----------|----------------|
| Norm | BatchNorm | LayerNorm | BN for CNN+large batch; LN for Transformer+variable seq |
| PE | Sinusoidal | RoPE | Sinusoidal for fixed-length; RoPE for long-context LLMs |
| Optimizer | SGD+momentum | AdamW | SGD for CV (better generalization); AdamW for LLMs |
| PEFT | LoRA | Adapters | LoRA if inference latency critical; Adapters for more flexibility |
| Parallelism | DDP | ZeRO-3 | DDP if model fits on 1 GPU; ZeRO-3 for multi-hundred B models |
| Quantization | INT8 | INT4 | INT8 for minimal accuracy loss; INT4 for maximum compression |
| Attention | Standard | Flash Attention | Always use FlashAttn if available — same output, less memory |
| Init | Xavier | Kaiming | Xavier for sigmoid/tanh; Kaiming for ReLU networks |
| Loss (imbalance) | CE | Focal Loss | Focal for severe class imbalance (< 1% positive rate) |
| Generation speed | Greedy | Speculative | Speculative for large model on constrained hardware |

---

## Common Interview Questions & Answers

**Q: Why does batch normalization help training?**
A: Reduces internal covariate shift, allows higher LR, acts as regularizer (noise from batch statistics), smooths the loss landscape. The learnable scale/shift allows recovering any representation the network needs.

**Q: What is the difference between attention and convolution?**
A: Convolution uses fixed, local, translation-equivariant filters. Attention is dynamic (query-dependent), global (all-to-all), and content-based. Transformers have O(N^2) complexity vs O(N*k) for conv. For long sequences, attention is expensive but more expressive for long-range dependencies.

**Q: How would you debug NaN losses?**
A: (1) Check for inf/nan in inputs. (2) Lower LR. (3) Add gradient clipping. (4) Check log of zero (add epsilon). (5) Use `torch.autograd.set_detect_anomaly(True)`. (6) Check for exploding activations with hooks. (7) Verify loss function numerics (e.g., CrossEntropyLoss expects raw logits, not log-softmax).

**Q: How does Adam work and what are its failure modes?**
A: Adam maintains first (m) and second (v) moment estimates of gradients, bias-corrected. Update: `theta -= lr * m_hat / (sqrt(v_hat) + eps)`. Failure modes: (1) weight decay is wrong (use AdamW), (2) can generalize worse than SGD in some settings due to per-parameter adaptive LR masking the noise that aids generalization, (3) requires more memory (2 extra buffers per param).

**Q: Explain the reparameterization trick in VAEs.**
A: Direct sampling `z ~ N(mu, sigma^2)` is not differentiable. Reparameterize as `z = mu + sigma * epsilon`, `epsilon ~ N(0,1)`. Gradients flow through mu and sigma; epsilon is a fixed random draw. This makes the sampling operation differentiable with respect to the network parameters.

**Q: What is gradient checkpointing and when do you use it?**
A: Gradient checkpointing discards intermediate activations during forward pass and recomputes them during backward when needed. Reduces activation memory from O(N) to O(sqrt(N)) for a sequence of N operations, at the cost of ~33% extra compute. Use when GPU OOM during training of large models — standard practice for fine-tuning LLMs.