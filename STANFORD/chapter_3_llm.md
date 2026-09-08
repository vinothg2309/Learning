chapter_3_llm.md
# Resources

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |

---

# Mixture of Experts (MoE)

## Table of Contents

### 1. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
   - [What is MoE?](#what-is-moe)
   - [Purpose & Benefits](#purpose--benefits)
   - [Architecture Overview](#architecture-overview)
   - [How It Works: Step-by-Step](#how-it-works-step-by-step)
   - [Critical Concept: Token-Level Routing](#️-critical-concept-token-level-routing)
   - [Simple Analogy](#simple-analogy)
   - [Real-World Examples](#real-world-examples)
   - [Key Advantages vs Traditional Models](#key-advantages-vs-traditional-models)
   - [Training Challenges](#training-challenges)
   - [Visual: Parameter Usage Comparison](#visual-parameter-usage-comparison)
   - [When to Use MoE?](#when-to-use-moe)

### 2. [Sparse MoE vs Dense MoE](#sparse-moe-vs-dense-moe)
   - [What are Sparse and Dense MoE?](#what-are-sparse-and-dense-moe)
   - [Architecture Comparison](#architecture-comparison)
   - [Key Differences](#key-differences)
   - [Visualizing Efficiency](#visualizing-efficiency)
   - [When to Use Which?](#when-to-use-which)

### 3. [MOE in Transformer-based Model](#moe-in-transformer-based-model)
   - [Each Expert = Complete FFN](#each-expert--complete-ffn)
   - [Do All Layers Use MoE?](#do-all-layers-use-moe)
   - [Parameter Efficiency (Mixtral 8x7B)](#parameter-efficiency-mixtral-8x7b)
   - [Routing Collapse](#routing-collapse)
     - [What is Routing Collapse?](#what-is-routing-collapse)
     - [Why Does This Happen?](#why-does-this-happen)
     - [Problems Caused by Routing Collapse](#problems-caused-by-routing-collapse)
     - [How to Mitigate Routing Collapse](#how-to-mitigate-routing-collapse)
     - [Real-World Implementation: Mixtral 8x7B](#real-world-implementation-example-mixtral-8x7b)

### 4. [Response Generation](#response-generation)
   - [How LLMs Generate Text](#how-llms-generate-text)
   - [Impact of Temperature](#impact-of-temperature-on-probability)
   - [Guided Decoding](#guided-decoding)

### 5. [Prompting Strategies](#prompting-strategies)
   - [How to Ask LLMs Effectively](#how-to-ask-llms-effectively)
   - [Prompt Structure](#prompt-structure)
   - [In-Context Learning](#in-context-learning)
   - [Context Rot](#context-rot)
   - [Chain of Thought](#chain-of-thoughts)
   - [Self-Consistency](#self-consistency)

### 6. [Inference Optimization](#inference-optimization)
   - [KV Cache](#kv-cache)
     - [What is KV Cache?](#what-is-kv-cache)
     - [K, V Generation](#k-v-generation)
     - [How KV Cache Saves Computation](#how-kv-cache-saves-computation)
     - [Memory Trade-off](#memory-trade-off)
     - [KV Cache Growth](#kv-cache-growth)
   - [Traditional Continuous Memory Allocation](#traditional-continuous-memory-allocation)
     - [The Problem: Wasted Memory](#the-problem-wasted-memory)
     - [Memory Cannot Be Reclaimed](#memory-cannot-be-reclaimed)
     - [Summary: Why This Is Inefficient](#summary-why-this-is-inefficient)
   - [PagedAttention](#sharing-attention-heads---pagedattention)
     - [What is PagedAttention?](#what-is-pagedattention)
     - [How PagedAttention Works](#how-pagedattention-works)
   - [KV Cache Dimension Reduction](#kv-cache-dimension-reduction)
     - [Multi-Latent Attention (MLA)](#multi-latent-attention)
       - [The Problem MLA Solves](#the-problem-mla-solves)
       - [How MLA Works: KV Compression](#how-mla-works-kv-compression)
       - [Memory Savings](#memory-savings-example)
       - [Trade-offs](#trade-offs)
       - [Real-World Impact: DeepSeek-V2](#real-world-impact-deepseek-v2)
   - [Speculative Sampling](#speeding-up-decoding-with-speculative-sampling)
     - [How It Works](#how-it-works)
     - [Why It's Faster](#why-its-faster)
   - [Multi-Token Prediction](#multi-token-prediction)
     - [How It Works](#how-it-works-1)
     - [Key Difference from Speculative Sampling](#key-difference-from-speculative-sampling)

### 7. [Sampling Strategies: Top-P, Top-K, and Temperature](#sampling-strategies-top-p-top-k-and-temperature)
   - [Temperature (N)](#temperature-n)
   - [Top-K Sampling](#top-k-sampling)
   - [Top-P (Nucleus) Sampling](#top-p-nucleus-sampling)
   - [Comparison & Trade-offs](#comparison--trade-offs)
   - [Real-World Examples](#real-world-examples)
   - [Best Practices](#best-practices)

---

## What is MoE?

**Mixture of Experts (MoE)** is a neural network architecture that divides a large model into multiple smaller "expert" networks. Instead of using all parameters for every input, a **gating network (router)** selectively activates only the most relevant experts for each specific input.

Think of it like a hospital: instead of every doctor examining every patient, a receptionist (router) directs patients to specialized doctors (experts) based on their symptoms.

## Purpose & Benefits

### 1. **Computational Efficiency**
- Only a fraction of the model's parameters are active per input
- Example: A 50B parameter model might use only 12B parameters per token
- Results in **faster inference** without sacrificing model capacity

### 2. **Specialization**
- Each expert learns to handle specific types of patterns or tasks
- Natural division of labor emerges during training
- One expert might specialize in math, another in creative writing

### 3. **Scalability**
- Can scale to trillions of parameters without proportional compute increase
- Add more experts without linearly increasing computation cost
- Enables massive models that are still practical to run

## Architecture Overview

`MoE is applied in Feed Foward Network(FFN) in transformer decoder`

```
                    Input Token
                         |
                         ↓
              ┌──────────────────────┐
              │   Router Network     │
              │  (Gating Function)   │
              └──────────────────────┘
                         |
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   ┌─────────┐      ┌─────────┐     ┌─────────┐
   │Expert 1 │      │Expert 2 │ ... │Expert N │
   │  (FFN)  │      │  (FFN)  │     │  (FFN)  │
   └─────────┘      └─────────┘     └─────────┘
        |                |                |
        └────────────────┼────────────────┘
                         ↓
              ┌──────────────────────┐
              │  Weighted Combine    │
              │  (Top-K Selection)   │
              └──────────────────────┘
                         |
                         ↓
                    Final Output
```

![MoE Architecture](images/moe_architecture.png)

### Components Explained:

1. **Input Layer**: Receives the token/input data (e.g., a word or sentence embedding)

2. **Router/Gating Network**: 
   - **Purpose**: Decides which experts are most relevant for this input
   - Outputs probability scores for each expert
   - Typically selects top-k experts (e.g., top-2)
   - Example: For "What is 2+2?", router gives Math Expert high score

3. **Expert Networks**: 
   - Multiple parallel feed-forward networks (FFN)
   - Each expert is a complete neural network layer (same architecture, different weights)
   - Experts naturally specialize during training
   - Example: Math Expert, Language Expert, Code Expert, etc.

4. **Output Combination**: 
   - Takes outputs from selected experts
   - Combines them using weighted sum (weights from router scores)
   - Formula: `Output = Σ(router_weight_i × expert_output_i)` for selected experts only

## How It Works: Step-by-Step

```
Input Token: "Calculate 25 + 37"
         ↓
   Router Network
   (analyzes input)
         ↓
   Expert Scores:
   Math Expert: 0.85 ✓ (selected)
   Language Expert: 0.10
   Code Expert: 0.65 ✓ (selected)
   History Expert: 0.05
         ↓
   Top-2 Experts Activated
   (Math + Code)
         ↓
   Weighted Combination
   0.85 × Math_output + 0.65 × Code_output
         ↓
   Final Output: "62"
```

## ⚠️ Critical Concept: Token-Level Routing

### MoE Routes EACH TOKEN Independently, Not Entire Queries!

This is a common misconception that needs clarification:

❌ **WRONG**: The entire user query "Calculate 25 + 37 and write a poem" goes to one expert  
✅ **CORRECT**: Each token is routed independently to potentially different experts

### How Token-Level Routing Works

```
User Query: "Calculate 25 + 37 and write a poem"

Tokenized: ["Calculate", "25", "+", "37", "and", "write", "a", "poem"]

Token-by-Token Routing:
┌───────────┬─────────────────────────────────────┐
│ Token     │ Experts Activated (Top-2)           │
├───────────┼─────────────────────────────────────┤
│ Calculate │ Math Expert + Code Expert           │
│ 25        │ Math Expert + Number Expert         │
│ +         │ Math Expert + Symbol Expert         │
│ 37        │ Math Expert + Number Expert         │
│ and       │ Language Expert + Logic Expert      │
│ write     │ Language Expert + Creative Expert   │
│ a         │ Language Expert + Grammar Expert    │
│ poem      │ Creative Expert + Language Expert   │
└───────────┴─────────────────────────────────────┘
```

### Visual Example

```
Query: "Python code for sorting"
         ↓
    [Tokenize]
         ↓
["Python", "code", "for", "sorting"]
    ↓        ↓       ↓        ↓
  Router   Router  Router   Router
    ↓        ↓       ↓        ↓
[E1,E3]  [E3,E5]  [E2,E7]  [E1,E3]
(Code)   (Code)   (Lang)   (Code)

Each token gets its own expert selection!
```

![Token-Level Routing](images/token_level_routing.png)

### Key Points

1. **Independent Routing**: Each token position in the sequence gets routed separately
2. **Different Experts per Token**: Token 1 might use Expert A+B, Token 2 might use Expert C+D
3. **Context-Aware**: The router considers the token AND its surrounding context to decide routing
4. **Dynamic**: Even the same word in different contexts can route to different experts

### Example: Same Word, Different Routing

```
Sentence 1: "Apple is a tech company"
Token "Apple" → Tech Expert + Business Expert

Sentence 2: "I ate an apple for lunch"  
Token "Apple" → Food Expert + Health Expert

Same word, different context → different experts!
```

### Why This Matters

- **Flexibility**: Model adapts expert selection to each part of the input
- **Efficiency**: Only activates needed expertise for each specific token
- **Specialization**: Experts can focus on very specific token types or contexts
- **Performance**: Enables fine-grained routing decisions throughout the sequence

### Complete Flow Through a Query

```
Input: "What's 5+3? Write a story about it."

Processing Layer by Layer:
┌─────────────────────────────────────────────────┐
│ Layer 1 (MoE Layer 1):                          │
│ Token "What's" → [Language Expert, Query Expert]│
│ Token "5"      → [Math Expert, Number Expert]   │
│ Token "+"      → [Math Expert, Symbol Expert]   │
│ Token "3"      → [Math Expert, Number Expert]   │
│ Token "?"      → [Query Expert, Syntax Expert]  │
│ Token "Write"  → [Creative Expert, Lang Expert] │
│ Token "a"      → [Grammar Expert, Lang Expert]  │
│ Token "story"  → [Creative Expert, Narrative]   │
│ ... and so on                                    │
└─────────────────────────────────────────────────┘

(This repeats for every MoE layer in the model)
```

### Summary

🔑 **Remember**: MoE doesn't route the entire query to one set of experts. Instead, it makes routing decisions **independently for each token**, allowing fine-grained specialization throughout the processing of your input.

## Simple Analogy

Imagine a **customer service center**:
- **Traditional Model**: Every agent handles every type of question (inefficient)
- **MoE Model**: 
  - Receptionist (router) listens to your question
  - Routes you to 1-2 specialized agents (experts)
  - Technical issues → Tech Support expert
  - Billing questions → Billing expert
  - Product questions → Product expert

## Real-World Examples

### 1. **Mixtral 8x7B** (Mistral AI)
- **8 experts** of 7B parameters each
- Activates **top-2 experts** per token
- Total: 47B parameters, but only ~13B active per token
- Performance of 47B model with speed of 13B model

### 2. **Switch Transformer** (Google)
- Up to **1.6 trillion parameters**
- Activates only **1 expert** per token (top-1 routing)
- Demonstrates extreme scaling with MoE

### 3. **GPT-4** (OpenAI - rumored)
- Believed to use MoE architecture with 8 experts
- Each expert ~220B parameters
- Explains the model's efficiency despite massive size

## Key Advantages vs Traditional Models

| Aspect | Traditional Dense Model | MoE Model |
|--------|------------------------|-----------|
| **Parameters Used** | All (100%) | Subset (10-20%) |
| **Inference Speed** | Slower | Faster |
| **Specialization** | General purpose | Task-specific experts |
| **Scaling** | Linear cost increase | Sub-linear cost increase |
| **Model Size** | Limited by compute | Can be much larger |

## Training Challenges

1. **Load Balancing**: Ensuring all experts are used (prevent "expert collapse")
2. **Router Training**: Teaching the router to make good decisions
3. **Communication Overhead**: Routing adds some computational cost
4. **Memory**: All experts must fit in memory even if not all are active

## Visual: Parameter Usage Comparison

### Traditional Dense Model
```
Input → [████████████████████] (100% of parameters active)
        All 100B parameters processing every token
```

### MoE Model (8 experts, top-1 routing)
```
Input → [██░░░░░░░░░░░░░░░░░░] (12.5% of parameters active)
        Only 12.5B out of 100B parameters active per token
        
Available Experts: [E1] [E2] [E3] [E4] [E5] [E6] [E7] [E8]
Active for Token:   ██  ░░  ░░  ░░  ░░  ░░  ░░  ░░
                   (only Expert 1 activated)
```

### Efficiency Visualization
```
┌─────────────────────────────────────────────────┐
│ Dense Model: 100B params                        │
│ ████████████████████████████████████████████████│ 100% active
│ Inference: SLOW ⏱️  (all params compute)        │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ MoE Model: 100B params (8 experts × 12.5B)      │
│ ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ 12.5% active
│ Inference: FAST ⚡ (1 expert computes)          │
└─────────────────────────────────────────────────┘
```

**Key Insight**: MoE achieves the capacity of a 100B model with the speed of a 12.5B model!

## When to Use MoE?

✅ **Good for:**
- Large-scale language models
- Multi-task learning scenarios
- When inference speed matters
- Resource-constrained deployment

❌ **Less ideal for:**
- Small models (overhead not worth it)
- Single-task fine-tuning
- Extremely memory-limited environments

## Key Takeaway

MoE is like having a **team of specialists** instead of one generalist. You get the knowledge of many experts but only "pay" (compute) for consulting the most relevant ones for each question. This makes massive models practical and efficient!

---

# Sparse MoE vs Dense MoE

## What are Sparse and Dense MoE?

### **Sparse MoE** (Standard MoE)
Only a **subset** of experts are activated for each input token.

![Sparse MoE Diagram](images/sparse_moe_diagram.png)

### **Dense MoE** (Fully Connected)
**All experts** are activated for every input token, but with different weights.

## Architecture Comparison

### Sparse MoE (Top-K Routing)
```
Input: "Hello world"
         ↓
    Router decides
         ↓
Expert Weights: [0.9, 0.7, 0.1, 0.05, 0.03, 0.02, 0.01, 0.0]
                 ✓    ✓    ✗    ✗     ✗     ✗     ✗     ✗
         ↓
Activates: Expert 1 + Expert 2 only (Top-2)
         ↓
Output = 0.9 × E1 + 0.7 × E2
```

### Dense MoE (All Experts)
```
Input: "Hello world"
         ↓
    Router decides
         ↓
Expert Weights: [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.015, 0.005]
                 ✓    ✓     ✓     ✓    ✓     ✓     ✓      ✓
         ↓
Activates: ALL 8 experts (weighted)
         ↓
Output = Σ(weight_i × Expert_i) for all experts
```

## Key Differences

| Aspect | Sparse MoE | Dense MoE |
|--------|-----------|-----------|
| **Active Experts** | Few (top-k, e.g., 2 out of 8) | All experts |
| **Computation** | Low (only k experts compute) | High (all experts compute) |
| **Speed** | Fast ⚡ | Slower 🐢 |
| **Specialization** | High (clear expert roles) | Moderate (blended expertise) |
| **Memory Efficiency** | High | Lower |

## Purpose

### Sparse MoE
- **Maximize efficiency**: Get large model capacity with minimal compute
- **Clear specialization**: Each expert focuses on specific tasks
- **Scale to extreme sizes**: Trillions of parameters feasible

### Dense MoE
- **Robust predictions**: All experts contribute to every decision
- **Better generalization**: No single expert can fail the model
- **Ensemble-like behavior**: Combines diverse perspectives

## Real-World Examples

### Sparse MoE Models
1. **Mixtral 8x7B**: 8 experts, activates top-2 (25% active)
2. **Switch Transformer**: 2048 experts, activates top-1 (0.05% active)
3. **GPT-4** (rumored): 8 experts, activates top-2

### Dense MoE Models
1. **Ensemble models**: All sub-models vote on predictions
2. **Traditional ensemble learning**: Random forests, boosting

## Visualizing Efficiency

```
┌──────────────────────────────────────────┐
│ Sparse MoE (8 experts, top-2)            │
│ Active: ██░░░░░░ (25% of experts)        │
│ Speed: ⚡⚡⚡⚡ Very Fast                    │
│ Specialization: ★★★★★ High               │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ Dense MoE (8 experts, all active)        │
│ Active: ████████ (100% of experts)       │
│ Speed: ⚡⚡ Slower                         │
│ Specialization: ★★★ Moderate             │
└──────────────────────────────────────────┘
```

## When to Use Which?

### Use Sparse MoE when:
✅ You need maximum inference speed  
✅ Model size is very large (billions/trillions of parameters)  
✅ Clear task specialization is desired  
✅ Running on resource-constrained hardware  

### Use Dense MoE when:
✅ Prediction quality is more important than speed  
✅ Model is relatively smaller  
✅ You want ensemble-like robustness  
✅ All experts should contribute their knowledge  

## Key Takeaway

**Sparse MoE** = Select few specialists for each question (fast & efficient)  
**Dense MoE** = Consult all specialists for each question (thorough but slower)

Most modern large language models (like Mixtral, GPT-4) use **Sparse MoE** because it provides the best balance of capacity, speed, and efficiency!

# MOE in Transformer-based Model

`MoE is done in FFN in transformer decoder.`

![MoE in Transformer](images/moe_in_transformer.png)

## Each Expert = Complete FFN

Yes — each expert is a **full independent FFN** with its own weight matrices:

```
Standard layer:  MHA → FFN
                         ↓
                  Linear(d→4d) → GeLU → Linear(4d→d)

MoE layer:       MHA → Router → picks 2 of 8 experts
                         ↓
          Expert 3: Linear(d→4d) → GeLU → Linear(4d→d)  ← full FFN
          Expert 7: Linear(d→4d) → GeLU → Linear(4d→d)  ← full FFN

          Output = weighted sum of Expert 3 + Expert 7
```

**MHA is never replaced** — only the FFN part becomes MoE.

## Do All Layers Use MoE?

**No — it depends on the model design.**

| Model | MoE Layers |
|---|---|
| Mixtral 8x7B | All 32 layers use MoE |
| Some Google models | Every other layer (alternating) |
| Standard dense models | No MoE at all |

Applying MoE to every layer multiplies FFN parameters by `num_experts` — costly. Alternating is a common compromise.

## Parameter Efficiency (Mixtral 8x7B)

```
Total params:   47B   (8 experts × ~6B each)
Active params:  ~12B  (only 2 experts fire per token)

→ Capacity of a 47B model at compute cost of ~12B
```

---

## Routing Collapse

### What is Routing Collapse?

**Routing Collapse** (also called **Expert Collapse**) is a training problem where the router learns to send most/all tokens to just one or two "favorite" experts, leaving other experts underutilized or completely ignored.

```
Before Collapse (Healthy):          After Collapse (Problem):
┌────────────────────────┐          ┌────────────────────────┐
│ Expert 1: 12% of tokens│          │ Expert 1: 85% ████████│
│ Expert 2: 14% of tokens│          │ Expert 2: 10% █       │
│ Expert 3: 13% of tokens│          │ Expert 3:  3% ▌       │
│ Expert 4: 11% of tokens│          │ Expert 4:  1%         │
│ Expert 5: 12% of tokens│          │ Expert 5:  0.5%       │
│ Expert 6: 13% of tokens│          │ Expert 6:  0.3%       │
│ Expert 7: 13% of tokens│          │ Expert 7:  0.1%       │
│ Expert 8: 12% of tokens│          │ Expert 8:  0.1%       │
└────────────────────────┘          └────────────────────────┘
   Balanced Usage                    Collapsed! (Waste)
```

### Why Does This Happen?

1. **Reinforcement Effect**: If Expert 1 performs well early in training, router sends more tokens to it
2. **Rich Get Richer**: Expert 1 gets more training data → improves faster → gets selected even more
3. **Poor Get Poorer**: Other experts get little data → don't improve → rarely selected → become useless
4. **Local Minimum**: Router gets stuck in a suboptimal state where it only trusts one expert

### Problems Caused by Routing Collapse

❌ **Wasted Parameters**: You have 8 experts but only using 1-2 → 75%+ of model is useless  
❌ **Loss of Specialization**: No division of labor, defeating the purpose of MoE  
❌ **Poor Performance**: Effectively becomes a smaller, weaker model  
❌ **Training Inefficiency**: Resources spent on unused experts  

### How to Mitigate Routing Collapse

#### 1. **Load Balancing Loss** (Most Common)
Add an auxiliary loss function that penalizes uneven expert usage:

```
Total Loss = Task Loss + λ × Load Balance Loss

Load Balance Loss = Encourages equal token distribution
```

**Example**: If Expert 1 gets 50% of tokens, add penalty to force router to use other experts

#### 2. **Expert Capacity Limits**
Set a maximum number of tokens each expert can process:

```
Expert Capacity = (Total Tokens / Number of Experts) × Capacity Factor

Example: 1000 tokens ÷ 8 experts × 1.25 = 156 tokens per expert max

If Expert 1 reaches capacity → Router MUST choose other experts
```

#### 3. **Random Routing (During Training)**
Occasionally route tokens randomly instead of using router's choice:

```
if random() < 0.1:  # 10% of the time
    route_randomly()
else:
    route_by_router_scores()
```

Forces all experts to stay trained and useful.

#### 4. **Expert Dropout**
Randomly disable certain experts during training:

```
Available Experts: [E1, E2, E3, E4, E5, E6, E7, E8]
This batch:        [E1, ✗ , E3, ✗ , E5, E6, ✗ , E8]
                       (E2, E4, E7 dropped)
```

Router must learn to work with different expert combinations.

### Real-World Implementation Example: Mixtral 8x7B

**Load Balancing Strategy**:
```
- Tracks how many tokens each expert receives per batch
- If imbalance detected (e.g., Expert 1 gets >20%, others <5%)
- Adds penalty to training loss
- Router adjusts to distribute more evenly
- Target: Each expert handles ~12.5% of tokens (1/8)
```

### Visualizing the Solution

```
Training Progress with Load Balancing:

Epoch 1 (Collapse starting):
Expert Usage: [45%, 30%, 10%, 5%, 4%, 3%, 2%, 1%]
Load Balance Penalty: HIGH ⚠️
             ↓
Epoch 5 (Correcting):
Expert Usage: [20%, 18%, 15%, 12%, 11%, 10%, 8%, 6%]
Load Balance Penalty: MEDIUM ⚠️
             ↓
Epoch 10 (Balanced):
Expert Usage: [13%, 13%, 12%, 12%, 13%, 12%, 13%, 12%]
Load Balance Penalty: LOW ✓
```

### Key Metrics to Monitor

- **Expert Utilization Rate**: % of tokens each expert processes
- **Router Entropy**: Higher entropy = more diverse routing (good!)
- **Unused Experts**: Count of experts getting <1% of tokens

### Simple Analogy

**Without Load Balancing**:
- Restaurant with 8 chefs
- All customers ask for Chef 1 because they're initially popular
- Other 7 chefs sit idle → waste of talent and resources

**With Load Balancing**:
- Restaurant manager ensures each chef gets equal customers
- All chefs stay practiced and skilled
- Better service and resource utilization

### Key Takeaway

**Routing Collapse** = Router gets lazy and only uses favorite experts  
**Solution** = Add load balancing mechanisms to force equal expert usage  
**Result** = All experts stay trained and useful, maximizing MoE benefits!


# Response Generation

## How LLMs Generate Text

**Response Generation** is how language models create text output, **one token at a time**.

### Process:
1. Input → Tokenize and process through transformer
2. Model outputs probability scores for each possible next token
3. Select next token based on probabilities (sampling)
4. Add token to sequence and repeat until done

### Example:
```
Input: "The cat sat on the"
→ Predicts "mat" → Add to sequence
→ "The cat sat on the mat"
→ Predicts "." → Add
→ "The cat sat on the mat."
→ Predicts [STOP]
```

This is **autoregressive generation** - each token depends on previous ones.

![Response Generation Process](images/response_generation_process.png)

## Impact of Temperature on probability

`Everything in the transformer architecture is deterministic except output sampling. Sampling (probability) of next token is only non-deterministic.`

### What is Temperature?

**Temperature** controls how random or predictable the model's output is.

- **Low Temperature (0.0 - 0.3)**: More predictable, picks most likely tokens
- **Medium Temperature (~1.0)**: Balanced (default)
- **High Temperature (1.5+)**: More creative/random

### Example:
```
Completing "The weather is"

Temperature = 0.1 (Predictable):
- "sunny" (95%) ← Almost always picks this
- "nice" (3%)
- "cloudy" (2%)

Temperature = 1.0 (Balanced):
- "sunny" (40%)
- "nice" (25%)
- "cloudy" (20%)
- "pleasant" (15%)

Temperature = 2.0 (Creative):
- "sunny" (22%)
- "nice" (20%)
- "mysterious" (15%) ← Unusual words appear
- "cloudy" (18%)
```

### When to Use:
- **Code/Math**: Low (0.0-0.2) - Need exact answers
- **Chatbot**: Medium (0.5-0.8) - Natural conversation
- **Creative Writing**: High (0.8-1.5) - Want variety

![Temperature Effect on Sampling](images/temperature_sampling.png)

## Guided Decoding

### What is Guided Decoding?

**Guided Decoding** forces the model to follow specific rules or formats while generating text.

### Purpose:
Ensure outputs are valid and usable (correct JSON, valid code, proper format)

### How it Works:
Instead of allowing any token, only tokens that maintain valid format are allowed.

### Example:
```
Task: "Generate user info"
Constraint: Must be valid JSON

Without Guided Decoding:
"The user name is John and age 30..." ✗ (not JSON)

With Guided Decoding:
{"name": "John", "age": 30} ✓ (guaranteed valid JSON)
```

### Use Cases:
- Generate valid code (Python, SQL, JSON)
- Create structured API responses
- Ensure format compliance (dates, emails)
- Block unwanted content

### Benefits:
✅ Output always in correct format  
✅ No parsing errors  
✅ Can use output directly in systems  

![Guided Decoding Illustration](images/guided_decoding.png)

# Prompting Strategies

## How to Ask LLMs Effectively

**Prompting Strategies** are different ways to structure your questions to get better responses from language models.

### Common Strategies:

#### 1. **Zero-Shot** (No examples)
Just ask directly:
```
"Translate 'Hello' to French."
→ "Bonjour"
```

#### 2. **Few-Shot** (With examples)
Show examples first:
```
English: Hello → French: Bonjour
English: Goodbye → French: Au revoir
English: Thank you → French: ?
→ "Merci"
```

#### 3. **Chain-of-Thought** (Show reasoning)
Ask to explain step-by-step:
```
"If 5 apples cost $10, how much for 8? Show steps."
→ Cost per apple: $10 ÷ 5 = $2
→ For 8 apples: $2 × 8 = $16
```

#### 4. **Role Prompting**
Assign expertise:
```
"You are a Python expert. Write a sorting function."
```

### When to Use:
| Strategy | Best For |
|----------|----------|
| Zero-Shot | Simple, straightforward tasks |
| Few-Shot | Complex tasks, specific formatting |
| Chain-of-Thought | Math, reasoning, logic problems |
| Role Prompting | Need domain expertise |

![Prompting Strategies Overview](images/prompting_strategies.png)

## Prompt structure

### How to Structure a Good Prompt

**Prompt Structure** is how you organize your prompt for best results.

### Key Components:

1. **Role** (Optional): "You are an expert data scientist"
2. **Context**: Background information
3. **Task**: What you want done
4. **Constraints**: Length, format, style requirements
5. **Examples**: Show desired format (if needed)

### Template:
```
┌─────────────────────────┐
│ [ROLE]                  │
│ You are a [expert]      │
├─────────────────────────┤
│ [TASK]                  │
│ Clear instruction       │
├─────────────────────────┤
│ [CONSTRAINTS]           │
│ Format, length, style   │
└─────────────────────────┘
```

### Example:

❌ **Bad** (vague):
```
"Tell me about functions"
```

✅ **Good** (structured):
```
Role: You are a Python tutor
Task: Explain functions to a beginner
Constraints: 
- Use simple language
- Include 1 code example
- Keep under 100 words
```

### Tips:
- Be specific about what you want
- Specify output format clearly
- Add examples for complex tasks
- Include length/style constraints

![Prompt Structure Diagram](images/prompt_structure.png)

## In-context learning

### What is In-Context Learning?

**In-Context Learning** is when a model learns a new task from examples provided directly in the prompt, without any fine-tuning or retraining.

### How it Works:

You give examples in the prompt, and the model learns the pattern:

```
Example 1: Input → Output
Example 2: Input → Output
Example 3: Input → Output
New Input: ? → Model predicts based on pattern
```

### Real Example:

```
Sentiment Analysis (In-Context Learning):

"I love this product!" → Positive
"This is terrible." → Negative
"It's okay, not great." → Neutral
"Best purchase ever!" → ?

Model Output: Positive
```

The model learned the sentiment analysis task from just 3 examples!

### Key Points:

- **No Training Required**: Model adapts on the fly
- **Flexible**: Can teach new tasks instantly
- **Temporary**: Learning only lasts for that conversation
- **More Examples = Better**: Usually 3-10 examples work well

### Benefits:

✅ Instant task adaptation  
✅ No model retraining needed  
✅ Works with any task that has clear patterns  
✅ Can quickly test new ideas  

### Limitations:

❌ Limited by context window size  
❌ Not as accurate as fine-tuning for specialized tasks  
❌ Requires good examples  

![In-context Learning Visualization](images/in_context_learning.png)

## Context Rot

### What is Context Rot?

**Context Rot** (also called **Context Dilution**) occurs when important information in the prompt gets "lost" or "forgotten" as the conversation or prompt gets longer, causing the model's performance to degrade.

### Why Does This Happen?

LLMs have a **limited context window** (e.g., 4K, 8K, 128K tokens). As more content is added:
1. Early information becomes "distant" from current processing
2. Model pays less attention to older parts
3. Important details get diluted by less relevant information

### Simple Analogy:

🧠 Imagine reading a very long document:
- First page: You remember clearly
- Page 50: Details get fuzzy
- Page 100: Early information nearly forgotten

The model experiences the same effect!

### Example of Context Rot:

```
Start of conversation (Token 1-100):
User: "My name is John and I work at Google."
Model: ✓ Remembers clearly

After long conversation (Token 10,000+):
User: "What's my name and where do I work?"
Model: "I don't have that information." ❌
(Context rotted - early info lost in long conversation)
```

### Real-World Example:

```
Prompt with 20,000 tokens:

Token 1-1000: [Important instructions: "Always format as JSON"]
Token 1000-15000: [Long examples and context]
Token 15000-20000: [Actual task to perform]

Result: Model forgets JSON format requirement ❌
(Instruction from beginning got "rotted away")
```

### When Context Rot Happens:

❌ **Very long conversations** (100+ messages)  
❌ **Large documents** in prompt  
❌ **Many examples** in few-shot learning  
❌ **Important info at start**, task at end  

### How to Prevent Context Rot:

#### 1. **Keep Prompts Concise**
Only include necessary information:
```
❌ Bad: 10,000 words of background
✅ Good: 200 words of key points
```

#### 2. **Put Important Info Near the Task**
Place critical instructions close to the query:
```
✅ Good Structure:
- Brief context
- Examples
- [IMPORTANT INSTRUCTIONS] ← Near the end
- [ACTUAL TASK] ← At the end
```

#### 3. **Repeat Key Information**
Restate important points:
```
Start: "Format as JSON"
Middle: [examples]
End: "Remember: Format as JSON" ← Repeat
```

#### 4. **Use System Messages**
Many APIs let you set persistent system instructions that don't rot:
```
System: "You are a helpful assistant. Always be concise."
(This stays active throughout conversation)
```

#### 5. **Summarize Long Conversations**
Periodically condense history:
```
Instead of: 50 messages (15,000 tokens)
Use: Summary of key points (500 tokens)
```

#### 6. **Break Into Smaller Chunks**
For long documents, process in pieces:
```
❌ One 50,000 token document → context rot
✅ Ten 5,000 token chunks → better retention
```

### Context Window vs Context Rot:

| Context Window | Context Rot |
|----------------|-------------|
| Maximum tokens model can process | Quality degradation within that window |
| 128K tokens = can fit | But early tokens may be "forgotten" |
| Hard limit | Soft degradation |

### Practical Tips:

1. **Monitor conversation length** - Restart when it gets too long
2. **Front-load critical info** - If possible, put key details near the task
3. **Test with position** - Important info at start vs end, see which works better
4. **Use retrieval** - For very long documents, use RAG (Retrieval Augmented Generation)

### Key Takeaway:

**Context Rot** = Information "fades" in long prompts/conversations  
**Solution** = Keep prompts concise, repeat key info, place important details near the task  
**Result** = Better, more consistent model performance

## Chain of Thoughts

### What is Chain-of-Thought (CoT)?

**Chain-of-Thought** is a prompting technique where you ask the model to show its reasoning step-by-step before giving the final answer, like showing your work in math class.

**Why it works**: Breaking down complex problems into smaller steps improves accuracy, especially for reasoning and math tasks.

**Example**:
```
Without CoT:
Q: "If 5 apples cost $10, how much do 8 apples cost?"
A: "$16" (might be wrong)

With CoT:
Q: "If 5 apples cost $10, how much do 8 apples cost? Think step by step."
A: "Step 1: Cost per apple = $10 ÷ 5 = $2
    Step 2: Cost for 8 apples = $2 × 8 = $16
    Answer: $16" ✓ (more likely correct)
```

**Key Benefits**:
- ✅ Better accuracy on complex problems
- ✅ Easier to verify reasoning
- ✅ Helps catch errors in logic

**Simple prompt addition**: Add "Let's think step by step" or "Show your reasoning"

![Chain-of-Thought Reasoning](images/chain_of_thought.png)

## Self-consistency

### What is Self-Consistency?

**Self-Consistency** is a technique where you generate multiple answers to the same question and pick the most common one (majority vote).

**How it works**:
1. Ask the same question multiple times (e.g., 5-10 times)
2. Collect all answers
3. Choose the answer that appears most frequently

**Example**:
```
Question: "What is 23 × 17?"

Run 1: 391 ✓
Run 2: 391 ✓
Run 3: 381 ✗
Run 4: 391 ✓
Run 5: 391 ✓

Most common answer: 391 (appears 4 times)
Final answer: 391 ✓
```

**Why it works**:
- Random errors are less likely to repeat
- Correct answer appears more consistently
- Like getting multiple opinions and taking the majority

**Use cases**:
- Math problems
- Logic puzzles
- Critical decisions where accuracy matters

**Trade-off**: Slower (multiple runs) but more accurate

![Self-Consistency Approach](images/self_consistency.png)

# Inference optimization

### What is Inference Optimization?

**Inference Optimization** refers to techniques that make language models run faster and use less memory during text generation (inference), without retraining the model.

**Goal**: Generate text faster while maintaining quality

**Common techniques**:
- KV Caching: Reuse computed values
- Quantization: Use smaller numbers (16-bit → 8-bit)
- Batching: Process multiple requests together
- Model pruning: Remove unnecessary parts

**Why it matters**: Production LLMs need to be fast and cost-efficient

![Inference Optimization Overview](images/inference_optimization_overview.png)

![Inference Optimization Techniques](images/inference_optimization_techniques.png)

## KV Cache

`KV caching is done in **Masked Self attention layer**`

**Context**: `In decoder masked attention, transformers must attend to all previous words to generate new tokens. KV (Key-Value) pairs of all previously generated words are cached for efficiency.`

### What is KV Cache?

**KV Cache** stores the Key (K) and Value (V) matrices from previous tokens so they don't need to be recomputed when generating each new token.

**Without KV Cache**:
```
Generate "Hello world"
Token 1: "Hello" → Compute K, V
Token 2: "world" → Recompute K, V for "Hello" + compute for "world" ❌ (wasteful)
```

**With KV Cache**:
```
Token 1: "Hello" → Compute K, V → Save in cache
Token 2: "world" → Reuse cached K, V for "Hello" + compute only for "world" ✓ (efficient)
```

**K, V Generation**

> K and V are **not raw embeddings** — they are learned projections via `Wk`, `Wv` weight matrices. Each layer has its own `Wk`, `Wv`, so K and V differ at every layer.

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

**Per-layer projection** — K, V are different at every layer:
```
Layer 1:  K¹_cat, V¹_cat   ← syntax-level
Layer 2:  K²_cat, V²_cat   ← semantic-level
Layer N:  Kᴺ_cat, Vᴺ_cat
```

KV cache stores `num_layers × num_heads × seq_len` vectors.

| | What it holds | Cached? |
|---|---|---|
| **Embedding** | Raw word meaning — "what word is this?" | No |
| **K (Key)** | "What do I offer for others to match against?" | Yes |
| **V (Value)** | "What do I pass to whoever attends to me?" | Yes |
| **Q (Query)** | "What am I looking for?" — only for current token | No |

![KV Cache Mechanism](images/kv_cache_concept.png)


### How KV Cache Saves Computation

**Example**: Generating 1000 tokens

**Without cache**: Recompute all K,V for all previous tokens each time
- Token 2: Compute 2 tokens
- Token 3: Compute 3 tokens
- Token 1000: Compute 1000 tokens
- Total: ~500,000 computations ❌

**With cache**: Compute K,V only once per token
- Each token: Compute once, cache it
- Total: 1,000 computations ✓ (500× faster!)

`K & V alone saved, "Q" isn't saved since previous word/token Question(Q) won't be used for next token.`

![KV Cache Efficiency](images/kv_cache_computation.png)

### Memory Trade-off

**Speed**: Much faster generation (up to 100× speedup)  
**Memory**: Requires storing K,V for all past tokens  
**Result**: Standard practice in all modern LLMs

![KV Cache Memory Tradeoff](images/kv_cache_memory_tradeoff.png)

### KV Cache Growth

As you generate more tokens, the cache grows:
```
Token 1: Cache size = 1 token
Token 10: Cache size = 10 tokens
Token 1000: Cache size = 1000 tokens
```

This is why long context generation uses more memory!

![KV Cache Growth Over Time](images/kv_cache_growth.png)

---

## Traditional Continuous Memory Allocation

### How Traditional KV Cache Memory Works

In traditional LLM inference, memory for KV cache is allocated as **one continuous block** for the maximum possible sequence length (input + output).

### Memory Allocation Example

**Scenario**: User input is 2K tokens, model can generate up to 2K tokens

**Step 1: Pre-allocate Maximum Memory**
```
Model context window: 4K tokens (2K input + 2K output max)
Each token needs: 2 matrices (K, V) × model_dim
Example: 2 × 4096 dimensions × 2 bytes = ~16KB per token

Total memory allocated: 4K tokens × 16KB = 64MB
(Allocated upfront as ONE continuous block)
```

**Visualization**:
```
┌─────────────────────────────────────────────────┐
│   Continuous Memory Block (64MB allocated)      │
├─────────────────────────┬───────────────────────┤
│  Input (2K tokens)      │  Output (2K tokens)   │
│  [Used immediately]     │  [Reserved, empty]    │
│  ████████████████████   │  ░░░░░░░░░░░░░░░░░░░░ │
│  32MB used              │  32MB reserved        │
└─────────────────────────┴───────────────────────┘
```

### The Problem: Wasted Memory

**Reality**: Most responses are much shorter than max length

```
Request 1: Input 2K → Output 200 tokens (not 2K)
┌─────────────────────────────────────────────────┐
│  Input (2K) │ Output (200) │    Unused (1.8K)   │
│  ████████   │ ██           │ ░░░░░░░░░░░░░░░░░░ │
│  32MB       │ 3.2MB        │ 28.8MB WASTED ❌   │
└─────────────────────────────────────────────────┘

Request 2: Input 1K → Output 500 tokens
┌─────────────────────────────────────────────────┐
│  Input (1K) │ Output (500) │   Unused (2.5K)    │
│  ████        │ ████         │ ░░░░░░░░░░░░░░░░░ │
│  16MB       │ 8MB          │ 40MB WASTED ❌     │
└─────────────────────────────────────────────────┘
```

### Why Continuous Memory?

**Traditional approach requires continuous memory because**:
1. Attention mechanism needs to access all previous tokens' KV values
2. GPU operations are optimized for contiguous memory access
3. Simpler implementation - just one pointer to the start

### The Waste Problem in Detail

**Scenario**: Serving 10 concurrent requests with 4K context window

```
Traditional Allocation:
─────────────────────────────────────────────
Request 1: [████████████░░░░░░░░░░] 64MB allocated, 40MB used
Request 2: [██████░░░░░░░░░░░░░░░░] 64MB allocated, 20MB used
Request 3: [████████████████░░░░░░] 64MB allocated, 50MB used
Request 4: [████░░░░░░░░░░░░░░░░░░] 64MB allocated, 15MB used
Request 5: [██████████░░░░░░░░░░░░] 64MB allocated, 30MB used
Request 6: [████████░░░░░░░░░░░░░░] 64MB allocated, 25MB used
Request 7: [██████████████░░░░░░░░] 64MB allocated, 45MB used
Request 8: [██░░░░░░░░░░░░░░░░░░░░] 64MB allocated, 10MB used
Request 9: [████████████░░░░░░░░░░] 64MB allocated, 40MB used
Request 10:[██████░░░░░░░░░░░░░░░░] 64MB allocated, 20MB used

Total allocated: 640MB
Total used: 295MB
Wasted: 345MB (54% waste!) ❌
```

### Real Example: 2K Input, Variable Output

**User sends 2K token prompt, model generates answer**

**Case 1: Short answer (100 tokens)**
```
Allocated: 2K input + 2K output slots = 4K slots = 64MB
Used: 2K input + 100 output = 2.1K slots = 33.6MB
Wasted: 1.9K slots = 30.4MB (47% wasted)
```

**Case 2: Medium answer (500 tokens)**
```
Allocated: 4K slots = 64MB
Used: 2K + 500 = 2.5K slots = 40MB
Wasted: 1.5K slots = 24MB (37% wasted)
```

**Case 3: Long answer (1.5K tokens)**
```
Allocated: 4K slots = 64MB
Used: 2K + 1.5K = 3.5K slots = 56MB
Wasted: 0.5K slots = 8MB (12% wasted)
```

### Memory Cannot Be Reclaimed

**Key issue**: Once allocated as continuous block, unused portion cannot be used for other requests

```
GPU Memory (Total: 500MB available)
┌────────────────────────────────────────────────┐
│ Request A: [████████░░░░░░░░] 64MB (30MB used) │
│ Request B: [████████░░░░░░░░] 64MB (25MB used) │
│ Request C: [████████░░░░░░░░] 64MB (40MB used) │
│ Request D: [████████░░░░░░░░] 64MB (20MB used) │
│ Request E: [████████░░░░░░░░] 64MB (35MB used) │
│ Request F: [████████░░░░░░░░] 64MB (30MB used) │
│ Request G: [████████░░░░░░░░] 64MB (45MB used) │
│ Request H: WAITING... (need 64MB but only 52MB free)│
├────────────────────────────────────────────────┤
│ Allocated: 448MB (7 requests × 64MB each)      │
│ Actually used: 225MB                           │
│ Wasted in gaps: 223MB                          │
│ Free: 52MB (can't fit 8th request!)           │
└────────────────────────────────────────────────┘

With better memory management:
  225MB used + 52MB free = 277MB could serve 13+ requests!
```

### Summary: Why This Is Inefficient

❌ **Pre-allocate max**: Must reserve for worst case (max output length)  
❌ **Cannot share**: Unused slots locked, can't be used by other requests  
❌ **Fragmentation**: Even with free total memory, may lack continuous block  
❌ **Low utilization**: Typically only 40-60% of allocated memory is used  

**This is the problem PagedAttention solves** → explained next ⬇️

---

## Sharing Attention Heads - PagedAttention

### What is PagedAttention?

**PagedAttention** is a memory management technique that stores KV cache in small "pages" (like virtual memory in operating systems) instead of continuous blocks, reducing memory waste.

**The Problem**: Traditional KV cache allocates continuous memory blocks, leading to fragmentation and waste when sequences have different lengths.

**The Solution**: Split KV cache into fixed-size pages that can be stored anywhere in memory and dynamically allocated/deallocated.

**Benefits**:
- ✅ Less memory waste (up to 90% reduction in fragmentation)
- ✅ Higher throughput (can serve more requests)
- ✅ Better GPU utilization
- 🔧 Used in: vLLM (popular inference engine)

**Analogy**: Like how your computer manages RAM - split into pages instead of requiring one big continuous chunk.

![PagedAttention Concept](images/paged_attention_overview.png)

### How PagedAttention Works

Instead of allocating one big continuous memory block for the entire sequence, it divides the KV cache into small fixed-size pages (e.g., 16 tokens per page).

**Traditional approach**:
```
Sequence of 100 tokens → Need 100 continuous memory slots
If only 95 available together → Fails ❌
```

**PagedAttention**:
```
Sequence of 100 tokens → Need 7 pages (16 tokens each)
Pages can be scattered: Page1[slot 10-25], Page2[slot 50-65], etc.
Works even with fragmented memory ✓
```

`internal Fragmentation - Reserved by Model to complete the request. Reserved(<resv>) is subset of it that corresponds to be used.`

![PagedAttention Internal Fragmentation](images/paged_attention_fragmentation.png)


`vLLM comes with **Paged Attention**. More vLLM deep dive is present in vLLM.md`

![vLLM PagedAttention](images/paged_attention_vllm.png)

## KV Cache dimension reduction


`DeepSeek has tried to tackle dimension reduction in **Multi-latent attention**`

![MLA KV Compression Part 1](images/mla_kv_compression_1.png)

![MLA KV Compression Part 2](images/mla_kv_compression_2.png)

### Multi-latent Attention

`"DeepSeek-V2: A Strong, Economical, and Eﬃcient Mixture-of-Experts Language Model", by DeepSeek, 2023.`

#### What is Multi-latent Attention (MLA)?

**Multi-latent Attention** is a KV cache compression technique developed by DeepSeek that drastically reduces memory usage by compressing Key and Value matrices into lower-dimensional "latent" representations.

**Core Innovation**: Instead of storing full high-dimensional K and V matrices for every token, MLA stores compressed low-dimensional representations and reconstructs them when needed.

#### The Problem MLA Solves

**What KV Cache Actually Stores**

```
For each token:
  Layer 1:   K¹, V¹   (from W_K¹, W_V¹)
  Layer 2:   K², V²   (from W_K², W_V²)
  ...
  Layer 80:  K⁸⁰, V⁸⁰ (from W_K⁸⁰, W_V⁸⁰)
```
Total KV cache = `num_layers × num_heads × seq_len × head_dim × 2`



Traditional attention stores massive KV cache:
```
Standard Attention (e.g., LLaMA-2 70B):
- Hidden dimension: 8192
- Each token stores: K (8192 dims) + V (8192 dims) = 16,384 values
- For 1000 tokens: 16,384 × 1000 = 16.4M values
- Memory: ~65MB per 1000 tokens (fp16)
```

**KV for all layers**

```
Per token, per layer:   16,384 values = 32KB  (fp16)
× 80 layers:            2,560KB = 2.5MB per token
× 1000 tokens:          2,500MB = ~2.5GB
× 100K tokens (128K ctx): ~250GB  ← why this is a serious problem
```


This becomes a bottleneck for:
- Long context windows (128K+ tokens)
- Serving many concurrent requests
- Limited GPU memory

#### How MLA Works: KV Compression

**Step 1: Compress to Latent Space**

Instead of storing full K and V, compress them into low-dimensional latent vectors:

```
Standard Attention:
Token → K (8192 dims) + V (8192 dims)
Cache size per token: 16,384 values

Multi-latent Attention:
Token → Compressed Latent (512 dims)
Cache size per token: 512 values
Compression ratio: 32× smaller! 🎯
```

**Step 2: Store Only Compressed Latents**

```
Traditional KV Cache:
┌─────────────────────────────────────┐
│ Token 1: K[8192] + V[8192]          │
│ Token 2: K[8192] + V[8192]          │
│ ...                                 │
│ Token N: K[8192] + V[8192]          │
└─────────────────────────────────────┘
Memory: N × 16,384 values

MLA Cache:
┌─────────────────────────────────────┐
│ Token 1: Latent[512]                │
│ Token 2: Latent[512]                │
│ ...                                 │
│ Token N: Latent[512]                │
└─────────────────────────────────────┘
Memory: N × 512 values (32× smaller)
```

**Step 3: Decompress During Attention**

When computing attention, decompress latents back to K and V:

```
1. Retrieve compressed latent (512 dims)
2. Apply learned projection: Latent → K (8192) + V (8192)
3. Compute attention with reconstructed K, V
4. Discard reconstructed K, V (don't store them)
```

#### Architecture Details

**Compression Mechanism**:
```
Input embedding (d_model = 8192)
        ↓
Down-project to latent space (learned linear projection)
   c_latent = W_c × input  (8192 → 512)
   W_c: Trainable weight matrix [512 × 8192]
        ↓
[Store in cache] ← Only this stored!
        ↓
When needed, up-project back (learned linear projections):
   K = W_k × c_latent  (512 → 8192)
   V = W_v × c_latent  (512 → 8192)
   W_k, W_v: Trainable weight matrices [8192 × 512]
        ↓
Use K, V for attention computation
```

**Dimension Reduction Method**:

MLA uses **learned linear projections**, **low-rank Matrix Factorization** (matrix multiplication) to compress dimensions:

```
Down-projection (Compression):
─────────────────────────────
Input: [8192 dimensions]
  × W_c [512 × 8192 matrix]
= Latent: [512 dimensions]

Think of it as: Project high-dim vector onto lower-dim subspace
The 512 dimensions capture the "essence" of the original 8192

Up-projection (Reconstruction):
─────────────────────────────
Latent: [512 dimensions]
  × W_k [8192 × 512 matrix]
= K: [8192 dimensions]

Latent: [512 dimensions]
  × W_v [8192 × 512 matrix]
= V: [8192 dimensions]
```

**Key Point**: W_c, W_k, and W_v are **learned during training**, not hand-crafted. The model learns how to:
1. Compress information optimally (W_c)
2. Reconstruct K and V accurately (W_k, W_v)

**Does Compression Cause Loss?**

**Yes, but minimal** (1-2% accuracy drop):

✅ **Why loss is small**:
- The compression matrices (W_c, W_k, W_v) are **trained jointly with the entire model**
- Model learns to compress in a way that preserves the most important information for attention
- 512 dimensions are enough to capture the essential patterns
- Training optimizes the trade-off between compression and accuracy

❌ **Why there's some loss**:
- **Information bottleneck**: 8192 dims → 512 dims = some information discarded
- Can't perfectly reconstruct original K, V from compressed latent
- Reconstructed K, V are approximations

**Loss Analogy**:
```
Original photo: 4K resolution (8192 × 8192 pixels)
Compressed JPEG: 512 × 512 pixels
You lose some detail, but the essential content remains recognizable

Same with MLA: You lose some subtle information, but attention patterns 
are preserved well enough for ~98-99% of original performance
```

**Typical Impact**:
- Benchmark scores: -1% to -2% vs full KV cache
- Perplexity: Slightly higher (+0.1 to +0.3)
- Real-world usage: Often imperceptible

**Trade-off Decision**:
```
Full KV Cache:     100% accuracy, 6.5GB memory
Multi-latent (MLA): 98% accuracy, 200MB memory (32× less)

For most applications: 2% accuracy loss << 32× memory savings!
```

#### Memory Savings Example

**Scenario**: 100K token context window, 8192 hidden dimension

**Standard Attention**:
- KV cache: 100K × 16,384 = 1.64B values
- Memory: ~6.5GB (fp16)

**Multi-latent Attention** (512 latent dim):
- Latent cache: 100K × 512 = 51.2M values
- Memory: ~200MB (fp16)
- **Saving: 6.3GB (97% reduction!)** 🚀

#### Key Benefits

✅ **Massive memory reduction**: 32× smaller KV cache (typical)  
✅ **Longer contexts**: Can handle 128K+ tokens with same memory  
✅ **Higher throughput**: Serve more concurrent requests  
✅ **Minimal quality loss**: Learned compression preserves information  
✅ **Faster attention**: Less data to move from memory  

#### Trade-offs

**Pros**:
- Dramatic memory savings (20-40× compression)
- Enables very long context windows
- Better GPU utilization

**Cons**:
- Slight computational overhead (compression/decompression)
- Need to train with MLA from scratch (can't retrofit)
- Small accuracy trade-off (~1-2% in some tasks)

#### Real-World Impact: DeepSeek-V2

DeepSeek-V2 uses MLA to achieve:
```
Model: 236B parameters (MoE)
Context: 128K tokens
KV cache per request: ~200MB (with MLA)
Without MLA: ~6GB per request

Result: Can serve 30× more requests on same hardware!
```

#### Comparison with Standard Attention

| Aspect | Standard Attention | Multi-latent Attention |
|--------|-------------------|------------------------|
| **KV Cache/Token** | 16,384 values | 512 values |
| **Memory (100K ctx)** | ~6.5GB | ~200MB |
| **Compression** | 1× | 32× |
| **Long Context** | Limited by memory | 128K+ feasible |
| **Throughput** | Lower | 20-30× higher |
| **Accuracy** | Baseline | ~98-99% of baseline |

#### When to Use MLA

**Use Multi-latent Attention when**:
- Building models for very long contexts (64K+ tokens)
- Memory efficiency is critical
- Serving high-throughput inference
- Working with MoE architectures

**Stick with standard attention when**:
- Short contexts (<4K tokens)
- Maximum accuracy is paramount
- Already have trained models (can't retrofit MLA)

#### Key Takeaway

**Multi-latent Attention** = Compress KV cache into low-dimensional latents (32× smaller) → Reconstruct when needed → Store only compressed version

**Result**: Same model capacity, 97% less KV cache memory, enabling longer contexts and higher throughput!

![Multi-latent Attention Architecture](images/multi_latent_attention_diagram.png)

## Speeding up decoding with speculative Sampling

Research Paper - `Better & Faster Large Language Models via Multi-token Prediction`

### What is Speculative Sampling?

**Speculative Sampling** speeds up text generation by using a small "draft" model to guess multiple tokens ahead, which are then verified by the large model in one pass.

**Simple Idea**: Guess several tokens, check them all at once, accept the correct ones.

**What are these models?**
- **Small Model (Draft)**: A separate, smaller LLM with fewer parameters (1B-7B params)
  - Example: LLaMA-7B used as draft for LLaMA-70B
  - It's a real LLM, just smaller and faster
  - Makes quick guesses (might be wrong sometimes)
  
- **Large Model (Target)**: The main LLM you want to use (70B+ params)
  - Example: LLaMA-70B, GPT-4
  - Slower but more accurate
  - Makes final decisions (verifies draft guesses)

Both are complete LLMs - the difference is size (parameter count) and speed.

### The Problem: One Token at a Time is Slow

```
Normal Generation (500ms total):
Step 1: "Hello"     (100ms) → wait
Step 2: "world"     (100ms) → wait
Step 3: "!"         (100ms) → wait
Step 4: "How"       (100ms) → wait
Step 5: "are"       (100ms) → wait

Problem: Must generate one by one ❌
```

### How It Works

```
Step 1: Small Model Guesses (40ms)
────────────────────────────────
Draft model: "Hello" → ["world", "!", "How", "are"]
(Fast but might be wrong)

Step 2: Big Model Checks All (100ms)
────────────────────────────────
Target model verifies: ✓ ✓ ✓ ✗
Accepts: ["world", "!", "How"]
Rejects: "are"

Result: 140ms for 3 tokens (vs 300ms normal)
Speedup: 2× faster!
```

### Simple Visual

```
Two LLMs Work Together:
┌────────────────────────────────────┐
│ Draft Model (Smaller LLM - 7B)     │ → Guesses 4 tokens fast
│ Example: LLaMA-7B, Mistral-7B      │    (10ms per token)
└────────────────────────────────────┘
            ↓ (passes guesses to)
┌────────────────────────────────────┐
│ Target Model (Larger LLM - 70B)    │ → Checks all at once
│ Example: LLaMA-70B, GPT-4          │    (100ms for all 4)
└────────────────────────────────────┘
            ↓
      Keep correct ones
```

**Both are complete LLMs**, just different sizes:
- 7B model: 7 billion parameters (small, fast, less accurate)
- 70B model: 70 billion parameters (large, slow, more accurate)

### Why It's Faster

**Key Insight**: The big model can check multiple tokens in ONE pass (parallel) instead of generating them one by one (sequential).

```
Normal: [Token1]--[Token2]--[Token3]  (3 passes = slow)
Speculative: [Token1,2,3 checked together] (1 pass = fast!)
```

### Benefits

✅ **2-3× faster** generation  
✅ **No quality loss** (big model still decides)  
✅ **Best for predictable text** (code, structured data)  

### When It Works Well

**Good** (high acceptance):
- Code completion: 92% acceptance → 2.8× faster
- Q&A answers: 78% acceptance → 2.3× faster

**Not as good** (low acceptance):
- Creative writing: 45% acceptance → 1.4× faster
- Unpredictable text: Many possible answers

### Example

```
Input: "def calculate_sum("

Draft guesses: "a, b):\n    return a + b"
Target checks:  ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
All accepted! → Very fast

Input: "Write a creative story about"

Draft guesses: "a magical forest with"
Target checks:  ✓ ✗ (prefers different words)
Only 1 accepted → Not much speedup
```

### Key Takeaway

**Speculative Sampling** = Small model guesses → Big model verifies all at once → 2-3× faster

**How it helps**: Like having a fast assistant make educated guesses, then an expert quickly checks them all together instead of thinking through each one separately.

**Best use**: Code, structured output, predictable text where guesses are often correct!

## Multi-Token Prediction

![Multi-Token Prediction Architecture](images/multi_token_prediction.png)


### What is Multi-Token Prediction?

**Multi-Token Prediction** is a training technique where the model learns to predict multiple future tokens at once, instead of just the next single token.

**Simple Idea**: Train the model to look further ahead during training, so it becomes faster and smarter at generation.

### Traditional vs Multi-Token Prediction

**Traditional (Next Token Prediction)**:
```
Input: "The cat sat on the"
Model predicts: "mat" (only 1 token ahead)
```

**Multi-Token Prediction**:
```
Input: "The cat sat on the"
Model predicts: ["mat", ",", "sleeping"] (3 tokens ahead)
```

During training, the model learns to predict not just the immediate next token, but also 2-3 tokens further ahead.

### How It Works

```
Training Phase:
──────────────
Traditional:
Input: "Hello"
Target: "world" (predict 1 token)

Multi-Token:
Input: "Hello"
Targets: ["world", "!", "How"] (predict 3 tokens)
           ↓       ↓     ↓
        Token+1  Token+2  Token+3

Model has 3 prediction heads (one for each position)
```

### Visual Comparison

```
Traditional Training:
┌─────────────────────────────────┐
│ Input: "The cat"                │
│         ↓                       │
│ Predict: "sat" (next token)     │
└─────────────────────────────────┘

Multi-Token Training:
┌─────────────────────────────────┐
│ Input: "The cat"                │
│         ↓                       │
│ Predict: "sat" (token +1)       │
│         "on"  (token +2)       │
│         "the" (token +3)       │
└─────────────────────────────────┘
Model learns to plan ahead!
```

### Benefits

✅ **Better planning**: Model learns longer-term dependencies  
✅ **Faster inference**: Can be combined with speculative sampling  
✅ **Improved quality**: Better understanding of context  
✅ **More coherent text**: Plans ahead instead of myopic next-token choices  

### Why It Helps

**Problem with traditional training**:
- Model only sees 1 token ahead
- Makes locally optimal but globally suboptimal choices
- "Shortsighted" predictions

```
Example:
Input: "I went to the"
Traditional: Predicts "store" (common but generic)
Multi-Token: Predicts "library" (better fits longer context)
             because it already planned "to borrow books"
```

**Multi-token advantage**:
- Model considers future consequences
- Makes better choices that fit longer sequences
- Like playing chess - thinking several moves ahead

### Real-World Impact

```
Traditional Model:
Input: "Write a function to"
Output: "calculate the sum of two numbers"
(Generic, common pattern)

Multi-Token Trained Model:
Input: "Write a function to"
Output: "efficiently process large datasets using parallel computation"
(More sophisticated, better planning)
```

### How It Speeds Up Inference

Multi-token prediction helps with **speculative sampling**:

```
1. Model already trained to predict multiple tokens
2. Uses multiple prediction heads during inference
3. Generates several tokens at once
4. Verifies them together

Result: 2-3× faster generation with better quality!
```

### Key Difference from Speculative Sampling

| Aspect | Speculative Sampling | Multi-Token Prediction |
|--------|---------------------|------------------------|
| **What** | Inference technique | Training technique |
| **When** | During generation | During training |
| **How** | Two models (draft + target) | One model, multiple heads |
| **Goal** | Speed up inference | Better planning + speed |

**They work together**: Multi-token trained models work great with speculative sampling!

### Simple Analogy

**Traditional Training**: Like learning to drive by only looking 5 feet ahead
- You can drive, but make jerky, reactive decisions
- No long-term planning

**Multi-Token Training**: Like learning to drive by looking 50 feet ahead
- Smoother driving, anticipate turns
- Better decisions because you see what's coming

### Example: Writing Code

```
Traditional (myopic):
"def calculate" → "(" → "x" → ")" → ":" → "\n" → "return"
Each token decided independently

Multi-Token (planning):
"def calculate" → model already knows it will be:
"(x, y):\n    return x + y"
Better structure from the start!
```

### When Multi-Token Helps Most

**Best for**:
- Long-form text generation (essays, stories)
- Code generation (better structure)
- Tasks requiring planning (reasoning, problem-solving)

**Less impact**:
- Very short responses (1-2 tokens)
- Random/creative tasks with no planning needed

### Key Takeaway

**Multi-Token Prediction** = Train model to predict 2-4 tokens ahead (not just 1) → Model learns to plan ahead → Better quality + faster inference

**How it helps**: Model becomes a better planner (not just reactive), makes smarter choices that fit longer sequences, and naturally supports faster inference techniques.

**Bottom line**: Thinking ahead during training = smarter, faster model!

---

# Sampling Strategies: Top-P, Top-K, and Temperature

## Overview

When generating text with LLMs, the model doesn't always pick the single most likely next token. Instead, it produces a **probability distribution** over all possible tokens. How we sample from this distribution dramatically affects output quality.

### The Problem We're Solving

```
Model outputs probabilities for next token:
"the"     → 45%
"a"       → 30%
"an"      → 15%
"some"    → 5%
"another" → 3%
"xyz"     → 2%

Question: Which token should we pick?
- Always pick "the" (45%)? → Boring, repetitive
- Include all tokens? → Gibberish ("xyz"?)
- Smart selection? → That's what Top-K and Top-P do!
```

## Table of Contents
- [Temperature (N)](#temperature-n)
- [Top-K Sampling](#top-k-sampling)
- [Top-P (Nucleus) Sampling](#top-p-nucleus-sampling)
- [Comparison & Trade-offs](#comparison--trade-offs)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)

---

## Temperature (N)

### What is Temperature?

Temperature controls how "confident" the model's decisions are. It's a scaling factor applied to the logits **before** softmax.

```
higher temperature → more randomness
lower temperature → more deterministic
```

### Formula

```
probability = softmax(logits / temperature)

temperature = 0.5  → "confident" (logits amplified)
temperature = 1.0  → "normal" (no change)
temperature = 2.0  → "uncertain" (logits reduced)
```

### Visual Explanation

```
Temperature = 0.3 (Very Deterministic)
┌─────────────────────────────────┐
│ "the"  ▓▓▓▓▓▓▓▓▓▓▓▓ 92%         │
│ "a"    ▓▓▓ 6%                   │
│ "an"   ▓▓ 2%                    │
│ others  0%                      │
└─────────────────────────────────┘
Result: Picks "the" most of the time

Temperature = 1.0 (Normal)
┌─────────────────────────────────┐
│ "the"  ▓▓▓▓▓ 45%                │
│ "a"    ▓▓▓ 30%                  │
│ "an"   ▓▓ 15%                   │
│ "some" ▓ 5%                     │
│ others ▓ 5%                     │
└─────────────────────────────────┘
Result: Diverse, balanced choices

Temperature = 2.0 (Very Random)
┌─────────────────────────────────┐
│ "the"  ▓▓▓ 20%                  │
│ "a"    ▓▓▓ 18%                  │
│ "an"   ▓▓▓ 17%                  │
│ "some" ▓▓▓ 16%                  │
│ "xyz"  ▓▓▓ 15%                  │
│ others ▓▓▓ 14%                  │
└─────────────────────────────────┘
Result: Often picks wrong tokens
```

### Use Cases

| Temperature | Use Case | Example |
|-------------|----------|---------|
| **0.0** | Deterministic, always same output | Code generation, factual Q&A, translation |
| **0.3 - 0.7** | Slightly creative, mostly consistent | Customer support, summaries, email drafts |
| **0.7 - 1.5** | Creative, varied | Creative writing, brainstorming, chat |
| **2.0+** | Very random, often gibberish | Usually not recommended |

### Key Insight

Temperature doesn't filter out bad tokens—it just changes **how often** each token is chosen. The "xyz" token at 2% still has a 15% chance with high temperature!

**This is where Top-K and Top-P come in.**

---

## Top-K Sampling

### What is Top-K?

**Top-K** means: "Only consider the K most likely tokens, ignore the rest."

It's a **hard cutoff** approach.

### How It Works

```
All tokens sorted by probability:
1. "the"     → 45%  ✓ (included)
2. "a"       → 30%  ✓ (included)
3. "an"      → 15%  ✓ (included)
4. "some"    → 5%   ✗ (excluded)
5. "another" → 3%   ✗ (excluded)
6. "xyz"     → 2%   ✗ (excluded)

With K=3: Only sample from ["the", "a", "an"]
Renormalize probabilities:
- "the"     → 45 / (45+30+15) = 60%
- "a"       → 30 / 90 = 33%
- "an"      → 15 / 90 = 17%
```

### Visual Example

```
Original distribution (K=all):
"the" 45% | "a" 30% | "an" 15% | "some" 5% | "another" 3% | "xyz" 2%
                                    ▲
                        Poor quality tokens included

After Top-K=3:
"the" 60% | "a" 33% | "an" 17% | ✗ ✗ ✗
                              ▲
                All probability mass on good tokens
```

### Advantages

✅ Simple and fast (just sort)
✅ Removes obviously bad tokens
✅ Predictable behavior

### Disadvantages

❌ **Fixed cutoff** - sometimes K=5 is perfect, sometimes it's too many/too few
❌ Doesn't account for **probability gaps**
   - With K=5, includes a token at 5% and one at 1% (huge difference!)
   - No distinction between:
     - "the" 45%, "a" 30%, "an" 15%, "some" 5%, "another" 1%
     - "the" 25%, "a" 24%, "an" 23%, "some" 22%, "another" 1%

### When to Use

- Simple generation tasks
- When you have a good K value for your domain
- When speed is critical (Top-K is O(n log k))

---

## Top-P (Nucleus) Sampling

### What is Top-P?

**Top-P** (also called **Nucleus Sampling**) means: "Keep sampling tokens until cumulative probability reaches P."

It's an **adaptive cutoff** approach—the number of tokens included depends on the probability distribution.

### How It Works

```
All tokens sorted by probability:
"the"     → 45%  [cumulative: 45%]    ✓
"a"       → 30%  [cumulative: 75%]    ✓
"an"      → 15%  [cumulative: 90%]    ✓
"some"    → 5%   [cumulative: 95%]    ✗
"another" → 3%   [cumulative: 98%]    ✗
"xyz"     → 2%   [cumulative: 100%]   ✗

With P=0.9: Include tokens until 90% cumulative
Selected: ["the", "a", "an"]
(automatically excluded "some", "another", "xyz")
```

### Adaptive Behavior

**Distribution 1** (concentrated):
```
"the" 70% | "a" 20% | "an" 8% | "some" 2%
                    ▲
         P=0.9 stops here → 2 tokens selected
```

**Distribution 2** (spread out):
```
"the" 25% | "a" 24% | "an" 23% | "some" 22% | "another" 6%
                                              ▲
                       P=0.9 stops here → 5 tokens selected
```

**Same P value, different number of tokens!** This is the key insight.

### Visual Example

```
Original probability space:
┌──────────────────────────────────────────────┐
│ "the"     ████████████████████░ 45%          │
│ "a"       ████████████░░░░░░░░░ 30%          │
│ "an"      ██████░░░░░░░░░░░░░░░ 15%          │
│ "some"    ██░░░░░░░░░░░░░░░░░░░ 5%           │
│ "another" ░░░░░░░░░░░░░░░░░░░░░ 3%           │
│ "xyz"     ░░░░░░░░░░░░░░░░░░░░░ 2%           │
└──────────────────────────────────────────────┘

Cumulative probability:
45% + 30% + 15% = 90% ≥ P (0.9)
So include only [45%, 30%, 15%]

Final distribution (renormalized):
┌──────────────────────────────────────┐
│ "the"  ██████████████░░░░░░░░ 60%     │
│ "a"    ███████░░░░░░░░░░░░░░░ 40%     │
│ "an"   █████░░░░░░░░░░░░░░░░░ 25%     │
│ others ░░░░░░░░░░░░░░░░░░░░░░░ 0%     │
└──────────────────────────────────────┘
```

### Advantages

✅ **Adaptive** - automatically adjusts to the distribution
✅ **Removes poor tokens** - no matter how many good ones there are
✅ **Balances diversity and quality** - sweet spot between top-k and full sampling
✅ **Works with any temperature** - complementary techniques

### Disadvantages

❌ Slightly more complex than Top-K
❌ Harder to predict behavior (depends on distribution)

### When to Use

- **Recommended for most applications**
- Creative writing (diversity)
- Conversational AI (quality + variety)
- Open-ended generation
- Most modern LLM APIs default to Top-P!

---

## Comparison & Trade-offs

### Head-to-Head Comparison

| Aspect | Temperature | Top-K | Top-P |
|--------|-------------|-------|-------|
| **What it does** | Scales probabilities | Hard cutoff (K tokens) | Soft cutoff (P cumulative) |
| **Randomness** | 0 = deterministic, high = random | Higher K = more random | Higher P = more diverse |
| **Filters bad tokens** | ❌ No | ✅ Yes | ✅ Yes |
| **Adaptive** | ❌ Fixed | ❌ Fixed K | ✅ Adapts to distribution |
| **Best for** | Creativity control | Specific domain + K | Most applications |
| **Computation** | Very fast | O(n log K) | O(n) |

### Common Parameter Combinations

```
1. Deterministic (Factual, Reproducible)
   temperature = 0.0
   top_p = 0.9
   top_k = 40
   Result: Always produces same output

2. Balanced (General purpose)
   temperature = 0.7
   top_p = 0.9
   top_k = 50
   Result: Good diversity + quality

3. Creative (Writing, Brainstorming)
   temperature = 0.9
   top_p = 0.95
   top_k = 50
   Result: More varied, creative outputs

4. Very Creative (Experimental)
   temperature = 1.2
   top_p = 0.99
   top_k = 100
   Result: High diversity, risk of gibberish

5. Code Generation (Precise)
   temperature = 0.2
   top_p = 0.95
   top_k = 40
   Result: Mostly correct syntax, slight variety
```

### When Multiple Techniques Combine

```
Step 1: Start with model's probability distribution
Step 2: Apply Temperature scaling
Step 3: Apply Top-K filtering (if enabled)
Step 4: Apply Top-P filtering (if enabled)
Step 5: Sample from final distribution
```

Example:
```
Original: ["the" 45%, "a" 30%, "an" 15%, "some" 5%, "xyz" 2%]
↓ After temperature = 0.5 (more deterministic)
["the" 62%, "a" 22%, "an" 12%, "some" 3%, "xyz" 1%]
↓ After top_k = 3
["the" 72%, "a" 26%, "an" 12%] (renormalized)
↓ After top_p = 0.9
["the" 72%, "a" 26%] (cumulative = 98% > 90%)
↓ Sample one token
Result: Pick "the" or "a" with probabilities 72% and 28%
```

---

## Real-World Examples

### Example 1: Customer Support Bot

**Goal**: Fast, accurate, professional responses

```
temperature = 0.3     # Low randomness
top_p = 0.9          # Standard filtering
top_k = 50           # Good quality tokens only

Query: "How do I reset my password?"
Output: "To reset your password, go to Settings > Security..."
        (Same response every time - good!)
```

### Example 2: Creative Writing Assistant

**Goal**: Varied, interesting, unpredictable

```
temperature = 1.0     # Balanced creativity
top_p = 0.95         # More diverse tokens
top_k = 100          # Allow more variation

Prompt: "Write an opening line for a mystery novel"
Output 1: "The old house stood silent on the hill..."
Output 2: "She found the key exactly where he'd hidden it..."
Output 3: "Rain hammered against the windows as lightning..."
         (Different every time - good!)
```

### Example 3: Code Generation

**Goal**: Correct syntax with minor variations

```
temperature = 0.2     # Mostly deterministic
top_p = 0.95         # High quality tokens
top_k = 40           # Very selective

Prompt: "Write a Python function to reverse a string"
Output: "def reverse_string(s):\n    return s[::-1]"
        (Usually correct, minor formatting variations)
```

### Example 4: Brainstorming Tool

**Goal**: Lots of different ideas, even weird ones

```
temperature = 1.2     # Very creative
top_p = 0.99         # Very permissive
top_k = 200          # Allow unexpected tokens

Prompt: "What are unusual uses for a paperclip?"
Output 1: "Emergency bookmark..."
Output 2: "Lockpick for tiny boxes..."
Output 3: "Antenna for a crystal radio..."
Output 4: "Constraint for bundled cables..."
         (Wide variety - good!)
```

---

## Best Practices

### 1. **Start with Top-P, Adjust Temperature**

```
✅ GOOD approach:
- Set top_p = 0.9 (standard filtering)
- Adjust temperature for creativity level

❌ AVOID:
- Using only temperature (doesn't filter bad tokens)
- Setting both top_k and top_p (conflicting)
```

### 2. **Match Parameters to Task**

| Task | Temperature | Top-P | Top-K |
|------|-------------|-------|-------|
| Factual QA | 0.0 - 0.3 | 0.9 | 40-50 |
| Summarization | 0.3 - 0.5 | 0.9 | 50 |
| General Chat | 0.6 - 0.8 | 0.9 | 50 |
| Creative Writing | 0.8 - 1.2 | 0.95 | 100+ |
| Brainstorming | 1.0 - 1.5 | 0.95 | 150+ |

### 3. **Monitor Output Quality**

```python
# Check for these patterns:
- Too deterministic? Lower temperature
- Too random/gibberish? Raise temperature or lower top_p
- All outputs identical? Lower temperature
- Outputs too similar? Raise temperature or top_p
```

### 4. **Consider Model Capability**

- **Smaller models**: More conservative settings (lower temp, lower top_p)
- **Larger models**: Can handle more creativity (higher temp, higher top_p)
- **Fine-tuned models**: May need custom tuning

### 5. **Test in Your Use Case**

```
Don't blindly copy parameters from examples!

Your domain might need:
- Higher temperature (if outputs are too repetitive)
- Lower top_p (if you see gibberish)
- Custom top_k (if you know your domain well)

Test with real prompts and users.
```

---

## Key Takeaways

### Temperature (N)
- **Controls confidence level** of probability distribution
- 0 = always pick highest probability
- Higher = flatter distribution, more randomness
- **Alone, not enough** to prevent bad tokens

### Top-K
- **Fixed cutoff**: "Take top K tokens only"
- Simple but inflexible
- K=40-50 is common
- **Doesn't adapt** to distribution shape

### Top-P (Nucleus)
- **Adaptive cutoff**: "Keep sampling until P% cumulative probability"
- Smartly handles both concentrated and spread distributions
- **Recommended** for most applications
- P=0.9 is the standard default

### The Golden Rule

```
Bad tokens? → Use Top-K or Top-P
Too deterministic? → Raise Temperature
Too random? → Lower Temperature or Top-P
Outputs too similar? → Raise Temperature
Gibberish appearing? → Lower Top-P or raise Temperature
```

### Recommended Defaults

```
For most applications:
temperature = 0.7
top_p = 0.9
(top_k can be disabled or set to 50+)
```

This balances **quality** (top-p filters), **diversity** (temperature), and **consistency** (not too random).

