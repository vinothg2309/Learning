chapter_6_llm_reasoning.md
# Resources

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |

---

<!-- TOC -->
## Table of Contents

- [Resources](#resources)
  - [Table of Contents](#table-of-contents)
- [LLM Reasoning](#llm-reasoning)
- [Overview](#overview)
- [Reasoning Model](#reasoning-model)
  - [Terminology](#terminology)
    - [Chain of thoughts(CoT)](#chain-of-thoughtscot)
- [Benchmark](#benchmark)
  - [Code](#code)
  - [Math](#math)
- [Metrics](#metrics)
    - [Pass@k Formula](#passk-formula)
    - [Understanding n vs k (Important!)](#understanding-n-vs-k-important)
    - [Step-by-Step Computation Example](#step-by-step-computation-example)
    - [Quick Calculation for Different k values](#quick-calculation-for-different-k-values)
  - [Temperature vs Pass@K](#temperature-vs-passk)
- [Scaling with RL](#scaling-with-rl)
- [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
  - [What is GRPO?](#what-is-grpo)
  - [Example Scenario: Calculate Advantages](#example-scenario-calculate-advantages)
    - [Formula:](#formula)
    - [Problem: "What is 25 × 4?"](#problem-what-is-25--4)
  - [Reference Model and Reward Model in GRPO](#reference-model-and-reward-model-in-grpo)
    - [Reference Model](#reference-model)
    - [Reward Model](#reward-model)
    - [How They Work Together](#how-they-work-together)
    - [When is GRPO Applied?](#when-is-grpo-applied)
    - [How Are Weights Adjusted Based on Policy?](#how-are-weights-adjusted-based-on-policy)
  - [GRPO vs PPO](#grpo-vs-ppo)
    - [Quick Comparison Table](#quick-comparison-table)
    - [PPO (Proximal Policy Optimization)](#ppo-proximal-policy-optimization)
    - [Formula](#formula-1)
    - [Which One Should You Use?](#which-one-should-you-use)
    - [The Bottom Line](#the-bottom-line)
- [Length Bias](#length-bias)
  - [What is Length Bias?](#what-is-length-bias)
  - [Image 24: Demonstrating Length Bias Problem](#image-24-demonstrating-length-bias-problem)
  - [Image 25: Length Bias Impact on Training](#image-25-length-bias-impact-on-training)
  - [Image 26: Solutions to Length Bias](#image-26-solutions-to-length-bias)
  - [Image 27: Practical Example](#image-27-practical-example)
  - [Bottom Line](#bottom-line)
- [DAPO (Direct Alignment from Preference Optimization)](#dapo-direct-alignment-from-preference-optimization)
  - [What is DAPO?](#what-is-dapo)
  - [GRPO vs DAPO: The Key Difference](#grpo-vs-dapo-the-key-difference)
    - [GRPO Approach:](#grpo-approach)
    - [DAPO Approach:](#dapo-approach)
  - [Example: Why DAPO Improves Over GRPO](#example-why-dapo-improves-over-grpo)
    - [GRPO Problem:](#grpo-problem)
    - [DAPO Solution:](#dapo-solution)
  - [Image 29: DAPO Architecture](#image-29-dapo-architecture)
- [Dr.GRPO (Dropout-Regularized GRPO)](#drgrpo-dropout-regularized-grpo)
  - [What is Dr.GRPO?](#what-is-drgrpo)
  - [How Dr.GRPO Works](#how-drgrpo-works)
  - [Example: GRPO vs Dr.GRPO](#example-grpo-vs-drgrpo)
    - [Regular GRPO (Overfitting):](#regular-grpo-overfitting)
    - [Dr.GRPO (Generalizes):](#drgrpo-generalizes)
  - [Visual Comparison: The Three Approaches](#visual-comparison-the-three-approaches)
    - [GRPO (Original):](#grpo-original)
    - [DAPO (Direct Alignment):](#dapo-direct-alignment)
    - [Dr.GRPO (Dropout Regularized):](#drgrpo-dropout-regularized)
  - [Real-World Example: All Three Methods](#real-world-example-all-three-methods)
    - [GRPO Training:](#grpo-training)
    - [DAPO Training:](#dapo-training)
    - [Dr.GRPO Training:](#drgrpo-training)
  - [Which Should You Use?](#which-should-you-use)
  - [Bottom Line](#bottom-line-1)
- [Applications](#applications)
  - [Deepseek R0](#deepseek-r0)
    - [Image 30: Deepseek R0 Architecture](#image-30-deepseek-r0-architecture)
    - [Image 31: R0 Training Results](#image-31-r0-training-results)
    - [Image 32: R0 Benchmark Performance](#image-32-r0-benchmark-performance)
    - [Image 33: R0 Example Outputs](#image-33-r0-example-outputs)
  - [Deepseek R1](#deepseek-r1)
    - [Image 34: R1 Architecture \& Improvements](#image-34-r1-architecture--improvements)
    - [Image 35: R1 Training Efficiency](#image-35-r1-training-efficiency)
    - [Image 36: R1 Benchmark Comparison](#image-36-r1-benchmark-comparison)
    - [Image 37: R1 Reasoning Examples](#image-37-r1-reasoning-examples)
    - [Image 38: R1 vs R0 Performance Summary](#image-38-r1-vs-r0-performance-summary)
  - [Bottom Line: R0 → R1 Evolution](#bottom-line-r0--r1-evolution)
- [Distillation](#distillation)
  - [What is Distillation?](#what-is-distillation)
  - [How Distillation Works (Simple 3 Steps)](#how-distillation-works-simple-3-steps)
  - [Why Use Distillation?](#why-use-distillation)
  - [Real-World Example](#real-world-example)
  - [Analogy](#analogy)
  - [How Training Actually Works in Distillation](#how-training-actually-works-in-distillation)
    - [Important Question: How Do We Get Probabilities from LLMs?](#important-question-how-do-we-get-probabilities-from-llms)
    - [What Happens Inside Every LLM:](#what-happens-inside-every-llm)
    - [Step-by-Step Training Process:](#step-by-step-training-process)
    - [How Distillation Gets Access to Probabilities:](#how-distillation-gets-access-to-probabilities)
    - [Real Training Code Example:](#real-training-code-example)
    - [Training Progress Example:](#training-progress-example)
    - [Why Learn from Probabilities Instead of Just Answers?](#why-learn-from-probabilities-instead-of-just-answers)
    - [Key Parameters:](#key-parameters)
    - [Bottom Line on Training:](#bottom-line-on-training)
  - [Image 40: Distillation Process](#image-40-distillation-process)
  - [Bottom Line](#bottom-line-2)
<!-- /TOC -->

---

# LLM Reasoning

# Overview

This section introduces the fundamental concepts of reasoning in Large Language Models (LLMs). We'll explore what "vanilla" LLMs can and cannot do, setting the stage for understanding why advanced reasoning techniques are necessary.

**The trade-offs of basic LLMs:** On the strength side, they excel at creative tasks like generating ideas and writing/debugging code. However, they have critical weaknesses: they struggle with complex reasoning, their knowledge doesn't update (it's frozen from training), they can't take actions in the real world, and their performance is difficult to measure objectively.

![alt text](images/01_overview_llm_tradeoffs.png)

# Reasoning Model

This section dives into specialized models designed to improve LLM reasoning capabilities. We'll learn about different techniques and terminology used to make LLMs "think" more carefully and logically through problems.

## Terminology

 Here we define the key terms and concepts you'll encounter when working with reasoning models. Understanding this vocabulary is essential for grasping how these advanced techniques work.

**Fundamental terminology in LLM reasoning research:** Clarifies the different components and approaches that researchers use to describe how models process information and arrive at answers.

![alt text](images/02_terminology_fundamentals.png)

### Chain of thoughts(CoT)

 Chain of Thought (CoT) is a breakthrough technique that makes LLMs "show their work" by thinking step-by-step. Instead of jumping directly to an answer, the model explains its reasoning process, similar to how a teacher solves a math problem on a whiteboard. This dramatically improves accuracy on complex tasks.

**How Chain of Thought prompting works in practice:** Shows the difference between asking an LLM for a direct answer versus prompting it to break down the problem into logical steps. Step-by-step reasoning leads to more accurate and reliable outputs.

![alt text](images/03_cot_how_it_works.png)

**Practical example of Chain of Thought in action:** A specific problem (such as a math word problem or logical puzzle) broken down into intermediate reasoning steps. Each step builds on the previous one, making the final answer more transparent and verifiable.

![alt text](images/04_cot_practical_example.png)

**Variations and extensions of the Chain of Thought technique:** Different prompting strategies, such as "few-shot CoT" (giving the model examples of step-by-step reasoning) or "zero-shot CoT" (simply asking the model to "think step by step"). These variations help us understand when and how to best apply CoT.

![alt text](images/05_cot_variations_extensions.png)

**Advanced concepts and results in Chain of Thought reasoning:** Performance comparisons (how much CoT improves accuracy), different types of reasoning tasks where CoT excels, and architectural improvements that enhance reasoning capabilities. Shows the real-world impact and effectiveness of reasoning techniques.

**More CoT techniques and improvements:** Additional methods that build on basic Chain of Thought to make reasoning even better.

![alt text](images/06_cot_advanced_techniques.png)

**Results showing CoT effectiveness:** Charts or tables comparing model performance with and without Chain of Thought reasoning.

![alt text](images/07_cot_results_effectiveness.png)

# Benchmark

Testing how well reasoning models perform on real-world tasks.

## Code

**Code reasoning benchmarks:** Popular datasets and tests used to evaluate how well models can solve programming problems.

![alt text](images/08_benchmark_code_datasets.png)

**Code benchmark results:** Performance scores showing which models are best at coding tasks.

![alt text](images/09_benchmark_code_results.png)

## Math

**Math reasoning benchmarks:** Standard math problem sets used to test model reasoning abilities on arithmetic, algebra, and word problems.

![alt text](images/10_benchmark_math_datasets.png)

**Math benchmark results:** Accuracy scores comparing different models on math questions.

![alt text](images/11_benchmark_math_results.png)

# Metrics

Metrics to compute the reasoning model - `Pass@k`

**Understanding Pass@k (Step-by-Step):**

Pass@k measures how often a model gets the right answer when given multiple tries. Here's how it works:

1. **Give the model a problem** - For example, a coding or math question
2. **Generate k solutions** - Ask the model to create k different answers (e.g., k=5 means 5 attempts)
3. **Test each solution** - Check if any of these k solutions is correct
4. **Count success** - If at least one solution out of k is correct, it "passes"
5. **Calculate the metric** - Pass@k = (number of problems that passed) / (total problems tested)

**Example:** 
- Pass@1: Model gets 1 try. If it's correct → pass ✓
- Pass@5: Model gets 5 tries. If any 1 of the 5 is correct → pass ✓
- Pass@10: Model gets 10 tries. If any 1 of the 10 is correct → pass ✓

Higher k values give the model more chances, so Pass@10 > Pass@5 > Pass@1

**Why it matters:** Pass@k helps us understand if a model can find the right answer when given multiple attempts, which is useful for real-world applications where we can generate and test multiple solutions.

---

### Pass@k Formula

The mathematical formula for Pass@k is:

$$\text{Pass@k} = \mathbb{E}_{\text{Problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

Where:
- $n$ = total number of samples generated per problem
- $c$ = number of correct samples among the $n$ samples
- $k$ = number of samples we consider (k ≤ n)
- $\binom{n}{k}$ = binomial coefficient "n choose k"

**Simplified interpretation:** The probability that at least one correct solution appears in k randomly selected samples.

---

### Understanding n vs k (Important!)

**Think of it like a lottery ticket analogy:**

**n (total samples):** `How many solutions the model actually generates`
- Example: Model generates n=10 different answers to a coding problem

**k (samples considered):** `How many of those solutions we randomly pick and check`
- Example: We randomly select k=3 out of the 10 to evaluate

**Why k ≤ n?** You can't check more solutions than you generated!

**Real-World Scenario:**

Imagine asking ChatGPT to solve a math problem:

1. **Generate n=10 solutions** 
   - Model creates 10 different attempts at solving the problem
   - Some might be correct, some wrong
   
2. **Check k solutions**
   - **Pass@1**: Randomly pick 1 solution from the 10 → Is it correct?
   - **Pass@3**: Randomly pick 3 solutions from the 10 → Is at least 1 correct?
   - **Pass@5**: Randomly pick 5 solutions from the 10 → Is at least 1 correct?

**Key Point:** 
- We **generate** `n solutions once (expensive computation)`
- We **evaluate** Pass@k for different values of k (1, 3, 5, etc.) from those same n solutions
- This lets us estimate: "If I only looked at k solutions, what's my chance of finding a correct one?"

**Why this matters:**
- Generating solutions costs time/money (API calls, GPU usage)
- By generating n=100 solutions once, we can calculate Pass@1, Pass@10, Pass@50 all from the same batch!
- Helps us understand the trade-off between computational cost (k) and success rate

---

### Step-by-Step Computation Example

Let's calculate Pass@3 for a coding problem:

**Given:**
- We generate $n = 10$ solutions for a problem
- Out of these 10, $c = 4$ solutions are correct
- We want Pass@3 (checking 3 random solutions)

**Step 1: Identify the values**
- $n = 10$ (total samples)
- $c = 4$ (correct samples)
- $k = 3$ (samples to check)

**Step 2: Calculate "ways to pick k samples with NO correct ones"**

This is $\binom{n-c}{k} = \binom{10-4}{3} = \binom{6}{3}$

$$\binom{6}{3} = \frac{6!}{3!} = \frac{6 \times 5 \times 4}{3 \times 2 \times 1} = 20$$

**Step 3: Calculate "total ways to pick k samples"**

This is $\binom{n}{k} = \binom{10}{3}$

$$\binom{10}{3} = \frac{10!}{3!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = 120$$

**Step 4: Calculate probability of picking NO correct solutions**

$$P(\text{no correct}) = \frac{\binom{6}{3}}{\binom{10}{3}} = \frac{20}{120} = 0.167$$

**Step 5: Calculate Pass@3 (probability of at least 1 correct)**

$$\text{Pass@3} = 1 - P(\text{no correct}) = 1 - 0.167 = 0.833$$

**Result:** Pass@3 = 83.3% 

This means if we randomly pick 3 solutions out of the 10 generated, there's an 83.3% chance at least one will be correct!

---

### Quick Calculation for Different k values

Using the same example ($n=10$, $c=4$):

| k | Formula | Calculation | Pass@k |
|---|---------|-------------|--------|
| 1 | $1 - \frac{\binom{6}{1}}{\binom{10}{1}}$ | $1 - \frac{6}{10}$ | **40%** |
| 3 | $1 - \frac{\binom{6}{3}}{\binom{10}{3}}$ | $1 - \frac{20}{120}$ | **83.3%** |
| 5 | $1 - \frac{\binom{6}{5}}{\binom{10}{5}}$ | $1 - \frac{6}{252}$ | **97.6%** |

**Key insight:** As k increases, Pass@k increases because we have more chances to find a correct solution!

![alt text](images/12_metrics_passk_visualization.png)

## Temperature vs Pass@K

**What is Temperature?**

Temperature controls how "creative" or "random" the model's outputs are:
- **Low temperature (0.0 - 0.3)**: Model is deterministic and conservative, picks the most likely words
- **High temperature (0.7 - 1.0+)**: Model is creative and diverse, explores different possibilities

**The Trade-off:**

`Low temperature provides good result but it doesn't provide diverse solutions`
- ✅ More accurate and consistent answers
- ❌ All n solutions look very similar (not diverse)
- ❌ Poor for Pass@k because if the first answer is wrong, others will likely be wrong too

`High temperature will also introduce adverse effect on your response`
- ✅ Very diverse solutions (good for Pass@k)
- ❌ Can produce nonsensical or incorrect answers
- ❌ Too much randomness hurts quality

**For Pass@k, we need a sweet spot:**
- **Temperature too low (0.2)**: Generate n=10 solutions, but they're all nearly identical → Pass@10 ≈ Pass@1 (no benefit!)
- **Temperature too high (1.5)**: Generate n=10 diverse solutions, but most are garbage → Pass@10 still low
- **Optimal temperature (0.6-0.8)**: Diverse AND reasonable solutions → Pass@k improves significantly

**Simple Example:**

Question: "What's 15% of 80?"

**Temperature = 0.0** (10 solutions):
1. "15% of 80 is 12"
2. "15% of 80 is 12"
3. "15% of 80 is 12"
... (all identical) → Pass@10 = Pass@1

**Temperature = 0.7** (10 solutions):
1. "15% of 80 = 0.15 × 80 = 12 ✓"
2. "To find 15% of 80: (15/100) × 80 = 12 ✓"
3. "80 × 0.15 = 12 ✓"
4. "15% means 15/100, so 80/100 × 15 = 12 ✓"
... (diverse approaches, multiple correct) → Pass@10 >> Pass@1

**Key Takeaway:** Temperature affects how diverse your n solutions are, which directly impacts Pass@k effectiveness. Find the balance between diversity and quality!

**Visual comparison of temperature settings:** Graph or chart showing how different temperature values affect Pass@k performance. Lower temperatures give consistent but similar outputs, while higher temperatures provide diversity but with quality trade-offs.

![alt text](images/13_temperature_vs_passk_chart.png)

**Temperature and diversity relationship:** Demonstrates the sweet spot - shows experimental results of Pass@k scores at various temperature settings, helping you choose the optimal temperature for your use case.

![alt text](images/14_temperature_vs_passk_analysis.png)

---

# Scaling with RL

Using Reinforcement Learning (RL) to improve reasoning models by training them with rewards and feedback, making them better over time.

**What is RL for LLMs:** Introduction to how reinforcement learning works with language models - the model generates solutions, gets feedback (rewards) on quality, and learns to produce better reasoning over time.

![alt text](images/15_scaling_rl_overview.png)

**RL training process for reasoning:** Shows the feedback loop - model generates reasoning steps, receives rewards for correct logic, and updates its parameters to improve future responses.

![alt text](images/16_scaling_rl_approach.png)

**Scaling laws with RL:** Demonstrates how reasoning performance improves as we increase model size, training data, or compute resources when using RL. Bigger models + RL = better reasoning.

![alt text](images/17_scaling_rl_methods.png)

**RL results and comparisons:** Performance metrics comparing models trained with and without RL. Shows concrete improvements in Pass@k, accuracy, or other benchmarks when RL is applied.

![alt text](images/18_scaling_rl_results.png)

**Advanced RL techniques or case studies:** Examples of specific RL approaches (like PPO, RLHF) applied to reasoning tasks, or real-world success stories showing how RL transforms model capabilities.

![alt text](images/19_scaling_rl_comparison.png)

---

# GRPO (Group Relative Policy Optimization)

## What is GRPO?

**GRPO** is a simpler and more efficient reinforcement learning technique for training reasoning models. Think of it as "learning by comparison within a group" - instead of using an external critic to judge each response, GRPO lets the model's own outputs compete against each other.

**Key Idea:** Generate multiple responses to the same problem, compare them, and learn from the best ones in the group.

**How GRPO Works (Simple 4-Step Process):**

1. **Generate Group Responses** 
   - Given a problem, generate multiple solutions (e.g., 4-8 different answers)
   - Example: "Solve 25 × 16" → Model generates 8 different solution attempts

2. **Evaluate Each Response**
   - Check which solutions are correct/incorrect
   - Give rewards based on correctness (correct answer = high reward, wrong = low reward)

3. **Compare Within the Group (This is the KEY!)**
   - Calculate **relative** advantage: How good is each response compared to the group average?
   - Don't need a separate critic model - just compare responses to each other
   - Example: If 3 out of 8 solutions are correct, those 3 get positive advantage, others get negative

4. **Update the Model**
   - Reinforce (strengthen) the better responses
   - Discourage (weaken) the worse responses
   - Model learns: "This approach worked better than my other attempts"

---

## Example Scenario: Calculate Advantages

**Advantage** = How much better (or worse) is this response compared to the average of all responses in the group?

### Formula:

```
Advantage = Individual Reward - Group Average Reward
```

### Problem: "What is 25 × 4?"

The model generates **8 responses**:

| Response | Answer | Correct? | Reward |
|----------|--------|----------|--------|
| 1        | 100    | ✓        | 1      |
| 2        | 100    | ✓        | 1      |
| 3        | 100    | ✓        | 1      |
| 4        | 90     | ✗        | 0      |
| 5        | 110    | ✗        | 0      |
| 6        | 100    | ✓        | 1      |
| 7        | 95     | ✗        | 0      |
| 8        | 80     | ✗        | 0      |

**Step 1: Calculate Group Average**

```
Group Average = (1 + 1 + 1 + 0 + 0 + 1 + 0 + 0) / 8 = 4 / 8 = 0.5
```

**Step 2: Calculate Each Response's Advantage**

```
Response 1: Advantage = 1 - 0.5 = +0.5  ✓ (above average)
Response 2: Advantage = 1 - 0.5 = +0.5  ✓ (above average)
Response 3: Advantage = 1 - 0.5 = +0.5  ✓ (above average)
Response 4: Advantage = 0 - 0.5 = -0.5  ✗ (below average)
Response 5: Advantage = 0 - 0.5 = -0.5  ✗ (below average)
Response 6: Advantage = 1 - 0.5 = +0.5  ✓ (above average)
Response 7: Advantage = 0 - 0.5 = -0.5  ✗ (below average)
Response 8: Advantage = 0 - 0.5 = -0.5  ✗ (below average)
```

**What This Means:**

- **Positive Advantage (+0.5)**: Response is BETTER than average → Increase probability
- **Negative Advantage (-0.5)**: Response is WORSE than average → Decrease probability

**Key Insight:** Judging **relative** to the group, not in isolation. This creates a baseline for learning.

---

**Why "Group Relative"?**
- **Group**: Work with multiple responses at once
- **Relative**: Judge each response relative to others in the group (not absolute scoring)
- **Policy Optimization**: Improve the model's strategy for generating responses

**Analogy:** Like having a student (actor) and a teacher (critic). The teacher evaluates each answer and helps the student learn.

**Pros:**
- ✅ Very stable training
- ✅ Proven effective across many domains
- ✅ Good for complex reward functions

**Cons:**
- ❌ Needs two models (actor + critic) = more memory
- ❌ Slower training
- ❌ More complex to implement

**Image 19 Explanation:**

![alt text](images/20_grpo_architecture.png)

*This image shows the GRPO training pipeline. You can see how the model generates multiple outputs for each input, these outputs are evaluated, and then the model is updated based on which outputs performed better relative to the group average.*

**Image 20 Explanation:**

![alt text](images/21_grpo_blocks_explanation.png)

*This diagram illustrates the mathematical foundation of GRPO - showing how advantages are calculated by comparing each response's reward to the group mean. Responses above average get positive reinforcement, those below get negative reinforcement.*

---

## Reference Model and Reward Model in GRPO

### Reference Model

**What is it?** A frozen copy of your policy model taken at the start of GRPO training.

**Purpose:**
- Prevents the model from changing too drastically during RL
- Calculates KL divergence (how much policy has drifted)
- Stops the model from "forgetting" SFT knowledge

**How it's created:**
```
SFT Model → Copy weights → Reference Model (FROZEN)
                        ↓
                   Policy Model (trainable)
```

**Why needed?** Without it, RL training might generate gibberish that "games" the reward while destroying language capabilities.

---

### Reward Model

**What is it?** Evaluates response quality and assigns numerical scores.

**Two Types:**

**1. Rule-Based (Simple) - No Training Needed**
- Math/Code: Just check correctness automatically
- `reward = 1 if correct else 0`

**2. Learned (Complex) - Requires Training**
- Used for: Open-ended tasks, helpfulness
- Training: Learn from human preference pairs
  ```
  Response A (preferred) → high score
  Response B (rejected) → low score
  ```

---

### How They Work Together

**GRPO Training Loop:**
```
1. Policy Model → generates N responses
2. Reward Model → scores each response
3. Calculate advantages (vs group average)
4. Reference Model → provides KL penalty (don't drift too far)
5. Update Policy: Loss = -advantage + β × KL(Policy || Reference)
```

**What is KL Divergence?**

**KL (Kullback-Leibler) Divergence** measures how different two probability distributions are.

**Simple Terms:** How much has the **Policy Model** (being trained) changed from the **Reference Model** (frozen)?

**The Two Models:**
- **Reference Model** = Frozen snapshot at training start (never changes)
- **New Model** = Policy Model being updated during GRPO (changes with training)

**Formula:** KL(Policy || Reference) = measures distribution difference

**Values:**
- KL = 0 → No change (Policy = Reference)
- KL = small → Slight change ✓
- KL = large → Too much change ⚠️

**Example:**

```
Same Input: "What animal?"

Reference Model (frozen):  ["cat": 60%, "dog": 30%, "bird": 10%]
New Model (after 100 steps): ["cat": 65%, "dog": 25%, "bird": 10%]
→ KL = 0.02 (small, good!)

New Model (after 500 steps): ["cat": 10%, "dog": 80%, "bird": 10%]
→ KL = 0.8 (large, triggers penalty!)
```

**Why KL Penalty?**

```python
Loss = -advantage + β × KL(Policy || Reference)
                    ↑
            Penalty if model changes too much
```

**Purpose:**
- ✅ Prevents model from "gaming" rewards with gibberish
- ✅ Keeps model coherent (doesn't forget language)
- ✅ Balances improvement vs stability

**Training Timeline:**
```
Pre-training → SFT → [Reward Model Training*] → GRPO
                 ↓                                ↓
            Reference (frozen)          Policy (updates)

*Only if using learned reward model
```

**📖 For comprehensive explanation with additional diagrams, configuration examples, and β (kl_penalty) vs KL divergence distinction, see:**
[Understanding KL Divergence in Alignment](../../../NeMo/Alignment/alignment.md#understanding-kl-divergence-in-alignment)

**Image 21 Explanation:**

![alt text](images/22_grpo_reference_reward_models.png)

In GRPO --> ` **Reference Model** and **Reward Model** as **Frozen Model(Not trained)** and only **Policy Model** is **TRAINED Model** `

*This visualization demonstrates the training results or algorithm flow for GRPO, showing how the model improves over iterations by consistently learning from group comparisons.*



### When is GRPO Applied?
`No, GRPO is NOT used during pre-training.`

Training Stages:

```
1. Pre-training (Base Model)
   ↓
   Uses: Next-token prediction on massive text data
   Goal: Learn language patterns, facts, grammar
   
2. Supervised Fine-tuning (SFT)
   ↓
   Uses: High-quality human demonstrations
   Goal: Learn task formats and basic reasoning
   
3. GRPO Training ← HERE IS WHERE GRPO HAPPENS
   ↓
   Uses: Reinforcement learning with group comparisons
   Goal: Optimize reasoning quality and correctness
```

**Why not during pre-training?**

- Pre-training is about learning language fundamentals
- GRPO needs a model that can already generate coherent responses
- GRPO optimizes for specific tasks (reasoning, math, coding)

### How Are Weights Adjusted Based on Policy?
Simple Explanation:

`GRPO adjusts weights to make "better responses" more likely and "worse responses" less likely.`

**Step-by-Step Process:**

```
1. Generate Responses
   Model → "Response 1, 2, 3, ..., 10"

2. Calculate Advantages
   Compare each response to group average
   Response 1 (correct): advantage = +0.7 ✓
   Response 8 (wrong): advantage = -0.3 ✗

3. Calculate Loss (Policy Gradient)
   For each response:
   Loss = -advantage × log(probability of that response)
   
   - Positive advantage → Lower loss → Increase probability
   - Negative advantage → Higher loss → Decrease probability

4. Backpropagation
   Gradients flow backward through the model
   Weights update to:
   ↑ Strengthen patterns that led to good responses
   ↓ Weaken patterns that led to bad responses
```

**Concrete Example:**
Problem: "What is 5 × 3?"

**Generated responses:**

Response A: "15" (correct, advantage = +0.8)
Response B: "8" (wrong, advantage = -0.2)

**Weight updates:**

Neurons that activated for Response A → weights increased
Neurons that activated for Response B → weights decreased
Next time: Model more likely to generate "15", less likely "8"

---

## GRPO vs PPO

### Quick Comparison Table

| Aspect | PPO (Proximal Policy Optimization) | GRPO (Group Relative Policy Optimization) |
|--------|-------------------------------------|-------------------------------------------|
| **Complexity** | More complex - needs critic model | Simpler - no critic needed |
| **Components** | Actor + Critic (2 models) | Actor only (1 model) |
| **Comparison** | Each response vs learned value function | Responses vs group average |
| **Memory** | High (stores critic model) | Lower (no critic) |
| **Training Speed** | Slower | Faster |
| **Stability** | Very stable | Stable enough for most tasks |

---

### PPO (Proximal Policy Optimization)

**What is PPO?**
PPO is a popular but more complex RL algorithm that uses a "critic" to judge how good each response is.

**How PPO Works:**

1. **Actor Model**: Generates responses (the LLM)
2. **Critic Model**: Evaluates "How good is this response?" (separate neural network)
3. **Training Loop**:
   - Actor generates response
   - Critic predicts expected reward
   - Compare actual reward vs predicted
   - Update both actor and critic

---

### Formula

![alt text](images/24_ppo_formula.png)

---

### Which One Should You Use?

**Use PPO when:**
- You need maximum stability
- You have complex, noisy rewards
- Memory/compute is not a constraint
- You're working on diverse tasks beyond reasoning

**Use GRPO when:**
- You're training reasoning models (math, code, logic)
- You want faster iteration and simpler code
- You have limited GPU memory
- You can generate multiple responses per problem

**Image 22 Explanation:**

![alt text](images/23_grpo_vs_ppo_comparison.png)

In GRPO --> ` **Reference Model** and **Reward Model** as **Frozen Model(Not trained)** and only **Policy Model** is **TRAINED Model** `
In PPO --> ` **Reference Model** and **Reward Model** as **Frozen Model(Not trained)** and both **Policy Model** and **value Model** are **TRAINED Model** `

*This comparison diagram shows the architectural differences between PPO and GRPO. On the left, PPO shows the actor-critic setup with two models, while on the right, GRPO shows the simpler single-model approach with group-based comparison. The diagram likely also includes performance comparisons showing that GRPO achieves similar or better results with less complexity.*

---

### The Bottom Line

**GRPO is gaining popularity because:**
- It achieves similar performance to PPO with less complexity
- Perfect for reasoning tasks where you naturally generate multiple solutions
- Faster to train and easier to implement
- Lower resource requirements

**Think of it this way:**
- **PPO** = Having a dedicated teacher evaluate every answer (accurate but expensive)
- **GRPO** = Peer review within a study group (efficient and effective)

Both work well, but GRPO is becoming the preferred choice for modern reasoning model training!

---

# Length Bias

## What is Length Bias?

**Length Bias** is a problem where reward models incorrectly prefer longer responses over shorter ones, even when the shorter response is better.

**The Problem:**
- Reward models often confuse "longer" with "better"
- A concise, correct 2-line answer might score lower than a verbose, incorrect 10-line answer
- This happens because reward models learn patterns like "helpful responses tend to be detailed"

**Why It Matters:**
- Models waste tokens generating unnecessarily long responses
- Longer responses cost more (API costs scale with length)
- User experience suffers (people want quick, direct answers)
- Training becomes inefficient

---

## Image 24: Demonstrating Length Bias Problem

![alt text](images/25_length_bias_problem.png)

**What this shows:**

This image demonstrates the core length bias problem with concrete examples:

**Left side (Short response):**
- Question: "What is 2+2?"
- Model A answer: "4" (correct, concise)
- Reward score: 6.5/10 ❌ (unfairly low!)

**Right side (Long response):**
- Same question: "What is 2+2?"
- Model B answer: "Well, let me break this down step by step. First, we need to understand addition. When we combine 2 and 2, we count 1, 2, 3, 4. Therefore the answer is 4." (correct but verbose)
- Reward score: 8.5/10 ✓ (higher score just for being longer!)

**Key Insight:** Both answers are correct, but the reward model favors length over quality. This is length bias in action!

---

## Image 25: Length Bias Impact on Training

![alt text](images/26_length_bias_training_impact.png)

**What this shows:**

This chart demonstrates how length bias affects model behavior during training:

**X-axis:** Training iterations (time/steps)
**Y-axis:** Average response length (in tokens)

**Three lines showing:**

1. **Blue line (No length penalty):** Response length keeps increasing
   - Model learns: "Longer = higher reward"
   - By iteration 10k, responses are 500+ tokens
   - Inefficient and expensive!

2. **Orange line (With length penalty):** Response length stays controlled
   - Penalty discourages unnecessary verbosity
   - Stays around 150-200 tokens
   - Efficient and cost-effective!

3. **Red dotted line (Optimal length):** Target response length
   - Shows where responses should ideally be
   - Around 100-150 tokens for most tasks

**Key Takeaway:** Without length penalty, models become increasingly verbose over training. With length penalty, they stay concise.

---

## Image 26: Solutions to Length Bias

![alt text](images/27_length_bias_solutions.png)

**What this shows:**

This image presents different techniques to combat length bias:

**Method 1: Length-Normalized Rewards**
```
Before: Reward = quality_score
After:  Reward = quality_score / (response_length^α)

where α = penalty strength (typically 0.1-0.3)
```
- Divides reward by length
- Longer responses need proportionally higher quality to match short ones

**Method 2: Length Penalty in Loss Function**
```
Loss = -advantage × log(π) + β × length_penalty

length_penalty = (num_tokens - target_length)²
```
- Adds explicit penalty for exceeding target length
- β controls penalty strength

**Method 3: Length-Aware Training Data**
- Include examples showing concise responses getting high rewards
- Train reward model on preference pairs that favor brevity
- Example pairs: "Brief correct answer" > "Verbose correct answer"

**Comparison Results Table:**

| Method | Avg Length | Quality Score | Cost |
|--------|------------|---------------|------|
| No penalty | 450 tokens | 8.2/10 | High |
| Length normalized | 180 tokens | 8.1/10 | Medium |
| Length penalty | 120 tokens | 8.0/10 | Low |
| Combined | 100 tokens | 8.3/10 | Lowest |

**Key Insight:** Using length penalties maintains quality while dramatically reducing response length and cost!

---

## Image 27: Practical Example

![alt text](images/28_length_bias_practical_example.png)

**Before vs After Length Bias Fix:**

**Without Length Bias Fix:**
```
User: "What's the capital of France?"
Model: "That's a great question! Let me provide you with a comprehensive 
answer. France, officially known as the French Republic, is a country 
in Western Europe. Its capital city, which has been the center of French 
culture and politics for centuries, is Paris. Paris, also known as the 
City of Light, is located in the north-central part of the country..."
(300+ tokens)
```

**With Length Bias Fix:**
```
User: "What's the capital of France?"
Model: "Paris."
(1 token)
```

**Result:**
- ✅ Both correct
- ✅ Second response 300x more efficient
- ✅ Better user experience
- ✅ Lower API costs

---

## Bottom Line

**Length bias** teaches us that "more is not always better." By implementing length penalties and normalization, we can train models that are:
- More efficient (shorter responses)
- More cost-effective (fewer tokens)
- Better user experience (get to the point)
- Still maintain high quality

This is especially important for production systems where every token costs money!


![alt text](images/29_length_bias_summary.png)

---

# DAPO (Direct Alignment from Preference Optimization)

## What is DAPO?

**DAPO** is an improved version of GRPO that addresses key limitations. It's designed to make preference-based training more efficient and stable.

**Key Problem DAPO Solves:**

In standard GRPO, when you generate multiple responses and calculate advantages, there's a problem:
- If most responses are bad, the "best" response might still be mediocre
- The model learns to prefer "less bad" instead of "truly good"
- This can lead to slow improvement

**DAPO's Solution:**

Instead of comparing responses only within a group, DAPO uses **explicit human preferences** to guide training more directly.

---

## GRPO vs DAPO: The Key Difference

### GRPO Approach:
```
Question: "What is 5 × 3?"

Generate 5 responses:
1. "15" ✓ (correct)
2. "8"  ✗ (wrong)
3. "12" ✗ (wrong)
4. "20" ✗ (wrong)
5. "10" ✗ (wrong)

Group average = 1/5 = 0.2
Response 1 gets advantage = +0.8 (best in group)
BUT: We're learning from 80% wrong answers!
```

### DAPO Approach:
```
Question: "What is 5 × 3?"

Use human preference data:
Preferred answer: "15" (definitely correct)
Rejected answer: "8" (definitely wrong)

Direct signal: Make "15" much more likely
              Make "8" much less likely

Result: Cleaner signal, faster learning
```

---

## Example: Why DAPO Improves Over GRPO

**Scenario:** Training a model on math problems

### GRPO Problem:
```
Problem: "Calculate 17 × 8"

Generated responses:
1. "136" ✓ (correct)
2. "134" ✗ (close but wrong)
3. "130" ✗ (wrong)
4. "140" ✗ (wrong)

GRPO: Compares within group
- Response 1 gets +0.75 advantage (best)
- Response 2 gets +0.25 advantage (second best)
- Model learns "134 is pretty good" (WRONG!)
```

### DAPO Solution:
```
DAPO: Uses explicit preferences
- "136" is preferred ✓
- "134" is rejected ✗
- Clear signal: Only "136" is correct
- Model learns: Don't generate "134" even though it's close
```

**Result:** DAPO prevents the model from learning bad patterns that happen to be "best in a bad group."

---

## Image 29: DAPO Architecture

![alt text](images/30_dapo_architecture.png)

*This image shows how DAPO improves upon GRPO by incorporating direct preference signals, leading to more efficient and accurate training for reasoning tasks.*

---

# Dr.GRPO (Dropout-Regularized GRPO)

## What is Dr.GRPO?

**Dr.GRPO** is another enhancement to GRPO that tackles the problem of **overfitting** during training.

**The Problem Dr.GRPO Solves:**

In regular GRPO, the model can start "memorizing" specific responses instead of learning general reasoning patterns:

```
Training problem: "What is 2 + 2?"
Model memorizes: "The answer is 4"

New problem: "Calculate 2 + 2"
Model struggles: Different wording → doesn't recognize pattern
```

---

## How Dr.GRPO Works

**Dr.GRPO adds "dropout" during training:**

**Dropout** = Randomly "turn off" some neurons during training

**Why this helps:**
- Forces the model to learn robust patterns
- Prevents over-reliance on specific neurons
- Makes the model generalize better to new problems

---

## Example: GRPO vs Dr.GRPO

**Training Scenario:** Teaching the model multiplication

### Regular GRPO (Overfitting):
```
Sees problem: "What is 5 × 3?"
Memorizes: "5 × 3 = 15"

New problem: "Calculate 5 times 3"
Gets confused: Never saw this exact phrasing
Performance: 60% accuracy on variations
```

### Dr.GRPO (Generalizes):
```
Sees problem: "What is 5 × 3?"
With dropout: Some neurons disabled randomly
Forces model to learn: multiplication concept, not exact phrase

New problem: "Calculate 5 times 3"
Handles well: Learned the general pattern
Performance: 85% accuracy on variations
```

---

## Visual Comparison: The Three Approaches

### GRPO (Original):
```
✅ Simple, works well for basic tasks
❌ Can learn from "best of bad" responses
❌ May overfit to training examples
```

### DAPO (Direct Alignment):
```
✅ Uses explicit human preferences
✅ Cleaner training signal
✅ Faster learning, fewer bad patterns
❌ Requires preference data
```

### Dr.GRPO (Dropout Regularized):
```
✅ Better generalization
✅ Less overfitting
✅ Works on new phrasings/variations
❌ Slightly slower training
```

---

## Real-World Example: All Three Methods

**Task:** Train model to solve word problems

**Problem:** "Sarah has 5 apples. She buys 3 more. How many does she have?"

### GRPO Training:
```
Generate 4 responses:
1. "8" ✓
2. "5" ✗
3. "3" ✗
4. "7" ✗

Learns: "8 is best in this group"
Issue: 75% responses were wrong, weak signal
```

### DAPO Training:
```
Use preferences:
Preferred: "8 apples. 5 + 3 = 8" ✓
Rejected: "5 apples" ✗

Learns: Clear correct vs wrong pattern
Benefit: Strong, clean signal
```

### Dr.GRPO Training:
```
Same as GRPO but with dropout:
Some neurons randomly disabled

Test: "Sarah has 5 oranges. She gets 3 more. Total?"
GRPO: Confused (different words!)
Dr.GRPO: "8" ✓ (learned general addition pattern)

Benefit: Handles variations better
```

---

## Which Should You Use?

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **GRPO** | Simple tasks, limited resources | Easy to implement | Can learn from bad groups |
| **DAPO** | When you have preference data | Fastest learning | Needs human preferences |
| **Dr.GRPO** | Need generalization | Robust to variations | Slightly slower |

**Practical Recommendation:**
- **Start with:** GRPO (simplest)
- **Upgrade to:** DAPO if you have preference data
- **Use:** Dr.GRPO if model overfits or struggles with variations

---

## Bottom Line

**GRPO → DAPO → Dr.GRPO** represents evolution in training reasoning models:

1. **GRPO:** Compare within groups (simple but can learn bad patterns)
2. **DAPO:** Use direct preferences (cleaner signal, better quality)
3. **Dr.GRPO:** Add dropout (better generalization, less overfitting)

Each method builds on the previous one to solve specific problems, making reasoning models more reliable and efficient!

---

# Applications

## Deepseek R0

**What is Deepseek R0?**

Deepseek R0 is an early reasoning model that demonstrates the application of RL techniques (like GRPO) to improve mathematical and logical reasoning capabilities.

---

### Image 30: Deepseek R0 Architecture

![alt text](images/31_deepseek_r0_architecture.png)

**What this shows:**

- **Model Structure:** Base model + RL training pipeline
- **Training Process:** How R0 uses reinforcement learning to improve reasoning
- **Key Components:** Shows the policy model, reward mechanism, and update cycle
- **Innovation:** Demonstrates how GRPO-style training works in practice

**Key Takeaway:** R0 architecture shows the practical implementation of RL-based reasoning training.

---

### Image 31: R0 Training Results

![alt text](images/32_deepseek_r0_training_results.png)

**What this shows:**

Performance improvements during training:
- **X-axis:** Training iterations/steps
- **Y-axis:** Accuracy on reasoning tasks (math, logic)
- **Curves:** Show how accuracy improves over time
- **Comparison:** Before RL vs After RL training

**Key Insight:** Clear improvement in reasoning ability as RL training progresses.

---

### Image 32: R0 Benchmark Performance

![alt text](images/33_deepseek_r0_benchmark_performance.png)

**What this shows:**

R0's performance on standard benchmarks:
- **Math benchmarks:** GSM8K, MATH dataset scores
- **Comparison:** R0 vs other models (GPT-3.5, etc.)
- **Pass@k metrics:** How often R0 gets correct answers

**Key Takeaway:** R0 demonstrates competitive performance, proving RL training works for reasoning.

---

### Image 33: R0 Example Outputs

![alt text](images/34_deepseek_r0_example_outputs.png)

**What this shows:**

Real examples of R0 solving problems:
- **Problem:** Math word problem or logical puzzle
- **R0's reasoning:** Step-by-step thought process
- **Final answer:** Correct solution with explanation

**Key Insight:** R0 shows its work (Chain of Thought), making reasoning transparent and verifiable.

---

## Deepseek R1

**What is Deepseek R1?**

Deepseek R1 is the improved successor to R0, incorporating advanced techniques like DAPO and better training strategies for even stronger reasoning capabilities.

---

### Image 34: R1 Architecture & Improvements

![alt text](images/35_deepseek_r1_architecture_improvements.png)

**What this shows:**

How R1 improves upon R0:
- **Enhanced Architecture:** Larger model, better training
- **New Techniques:** DAPO, Dr.GRPO, length bias fixes
- **Training Pipeline:** More sophisticated reward models
- **Comparison:** R0 architecture vs R1 architecture

**Key Takeaway:** R1 incorporates all the advanced techniques we've learned (DAPO, dropout, length penalties).

---

### Image 35: R1 Training Efficiency

![alt text](images/36_deepseek_r1_training_efficiency.png)

**What this shows:**

R1 trains more efficiently than R0:
- **X-axis:** Training compute/time
- **Y-axis:** Model performance
- **Two curves:** R0 training curve vs R1 training curve
- **Key difference:** R1 reaches higher performance with less training

**Key Insight:** R1's improvements (DAPO, etc.) make training faster and more effective.

---

### Image 36: R1 Benchmark Comparison

![alt text](images/37_deepseek_r1_benchmark_comparison.png)

**What this shows:**

R1 vs leading models on multiple benchmarks:
- **Benchmarks:** GSM8K, MATH, MMLU, HumanEval (code)
- **Models compared:** R1, GPT-4, Claude, Gemini
- **Results:** Bar charts or tables showing scores

**Key Takeaway:** R1 achieves state-of-the-art or competitive performance across diverse reasoning tasks.

---

### Image 37: R1 Reasoning Examples

![alt text](images/38_deepseek_r1_reasoning_examples.png)

**What this shows:**

Complex reasoning examples from R1:
- **Problem types:** Advanced math, multi-step logic, code generation
- **R1's approach:** Detailed step-by-step reasoning
- **Quality:** Clean, concise, correct reasoning chains

**Key Insight:** R1 handles harder problems and shows better reasoning quality than R0.

---

### Image 38: R1 vs R0 Performance Summary

![alt text](images/39_deepseek_r1_vs_r0_summary.png)

**What this shows:**

Direct comparison between R0 and R1:

| Metric | R0 | R1 | Improvement |
|--------|----|----|-------------|
| **Math accuracy** | 75% | 88% | +13% |
| **Training time** | 100 hours | 60 hours | 40% faster |
| **Response length** | 250 tokens | 120 tokens | 52% shorter |
| **Generalization** | Good | Excellent | Better |

**Key Takeaways:**
- R1 is more accurate
- R1 trains faster (thanks to DAPO)
- R1 produces shorter responses (length bias fix)
- R1 generalizes better (dropout regularization)

---

## Bottom Line: R0 → R1 Evolution

**Deepseek R0:**
- Proved RL (GRPO) works for reasoning
- Good performance on basic tasks
- Foundation for improvements

**Deepseek R1:**
- Incorporates DAPO (better training signal)
- Uses Dr.GRPO (better generalization)
- Fixes length bias (efficient responses)
- State-of-the-art reasoning performance

**Lesson:** R0 → R1 shows how theoretical improvements (DAPO, Dr.GRPO, length penalties) translate to real-world gains in accuracy, efficiency, and quality!

![alt text](images/40_applications_overview.png)

---

# Distillation

## What is Distillation?

**Distillation** is a technique to transfer knowledge from a large, powerful model (teacher) to a smaller, faster model (student). Think of it like a student learning from an expert teacher.

**The Idea:**
- **Teacher Model:** Large, accurate, but slow and expensive (e.g., GPT-4, DeepSeek R1)
- **Student Model:** Smaller, faster, cheaper, but initially less capable
- **Goal:** Make the student model perform almost as well as the teacher

---

## How Distillation Works (Simple 3 Steps)

**Step 1: Teacher Generates Examples**
```
Teacher model (R1): Solves problems with detailed reasoning
Problem: "What is 25 × 4?"
Teacher output: "Let me calculate: 25 × 4 = 100 ✓"
```

**Step 2: Student Learns from Teacher**
```
Student model: Trained to mimic teacher's outputs
Input: Same problem "What is 25 × 4?"
Target: Learn to produce "25 × 4 = 100" like teacher
```

**Step 3: Student Becomes Independent**
```
Student model: Now solves problems on its own
Uses knowledge learned from teacher
Runs much faster and cheaper!
```

---

## Why Use Distillation?

**Problem:** Large models are powerful but expensive

| Model | Size | Speed | Cost per query |
|-------|------|-------|----------------|
| Teacher (R1) | 70B params | Slow (2 sec) | $0.10 |
| Student (distilled) | 7B params | Fast (0.2 sec) | $0.01 |

**Solution:** Distillation gives you 80-90% of teacher's performance at 10% of the cost!

---

## Real-World Example

**Scenario:** You need a math-solving chatbot for your app

**Option 1: Use Teacher Model Directly**
```
User asks: "What is 15% of 200?"
R1 (teacher): Generates answer (slow, expensive)
Cost: $0.10 per question
For 10,000 users/day = $1,000/day 💸
```

**Option 2: Use Distilled Student Model**
```
Training: R1 solves 100,000 practice problems
Student learns from R1's solutions
Deployment: Student model answers questions
Cost: $0.01 per question
For 10,000 users/day = $100/day ✅
```

**Savings:** 90% cost reduction with minimal quality loss!

---

## Analogy

Think of it like this:

**Without Distillation:**
- Hiring Einstein to answer every math question (accurate but expensive)

**With Distillation:**
- Einstein teaches a smart student for a few months
- Student can then answer most questions correctly
- Much cheaper than calling Einstein every time!

---

## How Training Actually Works in Distillation

### Important Question: How Do We Get Probabilities from LLMs?

**You might wonder:** "LLMs just return text responses. How do we get probabilities for training?"

**The answer:** LLMs ALWAYS generate probabilities internally - that's how they work!

---

### What Happens Inside Every LLM:

**What you see (normal usage):**
```
User: "What is the capital of France?"
Model: "Paris"
```

**What actually happens inside:**
```
1. Model calculates probability for EVERY possible next token:
   - "Paris": 0.95 (95%)
   - "London": 0.02 (2%)
   - "Berlin": 0.01 (1%)
   - "The": 0.005 (0.5%)
   - ... (50,000+ other tokens)

2. Model picks highest probability: "Paris"

3. API returns to you: "Paris" (just the text)
```

**Key Point:** Probabilities exist at every step; they're just hidden from normal API users!

---

### Step-by-Step Training Process:

**Step 1: Teacher Model Generates "Soft" Outputs**

Instead of just the final answer, the teacher provides **probability distributions**:

```
Question: "What is 25 × 4?"

Teacher Model Internal Output:
- "100": 0.98 (98%)
- "110": 0.01 (1%)
- "90": 0.005 (0.5%)
- "25": 0.003 (0.3%)
- Other tokens: 0.002 (0.2%)
```

**Why probabilities matter:** The teacher shows not just the answer, but its confidence and uncertainty. This is much richer than just "100"!

---

**Step 2: Student Model Learns from Teacher's Probabilities**

```
Student Model (before training):
- "100": 0.40 (40%)
- "110": 0.25 (25%)
- "90": 0.20 (20%)
- "25": 0.15 (15%)

Student Model (after training):
- "100": 0.97 (97%) ← Now matches teacher!
- "110": 0.015 (1.5%)
- "90": 0.01 (1%)
- "25": 0.005 (0.5%)
```

---

**Step 3: Training Loss Function**

```python
# Distillation Loss (main signal)
distillation_loss = KL_divergence(student_probs, teacher_probs)
                    ↑
              How different is student from teacher?

# Hard Label Loss (ground truth backup)
hard_label_loss = cross_entropy(student_output, correct_answer)

# Combined Loss
total_loss = 0.9 × distillation_loss + 0.1 × hard_label_loss
             ↑                          ↑
        Learn from teacher         Learn from correct answers
```

---

### How Distillation Gets Access to Probabilities:

**Option 1: Direct Model Access (Open Source)**
```python
from transformers import AutoModel

# Load open-source models (Llama, Mistral, DeepSeek)
teacher = AutoModel.from_pretrained("deepseek-v2-70b")
student = AutoModel.from_pretrained("small-model-7b")

# Get full probability distributions
teacher_outputs = teacher(inputs)
teacher_probs = teacher_outputs.logits  # All 50K+ token probabilities!
```

**Option 2: Own the Model Weights**
```python
# If you trained the model yourself
teacher = load_local_model("my-70b-model")
# Full access to all internal states
```

**Option 3: Limited API Access**
```python
# Some APIs offer limited probability access
response = openai.chat.completion.create(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    logprobs=True,  # Request top token probabilities
    top_logprobs=5   # Show top 5 most likely tokens
)
```

**Note:** You CANNOT distill from ChatGPT API alone - you need actual model access!

---

### Real Training Code Example:

```python
# Simplified distillation training

teacher = load_model("teacher-70b")  # Needs direct access
student = initialize_model("student-7b")
teacher.eval()  # Freeze teacher weights

for batch in training_data:
    questions = batch['questions']
    
    # Get teacher's probability distribution
    with torch.no_grad():  # Don't train teacher
        teacher_logits = teacher(questions)
        teacher_probs = softmax(teacher_logits / temperature)
    
    # Get student's probability distribution
    student_logits = student(questions)
    student_probs = softmax(student_logits / temperature)
    
    # Make student match teacher
    distill_loss = kl_divergence(student_probs, teacher_probs)
    hard_loss = cross_entropy(student_logits, correct_answers)
    
    total_loss = 0.9 * distill_loss + 0.1 * hard_loss
    
    # Update only student weights
    total_loss.backward()
    optimizer.step()
```

---

### Training Progress Example:

| Week | Student Accuracy | Matches Teacher? | Loss |
|------|------------------|------------------|------|
| 0 (start) | 40% | ❌ Random | 2.8 |
| 1 | 65% | 🟡 Learning | 1.5 |
| 2 | 85% | 🟢 Close | 0.4 |
| 3 | 92% | ✅ Very close | 0.1 |
| 4 | 95% | ✅ Almost matches | 0.05 |

**Final Result:**
- Teacher: 97% accuracy, 70B params, 2000ms
- Student: 95% accuracy, 7B params, 200ms

---

### Why Learn from Probabilities Instead of Just Answers?

**Traditional Training (Hard Labels):**
```
Question: "What is 25 × 4?"
Label: "100" (just the answer)

Model learns: Binary right/wrong only
```

**Distillation (Soft Labels):**
```
Question: "What is 25 × 4?"
Teacher provides:
- "100": 98% ← Correct!
- "110": 1%  ← Common mistake
- "90": 0.5% ← Another error pattern

Model learns:
- The right answer AND
- Common mistakes to avoid AND
- Teacher's confidence level
```

**Benefits:**
- ✅ **Richer learning signal:** Understands teacher's reasoning
- ✅ **Better generalization:** Learns from teacher's uncertainty
- ✅ **Smoother training:** Gradual improvement vs binary feedback

---

### Key Parameters:

**Temperature (T):**
```
Low temperature (T=1): Sharp probabilities
- "100": 98%, "110": 1%, "90": 0.5%

High temperature (T=3): Softer probabilities
- "100": 85%, "110": 8%, "90": 4%

Sweet spot (T=2-3): Balances confidence with uncertainty
```

**Alpha (α) - Mixing ratio:**
```
α = 0.9 (typical): 90% learn from teacher, 10% from ground truth
α = 1.0: 100% learn from teacher (pure distillation)
α = 0.5: 50/50 split
```

---

### Bottom Line on Training:

**How smaller models are trained with larger models:**

1. ✅ **Teacher generates probability distributions** (not just text)
2. ✅ **Student learns to match teacher's probabilities** (richer than just answers)
3. ✅ **Requires direct model access** (not through APIs)
4. ✅ **Uses combined loss:** 90% teacher, 10% ground truth
5. ✅ **Student achieves 90-95% of teacher performance** at 10x smaller size

**Think of it like:** Instead of just copying the teacher's final answer, the student learns the teacher's thought process, confidence levels, and even common pitfalls to avoid - making for much better learning!

---

## Image 40: Distillation Process

![alt text](images/41_distillation_process.png)

**What this shows:**

- **Left side:** Large teacher model (complex architecture)
- **Arrow:** Knowledge transfer process
- **Right side:** Smaller student model (simplified architecture)
- **Results:** Performance comparison showing student achieves 85-90% of teacher's accuracy at fraction of the cost

**Key Insight:** Distillation makes powerful reasoning capabilities accessible and affordable for production use!

---

## Bottom Line

**Distillation = Making AI reasoning practical and affordable**

✅ **Keep the smarts:** Student learns reasoning patterns from teacher
✅ **Reduce the cost:** 10x smaller, 10x faster, 10x cheaper
✅ **Maintain quality:** Typically 85-95% of teacher's performance

Perfect for deploying reasoning models in real applications where speed and cost matter!

![alt text](images/42_distillation_summary_1.png)

![alt text](images/43_distillation_summary_2.png)
