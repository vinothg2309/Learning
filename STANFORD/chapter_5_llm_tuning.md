chapter_5_llm_tuning.md
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
- [Preference Tuning](#preference-tuning)
  - [What is Preference Tuning?](#what-is-preference-tuning)
  - [The Preference Tuning Process](#the-preference-tuning-process)
  - [Training with Preference Data](#training-with-preference-data)
- [Data Collection](#data-collection)
  - [How Do We Collect Preference Data?](#how-do-we-collect-preference-data)
  - [Comparison Methods for Preference Data](#comparison-methods-for-preference-data)
  - [Building Your Preference Dataset](#building-your-preference-dataset)
- [RLHF (Reinforcement Learning with Human Feedback)](#rlhf-reinforcement-learning-with-human-feedback)
  - [What is RLHF?](#what-is-rlhf)
  - [The RLHF Framework](#the-rlhf-framework)
  - [Step 1: Supervised Fine-Tuning (SFT)](#step-1-supervised-fine-tuning-sft)
  - [Step 2: Training the Reward Model](#step-2-training-the-reward-model)
  - [Step 3: Reinforcement Learning Optimization](#step-3-reinforcement-learning-optimization)
  - [The Reward Model in Action](#the-reward-model-in-action)
  - [The Reward Model Architecture and Its Role](#the-reward-model-architecture-and-its-role)
      - [Two Separate Models:](#two-separate-models)
      - [How They Work Together in RLHF:](#how-they-work-together-in-rlhf)
      - [Visual Flow:](#visual-flow)
      - [Key Points for Beginners:](#key-points-for-beginners)
      - [Where is the Reward Model "Included"?](#where-is-the-reward-model-included)
  - [How Does the LLM Update Its Parameters Based on Reward Scores?](#how-does-the-llm-update-its-parameters-based-on-reward-scores)
    - [Understanding the Update Process](#understanding-the-update-process)
    - [The Key Difference: RL vs Regular Training](#the-key-difference-rl-vs-regular-training)
    - [Step-by-Step: How Parameter Updates Work](#step-by-step-how-parameter-updates-work)
      - [Step 1: Generate Multiple Responses](#step-1-generate-multiple-responses)
      - [Step 2: Calculate the Policy Gradient](#step-2-calculate-the-policy-gradient)
      - [Step 3: Use PPO to Update Safely](#step-3-use-ppo-to-update-safely)
    - [Visual Example: Parameter Update Flow](#visual-example-parameter-update-flow)
    - [Does It Use Regular Backpropagation?](#does-it-use-regular-backpropagation)
    - [Example: Concrete Parameter Update](#example-concrete-parameter-update)
    - [Key Concepts for Beginners](#key-concepts-for-beginners)
    - [The Complete RLHF Training Loop](#the-complete-rlhf-training-loop)
  - [Bradley-Terry Formulation](#bradley-terry-formulation)
    - [What is Bradley-Terry?](#what-is-bradley-terry)
    - [The Bradley-Terry Formula](#the-bradley-terry-formula)
    - [Training the Reward Model with Bradley-Terry](#training-the-reward-model-with-bradley-terry)
    - [The Reward Model Architecture](#the-reward-model-architecture)
  - [How is the Reward Model Actually Built?](#how-is-the-reward-model-actually-built)
    - [Pairwise vs Pointwise Approach](#pairwise-vs-pointwise-approach)
    - [Building a Pairwise Reward Model - Step by Step](#building-a-pairwise-reward-model---step-by-step)
    - [Example: Pairwise Dataset Construction](#example-pairwise-dataset-construction)
    - [Why Pairwise Works Better](#why-pairwise-works-better)
<!-- /TOC -->

---

# Preference Tuning

## What is Preference Tuning?

Preference tuning is a technique to align language models with human preferences. Instead of just training on correct answers, we teach the model to generate responses that humans find more helpful, harmless, and honest. The model learns by comparing pairs of responses - one preferred and one rejected - and adjusts to favor the preferred style.

## The Preference Tuning Process

After pre-training and supervised fine-tuning, we use human feedback to further refine the model's behavior. This involves collecting preference data where humans rank different model responses, then using algorithms like RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization) to train the model.

![alt text](image.png)

## Training with Preference Data

During training, the model compares its predictions for both the chosen and rejected responses. The loss function encourages the model to assign higher likelihood to the preferred response. This gradient-based approach effectively teaches the model human preferences without explicit reward modeling.

![alt text](image-2.png)

# Data Collection

## How Do We Collect Preference Data?

To train models with human preferences, we need to collect feedback on model responses. There are different methods to gather this data, but pairwise comparison (choosing between two responses) is the most effective and widely used approach.

## Comparison Methods for Preference Data

Different ways to collect human feedback include: rating individual responses, ranking multiple responses, or comparing pairs. `Pairwise comparison is most preferred because it's easier for humans to choose between two options than to assign absolute scores.`

![alt text](image-1.png)

## Building Your Preference Dataset

Once you've collected pairwise preferences from humans, you create a dataset with prompts and their corresponding chosen/rejected response pairs. This dataset is what you'll use to train your model to align with human preferences.

![alt text](image-3.png)

# RLHF (Reinforcement Learning with Human Feedback)

## What is RLHF?

RLHF treats the language model as an "agent" that learns to take better actions (generate better text) based on human feedback. The model receives a prompt (state), generates a response (action), and gets a reward score based on how well it aligns with human preferences.

## The RLHF Framework

In reinforcement learning terms: the LLM is the agent, the prompt is the state, generating text is the action, and human feedback provides the reward signal. The model learns to maximize rewards by producing responses humans prefer.

![alt text](image-4.png)

## Step 1: Supervised Fine-Tuning (SFT)

Before RLHF, we start with a pre-trained model and fine-tune it on high-quality human demonstrations. This teaches the model to follow instructions and respond appropriately, creating a good starting point for reinforcement learning.

![alt text](image-5.png)

## Step 2: Training the Reward Model

We train a separate reward model to predict human preferences. It learns from comparison data (chosen vs. rejected responses) to assign scores that reflect how much humans would like a response. This reward model acts as a proxy for human judgment.

![alt text](image-6.png)

## Step 3: Reinforcement Learning Optimization

Using the reward model, we optimize the language model with RL algorithms (typically PPO - Proximal Policy Optimization). The model generates responses, receives reward scores, and updates its parameters to produce higher-reward (more preferred) outputs over time.

![alt text](image-7.png)

## The Reward Model in Action

The reward model takes a prompt and response as input and outputs a single score indicating quality. Higher scores mean the response better aligns with human preferences. This score guides the RL training process.

![alt text](image-8.png)

## The Reward Model Architecture and Its Role

**Important: The Reward Model is a SEPARATE model, not added to the LLM's response layer.**

**Understanding Where the Reward Model Fits:**

Many beginners think the reward model is part of the main LLM, but it's actually a **separate, independent model** that works alongside the LLM. Here's how they work together:

#### Two Separate Models:

1. **The Language Model (LLM)**: 
   - Generates responses to prompts
   - This is the model we want to improve
   - Examples: GPT, Claude, Llama

2. **The Reward Model (RM)**:
   - Acts as a "judge" or "critic"
   - Takes a prompt + response as input
   - Outputs a single score (how good the response is)
   - Built by modifying a language model's final layer to output a number instead of text

#### How They Work Together in RLHF:

**Step-by-Step Process:**

1. **LLM generates a response**
   ```
   Prompt: "Explain gravity"
   LLM → "Gravity is a force that pulls objects..."
   ```

2. **Reward Model scores the response**
   ```
   RM receives: (Prompt + LLM's response)
   RM outputs: Score = 0.75 (on a scale where higher = better)
   ```

3. **LLM learns from the score**
   - If score is high (0.8+) → LLM learns: "This type of response is good, do more of this"
   - If score is low (0.3-) → LLM learns: "This type of response is bad, avoid this"

4. **LLM updates its parameters**
   - Using reinforcement learning (PPO algorithm)
   - Adjusts weights to generate responses that get higher scores
   - Over many iterations, the LLM gets better at producing preferred responses

#### Visual Flow:

```
┌─────────────┐
│   Prompt    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Language Model │  ← This model is being trained/improved
│     (LLM)       │
└──────┬──────────┘
       │
       │ generates
       ▼
┌─────────────────┐
│   Response      │
└──────┬──────────┘
       │
       │ (Prompt + Response)
       ▼
┌─────────────────┐
│  Reward Model   │  ← Separate model that judges quality
│      (RM)       │
└──────┬──────────┘
       │
       │ outputs score
       ▼
┌─────────────────┐
│  Score: 0.75    │
└──────┬──────────┘
       │
       │ feedback
       ▼
┌─────────────────┐
│  Update LLM     │  ← LLM learns to get higher scores
│  Parameters     │
└─────────────────┘
```

#### Key Points for Beginners:

1. **Separate Models**: The reward model is NOT inside the LLM. They are two different models.

2. **Reward Model's Job**: It's trained FIRST (using pairwise comparisons) to predict what humans prefer.

3. **LLM's Learning**: The LLM uses the reward model's scores as a guide to improve itself.

4. **Think of it like**:
   - **LLM** = Student writing essays
   - **Reward Model** = Teacher grading essays
   - **Training Process** = Student learns to write better essays by getting grades from the teacher

5. **Why This Works**:
   - The reward model captures human preferences (what makes a good response)
   - The LLM learns to maximize its reward scores
   - Higher reward scores = responses more aligned with human preferences

#### Where is the Reward Model "Included"?

The reward model is:
- **NOT** embedded in the LLM's architecture
- **NOT** part of the LLM's response generation
- **Used during training only** as an external evaluator
- **Can be removed after training** - the improved LLM can work independently

Think of it like training wheels on a bicycle - they help during learning but aren't needed once the model is trained!

## How Does the LLM Update Its Parameters Based on Reward Scores?

### Understanding the Update Process

Yes, the LLM does use backpropagation, but it's different from regular supervised learning. In RLHF, we use **Reinforcement Learning (RL)** algorithms, specifically **PPO (Proximal Policy Optimization)**, to update the LLM's parameters based on reward scores.

### The Key Difference: RL vs Regular Training

**Regular Training (Supervised Learning):**
```
1. Model generates output
2. Compare with correct answer
3. Calculate loss (how wrong it was)
4. Backpropagate to update weights
```

**RLHF Training (Reinforcement Learning):**
```
1. LLM generates response
2. Reward model scores the response
3. Calculate "reward gradient" (how to improve)
4. Use PPO algorithm to update LLM weights
```

### Step-by-Step: How Parameter Updates Work

#### Step 1: Generate Multiple Responses

The LLM generates responses and gets them scored:

```
Prompt: "Explain machine learning"

Generation 1: "ML is when computers learn patterns..." 
→ Reward Score: 0.8

Generation 2: "Machine learning involves algorithms..." 
→ Reward Score: 0.6

Generation 3: "ML is AI that learns from data..." 
→ Reward Score: 0.9
```

#### Step 2: Calculate the Policy Gradient

The algorithm calculates which parameter changes would increase the probability of high-reward responses:

**Key Formula (Simplified):**
```
Gradient = Reward × ∇log P(response | prompt)
```

**What this means:**
- If reward is HIGH (0.9) → Increase probability of generating this type of response
- If reward is LOW (0.3) → Decrease probability of generating this type of response
- `∇log P` tells us which parameters to adjust to change these probabilities

#### Step 3: Use PPO to Update Safely

PPO ensures the model doesn't change too drastically in one step:

**Why PPO?**
- Prevents the model from making huge changes that could break it
- Uses a "trust region" - only allows small, safe updates
- Clips gradient updates to stay within bounds

**PPO Update Process:**
```
1. Calculate how much we want to change (policy gradient)
2. Clip the change to prevent extreme updates
3. Apply the clipped gradient using backpropagation
4. Update LLM parameters slightly toward higher rewards
```

### Visual Example: Parameter Update Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     Training Iteration                        │
└──────────────────────────────────────────────────────────────┘

Step 1: Generate Response
┌─────────────┐
│   Prompt    │ → "What is Python?"
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LLM (Current)  │
│   Parameters    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Response      │ → "Python is a programming language..."
└──────┬──────────┘

Step 2: Get Reward Score
       │
       ▼
┌─────────────────┐
│  Reward Model   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Score: 0.75    │
└──────┬──────────┘

Step 3: Calculate Gradient
       │
       ▼
┌─────────────────────────────────────────┐
│ PPO Algorithm Calculates:               │
│ - Policy gradient (which way to adjust) │
│ - Clip gradient (prevent big jumps)     │
│ - Final update direction                │
└──────┬──────────────────────────────────┘

Step 4: Backpropagate & Update
       │
       ▼
┌─────────────────┐
│ Backpropagation │ → Updates weights layer by layer
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  LLM (Updated)  │ → Slightly better at generating
│   Parameters    │    high-reward responses
└─────────────────┘
```

### Does It Use Regular Backpropagation?

**Yes and No:**

**YES - The mechanism is the same:**
- Gradients flow backward through the network
- Weights are updated using gradient descent
- Each layer adjusts based on the gradient

**NO - The gradient source is different:**
- Regular training: Gradient comes from comparing to correct answer
- RLHF: Gradient comes from reward signal + policy gradient

### Example: Concrete Parameter Update

**Before Update:**
```
Prompt: "Explain AI"
LLM generates: "AI is computers" (score: 0.4 - too short)
```

**What Happens:**
1. Reward model gives low score (0.4)
2. PPO calculates: "This response got low reward, decrease probability of generating short responses"
3. Backpropagation adjusts weights:
   - Neurons that led to "short response" → weights decreased
   - Neurons that could lead to "detailed response" → weights increased
4. Next time, LLM is slightly more likely to generate longer, detailed responses

**After Many Updates:**
```
Prompt: "Explain AI"
LLM generates: "AI is artificial intelligence, where computers learn from data to make decisions..." (score: 0.9 - much better!)
```

### Key Concepts for Beginners

1. **Reward Score → Gradient**: The reward score tells us the direction to update parameters
   - High reward = move parameters in this direction
   - Low reward = move parameters away from this direction

2. **PPO Prevents Breaking**: Without PPO, one bad update could ruin the model
   - PPO clips updates to safe ranges
   - Model improves gradually over many iterations

3. **Backpropagation is the Tool**: 
   - Same technique as regular training
   - But the "error signal" comes from rewards, not labels

4. **Iterative Process**:
   - Generate → Score → Update → Repeat
   - Each iteration makes the model slightly better
   - After thousands of iterations, the model aligns with human preferences

### The Complete RLHF Training Loop

```
for iteration in range(num_iterations):
    # 1. Generate responses
    responses = LLM.generate(prompts)
    
    # 2. Get reward scores
    rewards = RewardModel.score(prompts, responses)
    
    # 3. Calculate PPO gradient
    gradient = PPO.calculate_gradient(responses, rewards)
    
    # 4. Clip gradient for safety
    clipped_gradient = PPO.clip(gradient)
    
    # 5. Backpropagate and update LLM
    LLM.backpropagate(clipped_gradient)
    LLM.update_parameters()
    
    # 6. Model is now slightly better!
```

---

## Bradley-Terry Formulation

### What is Bradley-Terry?

The Bradley-Terry model is a mathematical formula that helps us predict which response humans will prefer. Instead of just knowing "humans liked A more than B," it gives us actual probability scores that we can use to train our reward model. This makes the training process more precise and effective.

### The Bradley-Terry Formula

The formula converts human preferences into probabilities. It calculates: "What's the probability that response W (winner) is preferred over response L (loser)?" The answer depends on the reward scores assigned to each response.

`We will have winning and losing response pair for each prompt/question`

![alt text](image-9.png)

**Formula Variables Explained:**
- **P(y_w > y_l | x)**: The probability that the winning response (y_w) is preferred over the losing response (y_l) for a given prompt (x)
- **y_w**: The winning/chosen response that humans preferred
- **y_l**: The losing/rejected response that humans didn't prefer
- **x**: The input prompt given to the model
- **r(x, y_w)**: The reward score for the winning response (higher is better)
- **r(x, y_l)**: The reward score for the losing response (lower is worse)
- **exp()**: The exponential function that converts score differences into probabilities

### Training the Reward Model with Bradley-Terry

We use the Bradley-Terry formula as our loss function when training the reward model. The model learns to assign higher scores to preferred responses so that the probability P(y_w > y_l) becomes close to 1 (100% confidence in the human preference).

![alt text](image-10.png)

**Key Points:**
- The reward model outputs scores r(x, y) for any prompt-response pair
- Higher reward scores = better alignment with human preferences
- The sigmoid function (1 / (1 + exp(-difference))) squashes the score difference into a probability between 0 and 1
- During training, we maximize this probability for all human preference pairs in our dataset

### The Reward Model Architecture

This diagram shows the structure of the reward model and how it processes input to produce reward scores.

![alt text](image-11.png)

**Understanding the Reward Model:**

The reward model is essentially a language model that has been modified to output a single number (the reward score) instead of generating text. Here's how it works:

1. **Input**: The model receives the prompt (question) concatenated with a response
2. **Processing**: The text goes through transformer layers (same as a regular language model)
3. **Output**: Instead of predicting the next word, the final layer outputs a single reward score
4. **Training**: The model learns to give higher scores to preferred responses and lower scores to rejected ones

**Key Concept**: Think of it like converting a text generator into a text evaluator. The model uses its understanding of language to judge the quality of responses rather than generate them.

## How is the Reward Model Actually Built?

### Pairwise vs Pointwise Approach

The reward model is built using a **pairwise approach**, not pointwise. Here's why:

- **Pointwise**: Would train the model to predict an absolute score for each response (e.g., "this response is 7/10"). This is hard because humans struggle to assign consistent absolute scores.

- **Pairwise** (used in practice): Trains the model by comparing two responses at a time. The model learns: "response A is better than response B" rather than predicting exact scores. This is much easier and more consistent for humans.

### Building a Pairwise Reward Model - Step by Step

**Step 1: Start with a Language Model**
- Take a pre-trained or fine-tuned language model (like GPT)
- Replace its final layer with a single value output (the reward score)

**Step 2: Prepare Pairwise Training Data**

For each training example, you need:
- One prompt (x)
- One chosen/winning response (y_w)
- One rejected/losing response (y_l)

**Step 3: Training Process**
1. Pass the prompt + winning response through the model → get score r_w
2. Pass the prompt + losing response through the model → get score r_l
3. Calculate the probability using Bradley-Terry: P = 1/(1 + exp(r_l - r_w))
4. The model learns to make r_w > r_l by maximizing this probability

### Example: Pairwise Dataset Construction

Let's see a concrete example:

**Original Prompt:**
```
"Explain what photosynthesis is"
```

**Model generates 2 responses:**

**Response A (Chosen/Winner):**
```
"Photosynthesis is the process where plants convert sunlight into energy. 
Plants use chlorophyll to capture light, then combine CO2 and water to 
create glucose (food) and release oxygen. This is how plants make their 
own food and provide oxygen for us to breathe."
```

**Response B (Rejected/Loser):**
```
"Photosynthesis is when plants eat sunlight. It happens in leaves."
```

**Why Response A is better:**
- More detailed and accurate
- Explains the process clearly
- Mentions key components (chlorophyll, CO2, water, glucose)
- Helpful and informative

**The Pairwise Training Example:**
```
{
  "prompt": "Explain what photosynthesis is",
  "chosen": "Photosynthesis is the process where plants...",
  "rejected": "Photosynthesis is when plants eat sunlight..."
}
```

**During Training:**
1. Reward model processes (prompt + chosen) → outputs score: 0.8
2. Reward model processes (prompt + rejected) → outputs score: 0.3
3. Bradley-Terry probability: P = 1/(1 + exp(0.3 - 0.8)) = 0.62
4. Loss function pushes the model to make this probability closer to 1.0
5. Over many examples, the model learns to give higher scores to better responses

### Why Pairwise Works Better

1. **Easier for humans**: Choosing between two options is simpler than rating
2. **More consistent**: Humans agree more on comparisons than absolute scores
3. **Relative understanding**: The model learns "what makes one response better" rather than "what is a perfect response"
4. **Scalable**: You can collect many pairwise comparisons efficiently




