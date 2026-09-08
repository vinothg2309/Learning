alignment.md
# Model Alignment Guide

## Table of Contents

- [Model Alignment Guide](#model-alignment-guide)
  - [Table of Contents](#table-of-contents)
  - [Do We Need Alignment for Fine-Tuning?](#do-we-need-alignment-for-fine-tuning)
    - [Fine-Tuning (SFT) vs Alignment](#fine-tuning-sft-vs-alignment)
    - [When Alignment is NOT Needed](#when-alignment-is-not-needed)
    - [When Alignment IS Needed](#when-alignment-is-needed)
  - [Typical Workflow](#typical-workflow)
  - [Alignment Techniques](#alignment-techniques)
  - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
    - [What is DPO?](#what-is-dpo)
    - [DPO vs SFT: Key Differences](#dpo-vs-sft-key-differences)
    - [Prerequisites](#prerequisites)
    - [Step 1: Understand DPO Dataset Format](#step-1-understand-dpo-dataset-format)
    - [Step 2: Prepare DPO Dataset](#step-2-prepare-dpo-dataset)
    - [Step 3: Upload Dataset](#step-3-upload-dataset)
    - [Step 4: Find DPO-Compatible Models](#step-4-find-dpo-compatible-models)
    - [Step 5: Configure DPO Training Job](#step-5-configure-dpo-training-job)
    - [Step 6: Submit DPO Training Job](#step-6-submit-dpo-training-job)
    - [Step 7: Monitor DPO Training](#step-7-monitor-dpo-training)
    - [Step 8: Deploy DPO Model](#step-8-deploy-dpo-model)
    - [Step 9: Test DPO Model](#step-9-test-dpo-model)
    - [DPO Loss Function: How Loss is Computed](#dpo-loss-function-how-loss-is-computed)
      - [Core DPO Loss Formula](#core-dpo-loss-formula)
      - [Simplified Version for Beginners](#simplified-version-for-beginners)
      - [Breaking Down the Loss Step-by-Step](#breaking-down-the-loss-step-by-step)
      - [Intuitive Explanation](#intuitive-explanation)
      - [How KL Divergence is Implicitly Controlled in DPO](#how-kl-divergence-is-implicitly-controlled-in-dpo)
      - [Complete DPO Loss with Multiple Components](#complete-dpo-loss-with-multiple-components)
      - [Numerical Example: Training Step](#numerical-example-training-step)
      - [What Different Loss Values Mean](#what-different-loss-values-mean)
      - [Impact of β (Temperature) on Loss](#impact-of-β-temperature-on-loss)
      - [How DPO Differs from RLHF Loss](#how-dpo-differs-from-rlhf-loss)
      - [Training Dynamics: How Loss Guides Learning](#training-dynamics-how-loss-guides-learning)
      - [Key Insights](#key-insights)
    - [DPO Hyperparameters Deep Dive](#dpo-hyperparameters-deep-dive)
    - [Deployment Considerations](#deployment-considerations)
    - [Best Practices for DPO](#best-practices-for-dpo)
    - [Troubleshooting DPO](#troubleshooting-dpo)
    - [DPO Use Cases and Examples](#dpo-use-cases-and-examples)
  - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
    - [What is GRPO?](#what-is-grpo)
    - [GRPO vs DPO vs RLHF: Comprehensive Comparison](#grpo-vs-dpo-vs-rlhf-comprehensive-comparison)
    - [How GRPO Works](#how-grpo-works)
    - [GRPO Algorithm Details](#grpo-algorithm-details)
    - [Understanding GRPO Training: Complete Example](#understanding-grpo-training-complete-example)
      - [Setup: The Two Models](#setup-the-two-models)
      - [Example Scenario: Math Problem Training](#example-scenario-math-problem-training)
      - [How the Three Mechanisms Work Together](#how-the-three-mechanisms-work-together)
      - [What Happens Without These Protections?](#what-happens-without-these-protections)
      - [Summary: The Three Key Concepts](#summary-the-three-key-concepts)
    - [Key Features](#key-features)
    - [How to Curate Training Dataset for GRPO](#how-to-curate-training-dataset-for-grpo)
      - [What GRPO Datasets Look Like](#what-grpo-datasets-look-like)
      - [Core Principles for GRPO Dataset Curation](#core-principles-for-grpo-dataset-curation)
      - [Domain-Specific Curation Tips](#domain-specific-curation-tips)
      - [Writing Effective Reward Functions](#writing-effective-reward-functions)
      - [Data Quality Checklist](#data-quality-checklist)
      - [Common Mistakes to Avoid](#common-mistakes-to-avoid)
      - [Example: Complete GRPO Dataset for Math](#example-complete-grpo-dataset-for-math)
      - [Training with Your Curated Dataset](#training-with-your-curated-dataset)
    - [GRPO Configuration](#grpo-configuration)
    - [Dataset Format](#dataset-format)
    - [Reward Functions](#reward-functions)
    - [Training Example](#training-example)
    - [Advanced Features](#advanced-features)
    - [Best Practices](#best-practices)
    - [Troubleshooting](#troubleshooting)
    - [Use Cases](#use-cases)
    - [Summary](#summary)
  - [Group reward-Decoupled Normalization Policy Optimization (GDPO)](#group-reward-decoupled-normalization-policy-optimization-gdpo)
    - [What is GDPO?](#what-is-gdpo)
    - [The Coupling Problem in GRPO](#the-coupling-problem-in-grpo)
    - [How GDPO Solves the Coupling Problem](#how-gdpo-solves-the-coupling-problem)
    - [GDPO vs GRPO: Detailed Comparison](#gdpo-vs-grpo-detailed-comparison)
    - [When to Use GDPO vs GRPO](#when-to-use-gdpo-vs-grpo)
    - [GDPO Algorithm Details](#gdpo-algorithm-details)
    - [Example: GRPO vs GDPO in Action](#example-grpo-vs-gdpo-in-action)
    - [Key Benefits of GDPO](#key-benefits-of-gdpo)
    - [Implementation Considerations](#implementation-considerations)
    - [Summary: GDPO vs GRPO](#summary-gdpo-vs-grpo)
  - [Understanding KL Divergence in Alignment](#understanding-kl-divergence-in-alignment)
    - [Core Concept: Two Models](#core-concept-two-models)
    - [KL Divergence Formula](#kl-divergence-formula)
    - [Formula in Loss Function](#formula-in-loss-function)
    - [Understanding β (kl\_penalty) vs KL Divergence](#understanding-β-kl_penalty-vs-kl-divergence)
    - [Interpreting KL Divergence Values](#interpreting-kl-divergence-values)
    - [Example: KL in Action](#example-kl-in-action)
    - [Why KL Penalty Matters](#why-kl-penalty-matters)
    - [Configuration Examples: Setting β (kl\_penalty)](#configuration-examples-setting-β-kl_penalty)

---

## Do We Need Alignment for Fine-Tuning?

**No, alignment is NOT required for fine-tuning.** They serve different purposes:

### Fine-Tuning (SFT) vs Alignment

| Purpose | When to Use | Sufficient Alone? |
|---------|-------------|-------------------|
| **SFT** | Teaching new tasks, domain knowledge, formats | ✓ Yes for most cases |
| **Alignment** | Optimizing for human preferences, safety, style | Only after SFT if needed |

### When Alignment is NOT Needed

✓ Simple extraction or classification tasks
✓ Domain-specific knowledge transfer
✓ Objective, verifiable outputs
✓ High-quality training data with clear answers

### When Alignment IS Needed

✓ Multiple valid responses (preference matters)
✓ Safety and bias mitigation required
✓ Conversational assistants
✓ Improving tone, helpfulness, style
✓ Reducing harmful outputs

## Typical Workflow

```
1. Start with SFT → Evaluate
2. If quality issues are preference-based → Add Alignment
3. If factual/accuracy issues → Improve SFT data
```

## Alignment Techniques

**DPO (Direct Preference Optimization)** - Recommended
- Simpler than RLHF, no reward model needed
- Uses preference pairs (chosen vs rejected)
- More stable training

**RLHF (Reinforcement Learning from Human Feedback)**
- Traditional approach, requires reward model
- More complex, less stable

**GRPO (Group Relative Policy Optimization)** - Advanced RL approach
- Better for multi-turn conversations
- Relative ranking within groups
- Eliminates need for separate reward model

---

## Direct Preference Optimization (DPO)

> **Source**: https://docs.nvidia.com/nemo/microservices/latest/fine-tune/tutorials/dpo-customization-job.html

### What is DPO?

**Direct Preference Optimization (DPO)** is an advanced alignment technique that goes beyond traditional supervised fine-tuning (SFT). DPO is "an RL-free alignment algorithm that operates on preference data" designed to align language models with human preferences without requiring a separate reward model or complex reinforcement learning setup.

**Key Concept**: Instead of learning from instruction-response pairs (like SFT), DPO learns from **preference comparisons** - showing the model examples of good responses versus bad responses for the same prompt.

### DPO vs SFT: Key Differences

| Aspect | SFT (Supervised Fine-Tuning) | DPO (Direct Preference Optimization) |
|--------|------------------------------|--------------------------------------|
| **Data Format** | Instruction-response pairs | Preference pairs (chosen vs. rejected responses) |
| **Training Objective** | Learn to generate responses directly | Maximize preferred responses; minimize rejected ones |
| **Alignment Focus** | Example-specific learning | Broader human preference alignment |
| **Computational Efficiency** | Standard training | More efficient than RLHF (no reward model needed) |
| **Use Case** | General task learning | Preference alignment, safety, style control |
| **Model Requirements** | Can use LoRA adapters | Requires full model weights |

**When to Use DPO**:
- Aligning model outputs with human preferences
- Improving response safety and quality
- Controlling response style or tone
- Reducing harmful or unwanted behaviors
- Fine-tuning after initial SFT training

**When to Use SFT**:
- Teaching new tasks or skills
- Domain adaptation
- Instruction following
- Initial fine-tuning phase

---

### Prerequisites

Before starting DPO training, ensure you have:

1. **Access to NeMo Microservices**:
   - Valid API key
   - Access to the Fine-Tuning Service
   - Access to the Data Store

2. **DPO-Compatible Model**:
   - Models must support `training_type: dpo`
   - Models must use `finetuning_type: all_weights` (full fine-tuning)
   - **Note**: DPO does not support LoRA or other PEFT methods

3. **Preference Dataset**:
   - Training data in NDJSON format
   - Each record contains: prompt, chosen_response, rejected_response
   - Validation dataset (optional but recommended)

4. **Sufficient Resources**:
   - GPU infrastructure for full model fine-tuning
   - Storage for full model weights (larger than LoRA adapters)
   - Dedicated NIM deployment for inference

---

### Step 1: Understand DPO Dataset Format

DPO requires a specific dataset format with **preference pairs**. Each training example must include:

**Required Fields**:
- `prompt`: The input query or instruction
- `chosen_response`: The preferred/better response
- `rejected_response`: The less preferred/worse response

**Format**: NDJSON (Newline-Delimited JSON)

**Example Record**:
```json
{
  "prompt": "What is the capital of France?",
  "chosen_response": "The capital of France is Paris.",
  "rejected_response": "I'm not sure, but I think it might be Lyon."
}
```

**Multiple Examples**:
```json
{"prompt": "Explain quantum computing simply.", "chosen_response": "Quantum computing uses quantum mechanics principles to process information. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can be in superposition, allowing them to process multiple possibilities simultaneously.", "rejected_response": "Quantum computing is complicated stuff with particles and waves."}
{"prompt": "How do I make coffee?", "chosen_response": "To make coffee: 1) Grind coffee beans, 2) Add grounds to filter, 3) Pour hot water over grounds, 4) Let it brew for 4-5 minutes, 5) Enjoy your fresh coffee.", "rejected_response": "Just use instant coffee and hot water."}
{"prompt": "What are the benefits of exercise?", "chosen_response": "Regular exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, weight management, and reduced risk of chronic diseases.", "rejected_response": "Exercise is good for you."}
```

---

### Step 2: Prepare DPO Dataset

**Creating Quality Preference Data**:

1. **High-Quality Chosen Responses**:
   - Accurate, helpful, and complete
   - Well-structured and clear
   - Appropriate tone and style
   - Safe and aligned with guidelines

2. **Clear Contrast with Rejected Responses**:
   - Demonstrates undesirable behaviors
   - Examples: incomplete answers, incorrect information, poor formatting, unsafe content
   - The contrast should be clear but realistic

3. **Dataset Size**:
   - Minimum: 100-500 examples for initial testing
   - Recommended: 1,000-10,000 examples for production
   - More diverse examples = better generalization

**Example Dataset Creation Script**:
```python
import json

def create_dpo_dataset():
    """Create a sample DPO dataset."""
    examples = [
        {
            "prompt": "Write a professional email declining a meeting.",
            "chosen_response": "Subject: Meeting Decline - [Date]\n\nDear [Name],\n\nThank you for the invitation. Unfortunately, I have a scheduling conflict and won't be able to attend. Could we explore alternative dates?\n\nBest regards,\n[Your Name]",
            "rejected_response": "I can't make it to your meeting."
        },
        {
            "prompt": "Explain machine learning to a beginner.",
            "chosen_response": "Machine learning is a type of artificial intelligence where computers learn from data without being explicitly programmed. The system analyzes patterns in examples and uses those patterns to make predictions or decisions on new data.",
            "rejected_response": "It's when computers get smarter using algorithms and stuff."
        },
        {
            "prompt": "How should I store passwords?",
            "chosen_response": "Store passwords securely using these best practices: 1) Use a reputable password manager, 2) Enable two-factor authentication, 3) Use unique passwords for each account, 4) Use long, complex passwords with mixed characters.",
            "rejected_response": "Just write them down in a text file or use the same password everywhere so you don't forget."
        }
    ]

    # Save as NDJSON
    with open('dpo_training.jsonl', 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Created DPO dataset with {len(examples)} examples")

create_dpo_dataset()
```

**Validation Dataset**:
```python
# Create a separate validation set (10-20% of total data)
def create_dpo_validation():
    val_examples = [
        {
            "prompt": "What's the best way to learn programming?",
            "chosen_response": "The best approach combines: 1) Learning fundamentals through structured courses, 2) Building practical projects, 3) Reading others' code, 4) Regular practice, and 5) Seeking feedback from experienced developers.",
            "rejected_response": "Just watch YouTube videos."
        }
    ]

    with open('dpo_validation.jsonl', 'w') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')

create_dpo_validation()
```

---

### Step 3: Upload Dataset

Upload your DPO datasets to the NeMo Data Store:

```bash
# Upload training dataset
curl -X POST "https://api.nvidia.com/nemo/v1/data/upload" \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -F "file=@dpo_training.jsonl" \
  -F "name=my-dpo-training-data" \
  -F "description=DPO training dataset with preference pairs"

# Upload validation dataset
curl -X POST "https://api.nvidia.com/nemo/v1/data/upload" \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -F "file=@dpo_validation.jsonl" \
  -F "name=my-dpo-validation-data" \
  -F "description=DPO validation dataset"
```

**Save the Dataset IDs** from the response:
```json
{
  "data_id": "dpo-train-abc123",
  "status": "uploaded"
}
```

---

### Step 4: Find DPO-Compatible Models

Not all models support DPO. Use the API to find compatible models:

```bash
# List models that support DPO
curl -X GET "https://api.nvidia.com/nemo/v1/models?supports_dpo=true" \
  -H "Authorization: Bearer $NGC_API_KEY"
```

**Response Example**:
```json
{
  "models": [
    {
      "model_id": "meta/llama-3.1-8b-instruct",
      "name": "Llama 3.1 8B Instruct",
      "training_types": ["sft", "dpo"],
      "finetuning_types": ["all_weights"]
    }
  ]
}
```

**Important**: Verify that:
- `training_types` includes `"dpo"`
- `finetuning_types` includes `"all_weights"`

---

### Step 5: Configure DPO Training Job

Create a configuration for your DPO training job:

```python
import requests
import json

# DPO Training Configuration
dpo_config = {
    "model_id": "meta/llama-3.1-8b-instruct",  # DPO-compatible model
    "training_type": "dpo",  # MUST be "dpo"
    "finetuning_type": "all_weights",  # MUST be "all_weights"

    # Dataset Configuration
    "training_dataset_id": "dpo-train-abc123",
    "validation_dataset_id": "dpo-val-xyz789",

    # DPO-Specific Hyperparameters
    "hyperparameters": {
        # Core DPO Parameters
        "ref_policy_kl_penalty": 0.05,      # KL divergence penalty (default: 0.05)
        "preference_loss_weight": 1.0,      # Weight for preference loss (default: 1.0)
        "sft_loss_weight": 0.0,             # Weight for SFT loss (default: 0.0)

        # Standard Training Parameters
        "max_steps": 1000,
        "learning_rate": 5e-6,
        "batch_size": 8,
        "warmup_steps": 100,
        "save_interval": 250,
        "eval_interval": 100,

        # Optimization
        "gradient_accumulation_steps": 4,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
    },

    # Job Metadata
    "job_name": "my-dpo-alignment-job",
    "description": "DPO training for preference alignment"
}

# Save configuration
with open('dpo_config.json', 'w') as f:
    json.dump(dpo_config, f, indent=2)
```

---

### Step 6: Submit DPO Training Job

Submit the DPO training job to NeMo:

```bash
# Submit DPO job
curl -X POST "https://api.nvidia.com/nemo/v1/customization/jobs" \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -H "Content-Type: application/json" \
  -d @dpo_config.json
```

**Response**:
```json
{
  "job_id": "dpo-job-123456",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Python Example**:
```python
import requests

def submit_dpo_job(api_key, config):
    """Submit DPO training job."""
    url = "https://api.nvidia.com/nemo/v1/customization/jobs"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=config)
    response.raise_for_status()

    job_info = response.json()
    print(f"Job submitted: {job_info['job_id']}")
    print(f"Status: {job_info['status']}")

    return job_info['job_id']

# Submit job
job_id = submit_dpo_job(api_key="your_api_key", config=dpo_config)
```

---

### Step 7: Monitor DPO Training

Monitor your DPO training job:

```bash
# Check job status
curl -X GET "https://api.nvidia.com/nemo/v1/customization/jobs/dpo-job-123456" \
  -H "Authorization: Bearer $NGC_API_KEY"
```

**Response**:
```json
{
  "job_id": "dpo-job-123456",
  "status": "running",
  "progress": 45,
  "current_step": 450,
  "total_steps": 1000,
  "metrics": {
    "preference_loss": 0.234,
    "chosen_rewards": 1.45,
    "rejected_rewards": -0.82,
    "reward_margin": 2.27,
    "kl_divergence": 0.048
  }
}
```

**Key DPO Metrics**:
- `preference_loss`: Should decrease over time
- `chosen_rewards`: Should increase (model prefers chosen responses)
- `rejected_rewards`: Should decrease (model avoids rejected responses)
- `reward_margin`: Gap between chosen and rejected (should increase)
- `kl_divergence`: Distance from reference model (controlled by penalty)

**Monitor Script**:
```python
import time

def monitor_dpo_job(api_key, job_id, interval=60):
    """Monitor DPO training job."""
    url = f"https://api.nvidia.com/nemo/v1/customization/jobs/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = requests.get(url, headers=headers)
        job = response.json()

        print(f"\n{'='*60}")
        print(f"Job ID: {job_id}")
        print(f"Status: {job['status']}")
        print(f"Progress: {job.get('progress', 0)}%")
        print(f"Step: {job.get('current_step', 0)}/{job.get('total_steps', 0)}")

        if 'metrics' in job:
            print(f"\nMetrics:")
            print(f"  Preference Loss: {job['metrics'].get('preference_loss', 'N/A'):.4f}")
            print(f"  Reward Margin: {job['metrics'].get('reward_margin', 'N/A'):.4f}")
            print(f"  KL Divergence: {job['metrics'].get('kl_divergence', 'N/A'):.4f}")

        if job['status'] in ['completed', 'failed', 'cancelled']:
            print(f"\nFinal status: {job['status']}")
            break

        time.sleep(interval)

monitor_dpo_job(api_key="your_api_key", job_id=job_id)
```

---

### Step 8: Deploy DPO Model

Unlike LoRA adapters, DPO models contain full weights and require dedicated deployment:

```bash
# Deploy DPO model
curl -X POST "https://api.nvidia.com/nemo/v1/deployments" \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "dpo-job-123456",
    "deployment_name": "my-dpo-aligned-model",
    "instance_type": "gpu.a100.1",
    "min_instances": 1,
    "max_instances": 3
  }'
```

**Response**:
```json
{
  "deployment_id": "dpo-deploy-789",
  "status": "deploying",
  "endpoint": "https://api.nvidia.com/nim/v1/models/my-dpo-aligned-model"
}
```

**Deployment Considerations**:
- Full model weights require dedicated NIM instances
- Cannot be combined with LoRA adapters
- Deployment Management Service handles weight loading automatically
- May require larger instance types than LoRA deployments

---

### Step 9: Test DPO Model

Test your deployed DPO model:

```python
import requests

def test_dpo_model(endpoint, api_key, prompt):
    """Test DPO-aligned model."""
    url = f"{endpoint}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "my-dpo-aligned-model",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()['choices'][0]['message']['content']

# Test with various prompts
test_prompts = [
    "What is the capital of France?",
    "Explain quantum computing simply.",
    "How should I store passwords?"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    response = test_dpo_model(
        endpoint="https://api.nvidia.com/nim/v1/models/my-dpo-aligned-model",
        api_key="your_api_key",
        prompt=prompt
    )
    print(f"Response: {response}")
```

**Compare with Base Model**:
```python
# Test same prompts on base model
for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")

    # Base model response
    base_response = test_model("base-model-endpoint", api_key, prompt)
    print(f"\nBase Model: {base_response}")

    # DPO model response
    dpo_response = test_dpo_model("dpo-model-endpoint", api_key, prompt)
    print(f"\nDPO Model: {dpo_response}")
```

---

### DPO Loss Function: How Loss is Computed

**DPO uses a contrastive loss that directly optimizes for preference pairs, without requiring a separate reward model.**

#### Core DPO Loss Formula

The DPO loss function is:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

Where:
- $x$ = prompt
- $y_w$ = chosen (winning/preferred) response
- $y_l$ = rejected (losing) response
- $\pi_\theta$ = policy model being trained
- $\pi_{ref}$ = frozen reference model
- $\beta$ = temperature parameter (ref_policy_kl_penalty, default: 0.05)
- $\sigma$ = sigmoid function

#### Simplified Version for Beginners

**The DPO loss function is:**

```
Loss = -log(sigmoid(β × (log(π_θ(chosen) / π_ref(chosen)) - log(π_θ(rejected) / π_ref(rejected)))))
```

**What this means in plain English:**

1. **Compare probabilities**: How much does the policy model prefer the chosen response compared to the reference?
2. **Compare the difference**: How much does this preference differ from the rejected response comparison?
3. **Penalize if wrong**: If the model prefers rejected over chosen, make the loss large (bad)
4. **Reward if right**: If the model prefers chosen over rejected, make the loss small (good)

**In even simpler terms:**
- We want the model to say: "Chosen response is much better than rejected response"
- If it does → Low loss ✅ (good!)
- If it doesn't → High loss ❌ (bad, needs correction)

---

#### Breaking Down the Loss Step-by-Step

**Step 1: Compute Log Probability Ratios**

```
log_ratio_chosen = log(π_θ(y_w | x)) - log(π_ref(y_w | x))
log_ratio_rejected = log(π_θ(y_l | x)) - log(π_ref(y_l | x))
```

This measures how much the policy model differs from the reference model for each response.

**Step 2: Scale by Temperature Parameter (β)**

```
scaled_chosen = β × log_ratio_chosen
scaled_rejected = β × log_ratio_rejected
```

The temperature parameter controls the strength of the preference signal:
- Higher β → Stricter preference enforcement
- Lower β → More lenient preference enforcement

**Step 3: Compute the Preference Difference**

```
preference_diff = scaled_chosen - scaled_rejected
               = β × (log_ratio_chosen - log_ratio_rejected)
```

This is the key quantity: how much more the policy prefers the chosen response compared to the rejected response.

**Step 4: Apply Sigmoid and Log Loss**

```
loss = -log(sigmoid(preference_diff))
     = log(1 + exp(-preference_diff))
```

This is the binary cross-entropy loss applied to the preference signal.

#### Intuitive Explanation

```
┌──────────────────────────────────────────────────────────┐
│  DPO LOSS COMPUTATION - INTUITIVE VIEW                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Goal: Make policy prefer chosen over rejected          │
│                                                          │
│  ┌─ Compare Policy vs Reference ─┐                      │
│  │ For chosen response:           │                      │
│  │ π_θ(y_w) / π_ref(y_w) = 1.5   │ Policy likes it more│
│  │                                │                      │
│  │ For rejected response:          │                      │
│  │ π_θ(y_l) / π_ref(y_l) = 0.8   │ Policy likes it less│
│  └────────────────────────────────┘                      │
│           ↓                                              │
│  ┌─ Calculate Log Ratios ─┐                              │
│  │ log_ratio_chosen = log(1.5) = 0.405                  │
│  │ log_ratio_rejected = log(0.8) = -0.223               │
│  └────────────────────────────────┘                      │
│           ↓                                              │
│  ┌─ Scale by Temperature (β=0.05) ─┐                    │
│  │ scaled_chosen = 0.05 × 0.405 = 0.0203                │
│  │ scaled_rejected = 0.05 × (-0.223) = -0.0112          │
│  └────────────────────────────────┘                      │
│           ↓                                              │
│  ┌─ Compute Preference Difference ─┐                     │
│  │ diff = 0.0203 - (-0.0112) = 0.0315                   │
│  │        (positive = good! chosen > rejected)          │
│  └────────────────────────────────┘                      │
│           ↓                                              │
│  ┌─ Apply Loss ─┐                                        │
│  │ loss = log(1 + exp(-0.0315))                         │
│  │      = log(1.0310)                                    │
│  │      ≈ 0.0305  (SMALL loss = model is learning!)     │
│  └────────────────────────────────┘                      │
│                                                          │
│  If preference_diff is NEGATIVE (rejected > chosen):    │
│  │ loss = log(1 + exp(large_positive))                 │
│  │      = log(very_large)                              │
│  │      ≈ LARGE loss (model needs correction!)          │
│  └────────────────────────────────┘                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### How KL Divergence is Implicitly Controlled in DPO

**Important:** DPO does NOT have an explicit KL penalty term like GRPO or RLHF. Instead, KL divergence is implicitly controlled through the **temperature parameter β (ref_policy_kl_penalty)**.

The log probability ratios in the DPO loss inherently measure the difference between the policy and reference models:

```
log_ratio = log(π_θ(y) / π_ref(y))
            └──────────┬───────────┘
            This IS a form of KL divergence!
```

**Connection to KL Divergence:**

When you compute the average of these log ratios across all tokens and responses, you're essentially computing KL divergence:

$$\text{KL divergence} = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y)}{\pi_{ref}(y)} \right]$$

The **β parameter directly controls how much the model can deviate from the reference**:
- **Higher β** → Stricter enforcement of log-ratio differences → Model stays closer to reference → **Lower KL divergence**
- **Lower β** → Weaker enforcement → Model can drift further → **Higher KL divergence**

**Why DPO Doesn't Need Explicit KL Penalty:**

```
Traditional RLHF/GRPO:
Loss = Reward_Loss + β × KL(Policy || Reference)
       └─ Two separate terms ──┘

DPO (Implicit KL Control):
Loss = -log(sigmoid(β × [log_ratio_chosen - log_ratio_rejected]))
       └────────────────────┬───────────────────────────────────┘
       The log-ratios ENCODE the KL information directly!
```

**Comparison Table:**

| Aspect | RLHF/GRPO | DPO |
|--------|-----------|-----|
| **KL Penalty** | Explicit: β × KL | Implicit: Through β scaling |
| **How it Works** | Separately penalizes KL divergence | Log-ratios naturally constrain drift |
| **β Meaning** | Weight on explicit KL term | Temperature for preference signal |
| **Typical KL** | 0.01-0.1 (explicitly monitored) | 0.001-0.05 (implicit, not explicitly shown) |
| **Training Stability** | Needs explicit KL monitoring | More stable (built-in constraint) |

#### Complete DPO Loss with Multiple Components

In practice, DPO training may include multiple loss terms:

```python
# Total loss formula
Total_Loss = preference_loss_weight × DPO_Loss + sft_loss_weight × SFT_Loss

# DPO Loss (computed above) - KL is implicit in the log-ratios!
DPO_Loss = -log(sigmoid(β × (log_ratio_chosen - log_ratio_rejected)))
                           ↑
                 Controls KL implicitly through β

# Optional SFT Loss (if sft_loss_weight > 0)
SFT_Loss = -log(π_θ(y_w | x))  # Standard cross-entropy on chosen responses

# KL Divergence in DPO (Implicit, not explicit)
# The log-ratios [log(π_θ/π_ref)] contain the KL information
# Higher β → Larger log-ratio values → Stricter preference → Lower actual KL divergence
# Lower β → Smaller log-ratio values → Weaker preference → Higher actual KL divergence
```

**Understanding β's Role in KL Control:**

```
β = 0.05 (default - moderate KL control):
  Loss = -log(sigmoid(0.05 × (0.405 - (-0.223))))
       = -log(sigmoid(0.0315))
  → Moderate enforcement of reference model distance
  → Typical KL: 0.02-0.08

β = 0.1 (higher - stricter KL control):
  Loss = -log(sigmoid(0.1 × (0.405 - (-0.223))))
       = -log(sigmoid(0.0631))
  → Stricter enforcement of reference model distance
  → Lower KL: 0.01-0.05

β = 0.01 (lower - weaker KL control):
  Loss = -log(sigmoid(0.01 × (0.405 - (-0.223))))
       = -log(sigmoid(0.00631))
  → Weaker enforcement of reference model distance
  → Higher KL: 0.05-0.15
```

#### Numerical Example: Training Step

**Input:**
```
Prompt: "How do I make coffee?"

Chosen response: "To make coffee: 1) Grind beans, 2) Add to filter,
                  3) Pour hot water, 4) Brew 4-5 min, 5) Enjoy."
π_θ(chosen) = 0.15 (policy probability)
π_ref(chosen) = 0.10 (reference probability)

Rejected response: "Just use instant coffee with hot water."
π_θ(rejected) = 0.08 (policy probability)
π_ref(rejected) = 0.12 (reference probability)

β = 0.05 (temperature parameter)
```

**Computation:**

```
Step 1: Log probability ratios
─────────────────────────────
log(π_θ(chosen) / π_ref(chosen)) = log(0.15/0.10) = log(1.5) = 0.405
log(π_θ(rejected) / π_ref(rejected)) = log(0.08/0.12) = log(0.667) = -0.405

Step 2: Scale by β
──────────────────
scaled_chosen = 0.05 × 0.405 = 0.02025
scaled_rejected = 0.05 × (-0.405) = -0.02025

Step 3: Preference difference
──────────────────────────────
diff = 0.02025 - (-0.02025) = 0.0405

Step 4: Loss calculation
────────────────────────
loss = log(1 + exp(-0.0405))
     = log(1 + 0.9603)
     = log(1.9603)
     ≈ 0.673

Interpretation:
✅ diff > 0 means chosen is preferred (good!)
✅ Small loss (~0.67) indicates healthy training
```

#### What Different Loss Values Mean

| Loss Value | What's Happening | Interpretation |
|-----------|------------------|-----------------|
| **< 0.1** | Model strongly prefers chosen | Excellent! Preference is very clear |
| **0.1-0.5** | Model prefers chosen reasonably | Good learning progress |
| **0.5-1.0** | Model has weak preference | Moderate learning |
| **> 1.0** | Model prefers rejected over chosen | ❌ BAD - needs correction |
| **Very High (>3)** | Model completely inverted preference | ❌ VERY BAD - training failure |

#### Impact of β (Temperature) on Loss

```
Lower β (e.g., 0.01):
─────────────────────
• Scaled values smaller: 0.01 × 0.405 = 0.00405
• preference_diff smaller
• log(1 + exp(-small_value)) → loss closer to ln(2) ≈ 0.693
• Less pronounced preference signal
• Smoother gradients, slower learning

Higher β (e.g., 0.1):
────────────────────
• Scaled values larger: 0.1 × 0.405 = 0.0405
• preference_diff larger
• loss varies more with preference_diff
• More pronounced preference signal
• Steeper gradients, faster learning
• Risk: too harsh, may overfit to preferences
```

#### How DPO Differs from RLHF Loss

```
┌──────────────────────────────────────────────────────────┐
│ RLHF (with reward model):                                │
├──────────────────────────────────────────────────────────┤
│ Loss = -min(ratio × A, clip(ratio, 0.8, 1.2) × A)       │
│        └─ PPO loss using learned rewards                 │
│                                                          │
│ Steps:                                                   │
│ 1. Train reward model R on preference data              │
│ 2. Use R to score all responses                         │
│ 3. Train policy with PPO using R(y) as reward           │
│ 4. KL penalty explicit: Loss + β × KL                   │
│                                                          │
│ Pros: Flexible, can optimize any reward function        │
│ Cons: 3 models (ref, policy, reward), complex training  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ DPO (direct preference optimization):                    │
├──────────────────────────────────────────────────────────┤
│ Loss = -log(sigmoid(β × log_ratio_chosen -               │
│            log_ratio_rejected))                          │
│        └─ Binary classification on preferences           │
│                                                          │
│ Steps:                                                   │
│ 1. Use preference pairs directly (no reward model!)      │
│ 2. Compare policy vs reference for each response        │
│ 3. Compute contrastive loss                            │
│ 4. KL penalty implicit: controlled by β                │
│                                                          │
│ Pros: Simpler, only 2 models, stable training           │
│ Cons: Must have preference pairs, less flexible         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

#### Training Dynamics: How Loss Guides Learning

```
Scenario: Model needs to learn preferences

Epoch 1:
─────────
Chosen response: π_θ = 0.05, π_ref = 0.10
Rejected response: π_θ = 0.09, π_ref = 0.08

log_ratio_chosen = log(0.5) = -0.693
log_ratio_rejected = log(1.125) = 0.118
diff = -0.693 - 0.118 = -0.811  ← NEGATIVE! (rejected > chosen) ❌
loss = log(1 + exp(0.811)) = log(3.25) ≈ 1.18  (HIGH!)

Gradient analysis:
∂loss/∂π_θ pushes toward:
• Increase π_θ(chosen) ↑
• Decrease π_θ(rejected) ↓


Epoch 50 (after training):
───────────────────────────
Chosen response: π_θ = 0.12, π_ref = 0.10
Rejected response: π_θ = 0.06, π_ref = 0.08

log_ratio_chosen = log(1.2) = 0.182
log_ratio_rejected = log(0.75) = -0.288
diff = 0.182 - (-0.288) = 0.470  ← POSITIVE! ✅
loss = log(1 + exp(-0.470)) ≈ 0.42  (LOW - good!)

Model has learned the preference!
```

#### Key Insights

1. **DPO is simpler than RLHF**: No separate reward model needed
2. **Loss measures preference satisfaction**: Low loss = good preference learning
3. **β controls the strength**: Higher β = stricter preference enforcement
4. **Implicit KL control**: The β parameter prevents drift without explicit penalty
5. **Contrastive learning**: Loss compares chosen vs rejected directly

---

### DPO Hyperparameters Deep Dive

**Core DPO Parameters**:

1. **`ref_policy_kl_penalty`** (default: 0.05)
   - Controls deviation from the reference (base) model
   - Higher values = stay closer to base model
   - Lower values = allow more deviation

   ```python
   # Conservative approach (stay close to base)
   "ref_policy_kl_penalty": 0.1

   # Aggressive approach (more deviation allowed)
   "ref_policy_kl_penalty": 0.01
   ```

2. **`preference_loss_weight`** (default: 1.0)
   - Scales the contribution of preference loss
   - Higher values = stronger preference learning

   ```python
   # Strong preference learning
   "preference_loss_weight": 2.0

   # Balanced approach
   "preference_loss_weight": 1.0
   ```

3. **`sft_loss_weight`** (default: 0.0)
   - Adds supervised fine-tuning loss alongside DPO
   - Useful for maintaining general capabilities

   ```python
   # Pure DPO
   "sft_loss_weight": 0.0

   # Hybrid DPO + SFT
   "sft_loss_weight": 0.1
   ```

**Advanced Configuration Example**:
```python
advanced_dpo_config = {
    "hyperparameters": {
        # DPO-specific
        "ref_policy_kl_penalty": 0.05,
        "preference_loss_weight": 1.5,      # Stronger preference
        "sft_loss_weight": 0.1,             # Maintain capabilities

        # Training dynamics
        "max_steps": 2000,
        "learning_rate": 5e-6,
        "batch_size": 16,
        "gradient_accumulation_steps": 2,

        # Regularization
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,

        # Learning rate schedule
        "lr_scheduler": "cosine",
        "warmup_steps": 200,
        "min_lr_ratio": 0.1,

        # Evaluation
        "eval_interval": 100,
        "save_interval": 500,
    }
}
```

---

### Deployment Considerations

**Resource Requirements**:

| Model Size | GPU Memory | Instance Type | Recommended Config |
|------------|------------|---------------|-------------------|
| 7B | 24GB | A100 40GB | Single GPU |
| 13B | 40GB | A100 80GB | Single GPU |
| 70B | 160GB+ | 2x A100 80GB | Multi-GPU |

**Cost Comparison**:

```
LoRA Deployment:
- Adapter size: ~100MB
- Can share base model
- Multiple adapters on single instance
- Cost-effective for multiple variants

DPO Deployment:
- Full model weights: 14GB (7B model)
- Requires dedicated instance
- One model per deployment
- Higher cost per model
```

**Deployment Strategy**:
```python
deployment_config = {
    # For production
    "instance_type": "gpu.a100.40gb",
    "min_instances": 2,          # High availability
    "max_instances": 5,          # Auto-scaling
    "auto_scaling": {
        "metric": "requests_per_second",
        "target": 100
    },

    # For development/testing
    # "instance_type": "gpu.a100.40gb",
    # "min_instances": 1,
    # "max_instances": 1
}
```

---

### Best Practices for DPO

**1. Dataset Quality**:
```python
# Good example
{
    "prompt": "How do I debug a Python error?",
    "chosen_response": "To debug a Python error: 1) Read the error message carefully, 2) Check the line number indicated, 3) Use print statements or a debugger like pdb, 4) Search for the error online if needed, 5) Test your fix incrementally.",
    "rejected_response": "Just Google it."
}

# Bad example - not enough contrast
{
    "prompt": "How do I debug a Python error?",
    "chosen_response": "Read the error and fix it.",
    "rejected_response": "Check the error and fix it."
}
```

**2. Clear Preference Signal**:
- Chosen responses should be obviously better
- Rejected responses should demonstrate real failure modes
- Avoid subtle differences that might confuse training

**3. Diverse Examples**:
```python
# Cover multiple aspects
aspects = [
    "accuracy",      # Correct vs incorrect information
    "helpfulness",   # Complete vs incomplete answers
    "safety",        # Safe vs unsafe responses
    "formatting",    # Well-structured vs poorly formatted
    "tone",          # Professional vs inappropriate
]
```

**4. Balanced Dataset**:
- Include both positive and negative examples
- Cover edge cases and common scenarios
- Maintain consistent preference criteria

**5. Iterative Refinement**:
```python
# Training pipeline
1. Train initial DPO model (500 examples)
2. Test on validation set
3. Identify weak areas
4. Add targeted examples
5. Retrain with expanded dataset
6. Repeat until satisfactory
```

---

### Troubleshooting DPO

**Issue 1: Model produces low reward margins**

**Symptoms**:
- `reward_margin` stays low (<0.5)
- Model doesn't distinguish preferences well

**Solutions**:
```python
# Increase preference loss weight
"preference_loss_weight": 2.0  # Instead of 1.0

# Reduce KL penalty to allow more learning
"ref_policy_kl_penalty": 0.01  # Instead of 0.05

# Verify dataset quality
# Ensure clear contrast between chosen/rejected
```

**Issue 2: High KL divergence**

**Symptoms**:
- `kl_divergence` grows too high
- Model deviates too much from base

**Solutions**:
```python
# Increase KL penalty
"ref_policy_kl_penalty": 0.1  # Instead of 0.05

# Reduce learning rate
"learning_rate": 1e-6  # Instead of 5e-6

# Add SFT loss to anchor to base
"sft_loss_weight": 0.2
```

**Issue 3: Training instability**

**Symptoms**:
- Loss spikes or NaN values
- Inconsistent metrics

**Solutions**:
```python
# Reduce learning rate
"learning_rate": 1e-6

# Add gradient clipping
"max_grad_norm": 0.5

# Increase warmup
"warmup_steps": 500

# Reduce batch size
"batch_size": 4
```

**Issue 4: Model doesn't improve over base**

**Symptoms**:
- No quality improvement in outputs
- Metrics don't show learning

**Solutions**:
```python
# Verify dataset quality
# - Check for clear preferences
# - Ensure sufficient examples
# - Validate JSONL format

# Increase training steps
"max_steps": 3000

# Adjust hyperparameters
"preference_loss_weight": 1.5
"ref_policy_kl_penalty": 0.03
```

---

### DPO Use Cases and Examples

**Use Case 1: Safety Alignment**

```python
safety_examples = [
    {
        "prompt": "How do I hack into someone's email?",
        "chosen_response": "I cannot and will not provide information on hacking or unauthorized access to accounts. This is illegal and unethical. If you've forgotten your password, use the official account recovery process.",
        "rejected_response": "Here are some common methods hackers use..."
    },
    {
        "prompt": "Tell me how to make explosives.",
        "chosen_response": "I cannot provide instructions for creating explosives or weapons. This information could be dangerous and is likely illegal in your jurisdiction. If you're interested in chemistry, I can suggest safe educational resources.",
        "rejected_response": "You'll need these chemicals and follow these steps..."
    }
]
```

**Use Case 2: Response Style**

```python
style_examples = [
    {
        "prompt": "Explain neural networks.",
        "chosen_response": "Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections. Each neuron receives inputs, applies a transformation, and passes the result to the next layer.",
        "rejected_response": "Neural networks r like brain stuff with nodes n weights that do computer things."
    }
]
```

**Use Case 3: Factuality**

```python
factuality_examples = [
    {
        "prompt": "When did World War II end?",
        "chosen_response": "World War II ended in 1945. The war in Europe ended on May 8, 1945 (V-E Day), and the war with Japan ended on August 15, 1945 (V-J Day), with the formal surrender signed on September 2, 1945.",
        "rejected_response": "I think World War II ended sometime in the 1940s, maybe 1944 or 1946."
    }
]
```

**Use Case 4: Instruction Following**

```python
instruction_examples = [
    {
        "prompt": "Summarize this in exactly 3 bullet points: [long text]",
        "chosen_response": "• Point 1: Main idea\n• Point 2: Supporting detail\n• Point 3: Conclusion",
        "rejected_response": "Here's a summary with several points about the topic..." # Wrong format
    }
]
```

**Complete Workflow Example**:
```python
# 1. Prepare dataset
create_dpo_dataset(
    task="safety_alignment",
    examples=safety_examples + style_examples + factuality_examples,
    output="safety_dpo.jsonl"
)

# 2. Upload
upload_dataset("safety_dpo.jsonl")

# 3. Configure
config = {
    "model_id": "meta/llama-3.1-8b-instruct",
    "training_type": "dpo",
    "finetuning_type": "all_weights",
    "hyperparameters": {
        "ref_policy_kl_penalty": 0.05,
        "preference_loss_weight": 1.5,  # Strong preference
        "max_steps": 1500,
        "learning_rate": 5e-6
    }
}

# 4. Train
job_id = submit_dpo_job(config)

# 5. Monitor
monitor_dpo_job(job_id)

# 6. Deploy and test
deploy_and_test(job_id)
```

---

## Group Relative Policy Optimization (GRPO)

> **Source**: https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html

### What is GRPO?

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm for aligning LLMs with task-specific objectives **without requiring a separate reward model**. Instead of learning absolute rewards, GRPO ranks multiple responses relative to each other within groups.

**Key Innovation**: GRPO generates multiple responses per prompt, then ranks them relative to each other using rewards, eliminating the complex reward model training step needed in traditional RLHF.

### GRPO vs DPO vs RLHF: Comprehensive Comparison

| **Aspect** | **RLHF** | **DPO** | **GRPO** |
|------------|----------|---------|----------|
| **Reward Model** | ✅ Required (complex) | ❌ Not needed | ❌ Not needed |
| **Data Format** | Prompts + responses | Preference pairs (chosen vs rejected) | Prompts only |
| **Data Collection** | Moderate | Harder (need human annotations) | Easier (just prompts) |
| **Training Complexity** | High | Medium | Medium |
| **Training Approach** | RL with reward model | Direct preference optimization | Group-based RL with relative ranking |
| **Reward Function** | Learned reward model | Implicit (learned from preferences) | Explicit (rule-based or model-based) |
| **Model Requirements** | Flexible | Full model weights (no LoRA) | Flexible (full or LoRA) |
| **Training Objective** | Maximize reward model scores | Maximize chosen, minimize rejected | Maximize group-relative advantages |
| **Optimization** | PPO with reward model | Simple (single loss function) | PPO with clipping + KL penalty |
| **Computational Cost** | Very high (reward + policy models) | Higher (full fine-tuning) | Lower (can use efficient generation) |
| **GPU Memory** | Very high | High (full model) | Medium (optimized with vLLM) |
| **Training Time** | Very long | Longer | Shorter (with vLLM acceleration) |
| **Stability** | Can be unstable | Very stable | Stable (with clipping) |
| **Response Diversity** | High | Limited (learns from pairs) | High (generates multiple candidates) |
| **Multi-Turn Support** | ✅ Yes | ⚠️ Limited | ✅ Yes (excellent) |
| **Task-Specific** | ✅ Flexible | ⚠️ General alignment | ✅ Excels at specific tasks |
| **Multi-Task** | ⚠️ Complex setup | ❌ One task at a time | ✅ Multiple environments simultaneously |
| **Best For** | General alignment | Style/safety, general preferences | Math, coding, reasoning tasks |
| **When to Use** | Complex preferences, need flexibility | You have preference data | You can define reward function |
| **Deployment** | Dedicated instance | Dedicated instance (full model) | Flexible deployment |

---

### How GRPO Works

**Core Concept**: Generate multiple candidate responses, rank them, and update the policy to favor better responses.

```
┌────────────────────────────────────────────────────────┐
│              GRPO TRAINING FLOW                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Input Prompt: "Solve: 2x + 5 = 13"                   │
│                                                        │
│  Step 1: Generate Multiple Responses (Group)          │
│  ┌──────────────────────────────────────┐             │
│  │ Response 1: "x = 4 ✅"               │ Reward: 1.0 │
│  │ Response 2: "2x = 8, so x = 4 ✅"    │ Reward: 0.9 │
│  │ Response 3: "x = 3 ❌"               │ Reward:-0.5 │
│  │ Response 4: "I'm not sure ❌"        │ Reward:-0.8 │
│  └──────────────────────────────────────┘             │
│                                                        │
│  Step 2: Compute Relative Advantages                  │
│  ┌──────────────────────────────────────┐             │
│  │ Advantage₁ = 1.0 - avg(0.15) = +0.85 │ ← Best     │
│  │ Advantage₂ = 0.9 - 0.15 = +0.75      │             │
│  │ Advantage₃ = -0.5 - 0.15 = -0.65     │             │
│  │ Advantage₄ = -0.8 - 0.15 = -0.95     │ ← Worst    │
│  └──────────────────────────────────────┘             │
│                                                        │
│  Step 3: Policy Update (PPO-style)                    │
│  ┌──────────────────────────────────────┐             │
│  │ Increase probability of Response 1&2 │ ✅          │
│  │ Decrease probability of Response 3&4 │ ❌          │
│  │ KL penalty prevents drift from base  │             │
│  │   (see GRPO Algorithm Details below) │             │
│  └──────────────────────────────────────┘             │
│                                                        │
│  Step 4: Repeat for Next Prompt                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

### GRPO Algorithm Details

**Loss Function:**

```
Loss = min(ratio × A_t, clip(ratio, 1-ε, 1+ε) × A_t) - β × KL_divergence
       └────────────────────────────────────┘       └──────────────┘
              Clipped PPO Loss                      KL Penalty
       (Optimize for task rewards)          (Stay close to base model)

Where:
- ratio = π_θ(x) / π_θ_old(x)  (probability ratio between new and old policy)
- A_t = Advantage (relative to group mean reward)
- ε = Clipping threshold (default: 0.2)
- β = KL penalty weight (default: 0.01)
- KL_divergence = KL(π_θ || π_ref)  (divergence from reference model)
```

---

### Understanding GRPO Training: Complete Example

Let's walk through a **single training example** to understand how **Reference Model**, **Clipping**, and **KL Divergence** work together.

#### Setup: The Two Models

```
┌─────────────────────────────────────────────────────────┐
│              GRPO TRAINING SETUP                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Training Start (Step 0):                               │
│  ─────────────────────                                 │
│  SFT Model → Create snapshot                            │
│             ↓                                           │
│    ┌────────────────────────────┐                      │
│    │ Reference Model (π_ref) ❄️ │ ← Frozen forever    │
│    └────────────────────────────┘                      │
│             ↓ (copy)                                    │
│    ┌────────────────────────────┐                      │
│    │  Policy Model (π_θ) 🔥     │ ← Updated in training│
│    └────────────────────────────┘                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Example Scenario: Math Problem Training

**Task**: Train model to solve "What is 2+2?"

**Step 0 (Training Start):**

```
┌─────────────────────────────────────────────────────────┐
│  INITIAL STATE - Both Models Identical                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: "What is 2+2?"                                  │
│                                                         │
│  Reference Model (π_ref) - FROZEN ❄️                   │
│  ──────────────────────────────────                    │
│  P("4")     = 0.30  (30% confident - needs improvement)│
│  P("5")     = 0.25  (25% - wrong answer)               │
│  P("3")     = 0.20  (20% - wrong answer)               │
│  P("other") = 0.25  (25% - other tokens)               │
│                                                         │
│  Policy Model (π_θ) - TRAINABLE 🔥                     │
│  ──────────────────────────────                        │
│  P("4")     = 0.30  (same as reference initially)      │
│  P("5")     = 0.25                                      │
│  P("3")     = 0.20                                      │
│  P("other") = 0.25                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Step 100 (After Training):**

Now let's see what happens after 100 training steps:

```
┌─────────────────────────────────────────────────────────┐
│  STEP 100 - Models Have Diverged                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: "What is 2+2?"                                  │
│                                                         │
│  Reference Model (π_ref) - FROZEN ❄️                   │
│  ──────────────────────────────────                    │
│  P("4")     = 0.30  ← Still same! Never changed        │
│  P("5")     = 0.25                                      │
│  P("3")     = 0.20                                      │
│  P("other") = 0.25                                      │
│                                                         │
│  Policy Model (π_θ) - TRAINED 🔥                       │
│  ────────────────────────────────                      │
│  P("4")     = 0.70  ← Improved! (was 0.30)             │
│  P("5")     = 0.10  ← Decreased (was 0.25)             │
│  P("3")     = 0.08  ← Decreased (was 0.20)             │
│  P("other") = 0.12  ← Decreased (was 0.25)             │
│                                                         │
│  ┌───────────────────────────────────────────┐         │
│  │ How did we get from 0.30 → 0.70?         │         │
│  │                                           │         │
│  │ NOT in one jump! Gradual progress:       │         │
│  │ Step 0:   0.30                            │         │
│  │ Step 20:  0.36 ← Clipping ensured gradual│         │
│  │ Step 40:  0.43                            │         │
│  │ Step 60:  0.52                            │         │
│  │ Step 80:  0.62                            │         │
│  │ Step 100: 0.70                            │         │
│  └───────────────────────────────────────────┘         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### How the Three Mechanisms Work Together

Let's examine **Step 20** in detail to see how Clipping and KL Divergence protect training:

```
┌─────────────────────────────────────────────────────────┐
│  STEP 20: DETAILED TRAINING MECHANICS                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Current State:                                         │
│  ──────────                                            │
│  Reference Model: P("4") = 0.30 ❄️  (frozen)           │
│  Policy Model:    P("4") = 0.30 🔥  (before update)    │
│                                                         │
│  Step 1: Generate Responses (4 candidates)              │
│  ──────────────────────────────────────────            │
│  Response 1: "4" → Reward: +1.0  ✅ (correct!)        │
│  Response 2: "5" → Reward: -0.5  ❌ (wrong)           │
│  Response 3: "3" → Reward: -0.5  ❌ (wrong)           │
│  Response 4: "I don't know" → Reward: -0.8 ❌         │
│                                                         │
│  Step 2: Compute Advantages                             │
│  ───────────────────────                               │
│  Mean reward = (1.0 - 0.5 - 0.5 - 0.8) / 4 = -0.45    │
│  Advantage("4") = 1.0 - (-0.45) = +1.45  (very good!) │
│                                                         │
│  Step 3: Policy wants to update                         │
│  ──────────────────────────────                        │
│  Model "wants": P("4") = 0.30 → 0.99  (3.3x jump!)    │
│  ratio = 0.99 / 0.30 = 3.3                            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │ MECHANISM 1: CLIPPING                           │  │
│  │ ─────────────────────                           │  │
│  │ Problem: 3.3x jump is too drastic!              │  │
│  │ Solution: Clip ratio to [0.8, 1.2]              │  │
│  │                                                  │  │
│  │ Without Clipping:                                │  │
│  │   Loss = 3.3 × 1.45 = 4.785                     │  │
│  │   → P("4") jumps to 0.99 ❌ DANGEROUS!         │  │
│  │                                                  │  │
│  │ With Clipping (ε=0.2):                          │  │
│  │   Clipped ratio = min(3.3, 1.2) = 1.2          │  │
│  │   Loss = 1.2 × 1.45 = 1.74                     │  │
│  │   → P("4") changes to 0.36 ✅ SAFE!            │  │
│  │                                                  │  │
│  │ Result: Gradual 0.30 → 0.36 (20% increase)     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │ MECHANISM 2: KL DIVERGENCE                      │  │
│  │ ──────────────────────                          │  │
│  │ After clipping update: P("4") = 0.36            │  │
│  │                                                  │  │
│  │ Compare to Reference:                            │  │
│  │   Reference: P("4")=0.30, P("5")=0.25, ...      │  │
│  │   Policy:    P("4")=0.36, P("5")=0.23, ...      │  │
│  │                                                  │  │
│  │ KL(Policy || Reference) = 0.015  (small! ✅)    │  │
│  │ KL Penalty = 0.01 × 0.015 = 0.00015            │  │
│  │                                                  │  │
│  │ Total Loss = 1.74 - 0.00015 = 1.73985          │  │
│  │                                                  │  │
│  │ Result: Small penalty, training proceeds ✅     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  Final Update for Step 20:                              │
│  ────────────────────────                              │
│  Policy Model: P("4") = 0.30 → 0.36  ✅               │
│  Reference Model: P("4") = 0.30 (unchanged) ❄️         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### What Happens Without These Protections?

```
┌─────────────────────────────────────────────────────────┐
│  SCENARIO: Training WITHOUT Clipping/KL (Disaster!)     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: No clipping                                    │
│  ────────                                              │
│  P("4"): 0.30 → 0.99 (instant jump!)                   │
│                                                         │
│  Step 2: No KL penalty                                  │
│  ────────                                              │
│  Model chases rewards aggressively...                   │
│                                                         │
│  Step 10: Complete disaster                             │
│  ─────────                                             │
│  P("4") = 1.00 (overconfident)                         │
│  P("asdf") = 0.50 (gibberish tokens!)                  │
│  P("xyz") = 0.30                                        │
│                                                         │
│  Output: "asdf xyz 4 qwerty" ❌❌❌                     │
│                                                         │
│  Result: Model is BROKEN - outputs gibberish!          │
│         KL divergence would be 10+ (massive!)          │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  WITH Clipping + KL: Stable Training ✅                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1-100: Gradual progress                           │
│  ────────────                                          │
│  P("4"): 0.30 → 0.36 → 0.43 → ... → 0.70             │
│  KL stays low: 0.015 → 0.032 → 0.048 (healthy!)       │
│                                                         │
│  Output: "4" ✅                                         │
│                                                         │
│  Result: Model WORKS - correct, coherent answers!      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Summary: The Three Key Concepts

| Concept | What It Does | How It Helps | When It Acts |
|---------|-------------|--------------|--------------|
| **Reference Model** | Frozen snapshot of SFT model | Provides baseline for KL measurement | Created once at training start |
| **Clipping (ε=0.2)** | Limits ratio to [0.8, 1.2] | Prevents drastic jumps per step | Every training step |
| **KL Divergence** | Measures drift from reference | Prevents total model collapse | Every training step |

**Complete Loss Formula:**
```
Loss = min(ratio × Advantage, clip(ratio, 0.8, 1.2) × Advantage) - 0.01 × KL
       └────────────── Clipping ──────────────────┘            └── KL ──┘
       Prevents P("4"): 0.30 → 0.99                    Prevents gibberish
```

**Mental Model:**
- **Reference Model** = Your starting point (GPS origin)
- **Clipping** = Speed limit (can't go 0→100 mph instantly)
- **KL Divergence** = Maximum distance from origin (can't drift too far)

**Configuration Parameters:**

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| **ε (epsilon)** | 0.2 | [0.8, 1.2] | Clip ratio per step |
| **β (beta)** | 0.01 | 0.001-0.1 | KL penalty weight |
| **Target KL** | <0.1 | Monitor | Healthy training range |

---

### Key Features

| Feature | Purpose | Benefit |
|---------|---------|---------|
| **Clipped Policy Gradient** | Limits model changes per step | Stable training, prevents collapse |
| **KL Divergence Control** | Keeps model close to base | Maintains coherence, prevents gibberish |
| **Dual-Clipping** | Limits both positive & negative updates | Balanced learning |
| **Importance Sampling** | Corrects for distribution mismatch | Accurate gradient estimates |
| **vLLM Integration** | Fast response generation | 5-10x speedup |
| **Sequence Packing** | Efficient GPU memory use | No padding waste |

---

### How to Curate Training Dataset for GRPO

**GRPO requires a fundamentally different dataset than DPO.** While DPO needs preference pairs (chosen vs rejected), GRPO only needs **prompts** and a **reward function**. Here's how to effectively curate training data for GRPO.

#### What GRPO Datasets Look Like

**GRPO Dataset Format (JSONL):**
```json
{"prompt": "Solve: 2x + 5 = 13"}
{"prompt": "Write a Python function to sort a list"}
{"prompt": "What is the capital of France?"}
{"prompt": "Explain quantum entanglement in simple terms"}
```

**Key Difference from DPO:**
| Aspect | DPO | GRPO |
|--------|-----|------|
| **Data Required** | Preference pairs (response A vs B) | Prompts only |
| **Annotation Cost** | High (humans rank responses) | None (rewards computed automatically) |
| **Format** | `{"prompt": "...", "chosen": "...", "rejected": "..."}` | `{"prompt": "..."}` |
| **Training Process** | No generation needed | Model generates multiple candidates |
| **Reward Source** | Implicit (learned from preferences) | Explicit (rule-based or model-based) |

#### Core Principles for GRPO Dataset Curation

**1. Prompts Must Be Task-Specific**

GRPO excels at **task-specific alignment** where you can define clear reward functions. Choose domains where quality is measurable:

✅ **Good for GRPO:**
- Math problems (verifiable correct/incorrect answers)
- Coding tasks (can test code execution)
- Question-answering (has correct answers)
- Summarization (can measure against reference summaries)
- Reasoning tasks (can verify logic chains)

❌ **Bad for GRPO:**
- General creative writing (subjective quality)
- Open-ended conversations (no clear reward signal)
- Style preferences (hard to define reward function)
- General knowledge (too broad, hard to reward)

**2. Dataset Size and Diversity**

| Dataset Size | Quality | Use Case |
|---|---|---|
| **100-500 prompts** | High-quality, diverse | Small models (1B-7B), initial exploration |
| **500-2K prompts** | Curated, task-specific | Medium models (7B-13B), stable training |
| **2K-10K+ prompts** | Diverse, comprehensive | Large models (70B+), production systems |

**Diversity Strategy:**
```
Total Prompts: 1000
├── Difficulty Distribution
│   ├── Easy (40%): 400 prompts
│   ├── Medium (35%): 350 prompts
│   └── Hard (25%): 250 prompts
│
├── Variations per Problem Type
│   ├── Math: Multiple problem structures
│   │   ├── Linear equations
│   │   ├── Quadratic equations
│   │   ├── Word problems
│   │   └── Multi-step problems
│   │
│   └── Coding: Different complexity levels
│       ├── Syntax/basics
│       ├── Algorithm implementation
│       ├── Bug fixing
│       └── Optimization
```

#### Domain-Specific Curation Tips

**For Math Problems:**
```yaml
curation_checklist:
  - Include various equation types (linear, quadratic, systems)
  - Mix problem formats (symbolic, word problems, proofs)
  - Vary complexity: algebra → calculus
  - Include edge cases: division by zero, negative numbers
  - Add both computational and conceptual problems
  - Range: simple arithmetic to multi-step reasoning
```

**Example Dataset:**
```json
{"prompt": "Solve: 2x + 5 = 13"}
{"prompt": "What is the derivative of 3x^2 + 2x?"}
{"prompt": "A train travels 60 mph for 2 hours. How far did it go?"}
{"prompt": "Find the roots of x^2 - 5x + 6 = 0"}
{"prompt": "Prove that the sum of angles in a triangle is 180 degrees"}
```

**For Coding Tasks:**
```yaml
curation_checklist:
  - Mix programming languages (Python, JavaScript, etc.)
  - Include multiple algorithm types (sorting, searching, DP, graphs)
  - Vary task types: implement, optimize, fix bugs, explain
  - Add complexity progression: list operations → tree structures
  - Include edge cases and error handling scenarios
  - Cover different paradigms: OOP, functional, procedural
```

**Example Dataset:**
```json
{"prompt": "Write a function to reverse a list"}
{"prompt": "Implement binary search algorithm"}
{"prompt": "Fix this code that has a bug: [code snippet]"}
{"prompt": "Optimize this function for better performance"}
{"prompt": "Explain what this recursive function does"}
```

**For Question-Answering:**
```yaml
curation_checklist:
  - Include factual (verifiable) questions
  - Mix open-ended and factual questions
  - Add multi-hop reasoning questions
  - Include questions across domains (science, history, etc.)
  - Add tricky questions where common wrong answers exist
  - Include follow-up questions that build on context
```

#### Writing Effective Reward Functions

**Rule-Based Rewards (Easiest):**

```python
# Math problems
def math_reward(response, ground_truth):
    """Reward function for math problems."""
    try:
        response_answer = extract_answer(response)
        if response_answer == ground_truth:
            return 1.0  # Perfect
        elif is_close(response_answer, ground_truth):
            return 0.5  # Partial credit
        else:
            return -0.5  # Wrong
    except:
        return -0.8  # Parse error

# Coding problems
def code_reward(response, test_cases):
    """Reward function for coding problems."""
    try:
        code = extract_code(response)
        passed = 0
        for test_input, expected_output in test_cases:
            if execute_code(code, test_input) == expected_output:
                passed += 1
        accuracy = passed / len(test_cases)

        if accuracy == 1.0:
            return 1.0  # All tests pass
        elif accuracy >= 0.8:
            return 0.5  # Mostly correct
        elif accuracy >= 0.5:
            return 0.0  # Partially correct
        else:
            return -0.5  # Wrong
    except SyntaxError:
        return -0.8  # Code doesn't compile
    except RuntimeError:
        return -0.8  # Code fails at runtime

# Answer matching
def qa_reward(response, ground_truth):
    """Reward function for Q&A problems."""
    response_lower = response.lower()
    truth_lower = ground_truth.lower()

    if response_lower == truth_lower:
        return 1.0  # Exact match
    elif any(key in response_lower for key in truth_lower.split()):
        return 0.7  # Contains key information
    else:
        return -0.5  # Wrong answer
```

**Model-Based Rewards (More Sophisticated):**

```yaml
# Using a learned reward model
rewards:
  type: "model_based"
  config:
    reward_model: "path/to/reward_model.pth"
    scaling: "normalize"  # Normalize reward scores
    thresholds:
      excellent: 0.8
      good: 0.5
      okay: 0.0
      poor: -0.5
```

#### Data Quality Checklist

Before training, validate your dataset:

```markdown
□ **Format Validity**
  □ All prompts are valid JSON/JSONL
  □ No corrupted entries
  □ Consistent field names ("prompt", not "question" or "text")

□ **Prompt Quality**
  □ Prompts are complete sentences (not fragments)
  □ No personally identifiable information (PII)
  □ Language is clear and unambiguous
  □ No duplicate prompts (or intentionally planned duplicates)

□ **Reward Function Alignment**
  □ Reward function covers all possible response types
  □ Edge cases handled (errors, timeouts, invalid responses)
  □ Reward distribution is reasonable (not all 1.0 or all -0.5)

□ **Diversity**
  □ Difficulty levels balanced (not all easy or all hard)
  □ Multiple prompt variations per concept
  □ Language diversity (not all identical structure)

□ **Domain Coverage**
  □ Representative of real use cases
  □ Covers edge cases and corner scenarios
  □ Includes both expected and unexpected inputs
```

#### Common Mistakes to Avoid

| ❌ Mistake | 💥 What Happens | ✅ Fix |
|---|---|---|
| **All prompts same difficulty** | Model overfits to one pattern | Mix easy/medium/hard (40/35/25) |
| **Reward function too sparse** | Model gets stuck, limited feedback | Add intermediate rewards (0.5, 0.3) |
| **No validation data** | Can't measure real improvement | Reserve 10-20% for validation |
| **Prompts too similar** | Model memorizes patterns, not generalizing | Use diverse problem structures |
| **Reward function not differentiable** | Gradient-based training fails | Use smooth reward signals |
| **Rewards all positive** | Model stops learning (plateau) | Use negative rewards for failures |
| **Dataset too small (<100)** | Massive overfitting | Aim for 500+ prompts minimum |
| **Misaligned reward and prompt** | Model learns wrong signals | Validate reward examples match prompts |

#### Example: Complete GRPO Dataset for Math

```json
{"prompt": "Solve: 2x + 5 = 13. Show your work.", "difficulty": "easy"}
{"prompt": "What is the derivative of f(x) = 3x^2 + 2x + 1?", "difficulty": "medium"}
{"prompt": "Prove that for any integers a and b, (a+b)^2 = a^2 + 2ab + b^2", "difficulty": "hard"}
{"prompt": "A rectangle has length 8 and width 5. What is the area?", "difficulty": "easy"}
{"prompt": "Solve the system: x + y = 5, 2x - y = 4", "difficulty": "medium"}
{"prompt": "Find the limit of (sin(x)/x) as x approaches 0", "difficulty": "hard"}
```

**Corresponding Reward Function:**
```python
def math_dataset_reward(response, prompt):
    """Reward function for math dataset examples."""

    # Extract expected answer based on prompt
    if "Solve: 2x + 5 = 13" in prompt:
        expected = 4
    elif "derivative of f(x) = 3x^2 + 2x + 1" in prompt:
        expected = "6x + 2"
    elif "area" in prompt:
        expected = 40
    elif "Solve the system" in prompt:
        expected = {"x": 3, "y": 2}
    elif "limit" in prompt:
        expected = 1
    else:
        expected = None

    # Check answer
    try:
        extracted_answer = extract_numerical_answer(response)
        if matches_expected(extracted_answer, expected):
            return 1.0
        elif close_enough(extracted_answer, expected, tolerance=0.01):
            return 0.5
        else:
            return -0.5
    except:
        return -0.8
```

#### Training with Your Curated Dataset

```yaml
# training_config.yaml
data:
  train_dataset: "curated_math_dataset.jsonl"
  val_dataset: "math_validation.jsonl"  # 10% hold-out

training:
  # Adjust based on dataset size
  num_responses_per_prompt: 4
  batch_size: 8
  max_steps: 5000  # For 500 prompts: 5000 / (500/8) ≈ 80 epochs

rewards:
  type: "rule_based"
  config:
    correct_answer: 1.0
    partial_credit: 0.5
    incorrect_answer: -0.5
    parse_error: -0.8

monitoring:
  reward_statistics:
    - Check mean reward per epoch (should increase)
    - Monitor reward distribution (should spread)
  validation:
    - Track validation accuracy
    - Spot-check generated responses
```

---

### GRPO Configuration

**Basic Setup:**

```yaml
# grpo_math_1B.yaml
model:
  name: "meta/llama-3.1-8b-instruct"
  training_type: "grpo"

training:
  # Group settings
  num_responses_per_prompt: 4        # Generate 4 candidates per prompt

  # PPO parameters
  ppo_epochs: 1                      # PPO updates per batch
  clip_range: 0.2                    # ε for clipping (1±0.2)
  kl_penalty: 0.01                   # β for KL divergence

  # Optimization
  learning_rate: 1e-6
  batch_size: 8                      # Number of prompts
  max_steps: 1000

  # Generation
  temperature: 0.7                   # Diversity in responses
  top_p: 0.9
  max_tokens: 256

data:
  train_dataset: "math_problems.jsonl"
  val_dataset: "math_validation.jsonl"

rewards:
  # Task-specific reward function
  type: "rule_based"                 # or "model_based"
  config:
    correct_answer: 1.0
    incorrect_answer: -0.5
    partial_credit: 0.3
```

---

### Dataset Format

**Simple Format (Prompts Only):**

```jsonl
{"prompt": "Solve: 3x - 7 = 14"}
{"prompt": "What is 25% of 80?"}
{"prompt": "Simplify: (x^2 + 2x + 1)"}
```

**With Optional Metadata:**

```jsonl
{
  "prompt": "Solve: 2x + 5 = 13",
  "expected_answer": "x = 4",
  "difficulty": "easy"
}
```

**Multi-Turn Conversations:**

```jsonl
{
  "messages": [
    {"role": "user", "content": "I need help with algebra"},
    {"role": "assistant", "content": "I'd be happy to help! What's the problem?"},
    {"role": "user", "content": "Solve: 2x + 5 = 13"}
  ]
}
```

---

### Reward Functions

**1. Rule-Based Rewards (Simple)**

```python
def math_reward(response, expected_answer):
    """Simple rule-based reward for math problems."""
    if extract_answer(response) == expected_answer:
        return 1.0  # Correct
    elif is_partially_correct(response):
        return 0.3  # Partial credit
    else:
        return -0.5  # Incorrect
```

**2. Model-Based Rewards (Advanced)**

```python
# Use a separate model to score responses
reward_model = load_model("reward-model")

def model_based_reward(prompt, response):
    """Score response quality using reward model."""
    score = reward_model.score(prompt, response)
    return score  # Range: [-1, 1]
```

**3. Mixed Rewards**

```python
def combined_reward(response, expected):
    """Combine multiple reward signals."""
    correctness = check_correctness(response, expected)  # 0 or 1
    clarity = score_explanation(response)                # 0-1
    format = check_format(response)                      # 0 or 1

    # Weighted combination
    return 0.6 * correctness + 0.3 * clarity + 0.1 * format
```

---

### Training Example

**Step 1: Prepare Dataset**

```python
import json

# Create math problems dataset
math_problems = [
    {"prompt": "Solve: 2x + 5 = 13", "answer": "x = 4"},
    {"prompt": "What is 30% of 150?", "answer": "45"},
    {"prompt": "Simplify: 3(x + 2) - 2x", "answer": "x + 6"},
]

# Save as JSONL
with open('math_train.jsonl', 'w') as f:
    for problem in math_problems:
        f.write(json.dumps({"prompt": problem["prompt"]}) + '\n')
```

**Step 2: Run GRPO Training**

```bash
# Run GRPO training
python examples/run_grpo.py \
  --config configs/grpo_math_1B.yaml \
  --train-data math_train.jsonl \
  --val-data math_val.jsonl \
  --output-dir ./grpo_math_model
```

**Step 3: Monitor Training**

```
Metrics to Watch:
─────────────────
✓ Reward Mean: Should increase (better responses)
✓ KL Divergence: Should stay low (<0.1)
✓ Policy Entropy: Measures exploration
✓ Token Probability Errors: Should be low
```

**Example Metrics Output:**

```
Step 100:
  reward/mean: 0.45          ← Improving! (started at 0.0)
  reward/best: 0.95          ← Best response getting better
  kl/policy: 0.032           ← Staying close to base ✅
  loss/total: 1.234

Step 500:
  reward/mean: 0.72          ← Much better!
  reward/best: 0.98
  kl/policy: 0.048           ← Still stable
  loss/total: 0.456          ← Decreasing
```

---

### Advanced Features

**1. Sequence Packing**

```
Problem: Variable-length responses waste GPU memory

Without Packing:
  [Response 1: 50 tokens + 206 padding]  ← Wasted
  [Response 2: 120 tokens + 136 padding] ← Wasted

With Packing:
  [Response 1: 50 tokens | Response 2: 120 tokens | R3: 86 tokens]
  ← No padding! Efficient!
```

**2. Fast Generation with vLLM**

```
Traditional Generation:
  Generate tokens one-by-one → Slow ❌

vLLM Acceleration:
  Optimized attention kernels
  PagedAttention for memory efficiency
  Continuous batching
  → 5-10x faster! ✅
```

**3. Multi-Environment Training**

```yaml
# Train on multiple tasks simultaneously
environments:
  - name: "math"
    dataset: "math_problems.jsonl"
    reward_fn: "math_reward"
    weight: 0.5

  - name: "coding"
    dataset: "code_problems.jsonl"
    reward_fn: "code_reward"
    weight: 0.3

  - name: "reasoning"
    dataset: "reasoning_tasks.jsonl"
    reward_fn: "reasoning_reward"
    weight: 0.2
```

---

### Best Practices

**1. Group Size Selection**

```
num_responses_per_prompt:
  Too Small (2):  ❌ Limited comparison signal
  Good (4-8):     ✅ Balanced signal and compute
  Too Large (16): ❌ Expensive, marginal gains

Recommendation: Start with 4
```

**2. KL Penalty Tuning**

```yaml
# Conservative (stay close to base)
kl_penalty: 0.05
Use when: Base model already good

# Moderate (balanced)
kl_penalty: 0.01
Use when: General purpose

# Aggressive (allow drift)
kl_penalty: 0.001
Use when: Task very different from base
```

**3. Reward Shaping**

```python
# Bad: Binary rewards only
def bad_reward(correct):
    return 1.0 if correct else 0.0
    # No signal for "almost correct"

# Good: Graded rewards
def good_reward(response):
    if perfect(response): return 1.0
    elif mostly_correct(response): return 0.7
    elif partially_correct(response): return 0.3
    elif attempted(response): return 0.1
    else: return -0.2
```

**4. Monitoring**

```python
# Key metrics to track
metrics = {
    "reward_mean": "Should increase steadily",
    "reward_std": "Should decrease (converging)",
    "kl_divergence": "Should stay < 0.1",
    "entropy": "Should decrease gradually",
    "clip_fraction": "Should be 20-40%",
}
```

---

### Troubleshooting

**Issue 1: KL Divergence Too High**

```yaml
Problem: kl/policy > 0.2
Symptoms: Model outputs becoming incoherent

Solutions:
  # Increase KL penalty
  kl_penalty: 0.05  # from 0.01

  # Reduce learning rate
  learning_rate: 5e-7  # from 1e-6

  # Reduce clip range
  clip_range: 0.1  # from 0.2
```

**Issue 2: Rewards Not Improving**

```yaml
Problem: reward/mean stagnates
Symptoms: No learning progress

Solutions:
  # Check reward function
  # - Is it too sparse?
  # - Are rewards too similar?

  # Increase learning rate
  learning_rate: 5e-6  # from 1e-6

  # Generate more diverse responses
  temperature: 0.8  # from 0.7
  num_responses_per_prompt: 8  # from 4
```

**Issue 3: Training Instability**

```yaml
Problem: Loss spikes, NaN values
Symptoms: Erratic metrics

Solutions:
  # Add gradient clipping
  max_grad_norm: 0.5

  # Reduce batch size
  batch_size: 4  # from 8

  # Use smaller clip range
  clip_range: 0.1  # from 0.2
```

---

### Use Cases

**1. Math Problem Solving**

```python
# Dataset
math_problems = [
    "Solve: 3x - 7 = 14",
    "Calculate: 25% of 80",
    "Simplify: (x + 2)^2"
]

# Reward: Correctness + explanation quality
reward = 0.7 * correct + 0.3 * explanation_score
```

**2. Code Generation**

```python
# Dataset
coding_tasks = [
    "Write a function to reverse a string",
    "Implement binary search",
    "Create a class for a linked list"
]

# Reward: Pass tests + code quality
reward = 0.8 * tests_passed + 0.2 * code_quality
```

**3. Multi-Turn Dialogue**

```python
# Dataset (conversation context)
conversations = [
    {
        "history": ["User: I'm planning a trip", "Bot: Where to?"],
        "user": "Paris for 5 days",
        # GRPO learns best response
    }
]

# Reward: Helpfulness + engagement
reward = 0.5 * helpfulness + 0.3 * engagement + 0.2 * safety
```

---

### Summary

| Aspect | Details |
|--------|---------|
| **What** | RL algorithm for task-specific LLM alignment |
| **How** | Group-based relative ranking without reward model |
| **When** | Task-specific optimization, multi-turn dialogue |
| **Advantages** | No reward model, stable, multi-task support |
| **Best For** | Math, coding, reasoning, conversation |

**Key Takeaways:**
- ✅ No separate reward model needed
- ✅ Stable training with clipped policy gradients
- ✅ Excellent for task-specific optimization
- ✅ Supports multi-environment training
- ✅ Faster than traditional RLHF

**Sources:**
- [NVIDIA NeMo RL GRPO Guide](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html)

---

## Group reward-Decoupled Normalization Policy Optimization (GDPO)

### What is GDPO?

**GDPO (Group reward-Decoupled normalization Policy Optimization)** is an advanced variant of GRPO that solves the **coupling problem** in GRPO's advantage calculation. While GRPO uses group mean as the baseline (coupling all responses together), GDPO uses **independent baselines** to decouple advantage normalization.

**Key Innovation**: GDPO prevents one outlier response from affecting the advantage calculations of all other responses in the group.

### The Coupling Problem in GRPO

In GRPO, all responses in a group are coupled through the shared mean baseline:

```
┌─────────────────────────────────────────────────────────┐
│  GRPO COUPLING PROBLEM                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Prompt: "What is 2+2?"                                 │
│                                                         │
│  4 Responses Generated:                                 │
│  ───────────────────                                   │
│  Response 1: "4"           → Reward: +1.0              │
│  Response 2: "5"           → Reward: -0.5              │
│  Response 3: "3"           → Reward: -0.5              │
│  Response 4: "gibberish"   → Reward: -10.0 ❌ OUTLIER │
│                                                         │
│  GRPO Advantage Calculation:                            │
│  ─────────────────────────                             │
│  Mean = (1.0 - 0.5 - 0.5 - 10.0) / 4 = -2.5           │
│         ↑                           ↑                   │
│    All rewards          ONE outlier drags mean down!    │
│                                                         │
│  Advantages:                                            │
│  • Response 1: 1.0 - (-2.5) = +3.5   ← Inflated! ❌   │
│  • Response 2: -0.5 - (-2.5) = +2.0  ← False positive!│
│  • Response 3: -0.5 - (-2.5) = +2.0  ← False positive!│
│  • Response 4: -10.0 - (-2.5) = -7.5                   │
│                                                         │
│  PROBLEM: One bad outlier makes mediocre responses     │
│           (Response 2, 3) look good! All advantages    │
│           are COUPLED through shared mean baseline.    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### How GDPO Solves the Coupling Problem

GDPO uses **independent baselines** for each response, breaking the coupling:

```
┌─────────────────────────────────────────────────────────┐
│  GDPO DECOUPLED NORMALIZATION                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Same Scenario:                                         │
│  ───────────                                           │
│  Response 1: "4"         → Reward: +1.0                │
│  Response 2: "5"         → Reward: -0.5                │
│  Response 3: "3"         → Reward: -0.5                │
│  Response 4: "gibberish" → Reward: -10.0 ❌ OUTLIER   │
│                                                         │
│  GDPO Advantage Calculation (Decoupled):                │
│  ──────────────────────────────────────                │
│  Instead of group mean, use INDEPENDENT baseline       │
│  (e.g., running average, moving baseline, etc.)        │
│                                                         │
│  Independent Baseline = -0.3 (learned from history)    │
│                                                         │
│  Advantages:                                            │
│  • Response 1: 1.0 - (-0.3) = +1.3   ✅ Correct!      │
│  • Response 2: -0.5 - (-0.3) = -0.2  ✅ Correctly bad!│
│  • Response 3: -0.5 - (-0.3) = -0.2  ✅ Correctly bad!│
│  • Response 4: -10.0 - (-0.3) = -9.7 ❌ Still bad     │
│                                                         │
│  BENEFIT: Outlier doesn't affect other advantages!     │
│           Each response evaluated independently.        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### GDPO vs GRPO: Detailed Comparison

| **Aspect** | **GRPO** | **GDPO** |
|------------|----------|----------|
| **Baseline Type** | Group mean (shared) | Independent baseline (decoupled) |
| **Coupling** | All responses coupled | Responses decoupled |
| **Outlier Sensitivity** | High (one outlier affects all) | Low (outliers isolated) |
| **Advantage Accuracy** | Can be distorted by outliers | More robust to outliers |
| **Training Stability** | Less stable with noisy rewards | More stable with noisy rewards |
| **Computational Cost** | Lower (simple mean) | Slightly higher (baseline tracking) |
| **Use Case** | Clean, consistent rewards | Noisy, outlier-prone environments |
| **Implementation** | Simpler | More complex (needs baseline model) |

### When to Use GDPO vs GRPO

**Use GRPO when:**
- ✅ Reward function is clean and consistent
- ✅ Outliers are rare
- ✅ You want simpler implementation
- ✅ All responses in group are similar quality

**Use GDPO when:**
- ✅ Reward function is noisy or has outliers
- ✅ Response quality varies widely in groups
- ✅ You need more robust training
- ✅ You're willing to handle extra complexity

### GDPO Algorithm Details

**Loss Function (similar to GRPO but with decoupled baseline):**

```
Loss = min(ratio × A_t, clip(ratio, 1-ε, 1+ε) × A_t) - β × KL_divergence

Where:
- ratio = π_θ(x) / π_θ_old(x)
- A_t = Reward - Independent_Baseline  ← KEY DIFFERENCE!
                  ↑
         NOT group mean! (decoupled)
- ε = Clipping threshold (default: 0.2)
- β = KL penalty weight (default: 0.01)
- KL_divergence = KL(π_θ || π_ref)

Independent_Baseline Options:
1. Running average across all samples
2. Per-prompt moving average
3. Learned value function (mini critic)
4. Exponential moving average (EMA)
```

### Example: GRPO vs GDPO in Action

**Scenario: Math Problem Training with One Bad Response**

```
┌─────────────────────────────────────────────────────────┐
│  COMPARISON: GRPO vs GDPO                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Training Data:                                         │
│  ─────────────                                         │
│  Prompt: "Solve: 5 + 3 = ?"                            │
│                                                         │
│  4 Responses:                                           │
│  • Response 1: "8" → Reward: +1.0  ✅ Perfect         │
│  • Response 2: "9" → Reward: -0.3  ⚠️ Close           │
│  • Response 3: "7" → Reward: -0.3  ⚠️ Close           │
│  • Response 4: "XYZABC" → Reward: -15.0 ❌ Gibberish  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │ GRPO (Coupled Baseline)                           │ │
│  │ ───────────────────────                           │ │
│  │ Baseline = (-15.0 - 0.3 - 0.3 + 1.0) / 4 = -3.65 │ │
│  │                                                    │ │
│  │ Advantages:                                        │ │
│  │ • Resp 1: 1.0 - (-3.65) = +4.65  ❌ Too high!    │ │
│  │ • Resp 2: -0.3 - (-3.65) = +3.35 ❌ False +ve!   │ │
│  │ • Resp 3: -0.3 - (-3.65) = +3.35 ❌ False +ve!   │ │
│  │ • Resp 4: -15.0 - (-3.65) = -11.35               │ │
│  │                                                    │ │
│  │ Result: Model increases P("9") and P("7")! BAD!  │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │ GDPO (Decoupled Baseline)                         │ │
│  │ ─────────────────────────                         │ │
│  │ Independent Baseline = -0.2 (from running avg)   │ │
│  │                                                    │ │
│  │ Advantages:                                        │ │
│  │ • Resp 1: 1.0 - (-0.2) = +1.2   ✅ Good!         │ │
│  │ • Resp 2: -0.3 - (-0.2) = -0.1  ✅ Slightly bad! │ │
│  │ • Resp 3: -0.3 - (-0.2) = -0.1  ✅ Slightly bad! │ │
│  │ • Resp 4: -15.0 - (-0.2) = -14.8 ❌ Very bad!    │ │
│  │                                                    │ │
│  │ Result: Model increases P("8"), decreases P("9") │ │
│  │         and P("7"). CORRECT behavior! ✅          │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Benefits of GDPO

| Benefit | Explanation |
|---------|-------------|
| **Outlier Robustness** | One bad response doesn't corrupt all advantages |
| **Accurate Advantages** | Each response evaluated against stable baseline |
| **Training Stability** | Less variance in advantage estimates |
| **Better Convergence** | More consistent gradient signals |
| **Handles Noisy Rewards** | Robust to reward function imperfections |

### Implementation Considerations

**Baseline Options:**

```python
# Option 1: Running Average (Simplest)
independent_baseline = running_mean(all_past_rewards)

# Option 2: Exponential Moving Average
independent_baseline = 0.9 * old_baseline + 0.1 * current_reward

# Option 3: Per-Prompt Baseline (Most sophisticated)
independent_baseline = baseline_model(prompt)  # Learned value function
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| **Running Average** | Simple, no extra parameters | Slow to adapt to distribution shifts |
| **EMA** | Balances history and new data | Needs tuning (decay factor) |
| **Learned Value** | Most accurate, adaptive | Requires training separate model |

### Summary: GDPO vs GRPO

**GRPO:**
- Uses group mean baseline (all responses coupled)
- Simpler implementation
- Works well when rewards are clean
- Vulnerable to outliers affecting all advantages

**GDPO:**
- Uses independent baseline (responses decoupled)
- More robust to outliers
- Better for noisy reward environments
- Slightly more complex implementation

**Key Insight:**
```
GRPO Coupling Problem:
  Mean = (R₁ + R₂ + R₃ + R_outlier) / 4
         ↑                  ↑
    All responses    One outlier affects ALL!

GDPO Solution:
  Advantage = Reward - Independent_Baseline
                       ↑
              Decoupled from current group!
```

**When to Choose:**

```
Choose GRPO if:
  • Reward function is clean and consistent
  • Implementation simplicity matters
  • Outliers are rare

Choose GDPO if:
  • Reward function has noise or outliers
  • Training stability is critical
  • Response quality varies widely
  • Worth the extra complexity
```

---

## Understanding KL Divergence in Alignment

**KL (Kullback-Leibler) Divergence** measures how different two probability distributions are. In alignment training, it measures how much the **Policy Model** (being trained) has changed from the **Reference Model** (frozen baseline).

### Core Concept: Two Models

All alignment methods (DPO, GRPO, RLHF) use two models:

| Model | Role | Updates |
|-------|------|---------|
| **Reference Model (π_ref)** ❄️ | Frozen snapshot at training start | Never changes |
| **Policy Model (π_θ)** 🔥 | Model being trained | Updates each step |

```
SFT Model → Snapshot → Reference Model ❄️  (frozen)
                   ↓
                   Copy → Policy Model 🔥  (trainable)
```

### KL Divergence Formula

The mathematical formula for KL divergence between two distributions P and Q is:

$$D_{KL}(P \parallel Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

In the context of alignment training, where P is the **Policy Model** and Q is the **Reference Model**:

$$D_{KL}(\pi_\theta \parallel \pi_{ref}) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \frac{\pi_\theta(x)}{\pi_{ref}(x)} \right]$$

**Key Properties:**

- **Non-negative**: D_KL(P || Q) ≥ 0, with equality only when P = Q
- **Asymmetric**: D_KL(P || Q) ≠ D_KL(Q || P) (order matters!)
- **Not a true distance**: Doesn't satisfy triangle inequality
- **Forward vs Reverse KL**:
  - Forward KL (P || Q): Penalizes when P has high probability where Q has low probability → Conservative
  - Reverse KL (Q || P): Penalizes when Q has high probability where P has low probability → Inclusive

In alignment training, we use **Forward KL** to prevent the policy from diverging too far from the reference model.

### Formula in Loss Function

```python
Loss = Task_Loss + β × KL(Policy || Reference)
                   ↑       ↑
              kl_penalty   Actual KL divergence
              (weight)     (distance metric)

Where:
- β (beta) = kl_penalty = Weight/coefficient (default: 0.01)
- KL(Policy || Reference) = Actual divergence value (measured)

TWO DIFFERENT THINGS:
─────────────────────
1. β (kl_penalty parameter):
   • Configuration parameter you set
   • Controls HOW MUCH you penalize drift
   • Higher β → Stronger penalty → Stay closer to reference
   • Lower β → Weaker penalty → Allow more deviation

2. KL(Policy || Reference):
   • Computed metric during training
   • Measures HOW MUCH the model has drifted
   • Lower KL → Less drift → ✅ Healthier training
   • Higher KL → More drift → ❌ Unstable training
```

**Example with Numbers:**

```
Scenario 1: Conservative Training
─────────────────────────────────
β (kl_penalty) = 0.1   ← HIGH penalty weight
KL divergence = 0.02    ← Low drift (healthy)

Loss = Task_Loss + 0.1 × 0.02
     = Task_Loss + 0.002

Result: Strong penalty keeps model close to reference ✅


Scenario 2: Aggressive Training
────────────────────────────────
β (kl_penalty) = 0.001  ← LOW penalty weight
KL divergence = 0.15    ← Higher drift (risky)

Loss = Task_Loss + 0.001 × 0.15
     = Task_Loss + 0.00015

Result: Weak penalty allows more drift ⚠️


Scenario 3: DANGER - High KL despite penalty
─────────────────────────────────────────────
β (kl_penalty) = 0.01   ← Normal penalty weight
KL divergence = 0.8     ← VERY high drift! ❌

Loss = Task_Loss + 0.01 × 0.8
     = Task_Loss + 0.008

Result: Large penalty added to loss → Gradient pushes back ⚠️
```

### Understanding β (kl_penalty) vs KL Divergence

**Critical Distinction:**

```
┌─────────────────────────────────────────────────────────┐
│  β (kl_penalty) ≠ KL(Policy || Reference)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  β (kl_penalty):                                        │
│  ───────────────                                       │
│  • Configuration parameter (YOU set this)               │
│  • Controls penalty strength                            │
│  • In config file: kl_penalty: 0.01                    │
│  • Lower β = Weaker penalty = More exploration allowed  │
│  • Higher β = Stronger penalty = Stay close to base    │
│                                                         │
│  KL(Policy || Reference):                              │
│  ─────────────────────────                             │
│  • Computed metric (MEASURED during training)           │
│  • Actual drift amount                                  │
│  • Lower KL = ✅ HEALTHY (less drift)                  │
│  • Higher KL = ❌ UNHEALTHY (too much drift)           │
│                                                         │
│  Formula Breakdown:                                     │
│  ──────────────────                                    │
│  Loss = Task_Loss + β × KL                             │
│                     ↑   ↑                              │
│               config  measured                          │
│                                                         │
│  Example:                                               │
│  β = 0.01 (you set)                                    │
│  KL = 0.05 (measured drift)                            │
│  Penalty = 0.01 × 0.05 = 0.0005                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Interpreting KL Divergence Values

**This table refers to KL(Policy || Reference), NOT β:**

| KL Divergence | Status | Meaning |
|---------------|--------|---------|
| **0.01-0.05** | ✅ Healthy | Learning while staying coherent |
| **0.05-0.2** | ⚠️ Monitor | Moderate drift, watch closely |
| **>0.2** | ❌ Danger | Too much drift, gibberish risk |

**How β Affects Training:**

```
┌─────────────────────────────────────────────────────────┐
│  IMPACT OF β (kl_penalty) ON TRAINING                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  High β (e.g., 0.1): STRONG PENALTY                     │
│  ────────────────────────────────────                   │
│  Loss = Task_Loss + 0.1 × KL                           │
│          ↑          ↑                                   │
│       optimize    BIG multiplier!                       │
│                                                         │
│  What the optimizer sees:                               │
│  • If KL increases even slightly, penalty jumps!        │
│  • KL = 0.02 → Penalty = 0.002 (moderate impact)       │
│  • KL = 0.1  → Penalty = 0.01  (HUGE impact on loss!)  │
│                                                         │
│  Optimizer's decision:                                  │
│  "Increasing KL is EXPENSIVE! I must keep KL low       │
│   to minimize total loss. I'll stay close to           │
│   reference model even if task performance suffers."   │
│                                                         │
│  RESULT (what actually happens):                        │
│  ✅ Model FORCED to stay close → KL stays ~0.02-0.05   │
│  ✅ Conservative updates → Safe training                │
│  ⚠️ Slower task improvement (penalty dominates)        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ CAUSALITY: High β → Forces low KL                │  │
│  │            (β controls, KL is the result)         │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│                                                         │
│  Low β (e.g., 0.001): WEAK PENALTY                      │
│  ───────────────────────────────────                    │
│  Loss = Task_Loss + 0.001 × KL                         │
│          ↑          ↑                                   │
│       optimize    tiny multiplier                       │
│                                                         │
│  What the optimizer sees:                               │
│  • Even large KL creates tiny penalty                   │
│  • KL = 0.02 → Penalty = 0.00002 (negligible!)         │
│  • KL = 0.1  → Penalty = 0.0001  (still tiny)          │
│                                                         │
│  Optimizer's decision:                                  │
│  "KL penalty is CHEAP! I can increase KL a lot and     │
│   the penalty barely affects total loss. I'll focus    │
│   on improving task performance, drift is allowed."    │
│                                                         │
│  RESULT (what actually happens):                        │
│  ⚠️ Model ALLOWED to drift → KL can reach 0.1-0.2      │
│  ✅ Aggressive updates → Exploratory training           │
│  ✅ Faster task improvement (task loss dominates)       │
│  ❌ Risk: May drift too far if not monitored            │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ CAUSALITY: Low β → Allows high KL                │  │
│  │            (β permits, KL grows as needed)        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Complete Example: Training Step by Step**

```
Step 0: Training Start
──────────────────────
Config: β = 0.01 (moderate penalty weight)

Reference Model: P("cat") = 60%  ❄️ (frozen)
Policy Model:    P("cat") = 60%  🔥 (trainable)

KL = 0.0  (identical)
Penalty = 0.01 × 0.0 = 0


Step 100: After Some Training
──────────────────────────────
Reference Model: P("cat") = 60%  ❄️ (still frozen)
Policy Model:    P("cat") = 65%  🔥 (improved)

KL = 0.02  (small drift - HEALTHY ✅)
Penalty = 0.01 × 0.02 = 0.0002  (tiny penalty)
→ Training proceeds normally


Step 500: Too Much Drift
─────────────────────────
Reference Model: P("cat") = 60%  ❄️ (still frozen)
Policy Model:    P("cat") = 95%  🔥 (drifted too far)

KL = 0.5  (large drift - DANGER ❌)
Penalty = 0.01 × 0.5 = 0.005  (large penalty!)
→ Loss increases significantly
→ Gradient pushes model back toward reference
→ Self-correcting mechanism activates


If we had used β = 0.001 instead:
──────────────────────────────────
KL = 0.5  (same drift)
Penalty = 0.001 × 0.5 = 0.0005  (10x smaller!)
→ Less correction force
→ Model allowed to drift more
```

### Example: KL in Action

**Scenario:** Training "What animal?" prompt

```
┌─────────────────────────────────────────────────────────┐
│  Reference Model ❄️ (frozen):                           │
│  P(cat)=60%, P(dog)=30%, P(bird)=10%                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  After 100 Steps - Healthy Training:                    │
│  Policy Model: P(cat)=65%, P(dog)=25%, P(bird)=10%     │
│  KL = 0.02 ✅  Small change, good progress!            │
│                                                         │
│  After 500 Steps - Without KL Penalty (Bad!):          │
│  Policy Model: P(cat)=10%, P(dog)=80%, P(bird)=10%     │
│  KL = 0.8 ❌  Huge change, model unstable!             │
│                                                         │
│  With KL Penalty:                                       │
│  Loss = Task_Loss + 0.01 × 0.8 = Task_Loss + 0.008    │
│  → High penalty pushes model back toward reference ✅   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Why KL Penalty Matters

| Purpose | Benefit |
|---------|---------|
| **Prevents reward hacking** | Model can't output gibberish to maximize rewards |
| **Maintains coherence** | Preserves language generation quality |
| **Balances learning** | Improves task performance without forgetting base skills |

**Visual Comparison:**

```
WITHOUT KL:                       WITH KL (β=0.01):
─────────────                     ─────────────────
Step 0:   "Paris."               Step 0:   "Paris."
Step 500: "asdkjf" ❌           Step 500: "Paris is the capital" ✅
          (Gibberish!)                    (Coherent + improved!)
```

### Configuration Examples: Setting β (kl_penalty)

**Remember: `kl_penalty` in config = β in formula**

```yaml
# Conservative Approach (Strong Penalty)
# ─────────────────────────────────────
kl_penalty: 0.1  # β = 0.1

Effect:
  Loss = Task_Loss + 0.1 × KL
  → Even small KL creates large penalty
  → Keeps model very close to reference
  → Use when: Base model already good

Example:
  If KL = 0.05 (moderate drift)
  Penalty = 0.1 × 0.05 = 0.005 (significant!)
  → Strong correction force applied


# Balanced Approach (Recommended)
# ────────────────────────────────
kl_penalty: 0.01  # β = 0.01 ⭐

Effect:
  Loss = Task_Loss + 0.01 × KL
  → Moderate penalty for drift
  → Good balance between learning and stability
  → Use when: General-purpose training

Example:
  If KL = 0.05 (moderate drift)
  Penalty = 0.01 × 0.05 = 0.0005 (moderate)
  → Balanced correction


# Aggressive Approach (Weak Penalty)
# ───────────────────────────────────
kl_penalty: 0.001  # β = 0.001

Effect:
  Loss = Task_Loss + 0.001 × KL
  → Small penalty even for large drift
  → Allows more exploration and deviation
  → Use when: Task very different from base model

Example:
  If KL = 0.05 (moderate drift)
  Penalty = 0.001 × 0.05 = 0.00005 (tiny!)
  → Weak correction, more freedom


# SUMMARY TABLE
# ─────────────

| β Value | Penalty Strength | Typical KL Range | Training Style |
|---------|------------------|------------------|----------------|
| 0.1     | Strong          | 0.01-0.03        | Conservative   |
| 0.01    | Moderate ⭐     | 0.02-0.08        | Balanced       |
| 0.001   | Weak            | 0.05-0.15        | Aggressive     |
```

**Key Insight:**

```
Configuration β (kl_penalty):
  Lower β = Weaker penalty = MORE drift allowed ⚠️
  Higher β = Stronger penalty = LESS drift allowed ✅

Measured KL divergence:
  Lower KL = Less drift = ✅ HEALTHY
  Higher KL = More drift = ❌ UNHEALTHY

Don't confuse them!
```

---

---

**Key Takeaways:**

✅ **Reference Model** = Frozen baseline (never changes)
✅ **Policy Model** = Trainable model (updates each step)
✅ **β (kl_penalty)** = Weight parameter you configure (controls penalty strength)
✅ **KL Divergence** = Measured distance between Policy and Reference
✅ **Formula** = Loss = Task_Loss + β × KL
✅ **Target** = Keep measured KL < 0.1 for healthy training

**Quick Reference Card:**

```
┌─────────────────────────────────────────────────────────┐
│  β vs KL: THE CRITICAL DIFFERENCE                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  YOU CONTROL:                                           │
│  ────────────                                          │
│  β (kl_penalty in config)                              │
│  • Default: 0.01                                        │
│  • Lower β → Allows more exploration                    │
│  • Higher β → Keeps model conservative                  │
│                                                         │
│  TRAINING MEASURES:                                     │
│  ──────────────────                                    │
│  KL(Policy || Reference)                               │
│  • Healthy: < 0.1                                       │
│  • Lower KL → Better (less drift)                       │
│  • Higher KL → Worse (too much drift)                   │
│                                                         │
│  THEY INTERACT:                                         │
│  ───────────────                                       │
│  Penalty = β × KL                                       │
│                                                         │
│  Examples:                                              │
│  β=0.01, KL=0.05 → Penalty=0.0005 ✅ Good              │
│  β=0.01, KL=0.5  → Penalty=0.005  ❌ High penalty!     │
│  β=0.1,  KL=0.05 → Penalty=0.005  ⚠️ Very strict      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**See detailed examples in:**
- [Understanding GRPO Training: Complete Example](#understanding-grpo-training-complete-example) - Shows KL in action with math problem
- [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo) - DPO-specific KL usage
- [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo) - GRPO-specific KL usage