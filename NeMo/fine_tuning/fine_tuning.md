fine_tuning.md



# NeMo Fine-Tuning: Training Dataset Format Guide

> **Source**: https://docs.nvidia.com/nemo/microservices/latest/fine-tune/tutorials/format-training-dataset.html

---

## Table of Contents

### Part 1: Dataset Formatting
- [Introduction](#introduction)
  - [What is Dataset Formatting?](#what-is-dataset-formatting)
  - [Why is Format Important?](#why-is-format-important)
  - [Key Requirement: JSONL Format](#key-requirement-jsonl-format)
- [Chat Models Dataset Format](#chat-models-dataset-format)
  - [What are Chat Models?](#what-are-chat-models)
  - [Basic Chat Format Structure](#basic-chat-format-structure)
  - [Understanding Roles](#understanding-roles)
  - [Important Rules](#important-rules)
  - [Chat Format Examples](#chat-format-examples)
  - [Full JSONL File Example for Chat](#full-jsonl-file-example-for-chat)
- [Completion Models Dataset Format](#completion-models-dataset-format)
  - [What are Completion Models?](#what-are-completion-models)
  - [Basic Completion Format Structure](#basic-completion-format-structure)
  - [Understanding Fields](#understanding-fields)
  - [Completion Format Examples](#completion-format-examples)
  - [Full JSONL File Example for Completion](#full-jsonl-file-example-for-completion)
- [Advanced Features](#advanced-features)
  - [1. Reasoning Support (Chain-of-Thought)](#1-reasoning-support-chain-of-thought)
  - [2. Tool Calling (Function Calling)](#2-tool-calling-function-calling)
    - [Method 1: Inline Tool Definition](#method-1-inline-tool-definition)
    - [Method 2: Shared Tool Configuration](#method-2-shared-tool-configuration)
  - [3. Multi-Modal Support (Images, etc.)](#3-multi-modal-support-images-etc)
- [Best Practices](#best-practices)
  - [1. Data Quality Over Quantity](#1-data-quality-over-quantity)
  - [2. Consistency is Key](#2-consistency-is-key)
  - [3. Message Order](#3-message-order)
  - [4. Complete Context](#4-complete-context)
  - [5. One Example Per Line](#5-one-example-per-line)
  - [6. Escaping Special Characters](#6-escaping-special-characters)
  - [7. Validation Before Training](#7-validation-before-training)
- [Common Mistakes to Avoid](#common-mistakes-to-avoid)
- [Choosing the Right Format](#choosing-the-right-format)
- [Inference Endpoints](#inference-endpoints)
- [Quick Reference](#quick-reference)
- [Additional Resources](#additional-resources)

### Part 2: Fine-Tuning Methods
- [LoRA Model Customization](#lora-model-customization)
  - [What is LoRA?](#what-is-lora)
  - [Prerequisites](#prerequisites)
  - [Step 1: Model Selection](#step-1-model-selection)
  - [Step 2: Dataset Preparation](#step-2-dataset-preparation)
  - [Step 3: Data Upload](#step-3-data-upload)
  - [Step 4: Job Creation and Submission](#step-4-job-creation-and-submission)
  - [Step 5: Job Monitoring](#step-5-job-monitoring)
  - [Step 6: Using the Fine-Tuned Model](#step-6-using-the-fine-tuned-model)
  - [Common Issues and Solutions](#common-issues-and-solutions)
  - [Integration with Weights & Biases](#integration-with-weights--biases)
- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [What is Full SFT?](#what-is-full-sft)
  - [Prerequisites](#prerequisites-1)
  - [Step 1: Model Selection](#step-1-model-selection-1)
  - [Step 2: Dataset Preparation](#step-2-dataset-preparation-1)
  - [Step 3: Data Upload](#step-3-data-upload-1)
  - [Step 4: Job Configuration](#step-4-job-configuration)
  - [Step 5: Submit Training Job](#step-5-submit-training-job)
  - [Step 6: Monitor Training](#step-6-monitor-training)
  - [Step 7: Model Deployment](#step-7-model-deployment)
  - [Step 8: Testing the Model](#step-8-testing-the-model)
  - [Step 9: Monitoring with Weights & Biases](#step-9-monitoring-with-weights--biases)
  - [Comparison: LoRA vs Full SFT](#comparison-lora-vs-full-sft)
  - [Best Practices](#best-practices-1)
  - [Troubleshooting](#troubleshooting)
  - [Next Steps](#next-steps)
  - [Appendix: Complete Example Scripts](#appendix-complete-example-scripts)

### Part 3: Knowledge Distillation
- [Knowledge Distillation](#knowledge-distillation)
  - [What is Knowledge Distillation?](#what-is-knowledge-distillation)
  - [Prerequisites](#prerequisites-2)
  - [Step 1: Find Distillation Configurations](#step-1-find-distillation-configurations)
  - [Step 2: Prepare Dataset](#step-2-prepare-dataset)
  - [Step 3: Upload Dataset](#step-3-upload-dataset)
  - [Step 4: Configure Distillation Job](#step-4-configure-distillation-job)
  - [Step 5: Submit Distillation Job](#step-5-submit-distillation-job)
  - [Step 6: Monitor Training](#step-6-monitor-training-1)
  - [Step 7: Deploy and Test](#step-7-deploy-and-test)
  - [Best Practices for Distillation](#best-practices-for-distillation)
  - [Troubleshooting Distillation](#troubleshooting-distillation)
  - [Monitoring with Weights & Biases](#monitoring-with-weights--biases)

### Part 4: Embedding Model Customization
- [Embedding Model Customization](#embedding-model-customization)
  - [What are Embedding Models?](#what-are-embedding-models)
  - [Prerequisites](#prerequisites-3)
  - [Step 1: Understand Embedding Dataset Format](#step-1-understand-embedding-dataset-format)
  - [Step 2: Prepare Embedding Dataset](#step-2-prepare-embedding-dataset)
  - [Step 3: Upload Dataset](#step-3-upload-dataset-1)
  - [Step 4: Find Embedding Model Configurations](#step-4-find-embedding-model-configurations)
  - [Step 5: Configure Embedding Training Job](#step-5-configure-embedding-training-job)
  - [Step 6: Submit Training Job](#step-6-submit-training-job-1)
  - [Step 7: Monitor Training](#step-7-monitor-training)
  - [Step 8: Deploy Embedding Model](#step-8-deploy-embedding-model)
  - [Step 9: Test Embedding Model](#step-9-test-embedding-model)
  - [Step 10: Quality Validation](#step-10-quality-validation)
  - [Use Cases and Examples](#use-cases-and-examples)
  - [Best Practices for Embedding Models](#best-practices-for-embedding-models)
  - [Troubleshooting Embedding Models](#troubleshooting-embedding-models)
  - [Summary](#summary)

### Part 5: Monitoring Training Metrics
- [Monitoring Training Metrics](#monitoring-training-metrics)
  - [What are Training Metrics?](#what-are-training-metrics)
  - [Understanding Loss Metrics](#understanding-loss-metrics)
  - [Step 1: Access Metrics via API](#step-1-access-metrics-via-api)
  - [Step 2: Monitor Metrics During Training](#step-2-monitor-metrics-during-training)
  - [Step 3: Integration with MLflow](#step-3-integration-with-mlflow)
  - [Step 4: Integration with Weights & Biases](#step-4-integration-with-weights--biases)
  - [Step 5: Interpreting Metrics](#step-5-interpreting-metrics)
  - [Step 6: Configure Metric Logging](#step-6-configure-metric-logging)
  - [Best Practices for Monitoring](#best-practices-for-monitoring)
  - [Troubleshooting Metrics Issues](#troubleshooting-metrics-issues)

### Part 6: Optimizing Training Performance
- [Optimizing Training Performance](#optimizing-training-performance)
  - [What is Tokens Per GPU?](#what-is-tokens-per-gpu)
  - [Why Optimization Matters](#why-optimization-matters)
  - [Primary Optimization: Sequence Packing](#primary-optimization-sequence-packing)
  - [Step 1: Enable Sequence Packing](#step-1-enable-sequence-packing)
  - [Step 2: Compare Performance](#step-2-compare-performance)
  - [Step 3: Analyze Performance Metrics](#step-3-analyze-performance-metrics)
  - [Step 4: Additional Optimization Techniques](#step-4-additional-optimization-techniques)
  - [Best Practices for Optimization](#best-practices-for-optimization)
  - [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

### Part 7: Importing Custom Models from HuggingFace
- [Importing Custom Models from HuggingFace](#importing-custom-models-from-huggingface)
  - [What is Model Import?](#what-is-model-import)
  - [Prerequisites](#prerequisites-4)
  - [Supported Model Architectures](#supported-model-architectures)
  - [Step 1: Download Model from HuggingFace](#step-1-download-model-from-huggingface)
  - [Step 2: Upload Model to NeMo Data Store](#step-2-upload-model-to-nemo-data-store)
  - [Step 3: Register Model in Entity Store](#step-3-register-model-in-entity-store)
  - [Step 4: Create Customization Target](#step-4-create-customization-target)
  - [Step 5: Create Fine-Tuning Configuration](#step-5-create-fine-tuning-configuration)
  - [Step 6: Prepare Training Dataset](#step-6-prepare-training-dataset)
  - [Step 7: Launch Fine-Tuning Job](#step-7-launch-fine-tuning-job)
  - [Step 8: Deploy and Test Model](#step-8-deploy-and-test-model)
  - [Complete End-to-End Example](#complete-end-to-end-example)
  - [Best Practices](#best-practices-2)
  - [Troubleshooting](#troubleshooting-1)

### Part 8: Foundational Concepts
- [Understanding Layers and LoRA Application](#understanding-layers-and-lora-application)
  - [What are "Layers" in LLMs?](#what-are-layers-in-llms)
  - [Inside One Transformer Block](#inside-one-transformer-block)
  - [How LoRA Scales with Layers](#how-lora-scales-with-layers)
  - [How LoRA Adapters Work: The A and B Matrices](#how-lora-adapters-work-the-a-and-b-matrices)
  - [Configuration Comparison](#configuration-comparison)
  - [Key Takeaways](#key-takeaways)
- [Transformer Block Internals: MHA, FFN, and Loss](#transformer-block-internals-mha-ffn-and-loss)
  - [Component Overview](#component-overview)
  - [1. Attention Scores: How Words Find Related Words](#1-attention-scores-how-words-find-related-words)
  - [2. Shape Flow Through Transformer](#2-shape-flow-through-transformer)
  - [3. FFN Purpose: The Expand-Compress Pattern](#3-ffn-purpose-the-expand-compress-pattern)
  - [4. Where Loss is Computed: End-to-End Flow](#4-where-loss-is-computed-end-to-end-flow)
  - [5. Backpropagation: How ALL Layers Learn](#5-backpropagation-how-all-layers-learn)
  - [6. LoRA Application in Transformer Blocks](#6-lora-application-in-transformer-blocks)
  - [7. How Loss Function is Computed](#7-how-loss-function-is-computed)
  - [Key Takeaways](#key-takeaways-1)

---

## Introduction

### What is Dataset Formatting?

When fine-tuning a language model with NVIDIA NeMo, you need to prepare your training data in a specific format so the model can learn from it effectively. Think of it as organizing your teaching materials in a way the student (the model) can understand.

### Why is Format Important?

- **Consistency**: Models learn better from consistently formatted data
- **Compatibility**: NeMo requires specific formats to process your data
- **Training Efficiency**: Properly formatted data reduces errors and training time

### Key Requirement: JSONL Format

**All NeMo training datasets must be in JSONL format** (JSON Lines).

**What is JSONL?**
- Each line in the file is a complete, valid JSON object
- One training example per line
- File extension: `.jsonl`

**Example JSONL file structure:**
```jsonl
{"messages": [{"role": "user", "content": "Hello"}]}
{"messages": [{"role": "user", "content": "How are you?"}]}
{"messages": [{"role": "user", "content": "Goodbye"}]}
```

---

## Chat Models Dataset Format

### What are Chat Models?

Chat models are designed for conversational interactions (like ChatGPT, Claude, etc.). They understand multi-turn conversations with different roles.

### Basic Chat Format Structure

Each line must contain a `messages` array with role-based conversation:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Understanding Roles

| Role | Purpose | When to Use |
|------|---------|-------------|
| **system** | Sets the behavior/context for the assistant | First message (optional but recommended) |
| **user** | Represents the human input/question | User queries or prompts |
| **assistant** | The model's response | What you want the model to learn to say |

### Important Rules

1. ✅ **Assistant message should be LAST** in the sequence
2. ✅ **System message (if present) should be FIRST**
3. ✅ The model learns to generate the assistant's responses

### Chat Format Examples

#### Example 1: Simple Q&A

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "What is 5 + 3?"},
    {"role": "assistant", "content": "5 + 3 = 8"}
  ]
}
```

#### Example 2: Multi-turn Conversation

```json
{
  "messages": [
    {"role": "system", "content": "You are a friendly customer support agent."},
    {"role": "user", "content": "I forgot my password."},
    {"role": "assistant", "content": "I can help you reset it. What's your email?"},
    {"role": "user", "content": "user@example.com"},
    {"role": "assistant", "content": "I've sent a password reset link to user@example.com. Please check your inbox."}
  ]
}
```

#### Example 3: Without System Message

```json
{
  "messages": [
    {"role": "user", "content": "Translate 'hello' to Spanish."},
    {"role": "assistant", "content": "Hola"}
  ]
}
```

### Full JSONL File Example for Chat

**File: `chat_training.jsonl`**
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence, which refers to computer systems that can perform tasks that typically require human intelligence."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is ML?"}, {"role": "assistant", "content": "ML stands for Machine Learning, a subset of AI that enables systems to learn from data without being explicitly programmed."}]}
{"messages": [{"role": "user", "content": "Define deep learning."}, {"role": "assistant", "content": "Deep learning is a type of machine learning that uses neural networks with multiple layers to learn complex patterns in data."}]}
```

---

## Completion Models Dataset Format

### What are Completion Models?

Completion models are designed for text completion tasks where you provide a prompt and the model generates a continuation. Think of them as "fill in the blank" models.

**Use Cases:**
- Text summarization
- Classification
- Code completion
- Simple text generation

### Basic Completion Format Structure

Each line contains a `prompt` and `completion`:

```json
{"prompt": "Summarize this: The cat sat on the mat.", "completion": "A cat rested on a mat."}
```

### Understanding Fields

| Field | Description | Example |
|-------|-------------|---------|
| **prompt** | The input text you provide | "Translate to French: Hello" |
| **completion** | The desired output the model should generate | " Bonjour" |

### Completion Format Examples

#### Example 1: Text Summarization

```json
{"prompt": "Summarize: Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI applications are improving efficiency and decision-making.", "completion": " AI is revolutionizing global industries by enhancing efficiency and decision-making across sectors like healthcare and finance."}
```

#### Example 2: Classification

```json
{"prompt": "Classify the sentiment: This product is amazing!\nSentiment:", "completion": " Positive"}
```

#### Example 3: Code Completion

```json
{"prompt": "def add_numbers(a, b):", "completion": "\n    return a + b"}
```

#### Example 4: Question Answering

```json
{"prompt": "Question: What is the boiling point of water?\nAnswer:", "completion": " 100 degrees Celsius or 212 degrees Fahrenheit at sea level."}
```

### Full JSONL File Example for Completion

**File: `completion_training.jsonl`**
```jsonl
{"prompt": "Translate to Spanish: Good morning", "completion": " Buenos días"}
{"prompt": "Translate to Spanish: Thank you", "completion": " Gracias"}
{"prompt": "Translate to Spanish: Goodbye", "completion": " Adiós"}
{"prompt": "Classify: I love this movie!\nSentiment:", "completion": " Positive"}
{"prompt": "Classify: This is terrible.\nSentiment:", "completion": " Negative"}
```

---

## Advanced Features

### 1. Reasoning Support (Chain-of-Thought)

Some models like **Llama Nemotron** support detailed reasoning where the model shows its thinking process.

#### Enabling Reasoning

Control via system message:

**Standard Response (No Reasoning):**
```json
{
  "messages": [
    {"role": "system", "content": "detailed thinking off"},
    {"role": "user", "content": "What is 15 * 24?"},
    {"role": "assistant", "content": "360"}
  ]
}
```

**With Reasoning (Step-by-Step):**
```json
{
  "messages": [
    {"role": "system", "content": "detailed thinking on"},
    {"role": "user", "content": "What is 15 * 24?"},
    {"role": "assistant", "content": "<think>Let me break this down: 15 * 24 = 15 * 20 + 15 * 4 = 300 + 60 = 360</think> The answer is 360."}
  ]
}
```

**When to Use:**
- Complex problem-solving tasks
- Mathematical reasoning
- Multi-step logical deductions
- Educational applications where showing work is important

### 2. Tool Calling (Function Calling)

Enable your model to call external functions/APIs. Useful for:
- Database queries
- API integrations
- Calculator functions
- Web searches

#### Method 1: Inline Tool Definition

Define tools directly in each example:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What's the weather in New York?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_1",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"New York\", \"unit\": \"fahrenheit\"}"
          }
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 72°F"},
    {"role": "assistant", "content": "The weather in New York is sunny and 72°F."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

#### Method 2: Shared Tool Configuration

Instead of repeating tool definitions in every training example, define them once at the dataset level when creating the fine-tuning job.

**Without Shared Configuration (Redundant):**

Imagine you have 1,000 training examples that all use the same tools (calculator, weather). You'd have to repeat the tool definitions in every single example:

```jsonl
{
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Performs mathematical calculations",
        "parameters": {...}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets current weather",
        "parameters": {...}
      }
    }
  ]
}
{
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Performs mathematical calculations",
        "parameters": {...}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets current weather",
        "parameters": {...}
      }
    }
  ]
}
{
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Performs mathematical calculations",
        "parameters": {...}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Gets current weather",
        "parameters": {...}
      }
    }
  ]
}
... (997 more times with the same tool definitions!)
```

This creates:
- **Huge file sizes**: Same tools repeated 1,000 times
- **Hard to update**: Change a tool schema? Update 1,000 places
- **Error-prone**: Inconsistencies across examples

**With Shared Configuration (Efficient):**

Define tools once when creating the fine-tuning job.

**What does "creating the fine-tuning job" mean?**

When you start fine-tuning a model, you make an API call to NVIDIA NeMo to create a "fine-tuning job". This job contains:
1. Your training data file
2. The base model to fine-tune
3. Configuration parameters (like tools)

**How to create a fine-tuning job with shared tools:**

You can use cURL, Python, or the NeMo CLI to send a request to the NeMo fine-tuning API:

**Option 1: Using cURL (Command Line)**

```bash
curl -X POST https://api.nemo.nvidia.com/v1/fine-tuning/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_NVIDIA_API_KEY" \
  -d '{
    "training_file": "file-abc123",
    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
    "dataset_parameters": {
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Gets current weather for a location",
            "parameters": {
              "type": "object",
              "properties": {
                "location": {"type": "string"}
              },
              "required": ["location"]
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "calculator",
            "description": "Performs mathematical calculations",
            "parameters": {
              "type": "object",
              "properties": {
                "expression": {"type": "string"}
              },
              "required": ["expression"]
            }
          }
        }
      ]
    }
  }'
```

**Option 2: Using Python (NeMo SDK)**

```python
import requests

# NeMo API configuration
NEMO_API_URL = "https://api.nemo.nvidia.com/v1/fine-tuning/jobs"
API_KEY = "YOUR_NVIDIA_API_KEY"

# Create fine-tuning job with shared tools
payload = {
    "training_file": "file-abc123",  # ID of your uploaded training file
    "model": "nvidia/llama-3.1-nemotron-70b-instruct",
    "dataset_parameters": {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Performs mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.post(NEMO_API_URL, json=payload, headers=headers)
job = response.json()

print(f"Fine-tuning job created: {job['id']}")
```

**Key Point:** The `dataset_parameters.tools` section is where you define the tools ONCE. These tools will be automatically applied to ALL examples in your training file.

**Now your training data is clean and simple:**

```jsonl
{
  "messages": [
    {"role": "user", "content": "What's 25 * 4?"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_1",
          "function": {
            "name": "calculator",
            "arguments": "{\"expression\": \"25 * 4\"}"
          }
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "100"},
    {"role": "assistant", "content": "The result is 100."}
  ]
}
{
  "messages": [
    {"role": "user", "content": "Weather in Paris?"},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_2",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Paris\"}"
          }
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "Sunny, 22°C"},
    {"role": "assistant", "content": "It's sunny and 22°C in Paris."}
  ]
}
```

Notice: **No tool definitions in the training data!** They're defined once in `dataset_parameters`.

**Benefits:**
- **Reduced file size**: No repeated tool definitions across thousands of examples
- **Easier maintenance**: Update tool schemas in one place
- **Consistency**: All examples use the same tool definitions
- **Cost efficient**: Faster uploads and processing

**When to use:**
- Multiple training examples using the same toolset
- Large datasets (100+ examples with tools)
- Fixed set of tools for a specific use case

### 3. Multi-Modal Support (Images, etc.)

Some models support images alongside text. Format varies by model capability.

---

## Best Practices

### 1. Data Quality Over Quantity

- **Better**: 100 high-quality, diverse examples
- **Worse**: 1000 repetitive or low-quality examples

### 2. Consistency is Key

✅ **Good - Consistent formatting:**
```jsonl
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]}
```

❌ **Bad - Inconsistent structure:**
```jsonl
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "Bye"}]}
```

### 3. Message Order

- **System message**: Always first (if used)
- **Assistant message**: Always last
- **Alternating user/assistant**: Preferred for conversations

### 4. Complete Context

Include all necessary context in the system message:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a medical assistant. Provide information based on verified medical knowledge. Always recommend consulting healthcare professionals for serious concerns."
    },
    {"role": "user", "content": "What causes headaches?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

### 5. One Example Per Line

Each line must be a complete, valid JSON object:

✅ **Correct:**
```jsonl
{"messages": [{"role": "user", "content": "Hello"}]}
{"messages": [{"role": "user", "content": "Hi"}]}
```

❌ **Wrong:**
```json
[
  {"messages": [{"role": "user", "content": "Hello"}]},
  {"messages": [{"role": "user", "content": "Hi"}]}
]
```

### 6. Escaping Special Characters

Use proper JSON escaping:

```json
{
  "messages": [
    {"role": "user", "content": "What does \"AI\" stand for?"},
    {"role": "assistant", "content": "AI stands for \"Artificial Intelligence\"."}
  ]
}
```

### 7. Validation Before Training

**Python validation script:**
```python
import json

def validate_jsonl(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                # Validate structure
                if 'messages' in data:  # Chat format
                    assert isinstance(data['messages'], list)
                    assert data['messages'][-1]['role'] == 'assistant'
                elif 'prompt' in data and 'completion' in data:  # Completion format
                    assert isinstance(data['prompt'], str)
                    assert isinstance(data['completion'], str)
                else:
                    print(f"Line {i}: Unknown format")
            except json.JSONDecodeError:
                print(f"Line {i}: Invalid JSON")
            except AssertionError as e:
                print(f"Line {i}: Structure error - {e}")

validate_jsonl('training_data.jsonl')
```

---

## Common Mistakes to Avoid

### Mistake 1: Using Regular JSON Instead of JSONL

❌ **Wrong:**
```json
[
  {"messages": [...]},
  {"messages": [...]}
]
```

✅ **Correct:**
```jsonl
{"messages": [...]}
{"messages": [...]}
```

### Mistake 2: Missing Assistant Response

❌ **Wrong:**
```json
{"messages": [{"role": "user", "content": "Hello"}]}
```

✅ **Correct:**
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
```

### Mistake 3: Wrong Role Order

❌ **Wrong:**
```json
{
  "messages": [
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Hi"}
  ]
}
```

✅ **Correct:**
```json
{
  "messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ]
}
```

### Mistake 4: Mixing Chat and Completion Formats

❌ **Wrong:**
```jsonl
{"messages": [{"role": "user", "content": "Hello"}]}
{"prompt": "Hi", "completion": " there"}
```

✅ **Correct - Pick one format:**
```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]}
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
```

### Mistake 5: Empty or Missing Content

❌ **Wrong:**
```json
{"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "Response"}]}
```

✅ **Correct:**
```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is Artificial Intelligence."}]}
```

### Mistake 6: Not Using Shared Tools When Available

For large datasets with repeated tool definitions, use shared configuration to reduce file size and maintain consistency.

---

## Choosing the Right Format

### Use Chat Format When:
- Building conversational AI
- Need multi-turn dialogues
- Want role-based context (system, user, assistant)
- Implementing chatbots or virtual assistants

### Use Completion Format When:
- Doing text completion tasks
- Building classifiers
- Simple prompt-response pairs
- Summarization or translation tasks

---

## Inference Endpoints

After fine-tuning, use the appropriate endpoint:

| Model Type | Endpoint | Example |
|------------|----------|---------|
| **Chat Models** | `/chat/completions` | Conversational AI |
| **Completion Models** | `/completions` | Text completion |

---

## Quick Reference

### Chat Format Template
```json
{
  "messages": [
    {"role": "system", "content": "<system_instruction>"},
    {"role": "user", "content": "<user_input>"},
    {"role": "assistant", "content": "<desired_response>"}
  ]
}
```

### Completion Format Template
```json
{"prompt": "<input_text>", "completion": "<desired_output>"}
```

### Validation Checklist

- [ ] File is in `.jsonl` format
- [ ] Each line is valid JSON
- [ ] One example per line
- [ ] Chat format: assistant message is last
- [ ] Chat format: system message is first (if present)
- [ ] No empty content fields
- [ ] Consistent format throughout file
- [ ] Special characters properly escaped
- [ ] File tested with validation script

---

## Additional Resources

- [NeMo Official Documentation](https://docs.nvidia.com/nemo/)
- [OpenAI Message Format Specification](https://platform.openai.com/docs/api-reference/chat)
- [JSON Lines Format](https://jsonlines.org/)

---


# NeMo Fine-Tuning Guide

This guide covers two main approaches to fine-tuning models using NVIDIA NeMo Microservices: LoRA (Low-Rank Adaptation) and SFT (Supervised Fine-Tuning).

---

# LoRA Model Customization

https://docs.nvidia.com/nemo/microservices/latest/fine-tune/tutorials/lora-customization-job.html

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a smart way to teach an LLM new skills without retraining the entire model.

### Simple Analogy
Think of a base model as a skilled employee. Instead of complete retraining for specialized tasks, you give them a small "cheat sheet" (LoRA adapter). They keep general knowledge while gaining new expertise.

### How It Works
- **Base Model**: Frozen/unchanged (like a reference book)
- **LoRA Adapter**: Small trainable module - only 2-3% of original size
- **Result**: New behavior without changing the original

### Quick Comparison

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Training Time | Days/Weeks | Hours |
| GPU Memory | 80GB+ | 16-24GB |
| Storage | 100GB+ | 10-50MB |
| Reversible | No | Yes |

### When to Use LoRA
✅ Limited GPU resources, quick experiments, multiple task versions, budget-conscious
❌ Complete behavior change needed, unlimited resources available

### NeMo Customizer

**NeMo Customizer** is NVIDIA's dedicated microservice for managing fine-tuning operations.

**What It Does**:
1. **Job Management** - Create and manage LoRA/SFT fine-tuning jobs
2. **Configuration** - Store hyperparameters and training configs
3. **Monitoring** - Track progress, metrics, and job status
4. **Model Registry** - Manage fine-tuned model artifacts

**Endpoint (CUSTOMIZER_BASE_URL)**:
- **Production**: `https://api.nemo.nvidia.com/customizer/v1` (NVIDIA Cloud)
- **Local**: `http://localhost:8080` (with port-forwarding to Kubernetes)
- **Enterprise**: `https://nemo-customizer.your-company.com` (self-hosted)

**Key API Operations**:
```bash
# List configurations
GET ${CUSTOMIZER_BASE_URL}/v1/customization/configs

# Create fine-tuning job
POST ${CUSTOMIZER_BASE_URL}/v1/customization/jobs

# Check job status
GET ${CUSTOMIZER_BASE_URL}/v1/customization/jobs/{job_id}

# List all jobs
GET ${CUSTOMIZER_BASE_URL}/v1/customization/jobs
```

**NeMo Microservices Architecture**

NeMo consists of separate, independent microservices working together:

| Service | Phase | Purpose |
|---------|-------|---------|
| **NeMo Customizer** | Training | Fine-tuning job management |
| **Entity Store** | Organization | Namespace/project management |
| **Data Store** | Storage | Dataset/model storage (S3) |
| **NIM** | Inference | Serving fine-tuned models |
| **Deployment** | Orchestration | Model deployment |

**Key Distinction**:
- **NeMo Customizer** ≠ **NIM** (they are separate services)
- Customizer handles **training/fine-tuning**
- NIM handles **inference/serving**

**Typical Workflow**:
```
1. Data Store → Upload training datasets
2. Customizer → Create & run fine-tuning jobs
3. Entity Store → Manage organization/namespaces
4. NIM → Serve fine-tuned models for inference
5. Deployment → Deploy models to production
```

**Detailed Architecture**

```
┌─────────────────────────────────────────────────────┐
│                 NeMo Platform                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. NeMo Customizer (CUSTOMIZER_BASE_URL)          │
│     - Fine-tuning jobs (LoRA, SFT)                 │
│     - Training orchestration                        │
│     - Job management                                │
│     Port: 8080                                      │
│                                                     │
│  2. Entity Store (ENTITY_STORE_BASE_URL)           │
│     - Namespace management                          │
│     - Project organization                          │
│     - Metadata storage                              │
│                                                     │
│  3. Data Store (DATA_STORE_BASE_URL)               │
│     - Dataset storage (S3/MinIO)                   │
│     - Model artifacts                               │
│     - Checkpoints                                   │
│     Port: 9000                                      │
│                                                     │
│  4. NIM - Inference (NIM_BASE_URL)                 │
│     - Model serving                                 │
│     - Inference endpoints                           │
│     - Chat completions                              │
│     Port: 8000                                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```
---

## Understanding Layers and LoRA Application

### What are "Layers" in LLMs?

**80 layers = 80 stacked transformer blocks**

```
┌──────────────────────────────┐
│  Layer 80 (Top)              │  Each layer contains:
├──────────────────────────────┤  • Multi-Head Attention (MHA)
│  Layer 79                    │  • Feed-Forward Network (FFN)
├──────────────────────────────┤  • Layer Normalization
│  Layer 78                    │
├──────────────────────────────┤
│         ...                  │
├──────────────────────────────┤
│  Layer 2                     │
├──────────────────────────────┤
│  Layer 1 (Bottom)            │
└──────────────────────────────┘
```

**Common Models:**

| Model | Total Layers | Components per Layer |
|-------|--------------|---------------------|
| Llama 2 7B | 32 | MHA (4 modules) + FFN (3 modules) |
| Llama 2 70B | 80 | MHA (4 modules) + FFN (3 modules) |
| GPT-3 175B | 96 | MHA (4 modules) + FFN (3 modules) |

**What are "modules"?** These are specific projection layers:
- **MHA modules (4)**: `q_proj`, `k_proj`, `v_proj`, `o_proj` (Query, Key, Value, Output projections)
- **FFN modules (3)**: `up_proj`, `gate_proj`, `down_proj` (Expansion and compression layers)

> **Note:** "Modules" refers to the specific weight matrices (linear layers) inside MHA and FFN:
> - **MHA modules (4):** `q_proj`, `k_proj`, `v_proj`, `o_proj`
> - **FFN modules (3):** `up_proj`, `gate_proj`, `down_proj`

---

### Inside One Transformer Block

```
┌─────────────────────────────────────────────┐
│         SINGLE TRANSFORMER LAYER            │
├─────────────────────────────────────────────┤
│                                             │
│  ┌────────────────────────────────────┐    │
│  │  Multi-Head Attention (MHA)        │    │
│  │  ────────────────────────────      │    │
│  │  • q_proj  ← LoRA adapter here    │    │
│  │  • k_proj  ← LoRA adapter here    │    │
│  │  • v_proj  ← LoRA adapter here    │    │
│  │  • o_proj  ← LoRA adapter here    │    │
│  └────────────────────────────────────┘    │
│              ↓                              │
│  ┌────────────────────────────────────┐    │
│  │  Feed-Forward Network (FFN)        │    │
│  │  ──────────────────────────         │    │
│  │  • up_proj    ← LoRA adapter here  │    │
│  │  • gate_proj  ← LoRA adapter here  │    │
│  │  • down_proj  ← LoRA adapter here  │    │
│  └────────────────────────────────────┘    │
│                                             │
└─────────────────────────────────────────────┘
```

**7 target modules per layer** = MHA (4) + FFN (3)

---

### How LoRA Scales with Layers

**Key Concept:** LoRA adapters are added to **ALL layers**!

```
Visual: 80-Layer Model with 7 Target Modules

Layer 80:  [q] [k] [v] [o] [up] [gate] [down]  ← 7 LoRA adapters
           LoRA LoRA LoRA LoRA LoRA LoRA LoRA

Layer 79:  [q] [k] [v] [o] [up] [gate] [down]  ← 7 LoRA adapters
           LoRA LoRA LoRA LoRA LoRA LoRA LoRA

Layer 78:  [q] [k] [v] [o] [up] [gate] [down]  ← 7 LoRA adapters
           LoRA LoRA LoRA LoRA LoRA LoRA LoRA

   ⋮          ⋮    ⋮    ⋮    ⋮    ⋮     ⋮     ⋮

Layer 2:   [q] [k] [v] [o] [up] [gate] [down]  ← 7 LoRA adapters
           LoRA LoRA LoRA LoRA LoRA LoRA LoRA

Layer 1:   [q] [k] [v] [o] [up] [gate] [down]  ← 7 LoRA adapters
           LoRA LoRA LoRA LoRA LoRA LoRA LoRA

───────────────────────────────────────────────
Total:     80 layers × 7 modules = 560 LoRA adapters
```

**Calculation Example:**
```
Llama 2 70B (80 layers):
  Target modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                   "up_proj", "gate_proj", "down_proj"]

  Total LoRA modules = 80 × 7 = 560 adapters
  Trainable params = ~146M (only 0.2% of 70B!)
```

---

### How LoRA Adapters Work: The A and B Matrices

**For Beginners:** Instead of updating millions of weights in the base model, LoRA adds small "adapter" matrices.

**Each module gets its own trainable A and B matrices:**

```
Original weight matrix (FROZEN):
W ∈ ℝ^(d×d)  (e.g., 4096×4096 = 16.7M parameters)

LoRA adds two small matrices:
A ∈ ℝ^(d×r)  (e.g., 4096×16 = 65K parameters)
B ∈ ℝ^(r×d)  (e.g., 16×4096 = 65K parameters)

where r (rank) is MUCH smaller than d (e.g., r=16 vs d=4096)
```

**How they work together:**

```
┌──────────────────────────────────────────────────────┐
│  Before LoRA (Full Fine-Tuning):                     │
│                                                       │
│  Input ──→ [W] ──→ Output                           │
│            ↑                                          │
│        Update ALL weights (expensive!)               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  With LoRA (Parameter-Efficient):                    │
│                                                       │
│  Input ──→ [W] ──→ Add ──→ Output                   │
│         (FROZEN)     ↑                                │
│                      │                                │
│            Input ──→[A]──→[B]                        │
│                     ↑      ↑                          │
│                  Train  Train                         │
│                  only   only                          │
│                  these! these!                        │
└──────────────────────────────────────────────────────┘
```

**Mathematical Formula:**
```
Output = (W + B·A) × Input

Where:
• W = Frozen base model weights
• B·A = Low-rank update (ΔW)
• B·A is MUCH smaller than W
```

**How adapters are added to the final model:**

**During Training:**
- Base model weights (W) stay frozen ❄️
- Only A and B matrices are trained 🔥
- Forward pass: `output = W(input) + B(A(input))`

**After Training (Merging):**
```python
# Option 1: Keep separate (for switching tasks)
output = base_model(input) + lora_adapter(input)

# Option 2: Merge into base model (deployment)
W_new = W_frozen + (B @ A)  # Matrix multiplication
# Now you have a single merged model!
```

**Real Example (q_proj in one layer):**
```
Base q_proj weight: 4096 × 4096 = 16,777,216 params (FROZEN)

LoRA adapter:
  Matrix A: 4096 × 16 = 65,536 params (trainable)
  Matrix B: 16 × 4096 = 65,536 params (trainable)
  Total trainable: 131,072 params

Efficiency: 131K / 16.7M = 0.78% of original size!
```

**Key Benefits:**
- ✅ Base model never changes (can reuse for multiple tasks)
- ✅ Adapters are tiny (easy to store/share)
- ✅ Can merge adapters into base model after training
- ✅ Can swap different adapters for different tasks

---

### Configuration Comparison

**Choose based on task complexity:**

| Configuration | Modules | Rank (r) | LoRA Adapters<br>(80 layers) | Params | Best For |
|---------------|---------|----------|------------------------------|--------|----------|
| **Minimal** | q_proj, v_proj | 8 | 2 × 80 = 160 | ~42M | Simple classification |
| **Standard** ⭐ | q, k, v, o_proj | 16 | 4 × 80 = 320 | ~84M | Most tasks |
| **Full** | All 7 modules | 32 | 7 × 80 = 560 | ~146M | Complex reasoning |

**Visual Comparison:**
```
Minimal:     Layer: [q✓] [k ] [v✓] [o ] [up ] [gate ] [down ]
             80×    LoRA      LoRA
             = 160 adapters

Standard:    Layer: [q✓] [k✓] [v✓] [o✓] [up ] [gate ] [down ]
             80×    LoRA LoRA LoRA LoRA
             = 320 adapters

Full:        Layer: [q✓] [k✓] [v✓] [o✓] [up✓] [gate✓] [down✓]
             80×    LoRA LoRA LoRA LoRA LoRA  LoRA   LoRA
             = 560 adapters
```

---

### Key Takeaways

✅ **80 layers** = 80 copies of the same transformer structure
✅ **Each layer** has 7 potential target modules (4 MHA + 3 FFN)
✅ **LoRA applies to ALL layers** - not just one!
✅ **Still efficient**: 560 adapters = only 0.2% of base model size
✅ **More adapters** = more capacity, but still parameter-efficient

---

## Transformer Block Internals: MHA, FFN, and Loss

### Component Overview

| **Component** | **Purpose** | **Input Shape** | **Output Shape** | **Computes Loss?** |
|---------------|-------------|-----------------|------------------|--------------------|
| **MHA** | Gathers context from related tokens | `[n × d]` | `[n × d]` | ❌ No |
| **FFN** | Transforms each token individually | `[n × d]` | `[n × d]` | ❌ No |
| **Add & Norm** | Stabilizes training (residual + normalize) | `[n × d]` | `[n × d]` | ❌ No |
| **Final Linear** | Projects to vocabulary space | `[n × d]` | `[n × vocab_size]` | ❌ No |
| **Loss Layer** | Compares prediction with true label | `[vocab_size]` | Scalar | ✅ **YES** |

Where: `n` = sequence length (tokens), `d` = hidden dimension (768, 4096, etc.)

---

### 1. Attention Scores: How Words Find Related Words

**Purpose**: MHA helps each token gather context from related tokens.

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
- `QK^T` dot product → Higher scores for related words
- Model **learns** these relationships during training
- "cat" + "sat" have high correlation (subject-verb)
- Output: `[n × d]` rich vectors, NOT `[n × 1]` scalars!

---

### 2. Shape Flow Through Transformer

**Question**: Does each token output a single number?

**Answer**: ❌ **No!** Each token maintains a **d-dimensional vector** (e.g., 768 or 4096 dimensions).

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

### 3. FFN Purpose: The Expand-Compress Pattern

**Purpose**: FFN is a **2-layer neural network** that processes each token independently.

**FFN Architecture:**

```
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

**Why Both MHA and FFN Are Needed:**

| **Aspect** | **MHA** | **FFN** |
|------------|---------|---------|
| **Role** | Communication between tokens | Computation within each token |
| **Operation** | Token attends to OTHER tokens | Token processes ITS OWN representation |
| **Analogy** | Reading a book (gathering info) | Thinking about it (processing) |
| **Formula** | `Attention(Q,K,V)` | `Linear₂(ReLU(Linear₁(x)))` |

**Without FFN**: ❌ Tokens gather context but can't process it (~30% accuracy drop)
**Without MHA**: ❌ No context understanding (~50% accuracy drop)
**With Both**: ✅ Full model capability

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

### 6. LoRA Application in Transformer Blocks

When you apply LoRA to `["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]`:

**MHA Modules** (4 modules per layer):
- `q_proj`, `k_proj`, `v_proj`: Create Query, Key, Value matrices
- `o_proj`: Output projection after attention

**FFN Modules** (3 modules per layer):
- `up_proj`: Expand dimension (d → 4d)
- `gate_proj`: Gating mechanism (for some architectures)
- `down_proj`: Compress dimension (4d → d)

**Visual: LoRA in One Layer**

```
┌────────────────────────────────────────────────┐
│              TRANSFORMER LAYER                 │
├────────────────────────────────────────────────┤
│                                                │
│  Multi-Head Attention (MHA)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ q_proj   │  │ k_proj   │  │ v_proj   │    │
│  │ [W+B·A] │  │ [W+B·A] │  │ [W+B·A] │    │
│  │  LoRA   │  │  LoRA   │  │  LoRA   │    │
│  └──────────┘  └──────────┘  └──────────┘    │
│         ↓           ↓           ↓              │
│  ┌──────────────────────────────────────┐     │
│  │      Attention Computation           │     │
│  └──────────────────────────────────────┘     │
│         ↓                                      │
│  ┌──────────┐                                 │
│  │ o_proj   │                                 │
│  │ [W+B·A] │  ← LoRA adapter                 │
│  └──────────┘                                 │
│                                                │
├────────────────────────────────────────────────┤
│                                                │
│  Feed-Forward Network (FFN)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ up_proj  │  │gate_proj │  │down_proj │    │
│  │ [W+B·A] │  │ [W+B·A] │  │ [W+B·A] │    │
│  │  LoRA   │  │  LoRA   │  │  LoRA   │    │
│  └──────────┘  └──────────┘  └──────────┘    │
│                                                │
└────────────────────────────────────────────────┘
           7 LoRA adapters per layer
```

**For 80-layer model**: 80 × 7 = 560 LoRA adapters

**How LoRA Helps with Training:**
- ✅ Base weights (W) stay **frozen** → preserve pre-trained knowledge
- ✅ Only small adapters (B·A) are **trainable** → efficient fine-tuning
- ✅ Adapters applied to **both** MHA and FFN → fine-tune communication AND computation
- ✅ Gradients flow through adapters during backpropagation → all 560 adapters learn!

---

### 7. How Loss Function is Computed

**Core Concept**: Model predicts next token, compares with actual token, computes error.

**Example: Input "The Cat" → Predict "sat"**

```
┌─────────────────────────────────────────────────┐
│           PREDICTION PROCESS                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Input Tokens: ["The", "Cat"]                  │
│  Target Token: "sat"                            │
│                                                 │
│  Step 1: Process Input                         │
│  ┌───────┐  ┌───────┐                          │
│  │  The  │  │  Cat  │                          │
│  └───┬───┘  └───┬───┘                          │
│      │          │                               │
│      ↓          ↓                               │
│  ┌─────────────────────────┐                   │
│  │  80 Transformer Layers  │                   │
│  │  (MHA + FFN in each)   │                   │
│  └─────────────────────────┘                   │
│            ↓                                    │
│  [2 × 4096] hidden states                      │
│            ↓                                    │
│  Step 2: Project to Vocabulary                 │
│  ┌─────────────────────────┐                   │
│  │  Linear: [2×4096]       │                   │
│  │       ↓                 │                   │
│  │  [2 × 50000] logits     │                   │
│  └─────────────────────────┘                   │
│            ↓                                    │
│  Extract LAST token's logits:                  │
│  Position "Cat" → [50,000 scores]              │
│            ↓                                    │
│  Step 3: Convert to Probabilities              │
│  ┌─────────────────────────┐                   │
│  │  Softmax                │                   │
│  └─────────────────────────┘                   │
│            ↓                                    │
│  P("sat")  = 0.15  ← Model's prediction        │
│  P("dog")  = 0.40  ← Wrong! (highest)          │
│  P("ran")  = 0.25                              │
│  P("sat")  = 0.15                              │
│  ...                                            │
│            ↓                                    │
│  Step 4: Compute Loss (Cross-Entropy)          │
│  ┌─────────────────────────┐                   │
│  │  True Label: "sat"      │                   │
│  │  Predicted P("sat")=0.15│                   │
│  │                         │                   │
│  │  Loss = -log(0.15)      │                   │
│  │       = 1.90 (HIGH!)    │                   │
│  └─────────────────────────┘                   │
│                                                 │
│  High loss = Bad prediction!                   │
│  Model predicted "dog" (0.40) but answer is    │
│  "sat" (0.15)                                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Loss Formula:**

```
Loss = -log(P(correct_token))

If P("sat") = 0.15:  Loss = -log(0.15) = 1.90  ❌ High (bad)
If P("sat") = 0.80:  Loss = -log(0.80) = 0.22  ✅ Low (good)
If P("sat") = 0.99:  Loss = -log(0.99) = 0.01  ✅ Very low (excellent)
```

**Training Goal**: Minimize loss by increasing probability of correct token.

**Visual Comparison:**

```
Before Training:               After Training:
─────────────────              ─────────────────
Input: "The Cat"               Input: "The Cat"
Prediction: "dog" (40%)        Prediction: "sat" (85%)
Actual: "sat" (15%)            Actual: "sat" (85%)
Loss = 1.90 ❌ HIGH            Loss = 0.16 ✅ LOW

Model learns to predict "sat" correctly!
```

**Multiple Tokens Example:**

```
Sequence: "The cat sat on"
Target:   "cat sat on the"

Position 1: "The" → Predict "cat"   → Loss₁
Position 2: "cat" → Predict "sat"   → Loss₂
Position 3: "sat" → Predict "on"    → Loss₃
Position 4: "on"  → Predict "the"   → Loss₄

Total Loss = Average(Loss₁, Loss₂, Loss₃, Loss₄)
```

**Key Points:**
- ✅ Loss computed **once per forward pass** (at the end)
- ✅ Compares model's prediction vs. actual next token
- ✅ Lower loss = better prediction
- ✅ Backpropagation uses this loss to update ALL layers

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
| **Why both MHA & FFN?** | MHA = communication, FFN = computation (both essential) |
| **LoRA adapters learn?** | ✅ Yes - all 560 adapters receive gradients from single loss |

**Mental Model:**
- **Loss**: Computed once at the end (like a final exam score)
- **Backpropagation**: Feedback distributed to ALL layers (like reviewing all your mistakes)
- **FFN/MHA**: Workers that ALL receive feedback and improve
- **LoRA**: Adds small trainable adapters to each worker instead of retraining entire workers

---

## Prerequisites

### Required Access
- Access to NeMo Customizer microservice
- Completed entity management setup (namespaces and projects)
- Understanding of organizational structure in NeMo

### Technical Setup
- Python environment with `huggingface_hub` package installed
- Environment variables configured
- Optional: Weights & Biases account for monitoring

### Environment Variables

You need to set the following environment variables:

```bash
# Core service endpoints
export CUSTOMIZER_BASE_URL="<your-customizer-service-url>"
export ENTITY_STORE_BASE_URL="<your-entity-store-url>"
export DATA_STORE_BASE_URL="<your-data-store-url>"

# Organization identifiers
export NAMESPACE="<your-namespace>"
export DATASET_NAME="<your-dataset-name>"

# Hugging Face integration
export HF_ENDPOINT="<huggingface-endpoint>"
export HF_TOKEN="<your-huggingface-token>"

# Optional: Weights & Biases monitoring
export WANDB_API_KEY="<your-wandb-api-key>"
```

---

## Step 1: Model Selection

### Initialize Client

**Using Python SDK (Recommended - Unified Client):**
```python
from nemo_microservices import NeMoMicroservices
import os

# Initialize unified NeMo client
client = NeMoMicroservices(base_url=os.environ['CUSTOMIZER_BASE_URL'])
```

### Find Available LoRA Configurations

Query the available configurations that support LoRA fine-tuning:

**Using Python SDK:**
```python
# Filter for LoRA-compatible configurations
configs = client.customization.configs.list(
    filter={"finetuning_type": "lora"}
)

# Review available configurations
for config in configs:
    print(f"Model: {config.name}")
    print(f"GPU Requirements: {config.gpu_requirements}")
    print(f"Dataset Schema: {config.dataset_schema}")
```

**Using cURL:**
```bash
curl -X GET "${CUSTOMIZER_BASE_URL}/v1/customization/configs?filter[finetuning_type]=lora" \
  -H "Authorization: Bearer ${API_KEY}"
```

### Review Configuration Details

Each configuration provides:
- **GPU Requirements**: Number and type of GPUs needed
- **Dataset Schema**: Expected data format (typically prompt/completion pairs)
- **Training Options**: Available hyperparameters

---

## Step 2: Dataset Preparation

### Model Selection for LoRA

**Always use chat/instruct models** (not base models):

✅ **Recommended**: `llama-3.1-nemotron-70b-instruct`, `mistral-large-instruct`

❌ **Avoid**: Base models like `llama-3.1-70b-base`

**Why?** Chat models are pre-aligned for instructions, understand conversation structure, and provide better starting points.

### Data Format

**Preferred: Chat format** (for instruct models):

**train.jsonl:**
```json
{"messages": [{"role": "system", "content": "You are a professional email writer."}, {"role": "user", "content": "Write an email to request a meeting"}, {"role": "assistant", "content": "Subject: Meeting Request\n\nDear [Name],\n\nI hope this email finds you well..."}]}
{"messages": [{"role": "system", "content": "You are a professional email writer."}, {"role": "user", "content": "Decline an invitation politely"}, {"role": "assistant", "content": "Subject: Re: Invitation\n\nDear [Name],\n\nThank you for the kind invitation..."}]}
```

**validation.jsonl:**
```json
{"messages": [{"role": "system", "content": "You are a professional email writer."}, {"role": "user", "content": "Write a thank you email"}, {"role": "assistant", "content": "Subject: Thank You\n\nDear [Name],\n\nI wanted to express my sincere gratitude..."}]}
```

**Alternative: Completion format** (legacy):
```json
{"prompt": "Write a meeting request", "completion": "Subject: Meeting Request..."}
```

### Important Data Rules

1. **Use chat format** with instruct models (system/user/assistant roles)
2. **One record per line**: Each JSON object on single line
3. **Consistent structure**: Same roles across all examples
4. **Separate files**: training and validation datasets
5. **Sufficient data**: 100+ minimum, 500+ recommended

---

## Step 3: Data Upload

### Understanding the Storage Architecture (For Beginners)

Think of NeMo's storage like organizing a research project:

| System | What It Stores | Role in Fine-Tuning | Real-World Analogy |
|--------|----------------|---------------------|-------------------|
| **Entity Store** | Namespaces, projects, permissions | Who can access what | Project directory (folder structure) |
| **Data Store** | Datasets (backup), checkpoints, models | Permanent storage & outputs | External hard drive (backup archive) |
| **Hugging Face** | Dataset files (JSONL) | Primary training data source | Cloud storage (Google Drive - active work) |

### Why Datasets Are Stored in TWO Places

Your training data lives in **both** Data Store and Hugging Face for different reasons:

#### Hugging Face (Primary - Used During Training)
**Purpose**: Active data source for training jobs

- ✅ **When**: Every training job reads from here
- ✅ **Why**: Fast downloads, version control, public/private access
- ✅ **How NeMo uses it**: Downloads JSONL files at job start → loads batches → trains model

**Think of it as**: Your working copy on Google Drive (fast, accessible, shareable)

#### Data Store (Backup - Long-term Archive)
**Purpose**: Permanent backup and company data retention

- ✅ **When**: Data synced here for archival
- ✅ **Why**: Compliance, disaster recovery, company policy
- ✅ **How NeMo uses it**: Fallback if HF unavailable, stores model outputs

**Think of it as**: Your external hard drive backup (permanent, secure, offline)

### Complete Fine-Tuning Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ BEFORE Training (Setup - You do this)                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Create train.jsonl on your computer                 │
│  2. Upload to Hugging Face → hf://namespace/dataset     │
│  3. Auto-synced to Data Store → s3://namespace/dataset  │
│                                                          │
│  Result: Dataset in BOTH places                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ DURING Training (NeMo Customizer does this)             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Job starts                                          │
│  2. Downloads from Hugging Face (using HF_TOKEN)        │
│  3. Caches dataset in training cluster                  │
│  4. Reads batches → GPU → Model training                │
│  5. Saves checkpoints to Data Store every N steps       │
│                                                          │
│  Primary source: Hugging Face ✓                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ AFTER Training (Outputs stored)                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Fine-tuned model → Saved to Data Store              │
│  2. Final checkpoint → Saved to Data Store              │
│  3. Training metrics → Logged (W&B/MLflow)              │
│                                                          │
│  Output location: Data Store (s3://...)                 │
└─────────────────────────────────────────────────────────┘
```

### Why This Two-Storage Design?

| Scenario | Solution |
|----------|----------|
| **Fast training start** | HF provides quick downloads globally |
| **Version control** | HF tracks dataset versions (v1, v2, v3) |
| **Sharing datasets** | HF allows team collaboration easily |
| **Company compliance** | Data Store keeps permanent copies |
| **HF is down** | Data Store serves as backup source |
| **Model outputs** | Data Store keeps fine-tuned models |

### Simple Workflow

```
You → Hugging Face (upload train.jsonl)  ← ONLY manual step
          ↓
    NeMo reads during training             ← Automatic
          ↓
    Also backed up in Data Store           ← Automatic
          ↓
    Fine-tuned model saved in Data Store   ← Automatic
```

**Important**: You ONLY upload to Hugging Face. NeMo Customizer automatically:
- Downloads data from HF when training starts
- Backs up to Data Store (optional, for compliance)
- Manages all caching and storage

**You do NOT manually push to Data Store!**

### Initialize Clients

**Option 1: Using Unified NeMo Client (Recommended):**
```python
from nemo_microservices import NeMoMicroservices
from huggingface_hub import HfApi
import os

# Initialize unified NeMo client (handles all services)
client = NeMoMicroservices(base_url=os.environ['CUSTOMIZER_BASE_URL'])
hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)
```

**Option 2: Using Service-Specific Clients (Legacy):**
```python
from nemo_customizer import Client
from huggingface_hub import HfApi

# Initialize separate clients for each service
entity_client = Client(base_url=ENTITY_STORE_BASE_URL)
data_client = Client(base_url=DATA_STORE_BASE_URL)
hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)
```

**Note**: The examples below show the legacy approach. When using `NeMoMicroservices`, replace:
- `entity_client.namespaces.*` with `client.entity_store.namespaces.*`
- `data_client.namespaces.*` with `client.data_store.namespaces.*`

### Step 1: Create Namespaces

**Purpose**: Organizational units for access control and resource isolation.

```python
# Entity Store: Stores namespace metadata (owner, permissions, projects)
entity_client.namespaces.create(name="my-namespace")

# Data Store: Creates S3 bucket/container for this namespace
data_client.namespaces.create(name="my-namespace")
```

**What gets stored**:
- **Entity Store**: `{"name": "my-namespace", "owner": "user@company.com", "created": "2024-01-01"}`
- **Data Store**: Empty S3 bucket/path: `s3://data-store/my-namespace/`

### Step 2: Create Hugging Face Dataset Repository

**Purpose**: Store actual training data files that NeMo will read during fine-tuning.

```python
# Create private dataset repository on Hugging Face
hf_api.create_repo(
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset",
    private=True  # Keep your data private
)
```

**What gets created**:
- **Hugging Face**: Repository at `https://huggingface.co/datasets/{NAMESPACE}/{DATASET_NAME}`
- **Accessibility**: NeMo Customizer can access this via your `HF_TOKEN`

### Step 3: Upload Training Files to Hugging Face

**Critical**: NeMo reads datasets from Hugging Face, so files must be in specific folders:

```python
# Upload training data to training/ folder
hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",  # Must be in training/ folder
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

# Upload validation data to validation/ folder
hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",  # Must be in validation/ folder
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)
```

**What gets stored**:
- **Hugging Face repository structure**:
  ```
  {NAMESPACE}/{DATASET_NAME}/
  ├── training/
  │   └── train.jsonl       ← NeMo reads this
  └── validation/
      └── validation.jsonl  ← NeMo reads this
  ```

### Step 4: Reference Dataset in Fine-Tuning Job

When creating a fine-tuning job, reference the Hugging Face dataset:

```python
{
  "training_file": "hf://{NAMESPACE}/{DATASET_NAME}/training/train.jsonl",
  "validation_file": "hf://{NAMESPACE}/{DATASET_NAME}/validation/validation.jsonl"
}
```

**How NeMo accesses it**:
1. NeMo Customizer uses your `HF_TOKEN` (from secrets)
2. Downloads files from Hugging Face during job setup
3. Caches data for training
4. Reads batches during fine-tuning

### Summary: What Goes Where

```
┌─────────────────────────────────────────────────────────┐
│ Entity Store (Metadata)                                 │
│ • Namespace: "my-namespace"                             │
│ • Projects: ["project-1", "project-2"]                  │
│ • Permissions: {"owner": "user@company.com"}            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Data Store (S3-compatible)                              │
│ • Namespace path: s3://data-store/my-namespace/         │
│ • Model artifacts: fine-tuned-model.pth                 │
│ • Checkpoints: checkpoint-epoch-3.ckpt                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Hugging Face (Actual Training Data)                     │
│ • Repository: hf://my-namespace/my-dataset              │
│ • Files:                                                │
│   - training/train.jsonl (your training examples)       │
│   - validation/validation.jsonl (your val examples)     │
│ • Access: NeMo reads directly via HF_TOKEN              │
└─────────────────────────────────────────────────────────┘
```

---

### Where Models and Checkpoints Are Saved

Understanding where everything is stored during and after fine-tuning:

#### Base Model (Source)
- **Location**: NVIDIA Model Registry (not your storage)
- **Access**: Referenced via model name (e.g., `nvidia/llama-3.1-nemotron-70b-instruct`)
- **Downloaded**: Temporarily to GPU cluster during training
- **Cost**: No storage cost (not copied to your account)

#### During Training - Checkpoints
**Saved to**: Data Store (S3-compatible)

```
Path: s3://data-store/{NAMESPACE}/jobs/{JOB_ID}/checkpoints/
├── checkpoint-step-100.ckpt   # Every N steps
├── checkpoint-step-200.ckpt
└── checkpoint-final.ckpt      # Training complete
```

**What's saved**: Model weights, optimizer state, training progress
**When**: Every N steps (configurable) + at completion
**Why**: Resume training if it crashes, track progress

#### After Training - Fine-Tuned Model
**Saved to**: Data Store

**Full SFT** (all weights updated):
```
Path: s3://data-store/{NAMESPACE}/models/{MODEL_ID}/
├── model.safetensors      # Complete model (140GB for 70B model)
├── config.json
└── tokenizer.json
```

**LoRA** (adapter only):
```
Path: s3://data-store/{NAMESPACE}/models/{MODEL_ID}/
├── adapter_model.safetensors  # LoRA weights only (10-50MB)
├── adapter_config.json
└── tokenizer.json
```

#### Storage Summary

| Item | Location | Size | When |
|------|----------|------|------|
| Base Model | NVIDIA Registry | 140GB | Referenced (not stored) |
| Checkpoints | Data Store | Full model size | During training |
| Fine-tuned (SFT) | Data Store | 140GB | After training |
| Fine-tuned (LoRA) | Data Store | 10-50MB | After training |
| Training Logs | Data Store | <1MB | During/after |

**Key Points**:
- LoRA saves **massive storage** (50MB vs 140GB)
- Checkpoints enable **training recovery** if interrupted
- Everything stored in **Data Store** (your permanent storage)
- Base model stays in **NVIDIA Registry** (just referenced)

---

## Step 4: Job Creation and Submission

### Construct Configuration URN

The configuration URN must include the version:

**Format:** `name@version+size`

**Example:** `meta/llama-3.2-1b-instruct@v1.0.0+8GB`

**Important**: Omitting the version will cause errors!

### Define Hyperparameters

Create a hyperparameters configuration for LoRA training:

```python
hyperparameters = {
    "training_type": "sft",           # Supervised fine-tuning
    "finetuning_type": "lora",        # Use LoRA adapters
    "epochs": 10,                      # Number of training epochs
    "batch_size": 16,                  # Training batch size
    "learning_rate": 0.0001,           # Learning rate
    "max_seq_length": 2048,            # Maximum sequence length
    "lora": {
        "adapter_dim": 8,              # LoRA adapter dimension
        "adapter_dropout": 0.01        # Dropout for adapters
    }
}
```

### Hyperparameter Explanations

- **epochs**: How many times the model sees the entire dataset
- **batch_size**: Number of examples processed together (lower = less memory)
- **learning_rate**: How quickly the model learns (lower = more stable, higher = faster)
- **max_seq_length**: Maximum input/output length
- **adapter_dim**: LoRA rank (higher = more capacity but more resources)
- **adapter_dropout**: Regularization to prevent overfitting

---

### Advanced Hyperparameters (Optional)

For experienced users who need fine-grained control:

#### 1. Target Modules (Which Layers to Fine-Tune)

By default, NeMo fine-tunes all attention layers. You can specify exactly which to target:

```python
hyperparameters = {
    # ... basic params ...
    "lora": {
        "adapter_dim": 8,
        "target_modules": [
            "q_proj",      # Query projection (attention)
            "k_proj",      # Key projection (attention)
            "v_proj",      # Value projection (attention)
            "o_proj",      # Output projection (attention)
            "gate_proj",   # Feed-forward gate
            "up_proj",     # Feed-forward up projection
            "down_proj"    # Feed-forward down projection
        ]
    }
}
```

**What each module does**:

| Module | Layer Type | Impact on Training | When to Include |
|--------|-----------|-------------------|-----------------|
| `q_proj`, `k_proj`, `v_proj` | Attention (QKV) | How model attends to tokens | Always (default, most important) |
| `o_proj` | Attention output | Attention result processing | High accuracy needs |
| `gate_proj`, `up_proj`, `down_proj` | Feed-forward (FFN) | Factual knowledge storage | Domain-specific knowledge* |

**Why FFN modules for domain knowledge?**

Transformer architecture has two key components with different roles:

**Attention Layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`):
- **Purpose**: Learn *how* to process language
- **What they capture**: Grammar, context, reasoning patterns, relationships between words
- **Example**: Understanding "bank" means financial institution vs river bank based on context

**Feed-Forward Layers (FFN)** (`gate_proj`, `up_proj`, `down_proj`):
- **Purpose**: Store *what* the model knows
- **What they capture**: Facts, concepts, domain terminology, specialized knowledge
- **Example**: Knowing "myocardial infarction" = heart attack, symptoms, treatment

**Real-world example**:
```
Medical domain fine-tuning:

Attention only (q_proj, k_proj, v_proj, o_proj):
Q: "What is myocardial infarction?"
A: "It's a medical term..." (understands it's medical, but lacks specifics)

Attention + FFN (all 7 modules):
Q: "What is myocardial infarction?"
A: "It's a heart attack caused by blocked coronary arteries,
   leading to cardiac tissue death. Symptoms include chest pain,
   shortness of breath..." (precise domain knowledge)
```

**When to include FFN modules**:
✅ Medical/Legal/Finance with specialized terminology
✅ Teaching facts the base model doesn't know
✅ New domains (e.g., rare programming languages, industry jargon)
❌ General conversation style adaptation (attention only is enough)

**Recommendations**:
- **Minimal (fastest)**: `["q_proj", "v_proj"]` - Only query/value
- **Standard (balanced)**: `["q_proj", "k_proj", "v_proj", "o_proj"]` - All attention (most use cases)
- **Maximum (domain-specific)**: All 7 modules - Attention + FFN (medical, legal, technical domains)

**Trade-off**: More modules = better quality but slower training and larger adapters

#### 2. Mixed Precision Training

LoRA supports mixed precision for 2x faster training and 50% memory savings:

```python
hyperparameters = {
    "precision": "bf16-mixed",  # Recommended (default in NeMo)
    "lora": {
        "adapter_dim": 16,
        "adapter_dropout": 0.01
    }
}
```

**Precision Options**:

| Precision | Speed | Memory | GPU Compatibility |
|-----------|-------|--------|-------------------|
| `bf16-mixed` | 2x | 50% less | Modern GPUs (A100, H100, RTX 40xx) - **Recommended** |
| `fp16-mixed` | 2x | 50% less | Older GPUs (V100, RTX 30xx) |
| `32` | 1x | Baseline | CPU/All GPUs - **Required for CPU** |

**How Mixed Precision Works**:

Mixed precision keeps **master weights in FP32** but computes in BF16/FP16:

| Component | Precision | Reason |
|-----------|-----------|--------|
| **Matrix Multiplications** | BF16/FP16 | 2x speed via Tensor Cores |
| **Activations** | BF16/FP16 | 50% memory savings |
| **LayerNorm/BatchNorm** | FP32 | Numerical stability |
| **Softmax/Loss** | FP32 | Prevent underflow/overflow |
| **Master Weights** | FP32 | Optimizer maintains this |
| **Gradient Accumulation** | FP32 | Precise updates |
| **Weight Updates** | FP32 | Prevent precision drift |

**Training Loop**:
```
1. Cast FP32 master weights → BF16/FP16
2. Forward pass: MatMuls + activations in BF16/FP16
3. Backward pass: Gradients computed in BF16/FP16
4. Cast gradients → FP32
5. Update master weights in FP32
6. Repeat
```

**Layer-Specific Precision (bf16-mixed/fp16-mixed)**:
- **Attention layers (Q, K, V, O projections)**: Compute in BF16/FP16
- **FFN layers (Gate, Up, Down projections)**: Compute in BF16/FP16
- **Embedding layers**: BF16/FP16
- **LayerNorm**: FP32 (stability critical)
- **RMSNorm**: FP32 (stability critical)
- **Final LM head**: BF16/FP16 for matmul, FP32 for softmax/loss

**BF16 vs FP16**:
- **BF16 (bfloat16)**: Same range as FP32 (8-bit exponent), no gradient scaling needed → **Recommended**
- **FP16 (float16)**: Narrower range (5-bit exponent), requires gradient scaling to prevent underflow → Older GPUs only

**Key Insight**: All weights have TWO copies - FP32 master (optimizer) + BF16/FP16 runtime (compute)

**Complete Example**:

```python
# LoRA with mixed precision
hyperparameters = {
    "training_type": "sft",
    "finetuning_type": "lora",
    "precision": "bf16-mixed",
    "batch_size": 16,        # Can increase due to memory savings
    "learning_rate": 0.0001,
    "lora": {
        "adapter_dim": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
}
```

**QLoRA with Mixed Precision**:

```python
# 4-bit base model + bf16 LoRA adapters
hyperparameters = {
    "precision": "bf16-mixed",
    "quantization": {
        "algorithm": "nf4",
        "compute_dtype": "bfloat16"
    },
    "lora": {"adapter_dim": 64}
}
```

**Memory Impact**:

| Config | 7B Model | 70B Model |
|--------|----------|-----------|
| LoRA (FP32) | 56 GB | 560 GB |
| LoRA (BF16) | 28 GB | 280 GB |
| QLoRA (4-bit + BF16) | 7 GB | 70 GB |

**Important**: CPU training requires `precision: "32"` (no mixed precision support)

#### 3. How to Check if Model Fits in GPU Memory

Before choosing parallelism strategy, calculate if your model fits in available GPU memory.

**Quick Formula**:
```
GPU Memory Needed = Model Size × Precision Factor × Training Overhead

Model Size (GB) = Parameters (B) × 2 bytes (FP16)
Training Overhead ≈ 4x (for gradients, optimizer states, activations)
```

**Examples**:

| Model | Parameters | FP16 Size | Training Memory | Fits in GPU? |
|-------|-----------|-----------|-----------------|--------------|
| Llama 3.1 8B | 8B | 16 GB | 64 GB | ✅ A100 80GB |
| Llama 3.1 70B | 70B | 140 GB | 560 GB | ❌ Needs 8× A100 |
| Mistral 7B | 7B | 14 GB | 56 GB | ✅ A100 80GB |
| GPT-3 175B | 175B | 350 GB | 1.4 TB | ❌ Needs 20× A100 |

**How to check**:
```python
# Calculate model memory
parameters = 70_000_000_000  # 70B model
fp16_size = parameters * 2 / (1024**3)  # Convert to GB
training_memory = fp16_size * 4  # Training overhead

print(f"Model size: {fp16_size:.1f} GB")
print(f"Training memory needed: {training_memory:.1f} GB")

# Check if fits in single GPU
gpu_memory = 80  # A100 80GB
if training_memory <= gpu_memory:
    print("✅ Fits in single GPU - No parallelism needed")
else:
    gpus_needed = int(training_memory / gpu_memory) + 1
    print(f"❌ Needs {gpus_needed}+ GPUs - Use Tensor Parallelism")
```

**Special case - LoRA**:
```
LoRA doesn't train full model, only small adapters!

LoRA Memory = Base Model (inference mode) + Adapter Weights + Gradients
            ≈ 2x model size (instead of 4x)

Example: 70B model with LoRA
- Full SFT: 560 GB (needs 8 GPUs)
- LoRA: 280 GB (needs 4 GPUs or less with optimizations)
```

**How to check if individual layers fit in GPU**:

Tensor Parallelism splits **individual layers** across GPUs. Check if one layer fits:

```python
# Approximate layer size for Transformer
model_size = 140  # GB (70B model)
num_layers = 80   # Typical for large models
layer_size = model_size / num_layers

print(f"Single layer size: {layer_size:.1f} GB")

if layer_size > 80:  # A100 80GB
    print("❌ Single layer too large - Tensor Parallelism required")
else:
    print("✅ Layers fit in GPU - TP optional (for speed)")
```

**Practical GPU requirements**:

| Scenario | Model | Method | GPU Setup | Recommendation |
|----------|-------|--------|-----------|----------------|
| Small model | 7B | LoRA | 1× A100 40GB | No parallelism |
| Medium model | 13B | LoRA | 1× A100 80GB | No parallelism |
| Large model | 70B | LoRA | 4× A100 80GB | TP=2, DP=2 |
| Large model | 70B | Full SFT | 8× A100 80GB | TP=4, DP=2 |
| Huge model | 175B | LoRA | 8× A100 80GB | TP=4, DP=2 |

**When you don't know**:
- Try without parallelism first
- If you get "CUDA out of memory" error → Add Tensor Parallelism
- NeMo often auto-configures parallelism based on available GPUs

---

#### 3. Parallelism Strategies (Multi-GPU Training)

For large models or datasets, distribute training across GPUs:

```python
hyperparameters = {
    # ... basic params ...
    "parallelism": {
        "tensor_parallel": 2,      # Split model layers across GPUs
        "pipeline_parallel": 2,    # Split model stages across GPUs
        "data_parallel": 4,        # Replicate model, split data
    }
}
```

**Parallelism Types**:

| Type | What It Splits | Use Case | Example |
|------|---------------|----------|---------|
| **Tensor Parallelism (TP)** | Individual layers across GPUs | Model too large for one GPU | 70B model: TP=4 (4 GPUs per layer) |
| **Pipeline Parallelism (PP)** | Model stages (blocks) across GPUs | Very deep models | 12-layer model: PP=3 (4 layers per GPU) |
| **Data Parallelism (DP)** | Training data batches | Speed up training | DP=8 (8 GPUs process different batches) |

**Visual Example**:
```
Single GPU:  [Full Model] → One batch at a time

Tensor Parallel (TP=2):
GPU 0: [Layer 0 (half)] [Layer 1 (half)]
GPU 1: [Layer 0 (half)] [Layer 1 (half)]

Pipeline Parallel (PP=2):
GPU 0: [Layers 0-5]
GPU 1: [Layers 6-11]

Data Parallel (DP=4):
GPU 0: [Full Model] → Batch 1
GPU 1: [Full Model] → Batch 2
GPU 2: [Full Model] → Batch 3
GPU 3: [Full Model] → Batch 4
```

**When to use**:

| Scenario | Strategy | Configuration |
|----------|----------|---------------|
| Model fits in 1 GPU | None needed | Just set `batch_size` |
| Model too large | Tensor Parallel | `tensor_parallel: 2-8` |
| Extremely deep model | Pipeline Parallel | `pipeline_parallel: 2-4` |
| Speed up training | Data Parallel | `data_parallel: num_gpus` |
| 70B model, 8 GPUs | Combined | `TP=4, DP=2` |

**Rules**:
- `TP × PP × DP` must equal total GPUs
- LoRA often only needs DP (adapters are small)
- Full SFT needs TP/PP for large models

#### 3. Complete Advanced Example

```python
# Advanced LoRA configuration for production
hyperparameters = {
    # Basic settings
    "training_type": "sft",
    "finetuning_type": "lora",
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "max_seq_length": 4096,

    # LoRA advanced
    "lora": {
        "adapter_dim": 16,              # Higher rank for complex tasks
        "adapter_dropout": 0.05,
        "target_modules": [             # Fine-tune attention + FFN
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    },

    # Multi-GPU parallelism (8 GPUs)
    "parallelism": {
        "tensor_parallel": 1,           # LoRA: model fits in 1 GPU
        "data_parallel": 8              # Use all 8 GPUs for speed
    },

    # Optimization
    "gradient_accumulation_steps": 4,  # Simulate larger batches
    "warmup_steps": 100,                # Learning rate warmup
    "weight_decay": 0.01,               # Regularization

    # Checkpointing
    "save_steps": 500,                  # Save every 500 steps
    "eval_steps": 100                   # Evaluate every 100 steps
}
```

**When to use advanced settings**:
- **Target modules**: Domain-specific fine-tuning (e.g., medical, legal)
- **Parallelism**: Multi-GPU setup, large models (13B+)
- **Gradient accumulation**: Limited GPU memory
- **Advanced settings**: Production deployments

**Beginners**: Stick to basic hyperparameters - NeMo's defaults work well for most cases!

---

### Submit the Job

#### Understanding Dataset Reference

The `dataset` parameter uses namespace + name to automatically construct the Hugging Face path:

```python
# What you provide:
dataset = {"name": "my-dataset", "namespace": "my-namespace"}

# What NeMo constructs:
hf_path = "hf://my-namespace/my-dataset"
training_file = "hf://my-namespace/my-dataset/training/train.jsonl"
validation_file = "hf://my-namespace/my-dataset/validation/validation.jsonl"
```

**Method 1: Implicit Reference (Recommended)**

**Using Python SDK (Unified Client):**
```python
from nemo_microservices import NeMoMicroservices
import os

# Initialize client
client = NeMoMicroservices(base_url=os.environ['CUSTOMIZER_BASE_URL'])

# Create fine-tuning job
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+8GB",
    dataset={
        "name": DATASET_NAME,        # e.g., "my-dataset"
        "namespace": NAMESPACE        # e.g., "my-namespace"
    },
    hyperparameters=hyperparameters
)

print(f"Job ID: {job.id}")
print(f"Output Model: {job.output_model}")
# NeMo automatically uses: hf://my-namespace/my-dataset
```

**Alternative - Using Legacy Service-Specific Client:**
```python
from nemo_customizer import Client

client = Client(base_url=CUSTOMIZER_BASE_URL)
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+8GB",
    dataset={"name": DATASET_NAME, "namespace": NAMESPACE},
    hyperparameters=hyperparameters
)
```

**Method 2: Explicit File Paths (Alternative)**

```python
# Option A: Explicit Hugging Face paths
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+8GB",
    training_file="hf://my-namespace/my-dataset/training/train.jsonl",
    validation_file="hf://my-namespace/my-dataset/validation/validation.jsonl",
    hyperparameters=hyperparameters
)

# Option B: Data Store (S3) paths
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+8GB",
    training_file="s3://data-store/my-namespace/datasets/train.jsonl",
    validation_file="s3://data-store/my-namespace/datasets/validation.jsonl",
    hyperparameters=hyperparameters
)
```

**Why implicit is recommended**:
- ✅ Cleaner code
- ✅ Prevents path mismatch errors
- ✅ Consistent with Entity Store organization

**Using cURL:**
```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "config": "meta/llama-3.2-1b-instruct@v1.0.0+8GB",
    "dataset": {
      "name": "'${DATASET_NAME}'",
      "namespace": "'${NAMESPACE}'"
    },
    "hyperparameters": {
      "training_type": "sft",
      "finetuning_type": "lora",
      "epochs": 10,
      "batch_size": 16,
      "learning_rate": 0.0001,
      "lora": {
        "adapter_dim": 8,
        "adapter_dropout": 0.01
      }
    }
  }'
```

---

## Step 5: Job Monitoring

### Check Job Status

**Using Python SDK:**
```python
# Get job status
status = client.customization.jobs.get(job_id=job.id)

print(f"Status: {status.status}")
print(f"Progress: {status.progress}%")
print(f"Current Epoch: {status.current_epoch}")
print(f"Training Loss: {status.training_loss}")
print(f"Validation Loss: {status.validation_loss}")
```

**Using cURL:**
```bash
curl -X GET "${CUSTOMIZER_BASE_URL}/v1/customization/jobs/${JOB_ID}" \
  -H "Authorization: Bearer ${API_KEY}"
```

### Understanding Job Metrics

- **Status**: `pending`, `running`, `completed`, `failed`
- **Progress**: Percentage completion (0-100%)
- **Current Epoch**: Which training epoch is currently running
- **Training Loss**: Loss on training data (should decrease)
- **Validation Loss**: Loss on validation data (should decrease without overfitting)

### Early Stopping Feature

Training automatically stops if validation loss shows no improvement for 10 consecutive epochs. This prevents overfitting and saves resources.

**What this means:**
- Your job might complete before all specified epochs finish
- This is a good thing - it means the model has learned as much as it can
- Check the logs to see if early stopping was triggered

### Monitor Training Progress

```python
import time

while True:
    status = client.customization.jobs.get(job_id=job.id)

    if status.status == "completed":
        print("Training completed successfully!")
        break
    elif status.status == "failed":
        print(f"Training failed: {status.error_message}")
        break
    else:
        print(f"Progress: {status.progress}% | Epoch: {status.current_epoch}")
        time.sleep(60)  # Check every minute
```

---

## Step 6: Using the Fine-Tuned Model

### Output Model Identifier

After successful completion, you'll receive an output model identifier:

**Format:** `namespace/base-model-name@job-id`

**Example:** `my-namespace/llama-3.2-1b-instruct@abc123-job-id`

### Deployment with LoRA Adapters

LoRA adapters are served alongside the base model, allowing multiple adapters to share the same base model:

```python
# The adapter is automatically registered and ready to use
# You can reference it by the output_model identifier
model_id = job.output_model
```

### Next Steps

1. **Evaluate Performance**: Use the Evaluator service to test your model
2. **Review Metrics**: Check detailed training metrics
3. **Deploy for Inference**: Use the model identifier in your applications
4. **Iterate**: Adjust hyperparameters based on results

---

## Common Issues and Solutions

### Issue: "Config must include version"
**Solution**: Add version to your config URN: `model@v1.0.0+size`

### Issue: Dataset not found
**Solution**: Ensure namespaces match exactly between Entity Store and Data Store

### Issue: Files not uploading
**Solution**: Verify files are placed in `training/` and `validation/` directories

### Issue: Training loss not decreasing
**Solution**:
- Increase learning rate
- Check data quality and format
- Ensure sufficient training examples

### Issue: Validation loss increasing
**Solution**:
- Reduce learning rate
- Add more validation data
- Check for data quality issues
- Early stopping should prevent this

---

## Integration with Weights & Biases

For enhanced monitoring, integrate with Weights & Biases:

```python
import os
os.environ["WANDB_API_KEY"] = "your-api-key"

# When creating job, metrics automatically upload to wandb.ai
# Project name: nvidia-nemo-customizer
```

Access your training metrics at [wandb.ai](https://wandb.ai) to visualize:
- Loss curves
- Learning rate schedules
- GPU utilization
- Custom metrics

---

# Supervised Fine-Tuning (SFT)

https://docs.nvidia.com/nemo/microservices/latest/fine-tune/tutorials/sft-customization-job.html

## What is Full SFT?

Supervised Fine-Tuning (SFT) with all weights updates every parameter in the model, not just adapters. This provides maximum flexibility but requires more computational resources.

### Key Differences from LoRA

| Aspect | LoRA | Full SFT |
|--------|------|----------|
| **Parameters Updated** | Small adapter layers only | All model weights |
| **Resource Requirements** | Low (few GB VRAM) | High (tens to hundreds of GB) |
| **Training Time** | Minutes to hours | Hours to days |
| **Model Size** | Small adapter files | Full model checkpoint |
| **Deployment** | Adapter serving | Dedicated NIM deployment |
| **Use Case** | Task adaptation | Complete model customization |

### When to Use Full SFT

- Need maximum performance on specific tasks
- Have sufficient GPU resources
- Require complete control over model behavior
- Working with proprietary or sensitive data that needs isolated models
- Base model needs significant behavioral changes

---

## Prerequisites

### Required Access
- Access to NeMo Customizer microservice
- Sufficient GPU resources (check model requirements)
- `huggingface_hub` Python package installed
- Completed entity management and namespace setup

### Environment Variables

Same as LoRA setup:

```bash
export CUSTOMIZER_BASE_URL="<your-customizer-service-url>"
export ENTITY_STORE_BASE_URL="<your-entity-store-url>"
export DATA_STORE_BASE_URL="<your-data-store-url>"
export NAMESPACE="<your-namespace>"
export DATASET_NAME="<your-dataset-name>"
export HF_ENDPOINT="<huggingface-endpoint>"
export HF_TOKEN="<your-huggingface-token>"
export WANDB_API_KEY="<your-wandb-api-key>"  # Optional
```

---

## Step 1: Model Selection

### Query Available SFT Configurations

**Using Python SDK:**
```python
from nemo_customizer import Client

client = Client(base_url=CUSTOMIZER_BASE_URL)

# Filter for all_weights configurations
configs = client.customization.configs.list(
    filter={"finetuning_type": "all_weights"}
)

# Review configurations
for config in configs:
    print(f"Model: {config.name}")
    print(f"Version: {config.version}")
    print(f"GPU Requirements: {config.gpu_requirements}")
    print(f"Recommended Batch Size: {config.recommended_batch_size}")
    print(f"Dataset Schema: {config.dataset_schema}")
    print("---")
```

**Using cURL:**
```bash
curl -X GET "${CUSTOMIZER_BASE_URL}/v1/customization/configs?filter[finetuning_type]=all_weights" \
  -H "Authorization: Bearer ${API_KEY}"
```

### Understanding GPU Requirements

Full SFT requires significantly more resources:
- **1B parameter model**: ~80GB VRAM
- **7B parameter model**: ~200GB VRAM
- **13B+ parameter models**: Multiple high-end GPUs

Plan your resources accordingly!

---

## Step 2: Dataset Preparation

### Model Selection for SFT

**Always use chat/instruct models** (same as LoRA):

✅ **Recommended**: `llama-3.1-nemotron-70b-instruct`, `mistral-large-instruct`
❌ **Avoid**: Base models (use base models only for initial instruction-tuning)

### Data Format

**Preferred: Chat format**:

**train.jsonl:**
```json
{"messages": [{"role": "system", "content": "You are a French translator."}, {"role": "user", "content": "Translate: Hello, how are you?"}, {"role": "assistant", "content": "Bonjour, comment allez-vous?"}]}
{"messages": [{"role": "system", "content": "You are a French translator."}, {"role": "user", "content": "Translate: What is your name?"}, {"role": "assistant", "content": "Comment vous appelez-vous?"}]}
{"messages": [{"role": "system", "content": "You are a French translator."}, {"role": "user", "content": "Translate: I love programming."}, {"role": "assistant", "content": "J'adore la programmation."}]}
```

**validation.jsonl:**
```json
{"messages": [{"role": "system", "content": "You are a French translator."}, {"role": "user", "content": "Translate: Good morning!"}, {"role": "assistant", "content": "Bonjour!"}]}
{"messages": [{"role": "system", "content": "You are a French translator."}, {"role": "user", "content": "Translate: See you tomorrow."}, {"role": "assistant", "content": "À demain."}]}
```

**Alternative: Completion format** (legacy):
```json
{"prompt": "Translate to French: Hello", "completion": "Bonjour"}
```

### Data Quality Considerations

For SFT, data quality is critical:

1. **Use chat format** with system/user/assistant roles
2. **Diverse Examples**: Cover all expected use cases
3. **Consistent Format**: Maintain uniform structure
4. **Sufficient Volume**: 1,000+ examples recommended
5. **Balanced Distribution**: Equal representation across categories
6. **Clean Data**: Remove duplicates and errors

---

## Step 3: Data Upload

### Initialize Clients

```python
from nemo_customizer import Client
from huggingface_hub import HfApi

# Initialize clients
entity_client = Client(base_url=ENTITY_STORE_BASE_URL)
data_client = Client(base_url=DATA_STORE_BASE_URL)
hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)
```

### Create Namespaces

```python
# Create namespace in Entity Store
entity_client.namespaces.create(name="my-sft-namespace")

# Create namespace in Data Store
data_client.namespaces.create(name="my-sft-namespace")
```

### Create and Upload Dataset

```python
# Create dataset repository
hf_api.create_repo(
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset",
    private=True
)

# Upload training data to training/ folder
hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

# Upload validation data to validation/ folder
hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)
```

**Critical**: Files must be in `training/` and `validation/` folders respectively!

---

## Step 4: Job Configuration

### Define Hyperparameters

Full SFT typically uses different hyperparameters than LoRA:

```python
hyperparameters = {
    "training_type": "sft",              # Supervised fine-tuning
    "finetuning_type": "all_weights",    # Update all weights
    "epochs": 3,                         # Fewer epochs needed
    "batch_size": 4,                     # Smaller batch due to memory
    "learning_rate": 0.00005,            # Lower learning rate
    "max_seq_length": 2048,              # Maximum sequence length
    "gradient_accumulation_steps": 4,    # Simulate larger batch
    "warmup_steps": 100,                 # Learning rate warmup
    "weight_decay": 0.01                 # Regularization
}
```

### Hyperparameter Explanations for SFT

- **epochs**: Usually 3-5 for full SFT (more can cause overfitting)
- **batch_size**: Limited by GPU memory (typically 1-8)
- **learning_rate**: Lower than LoRA to prevent catastrophic forgetting (0.00001-0.0001)
- **gradient_accumulation_steps**: Accumulate gradients to simulate larger batches
- **warmup_steps**: Gradually increase learning rate at start
- **weight_decay**: Prevent overfitting by penalizing large weights

### Construct Configuration URN

**Critical Requirement**: Must include version!

**Format:** `model-name@version+size`

**Example:** `meta/llama-3.2-1b-instruct@v1.0.0+80GB`

---

## Step 5: Submit Training Job

### Using Python SDK

```python
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    dataset={
        "name": DATASET_NAME,
        "namespace": NAMESPACE
    },
    hyperparameters=hyperparameters
)

print(f"Job ID: {job.id}")
print(f"Output Model: {job.output_model}")
print(f"Status: {job.status}")
```

### Using cURL

```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "config": "meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    "dataset": {
      "name": "'${DATASET_NAME}'",
      "namespace": "'${NAMESPACE}'"
    },
    "hyperparameters": {
      "training_type": "sft",
      "finetuning_type": "all_weights",
      "epochs": 3,
      "batch_size": 4,
      "learning_rate": 0.00005,
      "max_seq_length": 2048,
      "gradient_accumulation_steps": 4,
      "warmup_steps": 100,
      "weight_decay": 0.01
    }
  }'
```

### Save Job Information

```python
# Save job details for later reference
import json

job_info = {
    "job_id": job.id,
    "output_model": job.output_model,
    "config": "meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    "dataset": f"{NAMESPACE}/{DATASET_NAME}",
    "created_at": job.created_at
}

with open("job_info.json", "w") as f:
    json.dump(job_info, f, indent=2)

print(f"Job information saved to job_info.json")
```

---

## Step 6: Monitor Training

### Poll Job Status

```python
import time

def monitor_job(job_id, poll_interval=60):
    """Monitor job progress until completion or failure"""

    while True:
        status = client.customization.jobs.get(job_id=job_id)

        print(f"\n{'='*50}")
        print(f"Status: {status.status}")
        print(f"Progress: {status.progress}%")

        if hasattr(status, 'current_epoch'):
            print(f"Current Epoch: {status.current_epoch}/{status.total_epochs}")

        if hasattr(status, 'training_loss'):
            print(f"Training Loss: {status.training_loss:.4f}")

        if hasattr(status, 'validation_loss'):
            print(f"Validation Loss: {status.validation_loss:.4f}")

        if status.status == "completed":
            print(f"\n✓ Training completed successfully!")
            print(f"Output Model: {status.output_model}")
            return status

        elif status.status == "failed":
            print(f"\n✗ Training failed!")
            if hasattr(status, 'error_message'):
                print(f"Error: {status.error_message}")
            return status

        else:
            print(f"\nStill running... checking again in {poll_interval} seconds")
            time.sleep(poll_interval)

# Start monitoring
completed_job = monitor_job(job.id)
```

### Understanding Training Metrics

**Training Loss**: Measures how well the model fits the training data
- Should steadily decrease
- Very low values might indicate overfitting

**Validation Loss**: Measures performance on unseen validation data
- Should decrease but not as low as training loss
- If increasing while training loss decreases = overfitting
- Ideally stays close to training loss

**Good Training Pattern:**
```
Epoch 1: Train Loss: 2.5, Val Loss: 2.6
Epoch 2: Train Loss: 1.8, Val Loss: 1.9
Epoch 3: Train Loss: 1.3, Val Loss: 1.4
```

**Overfitting Pattern:**
```
Epoch 1: Train Loss: 2.5, Val Loss: 2.6
Epoch 2: Train Loss: 1.2, Val Loss: 2.1
Epoch 3: Train Loss: 0.5, Val Loss: 2.8  ← Overfitting!
```

---

## Step 7: Model Deployment

### Important Distinction

**LoRA**: Uses adapter serving alongside base model
**Full SFT**: Requires dedicated NIM (NVIDIA Inference Microservice) deployment

### Create Deployment Configuration

```python
from nemo_deployment import DeploymentClient

deployment_client = DeploymentClient(base_url=DEPLOYMENT_BASE_URL)

# Create deployment configuration
deployment_config = deployment_client.configs.create(
    name="my-sft-model-deployment",
    model=job.output_model,  # Use the output model from training
    container={
        "image": "nvcr.io/nvidia/nim:latest",
        "resources": {
            "gpu": 1,
            "memory": "80Gi"
        }
    }
)

print(f"Deployment Config ID: {deployment_config.id}")
```

### Create Deployment

```python
# Create the actual deployment
deployment = deployment_client.deployments.create(
    config_id=deployment_config.id,
    namespace=NAMESPACE
)

print(f"Deployment ID: {deployment.id}")
print(f"Status: {deployment.status}")
```

### Wait for Deployment to be Ready

```python
def wait_for_deployment(deployment_id, timeout=1800):
    """Wait for deployment to become ready"""

    import time
    start_time = time.time()

    while time.time() - start_time < timeout:
        deployment = deployment_client.deployments.get(deployment_id)

        print(f"Deployment Status: {deployment.status}")

        if deployment.status == "ready":
            print(f"\n✓ Deployment is ready!")
            print(f"Endpoint: {deployment.endpoint}")
            return deployment

        elif deployment.status == "failed":
            print(f"\n✗ Deployment failed!")
            print(f"Error: {deployment.error_message}")
            return None

        print(f"Waiting for deployment... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(30)

    print(f"\n✗ Deployment timed out after {timeout}s")
    return None

# Wait for deployment
ready_deployment = wait_for_deployment(deployment.id)
```

### What Happens During Deployment

The service automatically:
1. Downloads the fine-tuned model weights
2. Provisions storage and GPU resources
3. Configures the NIM container
4. Starts the inference server
5. Performs health checks
6. Exposes the endpoint

---

## Step 8: Testing the Model

### Using OpenAI-Compatible Client

```python
from openai import OpenAI

# Initialize OpenAI client pointing to NIM endpoint
openai_client = OpenAI(
    base_url=ready_deployment.endpoint,
    api_key="dummy-key"  # NIM uses different auth
)

# Test the model
response = openai_client.completions.create(
    model=job.output_model,
    prompt="Translate to French: Hello, how are you?",
    max_tokens=128,
    temperature=0.7
)

print(f"Response: {response.choices[0].text}")
```

### Test with Chat Format

```python
# For instruction-tuned models, use chat format
response = openai_client.chat.completions.create(
    model=job.output_model,
    messages=[
        {"role": "system", "content": "You are a helpful translation assistant."},
        {"role": "user", "content": "Translate to French: Hello, how are you?"}
    ],
    max_tokens=128,
    temperature=0.7
)

print(f"Response: {response.choices[0].message.content}")
```

### Comprehensive Testing Script

```python
def test_model(model_name, endpoint, test_prompts):
    """Test model with multiple prompts"""

    client = OpenAI(base_url=endpoint, api_key="dummy")

    results = []
    for prompt in test_prompts:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=128,
            temperature=0.7
        )

        result = {
            "prompt": prompt,
            "response": response.choices[0].text,
            "tokens_used": response.usage.total_tokens
        }
        results.append(result)

        print(f"\nPrompt: {prompt}")
        print(f"Response: {result['response']}")
        print(f"Tokens: {result['tokens_used']}")

    return results

# Test prompts
test_prompts = [
    "Translate to French: Good morning!",
    "Translate to French: How are you today?",
    "Translate to French: I love learning new languages."
]

# Run tests
test_results = test_model(
    model_name=job.output_model,
    endpoint=ready_deployment.endpoint,
    test_prompts=test_prompts
)
```

---

## Step 9: Monitoring with Weights & Biases

### Enable W&B Integration

Set the API key before creating the job:

```python
import os
os.environ["WANDB_API_KEY"] = "your-wandb-api-key"

# Create job (W&B integration happens automatically)
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    dataset={"name": DATASET_NAME, "namespace": NAMESPACE},
    hyperparameters=hyperparameters
)
```

### Access W&B Dashboard

1. Go to [wandb.ai](https://wandb.ai)
2. Navigate to project: `nvidia-nemo-customizer`
3. Find your run by job ID

### Available Metrics

W&B automatically tracks:
- Training and validation loss curves
- Learning rate schedule
- GPU utilization and memory
- Throughput (tokens/second)
- Custom metrics
- System metrics

### Analyze Training

Use W&B to:
- Compare different hyperparameter configurations
- Identify overfitting early
- Debug training issues
- Share results with team
- Create reports

---

## Comparison: LoRA vs Full SFT

### Quick Decision Guide

**Choose LoRA if:**
- Limited GPU resources (< 80GB VRAM)
- Need fast iteration cycles
- Want to maintain multiple task-specific versions
- Working with well-aligned base models
- Budget constraints

**Choose Full SFT if:**
- Need maximum task performance
- Have sufficient GPU resources (80GB+ VRAM)
- Require significant behavioral changes
- Working with proprietary/sensitive data requiring isolation
- Can afford longer training times

### Resource Comparison Example

For Llama 3.2 1B model:

| Aspect | LoRA | Full SFT |
|--------|------|----------|
| **VRAM Required** | 8-16 GB | 80+ GB |
| **Training Time** | 1-2 hours | 4-8 hours |
| **Disk Space** | 50-100 MB | 4-8 GB |
| **Deployment** | Adapter serving | Dedicated NIM |
| **Cost** | $ | $$$ |

---

## Best Practices

### Data Preparation
1. **Quality over Quantity**: 1,000 high-quality examples > 10,000 noisy ones
2. **Balanced Distribution**: Ensure all categories are well-represented
3. **Validation Split**: Use 10-20% of data for validation
4. **Format Consistency**: Maintain uniform structure across all examples
5. **Remove Duplicates**: Clean your dataset before training

### Hyperparameter Tuning
1. **Start Conservative**: Begin with recommended values
2. **One Change at a Time**: Isolate the effect of each hyperparameter
3. **Monitor Validation Loss**: Primary metric for model quality
4. **Use Learning Rate Warmup**: Prevents early instability
5. **Save Checkpoints**: Enable recovery from failures

### Training Monitoring
1. **Check Regularly**: Don't just submit and forget
2. **Watch for Overfitting**: Validation loss increasing is a red flag
3. **Use W&B**: Visual monitoring is easier than raw logs
4. **Set Up Alerts**: Get notified of completion or failures
5. **Document Results**: Keep track of what works

### Production Deployment
1. **Test Thoroughly**: Validate on diverse examples before production
2. **A/B Testing**: Compare with base model performance
3. **Resource Planning**: Ensure sufficient infrastructure
4. **Monitoring**: Track inference latency and quality
5. **Version Control**: Keep track of model versions and datasets

---

## Troubleshooting

### Training Issues

**Problem: Training loss not decreasing**
- Increase learning rate (try 2-5x current value)
- Check data format and quality
- Ensure sufficient training examples (100+ minimum)
- Verify dataset schema matches model expectations

**Problem: Validation loss increasing**
- Decrease learning rate (try 0.5x current value)
- Reduce number of epochs
- Add more diverse validation data
- Check for data leakage between train/val splits

**Problem: Out of memory errors**
- Reduce batch size (try half)
- Reduce max_seq_length
- Enable gradient checkpointing
- Use gradient accumulation to simulate larger batches

**Problem: Job stuck in pending**
- Check GPU resource availability
- Verify quota limits
- Contact support if waiting > 30 minutes

### Deployment Issues

**Problem: Deployment fails to start**
- Verify output model identifier is correct
- Check GPU resources are available
- Ensure namespace permissions
- Review deployment logs

**Problem: Inference is slow**
- Check GPU utilization (should be high)
- Reduce max_tokens in requests
- Consider batching requests
- Review model quantization options

**Problem: Quality degraded from training**
- Test with same prompts used in validation
- Check if different sampling parameters needed
- Verify correct model version deployed
- Compare with base model behavior

### Data Issues

**Problem: Dataset upload fails**
- Verify file paths use correct directories (`training/`, `validation/`)
- Check file format is valid NDJSON
- Ensure namespace exists in both stores
- Verify Hugging Face token has write permissions

**Problem: Schema validation errors**
- Review dataset schema from config
- Ensure all required fields present
- Check for extra/missing fields
- Validate JSON syntax (use online validators)

---

## Next Steps

### After Successful Fine-Tuning

1. **Evaluate Performance**
   - Use NeMo Evaluator service
   - Test on held-out test set
   - Compare with base model baseline
   - Gather user feedback

2. **Optimize for Production**
   - Fine-tune hyperparameters based on results
   - Consider quantization for faster inference
   - Set up monitoring and logging
   - Implement safety guardrails

3. **Iterate and Improve**
   - Collect production data for next iteration
   - Expand dataset with edge cases
   - Try different hyperparameter combinations
   - Explore advanced techniques (DPO, distillation)

4. **Scale Deployment**
   - Set up auto-scaling policies
   - Implement load balancing
   - Monitor costs and resource usage
   - Plan for model updates

### Additional Customization Techniques

NeMo supports other fine-tuning methods:
- **DPO (Direct Preference Optimization)**: Align models with human preferences
- **Distillation**: Create smaller, faster models
- **Embedding Models**: Custom embedding for retrieval
- **Multi-Task Learning**: Train on multiple tasks simultaneously

### Resources

- [NeMo Documentation](https://docs.nvidia.com/nemo/)
- [Model Configurations](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/configs.html)
- [Hyperparameter Guide](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/hyperparameters.html)
- [Job Metrics Tutorial](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/tutorials/job-metrics.html)
- [Evaluator Service](https://docs.nvidia.com/nemo/microservices/latest/evaluator/)

---

## Appendix: Complete Example Scripts

### End-to-End LoRA Example

```python
#!/usr/bin/env python3
"""
Complete LoRA fine-tuning example
"""
import os
import time
from nemo_customizer import Client
from huggingface_hub import HfApi

# Configuration
CUSTOMIZER_BASE_URL = os.environ["CUSTOMIZER_BASE_URL"]
ENTITY_STORE_BASE_URL = os.environ["ENTITY_STORE_BASE_URL"]
DATA_STORE_BASE_URL = os.environ["DATA_STORE_BASE_URL"]
HF_ENDPOINT = os.environ["HF_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]
NAMESPACE = "my-namespace"
DATASET_NAME = "email-writing-dataset"

# Initialize clients
customizer = Client(base_url=CUSTOMIZER_BASE_URL)
entity_client = Client(base_url=ENTITY_STORE_BASE_URL)
data_client = Client(base_url=DATA_STORE_BASE_URL)
hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)

# Step 1: Create namespaces
print("Creating namespaces...")
entity_client.namespaces.create(name=NAMESPACE)
data_client.namespaces.create(name=NAMESPACE)

# Step 2: Upload dataset
print("Uploading dataset...")
hf_api.create_repo(
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset",
    private=True
)

hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

# Step 3: Create job
print("Creating fine-tuning job...")
job = customizer.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+8GB",
    dataset={"name": DATASET_NAME, "namespace": NAMESPACE},
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "lora": {
            "adapter_dim": 8,
            "adapter_dropout": 0.01
        }
    }
)

print(f"Job ID: {job.id}")
print(f"Output Model: {job.output_model}")

# Step 4: Monitor job
print("Monitoring job...")
while True:
    status = customizer.customization.jobs.get(job_id=job.id)
    print(f"Status: {status.status} | Progress: {status.progress}%")

    if status.status in ["completed", "failed"]:
        break

    time.sleep(60)

# Step 5: Results
if status.status == "completed":
    print(f"\n✓ Training completed!")
    print(f"Use model: {status.output_model}")
else:
    print(f"\n✗ Training failed: {status.error_message}")
```

### End-to-End Full SFT Example

```python
#!/usr/bin/env python3
"""
Complete Full SFT fine-tuning and deployment example
"""
import os
import time
from nemo_customizer import Client
from nemo_deployment import DeploymentClient
from huggingface_hub import HfApi
from openai import OpenAI

# Configuration
CUSTOMIZER_BASE_URL = os.environ["CUSTOMIZER_BASE_URL"]
ENTITY_STORE_BASE_URL = os.environ["ENTITY_STORE_BASE_URL"]
DATA_STORE_BASE_URL = os.environ["DATA_STORE_BASE_URL"]
DEPLOYMENT_BASE_URL = os.environ["DEPLOYMENT_BASE_URL"]
HF_ENDPOINT = os.environ["HF_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]
NAMESPACE = "my-sft-namespace"
DATASET_NAME = "translation-dataset"

# Initialize clients
customizer = Client(base_url=CUSTOMIZER_BASE_URL)
entity_client = Client(base_url=ENTITY_STORE_BASE_URL)
data_client = Client(base_url=DATA_STORE_BASE_URL)
deployment_client = DeploymentClient(base_url=DEPLOYMENT_BASE_URL)
hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)

print("=== Full SFT Fine-Tuning Pipeline ===\n")

# Step 1: Setup
print("Step 1: Setting up namespaces and dataset...")
entity_client.namespaces.create(name=NAMESPACE)
data_client.namespaces.create(name=NAMESPACE)

hf_api.create_repo(
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset",
    private=True
)

hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

# Step 2: Create training job
print("\nStep 2: Creating fine-tuning job...")
job = customizer.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    dataset={"name": DATASET_NAME, "namespace": NAMESPACE},
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "all_weights",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 0.00005,
        "max_seq_length": 2048,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 100,
        "weight_decay": 0.01
    }
)

print(f"Job ID: {job.id}")
print(f"Output Model: {job.output_model}")

# Step 3: Monitor training
print("\nStep 3: Monitoring training...")
while True:
    status = customizer.customization.jobs.get(job_id=job.id)

    if hasattr(status, 'training_loss') and hasattr(status, 'validation_loss'):
        print(f"Epoch {status.current_epoch} | "
              f"Train Loss: {status.training_loss:.4f} | "
              f"Val Loss: {status.validation_loss:.4f}")
    else:
        print(f"Status: {status.status} | Progress: {status.progress}%")

    if status.status in ["completed", "failed"]:
        break

    time.sleep(60)

if status.status == "failed":
    print(f"\n✗ Training failed: {status.error_message}")
    exit(1)

print(f"\n✓ Training completed!")

# Step 4: Create deployment
print("\nStep 4: Creating deployment...")
deployment_config = deployment_client.configs.create(
    name=f"{DATASET_NAME}-deployment",
    model=job.output_model,
    container={
        "image": "nvcr.io/nvidia/nim:latest",
        "resources": {"gpu": 1, "memory": "80Gi"}
    }
)

deployment = deployment_client.deployments.create(
    config_id=deployment_config.id,
    namespace=NAMESPACE
)

print(f"Deployment ID: {deployment.id}")

# Step 5: Wait for deployment
print("\nStep 5: Waiting for deployment to be ready...")
while True:
    dep_status = deployment_client.deployments.get(deployment.id)
    print(f"Deployment Status: {dep_status.status}")

    if dep_status.status == "ready":
        break
    elif dep_status.status == "failed":
        print(f"✗ Deployment failed: {dep_status.error_message}")
        exit(1)

    time.sleep(30)

print(f"✓ Deployment ready!")
print(f"Endpoint: {dep_status.endpoint}")

# Step 6: Test the model
print("\nStep 6: Testing the model...")
openai_client = OpenAI(
    base_url=dep_status.endpoint,
    api_key="dummy"
)

test_prompts = [
    "Translate to French: Hello!",
    "Translate to French: How are you?",
    "Translate to French: Good morning!"
]

for prompt in test_prompts:
    response = openai_client.completions.create(
        model=job.output_model,
        prompt=prompt,
        max_tokens=64,
        temperature=0.7
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response.choices[0].text}")

print("\n=== Pipeline Complete! ===")
print(f"Model: {job.output_model}")
print(f"Endpoint: {dep_status.endpoint}")
```

---

# Knowledge Distillation

## What is Knowledge Distillation?

Knowledge distillation is a technique for transferring knowledge from a large, high-capacity "teacher" model to a smaller "student" model. This approach enables smaller models to achieve comparable accuracy to larger ones while requiring significantly fewer computational resources.

### Key Benefits
- **Reduced model size**: Deploy smaller models with similar performance
- **Lower inference costs**: Smaller models = less compute needed
- **Faster inference**: Smaller models respond quicker
- **Easier deployment**: Fits on resource-constrained devices

### When to Use Distillation
- Need to deploy models on edge devices or mobile
- Want to reduce API costs for production inference
- Have a well-performing large model you want to compress
- Need faster response times while maintaining quality

### Important Limitations
- ⚠️ **Only logit-pair distillation is currently supported**
- ⚠️ **LoRA adapters cannot be used as teacher models**
- Requires access to both teacher and student models

---

## Prerequisites

### Required Access
- Access to NeMo Customizer microservice
- Python packages: `huggingface_hub` and `requests`
- Teacher model available as a customization target
- Student model (smaller architecture) ready for training

### Environment Variables

Same setup as LoRA/SFT:

```bash
export CUSTOMIZER_BASE_URL="<your-customizer-service-url>"
export ENTITY_STORE_BASE_URL="<your-entity-store-url>"
export DATA_STORE_BASE_URL="<your-data-store-url>"
export NAMESPACE="<your-namespace>"
export DATASET_NAME="<your-dataset-name>"
export HF_ENDPOINT="<huggingface-endpoint>"
export HF_TOKEN="<your-huggingface-token>"
export WANDB_API_KEY="<your-wandb-api-key>"  # Optional
```

---

## Step 1: Find Distillation Configurations

### Query Available Configs

**Using Python SDK:**
```python
from nemo_customizer import Client

client = Client(base_url=CUSTOMIZER_BASE_URL)

# Filter for distillation configurations
configs = client.customization.configs.list(
    filter={
        "training_type": "distillation",
        "finetuning_type": "all_weights"
    }
)

# Review configurations
for config in configs:
    print(f"Student Model: {config.name}")
    print(f"Version: {config.version}")
    print(f"GPU Requirements: {config.gpu_requirements}")
    print(f"Max Sequence Length: {config.max_seq_length}")
    print("---")
```

**Using cURL:**
```bash
curl -X GET "${CUSTOMIZER_BASE_URL}/v1/customization/configs?filter[training_type]=distillation&filter[finetuning_type]=all_weights" \
  -H "Authorization: Bearer ${API_KEY}"
```

### Understanding the Configuration

- **Student model**: The small model you're training (e.g., Llama 3.2 1B)
- **Teacher model**: The large model you're learning from (specified in hyperparameters)
- **Training precision**: bf16 recommended for best performance

---

## Step 2: Prepare Dataset

### Data Format

**Critical**: Use the **same dataset** that fine-tuned the teacher model!

Dataset format is identical to SFT (JSONL with chat or completion format):

**train.jsonl:**
```json
{"messages": [{"role": "user", "content": "What is photosynthesis?"}, {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight into chemical energy..."}]}
{"messages": [{"role": "user", "content": "Explain gravity."}, {"role": "assistant", "content": "Gravity is the force that attracts objects with mass toward each other..."}]}
```

### Why Use the Same Dataset?

The student model learns to mimic the teacher's outputs on this specific data. Using different data may reduce distillation effectiveness.

---

## Step 3: Upload Dataset

Same process as LoRA/SFT:

```python
from huggingface_hub import HfApi

hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)

# Create dataset repository
hf_api.create_repo(
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset",
    private=True
)

# Upload training data
hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)

# Upload validation data
hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",
    repo_id=f"{NAMESPACE}/{DATASET_NAME}",
    repo_type="dataset"
)
```

---

## Step 4: Configure Distillation Job

### Define Hyperparameters

```python
hyperparameters = {
    "training_type": "distillation",     # Distillation mode
    "finetuning_type": "all_weights",    # Full weight updates
    "epochs": 3,                         # Usually fewer epochs needed
    "batch_size": 4,                     # Adjust based on GPU memory
    "learning_rate": 0.00005,            # Lower learning rate
    "max_seq_length": 2048,              # Match teacher's length
    "distillation": {
        "teacher": "meta/llama-3.2-3b-instruct@v1.0.0"  # Teacher model
    }
}
```

### Hyperparameter Explanations

- **teacher**: Full path to teacher model with version (required!)
- **epochs**: Typically 3-5 for distillation (fewer than standard training)
- **batch_size**: Limited by student model size + teacher inference
- **learning_rate**: Lower than SFT to ensure stable knowledge transfer

### Teacher-Student Pairing Examples

| Teacher Model | Student Model | Use Case |
|---------------|---------------|----------|
| Llama 3.2 3B | Llama 3.2 1B | General compression |
| Llama 3.1 8B | Llama 3.2 3B | Moderate compression |
| Custom 70B | Llama 3.1 8B | Maximum compression |

---

## Step 5: Submit Distillation Job

### Using Python SDK

```python
job = client.customization.jobs.create(
    config="meta/llama-3.2-1b-instruct@v1.0.0+80GB",  # Student model
    dataset={
        "name": DATASET_NAME,
        "namespace": NAMESPACE
    },
    hyperparameters=hyperparameters
)

print(f"Job ID: {job.id}")
print(f"Output Model: {job.output_model}")
print(f"Teacher: {hyperparameters['distillation']['teacher']}")
print(f"Student: meta/llama-3.2-1b-instruct")
```

### Using cURL

```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "config": "meta/llama-3.2-1b-instruct@v1.0.0+80GB",
    "dataset": {
      "name": "'${DATASET_NAME}'",
      "namespace": "'${NAMESPACE}'"
    },
    "hyperparameters": {
      "training_type": "distillation",
      "finetuning_type": "all_weights",
      "epochs": 3,
      "batch_size": 4,
      "learning_rate": 0.00005,
      "max_seq_length": 2048,
      "distillation": {
        "teacher": "meta/llama-3.2-3b-instruct@v1.0.0"
      }
    }
  }'
```

**Critical**: Must include version in config (e.g., `@v1.0.0+80GB`)!

---

## Step 6: Monitor Training

### Check Job Status

```python
import time

def monitor_distillation(job_id, interval=60):
    """Monitor distillation job progress"""

    while True:
        status = client.customization.jobs.get(job_id=job_id)

        print(f"\n{'='*50}")
        print(f"Status: {status.status}")
        print(f"Progress: {status.progress}%")

        if hasattr(status, 'current_epoch'):
            print(f"Epoch: {status.current_epoch}/{status.total_epochs}")

        if hasattr(status, 'distillation_loss'):
            print(f"Distillation Loss: {status.distillation_loss:.4f}")

        if status.status == "completed":
            print(f"\n✓ Distillation completed!")
            print(f"Student Model: {status.output_model}")
            return status

        elif status.status == "failed":
            print(f"\n✗ Distillation failed!")
            if hasattr(status, 'error_message'):
                print(f"Error: {status.error_message}")
            return status

        time.sleep(interval)

# Start monitoring
completed_job = monitor_distillation(job.id)
```

### Understanding Distillation Metrics

**Distillation Loss**: Measures how well the student mimics the teacher
- Should decrease steadily
- Lower values = better knowledge transfer
- Compare to teacher's original performance

**Good Distillation Pattern:**
```
Epoch 1: Distillation Loss: 1.8
Epoch 2: Distillation Loss: 1.2
Epoch 3: Distillation Loss: 0.9
```

---

## Step 7: Deploy and Test

### Deployment

Distilled models require full NIM deployment (like SFT):

```python
from nemo_deployment import DeploymentClient

deployment_client = DeploymentClient(base_url=DEPLOYMENT_BASE_URL)

# Create deployment
deployment_config = deployment_client.configs.create(
    name="distilled-model-deployment",
    model=job.output_model,
    container={
        "image": "nvcr.io/nvidia/nim:latest",
        "resources": {
            "gpu": 1,
            "memory": "40Gi"  # Smaller than teacher!
        }
    }
)

deployment = deployment_client.deployments.create(
    config_id=deployment_config.id,
    namespace=NAMESPACE
)

print(f"Deployment ID: {deployment.id}")
```

### Testing the Distilled Model

```python
from openai import OpenAI

openai_client = OpenAI(
    base_url=deployment.endpoint,
    api_key="dummy"
)

# Test with same prompts used for teacher
test_prompts = [
    "What is photosynthesis?",
    "Explain gravity.",
    "What is machine learning?"
]

for prompt in test_prompts:
    response = openai_client.chat.completions.create(
        model=job.output_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        temperature=0.7
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response.choices[0].message.content}")
```

### Comparing with Teacher

```python
# Compare response quality
teacher_response = get_teacher_response(prompt)
student_response = get_student_response(prompt)

print(f"Teacher (3B): {teacher_response}")
print(f"Student (1B): {student_response}")
print(f"Quality maintained: {'Yes' if similar(teacher_response, student_response) else 'No'}")
```

---

## Best Practices for Distillation

### 1. Teacher Model Selection
- Use a well-performing teacher (fine-tuned on your task)
- Teacher should be 2-4x larger than student
- Ensure teacher model is stable and reliable

### 2. Student Model Selection
- Choose appropriate size for deployment constraints
- Ensure architectural compatibility
- Consider inference speed vs. quality tradeoff

### 3. Dataset Quality
- Use the same high-quality data that trained the teacher
- Include diverse examples covering all use cases
- Validate data format matches teacher's training data

### 4. Hyperparameter Tuning
- Start with lower learning rates (0.00003-0.00005)
- Use fewer epochs than standard training (3-5)
- Monitor distillation loss closely

### 5. Evaluation
- Compare student outputs directly with teacher
- Test on held-out evaluation set
- Measure both quality and inference speed

---

## Troubleshooting Distillation

### Issue: High distillation loss not decreasing
**Solution**:
- Reduce learning rate by 50%
- Increase batch size if GPU memory allows
- Ensure dataset matches teacher's training data
- Check teacher model version is correct

### Issue: Student performs worse than expected
**Solution**:
- Verify teacher model quality first
- Increase training epochs
- Try different student architecture size
- Check for data quality issues

### Issue: Out of memory during training
**Solution**:
- Reduce batch size
- Reduce max_seq_length
- Use gradient checkpointing
- Choose smaller student model

### Issue: Teacher model not found
**Solution**:
- Verify teacher model path includes version
- Check teacher model is available in your namespace
- Ensure teacher model type is compatible

---

## Monitoring with Weights & Biases

Enable W&B for detailed distillation metrics:

```python
import os
os.environ["WANDB_API_KEY"] = "your-api-key"

# Metrics automatically tracked:
# - Distillation loss
# - Student vs teacher output divergence
# - Training efficiency
# - GPU utilization
```

Access at [wandb.ai](https://wandb.ai) under project `nvidia-nemo-customizer`.

---

# Embedding Model Customization

## What are Embedding Models?

Embedding models are specialized neural networks that convert text into dense numerical vectors (embeddings). These vectors capture semantic meaning, allowing computers to understand text similarity and relationships.

### What are Embeddings?

Think of embeddings as coordinates in a multi-dimensional space where similar concepts are close together:

```
"cat" → [0.2, 0.8, 0.1, ...]
"dog" → [0.3, 0.7, 0.2, ...]  ← Close to "cat"
"car" → [0.9, 0.1, 0.8, ...]  ← Far from "cat"
```

### Key Use Cases
- **Semantic Search**: Find documents by meaning, not just keywords
- **Question Answering**: Match questions to relevant answers
- **RAG Pipelines**: Retrieve relevant context for LLMs
- **Document Retrieval**: Find similar documents
- **Clustering**: Group similar texts together
- **Recommendation Systems**: Suggest similar content

### When to Customize Embedding Models
- Domain-specific vocabulary (medical, legal, technical)
- Better retrieval accuracy for your specific data
- Improved performance on specialized tasks
- Custom similarity metrics for your use case

---

## Prerequisites

### Required Access
- Access to NeMo Customizer microservice
- Access to Deployment Management Service
- Python packages: `huggingface_hub`, `requests`
- Established namespaces in Entity Store
- Pre-uploaded embedding datasets

### Environment Variables

```bash
export CUSTOMIZER_BASE_URL="<your-customizer-service-url>"
export ENTITY_STORE_BASE_URL="<your-entity-store-url>"
export DATA_STORE_BASE_URL="<your-data-store-url>"
export DEPLOYMENT_BASE_URL="<your-deployment-url>"
export HF_ENDPOINT="${DATA_STORE_BASE_URL}/v1/hf"
export HF_TOKEN="<your-huggingface-token>"
export NAMESPACE="<your-namespace>"
```

---

## Step 1: Understand Embedding Dataset Format

### Triplet Format

Embedding models are trained using **triplet loss** with three components:

| Field | Purpose | Example |
|-------|---------|---------|
| **query** | Anchor text for matching | "What is machine learning?" |
| **pos_doc** | Positive match (relevant) | "Machine learning is a subset of AI..." |
| **neg_doc** | Negative matches (not relevant) | ["Cooking recipes", "Car mechanics"] |

### Complete Example

**train.jsonl:**
```json
{"query": "machine learning algorithms", "pos_doc": "Deep learning approaches to classification including neural networks and transformers", "neg_doc": ["Quantum computing cryptography methods", "Traditional database indexing techniques"]}
{"query": "natural language processing", "pos_doc": "NLP techniques for text analysis including tokenization and embeddings", "neg_doc": ["Image recognition algorithms", "Audio signal processing"]}
{"query": "computer vision basics", "pos_doc": "Fundamentals of image processing and object detection", "neg_doc": ["Speech synthesis methods", "Network protocols"]}
```

### Important Rules

1. ✅ **query**: User's search or question
2. ✅ **pos_doc**: Document that SHOULD match the query
3. ✅ **neg_doc**: Array of documents that SHOULD NOT match
4. ✅ Each line is a complete JSON object (JSONL format)
5. ✅ Provide multiple negative examples for better learning

### Why Triplets?

The model learns to:
- Place query and pos_doc close together in embedding space
- Push query and neg_doc far apart in embedding space
- Create meaningful semantic representations

---

## Step 2: Prepare Embedding Dataset

### Create Training Data

```python
import json

# Example: Creating embedding training data
training_examples = [
    {
        "query": "How to train a neural network?",
        "pos_doc": "Neural network training involves forward propagation, loss calculation, and backpropagation to update weights.",
        "neg_doc": [
            "Recipe for chocolate cake with frosting",
            "Guide to gardening in spring season"
        ]
    },
    {
        "query": "What is semantic search?",
        "pos_doc": "Semantic search finds results based on meaning and context rather than just keyword matching.",
        "neg_doc": [
            "How to change a car tire step by step",
            "History of ancient civilizations"
        ]
    }
]

# Save as JSONL
with open("train.jsonl", "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")
```

### Validation Data

Create separate validation.jsonl with different examples:

```json
{"query": "deep learning frameworks", "pos_doc": "Popular deep learning frameworks include PyTorch, TensorFlow, and JAX", "neg_doc": ["Cooking utensils and tools", "Automotive repair manuals"]}
```

### Data Quality Tips

1. **Diverse queries**: Cover all expected search patterns
2. **Relevant positives**: Ensure pos_doc truly matches the query
3. **Hard negatives**: Use plausible but incorrect neg_doc
4. **Sufficient volume**: At least 1,000+ examples for good results
5. **Domain coverage**: Include all important topics

---

## Step 3: Upload Dataset

```python
from huggingface_hub import HfApi

hf_api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)

# Create dataset repository
hf_api.create_repo(
    repo_id=f"{NAMESPACE}/embedding-dataset",
    repo_type="dataset",
    private=True
)

# Upload training data
hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",
    repo_id=f"{NAMESPACE}/embedding-dataset",
    repo_type="dataset"
)

# Upload validation data
hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",
    repo_id=f"{NAMESPACE}/embedding-dataset",
    repo_type="dataset"
)
```

---

## Step 4: Find Embedding Model Configurations

### Query Available Configs

**Using Python SDK:**
```python
from nemo_customizer import Client

client = Client(base_url=CUSTOMIZER_BASE_URL)

# Filter for lora_merged configurations (embedding models)
configs = client.customization.configs.list(
    filter={"finetuning_type": "lora_merged"}
)

# Review configurations
for config in configs:
    print(f"Model: {config.name}")
    print(f"Version: {config.version}")
    print(f"Type: {config.finetuning_type}")
    print(f"GPU Requirements: {config.gpu_requirements}")
    print("---")
```

**Using cURL:**
```bash
curl -X GET "${CUSTOMIZER_BASE_URL}/v1/customization/configs?filter[finetuning_type]=lora_merged" \
  -H "Accept: application/json"
```

### Understanding lora_merged

**What is lora_merged?**
- Hybrid approach combining LoRA efficiency with full-weight simplicity
- LoRA weights are merged into base model during training
- Creates a complete model artifact (not just adapters)
- Easier deployment than separate adapter management

**Benefits:**
- Efficient training like LoRA
- Single model file for deployment
- No adapter loading complexity
- Better for embedding models

---

## Step 5: Configure Embedding Training Job

### Define Hyperparameters

```python
hyperparameters = {
    "training_type": "sft",              # Supervised fine-tuning
    "finetuning_type": "lora_merged",    # Merge LoRA during training
    "epochs": 3,                         # Number of epochs
    "batch_size": 8,                     # Batch size
    "learning_rate": 5e-5,               # Learning rate (0.00005)
    "max_seq_length": 512,               # Shorter for embeddings
    "lora": {
        "adapter_dim": 16,               # LoRA rank
        "alpha": 32,                     # LoRA alpha (scaling)
        "adapter_dropout": 0.01          # Dropout rate
    }
}
```

### Hyperparameter Explanations for Embeddings

- **epochs**: 3-5 typical for embedding models
- **batch_size**: 8-16 works well for triplet training
- **learning_rate**: 5e-5 to 1e-4 for embedding models
- **max_seq_length**: 512 typical (embeddings don't need long context)
- **adapter_dim**: 16-32 for embedding models (lower than chat models)
- **alpha**: Usually 2x adapter_dim

---

## Step 6: Submit Training Job

### Using Python SDK

```python
job = client.customization.jobs.create(
    config="nvidia/llama-3.2-nv-embedqa-1b@v2+80GB",
    dataset={
        "namespace": NAMESPACE,
        "name": "embedding-dataset"
    },
    hyperparameters=hyperparameters,
    output_model=f"{NAMESPACE}/my-embedding-model@v1"
)

print(f"Job ID: {job.id}")
print(f"Output Model: {job.output_model}")
print(f"Status: {job.status}")
```

### Using cURL

```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "config": "nvidia/llama-3.2-nv-embedqa-1b@v2+80GB",
    "dataset": {
      "namespace": "'${NAMESPACE}'",
      "name": "embedding-dataset"
    },
    "hyperparameters": {
      "training_type": "sft",
      "finetuning_type": "lora_merged",
      "epochs": 3,
      "batch_size": 8,
      "learning_rate": 5e-5,
      "max_seq_length": 512,
      "lora": {
        "adapter_dim": 16,
        "alpha": 32,
        "adapter_dropout": 0.01
      }
    },
    "output_model": "'${NAMESPACE}'/my-embedding-model@v1"
  }'
```

**Important**: Config must include version (e.g., `@v2+80GB`)!

---

## Step 7: Monitor Training

### Poll Job Status

```python
import time

def monitor_embedding_training(job_id, interval=60):
    """Monitor embedding model training"""

    while True:
        job_details = client.customization.jobs.get(job_id=job_id)

        print(f"\n{'='*50}")
        print(f"Status: {job_details.status}")
        print(f"Progress: {job_details.progress}%")

        if hasattr(job_details, 'current_epoch'):
            print(f"Epoch: {job_details.current_epoch}/{job_details.total_epochs}")

        if hasattr(job_details, 'triplet_loss'):
            print(f"Triplet Loss: {job_details.triplet_loss:.4f}")

        if job_details.status == "completed":
            print(f"\n✓ Training completed!")
            print(f"Embedding Model: {job_details.output_model}")
            return job_details

        elif job_details.status == "failed":
            print(f"\n✗ Training failed!")
            if hasattr(job_details, 'error_message'):
                print(f"Error: {job_details.error_message}")
            return job_details

        time.sleep(interval)

# Monitor the job
completed_job = monitor_embedding_training(job.id)
```

### Understanding Training Metrics

**Triplet Loss**: Measures embedding quality
- Should decrease over epochs
- Lower values = better semantic separation
- Typical range: 0.1 - 1.0 after training

**Good Training Pattern:**
```
Epoch 1: Triplet Loss: 0.8
Epoch 2: Triplet Loss: 0.4
Epoch 3: Triplet Loss: 0.2
```

---

## Step 8: Deploy Embedding Model

### Create Deployment

Unlike LoRA adapters, `lora_merged` models need full NIM deployment:

```python
from nemo_deployment import DeploymentClient

deployment_client = DeploymentClient(base_url=DEPLOYMENT_BASE_URL)

# Create deployment configuration
deployment_config = deployment_client.configs.create(
    name="embedding-model-deployment",
    model=job.output_model,
    container={
        "image": "nvcr.io/nvidia/nim-embedding:latest",  # Embedding-specific image
        "resources": {
            "gpu": 1,
            "memory": "40Gi"
        }
    }
)

print(f"Deployment Config ID: {deployment_config.id}")

# Create deployment
deployment = deployment_client.deployments.create(
    config_id=deployment_config.id,
    namespace=NAMESPACE
)

print(f"Deployment ID: {deployment.id}")
```

### Wait for Deployment

```python
def wait_for_embedding_deployment(deployment_id, timeout=1800):
    """Wait for embedding model deployment"""

    import time
    start_time = time.time()

    while time.time() - start_time < timeout:
        deployment = deployment_client.deployments.get(deployment_id)

        print(f"Deployment Status: {deployment.status}")

        if deployment.status == "ready":
            print(f"\n✓ Deployment ready!")
            print(f"Endpoint: {deployment.endpoint}")
            return deployment

        elif deployment.status == "failed":
            print(f"\n✗ Deployment failed!")
            return None

        time.sleep(30)

    print(f"\n✗ Deployment timed out!")
    return None

# Wait for deployment
ready_deployment = wait_for_embedding_deployment(deployment.id)
```

---

## Step 9: Test Embedding Model

### Basic Testing

```python
from nemo_customizer import Client

# Initialize client
embedding_client = Client(base_url=ready_deployment.endpoint)

# Generate embeddings
response = embedding_client.embeddings.create(
    input="What is machine learning?",
    model=job.output_model
)

# Get the embedding vector
embedding = response.data[0].embedding

print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
```

### Similarity Testing

Test if similar texts have similar embeddings:

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Test queries
query = "machine learning algorithms"
pos_text = "neural networks and deep learning"
neg_text = "cooking recipes and food"

# Get embeddings
query_emb = embedding_client.embeddings.create(
    input=query, model=job.output_model
).data[0].embedding

pos_emb = embedding_client.embeddings.create(
    input=pos_text, model=job.output_model
).data[0].embedding

neg_emb = embedding_client.embeddings.create(
    input=neg_text, model=job.output_model
).data[0].embedding

# Calculate similarities
pos_sim = cosine_similarity(query_emb, pos_emb)
neg_sim = cosine_similarity(query_emb, neg_emb)

print(f"Query-Positive similarity: {pos_sim:.4f}")
print(f"Query-Negative similarity: {neg_sim:.4f}")
print(f"Difference: {pos_sim - neg_sim:.4f}")

# Success criteria
if pos_sim > neg_sim and (pos_sim - neg_sim) > 0.1:
    print("✓ Model working correctly!")
else:
    print("✗ Model may need more training")
```

### Batch Testing

Test multiple queries at once:

```python
# Batch of queries
queries = [
    "deep learning frameworks",
    "natural language processing",
    "computer vision techniques"
]

# Get all embeddings in one call
batch_response = embedding_client.embeddings.create(
    input=queries,
    model=job.output_model
)

# Extract embeddings
embeddings = [data.embedding for data in batch_response.data]

print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")
```

---

## Step 10: Quality Validation

### Comprehensive Testing Script

```python
def evaluate_embedding_model(model_name, endpoint, test_cases):
    """
    Evaluate embedding model quality

    Args:
        model_name: Name of the embedding model
        endpoint: Deployment endpoint
        test_cases: List of (query, pos_doc, neg_docs) tuples
    """

    client = Client(base_url=endpoint)

    results = []
    correct = 0
    total = len(test_cases)

    for query, pos_doc, neg_docs in test_cases:
        # Get query embedding
        query_emb = client.embeddings.create(
            input=query, model=model_name
        ).data[0].embedding

        # Get positive document embedding
        pos_emb = client.embeddings.create(
            input=pos_doc, model=model_name
        ).data[0].embedding

        # Get negative document embeddings
        neg_embs = []
        for neg_doc in neg_docs:
            neg_emb = client.embeddings.create(
                input=neg_doc, model=model_name
            ).data[0].embedding
            neg_embs.append(neg_emb)

        # Calculate similarities
        pos_sim = cosine_similarity(query_emb, pos_emb)
        neg_sims = [cosine_similarity(query_emb, neg) for neg in neg_embs]
        max_neg_sim = max(neg_sims)

        # Check if positive is closer than all negatives
        is_correct = pos_sim > max_neg_sim
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "pos_similarity": pos_sim,
            "max_neg_similarity": max_neg_sim,
            "correct": is_correct
        })

        print(f"\nQuery: {query[:50]}...")
        print(f"Positive Similarity: {pos_sim:.4f}")
        print(f"Max Negative Similarity: {max_neg_sim:.4f}")
        print(f"Correct: {'✓' if is_correct else '✗'}")

    accuracy = correct / total
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})")

    return results, accuracy

# Define test cases
test_cases = [
    (
        "What is deep learning?",
        "Deep learning uses neural networks with multiple layers to learn patterns",
        ["Cooking pasta step by step", "History of ancient Rome"]
    ),
    (
        "How to train AI models?",
        "Training AI models involves preparing data, selecting architecture, and optimization",
        ["Car maintenance guide", "Gardening tips for beginners"]
    )
]

# Evaluate
results, accuracy = evaluate_embedding_model(
    model_name=job.output_model,
    endpoint=ready_deployment.endpoint,
    test_cases=test_cases
)

# Success criteria: ≥80% accuracy with average improvement >0.1
if accuracy >= 0.8:
    print("\n✓ Model meets quality standards!")
else:
    print("\n✗ Model may need additional training")
```

---

## Use Cases and Examples

### Use Case 1: Semantic Search

```python
def semantic_search(query, documents, model_name, endpoint, top_k=3):
    """Find most relevant documents for a query"""

    client = Client(base_url=endpoint)

    # Get query embedding
    query_emb = client.embeddings.create(
        input=query, model=model_name
    ).data[0].embedding

    # Get document embeddings
    doc_embs = []
    for doc in documents:
        doc_emb = client.embeddings.create(
            input=doc, model=model_name
        ).data[0].embedding
        doc_embs.append(doc_emb)

    # Calculate similarities
    similarities = [
        cosine_similarity(query_emb, doc_emb)
        for doc_emb in doc_embs
    ]

    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [
        {"document": documents[i], "score": similarities[i]}
        for i in top_indices
    ]

    return results

# Example usage
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "Neural networks are inspired by biological neurons",
    "Database systems store and manage data"
]

results = semantic_search(
    query="What is AI?",
    documents=documents,
    model_name=job.output_model,
    endpoint=ready_deployment.endpoint
)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['document']}")
    print(f"   Score: {result['score']:.4f}\n")
```

### Use Case 2: Document Clustering

```python
from sklearn.cluster import KMeans

def cluster_documents(documents, model_name, endpoint, n_clusters=3):
    """Cluster documents by semantic similarity"""

    client = Client(base_url=endpoint)

    # Get embeddings for all documents
    embeddings = []
    for doc in documents:
        emb = client.embeddings.create(
            input=doc, model=model_name
        ).data[0].embedding
        embeddings.append(emb)

    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Group documents by cluster
    clustered_docs = {i: [] for i in range(n_clusters)}
    for doc, cluster in zip(documents, clusters):
        clustered_docs[cluster].append(doc)

    return clustered_docs

# Example usage
documents = [
    "Deep learning neural networks",
    "Machine learning algorithms",
    "Cooking pasta recipes",
    "Baking bread at home",
    "Convolutional neural networks",
    "Italian cuisine guide"
]

clusters = cluster_documents(
    documents=documents,
    model_name=job.output_model,
    endpoint=ready_deployment.endpoint,
    n_clusters=2
)

for cluster_id, docs in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
```

### Use Case 3: RAG Pipeline

```python
def rag_retrieve(query, knowledge_base, model_name, endpoint, top_k=3):
    """Retrieve relevant context for RAG pipeline"""

    # Get most relevant documents
    relevant_docs = semantic_search(
        query=query,
        documents=knowledge_base,
        model_name=model_name,
        endpoint=endpoint,
        top_k=top_k
    )

    # Combine into context
    context = "\n\n".join([doc["document"] for doc in relevant_docs])

    return context, relevant_docs

# Example usage
knowledge_base = [
    "PyTorch is a deep learning framework developed by Meta",
    "TensorFlow is Google's machine learning framework",
    "JAX is a high-performance numerical computing library",
    "Scikit-learn provides classical machine learning algorithms"
]

query = "What deep learning framework should I use?"
context, sources = rag_retrieve(
    query=query,
    knowledge_base=knowledge_base,
    model_name=job.output_model,
    endpoint=ready_deployment.endpoint
)

print(f"Query: {query}\n")
print(f"Retrieved Context:\n{context}\n")
print(f"Sources ({len(sources)}):")
for i, source in enumerate(sources, 1):
    print(f"{i}. {source['document']} (score: {source['score']:.4f})")
```

---

## Best Practices for Embedding Models

### 1. Dataset Preparation
- **Diverse triplets**: Cover all domain vocabulary
- **Hard negatives**: Use similar but incorrect documents
- **Balanced examples**: Equal representation of all topics
- **Quality over quantity**: 1,000 high-quality triplets > 10,000 low-quality

### 2. Training Configuration
- **Learning rate**: Start with 5e-5, adjust based on loss
- **Batch size**: 8-16 works well for triplet training
- **Epochs**: 3-5 usually sufficient
- **Sequence length**: 512 typical for embeddings

### 3. Evaluation
- **Similarity testing**: Query-positive > query-negative
- **Retrieval accuracy**: Top-k retrieval contains relevant docs
- **Domain coverage**: Test on all important topics
- **Comparison**: Benchmark against base model

### 4. Deployment
- **Batch processing**: Generate embeddings in batches
- **Caching**: Cache frequently used embeddings
- **Normalization**: Normalize embeddings for cosine similarity
- **Dimensionality**: Consider dimensionality reduction if needed

---

## Troubleshooting Embedding Models

### Issue: Poor retrieval accuracy
**Solution**:
- Add more training examples
- Use harder negative examples
- Increase training epochs
- Check dataset quality and diversity

### Issue: All embeddings too similar
**Solution**:
- Increase adapter_dim (e.g., 32 instead of 16)
- Lower learning rate
- Add more diverse negative examples
- Check for data quality issues

### Issue: Slow inference
**Solution**:
- Use batching for multiple queries
- Cache common embeddings
- Consider model quantization
- Use GPU acceleration

### Issue: Model not learning
**Solution**:
- Increase learning rate
- Check triplet format is correct
- Ensure positive and negative docs are distinct
- Validate dataset upload was successful

---

## Summary

This guide covered two powerful fine-tuning approaches with NVIDIA NeMo:

### LoRA (Low-Rank Adaptation)
- **Best for**: Resource-constrained scenarios, quick iterations
- **Resources**: Low (8-16GB VRAM)
- **Time**: Hours
- **Output**: Small adapter files
- **Deployment**: Adapter serving

### Full SFT (Supervised Fine-Tuning)
- **Best for**: Maximum performance, complete customization
- **Resources**: High (80GB+ VRAM)
- **Time**: Hours to days
- **Output**: Full model checkpoint
- **Deployment**: Dedicated NIM

Both approaches follow similar workflows:
1. Select model configuration
2. Prepare training data
3. Upload dataset
4. Configure and submit job
5. Monitor training
6. Deploy and test

Choose the approach that best fits your resources, timeline, and performance requirements. Start with LoRA for initial experiments, then move to full SFT when you need maximum performance.

---

# Monitoring Training Metrics

## What are Training Metrics?

Training metrics are measurements that track how well your model is learning during fine-tuning. They help you understand if training is progressing correctly and when it's complete.

### Why Metrics Matter

- **Track progress**: See if your model is improving
- **Detect problems**: Identify overfitting or training failures early
- **Optimize hyperparameters**: Make data-driven adjustments
- **Validate quality**: Ensure model meets performance requirements

### Available Metrics

NeMo Customizer tracks two primary metrics:

| Metric | When Recorded | What It Measures |
|--------|---------------|------------------|
| **Training Loss** | Every 10 steps (configurable) | How well model fits training data |
| **Validation Loss** | Each validation interval | How well model generalizes to new data |

---

## Understanding Loss Metrics

### Training Loss

**What it is**: Measures prediction errors on training data

**How to interpret**:
- **Decreasing**: Model is learning ✓
- **Stable/Low**: Model has learned the training data well ✓
- **Increasing**: Something is wrong (check learning rate) ✗
- **Very low (near 0)**: Might be overfitting ⚠️

**Typical pattern**:
```
Step 10:  Loss: 2.5
Step 20:  Loss: 2.1
Step 30:  Loss: 1.8
Step 40:  Loss: 1.6
...
Step 100: Loss: 0.8  ← Good progress
```

### Validation Loss

**What it is**: Measures prediction errors on data the model hasn't seen

**How to interpret**:
- **Decreasing with training loss**: Excellent! Model generalizes well ✓
- **Stable while training decreases**: Model has reached its capacity ✓
- **Increasing while training decreases**: Overfitting! ✗
- **Much higher than training loss**: Likely overfitting ⚠️

**Healthy pattern**:
```
Epoch 1: Train Loss: 2.1, Val Loss: 2.2  ← Close together
Epoch 2: Train Loss: 1.5, Val Loss: 1.6  ← Both decreasing
Epoch 3: Train Loss: 1.2, Val Loss: 1.3  ← Still close
```

**Overfitting pattern**:
```
Epoch 1: Train Loss: 2.1, Val Loss: 2.2
Epoch 2: Train Loss: 1.2, Val Loss: 2.0  ← Val increasing
Epoch 3: Train Loss: 0.6, Val Loss: 2.4  ← Getting worse!
```

---

## Step 1: Access Metrics via API

### Using Python SDK

**Basic Retrieval:**

```python
from nemo_customizer import Client
import os

# Initialize client
client = Client(base_url=os.environ['CUSTOMIZER_BASE_URL'])

# Get job status with metrics
job_id = "your-job-id-here"
job_status = client.customization.jobs.get(job_id=job_id)

# Check if metrics are available
if hasattr(job_status, 'metrics') and job_status.metrics:
    metrics = job_status.metrics

    # Extract training losses
    train_losses = metrics.get("train_loss", [])

    # Extract validation losses
    val_losses = metrics.get("val_loss", [])

    print(f"Training samples: {len(train_losses)}")
    print(f"Validation samples: {len(val_losses)}")

    # Show latest values
    if train_losses:
        print(f"Latest training loss: {train_losses[-1]}")
    if val_losses:
        print(f"Latest validation loss: {val_losses[-1]}")
else:
    print("Metrics not yet available")
```

### Using cURL

```bash
# Get job status including metrics
curl "${CUSTOMIZER_BASE_URL}/v1/customization/jobs/${JOB_ID}/status" \
  -H "Authorization: Bearer ${API_KEY}" | jq

# Extract just the metrics
curl "${CUSTOMIZER_BASE_URL}/v1/customization/jobs/${JOB_ID}/status" \
  -H "Authorization: Bearer ${API_KEY}" | jq '.status_details.metrics'
```

### Understanding the Response

**Response structure:**

```json
{
  "job_id": "abc-123",
  "status": "running",
  "progress": 45,
  "status_details": {
    "metrics": {
      "train_loss": [
        {"step": 10, "value": 2.5, "timestamp": "2024-01-15T10:00:00Z"},
        {"step": 20, "value": 2.1, "timestamp": "2024-01-15T10:05:00Z"},
        {"step": 30, "value": 1.8, "timestamp": "2024-01-15T10:10:00Z"}
      ],
      "val_loss": [
        {"epoch": 1, "value": 2.2, "timestamp": "2024-01-15T10:12:00Z"},
        {"epoch": 2, "value": 1.9, "timestamp": "2024-01-15T10:24:00Z"}
      ]
    }
  }
}
```

---

## Step 2: Monitor Metrics During Training

### Real-Time Monitoring Script

```python
import time
import matplotlib.pyplot as plt

def monitor_training_metrics(job_id, interval=30, max_iterations=100):
    """
    Monitor and visualize training metrics in real-time

    Args:
        job_id: Customization job ID
        interval: Seconds between checks
        max_iterations: Maximum monitoring cycles
    """

    train_losses = []
    val_losses = []

    for i in range(max_iterations):
        # Get current status
        status = client.customization.jobs.get(job_id=job_id)

        print(f"\n{'='*60}")
        print(f"Iteration {i+1} | Status: {status.status} | Progress: {status.progress}%")

        # Extract metrics if available
        if hasattr(status, 'metrics') and status.metrics:
            metrics = status.metrics

            # Get latest training loss
            train_loss_data = metrics.get("train_loss", [])
            if train_loss_data:
                latest_train = train_loss_data[-1]
                train_losses.append(latest_train)
                print(f"Latest Training Loss: {latest_train['value']:.4f} (Step {latest_train['step']})")

            # Get latest validation loss
            val_loss_data = metrics.get("val_loss", [])
            if val_loss_data:
                latest_val = val_loss_data[-1]
                val_losses.append(latest_val)
                print(f"Latest Validation Loss: {latest_val['value']:.4f} (Epoch {latest_val.get('epoch', 'N/A')})")

            # Check for completion
            if status.status == "completed":
                print(f"\n✓ Training completed successfully!")
                break
            elif status.status == "failed":
                print(f"\n✗ Training failed!")
                break

        # Wait before next check
        time.sleep(interval)

    return train_losses, val_losses

# Start monitoring
train_history, val_history = monitor_training_metrics(
    job_id="your-job-id",
    interval=60  # Check every minute
)
```

### Plotting Training Progress

```python
def plot_training_metrics(train_losses, val_losses):
    """Visualize training and validation losses"""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract values and steps
    if train_losses:
        train_steps = [item['step'] for item in train_losses]
        train_values = [item['value'] for item in train_losses]
        ax.plot(train_steps, train_values, 'b-', label='Training Loss', linewidth=2)

    if val_losses:
        val_epochs = [item.get('epoch', i) for i, item in enumerate(val_losses)]
        val_values = [item['value'] for item in val_losses]

        # Convert epochs to approximate steps for plotting
        # Assuming validation happens at epoch boundaries
        steps_per_epoch = max(train_steps) / max(val_epochs) if val_epochs else 1
        val_steps = [epoch * steps_per_epoch for epoch in val_epochs]

        ax.plot(val_steps, val_values, 'r-', label='Validation Loss',
                linewidth=2, marker='o', markersize=8)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150)
    print("\n✓ Plot saved as 'training_metrics.png'")
    plt.show()

# Create visualization
plot_training_metrics(train_history, val_history)
```

---

## Step 3: Integration with MLflow

### What is MLflow?

MLflow is an open-source platform for managing machine learning lifecycle, including experiment tracking and model registry.

### Accessing Metrics in MLflow

If your NeMo deployment has MLflow enabled:

**1. Access MLflow UI:**
```
http://<mlflow-server-url>
```

**2. Find Your Experiment:**
- Search by output model name
- Look for experiments matching your job

**3. Locate Your Run:**
- Filter by job ID
- Click on the run to see details

**4. View Metrics:**
- Navigate to "Metrics" tab
- Examine loss curves and other tracked parameters
- Compare with other runs

### MLflow Python API

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://<mlflow-server-url>")

# Search for your experiment
experiment = mlflow.get_experiment_by_name("your-model-name")

# Get runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.job_id = '{job_id}'"
)

# Access metrics
for index, run in runs.iterrows():
    run_id = run.run_id
    metrics = mlflow.get_run(run_id).data.metrics
    print(f"Run: {run_id}")
    print(f"Training Loss: {metrics.get('train_loss')}")
    print(f"Validation Loss: {metrics.get('val_loss')}")
```

---

## Step 4: Integration with Weights & Biases

### What is Weights & Biases?

Weights & Biases (W&B) is a powerful ML experiment tracking platform with advanced visualization and collaboration features.

### Enable W&B Tracking

**During Job Creation:**

```python
import os

# Set W&B API key
wandb_api_key = os.getenv('WANDB_API_KEY')

# Create job with W&B integration
job = client.customization.jobs.create(
    config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
    dataset={
        "name": "my-dataset",
        "namespace": "default"
    },
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.0001
    },
    extra_headers={
        'wandb-api-key': wandb_api_key
    }
)

print(f"Job created: {job.id}")
print(f"View metrics at: https://wandb.ai/nvidia-nemo-customizer")
```

### Accessing W&B Dashboard

1. **Go to wandb.ai**
2. **Navigate to project**: `nvidia-nemo-customizer`
3. **Find your run**: Search by job ID or model name
4. **Explore metrics**:
   - Training and validation loss curves
   - Learning rate schedule
   - GPU utilization
   - System metrics (memory, CPU)
   - Custom metrics

### W&B Features

**Loss Curves:**
- Interactive plots with zoom and pan
- Compare multiple runs
- Smooth curves or show raw data

**System Metrics:**
- GPU utilization percentage
- GPU memory usage
- CPU and RAM usage
- Network I/O

**Custom Tracking:**
```python
# W&B automatically tracks:
# - Training loss (every log interval)
# - Validation loss (every validation epoch)
# - Learning rate (every step)
# - Gradient norms
# - Model parameters
```

### Important Notes

**Privacy:**
- W&B API keys are encrypted
- Keys are not logged by NeMo
- Review W&B terms of service
- Data sent to W&B servers

**Project Organization:**
- All NeMo jobs appear under `nvidia-nemo-customizer` project
- Runs are named with job IDs
- Tags include model name and configuration

---

## Step 5: Interpreting Metrics

### Healthy Training Signs

✅ **Good indicators:**
```python
# Training loss steadily decreasing
train_loss: [2.5, 2.0, 1.6, 1.3, 1.0, 0.8]

# Validation loss following training loss
val_loss: [2.6, 2.1, 1.7, 1.4, 1.1, 0.9]

# Small gap between train and validation
gap = val_loss - train_loss  # Should be < 0.3
```

### Problem Patterns

❌ **Warning signs:**

**1. Overfitting:**
```python
train_loss: [2.5, 1.5, 0.8, 0.3, 0.1]  # Too fast, too low
val_loss:   [2.6, 2.0, 2.1, 2.3, 2.5]  # Increasing!
```

**Solution**:
- Reduce learning rate
- Add more training data
- Increase dropout
- Use early stopping

**2. Underfitting:**
```python
train_loss: [2.5, 2.4, 2.3, 2.3, 2.3]  # Not decreasing enough
val_loss:   [2.6, 2.5, 2.4, 2.4, 2.4]  # Also stuck
```

**Solution**:
- Increase learning rate
- Train for more epochs
- Increase model capacity
- Check data quality

**3. Unstable Training:**
```python
train_loss: [2.5, 1.8, 3.2, 1.5, 4.1, 1.2]  # Jumping around
val_loss:   [2.6, 1.9, 3.5, 1.7, 4.3, 1.4]  # Also unstable
```

**Solution**:
- Reduce learning rate significantly
- Increase batch size
- Use gradient clipping
- Check for data issues

---

## Step 6: Configure Metric Logging

### Adjust Logging Frequency

Control how often metrics are recorded:

```python
hyperparameters = {
    "training_type": "sft",
    "finetuning_type": "lora",
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 0.0001,

    # Metric logging configuration
    "log_interval": 10,        # Log training loss every 10 steps
    "val_check_interval": 1.0  # Validate every 1 epoch (1.0 = 100%)
}
```

### Logging Parameters Explained

| Parameter | Default | Description | Example Values |
|-----------|---------|-------------|----------------|
| `log_interval` | 10 | Steps between training loss logs | 10, 50, 100 |
| `val_check_interval` | 1.0 | Fraction of epoch between validations | 0.5 (half epoch), 1.0 (full epoch) |
| `limit_val_batches` | 1.0 | Fraction of validation data to use | 0.1 (10%), 1.0 (all) |

### Example Configurations

**Frequent Logging (for debugging):**
```python
hyperparameters = {
    "log_interval": 5,         # Log every 5 steps
    "val_check_interval": 0.25 # Validate 4 times per epoch
}
```

**Standard Logging:**
```python
hyperparameters = {
    "log_interval": 10,        # Log every 10 steps
    "val_check_interval": 1.0  # Validate once per epoch
}
```

**Minimal Logging (for long jobs):**
```python
hyperparameters = {
    "log_interval": 100,       # Log every 100 steps
    "val_check_interval": 1.0  # Validate once per epoch
}
```

---

## Best Practices for Monitoring

### 1. Always Monitor Both Metrics

```python
# Track both training and validation
def check_health(train_loss, val_loss):
    gap = val_loss - train_loss

    if gap > 0.5:
        print("⚠️ Warning: Large gap suggests overfitting")
    elif gap < 0:
        print("⚠️ Warning: Validation better than training (data issue?)")
    else:
        print("✓ Healthy training/validation relationship")

    return gap
```

### 2. Set Up Alerts

```python
def alert_on_issues(job_id, check_interval=300):
    """Alert if training shows problems"""

    last_train_loss = None

    while True:
        status = client.customization.jobs.get(job_id=job_id)

        if hasattr(status, 'metrics') and status.metrics:
            train_losses = status.metrics.get("train_loss", [])

            if train_losses and last_train_loss:
                current = train_losses[-1]['value']

                # Alert if loss increases
                if current > last_train_loss * 1.2:
                    print(f"🚨 ALERT: Loss increased from {last_train_loss:.4f} to {current:.4f}")
                    # Send email, Slack notification, etc.

                last_train_loss = current
            elif train_losses:
                last_train_loss = train_losses[-1]['value']

        if status.status in ["completed", "failed"]:
            break

        time.sleep(check_interval)
```

### 3. Compare with Baselines

```python
# Save baseline metrics
baseline_metrics = {
    "final_train_loss": 0.85,
    "final_val_loss": 0.92,
    "training_time": 3600  # seconds
}

# Compare new run
def compare_with_baseline(job_id, baseline):
    status = client.customization.jobs.get(job_id=job_id)

    if status.status == "completed":
        metrics = status.metrics
        final_train = metrics["train_loss"][-1]['value']
        final_val = metrics["val_loss"][-1]['value']

        print(f"Training Loss: {final_train:.4f} vs Baseline: {baseline['final_train_loss']:.4f}")
        print(f"Validation Loss: {final_val:.4f} vs Baseline: {baseline['final_val_loss']:.4f}")

        if final_val < baseline['final_val_loss']:
            print("✓ New model outperforms baseline!")
        else:
            print("⚠️ Baseline still better")
```

### 4. Document Everything

```python
import json
from datetime import datetime

def save_training_report(job_id, hyperparameters, metrics):
    """Save comprehensive training report"""

    report = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": hyperparameters,
        "metrics": {
            "final_train_loss": metrics["train_loss"][-1]['value'],
            "final_val_loss": metrics["val_loss"][-1]['value'],
            "min_val_loss": min(m['value'] for m in metrics["val_loss"]),
            "total_steps": metrics["train_loss"][-1]['step']
        },
        "convergence": {
            "converged": True,  # Based on your criteria
            "epochs_to_converge": 8
        }
    }

    filename = f"training_report_{job_id}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Report saved: {filename}")
```

---

## Troubleshooting Metrics Issues

### Issue: Metrics not appearing

**Possible causes:**
- Job hasn't started training yet
- Metrics endpoint not supported in older versions
- Insufficient permissions

**Solutions:**
```python
# Check job status first
status = client.customization.jobs.get(job_id=job_id)
print(f"Status: {status.status}")
print(f"Progress: {status.progress}%")

# Metrics appear after first log_interval
if status.progress < 1:
    print("Training not started yet, please wait...")
```

### Issue: Incomplete metrics data

**Possible causes:**
- Training still in progress
- Validation not run yet
- Network issues

**Solutions:**
```python
# Gracefully handle missing data
metrics = status.metrics or {}
train_losses = metrics.get("train_loss", [])
val_losses = metrics.get("val_loss", [])

if not train_losses:
    print("Training losses not yet available")
if not val_losses:
    print("Validation not run yet (normal early in training)")
```

### Issue: Metrics show unexpected values

**Possible causes:**
- NaN or Inf values (numerical instability)
- Wrong scale (check if logged in correct units)
- Data preprocessing issues

**Solutions:**
```python
# Validate metric values
def validate_metrics(metrics):
    for loss in metrics.get("train_loss", []):
        value = loss['value']

        if value != value:  # Check for NaN
            print(f"⚠️ NaN detected at step {loss['step']}")
            return False

        if value > 1000:  # Unreasonably high
            print(f"⚠️ Extremely high loss: {value}")
            return False

    return True
```

---

# Optimizing Training Performance

## What is Tokens Per GPU?

Tokens per GPU measures how many tokens your GPU processes during training. Higher throughput means:
- **Faster training**: Complete jobs quicker
- **Lower costs**: Use GPU time more efficiently
- **Better scaling**: Process more data in same time

### Understanding the Metric

**Tokens per GPU = (Batch Size × Sequence Length) / Training Time**

Example:
```
Batch size: 16
Sequence length: 2048
Training step time: 2 seconds

Tokens per step = 16 × 2048 = 32,768 tokens
Throughput = 32,768 / 2 = 16,384 tokens/second
```

---

## Why Optimization Matters

### Cost Impact

**Example comparison:**

| Configuration | Tokens/GPU | Training Time | GPU Cost |
|---------------|------------|---------------|----------|
| Unoptimized | 8,000/sec | 10 hours | $80 |
| Optimized | 16,000/sec | 5 hours | $40 |

**Savings**: 50% reduction in time and cost!

### Resource Utilization

**Poor utilization:**
```
GPU Usage: 40%
Memory: 30% allocated
Tokens/sec: 5,000
→ Wasting 60% of GPU capacity!
```

**Good utilization:**
```
GPU Usage: 85%
Memory: 75% allocated
Tokens/sec: 15,000
→ Efficient use of resources ✓
```

---

## Primary Optimization: Sequence Packing

### What is Sequence Packing?

**Without packing:**
```
Example 1: "Hello world" + padding [512 tokens total, 3 useful]
Example 2: "How are you" + padding [512 tokens total, 4 useful]
Example 3: "Good morning" + padding [512 tokens total, 3 useful]

→ Only 10 tokens useful out of 1,536 total (0.65% efficiency!)
```

**With packing:**
```
Packed: "Hello world [SEP] How are you [SEP] Good morning" + padding
[512 tokens total, 10 useful + 3 separators]

→ 13 tokens useful out of 512 total (2.5% efficiency - 4x better!)
```

### How Sequence Packing Works

1. **Collect short sequences**: Gather multiple training examples
2. **Combine intelligently**: Pack multiple sequences into one batch item
3. **Add separators**: Insert special tokens between sequences
4. **Maximize utilization**: Fill batches to max_seq_length
5. **Train efficiently**: Process more actual data per GPU cycle

### Benefits

✅ **Faster training**: 2-3x speedup typical
✅ **Better GPU utilization**: From 40% to 85%+
✅ **Higher memory efficiency**: More useful work per GB
✅ **Same model quality**: No accuracy loss
✅ **Lower costs**: Fewer GPU hours needed

---

## Step 1: Enable Sequence Packing

### Configuration

Add to your hyperparameters:

```python
hyperparameters = {
    "training_type": "sft",
    "finetuning_type": "lora",
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 0.0001,
    "max_seq_length": 2048,

    # Enable sequence packing
    "sequence_packing_enabled": True,

    "lora": {
        "adapter_dim": 16,
        "adapter_dropout": 0.01
    }
}
```

### Complete Job Creation Example

```python
from nemo_customizer import Client
import os

client = Client(base_url=os.environ['CUSTOMIZER_BASE_URL'])

# Create job with sequence packing
job = client.customization.jobs.create(
    config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
    dataset={
        "name": "my-dataset",
        "namespace": "default"
    },
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "max_seq_length": 2048,
        "sequence_packing_enabled": True,  # Enable packing
        "lora": {
            "adapter_dim": 16,
            "adapter_dropout": 0.01
        }
    }
)

print(f"Job ID: {job.id}")
print(f"Sequence packing: ENABLED")
```

---

## Step 2: Compare Performance

### Run Parallel Experiments

**Create two identical jobs:**

```python
# Job 1: Without sequence packing
job_unpacked = client.customization.jobs.create(
    config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
    dataset={"name": "my-dataset", "namespace": "default"},
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "sequence_packing_enabled": False  # Disabled
    }
)

# Job 2: With sequence packing
job_packed = client.customization.jobs.create(
    config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
    dataset={"name": "my-dataset", "namespace": "default"},
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "sequence_packing_enabled": True  # Enabled
    }
)

print(f"Unpacked job: {job_unpacked.id}")
print(f"Packed job: {job_packed.id}")
```

### Monitor Both Jobs

```python
import time

def compare_jobs(job_id_1, job_id_2, label_1="Job 1", label_2="Job 2"):
    """Compare two training jobs side by side"""

    start_time = time.time()

    while True:
        # Get status of both jobs
        status_1 = client.customization.jobs.get(job_id=job_id_1)
        status_2 = client.customization.jobs.get(job_id=job_id_2)

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"Elapsed Time: {elapsed/60:.1f} minutes")
        print(f"\n{label_1}:")
        print(f"  Status: {status_1.status} | Progress: {status_1.progress}%")

        print(f"\n{label_2}:")
        print(f"  Status: {status_2.status} | Progress: {status_2.progress}%")

        # Check if both completed
        if status_1.status in ["completed", "failed"] and \
           status_2.status in ["completed", "failed"]:
            print(f"\n{'='*70}")
            print("Both jobs finished!")
            break

        time.sleep(60)  # Check every minute

# Compare the jobs
compare_jobs(
    job_unpacked.id,
    job_packed.id,
    "Unpacked (Baseline)",
    "Packed (Optimized)"
)
```

---

## Step 3: Analyze Performance Metrics

### Using Weights & Biases

Enable W&B for both jobs to see detailed comparisons:

```python
# Create jobs with W&B tracking
import os

wandb_key = os.getenv('WANDB_API_KEY')

job_unpacked = client.customization.jobs.create(
    config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
    dataset={"name": "my-dataset", "namespace": "default"},
    hyperparameters={
        "sequence_packing_enabled": False,
        # ... other hyperparameters
    },
    extra_headers={'wandb-api-key': wandb_key}
)

job_packed = client.customization.jobs.create(
    config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
    dataset={"name": "my-dataset", "namespace": "default"},
    hyperparameters={
        "sequence_packing_enabled": True,
        # ... other hyperparameters
    },
    extra_headers={'wandb-api-key': wandb_key}
)
```

### Key Metrics to Compare

**1. Training Time:**
```
Unpacked: 8 hours
Packed:   4 hours
→ 2x speedup! ✓
```

**2. GPU Utilization:**
```
Unpacked: 45% average
Packed:   82% average
→ 82% improvement ✓
```

**3. GPU Memory:**
```
Unpacked: 35 GB used
Packed:   58 GB used
→ Better memory utilization ✓
```

**4. Tokens/Second:**
```
Unpacked: 8,500 tokens/sec
Packed:   18,200 tokens/sec
→ 2.14x throughput! ✓
```

**5. Final Validation Loss:**
```
Unpacked: 0.85
Packed:   0.84
→ Same quality ✓
```

---

## Step 4: Additional Optimization Techniques

### 1. Batch Size Tuning

**Find optimal batch size:**

```python
# Test different batch sizes
batch_sizes = [8, 16, 32, 64]

for batch_size in batch_sizes:
    try:
        job = client.customization.jobs.create(
            config="meta/llama-3.1-8b-instruct@v1.0.0+80GB",
            dataset={"name": "my-dataset", "namespace": "default"},
            hyperparameters={
                "batch_size": batch_size,
                "sequence_packing_enabled": True,
                # ... other params
            }
        )
        print(f"✓ Batch size {batch_size}: Job {job.id} created")
    except Exception as e:
        print(f"✗ Batch size {batch_size}: Failed - {e}")
        break  # Out of memory, stop increasing
```

**Guidelines:**
- **Start small**: Begin with batch_size=8
- **Double gradually**: Try 8, 16, 32, 64
- **Watch for OOM**: Out of memory errors mean you've gone too far
- **Monitor quality**: Ensure validation loss doesn't degrade

### 2. Gradient Accumulation

Simulate larger batches without OOM:

```python
hyperparameters = {
    "batch_size": 8,                      # Physical batch
    "gradient_accumulation_steps": 4,     # Accumulate 4 batches
    # Effective batch = 8 × 4 = 32
    "sequence_packing_enabled": True
}
```

**Benefits:**
- Train with larger effective batch sizes
- Avoid out-of-memory errors
- Smoother gradient updates

**Tradeoff:**
- Slower per-step updates
- Same total throughput

### 3. Mixed Precision Training

Use bf16 or fp16 for faster training:

```python
hyperparameters = {
    "precision": "bf16",  # Or "fp16"
    "batch_size": 16,
    "sequence_packing_enabled": True
}
```

**Benefits:**
- 2x faster computation
- 2x less memory usage
- Can increase batch size

**Note**: Most NeMo configs default to bf16 already

### 4. Efficient Data Loading

Optimize dataset configuration:

```python
hyperparameters = {
    "num_workers": 4,              # Parallel data loading
    "pin_memory": True,            # Faster CPU→GPU transfer
    "persistent_workers": True,    # Keep workers alive
    "sequence_packing_enabled": True
}
```

---

## Best Practices for Optimization

### 1. Always Start with Sequence Packing

```python
# Default to enabled
default_hyperparameters = {
    "sequence_packing_enabled": True,  # Always use this!
    "batch_size": 16,
    "learning_rate": 0.0001
}
```

### 2. Benchmark Before Production

```python
def benchmark_configuration(config, runs=3):
    """Run multiple times to get reliable metrics"""

    durations = []

    for i in range(runs):
        job = client.customization.jobs.create(
            config=config["model"],
            dataset=config["dataset"],
            hyperparameters=config["hyperparameters"]
        )

        # Wait for completion
        start = time.time()
        while True:
            status = client.customization.jobs.get(job_id=job.id)
            if status.status in ["completed", "failed"]:
                break
            time.sleep(60)

        duration = time.time() - start
        durations.append(duration)
        print(f"Run {i+1}: {duration/3600:.2f} hours")

    avg_duration = sum(durations) / len(durations)
    print(f"\nAverage: {avg_duration/3600:.2f} hours")

    return avg_duration
```

### 3. Monitor Resource Usage

```python
# Check if you're hitting limits
def check_resource_limits(job_id):
    """Monitor for resource constraints"""

    status = client.customization.jobs.get(job_id=job_id)

    warnings = []

    # Check GPU memory
    if hasattr(status, 'gpu_memory_used'):
        usage_pct = status.gpu_memory_used / status.gpu_memory_total
        if usage_pct > 0.95:
            warnings.append("GPU memory >95% - may OOM")
        elif usage_pct < 0.50:
            warnings.append("GPU memory <50% - can increase batch size")

    # Check GPU utilization
    if hasattr(status, 'gpu_utilization'):
        if status.gpu_utilization < 60:
            warnings.append("Low GPU utilization - check data loading")

    return warnings
```

### 4. Document Optimizations

```python
optimization_log = {
    "experiment": "sequence_packing_comparison",
    "baseline": {
        "sequence_packing": False,
        "batch_size": 16,
        "training_time_hours": 8.2,
        "tokens_per_second": 8500,
        "final_val_loss": 0.85
    },
    "optimized": {
        "sequence_packing": True,
        "batch_size": 16,
        "training_time_hours": 4.1,
        "tokens_per_second": 18200,
        "final_val_loss": 0.84
    },
    "improvement": {
        "speedup": "2.0x",
        "throughput_increase": "2.14x",
        "cost_savings": "50%"
    }
}

# Save for future reference
import json
with open("optimization_results.json", "w") as f:
    json.dump(optimization_log, f, indent=2)
```

---

## Troubleshooting Performance Issues

### Issue: Sequence packing not improving performance

**Possible causes:**
- Dataset has mostly long sequences already
- Batch size too small to see benefit
- Data loading bottleneck

**Solutions:**
```python
# Check your data distribution
def analyze_sequence_lengths(dataset_path):
    """Analyze if packing will help"""

    import json
    lengths = []

    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            # Approximate token count
            text = data.get('prompt', '') + data.get('completion', '')
            lengths.append(len(text.split()))

    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)

    print(f"Average length: {avg_len:.0f} tokens")
    print(f"Max length: {max_len} tokens")
    print(f"Sequences > 512 tokens: {sum(1 for l in lengths if l > 512)/len(lengths)*100:.1f}%")

    if avg_len > 1500:
        print("⚠️ Most sequences are long - packing may not help much")
    else:
        print("✓ Good candidate for sequence packing!")

    return lengths

# Run analysis
analyze_sequence_lengths("train.jsonl")
```

### Issue: Out of memory with optimizations

**Solutions:**
```python
# Reduce batch size
hyperparameters = {
    "batch_size": 8,  # Half the batch size
    "sequence_packing_enabled": True
}

# Or use gradient accumulation
hyperparameters = {
    "batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "sequence_packing_enabled": True
}
```

### Issue: Training slower with optimizations

**Possible causes:**
- Data loading bottleneck
- Too many workers
- Network latency

**Solutions:**
```python
# Optimize data loading
hyperparameters = {
    "num_workers": 2,  # Reduce if too many
    "prefetch_factor": 2,
    "persistent_workers": True,
    "sequence_packing_enabled": True
}
```

---

# Importing Custom Models from HuggingFace

## What is Model Import?

Model importing allows you to bring your own private or custom models from HuggingFace Hub into NeMo Customizer for fine-tuning. This enables you to use proprietary models or community models not in NeMo's standard catalog.

### Why Import Custom Models?

- **Private models**: Use your organization's proprietary models
- **Custom architectures**: Fine-tune models you've pre-trained
- **Community models**: Access cutting-edge models from HuggingFace
- **Experimentation**: Test different base models for your use case
- **Compliance**: Keep models within your infrastructure

### How It Works

```
HuggingFace Hub → Download → Upload to NeMo → Register → Fine-tune
```

1. Download model from HuggingFace
2. Upload to NeMo Data Store
3. Register in NeMo Entity Store
4. Create customization target
5. Define training configuration
6. Launch fine-tuning jobs

---

## Prerequisites

### Required Access

- **NeMo Customizer**: Fine-tuning service access
- **Entity Store**: Model metadata management
- **Data Store**: Model file storage
- **Deployment Manager**: Model deployment (optional)

### Required Tools

```bash
# Install required packages
pip install huggingface_hub

# Authenticate with HuggingFace
huggingface-cli login
# Or
hf auth login
```

### Storage Requirements

| Model Size | Storage Needed |
|------------|----------------|
| 1-3B parameters | 5-15 GB |
| 7-8B parameters | 20-35 GB |
| 13B+ parameters | 40-100 GB |

### GPU Requirements

| Model Size | Minimum GPU Memory |
|------------|-------------------|
| 1-3B | 8 GB |
| 7-8B | 16 GB |
| 13B+ | 24+ GB |

### Environment Setup

```bash
# Set environment variables
export CUSTOMIZER_BASE_URL="<your-customizer-url>"
export ENTITY_STORE_BASE_URL="<your-entity-store-url>"
export DATA_STORE_BASE_URL="<your-data-store-url>"
export HF_ENDPOINT="${DATA_STORE_BASE_URL}/v1/hf"
export NAMESPACE="<your-namespace>"
```

---

## Supported Model Architectures

### ✅ Compatible Models

**Recommended architectures:**
- **Llama models**: llama-3.1, llama-3.2, llama-3.3
- **Nemotron models**: nvidia/nemotron variants
- **Phi models**: microsoft/phi-2, phi-3
- **Gemma models**: google/gemma-2-2b, gemma-7b
- **Mistral models**: mistralai/mistral-7b

**Example compatible models:**
```python
compatible_models = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2"
]
```

### ❌ Incompatible Models

**Models with Conv1D layers FAIL:**
- ❌ microsoft/DialoGPT-small
- ❌ microsoft/DialoGPT-medium
- ❌ microsoft/DialoGPT-large
- ❌ openai-gpt variants

**Error you'll see:**
```
AttributeError: 'Conv1D' object has no attribute 'config'
```

**Why they fail**: Conv1D layers lack the linear layer structure NeMo's LoRA transformation requires.

---

## Step 1: Download Model from HuggingFace

### Using HuggingFace CLI

```bash
# Define model details
export MODEL_NAME="google/gemma-2-2b-it"
export LOCAL_MODEL_PATH="./downloaded_models/gemma-2-2b-it"

# Download the model
huggingface-cli download ${MODEL_NAME} \
    --local-dir ${LOCAL_MODEL_PATH} \
    --local-dir-use-symlinks False

# Or using hf command
hf download ${MODEL_NAME} --local-dir ${LOCAL_MODEL_PATH}
```

### Using Python

```python
from huggingface_hub import snapshot_download

# Download model
model_name = "google/gemma-2-2b-it"
local_path = "./downloaded_models/gemma-2-2b-it"

snapshot_download(
    repo_id=model_name,
    local_dir=local_path,
    local_dir_use_symlinks=False
)

print(f"✓ Model downloaded to: {local_path}")
```

### What Gets Downloaded

```
downloaded_models/gemma-2-2b-it/
├── config.json              # Model configuration
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
├── special_tokens_map.json  # Special token mappings
├── model-00001-of-00002.safetensors  # Model weights (part 1)
├── model-00002-of-00002.safetensors  # Model weights (part 2)
└── model.safetensors.index.json      # Weight index
```

---

## Step 2: Upload Model to NeMo Data Store

### Create Model Repository

**Using Python SDK:**

```python
from huggingface_hub import HfApi

# Initialize Data Store HF API
hf_api = HfApi(
    endpoint=f"{DATA_STORE_BASE_URL}/v1/hf",
    token=""  # Empty token for internal Data Store
)

# Create repository
repo_id = f"{NAMESPACE}/{MODEL_NAME}"

hf_api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True
)

print(f"✓ Repository created: {repo_id}")
```

**Using cURL:**

```bash
# Extract model name without org
MODEL_SHORT_NAME=$(echo ${MODEL_NAME} | cut -d'/' -f2)

curl -X POST "${DATA_STORE_BASE_URL}/v1/hf/api/repos/create" \
  -H "Content-Type: application/json" \
  -d '{
      "organization": "'${NAMESPACE}'",
      "name": "'${MODEL_SHORT_NAME}'",
      "type": "model"
  }'
```

### Upload Model Files

**Using Python:**

```python
# Upload entire model directory
hf_api.upload_folder(
    repo_id=repo_id,
    folder_path=local_path,
    repo_type="model",
    revision="main",
    commit_message=f"Upload {MODEL_NAME} model files"
)

print(f"✓ Model uploaded to: {repo_id}")
```

**Monitor upload progress:**

```python
import os

def get_dir_size(path):
    """Calculate directory size in GB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total += os.path.getsize(filepath)
    return total / (1024**3)  # Convert to GB

size_gb = get_dir_size(local_path)
print(f"Uploading {size_gb:.2f} GB...")
print("This may take several minutes...")
```

---

## Step 3: Register Model in Entity Store

### Create Model Entity

**Using Python SDK:**

```python
from nemo_customizer import Client

# Initialize Entity Store client
client = Client(base_url=ENTITY_STORE_BASE_URL)

# Create model metadata
model = client.models.create(
    name=MODEL_SHORT_NAME,
    namespace=NAMESPACE,
    artifact={
        "files_url": f"hf://models/{repo_id}",
        "backend_engine": "hugging_face",
        "status": "upload_completed"
    },
    spec={
        "num_parameters": 2000000000,  # 2B parameters for gemma-2-2b
        "context_size": 8192,
        "is_chat": True
    }
)

print(f"✓ Model registered: {model.name}")
print(f"  Namespace: {model.namespace}")
print(f"  Parameters: {model.spec.num_parameters:,}")
```

**Using cURL:**

```bash
curl -X POST "${ENTITY_STORE_BASE_URL}/v1/models" \
  -H "Content-Type: application/json" \
  -d '{
      "name": "'${MODEL_SHORT_NAME}'",
      "namespace": "'${NAMESPACE}'",
      "artifact": {
        "files_url": "hf://models/'${NAMESPACE}'/'${MODEL_SHORT_NAME}'",
        "backend_engine": "hugging_face",
        "status": "upload_completed"
      },
      "spec": {
        "num_parameters": 2000000000,
        "context_size": 8192,
        "is_chat": true
      }
  }'
```

### Understanding Model Spec

**Key fields:**

| Field | Description | Example |
|-------|-------------|---------|
| `num_parameters` | Total model parameters | 2000000000 (2B) |
| `context_size` | Maximum context window | 8192 tokens |
| `is_chat` | Chat vs completion model | true/false |

**Finding model specs:**

```python
# Check model config
import json

with open(f"{local_path}/config.json") as f:
    config = json.load(f)

# Extract key info
print(f"Hidden size: {config.get('hidden_size')}")
print(f"Num layers: {config.get('num_hidden_layers')}")
print(f"Vocab size: {config.get('vocab_size')}")
print(f"Max position: {config.get('max_position_embeddings')}")
```

---

## Step 4: Create Customization Target

### What is a Customization Target?

A target represents a base model ready for fine-tuning. It includes:
- Model weights location
- Training-specific configurations
- GPU requirements
- Precision settings

### Create Target

**Using Python SDK:**

```python
# Define version
MODEL_VERSION = "1.0.0"

# Create customization target
target = client.customization.targets.create(
    name=f"{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
    namespace=NAMESPACE,
    model_uri=f"hf://{NAMESPACE}/{MODEL_SHORT_NAME}",
    num_parameters=2000000000,
    precision="bf16-mixed"  # Or "fp16-mixed"
)

print(f"✓ Target created: {target.name}")
print(f"  Status: {target.status}")
```

**Using cURL:**

```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/targets" \
  -H "Content-Type: application/json" \
  -d '{
      "name": "'${MODEL_SHORT_NAME}'@v'${MODEL_VERSION}'",
      "namespace": "'${NAMESPACE}'",
      "model_uri": "hf://'${NAMESPACE}'/'${MODEL_SHORT_NAME}'",
      "num_parameters": 2000000000,
      "precision": "bf16-mixed"
  }'
```

### Monitor Target Preparation

The target needs to be "ready" before fine-tuning:

```python
import time

def wait_for_target_ready(target_name, namespace, timeout=1800):
    """Wait for target to be ready"""

    start_time = time.time()

    while time.time() - start_time < timeout:
        target = client.customization.targets.get(
            name=target_name,
            namespace=namespace
        )

        print(f"Target Status: {target.status}")

        if target.status == "ready":
            print(f"\n✓ Target ready for fine-tuning!")
            return target
        elif target.status == "failed":
            print(f"\n✗ Target preparation failed!")
            print(f"Error: {target.error_message}")
            return None

        # Status can be: downloading, processing, validating, ready
        time.sleep(30)

    print(f"\n✗ Timeout after {timeout/60:.0f} minutes")
    return None

# Wait for target
ready_target = wait_for_target_ready(
    target_name=f"{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
    namespace=NAMESPACE
)
```

---

## Step 5: Create Fine-Tuning Configuration

### Define LoRA Configuration

**Using Python SDK:**

```python
# Create LoRA fine-tuning config
config = client.customization.configs.create(
    name=f"{MODEL_SHORT_NAME}-lora-config@v{MODEL_VERSION}",
    namespace=NAMESPACE,
    target=f"{NAMESPACE}/{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
    training_options=[{
        "training_type": "sft",
        "finetuning_type": "lora",
        "num_gpus": 1,
        "micro_batch_size": 1,
        "global_batch_size": 8
    }],
    training_precision="bf16-mixed",
    max_seq_length=1024
)

print(f"✓ Config created: {config.name}")
```

**Using cURL:**

```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/configs" \
  -H "Content-Type: application/json" \
  -d '{
      "name": "'${MODEL_SHORT_NAME}'-lora-config@v'${MODEL_VERSION}'",
      "namespace": "'${NAMESPACE}'",
      "target": "'${NAMESPACE}'/'${MODEL_SHORT_NAME}'@v'${MODEL_VERSION}'",
      "training_options": [{
        "training_type": "sft",
        "finetuning_type": "lora",
        "num_gpus": 1,
        "micro_batch_size": 1,
        "global_batch_size": 8
      }],
      "training_precision": "bf16-mixed",
      "max_seq_length": 1024
  }'
```

### Configuration Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `training_type` | Training approach | "sft", "distillation" |
| `finetuning_type` | Efficiency method | "lora", "all_weights" |
| `num_gpus` | GPUs to use | 1, 2, 4, 8 |
| `micro_batch_size` | Per-GPU batch size | 1, 2, 4 |
| `global_batch_size` | Total effective batch | 8, 16, 32 |
| `training_precision` | Numerical precision | "bf16-mixed", "fp16-mixed" |
| `max_seq_length` | Maximum tokens | 512, 1024, 2048 |

---

## Step 6: Prepare Training Dataset

### Dataset Format

Use the same JSONL format as other NeMo fine-tuning:

**train.jsonl:**
```json
{"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm doing well, thank you for asking!"}]}
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}]}
```

### Create and Upload Dataset

```python
import json

# Create training data
training_data = [
    {
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What can you help me with?"},
            {"role": "assistant", "content": "I can help with many tasks including..."}
        ]
    }
]

# Save as JSONL
with open("train.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

print("✓ Training data created")
```

### Upload Dataset to Data Store

```python
from huggingface_hub import HfApi

# Create dataset repository
dataset_name = "my-custom-dataset"
hf_api.create_repo(
    repo_id=f"{NAMESPACE}/{dataset_name}",
    repo_type="dataset",
    exist_ok=True
)

# Upload training file
hf_api.upload_file(
    path_or_fileobj="train.jsonl",
    path_in_repo="training/train.jsonl",
    repo_id=f"{NAMESPACE}/{dataset_name}",
    repo_type="dataset"
)

# Upload validation file
hf_api.upload_file(
    path_or_fileobj="validation.jsonl",
    path_in_repo="validation/validation.jsonl",
    repo_id=f"{NAMESPACE}/{dataset_name}",
    repo_type="dataset"
)

print(f"✓ Dataset uploaded: {dataset_name}")
```

### Register Dataset Entity

```python
# Create dataset entity
dataset = client.datasets.create(
    name=dataset_name,
    namespace=NAMESPACE,
    artifact={
        "files_url": f"hf://datasets/{NAMESPACE}/{dataset_name}",
        "backend_engine": "hugging_face",
        "status": "upload_completed"
    }
)

print(f"✓ Dataset registered: {dataset.name}")
```

---

## Step 7: Launch Fine-Tuning Job

### Submit Job

**Using Python SDK:**

```python
# Create fine-tuning job
job = client.customization.jobs.create(
    config=f"{NAMESPACE}/{MODEL_SHORT_NAME}-lora-config@v{MODEL_VERSION}",
    dataset={
        "name": dataset_name,
        "namespace": NAMESPACE
    },
    hyperparameters={
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "lora": {
            "adapter_dim": 16,
            "adapter_dropout": 0.01
        }
    }
)

print(f"✓ Job created: {job.id}")
print(f"  Output model: {job.output_model}")
```

**Using cURL:**

```bash
curl -X POST "${CUSTOMIZER_BASE_URL}/v1/customization/jobs" \
  -H "Content-Type: application/json" \
  -d '{
      "config": "'${NAMESPACE}'/'${MODEL_SHORT_NAME}'-lora-config@v'${MODEL_VERSION}'",
      "dataset": {
        "name": "'${dataset_name}'",
        "namespace": "'${NAMESPACE}'"
      },
      "hyperparameters": {
        "training_type": "sft",
        "finetuning_type": "lora",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "lora": {
          "adapter_dim": 16,
          "adapter_dropout": 0.01
        }
      }
  }'
```

### Monitor Job Progress

```python
import time

def monitor_job(job_id, interval=60):
    """Monitor fine-tuning job"""

    while True:
        status = client.customization.jobs.get(job_id=job_id)

        print(f"\n{'='*60}")
        print(f"Status: {status.status}")
        print(f"Progress: {status.progress}%")

        if hasattr(status, 'metrics') and status.metrics:
            train_loss = status.metrics.get("train_loss", [])
            if train_loss:
                print(f"Latest Training Loss: {train_loss[-1]['value']:.4f}")

        if status.status == "completed":
            print(f"\n✓ Fine-tuning completed!")
            print(f"  Output: {status.output_model}")
            break
        elif status.status == "failed":
            print(f"\n✗ Job failed!")
            if hasattr(status, 'error_message'):
                print(f"  Error: {status.error_message}")
            break

        time.sleep(interval)

# Start monitoring
monitor_job(job.id)
```

---

## Step 8: Deploy and Test Model

### Deploy Base Model with LoRA Support

```python
from nemo_deployment import DeploymentClient

deployment_client = DeploymentClient(base_url=DEPLOYMENT_BASE_URL)

# Create deployment config
deployment_config = deployment_client.configs.create(
    name=f"{MODEL_SHORT_NAME}-deployment-config",
    namespace=NAMESPACE,
    model=f"{NAMESPACE}/{MODEL_SHORT_NAME}",
    nim_deployment={
        "image_name": "nvcr.io/nim/nvidia/llm-nim:latest",
        "gpu": 1,
        "disable_lora_support": False,  # Enable LoRA adapters
        "additional_envs": {
            "NIM_PEFT_SOURCE": f"http://{ENTITY_STORE_BASE_URL}",
            "NIM_PEFT_REFRESH_INTERVAL": "30"
        }
    }
)

# Create deployment
deployment = deployment_client.deployments.create(
    config_id=deployment_config.id,
    namespace=NAMESPACE
)

print(f"✓ Deployment created: {deployment.id}")
```

### Test Base Model

```python
from openai import OpenAI

# Initialize client
openai_client = OpenAI(
    base_url=deployment.endpoint,
    api_key="dummy"
)

# Test base model
base_response = openai_client.chat.completions.create(
    model=f"{NAMESPACE}/{MODEL_SHORT_NAME}",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(f"Base Model Response:")
print(base_response.choices[0].message.content)
```

### Test Fine-Tuned Model

```python
# Test with LoRA adapter
lora_response = openai_client.chat.completions.create(
    model=job.output_model,  # Fine-tuned model with LoRA
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(f"\nFine-Tuned Model Response:")
print(lora_response.choices[0].message.content)
```

### Compare Responses

```python
def compare_models(prompt, base_model, lora_model):
    """Compare base vs fine-tuned model responses"""

    print(f"\nPrompt: {prompt}")
    print(f"\n{'='*60}")

    # Base model
    base_resp = openai_client.chat.completions.create(
        model=base_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    print(f"BASE MODEL:")
    print(base_resp.choices[0].message.content)

    print(f"\n{'='*60}")

    # Fine-tuned model
    lora_resp = openai_client.chat.completions.create(
        model=lora_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    print(f"FINE-TUNED MODEL:")
    print(lora_resp.choices[0].message.content)

# Test comparison
test_prompts = [
    "What can you help me with?",
    "Tell me about yourself.",
    "How should I greet someone?"
]

for prompt in test_prompts:
    compare_models(
        prompt=prompt,
        base_model=f"{NAMESPACE}/{MODEL_SHORT_NAME}",
        lora_model=job.output_model
    )
```

---

## Complete End-to-End Example

### Full Import and Fine-Tune Script

```python
#!/usr/bin/env python3
"""
Complete example: Import HuggingFace model and fine-tune with NeMo
"""

import os
import time
import json
from huggingface_hub import HfApi, snapshot_download
from nemo_customizer import Client

# Configuration
MODEL_NAME = "google/gemma-2-2b-it"
MODEL_SHORT_NAME = "gemma-2-2b-it"
MODEL_VERSION = "1.0.0"
NAMESPACE = os.environ["NAMESPACE"]
LOCAL_MODEL_PATH = f"./models/{MODEL_SHORT_NAME}"

# Initialize clients
entity_client = Client(base_url=os.environ["ENTITY_STORE_BASE_URL"])
customizer_client = Client(base_url=os.environ["CUSTOMIZER_BASE_URL"])
hf_api = HfApi(
    endpoint=f"{os.environ['DATA_STORE_BASE_URL']}/v1/hf",
    token=""
)

print("="*70)
print("NVIDIA NeMo Model Import and Fine-Tuning Pipeline")
print("="*70)

# Step 1: Download from HuggingFace
print("\n[1/8] Downloading model from HuggingFace...")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_MODEL_PATH,
    local_dir_use_symlinks=False
)
print(f"✓ Downloaded to: {LOCAL_MODEL_PATH}")

# Step 2: Create Data Store repo
print("\n[2/8] Creating Data Store repository...")
repo_id = f"{NAMESPACE}/{MODEL_SHORT_NAME}"
hf_api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print(f"✓ Repository: {repo_id}")

# Step 3: Upload model
print("\n[3/8] Uploading model to Data Store...")
hf_api.upload_folder(
    repo_id=repo_id,
    folder_path=LOCAL_MODEL_PATH,
    repo_type="model",
    commit_message=f"Upload {MODEL_NAME}"
)
print(f"✓ Uploaded")

# Step 4: Register model entity
print("\n[4/8] Registering model in Entity Store...")
model = entity_client.models.create(
    name=MODEL_SHORT_NAME,
    namespace=NAMESPACE,
    artifact={
        "files_url": f"hf://models/{repo_id}",
        "backend_engine": "hugging_face",
        "status": "upload_completed"
    },
    spec={
        "num_parameters": 2000000000,
        "context_size": 8192,
        "is_chat": True
    }
)
print(f"✓ Registered: {model.name}")

# Step 5: Create customization target
print("\n[5/8] Creating customization target...")
target = customizer_client.customization.targets.create(
    name=f"{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
    namespace=NAMESPACE,
    model_uri=f"hf://{NAMESPACE}/{MODEL_SHORT_NAME}",
    num_parameters=2000000000,
    precision="bf16-mixed"
)
print(f"✓ Target: {target.name}")

# Wait for target to be ready
print("   Waiting for target preparation...")
while True:
    target = customizer_client.customization.targets.get(
        name=f"{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
        namespace=NAMESPACE
    )
    if target.status == "ready":
        break
    print(f"   Status: {target.status}")
    time.sleep(30)
print("✓ Target ready!")

# Step 6: Create config
print("\n[6/8] Creating fine-tuning configuration...")
config = customizer_client.customization.configs.create(
    name=f"{MODEL_SHORT_NAME}-lora@v{MODEL_VERSION}",
    namespace=NAMESPACE,
    target=f"{NAMESPACE}/{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
    training_options=[{
        "training_type": "sft",
        "finetuning_type": "lora",
        "num_gpus": 1
    }],
    training_precision="bf16-mixed",
    max_seq_length=1024
)
print(f"✓ Config: {config.name}")

# Step 7: Submit job (assuming dataset already uploaded)
print("\n[7/8] Submitting fine-tuning job...")
job = customizer_client.customization.jobs.create(
    config=f"{NAMESPACE}/{config.name}",
    dataset={"name": "my-dataset", "namespace": NAMESPACE},
    hyperparameters={
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "lora": {"adapter_dim": 16}
    }
)
print(f"✓ Job ID: {job.id}")

# Step 8: Monitor completion
print("\n[8/8] Monitoring job progress...")
while True:
    status = customizer_client.customization.jobs.get(job_id=job.id)
    print(f"   Status: {status.status} | Progress: {status.progress}%")

    if status.status in ["completed", "failed"]:
        break
    time.sleep(60)

if status.status == "completed":
    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"Output Model: {status.output_model}")
    print(f"Ready for deployment and inference!")
else:
    print(f"\n✗ Job failed: {status.error_message}")
```

---

## Best Practices

### 1. Verify Model Compatibility

Before downloading large models:

```python
from transformers import AutoConfig

def check_model_compatibility(model_name):
    """Check if model architecture is compatible"""

    try:
        config = AutoConfig.from_pretrained(model_name)
        arch = config.architectures[0] if config.architectures else "Unknown"

        print(f"Model: {model_name}")
        print(f"Architecture: {arch}")

        # Known incompatible architectures
        incompatible = ["GPT2LMHeadModel"]  # Uses Conv1D

        if any(inc in arch for inc in incompatible):
            print("⚠️ WARNING: This model may not be compatible!")
            print("   Conv1D layers are not supported")
            return False

        print("✓ Architecture appears compatible")
        return True

    except Exception as e:
        print(f"✗ Error checking model: {e}")
        return False

# Check before downloading
check_model_compatibility("google/gemma-2-2b-it")
```

### 2. Clean Up After Import

```python
import shutil

def cleanup_local_model(local_path):
    """Remove local model files after upload"""

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        print(f"✓ Cleaned up: {local_path}")

# After successful upload
cleanup_local_model(LOCAL_MODEL_PATH)
```

### 3. Document Model Metadata

```python
model_metadata = {
    "original_source": MODEL_NAME,
    "import_date": datetime.now().isoformat(),
    "nemo_model_uri": f"hf://{NAMESPACE}/{MODEL_SHORT_NAME}",
    "version": MODEL_VERSION,
    "parameters": 2000000000,
    "context_size": 8192,
    "compatible_configs": [
        f"{MODEL_SHORT_NAME}-lora@v{MODEL_VERSION}"
    ],
    "notes": "Imported from HuggingFace for custom fine-tuning"
}

with open(f"{MODEL_SHORT_NAME}_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

print("✓ Metadata saved")
```

---

## Troubleshooting

### Issue: Conv1D error during fine-tuning

**Error:**
```
AttributeError: 'Conv1D' object has no attribute 'config'
```

**Cause**: Model uses Conv1D layers incompatible with LoRA

**Solution**:
- Choose different base model
- Use models with standard linear layers
- Check architecture before importing

### Issue: Target stuck in "downloading" status

**Possible causes**:
- Large model taking time to download internally
- Network issues
- Insufficient storage

**Solutions**:
```python
# Check target details
target = client.customization.targets.get(
    name=f"{MODEL_SHORT_NAME}@v{MODEL_VERSION}",
    namespace=NAMESPACE
)

print(f"Status: {target.status}")
print(f"Details: {target.status_message}")

# Wait longer for large models (can take 30+ minutes)
```

### Issue: Upload fails with timeout

**Solutions**:
```python
# Upload in smaller chunks
from huggingface_hub import upload_file

# Upload files individually
import os
for root, dirs, files in os.walk(LOCAL_MODEL_PATH):
    for file in files:
        filepath = os.path.join(root, file)
        repo_path = os.path.relpath(filepath, LOCAL_MODEL_PATH)

        upload_file(
            path_or_fileobj=filepath,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✓ Uploaded: {repo_path}")
```

### Issue: Out of memory during fine-tuning

**Solutions**:
```python
# Reduce batch sizes in config
hyperparameters = {
    "batch_size": 4,  # Smaller batch
    "gradient_accumulation_steps": 2,
    "lora": {
        "adapter_dim": 8  # Smaller LoRA rank
    }
}
```

---

Happy fine-tuning!