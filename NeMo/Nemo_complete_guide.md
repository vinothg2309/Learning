# NVIDIA NeMo Framework: Complete E2E Learning Guide (Beginner to Expert)

## 📋 TABLE OF CONTENTS

### 1. [What is NVIDIA NeMo?](#what-is-nvidia-nemo)
- [Key Characteristics](#key-characteristics)
- [Primary Use Cases](#primary-use-cases)

### 2. [Architecture & Components](#architecture--components)
- [Core Architecture Overview](#core-architecture-overview)
- [Key Components Explained](#key-components-explained)

### 3. [Learning Path (Week-by-Week)](#learning-path-week-by-week)
- **[🎯 Beginner Track (4 Weeks)](#-beginner-track-4-weeks)**
  - [Week 1: Foundations & Setup](#week-1-foundations--setup)
  - [Week 2: Data Preparation & Basics](#week-2-data-preparation--basics)
  - [Week 3: LLM Fine-Tuning (LoRA)](#week-3-llm-fine-tuning-lora)
  - [Week 4: Inference & Deployment](#week-4-inference--deployment)
- **[🚀 Intermediate Track (3 Weeks)](#-intermediate-track-3-weeks)**
  - [Week 5: Advanced Training Techniques](#week-5-advanced-training-techniques)
  - [Week 6: NeMo Agent & RAG](#week-6-nemo-agent--rag)
  - [Week 7: Production Deployment](#week-7-production-deployment)
- **[🎓 Expert Track (4+ Weeks)](#-expert-track-4-weeks)**

### 4. [YouTube Video Resources](#youtube-video-resources)
- [⭐ ESSENTIAL Videos (Watch First)](#-essential-videos-watch-first)
- [🎬 Playlist Collections](#-playlist-collections)
- [📝 Video Learning Tips](#-video-learning-tips)

### 5. [Official Documentation](#official-documentation)
- [📖 Primary Documentation Sources](#-primary-documentation-sources)
- [🔗 Key Documentation Pages](#-key-documentation-pages)

### 6. [Written Tutorials & Blogs](#written-tutorials--blogs)
- [🟢 Beginner-Friendly Articles](#-beginner-friendly-articles)
- [🟡 Intermediate Articles](#-intermediate-articles)
- [🔴 Advanced Articles](#-advanced-articles)
- [📰 Community & Blogs](#-community--blogs)

### 7. [E2E Projects with Working Demos](#e2e-projects-with-working-demos)
- [🎯 Project 1: Sentiment Analysis (Beginner) - Week 2](#-project-1-sentiment-analysis-beginner---week-2)
- [🎯 Project 2: Llama 3 Fine-Tuning with LoRA (Intermediate) - Week 3](#-project-2-llama-3-fine-tuning-with-lora-intermediate---week-3)
- [🎯 Project 3: Multi-RAG Agent with FastAPI (Advanced) - Week 6](#-project-3-multi-rag-agent-with-fastapi-advanced---week-6)
- [🎯 Project 4: Production Deployment with TensorRT (Advanced)](#-project-4-production-deployment-with-tensorrt-advanced)

### 8. [Advanced Topics](#advanced-topics)

#### 8.1 [Megatron Core: High-Performance Training Engine](#1-megatron-core-high-performance-training-engine)
- [What is Megatron Core?](#what-is-megatron-core)
- [Core Problems Megatron Core Solves](#core-problems-megatron-core-solves)
- [Parallelism Types in Megatron Core](#parallelism-types-in-megatron-core)
  - [1. Tensor Parallelism (TP) - Split Individual Layers Horizontally](#1-tensor-parallelism-tp---split-individual-layers-horizontally)
  - [2. Pipeline Parallelism (PP) - Split Layers Vertically (Assembly Line)](#2-pipeline-parallelism-pp---split-layers-vertically-assembly-line)
  - [3. Data Parallelism (DP) - Replicate Model, Different Data](#3-data-parallelism-dp---replicate-model-different-data)
  - [4. Sequence Parallelism (SP) - Split Long Sequences](#4-sequence-parallelism-sp---split-long-sequences)
  - [5. Context Parallelism (CP) - Handle Million+ Token Contexts](#5-context-parallelism-cp---handle-million-token-contexts)
  - [6. 3D Parallelism (DP + TP + PP) - Combine All Three](#6-3d-parallelism-dp--tp--pp---combine-all-three)
- [Quick Decision Guide](#quick-decision-guide)
- [Megatron Core in NeMo](#megatron-core-in-nemo)
- [When to Use Different Parallelism Strategies](#when-to-use-different-parallelism-strategies)
- [Key Features](#key-features)
- [Performance Benefits](#performance-benefits)
- [Code Example: Using Megatron Core in NeMo](#code-example-using-megatron-core-in-nemo)
- [Monitoring Megatron Training](#monitoring-megatron-training)
- [Best Practices](#best-practices)
- [Resources](#resources)

#### 8.2 [Megatron Bridge: Framework Conversion Tool](#2-megatron-bridge-framework-conversion-tool)
- [What is Megatron Bridge?](#what-is-megatron-bridge)
- [The Problem It Solves](#the-problem-it-solves)
- [How Megatron Bridge Works](#how-megatron-bridge-works)
- [Supported Conversions](#supported-conversions)
- [Conversion Example: Megatron → HuggingFace](#conversion-example-megatron--huggingface)
- [Conversion Example: HuggingFace → NeMo](#conversion-example-huggingface--nemo)
- [Key Conversion Operations](#key-conversion-operations)
- [Advanced Usage: Custom Architectures](#advanced-usage-custom-architectures)
- [Real-World Workflow](#real-world-workflow)
- [Validation After Conversion](#validation-after-conversion)
- [Common Issues & Solutions](#common-issues--solutions)
- [Performance Considerations](#performance-considerations)
- [Integration with NeMo Workflow](#integration-with-nemo-workflow)
- [Quick Reference Commands](#quick-reference-commands)
- [Resources](#resources-1)

#### 8.3 [Distributed Training](#3-distributed-training)

#### 8.4 [NeMo-Run Orchestration & Ray Integration](#2-nemo-run-orchestration--ray-integration)

#### 8.5 [NeMo Skills: Improving LLM Capabilities](#3-nemo-skills-improving-llm-capabilities)
- [What is NeMo Skills?](#what-is-nemo-skills)
- [The Problem NeMo Skills Solves](#the-problem-nemo-skills-solves)
- [How NeMo Skills Works](#how-nemo-skills-works)
- [Code Examples](#code-example-1-synthetic-data-generation)
- [When to Use NeMo Skills](#when-to-use-nemo-skills)

#### 8.6 [Multimodal Models](#4-multimodal-models)

#### 8.7 [Custom Model Architectures](#5-custom-model-architectures)

#### 8.8 [Production Observability](#6-production-observability)

#### 8.9 [Guardrails & Safety](#7-guardrails--safety)

#### 8.10 [Understanding Checkpoints in LLMs](#understanding-checkpoints-in-llms)
- [What is a Checkpoint?](#what-is-a-checkpoint)
- [Checkpoint Contents](#checkpoint-contents)
- [Where Checkpoints are Saved in Training](#where-checkpoints-are-saved-in-training)
- [Checkpoint Fine-Tuning vs Full Training](#checkpoint-fine-tuning-vs-full-training)
- [Understanding LoRA Target Modules](#understanding-lora-target-modules)
  - [Attention Projection Matrices](#attention-projection-matrices)
  - [Feed-Forward Network (FFN) Projections](#feed-forward-network-ffn-projections)
  - [Other Common Target Modules](#other-common-target-modules)
  - [Choosing Target Modules](#choosing-target-modules)

#### 8.11 [TensorRT-LLM vs vLLM: Inference Engine Comparison](#tensorrt-llm-vs-vllm-inference-engine-comparison)
- [What is TensorRT-LLM?](#what-is-tensorrt-llm)
- [Quick Comparison](#quick-comparison)
- [Detailed Breakdown](#detailed-breakdown)
- [Performance Benchmarks](#performance-llama-2-70b)
- [Code Examples](#code-examples)
- [Recommendation](#recommendation-for-nemo-users)

#### 8.12 [Mamba vs Transformer: State Space Models Explained](#mamba-vs-transformer-state-space-models-explained)
- [What is Mamba?](#what-is-mamba)
- [The Core Problem with Transformers](#the-core-problem-with-transformers)
- [How Mamba Solves It](#how-mamba-solves-it)
- [Key Differences: Transformer vs Mamba](#key-differences-transformer-vs-mamba)
- [Performance Comparison](#performance-comparison-1)
- [When to Use Each](#when-to-use-each)

---

## What is NVIDIA NeMo?

**NVIDIA NeMo** (Neural Modules) is a **modular, enterprise-grade, cloud-native framework** for building, training, customizing, and deploying generative AI models. [web:409][web:412]

### Key Characteristics:
- **Open-source** (Apache 2.0 license)
- **Production-ready** enterprise solution
- **Multi-domain support**: LLMs, ASR (speech recognition), TTS (text-to-speech), Multimodal
- **GPU-accelerated** (requires NVIDIA GPUs)
- **Cloud-native** (Kubernetes, Docker support)
- **Configuration-driven** (YAML-based setup)
- **2.0 release** introduces NeMo-Run for scalable orchestration

### Primary Use Cases:
✅ Fine-tune large language models (Llama, Nemotron, etc.)
✅ Build custom RAG systems
✅ Deploy speech recognition systems
✅ Create text-to-speech applications
✅ Build AI agents
✅ Scale training to thousands of GPUs

---

## Architecture & Components

### Core Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    NeMo Framework                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  NeMo Core     │  │ Collections  │  │ NeMo-Run     │   │
│  │  (Foundation)  │  │ (Models)     │  │ (Orchestr.)  │   │
│  └────────────────┘  └──────────────┘  └──────────────┘   │
│                                                              │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  NeMo Curator  │  │   NeMo Agent │  │ NeMo Skills  │   │
│  │  (Data)        │  │  (Agents)    │  │ (Training)   │   │
│  └────────────────┘  └──────────────┘  └──────────────┘   │
│                                                              │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   TensorRT     │  │  vLLM        │  │  Triton      │   │
│  │   (Inference)  │  │  (Inference) │  │  (Serving)   │   │
│  └────────────────┘  └──────────────┘  └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         Built on: PyTorch, Hydra, PyTorch Lightning
```

### Key Components Explained [web:406][web:408]

| Component | Purpose | Examples |
|-----------|---------|----------|
| **NeMo Core** | Foundational elements for training/inference | Neural Module Factory, experiment management |
| **Collections** | Domain-specific models and training scripts | LLM, ASR, TTS, Multimodal collections |
| **Neural Modules** | Building blocks (encoders, decoders, etc.) | Interconnectable trainable components |
| **NeMo Curator** | Data processing at scale | Data cleaning, filtering, synthetic generation |
| **NeMo-Run** | Experiment orchestration | Distributed training, hyperparameter sweeps |
| **NeMo Agent** | Agentic workflows | RAG agents, multi-tool workflows, FastAPI integration |
| **NeMo Skills** | Training pipeline improvements | Math reasoning, synthetic data generation |
| **Inference Libraries** | Optimized deployment | TensorRT-LLM, vLLM, Triton Inference Server |

### NeMo Microservices (Production Deployment)

NeMo also provides **microservices architecture** for production fine-tuning and inference workflows:

| Service | Phase | Purpose | Endpoint |
|---------|-------|---------|----------|
| **NeMo Customizer** | Training | Fine-tuning job management (LoRA/SFT) | `CUSTOMIZER_BASE_URL` |
| **Entity Store** | Organization | Namespace/project management | `ENTITY_STORE_BASE_URL` |
| **Data Store** | Storage | Training dataset storage (S3-compatible) | `DATA_STORE_BASE_URL` |
| **NIM (NVIDIA Inference)** | Inference | Serving fine-tuned models | `NIM_BASE_URL` |
| **Deployment** | Orchestration | Model deployment management | `DEPLOYMENT_BASE_URL` |

**Key Distinction**:
- **NeMo Customizer** ≠ **NIM** (separate independent services)
- **Customizer** = Training/fine-tuning phase
- **NIM** = Inference/serving phase (uses fine-tuned models)

**Typical Workflow**:
```
1. Data Store → Upload training datasets
2. Customizer → Create & run fine-tuning jobs
3. Entity Store → Manage organization/namespaces
4. NIM → Serve fine-tuned models for inference
5. Deployment → Deploy models to production
```

---

## Learning Path (Week-by-Week)

### 🎯 Beginner Track (4 Weeks)

#### **Week 1: Foundations & Setup**

**Days 1-2: Conceptual Understanding**
- Watch: "What is NVIDIA NeMo?" (2-min intro)
- Read: NeMo Framework Overview [web:413]
- Key concepts: Neural modules, collections, configurations, YAML

**Days 3-4: Installation & Environment**
- Install NeMo locally or Docker
- Run verification scripts
- Explore NGC models

**Days 5-7: First Model Inference**
- Download pre-trained model
- Run inference with CLI
- Understand model outputs

**Resources**:
- Video: "Building and Deploying Generative AI Models" [web:393]
- Docs: NeMo Fundamentals [web:412]
- Tutorial: "Intro to NVIDIA NeMo" [web:405]

---

#### **Week 2: Data Preparation & Basics**

**Days 1-3: Data Curation**
- Learn NeMo Curator fundamentals
- Process datasets (TinyStories, Enron)
- Filter, clean, augment data

**Days 4-7: First Training Run**
- Download a small dataset
- Configure YAML file
- Run simple fine-tuning job
- Monitor training logs

**Resources**:
- Tutorial: "Text Classification with NeMo" [web:405]
- Docs: NeMo Curator API [web:396]
- Example: Fine-tune on SQuAD dataset [web:394]

**Quick Project**: Fine-tune sentiment classifier on 100K movie reviews

---

#### **Week 3: LLM Fine-Tuning (LoRA)**

**Days 1-3: Parameter-Efficient Fine-Tuning (PEFT)**
- Understand LoRA, P-tuning, adapters
- Watch PEFT tutorials
- Compare parameter efficiency

**Days 4-7: Hands-On LoRA**
- Fine-tune Llama 3 8B with LoRA
- Configure optimizer, learning rate, batch size
- Evaluate on validation set

**Resources**:
- Video: "Improve LLM with NeMo-Skills" [web:391]
- Tutorial: "PEFT For LLM Using NeMo" [web:403]
- Example: SQuAD fine-tuning [web:394]
- Docs: LLM Collection [web:397]

**Quick Project**: Fine-tune Llama 3 on custom Q&A dataset using LoRA

---

#### **Week 4: Inference & Deployment**

**Days 1-3: Inference Options**
- Learn deployment paths (TensorRT, vLLM, Triton)
- Export models
- Compare inference performance

**Days 4-7: Deploy Model**
- Export fine-tuned model to TensorRT-LLM
- Run Triton Inference Server
- Create REST API endpoint

**Resources**:
- Docs: Deploy NeMo Models [web:410]
- Docs: LLM Deployment [web:407]
- Tutorial: NeMo Microservices [web:398]

**Quick Project**: Deploy fine-tuned model as REST API

---

### 🚀 Intermediate Track (3 Weeks)

#### **Week 5: Advanced Training Techniques**

**Topics**:
- Distributed training (TP, DP, PP)
- Quantization (FP8, INT8)
- Checkpointing & resuming
- Hyperparameter tuning

**Resources**:
- Video: "LLMs at Scale on Azure" [web:417]
- Docs: Advanced Training [web:397]

---

#### **Week 6: NeMo Agent & RAG**

**Topics**:
- Build RAG agents
- Tool integration
- Multi-agent workflows
- FastAPI deployment

**Resources**:
- Video: "Create AI Agents with NeMo" [web:390]
- Video: "Production-Ready NeMo Agents" [web:392]
- Docs: NeMo Agent Toolkit [web:415]

**Project**: Build RAG agent that answers domain-specific questions

---

#### **Week 7: Production Deployment**

**Topics**:
- NVIDIA NIM for enterprise
- Monitoring & observability
- Rate limiting, caching, auth
- Multi-node orchestration

**Resources**:
- Docs: Production Deployment [web:410]
- Docs: NIM Integration [web:413]

**Project**: Deploy multi-agent system with monitoring

---

### 🎓 Expert Track (4+ Weeks)

**Topics**:
- Custom model architectures
- Pre-training from scratch
- Multimodal models (LLM + vision)
- Research contributions

---

## YouTube Video Resources

### ⭐ ESSENTIAL Videos (Watch First)

| Video Title | Channel | Duration | Level | Key Topics | Link |
|------------|---------|----------|-------|-----------|------|
| **Building and Deploying Generative AI Models** | NVIDIA Developer | 2:15 min | Beginner | Framework overview, multi-modality, guardrails | [Watch](https://www.youtube.com/watch?v=gTwLLhebOcQ) |
| **Build Your Own Custom Generative AI Model** | Voicebot + Synthedia | 19:32 min | Beginner | Data curation, training, guardrails, monitoring | [Watch](https://www.youtube.com/watch?v=fijcGA6Kk5k) |
| **Improve LLM Abilities Using NeMo-Skills** | NVIDIA Developer | 18:26 min | Intermediate | NeMo-Skills pipeline, synthetic data, math reasoning | [Watch](https://www.youtube.com/watch?v=rwpsofHdAOo) |
| **Create Your Own AI Agent with NeMo** | NVIDIA Developer | 16:50 min | Intermediate | Multi-RAG agents, FastAPI, configuration workflows | [Watch](https://www.youtube.com/watch?v=NsogD7UhZ4Q) |
| **Make AI Agents Production-Ready** | NVIDIA Developer | 8+ hours | Advanced | Observability, evaluation, deployment, multi-agent orchestration | [Watch](https://www.youtube.com/watch?v=N72-fAI5V9k) |
| **NVIDIA NeMo Microservices Tutorial** | Mervin Praison | 8:57 min | Intermediate | Fine-tuning with microservices, Docker, data preparation | [Watch](https://www.youtube.com/watch?v=J60tNkHkOHY) |
| **LLMs at Scale with NeMo on Azure** | NVIDIA Developer | 29:15 min | Advanced | Distributed training, deployment, scaling strategies | [Watch](https://www.youtube.com/watch?v=QWvrCuuFsjg) |
| **Introduction to NVIDIA NeMo** | NVIDIA Developer | Legacy | Beginner | NeMo basics, neural modules, collections | [Watch](https://www.youtube.com/watch?v=2kTZ0oST8wg) |

### 🎬 Playlist Collections

| Playlist | Link | Content |
|----------|------|---------|
| **NVIDIA NeMo Official** | [YouTube Channel](https://www.youtube.com/@NVIDIADeveloper) | All official NeMo videos, live sessions, announcements |
| **NeMo Developer Talks** | Search YouTube | Community talks, use cases, tips |

### 📝 Video Learning Tips

1. **Watch at 1.5x speed** for efficiency
2. **Code along** - pause and implement examples
3. **Take notes** on configuration patterns
4. **Bookmark** parts you need to review

---

## Official Documentation

### 📖 Primary Documentation Sources

| Resource | URL | Best For |
|----------|-----|----------|
| **NeMo Framework User Guide** | [docs.nvidia.com/nemo-framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/) | Complete reference, all topics |
| **NeMo LLM Collection** | [docs.nvidia.com/nemo-framework/.../llms](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/index.html) | LLM-specific guides, models, training |
| **NeMo Tutorials** | [docs.nvidia.com/nemo-framework/.../playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) | Step-by-step hands-on tutorials [web:396] |
| **NeMo Deployment Guide** | [docs.nvidia.com/nemo-framework/.../deployment](https://docs.nvidia.com/nemo-framework/user-guide/24.12/deployment/index.html) | Inference, TensorRT, vLLM, Triton |
| **NeMo Models Reference** | [docs.nvidia.com/nemo-framework/.../models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/core.html) | Model configuration, architecture details |
| **NeMo Fundamentals** | [docs.nvidia.com/nemo-framework/.../fundamentals](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/fundamentals.html) | Core concepts, training basics |

### 🔗 Key Documentation Pages

**Getting Started**:
- [Overview](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) [web:413]
- [Fundamentals](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/fundamentals.html) [web:412]
- [Tutorials Index](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) [web:396]

**LLM Specific**:
- [LLM Collection](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/index.html) [web:397]
- [LLM Deployment](https://docs.nvidia.com/nemo-framework/user-guide/24.12/deployment/llm/index.html) [web:407]

**Data & Training**:
- [NeMo Curator API Documentation](https://docs.nvidia.com/nemo-curator/)
- [NeMo-Run Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/index.html)

**Deployment**:
- [Deployment Paths](https://docs.nvidia.com/nemo-framework/user-guide/24.12/deployment/index.html) [web:410]
- [NIM Integration](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) [web:413]

---

## Written Tutorials & Blogs

### 🟢 Beginner-Friendly Articles

| Title | Source | URL | Focus |
|-------|--------|-----|-------|
| **Intro to NVIDIA NeMo Tutorial** | Exxact Corp | [Read](https://www.exxactcorp.com/blog/Deep-Learning/Intro-to-NVIDIA-NeMo) | Step-by-step setup, sentiment analysis demo [web:405] |
| **Understanding NVIDIA NeMo** | Scaleway | [Read](https://www.scaleway.com/en/docs/gpu/reference-content/understanding-nvidia-nemo/) | Components overview, features, architecture [web:406] |
| **NVIDIA NeMo: How It Works** | Weka | [Read](https://www.weka.io/learn/glossary/ai-ml/nvidia-nemo/) | Framework explanation, use cases [web:409] |

### 🟡 Intermediate Articles

| Title | Source | URL | Focus |
|-------|--------|-----|-------|
| **Fine-tune GPT-OSS with NeMo** | LinkedIn/Medium | [Read](https://www.linkedin.com/posts/amandamsaunders_specializedai-aiagents-openai-activity-7361896335412801536-oQ9l) | Docker setup, fine-tuning workflow [web:399] |
| **LLM Fine-Tuning with NeMo Framework** | PaaS-up | [Read](https://ideas.paasup.io/global/nemo2en/) | SQuAD dataset, LoRA, NeMo-Run configuration [web:394] |
| **PEFT For LLM Using NeMo** | CDAC Presentation | [PDF Download](https://airawat.cdac.in/static/media/Day-6_Session-2-1_PEFT_For_LLM_Using_NeMo.30870735cc1e3ccf3c49.pdf) | LoRA, adapters, training recipes [web:403] |

### 🔴 Advanced Articles

| Title | Source | URL | Focus |
|-------|--------|-----|-------|
| **Distributed Training with NeMo on EKS** | AWS Blog | [Read](https://aws.amazon.com/blogs/machine-learning/accelerate-your-generative-ai-distributed-training-workloads-with-the-nvidia-nemo-framework/) | Multi-node setup, Kubernetes, scaling [web:400] |

### 📰 Community & Blogs

- **NVIDIA Official Blog**: nvidia.com/en-us/ai-data-science/
- **Hugging Face NeMo**: huggingface.co/docs (NeMo integration)
- **Reddit r/NeMo**: Community discussions
- **Medium**: Search "NVIDIA NeMo" for community articles

---

## E2E Projects with Working Demos

### 🎯 Project 1: Sentiment Analysis (Beginner) - Week 2

**Goal**: Fine-tune NeMo text classifier on movie reviews

**Duration**: 3-4 hours

**Components**:
- Data: Download sentiment dataset
- Model: Use TextClassification collection
- Training: Fine-tune on GPU
- Evaluation: Compute accuracy, F1 score
- Deployment: Simple inference script

**Code Example**:
```python
# Step 1: Install and import
import nemo
from nemo.collections import nlp

# Step 2: Download pre-trained model
pretrained_model = "text_classification_domain_adapter"
model = nlp.models.TextClassificationModel.restore_from_pretrained(
    model_name=pretrained_model
)

# Step 3: Fine-tune
# Configure YAML (training_config.yaml)
# Run training script
!python -m examples.nlp.text_classification.text_classification_with_pretrained_models \
  --config-name=text_classification_config.yaml

# Step 4: Inference
logits = model.predict(queries=["I love this movie!"])
```

**Deliverables**:
- ✅ Trained model checkpoint
- ✅ Evaluation metrics report
- ✅ Inference script

**Resources**:
- Tutorial: Text Classification [web:405]
- Example: NeMo examples/nlp/text_classification

---

### 🎯 Project 2: Llama 3 Fine-Tuning with LoRA (Intermediate) - Week 3

**Goal**: Fine-tune Llama 3 8B on custom Q&A dataset using LoRA

**Duration**: 6-8 hours

**Components**:
- Model: Llama 3 8B (from Hugging Face)
- Dataset: SQuAD or custom Q&A
- Method: LoRA (Parameter-Efficient Fine-Tuning)
- Optimization: Adam, learning rate scheduling
- Evaluation: Validation loss, BLEU score

**Code Example**:
```python
from nemo.collections import llm
from nemo_run import run, Partial
import torch

# Step 1: Define model config
def llama3_8b():
    return llm.Llama3Model.configure(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        vocab_size=128256
    )

# Step 2: Configure LoRA
def lora_config():
    return llm.peft.LoRAConfig(
        target_modules=["linear_qkv", "linear_fc1", "linear_fc2"],
        r=8,  # rank
        lora_alpha=16,
        dropout=0.1
    )

# Step 3: Setup training
def trainer_config():
    return run.Config(
        name="llama3_lora_finetuning",
        max_epochs=3,
        learning_rate=1e-4,
        batch_size=8
    )

# Step 4: Run fine-tuning
recipe = run.Partial(
    llm.finetune,
    model=llama3_8b(),
    peft=lora_config(),
    trainer=trainer_config()
)

run.run(recipe)
```

**Deliverables**:
- ✅ Fine-tuned LoRA adapter
- ✅ Model performance comparison
- ✅ Inference with merged weights

**Resources**:
- Tutorial: SQuAD Fine-tuning [web:394]
- Tutorial: NeMo-Skills [web:391]
- Video: "Improve LLM Abilities" [web:391]
- Docs: LLM Training [web:397]

---

### 🎯 Project 3: Multi-RAG Agent with FastAPI (Advanced) - Week 6

**Goal**: Build multi-tool RAG agent with NeMo Agent toolkit and deploy as microservice

**Duration**: 8-12 hours

**Components**:
- RAG Engines: 3 different knowledge sources
- Agent: React agent with tool calling
- LLM: Llama 3 70B instruct
- Embedding Model: Text embedding service
- API: FastAPI microservice
- Frontend: Basic UI (optional)

**Architecture**:
```
User Request → FastAPI Server → NeMo Agent
                                    ├─ LLM (Llama 3 70B)
                                    ├─ RAG Tool 1 (Docs)
                                    ├─ RAG Tool 2 (Code)
                                    └─ RAG Tool 3 (QA)
                                        ↓
                                    Responses
```

**Code Example**:
```python
from nemo.agent import ReactAgent, Tool, RAG
from nemo.collections import llm
from fastapi import FastAPI
from pydantic import BaseModel

# Step 1: Define RAG Tools
doc_rag = RAG(
    knowledge_base="docs.txt",
    embedding_model="nvidia/NV-Embed-v2"
)

code_rag = RAG(
    knowledge_base="codebase.txt",
    embedding_model="nvidia/NV-Embed-v2"
)

# Step 2: Create tools
tools = [
    Tool(name="search_docs", func=doc_rag.search),
    Tool(name="search_code", func=code_rag.search),
    Tool(name="calculator", func=eval),  # Simple calculator
]

# Step 3: Initialize agent
llm_client = llm.Llama3(
    model_name="meta-llama/Llama-3-70b-instruct"
)

agent = ReactAgent(
    llm=llm_client,
    tools=tools,
    prompt_template="You are a helpful assistant..."
)

# Step 4: Create FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query(request: QueryRequest):
    response = agent.run(request.query)
    return {"response": response}

# Step 5: Run
# uvicorn main:app --reload
```

**YAML Configuration** (config.yaml):
```yaml
general:
  frontend: fastapi
  
llm:
  model_name: meta-llama/Llama-3-70b-instruct
  
embedder:
  model_name: nvidia/NV-Embed-v2

functions:
  doc_search:
    type: RAG
    knowledge_base: docs.txt
  
  code_search:
    type: RAG
    knowledge_base: codebase.txt
```

**Deliverables**:
- ✅ Functional multi-agent system
- ✅ FastAPI REST endpoints
- ✅ Inference latency benchmarks
- ✅ Docker container
- ✅ Documentation

**Resources**:
- Video: "Create AI Agents" [web:390]
- Video: "Production-Ready Agents" [web:392]
- Tutorial: NeMo Microservices [web:398]
- Example: nemo/examples/llm/agents

---

### 🎯 Project 4: Production Deployment with TensorRT (Advanced)

**Goal**: Export fine-tuned model and deploy with TensorRT-LLM for optimal inference

**Duration**: 4-6 hours

**Components**:
- Export: Convert NeMo → TensorRT-LLM format
- Quantization: FP8 or INT8
- Server: Triton Inference Server
- Client: REST API consumer

**Code Example**:
```python
from nemo import export
import torch

# Step 1: Export to TensorRT-LLM
nemo_model_path = "llama3_finetuned.nemo"
export_path = "llama3_trt"

export.export_llm_to_tensorrt_llm(
    nemo_checkpoint_path=nemo_model_path,
    export_dir=export_path,
    tensor_parallel_size=1,
    quantization="fp8"  # or "int8"
)

# Step 2: Deploy with Triton
# Use: tritonserver --model-repository ./triton_model_repo

# Step 3: Query model
from tritonclient.grpc import InferenceServerClient

client = InferenceServerClient(url="localhost:8001")
response = client.infer(
    model_name="llama3",
    inputs=[...],
    outputs=[...]
)
```

**Deliverables**:
- ✅ TensorRT-LLM exported model
- ✅ Triton deployment config
- ✅ Benchmark report (latency, throughput)
- ✅ Docker compose setup

**Resources**:
- Docs: LLM Deployment [web:407]
- Docs: TensorRT-LLM Export [web:410]

---

## Advanced Topics

### 1. **Megatron Core: High-Performance Training Engine**

#### What is Megatron Core?

**Megatron Core** is NVIDIA's **high-performance library for training massive language models** (GPT, BERT, T5, etc.). It's the underlying engine that powers NeMo's ability to train models with billions to trillions of parameters efficiently across thousands of GPUs.

**Key Purpose**: Solve the fundamental problem of training models too large to fit on a single GPU.

#### Core Problems Megatron Core Solves

| Problem | Megatron Solution |
|---------|------------------|
| Model > GPU memory (e.g., 175B params needs ~700GB, but GPU has 80GB) | Split model using tensor/pipeline parallelism |
| Extremely slow training | Distributed training with optimal communication |
| Inefficient GPU utilization | Advanced parallelism strategies |
| Memory bottlenecks | Sequence parallelism, context parallelism |

#### Parallelism Types in Megatron Core

Think of training a huge model like building a massive LEGO castle that's too big for one person. Different parallelism types are different strategies for dividing the work.

---

**1. Tensor Parallelism (TP)** - Split Individual Layers Horizontally

**Analogy**: Cutting a pizza horizontally into slices - each person gets one slice of the same pizza.

**How it works**: Each layer's weight matrix is split across multiple GPUs. All GPUs work on the same data but compute different parts of each layer.

**Visual Example - Attention Layer**:
```
Without TP (Single GPU):
┌─────────────────────────────────────┐
│  Attention Layer (12GB)             │
│  ├── Query weights:  4GB            │  All on GPU 1
│  ├── Key weights:    4GB            │
│  └── Value weights:  4GB            │
└─────────────────────────────────────┘

With TP = 4 GPUs (Split horizontally):
GPU 1: [Q: 1GB | K: 1GB | V: 1GB] = 3GB
GPU 2: [Q: 1GB | K: 1GB | V: 1GB] = 3GB
GPU 3: [Q: 1GB | K: 1GB | V: 1GB] = 3GB
GPU 4: [Q: 1GB | K: 1GB | V: 1GB] = 3GB

Process:
1. Input data is broadcast to all 4 GPUs
2. Each GPU computes partial attention (1/4 of the heads)
3. Results are gathered and combined
4. Final output sent to next layer
```

**Real Example - Matrix Multiplication**:
```python
# Original (Single GPU):
Input (batch=8, seq=2048, hidden=8192) × Weight (8192×8192) = Output (8, 2048, 8192)
Memory: ~256MB input + ~256MB weight + ~256MB output = ~768MB

# With TP=4 (Split across 4 GPUs):
GPU 1: Input × Weight[0:2048, :]    = Partial_Output_1 (8, 2048, 2048)
GPU 2: Input × Weight[2048:4096, :] = Partial_Output_2 (8, 2048, 2048)
GPU 3: Input × Weight[4096:6144, :] = Partial_Output_3 (8, 2048, 2048)
GPU 4: Input × Weight[6144:8192, :] = Partial_Output_4 (8, 2048, 2048)

Final: Concatenate [Partial_1, Partial_2, Partial_3, Partial_4] = Full Output
Memory per GPU: ~256MB input + ~64MB weight + ~64MB output = ~384MB
Memory saved: 768MB → 384MB per GPU (50% reduction)
```

**When to use**: Model's single layer doesn't fit in one GPU, GPUs are on the same machine (requires fast interconnect like NVLink).

---

**2. Pipeline Parallelism (PP)** - Split Layers Vertically (Assembly Line)

**Analogy**: Car assembly line - Station 1 builds the frame, Station 2 adds the engine, Station 3 adds wheels, Station 4 paints.

**How it works**: Different layers of the model are placed on different GPUs. Data flows through GPUs like a pipeline.

**Visual Example - 32-Layer Transformer**:
```
Without PP (impossible for large models):
GPU 1: [Layers 1-32] ← All 32 layers

With PP = 4 GPUs (Split vertically):
┌─────────────┐
│   GPU 1     │  Layers 1-8    (Embedding + First 8 transformer blocks)
└──────┬──────┘
       │ Forward pass data flows down ↓
┌──────▼──────┐
│   GPU 2     │  Layers 9-16   (Next 8 transformer blocks)
└──────┬──────┘
       │
┌──────▼──────┐
│   GPU 3     │  Layers 17-24  (Next 8 transformer blocks)
└──────┬──────┘
       │
┌──────▼──────┐
│   GPU 4     │  Layers 25-32  (Last 8 blocks + LM head)
└─────────────┘
```

**Problem - GPU Bubble (Inefficiency)**:

**What is a "GPU Bubble"?**

A GPU bubble is **wasted time when GPUs are idle** waiting for data from previous pipeline stages. Think of it like a traffic jam where cars (data) are stuck waiting, and the road (GPU) sits empty.

**Why Does This Happen?**

In pipeline parallelism, data must flow sequentially through stages:
1. GPU 1 finishes → sends data to GPU 2
2. GPU 2 finishes → sends data to GPU 3
3. GPU 3 finishes → sends data to GPU 4

During this flow, earlier GPUs sit idle waiting for the pipeline to complete!

**Beginner's Example: Assembly Line Analogy**

Imagine a car assembly line with 4 stations building ONE car:

```
Station 1 (GPU 1): Install engine    [Takes 4 minutes]
Station 2 (GPU 2): Install wheels    [Takes 4 minutes]
Station 3 (GPU 3): Paint car         [Takes 4 minutes]
Station 4 (GPU 4): Final inspection  [Takes 4 minutes]

Timeline for 1 car:
Minutes 1-4:   Station 1 [Working] → Station 2 [IDLE] → Station 3 [IDLE] → Station 4 [IDLE]
Minutes 5-8:   Station 1 [IDLE]    → Station 2 [Working] → Station 3 [IDLE] → Station 4 [IDLE]
Minutes 9-12:  Station 1 [IDLE]    → Station 2 [IDLE] → Station 3 [Working] → Station 4 [IDLE]
Minutes 13-16: Station 1 [IDLE]    → Station 2 [IDLE] → Station 3 [IDLE] → Station 4 [Working]

Total time: 16 minutes
Efficiency: Only 1 station working at a time = 25% utilization!
Wasted capacity: 75% (12 minutes of idle time across 4 stations)
```

**In GPU Terms (Without Micro-batching)**:
```
Processing 1 batch through 4-GPU pipeline:

Time 1-4:   GPU1 [████] → GPU2 [    ] → GPU3 [    ] → GPU4 [    ]
            ↑ Working      ↑ Idle      ↑ Idle       ↑ Idle
Time 5-8:   GPU1 [    ] → GPU2 [████] → GPU3 [    ] → GPU4 [    ]
            ↑ Idle         ↑ Working   ↑ Idle       ↑ Idle
Time 9-12:  GPU1 [    ] → GPU2 [    ] → GPU3 [████] → GPU4 [    ]
            ↑ Idle         ↑ Idle      ↑ Working    ↑ Idle
Time 13-16: GPU1 [    ] → GPU2 [    ] → GPU3 [    ] → GPU4 [████]
            ↑ Idle         ↑ Idle      ↑ Idle       ↑ Working

Problem: Only 1 GPU active at a time!
GPU Utilization: 25% (1 out of 4 GPUs working)
GPU Bubble: 75% (wasted idle time)
```

**Why This Is Terrible:**
- You have 4 expensive GPUs but only 1 is working at any time
- Training takes 4× longer than necessary
- You're paying for 4 GPUs but getting the performance of 1 GPU
- The "bubble" is the empty/idle time (shown as `[    ]`)

**Solution - Micro-batching (Filling the Pipeline)**:

**The Fix:** Instead of processing ONE big batch, split it into many small **micro-batches** and keep the pipeline full!


**In GPU Terms (With Micro-batching)**:

Split batch into 8 micro-batches (MB1-MB8):

```
Time 1:  GPU1[MB1] → GPU2[    ] → GPU3[    ] → GPU4[    ]  Utilization: 25%
Time 2:  GPU1[MB2] → GPU2[MB1] → GPU3[    ] → GPU4[    ]  Utilization: 50%
Time 3:  GPU1[MB3] → GPU2[MB2] → GPU3[MB1] → GPU4[    ]  Utilization: 75%
Time 4:  GPU1[MB4] → GPU2[MB3] → GPU3[MB2] → GPU4[MB1]  Utilization: 100% ✅
Time 5:  GPU1[MB5] → GPU2[MB4] → GPU3[MB3] → GPU4[MB2]  Utilization: 100% ✅
Time 6:  GPU1[MB6] → GPU2[MB5] → GPU3[MB4] → GPU4[MB3]  Utilization: 100% ✅
Time 7:  GPU1[MB7] → GPU2[MB6] → GPU3[MB5] → GPU4[MB4]  Utilization: 100% ✅
Time 8:  GPU1[MB8] → GPU2[MB7] → GPU3[MB6] → GPU4[MB5]  Utilization: 100% ✅
Time 9:  GPU1[    ] → GPU2[MB8] → GPU3[MB7] → GPU4[MB6]  Utilization: 75%
Time 10: GPU1[    ] → GPU2[    ] → GPU3[MB8] → GPU4[MB7]  Utilization: 50%
Time 11: GPU1[    ] → GPU2[    ] → GPU3[    ] → GPU4[MB8]  Utilization: 25%

Total GPU-steps: 44 working units
Possible GPU-steps: 11 time steps × 4 GPUs = 44
Average utilization: 44/44 × (28 working)/(44 possible) ≈ 73%

Result: 4 GPUs working simultaneously most of the time!
GPU utilization: ~70-80% (vs 25% before) 🚀
```

**Key Insight:**
- **Warmup phase** (Time 1-3): Pipeline filling up
- **Steady state** (Time 4-8): All GPUs working! Maximum efficiency!
- **Cooldown phase** (Time 9-11): Pipeline draining

**The Math:**
```
Without micro-batching:
- 4 GPUs, 1 batch
- Total time: 4 time units
- GPU utilization: 1/4 = 25%
- Bubble: 75%

With micro-batching (8 micro-batches):
- 4 GPUs, 8 micro-batches
- Total time: 11 time units
- Working GPU-time: 32 units (8 MB × 4 stages)
- Possible GPU-time: 44 units (11 time × 4 GPUs)
- GPU utilization: 32/44 ≈ 73%
- Bubble: 27% (much better!)

Bubble formula: (p-1)/(m+p-1)
Where: p = pipeline stages (4), m = micro-batches (8)
Bubble = (4-1)/(8+4-1) = 3/11 ≈ 27% ✅
```

**More Micro-batches = Less Bubble:**
```
4 micro-batches:  Bubble = 3/7  = 43%
8 micro-batches:  Bubble = 3/11 = 27%
16 micro-batches: Bubble = 3/19 = 16%
32 micro-batches: Bubble = 3/35 = 9%

Trade-off: More micro-batches need more memory (store activations)
Sweet spot: Usually 4-8× the number of pipeline stages
```

**When to use**: Model has many layers, going multi-node (slower interconnect between nodes), can tolerate some pipeline bubble overhead.

---

**3. Data Parallelism (DP)** - Replicate Model, Different Data

**Analogy**: 4 chefs making the same recipe, each cooking different batches of food, then sharing tips to improve the recipe.

**How it works**: Each GPU has a complete copy of the model. Each processes different data batches. Gradients are synchronized across all GPUs.

**Visual Example**:
```
Dataset: 128 samples total (Batch size = 128)

With DP = 4 GPUs:
┌────────────────────┐
│ GPU 1: [Full Model]│ ← Processes samples 1-32
│ Batch: [32 samples]│
└────────────────────┘
┌────────────────────┐
│ GPU 2: [Full Model]│ ← Processes samples 33-64
│ Batch: [32 samples]│
└────────────────────┘
┌────────────────────┐
│ GPU 3: [Full Model]│ ← Processes samples 65-96
│ Batch: [32 samples]│
└────────────────────┘
┌────────────────────┐
│ GPU 4: [Full Model]│ ← Processes samples 97-128
│ Batch: [32 samples]│
└────────────────────┘

Training Step:
1. Forward pass: Each GPU processes its batch independently
2. Backward pass: Each GPU computes gradients
3. AllReduce: Gradients averaged across all GPUs
   GPU1_grad + GPU2_grad + GPU3_grad + GPU4_grad
   Average_grad = ────────────────────────────────
                              4
4. Update: All GPUs update weights with same average gradient
5. Result: All 4 models stay synchronized

Effective batch size = 32 × 4 = 128 (4x larger!)
Training speed: 4x faster (4 batches processed simultaneously)
```

**When to use**: Model fits in one GPU, want to process more data faster, easiest to implement.

---

**4. Sequence Parallelism (SP)** - Split Long Sequences

**Analogy**: Reading a long book - each person reads different chapters simultaneously, then shares summaries.

**How it works**: Long input sequences are split across GPUs. Combined with Tensor Parallelism to reduce activation memory.

**Visual Example - 100K Token Sequence**:
```
Without SP (TP=4 only):
GPU 1: [All 100K tokens] + [Weight slice 1]  ← 100K tokens in memory
GPU 2: [All 100K tokens] + [Weight slice 2]  ← 100K tokens in memory
GPU 3: [All 100K tokens] + [Weight slice 3]  ← 100K tokens in memory
GPU 4: [All 100K tokens] + [Weight slice 4]  ← 100K tokens in memory
Total activation memory: 4 × 100K = 400K token-worth

With SP + TP=4:
GPU 1: [Tokens 1-25K]    + [Weight slice 1]  ← 25K tokens in memory
GPU 2: [Tokens 25K-50K]  + [Weight slice 2]  ← 25K tokens in memory
GPU 3: [Tokens 50K-75K]  + [Weight slice 3]  ← 25K tokens in memory
GPU 4: [Tokens 75K-100K] + [Weight slice 4]  ← 25K tokens in memory
Total activation memory: 4 × 25K = 100K token-worth

Memory saved: 400K → 100K (4x reduction!) ✅
```

**Real Example - Processing Long Document**:
```python
# Document: "The quick brown fox jumps over the lazy dog..."
# Total tokens: 100,000

# Without SP (4 GPUs, TP=4):
GPU 1: Processes all 100K tokens, computes 1/4 of attention heads
GPU 2: Processes all 100K tokens, computes 1/4 of attention heads
GPU 3: Processes all 100K tokens, computes 1/4 of attention heads
GPU 4: Processes all 100K tokens, computes 1/4 of attention heads
Activation memory per GPU: ~8GB (for 100K tokens)

# With SP+TP (4 GPUs):
GPU 1: "The quick brown..." (tokens 1-25K), 1/4 attention heads
GPU 2: "fox jumps over..." (tokens 25K-50K), 1/4 attention heads
GPU 3: "the lazy dog..." (tokens 50K-75K), 1/4 attention heads
GPU 4: "and runs fast..." (tokens 75K-100K), 1/4 attention heads
Activation memory per GPU: ~2GB (for 25K tokens)

Result: 75% memory saved! Can now handle 4x longer sequences.
```

**When to use**: Very long sequences (100K+ tokens), combined with TP, memory bottleneck from activations.

---

**5. Context Parallelism (CP)** - Handle Million+ Token Contexts

**Analogy**: Multiple people reading different sections of a massive encyclopedia simultaneously, using a smart system to share information.

**How it works**: Extremely long contexts split across GPUs using Ring Attention algorithm. Each GPU computes attention for its segment while passing information in a ring.

**Visual Example - 1 Million Token Context**:
```
Processing entire codebase (1M tokens) across 8 GPUs:

┌──────────────┐
│ GPU 1        │  Tokens 1-125K     (Files: main.py, utils.py, ...)
└──────┬───────┘
       │ Ring communication ↓
┌──────▼───────┐
│ GPU 2        │  Tokens 125K-250K  (Files: models.py, train.py, ...)
└──────┬───────┘
       │
┌──────▼───────┐
│ GPU 3        │  Tokens 250K-375K  (Files: data.py, config.py, ...)
└──────┬───────┘
       │
┌──────▼───────┐
│ GPU 4        │  Tokens 375K-500K  (Files: tests/, docs/, ...)
└──────┬───────┘
       │
┌──────▼───────┐
│ GPU 5        │  Tokens 500K-625K
└──────┬───────┘
       │
┌──────▼───────┐
│ GPU 6        │  Tokens 625K-750K
└──────┬───────┘
       │
┌──────▼───────┐
│ GPU 7        │  Tokens 750K-875K
└──────┬───────┘
       │
┌──────▼───────┐
│ GPU 8        │  Tokens 875K-1M
└──────────────┘

Ring Attention Process:
Step 1: GPU 1 computes attention for segment 1
        Passes KV cache to GPU 2 →
Step 2: GPU 2 computes attention for segment 2 + receives KV from GPU 1
        Passes combined KV to GPU 3 →
...continues around the ring...

Result: Full attention computed across all 1M tokens
Memory per GPU: ~125K tokens (1/8 of total)
```

**When to use**: Context length > 100K tokens, analyzing multiple documents, long-form generation (novels, research papers).

---

**6. 3D Parallelism (DP + TP + PP)** - Combine All Three

**What is 3D Parallelism?**

3D Parallelism combines **Data Parallelism (DP)**, **Tensor Parallelism (TP)**, and **Pipeline Parallelism (PP)** to train models that are:
- Too large for individual layers to fit on one GPU (need TP)
- Have too many layers to fit even with TP (need PP)
- Need high throughput and larger effective batch sizes (need DP)

**Think of it as 3 dimensions of splitting:**
1. **TP (Horizontal)**: Split each layer's weights across GPUs
2. **PP (Vertical)**: Split layers across GPUs (like floors in a building)
3. **DP (Replication)**: Multiple identical copies processing different data


---

### Why We Need 3D Parallelism

**Problem Scenario: Training Llama 3 70B**

```
Model: 70 billion parameters
Single GPU memory: 80 GB
Model size: ~280 GB (FP16)

❌ Won't fit on 1 GPU
❌ TP=8 alone: 280GB/8 = 35GB per GPU - fits, but...
   • Limited to 8 GPUs (TP usually maxes at 8 due to communication overhead)
   • Can't scale beyond single node
   • Small effective batch size → slow training

✅ Solution: 3D Parallelism (TP=8, PP=4, DP=2) = 64 GPUs
   • Model fits: 280GB / (8×4) = 8.75GB per GPU
   • Scales across nodes: PP allows multi-node
   • 2x larger batch: DP=2 doubles throughput
   • Training speed: 40,000 tokens/sec
```

---

### How TP and PP Work Together - The Key Insight

**The Confusion:** "If PP splits layers sequentially across GPUs, how can TP split each layer across multiple GPUs?"

**The Answer:** TP and PP operate at **different granularities** on **different sets of GPUs**.

#### Visual Breakdown - Single Pipeline Stage

**Llama 3 70B has 80 transformer layers. With PP=4:**
```
Pipeline Stage 1: Layers 1-20   (on GPU group 1)
Pipeline Stage 2: Layers 21-40  (on GPU group 2)
Pipeline Stage 3: Layers 41-60  (on GPU group 3)
Pipeline Stage 4: Layers 61-80  (on GPU group 4)
```

**Now, WITHIN each pipeline stage, we apply TP=8:**

```
Pipeline Stage 1 (Layers 1-20) - Split across 8 GPUs using TP:

Layer 1:
├─ GPU 0: 1/8 of layer weights (columns 0-511 of weight matrix)
├─ GPU 1: 1/8 of layer weights (columns 512-1023)
├─ GPU 2: 1/8 of layer weights (columns 1024-1535)
├─ GPU 3: 1/8 of layer weights (columns 1536-2047)
├─ GPU 4: 1/8 of layer weights (columns 2048-2559)
├─ GPU 5: 1/8 of layer weights (columns 2560-3071)
├─ GPU 6: 1/8 of layer weights (columns 3072-3583)
└─ GPU 7: 1/8 of layer weights (columns 3584-4095)

Layer 2: Same 8-way split across same GPUs
Layer 3: Same 8-way split across same GPUs
...
Layer 20: Same 8-way split across same GPUs
```

**Key Point:** Each pipeline stage uses a **group of GPUs** (8 in this case), and TP splits the layers **within that stage** across those GPUs.

---

#### GPU Organization Summary

**Complete 64-GPU Layout (TP=8, PP=4, DP=2):**

```
┌──────────────┬─────────────┬──────────────┬────────────────┐
│ DP Replica   │ PP Stage    │ GPUs         │ Layers Held    │
├──────────────┼─────────────┼──────────────┼────────────────┤
│ Replica 1    │ Stage 1     │ GPUs 0-7     │ Layers 1-20    │
│ (GPUs 0-31)  │ Stage 2     │ GPUs 8-15    │ Layers 21-40   │
│              │ Stage 3     │ GPUs 16-23   │ Layers 41-60   │
│              │ Stage 4     │ GPUs 24-31   │ Layers 61-80   │
├──────────────┼─────────────┼──────────────┼────────────────┤
│ Replica 2    │ Stage 1     │ GPUs 32-39   │ Layers 1-20    │
│ (GPUs 32-63) │ Stage 2     │ GPUs 40-47   │ Layers 21-40   │
│              │ Stage 3     │ GPUs 48-55   │ Layers 41-60   │
│              │ Stage 4     │ GPUs 56-63   │ Layers 61-80   │
└──────────────┴─────────────┴──────────────┴────────────────┘

Within EACH group of 8 GPUs (e.g., GPUs 0-7):
┌──────┬────────────────────────────────────────────┐
│ GPU  │ Weight Columns Held (for each layer)      │
├──────┼────────────────────────────────────────────┤
│ GPU 0│ Columns 0-1023     (1/8 of layer)         │
│ GPU 1│ Columns 1024-2047  (1/8 of layer)         │
│ GPU 2│ Columns 2048-3071  (1/8 of layer)         │
│ GPU 3│ Columns 3072-4095  (1/8 of layer)         │
│ GPU 4│ Columns 4096-5119  (1/8 of layer)         │
│ GPU 5│ Columns 5120-6143  (1/8 of layer)         │
│ GPU 6│ Columns 6144-7167  (1/8 of layer)         │
│ GPU 7│ Columns 7168-8191  (1/8 of layer)         │
└──────┴────────────────────────────────────────────┘
```

**The 3D Split Visualized:**

```
           TP Dimension (Horizontal - splits each layer)
                    ↔ 8 GPUs ↔

PP Dimension     ┌───┬───┬───┬───┬───┬───┬───┬───┐
(Vertical -      │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ Stage 1 (L1-20)
splits layers)   ├───┼───┼───┼───┼───┼───┼───┼───┤
    ↕            │ 8 │ 9 │10 │11 │12 │13 │14 │15 │ Stage 2 (L21-40)
                 ├───┼───┼───┼───┼───┼───┼───┼───┤
                 │16 │17 │18 │19 │20 │21 │22 │23 │ Stage 3 (L41-60)
                 ├───┼───┼───┼───┼───┼───┼───┼───┤
                 │24 │25 │26 │27 │28 │29 │30 │31 │ Stage 4 (L61-80)
                 └───┴───┴───┴───┴───┴───┴───┴───┘
                       ⬆ Replica 1 (32 GPUs)

                 ┌───┬───┬───┬───┬───┬───┬───┬───┐
                 │32 │33 │34 │35 │36 │37 │38 │39 │ Stage 1 (L1-20)
                 ├───┼───┼───┼───┼───┼───┼───┼───┤
                 │40 │41 │42 │43 │44 │45 │46 │47 │ Stage 2 (L21-40)
  DP Dimension   ├───┼───┼───┼───┼───┼───┼───┼───┤
  (Depth -       │48 │49 │50 │51 │52 │53 │54 │55 │ Stage 3 (L41-60)
  replicates)    ├───┼───┼───┼───┼───┼───┼───┼───┤
      ↕          │56 │57 │58 │59 │60 │61 │62 │63 │ Stage 4 (L61-80)
                 └───┴───┴───┴───┴───┴───┴───┴───┘
                       ⬆ Replica 2 (32 GPUs)
```

---

#### Concrete Example - Processing One Token Through Pipeline Stage 1

Let's see exactly what happens when data flows through one pipeline stage with TP=8:

```
Input: Token embedding vector [8192 dimensions]

Pipeline Stage 1 - Layer 1 Self-Attention:
┌─────────────────────────────────────────────────────────────┐
│ Weight Matrix W_q [8192 × 8192] split across 8 GPUs:       │
├─────────────────────────────────────────────────────────────┤
│ GPU 0: W_q[:, 0:1024]     ← Columns 0-1023                 │
│ GPU 1: W_q[:, 1024:2048]  ← Columns 1024-2047              │
│ GPU 2: W_q[:, 2048:3072]  ← Columns 2048-3071              │
│ GPU 3: W_q[:, 3072:4096]  ← Columns 3072-4095              │
│ GPU 4: W_q[:, 4096:5120]  ← Columns 4096-5119              │
│ GPU 5: W_q[:, 5120:6144]  ← Columns 5120-6143              │
│ GPU 6: W_q[:, 6144:7168]  ← Columns 6144-7167              │
│ GPU 7: W_q[:, 7168:8192]  ← Columns 7168-8191              │
└─────────────────────────────────────────────────────────────┘

Forward Pass:
1. Input [8192] broadcast to all 8 GPUs
2. Each GPU computes: partial_output = input @ W_q_slice
   GPU 0: [8192] @ [8192 × 1024] = [1024] (partial query)
   GPU 1: [8192] @ [8192 × 1024] = [1024] (partial query)
   ...
   GPU 7: [8192] @ [8192 × 1024] = [1024] (partial query)

3. All-Gather: Concatenate all partial outputs
   Full Query = [GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | GPU6 | GPU7]
              = [1024 + 1024 + 1024 + 1024 + 1024 + 1024 + 1024 + 1024]
              = [8192] ✅

4. Repeat for K, V, and all layers 1-20

5. Output of Layer 20 → Sent to Pipeline Stage 2 (GPUs 8-15)
```

**Why This Works:**
- **Matrix multiplication is column-wise parallelizable**: Each GPU computes a portion of the output
- **All-Gather combines results**: Final output is same as if computed on one GPU
- **Same process for all 20 layers** in this pipeline stage
- **Output goes to next pipeline stage** as a complete activation

---

### Step-by-Step: How 3D Parallelism Works

**Real Example - Training Llama 3 70B on 64 GPUs**:
```
Configuration:
- Tensor Parallel (TP) = 8    (split each layer across 8 GPUs)
- Pipeline Parallel (PP) = 4   (split 80 layers into 4 stages)
- Data Parallel (DP) = 2       (2 complete model replicas)
- Total: 8 × 4 × 2 = 64 GPUs

Visual Layout:
┌─────────────────────────────────────────────────────────────┐
│ Data Parallel Replica 1 (32 GPUs)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pipeline Stage 1 (8 GPUs - Layers 1-20):                  │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │GPU 0│GPU 1│GPU 2│GPU 3│GPU 4│GPU 5│GPU 6│GPU 7│ ← TP=8 │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
│         Each GPU has 1/8 of each layer's weights            │
│                           ↓                                  │
│  Pipeline Stage 2 (8 GPUs - Layers 21-40):                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │GPU 8│GPU 9│...  │...  │...  │...  │...  │GPU15│ ← TP=8 │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
│                           ↓                                  │
│  Pipeline Stage 3 (8 GPUs - Layers 41-60):                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │GPU16│GPU17│...  │...  │...  │...  │...  │GPU23│ ← TP=8 │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
│                           ↓                                  │
│  Pipeline Stage 4 (8 GPUs - Layers 61-80):                 │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐        │
│  │GPU24│GPU25│...  │...  │...  │...  │...  │GPU31│ ← TP=8 │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘        │
└─────────────────────────────────────────────────────────────┘
                    Processes Batch A

┌─────────────────────────────────────────────────────────────┐
│ Data Parallel Replica 2 (32 GPUs)                           │
├─────────────────────────────────────────────────────────────┤
│  Same structure as Replica 1 (GPUs 32-63)                  │
│  Pipeline: 4 stages × TP: 8 GPUs each                       │
└─────────────────────────────────────────────────────────────┘
                    Processes Batch B

Training Flow:
1. Batch A goes through Replica 1 (GPUs 0-31)
2. Batch B goes through Replica 2 (GPUs 32-63)
3. Gradients from both replicas are averaged (DP synchronization)
4. All 64 GPUs update with the same averaged gradients

Benefits:
- Each GPU only holds: 70B params / (2×4×8) = 1.1B params (~4.4GB)
- Effective batch size: 2× larger (from DP=2)
- Training speed: ~40,000 tokens/sec
- Can train on 2 trillion tokens in ~2 weeks
```

**Memory Breakdown per GPU**:
```
Model Parameters: 70B / 64 = 1.1B params × 2 bytes (FP16) = 2.2 GB
Optimizer States:  1.1B params × 8 bytes = 8.8 GB
Gradients:         1.1B params × 2 bytes = 2.2 GB
Activations:       ~10 GB (depends on batch size)
───────────────────────────────────────────────
Total per GPU:     ~23 GB (fits in 80GB A100!)
```

---

### End-to-End Data Flow Example

**How a batch flows through the entire 3D parallel system:**

```
Configuration: TP=8, PP=4, DP=2 (64 GPUs total)

Input Batch: "The quick brown fox jumps over the lazy dog"
Tokenized: [128 tokens, batch_size=4]

┌─────────────────────────────────────────────────────────────┐
│ Data Parallel Replica 1 (processes batch samples 1-2)      │
└─────────────────────────────────────────────────────────────┘

Step 1: Pipeline Stage 1 (GPUs 0-7) - Layers 1-20
┌────────────────────────────────────────────────────────┐
│ Input: [batch=2, seq=128, hidden=8192]                │
│                                                        │
│ GPU 0: Layer 1-20 weights columns 0-1023              │
│ GPU 1: Layer 1-20 weights columns 1024-2047           │
│ GPU 2: Layer 1-20 weights columns 2048-3071           │
│ ...                                                    │
│ GPU 7: Layer 1-20 weights columns 7168-8191           │
│                                                        │
│ Process:                                               │
│ 1. Broadcast input to all 8 GPUs                      │
│ 2. Each GPU computes partial forward pass             │
│ 3. All-Reduce to combine results                      │
│ 4. Output: [batch=2, seq=128, hidden=8192]            │
└────────────────────────────────────────────────────────┘
                    ↓ Send to Stage 2
┌────────────────────────────────────────────────────────┐
│ Step 2: Pipeline Stage 2 (GPUs 8-15) - Layers 21-40   │
│ Same TP process for layers 21-40                       │
└────────────────────────────────────────────────────────┘
                    ↓ Send to Stage 3
┌────────────────────────────────────────────────────────┐
│ Step 3: Pipeline Stage 3 (GPUs 16-23) - Layers 41-60  │
│ Same TP process for layers 41-60                       │
└────────────────────────────────────────────────────────┘
                    ↓ Send to Stage 4
┌────────────────────────────────────────────────────────┐
│ Step 4: Pipeline Stage 4 (GPUs 24-31) - Layers 61-80  │
│ Same TP process for layers 61-80                       │
│ Output: Logits [batch=2, seq=128, vocab=128K]         │
└────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Data Parallel Replica 2 (GPUs 32-63)                       │
│ Processes batch samples 3-4 through same pipeline          │
└─────────────────────────────────────────────────────────────┘

Backward Pass:
1. Stage 4 computes gradients → sends to Stage 3
2. Stage 3 computes gradients → sends to Stage 2
3. Stage 2 computes gradients → sends to Stage 1
4. Stage 1 computes gradients

Gradient Synchronization (DP):
1. Replica 1 (GPUs 0-31) has gradients for samples 1-2
2. Replica 2 (GPUs 32-63) has gradients for samples 3-4
3. All-Reduce: Average gradients across replicas
4. All 64 GPUs update with same averaged gradient
```

**Communication Patterns:**

1. **Within Pipeline Stage (TP communication)**:
   - All-Reduce across 8 GPUs (fast - NVLink)
   - Happens at every layer
   - High frequency, low latency

2. **Between Pipeline Stages (PP communication)**:
   - Point-to-point send/receive
   - Stage 1 → Stage 2 → Stage 3 → Stage 4
   - Moderate frequency, moderate latency
   - Can use InfiniBand across nodes

3. **Between Data Replicas (DP communication)**:
   - All-Reduce across 2 replicas (32 GPUs each)
   - Once per training step
   - Low frequency, high volume
   - Usually across InfiniBand

**When to use**: Very large models (70B+), have many GPUs (64+), production training at scale.

---

### Quick Decision Guide

```
┌─────────────────────────────────────────┐
│ Does model fit on 1 GPU?                │
├─────────────────────────────────────────┤
│ YES → Use Data Parallelism (DP)        │
│       • Simplest approach                │
│       • 1-8 GPUs                        │
│       • Example: Llama 7B              │
└─────────────────────────────────────────┘
         │
         │ NO
         ↓
┌─────────────────────────────────────────┐
│ Do individual layers fit on 1 GPU?     │
├─────────────────────────────────────────┤
│ YES → Use Pipeline Parallelism (PP)    │
│       • Split layers across GPUs        │
│       • Good for multi-node             │
│       • Example: GPT-3 175B            │
└─────────────────────────────────────────┘
         │
         │ NO
         ↓
┌─────────────────────────────────────────┐
│ Use Tensor Parallelism (TP)            │
│ • Split individual layers               │
│ • Requires fast interconnect (NVLink)  │
│ • Usually 2-8 GPUs per layer           │
│ • Example: Llama 70B with TP=8        │
└─────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────┐
│ For Production / Very Large Scale:     │
│ Use 3D Parallelism (DP + TP + PP)     │
│ • Combine all three                     │
│ • 64-512+ GPUs                         │
│ • Example: Training Llama 3 405B      │
└─────────────────────────────────────────┘

Special Cases:
• Long sequences (100K+)?    → Add Sequence Parallelism (SP)
• Ultra-long (1M+ tokens)?   → Add Context Parallelism (CP)
• Mixture of Experts (MoE)?  → Add Expert Parallelism
```

#### Megatron Core in NeMo

NeMo Framework uses Megatron Core as its training backend for large models. When you configure parallelism in NeMo, you're using Megatron Core under the hood.

**NeMo Configuration Example**:
```python
from nemo.collections import llm
from nemo_run import run

# Configure Megatron parallelism through NeMo
def llama3_70b_distributed():
    return llm.Llama3Model.configure(
        tensor_model_parallel_size=8,      # TP: 8 GPUs per layer
        pipeline_model_parallel_size=4,    # PP: 4 pipeline stages
        data_parallel_size=2,              # DP: 2 replicas
        sequence_parallel=True,            # Enable SP
        # Total GPUs = 8 × 4 × 2 = 64 GPUs
    )
```

**YAML Configuration**:
```yaml
model:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 4
  micro_batch_size: 1
  global_batch_size: 128

  # Megatron optimizations
  sequence_parallel: true
  activations_checkpoint_granularity: selective
  activations_checkpoint_method: uniform
```

#### When to Use Different Parallelism Strategies

| Model Size | Sequence Length | Recommended Strategy | GPUs |
|------------|----------------|---------------------|------|
| < 7B | < 8K | Data Parallel only | 1-8 |
| 7-13B | < 8K | DP + optional TP=2 | 2-16 |
| 13-70B | < 32K | DP + TP=4-8 | 8-64 |
| 70-175B | < 32K | DP + TP=8 + PP=4 | 64-256 |
| 175B+ | < 32K | 3D Parallelism | 256+ |
| Any | 100K+ | Add SP to above | Same |
| Any | 1M+ | Add CP to above | Same |

#### Key Features

- **6D Parallelism**: DP, TP, PP, SP, CP, Expert Parallelism (for MoE)
- **Mixed Precision Training**: FP16, BF16, FP8
- **Flash Attention**: Memory-efficient attention computation
- **Gradient Accumulation**: Simulate larger batches
- **Activation Checkpointing**: Trade compute for memory
- **Distributed Optimizer**: ZeRO-style optimizer state sharding

#### Performance Benefits

```
Example: Training Llama 3 70B

Without Megatron (impossible):
- 70B params × 4 bytes = 280GB
- Single 80GB GPU: ❌ Cannot fit

With Megatron (TP=8):
- 280GB / 8 GPUs = 35GB per GPU
- Each 80GB GPU: ✅ Fits with room for activations
- Training speed: ~5000 tokens/sec/GPU

With 3D Parallelism (TP=8, PP=4, DP=2):
- 64 GPUs total
- Effective batch size: 128
- Training speed: 40,000 tokens/sec
- Time to train on 2T tokens: ~2 weeks
```

#### Code Example: Using Megatron Core in NeMo

```python
from nemo.collections import llm
from nemo_run import run, Config

# Define model with Megatron parallelism
def configure_megatron_training():
    # Model configuration
    model = llm.Llama3Model.configure(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=4,
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        sequence_parallel=True,
        use_flash_attention=True,
    )

    # Optimizer configuration
    optimizer = Config(
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Training configuration
    trainer = Config(
        max_steps=100000,
        val_check_interval=1000,
        precision="bf16-mixed",  # BF16 mixed precision
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )

    return model, optimizer, trainer

# Run distributed training
recipe = run.Partial(
    llm.pretrain,
    model=configure_megatron_training()[0],
    optimizer=configure_megatron_training()[1],
    trainer=configure_megatron_training()[2],
)

# Execute on Slurm cluster
run.run(recipe, executor=run.SlurmExecutor(
    nodes=16,  # 16 nodes × 8 GPUs = 128 GPUs
    gpus_per_node=8,
    time_limit="48:00:00",
))
```

#### Monitoring Megatron Training

```python
# Key metrics to monitor
import torch.distributed as dist

if dist.is_initialized():
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Check parallelism setup
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    dp_size = parallel_state.get_data_parallel_world_size()

    print(f"Rank {rank}/{world_size}")
    print(f"TP: {tp_size}, PP: {pp_size}, DP: {dp_size}")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

#### Best Practices

1. **Start with TP only**: Use tensor parallelism first (8 GPUs per node)
2. **Add PP for scaling**: Add pipeline parallelism when going multi-node
3. **Use DP for throughput**: Add data parallelism to increase batch processing
4. **Enable SP for long sequences**: Reduces memory for 100K+ token sequences
5. **Monitor GPU utilization**: Aim for 80%+ GPU utilization
6. **Tune micro-batch size**: Find largest batch that fits in memory
7. **Use Flash Attention**: Always enable for memory savings

#### Resources

- **Megatron-LM GitHub**: https://github.com/NVIDIA/Megatron-LM
- **Megatron Core Docs**: https://docs.nvidia.com/megatron-core/
- **NeMo Parallelism Guide**: Part of NeMo Framework documentation

---

### 2. **Megatron Bridge: Framework Conversion Tool**

#### What is Megatron Bridge?

**Megatron Bridge** is a **conversion tool** that translates model checkpoints between different training/inference frameworks. Think of it as a universal adapter for AI models.

**Key Purpose**: Enable seamless movement of models between Megatron, NeMo, HuggingFace, and other frameworks.

#### The Problem It Solves

Different frameworks save models in incompatible formats:

```
Training Framework          Checkpoint Format          What's Inside & Purpose                         Typical Size
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Megatron-LM/Core     →     Raw distributed          • mp_rank_XX_YYY.pt files                       7B:  ~28GB (TP=4)
                            format                     - Sharded model weights across TP/PP ranks    70B: ~280GB (TP=8)
                            (Native training           - Each file = slice of layers for one rank    (Distributed across
                            checkpoint, raw            - Purpose: Parallel loading during training   TP×PP files)
                            Megatron-Core)               resumption, no network transfer needed
                                                     • model_optim_rng.pt (per rank)
                                                       - Adam optimizer states (momentum, variance)
                                                       - RNG states for reproducibility
                                                       - Purpose: Resume training exactly from step
                                                     • latest_checkpointed_iteration.txt
                                                       - Training step number
                                                       - Purpose: Track progress, auto-resume
                                                     • No config file (architecture in training script)

NVIDIA NeMo          →     .nemo format             • model_weights.ckpt                            7B:  ~14GB (consolidated)
                            (NeMo's packaging          - PyTorch state_dict (consolidated OR          7B:  ~28GB (distributed)
                            of Megatron-Core)            distributed mp_rank files)                  70B: ~140GB (consolidated)
                            **SUPPORTS                 - All layer weights (attention, FFN, embed)   70B: ~280GB (distributed)
                            PARALLELISM!**             - Purpose: Model inference or training        (Packaged in .tar.gz)
                                                     • model_config.yaml
                                                       - Architecture: layers, hidden_size, heads
                                                       - Parallelism: tensor_model_parallel_size,
                                                         pipeline_model_parallel_size
                                                       - Purpose: Initialize model correctly
                                                     • tokenizer.model
                                                       - SentencePiece/HF tokenizer weights
                                                       - Vocabulary mappings (token↔ID)
                                                       - Purpose: Text preprocessing
                                                     • hparams.yaml
                                                       - Training hyperparams (LR, batch, precision)
                                                       - Optimizer config
                                                       - Purpose: Reproduce training setup
                                                     • (Optional) mp_rank_XX files for distributed ckpt
                                                     • All packaged in single .tar.gz archive

                            **NOTE**: NeMo uses Megatron-Core internally. During training,
                            it saves distributed mp_rank files inside .nemo archive. For
                            deployment, it can save consolidated weights instead.

HuggingFace          →     safetensors / bin        • model.safetensors OR pytorch_model.bin        7B:  ~13-15GB
                            format                     - Single consolidated file with ALL weights   70B: ~130-140GB
                            (Ecosystem                 - Layers (model.layers.X.self_attn.q_proj)    (safetensors ~10%
                            standard)                  - Embeddings, layer norms, output head        smaller + safer
                            **NO parallelism           - FP16/BF16/FP32 tensors                      than .bin)
                            support**                  - Purpose: Load full model for inference
                                                     • config.json
                                                       - Model architecture in JSON
                                                       - {hidden_size, num_layers, num_heads, ...}
                                                       - Purpose: Build model architecture
                                                     • tokenizer.json
                                                       - Tokenizer vocab + algorithm (BPE, WordPiece)
                                                       - Normalization rules
                                                       - Purpose: Text→tokens conversion
                                                     • tokenizer_config.json
                                                       - Tokenizer class name & init params
                                                       - Purpose: Load correct tokenizer
                                                     • special_tokens_map.json
                                                       - BOS, EOS, PAD, UNK token IDs
                                                       - Purpose: Handle special sequences
                                                     • generation_config.json (optional)
                                                       - Default inference params (temp, top_p, max_len)
                                                       - Purpose: Generation defaults

                            **NOTE**: HF format is ALWAYS consolidated. Use vLLM for
                            distributed inference, but checkpoint itself is not sharded.

PyTorch              →     .pt / .pth format        • Custom state_dict                             7B:  ~14-28GB
                            (Vanilla PyTorch)          - Arbitrary nested dict structure             70B: ~140-280GB
                                                       - Typically model.state_dict() output         (Larger if includes
                                                       - Purpose: Flexible custom checkpointing      optimizer: 2-3x size)
                                                     • Common keys:
                                                       - 'model': model weights
                                                       - 'optimizer': Adam/SGD states
                                                       - 'scheduler': LR scheduler state
                                                       - 'epoch': training epoch counter
                                                       - Purpose: Full training state snapshot
                                                     • No standardized structure or config
                                                     • Framework-agnostic, maximum flexibility
```

**Key Insights**:

1. **NeMo DOES support parallelism** - It's built on Megatron-Core and can save/load distributed checkpoints
   - During training: Saves distributed `mp_rank_XX` files (same as raw Megatron)
   - For deployment: Can consolidate weights into single file
   - The `.nemo` archive is just packaging (tar.gz) - contents can be distributed or consolidated

2. **Why Megatron-LM/Core format exists**:
   - It's the **native format** used during training (lowest level)
   - NeMo adds a **packaging layer** on top (configs + tokenizer + tar.gz)
   - Pure Megatron-LM training (without NeMo framework) uses raw format directly
   - During NeMo training, internally it's still using Megatron-Core's distributed checkpoints

3. **Format Comparison**:
   ```
   During Training:
   ─────────────────────────────────────────────────────────────
   Raw Megatron      → mp_rank_00_000.pt, mp_rank_01_000.pt, ...
   NeMo Training     → Same mp_rank files, packaged in .nemo.tar.gz

   For Deployment:
   ─────────────────────────────────────────────────────────────
   Megatron          → Keep distributed (requires Megatron to load)
   NeMo              → Consolidated model_weights.ckpt in .nemo
   HuggingFace       → Consolidated model.safetensors
   ```

**Problem**: You train with Megatron/NeMo (distributed, fastest) but want to deploy with HuggingFace (ecosystem, consolidated) → **Megatron Bridge solves this**.

#### How Megatron Bridge Works

```
┌──────────────────────┐
│  Megatron Checkpoint │  (Distributed, TP=8, PP=4)
│  32 checkpoint files │
└──────────┬───────────┘
           │
           │ Megatron Bridge
           │ • Merge distributed weights
           │ • Rename layer keys
           │ • Adjust architecture config
           ▼
┌──────────────────────┐
│ HuggingFace Format   │  (Single consolidated checkpoint)
│ model.safetensors    │
│ config.json          │
└──────────────────────┘
```

#### Supported Conversions

| From | To | Use Case |
|------|-----|----------|
| Megatron → HuggingFace | Deploy with Transformers, vLLM, TGI | Most common |
| HuggingFace → Megatron | Continue training with efficiency | Pre-trained → fine-tune |
| NeMo → HuggingFace | NeMo training → HF deployment | NeMo users |
| HuggingFace → NeMo | HF models → NeMo framework | Import community models |
| Megatron → NeMo | Megatron-LM → NeMo workflow | Framework migration |

#### Conversion Example: Megatron → HuggingFace

**Scenario**: You trained Llama 3 70B with NeMo/Megatron (TP=8, PP=4 on 32 GPUs), now want to deploy with vLLM.

```python
# Method 1: Using NeMo's built-in conversion
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel
from transformers import AutoModelForCausalLM

# Load NeMo checkpoint
nemo_model = MegatronGPTModel.restore_from(
    restore_path="llama3_70b_nemo.nemo",
    trainer=None,
)

# Convert to HuggingFace
nemo_model.save_to_hf(
    output_path="llama3_70b_hf",
    precision="bf16",
)

# Verify HuggingFace model
hf_model = AutoModelForCausalLM.from_pretrained("llama3_70b_hf")
print(f"Loaded HF model: {hf_model.config}")
```

**Method 2: Using Megatron Bridge CLI**
```bash
# Convert Megatron checkpoint to HuggingFace
python nemo/scripts/nlp_language_modeling/convert_megatron_to_hf.py \
    --input-name-or-path /path/to/megatron_checkpoint \
    --output-path /path/to/hf_output \
    --model-type llama \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --precision bf16

# Result: HuggingFace-compatible checkpoint
# ├── config.json
# ├── model.safetensors (or pytorch_model.bin)
# └── tokenizer files
```

#### Conversion Example: HuggingFace → NeMo

**Scenario**: Start with HuggingFace Llama 3 8B, fine-tune efficiently with NeMo.

```python
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel

# Convert HuggingFace to NeMo
MegatronGPTModel.convert_hf_to_nemo(
    hf_model_path="meta-llama/Llama-3-8b-hf",
    output_path="llama3_8b.nemo",
    precision="bf16",
)

# Now use in NeMo training
model = MegatronGPTModel.restore_from("llama3_8b.nemo")
```

**CLI Method**:
```bash
python nemo/scripts/nlp_language_modeling/convert_hf_to_nemo.py \
    --input-name-or-path meta-llama/Llama-3-8b-hf \
    --output-path llama3_8b.nemo \
    --precision bf16
```

#### Key Conversion Operations

**1. Weight Merging** (Distributed → Single)
```python
# Megatron: Weights split across 8 GPUs
GPU 0: layer.qkv.weight[0:1024, :]
GPU 1: layer.qkv.weight[1024:2048, :]
GPU 2: layer.qkv.weight[2048:3072, :]
...
GPU 7: layer.qkv.weight[7168:8192, :]

# After Bridge: Single consolidated weight
HF: layer.qkv.weight[0:8192, :]  # Merged
```

**2. Key Renaming**
```python
# Megatron naming
"transformer.layers.0.attention.query_key_value.weight"

# HuggingFace naming
"model.layers.0.self_attn.qkv_proj.weight"

# Bridge handles automatic mapping
```

**3. Config Translation**
```python
# Megatron config
{
    "num_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "tensor_model_parallel_size": 8,
}

# HuggingFace config
{
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    # TP info removed (not needed for inference)
}
```

#### Advanced Usage: Custom Architectures

```python
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel

# Define custom conversion mapping
custom_mapping = {
    "megatron_key_pattern": "hf_key_pattern",
    "transformer.layers.*.attention.dense": "model.layers.*.self_attn.o_proj",
    "transformer.layers.*.mlp.dense_h_to_4h": "model.layers.*.mlp.gate_up_proj",
}

# Convert with custom mapping
MegatronGPTModel.convert_with_mapping(
    input_path="custom_megatron.nemo",
    output_path="custom_hf",
    key_mapping=custom_mapping,
)
```

#### Real-World Workflow

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Training Phase                                  │
├─────────────────────────────────────────────────────────┤
│ • Start with HuggingFace Llama 3 70B                    │
│ • Convert to NeMo format (Megatron Bridge)              │
│ • Train with NeMo on 64 GPUs (TP=8, PP=4, DP=2)        │
│ • Achieves 5000 tokens/sec training speed               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: Conversion Phase                                │
├─────────────────────────────────────────────────────────┤
│ • Use Megatron Bridge to convert back to HuggingFace    │
│ • Merge distributed checkpoints into single file        │
│ • Validate conversion (compare outputs)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: Deployment Phase                                │
├─────────────────────────────────────────────────────────┤
│ • Deploy with vLLM (high throughput inference)          │
│ • Or TensorRT-LLM (optimized for NVIDIA GPUs)          │
│ • Or Triton Inference Server (production serving)       │
│ • Access full HuggingFace ecosystem                     │
└─────────────────────────────────────────────────────────┘
```

#### Validation After Conversion

Always validate that conversion preserved model behavior:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load converted model
model = AutoModelForCausalLM.from_pretrained("llama3_70b_hf")
tokenizer = AutoTokenizer.from_pretrained("llama3_70b_hf")

# Test inference
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0])

print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")

# Compare with original NeMo model output
# Should be identical (or very similar with temperature=0)
```

#### Common Issues & Solutions

**Issue 1: Checkpoint Size Mismatch**
```bash
# Problem: Megatron checkpoint has different shape
# Solution: Check TP/PP sizes match
python convert.py \
    --tensor-parallel-size 8 \  # Must match training config
    --pipeline-parallel-size 4
```

**Issue 2: Missing Configuration**
```python
# Problem: Some config values not auto-detected
# Solution: Manually specify
MegatronGPTModel.convert_hf_to_nemo(
    hf_model_path="model_path",
    output_path="output.nemo",
    vocab_size=128256,  # Manually specify
    rotary_percentage=0.5,  # Custom architecture
)
```

**Issue 3: Different Precision**
```bash
# Problem: Training was BF16, deployment needs FP16
# Solution: Convert precision during bridge
python convert.py \
    --input-precision bf16 \
    --output-precision fp16
```

#### Performance Considerations

| Aspect | Details |
|--------|---------|
| **Conversion Time** | ~5-30 minutes for 70B model (depends on I/O speed) |
| **Disk Space** | Need 2-3x model size temporarily (original + converted) |
| **Memory** | Minimal (processes chunk-by-chunk) |
| **Accuracy** | Bit-exact if same precision, negligible diff if converting precision |

#### Integration with NeMo Workflow

NeMo 2.0+ includes Megatron Bridge functionality built-in:

```python
from nemo.collections import llm
from nemo_run import run

# Configure training with auto-conversion
recipe = run.Partial(
    llm.finetune,
    model="meta-llama/Llama-3-8b-hf",  # Auto-converts from HF
    # ... training config ...
)

# After training, export back to HF
from nemo.export import export_to_hf
export_to_hf(
    nemo_checkpoint="trained_model.nemo",
    output_path="trained_model_hf",
)
```

#### Quick Reference Commands

```bash
# 1. Megatron → HuggingFace
python nemo/scripts/nlp_language_modeling/convert_megatron_to_hf.py \
    --input-name-or-path /megatron/ckpt \
    --output-path /hf/output

# 2. HuggingFace → NeMo
python nemo/scripts/nlp_language_modeling/convert_hf_to_nemo.py \
    --input-name-or-path meta-llama/Llama-3-8b \
    --output-path llama3.nemo

# 3. NeMo → TensorRT-LLM
python nemo/scripts/deploy/export_to_tensorrt_llm.py \
    --nemo-checkpoint llama3.nemo \
    --output-dir trt_llm_engine

# 4. Verify conversion
python verify_conversion.py \
    --original-checkpoint megatron_ckpt \
    --converted-checkpoint hf_ckpt \
    --test-prompts prompts.txt
```

#### Resources

- **NeMo Conversion Scripts**: `NeMo/scripts/nlp_language_modeling/`
- **Documentation**: NeMo Framework User Guide → Model Conversion
- **Examples**: `NeMo/examples/nlp/language_modeling/checkpoint_conversion/`

---

### 3. **Distributed Training** [web:417]
- Tensor Parallelism (TP)
- Data Parallelism (DP)
- Pipeline Parallelism (PP)
- 3D Parallelism (TP + DP + PP)

**Learning Resource**: "LLMs at Scale on Azure" [web:417]

### 2. **NeMo-Run Orchestration & Ray Integration** [web:391]

#### What is Ray?

**Ray** is a distributed computing framework that NeMo uses to orchestrate and scale training/experiments across multiple machines.

**Purpose in NeMo:**
- **Multi-node training**: Coordinate training across clusters (10s to 1000s of GPUs)
- **Hyperparameter tuning**: Run parallel experiments with different configs
- **Resource management**: Allocate CPUs, GPUs, memory across nodes
- **Fault tolerance**: Handle node failures during long training runs
- **Experiment tracking**: Monitor and compare experiments

**Think of Ray as:** The "conductor" that orchestrates a symphony of GPUs across multiple machines.

---

#### How Ray Works in NeMo

```
Without Ray (Single Node):
┌─────────────────┐
│  Your Machine   │
│  ├─ GPU 0       │  ← Limited to local GPUs
│  ├─ GPU 1       │
│  └─ GPU 2       │
└─────────────────┘

With Ray (Multi-Node Cluster):
┌──────────────────────────────────────────────────────┐
│  Ray Cluster                                         │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Node 1   │  │ Node 2   │  │ Node 3   │          │
│  │ 8x GPUs  │  │ 8x GPUs  │  │ 8x GPUs  │  ...     │
│  └──────────┘  └──────────┘  └──────────┘          │
│         ↑           ↑           ↑                   │
│         └───────────┴───────────┘                   │
│              Ray Orchestration                      │
└──────────────────────────────────────────────────────┘
```

---

#### NeMo-Run: Built on Ray

**NeMo-Run** is NeMo's interface to Ray that simplifies distributed training.

**Key Features:**
- One-line switch: Local → Multi-node
- Experiment versioning
- Auto-resume on failures
- Slurm cluster integration
- Hyperparameter sweeps

---

#### Code Example 1: Local to Distributed Training

**Local Training (Single GPU):**
```python
from nemo.collections import llm
from nemo import lightning as nl

# Define training recipe
recipe = llm.llama3_8b.finetune_recipe(
    dir="./checkpoints",
    num_nodes=1,  # Single machine
    num_gpus_per_node=1
)

# Train locally
trainer = nl.Trainer(
    devices=1,
    max_epochs=3
)
trainer.fit(model, data)
```

**Distributed Training with Ray (64 GPUs across 8 nodes):**
```python
import nemo_run as run
from nemo.collections import llm

# Define the SAME recipe
recipe = llm.llama3_8b.finetune_recipe(
    dir="./checkpoints",
    num_nodes=8,        # 8 machines
    num_gpus_per_node=8  # 64 total GPUs
)

# Configure Ray cluster
executor = run.SlurmExecutor(
    account="your_account",
    partition="gpu",
    nodes=8,
    ntasks_per_node=8,
    time="24:00:00"
)

# Submit to cluster - ONE LINE CHANGE!
with run.Experiment("llama3_finetune") as exp:
    exp.add(
        recipe,
        executor=executor,
        name="8node_64gpu_run"
    )
    exp.run()
```

**That's it!** Ray handles:
- Job submission to Slurm
- GPU allocation across 8 nodes
- Inter-node communication setup
- Checkpointing & recovery

---

#### Code Example 2: Hyperparameter Tuning with Ray

```python
import nemo_run as run
from nemo.collections import llm

# Define hyperparameter grid
learning_rates = [1e-5, 5e-5, 1e-4]
batch_sizes = [32, 64, 128]
lora_ranks = [8, 16, 32]

# Ray runs ALL combinations in parallel!
experiments = []

for lr in learning_rates:
    for bs in batch_sizes:
        for rank in lora_ranks:
            # Create experiment config
            recipe = llm.llama3_8b.finetune_recipe(
                dir=f"./experiments/lr{lr}_bs{bs}_rank{rank}",
                num_nodes=1,
                num_gpus_per_node=4
            )

            # Customize hyperparameters
            recipe.trainer.max_epochs = 3
            recipe.optim.lr = lr
            recipe.data.global_batch_size = bs
            recipe.model.peft.lora_rank = rank

            experiments.append(recipe)

# Submit all experiments - Ray runs them in parallel!
with run.Experiment("hyperparameter_sweep") as exp:
    for i, recipe in enumerate(experiments):
        exp.add(
            recipe,
            executor=run.LocalExecutor(gpus=4),
            name=f"exp_{i}"
        )

    # Ray automatically schedules experiments based on GPU availability
    results = exp.run()

# Compare results
best_config = max(results, key=lambda x: x.metrics['val_accuracy'])
print(f"Best config: LR={best_config.lr}, BS={best_config.bs}")
```

**Benefits:**
- **Parallel execution**: All 27 experiments (3×3×3) run simultaneously
- **Resource optimization**: Ray schedules jobs based on available GPUs
- **Auto-tracking**: Results logged for each experiment

---

#### Code Example 3: Fault-Tolerant Training

```python
import nemo_run as run
from nemo.collections import llm

# Long training job (5 days)
recipe = llm.llama3_70b.pretrain_recipe(
    dir="./checkpoints",
    num_nodes=32,
    num_gpus_per_node=8,
    max_steps=100000
)

# Configure checkpointing
recipe.trainer.val_check_interval = 1000
recipe.trainer.save_top_k = 3  # Keep best 3 checkpoints

# Ray handles failures automatically
executor = run.SlurmExecutor(
    nodes=32,
    time="120:00:00",  # 5 days
    # Ray auto-resumes from last checkpoint if node fails!
)

with run.Experiment("llama3_pretrain") as exp:
    exp.add(
        recipe,
        executor=executor,
        name="production_run",
        # Auto-resume on failure
        auto_resume=True,
        max_retries=3
    )
    exp.run()
```

**What Ray Does:**
1. Node fails at step 45,000
2. Ray detects failure
3. Reallocates resources
4. Loads checkpoint from step 44,000
5. Resumes training automatically

---

#### When to Use Ray/NeMo-Run

| Scenario | Use Ray? | Why |
|----------|----------|-----|
| Single GPU training | ❌ No | Unnecessary overhead |
| 1-8 GPUs on single machine | ❌ No | PyTorch DDP is simpler |
| 8+ GPUs across multiple nodes | ✅ Yes | Need cluster orchestration |
| Hyperparameter tuning (5+ experiments) | ✅ Yes | Parallel execution |
| Training > 24 hours | ✅ Yes | Fault tolerance critical |
| Slurm/Kubernetes cluster | ✅ Yes | Built-in integration |

---

#### NeMo-Run Quick Reference

```python
# 1. Local execution (for testing)
executor = run.LocalExecutor(gpus=1)

# 2. Slurm cluster
executor = run.SlurmExecutor(
    account="myaccount",
    partition="gpu",
    nodes=4,
    ntasks_per_node=8,
    time="48:00:00"
)

# 3. Kubernetes
executor = run.KubernetesExecutor(
    namespace="nemo-training",
    image="nvcr.io/nvidia/nemo:24.01",
    gpus_per_node=8
)

# 4. SSH cluster
executor = run.SSHExecutor(
    nodes=["gpu-node-1", "gpu-node-2"],
    user="username"
)
```

---

#### Monitoring Ray Jobs

```bash
# Start Ray dashboard (runs on port 8265)
ray start --head --dashboard-host=0.0.0.0

# View in browser
http://your-server:8265

# Dashboard shows:
# - Active jobs
# - GPU utilization per node
# - Memory usage
# - Task timeline
# - Logs per experiment
```

---

### 3. **NeMo Skills: Improving LLM Capabilities**

#### Reference

https://github.com/NVIDIA-NeMo/Skills

https://developer.nvidia.com/blog/how-to-streamline-complex-llm-workflows-using-nvidia-nemo-skills/

#### What is NeMo Skills?

**NeMo Skills** is a toolkit for **improving LLM performance on specific tasks** through:
- **Synthetic data generation**: Create high-quality training data automatically
- **Iterative improvement**: Train → Evaluate → Generate better data → Retrain
- **Domain specialization**: Math reasoning, coding, complex problem-solving
- **Pipeline orchestration**: End-to-end workflow from data generation to deployment

**Core Purpose:** Transform a general-purpose LLM into a domain expert through automated data generation and iterative training.

---

#### The Problem NeMo Skills Solves

**Traditional Fine-Tuning Challenges:**

```
Problem 1: Lack of Quality Training Data
- Math reasoning datasets: Limited, often low quality
- Coding problems: Need diverse, challenging examples
- Domain-specific tasks: Expensive to create manually

Problem 2: Data Quality Issues
- Human-annotated data: Expensive ($50-200 per example)
- Crowdsourced data: Inconsistent quality
- Static datasets: Don't adapt to model weaknesses

Problem 3: Iterative Improvement is Manual
- Find weak areas → Create data → Train → Repeat
- Time-consuming, requires expertise
- Hard to scale
```

**NeMo Skills Solution:**

```
✅ Automated synthetic data generation
✅ Self-improving pipeline (find weaknesses, generate targeted data)
✅ High-quality examples at scale (thousands/millions)
✅ Orchestrated with NeMo-Run (distributed generation/training)
```

---

#### How NeMo Skills Works

**The Pipeline:**

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: Base Model (e.g., Llama 3 8B)                  │
│ Performance on math: 45% accuracy                       │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Generate Synthetic Training Data               │
│ - Use strong model (GPT-4) to create problems          │
│ - Generate step-by-step solutions                      │
│ - Create 100K synthetic math problems                  │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Fine-Tune on Synthetic Data                    │
│ - Train Llama 3 8B on generated problems               │
│ - Use LoRA for efficiency                              │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Evaluate & Find Weaknesses                     │
│ - Test on benchmark (e.g., GSM8K)                      │
│ - Identify error patterns (geometry? algebra?)         │
│ - Performance: 68% accuracy (improved!)                │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 5: Targeted Data Generation                       │
│ - Generate MORE data on weak areas                     │
│ - Focus on geometry, word problems                     │
│ - Create 50K additional targeted examples              │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│ Step 6: Retrain (Iteration 2)                          │
│ - Fine-tune on targeted data                           │
│ - Final performance: 82% accuracy! 🎯                  │
└─────────────────────────────────────────────────────────┘
```

**Key Innovation:** The pipeline **automatically identifies weaknesses** and **generates targeted data** to improve them.

---

#### Code Example 1: Synthetic Data Generation

```python
from nemo_skills.synthetic_data import MathProblemGenerator
from nemo_skills.inference import LLMClient

# Step 1: Initialize generator with a strong model
generator = MathProblemGenerator(
    model="gpt-4",  # Use strong model to generate problems
    api_key="your-openai-key"
)

# Step 2: Generate synthetic math problems
problems = generator.generate(
    num_samples=10000,
    difficulty_levels=["easy", "medium", "hard"],
    topics=["algebra", "geometry", "word_problems"],
    format="step_by_step"  # Include reasoning steps
)

# Generated example:
# {
#   "problem": "A train travels 120 miles in 2 hours.
#               If it continues at the same speed, how far will it travel in 5 hours?",
#   "solution": "Step 1: Calculate speed = 120 miles / 2 hours = 60 mph
#                Step 2: Distance = speed × time = 60 mph × 5 hours = 300 miles
#                Answer: 300 miles",
#   "answer": 300,
#   "difficulty": "easy",
#   "topic": "word_problems"
# }

# Step 3: Save to JSONL format for training
generator.save(problems, "synthetic_math_train.jsonl")
```

---

#### Code Example 2: End-to-End Training Pipeline

```python
import nemo_run as run
from nemo.collections import llm
from nemo_skills.training import SkillsTrainer
from nemo_skills.evaluation import GSM8KEvaluator

# Step 1: Generate synthetic data
run.execute(
    name="generate_synthetic_data",
    script="nemo_skills.generate_math_data",
    config={
        "num_samples": 100000,
        "output_file": "data/synthetic_math.jsonl"
    }
)

# Step 2: Fine-tune model on synthetic data
recipe = llm.llama3_8b.finetune_recipe(
    dir="./checkpoints/math_iteration1",
    num_nodes=2,
    num_gpus_per_node=8
)

recipe.data = "data/synthetic_math.jsonl"
recipe.model.peft = llm.peft.LoRAConfig(
    target_modules=['q_proj', 'v_proj', 'gate_proj', 'up_proj'],
    rank=32
)

with run.Experiment("math_skills_training") as exp:
    exp.add(recipe, name="iteration_1")
    exp.run()

# Step 3: Evaluate on benchmark
evaluator = GSM8KEvaluator(
    model_path="./checkpoints/math_iteration1/final.nemo"
)

results = evaluator.evaluate()
print(f"Accuracy: {results['accuracy']}")  # e.g., 68%

# Step 4: Analyze errors
weak_areas = evaluator.analyze_errors()
# {'geometry': 45%, 'complex_word_problems': 52%, 'algebra': 75%}

# Step 5: Generate targeted data for weak areas
targeted_generator = MathProblemGenerator(
    focus_topics=["geometry", "complex_word_problems"],
    num_samples=50000
)
targeted_data = targeted_generator.generate()

# Step 6: Retrain (Iteration 2)
recipe.data = "data/targeted_math.jsonl"
recipe.dir = "./checkpoints/math_iteration2"

with run.Experiment("math_skills_iteration2") as exp:
    exp.add(recipe, name="iteration_2")
    exp.run()

# Final evaluation
final_results = evaluator.evaluate("./checkpoints/math_iteration2/final.nemo")
print(f"Final Accuracy: {final_results['accuracy']}")  # e.g., 82%!
```

---

#### Code Example 3: Custom Domain Specialization

```python
from nemo_skills.synthetic_data import CustomDomainGenerator

# Example: Create medical diagnosis training data
medical_generator = CustomDomainGenerator(
    domain="medical_diagnosis",
    expert_model="gpt-4",
    template="""
    Generate a medical case with:
    - Patient symptoms
    - Medical history
    - Step-by-step diagnostic reasoning
    - Final diagnosis
    - Treatment plan
    """
)

medical_cases = medical_generator.generate(
    num_samples=5000,
    validation_keywords=["diagnosis", "symptoms", "treatment"]
)

# Train medical specialist model
medical_recipe = llm.llama3_8b.finetune_recipe(
    dir="./checkpoints/medical_specialist",
    data="medical_cases.jsonl"
)

# Result: General Llama → Medical diagnosis specialist
```

---

#### NeMo Skills Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Synthetic Data Generation** | Auto-generate training examples | No manual annotation needed |
| **Multi-Model Support** | Use GPT-4, Claude, Llama for generation | Leverage best models for data creation |
| **Iterative Improvement** | Train → Eval → Generate → Retrain | Continuously improve on weak areas |
| **Domain Templates** | Pre-built templates (math, code, reasoning) | Quick start for common tasks |
| **Evaluation Suite** | Built-in benchmarks (GSM8K, MATH, HumanEval) | Track progress systematically |
| **Ray Integration** | Distributed data generation/training | Scale to millions of examples |
| **Quality Filtering** | Automatic filtering of low-quality data | Only train on good examples |

---

#### Real-World Results

**Math Reasoning (GSM8K Benchmark):**

```
Base Llama 3 8B:          45% accuracy
+ NeMo Skills (1 iter):   68% accuracy  (+23% improvement)
+ NeMo Skills (3 iters):  82% accuracy  (+37% improvement)

Training cost: $500 (vs $50K+ for human annotation)
Training time: 2 days (vs months of manual data creation)
```

**Code Generation (HumanEval):**

```
Base Model:               35% pass@1
+ NeMo Skills:            58% pass@1  (+23% improvement)

Generated 100K coding problems automatically
```

---

#### When to Use NeMo Skills

| Use Case | NeMo Skills? | Why |
|----------|--------------|-----|
| General chatbot | ❌ No | Pre-trained models sufficient |
| Math reasoning | ✅ Yes | Need specialized training data |
| Code generation | ✅ Yes | Diverse problem generation |
| Domain specialization (medical, legal) | ✅ Yes | Expensive to get expert annotations |
| Limited training data | ✅ Yes | Generate synthetic data |
| Iterative improvement needed | ✅ Yes | Automated weakness identification |

---

#### NeMo Skills vs Traditional Fine-Tuning

| Aspect | Traditional Fine-Tuning | NeMo Skills |
|--------|-------------------------|-------------|
| **Data Source** | Manual annotation | Automated generation |
| **Data Quality** | Variable | Consistent (filtered) |
| **Cost** | $50-200 per example | $0.01-0.10 per example |
| **Iteration** | Manual | Automated pipeline |
| **Weak Area Focus** | Manual analysis | Automatic identification |
| **Scale** | 1K-10K examples | 100K-1M+ examples |
| **Time** | Months | Days |

---

#### Quick Start: NeMo Skills Setup

```bash
# Install NeMo Skills
pip install nemo-skills

# Clone examples
git clone https://github.com/NVIDIA/NeMo-Skills.git
cd NeMo-Skills

# Generate math data
python scripts/generate_math_data.py \
    --num-samples 10000 \
    --output data/math_train.jsonl

# Fine-tune
python scripts/train.py \
    --model meta-llama/Llama-3-8B \
    --data data/math_train.jsonl \
    --output checkpoints/math_model

# Evaluate
python scripts/evaluate.py \
    --model checkpoints/math_model \
    --benchmark gsm8k
```

---

#### Advanced: Custom Skill Pipeline

```python
from nemo_skills import SkillsPipeline

# Define end-to-end pipeline
pipeline = SkillsPipeline(
    base_model="llama3-8b",
    skill="custom_reasoning",
    iterations=3,  # 3 rounds of improvement

    # Data generation config
    generation={
        "generator_model": "gpt-4",
        "num_samples_per_iter": 50000,
        "templates": "custom_templates/"
    },

    # Training config
    training={
        "peft": "lora",
        "rank": 32,
        "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj"],
        "num_gpus": 16
    },

    # Evaluation config
    evaluation={
        "benchmarks": ["custom_benchmark_1", "custom_benchmark_2"],
        "weak_area_threshold": 0.60  # Focus on areas below 60%
    }
)

# Run entire pipeline
results = pipeline.run()

# Output:
# Iteration 1: 55% → 68% (+13%)
# Iteration 2: 68% → 78% (+10%)
# Iteration 3: 78% → 85% (+7%)
# Final model: checkpoints/custom_reasoning_final.nemo
```

---

#### Key Takeaways

1. **NeMo Skills = Automated Specialization**: Turn general models into domain experts
2. **Synthetic Data at Scale**: Generate 100K+ high-quality examples automatically
3. **Self-Improving Loop**: Automatically find weaknesses and generate targeted data
4. **Cost-Effective**: 100x cheaper than manual annotation
5. **Proven Results**: 37% improvement on math reasoning, 23% on code generation

**Best Use Case:** When you need specialized capabilities but lack training data.

---

**Learning Resource**: "Improve LLM Abilities Using NeMo-Skills" [web:391]

### 4. **Multimodal Models**
- Vision-Language Models
- Text-to-Image generation
- Multi-modal embeddings

**Docs**: NeMo Multimodal Collection

### 4. **Custom Model Architectures**
- Extending Neural Modules
- Custom loss functions
- Registered modules

**Docs**: NeMo Model Development

### 5. **Production Observability**
- OpenTelemetry integration
- Phoenix tracing
- Monitoring dashboards

**Video**: "Production-Ready Agents" [web:392]

### 6. **Guardrails & Safety**
- Content filtering
- Toxicity detection
- Custom guardrails

**Video**: "Build Custom Generative AI Model" [web:395]

---

## Understanding Checkpoints in LLMs

### What is a Checkpoint?

A **checkpoint** is a **snapshot of a model's state** saved during or after training, allowing you to:
- Resume training from a specific point
- Use a pre-trained model for inference
- Fine-tune a model on new data
- Share trained models with others

**Think of it as:** A save point in a video game - you can reload from that exact state later.

---

### Checkpoint Contents

A complete LLM checkpoint contains:

#### 1. **Model Weights (Parameters)** 🎯
The learned values of all neural network parameters.

**In Transformer Architecture:**

```
📦 Checkpoint File(s)
├── Embedding Layers
│   ├── token_embeddings.weight [vocab_size × hidden_dim]
│   └── position_embeddings.weight [max_seq_len × hidden_dim]
│
├── Transformer Blocks (repeated N times)
│   ├── Block 0
│   │   ├── Self-Attention
│   │   │   ├── query.weight [hidden_dim × hidden_dim]
│   │   │   ├── query.bias [hidden_dim]
│   │   │   ├── key.weight [hidden_dim × hidden_dim]
│   │   │   ├── key.bias [hidden_dim]
│   │   │   ├── value.weight [hidden_dim × hidden_dim]
│   │   │   ├── value.bias [hidden_dim]
│   │   │   └── output.weight [hidden_dim × hidden_dim]
│   │   │
│   │   ├── Layer Norm 1
│   │   │   ├── gamma [hidden_dim]
│   │   │   └── beta [hidden_dim]
│   │   │
│   │   ├── Feed-Forward Network
│   │   │   ├── fc1.weight [hidden_dim × ffn_dim]
│   │   │   ├── fc1.bias [ffn_dim]
│   │   │   ├── fc2.weight [ffn_dim × hidden_dim]
│   │   │   └── fc2.bias [hidden_dim]
│   │   │
│   │   └── Layer Norm 2
│   │       ├── gamma [hidden_dim]
│   │       └── beta [hidden_dim]
│   │
│   ├── Block 1 (same structure)
│   ├── Block 2 (same structure)
│   └── ... (up to N blocks)
│
├── Final Layer Norm
│   ├── gamma [hidden_dim]
│   └── beta [hidden_dim]
│
└── Output Head (LM Head)
    └── lm_head.weight [vocab_size × hidden_dim]
```

**Example - Llama 2 7B Checkpoint:**
- 32 transformer blocks
- 4,096 hidden dimensions
- 32,000 vocabulary size
- **Total: ~7 billion parameters**
- **File size: ~13GB** (FP16 format)

#### 2. **Optimizer State** (Training Checkpoints Only)
Momentum, variance, learning rates for each parameter.

```python
# Adam optimizer state (most common)
{
    'momentum': {...},      # First moment estimates
    'variance': {...},      # Second moment estimates
    'step': 15000          # Training step count
}
```

**Size:** Often **2-3x larger** than model weights alone!
- Model weights: 13GB
- Full checkpoint with optimizer: **40GB+**

#### 3. **Training Configuration**
```python
{
    'epoch': 3,
    'global_step': 15000,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'model_config': {
        'hidden_size': 4096,
        'num_layers': 32,
        'num_heads': 32,
        'vocab_size': 32000
    }
}
```

#### 4. **Tokenizer Files** (Sometimes Separate)
- Vocabulary mappings
- Special tokens
- Tokenization rules

---

### Where Checkpoints are Saved in Training

#### Training Loop with Checkpointing

```python
# Typical training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(batch)
        loss = compute_loss(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # 🔸 Checkpoint at intervals
        if step % checkpoint_interval == 0:
            save_checkpoint({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'loss': loss.item()
            }, f'checkpoint_step_{step}.pt')
```

#### Common Checkpointing Strategies

1. **Periodic Checkpoints** - Every N steps/epochs
   ```
   checkpoint_step_1000.pt
   checkpoint_step_2000.pt
   checkpoint_step_3000.pt
   ```

2. **Best Model Checkpoint** - Lowest validation loss
   ```
   best_model.pt (saved when val_loss improves)
   ```

3. **Last Checkpoint** - Most recent state
   ```
   last.pt (continuously overwritten)
   ```

4. **End-of-Epoch Checkpoints**
   ```
   epoch_1.pt
   epoch_2.pt
   epoch_3.pt
   ```

#### NeMo Checkpoint Example

```python
from nemo.collections import llm

# NeMo automatically saves checkpoints
trainer = Trainer(
    max_epochs=3,
    save_checkpoint_interval=1000,  # Every 1000 steps
    checkpoint_dir='./checkpoints'
)

# Checkpoints saved as:
# checkpoints/
#   ├── step_1000.nemo
#   ├── step_2000.nemo
#   └── last.nemo
```

---

### Checkpoint Fine-Tuning vs Full Training

#### Full Training (From Scratch)

```
Random Initialization
    ↓
Train on Large Corpus (e.g., Wikipedia, Books, Web)
    ↓ (weeks/months on thousands of GPUs)
Pre-trained Model Checkpoint
```

**Characteristics:**
- ✅ Learns general language understanding
- ❌ Extremely expensive ($1M - $10M+)
- ❌ Requires massive compute (10,000+ GPU days)
- ❌ Needs huge datasets (trillions of tokens)

**Example:** Training GPT-3 from scratch

---

#### Checkpoint Fine-Tuning (Transfer Learning)

```
Pre-trained Checkpoint (e.g., Llama 2 7B)
    ↓
Load weights
    ↓
Fine-tune on Task-Specific Data (e.g., medical Q&A)
    ↓ (hours/days on few GPUs)
Fine-tuned Model
```

**Characteristics:**
- ✅ Starts with learned knowledge
- ✅ Much cheaper ($100 - $1,000)
- ✅ Faster (hours to days)
- ✅ Needs smaller datasets (thousands to millions of examples)

**Example:** Fine-tuning Llama 2 for customer support

---

### Comparison Table

| Aspect | Full Training | Checkpoint Fine-Tuning |
|--------|---------------|------------------------|
| **Starting Point** | Random weights | Pre-trained checkpoint |
| **Compute** | 10,000+ GPU days | 10-100 GPU hours |
| **Cost** | $1M - $10M+ | $100 - $10K |
| **Data Required** | Trillions of tokens | Thousands - millions |
| **Time** | Weeks to months | Hours to days |
| **What Updates** | All 7B parameters | All or subset (LoRA) |
| **Use Case** | Create foundation model | Adapt to specific task |
| **Example** | Training GPT-4 | Fine-tune for medical use |

---

### Fine-Tuning Approaches

#### 1. **Full Fine-Tuning**
Update **all** model parameters.

```python
# Load checkpoint
model = load_checkpoint('llama-2-7b.pt')

# Train all parameters
for param in model.parameters():
    param.requires_grad = True  # All params trainable

# Fine-tune
train(model, custom_dataset)
```

**Pros:** Maximum adaptation to new task
**Cons:** Memory-intensive, risk of catastrophic forgetting

---

#### 2. **Parameter-Efficient Fine-Tuning (PEFT)**
Update only a **small subset** of parameters.

##### **LoRA (Low-Rank Adaptation)** - Most Popular

Instead of updating full weight matrices, add small trainable "adapter" matrices:

```
Original Weight: W [4096 × 4096] = 16M parameters

LoRA Decomposition:
W_new = W (frozen) + A × B
  where A [4096 × 8], B [8 × 4096]
  Trainable params: 4096×8 + 8×4096 = ~65K parameters

Reduction: 16M → 65K (250x fewer parameters!)
```

---

### Understanding LoRA Target Modules

When configuring LoRA, you specify which model components to adapt via `target_modules`. Here's what each module does:

#### Attention Projection Matrices

These are the most commonly targeted modules because they're the largest parameter matrices in transformers:

| Module | Full Name | Purpose | Shape Example |
|--------|-----------|---------|---------------|
| `q_proj` | Query Projection | Transforms input to query vectors for attention | [4096 × 4096] |
| `k_proj` | Key Projection | Transforms input to key vectors for attention | [4096 × 4096] |
| `v_proj` | Value Projection | Transforms input to value vectors for attention | [4096 × 4096] |
| `o_proj` | Output Projection | Combines attention heads back to hidden size | [4096 × 4096] |

**How They Work in Self-Attention:**
```
Input Embedding [batch, seq_len, 4096]
    ↓
Q = Input × q_proj  →  Query vectors
K = Input × k_proj  →  Key vectors
V = Input × v_proj  →  Value vectors
    ↓
Attention = softmax(Q·K^T / √d) · V
    ↓
Output = Attention × o_proj  →  [batch, seq_len, 4096]
```

---

#### Feed-Forward Network (FFN) Projections

After attention, transformers use a **Feed-Forward Network (FFN)** to process information. Modern architectures like Llama use a **Gated FFN** (also called SwiGLU) instead of the traditional FFN.

##### Traditional FFN (GPT-2, BERT style)

```
Input [4096]
    ↓
fc1 (expand): [4096] → [16384]  (4x expansion)
    ↓
Activation (ReLU/GELU)
    ↓
fc2 (compress): [16384] → [4096]
    ↓
Output [4096]
```

**Modules:** `fc1`, `fc2` or `mlp.c_fc`, `mlp.c_proj`

---

##### Gated FFN (Llama, Mistral, Mixtral style)

Modern models use a **gated mechanism** for better performance:

```
Input [4096]
    ↓
    ├─> gate_proj [4096 × 14336] → Gate values
    │
    └─> up_proj [4096 × 14336]   → Up-projected values
         ↓
    Gate ⊗ Up (element-wise multiply with SiLU activation)
         ↓
    down_proj [14336 × 4096]     → Compress back
         ↓
Output [4096]
```

**The Three Projections Explained:**

| Module | Full Name | Purpose | Shape Example | What It Does |
|--------|-----------|---------|---------------|--------------|
| `gate_proj` | Gate Projection | Controls information flow (gating mechanism) | [4096 × 14336] | Creates "gate" values that decide what to pass through |
| `up_proj` | Up Projection | Expands to intermediate dimension | [4096 × 14336] | Projects input to larger space for processing |
| `down_proj` | Down Projection | Compresses back to model dimension | [14336 → 4096] | Projects back to original hidden size |

---

#### Visual Example: How Gated FFN Works

```python
# Simplified Llama FFN forward pass

class LlamaFFN:
    def forward(self, x):
        # x shape: [batch, seq_len, 4096]

        # 1. Gate projection - learns what to "gate" (filter)
        gate = self.gate_proj(x)  # [batch, seq_len, 14336]
        gate = silu(gate)          # SiLU activation: x * sigmoid(x)

        # 2. Up projection - expands input
        up = self.up_proj(x)      # [batch, seq_len, 14336]

        # 3. Element-wise multiply (gating!)
        intermediate = gate * up   # [batch, seq_len, 14336]

        # 4. Down projection - compress back
        output = self.down_proj(intermediate)  # [batch, seq_len, 4096]

        return output
```

**Concrete Example with Numbers:**

```
Input vector: [0.5, 0.2, 0.8, 0.1, ...] (4096 values)

Step 1 - Gate Projection:
  gate_proj: [4096 × 14336] matrix multiply
  → [0.7, 0.3, 0.9, 0.1, ...] (14336 values)
  → Apply SiLU activation
  → [0.52, 0.15, 0.74, 0.05, ...] (gate values)

Step 2 - Up Projection:
  up_proj: [4096 × 14336] matrix multiply
  → [0.9, 0.8, 0.6, 0.4, ...] (14336 values)

Step 3 - Gating (element-wise multiply):
  gate ⊗ up
  → [0.52×0.9, 0.15×0.8, 0.74×0.6, 0.05×0.4, ...]
  → [0.468, 0.12, 0.444, 0.02, ...] (14336 values)

  👆 Gate controls what information passes through!

Step 4 - Down Projection:
  down_proj: [14336 × 4096] matrix multiply
  → [0.6, 0.3, 0.7, 0.2, ...] (4096 values - back to original size)
```

---

#### Why Gated FFN? (vs Traditional FFN)

| Aspect | Traditional FFN | Gated FFN (Llama) |
|--------|-----------------|-------------------|
| **Mechanism** | fc1 → ReLU → fc2 | gate_proj ⊗ up_proj → down_proj |
| **Parameters** | 2 matrices (fc1, fc2) | 3 matrices (gate, up, down) |
| **Activation** | ReLU or GELU | SiLU (Swish) |
| **Gating** | No gating | Dynamic gating mechanism |
| **Performance** | Good | Better (5-10% improvement) |
| **Use Case** | Older models (GPT-2, BERT) | Modern models (Llama, Mistral) |

**Why it's better:**
- **Dynamic gating**: Learns to selectively filter information
- **Better gradient flow**: SiLU activation helps training
- **Improved capacity**: More expressive with same parameter count
- **State-of-the-art**: Used in Llama 2/3, Mistral, Mixtral

---

#### Complete Transformer Block (Llama Style)

```
Input [batch, seq_len, 4096]
    ↓
┌─────────────────────────────────────────┐
│ 1. Self-Attention Block                │
│    ├─ q_proj [4096 × 4096]             │
│    ├─ k_proj [4096 × 4096]             │
│    ├─ v_proj [4096 × 4096]             │
│    └─ o_proj [4096 × 4096]             │
└─────────────────────────────────────────┘
    ↓ (residual connection + layer norm)
┌─────────────────────────────────────────┐
│ 2. Gated Feed-Forward Block             │
│    ├─ gate_proj [4096 × 14336]         │
│    ├─ up_proj [4096 × 14336]           │
│    └─ down_proj [14336 × 4096]         │
└─────────────────────────────────────────┘
    ↓ (residual connection + layer norm)
Output [batch, seq_len, 4096]
```

**Parameter Count (One Llama Block):**
```
Attention:
  q_proj: 4096 × 4096 = 16.8M
  k_proj: 4096 × 4096 = 16.8M
  v_proj: 4096 × 4096 = 16.8M
  o_proj: 4096 × 4096 = 16.8M
  Subtotal: 67.2M parameters

FFN (Gated):
  gate_proj: 4096 × 14336 = 58.7M
  up_proj:   4096 × 14336 = 58.7M
  down_proj: 14336 × 4096 = 58.7M
  Subtotal: 176.1M parameters

Total per block: 243.3M parameters
```

**💡 Key Insight:** FFN has **2.6x more parameters** than attention! This is why targeting `gate_proj`, `up_proj`, `down_proj` in LoRA can significantly improve adaptation for domain-specific tasks.


---

#### Other Common Target Modules

Beyond attention projections, you can target:

| Module Pattern | Component | Why Target It |
|----------------|-----------|---------------|
| `mlp`, `fc1`, `fc2` | Feed-Forward Network layers | Dense transformations after attention |
| `gate_proj`, `up_proj`, `down_proj` | Gated FFN (Llama style) | Alternative FFN architecture |
| `embed_tokens` | Token Embeddings | Input vocabulary representations |
| `lm_head` | Language Model Head | Output logits projection |
| `.*` (regex pattern) | All Linear Layers | Maximum adaptation (more params) |

#### Choosing Target Modules

**Common Strategies:**

1. **Attention-Only** (Most Popular):
   ```python
   target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
   ```
   - ✅ Good balance: performance vs. parameter count
   - ✅ Captures most semantic adaptations
   - 📊 Params: ~0.5-1% of model

2. **Query-Value Only** (Lightweight):
   ```python
   target_modules=['q_proj', 'v_proj']
   ```
   - ✅ Minimal parameters (~0.25% of model)
   - ✅ Still effective for many tasks
   - ⚡ Fastest training

3. **Attention + FFN** (Comprehensive):
   ```python
   target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
   ```
   - ✅ Better performance on complex tasks
   - 📊 Params: ~1-2% of model
   - ⏱️ Longer training time

4. **All-Linear** (Maximum Adaptation):
   ```python
   target_modules='.*'  # Regex for all layers
   ```
   - ✅ Best performance
   - ❌ More parameters (2-3% of model)
   - ❌ Higher memory usage

**Rule of Thumb:**
- Start with `['q_proj', 'v_proj']` for simple tasks
- Use all 4 attention projections for most cases
- Add FFN layers only for domain-heavy adaptation (legal, medical, code)
---

#### Why Expand-Compress(up_proj/down_proj)? (4096 → 14336 → 4096)

**The Question:** Why expand to 14336 and compress back to 4096?

**Answer:** **Representational capacity** - Higher dimensions enable non-linear transformations.

**Without Expansion:**
```python
output = W × input  # [4096 × 4096] - LINEAR only!
❌ Cannot learn complex patterns (e.g., XOR, disambiguation)
```

**With Expansion:**
```python
intermediate = W1 × input              # [4096 × 14336] - Expand
intermediate = SiLU(gate) * up         # Non-linearity + gating
output = W2 × intermediate             # [14336 × 4096] - Compress
✅ Can learn complex, non-linear functions
```

**Analogy:** "Thinking space"
- **4096**: Compressed representation (like a summary)
- **14336**: Expanded space to process details (like zooming in)
- **Gate**: Filters what's important (dynamic attention to features)
- **4096**: Refined output (like zooming back out)

**Example - Disambiguating "bank":**
```
Input: "bank" [4096 dims] - ambiguous
  ↓ up_proj expands to 14336 dims
  - Neuron 100: "financial context" [high]
  - Neuron 200: "river context" [low]
  - Neuron 500: "preceded by 'money'" [high]
  ↓ gate_proj filters important features
  ↓ down_proj compresses to 4096 dims
Output: "bank" [4096 dims] - clearly means "financial institution"
```

**Why 3.5-4x expansion?** Empirically optimal:
- **2x**: Too small, poor performance
- **4x**: Sweet spot (used in GPT, BERT, Llama)
- **8x**: Diminishing returns, wasted memory

**Performance Impact:**
```
No expansion (4096→4096):    62% accuracy, 4B params
With expansion (4096→14336): 85% accuracy, 7B params
```
Expansion is **critical** - even with fewer params, no expansion = much worse performance!
---

**NeMo LoRA Example:**

```python
from nemo.collections import llm

# Load base checkpoint
model = llm.Llama3Model.from_pretrained('llama-2-7b')

# Add LoRA adapters
lora_config = llm.peft.LoRAConfig(
    target_modules=['q_proj', 'v_proj'],  # Only adapt attention
    rank=8,                                # Low-rank dimension
    alpha=16
)

# Fine-tune (only LoRA params update)
recipe = llm.finetune(
    model=model,
    peft=lora_config,
    data=custom_dataset
)

# Result: Adapter weights only (~50MB vs 13GB full model)
```

**Benefits:**
- 💾 **Memory**: 3-10x less GPU memory
- 🚀 **Speed**: 2-3x faster training
- 💰 **Cost**: Run on single GPU vs multi-GPU
- 📦 **Storage**: Small adapter files (MBs vs GBs)

---

#### 3. **Prefix Tuning**
Add trainable "prefix" tokens to each layer.

#### 4. **Adapter Layers**
Insert small trainable layers between frozen transformer blocks.

---

### Practical Example: NeMo Workflow

#### Scenario: Fine-tune Llama 2 7B for Medical Q&A

```python
from nemo.collections import llm

# 1. Download pre-trained checkpoint
base_checkpoint = "meta-llama/Llama-2-7b-hf"

# 2. Setup LoRA fine-tuning
lora_config = llm.peft.LoRAConfig(
    rank=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
)

# 3. Fine-tune
recipe = llm.finetune(
    model=base_checkpoint,
    peft=lora_config,
    data='medical_qa_dataset.jsonl',
    trainer=Trainer(
        max_epochs=3,
        gpus=1,  # Single GPU sufficient!
        checkpoint_interval=500
    )
)

# 4. Checkpoints saved
# checkpoints/
#   ├── lora_step_500.pt     (adapter only, ~50MB)
#   ├── lora_step_1000.pt
#   └── lora_final.pt

# 5. Inference: Merge adapter with base
merged_model = merge_lora_checkpoint(
    base='llama-2-7b',
    adapter='lora_final.pt'
)
```

**Results:**
- Training time: 4 hours (vs 2 weeks full training)
- Cost: $20 (vs $500,000+)
- GPU: 1x A100 (vs 256+ GPUs)
- Checkpoint size: 50MB adapter + 13GB base (shareable)

---

### Key Takeaways

1. **Checkpoints = Saved Model State**: Weights + optimizer + config
2. **Location in Transformer**: Every layer has parameters saved (embeddings → blocks → head)
3. **Fine-Tuning ≠ Full Training**: Start with pre-trained knowledge vs random initialization
4. **LoRA = Efficient Fine-Tuning**: Train <1% of parameters, 10x less memory
5. **NeMo Benefits**: Built-in checkpointing, LoRA support, easy conversion

---

## TensorRT-LLM vs vLLM: Inference Engine Comparison

### What is TensorRT-LLM?

**TensorRT-LLM** is NVIDIA's high-performance inference library that optimizes Large Language Models for deployment on NVIDIA GPUs.

**Core Purpose:**
- **Model Optimization**: Compiles LLMs into optimized engines using TensorRT (NVIDIA's deep learning compiler)
- **Maximum Performance**: Extracts peak performance from NVIDIA GPUs through kernel fusion, quantization, and hardware-specific optimizations
- **Production Deployment**: Enterprise-grade solution for serving models at scale with minimal latency

**How It Works:**
```
Trained Model (PyTorch/NeMo)
    → Convert to TensorRT format
    → Optimize & Compile (graph optimizations, kernel fusion)
    → Build Engine (GPU-specific binary)
    → Deploy (ultra-fast inference)
```

**Key Capabilities:**
- Advanced quantization: FP8, INT8, INT4 (2-4x speedup)
- Multi-GPU inference with tensor/pipeline parallelism
- In-flight batching for dynamic workloads
- Integrates with Triton Inference Server

---

### Quick Comparison

| Aspect | TensorRT-LLM | vLLM |
|--------|--------------|------|
| **Developer** | NVIDIA | UC Berkeley |
| **Setup** | Complex (30-60 min) | Easy (5 min, pip install) |
| **Best For** | Low latency, enterprise | High throughput, prototyping |
| **Latency (single)** | 180ms ⚡ | 250ms |
| **Throughput (batch)** | 2,400 tok/s | 2,800 tok/s ⚡ |
| **GPU Support** | NVIDIA only | NVIDIA, AMD ROCm, CPU |
| **HuggingFace** | Needs conversion | Direct support |
| **Quantization** | FP8/INT8/INT4 | AWQ/GPTQ |
| **OpenAI API** | Via Triton | Native ✅ |

---

### Detailed Breakdown

#### TensorRT-LLM

**What:** NVIDIA's production-grade compiler that converts models into optimized GPU engines.

**Strengths:**
- ⚡ **Speed**: 20-40% faster latency than vLLM (optimized CUDA kernels)
- 🔧 **Quantization**: FP8/INT8/INT4 support (2-4x speedup, minimal accuracy loss)
- 🏢 **Enterprise**: Official NVIDIA support, Triton integration, production-ready
- 💾 **Efficiency**: Optimized memory layout, kernel fusion

**Limitations:**
- ⏱️ **Setup**: 30-60 min build time per model
- 🔒 **GPU Lock-in**: NVIDIA-only (no AMD/CPU)
- 📅 **Model Support**: Slower to support new architectures

**Best For:** Low-latency production, enterprise deployments, aggressive optimization

---

#### vLLM

**What:** Open-source inference engine with revolutionary PagedAttention memory management.

**Strengths:**
- 🚀 **Easy**: `pip install vllm` - works in minutes
- 💡 **PagedAttention**: 60-80% less KV cache waste (like OS virtual memory)
- 📈 **Throughput**: Better batch performance for concurrent users
- 🤗 **HuggingFace**: Direct compatibility, no conversion needed
- ⚙️ **Flexibility**: Fast new model support, cross-platform (NVIDIA/AMD/CPU)

**Limitations:**
- 🐢 **Latency**: 20-30% slower per request vs TensorRT-LLM
- 📦 **Quantization**: Limited (AWQ/GPTQ only, no FP8)
- 🧪 **Maturity**: Younger project, community support

**Best For:** High-throughput serving, rapid prototyping, OpenAI API replacement

---

### Performance (Llama 2 70B)

| Metric | TensorRT-LLM | vLLM |
|--------|--------------|------|
| Latency (single) | 180ms | 250ms |
| Throughput (batch=64) | 2,400 tok/s | 2,800 tok/s |
| Memory (batch=32) | 48GB | 52GB |

**Insight:** TensorRT-LLM → low latency, vLLM → high throughput

---

### Code Examples

**vLLM** (5 min):
```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-2-70b-hf")
outputs = llm.generate(["Hello"], SamplingParams(temp=0.8))
```

**TensorRT-LLM** (30-60 min):
```bash
# 1. Convert
python convert_checkpoint.py --model_dir ./llama-70b-hf --output_dir ./trt

# 2. Build
trtllm-build --checkpoint_dir ./trt --output_dir ./engine

# 3. Run
python run.py --engine_dir ./engine
```

---

### Recommendation for NeMo Users

- **Training**: NeMo Framework
- **Dev/Testing**: vLLM (export HF format)
- **Production**: TensorRT-LLM (NeMo export)
- **Hybrid**: vLLM (dev) + TensorRT-LLM (prod)

---

## Mamba vs Transformer: State Space Models Explained

### What is Mamba?

**Mamba** is a new type of architecture for language models that uses **State Space Models (SSMs)** instead of the attention mechanism used in Transformers.

**Think of it like this:**
- **Transformer**: Looks at ALL previous words every time (like re-reading the entire book to understand each new sentence)
- **Mamba**: Maintains a compact "memory summary" and updates it as it reads (like keeping running notes while reading)

**Published:** December 2023 by researchers at Carnegie Mellon University and Princeton
**Key Innovation:** Selective State Space Models with hardware-aware algorithms

---

### The Core Problem with Transformers

#### Problem 1: Quadratic Memory & Computation Cost

**Transformer Attention** looks at every token when processing each new token:

```
Processing 1,000 tokens:
- Token 1 looks at: Token 1 (1 comparison)
- Token 2 looks at: Tokens 1-2 (2 comparisons)
- Token 3 looks at: Tokens 1-3 (3 comparisons)
...
- Token 1000 looks at: Tokens 1-1000 (1,000 comparisons)

Total comparisons: 1 + 2 + 3 + ... + 1,000 = 500,500 comparisons
Formula: n²/2 where n = sequence length
```

**Cost:** O(n²) - doubles with sequence length!

**Real Impact:**
```
Sequence Length  | Memory (Attention) | Inference Time
─────────────────┼────────────────────┼────────────────
1,000 tokens     | 4 MB               | 10 ms
10,000 tokens    | 400 MB             | 1,000 ms (1 sec)
100,000 tokens   | 40 GB              | 100,000 ms (100 sec!)
1,000,000 tokens | 4 TB (impossible!) | Too slow
```

**Why this matters:**
- Long documents (100K+ tokens) are slow or impossible
- Cost scales quadratically with context length
- Memory becomes the bottleneck

---

### How Mamba Solves It

Mamba uses a **State Space Model** that processes sequences in **linear time** O(n).

#### The Big Idea: Compressed State

Instead of storing all previous tokens, Mamba maintains a small **hidden state** that summarizes the past.

```
Transformer Approach (O(n²)):
Token 1000 needs to look at all 999 previous tokens
Memory: Must store attention for all token pairs

Mamba Approach (O(n)):
Token 1000 only needs:
1. Current token
2. Small hidden state (summary of tokens 1-999)
Memory: Fixed-size state (e.g., 16 KB)
```

**Visual Comparison:**

```
TRANSFORMER (Attention Mechanism):
Input:  [The] [quick] [brown] [fox] [jumps]
         ↓      ↓       ↓       ↓      ↓
       Attention looks at ALL previous tokens:

       [jumps] → Attention → [The, quick, brown, fox, jumps]
                              ↑__________________________|
                              Must compute attention scores
                              for ALL pairs (expensive!)

Memory grows with sequence: O(n²)


MAMBA (State Space Model):
Input:  [The] [quick] [brown] [fox] [jumps]
         ↓      ↓       ↓       ↓      ↓
        Update small hidden state at each step:

Step 1: [The]   → State₁ [compressed info about "The"]
Step 2: [quick] + State₁ → State₂ [compressed info about "The quick"]
Step 3: [brown] + State₂ → State₃ [compressed info about "The quick brown"]
Step 4: [fox]   + State₃ → State₄ [compressed info so far]
Step 5: [jumps] + State₄ → State₅ [all context compressed]
                   ↑
            Fixed-size state (e.g., 4096 dimensions)

Memory is constant: O(1) per token
Total: O(n) for sequence
```

---

### Key Differences: Transformer vs Mamba

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| **Core Mechanism** | Self-Attention | State Space Model (SSM) |
| **Looks at** | All previous tokens | Compressed hidden state |
| **Computation** | O(n²) quadratic | O(n) linear |
| **Memory** | O(n²) for attention | O(n) for states |
| **Long Sequences** | Slow & memory-heavy | Fast & efficient |
| **Context Length** | Limited (4K-128K) | Can handle millions |
| **Inference Speed** | Slower for long text | 5-9x faster |
| **Training** | Well-optimized (Flash Attention) | Newer, improving |
| **Quality (Short)** | Excellent | Comparable |
| **Quality (Long)** | Good | Better on very long context |

---

### How Mamba Works (Simplified)

#### Step 1: Selective State Space Model

```python
# Simplified Mamba processing

# Traditional SSM (not selective):
# h_t = A * h_{t-1} + B * x_t    # Fixed A, B matrices
# y_t = C * h_t                  # Fixed C matrix

# Mamba (selective - A, B, C change based on input!):
for token in sequence:
    # Make A, B, C depend on current input (selective!)
    A_t = compute_A(token)  # Changes based on token
    B_t = compute_B(token)  # Changes based on token
    C_t = compute_C(token)  # Changes based on token

    # Update state
    hidden_state = A_t * hidden_state + B_t * token

    # Produce output
    output = C_t * hidden_state
```

**Why "Selective"?**
- Traditional SSMs use fixed weights (A, B, C)
- Mamba makes them **input-dependent** - they change based on what token you're processing
- This lets Mamba "decide" what to remember and what to forget

#### Step 2: Hardware-Aware Algorithm

Mamba uses a special algorithm that runs efficiently on GPUs:
- Processes sequences in **chunks** (like Flash Attention)
- Minimizes memory reads/writes
- Optimized kernel fusion

---

### Performance Comparison

#### Inference Speed (Generating 1000 tokens)

```
Model Size: 7B parameters
Context Length: 100,000 tokens

Architecture    | Inference Time | Memory
────────────────┼────────────────┼────────
Transformer     | 45 seconds     | 48 GB
Mamba          | 5 seconds      | 8 GB

Speedup: 9x faster, 6x less memory! ⚡
```

#### Quality Comparison

```
Task                    | Transformer | Mamba
────────────────────────┼─────────────┼───────
Short prompts (<4K)     | 85%         | 84% (similar)
Long context (50K)      | 78%         | 82% (better!)
Million-token docs      | Impossible  | 79% (works!)
General language        | 87%         | 85% (slightly lower)
```

---

### Concrete Example: Processing a Long Document

**Scenario:** Summarizing a 100,000-token legal document

```
TRANSFORMER:
1. Token 1: Compute attention over [Token 1]
2. Token 2: Compute attention over [Tokens 1-2]
...
100,000. Token 100K: Compute attention over [Tokens 1-100,000]

Total attention computations: 5 billion (100K² / 2)
Memory: 40 GB for attention matrices
Time: 120 seconds
❌ Often runs out of memory!


MAMBA:
1. Token 1: Update state with Token 1 info
2. Token 2: Update state with Token 2 info
...
100,000. Token 100K: Update state with Token 100K info

Total state updates: 100,000 (linear!)
Memory: 8 GB (fixed-size states)
Time: 12 seconds
✅ Fast and efficient!
```

---

### When to Use Each

#### Use Transformers When:
✅ Sequence length < 10,000 tokens
✅ Need absolute best quality on general tasks
✅ Using existing pre-trained models (Llama, GPT)
✅ Well-established training recipes
✅ Rich ecosystem (HuggingFace, tooling)

**Example Use Cases:**
- Chatbots with moderate context
- Code completion
- Translation
- General Q&A

---

#### Use Mamba When:
✅ Very long sequences (100K+ tokens)
✅ Need fast inference
✅ Memory-constrained environments
✅ Processing entire books, codebases, or documents
✅ Real-time applications

**Example Use Cases:**
- Analyzing entire books (500K+ tokens)
- Processing full codebases (1M+ tokens)
- Long-form document summarization
- DNA sequence analysis (millions of base pairs)
- Real-time audio processing

---

### Hybrid Models

**Recent Development:** Combining both!

Some new architectures mix Transformers and Mamba:
- Use Mamba layers for processing long sequences
- Use Transformer layers for complex reasoning
- Best of both worlds!

**Example: Jamba (AI21 Labs)**
```
Architecture:
- 80% Mamba layers (for efficiency)
- 20% Transformer layers (for quality)
- Handles 256K context length
- 3x faster than pure Transformers
```

---

### Mamba in NeMo

**NeMo Support:** Experimental (as of January 2026)

```python
from nemo.collections.nlp.models import MambaModel

# Load Mamba model
model = MambaModel.from_pretrained("state-spaces/mamba-2.8b")

# Inference on long document
long_text = "..." # 100K tokens
output = model.generate(long_text, max_length=1000)

# Training Mamba
from nemo.collections.nlp.models import MambaConfig

config = MambaConfig(
    vocab_size=50000,
    d_model=2048,      # Hidden state size
    n_layers=48,
    ssm_cfg={
        "d_state": 16,  # State dimension
        "d_conv": 4,    # Convolution width
    }
)

# Train on your data
trainer = Trainer(...)
model = MambaModel(config)
trainer.fit(model, train_dataloader)
```

---

### Summary: The Big Picture

**Transformers = Attention = Look at Everything**
- Great quality
- Slow on long sequences
- Memory-hungry
- O(n²) complexity

**Mamba = State Space = Compress & Update**
- Good quality (improving)
- Fast on long sequences
- Memory-efficient
- O(n) complexity

**The Future:**
- Transformers still dominate (for now)
- Mamba is catching up fast
- Hybrid models show promise
- Watch this space!

---

*Last Updated: January 2026*
*NeMo Version: 2.0+*
