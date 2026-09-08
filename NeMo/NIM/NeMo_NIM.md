NIM.md
# NVIDIA NIM (NVIDIA Inference Microservices)

## Table of Contents
- [What is NVIDIA NIM?](#what-is-nvidia-nim)
- [Why NIM?](#why-nim)
- [Core Architecture](#core-architecture)
- [Key Features & Capabilities](#key-features--capabilities)
- [Supported Model Categories](#supported-model-categories)
- [Deployment Options](#deployment-options)
- [Deploying NIM with Deployment Management](#deploying-nim-with-deployment-management)
  - [What is Deployment Management?](#what-is-deployment-management)
  - [How is Deployment Management Service Hosted in Enterprises?](#how-is-deployment-management-service-hosted-in-enterprises)
- [Inference Engine vs Serving Platform](#inference-engine-vs-serving-platform)
- [TensorRT vs TensorRT-LLM vs Triton](#tensorrt-vs-tensorrt-llm-vs-triton)
- [TensorRT-LLM/Triton vs vLLM](#tensorrt-llmtriton-vs-vllm)
- [NIM Operator vs GPU Operator](#nim-operator-vs-gpu-operator)
- [Batching in NIM](#batching-in-nim)
- [Monitoring and Observability](#monitoring-and-observability)
- [Scaling NIM for High Traffic (Enterprise)](#scaling-nim-for-high-traffic-enterprise)
- [How NIM Works](#how-nim-works)
- [Use Cases](#use-cases)
- [Getting Started](#getting-started)
- [Benefits for Enterprises](#benefits-for-enterprises)
- [NIM vs Traditional Inference Serving](#nim-vs-traditional-inference-serving)
- [Resources](#resources)
- [NIM Ecosystem Overview](#nim-ecosystem-overview)
- [Summary](#summary)
- [Your NIM Journey: From Development to Production](#your-nim-journey-from-development-to-production)
- [Disaggregated Prefill and Decode](#disaggregated-prefill-and-decode-prefill-decode-disaggregation)
  - [What is Prefill and Decode?](#what-is-prefill-and-decode-the-basics-first)
  - [The Problem: One GPU Trying to Do Both](#the-problem-one-gpu-trying-to-do-both)
  - [The Solution: Disaggregated Prefill and Decode](#the-solution-disaggregated-prefill-and-decode)
  - [Why Does This Work So Well?](#why-does-this-work-so-well)
  - [Before vs After: Performance Impact](#before-vs-after-performance-impact)
  - [How NIM Supports Disaggregated Prefill-Decode](#how-nim-supports-disaggregated-prefill-decode)
  - [When Should You Use Disaggregation?](#when-should-you-use-disaggregation)
  - [NVIDIA Dynamo: The Next Evolution](#nvidia-dynamo-the-next-evolution)
  - [Summary: The Big Picture](#summary-the-big-picture)
  - [Infrastructure Configuration: How to Wire Up P and D GPU Clusters](#infrastructure-configuration-how-to-wire-up-p-and-d-gpu-clusters)

---

## What is NVIDIA NIM?

**https://docs.nvidia.com/nim/large-language-models/latest/introduction.html**


NVIDIA NIM (NVIDIA Inference Microservices) is a set of **optimized, cloud-native microservices** designed to accelerate and simplify the deployment of foundation models and generative AI applications. NIM is part of **NVIDIA AI Enterprise**, providing production-grade runtimes with enterprise support, security updates, and stable APIs.

In simple terms, NIM takes complex AI models and packages them into easy-to-deploy containers that can run anywhere - from cloud platforms to on-premises data centers - while maintaining high performance, security, and reliability.

```
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA NIM Overview                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Complex AI Models  ──┐                                     │
│  (LLMs, VLMs, etc.)   │                                     │
│                       │                                     │
│  Inference Engines  ──┼──►  NIM Container  ──►  Deploy     │
│  (TensorRT, etc.)     │     (Optimized)         Anywhere!  │
│                       │                                     │
│  Enterprise Runtime ──┘                                     │
│  (Security, APIs)                                           │
│                                                             │
│  🚀 Minutes to deploy  │  🔒 Enterprise security           │
│  ⚡ High performance   │  🌍 Run anywhere                   │
└─────────────────────────────────────────────────────────────┘
```

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    NVIDIA NIM - At A Glance                           ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  What is it?                                                          ║
║  → Optimized AI model deployment in containers                       ║
║  → Pre-packaged inference engines + enterprise runtime               ║
║  → Standard APIs for easy integration                                ║
║                                                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐ ║
║  │                                                                 │ ║
║  │   Traditional: [Weeks of Setup] → [Manual Optimization]        │ ║
║  │                → [Custom APIs] → [Security DIY] → [Production] │ ║
║  │                                                                 │ ║
║  │   With NIM:    [Pull Container] → [Run] → [Production Ready!] │ ║
║  │                     (5 minutes)                                 │ ║
║  │                                                                 │ ║
║  └─────────────────────────────────────────────────────────────────┘ ║
║                                                                       ║
║  Key Numbers:                                                         ║
║  • 100+ Pre-optimized Models    • 5 Minute Deployment                ║
║  • 1000+ Tokens/sec Throughput  • 10-100ms Latency                   ║
║                                                                       ║
║  Deploy Anywhere:                                                     ║
║  [☁️  Cloud] [🏢 Data Center] [💻 Workstation] [📱 Edge]             ║
║                                                                       ║
║  Top Use Cases:                                                       ║
║  💬 Chatbots  📄 RAG  🎨 GenAI  🏥 Healthcare  🚗 Automotive         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Core Value Proposition

NIM helps organizations:
- **Reduce time-to-market** for AI applications
- **Simplify deployment** of complex foundation models
- **Maintain data security** and sovereignty
- **Achieve optimal performance** on NVIDIA-accelerated infrastructure
- **Get enterprise-grade support** and ongoing security updates

---

## Why NIM?

### Traditional Challenges in AI Deployment

Before NIM, deploying AI models to production involved:
- Complex infrastructure setup and optimization
- Manual model optimization for different hardware
- Security vulnerability management
- API stability concerns
- Lack of enterprise support
- Performance tuning challenges

### The NIM Solution

NIM addresses these challenges by providing:
- **Pre-optimized containers** with validated configurations
- **Industry-standard APIs** that work out-of-the-box
- **Continuous security updates** and vulnerability fixes
- **Enterprise-grade support** with SLAs
- **Hardware-accelerated inference** automatically configured
- **Flexible deployment options** for any infrastructure

```
┌───────────────────────────────────────────────────────────────────┐
│              Traditional AI Deployment vs NIM                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  TRADITIONAL APPROACH:                                            │
│  ┌──────────────────────────────────────────────────────┐        │
│  │ Weeks/Months of Work ────────────────────────────────┤        │
│  ├──────────────────────────────────────────────────────┤        │
│  │ 1. Setup Infrastructure        [████░░░░░] Complex   │        │
│  │ 2. Optimize Model              [████░░░░░] Manual    │        │
│  │ 3. Build APIs                  [████░░░░░] Custom    │        │
│  │ 4. Security Hardening          [████░░░░░] DIY       │        │
│  │ 5. Monitoring & Support        [████░░░░░] Limited   │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                   │
│  NIM APPROACH:                                                    │
│  ┌──────────────────────────────────────────────────────┐        │
│  │ Minutes to Production ────────────────────────────────┤        │
│  ├──────────────────────────────────────────────────────┤        │
│  │ 1. Pull Container              [██████████] ✓ Done   │        │
│  │ 2. Run Single Command          [██████████] ✓ Done   │        │
│  │ 3. Use Standard APIs           [██████████] ✓ Done   │        │
│  │ 4. Auto-Updates                [██████████] ✓ Done   │        │
│  │ 5. Enterprise Support          [██████████] ✓ Done   │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Core Architecture

NIM is packaged as a containerized inference service containing four essential components:

```
┌─────────────────────────────────────────────────────────────────┐
│                   NIM Container Architecture                    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Layer 1: Industry-Standard APIs              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐      │ │
│  │  │  REST API   │  │   gRPC      │  │  OpenAI API  │      │ │
│  │  └─────────────┘  └─────────────┘  └──────────────┘      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Layer 2: Domain-Specific Code                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │ Preprocessing│  │   Business   │  │Postprocessing│    │ │
│  │  │   Pipeline   │  │     Logic    │  │   Pipeline   │    │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Layer 3: Optimized Inference Engines              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │  TensorRT    │  │ TensorRT-LLM │  │    Triton    │    │ │
│  │  │   (DL Opt)   │  │   (LLM Opt)  │  │  (Serving)   │    │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                           ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │            Layer 4: Enterprise Runtime                    │ │
│  │  [Security] [Monitoring] [Logging] [Health Checks] [SLA] │ │
│  └───────────────────────────────────────────────────────────┘ │
│                           ▼                                     │
│                 ┌─────────────────────┐                         │
│                 │   NVIDIA GPU(s)     │                         │
│                 │  (A100, H100, RTX)  │                         │
│                 └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Industry-Standard APIs
- Domain-specific interfaces (REST, gRPC)
- OpenAI-compatible endpoints for LLMs
- Minimal integration code required
- Consistent API contracts across deployments

### 2. Domain-Specific Code

Layer 2 contains customizable business logic that sits between the API layer and the inference engine. This is where you can implement domain-specific transformations, validations, and custom workflows.

**What It Does:**
- Specialized implementations for different AI domains
- Optimized preprocessing and postprocessing pipelines
- Custom logic for specific model types (LLM, VLM, Speech, etc.)

**Architecture of Layer 2:**

```
┌──────────────────────────────────────────────────────────────────┐
│              Layer 2: Domain-Specific Code Flow                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request from Layer 1 (API)                                      │
│         │                                                         │
│         ↓                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ PREPROCESSING PIPELINE                                   │   │
│  │ ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │
│  │ │ Input      │→ │ Custom     │→ │ Format     │          │   │
│  │ │ Validation │  │ Transform  │  │ Conversion │          │   │
│  │ └────────────┘  └────────────┘  └────────────┘          │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ↓                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BUSINESS LOGIC                                           │   │
│  │ ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │
│  │ │ Prompt     │→ │ Safety     │→ │ Context    │          │   │
│  │ │ Engineering│  │ Checks     │  │ Management │          │   │
│  │ └────────────┘  └────────────┘  └────────────┘          │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ↓                                        │
│                  Layer 3 (Inference)                             │
│                         ↓                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ POSTPROCESSING PIPELINE                                  │   │
│  │ ┌────────────┐  ┌────────────┐  ┌────────────┐          │   │
│  │ │ Parse      │→ │ Filter     │→ │ Format     │          │   │
│  │ │ Output     │  │ & Sanitize │  │ Response   │          │   │
│  │ └────────────┘  └────────────┘  └────────────┘          │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ↓                                        │
│  Response to Layer 1 (API)                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Available Functionalities

**For complete implementation details, code examples, and configuration methods, see the expanded Layer 2 documentation at:** [NeMo/Layer2-DomainSpecificCode.md](Layer2-DomainSpecificCode.md)

**Quick Reference:**

| Component | Key Features | Configuration Method |
|-----------|-------------|---------------------|
| **Preprocessing** | • Input validation<br>• Tokenization<br>• Prompt templates<br>• Context injection<br>• PII masking | Env vars, YAML config, Python hooks |
| **Business Logic** | • Model routing<br>• Response caching<br>• Guardrails integration<br>• Rate limiting<br>• A/B testing | Python hooks, custom middleware |
| **Postprocessing** | • Output parsing<br>• PII removal<br>• Format conversion<br>• Metadata injection<br>• Citation management | YAML config, Python hooks |

**Common Use Cases:**

- Customer support with CRM integration
- Code generation with syntax validation
- RAG systems with citation tracking
- Multi-tenant apps with per-tenant rules
- Content moderation and safety
- Cost tracking and optimization

**Configuration Example:**

```yaml
# nim-domain-config.yaml
domain_specific:
  preprocessing:
    system_prompt: "You are a helpful AI assistant."
    max_tokens: 4096
    pii_masking: true

  business_logic:
    caching:
      enabled: true
      ttl: 3600
    guardrails:
      enabled: true
    rate_limiting:
      requests_per_minute: 60

  postprocessing:
    pii_removal: true
    output_format: markdown
    include_metadata: true
```

**Note:** Layer 2 is where NIM differs from generic inference servers - it provides domain-optimized pipelines out-of-the-box while allowing full customization for your specific use case.

### 3. Optimized Inference Engines
- **NVIDIA TensorRT** for deep learning inference
- **TensorRT-LLM** for large language models
- **Triton Inference Server** for multi-model serving
- Community-contributed engines
- Hardware-specific optimizations for maximum throughput and minimum latency

### 4. Enterprise Runtime
- Production-grade foundation with security hardening
- Continuous vulnerability scanning and patching
- Service-level agreements (SLAs)
- Monitoring and logging capabilities
- Health check endpoints

---

## Key Features & Capabilities

### 1. **Enterprise-Grade Security**
- Regular security updates and patches
- Data privacy and sovereignty protection
- Secure APIs with authentication options
- Compliance with enterprise security standards

### 2. **High Performance**
- Low-latency, high-throughput inference
- Optimized for NVIDIA GPU infrastructure
- Automatic batching and model optimization
- Efficient memory management

### 3. **Broad Model Support**
- 100+ pre-optimized models available
- Support for custom models
- Multi-framework compatibility (PyTorch, TensorFlow, JAX, etc.)

### 4. **Flexible Deployment**
- Cloud-agnostic (AWS, Azure, GCP, Oracle Cloud)
- On-premises data centers
- Edge devices and workstations
- Kubernetes-native with Helm charts

### 5. **Developer-Friendly**
- Simple container deployment (Docker/Podman)
- Standard API interfaces
- Comprehensive documentation
- Quick-start templates and examples
- 5-minute deployment guides

### 6. **Production-Ready**
- Stable APIs backed by SLAs
- Monitoring and observability built-in
- Autoscaling capabilities
- Load balancing support

---

## Supported Model Categories

NIM supports a wide range of AI model types across various domains:

```
┌────────────────────────────────────────────────────────────────────┐
│                    NIM Supported Model Types                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│   │     LLMs     │   │     VLMs     │   │    Speech    │         │
│   │   💬 Chat    │   │  🖼️  Vision  │   │   🎤 Audio   │         │
│   │   📝 Code    │   │  📊 Charts   │   │   🔊 TTS     │         │
│   └──────────────┘   └──────────────┘   └──────────────┘         │
│                                                                    │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│   │  Embeddings  │   │   Computer   │   │   Biology    │         │
│   │   🔍 RAG     │   │    Vision    │   │  🧬 Proteins │         │
│   │   📌 Search  │   │  🎨 GenAI    │   │  💊 Drugs    │         │
│   └──────────────┘   └──────────────┘   └──────────────┘         │
│                                                                    │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│   │   Climate    │   │   Digital    │   │    Safety    │         │
│   │  🌍 Weather  │   │    Humans    │   │  🛡️ Guard   │         │
│   │  🌊 Ocean    │   │  👤 Avatars  │   │  ⚠️ Moderate│         │
│   └──────────────┘   └──────────────┘   └──────────────┘         │
│                                                                    │
│                  100+ Pre-Optimized Models Available               │
└────────────────────────────────────────────────────────────────────┘
```

### 1. **Large Language Models (LLMs)**
- Text generation and completion
- Question answering and chat
- Code generation
- Summarization and translation
- Examples: Llama, Mistral, GPT models, Nemotron

### 2. **Vision Language Models (VLMs)**
- Image understanding and captioning
- Visual question answering
- Multimodal reasoning
- Examples: CLIP, BLIP, LLaVA

### 3. **Speech & Audio**
- Automatic Speech Recognition (ASR) via **NVIDIA Riva**
- Text-to-Speech (TTS)
- Audio-to-Face animation (**NVIDIA Maxine**)
- Speech translation

### 4. **Computer Vision**
- Image generation and editing
- Object detection and segmentation
- Video analysis and generation
- 3D modeling

### 5. **Retrieval & Embeddings**
- Text embedding models (**NeMo Retriever**)
- Reranking services
- Vector search optimization

### 6. **Specialized Domains**

#### **Biology & Healthcare**
- **BioNeMo models**: Protein folding, molecular generation
- Medical imaging (**MAISI**, **VISTA-3D**)
- Drug discovery applications

#### **Climate & Weather**
- Earth science simulations
- Weather prediction models

#### **Digital Humans**
- Real-time avatar animation
- Conversational AI with visual representation

#### **Safety & Moderation**
- **NemoGuard**: Content safety and moderation
- Guardrails for responsible AI

---

## Deployment Options

```
┌─────────────────────────────────────────────────────────────────────┐
│                   NIM Deployment Flexibility                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Option 1: SERVERLESS CLOUD                                         │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  🌐 build.nvidia.com                                       │     │
│  │  ✓ Free for development                                   │     │
│  │  ✓ No infrastructure setup                                │     │
│  │  ✓ Instant access to 100+ models                          │     │
│  │  ✓ Powered by DGX Cloud                                   │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Option 2: CLOUD PLATFORMS                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐      │
│  │    AWS     │ │   Azure    │ │    GCP     │ │  Oracle CI │      │
│  │            │ │            │ │            │ │            │      │
│  │  Deploy    │ │  Deploy    │ │  Deploy    │ │  Deploy    │      │
│  │  NIM       │ │  NIM       │ │  NIM       │ │  NIM       │      │
│  │  Anywhere  │ │  Anywhere  │ │  Anywhere  │ │  Anywhere  │      │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘      │
│                                                                     │
│  Option 3: ON-PREMISES / PRIVATE CLOUD                              │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  🏢 Your Data Center                                       │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │     │
│  │  │ DGX A100 │  │ DGX H100 │  │  Custom  │                │     │
│  │  │  System  │  │  System  │  │  Servers │                │     │
│  │  └──────────┘  └──────────┘  └──────────┘                │     │
│  │  ✓ Complete data sovereignty                              │     │
│  │  ✓ No data leaves your environment                        │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Option 4: KUBERNETES                                               │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  ☸  Kubernetes Cluster                                     │     │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                  │     │
│  │  │ Pod  │  │ Pod  │  │ Pod  │  │ Pod  │  Auto-scaling    │     │
│  │  │ NIM  │  │ NIM  │  │ NIM  │  │ NIM  │  Load Balancing  │     │
│  │  └──────┘  └──────┘  └──────┘  └──────┘                  │     │
│  │  ✓ NIM Operator   ✓ Helm Charts   ✓ Multi-node          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Option 5: WORKSTATIONS & EDGE                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  💻 NVIDIA RTX Workstations                                │     │
│  │  📱 Edge Devices with NVIDIA GPUs                          │     │
│  │  🪟 Windows Subsystem for Linux (WSL)                      │     │
│  │  ✓ Local development   ✓ Edge AI deployment              │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deploying NIM with Deployment Management

NVIDIA provides a Deployment Management Service that simplifies deploying and managing NIM microservices at scale. This is especially useful for enterprise deployments requiring centralized orchestration.

### What is Deployment Management?

The Deployment Management Service is a centralized platform that orchestrates multiple NIM microservices across Kubernetes clusters.

**Key Capabilities:**
- Orchestrates NIM deployments across clusters
- Manages model-to-NIM mappings automatically
- Provides RESTful APIs and Python SDK for automation
- Tracks deployment status and health
- Supports both synchronous and asynchronous deployments

**Important Distinctions:**

| Component | URL Type | Purpose | Maps To |
|-----------|----------|---------|---------|
| **Deployment Management Service** | `DEPLOYMENT_BASE_URL` | Orchestrate/manage NIMs | Management pods (NOT NIM) |
| **NIM Microservice** | `deployment.url` | Run inference | NIM pods/containers (actual AI) |

```
DEPLOYMENT_BASE_URL (Control Plane)
        ↓
┌─────────────────────────────────────────────┐
│ Deployment Management Service Pod          │
│ (Kubernetes Service)                        │
└──────────────────┬──────────────────────────┘
                   │ Creates & manages
                   ↓
┌──────────────────────────────────────────────────────────┐
│           Multiple NIM Deployments (Data Plane)          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ NIM 1: Llama-3-8B              NIM 2: Embedding Model    │
│ ┌────────────────────┐         ┌────────────────────┐   │
│ │ K8s Service        │         │ K8s Service        │   │
│ │ nim-llama-svc:8000 │         │ nim-embed-svc:8000 │   │
│ └────┬───────────────┘         └────┬───────────────┘   │
│      ↓                              ↓                    │
│ ┌─────────┐ ┌─────────┐       ┌─────────┐               │
│ │NIM Pod 1│ │NIM Pod 2│       │NIM Pod 1│               │
│ │(Docker) │ │(Docker) │       │(Docker) │               │
│ └─────────┘ └─────────┘       └─────────┘               │
│                                                          │
│ NIM 3: Llama-3-70B             NIM 4: NeMo Guard         │
│ ┌────────────────────┐         ┌────────────────────┐   │
│ │ K8s Service        │         │ K8s Service        │   │
│ │ nim-70b-svc:8000   │         │ nim-guard-svc:8000 │   │
│ └────┬───────────────┘         └────┬───────────────┘   │
│      ↓                              ↓                    │
│ ┌─────────┐ ┌─────────┐       ┌─────────┐               │
│ │NIM Pod 1│ │NIM Pod 2│       │NIM Pod 1│               │
│ │(Docker) │ │(Docker) │       │(Docker) │               │
│ └─────────┘ └─────────┘       └─────────┘               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**How It Works:**

1. **You call `DEPLOYMENT_BASE_URL`** → Connects to Deployment Management Service (orchestrator)
2. **Management Service creates** → Kubernetes Deployment + Service for NIM
3. **Kubernetes creates** → NIM Pods (each pod = Docker container with NIM)
4. **Kubernetes Service** → Load balances traffic across NIM pods
5. **You get `deployment.url`** → Points to Kubernetes Service for inference

**Can 1 Deployment Management Orchestrate Multiple NIMs?**

✅ **YES! One Deployment Management Service manages MANY NIM microservices:**

| Aspect | Details |
|--------|---------|
| **Management Service** | 1 instance (centralized orchestrator) |
| **NIM Deployments** | Multiple (as many as needed) |
| **Models per Deployment** | Can deploy single or multiple models per NIM |
| **Example** | 1 Management Service → 100+ different NIM deployments |

**Example: Multiple NIMs from One Management Service**

```python
from nemo_microservices import NeMoMicroservices

# One management client
client = NeMoMicroservices(
    base_url="https://deployment-service.com",  # ← Single DEPLOYMENT_BASE_URL
    api_key="your-api-key"
)

# Deploy multiple different NIMs
llama8b = client.deployments.create(
    name="llama-8b-prod",
    models=["meta/llama3-8b-instruct"]
)

llama70b = client.deployments.create(
    name="llama-70b-prod",
    models=["meta/llama3-70b-instruct"]
)

embeddings = client.deployments.create(
    name="embeddings-prod",
    models=["nvidia/embed-qa-4"]
)

guardrails = client.deployments.create(
    name="guard-prod",
    models=["nvidia/nemoguard"]
)

# Each deployment gets its own endpoint
print(llama8b.url)      # http://nim-llama8b-svc:8000
print(llama70b.url)     # http://nim-llama70b-svc:8000
print(embeddings.url)   # http://nim-embed-svc:8000
print(guardrails.url)   # http://nim-guard-svc:8000

# One management service → 4 separate NIM microservices
```

**Summary:**
- **`DEPLOYMENT_BASE_URL`** → Management Service (orchestrator, not NIM)
- **`deployment.url`** → Kubernetes Service → NIM Pods → Docker Containers
- **One Management Service** → Manages **many NIM deployments**
- Each NIM deployment → Kubernetes Service + multiple Pods/Containers

---

## How is Deployment Management Service Hosted in Enterprises?

The Deployment Management Service itself must be deployed before you can use it to manage NIMs. Enterprises have three main options:

### Deployment Options

```
┌────────────────────────────────────────────────────────────────┐
│         Deployment Management Service Hosting Options          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Option 1: NVIDIA-Hosted SaaS (Managed)                        │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  ☁️  NVIDIA's Cloud Infrastructure                    │     │
│  │  • Fully managed by NVIDIA                           │     │
│  │  • No installation required                          │     │
│  │  • Access via API key                                │     │
│  │  • Suitable for: Quick start, small teams            │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Option 2: Part of NeMo Microservices Platform (On-Prem)      │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  🏢 Your Kubernetes Cluster                           │     │
│  │  ┌────────────────────────────────────────────┐      │     │
│  │  │ NeMo Microservices Platform (Helm Chart)  │      │     │
│  │  │ ┌──────────────────────────────────────┐  │      │     │
│  │  │ │ Deployment Management Service        │  │      │     │
│  │  │ │ NeMo Retriever                       │  │      │     │
│  │  │ │ NeMo Evaluator                       │  │      │     │
│  │  │ │ Other NeMo Services                  │  │      │     │
│  │  │ └──────────────────────────────────────┘  │      │     │
│  │  └────────────────────────────────────────────┘      │     │
│  │  • Complete platform installation                    │     │
│  │  • Suitable for: Full NeMo ecosystem users           │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  Option 3: Standalone Installation (On-Prem)                  │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  🏢 Your Kubernetes Cluster                           │     │
│  │  ┌────────────────────────────────────────────┐      │     │
│  │  │ Just Deployment Management Service         │      │     │
│  │  │ (Helm Chart or Docker Compose)             │      │     │
│  │  └────────────────────────────────────────────┘      │     │
│  │  • Lightweight, only management service              │     │
│  │  • Suitable for: NIM-only deployments                │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Option 1: NVIDIA-Hosted SaaS (Easiest)

**What it is:** NVIDIA hosts the Deployment Management Service in their cloud.

**How to access:**
```bash
# Simply use NVIDIA's hosted endpoint
export DEPLOYMENT_BASE_URL="https://api.nvidia.com/nemo/deployments"
export API_KEY="your-nvidia-api-key"

# No installation needed!
```

**Pros & Cons:**

| Pros | Cons |
|------|------|
| ✅ Zero setup | ❌ Requires internet connectivity |
| ✅ Always up-to-date | ❌ Data leaves your network |
| ✅ Managed by NVIDIA | ❌ Less control |
| ✅ Quick start | ❌ May have rate limits |

### Option 2: NeMo Microservices Platform (Full Suite)

**What it is:** Deploy the complete NeMo platform (includes Deployment Management + other services).

**Installation Steps:**

```bash
# 1. Add NeMo Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# 2. Create namespace
kubectl create namespace nemo-platform

# 3. Create NGC API key secret
kubectl create secret generic ngc-api-secret \
  --from-literal=NGC_API_KEY=your-ngc-api-key \
  -n nemo-platform

# 4. Install NeMo Microservices Platform
helm install nemo-platform nvidia/nemo-microservices \
  --namespace nemo-platform \
  --set global.ngcApiKey=your-ngc-api-key \
  --set deploymentManagement.enabled=true \
  --set deploymentManagement.replicas=3 \
  --set deploymentManagement.resources.limits.cpu=4 \
  --set deploymentManagement.resources.limits.memory=8Gi

# 5. Wait for deployment
kubectl wait --for=condition=ready pod \
  -l app=deployment-management \
  -n nemo-platform \
  --timeout=300s

# 6. Get the service endpoint
kubectl get svc deployment-management-service -n nemo-platform
```

**What gets deployed:**

```yaml
# Deployment Management Service components
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-management
  namespace: nemo-platform
spec:
  replicas: 3  # HA deployment
  selector:
    matchLabels:
      app: deployment-management
  template:
    metadata:
      labels:
        app: deployment-management
    spec:
      containers:
      - name: deployment-service
        image: nvcr.io/nvidia/nemo/deployment-management:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:5432/deployments"
        - name: KUBERNETES_NAMESPACE
          value: "production"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"

---
apiVersion: v1
kind: Service
metadata:
  name: deployment-management-service
  namespace: nemo-platform
spec:
  type: LoadBalancer
  selector:
    app: deployment-management
  ports:
  - port: 443
    targetPort: 8080
```

**Infrastructure Requirements:**

| Component | Requirement |
|-----------|-------------|
| **Kubernetes** | v1.24+ |
| **CPU** | 2-4 cores per replica |
| **Memory** | 4-8 GB per replica |
| **Storage** | 50-100 GB (for metadata DB) |
| **Database** | PostgreSQL 13+ (for deployment state) |
| **Replicas** | 3 (for HA) |

### Option 3: Standalone Installation (Lightweight)

**What it is:** Deploy only the Deployment Management Service (not full NeMo platform).

**Method A: Helm Chart (Kubernetes)**

```bash
# 1. Create namespace
kubectl create namespace deployment-mgmt

# 2. Install standalone deployment management
helm install deployment-mgmt nvidia/deployment-management \
  --namespace deployment-mgmt \
  --set replicas=3 \
  --set ingress.enabled=true \
  --set ingress.hostname=deployment.yourcompany.com \
  --set database.type=postgresql \
  --set database.host=postgres.yourcompany.com

# 3. Access the service
export DEPLOYMENT_BASE_URL="https://deployment.yourcompany.com"
```

**Method B: Docker Compose (Simple Setup)**

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: deployments
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secure-password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  deployment-management:
    image: nvcr.io/nvidia/nemo/deployment-management:latest
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://admin:secure-password@postgres:5432/deployments
      KUBERNETES_CONFIG: /root/.kube/config
      LOG_LEVEL: INFO
    volumes:
      - ~/.kube:/root/.kube:ro
    depends_on:
      - postgres
    deploy:
      replicas: 2

  nginx:
    image: nginx:latest
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - deployment-management

volumes:
  postgres-data:
```

```bash
# Start services
docker-compose up -d

# Access
export DEPLOYMENT_BASE_URL="https://localhost"
```

### Enterprise Architecture Example

```
┌─────────────────────────────────────────────────────────────────┐
│              Enterprise Deployment Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ External Access (via Ingress/Load Balancer)            │    │
│  │ https://deployment-api.company.com                     │    │
│  └──────────────────────┬─────────────────────────────────┘    │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Kubernetes Cluster: Management                          │   │
│  │                                                         │   │
│  │  ┌──────────────────────────────────────────────┐      │   │
│  │  │ Deployment Management Service (HA)           │      │   │
│  │  │ ┌────────┐  ┌────────┐  ┌────────┐          │      │   │
│  │  │ │ Pod 1  │  │ Pod 2  │  │ Pod 3  │          │      │   │
│  │  │ └────────┘  └────────┘  └────────┘          │      │   │
│  │  │         LoadBalanced via Service            │      │   │
│  │  └──────────────────┬───────────────────────────┘      │   │
│  │                     │                                   │   │
│  │                     ↓ Stores state                      │   │
│  │  ┌──────────────────────────────────┐                  │   │
│  │  │ PostgreSQL Database (Primary)    │                  │   │
│  │  │ + Read Replicas                  │                  │   │
│  │  └──────────────────────────────────┘                  │   │
│  │                     │                                   │   │
│  └─────────────────────┼───────────────────────────────────┘   │
│                        │                                       │
│                        │ Creates/manages NIMs                  │
│                        ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Kubernetes Cluster: Production (NIMs)                  │   │
│  │                                                         │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐          │   │
│  │  │ NIM       │  │ NIM       │  │ NIM       │          │   │
│  │  │ Llama-8B  │  │ Llama-70B │  │ Embed     │  ...     │   │
│  │  └───────────┘  └───────────┘  └───────────┘          │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### High Availability Configuration

For enterprise production deployments:

```yaml
# High-availability setup
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-management
spec:
  replicas: 3  # Minimum 3 for HA
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1

  template:
    spec:
      affinity:
        # Spread across availability zones
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - deployment-management
            topologyKey: topology.kubernetes.io/zone

      containers:
      - name: deployment-service
        image: nvcr.io/nvidia/nemo/deployment-management:latest
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Access & Security

**1. Authentication Options:**

```yaml
# Option A: API Key
env:
- name: AUTH_METHOD
  value: "api_key"
- name: API_KEYS_SECRET
  valueFrom:
    secretKeyRef:
      name: deployment-api-keys
      key: keys

# Option B: OAuth2/OIDC
env:
- name: AUTH_METHOD
  value: "oidc"
- name: OIDC_ISSUER
  value: "https://auth.yourcompany.com"
- name: OIDC_CLIENT_ID
  value: "deployment-mgmt-client"
```

**2. Network Access:**

```yaml
# Internal only (within cluster)
apiVersion: v1
kind: Service
spec:
  type: ClusterIP  # Only accessible within K8s cluster

# External access (via Ingress)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/whitelist-source-range: "10.0.0.0/8"
spec:
  tls:
  - hosts:
    - deployment-api.company.com
    secretName: deployment-tls
  rules:
  - host: deployment-api.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: deployment-management-service
            port:
              number: 8080
```

### Comparison Table

| Aspect | SaaS (NVIDIA) | NeMo Platform | Standalone |
|--------|---------------|---------------|------------|
| **Setup Time** | Minutes | 1-2 hours | 30 mins |
| **Control** | Low | High | High |
| **Data Privacy** | External | On-prem | On-prem |
| **Cost** | Pay-per-use | Infrastructure | Infrastructure |
| **Updates** | Automatic | Manual | Manual |
| **Best For** | POC, small teams | Full NeMo users | NIM-only users |
| **HA Built-in** | Yes | Configure | Configure |

### Summary

**Deployment Management Service is hosted as:**

1. ✅ **SaaS** - NVIDIA-hosted, zero setup
2. ✅ **Kubernetes pods** - Part of NeMo platform or standalone
3. ✅ **Docker containers** - Simple Docker Compose setup

**Enterprise typically chooses:**
- **On-premises Kubernetes** with 3+ replicas for HA
- **PostgreSQL database** for storing deployment state
- **Ingress/Load Balancer** for external access
- **Multi-zone deployment** for resilience

**The service itself is just another Kubernetes application** that orchestrates NIM deployments!

---

### Prerequisites

Before deploying via Deployment Management, ensure you have:

1. **Deployment Management Service Access**
   - Deployed as part of NeMo Microservices platform, or
   - Installed independently
   - Base URL stored in `DEPLOYMENT_BASE_URL` environment variable

2. **Authentication**
   - API credentials or token
   - Appropriate permissions for deployment operations

3. **Model Information**
   - Know which models you want to deploy
   - Understand model-to-NIM compatibility mappings
   - Review the Cross-service Compatibility documentation

4. **Target Infrastructure**
   - Kubernetes cluster with GPU Operator installed
   - Sufficient GPU resources for models
   - Network connectivity to Deployment Management Service

### Deployment Methods

#### Method 1: Python SDK (Recommended)

```python
from nemo_microservices import NeMoMicroservices

# Initialize client
client = NeMoMicroservices(
    base_url="https://your-deployment-service.com",
    api_key="your-api-key"
)

# Deploy a NIM microservice
deployment = client.deployments.create(
    name="llama3-8b-production",
    namespace="production",
    models=["meta/llama3-8b-instruct"],
    config="optimized",  # Configuration profile
    project="chatbot-app",
    async_enabled=True,  # Enable async processing
    ownership={
        "creator": "user@company.com",
        "team": "ai-platform"
    }
)

print(f"Deployment created: {deployment.name}")
print(f"Status: {deployment.status}")
print(f"Endpoint: {deployment.url}")

# Monitor deployment status
status = client.deployments.get(deployment.id)
print(f"Current status: {status.status}")
```

#### Method 2: REST API (cURL)

```bash
# Set environment variables
export DEPLOYMENT_BASE_URL="https://your-deployment-service.com"
export API_KEY="your-api-key"

# Create deployment using REST API
curl -X POST "${DEPLOYMENT_BASE_URL}/v1/deployments" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama3-8b-production",
    "namespace": "production",
    "models": ["meta/llama3-8b-instruct"],
    "config": "optimized",
    "project": "chatbot-app",
    "async_enabled": true,
    "ownership": {
      "creator": "user@company.com",
      "team": "ai-platform"
    }
  }'
```

### Key Configuration Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `name` | string | Unique deployment identifier | Yes |
| `namespace` | string | Kubernetes namespace for deployment | Yes |
| `models` | array | List of models to deploy (e.g., ["meta/llama3-8b"]) | Yes |
| `config` | string | Configuration profile (optimized, balanced, low-latency) | No |
| `project` | string | Associated project name for organization | No |
| `async_enabled` | boolean | Enable asynchronous processing | No (default: false) |
| `ownership` | object | Creator and team information for access control | No |

### Configuration Profiles

The Deployment Management Service supports pre-defined configuration profiles:

| Profile | Description | Best For | Trade-offs |
|---------|-------------|----------|------------|
| **optimized** | Maximum throughput settings | Batch processing, high load | Slightly higher latency |
| **balanced** | Default settings | General-purpose applications | Balanced latency/throughput |
| **low-latency** | Minimum response time | Real-time chat, interactive apps | Lower throughput |
| **custom** | User-defined configuration | Specific requirements | Requires tuning expertise |

### Monitoring Deployment Progress

Deployments typically take a few minutes to complete. Monitor status using:

```python
# Python SDK
import time

deployment_id = "your-deployment-id"

while True:
    status = client.deployments.get(deployment_id)

    print(f"Status: {status.status}")
    print(f"Progress: {status.progress}%")

    if status.status == "RUNNING":
        print(f"✓ Deployment ready at: {status.url}")
        break
    elif status.status == "FAILED":
        print(f"✗ Deployment failed: {status.error_message}")
        break

    time.sleep(10)  # Check every 10 seconds
```

```bash
# REST API
curl -X GET "${DEPLOYMENT_BASE_URL}/v1/deployments/${DEPLOYMENT_ID}" \
  -H "Authorization: Bearer ${API_KEY}"
```

### Deployment States

| State | Description | Next Steps |
|-------|-------------|------------|
| **CREATING** | Initial deployment being provisioned | Wait for completion |
| **PENDING** | Waiting for resources | Check cluster capacity |
| **PULLING** | Downloading container images | Wait (can take minutes) |
| **RUNNING** | Deployment active and serving | Ready to use |
| **FAILED** | Deployment encountered error | Check logs, retry |
| **DELETING** | Deployment being removed | Wait for cleanup |

### Managing Deployments

#### List All Deployments

```python
# Python SDK
deployments = client.deployments.list(namespace="production")

for dep in deployments:
    print(f"{dep.name}: {dep.status} - {dep.url}")
```

```bash
# REST API
curl -X GET "${DEPLOYMENT_BASE_URL}/v1/deployments?namespace=production" \
  -H "Authorization: Bearer ${API_KEY}"
```

#### Update Deployment

```python
# Python SDK
client.deployments.update(
    deployment_id="your-deployment-id",
    config="low-latency",  # Switch to different profile
    replicas=3  # Scale to 3 instances
)
```

#### Delete Deployment

```python
# Python SDK
client.deployments.delete("your-deployment-id")
```

```bash
# REST API
curl -X DELETE "${DEPLOYMENT_BASE_URL}/v1/deployments/${DEPLOYMENT_ID}" \
  -H "Authorization: Bearer ${API_KEY}"
```

### Best Practices

| Practice | Why | How |
|----------|-----|-----|
| **Use namespaces** | Organize by environment | dev, staging, production |
| **Set ownership metadata** | Track who owns what | Include creator, team info |
| **Enable async for large models** | Prevent timeout issues | Set `async_enabled: true` |
| **Monitor deployment status** | Catch failures early | Poll status endpoint |
| **Use configuration profiles** | Optimize for use case | Start with balanced, tune later |
| **Name deployments clearly** | Easy identification | Use format: model-env-purpose |

### Common Deployment Patterns

#### Pattern 1: Blue-Green Deployment

```python
# Deploy new version (green)
green = client.deployments.create(
    name="llama3-8b-green",
    namespace="production",
    models=["meta/llama3-8b-instruct:v2.0"]
)

# Test green deployment
# If successful, switch traffic and delete blue

# Delete old version (blue)
client.deployments.delete("llama3-8b-blue")
```

#### Pattern 2: Multi-Model Deployment

```python
# Deploy multiple models in one deployment
deployment = client.deployments.create(
    name="multimodal-service",
    namespace="production",
    models=[
        "meta/llama3-8b-instruct",  # LLM
        "nvidia/embed-qa-4",         # Embeddings
        "nvidia/nemoguard"           # Safety
    ],
    config="optimized"
)
```

#### Pattern 3: Environment Progression

```python
# 1. Deploy to dev
dev = client.deployments.create(
    name="llama3-8b", namespace="dev", config="balanced"
)

# 2. Test in dev, then promote to staging
staging = client.deployments.create(
    name="llama3-8b", namespace="staging", config="balanced"
)

# 3. Final promotion to production
prod = client.deployments.create(
    name="llama3-8b", namespace="production", config="optimized"
)
```

### Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| **Deployment stuck in PENDING** | Insufficient GPU resources | Check cluster capacity with `kubectl describe nodes` |
| **Deployment FAILED** | Invalid model name or config | Verify model compatibility, check logs |
| **Slow deployment** | Large model download | Normal for first deployment, subsequent deploys are faster |
| **Connection timeout** | Network issues | Check `DEPLOYMENT_BASE_URL`, verify API credentials |
| **401 Unauthorized** | Invalid API key | Refresh credentials, check permissions |

### Example: Complete Deployment Workflow

```python
from nemo_microservices import NeMoMicroservices
import time

# Initialize client
client = NeMoMicroservices(
    base_url="https://deployment.example.com",
    api_key="your-api-key"
)

try:
    # Create deployment
    print("Creating deployment...")
    deployment = client.deployments.create(
        name="chatbot-llm",
        namespace="production",
        models=["meta/llama3-70b-instruct"],
        config="optimized",
        async_enabled=True,
        project="customer-service"
    )

    print(f"Deployment ID: {deployment.id}")

    # Wait for deployment to be ready
    print("Waiting for deployment...")
    while deployment.status != "RUNNING":
        time.sleep(15)
        deployment = client.deployments.get(deployment.id)
        print(f"Status: {deployment.status}")

        if deployment.status == "FAILED":
            raise Exception(f"Deployment failed: {deployment.error_message}")

    # Deployment ready
    print(f"✓ Deployment ready!")
    print(f"Endpoint: {deployment.url}")

    # Test the deployment
    response = client.inference.chat_completion(
        deployment_url=deployment.url,
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    print(f"Test response: {response.choices[0].message.content}")

except Exception as e:
    print(f"Error: {e}")
```

### Summary

**Deployment Management Service provides:**
- ✅ Centralized orchestration for NIM deployments
- ✅ Python SDK and REST API for automation
- ✅ Configuration profiles for different use cases
- ✅ Status tracking and health monitoring
- ✅ Multi-model and multi-environment support

**Key Takeaways:**
1. Use Python SDK for programmatic deployments
2. Monitor deployment status - deployments take a few minutes
3. Organize using namespaces (dev, staging, prod)
4. Start with pre-defined config profiles
5. Check Cross-service Compatibility docs for model-to-NIM mappings

---

## Inference Engine vs Serving Platform

Two key concepts for AI deployment:

### Core Concepts

| Component | Inference Engine | Serving Platform |
|-----------|------------------|------------------|
| **Purpose** | Runs the model, produces predictions | Provides APIs and manages production deployment |
| **Scope** | Just computation (the "brain") | Full infrastructure (the "body") |
| **What it does** | • Loads model to GPU/CPU<br>• Runs neural network<br>• Optimizes computation | • HTTP/gRPC APIs<br>• Load balancing, batching<br>• Monitoring, security, scaling |
| **Input/Output** | Raw tensors/data | HTTP requests → JSON responses |
| **Production Ready** | No (computation only) | Yes (complete system) |
| **Examples** | TensorRT-LLM, PyTorch, ONNX Runtime, vLLM core | Triton, TorchServe, TensorFlow Serving, vLLM API |

### How They Work Together

```
Client → Serving Platform → Inference Engine → GPU
         (manages requests)  (runs model)
```

**Flow**: HTTP request → Serving platform queues/batches → Inference engine computes → Serving platform returns JSON

### Common Architectures

| Stack | Inference Engine | Serving Platform | Approach |
|-------|------------------|------------------|----------|
| **NVIDIA** | TensorRT-LLM | Triton | Separate (max flexibility) |
| **vLLM** | vLLM engine | vLLM API server | Combined (easy setup) |
| **Custom** | PyTorch/Any | FastAPI/Flask | DIY (full control) |

### Analogy
- **Inference Engine** = Restaurant kitchen (prepares food)
- **Serving Platform** = Front-of-house (takes orders, serves customers, manages payments)

**You need BOTH for production AI systems!**

---

## TensorRT vs TensorRT-LLM vs Triton

Understanding these three NVIDIA technologies is essential for working with NIM:

| Technology | Type | Purpose | Key Features | Use Cases | Analogy |
|------------|------|---------|--------------|-----------|---------|
| **TensorRT** | Inference Optimizer | Speed optimizer for any deep learning model | • Compresses & fuses operations<br>• Specialized GPU instructions<br>• Works with vision, speech, any DL model<br>• FP16/INT8 precision | Non-LLM models (ResNet, YOLO, Object Detection) | Turbocharger for any car |
| **TensorRT-LLM** | LLM Inference Engine | Specialized optimizer built ON TOP of TensorRT for LLMs | • In-flight batching<br>• KV cache management<br>• Multi-GPU support<br>• Quantization (INT8, FP8, INT4)<br>• For GPT, Llama, Mistral, etc. | Deploying LLMs with maximum performance and low latency | Turbocharger specifically designed for trucks |
| **Triton Inference Server** | Serving Platform | Production model deployment & serving | • HTTP/gRPC APIs<br>• Dynamic batching<br>• Load balancing across GPUs<br>• Model versioning<br>• Monitoring & metrics<br>• Works with TensorRT, PyTorch, TF, ONNX | Production API serving multiple models with any backend | Service center managing multiple vehicles |

### How They Work Together

**Typical Production Flow:**
```
1. Train LLM (Llama, GPT, etc.)
         ↓
2. Optimize with TensorRT-LLM (creates optimized engine)
         ↓
3. Deploy using Triton Server (serves via APIs)
         ↓
4. Applications call API → Fast responses
```

### When to Use What

| Scenario | Tools to Use | Why |
|----------|--------------|-----|
| **Non-LLM inference optimization** | TensorRT alone | Optimizes vision, speech, or other DL models |
| **LLM deployment** | TensorRT-LLM | LLM-specific optimizations (KV cache, batching) |
| **Production serving** | Triton | Provides APIs, load balancing, monitoring |
| **Production LLM (optimal)** | TensorRT-LLM + Triton | Maximum performance + production features |

### Performance Example

| Scenario | Before Optimization | With TensorRT/TensorRT-LLM |
|----------|---------------------|----------------------------|
| Image Classification | 100ms | 10ms (10x faster) |
| LLM Text Generation | 5 tokens/sec | 50+ tokens/sec (10x faster) |
| Multi-model Serving | Manual load balancing | Auto-batching + GPU balancing |

### In NIM Context

**NIM containers use TensorRT-LLM (for LLMs) and Triton (for serving) under the hood**, providing you with:
- Pre-optimized inference engines
- Production-ready APIs
- No manual configuration required
- Enterprise-grade performance out of the box

---

## TensorRT-LLM/Triton vs vLLM

**vLLM** is another popular LLM inference solution. Key comparison:

| Aspect | vLLM | TensorRT-LLM + Triton |
|--------|------|----------------------|
| **Type** | All-in-one (inference + serving) | Separate (inference engine + serving platform) |
| **Developer** | UC Berkeley (Open Source) | NVIDIA (Enterprise) |
| **License** | Apache 2.0 (Free) | NVIDIA AI Enterprise (Commercial) |
| **Setup** | Single command, very easy | Complex, requires integration |
| **Hardware** | NVIDIA, AMD, CPU | NVIDIA GPUs only |
| **Serving** | Built-in OpenAI-compatible API | Requires Triton |
| **KV Cache** | PagedAttention | Custom optimization |
| **Batching** | Continuous batching | In-flight batching |
| **Performance** | ~80-90% peak (excellent) | ~90-95% peak on NVIDIA (best-in-class) |
| **Memory** | Excellent (PagedAttention) | Excellent (optimized) |
| **Cost** | Free | License required |

### When to Use Each

| Choose vLLM | Choose TensorRT-LLM + Triton |
|-------------|------------------------------|
| ✅ Quick prototyping | ✅ Maximum NVIDIA GPU performance |
| ✅ Open-source requirement | ✅ Enterprise support & SLAs |
| ✅ AMD GPUs or CPU fallback | ✅ Production at scale |
| ✅ Easy setup needed | ✅ Multi-model serving features |
| ✅ Startups, research | ✅ NVIDIA AI Enterprise stack |

### Can They Work Together?

**Yes!** Triton can use vLLM as a backend, giving you vLLM's inference + Triton's production features (best of both worlds).

### Quick Examples

```bash
# vLLM (5 minutes)
pip install vllm
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B

# NIM (TensorRT-LLM + Triton pre-configured)
docker run --gpus all -p 8000:8000 nvcr.io/nim/meta/llama3-8b-instruct:latest
```

**Bottom Line**: vLLM = Swiss Army knife (versatile, easy). TensorRT-LLM = Racing engine (max performance, expertise needed).

---

## NIM Operator vs GPU Operator

**No, they are different Kubernetes operators with different purposes:**

| Aspect | NVIDIA GPU Operator | NIM Operator |
|--------|---------------------|--------------|
| **Purpose** | Manages GPU infrastructure in Kubernetes | Manages NIM deployments in Kubernetes |
| **What it does** | • Installs GPU drivers<br>• Deploys device plugins<br>• Configures GPU monitoring<br>• Manages GPU resources<br>• Handles GPU scheduling | • Deploys NIM containers<br>• Manages model serving<br>• Handles auto-scaling<br>• Model version management<br>• Load balancing |
| **Scope** | Infrastructure layer (GPU hardware) | Application layer (AI model serving) |
| **When to use** | First - to enable GPUs in K8s | After GPU Operator - to deploy NIMs |
| **Analogy** | Electrician (installs power infrastructure) | Chef (uses that power to run kitchen equipment) |
| **Required for** | Any GPU workload in Kubernetes | NIM deployments specifically |

### How They Work Together

```
┌─────────────────────────────────────────────────────┐
│           Kubernetes Cluster                        │
│                                                     │
│  Step 1: NVIDIA GPU Operator                        │
│  ┌───────────────────────────────────────────────┐  │
│  │  ✓ Installs GPU drivers                       │  │
│  │  ✓ Configures GPU device plugins              │  │
│  │  ✓ Sets up GPU monitoring (DCGM)              │  │
│  │  → Makes GPUs available to pods               │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                              │
│  Step 2: NIM Operator                               │
│  ┌───────────────────────────────────────────────┐  │
│  │  ✓ Deploys NIM containers                     │  │
│  │  ✓ Requests GPUs from GPU Operator            │  │
│  │  ✓ Manages model lifecycle                    │  │
│  │  ✓ Handles scaling & load balancing           │  │
│  │  → Runs AI inference workloads                │  │
│  └───────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Typical Deployment Order

```bash
# 1. Install GPU Operator (enables GPU support)
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace

# 2. Install NIM Operator (deploys NIM workloads)
helm install nim-operator nvidia/nim-operator \
  --namespace nim-operator --create-namespace

# 3. Deploy your NIM
kubectl apply -f llama-nim.yaml
```

### Key Differences

| Question | GPU Operator | NIM Operator |
|----------|--------------|--------------|
| **Do I always need it?** | Yes, for any GPU workload | Only if deploying NIMs |
| **Can I use without the other?** | Yes (for non-NIM GPU apps) | No (requires GPU Operator) |
| **What layer?** | Infrastructure/Platform | Application |
| **Manages what?** | GPU hardware & drivers | AI model containers |

### Summary

- **GPU Operator** = Foundation layer that makes GPUs work in Kubernetes
- **NIM Operator** = Application layer that deploys and manages NIM inference services

**You need GPU Operator first**, then optionally use NIM Operator for simplified NIM deployments.

---

## Batching in NIM

**Yes, batching is performed automatically by default** in NIM through Triton Inference Server's dynamic batching feature.

### What is Batching?

**Batching** combines multiple inference requests into a single batch to process them together on the GPU, maximizing throughput and GPU utilization.

```
Without Batching:                  With Batching:
Request 1 → GPU → Response         Requests 1,2,3,4 → GPU → Responses
Request 2 → GPU → Response         (processed together)
Request 3 → GPU → Response
Request 4 → GPU → Response

Throughput: Low                    Throughput: High
Latency: Low                       Latency: Slightly higher
GPU Util: 20%                      GPU Util: 80%
```

### Types of Batching in NIM

#### Detailed Comparison Table

| Type | How it Works | When Used | Trade-off |
|------|--------------|-----------|-----------|
| **Dynamic Batching** | **Server-side batching at request arrival:**<br>• Requests arrive at different times<br>• Server waits for a short delay (e.g., 500 microseconds)<br>• Groups together all requests that arrive during this window<br>• Processes the group as a single batch on GPU<br>• Each request gets its own response<br><br>*Example:* 4 requests arrive within 500μs → Server groups them → Processes all 4 together → Returns 4 individual responses | Default in NIM for all models | First request waits slightly longer (adds max_queue_delay), but overall throughput is much higher |
| **In-flight Batching**<br>(aka Continuous Batching) | **Dynamic batching during text generation:**<br>• Used specifically for LLMs generating text token-by-token<br>• New requests join ongoing generation batches<br>• When a request finishes generating, it leaves the batch<br>• New requests immediately fill the empty slot<br>• No need to wait for entire batch to complete<br><br>*Example:* Request A (50 tokens left) + Request B (100 tokens left) → B finishes → Request C (80 tokens) joins A in the same batch immediately | Automatic in LLM NIMs (TensorRT-LLM) | Maximizes GPU utilization; complex to implement (handled automatically) |
| **Static Batching** | **Client-side batching before sending:**<br>• Application groups requests together manually<br>• Sends multiple inputs as a single batch request<br>• Server processes them as one batch (no waiting)<br>• Returns batch of outputs<br>• Client is responsible for grouping<br><br>*Example:* Your app collects 10 user queries → Sends all 10 at once as batch → Gets 10 responses back | Rare, only when you have full control and pre-collected data | You control everything but must wait to collect enough requests; not flexible for real-time |


### Is Batching Automatic?

**Yes!** NIM automatically enables dynamic batching with sensible defaults:

| Component | Batching Type | Automatic? | Configuration |
|-----------|---------------|------------|---------------|
| **Triton Server** | Dynamic batching | ✅ Yes (enabled by default) | Via environment variables or config |
| **TensorRT-LLM** | In-flight batching | ✅ Yes (for LLMs) | Pre-configured in NIM |
| **Client Side** | Optional explicit batching | ⚠️ No (but not needed) | Client can send batch requests |

### Default Batching Configuration

NIM comes with pre-configured batching optimized for most use cases:

```bash
# NIM runs with these default batching settings:
# - Dynamic batching: ENABLED
# - Max batch size: 128-256 (model dependent)
# - Preferred batch sizes: [1, 2, 4, 8, 16, 32, 64, 128]
# - Max queue delay: 100-500 microseconds
```

### How to Configure Batching

#### Method 1: Environment Variables (Simplest)

```bash
# Deploy NIM with custom batching configuration
docker run --gpus all -p 8000:8000 \
  -e MAX_BATCH_SIZE=64 \
  -e PREFERRED_BATCH_SIZE="[8,16,32]" \
  -e MAX_QUEUE_DELAY_MICROSECONDS=1000 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```

#### Method 2: Triton Model Configuration

For advanced users who need fine-grained control, you can customize the Triton model configuration file.

**Where the Configuration File is Located:**

```
NIM Container Directory Structure:
/opt/nim/
├── models/
│   └── llama-3-8b/              # Model directory
│       ├── 1/                    # Version 1 (Triton versioning)
│       │   └── model files...
│       └── config.pbtxt          # ← Configuration file location
└── ...
```

**How to Persist Custom Configuration:**

```bash
# Option 1: Mount custom config when running NIM container
docker run --gpus all -p 8000:8000 \
  -v /path/to/your/config.pbtxt:/opt/nim/models/llama-3-8b/config.pbtxt \
  -e NGC_API_KEY=$NGC_API_KEY \
  nvcr.io/nim/meta/llama3-8b-instruct:latest

# Option 2: Create custom NIM image with modified config
# Dockerfile
FROM nvcr.io/nim/meta/llama3-8b-instruct:latest
COPY custom-config.pbtxt /opt/nim/models/llama-3-8b/config.pbtxt

# Option 3: Kubernetes ConfigMap (recommended for K8s)
kubectl create configmap nim-triton-config \
  --from-file=config.pbtxt=/path/to/config.pbtxt

# Then mount in pod spec:
# volumes:
#   - name: triton-config
#     configMap:
#       name: nim-triton-config
# volumeMounts:
#   - name: triton-config
#     mountPath: /opt/nim/models/llama-3-8b/config.pbtxt
#     subPath: config.pbtxt
```

**Example config.pbtxt File:**

```protobuf
# config.pbtxt - Triton model configuration
name: "llama-3-8b"
platform: "tensorrt_llm"
max_batch_size: 128

# Dynamic batching settings
dynamic_batching {
  # Preferred batch sizes (Triton will try to form these)
  preferred_batch_size: [ 8, 16, 32, 64, 128 ]

  # Max time to wait for batch formation (microseconds)
  max_queue_delay_microseconds: 500

  # Preserve ordering of requests
  preserve_ordering: true

  # Priority levels for requests
  priority_levels: 2
  default_priority_level: 1

  # Queue policy
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 10000000
    allow_timeout_override: true
    max_queue_size: 512
  }
}
```

**Important Notes:**

| Aspect | Details |
|--------|---------|
| **Default Config** | NIM comes with pre-optimized config.pbtxt already |
| **When to Modify** | Only if you need custom batching beyond environment variables |
| **File Format** | Protocol Buffer text format (.pbtxt) |
| **Validation** | Triton validates config on startup; check logs for errors |
| **Model Name** | Must match the model directory name |
| **Persistence** | Use volume mounts (Docker) or ConfigMaps (Kubernetes) |

**Quick Verification:**

```bash
# Check if your custom config is loaded
docker exec <container-id> cat /opt/nim/models/llama-3-8b/config.pbtxt

# View Triton server logs to verify config
docker logs <container-id> | grep "config.pbtxt"
```

### Client Code Examples

#### Example 1: Single Request (Benefits from Auto-Batching)

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"
)

# Single request - NIM automatically batches with other concurrent requests
response = client.chat.completions.create(
    model="meta/llama3-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

#### Example 2: Concurrent Requests (Maximizes Batching)

```python
import asyncio
import openai

client = openai.AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"
)

async def send_request(prompt):
    response = await client.chat.completions.create(
        model="meta/llama3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content

# Send 10 requests concurrently - NIM batches them automatically
async def main():
    prompts = [f"Tell me about topic {i}" for i in range(10)]

    # All 10 requests sent at once - NIM batches them together
    responses = await asyncio.gather(*[send_request(p) for p in prompts])

    for i, response in enumerate(responses):
        print(f"Response {i}: {response}")

asyncio.run(main())
```

#### Example 3: Manual Batch Request (Advanced)

```python
import requests

# You can also send explicit batch requests via Triton's HTTP/REST API
url = "http://localhost:8000/v2/models/llama-3-8b/infer"

# Batch of 4 inputs
batch_data = {
    "inputs": [
        {"name": "INPUT", "shape": [4, 512], "datatype": "INT32", "data": [...]},
    ],
    "outputs": [
        {"name": "OUTPUT"}
    ]
}

response = requests.post(url, json=batch_data)
print(response.json())
```

#### Example 4: Using Python SDK with Batching

```python
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

# Connect to NIM (Triton endpoint)
client = InferenceServerClient(url="localhost:8000")

# Prepare batch of inputs
batch_size = 8
inputs = []
input_data = InferInput("text_input", [batch_size, 1], "BYTES")

# Set input data for batch
batch_texts = [f"Prompt {i}".encode('utf-8') for i in range(batch_size)]
input_data.set_data_from_numpy(np.array(batch_texts).reshape(batch_size, 1))
inputs.append(input_data)

# Request output
outputs = [InferRequestedOutput("text_output")]

# Send batched request
response = client.infer(model_name="llama-3-8b", inputs=inputs, outputs=outputs)

# Get batched results
results = response.as_numpy("text_output")
print(results)
```

### Monitoring Batch Performance

```python
# Check batching effectiveness via Triton metrics
import requests

metrics = requests.get("http://localhost:8002/metrics").text

# Key metrics to monitor:
# nv_inference_request_success - total requests
# nv_inference_exec_count - actual batch executions
# nv_inference_queue_duration_us - time spent waiting in queue

# Calculate average batch size:
# avg_batch_size = total_requests / batch_executions
```

### Batching Configuration Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `max_batch_size` | 128-256 | Maximum requests in a batch | Higher = more throughput, more memory |
| `preferred_batch_size` | [1,2,4,8,16...] | Sizes Triton tries to form | Optimize for common workloads |
| `max_queue_delay_microseconds` | 100-500 | Max wait time to form batch | Higher = more batching, more latency |
| `preserve_ordering` | true | Maintain request order | True = FIFO, False = max performance |

### Batching Trade-offs

```
┌────────────────────────────────────────────────────┐
│         Batching Configuration Trade-offs          │
├────────────────────────────────────────────────────┤
│                                                    │
│  Low Latency Configuration:                        │
│  • max_queue_delay: 100 µs                         │
│  • max_batch_size: 8                               │
│  → Fast response, lower GPU utilization            │
│                                                    │
│  Balanced Configuration (Default):                 │
│  • max_queue_delay: 500 µs                         │
│  • max_batch_size: 32-128                          │
│  → Good latency, good throughput                   │
│                                                    │
│  High Throughput Configuration:                    │
│  • max_queue_delay: 1000+ µs                       │
│  • max_batch_size: 256                             │
│  → Maximum throughput, higher latency              │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Best Practices

| Practice | Why | How |
|----------|-----|-----|
| **Start with defaults** | Pre-optimized for most cases | Use NIM out-of-the-box |
| **Monitor actual batch sizes** | Understand real behavior | Check Triton metrics |
| **Tune for your workload** | Low traffic = lower delays | Adjust `max_queue_delay` |
| **Send concurrent requests** | Maximize batching benefits | Use async clients |
| **Match batch to GPU memory** | Prevent OOM errors | Monitor GPU memory usage |
| **Test different configs** | Find optimal settings | Use load testing tools |

### Example: Tuning Batching

```bash
# Scenario 1: Low latency (chatbot, real-time)
docker run --gpus all -p 8000:8000 \
  -e MAX_BATCH_SIZE=8 \
  -e MAX_QUEUE_DELAY_MICROSECONDS=100 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest

# Scenario 2: High throughput (batch processing)
docker run --gpus all -p 8000:8000 \
  -e MAX_BATCH_SIZE=128 \
  -e MAX_QUEUE_DELAY_MICROSECONDS=2000 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest

# Scenario 3: Balanced (general purpose)
docker run --gpus all -p 8000:8000 \
  -e MAX_BATCH_SIZE=64 \
  -e MAX_QUEUE_DELAY_MICROSECONDS=500 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```

### Kubernetes Batching Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-llama
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-8b-instruct:latest
        env:
        - name: MAX_BATCH_SIZE
          value: "64"
        - name: MAX_QUEUE_DELAY_MICROSECONDS
          value: "500"
        - name: PREFERRED_BATCH_SIZE
          value: "[8,16,32,64]"
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Summary

**Key Takeaways:**

1. ✅ **Batching is automatic** - NIM enables dynamic batching by default
2. 🎯 **No client changes needed** - Send individual requests, NIM batches them
3. ⚙️ **Configurable** - Tune via environment variables for your workload
4. 📊 **Monitor metrics** - Track batch sizes and queue times
5. 🔄 **Trade-off** - Latency vs Throughput (adjust `max_queue_delay`)

**Default Behavior:**
```python
# Your code - just send requests normally
response = client.chat.completions.create(...)

# NIM automatically:
# 1. Queues your request
# 2. Waits up to 500µs for more requests
# 3. Batches them together (up to 128 requests)
# 4. Processes batch on GPU
# 5. Returns individual responses
```

**When to Tune:**
- Low traffic → Decrease `max_queue_delay` (reduce waiting)
- High traffic → Increase `max_batch_size` (maximize GPU usage)
- Strict latency SLA → Lower both parameters
- Batch jobs → Increase both parameters

---

## Monitoring and Observability

Monitoring NIM deployments involves tracking GPU utilization, inference performance, and system health.

### What to Monitor

| Category | Metrics | Why Important |
|----------|---------|---------------|
| **GPU Metrics** | • GPU utilization %<br>• GPU memory usage<br>• GPU temperature<br>• Power consumption | Optimize resource usage, prevent overheating, cost management |
| **Inference Metrics** | • Request latency (P50, P95, P99)<br>• Throughput (requests/sec)<br>• Token generation rate<br>• Queue depth | Performance optimization, SLA compliance |
| **Model Metrics** | • Model loading time<br>• Batch size<br>• Sequence length<br>• Cache hit rate | Understand bottlenecks, tune configuration |
| **System Metrics** | • CPU usage<br>• Memory usage<br>• Network I/O<br>• Disk I/O | Overall system health |
| **Error Metrics** | • Error rate<br>• Timeout rate<br>• Failed requests | Reliability and debugging |

### Monitoring Stack

```
┌─────────────────────────────────────────────────────────┐
│               NIM Monitoring Architecture               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │            NIM Container                         │   │
│  │  ┌────────────────┐  ┌──────────────────────┐   │   │
│  │  │ Triton Server  │  │ TensorRT-LLM Engine │   │   │
│  │  │ (serves APIs)  │  │ (runs on GPU)       │   │   │
│  │  └────────┬───────┘  └──────────┬───────────┘   │   │
│  │           │                     │               │   │
│  │           ↓                     ↓               │   │
│  │  ┌─────────────────────────────────────────┐   │   │
│  │  │      Metrics Exporters                  │   │   │
│  │  │  • Triton Metrics (HTTP /metrics)       │   │   │
│  │  │  • DCGM Exporter (GPU metrics)          │   │   │
│  │  └──────────────────┬──────────────────────┘   │   │
│  └───────────────────────┼──────────────────────────┘   │
│                          ↓                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Prometheus (Metrics Collection)           │  │
│  │  • Scrapes metrics every 15s                      │  │
│  │  • Stores time-series data                        │  │
│  │  • Enables queries and alerts                     │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Grafana (Visualization)                   │  │
│  │  • Dashboards for GPU utilization                 │  │
│  │  • Inference performance graphs                   │  │
│  │  • Real-time alerts                               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Monitoring Tools

| Tool | Purpose | What it Monitors | Access Method |
|------|---------|------------------|---------------|
| **NVIDIA DCGM Exporter** | GPU metrics collector | GPU utilization, memory, temperature, power | Prometheus endpoint |
| **Triton Metrics Server** | Inference metrics | Request latency, throughput, queue depth, model stats | HTTP `/metrics` endpoint |
| **Prometheus** | Metrics storage & querying | Time-series database for all metrics | HTTP API, PromQL queries |
| **Grafana** | Visualization & alerting | Dashboards, graphs, alerts | Web UI (port 3000) |
| **NVIDIA GPU Dashboard** | Pre-built Grafana dashboard | Comprehensive GPU monitoring | Grafana import |

### Quick Setup Example

```bash
# 1. Deploy NIM with metrics enabled
docker run --gpus all -p 8000:8000 -p 8002:8002 \
  -e METRICS_ENABLED=true \
  nvcr.io/nim/meta/llama3-8b-instruct:latest

# 2. Deploy DCGM Exporter for GPU metrics
docker run -d --gpus all -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:3.1.8-3.1.5-ubuntu20.04

# 3. Configure Prometheus to scrape metrics
# prometheus.yml
scrape_configs:
  - job_name: 'triton'
    static_configs:
      - targets: ['localhost:8002']

  - job_name: 'dcgm'
    static_configs:
      - targets: ['localhost:9400']

# 4. Access metrics
curl http://localhost:8002/metrics  # Triton metrics
curl http://localhost:9400/metrics  # GPU metrics
```

### Critical Metrics to Watch

#### GPU Metrics (via DCGM)

```promql
# GPU Utilization (%)
DCGM_FI_DEV_GPU_UTIL

# GPU Memory Usage (MB)
DCGM_FI_DEV_FB_USED

# GPU Temperature (°C)
DCGM_FI_DEV_GPU_TEMP

# Power Usage (Watts)
DCGM_FI_DEV_POWER_USAGE
```

#### Triton Inference Metrics

```promql
# Request throughput (requests/sec)
rate(nv_inference_request_success[1m])

# Request latency P95 (microseconds)
histogram_quantile(0.95, nv_inference_request_duration_us_bucket)

# Queue time (how long requests wait)
nv_inference_queue_duration_us

# Model execution time
nv_inference_compute_infer_duration_us

# Active requests
nv_inference_pending_request_count
```

### Sample Grafana Dashboard

```
┌─────────────────────────────────────────────────────────┐
│           NIM Performance Dashboard                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  GPU Utilization                    GPU Memory         │
│  ┌────────────────┐                ┌─────────────────┐ │
│  │ ▁▂▃▅▆█████████ │                │ 18.5 GB / 24 GB │ │
│  │      85%       │                │      77%        │ │
│  └────────────────┘                └─────────────────┘ │
│                                                         │
│  Requests/Second                    P95 Latency        │
│  ┌────────────────┐                ┌─────────────────┐ │
│  │ ▂▃▄▅▆▇████▇▆▅  │                │   125 ms        │ │
│  │     1,234      │                │  ▁▂▂▃▄▅▆▅▄▃▂▁   │ │
│  └────────────────┘                └─────────────────┘ │
│                                                         │
│  Token Generation Rate              Error Rate         │
│  ┌────────────────┐                ┌─────────────────┐ │
│  │  45 tokens/sec │                │    0.02%        │ │
│  │  ▃▄▅▆▇████▇▆▅  │                │   ▁▁▁▁▁▁▁▁▁    │ │
│  └────────────────┘                └─────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Kubernetes Monitoring (with NIM Operator)

```yaml
# ServiceMonitor for Prometheus Operator
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: nim-metrics
spec:
  selector:
    matchLabels:
      app: nim-llama
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

### Alerting Rules Example

```yaml
# Prometheus alerts
groups:
  - name: nim_alerts
    rules:
      # GPU utilization too low (underutilization)
      - alert: GPUUnderutilized
        expr: DCGM_FI_DEV_GPU_UTIL < 30
        for: 10m
        annotations:
          summary: "GPU utilization below 30% for 10 minutes"

      # High latency
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, nv_inference_request_duration_us_bucket) > 500000
        for: 5m
        annotations:
          summary: "P95 latency above 500ms"

      # GPU temperature warning
      - alert: GPUTemperatureHigh
        expr: DCGM_FI_DEV_GPU_TEMP > 80
        for: 5m
        annotations:
          summary: "GPU temperature above 80°C"

      # Error rate spike
      - alert: HighErrorRate
        expr: rate(nv_inference_request_failure[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Error rate above 5%"
```

### Best Practices

| Practice | Why | How |
|----------|-----|-----|
| **Monitor P95/P99 latency** | P50 doesn't show tail latencies | Use Prometheus histograms |
| **Track GPU memory trends** | Prevent OOM errors | Alert when usage > 85% |
| **Set up alerts** | Proactive issue detection | Use Prometheus Alertmanager |
| **Use pre-built dashboards** | Save time, proven metrics | Import NVIDIA GPU dashboard |
| **Monitor batch efficiency** | Optimize throughput | Track batch size vs latency |
| **Log all errors** | Debugging production issues | Centralize logs (ELK, Loki) |

### Logging and Tracing

```bash
# Enable verbose logging in NIM
docker run --gpus all -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  -e TRITON_LOG_VERBOSE=1 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest

# View logs
docker logs <container-id>

# For distributed tracing (OpenTelemetry)
# Configure Triton to export traces
--trace-config=opentelemetry,url=http://jaeger:4317
```

### Commercial Monitoring Solutions

| Solution | Features | Best For |
|----------|----------|----------|
| **NVIDIA Fleet Command** | Centralized GPU fleet management | Enterprise multi-GPU deployments |
| **Datadog** | Full-stack monitoring with AI/ML dashboards | Cloud-native environments |
| **New Relic** | APM with inference monitoring | Application performance tracking |
| **Prometheus + Grafana** | Open-source, customizable | Self-managed, cost-conscious |

### Quick Health Check

```bash
# Check if NIM is healthy
curl http://localhost:8000/v2/health/ready

# Get current metrics snapshot
curl http://localhost:8002/metrics | grep nv_inference

# Check GPU status
nvidia-smi
```

### Summary

**Essential Monitoring Stack for NIM:**
1. **DCGM Exporter** → GPU metrics
2. **Triton Metrics** → Inference performance
3. **Prometheus** → Metrics storage
4. **Grafana** → Visualization & alerts

**Key Metrics:**
- GPU utilization & memory
- P95/P99 request latency
- Throughput (requests/sec, tokens/sec)
- Error rates

**Pro Tip**: Start with pre-built NVIDIA Grafana dashboards, then customize based on your SLAs.

---

## Scaling NIM for High Traffic (Enterprise)

### When to Scale

**Traffic Indicators**: Req/sec > 100, P95 latency > 500ms, GPU util > 80%, growing queues
**Business Indicators**: Slow responses, peak degradation, regional expansion

### Scaling Strategies

| Strategy | Best For | Cost | Complexity |
|----------|----------|------|------------|
| **Horizontal** | High availability, traffic bursts | Medium | Medium |
| **Vertical** | Single instance performance (A100 → H100) | High | Low |
| **Multi-GPU** | Large models (tensor parallelism) | High | Medium |
| **Multi-Region** | Global traffic, low latency | Very High | High |

### 1. Horizontal Scaling

```
┌──────────────────────────────────────────────────────────────────┐
│             Horizontal Scaling Architecture                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    User Traffic (1000 req/sec)                   │
│                              │                                   │
│                              ▼                                   │
│                  ┌───────────────────────┐                       │
│                  │   Load Balancer       │                       │
│                  │   (NGINX/Istio/ALB)   │                       │
│                  └───────────┬───────────┘                       │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐              │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│   ┌──────────┐         ┌──────────┐         ┌──────────┐        │
│   │  NIM 1   │         │  NIM 2   │         │  NIM 3   │        │
│   │  (GPU 1) │         │  (GPU 2) │         │  (GPU 3) │        │
│   │ 333 r/s  │         │ 333 r/s  │         │ 333 r/s  │        │
│   └──────────┘         └──────────┘         └──────────┘        │
│                                                                  │
│  ✓ Each instance handles manageable load                        │
│  ✓ High availability (one fails, others continue)               │
│  ✓ Easy to scale (add/remove replicas)                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Kubernetes HPA (GPU-based):**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nim-gpu-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: nim-llama-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        averageValue: "75"
  - type: Pods
    pods:
      metric:
        name: nv_inference_pending_request_count
      target:
        averageValue: "10"
```

### 2. Load Balancing Options

**K8s Service** (Round-robin), **NGINX Ingress** (rate limiting, timeouts), **Istio** (canary, session affinity)

### 3. Multi-GPU Scaling

For large models (e.g., LLaMA 405B), use tensor parallelism across multiple GPUs per pod:

```yaml
env:
- name: TENSOR_PARALLEL_SIZE
  value: "4"
resources:
  limits:
    nvidia.com/gpu: 4
```

```
┌──────────────────────────────────────────────────────────┐
│        Multi-GPU Deployment (Large Models)               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Pod 1                          Pod 2                    │
│  ┌──────────────────────┐      ┌──────────────────────┐ │
│  │  NIM Instance 1      │      │  NIM Instance 2      │ │
│  │  ┌────┬────┬────┬────┤      │  ┌────┬────┬────┬────┤ │
│  │  │GPU1│GPU2│GPU3│GPU4│      │  │GPU5│GPU6│GPU7│GPU8│ │
│  │  └────┴────┴────┴────┤      │  └────┴────┴────┴────┤ │
│  │  Tensor Parallel     │      │  Tensor Parallel     │ │
│  │  (Model split across)│      │  (Model split across)│ │
│  └──────────────────────┘      └──────────────────────┘ │
│                                                          │
│  ✓ Each pod handles large model with 4-way parallelism  │
│  ✓ 2 pods provide redundancy and 2x throughput          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4. Multi-Region Deployment

```
┌────────────────────────────────────────────────────────────┐
│         Multi-Region Global Deployment                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│              Global Load Balancer (CloudFlare/AWS)         │
│                    (Geo-routing + Failover)                │
│                            │                               │
│        ┌───────────────────┼───────────────────┐           │
│        │                   │                   │           │
│        ▼                   ▼                   ▼           │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐     │
│  │ US-East  │        │ EU-West  │        │   APAC   │     │
│  │ Region   │        │ Region   │        │  Region  │     │
│  │          │        │          │        │          │     │
│  │ 5 Pods   │        │ 5 Pods   │        │ 3 Pods   │     │
│  │ 5 GPUs   │        │ 5 GPUs   │        │ 3 GPUs   │     │
│  └──────────┘        └──────────┘        └──────────┘     │
│                                                            │
│  ✓ Users routed to nearest region (low latency)           │
│  ✓ Automatic failover if region unavailable               │
│  ✓ Independent scaling per region                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 5. Traffic Management Patterns

#### Pattern 1: Peak Hour Auto-Scaling

```yaml
# Scale up during business hours, scale down at night
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nim-time-based-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nim-deployment
  minReplicas: 2   # Night: 2 replicas
  maxReplicas: 20  # Day: up to 20 replicas
  # Use CronJobs to adjust min/max dynamically
```

```yaml
# CronJob to scale up for peak hours
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nim-scale-up
spec:
  schedule: "0 8 * * 1-5"  # 8 AM weekdays
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: scaler
            image: bitnami/kubectl
            command:
            - /bin/sh
            - -c
            - |
              kubectl patch hpa nim-hpa -n production \
                --type='json' \
                -p='[{"op": "replace", "path": "/spec/minReplicas", "value": 10}]'
          restartPolicy: OnFailure

---
# CronJob to scale down after hours
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nim-scale-down
spec:
  schedule: "0 18 * * 1-5"  # 6 PM weekdays
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: scaler
            image: bitnami/kubectl
            command:
            - /bin/sh
            - -c
            - |
              kubectl patch hpa nim-hpa -n production \
                --type='json' \
                -p='[{"op": "replace", "path": "/spec/minReplicas", "value": 2}]'
          restartPolicy: OnFailure
```

#### Pattern 2: Burst Handling with Queue

```yaml
# Use message queue for burst traffic
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-worker
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: worker
        image: your-worker-image
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: NIM_ENDPOINT
          value: "http://nim-service:8000"
        # Worker pulls from queue, sends to NIM
```

**Queue-based Architecture:**

```
┌────────────────────────────────────────────────────────────┐
│         Queue-based Burst Handling                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  High Traffic Burst                                        │
│  (1000s of requests)                                       │
│         │                                                  │
│         ▼                                                  │
│  ┌─────────────┐                                           │
│  │ API Gateway │                                           │
│  └──────┬──────┘                                           │
│         │                                                  │
│         ▼                                                  │
│  ┌─────────────────┐                                       │
│  │ Redis Queue     │ ← Requests buffered here            │
│  │ (Fast writes)   │                                       │
│  └────────┬────────┘                                       │
│           │                                                │
│           ▼                                                │
│  ┌───────────────────────────┐                            │
│  │   Worker Pool (5 workers) │                            │
│  │   Pull → Process → NIM    │                            │
│  └───────────┬───────────────┘                            │
│              │                                             │
│              ▼                                             │
│  ┌────────────────────────┐                               │
│  │  NIM Pods (3 replicas) │ ← Steady, manageable load    │
│  │  Handle requests       │                               │
│  └────────────────────────┘                               │
│                                                            │
│  ✓ Absorbs traffic spikes without overwhelming NIM        │
│  ✓ Workers auto-scale based on queue depth                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 6. Performance Optimization for Scale

| Optimization | Impact | Implementation | When to Use |
|--------------|--------|----------------|-------------|
| **Model Quantization** | 2-4x throughput increase | INT8/FP8 precision | Cost-sensitive, slight quality trade-off OK |
| **Batching Tuning** | 3-10x throughput increase | Increase max_batch_size, adjust delays | High concurrent traffic |
| **Request Caching** | 10-100x for cached hits | Redis/Memcached for common queries | Repeated queries (FAQ bots) |
| **Model Warmup** | Eliminate cold starts | Pre-load models on pod start | Critical for latency SLA |
| **Connection Pooling** | Reduce overhead | HTTP keep-alive, persistent connections | High request rate |

**Example: Implementing Request Caching:**

```python
import redis
import hashlib
import json

# Initialize Redis client
cache = redis.Redis(host='redis-service', port=6379, db=0)
CACHE_TTL = 3600  # 1 hour

def get_cached_response(prompt):
    """Check if response exists in cache"""
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

def cache_response(prompt, response):
    """Store response in cache"""
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()
    cache.setex(cache_key, CACHE_TTL, json.dumps(response))

def query_nim_with_cache(prompt, nim_client):
    """Query NIM with caching layer"""
    # Check cache first
    cached_response = get_cached_response(prompt)
    if cached_response:
        print("Cache hit!")
        return cached_response

    # Cache miss - query NIM
    response = nim_client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    # Cache the response
    cache_response(prompt, response.choices[0].message.content)

    return response.choices[0].message.content
```

### 7. Cost Optimization at Scale

| Strategy | Cost Savings | Complexity | Best For |
|----------|--------------|------------|----------|
| **Spot/Preemptible Instances** | 60-80% | High | Batch workloads, fault-tolerant apps |
| **Right-sizing GPUs** | 30-50% | Low | Match GPU to model size |
| **Auto-scaling** | 40-60% | Medium | Variable traffic patterns |
| **Shared GPU (MIG)** | 20-40% | Medium | Multiple small models |
| **Reserved Instances** | 30-50% | Low | Predictable baseline load |

**Example: Mixed Instance Strategy:**

```yaml
# Production: On-demand instances for baseline
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-baseline
spec:
  replicas: 3
  template:
    spec:
      nodeSelector:
        instance-type: on-demand
      # ... NIM configuration

---
# Burst capacity: Spot instances
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-burst
spec:
  replicas: 0  # Scaled by HPA
  template:
    spec:
      nodeSelector:
        instance-type: spot
      tolerations:
      - key: "spot"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      # ... NIM configuration
```

### 8. Monitoring at Scale

**Key Metrics Dashboard for Scaled Deployments:**

```yaml
# Prometheus queries for scaled NIM deployments

# Total requests per second across all pods
sum(rate(nv_inference_request_success[1m]))

# Per-pod request rate
sum(rate(nv_inference_request_success[1m])) by (pod)

# Average GPU utilization across cluster
avg(DCGM_FI_DEV_GPU_UTIL)

# P95 latency across all instances
histogram_quantile(0.95, sum(rate(nv_inference_request_duration_us_bucket[1m])) by (le))

# Number of active NIM pods
count(up{job="nim-metrics"} == 1)

# Requests per dollar (cost efficiency)
sum(rate(nv_inference_request_success[1m])) /
  (count(up{job="nim-metrics"} == 1) * GPU_COST_PER_HOUR)
```

### 9. Real-World Scaling Example

**Scenario: E-commerce chatbot serving 10,000 concurrent users**

```yaml
# Complete production-ready configuration
---
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: nim-production

---
# NGC API Key Secret
apiVersion: v1
kind: Secret
metadata:
  name: ngc-secret
  namespace: nim-production
type: Opaque
stringData:
  api-key: "your-ngc-api-key"

---
# NIM Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-chatbot
  namespace: nim-production
spec:
  replicas: 5  # Start with 5
  selector:
    matchLabels:
      app: nim-chatbot
  template:
    metadata:
      labels:
        app: nim-chatbot
        version: stable
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8002"
    spec:
      affinity:
        # Spread across nodes for HA
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - nim-chatbot
              topologyKey: kubernetes.io/hostname
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-70b-instruct:1.0.0
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8002
          name: metrics
        env:
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ngc-secret
              key: api-key
        - name: MAX_BATCH_SIZE
          value: "128"
        - name: MAX_QUEUE_DELAY_MICROSECONDS
          value: "500"
        - name: TRITON_MAX_QUEUE_DELAY_MICROSECONDS
          value: "500"
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "16"
            memory: "64Gi"
          limits:
            nvidia.com/gpu: 1
            cpu: "32"
            memory: "128Gi"
        livenessProbe:
          httpGet:
            path: /v2/health/live
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /v2/health/ready
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: nim-chatbot-service
  namespace: nim-production
spec:
  type: ClusterIP
  selector:
    app: nim-chatbot
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 8002
    targetPort: 8002
    name: metrics

---
# HPA with custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nim-chatbot-hpa
  namespace: nim-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nim-chatbot
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "150"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120

---
# Ingress with rate limiting
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nim-chatbot-ingress
  namespace: nim-production
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
spec:
  ingressClassName: nginx
  rules:
  - host: chatbot-api.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nim-chatbot-service
            port:
              number: 8000
```

**Expected Performance:**

| Metric | Value | Notes |
|--------|-------|-------|
| **Concurrent Users** | 10,000 | Peak capacity |
| **Requests/sec** | 750 (150/replica × 5) | With 5 replicas |
| **P95 Latency** | < 300ms | Within SLA |
| **Availability** | 99.95% | Multi-replica HA |
| **Cost** | ~$50/hour | 5× A100 GPUs at $10/hour |
| **Auto-scale Time** | 2-3 minutes | Pod startup + model loading |

### 10. Best Practices for Enterprise Scaling

| Practice | Why | How |
|----------|-----|-----|
| **Over-provision initially** | Avoid cold starts during traffic spikes | Start with 2x expected replicas, then optimize |
| **Use PodDisruptionBudgets** | Maintain availability during updates | Ensure min 50% pods always running |
| **Implement circuit breakers** | Prevent cascade failures | Timeout + retry logic in clients |
| **Monitor cost per request** | Optimize ROI | Track GPU cost vs request volume |
| **Test failure scenarios** | Ensure resilience | Chaos engineering (kill random pods) |
| **Use blue-green deployments** | Zero-downtime updates | Run old + new versions, switch traffic |
| **Set resource limits** | Prevent resource exhaustion | Always set CPU/memory/GPU limits |
| **Implement request queuing** | Handle bursts gracefully | Use Redis/RabbitMQ for async processing |

### 11. Troubleshooting Scaling Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Slow scale-up** | HPA takes 5+ minutes | Reduce stabilization window, increase scale-up rate |
| **Pods stuck pending** | New pods won't start | Check node GPU availability, add nodes |
| **Uneven load distribution** | Some pods at 100%, others idle | Check load balancer algorithm, use session affinity carefully |
| **OOM errors during scale** | Pods crash with out-of-memory | Increase memory limits, reduce batch size |
| **High latency despite scaling** | P95 latency still high | Check batching config, consider vertical scaling |

### Summary

**Key Takeaways for Enterprise Scaling:**

1. ✅ **Start with horizontal scaling** - Most cost-effective and flexible
2. ✅ **Use Kubernetes HPA** - Automatic scaling based on metrics
3. ✅ **Implement proper load balancing** - Distribute traffic evenly
4. ✅ **Monitor custom metrics** - GPU utilization, request rate, queue depth
5. ✅ **Optimize costs** - Mix on-demand/spot, right-size GPUs, auto-scale aggressively
6. ✅ **Test at scale** - Load test before production
7. ✅ **Plan for failures** - Multi-replica, multi-zone, circuit breakers
8. ✅ **Cache aggressively** - Reduce redundant inference calls

**Scaling Checklist:**

```
☐ Kubernetes cluster with GPU Operator installed
☐ Prometheus + custom metrics exporter configured
☐ HPA configured with appropriate metrics
☐ Load balancer/ingress configured
☐ Monitoring dashboards set up
☐ Alerts configured for latency/errors/costs
☐ Load testing completed
☐ Disaster recovery plan documented
☐ Cost tracking and optimization in place
```

---

## GPU Partitioning & MIG (Multi-Instance GPU) for NIM

### What is MIG?

**MIG (Multi-Instance GPU)** allows you to partition a single NVIDIA GPU (A100, H100, L40S) into **multiple independent GPU instances**, each with dedicated compute, memory, and memory bandwidth. This is useful for running multiple smaller NIM deployments on a single GPU.

```
Traditional GPU Setup:        MIG-Enabled GPU:
┌──────────────────┐         ┌─────┬─────┬─────┐
│   H100           │         │ MIG │ MIG │ MIG │
│   80GB VRAM      │    vs   │ #0  │ #1  │ #2  │
│   Shared by      │         │     │     │     │
│   all models     │         │20GB │30GB │30GB │
└──────────────────┘         └─────┴─────┴─────┘

Single large model          Multiple smaller models
(LLaMA 70B only)           (LLaMA 8B + Embed + Guard)
```

### When to Use MIG with NIM

| Scenario | Use MIG? | Reasoning |
|----------|----------|-----------|
| **Single large model** (LLaMA 70B, GPT-4 scale) | ❌ No | Needs full GPU for maximum throughput and latency |
| **Multiple small/medium models** (embed + chat + guard) | ✅ Yes | Share GPU, reduce infrastructure cost 30-40% |
| **Disaggregated P/D setup** | ❌ No | P/D need dedicated GPUs for performance, NVLink disabled in MIG |
| **Development/testing environment** | ✅ Yes | Sufficient throughput, lower cost for non-prod |
| **Multi-tenant deployment** (isolation required) | ✅ Yes | MIG provides hardware-level isolation |

### MIG Configuration Profiles

NVIDIA provides preset MIG modes that partition the GPU automatically:

| Mode | Profile | Use Case |
|------|---------|----------|
| **1g.10gb** | 7× tiny instances (10GB each) | Lightweight embeddings, spell-check |
| **1g.20gb** | 4× small instances (20GB each) | Small LLMs (Llama 8B), embedding models |
| **2g.20gb** | 2× medium instances (20GB each) | Medium models with redundancy |
| **1g.40gb** | 2× large instances (40GB each) | Large models (Llama 70B variants) |
| **No MIG** | 1× full instance (80GB) | Maximum performance for largest models |

### Example: Multi-Model NIM Deployment with MIG

```bash
# Enable MIG mode on the GPU
nvidia-smi -mig 1

# Configure H100 into 2×40GB MIG instances
nvidia-smi -lgc G.1g,G.1g,G.1g,G.1g

# Verify MIG instances created
nvidia-smi -L
# Output:
# GPU 0 MIG 0: 40GB (1g.40gb)
# GPU 0 MIG 1: 40GB (1g.40gb)

# Deploy NIM 1 on MIG instance 0
docker run --gpus device=0:0 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama3-70b-instruct:latest

# Deploy NIM 2 on MIG instance 1 (different model)
docker run --gpus device=0:1 \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8001:8000 \
  nvcr.io/nim/nvidia/embed-qa-4:latest
```

**Result:** Two independent NIM services running on a single H100, each with 40GB dedicated VRAM.

### GPU Partitioning Strategies for NIM at Scale

#### Strategy 1: **Full GPU per NIM** (Maximum Performance)

```yaml
# Kubernetes deployment - one H100 per NIM
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-llama-70b
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-70b-instruct:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # Full H100 per replica
```

**Pros:**
- ✅ Maximum latency and throughput
- ✅ No resource contention
- ✅ Simplest configuration

**Cons:**
- ❌ Higher cost per model
- ❌ GPU under-utilization if model is small

#### Strategy 2: **MIG Partitioning** (Cost-Efficient Multi-Model)

```yaml
# Pre-partition GPU into MIG instances (done once via nvidia-smi)
# Then deploy multiple NIMs to different MIG partitions

---
# NIM 1: Embedding model on MIG instance 0
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-embeddings
spec:
  template:
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/nvidia/embed-qa-4:latest
        resources:
          limits:
            nvidia.com/gpu: "2g.20gb"  # MIG partition

---
# NIM 2: Small LLM on MIG instance 1
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-llama-8b
spec:
  template:
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-8b-instruct:latest
        resources:
          limits:
            nvidia.com/gpu: "2g.20gb"  # Different MIG partition

---
# NIM 3: Safety model on MIG instance 2
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-guard
spec:
  template:
    spec:
      containers:
      - name: nim
        image: nvcr.io/nim/nvidia/nemoguard:latest
        resources:
          limits:
            nvidia.com/gpu: "2g.20gb"  # Another MIG partition
```

**Pros:**
- ✅ 30-40% cost reduction per GPU
- ✅ Efficient for multi-model systems
- ✅ Hardware isolation between models

**Cons:**
- ❌ Throughput per model is lower
- ❌ Not suitable for single large models

#### Strategy 3: **Disaggregated P/D** (No MIG)

```yaml
# Disaggregated prefill/decode requires FULL GPUs
# NVLink is disabled in MIG mode → use separate GPUs

---
# Prefill cluster (NO MIG)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-prefill
spec:
  template:
    spec:
      nodeSelector:
        role: prefill
      containers:
      - name: nim
        env:
        - name: NIM_DISAGG_ROLE
          value: "prefill"
        resources:
          limits:
            nvidia.com/gpu: 1  # Full H100 (NO MIG)

---
# Decode cluster (NO MIG)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-decode
spec:
  replicas: 4  # More decode than prefill
  template:
    spec:
      nodeSelector:
        role: decode
      containers:
      - name: nim
        env:
        - name: NIM_DISAGG_ROLE
          value: "decode"
        resources:
          limits:
            nvidia.com/gpu: 1  # Full H100 (NO MIG)
```

**Pros:**
- ✅ Extreme throughput (2-4x higher)
- ✅ Lowest latency (10-25x TTFT improvement)
- ✅ Optimized for long-prompt workloads

**Cons:**
- ❌ Highest cost (requires 2x GPUs)
- ❌ Complex setup (P/D coordination)
- ❌ Only for high-traffic production

### Comparing the Three Strategies

| Metric | Full GPU per NIM | MIG Multi-Model | Disaggregated P/D |
|--------|-----------------|-----------------|-------------------|
| **Cost per GPU** | Baseline | -35% | +100% |
| **Throughput** | Medium | Medium-Low | Very High |
| **TTFT Latency** | Medium | Medium | Very Low (10-25x) |
| **Complexity** | Simple | Medium | High |
| **Best For** | Production single models | Cost-saving multi-model | Extreme throughput |
| **Models per GPU** | 1 | 3-4 | N/A (separate clusters) |

### MIG Limitations with NIM

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| **NVLink disabled** | P2P transfers 10x slower, disaggregation impractical | Use full GPUs for P/D separation |
| **Aggregate throughput lower** | Each instance ~25-30% slower than full GPU | Use full GPUs if throughput is critical |
| **GPU memory fragmentation** | Unused partitions waste space | Right-size MIG partitions for your models |
| **Not all models fit** | Large models need full GPU (LLaMA 405B, GPT-4) | Use A100/H100 without MIG |

### Decision Tree: Choose Your GPU Strategy

```
START
  │
  ├─ Deploying single LARGE model (>50B params)?
  │  ├─ YES → Use FULL GPU per NIM
  │  │        [Max throughput, simplest]
  │  │
  │  └─ NO → Deploying multiple SMALL models?
  │     ├─ YES → Consider MIG partitioning
  │     │        [Cost-efficient]
  │     │
  │     └─ NO → Need extreme throughput (>1000 tok/sec)?
  │        ├─ YES → Use Disaggregated P/D
  │        │        [2-4x throughput, complex]
  │        │
  │        └─ NO → Use full GPU per NIM
  │               [Balanced approach]
```

### Practical Cost Example

```
Scenario: Deploy 3 models (LLaMA 8B, Embeddings, Guard)

OPTION A: Full GPU per NIM
  Cost: 3 × H100 @ $10/hour = $30/hour
  Monthly: ~$21,600

OPTION B: MIG on 1 H100 (2+1 partition)
  Cost: 1 × H100 @ $10/hour = $10/hour
  Monthly: ~$7,200
  Savings: 66%
  Trade-off: Slightly higher latency per model

OPTION C: MIG on 2 H100s (load-balanced)
  Cost: 2 × H100 @ $10/hour = $20/hour
  Monthly: ~$14,400
  Savings: 33%
  Trade-off: Better latency than single H100 MIG
```

### Best Practices for GPU Partitioning

| Practice | Why | How |
|----------|-----|-----|
| **Right-size MIG partitions** | Avoid wasting GPU memory | Test model VRAM requirements first |
| **Monitor per-partition utilization** | Detect bottlenecks | Use nvidia-smi, Prometheus metrics |
| **Consider fault tolerance** | Single GPU failure affects multiple models | Add redundancy (multiple H100s with MIG) |
| **Use Kubernetes resource limits** | Prevent resource starvation | Set `nvidia.com/gpu: "2g.20gb"` |
| **Test with your workload** | MIG performance varies by model | Benchmark before production |
| **Document your MIG config** | Team needs to know the layout | Keep infrastructure-as-code (YAML) |

### Kubernetes with MIG: Full Example

```yaml
# First, enable MIG on the node (one-time)
# nvidia-smi -mig 1
# nvidia-smi -lgc G.1g,G.1g,G.1g,G.1g

---
# NIM Embedding (MIG instance 0)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-embed
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nim-embed
  template:
    metadata:
      labels:
        app: nim-embed
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"          # GPU nodes
      containers:
      - name: nim
        image: nvcr.io/nim/nvidia/embed-qa-4:latest
        ports:
        - containerPort: 8000
        env:
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ngc-secret
              key: api-key
        resources:
          requests:
            nvidia.com/gpu: "2g.20gb"   # MIG partition
          limits:
            nvidia.com/gpu: "2g.20gb"

---
apiVersion: v1
kind: Service
metadata:
  name: nim-embed-service
spec:
  selector:
    app: nim-embed
  ports:
  - port: 8000
    targetPort: 8000

---
# NIM LLaMA 8B (MIG instance 1)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-llama-8b
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nim-llama-8b
  template:
    metadata:
      labels:
        app: nim-llama-8b
    spec:
      nodeSelector:
        nvidia.com/gpu: "true"
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama3-8b-instruct:latest
        ports:
        - containerPort: 8000
        env:
        - name: NGC_API_KEY
          valueFrom:
            secretKeyRef:
              name: ngc-secret
              key: api-key
        resources:
          requests:
            nvidia.com/gpu: "2g.20gb"
          limits:
            nvidia.com/gpu: "2g.20gb"

---
apiVersion: v1
kind: Service
metadata:
  name: nim-llama-service
spec:
  selector:
    app: nim-llama-8b
  ports:
  - port: 8000
    targetPort: 8000
```

### Summary

**GPU Partitioning Strategy Selection:**

```
Choose:
┌─────────────────────────────────────────┐
│ Full GPU per NIM     → Production SLA,  │
│                        single model, max│
│                        throughput        │
├─────────────────────────────────────────┤
│ MIG Partitioning     → Multi-model,     │
│                        cost-optimization│
│                        good enough      │
│                        latency          │
├─────────────────────────────────────────┤
│ Disaggregated P/D    → Extreme          │
│                        throughput,      │
│                        long prompts,    │
│                        large budget     │
└─────────────────────────────────────────┘
```

**Key Takeaway:** MIG is a powerful cost-reduction tool for multi-model NIM deployments, but sacrifices some throughput. For single critical models or disaggregated serving, use full GPUs.

---

## How NIM Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                   NIM Request Flow Diagram                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Your Application                                                  │
│   ┌─────────────────┐                                               │
│   │  HTTP Request   │  "What is AI?"                                │
│   │  POST /v1/chat  │                                               │
│   └────────┬────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              NIM Container                                  │   │
│   │                                                             │   │
│   │  Step 1: API Layer                                         │   │
│   │  ┌──────────────────────────────────┐                      │   │
│   │  │ REST/gRPC Endpoint               │                      │   │
│   │  │ - Authentication                 │                      │   │
│   │  │ - Validation                     │                      │   │
│   │  └──────────┬───────────────────────┘                      │   │
│   │             │                                               │   │
│   │             ▼                                               │   │
│   │  Step 2: Preprocessing                                     │   │
│   │  ┌──────────────────────────────────┐                      │   │
│   │  │ - Tokenization                   │                      │   │
│   │  │ - Format conversion              │                      │   │
│   │  │ - Batching                       │                      │   │
│   │  └──────────┬───────────────────────┘                      │   │
│   │             │                                               │   │
│   │             ▼                                               │   │
│   │  Step 3: Inference Engine                                  │   │
│   │  ┌──────────────────────────────────┐                      │   │
│   │  │  ┌────────────────────────────┐  │                      │   │
│   │  │  │  TensorRT / TensorRT-LLM   │  │                      │   │
│   │  │  │                            │  │                      │   │
│   │  │  │  ╔════════════════════╗    │  │ ⚡ GPU Processing  │   │
│   │  │  │  ║   NVIDIA GPU       ║    │  │                      │   │
│   │  │  │  ║   A100/H100/RTX    ║    │  │  - FP16/INT8 Opt    │   │
│   │  │  │  ║   Computing...     ║    │  │  - Kernel Fusion    │   │
│   │  │  │  ╚════════════════════╝    │  │  - Memory Opt       │   │
│   │  │  └────────────────────────────┘  │                      │   │
│   │  └──────────┬───────────────────────┘                      │   │
│   │             │                                               │   │
│   │             ▼                                               │   │
│   │  Step 4: Postprocessing                                    │   │
│   │  ┌──────────────────────────────────┐                      │   │
│   │  │ - Detokenization                 │                      │   │
│   │  │ - Format response                │                      │   │
│   │  │ - Add metadata                   │                      │   │
│   │  └──────────┬───────────────────────┘                      │   │
│   │             │                                               │   │
│   └─────────────┼───────────────────────────────────────────────┘   │
│                 │                                                   │
│                 ▼                                                   │
│   ┌─────────────────────┐                                           │
│   │  JSON Response      │  "AI stands for..."                      │
│   │  Streamed Output    │  (Low Latency!)                          │
│   └─────────────────────┘                                           │
│                                                                     │
│   ⏱️  Typical Latency: 10-100ms  │  🔥 Throughput: 1000+ tok/sec   │
└─────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Workflow

#### **1. Model Selection**
- Choose from 100+ pre-optimized models at build.nvidia.com
- Or bring your own custom model

#### **2. Container Download**
- Pull the NIM container image from NVIDIA NGC catalog
- Container includes model, runtime, and all dependencies

#### **3. Deployment**
```bash
# Example: Deploy Llama 3 8B model
docker run --gpus all -p 8000:8000 \
  -e NGC_API_KEY=<your-key> \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```

#### **4. API Access**
```python
# Use standard OpenAI-compatible API
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used"
)

response = client.chat.completions.create(
    model="meta/llama3-8b-instruct",
    messages=[{"role": "user", "content": "What is AI?"}]
)
```

#### **5. Production Scaling**
- Deploy with Kubernetes for auto-scaling
- Add monitoring and logging
- Configure load balancing
- Enable authentication and security

### Behind the Scenes

1. **Request arrives** at the NIM container
2. **Preprocessing** happens with domain-optimized code
3. **Inference engine** (TensorRT/TensorRT-LLM) processes on GPU
4. **Postprocessing** formats the output
5. **Response returns** via standard API

---

## Use Cases

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NIM Real-World Use Cases                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ENTERPRISE & BUSINESS                                              │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │   💬 Chatbots      │  │   📄 RAG Systems   │                    │
│  │  Customer Service  │  │  Document Search   │                    │
│  │  Internal Support  │  │  Q&A Systems       │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                     │
│  CREATIVE & CONTENT                                                 │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │  ✍️  Content Gen   │  │   💻 Code Gen      │                    │
│  │  Marketing Copy    │  │  Auto-complete     │                    │
│  │  Report Writing    │  │  Bug Detection     │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                     │
│  MULTIMODAL & VISION                                                │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │  🖼️  Visual Q&A    │  │   🎨 Image Gen     │                    │
│  │  Chart Analysis    │  │  Video Analysis    │                    │
│  │  Doc Understanding │  │  3D Modeling       │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                     │
│  INDUSTRY-SPECIFIC                                                  │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │  🏥 Healthcare     │  │   🚗 Automotive    │                    │
│  │  Medical Imaging   │  │  Autonomous Driving│                    │
│  │  Drug Discovery    │  │  In-car Assistants │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                     │
│  ┌────────────────────┐  ┌────────────────────┐                    │
│  │  💰 Finance        │  │   🎮 Gaming        │                    │
│  │  Fraud Detection   │  │  NPC Dialogue      │                    │
│  │  Risk Assessment   │  │  Dynamic Content   │                    │
│  └────────────────────┘  └────────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. **Enterprise Chatbots & Virtual Assistants**
- Customer service automation
- Internal knowledge assistants
- Conversational AI applications

### 2. **Retrieval-Augmented Generation (RAG)**
- Enterprise search with LLMs
- Document Q&A systems
- Knowledge base integration

```
┌─────────────────────────────────────────────────────────────┐
│            RAG System with NIM (Example)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query: "What is our refund policy?"                   │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────┐                                   │
│  │  Embedding NIM       │ Convert query to vector           │
│  │  (NeMo Retriever)    │                                   │
│  └──────────┬───────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌──────────────────────┐                                   │
│  │  Vector Database     │ Find relevant docs                │
│  │  (Retrieve top-k)    │                                   │
│  └──────────┬───────────┘                                   │
│             │                                               │
│             ▼                                               │
│  ┌──────────────────────┐                                   │
│  │  LLM NIM             │ Generate answer with context      │
│  │  (Llama 3 70B)       │                                   │
│  └──────────┬───────────┘                                   │
│             │                                               │
│             ▼                                               │
│  Response: "Our refund policy allows..."                    │
│                                                             │
│  ✓ All NIM components with standard APIs                   │
│  ✓ Enterprise security & performance                       │
└─────────────────────────────────────────────────────────────┘
```

### 3. **Content Generation**
- Marketing copy and creative writing
- Code generation and assistance
- Report and summary generation

### 4. **Multimodal Applications**
- Visual question answering
- Image and video analysis with text
- Document understanding with vision + text

### 5. **Healthcare & Life Sciences**
- Medical image analysis
- Drug discovery and molecular design
- Clinical decision support

### 6. **Automotive**
- Autonomous vehicle perception
- Driver assistance systems
- In-vehicle conversational AI

### 7. **Financial Services**
- Document processing and analysis
- Risk assessment and fraud detection
- Customer interaction automation

### 8. **Gaming & Entertainment**
- NPC (Non-Player Character) dialogue
- Dynamic content generation
- Digital human avatars

---

## Getting Started

```
┌─────────────────────────────────────────────────────────────────────┐
│              NIM Quick Start Journey (5 Minutes!)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  STEP 1: Prerequisites ✓                                            │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  [√] NVIDIA GPU (RTX 4090, A100, H100, L40S)             │     │
│  │  [√] Docker + NVIDIA Container Toolkit                    │     │
│  │  [√] NGC Account & API Key                               │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          ▼                                          │
│  STEP 2: Get API Key 🔑                                             │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  $ export NGC_API_KEY=<your-api-key>                     │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          ▼                                          │
│  STEP 3: Deploy NIM 🚀                                              │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  $ docker run --gpus all -p 8000:8000 \                  │     │
│  │      -e NGC_API_KEY=$NGC_API_KEY \                       │     │
│  │      nvcr.io/nim/meta/llama3-8b-instruct:latest          │     │
│  │                                                           │     │
│  │  [████████████████████] Downloading model...             │     │
│  │  [████████████████████] Loading to GPU...                │     │
│  │  ✓ Server running on http://localhost:8000               │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          ▼                                          │
│  STEP 4: Test API ⚡                                                │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  $ curl -X POST http://localhost:8000/v1/chat/completions│     │
│  │      -H "Content-Type: application/json" \               │     │
│  │      -d '{"model":"meta/llama3-8b-instruct",             │     │
│  │           "messages":[{"role":"user",                    │     │
│  │                        "content":"Hello!"}]}'            │     │
│  │                                                           │     │
│  │  ✓ Response: "Hello! How can I help you today?"          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          ▼                                          │
│  STEP 5: Build Your App! 🎉                                         │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Now integrate with your application using:              │     │
│  │  • Python (OpenAI SDK, LangChain)                        │     │
│  │  • JavaScript (fetch, axios)                             │     │
│  │  • Any language with HTTP support!                       │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  💡 TIP: Start with serverless API at build.nvidia.com             │
│      No GPU required for development!                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

1. **Hardware Requirements**
   - NVIDIA GPU with sufficient VRAM (varies by model)
   - Common: RTX 4090, A100, H100, L40S

2. **Software Requirements**
   - Docker or Podman
   - NVIDIA Container Toolkit
   - NVIDIA GPU drivers

3. **NVIDIA NGC Account**
   - Sign up at ngc.nvidia.com
   - Generate API key for container access

### Quick Start (5 Minutes)

#### **Step 1: Install Prerequisites**
```bash
# Install NVIDIA Container Toolkit
# (Instructions vary by OS)
```

#### **Step 2: Get NGC API Key**
```bash
export NGC_API_KEY=<your-api-key>
```

#### **Step 3: Deploy a NIM**
```bash
# Deploy Llama 3 8B model
docker run --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```

#### **Step 4: Test the API**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta/llama3-8b-instruct",
    "messages": [{"role":"user", "content":"Hello!"}],
    "max_tokens": 100
  }'
```

### Development vs Production

| Aspect | Development | Production |
|--------|-------------|------------|
| **Hosting** | Serverless APIs (free) | Self-hosted or cloud |
| **Scale** | Limited requests | Auto-scaling |
| **Security** | Public APIs | Private infrastructure |
| **Support** | Community | Enterprise SLA |
| **Cost** | Free tier available | License-based |

---

## Benefits for Enterprises

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Why Enterprises Choose NIM                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   🚀 FASTER TIME-TO-MARKET          💰 COST OPTIMIZATION            │
│   ┌───────────────────────┐        ┌───────────────────────┐       │
│   │ Months → Minutes      │        │ ↑ GPU Utilization     │       │
│   │ Pre-optimized models  │        │ ↓ Infrastructure cost │       │
│   │ Standard APIs         │        │ ↓ Engineering hours   │       │
│   └───────────────────────┘        └───────────────────────┘       │
│                                                                     │
│   🔒 DATA SECURITY                  🎯 ENTERPRISE SUPPORT           │
│   ┌───────────────────────┐        ┌───────────────────────┐       │
│   │ Private deployment    │        │ SLA guarantees        │       │
│   │ Data sovereignty      │        │ 24/7 support          │       │
│   │ Compliance ready      │        │ Regular updates       │       │
│   └───────────────────────┘        └───────────────────────┘       │
│                                                                     │
│   🔮 FUTURE-PROOF                   ⚙️  SIMPLIFIED OPS             │
│   ┌───────────────────────┐        ┌───────────────────────┐       │
│   │ Easy upgrades         │        │ Kubernetes native     │       │
│   │ API stability         │        │ Auto-scaling          │       │
│   │ Latest innovations    │        │ Built-in monitoring   │       │
│   └───────────────────────┘        └───────────────────────┘       │
│                                                                     │
│          ROI: Months of engineering saved per deployment!          │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. **Faster Time-to-Market**
- Pre-optimized models eliminate months of engineering
- Standard APIs reduce integration complexity
- Validated configurations ensure reliability

### 2. **Cost Optimization**
- Maximum GPU utilization through optimizations
- Reduced infrastructure costs
- Lower operational overhead

### 3. **Data Security & Sovereignty**
- Deploy on private infrastructure
- No data leaves your environment
- Compliance with regulations (GDPR, HIPAA, etc.)

### 4. **Enterprise Support**
- SLAs and support contracts
- Regular security updates
- Technical assistance from NVIDIA

### 5. **Future-Proof**
- Easy model upgrades
- API stability guarantees
- Access to latest NVIDIA innovations

### 6. **Simplified Operations**
- Kubernetes-native deployment
- Standard monitoring and logging
- DevOps-friendly containerization

---

## NIM vs Traditional Inference Serving

| Feature | Traditional Approach | NVIDIA NIM |
|---------|---------------------|------------|
| **Setup Time** | Weeks to months | Minutes to hours |
| **Optimization** | Manual tuning required | Pre-optimized |
| **APIs** | Custom implementation | Industry-standard |
| **Security Updates** | Self-managed | Continuous updates |
| **Support** | Community only | Enterprise SLA |
| **Model Coverage** | Limited | 100+ models |
| **Hardware Optimization** | Generic | NVIDIA GPU-specific |
| **Enterprise Features** | Build yourself | Built-in |
| **Documentation** | Varies | Comprehensive |

---

## Resources

### Official Documentation
- NVIDIA NIM Documentation: https://docs.nvidia.com/nim/
- NVIDIA AI Enterprise: https://www.nvidia.com/en-us/data-center/products/ai-enterprise/

### Try NIM
- NVIDIA AI Playground: https://build.nvidia.com/explore/discover
- NGC Catalog: https://catalog.ngc.nvidia.com/

### Learning Resources
- Developer Blog: https://developer.nvidia.com/blog
- Technical Forums: https://forums.developer.nvidia.com/
- GitHub Examples: https://github.com/nvidia

### Community
- NVIDIA Developer Program: https://developer.nvidia.com/
- Discord Community
- Technical Support Portal

---

## NIM Ecosystem Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  NVIDIA NIM Ecosystem Map                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                         APPLICATIONS                                │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│   │  Chatbot │  │    RAG   │  │  Vision  │  │  Custom  │          │
│   │   Apps   │  │  Systems │  │   Apps   │  │   Apps   │          │
│   └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│         │             │             │             │                │
│         └─────────────┴─────────────┴─────────────┘                │
│                          │                                          │
│                          ▼                                          │
│              ┌────────────────────────┐                             │
│              │    Standard APIs       │                             │
│              │  (OpenAI-compatible)   │                             │
│              └───────────┬────────────┘                             │
│                          │                                          │
│         ┌────────────────┼────────────────┐                         │
│         ▼                ▼                ▼                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                     │
│   │   LLM    │    │   VLM    │    │  Speech  │  ... 100+ Models   │
│   │   NIM    │    │   NIM    │    │   NIM    │                     │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘                     │
│        │               │               │                            │
│        └───────────────┼───────────────┘                            │
│                        ▼                                            │
│            ┌────────────────────────┐                               │
│            │  NVIDIA AI Enterprise  │                               │
│            │  Runtime & Support     │                               │
│            └───────────┬────────────┘                               │
│                        │                                            │
│       ┌────────────────┼────────────────┐                           │
│       ▼                ▼                ▼                           │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐                       │
│  │  Cloud  │    │ On-Prem  │    │   Edge   │                       │
│  │  (AWS,  │    │  (DGX,   │    │  (RTX,   │                       │
│  │  Azure) │    │  Custom) │    │  Jetson) │                       │
│  └─────────┘    └──────────┘    └──────────┘                       │
│                                                                     │
│       🌐 build.nvidia.com  │  📦 NGC Catalog  │  📚 Docs            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary

**NVIDIA NIM** represents a paradigm shift in AI deployment, transforming what was once a complex, time-consuming process into a streamlined, production-ready solution. By packaging optimized inference engines, enterprise runtimes, and industry-standard APIs into containerized microservices, NIM enables organizations to:

- Deploy AI models in minutes instead of months
- Maintain data security on private infrastructure
- Achieve optimal performance on NVIDIA hardware
- Access enterprise support and continuous updates
- Focus on building applications rather than managing infrastructure

Whether you're a startup experimenting with AI or an enterprise deploying at scale, NIM provides the tools, performance, and reliability needed to bring generative AI applications to production.

---

**Getting Started is Easy:** Visit [build.nvidia.com](https://build.nvidia.com) to try NIM APIs for free, or download containers from NGC to deploy on your own infrastructure.

**For Enterprise:** Contact NVIDIA about AI Enterprise licensing for production deployments with SLA support.

---

## Your NIM Journey: From Development to Production

```
┌─────────────────────────────────────────────────────────────────────┐
│                    The Complete NIM Journey                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: EXPLORATION (Day 1)                                       │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  🌐 Visit build.nvidia.com                                │     │
│  │  ✓ Try 100+ models via serverless APIs                    │     │
│  │  ✓ No GPU needed                                          │     │
│  │  ✓ Free tier available                                    │     │
│  │  → Test your use case, find the right model               │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          │                                          │
│                          ▼                                          │
│  PHASE 2: DEVELOPMENT (Week 1)                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  💻 Deploy on local workstation                           │     │
│  │  $ docker run --gpus all -p 8000:8000 \                  │     │
│  │      nvcr.io/nim/meta/llama3-8b-instruct:latest          │     │
│  │  ✓ Build your application with OpenAI-compatible API      │     │
│  │  ✓ Test locally with your data                           │     │
│  │  → Develop and iterate quickly                            │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          │                                          │
│                          ▼                                          │
│  PHASE 3: STAGING (Week 2-3)                                        │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  ☸  Deploy on Kubernetes cluster                          │     │
│  │  ✓ Use NIM Operator + Helm charts                        │     │
│  │  ✓ Configure auto-scaling                                │     │
│  │  ✓ Add monitoring & logging                              │     │
│  │  ✓ Test with realistic load                              │     │
│  │  → Validate performance at scale                          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          │                                          │
│                          ▼                                          │
│  PHASE 4: PRODUCTION (Week 4+)                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  🚀 Deploy to production environment                      │     │
│  │  ✓ Private cloud or on-premises                          │     │
│  │  ✓ Enterprise SLA & support active                       │     │
│  │  ✓ Security updates automated                            │     │
│  │  ✓ Multi-region deployment                               │     │
│  │  → Serve millions of requests reliably                    │     │
│  └───────────────────────────────────────────────────────────┘     │
│                          │                                          │
│                          ▼                                          │
│  PHASE 5: OPTIMIZATION (Ongoing)                                    │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  📊 Monitor & improve                                      │     │
│  │  ✓ Analyze performance metrics                           │     │
│  │  ✓ Upgrade to newer models                               │     │
│  │  ✓ Fine-tune for specific use cases                      │     │
│  │  ✓ Expand to additional NIMs                             │     │
│  │  → Continuous improvement with latest AI advances         │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Timeline: Prototype → Production in 3-4 weeks!                    │
│  (vs. 3-6 months with traditional approaches)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Disaggregated Prefill and Decode (Prefill-Decode Disaggregation)

### What is Prefill and Decode? (The Basics First)

Before understanding disaggregation, you need to understand how an LLM generates text. Every time you send a prompt to an LLM, it goes through **two distinct phases**:

```
YOUR PROMPT:  "Explain quantum computing in simple terms"
                              │
              ┌───────────────▼────────────────┐
              │         PHASE 1: PREFILL        │
              │  Process all input tokens at    │
              │  once (parallel computation)    │
              │  "Explain quantum computing in  │
              │   simple terms" → KV Cache      │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │         PHASE 2: DECODE         │
              │  Generate output tokens ONE     │
              │  at a time (sequential)         │
              │  "Quantum" → "computing" →      │
              │  "uses" → "qubits" → ...        │
              └───────────────┬────────────────┘
                              │
              ┌───────────────▼────────────────┐
              │           RESPONSE              │
              │  "Quantum computing uses        │
              │   qubits instead of bits..."    │
              └────────────────────────────────┘
```

| Phase | What it does | Speed | GPU Usage |
|-------|-------------|-------|-----------|
| **Prefill** | Reads & understands your entire prompt at once | Fast (parallel) | Compute-heavy (high GPU utilization) |
| **Decode** | Generates response token by token | Slow (sequential, one word at a time) | Memory-heavy (low GPU utilization) |

---

### The Problem: One GPU Trying to Do Both

In traditional LLM inference, **the same GPU(s) handle both prefill AND decode** for every request. This causes a fundamental conflict:

```
TRADITIONAL (COUPLED) APPROACH:
┌──────────────────────────────────────────────────────────────┐
│                     SINGLE GPU CLUSTER                        │
│                                                              │
│  Request A: [PREFILL████████] [DECODE░░░░░░░░░░░░░░░░░░░░]  │
│  Request B:                  [PREFILL████████] [DECODE░░░░]  │
│  Request C:                            [PREFILL████] [DECODE]│
│                                                              │
│  ████ = High GPU compute (prefill is efficient)             │
│  ░░░░ = Low GPU compute (decode wastes GPU capacity)        │
└──────────────────────────────────────────────────────────────┘

PROBLEM:
- During DECODE, the GPU is mostly idle (waiting for memory)
- Prefill interrupts decode of other requests → "prefill interference"
- Long prompts stall short responses → HIGH LATENCY (TTFT)
- GPU is never fully utilized
```

**Two key metrics get hurt:**
- **TTFT (Time To First Token)**: How long until you see the first word of the response. High TTFT = bad user experience.
- **TBT (Time Between Tokens)**: How fast tokens stream after the first one. High TBT = choppy, slow streaming.

---

### The Solution: Disaggregated Prefill and Decode

**Disaggregation** means splitting prefill and decode onto **separate, specialized GPU pools**. Each pool is optimized for its specific job.

```
DISAGGREGATED APPROACH:
┌─────────────────────────────────────────────────────────────────┐
│                        LOAD BALANCER / ROUTER                   │
│              (Routes requests intelligently)                     │
└──────────────────┬──────────────────────────┬───────────────────┘
                   │                          │
       ┌───────────▼───────────┐  ┌───────────▼───────────┐
       │   PREFILL GPU POOL    │  │   DECODE GPU POOL      │
       │   (P-Instances)       │  │   (D-Instances)        │
       │                       │  │                        │
       │  Optimized for:       │  │  Optimized for:        │
       │  ✓ High compute       │  │  ✓ Large KV cache      │
       │  ✓ Parallel ops       │  │  ✓ Memory bandwidth    │
       │  ✓ Fast matrix math   │  │  ✓ Low latency output  │
       │                       │  │                        │
       │  GPU: High-end        │  │  GPU: Can use lower    │
       │  (H100, A100)         │  │  tier if needed        │
       └─────────┬─────────────┘  └────────────┬───────────┘
                 │                              │
                 │   KV Cache Transfer          │
                 └──────────────────────────────┘
                 (Prefill sends KV cache to Decode)
```

**Why this works:** Prefill is compute-bound (parallel, fast). Decode is memory-bound (sequential, slow). Their resource needs are **opposite** — keeping them on the same GPU causes each phase to hurt the other. Separating them lets each pool specialize: P-instances tuned for peak compute throughput, D-instances tuned for fast memory access and KV cache storage.

---

### Before vs After: Performance Impact

```
BEFORE DISAGGREGATION (Coupled):
Timeline ────────────────────────────────────────────►
         [Request A: Prefill][   Request A: Decode   ]
                             [Request B: Prefill][B:Dec]
                                    ↑
                            Prefill of B stalls
                            decode of A → jitter!

TTFT for B = wait for A's decode + B's prefill = SLOW


AFTER DISAGGREGATION:
P-Pool:  [A:Pre][B:Pre][C:Pre][D:Pre]  ← always busy doing prefill
D-Pool:  [A: Decode...][B: Decode...] ← always busy doing decode
                ↑
         No interference!
         P-pool immediately processes next prompt
         D-pool continuously streams tokens

TTFT = only B's prefill time (no waiting) = FAST
TBT  = no prefill interruptions = SMOOTH streaming
```

**Results you get:**
| Metric | Coupled | Disaggregated | Improvement |
|--------|---------|---------------|-------------|
| TTFT (short prompts) | ~500ms | ~50ms | **10x faster** |
| TTFT (long prompts) | ~5000ms | ~200ms | **25x faster** |
| Throughput (tokens/sec) | Baseline | 2–4x higher | **2-4x** |
| GPU Utilization | ~40-60% | ~85-95% | **Much better** |

### How NIM Supports Disaggregated Prefill-Decode

NVIDIA NIM (with TensorRT-LLM backend) has **native support** for disaggregated serving:

- TensorRT-LLM's `executor` API supports P/D split natively via `ExecutorType.DISAGG_PREFILL` / `DISAGG_DECODE`
- Automatic KV Cache routing between P and D instances
- Dynamic scaling: adjust P vs D ratio based on workload
- Works with NVIDIA Dynamo (next-gen serving runtime)
- KV transfer over NVLink (within node ~900 GB/s) or InfiniBand (across nodes ~400 GB/s) via NIXL

> For full infrastructure wiring (process flags, network topology, K8s manifests) see [Infrastructure Configuration](#infrastructure-configuration-how-to-wire-up-p-and-d-gpu-clusters) below.

---

### When Should You Use Disaggregation?

```
USE DISAGGREGATION WHEN:                 SKIP IT WHEN:
──────────────────────────               ──────────────────
✓ Long prompts (RAG, documents)          ✗ Short prompts only
✓ Low TTFT is critical (chatbots)        ✗ Batch offline processing
✓ High concurrent users                  ✗ Small scale / dev environment
✓ Mixed long/short prompt workloads      ✗ Budget constraints (needs more GPUs)
✓ Production enterprise deployments      ✗ Simple single-GPU setups
✓ SLA requirements on first-token latency
```

---

### NVIDIA Dynamo: The Next Evolution

NVIDIA **Dynamo** (announced 2025) is NVIDIA's open-source serving framework built from the ground up for disaggregated inference at scale:

```
┌──────────────────────────────────────────────────────────┐
│                    NVIDIA DYNAMO                          │
│           (Disaggregation-Native Runtime)                │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Prefill  │  │ Prefill  │  │ Prefill  │  ← P-Pool     │
│  │ Worker   │  │ Worker   │  │ Worker   │               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │              │                     │
│       └──────────────┴──────────────┘                    │
│                      │ KV Cache Bus                      │
│       ┌──────────────┴──────────────┐                    │
│       │              │              │                     │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐               │
│  │ Decode   │  │ Decode   │  │ Decode   │  ← D-Pool     │
│  │ Worker   │  │ Worker   │  │ Worker   │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│                                                          │
│  Features:                                               │
│  • Distributed KV cache across GPU memory                │
│  • Smart request routing based on real-time load         │
│  • Auto-scale P and D pools independently                │
│  • Works with TensorRT-LLM, vLLM backends                │
└──────────────────────────────────────────────────────────┘
```

---

### Summary: The Big Picture

```
DISAGGREGATED PREFILL-DECODE IN ONE DIAGRAM:

User Prompt                                    Response
    │                                              ▲
    ▼                                              │
┌───────────────────────────────────────────────────────┐
│                    NIM / Dynamo Router                 │
└──────────────┬────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐   KV Cache   ┌──────────────────┐
    │   PREFILL POOL      │─────────────►│   DECODE POOL    │
    │                     │              │                  │
    │ Job: Understand the │              │ Job: Write the   │
    │ prompt (FAST)       │              │ response (SLOW   │
    │                     │              │ but smooth)      │
    │ Strength: Compute   │              │ Strength: Memory │
    └─────────────────────┘              └──────────────────┘

KEY INSIGHT:
"Don't make a chef cook AND wash dishes at the same time.
 Have a chef (prefill) prepare the meal and a waiter
 (decode) serve it — simultaneously, without blocking!"

BENEFIT SUMMARY:
┌────────────────────────────────────────┐
│ ✓ 10-25x lower TTFT                   │
│ ✓ 2-4x higher throughput              │
│ ✓ Smooth token streaming (no jitter)  │
│ ✓ Better GPU utilization              │
│ ✓ Independent scaling of P and D      │
│ ✓ Handles long prompts gracefully     │
└────────────────────────────────────────┘
```

---

### Infrastructure Configuration: How to Wire Up P and D GPU Clusters

This is the critical question: **how does the system actually know which GPU runs prefill and which runs decode?** There are three layers to this:

```
LAYER 1: Process/Instance Configuration  ← tell each process its role
LAYER 2: Networking & KV Transfer        ← connect P and D clusters
LAYER 3: Router / Orchestration          ← direct traffic to right cluster
```

---

#### Layer 1: Telling Each GPU Instance Its Role

Each NIM / TensorRT-LLM / Dynamo **worker process** is launched with a flag that declares its role. There is no "auto-detection" — you explicitly tag each process.

**With TensorRT-LLM (executor API):**

```python
# PREFILL INSTANCE startup config
executor_config = trtllm.ExecutorConfig(
    max_beam_width=1,
    kv_cache_config=trtllm.KvCacheConfig(
        enable_block_reuse=True,
        max_tokens=8192
    ),
    executor_type=trtllm.ExecutorType.DISAGG_PREFILL,  # ← THIS FLAG
)

# DECODE INSTANCE startup config
executor_config = trtllm.ExecutorConfig(
    max_beam_width=1,
    kv_cache_config=trtllm.KvCacheConfig(
        enable_block_reuse=True,
        max_tokens=32768          # ← larger cache for decode
    ),
    executor_type=trtllm.ExecutorType.DISAGG_DECODE,   # ← THIS FLAG
)
```

**With NVIDIA Dynamo (YAML config):**

```yaml
# prefill_worker.yaml  — runs on P-cluster nodes
Frontend:
  host: 0.0.0.0
  port: 8080

VllmWorker:                         # or TrtLlmWorker
  model: meta-llama/Llama-3.1-70B
  tensor_parallel_size: 4           # 4 GPUs for this worker
  max_num_seqs: 32
  role: prefill                     # ← ROLE DECLARATION
  kv_transfer_config:
    kv_connector: NixlConnector     # NVIDIA's fast KV transfer lib
    kv_role: kv_producer            # P-instance PRODUCES KV cache
    kv_rank: 0
    kv_port: 14579                  # port P uses to send KV cache

---
# decode_worker.yaml  — runs on D-cluster nodes
VllmWorker:
  model: meta-llama/Llama-3.1-70B
  tensor_parallel_size: 4
  max_num_seqs: 64                  # more sequences, decode is slower
  role: decode                      # ← ROLE DECLARATION
  kv_transfer_config:
    kv_connector: NixlConnector
    kv_role: kv_consumer            # D-instance CONSUMES KV cache
    kv_rank: 1
    kv_port: 14579                  # port D uses to receive KV cache
```

**Key point:** The `role: prefill` / `role: decode` flag (or `ExecutorType.DISAGG_PREFILL`) is what makes a GPU a P-instance or D-instance. The model weights loaded are **identical** — only the role differs.

---

#### Layer 2: Network & KV Cache Transfer — How Data Moves Between Clusters

This is the hardest part. After prefill finishes, the KV cache (can be GBs of data) must move to the decode GPU **fast enough** that the user doesn't notice a gap.

```
PHYSICAL NETWORK TOPOLOGY:

Node A (P-instance): GPU 0,1,2,3
┌───────────────────────────────┐
│  GPU0 ──NVLink──► GPU1        │
│   │                │          │
│  GPU2 ──NVLink──► GPU3        │
│         │                     │
│    NIC (InfiniBand / RoCE)    │
└──────────────┬────────────────┘
               │  High-speed fabric
               │  (InfiniBand HDR: 200Gb/s)
               │  (NVIDIA Quantum switch)
┌──────────────▼────────────────┐
│    NIC (InfiniBand / RoCE)    │
│         │                     │
│  GPU4 ──NVLink──► GPU5        │
│   │                │          │
│  GPU6 ──NVLink──► GPU7        │
└───────────────────────────────┘
Node B (D-instance): GPU 4,5,6,7

Transfer path:
GPU0 (P) ──NVLink──► CPU mem ──NIC──► Switch ──NIC──► CPU mem ──NVLink──► GPU4 (D)

Or with GPU-Direct RDMA:
GPU0 (P) ──NIC──► Switch ──NIC──► GPU4 (D)   ← bypasses CPU entirely!
```

**NVIDIA NIXL (NVIDIA Inference Xfer Library)** is the library that manages this:

```
NIXL handles:
┌─────────────────────────────────────────────────────┐
│                    NIXL                             │
│                                                     │
│  • GPU-Direct RDMA: GPU ──► GPU without CPU         │
│  • NVLink transfers (within node): ~900 GB/s        │
│  • InfiniBand transfers (across nodes): ~400 GB/s   │
│  • Automatic path selection (NVLink vs IB)          │
│  • Zero-copy transfers where possible               │
│  • Handles tensor parallelism sharding              │
│    (if model is split across 8 GPUs, KV cache       │
│     is also split — NIXL reassembles on decode side)│
└─────────────────────────────────────────────────────┘
```

**Practical transfer time:**
```
KV cache for 4096-token prompt (LLaMA-70B, FP8):
  ≈ 4096 tokens × 80 layers × 2 (K+V) × 1024 dim × 1 byte
  ≈ ~670 MB

Over NVLink (900 GB/s):   ~0.7ms   ✓ nearly instant
Over InfiniBand (400Gb/s): ~13ms   ✓ acceptable
Over standard Ethernet:    ~50ms+  ✗ too slow, defeats the purpose
```

**This is why disaggregation requires high-speed GPU interconnects (NVLink / InfiniBand). It does NOT work well over standard Ethernet.**

---

#### Layer 3: Router — How Traffic Is Directed to the Right Cluster

The router sits in front of everything and decides: "this request goes to a P-instance, then a D-instance."

```
REQUEST LIFECYCLE THROUGH THE ROUTER:

Client ──► Router
              │
              ├─ Step 1: Pick an available P-instance
              │          (based on load, queue depth)
              │
              ├─ Step 2: Send full prompt to P-instance
              │          P-instance runs prefill
              │          P-instance sends KV cache to chosen D-instance
              │
              └─ Step 3: Router tells D-instance: "start decoding"
                         D-instance streams tokens back to client


ROUTER IMPLEMENTATION OPTIONS:
┌──────────────────────┬──────────────────────────────────────────┐
│  Option              │  How it works                            │
├──────────────────────┼──────────────────────────────────────────┤
│ Dynamo Router        │ Built-in, disaggregation-aware,          │
│ (recommended)        │ knows P/D topology natively              │
├──────────────────────┼──────────────────────────────────────────┤
│ NGINX / Envoy        │ Simple L7 load balancer, routes by       │
│ (basic)              │ URL path (/prefill vs /decode)           │
├──────────────────────┼──────────────────────────────────────────┤
│ Kubernetes Ingress   │ Routes via Service labels                │
│                      │ (app: nim-prefill vs app: nim-decode)    │
├──────────────────────┼──────────────────────────────────────────┤
│ Custom gRPC Router   │ Full control, enterprise deployments     │
└──────────────────────┴──────────────────────────────────────────┘
```

---

#### Putting It All Together: Full Infrastructure View

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KUBERNETES CLUSTER                               │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      INGRESS / ROUTER                          │    │
│  │              (Dynamo Router or NGINX)                          │    │
│  └───────────────┬──────────────────────────────┬────────────────┘    │
│                  │                              │                       │
│     ┌────────────▼──────────────┐  ┌────────────▼──────────────┐      │
│     │   PREFILL NODE POOL       │  │   DECODE NODE POOL        │      │
│     │                           │  │                           │      │
│     │  ┌─────────────────────┐  │  │  ┌─────────────────────┐ │      │
│     │  │ Pod: nim-prefill-0  │  │  │  │ Pod: nim-decode-0   │ │      │
│     │  │ GPU: 8x H100        │  │  │  │ GPU: 8x H100        │ │      │
│     │  │ role: prefill       │  │  │  │ role: decode        │ │      │
│     │  │ kv_role: producer   │  │  │  │ kv_role: consumer   │ │      │
│     │  └─────────────────────┘  │  │  └─────────────────────┘ │      │
│     │  ┌─────────────────────┐  │  │  ┌─────────────────────┐ │      │
│     │  │ Pod: nim-prefill-1  │  │  │  │ Pod: nim-decode-1   │ │      │
│     │  │ GPU: 8x H100        │◄─┼──┼──┤ (receives KV cache) │ │      │
│     │  │ role: prefill       │  │  │  │ role: decode        │ │      │
│     │  └─────────────────────┘  │  │  └─────────────────────┘ │      │
│     │                           │  │  ┌─────────────────────┐ │      │
│     │  Node labels:             │  │  │ Pod: nim-decode-2   │ │      │
│     │  nim-role: prefill        │  │  │ ...                 │ │      │
│     │  nvidia.com/gpu: H100     │  │  └─────────────────────┘ │      │
│     └───────────────────────────┘  │                           │      │
│                                    │  Node labels:             │      │
│              KV Cache Transfer     │  nim-role: decode         │      │
│         (InfiniBand fabric) ───────►  nvidia.com/gpu: H100     │      │
│                                    └───────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Kubernetes manifest snippet (node affinity to force P/D separation):**

```yaml
# nim-prefill deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-prefill
spec:
  replicas: 2
  template:
    spec:
      nodeSelector:
        nim-role: prefill          # ← only schedule on prefill nodes
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama-3.1-70b-instruct:latest
        env:
        - name: NIM_DISAGG_ROLE
          value: "prefill"         # ← env var sets role inside container
        - name: NIM_KV_ROLE
          value: "producer"
        - name: NIM_KV_PORT
          value: "14579"
        resources:
          limits:
            nvidia.com/gpu: 8

---
# nim-decode deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nim-decode
spec:
  replicas: 6                      # ← typically more decode than prefill
  template:
    spec:
      nodeSelector:
        nim-role: decode           # ← only schedule on decode nodes
      containers:
      - name: nim
        image: nvcr.io/nim/meta/llama-3.1-70b-instruct:latest
        env:
        - name: NIM_DISAGG_ROLE
          value: "decode"          # ← same image, different role
        - name: NIM_KV_ROLE
          value: "consumer"
        - name: NIM_KV_PORT
          value: "14579"
        resources:
          limits:
            nvidia.com/gpu: 8
```

**The same container image is used for both P and D — the env var `NIM_DISAGG_ROLE` is what makes the difference.**

---

#### How Does a P-Instance Know Which D-Instance to Send KV Cache To?

This is the "pairing" problem. There are two approaches:

```
APPROACH 1: STATIC PAIRING (simple, less flexible)
──────────────────────────────────────────────────
P-instance 0 always sends to D-instance 0,1,2
P-instance 1 always sends to D-instance 3,4,5

Configured via:
  decode_endpoints: ["d-instance-0:14579", "d-instance-1:14579"]

Pro: simple, predictable
Con: no dynamic load balancing between P and D


APPROACH 2: DYNAMIC PAIRING via Router (recommended)
──────────────────────────────────────────────────────

  Client ──► Router
                │
    1. Router picks least-loaded P-instance
    2. Router picks least-loaded D-instance
    3. Router tells P: "send KV cache to D-instance-3"
    4. P runs prefill, pushes KV cache to D-instance-3
    5. P notifies Router: "done, token 1 ready"
    6. Router connects client stream to D-instance-3
    7. D streams response

This is what NVIDIA Dynamo's scheduler does natively.
```

---

#### Ratio of P to D Instances: How Many of Each?

This is a tuning decision based on your workload:

```
RULE OF THUMB:
─────────────
Prefill is FAST (milliseconds)
Decode is SLOW (seconds, one token at a time)

So you typically need MORE decode instances than prefill.

Common ratios:
┌────────────────────┬────────┬────────┬──────────────────────────┐
│ Workload           │ P pods │ D pods │ Reason                   │
├────────────────────┼────────┼────────┼──────────────────────────┤
│ Short prompts,     │  1     │  3     │ Prefill is very fast,    │
│ short responses    │        │        │ decode bottlenecks        │
├────────────────────┼────────┼────────┼──────────────────────────┤
│ Long prompts (RAG) │  2     │  4     │ Prefill takes longer     │
│ short responses    │        │        │ for big docs             │
├────────────────────┼────────┼────────┼──────────────────────────┤
│ Long prompts,      │  2     │  8     │ Both heavy; decode       │
│ long responses     │        │        │ dominates                │
└────────────────────┴────────┴────────┴──────────────────────────┘

You can auto-scale P and D independently with Kubernetes HPA
based on GPU utilization or queue depth metrics.
```

---

#### Quick Reference: What Enforces the Separation?

```
MECHANISM          WHERE               WHAT IT DOES
─────────────────────────────────────────────────────────────
ExecutorType flag  TensorRT-LLM code   Sets process mode (P or D)
role: prefill/     Dynamo YAML         Same — sets process mode
decode env var

nodeSelector       Kubernetes YAML     Ensures P pods only go to
nim-role: prefill                      P-labeled nodes (physical
                                       GPU isolation)

NVLink / IB fabric Physical hardware   Fast KV cache transfer
                                       between P and D nodes

NIXL library       Runtime             Manages actual GPU-to-GPU
                                       data movement

Router / Scheduler Dynamo / custom     Pairs requests: P-instance
                                       → D-instance dynamically
```