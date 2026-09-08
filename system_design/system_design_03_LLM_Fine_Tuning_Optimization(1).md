03_LLM_Fine_Tuning_Optimization.md
# LLM Fine-Tuning & Optimization Interview Questions

## Q1: Design a Production LLM Fine-Tuning Pipeline for Domain Adaptation

**Context**: Fine-tune an open-source LLM (Llama 3.1) to become an expert in payment domain tasks (dispute classification, risk assessment, merchant support). Must handle 100K+ training examples, reduce inference cost by 40%, and maintain safety.

### Follow-up Questions:
- What's your approach to data collection and curation?
- Which fine-tuning technique (LoRA, QLoRA, Full) would you choose and why?
- How do you prevent catastrophic forgetting?
- What's your evaluation strategy?
- How do you handle class imbalance in training data?

### Architecture Diagram:
```
┌───────────────────────────────────────────────────────────┐
│      PRODUCTION LLM FINE-TUNING PIPELINE                  │
│            (NVIDIA NeMo + Megatron Core)                  │
└───────────────────────────────────────────────────────────┘

1. DATA PREPARATION STAGE
┌─────────────────────────────────────────┐
│  Raw Data Collection                    │
│  ├─ Disputes dataset (50K examples)     │
│  ├─ Chat logs (labeled interactions)    │
│  ├─ Policy documents (context)          │
│  └─ Expert annotations (gold labels)    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Data Curation & Filtering              │
│  ├─ Remove duplicates                   │
│  ├─ Filter low-quality examples         │
│  ├─ Balance class distribution          │
│  ├─ Remove PII (before training!)       │
│  └─ Format into {input, output} pairs   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Data Splitting                         │
│  ├─ Train: 80% (80K examples)          │
│  ├─ Validation: 10% (10K)              │
│  ├─ Test: 10% (10K)                    │
│  └─ Stratified by intent class          │
└────────────────┬────────────────────────┘

2. FINE-TUNING TECHNIQUE SELECTION
┌─────────────────────────────────────────┐
│  Decision Matrix:                       │
│                                         │
│  LoRA (Low-Rank Adaptation):           │
│  ├─ Cost: $500 (single GPU)            │
│  ├─ Time: 6-8 hours                    │
│  ├─ Memory: 16GB GPU                   │
│  ├─ Inference latency: +5-10%          │
│  ├─ Accuracy gain: 4-6%                │
│  └─ Chosen: YES (cost/benefit optimal) │
│                                         │
│  QLoRA (Quantized LoRA):               │
│  ├─ Cost: $200 (single GPU)            │
│  ├─ Time: 12 hours                     │
│  ├─ Memory: 8GB GPU                    │
│  ├─ Inference latency: +15-20%         │
│  ├─ Accuracy gain: 3-5%                │
│  └─ Chosen: NO (overkill for budget)   │
│                                         │
│  Full Fine-tune:                       │
│  ├─ Cost: $5K (8x GPU cluster)         │
│  ├─ Time: 48 hours                     │
│  ├─ Memory: 80GB GPU cluster           │
│  ├─ Inference latency: No overhead     │
│  ├─ Accuracy gain: 8-12%               │
│  └─ Chosen: NO (too expensive)         │
└────────────────┬────────────────────────┘

3. TRAINING INFRASTRUCTURE
┌──────────────────────────────────────────┐
│  LoRA Configuration                      │
│  ├─ Base Model: Llama 3.1 (8B)          │
│  ├─ r (LoRA rank): 16                   │
│  ├─ alpha (scaling): 32                 │
│  ├─ target_modules: [q_proj, v_proj]   │
│  ├─ dropout: 0.1                        │
│  └─ Learnable params: ~2.1M (0.3% of base) │
└────────────────┬───────────────────────┘

┌──────────────────────────────────────────┐
│  Training Configuration                  │
│  ├─ Optimizer: AdamW (lr=5e-4)          │
│  ├─ Batch size: 32 (per GPU)            │
│  ├─ Gradient accumulation: 4            │
│  ├─ Max steps: 3000                     │
│  ├─ Warmup steps: 300 (10%)             │
│  ├─ Learning rate schedule: cosine      │
│  ├─ Weight decay: 0.1                   │
│  ├─ Gradient clipping: 1.0              │
│  ├─ FP16 mixed precision: enabled       │
│  └─ Distributed training: FSDP (1 GPU) │
└────────────────┬───────────────────────┘

┌──────────────────────────────────────────┐
│  Training Loop                           │
│                                          │
│  For each epoch:                         │
│  1. Forward pass (batch of sequences)   │
│  2. Compute loss (cross-entropy)        │
│  3. Backward pass (compute gradients)   │
│  4. Update LoRA weights (frozen base)   │
│  5. Validation check every 500 steps    │
│  6. Checkpoint best model (by val loss) │
│  7. Early stopping if no improvement    │
└────────────────┬───────────────────────┘

4. TRAINING MONITORING
┌───────────────────────────────────────┐
│  Real-time Metrics (W&B Dashboard)    │
│  ├─ Training loss (target: <0.5)      │
│  ├─ Validation loss (target: <0.6)    │
│  ├─ Learning rate schedule            │
│  ├─ Batch processing time             │
│  ├─ GPU memory utilization            │
│  ├─ GPU compute utilization           │
│  └─ Gradient norms (check for NaN)   │
│                                       │
│  Alerts:                              │
│  ├─ Loss not decreasing > 5 steps    │
│  ├─ GPU OOM                          │
│  ├─ Validation loss increasing       │
│  └─ Gradient explosion (norm > 10)   │
└───────────────┬───────────────────────┘

5. EVALUATION STAGE
┌────────────────────────────────────────┐
│  Test Set Evaluation (Llama 3.1 Base)  │
│  ├─ Accuracy: 84%                      │
│  ├─ F1-score (macro): 0.81             │
│  ├─ Confusion matrix analysis          │
│  └─ Per-class precision/recall         │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Test Set Evaluation (LoRA Fine-tuned)  │
│  ├─ Accuracy: 88% (+4%)                │
│  ├─ F1-score (macro): 0.86 (+0.05)    │
│  ├─ Dispute classification: 91% → 95% │
│  ├─ Risk assessment: 82% → 88%        │
│  └─ Merchant support intent: 86%→89%  │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│  Human Evaluation (A/B Test)            │
│  ├─ Sample 500 predictions              │
│  ├─ Expert review (base vs. fine-tuned)│
│  ├─ Preference: Fine-tuned wins 72%    │
│  ├─ Consistency improvement: 84%→91%   │
│  └─ Safety issues: 0 found             │
└────────────────┬───────────────────────┘

┌────────────────▼───────────────────────┐
│  Catastrophic Forgetting Check         │
│  ├─ Eval on held-out general QA set    │
│  ├─ General capability: 94% (vs 96%)   │
│  ├─ Acceptable degradation: <3%        │
│  └─ Mitigation: Reduce LoRA rank if >5%│
└────────────────┬───────────────────────┘

6. DEPLOYMENT
┌────────────────────────────────────────┐
│  Model Artifacts                        │
│  ├─ Base model: Llama 3.1 (3.2GB)      │
│  ├─ LoRA weights: 33MB                 │
│  ├─ Total size: 3.25GB                 │
│  ├─ Config: adapter_config.json         │
│  └─ Training details: training_log.json│
└────────────────┬───────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  vLLM/TensorRT-LLM Optimization         │
│  ├─ Merge LoRA into base (optional)    │
│  ├─ Quantize to INT8 (optional)        │
│  ├─ Compile to CUDA kernels            │
│  ├─ Enable paged attention             │
│  └─ Inference latency: 50ms/request    │
└────────────────┬───────────────────────┘

┌────────────────▼────────────────────────┐
│  Serving Architecture (GKE)             │
│  ├─ vLLM pod (auto-scale: 2-10)        │
│  ├─ GPU: NVIDIA L4 (cost-effective)    │
│  ├─ Batch size: 32 (dynamic batching)  │
│  ├─ Max concurrent requests: 256       │
│  ├─ Throughput: 500 requests/sec       │
│  ├─ p95 latency: <200ms                │
│  └─ Cost: $2K/month per model          │
└──────────────────────────────────────────┘
```

### Data Preparation Details:

**Training Data Format** (JSON Lines):
```json
{"input": "Classify dispute: Customer claims unauthorized transaction on merchant ABC", "output": "Chargeback - Unauthorized"}
{"input": "What is the refund policy for Digital goods?", "output": "Digital goods are non-refundable per policy section 4.2"}
{"input": "Assess risk: High-value payment from new account to risky merchant", "output": "Risk Level: HIGH (new account + risky merchant) → Manual review required"}
```

**Class Imbalance Handling**:
```
Original distribution:
├─ Dispute - Unauthorized: 40% (most common)
├─ Dispute - Item not received: 30%
├─ Dispute - Merchant mismatch: 20%
├─ Dispute - Billing error: 10% (least common)

Solution (Weighted loss):
├─ Weight 1: Unauthorized (common)
├─ Weight 1.3: Item not received
├─ Weight 1.6: Merchant mismatch
├─ Weight 2.5: Billing error (rare)

Prevents model from overfitting to majority class
```

**Preventing Catastrophic Forgetting**:
- Keep LoRA rank low (16-32) to preserve base knowledge
- Include general instruction examples in training data (~10%)
- Evaluate on general QA held-out set regularly
- Monitor perplexity drift on original base model's test set

---

## Q2: Design an Inference Optimization Pipeline for Production LLMs

**Context**: Optimize Llama 3.1 (8B) for sub-50ms latency inference serving 500 requests/sec with 99.9% uptime. Must support batch inference, streaming, and dynamic batching.

### Follow-up Questions:
- What optimization techniques would you apply?
- How do you handle variable sequence lengths?
- What's your batching strategy?
- How do you reduce memory footprint?

### Optimization Techniques:
```
1. QUANTIZATION (Reduce Model Size)
   ├─ INT8: 50% memory reduction, minimal accuracy loss
   ├─ INT4: 75% memory reduction, ~2% accuracy loss
   ├─ FP8: 50% memory reduction, better accuracy than INT8
   └─ Chosen: INT8 (best cost/quality trade-off)

2. PRUNING (Remove Unnecessary Parameters)
   ├─ Structured: Remove attention heads (12 → 8 heads)
   ├─ Unstructured: Magnitude-based weight pruning
   ├─ Layer-wise: Remove less important layers
   └─ Chosen: Structured head pruning (20% faster, 1% accuracy loss)

3. KNOWLEDGE DISTILLATION (Smaller Student Model)
   ├─ Teacher: Llama 3.1 (8B)
   ├─ Student: Llama 3.2 (1B)
   ├─ Distillation loss: KL divergence on teacher/student outputs
   └─ Result: 40% faster, 3-5% accuracy loss (acceptable)

4. HARDWARE ACCELERATION
   ├─ TensorRT-LLM: Custom CUDA kernels (+30% throughput)
   ├─ Flash Attention 2: Optimized attention (+50% speed)
   ├─ Paged Attention: Memory-efficient KV cache (+2x batch size)
   └─ Fused operators: Reduce kernel launches

5. SERVING OPTIMIZATION
   ├─ vLLM: Dynamic batching, continuous batching
   ├─ Prefix caching: Cache common prefixes
   ├─ Rope scaling: Extend context window efficiently
   └─ Token streaming: Start returning output early
```

---

## Q3: Design a Distributed Fine-Tuning System for Multi-GPU Training

**Context**: Fine-tune a 70B parameter model across 8 H100 GPUs with FSDP (Fully Sharded Data Parallel). Must reduce training time from 7 days to 24 hours.

### Follow-up Questions:
- How would you use FSDP vs. DDP vs. Pipeline Parallelism?
- What's your communication overhead strategy?
- How do you handle mixed precision training at scale?
- What's your approach to handling a GPU failure mid-training?

### Architecture:
```
FULLY SHARDED DATA PARALLEL (FSDP):

┌─────────────────────────────────────┐
│   Gradient Computation Phase        │
│   Each GPU processes batch          │
└────────────────┬────────────────────┘
                 │
┌────────────────▼──────────────────┐
│  Gradient Sync (Ring AllReduce)    │
│  ├─ GPU 0: owns params[0:10%]     │
│  ├─ GPU 1: owns params[10:20%]    │
│  ├─ ...                            │
│  └─ GPU 7: owns params[70:80%]    │
│                                    │
│  All-reduce over ring topology     │
│  Reduces communication bandwidth   │
│  O(N) instead of O(N²)             │
└────────────────┬──────────────────┘

Communication Timeline:
├─ Compute phases: 0.8s (overlapped with comm)
├─ Communication: 0.2s
└─ Total per step: 1.0s (vs 1.2s DDP)

Scaling Efficiency:
├─ 1x GPU: 1.0s/step → 100% efficiency
├─ 2x GPU: 0.55s/step → 91% efficiency
├─ 4x GPU: 0.30s/step → 83% efficiency
├─ 8x GPU: 0.16s/step → 78% efficiency
└─ Total time: 24 hours (7 days / 6.5x speedup)

Memory Efficiency:
├─ Per-GPU: 40GB/8 = 5GB for model weights
├─ Optimizer states (Adam): 3 copies per param
├─ Gradient buffer: 2.5GB
├─ Activation memory: 5GB
└─ Total per GPU: 40GB (fits on H100)
```

---

## Q4: Design RLHF (Reinforcement Learning from Human Feedback) Pipeline

**Context**: Improve model output quality using human feedback. Train a reward model, then optimize policy using PPO (Proximal Policy Optimization).

### Follow-up Questions:
- How do you collect human feedback at scale?
- What's your reward model architecture?
- How do you prevent reward hacking?
- What's your KL divergence penalty strategy?

### Pipeline:
```
Stage 1: Supervised Fine-Tuning (SFT)
├─ Fine-tune base model on high-quality examples
├─ Llama 3.1 → dispute classification accuracy +4%
└─ This becomes the policy model (π)

Stage 2: Reward Model Training
├─ Collect human comparisons: "Output A is better than B"
├─ Train binary classifier: reward_model(output) → score
├─ Accuracy: 95% agreement with human preferences
└─ Serves as signal for policy optimization

Stage 3: RLHF (PPO Training)
├─ Generate outputs from policy model
├─ Score with reward model
├─ Optimize policy to maximize reward
├─ KL penalty: prevent diverging too far from SFT model
└─ Result: Better outputs aligned with human preferences

Cost Breakdown:
├─ Human feedback collection: $2K (500 comparisons)
├─ Reward model training: $100 (single GPU, 4h)
├─ PPO training: $500 (3 GPUs, 8h)
└─ Total: $2.6K (vs full fine-tuning: $2K)
```

---

## Q5: Design Multi-Task Learning System for LLMs

**Context**: Train a single model on multiple tasks (dispute classification, merchant risk assessment, customer intent detection) with shared parameters but task-specific heads.

### Follow-up Questions:
- How do you balance training across different tasks?
- What's your architecture for task-specific adapters?
- How do you handle tasks with different loss scales?
- What's your evaluation strategy?

### Architecture:
```
Shared Base Model (Llama 3.1):
├─ Layers 0-30: Frozen (general knowledge)
└─ Layers 31-32: Fine-tuned (task adaptation)

Task-Specific Adapters (LoRA):
├─ Dispute Classification Adapter
│  ├─ LoRA rank: 8
│  ├─ Unique params: 0.8M
│  └─ Head: LinearLayer (input_dim → 4 classes)
│
├─ Risk Assessment Adapter
│  ├─ LoRA rank: 8
│  ├─ Unique params: 0.8M
│  └─ Head: LinearLayer (input_dim → 3 classes)
│
└─ Intent Detection Adapter
   ├─ LoRA rank: 8
   ├─ Unique params: 0.8M
   └─ Head: LinearLayer (input_dim → 8 classes)

Training Loop:
├─ Task 1 batch: loss_1 (weight: 0.4)
├─ Task 2 batch: loss_2 (weight: 0.4)
├─ Task 3 batch: loss_3 (weight: 0.2)
└─ Total loss: 0.4 * loss_1 + 0.4 * loss_2 + 0.2 * loss_3

Benefits:
├─ Single shared model (3.2GB instead of 9.6GB)
├─ Cross-task knowledge transfer
├─ Reduced inference latency (single forward pass)
└─ Faster training (shared gradients)
```

---

## Interview Tips for Top Companies:

1. **Understand Trade-offs**: Cost vs. Accuracy, Speed vs. Memory, Training Time vs. Inference Speed
2. **Production Focus**: Always mention monitoring, alerting, rollback strategy
3. **Data Quality**: Garbage in = garbage out. Talk about curation, filtering, validation
4. **Reproducibility**: Fixed seeds, version control, experiment tracking (W&B)
5. **Safe Rollout**: Blue-green deployment, canary releases, feature flags

---

## References to Your pp Experience:

- Reduced production retrieval latency by **50%** via fine-tuned Llama 3.1 Nemotron
- LoRA/PEFT fine-tuning for domain-specific embeddings
- Implemented HyDE-based query expansion
- Inference optimization achieving sub-2s p95 latency
- Production LLM serving across 50+ AI services at scale

