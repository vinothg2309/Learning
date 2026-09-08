04_MLOps_Infrastructure_Scale.md
# MLOps & Infrastructure at Scale Interview Questions

## Q1: Design a Scalable ML Platform for 50+ Concurrent AI Services

**Context**: Design a production platform serving 50+ AI microservices with sub-2s p95 latency, 99.9% uptime, 45% infrastructure cost reduction, and high-frequency deployment (8+ services/day).

### Follow-up Questions:
- How would you architect the serving layer?
- What's your strategy for GPU allocation and scheduling?
- How do you handle traffic spikes and load balancing?
- What's your deployment strategy?
- How do you monitor 50+ models simultaneously?

### Architecture Diagram:
```
┌─────────────────────────────────────────────────────────────┐
│          SCALABLE ML PLATFORM (GCP/GKE)                     │
│       Serving 50+ AI Services @ pp Scale               │
└─────────────────────────────────────────────────────────────┘

INGRESS LAYER
┌──────────────────────────────────────┐
│  Cloud Load Balancer                 │
│  ├─ Geolocation-based routing        │
│  ├─ Health check probes (path: /v1/) │
│  ├─ SSL/TLS termination              │
│  └─ Rate limiting (per API key)      │
└────────────────┬─────────────────────┘
                 │
┌────────────────▼──────────────────────┐
│  Service Mesh (Istio)                 │
│  ├─ Traffic policies (A/B test split)│
│  ├─ Retry logic (max 3 attempts)     │
│  ├─ Timeout enforcement (2s max)     │
│  ├─ Circuit breaker (fail open)      │
│  └─ Distributed tracing (Jaeger)     │
└────────────────┬──────────────────────┘

ROUTING LAYER
┌────────────────┬──────────────────────┐────────────┐
│                │                      │            │
│         ┌──────▼────────┐    ┌────────▼────┐     │
│         │  Dispute      │    │  Merchant   │     │
│         │  Resolution   │    │  Risk       │     │
│         │  Service      │    │  Service    │     │
│         └───────────────┘    └─────────────┘     │
│                                                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │  Payment │  │ Fraud    │  │ Customer │      │
│   │ Insights │  │ Detection│  │ Intent   │      │
│   │ Service  │  │ Service  │  │ Service  │      │
│   └──────────┘  └──────────┘  └──────────┘      │
│                                                   │
│   ... (50+ services total)                        │
│                                                   │
└───────────────────────────────────────────────────┘

COMPUTE LAYER (GKE Cluster)
┌────────────────────────────────────────────────┐
│  Node Pool 1: GPU-Optimized (A100/H100)        │
│  ├─ Nodes: 10-20 (auto-scale)                 │
│  ├─ GPU per node: 8 (256GB total)             │
│  ├─ Workload: Large models (LLMs)             │
│  ├─ Node pool taints/tolerations              │
│  └─ Preemptible nodes: 30% (cost savings)     │
└────────────────┬─────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────┐
│  Node Pool 2: CPU-Optimized (Standard)        │
│  ├─ Nodes: 30-50 (auto-scale)                │
│  ├─ CPU per node: 32 cores, 128GB RAM        │
│  ├─ Workload: Embeddings, small models       │
│  └─ Spot instances: 40% (cost savings)       │
└────────────────┬─────────────────────────────┘

INFERENCE SERVING LAYER
┌────────────────────────────────────────────────┐
│  Serving Framework: vLLM + Triton Inference   │
│                                                 │
│  vLLM Deployment:                              │
│  ├─ Pod autoscaling (requests/min metric)    │
│  ├─ Min replicas: 2 (HA)                     │
│  ├─ Max replicas: 10 (burst capacity)        │
│  ├─ Container resource:                      │
│  │  ├─ Request: 1 GPU, 30GB RAM              │
│  │  ├─ Limit: 1 GPU, 40GB RAM                │
│  │  └─ Startup probe: 30s timeout            │
│  ├─ Dynamic batching:                        │
│  │  ├─ Batch size: 32 (request/latency trade)│
│  │  ├─ Wait time: 5ms (batch timeout)        │
│  │  ├─ Max queue: 256 requests               │
│  │  └─ Throughput: 500-700 req/sec/replica  │
│  ├─ Quantization: INT8 (50% memory, 3% loss)│
│  ├─ Flash Attention 2: Enabled               │
│  └─ Prefix caching: Common prompts (20% hit) │
│                                               │
│  Triton Deployment (Lightweight models):     │
│  ├─ Ensemble pipelines (e.g., embed + rerank)│
│  ├─ CPU-only inference (cost-effective)      │
│  ├─ Model versioning (seamless updates)      │
│  └─ Request batching across models           │
└────────────────┬────────────────────────────┘

CACHING LAYER
┌────────────────────────────────────────────────┐
│  L1: Redis Semantic Cache (Hot)                │
│  ├─ Query embeddings + top results            │
│  ├─ TTL: 24h for common queries               │
│  ├─ Hit rate: 30-40% (reduces LLM calls)     │
│  ├─ Cost: $2K/month (Memorystore)             │
│  └─ Savings: $7M+ annually (cost opt)         │
│                                                 │
│  L2: vLLM KV Cache (Warm)                     │
│  ├─ PagedAttention: Efficient GPU memory     │
│  ├─ Hit rate on sequential requests: 60%     │
│  └─ Extends batch size capability             │
│                                                 │
│  L3: CDN for static content                   │
│  ├─ Model artifacts, configs                 │
│  └─ Reduces model loading time                │
└────────────────┬────────────────────────────┘

DATA LAYER
┌────────────────────────────────────────────────┐
│  Vector Database (Milvus/Pinecone)            │
│  ├─ 1M+ embeddings indexed                    │
│  ├─ Read-optimized (for retrieval)            │
│  ├─ Write: Batch updates nightly              │
│  └─ Query latency: <50ms p95                  │
│                                                 │
│  Feature Store (Vertex AI)                    │
│  ├─ Real-time features (current balance)     │
│  ├─ Batch features (historical patterns)     │
│  ├─ Online serving (sub-50ms)                │
│  └─ Lineage tracking (compliance)             │
│                                                 │
│  Metadata Store (PostgreSQL)                  │
│  ├─ Vector document mappings                  │
│  ├─ Citation tracking                        │
│  ├─ Compliance audit logs                    │
│  └─ TTL-based cleanup (GDPR)                 │
└────────────────┬────────────────────────────┘

MONITORING & OBSERVABILITY
┌────────────────────────────────────────────────┐
│  Metrics (Prometheus + Cloud Monitoring)      │
│  ├─ Latency: p50/p95/p99 per service         │
│  ├─ Throughput: requests/sec per model        │
│  ├─ Error rate: 4xx/5xx responses             │
│  ├─ Queue depth: pending requests             │
│  ├─ GPU utilization: per node (target: 70%)  │
│  ├─ Model inference accuracy drift            │
│  └─ Cost per request ($0.0005 target)         │
│                                                 │
│  Logging (Cloud Logging)                      │
│  ├─ Request trace ID (end-to-end debugging)  │
│  ├─ Model prediction + confidence             │
│  ├─ Tool calls (for agents)                   │
│  ├─ Error stack traces                        │
│  └─ Compliance events (PII access)            │
│                                                 │
│  Tracing (Cloud Trace)                        │
│  ├─ End-to-end request latency breakdown     │
│  ├─ Service-to-service dependencies          │
│  ├─ Bottleneck identification                │
│  └─ Flame graphs (CPU profiling)              │
│                                                 │
│  Alerting                                      │
│  ├─ p95 latency > 2s (page on-call)          │
│  ├─ Error rate > 0.1% (critical)             │
│  ├─ GPU OOM events (restart pod)             │
│  ├─ Queue depth > 1000 (add replicas)        │
│  └─ Model drift detected (retrain trigger)   │
└────────────────────────────────────────────────┘

DEPLOYMENT PIPELINE
┌────────────────────────────────────────────────┐
│  CI/CD (Cloud Build)                           │
│  ├─ Git push → Build Docker image             │
│  ├─ Scan for vulnerabilities (Artifact Reg)   │
│  ├─ Run unit tests (pytest)                   │
│  ├─ Run integration tests (mock APIs)         │
│  └─ Push to artifact registry                 │
│                                                 │
│  Canary Deployment (Istio)                    │
│  ├─ 1. Deploy to 5% of traffic (5 min)       │
│  ├─ 2. Monitor error rate + latency           │
│  ├─ 3. Gradual rollout: 25% → 50% → 100%    │
│  ├─ 4. Rollback if metrics degrade           │
│  └─ 5. Total time: 20 min (safe rollout)     │
│                                                 │
│  Frequency: 8+ deployments/day (safe due to  │
│  canary approach + comprehensive testing)     │
└────────────────────────────────────────────────┘

COST OPTIMIZATION STRATEGIES
┌────────────────────────────────────────────────┐
│  1. GPU Utilization (→ 45% cost reduction)    │
│  ├─ Dynamic batching: +40% throughput        │
│  ├─ Quantization: -50% memory                │
│  ├─ Spot instances: -70% compute cost        │
│  └─ Savings: $5M+ annually                    │
│                                                 │
│  2. Caching Strategy                          │
│  ├─ Redis semantic cache: 30-40% hit rate    │
│  ├─ Reduces LLM invocations: -30%             │
│  └─ Savings: $2M+ annually                    │
│                                                 │
│  3. Model Right-Sizing                        │
│  ├─ Llama 3.1 (8B) vs 70B for 95% of tasks   │
│  ├─ Smaller model + few-shot > large model   │
│  └─ 4x cheaper inference                      │
│                                                 │
│  4. Infrastructure Efficiency                 │
│  ├─ GCP Commitment discounts: -30%            │
│  ├─ Reserved instances: -25%                  │
│  └─ Combined: 45% total reduction             │
└────────────────────────────────────────────────┘
```

### Key Metrics Achieved:
- **Latency**: Sub-2s p95 (99% of requests)
- **Uptime**: 99.9% (4 hours downtime/month)
- **Throughput**: 500+ requests/sec per service
- **Cost Reduction**: 45% infrastructure savings
- **Deployment Speed**: 3× faster (vs previous)
- **Model Deployment**: <2 hours for new models across 8 prod workflows

---

## Q2: Design a CI/CD Pipeline for ML Models with Automated Testing

**Context**: Design a robust pipeline for deploying ML models safely with automated validation, quality gates, and canary releases.

### Follow-up Questions:
- What tests would you run before production deployment?
- How do you validate model performance?
- What's your rollback strategy?
- How do you prevent data drift?

### Testing Strategy:
```
UNIT TESTS (Code Level)
├─ Input validation (shape, dtype, ranges)
├─ Output shape correctness
├─ Edge cases (empty input, null values)
├─ Function logic (correctness)
└─ Latency requirements (<100ms per call)

INTEGRATION TESTS
├─ API endpoint returns valid responses
├─ Database queries work correctly
├─ External service calls succeed
├─ Error handling works (bad input → error)
└─ Concurrent requests handled properly

MODEL TESTS (ML-Specific)
├─ Performance on test set: accuracy ≥ 0.88
├─ Per-class metrics: precision/recall > 0.85
├─ Fairness: No demographic disparity
├─ Robustness: Adversarial examples handled
├─ Explainability: SHAP values computed
└─ Latency: p95 < 100ms

REGRESSION TESTS
├─ New model vs old model on benchmark set
├─ Minimum performance improvement: +1%
├─ No accuracy regression on held-out set
└─ Error rate: <0.1%

PRODUCTION VALIDATION (Canary)
├─ 5% traffic for 5 minutes
├─ Error rate increase < 0.05%
├─ Latency increase < 50ms
├─ User complaint rate (CSAT)
├─ If all green: Continue rollout
├─ If red: Rollback immediately
```

---

## Q3: Design a Feature Store for Real-Time ML

**Context**: Build a feature store that provides consistent, low-latency features for training and serving across 50+ models.

### Architecture:
```
Feature Store Layer (Vertex AI)

Online Store (Serving):
├─ Redis backend
├─ Query latency: <50ms
├─ Consistency: Strong
├─ Features updated: Real-time/hourly
└─ Use case: Inference-time serving

Batch Store (Training):
├─ BigQuery backend
├─ Query latency: <5s
├─ Consistency: Eventual
├─ Features refreshed: Daily/weekly
└─ Use case: Model training, historical analysis

Feature Pipeline (Streaming):
├─ Apache Beam / Dataflow jobs
├─ Ingest from: Payment API, Customer DB
├─ Compute derived features: Balance, fraud score
├─ Write to: Online store (immediate), Batch store (daily)
└─ Monitoring: Feature freshness, data quality

Benefits:
├─ Training-serving skew eliminated
├─ Reusable features across models
├─ Single source of truth
└─ Compliance audit trail (lineage)
```

---

## Q4: Design GPU Resource Scheduling for Multi-Tenant ML Platform

**Context**: Allocate K80/H100/L4 GPUs efficiently across 50+ teams running experiments and serving models. Handle priority scheduling, preemption, and cost allocation.

### Key Components:

```
GPU Scheduler (Karpenter + Custom Policy)

Priority Tiers:
├─ P1: Production inference (non-preemptible)
├─ P2: Batch training (low priority)
├─ P3: Ad-hoc experiments (preemptible)
└─ P4: Research (preemptible, lowest priority)

Scheduling Logic:
├─ P1 requests: Immediate allocation (SLA: <10s)
├─ P2 requests: Bin packing (cost optimization)
├─ P3 requests: Fill remaining capacity
├─ P4 requests: Scavenger mode (use idle GPUs)

Preemption Policy:
├─ If P1 needs GPU: Preempt P3 (immediate)
├─ Preemption grace: 30s (save checkpoint)
├─ Preempted job: Queued for retry
└─ Cost tracking: Charged based on time held

Cost Allocation:
├─ P1: $5/GPU hour (charged to product)
├─ P2: $2/GPU hour (batch discount)
├─ P3: $1/GPU hour (spot price)
└─ P4: $0.50/GPU hour (research budget)

Monitoring:
├─ GPU utilization: Target 85%+
├─ Fragmentation: <10% (unused GPU capacity)
├─ Wait time: P1 ≤ 10s, P2 ≤ 5 min
├─ Cost per request: Track per team
└─ Alert: Idle GPUs (investigate)
```

---

## Q5: Design a Model Registry & Versioning System

**Context**: Manage 50+ models with different versions, tracking lineage, metadata, and enabling rollbacks.

### Architecture:

```
Model Registry (MLflow/Vertex AI)

Model Storage:
├─ Artifacts: Model weights, config (GCS)
├─ Metadata: Author, creation date, metrics
├─ Code: Training script, dependencies
├─ Data: Training dataset hash (reproducibility)
└─ Environment: Python version, library versions

Versioning Strategy:
├─ Semantic: v1.0 (major.minor.patch)
├─ Major: Breaking changes, new architecture
├─ Minor: New feature, improved accuracy
├─ Patch: Bug fix, minor optimization
│
├─ Tags: prod, staging, candidate, deprecated
└─ Promotion: candidate → staging → prod

Lineage Tracking:
├─ Training data: Which dataset version?
├─ Hyperparameters: Learning rate, batch size
├─ Dependencies: sklearn 1.2, transformers 4.25
├─ Evaluation: Test metrics, confusion matrix
└─ Author: Who trained it?

Governance:
├─ Approval workflow: 2 reviewers before prod
├─ Audit log: Who promoted what, when
├─ Rollback: One-click revert to previous version
├─ Retirement: Archive old models (1-year retention)
└─ Compliance: PII handling validated

CI/CD Integration:
├─ Register model after training
├─ Auto-promote to staging if metrics > threshold
├─ Manual approval for production
├─ Deploy to serving infrastructure
└─ Monitor for data/concept drift
```

---

## Interview Tips for Top Companies:

1. **Focus on Reliability**: 99.9% uptime, graceful degradation, health checks
2. **Cost Awareness**: Show how you'd reduce expenses (spot instances, caching, quantization)
3. **Observability**: Discuss tracing, logging, monitoring—how you'd debug at 3am
4. **Scalability**: Talk about auto-scaling, load distribution, handling 10x traffic
5. **Security**: Encryption, access control, compliance (PCI-DSS, GDPR)

---

## References to Your pp Experience:

- **45% infrastructure cost reduction** through optimization
- **Sub-2s p95 latency** across 50+ AI services
- **99.9% uptime** on production deployment
- **3× faster model deployment cycles** (vLLM optimization)
- **8+ daily deployments** safely via canary releases
- **GCP expertise**: Vertex AI, Agent Builder, GKE, Cloud Run, GCS
- **$10M+ recurring cost optimization**, **$7M+ annual savings**
