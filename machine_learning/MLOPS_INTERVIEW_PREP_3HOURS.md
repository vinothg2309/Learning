# MLOps Interview Prep: 3-Hour Crash Course

**Last Updated:** June 2, 2026  
**Duration:** 3 hours (1.5 hours concepts, 1.5 hours questions)  
**Format:** Concepts First → Interview Questions

---

## Table of Contents

### Part 1: Core Concepts (1.5 Hours)
1. [MLOps in 30 Seconds](#1-mlops-in-30-seconds)
2. [The Three Layers of Production ML](#2-the-three-layers-of-production-ml)
3. [Data Management Concepts](#3-data-management-concepts)
4. [Model Deployment Concepts](#4-model-deployment-concepts)
5. [Monitoring & Drift Concepts](#5-monitoring--drift-concepts)
6. [Cost & Scale Concepts](#6-cost--scale-concepts)

### Part 2: Interview Questions (1.5 Hours)
7. [Question 1: Design 100M Requests/Day System](#question-1-design-100m-requestsday-system)
8. [Question 2: Model Accuracy Dropped 5%](#question-2-model-accuracy-dropped-5)
9. [Question 3: Reduce LLM Inference Cost](#question-3-reduce-llm-inference-cost)
10. [Question 4: Choose GCP vs Azure](#question-4-choose-gcp-vs-azure)
11. [Question 5: Detect & Respond to Data Drift](#question-5-detect--respond-to-data-drift)
12. [Question 6: MLflow on GCP & Large-Scale Training](#question-6-integrate-mlflow-with-gcp--train-large-data)

### Part 3: Quick References
12. [Red Flags: What NOT to Say](#red-flags-what-not-to-say)
13. [Talking Point Templates](#talking-point-templates)
14. [Decision Frameworks](#decision-frameworks)
15. [Key Numbers to Remember](#key-numbers-to-remember)

---

# Part 1: Core Concepts

## 1. MLOps in 30 Seconds

### Definition

**MLOps** = Machine Learning Operations = the discipline of operationalizing ML models from experimentation to production to monitoring to retraining.

**Simple Analogy:**
- **Software Delivery:** Code → Build → Test → Deploy → Monitor
- **MLOps:** Data + Code → Train → Test → Deploy Model → Monitor + Retrain

### Why It Matters

| Aspect | Impact |
|--------|--------|
| **Scale** | A model in a notebook handles 1 request/sec. Production needs 1000s/sec. |
| **Reliability** | Models degrade silently (data drift). Without monitoring, nobody knows. |
| **Cost** | Unoptimized ML systems cost $100K+/month. Good MLOps reduces to $10K. |
| **Business** | Bad predictions cost real money (wrong loans, unfair hiring, poor recommendations). |

### Core Challenge

Unlike software (deterministic - same code = same output), ML is **non-deterministic**:
- Same model may give different results with different data
- Models degrade as real-world data changes
- Hard to catch failures (model says "I'm confident" but wrong)

**Therefore, MLOps must handle:**
- ✓ Data versioning (not just code versioning)
- ✓ Model monitoring (not just system metrics)
- ✓ Automatic retraining (when data/world changes)
- ✓ Graceful degradation (fallbacks when model fails)

---

## 2. The Three Layers of Production ML

### Architecture Overview

```
┌─────────────────────────────────────┐
│   Application Layer                 │
│   (User-facing: API, UI, etc.)      │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   Model Serving Layer               │
│   (Inference, caching, LB)          │
├────────────────────────────────────┤
│   Online Serving:                   │
│   Request → Feature Lookup →        │
│   Model Inference → Response        │
│   Latency: <100ms                   │
│                                    │
│   Batch Serving:                    │
│   Daily Precomputation →            │
│   Cache Results → Lookup            │
│   Latency: Precomputed              │
└────────────────┬────────────────────┘
                 │
┌────────────────▼────────────────────┐
│   ML Pipeline Layer                 │
│   (Data prep, training, monitoring) │
├────────────────────────────────────┤
│   Data Ingestion → Validation →    │
│   Feature Engineering → Training →  │
│   Model Registry → Monitoring →     │
│   Drift Detection → Retraining      │
└─────────────────────────────────────┘
```

### Layer 1: Model Serving Layer

**What it does:** Takes user requests and returns predictions

**Two patterns:**

**A) Batch Serving (Offline Predictions)**
```
When: Results needed in hours, not milliseconds
How:
  1. Every night: Compute predictions for all users
  2. Store results in database/cache
  3. User request: Simple lookup (no model inference)
  4. Latency: <10ms (database lookup)

Cost: Low (compute only during batch window)
Complexity: Low (simple lookup)

Example: Email recommendations (user sees recs when opening email)
```

**B) Online Serving (Real-time Predictions)**
```
When: Results needed in milliseconds, personalized per request
How:
  1. User sends request
  2. Fetch user features from Feature Store (<10ms)
  3. Run model inference (<50ms)
  4. Return response (<100ms total)

Cost: High (servers always running, handling concurrent requests)
Complexity: High (load balancing, caching, auto-scaling)

Example: Product recommendation when user browses (personalized to browsing history)
```

### Layer 2: Model Serving Infrastructure

**Components:**

```
Load Balancer
  ├─ Routes requests across servers
  └─ Provides failover (if one server dies, others handle traffic)

Model Server (TensorFlow Serving, vLLM, FastAPI)
  ├─ Loads model into memory
  ├─ Handles concurrent requests
  └─ Returns predictions

Cache Layer (Redis)
  ├─ Stores frequently-accessed features
  ├─ Stores previous predictions
  └─ Lookup: <1ms

Monitoring
  ├─ Latency (p50, p95, p99)
  ├─ Error rate
  ├─ Model accuracy (if ground truth available)
  └─ Resource utilization (CPU, GPU, memory)
```

### Layer 3: ML Pipeline Layer

**What it does:** Prepares data, trains models, monitors for degradation, retrains

**Pipeline steps:**

```
1. Data Ingestion
   ├─ Fetch data from sources (database, APIs, logs)
   ├─ Store in data lake/warehouse
   └─ Scheduled: Daily/hourly

2. Data Validation
   ├─ Schema checks (right columns, right types)
   ├─ Quality checks (no nulls, no outliers)
   ├─ Drift checks (input distribution changed?)
   └─ Stop pipeline if checks fail

3. Feature Engineering
   ├─ Create ML-ready features from raw data
   ├─ Example: from [birth_date] → [age], [age_group]
   └─ Store in Feature Store (reusable)

4. Model Training
   ├─ Load features
   ├─ Train model on historical data
   ├─ Log hyperparameters & metrics
   ├─ Register model if metrics improve
   └─ Takes minutes to hours (depending on data size)

5. Model Evaluation
   ├─ Compare new model vs. baseline
   ├─ Check offline metrics (accuracy, F1, etc.)
   ├─ Check business metrics (conversion lift)
   └─ Only deploy if improvement > threshold

6. Deployment
   ├─ Stage new model (5% traffic)
   ├─ Monitor for 1 hour
   ├─ Gradually increase traffic
   ├─ Or rollback if metrics degrade
   └─ Takes 5 minutes to 24 hours (depending on strategy)

7. Monitoring
   ├─ Track model performance in production
   ├─ Detect data drift, prediction drift
   ├─ Alert if metrics degrade
   ├─ Log predictions for debugging
   └─ Continuous (24/7)

8. Retraining (triggered by monitoring)
   ├─ If drift detected → retrain on new data
   ├─ If accuracy drops → investigate then retrain
   ├─ If ground truth arrives → improve labels
   └─ Loop back to step 1
```

---

## 3. Data Management Concepts

### Feature Store

**What is it?**

A centralized repository that stores computed features (pre-calculated values used by ML models).

**Problem it solves:**

```
WITHOUT Feature Store:
┌─────────────────────────┐
│  Training Pipeline      │
│  Compute "user_age"     │ ← Takes 2 hours
│  Compute "avg_purchase" │
└─────────────────────────┘
                │
                └─→ Train model

┌─────────────────────────┐
│  Serving (real-time)    │
│  Compute "user_age"     │ ← Takes 5 seconds (TOO SLOW!)
│  Compute "avg_purchase" │
└─────────────────────────┘
                │
                └─→ Return prediction

PROBLEM: Features computed twice, serving takes 5 seconds (need <100ms)


WITH Feature Store:
┌─────────────────────────┐
│  Feature Engineering    │
│  Compute features once  │ ← Takes 2 hours
│  Store in Feature Store │
└──────────┬──────────────┘
           │
    ┌──────▼──────┐
    │ Feature     │
    │ Store       │
    │ (Database)  │
    └──────┬──────┘
           │
    ┌──────┴──────────────────┬──────────────────┐
    │                         │                  │
    ▼                         ▼                  ▼
Training Pipeline      Online Serving      Batch Scoring
Lookup features       (real-time)         (offline)
<10ms                 <10ms lookup        Bulk fetch
                      + inference
```

**Types:**

```
Batch Features (offline):
├─ Computed daily/weekly
├─ Stored in data warehouse (BigQuery, Synapse)
├─ Used for training, batch scoring
└─ Latency: Precomputed (doesn't matter)

Online Features (real-time):
├─ Computed daily, kept in cache (Redis)
├─ Served in <10ms
├─ Used during real-time inference
└─ Latency-critical: Must be <10ms
```

**Tools:**
- Feast (open-source, popular)
- Tecton (enterprise)
- Vertex Feature Store (GCP managed)
- Azure Feature Store (Azure managed)

### Data Drift

**What is it?**

Input data distribution changed (in production vs. training).

**Example:**

```
Training Data (2023):
├─ 60% of users aged 25-35
├─ Average income: $80K
└─ Mostly urban areas

Production Data (2024):
├─ 40% of users aged 25-35 (shift to older!)
├─ Average income: $60K (economic downturn)
└─ 30% rural (expansion to new market)

PROBLEM: Model was trained on young, high-income urban users
Now receiving old, low-income rural users
Model performance drops because it's out of distribution
```

**Detection:**

```python
from scipy.stats import ks_2samp

# Compare distributions
statistic, p_value = ks_2samp(training_data, current_data)

# p_value < 0.05: Distributions are significantly different (drift detected!)
# p_value > 0.05: No significant change
```

**Response:**

```
IF drift detected:
├─ Classify: Is it seasonal? New cohort? Quality issue?
├─ If seasonal: Normal, no action
├─ If new cohort: Retrain on new data
├─ If quality issue: Fix pipeline first
└─ Monitor: Watch metrics after retrain
```

### Model Registry

**What is it?**

Version control for trained models. Track which model is in production, its performance, and rollback if needed.

**Structure:**

```
Model: "recommendation-model"
├─ Version 1
│  ├─ Accuracy: 0.87
│  ├─ Training data: v1.2
│  ├─ Hyperparameters: lr=0.001, epochs=10
│  └─ Status: Old
│
├─ Version 2
│  ├─ Accuracy: 0.90 ✓ BEST
│  ├─ Training data: v1.5
│  ├─ Hyperparameters: lr=0.0005, epochs=20
│  └─ Status: Production (Currently serving 100% traffic)
│
└─ Version 3
   ├─ Accuracy: 0.88 ✗ WORSE
   ├─ Training data: v1.6
   ├─ Hyperparameters: lr=0.0001, epochs=50
   └─ Status: Archived (Bad experiment)
```

**Key operations:**

```
Register new model:
  registry.register("recommendation-model", model_file, metrics)
  ↓
  Model added as new version

Promote to staging:
  registry.transition("recommendation-model", version=3, stage="Staging")
  ↓
  Model v3 moved to Staging for testing

Promote to production:
  registry.transition("recommendation-model", version=3, stage="Production")
  ↓
  Model v3 now serves 100% traffic

Rollback (if v3 bad):
  registry.transition("recommendation-model", version=2, stage="Production")
  ↓
  Revert to v2 in <5 minutes
```

---

## 4. Model Deployment Concepts

### Batch vs. Online Serving Decision

**Latency SLA drives architecture:**

```
┌─────────────────────────────────────────────────────────┐
│ Latency Requirement                                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ < 50ms:    Must be ONLINE (in-memory, GPU)            │
│            Example: Real-time bidding for ads          │
│                                                         │
│ 50-500ms:  Preferably ONLINE (still user-facing)      │
│            Example: Product recommendations on browse  │
│                                                         │
│ 1-10 seconds: Could be cached ONLINE results           │
│            Example: Email personalization              │
│                                                         │
│ 1+ hours:  BATCH (precompute overnight)               │
│            Example: Weekly newsletter recommendations  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Decision Tree:**

```
Can you precompute overnight?
├─ YES → Batch Serving
│        ├─ Compute predictions daily
│        ├─ Store in cache/database
│        └─ Lookup at request time (fast)
│
└─ NO → Online Serving
         ├─ Run inference per request
         ├─ Must be fast (<100ms)
         └─ Need GPU/optimization
```

### Deployment Strategies

#### 1. Blue-Green Deployment

**How it works:**

```
BEFORE:
┌──────────────────┐
│   Requests       │
│   (User Traffic) │
└────────┬─────────┘
         │
    ┌────▼────┐
    │  Blue   │
    │ Model   │
    │ v2      │
    │ (Old)   │
    └─────────┘

DEPLOYMENT:
┌──────────────────┐
│   Requests       │
└────────┬─────────┘
         │
    ┌────▼────────────────┐
    │                     │
┌───▼────┐          ┌─────▼──┐
│  Blue  │          │ Green  │
│ Model  │          │ Model  │
│ v2     │          │ v3     │
│ (Old)  │          │ (New)  │
└────────┘          └────────┘
(Still serving)      (Testing)

SWITCH TRAFFIC:
    ┌──────────────────┐
    │   Requests       │
    └────────┬─────────┘
             │
        ┌────▼────────────────┐
        │                     │
    ┌───────┐          ┌─────▼──┐
    │  Blue │          │ Green  │
    │ Model │          │ Model  │
    │ v2    │          │ v3     │
    │ (Old) │          │ (New)  │
    └───────┘          └──▲─────┘
   (No traffic)   (100% traffic - INSTANT!)
```

**Advantages:**
- ✓ Zero downtime (switch happens instantly)
- ✓ Easy rollback (switch back to blue instantly)

**Disadvantages:**
- ✗ Need 2x infrastructure (two full environments)
- ✗ Expensive (idle capacity while testing)

#### 2. Canary Deployment

**How it works:**

```
Hour 0: Deploy v3 to 5% traffic
├─ 95% traffic → Model v2 (old)
└─  5% traffic → Model v3 (new)
    Monitor: latency, error rate, accuracy

Hour 1: If metrics good, increase to 10%
├─ 90% traffic → Model v2
├─ 10% traffic → Model v3
└─ Monitor more

Hour 4: If still good, increase to 50%
├─ 50% traffic → Model v2
├─ 50% traffic → Model v3
└─ Monitor more

Hour 8: If metrics excellent, increase to 100%
└─ 100% traffic → Model v3
   Rollback to v2 if at any point metrics degrade
```

**Advantages:**
- ✓ Low risk (catch bad models early)
- ✓ Natural A/B test (measure business impact)
- ✓ Normal infrastructure usage

**Disadvantages:**
- ✗ Slower rollout (takes hours/days, not minutes)
- ✗ Complex to implement (traffic splitting logic)

### Model Optimization

**Quantization: Compress Model**

```
Original Model (float32):
├─ Size: 1 GB (4 bytes per weight)
├─ Inference: 100ms
└─ Accuracy: 92%

Quantized Model (int8):
├─ Size: 250 MB (1 byte per weight) ✓ 4x smaller!
├─ Inference: 40ms ✓ 2.5x faster!
└─ Accuracy: 90.5% (only 1.5% loss, acceptable!)

USE CASE:
├─ Mobile devices (smaller = less battery)
├─ Inference cost (faster = fewer servers needed)
└─ Real-time serving (faster = meet latency SLA)
```

---

## 5. Monitoring & Drift Concepts

### What to Monitor

**Four types of metrics:**

#### A) System Metrics (Infrastructure)

```
Latency:
├─ p50 latency: 50% of requests faster than this
├─ p95 latency: 95% of requests faster than this
├─ p99 latency: 99% of requests faster than this
└─ SLA: p99 < 200ms

Error Rate:
├─ % of requests that failed
├─ SLA: < 0.1% (99.9% success rate)

Resource Utilization:
├─ CPU: Should be <80% (headroom for spikes)
├─ GPU: Should be <90% (avoid OOM errors)
├─ Memory: Should be <80%
└─ Disk: Should be <85%
```

#### B) Model Metrics (Prediction Quality)

```
Accuracy:
├─ % of predictions that were correct
├─ SLA: >= 90% (or whatever baseline)
└─ Alert if drops > 5%

Prediction Confidence:
├─ Average confidence of model
├─ If drops, model is uncertain
└─ Investigate if dropping

Feature Importance:
├─ Which features matter most
├─ Alert if importance changes (distribution shift)
└─ Example: "age" usually important, suddenly not
```

#### C) Data Quality Metrics

```
Data Completeness:
├─ % of features present (no nulls)
├─ SLA: > 99%

Data Freshness:
├─ Age of oldest feature
├─ Example: "purchase_history" should be <24h old
├─ SLA: Depends on use case

Data Drift:
├─ Has input distribution changed?
├─ Measured by KS test p-value
├─ Alert if p < 0.05 (significant change)
```

#### D) Business Metrics

```
Conversion Rate:
├─ % of users who completed desired action
├─ Example: % who clicked recommendation
├─ Alert if drops > 2%

Revenue per User:
├─ Direct measurement of model business impact
├─ Alert if drops > 5%

Cost per Prediction:
├─ Total monthly cost / total predictions
├─ Alert if increases > 20%
```

### Alert Severity Levels

```
P0 (Critical - Page immediately):
├─ Error rate > 10% for 2 minutes
├─ Model completely unavailable
├─ Predictions nonsensical (NaN, out of range)
└─ SLA breach: p99 latency > 500ms

P1 (High - Page within 15 min):
├─ Error rate > 5% for 5 minutes
├─ Latency > 200ms
├─ Data drift p-value < 0.01
└─ Accuracy drop > 10%

P2 (Medium - Email today):
├─ Data drift p-value < 0.05
├─ Accuracy drop 5-10%
├─ Cost increased > 20%
└─ Non-critical feature unavailable

P3 (Low - Email weekly):
├─ Accuracy drift 2-5%
├─ Minor data quality issues
├─ Deprecation warnings
```

---

## 6. Cost & Scale Concepts

### Cost Drivers

**Where does ML cost come from?**

```
100% = $50K/month typical ML system

├─ 60% Compute ($30K)
│  ├─ Training: $10K/month (GPUs running 24/7)
│  ├─ Serving: $18K/month (inference servers)
│  └─ Experimentation: $2K/month (dev/test)
│
├─ 20% Storage ($10K)
│  ├─ Data warehouse: $5K/month
│  ├─ Model artifacts: $2K/month
│  └─ Logs & monitoring: $3K/month
│
├─ 10% Tools ($5K)
│  ├─ MLflow, feature store, monitoring tools
│  └─ Cloud platform management
│
└─ 10% Team ($5K)
   ├─ ML engineers (salary prorated to this project)
   └─ Data engineers, MLOps
```

### Cost Optimization Levers

**Quick wins (implement first):**

```
1. Use Spot Instances for Training
   Before: $100/hour GPU compute
   After: $30/hour (70% cheaper!)
   Trade-off: Can be interrupted, need checkpointing
   ROI: Easy, first thing to do

2. Batch Requests Together
   Before: 1 request at a time = $0.001 each
   After: 100 requests together = $0.0005 each (50% savings)
   Trade-off: Latency (batch takes 5 minutes)
   ROI: Good, especially for offline workloads

3. Cache Predictions
   Before: Recompute prediction for same user every time
   After: Store in Redis, lookup <1ms
   Savings: Avoid GPU inference 90% of time
   ROI: Excellent, minimal complexity
```

**Advanced optimizations (implement second):**

```
4. Model Quantization
   Before: 1GB model, 100ms inference
   After: 250MB model, 40ms inference
   Savings: Faster = fewer servers needed (4x fewer)
   Trade-off: 1-2% accuracy loss
   ROI: Excellent, saves $10K+/month

5. Model Distillation
   Before: Large model (complex, slow)
   After: Small model trained to mimic large one
   Savings: 10x smaller, 5x faster inference
   Trade-off: 3-5% accuracy loss
   ROI: Good for cost-sensitive applications

6. Smaller Model Selection
   Before: GPT-4 for all tasks ($0.03 per call)
   After: GPT-3.5 for 70% tasks ($0.001), GPT-4 for 30%
   Savings: 30x cheaper on average
   Trade-off: 5-10% quality loss
   ROI: Excellent
```

### Scaling Architecture

**How to serve 100M requests/day:**

```
Daily Volume: 100M = ~1,200 requests/second average
Peak Traffic: 3-5x average = ~6,000 requests/second

Solution:
├─ Batch Layer (precompute 80%):
│  ├─ Nightly compute predictions for all users
│  ├─ Store in Redis cache (100GB)
│  ├─ Lookup latency: <10ms (cache hit)
│  └─ Cost: $5K/month (batch compute at night)
│
├─ Online Layer (real-time 20%):
│  ├─ 50 GPU servers (auto-scale 10-50)
│  ├─ Load balancer distributes requests
│  ├─ Inference latency: <50ms per server
│  └─ Cost: $20K/month (always-on servers)
│
├─ Feature Cache:
│  ├─ Redis cluster for feature lookup
│  ├─ Latency: <5ms per lookup
│  └─ Cost: $3K/month
│
└─ Multi-region (for failover):
   ├─ Same setup in 2-3 regions
   ├─ Automatic failover if region dies
   └─ Cost: 2-3x total cost
```

---

# Part 2: Interview Questions

## Question 1: Design 100M Requests/Day System

### Problem Statement

**"Design an ML system to serve 100M predictions per day with <100ms latency SLA."**

### Answer Framework

#### Step 1: Clarify Requirements

Before jumping into architecture, ask clarifying questions:

```
LATENCY:
├─ Is <100ms p50, p95, or p99?
│  (assume p99 based on "SLA")
├─ Can we tolerate occasional 500ms?
│  (no, users notice)

SCALE:
├─ 100M/day = ~1,200 req/sec average
├─ Peak traffic likely 3-5x = ~6,000 req/sec
├─ Spike traffic possible? (special events = 10x)

TRAFFIC PATTERN:
├─ Is traffic uniform 24/7?
│  (likely no - peak during business hours)
├─ Can we batch some predictions offline?
│  (yes, likely 80% can be precomputed)

ACCURACY REQUIREMENT:
├─ What baseline accuracy needed?
│  (assume 90%+)
├─ Tolerance for model unavailability?
│  (should have fallback to cached results)

COST BUDGET:
├─ Any budget constraint?
│  (assume standard cloud infrastructure)
```

#### Step 2: Propose Architecture

**Two-Tier Architecture (Batch + Online):**

```
┌─────────────────────────────────────────────────────────┐
│                   USER REQUEST                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Is this a "common" user?
                     │ (in precomputed set)
                     │
            ┌────────▼────────┐
            │  Check Cache    │
            │  (Redis)        │
            └────┬────────┬───┘
                 │        │
            YES  │        │  NO
                 │        │
            ┌────▼─┐  ┌──▼──────────────────┐
            │Cache │  │ Online Inference    │
            │Hit   │  │ 1. Fetch features   │
            │<10ms │  │    (<10ms)          │
            └──────┘  │ 2. Model inference  │
                      │    (<50ms)          │
                      │ 3. Format response  │
                      │    (<10ms)          │
                      └─────────┬───────────┘
                                │
                        ┌───────▼──────┐
                        │  Response    │
                        │  Total: <100ms
                        └──────────────┘
```

#### Step 3: Detail Each Layer

**Layer 1: Batch Precomputation (80% of traffic)**

```
Nightly Batch Job (runs 2am-6am):

INPUT:  100M users (yesterday's activity)
        ├─ Active users list
        └─ User features (updated daily)

COMPUTATION:
├─ For each user:
│  ├─ Fetch user features from warehouse
│  ├─ Run model inference (batch GPU)
│  └─ Get top 20 recommendations
│
└─ Takes 4 hours on 10 GPUs

OUTPUT: Redis Cache
├─ user:123 → [item_id_1, item_id_2, ..., item_id_20]
├─ user:456 → [item_id_5, item_id_10, ...]
└─ ~100GB total data

SERVING:
├─ User request comes in
├─ Lookup user:123 in Redis
├─ Return cached results
└─ Latency: <10ms (cache lookup)

COST:
├─ GPU compute: 4 hours × 10 GPUs × $0.35/hour = $14
├─ Total daily cost: ~$420/month
└─ Cost per prediction: $420M / 80M = $0.000005 (CHEAP!)
```

**Layer 2: Online Inference (20% of traffic)**

```
Real-time Serving for Edge Cases:

EXAMPLES WHERE BATCH INSUFFICIENT:
├─ New user (not in yesterday's precompute)
├─ User with different filters/constraints
├─ Personalized in-session recommendations
└─ Requests requiring latest data

ARCHITECTURE:

Load Balancer
  ├─ Distributes requests across servers
  ├─ Handles failover (if server dies, route elsewhere)
  └─ SSL termination

API Server (FastAPI)
  ├─ Receives HTTP requests
  ├─ Validates input
  └─ Calls feature service + model service

Feature Service:
  ├─ Fetches user features from cache/database
  ├─ Latency: <10ms (cached) or 50ms (lookup)
  └─ Could be Vertex Feature Store or Feast

Model Server:
  ├─ Loads model in memory (GPU)
  ├─ Runs inference on features
  ├─ Batch multiple requests if possible
  └─ Latency: <50ms per batch

INFRASTRUCTURE:

Base Setup:
├─ 10 GPU servers (baseline)
├─ 1 Load Balancer
└─ 1 Redis cache for intermediate results

Auto-Scaling (peak traffic):
├─ Monitor: If p99 latency > 150ms, scale up
├─ Add servers: 10 → 50 over 5 minutes
├─ Monitor: If p99 latency < 100ms, scale down
└─ Remove servers: 50 → 10 over 10 minutes

Total servers at peak: 50 GPU machines
```

**Layer 3: Feature Store Management**

##### 3.1 What Gets Stored in Feature Store

A Feature Store is a centralized repository that stores 4 types of artifacts:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Feature Definitions** | Schema + metadata for each feature | `age: Int, range 0-150, computed daily` |
| **Online Store** | Real-time serving (Redis/Spanner, <10ms) | `user:123 → {age: 30, purchases: 250.5}` |
| **Offline Store** | Historical data for training (BigQuery, 5-30s) | All user features with timestamps |
| **Transformation Code** | Shared logic for computing features | `def compute_age(birth_date): return TODAY - birth_date` |
| **Metadata & Stats** | Data quality metrics + versions | `mean: 35.2, std: 12.1, version: v1.2` |

##### 3.2 Prediction-Time Transformation: How to Get Features

**Problem:** How do we ensure features computed during serving match training?

**Recommended Solution: Hybrid Approach**

```
PREDICTION REQUEST (user_id = 123)
│
├─ TRY: FETCH from Feature Store (FAST, <10ms)
│  ├─ Query Online Store: user:123
│  ├─ Returns: {age: 30, recent_purchases: 250.5}
│  ├─ Guarantee: Features match training (pre-computed)
│  └─ ❌ Problem: New user not in store yet
│
├─ FALLBACK: COMPUTE on-the-fly (SLOW, 50-500ms)
│  ├─ Use shared transformation code:
│  │  ├─ age = compute_age(user.birth_date)
│  │  ├─ purchases = compute_purchases_7d(user_id)
│  │  └─ rating = compute_avg_rating(user_id)
│  ├─ Latency: 50-500ms (DB queries + computation)
│  └─ ✅ Works for new users
│
└─ RUN INFERENCE on either path (same format)
   └─ Model doesn't know which path was used
```

**Key Code Example:**

```python
# SHARED transformation library (version controlled)
# Used by BOTH training pipeline AND serving

class SharedFeatureTransformations:
    """Features computed exactly same way everywhere"""
    
    @staticmethod
    def compute_age(birth_date: str) -> int:
        """Same computation: training, online store, serving"""
        from datetime import date
        birth = date.fromisoformat(birth_date)
        today = date.today()
        return (today - birth).days // 365
    
    @staticmethod
    def compute_purchases_7d(user_id: str) -> float:
        """Query database for feature"""
        query = f"""
        SELECT SUM(amount) FROM purchases
        WHERE user_id = '{user_id}'
        AND date >= TODAY() - INTERVAL 7 DAY
        """
        return run_query(query) or 0.0


# During serving: Hybrid path
def predict(user_id: str):
    try:
        # FAST: Fetch from feature store
        fs = FeatureStore()
        features = fs.get_online_features(
            features=["user_features:age", "user_features:recent_purchases_7d"],
            entity_rows=[{"user_id": user_id}]
        )
        if features["age"] is not None:
            input_data = {
                "age": features["age"],
                "purchases_7d": features["recent_purchases_7d"]
            }
            return model.predict(input_data)
    except:
        pass
    
    # SLOW: Compute for new users (uses shared code)
    user = query_user(user_id)
    input_data = {
        "age": SharedFeatureTransformations.compute_age(user['birth_date']),
        "purchases_7d": SharedFeatureTransformations.compute_purchases_7d(user_id)
    }
    return model.predict(input_data)
```

##### 3.3 Training-Serving Consistency (Critical)

**Problem: Time-Based Features Can Break**

```
Training (May 2025):
├─ Feature: days_since_purchase = today - last_purchase_date
├─ Example: last purchase 2025-04-22, today 2025-05-02 = 10 days
└─ Model learned: "10 days → recent customer → high engagement"

Serving WITHOUT Feature Store (June 2026, WRONG):
├─ Same user, same last_purchase_date: 2025-04-22
├─ Feature: days_since_purchase = 2026-06-02 - 2025-04-22 = 407 days!
└─ Model predicts: "407 days → old customer → low engagement" ❌
   This is WRONG! Same user, completely different prediction.

Serving WITH Feature Store (June 2026, CORRECT):
├─ Feature Store recomputes ALL features daily (2am-6am)
├─ Today's batch (2026-06-02) computes fresh features
├─ days_since_purchase: still relative to THAT day's context
└─ Model predicts: Correct! Features match training distribution ✅
```

##### 3.4 Common Pitfalls to Avoid

```
❌ MISTAKE 1: Different preprocessing
   Training: age = (today - birth_date) / 365.25
   Serving:  age = today.year - birth.year
   → 1 year error! Use shared transformation code.
   
✅ SOLUTION: Store transformation code in feature store
             Reuse in both training + serving

❌ MISTAKE 2: Missing value handling mismatch
   Training: missing age → use average (30)
   Serving:  missing age → crash or return NULL
   → Features have completely different distributions!
   
✅ SOLUTION: Define strategy in feature definition
             Apply consistently everywhere

❌ MISTAKE 3: Feature version mismatch
   Training: used age_feature v1.0
   Serving:  using age_feature v1.2 (bug fixed)
   → v1.2 might compute differently, model won't work right!
   
✅ SOLUTION: Track which feature versions used in training
             Only use same versions for serving
             Plan retraining when upgrading features

❌ MISTAKE 4: External data changes
   Training: HOLIDAYS list includes 2025 holidays
   Serving:  HOLIDAYS list not updated with 2026 holidays
   → Model doesn't recognize new holidays!
   
✅ SOLUTION: Use authoritative external sources (not hardcoded)
             Regular updates to reference data
```

##### 3.5 Real-World Example: E-Commerce Recommendations

**What Features Are Stored:**

```
User Features:
├─ age (computed from birth_date)
├─ total_purchases (lifetime count)
├─ avg_order_value (average $$ per purchase)
└─ days_since_last_purchase (recency)

Product Features:
├─ price (current price)
├─ avg_rating (customer reviews)
└─ inventory_count (stock level)

Interaction Features (user-product pairs):
├─ user_viewed_product_7d (did user view recently?)
└─ user_price_sensitivity (buys at full price?)

During Prediction:
├─ User clicks product
├─ Fetch from Feature Store (<10ms): all 10 features
├─ Run recommendation model
└─ Return ranking (~50ms total latency)
```

##### 3.6 Daily Feature Computation Pipeline

**How features are created and stored:**

```
Daily Batch Job (2:00am - 6:00am)
├─ Data Warehouse extracts:
│  ├─ Raw users (birthdate, demographics)
│  ├─ Raw transactions (purchases, dates, amounts)
│  └─ Raw reviews (ratings, comments)
│
├─ Spark job transforms:
│  ├─ For each user: compute age, purchase count, avg value, etc.
│  ├─ For each product: compute price, rating, inventory, etc.
│  └─ For each user-product: compute viewing history, interactions
│
├─ Store in two places:
│  ├─ Online Store (Redis):
│  │  └─ Latest values only (fastest, <10ms lookup)
│  │  └─ Used during real-time prediction
│  │
│  └─ Offline Store (BigQuery):
│     └─ Historical data with timestamps
│     └─ Used for training next model
│
└─ Tools available:
   ├─ Feast (open-source)
   ├─ Vertex Feature Store (GCP)
   ├─ Azure Feature Store
   └─ Custom Redis + Spark

COST BREAKDOWN:
├─ Feature computation: 4 hours × 50 GPUs = $35/day = $1K/month
├─ Online storage (Redis): 100GB = $2K/month
├─ Offline storage (BigQuery): $1K/month
└─ Total: ~$4K/month
```

**Summary:**
- **Fetch from store** (fast, pre-computed) for existing users
- **Compute on-the-fly** (slow, but reliable) for new users
- **Use shared code** everywhere to guarantee consistency
- **Version features** to track training-serving compatibility

#### Step 4: Handle Failures

**Reliability measures:**

```
IF MODEL SERVER DIES:
├─ Load balancer detects (health check fails)
├─ Automatically route to another server
├─ User request goes to different server
└─ Failover: <5 seconds

IF FEATURE CACHE DIES:
├─ Fall back to database query
├─ Latency: 10ms → 50ms (slightly slower)
├─ Acceptable because latency still <100ms
└─ Alerts fire, team rebuilds cache

IF ALL SERVERS DIE:
├─ Return last known prediction (stale but OK)
├─ OR return most popular items (generic fallback)
└─ Alert: Page oncall, restore services ASAP

IF BATCH JOB FAILS:
├─ Use yesterday's cache (1 day stale)
├─ Resume batch job for today
└─ Latency impact: User gets slightly old recs
```

#### Step 5: Cost Breakdown

```
MONTHLY COST ESTIMATE:

Batch Layer:
├─ Compute: $420 (nightly batches)
└─ Storage: $2K (Redis cache)

Online Inference:
├─ GPU servers: 10 × $0.35/hour × 730 = $2,555/month
├─ Scale to 50 during peaks: +$12,775/month
├─ Average: ~$8K/month (factoring peaks)
└─ Includes load balancer, networking

Data Management:
├─ Data warehouse: $3K (BigQuery/Synapse)
├─ Feature store: $2K (computation + storage)
└─ Monitoring: $1K

Tools & Infrastructure:
├─ MLflow, monitoring tools: $1K
├─ Kubernetes management: $1K
└─ CI/CD pipelines: $500

TOTAL: ~$22K/month

Cost per prediction: $22K / 100M = $0.00022
```

#### Step 6: Key Talking Points

```
STRENGTHS OF ARCHITECTURE:
✓ "80% batch = cheap ($420/month)"
✓ "20% online = real-time personalization"
✓ "Multi-region failover = 99.9% availability"
✓ "Cache layer = fast lookup for common users"
✓ "Feature store = reusable features for ML teams"

TRADE-OFFS:
✓ "Batch predictions are 1 day stale (acceptable for recs)"
✓ "Online inference slower than cached (50ms vs 10ms)"
✓ "Complexity: harder to operate than single-model serving"

OPTIMIZATION POINTS:
✓ "Quantize model: 4x smaller, 2x faster"
✓ "Use spot instances for batch: 70% cheaper"
✓ "Request batching: GPU processes 10 requests together"
✓ "Add more caching layers (CDN for static data)"
```

---

## Question 2: Model Accuracy Dropped 5%

### Problem Statement

**"Your model's accuracy dropped from 92% to 87% in production. How do you diagnose and fix?"**

### Answer Framework

#### Step 1: Immediate Actions (First 5 minutes)

```
PRIMARY GOAL: Restore service, minimize user impact

1. ALERT & ASSESS
   ├─ Is model completely broken (all wrong)?
   │  └─ YES: Rollback immediately to previous version
   │          (5 minute recovery, best we can do)
   │
   ├─ Is accuracy slightly degraded (still 87%)?
   │  └─ YES: Keep running, investigate offline
   │          (can diagnose while serving)

2. ROLLBACK IF CRITICAL
   Command: kubectl rollout undo deployment/model-serving
   
   Timeline:
   ├─ Send rollback command: 30 seconds
   ├─ Stop new model: 30 seconds
   ├─ Start old model: 30 seconds
   ├─ Health checks pass: 30 seconds
   ├─ Route traffic to old: 30 seconds
   └─ Total: ~3 minutes

3. CHECK IF ROLLBACK HELPED
   ├─ Did accuracy return to 92%?
   ├─ If YES: Old model still works, problem in new model
   ├─ If NO: Problem not in model code (data? labeling?)
   └─ Continue investigation with old model serving (safe)
```

#### Step 2: Investigation (First 30 minutes)

**Checklist - Rule out common causes:**

```
HYPOTHESIS 1: Serving Discrepancy
(Model works in training, different in production)

Tests:
├─ Run inference locally on test data
│  └─ Compare: local result vs. production result
│
├─ Check preprocessing:
│  ├─ Are input features computed same way in training vs. serving?
│  ├─ Example: training used (height_cm), serving uses (height_inches)?
│  └─ Compare feature values for sample request
│
├─ Check quantization artifacts:
│  ├─ If using quantized (int8) model, might lose precision
│  ├─ Test: run original (float32) in production
│  └─ If accuracy returns to 92%, quantization was problem
│
└─ Check dependencies:
   ├─ Did Python package versions change?
   ├─ Did model file get corrupted?
   └─ Run: python -c "import model; print(model.version())"


HYPOTHESIS 2: Data Drift
(Input data distribution changed, model out of distribution)

Tests:
├─ Extract last 24h production data
├─ Compare with training data:
│  ├─ Use KS test: ks_2samp(training_data, production_data)
│  ├─ For each feature:
│  │  ├─ user_age: training mean=35, production mean=38 (slight drift)
│  │  ├─ income: training mean=$80K, production mean=$65K (BIG DRIFT!)
│  │  └─ location: training 60% urban, production 45% urban (drift)
│
├─ Visualize distributions:
│  ├─ Create histogram for training vs. production
│  ├─ Overlay histograms, look for shifts
│  └─ Easy to spot in Jupyter notebook
│
├─ Results:
│  ├─ If major drift found: retrain on new data
│  ├─ If no drift: problem elsewhere
│  └─ If seasonal drift: expected, monitor more
```

```
HYPOTHESIS 3: Label Shift
(Class distribution changed, not feature distribution)

Tests:
├─ If classification task (not regression):
│  ├─ Training: 60% positive, 40% negative
│  ├─ Production: 30% positive, 70% negative (BIG SHIFT!)
│  └─ Could cause accuracy drop
│
├─ Check:
│  ├─ What % of production predictions are positive?
│  ├─ Compare with training baseline
│  └─ Query: SELECT COUNT(*) WHERE prediction=1 / COUNT(*) FROM predictions
│
├─ Possible causes:
│  ├─ Marketing changed target audience
│  ├─ Product changed (easier/harder to use)
│  ├─ User behavior naturally changed
│  └─ Labeling pipeline changed
│
└─ Fix:
   ├─ Retrain with new class distribution
   ├─ Or adjust decision threshold
   └─ Example: Lower threshold from 0.5 to 0.35 if positive class rare
```

```
HYPOTHESIS 4: Feedback Loop Contamination
(Model's old predictions used as training labels for retraining)

Tests:
├─ Check retraining data source:
│  ├─ Where do ground truth labels come from?
│  ├─ Is it manual feedback from users? Good.
│  ├─ Is it model's own predictions? BAD!
│
├─ Example feedback loop:
│  ├─ Old model: "user A likely to click"
│  ├─ We show recommendation to user A
│  ├─ User clicks (because we recommended!)
│  ├─ Label it as "positive" (training data)
│  ├─ New model learns same pattern
│  ├─ But for new users without recommendation?
│  ├─ Model predicts "will click" even if we don't show
│  ├─ Results in accuracy drop on unbiased data
│
└─ Fix:
   ├─ Use only manual labels (user feedback)
   ├─ Not impressions/clicks from our recommendations
   ├─ Retrain with clean labels
```

```
HYPOTHESIS 5: Concept Drift
(Ground truth changed, not data)

Tests:
├─ Did the world change?
│  ├─ Example: "How likely will user default on loan?"
│  ├─ Training: 2019 (low defaults, economy good)
│  ├─ Production: 2024 (high defaults, recession)
│  ├─ Ground truth literally changed
│
├─ Not detectable by KS test:
│  ├─ KS test compares INPUT distribution
│  ├─ Concept drift is about OUTPUT (labels)
│  ├─ Same users now behave differently
│
├─ Detection methods:
│  ├─ If ground truth available, compare:
│  │  ├─ Training set accuracy: 92%
│  │  ├─ Production set accuracy: 87%
│  ├─ Ask domain expert:
│  │  ├─ "Has the prediction target changed?"
│  │  ├─ Example: "default rate went from 5% to 10%"
│
└─ Fix:
   ├─ Retrain on recent data (model adapts)
   ├─ May need manual review (is new behavior acceptable?)
   ├─ May need business rule change (lending criteria)
```
```

#### Step 3: Determine Root Cause

**Decision tree:**

```
                    Accuracy Dropped 5%
                            │
                ┌───────────┬┴┬───────────┐
                │           │ │           │
        ┌───────▼─┐  ┌──────▼┐ │  ┌──────▼┐
        │ Rollback │  │ Data  │ │  │ Label │
        │ helps?   │  │ Drift?│ │  │ Shift?│
        └────┬─────┘  └───┬──┘ │  └───┬───┘
             │            │    │      │
            YES           YES   │      YES
             │            │    │      │
     Problem in       Retrain on  Adjust
     New Model        New Data    Threshold
             │            │      or Retrain
          │Check:      │Fix:   │Fix:
          │  - Quantization    - Add/remove features
          │  - Preprocessing   - Adjust threshold
          │  - Dependencies    - Retrain
          └─────────────────────────────────┘


                            NO
                            │
                    ┌───────┴────────┐
                    │                │
            ┌───────▼──┐      ┌──────▼───┐
            │ Feedback │      │ Concept  │
            │ Loop?    │      │ Drift?   │
            └────┬─────┘      └────┬─────┘
                 │                 │
                YES               YES
                 │                 │
          Fix Data Pipeline  Manual Review
          Remove Self-Pred   + Retrain
          Retrain
                 │                 │
                 └────────┬────────┘
                          │
                   ┌──────▼───────┐
                   │ Monitor 1w   │
                   │ If stable:OK │
                   │ If drops:ROI │
                   └──────────────┘


#### Step 4: Root Cause Examples

**Example 1: Preprocessing Bug**

```
SCENARIO:
├─ New model deployed Friday
├─ Monday morning: accuracy dropped 92% → 87%
├─ Only 2 days difference, can't be data drift
├─ (Would need weeks for distribution to change)

ROOT CAUSE:
├─ Preprocessing code changed
├─ Old: normalize_feature(x) = (x - 0.5) / 0.5
├─ New: normalize_feature(x) = (x - 0) / 1.0
├─ Model trained on old preprocessing, now using new
├─ Features out of expected range
└─ Model performs poorly

DIAGNOSIS:
├─ Step 1: Compare preprocessing (train vs. serve)
├─ Step 2: Run inference with old preprocessing
├─ Step 3: Accuracy returns to 92% ✓
└─ Step 4: Fix code, redeploy

FIX TIME: 1 hour (identify + fix + test)
```

**Example 2: Data Drift**

```
SCENARIO:
├─ Model trained on 2023 user data
├─ Now it's 2024, user demographics shifted
├─ Training: 70% urban, age 25-40, income $80K
├─ Production: 50% urban, age 30-50, income $60K
├─ Model out of distribution

ROOT CAUSE:
├─ KS test detects: p-value < 0.01 (significant drift)
├─ Feature importance changed:
│  ├─ Old: "age" most important
│  ├─ New: "location" most important
├─ Model was optimized for old distribution
└─ Now on new distribution, performs worse

DIAGNOSIS:
├─ Step 1: Extract 1 week production data
├─ Step 2: KS test against training data
├─ Step 3: Visualize distributions (overlapping histograms)
├─ Step 4: Confirm major shift in 2-3 key features
└─ Step 5: Decide: Retrain or wait and monitor

FIX TIME: 4-24 hours (retrain on new data)
```

**Example 3: Feedback Loop**

```
SCENARIO:
├─ Model trained on labels = "user clicked recommendation"
├─ Used same training pipeline for retraining
├─ New model keeps learning from old predictions
├─ System spirals into overconfidence
└─ Accuracy drops on unbiased test set

ROOT CAUSE:
├─ Training labels are impressions (we recommend, user clicks)
├─ Not ground truth (would they click if we didn't recommend?)
├─ Selection bias: model only sees biased training examples
├─ When used on random users (not recommended), performs worse

DIAGNOSIS:
├─ Step 1: Inspect training data labels source
├─ Step 2: Check if labels = model's past predictions
├─ Step 3: Confirm selection bias in data
├─ Step 4: Check A/B test results (old vs. new model)
│  ├─ Offline accuracy: new model is better
│  ├─ Online A/B: new model is worse!
│  └─ Classic sign of feedback loop
└─ Step 5: Fix labeling pipeline

FIX TIME: 2-8 hours (fix pipeline + retrain)
```

#### Step 5: General Response Template

```
"First, I'd determine if this is critical (rollback immediately) 
or can be investigated offline (keep serving while diagnosing).

Then I'd run through a checklist:
1. Serving discrepancy? (preprocessing, quantization)
   └─ Compare local inference vs. production
2. Data drift? (input distribution changed)
   └─ KS test production vs. training data
3. Label shift? (class distribution changed)
   └─ Compare % positive in production vs. training
4. Feedback loop? (labels contaminated)
   └─ Check where ground truth comes from
5. Concept drift? (world changed)
   └─ Ask domain expert, check accuracy curve over time

Once I identify root cause, I'd:
├─ Fix the problem (retrain, fix preprocessing, adjust threshold)
├─ Validate the fix (test on holdout data)
├─ Deploy with monitoring (watch metrics for 1 week)
├─ Post-incident: Why did we miss this?
│  └─ Add automated detection (e.g., KS test daily)
└─ Document: Runbook for next time
"
```

---

## Question 3: Reduce LLM Inference Cost

### Problem Statement

**"Your LLM inference costs $10K/day. How would you reduce it to $1K/day (10x reduction)?"**

### Answer Framework

#### Step 1: Understand Current Costs

**Baseline metrics:**

```
Current Spend: $10,000/day

Assumptions (work backwards):
├─ 1,000,000 API calls/day
├─ Average cost per call: $0.01
└─ Breakdown:
   ├─ 500K calls to GPT-4: $0.03 each = $15K (but we said $10K total?)
   ├─ Let me recalculate...
   └─ Maybe: 500K calls × $0.005 avg = $2.5K
      + 500K calls × $0.015 avg = $7.5K = $10K total

More realistic breakdown:
├─ 1M API calls/day
├─ Mix of models: GPT-4 ($0.03), GPT-3.5 ($0.001), Claude ($0.008)
├─ Average call: 2000 tokens input + 500 tokens output
├─ Weighted average cost: $0.01/call
└─ Total: 1M × $0.01 = $10K/day
```

#### Step 2: Identify Cost Drivers

**Where is the $10K going?**

```
┌──────────────────────────────────────────┐
│ Cost Breakdown: $10,000/day              │
├──────────────────────────────────────────┤
│                                          │
│ Input Tokens: 2B tokens/day              │
│ ├─ At $0.00000050 per input token       │
│ └─ Cost: 2B × $0.0000005 = $1,000       │
│                                          │
│ Output Tokens: 500M tokens/day           │
│ ├─ At $0.00001500 per output token      │
│ └─ Cost: 500M × $0.00001500 = $7,500    │
│                                          │
│ Model Mix Surcharge: 10%                 │
│ ├─ Using expensive models (GPT-4)       │
│ ├─ Cost: $850                           │
│ └─ Could use cheaper models             │
│                                          │
│ Other: 10% overhead                      │
│ ├─ Calls for monitoring, testing        │
│ └─ Cost: $650                           │
│                                          │
└──────────────────────────────────────────┘

KEY INSIGHT: Output tokens = 75% of cost!
           (1,500 tokens output × expensive price)
```

#### Step 3: Optimization Strategies (In Priority Order)

### Strategy 1: Prompt Caching (90% Cost Savings on Context)

**Problem:**

```
Current flow per request:
┌──────────────────────────────────────┐
│ System Prompt (10K tokens)           │ ← Included in EVERY request
├──────────────────────────────────────┤
│ Company Documentation (100K tokens)  │ ← Included in EVERY request
├──────────────────────────────────────┤
│ User Query (100 tokens)              │ ← Different each time
└──────────────────────────────────────┘

Total: 110K tokens per request
Cost: 1M requests × 110K tokens = 110B tokens/day × $0.000001 = $110K!

But that doesn't match $10K...
Actually: Maybe only 10K tokens of context per request:
          1M requests × 10K tokens = 10B tokens = $10K ✓
```

**Solution: Cache Reusable Context**

```
Claude API or OpenAI API support prompt caching:

First request with cache:
├─ System prompt: 5K tokens (cached)
├─ Documentation: 5K tokens (cached)
│  ├─ Cache overhead: +100 tokens
│  └─ Total: 10,100 tokens (cache creation is cheap)
├─ User query: 100 tokens
└─ Cost: (10K × $0.000001) + (100 × $0.000005) = $0.01 + $0.0005 = $0.0105

Second request (cache hit):
├─ System prompt: 5K tokens (FROM CACHE)
│  ├─ Price: $0.00000009 (90% cheaper!)
│  └─ Cost: 5K × $0.0000001 = $0.0005
├─ Documentation: 5K tokens (FROM CACHE)
│  ├─ Cost: 5K × $0.0000001 = $0.0005
├─ User query: 100 tokens (new)
│  ├─ Cost: 100 × $0.000005 = $0.0005
└─ Total cost: $0.0015 (85% CHEAPER than uncached!)

IMPACT ON $10K/day:
├─ 80% of requests have cache hit
├─ 20% miss (first time, new cache)
├─ Average cost reduction: 0.2×(-10%) + 0.8×(-85%) = 70% savings
├─ New cost: $10K × 0.3 = $3K/day ✓
└─ Savings: $7K/day
```

**Implementation:**

```python
from anthropic import Anthropic

client = Anthropic()

# Define reusable context
system_prompt = "You are a customer service agent for TechCorp."
company_docs = """
Company: TechCorp
Products: Software, Hardware, Services
Contact: support@techcorp.com
...100KB of documentation...
"""

# Request 1: Cache creation (higher cost)
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": system_prompt,
        },
        {
            "type": "text",
            "text": company_docs,
            "cache_control": {"type": "ephemeral"}  # Cache this!
        }
    ],
    messages=[
        {"role": "user", "content": "How do I reset my password?"}
    ]
)

# Request 2-1000: Cache hit (90% cheaper)
for i in range(1000):
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
            },
            {
                "type": "text",
                "text": company_docs,
                "cache_control": {"type": "ephemeral"}  # Reuse cache!
            }
        ],
        messages=[
            {"role": "user", "content": f"Question {i}..."}
        ]
    )
    # Cost: 90% cheaper because docs are cached!
```

### Strategy 2: Model Selection (10x Cheaper for Simple Tasks)

**Problem:**

```
Using GPT-4 for ALL requests:
├─ GPT-4: $0.03 per call (most expensive)
├─ 1M calls/day × $0.03 = $30K/day (expensive!)
├─ Gets you 95% accuracy on all tasks
└─ But overkill for simple tasks
```

**Solution: Right-size Model by Task**

```
Classification task (is this spam?):
├─ Complexity: Low (binary decision)
├─ GPT-4: 95% accuracy, $0.03/call
├─ GPT-3.5: 92% accuracy, $0.001/call
├─ Trade-off: 3% accuracy loss, 30x cheaper!
└─ Decision: Use GPT-3.5 ✓

Summarization task:
├─ Complexity: Medium
├─ GPT-4: 90% quality, $0.03/call
├─ Claude 3 Haiku: 85% quality, $0.002/call
├─ Trade-off: 5% quality loss, 15x cheaper
└─ Decision: Use Haiku ✓

Complex reasoning task:
├─ Complexity: High (multi-step reasoning)
├─ Haiku: 60% accuracy, $0.002/call (not enough!)
├─ GPT-4: 95% accuracy, $0.03/call
└─ Decision: Use GPT-4 ✓

PORTFOLIO APPROACH:
├─ 70% of tasks: Simple (use cheapest model)
│  └─ Spam detection, sentiment, classification
├─ 20% of tasks: Medium (use mid-tier model)
│  └─ Summarization, extraction
└─  10% of tasks: Complex (use best model)
   └─ Reasoning, multi-step logic
```

**Cost impact:**

```
Before: 1M calls × $0.03 (all GPT-4) = $30K

After:
├─ 700K calls × $0.001 (GPT-3.5) = $700
├─ 200K calls × $0.003 (Claude Haiku) = $600
├─ 100K calls × $0.03 (GPT-4) = $3,000
└─ Total: $4,300/day

Savings: $30K → $4.3K = 86% cheaper!
But we said budget is $10K, so:

If current is $10K (already optimized mix):
├─ Optimize to 70/20/10 model split: Save 30%
├─ New cost: $10K × 0.7 = $7K/day ✓
└─ Savings: $3K/day
```

### Strategy 3: Fallback Logic (Rules Before LLM)

**Problem:**

```
Every request goes to LLM, even trivial ones:
├─ User asks: "How do I reset my password?"
│  └─ Send to GPT-4 ($0.03) for answer
│  └─ But answer is always same: "Click here"
│
├─ User asks: "What is your company name?"
│  └─ Send to Claude ($0.002)
│  └─ But answer is in first line of docs
│
└─ Waste of API calls
```

**Solution: Rule-based Routing**

```
Request comes in:
├─ Is it a FAQ? (password reset, hours, pricing?)
│  └─ YES: Return cached FAQ answer (FREE!)
│  └─ NO: Continue
│
├─ Is it a simple lookup? (product info, contact?)
│  └─ YES: Query knowledge base (FREE!)
│  └─ NO: Continue
│
├─ Is it a simple classification? (spam yes/no?)
│  └─ YES: Use regex or simple ML (FREE!)
│  └─ NO: Continue
│
└─ Complex question? → Send to LLM (charge $)

COST SAVINGS:
├─ FAQs (30% of requests): Free instead of $0.002 each
│  └─ Savings: 300K × $0.002 = $600/day
│
├─ Lookups (20% of requests): Free instead of $0.001 each
│  └─ Savings: 200K × $0.001 = $200/day
│
├─ Classifications (20% of requests): Free instead of $0.001 each
│  └─ Savings: 200K × $0.001 = $200/day
│
└─ Only 30% sent to LLM: 300K × average $0.01 = $3K/day

Total: $3,200/day (vs. $10K before!)
```

### Strategy 4: Local Models (Free After Infrastructure)

**Problem:**

```
API calls cost $10K/day = $3.65M/year

Alternatives:
├─ Run Llama 2 locally on your GPU
├─ Quality: 85% of GPT-4 (pretty good)
├─ Cost: $500/month GPU rental = $0.017/day hardware
└─ Savings: $10K → $0.017/day = infinite ROI!
```

**When to use:**

```
Local (Llama) vs. API (GPT-4):

Use Local:
├─ High volume (millions of requests/day)
├─ Latency not critical (5-10s OK)
├─ Privacy critical (data doesn't leave your servers)
├─ Cost-sensitive
└─ Accuracy 85%+ is acceptable

Use API:
├─ High accuracy required (95%+)
├─ Low latency required (<1s)
├─ Variable volume (don't want to buy GPUs)
└─ Don't want operational overhead
```

**Hybrid approach:**

```
Tier requests by importance:
├─ Important queries: Send to GPT-4 (accurate)
├─ Standard queries: Use Llama locally (fast, cheap)
└─ Spam/low-quality: Rule-based (free)

Cost: Could save 80-90% depending on split
```

### Strategy 5: Batch API (50% Discount)

**Problem:**

```
Processing 1M requests/day:
├─ Real-time API: 1M requests × $0.01 = $10K
└─ Batch API: 1M requests × $0.005 = $5K

Trade-off: Latency
├─ Real-time: Response in 5 seconds
├─ Batch: Response in 12-24 hours
```

**When to use:**

```
Use Batch API if:
├─ Offline analysis (not user-facing)
├─ Processing historical data
├─ Can tolerate 12-24 hour delay
├─ Want 50% cost savings

Use Real-time if:
├─ User is waiting for response
├─ Latency < 5 seconds required
├─ Can afford 2x cost
```

#### Step 4: Combined Optimization

**Implement all strategies together:**

```
BEFORE:
├─ 1M calls/day
├─ All models: GPT-4 average
├─ All real-time
├─ No caching, no rules
└─ Cost: $10,000/day

AFTER:
├─ 30% of requests eliminated (FAQ/rules): 0 cost
├─ 50% of requests cheaper model (GPT-3.5): -70% cost
├─ 15% of requests cached: -75% cost
├─ 5% of requests local model: -95% cost
└─ Cost per remaining request: $0.002 (vs. $0.01)

CALCULATION:
├─ 300K rules: $0
├─ 500K GPT-3.5: 500K × $0.001 = $500
├─ 150K cached: 150K × $0.002 = $300
├─  50K local: 50K × $0.0001 = $5
└─ Total: $805/day

SAVINGS: $10,000 → $805 = 92% reduction ✓

More conservatively (being practical):
├─ 20% rules elimination: 200K × $0.01 saved = $2K
├─ 30% model downgrade: 300K × $0.02 saved = $6K
├─ 40% caching: 400K × $0.005 saved = $2K
└─ Cost: $10,000 - $2K - $6K - $2K = $0K (wait, negative?)

Let me recalculate more carefully:
Before: 1M × $0.010 = $10K
After:
├─ 20% to rules (free): 200K requests, $0
├─ 50% to GPT-3.5 (cheap): 500K × $0.001 = $500
├─ 20% to cached: 200K × $0.002 = $400
├─ 10% to local: 100K × $0.0001 = $10
└─ Cost: $910/day ✓ (Close to $1K goal!)
```

#### Step 5: Talking Points

```
"I'd implement a tiered strategy:

1. PROMPT CACHING (immediate, 70% savings)
   - Cache system prompt + docs
   - Reuse across 1000s of requests
   - Cost: $10K → $3K

2. MODEL SELECTION (medium effort, 30% savings)
   - Use GPT-3.5 for classification (cheaper)
   - Use Claude Haiku for summarization
   - Reserve GPT-4 for complex tasks
   - Cost: $3K → $2K

3. RULE-BASED ROUTING (medium effort, 20% savings)
   - FAQ detection → static answers
   - Simple lookups → knowledge base
   - Only send complex queries to LLM
   - Cost: $2K → $1.6K

4. LOCAL MODELS (optional, depends on latency tolerance)
   - Run Llama 2 for non-critical paths
   - Could save another 50-80%

REALISTIC GOAL: 8-10x cost reduction achievable
- Combined strategies can get from $10K to $1K-1.2K/day
- Requires some quality trade-offs (use cheaper models, add fallbacks)
- Trade-off: Speed vs. Cost (some features slower, cached)
"
```

---

## Question 4: Choose GCP vs Azure

### Problem Statement

**"We're building a recommendation system for 50M users. Should we use GCP Vertex AI or Azure ML?"**

### Answer Framework

#### Step 1: Evaluate on Key Dimensions

**Scoring Matrix:**

```
┌────────────────────────────────────────────────────────────┐
│ Evaluation Criteria         │ GCP Vertex │ Azure ML │ Winner │
├────────────────────────────────────────────────────────────┤
│ Analytics Performance       │    ★★★★★   │   ★★★★   │ GCP    │
│ (BigQuery vs Synapse)       │            │          │        │
├────────────────────────────────────────────────────────────┤
│ Feature Store Maturity      │    ★★★★★   │   ★★★    │ GCP    │
│ (Production vs Preview)     │            │          │        │
├────────────────────────────────────────────────────────────┤
│ Real-time Serving           │    ★★★★★   │   ★★★★   │ GCP    │
│ (Latency, Throughput)       │            │          │        │
├────────────────────────────────────────────────────────────┤
│ Cost                        │    ★★★★★   │   ★★★★   │ GCP    │
│ (30% cheaper overall)       │            │          │        │
├────────────────────────────────────────────────────────────┤
│ Enterprise Compliance       │    ★★★★    │   ★★★★★   │ Azure  │
│ (HIPAA, FedRAMP)            │            │          │        │
├────────────────────────────────────────────────────────────┤
│ Integration Ecosystem       │    ★★★★    │   ★★★★★   │ Azure  │
│ (Office 365, Power BI, CRM) │            │          │        │
├────────────────────────────────────────────────────────────┤
│ LLM / Generative AI         │    ★★★★    │   ★★★★★   │ Azure  │
│ (Gemini vs Azure OpenAI)    │            │          │        │
├────────────────────────────────────────────────────────────┤
│ Developer Experience        │    ★★★★    │   ★★★★   │ Tie    │
│                             │            │          │        │
└────────────────────────────────────────────────────────────┘

OVERALL: GCP better for ML, Azure better for Enterprise
```

#### Step 2: Detailed Comparison

**A) Data Management**

```
TASK: Prepare 50M user records for training

GCP Approach:
├─ Data stored in BigQuery (data warehouse)
├─ Query exploration (super fast SQL):
│  ├─ "SELECT COUNT(*) WHERE age > 30" → 2 seconds (on 10B rows!)
│  ├─ BigQuery is optimized for analytical queries
│  └─ Dev-friendly, iterate fast in Jupyter
├─ Export to training → 5 minutes
└─ Cost: ~$1K/month (BigQuery + storage)

Azure Approach:
├─ Data stored in ADLS Gen2 (data lake)
├─ Query via Synapse Analytics (SQL warehouse)
│  ├─ "SELECT COUNT(*) WHERE age > 30" → 5 seconds (slower)
│  ├─ Synapse requires more setup (cluster configuration)
│  └─ More powerful for complex transforms (Spark)
├─ Export to training → 10 minutes
└─ Cost: ~$1.5K/month (Synapse compute + storage)

WINNER: GCP (faster analytics, simpler, cheaper)
```

**B) Feature Store**

```
REQUIREMENT: Serve features in <10ms for real-time inference

GCP Vertex Feature Store:
├─ Status: Fully managed, production-ready
├─ Online serving latency: <10ms (optimized)
├─ Offline batch serving: BigQuery integration
├─ Setup time: 30 minutes
└─ Cost: $500/month (online serving) + storage

Azure Feature Store:
├─ Status: In preview (still being developed)
├─ Online serving latency: <50ms (slower)
├─ Offline batch serving: Synapse integration
├─ Setup time: 1-2 hours (more complex)
└─ Cost: $500/month + custom infrastructure

WINNER: GCP (production-ready, faster, simpler)
```

**C) Model Training**

```
TASK: Train recommendation model on 50M users

Both platforms similar:
├─ Can use pre-built containers (PyTorch, TensorFlow)
├─ Can use custom Docker images
├─ Can scale to multiple GPUs
├─ Cost similar (~$0.35-0.50 per GPU-hour)

GCP Vertex Training:
├─ Setup: 10 minutes
├─ Launch: gcp.run_training(...)
├─ Typical time: 2 hours
└─ Cost: 2h × 8 GPUs × $0.35 = $5.60

Azure ML Training:
├─ Setup: 15 minutes
├─ Launch: ml_client.create_job(...)
├─ Typical time: 2 hours
└─ Cost: 2h × 8 GPUs × $0.50 = $8

WINNER: Slight tie (GCP 30% cheaper)
```

**D) Model Serving**

```
REQUIREMENT: Serve to 50M users, <100ms latency

GCP Vertex Endpoints:
├─ Protocol: gRPC (low-latency binary)
├─ Latency: <50ms per request (fast)
├─ Auto-scaling: Configured in minutes
├─ Cost: $0.05 per replica-hour

Azure ML Endpoints:
├─ Protocol: REST/HTTP (higher overhead)
├─ Latency: <100ms per request (slower)
├─ Auto-scaling: Configured in minutes
├─ Cost: $0.08 per replica-hour

WINNER: GCP (faster protocol, cheaper)
```

**E) Enterprise & Compliance**

```
REQUIREMENT: HIPAA compliance for medical data

GCP:
├─ Supports HIPAA (with BAA agreement)
├─ Audit logging: Available
├─ Data residency: Limited regions
└─ Compliance: OK but not primary focus

Azure:
├─ Native HIPAA support (built-in)
├─ Audit logging: Comprehensive
├─ Data residency: Many region options
├─ FedRAMP certified
└─ Compliance: Primary strength

WINNER: Azure (better for regulated data)
```

**F) LLM / Generative AI**

```
RECOMMENDATION SYSTEM with LLM ranking (2024 trend)

GCP:
├─ Vertex Generative AI API:
│  ├─ Access Gemini, Claude
│  ├─ Pricing: Competitive
│  └─ Setup: Simple
├─ Model Garden: Open-weight models (Llama, Mistral)
└─ Integration: Works with Vertex Endpoints

Azure:
├─ Azure OpenAI Service:
│  ├─ Access GPT-4, GPT-3.5
│  ├─ Dedicated capacity available
│  ├─ Pricing: Higher (Azure markup)
│  └─ Setup: Requires Azure OpenAI resource
├─ Model Catalog: Open-weight models
├─ Prompt Flow: Visual tool for LLM chains
│  └─ Powerful for prototyping
└─ Better for LLM-first applications

WINNER: Azure (better LLM integration, Prompt Flow)
```

#### Step 3: Architecture Decision

**For Recommendation System (50M users):**

**Option A: GCP-first (Recommended if ML-focused)**

```
Architecture:
┌─────────────────────────────────────────┐
│   BigQuery                              │
│   └─ Raw user data, interactions       │
├─────────────────────────────────────────┤
│   Vertex AI Feature Store               │
│   ├─ Online: Feature serving <10ms     │
│   └─ Offline: Batch feature export     │
├─────────────────────────────────────────┤
│   Vertex AI Training                    │
│   └─ Train recommendation model         │
├─────────────────────────────────────────┤
│   Model Registry                        │
│   └─ Version & track models            │
├─────────────────────────────────────────┤
│   Vertex AI Endpoints                   │
│   └─ Serve predictions, gRPC, <50ms   │
├─────────────────────────────────────────┤
│   Cloud Monitoring + Logging            │
│   └─ Monitor drift, latency             │
└─────────────────────────────────────────┘

Cost: ~$30K/month
├─ BigQuery: $5K
├─ Feature Store: $2K
├─ Training: $5K (nightly batches)
├─ Serving: $15K (inference servers)
└─ Monitoring: $3K

Strengths:
✓ Cohesive platform (everything works together)
✓ Fast analytics (BigQuery)
✓ Lowest cost
✓ Production-ready feature store

Weaknesses:
✗ Less enterprise integration (no Office 365, CRM)
✗ Limited LLM options (if wanting GPT-4)
```

**Option B: Azure-first (Recommended if Enterprise-focused)**

```
Architecture:
┌─────────────────────────────────────────┐
│   Azure Synapse                         │
│   └─ Data warehouse (SQL + Spark)      │
├─────────────────────────────────────────┤
│   Azure Feature Store (Preview)         │
│   ├─ Online: Feature serving <50ms     │
│   └─ Offline: Synapse integration      │
├─────────────────────────────────────────┤
│   Azure ML Training                     │
│   └─ Train recommendation model         │
├─────────────────────────────────────────┤
│   Model Registry                        │
│   └─ Version & track models            │
├─────────────────────────────────────────┤
│   Azure ML Endpoints                    │
│   └─ Serve predictions, REST, <100ms  │
├─────────────────────────────────────────┤
│   Prompt Flow (optional, for LLM)       │
│   └─ Rank recommendations with LLM     │
├─────────────────────────────────────────┤
│   Application Insights                  │
│   └─ Monitor drift, latency, fairness  │
├─────────────────────────────────────────┤
│   Power BI                              │
│   └─ Executive dashboards              │
└─────────────────────────────────────────┘

Cost: ~$40K/month
├─ Synapse: $8K
├─ Feature Store: $2K
├─ Training: $8K (nightly batches)
├─ Serving: $18K (inference servers)
├─ Monitoring: $2K
└─ Power BI: $2K

Strengths:
✓ Enterprise integration (Synapse, Power BI, Office)
✓ Better LLM support (Azure OpenAI, Prompt Flow)
✓ HIPAA, FedRAMP compliance
✓ Better for regulated industries

Weaknesses:
✗ Higher cost (33% more expensive)
✗ Slower feature serving (<50ms vs <10ms)
✗ Feature store still in preview
✗ Steeper learning curve
```

#### Step 4: Decision Tree

```
Do you already use GCP or Azure?
├─ YES (GCP) → Use Vertex AI
├─ YES (Azure) → Use Azure ML
└─ NO → Continue below

Is LLM ranking a core feature?
├─ YES → Prefer Azure (OpenAI integration, Prompt Flow)
└─ NO → Prefer GCP (simpler, cheaper)

Do you need strict compliance (HIPAA, FedRAMP)?
├─ YES → Use Azure ML
└─ NO → Prefer GCP

Is cost a primary driver?
├─ YES → GCP Vertex AI (30% cheaper)
└─ NO → Azure ML (better integration)

Do you need real-time <50ms feature serving?
├─ YES → GCP (production-ready feature store)
└─ NO → Azure acceptable (in preview, <50ms)

FINAL DECISION:
├─ For ML teams: GCP Vertex AI ✓
├─ For enterprises: Azure ML ✓
├─ For cost-sensitive: GCP ✓
└─ For compliance-heavy: Azure ✓
```

#### Step 5: Talking Points

```
"I'd recommend GCP Vertex AI for this project because:

1. BIGQUERY ADVANTAGE
   - Recommendation systems need heavy analytics
   - BigQuery is faster and cheaper than Synapse
   - SQL exploration is 2-3x faster
   - Save $1K-2K/month on analytics

2. FEATURE STORE
   - Vertex Feature Store is production-ready
   - Azure's is still in preview (risky)
   - Serves features in <10ms (critical for real-time)
   - Integration with BigQuery is seamless

3. COST
   - 30% cheaper than Azure overall
   - $30K/month vs $40K/month
   - Over 3 years: $360K savings

4. SIMPLICITY
   - All components from one vendor
   - Fewer integrations to manage
   - Easier to operate

HOWEVER, if this company is Azure-first or needs:
- Compliance (HIPAA, FedRAMP) → Azure
- LLM ranking (using GPT-4) → Azure
- Office 365 / Power BI integration → Azure

Then Azure ML is the right choice, despite higher cost.
"
```

---

## Question 5: Detect & Respond to Data Drift

### Problem Statement

**"Design a system to detect when your model's input data has drifted from training data. How would you respond?"**

### Answer Framework

#### Step 1: Drift Detection Architecture

**System Overview:**

```
┌─────────────────────────────────────────────────────────┐
│                 Production Data Stream                  │
│              (1M requests/day from users)               │
└────────────────┬────────────────────────────────────────┘
                 │
                 │ Sample 10K requests/day
                 │
        ┌────────▼────────┐
        │  Drift Detection │
        │  Job (Daily)     │
        └────────┬────────┘
                 │
         ┌───────┴───────┐
         │               │
    ┌────▼────┐    ┌─────▼─────┐
    │ Compare │    │ Run Tests  │
    │ Distrib │    │ (KS, Chi)  │
    └────┬────┘    └─────┬─────┘
         │               │
         └───────┬───────┘
                 │
          ┌──────▼───────┐
          │ p-value < ?  │
          └──────┬───────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
  <0.01        0.01-0.05    >0.05
    │            │            │
CRITICAL      MEDIUM        OK
    │            │            │
  PAGE       EMAIL TEAM    LOG &
  ONCALL     INVESTIGATE  MONITOR
```

#### Step 2: Baseline Computation (Training Time)

**During model training:**

```python
import numpy as np
from scipy.stats import ks_2samp

# After training, compute baseline statistics
training_data = load_training_data()

baseline_stats = {}
for feature in training_data.columns:
    baseline_stats[feature] = {
        'mean': training_data[feature].mean(),
        'std': training_data[feature].std(),
        'p25': training_data[feature].quantile(0.25),
        'p50': training_data[feature].quantile(0.50),
        'p75': training_data[feature].quantile(0.75),
        'p95': training_data[feature].quantile(0.95),
        'min': training_data[feature].min(),
        'max': training_data[feature].max(),
    }

# Store in version control (immutable)
save_baseline(baseline_stats, model_version='v2.1')

# Example output:
# {
#     'age': {
#         'mean': 35.2,
#         'std': 12.1,
#         'p95': 60,
#         ...
#     },
#     'income': {
#         'mean': 75000,
#         'std': 45000,
#         ...
#     }
# }
```

#### Step 3: Daily Drift Detection Job

**Runs every morning at 6am:**

```python
# Load production data from last 24 hours
production_data = load_production_data(days=1)

# Load baseline
baseline = load_baseline('v2.1')

# Run drift tests
drift_results = {}

for feature in baseline.keys():
    prod_values = production_data[feature]
    baseline_values = training_data[feature]  # Recompute or store
    
    # Run Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(baseline_values, prod_values)
    
    drift_results[feature] = {
        'p_value': p_value,
        'severity': 'CRITICAL' if p_value < 0.01
                   else 'MEDIUM' if p_value < 0.05
                   else 'OK',
        'baseline_mean': baseline[feature]['mean'],
        'prod_mean': prod_values.mean(),
        'mean_diff_pct': abs(prod_values.mean() - baseline[feature]['mean']) 
                        / baseline[feature]['mean'] * 100
    }

# Example output:
# {
#     'age': {
#         'p_value': 0.0001,
#         'severity': 'CRITICAL',
#         'baseline_mean': 35.2,
#         'prod_mean': 38.1,
#         'mean_diff_pct': 8.2%
#     },
#     'income': {
#         'p_value': 0.32,
#         'severity': 'OK',
#         ...
#     }
# }
```

#### Step 4: Alerting & Routing

**Based on drift severity:**

```
┌─────────────────────────────────────┐
│        Drift Detection Results      │
└────────────────┬────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
   ┌────▼────┐      ┌──────▼──────┐
   │ Feature │      │   Feature   │
   │ A Drift │      │   B Drift   │
   │p=0.0001 │      │  p=0.08     │
   │CRITICAL │      │   MEDIUM    │
   └────┬────┘      └──────┬──────┘
        │                  │
        │            ┌─────▼──────┐
        │            │ Email:     │
        │            │ data_team  │
        │            │ Investigate│
        │            └────────────┘
   ┌────▼────────────────┐
   │ Page Oncall Alert   │
   │ "Feature A drifted" │
   │ "Age: 35 → 38"      │
   │ "Investigate ASAP"  │
   └────────────────────┘
```

#### Step 5: Investigation Playbook

**IF drift detected: What does data team do?**

```
STEP 1: Classify Drift Type (10 minutes)

Seasonal Drift?
├─ Does this pattern repeat annually?
├─ Example: "Holiday shopping" → higher income, younger users
├─ Action: EXPECTED, log and monitor
└─ No immediate action

New Cohort?
├─ Did we expand to new market?
├─ Example: "We're now in rural areas" → lower income
├─ Action: Investigate business context
└─ Probably need retraining

Data Quality Issue?
├─ Null values spike?
├─ Out-of-range values (age = 999)?
├─ Missing features?
├─ Action: FIX PIPELINE FIRST
└─ Then retrain

Feedback Loop?
├─ Is model's past prediction affecting current data?
├─ Example: Model recommends product → user clicks → label as positive
├─ Action: Fix labeling pipeline
└─ Retrain with clean labels
```

**STEP 2: Root Cause Analysis (30 minutes)**

```
Drift in 'age' feature: 35 → 38 (slightly older users)

Possible causes:
├─ 1. Seasonal (expected): Do users age 5% in summer?
│     → Probably not, monthly variance
│
├─ 2. Marketing changed: Targeting older demographic?
│     → Check: Did marketing team change campaigns?
│     → Query marketing DB for recent changes
│
├─ 3. Data collection issue: Age calculated differently?
│     → Check: How is age_feature computed?
│     → Compare: definition in training vs. serving
│
├─ 4. Sampling bias: Only older users responded?
│     → Check: What % of users responded?
│     → Is sampling representative?
│
└─ 5. Natural shift: User base aging (time passing)?
    → Check: Is trend consistent over weeks?
    → Acceptable if slow natural change
```

**STEP 3: Decision: Retrain or Monitor? (1 hour)**

```
Decision Tree:

Drift is seasonal?
├─ YES → No action, continue monitoring
└─ NO → Continue below

Accuracy on new data (if ground truth available)?
├─ >= 90% (acceptable) → Monitor closely, retrain next cycle
├─ 85-90% (minor drop) → Retrain this week
└─ < 85% (major drop) → Retrain immediately

Can we understand root cause?
├─ YES (knew marketing changed) → Retrain is safe
├─ NO (mysterious drift) → Monitor carefully, investigate more

Is drift temporary (weather, holiday)?
├─ YES → No retrain needed, use adaptive threshold
└─ NO → Retrain

EXAMPLES:
├─ Seasonal drift + no accuracy drop → No action
├─ New cohort + 5% accuracy drop → Retrain this week
├─ Mystery drift + 10% drop → Retrain immediately, post-incident
└─ Data quality bug + 15% drop → Fix pipeline, retrain immediately
```

#### Step 6: Retraining Response

**IF decision is: RETRAIN**

```
Timeline:
├─ Decision (t=0): "Need to retrain"
├─ Trigger training (t=0-10 min): Submit new training job
├─ Training (t=10 min - t=2 hours): Model trains on recent data
├─ Evaluation (t=2h - t=2.5h):
│  ├─ Compare new model vs. old on test set
│  ├─ Confirm improvement
│  └─ If accuracy < old model: investigate before deploying
├─ Canary deploy (t=2.5h - t=6h): 5% traffic to new model
├─ Monitor (t=6h - t=12h): Watch latency, errors, accuracy
├─ Full deploy (t=12h): If all good, send 100% traffic
└─ Done (t=12h): New model in production

Cost: ~$100 (training compute)
Risk: Low (canary deployment)
Time: 12 hours total
```

#### Step 7: Prevention

**To prevent similar drifts:**

```
1. Add automated checks:
   ├─ Daily drift detection (already doing)
   ├─ Weekly fairness audits (new subgroups?)
   └─ Monthly analysis review (team discussion)

2. Data quality improvements:
   ├─ Add schema validation (catch null values early)
   ├─ Add range checks (age should be 0-150)
   ├─ Add anomaly detection (sudden changes)
   └─ Add missing value alerts

3. Monitoring enhancements:
   ├─ Track feature importance over time
   ├─ Alert if important feature missing
   ├─ Alert if new values not seen in training
   └─ Track ground truth (when available)

4. Process improvements:
   ├─ Document why drift happened
   ├─ Update runbook for next time
   ├─ Brief team on learnings
   └─ Prevent same root cause

5. Model improvements:
   ├─ Consider robust training (handles distribution shift)
   ├─ Consider online learning (continuous adaptation)
   └─ Consider ensemble (multiple models robust to drift)
```

#### Step 8: Example Response

**What to say in interview:**

```
"I'd design a comprehensive drift detection system:

1. BASELINE (at training time)
   - Compute distribution statistics on training data
   - Store: mean, std, percentiles for each feature
   - Version this baseline with the model

2. DAILY MONITORING
   - Extract last 24h production data (sample 10K requests)
   - Run KS test: Compare to baseline distribution
   - p-value < 0.01: Critical (page oncall)
   - p-value < 0.05: Medium (email team)
   - p-value > 0.05: Normal (log and continue)

3. INVESTIGATION
   - Classify drift type (seasonal, new cohort, data issue)
   - Root cause analysis (marketing change? pipeline bug?)
   - Decision: Retrain or monitor?

4. RESPONSE
   - If critical: Retrain immediately
   - If medium: Investigate, plan retrain for next cycle
   - If seasonal: Expected, continue monitoring

5. PREVENTION
   - Add data quality checks
   - Track feature importance changes
   - Post-mortem after each drift event
   - Update runbooks

The key insight is: drift detection is a SYSTEM, not just a single check.
You need automated detection, human investigation, and continuous improvement.
"
```

---

## Question 6: Integrate MLflow with GCP & Train Large Data

### Problem Statement

**"How would you integrate MLflow with GCP for experiment tracking? How would you train a model on 100GB+ of data on Vertex AI?"**

### Answer Framework

#### Step 1: MLflow + GCP Architecture

**Overall Setup:**

```
┌──────────────────────────────────────────────────────┐
│            MLflow + GCP Integration                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Local/Notebook (Development)                        │
│  ├─ Train small data locally                         │
│  ├─ Log experiments to MLflow                        │
│  └─ Tag best model                                   │
│           ↓                                          │
│  MLflow Server (Centralized)                         │
│  ├─ Runs database (track experiments)               │
│  ├─ Model registry (store versions)                 │
│  └─ Artifacts storage (GCS bucket)                  │
│           ↓                                          │
│  Vertex AI Training (Distributed)                    │
│  ├─ Train on 100GB+ data                            │
│  ├─ Log metrics to MLflow (real-time)               │
│  ├─ Register model automatically                    │
│  └─ Store model in Model Registry                   │
│           ↓                                          │
│  Model Serving (Vertex Endpoints)                    │
│  ├─ Deploy best model from registry                 │
│  ├─ A/B test against previous                       │
│  └─ Monitor performance                             │
│                                                      │
└──────────────────────────────────────────────────────┘
```

#### Step 2: Setting Up MLflow on GCP

**Option A: MLflow Server (Self-Managed)**

```python
# Step 1: Create MLflow server on Compute Engine

# Cloud Shell commands:
gcloud compute instances create mlflow-server \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-2 \
    --zone=us-central1-a

# Step 2: SSH into instance and install MLflow
gcloud compute ssh mlflow-server --zone=us-central1-a

# On instance:
pip install mlflow google-cloud-storage

# Step 3: Create GCS bucket for artifacts
gsutil mb gs://my-mlflow-artifacts/

# Step 4: Start MLflow server
mlflow server \
    --backend-store-uri postgresql://user:password@db-host/mlflow \
    --default-artifact-root gs://my-mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000

# Step 5: Expose via Cloud Load Balancer (make accessible)
# Set firewall rules, add SSL certificate, etc.
```

**Option B: Managed MLflow (Vertex AI Experiments) - Simpler**

```python
# Google Cloud's native MLflow integration (Recommended)

from google.cloud import aiplatform
from google.cloud.aiplatform import start_run, log_metrics

# Initialize Vertex AI
aiplatform.init(
    project="my-project",
    location="us-central1"
)

# Log experiments to Vertex AI (native alternative to MLflow)
with aiplatform.start_run(run="my-experiment-1"):
    # Train model
    model = train_model(data)
    
    # Log metrics (auto-synced to Vertex AI)
    aiplatform.log_metrics({
        "accuracy": 0.95,
        "loss": 0.05,
        "f1": 0.93
    })
    
    # Log model (stored in Model Registry)
    aiplatform.log_model(
        model,
        artifact_id="my-model",
        artifact_uri="gs://my-bucket/models/model.pkl"
    )
```

**Recommended Approach:**
- Use **Vertex AI Experiments** (managed, integrates with Vertex Training)
- Falls back to **MLflow** if need portability across clouds

#### Step 3: Training 100GB+ Data on Vertex AI

**Challenge: Data too large for single machine**

```
Problem:
├─ 100GB dataset doesn't fit in memory
├─ Single GPU/CPU would take days to train
└─ Need distributed training

Solution: Vertex AI Training with distributed approach
├─ Split data across Spot Instances (70% cheaper)
├─ Use Spot VM failures are OK for batch jobs
└─ Autoscaling: start with 10, scale to 100 if needed
```

**Architecture:**

```
┌────────────────────────────────────────────────────┐
│          Vertex AI Training (Distributed)         │
├────────────────────────────────────────────────────┤
│                                                   │
│ Data Layer:                                       │
│ ├─ BigQuery: 100GB dataset (SQL-queryable)       │
│ ├─ Cloud Storage: Training data in TFRecord      │
│ └─ Data API: Streaming to workers (5GB/min)      │
│                                                   │
│ Training Layer (Spot Instances):                 │
│ ├─ Worker 1: Process chunk 1-10GB                │
│ ├─ Worker 2: Process chunk 10-20GB               │
│ ├─ Worker 3: Process chunk 20-30GB               │
│ └─ 10 total workers (auto-scale to 20-50)        │
│                                                   │
│ Coordination:                                     │
│ ├─ Parameter Server: aggregates model updates    │
│ ├─ Sync: All workers sync gradients every step   │
│ └─ Failover: If worker dies, auto-restart        │
│                                                   │
│ Logging:                                          │
│ ├─ Real-time: Loss/accuracy to MLflow            │
│ ├─ Frequency: Every 100 steps                    │
│ └─ Storage: Cloud Logging + BigQuery             │
│                                                   │
└────────────────────────────────────────────────────┘
```

#### Step 4: Code Example - Training 100GB on Vertex AI

**Training Script (runs on Vertex AI):**

```python
# train.py - Submitted to Vertex AI Training

import os
import json
import mlflow
from tensorflow.keras import models, layers
from google.cloud import bigquery, storage
import tensorflow as tf

# Step 1: Connect to MLflow (logs to Vertex AI)
mlflow.set_tracking_uri("gs://my-mlflow-artifacts")  # GCS backend

# Step 2: Read data from BigQuery (streaming, doesn't fit in memory)
def load_training_data():
    """Load 100GB data in batches"""
    client = bigquery.Client()
    query = """
    SELECT 
        feature_1, feature_2, feature_3, ..., label
    FROM `project.dataset.training_data`
    LIMIT 1000000  -- Load 1M rows at a time
    """
    
    # Returns streaming iterator
    return client.query(query).to_arrow()

# Step 3: Distributed training (data parallelism)
def train_distributed():
    strategy = tf.distribute.MirroredStrategy()  # Multi-GPU on single machine
    # OR tf.distribute.MultiWorkerMirroredStrategy() for multi-machine
    
    with strategy.scope():
        # Build model
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(100,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    # Load data in batches (doesn't require all 100GB in memory)
    data = load_training_data()
    
    # Train with MLflow logging
    with mlflow.start_run() as run:
        mlflow.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam"
        })
        
        # Training loop with logging
        history = model.fit(
            data,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[
                # Log metrics every epoch
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: mlflow.log_metrics({
                        "loss": logs['loss'],
                        "accuracy": logs['accuracy'],
                        "val_loss": logs['val_loss'],
                        "val_accuracy": logs['val_accuracy']
                    }, step=epoch)
                )
            ]
        )
        
        # Log final model
        mlflow.keras.log_model(model, "model")
        mlflow.log_artifact("model_summary.txt")

if __name__ == "__main__":
    train_distributed()
```

**Submit to Vertex AI:**

```bash
# Create Python package
mkdir training_app
cp train.py training_app/
cp requirements.txt training_app/

# Build Docker image (optional, or use pre-built)
gcloud builds submit --tag gcr.io/my-project/training-image training_app/

# Submit training job
gcloud ai custom-training create \
    --display-name="large-model-training" \
    --script-path=training_app/train.py \
    --machine-type=n1-highmem-8 \
    --accelerator-type=NVIDIA_TESLA_V100 \
    --accelerator-count=2 \
    --training-container-image-uri=gcr.io/my-project/training-image \
    --replica-count=10 \  # 10 workers
    --enable-autoscaling \
    --max-replicas=50 \  # Scale up to 50 if needed
    --config=training_config.yaml
```

#### Step 5: Cost Optimization for Large Training

**Cost Breakdown (100GB data, 10 workers, 8 hours training):**

```
Spot Instances (70% discount):
├─ 10 workers × 8 hours × $0.50/hour (discounted) = $40
├─ vs. on-demand: $0.50 × 10 × 8 = $40 (before discount, would be $133)
└─ Savings: $93 for this job alone!

Data Transfer:
├─ BigQuery → Vertex: 100GB × $0.05/GB = $5 (outbound data)
├─ Store results: 1GB model × $0.02/GB = $0.02
└─ Logging: negligible

Total: ~$45-50 for entire training job

Comparison:
├─ Training locally: N/A (can't fit 100GB in memory)
├─ On-demand instances: $133 (very expensive)
├─ Spot instances: $40 (best)
└─ ROI: Model accuracy improves → saves $1000s in bad predictions
```

**Optimization Strategies:**

```
1. USE SPOT INSTANCES (save 70%)
   ├─ Worker failures OK (can retry)
   ├─ But turn off for final 10 epochs (use on-demand)
   └─ Cost: ~$50 instead of $150

2. REDUCE DATA (if possible)
   ├─ Stratified sampling: 10GB instead of 100GB
   ├─ Maybe 90% same accuracy with 10x less data
   └─ Training: 1 hour instead of 8 hours

3. OPTIMIZE MODEL
   ├─ Smaller model first (fewer parameters)
   ├─ Train on 10GB, then fine-tune on full 100GB
   ├─ Example: ResNet-50 → ResNet-18 (less compute)
   └─ Cost: 3x training speedup

4. BATCH SIZE & LEARNING RATE
   ├─ Larger batch size: 32 → 128 (fewer gradient updates)
   ├─ Higher learning rate: 0.001 → 0.01 (converges faster)
   └─ But careful: too aggressive → doesn't converge

5. MIXED PRECISION TRAINING
   ├─ Use fp16 for faster computation
   ├─ 2x speedup with minimal accuracy loss
   └─ Code: model.mixed_precision.Policy('mixed_float16')
```

#### Step 6: Monitoring Training Progress

**Real-Time Monitoring:**

```python
# View training progress live
mlflow.search_runs(
    experiment_ids=[run_id],
    filter_string="",
    max_results=1
)

# Or use Vertex AI dashboard
# gcloud ai custom-training describe {job-id}

# Metrics tracked:
├─ Loss: decreasing over time
├─ Accuracy: increasing over time
├─ Training time: 2h, 4h, 6h, 8h
├─ GPU memory: stable around 80%
├─ Data throughput: 1.5GB/min from BigQuery
└─ Worker failures: auto-recovered (3 total)
```

#### Step 7: Talking Points

```
STRENGTHS:
✓ "Vertex AI handles distributed training automatically"
✓ "Data streaming from BigQuery (doesn't need all in memory)"
✓ "Spot instances save 70% (job-level failure tolerance)"
✓ "MLflow integrates seamlessly (track all experiments)"
✓ "Autoscaling: start 10, scale to 50 workers if needed"
✓ "Model registry: automatic versioning of best models"

TRADE-OFFS:
✓ "Setup complexity: requires Docker, config files"
✓ "Spot failures: occasional restarts (OK for training)"
✓ "Cost still high: $50/job × 100s of experiments = overhead"

OPTIMIZATION POINTS:
✓ "Reduce data: 90% accuracy on 10GB vs 100GB?"
✓ "Smaller model: ResNet-18 instead of ResNet-50"
✓ "Batch size: balance speed vs. convergence"
✓ "Mixed precision: 2x faster with fp16"
✓ "Cache features: if same preprocessing needed"
```

---

# Part 3: Quick References

## Red Flags: What NOT to Say

### ❌ Anti-Pattern: Reactive Operations

**DON'T SAY:**
> "We just retrain the model when accuracy drops"

**WHY BAD:**
- No monitoring (don't know accuracy dropped)
- Manual process (slow, error-prone)
- No root cause analysis (might fix wrong thing)

**DO SAY:**
> "We automatically detect drift using daily KS tests. When accuracy drops, we investigate root cause (data drift vs. concept drift vs. feedback loop), then decide to retrain, adjust threshold, or rollback. All changes go through canary deployment with monitoring."

---

### ❌ Anti-Pattern: Monolithic Deployment

**DON'T SAY:**
> "We test the model locally, then push to production"

**WHY BAD:**
- No automated testing
- No gradual rollout (all or nothing)
- Hard to rollback

**DO SAY:**
> "We have CI/CD: data validation → model tests → A/B test on production traffic (5% canary) → monitor metrics → gradually increase to 100%. If metrics degrade, auto-rollback."

---

### ❌ Anti-Pattern: No Fallback

**DON'T SAY:**
> "The model runs inference for every request"

**WHY BAD:**
- If model crashes, entire service down
- No graceful degradation

**DO SAY:**
> "We have multi-layer serving: batch precomputation for 80% (cache hit), online for 20% (personalized). If model unavailable, fall back to cached results from yesterday."

---

### ❌ Anti-Pattern: Single Metric

**DON'T SAY:**
> "We monitor accuracy"

**WHY BAD:**
- Accuracy doesn't capture latency, fairness, business impact
- Offline accuracy doesn't predict production performance

**DO SAY:**
> "We monitor system metrics (latency p99, error rate), model metrics (accuracy, calibration), data quality (drift, missing values), and business metrics (conversion rate, revenue impact)."

---

### ❌ Anti-Pattern: Vague Cost Management

**DON'T SAY:**
> "We use cloud resources"

**WHY BAD:**
- No cost visibility
- Runaway costs

**DO SAY:**
> "We have clear cost breakdown: 60% compute, 20% storage, 10% tools, 10% team. We optimize with spot instances, batch processing, model quantization. Cost per prediction: $0.0002."

---

## Talking Point Templates

### When asked "Tell me about a project you built"

**Template:**

```
"I built a [recommendation/fraud detection/churn prediction] system that:

1. SCALE
   - Processes [100K/1M/100M] [users/transactions/requests] [daily/monthly]
   - Serves [500/5K/50K] predictions per second

2. APPROACH
   - Architecture: [Batch + Online / Online only]
   - Model type: [XGBoost / Neural Network / LLM-based]
   - Key challenge: [Latency / Cost / Fairness]

3. OPERATIONS
   - Training: Automated pipeline, retrain [daily/weekly/on-demand]
   - Deployment: Canary rollout, [X] hour monitoring period
   - Monitoring: [Drift detection / Fairness audits / Cost tracking]

4. BUSINESS IMPACT
   - Improved [metric]: from [X]% to [Y]% (Z% lift)
   - Saved [time/money]: $[X]K annually
   - Prevented [bad outcomes]: [fraud cases/churn/etc]

5. LEARNINGS
   - Key challenge: [feedback loops / data quality / cold start]
   - Solution: [approach, tool, process]
   - What I'd do differently: [optimization]
"
```

**Example:**

```
"I built a recommendation system for an e-commerce platform:

1. SCALE
   - Processes 50M user interactions daily
   - Serves 5K recommendations per second

2. APPROACH
   - 80% batch (nightly precompute), 20% online (real-time)
   - Neural network with embeddings
   - Challenge: Serving <100ms with large model

3. OPERATIONS
   - Retrain daily on fresh user data
   - Canary deploy (5% traffic for 4 hours)
   - Monitor for drift via KS test

4. BUSINESS IMPACT
   - Increased conversion: 1.5% → 2.1% (40% lift)
   - $5M incremental revenue annually
   - Paid for infrastructure 10x over

5. LEARNINGS
   - Biggest issue: feedback loop (our recs bias the labels)
   - Fixed by using only unbiased user feedback
   - Retrain latency critical → quantized model (4x faster)
"
```

---

### When asked about handling failures

**Template:**

```
"We have defense-in-depth for failures:

1. DETECTION (automated alerts)
   - System: latency > 200ms, error rate > 5%
   - Model: accuracy drop > 5%, drift p-value < 0.05
   - Data: null values spike, feature missing

2. RESPONSE (fast remediation)
   - If critical: Rollback model [minutes]
   - If data drift: Trigger retrain [hours]
   - If infrastructure: Auto-scale capacity

3. FALLBACK (graceful degradation)
   - Model unavailable: Serve cached results
   - Feature unavailable: Use default value
   - Service overloaded: Return approximate results

4. POST-MORTEM (prevent recurrence)
   - Incident review within 24 hours
   - Root cause analysis
   - Process improvement
   - Runbook update
"
```

---

### When asked to scale to 10x users

**Template:**

```
"Three levers to scale 10x:

1. INFRASTRUCTURE (3-5x cost increase)
   - Training: Distributed training across 16 GPUs (was 4)
   - Serving: 50 GPU servers (was 10), auto-scale to 200
   - Data: Shard across regions, local caches

2. COMPUTE (2-3x speedup)
   - Quantize model: 4x faster inference
   - Batch requests: GPU processes 32 at once (was 4)
   - Cache embeddings: Avoid recomputation

3. MODEL (slight accuracy trade-off)
   - Smaller model: 10x fewer parameters
   - Pruning: Remove unimportant weights
   - Distillation: Student model mimics large model

TRADE-OFFS:
- Cost: Increases 3-5x
- Latency: Stays same (optimization)
- Accuracy: Slight drop (1-2%) if needed
- Complexity: Increases significantly
"
```

---

### When asked about costs

**Template:**

```
"MLOps is expensive, but manageable with discipline:

COST BREAKDOWN:
├─ Compute (training + serving): 60% = $30K/month
├─ Storage (data + models + logs): 20% = $10K/month
├─ Tools (MLflow, monitoring, etc): 10% = $5K/month
└─ Team (salary allocation): 10% = $5K/month
   Total: $50K/month

OPTIMIZATION STRATEGIES:

Quick Wins:
├─ Spot instances for training: -70%
├─ Batch processing: -50%
├─ Cache predictions: -80%
└─ Easy, immediate savings

Medium Effort:
├─ Model quantization: -50% latency, -75% storage
├─ Feature store: -30% recomputation
├─ Smaller models: -80% cost (GPT-3.5 vs GPT-4)

Advanced:
├─ Local models: -95% (after infra cost)
├─ Prompt caching: -90% (context reuse)

REALISTIC GOAL:
- Start: $50K
- After optimization: $10-15K (70-80% savings)
- Target: <$0.0001 per prediction
"
```

---

## Decision Frameworks

### Decision: Batch vs. Online Serving?

```
START: What's the latency SLA?

< 100ms:
└─ ONLINE REQUIRED
   ├─ Inference per request
   ├─ GPU/fast compute
   └─ Cost: High ($50-100K/month)

100ms - 1s:
├─ PREFERABLY ONLINE
│  └─ User still waiting, need <1s
├─ BUT: Could use cached batch results
└─ Cost: Medium-high ($20-50K/month)

1s - 10s:
├─ EITHER works
│  ├─ Online + cache
│  └─ Pre-computed batch
└─ Cost: Medium ($10-20K/month)

> 10s:
└─ BATCH preferred
   ├─ Compute overnight
   ├─ Lookup during day
   └─ Cost: Low ($5-10K/month)

EXAMPLE DECISIONS:
- Ads (bidding): <100ms → Online
- E-commerce (checkout): <500ms → Online
- Email personalization: 1 hour OK → Batch
- Weekly digest: 24 hour OK → Batch
```

---

### Decision: Fine-tune vs. Prompt Engineer?

```
START: Can you achieve goal with prompts alone?

YES:
└─ PROMPT ENGINEER
   ├─ Create system prompt
   ├─ Add examples (few-shot)
   ├─ Iterate on phrasing
   ├─ Cost: Free (API calls only)
   ├─ Speed: Hours to days
   └─ Use for: Simple tasks, rapidly changing needs

NO (accuracy ceiling hit):
├─ Can you afford fine-tuning cost?
│  ├─ YES → FINE-TUNE
│  │       ├─ Collect domain-specific data
│  │       ├─ Use LoRA (parameter-efficient)
│  │       ├─ Cost: $1-10K compute
│  │       ├─ Speed: 1-2 weeks
│  │       └─ Use for: Critical applications, specific domain
│  │
│  └─ NO → USE BETTER PROMPT + LARGER MODEL
│         ├─ Try GPT-4 (more capable)
│         ├─ Add retrieval (RAG)
│         ├─ Cost: Higher API calls
│         └─ Use for: Cost-sensitive, good enough accuracy

EXAMPLE DECISIONS:
- Customer support FAQ: Prompt engineering (can't afford delay)
- Email classification: Prompt + few-shot (simple task)
- Domain-specific Q&A: RAG + fine-tune (needs precision)
- Medical diagnosis: Fine-tune on domain data (critical accuracy)
```

---

### Decision: Which Cloud Platform?

```
START: What's your primary constraint?

COST most important:
└─ GCP Vertex AI
   ├─ 30% cheaper overall
   ├─ Excellent analytics (BigQuery)
   ├─ Production feature store
   └─ Recommended for: Startups, cost-conscious

COMPLIANCE must-have:
└─ Azure ML
   ├─ Native HIPAA, FedRAMP
   ├─ Enterprise integration
   └─ Recommended for: Healthcare, finance, regulated

ALREADY invested in platform:
├─ Using Azure? → Azure ML
├─ Using GCP? → Vertex AI
├─ Using AWS? → SageMaker
└─ Using None? → See above

LLM is core feature:
├─ Want GPT-4? → Azure OpenAI
├─ Want Gemini? → Vertex AI
├─ Don't care? → Either works

ANALYTICS heavy:
└─ GCP Vertex AI
   ├─ BigQuery is faster (2-3x) than Synapse
   ├─ SQL iteration is quicker
   └─ Recommended for: Data science teams

ENTERPRISE integration:
└─ Azure ML
   ├─ Office 365, Power BI, Dynamics
   ├─ Active Directory integration
   └─ Recommended for: Large organizations

FINAL SCORECARD:
- ML-first, cost-conscious: GCP ✓
- Enterprise-first, compliance: Azure ✓
- Balanced: GCP (save 30%)
```

---

### Decision: Retraining Frequency?

```
START: How fast does your data change?

Daily (high-velocity data):
├─ Stock prices, currency rates, ads
├─ User behavior (new trends daily)
└─ Retrain: Daily or even hourly
   ├─ Cost: $100-500/day (nightly batches)
   └─ Benefit: Catch trends immediately

Weekly (moderate-velocity):
├─ E-commerce (user preferences gradually change)
├─ Social media (content trends shift weekly)
└─ Retrain: Weekly on Sunday night
   ├─ Cost: $20-100/day (weekly batches)
   └─ Benefit: Stay relevant, avoid staleness

Monthly (slow-velocity):
├─ Marketing campaigns (content stable)
├─ User demographics (age, location, interests)
└─ Retrain: Monthly, on demand
   ├─ Cost: $5-20/month
   └─ Benefit: Easy to manage

As-needed (domain-dependent):
├─ Trigger when metrics degrade
├─ Trigger when drift detected
├─ Trigger after major business change
└─ Cost: Highly variable
   ├─ Good for: Cost-conscious, stable domains
   └─ Risk: Might miss gradual degradation

EXAMPLE DECISIONS:
- Real-time bidding: Hourly retrain
- E-commerce: Daily retrain
- Recommendation: Weekly retrain
- Churn prediction: Monthly retrain
- Fraud detection: Hourly retrain (attackers evolve daily)
```

---

## Key Numbers to Remember

**Memorize these for quick calculations:**

```
SCALE METRICS:
├─ 1M requests/day = 12 req/sec average = 60 req/sec peak
├─ 100M requests/day = 1.2K req/sec average = 6K req/sec peak
├─ 1B requests/day = 12K req/sec average = 60K req/sec peak

LATENCY TARGETS:
├─ <100ms: User feels instant (interactive)
├─ 100-500ms: User notices slight lag
├─ >1s: User becomes frustrated
├─ <1ms: Cache lookup (in-memory)
├─ <50ms: GPU inference (fast model)
├─ <200ms: Database query

COST BENCHMARKS:
├─ GPU: $0.35/hour (GCP) to $1.50/hour (on-demand)
├─ Spot instances: 70% cheaper (interruption risk)
├─ BigQuery: $7.25/TB scanned (after 1TB free)
├─ Redis cache: $2K-10K/month (100GB-1TB)
├─ Model serving: $0.05-0.15 per replica-hour

EFFICIENCY METRICS:
├─ Model quantization: 4x smaller, 2-3x faster
├─ Model pruning: 10x fewer parameters
├─ Prompt caching: 90% savings on reused context
├─ Batch processing: 50% cost savings vs. real-time
├─ Spot instances: 70% cost savings (for training)

STATISTICAL THRESHOLDS:
├─ p-value < 0.01: Highly significant (alert!)
├─ p-value < 0.05: Statistically significant (investigate)
├─ p-value > 0.05: Not significant (normal variation)
├─ >5% accuracy drop: Alert (investigate)
├─ <2% accuracy drop: Monitor (maybe seasonal)

RELIABILITY TARGETS:
├─ 99% availability: 7 hours/year downtime
├─ 99.9% availability: 43 minutes/year downtime
├─ 99.99% availability: 4 minutes/year downtime
├─ p99 latency < 200ms: Most users happy
├─ p95 latency < 100ms: Users feel snappy

TIME TARGETS:
├─ Incident detection: <5 minutes (automated alerts)
├─ Incident response: <15 minutes (rollback)
├─ Model retraining: 2-24 hours (depending on size)
├─ Canary deployment: 4 hours (minimum observation)
├─ Full rollout: 24 hours (gradual)

TEAM METRICS:
├─ Cost per engineer: $200K/year
├─ ML engineer ratio: 1 MLOps per 5 ML engineers
├─ Oncall rotation: 1 week per month per engineer
├─ Post-incident review: 24 hours after incident
```

---

**You're ready! 🚀**

- Concepts: ✓ (1.5 hours)
- Questions: ✓ (1.5 hours)
- Decision frameworks: ✓ (quick reference)
- Key numbers: ✓ (memorized)

**Final tips:**
- Start with architecture, not details
- Ask clarifying questions (scale, latency, budget)
- State trade-offs explicitly
- Give numbers (don't vague)
- Be honest about unknowns
- Use drawing/diagrams if explaining complex systems

**Go crush this interview! 💪**