# MLOps & LLMOps: Staff/Architect Level Guide

**Target Audience:** Staff Engineers, Technical Architects, and ML beginners  
**Structure:** Concepts first (beginner-friendly), then Tooling details for each section  
**Focus Areas:** Production ML Systems, LLMOps, Scalability, Reliability, Cost Optimization

---

## Table of Contents

### Part A: MLOps Fundamentals
1. [ML System Design Fundamentals](#1-ml-system-design-fundamentals)
2. [Data Management at Scale](#2-data-management-at-scale)
3. [Model Development & Experimentation](#3-model-development--experimentation)
4. [Model Deployment & Serving](#4-model-deployment--serving)
5. [Monitoring, Observability & Alerting](#5-monitoring-observability--alerting)
6. [ML Infrastructure & Operations](#6-ml-infrastructure--operations)

### Part B: LLMOps & Specialized Topics
7. [LLMOps Specific Patterns](#7-llmops-specific-patterns)
8. [Governance & Compliance](#8-governance--compliance)
9. [Cost Management & Economics](#9-cost-management--economics)
10. [Production Incident Management](#10-production-incident-management)

### Part C: Organization & Tools
11. [Team & Process](#11-team--process)
12. [Tools Ecosystem](#12-tools-ecosystem)

### Part D: Cloud Platforms
13. [GCP MLOps & Vertex AI](#13-gcp-mlops--vertex-ai)
14. [Azure MLOps & Azure ML](#14-azure-mlops--azure-ml)
15. [GCP vs. Azure: Decision Framework](#15-gcp-vs-azure-decision-framework)

### Part E: Interview & Summary
16. [Interview Q&A Examples](#16-interview-qa-examples)
17. [Key Takeaways](#17-key-takeaways)

---

# Part A: MLOps Fundamentals

## 1. ML System Design Fundamentals

### Concepts & Explanation

**What is MLOps?**  
MLOps (Machine Learning Operations) is the practice of putting machine learning models into production and maintaining them reliably. Think of it like DevOps but for ML: instead of just deploying code, you're deploying code + trained models + data pipelines. The challenge is that models aren't deterministic—the same input can behave differently depending on the data it was trained on, so you need special processes to manage this.

**Why does MLOps matter?**  
A machine learning model in a notebook is just a prototype. To deliver business value, it needs to:
- Handle millions of requests reliably
- Automatically retrain when performance degrades
- Work with data that changes over time
- Be explainable and compliant with regulations
- Cost-effectively scale without breaking the budget

**Key Difference: MLOps vs DevOps**

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| **Versioning** | Code only | Code + Data + Models |
| **Testing** | Unit, integration tests | Model quality, data validity |
| **Deployment** | Deterministic (same code = same behavior) | Non-deterministic (models vary with data) |
| **Monitoring** | Server uptime, error rates | Business KPIs + model accuracy |
| **Reproducibility** | Straightforward (git hash) | Complex (need data version + random seed) |

### Tooling & Implementation

**ML Lifecycle Stages:**
1. **Data Engineering**: Collection, validation, versioning, feature engineering
   - Tools: Apache Spark, dbt, Airflow, Pandas
   
2. **Model Development**: Experimentation, training, evaluation, hyperparameter tuning
   - Tools: PyTorch, TensorFlow, Scikit-learn, Jupyter notebooks
   
3. **Model Deployment**: Containerization, versioning, serving infrastructure
   - Tools: Docker, Kubernetes, FastAPI, TensorFlow Serving
   
4. **Monitoring & Observability**: Drift detection, performance metrics, feedback loops
   - Tools: Prometheus, Grafana, DataDog, Weights & Biases
   
5. **Retraining & Iteration**: Continuous improvement, A/B testing, rollback
   - Tools: Airflow, Kubeflow, GitHub Actions, MLflow

---

## 2. Data Management at Scale

### Concepts & Explanation

**What is a Data Pipeline?**  
A data pipeline is an automated workflow that moves data from raw sources (databases, APIs, logs) through multiple transformation steps until it's ready for ML models. Think of it like an assembly line for data: each step cleans, validates, or transforms the data slightly until it's in the exact format the model needs.

**Why Data Management is Critical:**  
- **Quality**: Bad data = bad predictions (garbage in, garbage out)
- **Scalability**: You need pipelines that handle millions of rows daily
- **Reproducibility**: When a model is retrained, you need the exact same data
- **Governance**: You need to know where sensitive data lives and who can access it

**Data Pipeline Architecture Example:**
```
Raw Data Sources (APIs, Databases) 
  ↓
Data Ingestion (batch/streaming) 
  ↓
Data Lake/Warehouse (structured storage, versioned)
  ↓
Feature Computation (create ML-ready features)
  ↓
Feature Store (reusable feature repository)
  ↓
Model Training & Serving
  ↓
Monitoring & Feedback Loop
```

### Tooling & Implementation

#### 2.1 Data Versioning & Lineage

**Concept**: Track every version of your data, so you can always reproduce past model training.

**Tools:**
- **Delta Lake**: Versioned data format (Databricks), supports time travel (`SELECT * FROM table VERSION AS OF 'v1.5'`)
- **Apache Iceberg**: Netflix's versioned table format, efficient schema evolution
- **Hugging Face Datasets**: Version control for ML datasets, easy sharing
- **DVC (Data Version Control)**: Git-like system for large data files

**Beginner Example - Delta Lake:**
```python
# Write data with version history
data = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])
data.write.format("delta").mode("overwrite").save("/data/users")

# Travel back in time
spark.read.format("delta").option("versionAsOf", 1).load("/data/users").show()
```

#### 2.2 Feature Management

**Concept**: Instead of computing the same features repeatedly, store them in a central "Feature Store" that both training and serving can use.

**Example:** For a recommendation system, "user_purchase_history" and "average_rating" are features. Instead of computing them separately for training and serving, compute once, store, reuse.

**Tools:**
- **Feast**: Open-source feature store (Uber), supports online + offline
- **Tecton**: Enterprise feature store (great for ML teams)
- **Feature Pro**: Feature management with versioning
- **Vertex Feature Store**: Managed by Google Cloud

**Beginner Example - Feast:**
```python
from feast import FeatureStore

# Define a feature
feature_store = FeatureStore(repo_path=".")

# Get features for training
training_data = feature_store.get_historical_features(
    features=["user_features:age", "user_features:country"],
    entity_df=pd.read_csv("user_ids.csv")
)

# Get features for serving (real-time, <100ms)
online_features = feature_store.get_online_features(
    features=["user_features:age"],
    entity_rows=[{"user_id": 123}]
)
```

#### 2.3 Data Quality & Validation

**Concept**: Automate checks that your data hasn't corrupted or drifted. Catch issues before they break your model.

**Key Checks:**
- Schema validation: columns exist, right data types
- Distribution checks: age is 18-100, not 5000
- Null checks: not too many missing values
- Drift detection: input distribution changed vs. training data

**Tools:**
- **Great Expectations**: Framework for data quality assertions
- **Deequ**: AWS's data validation tool
- **TensorFlow Data Validation**: Schema + anomaly detection

**Beginner Example - Great Expectations:**
```python
import great_expectations as ge

# Load data and validate
df = ge.read_csv("users.csv")
df.expect_column_values_to_be_in_set("country", ["US", "UK", "CA"])
df.expect_column_values_to_be_between("age", 0, 150)
validation_results = df.validate()
```

---

## 3. Model Development & Experimentation

### Concepts & Explanation

**What is Experiment Tracking?**  
When developing ML models, you try hundreds of variations:
- Different algorithms
- Different hyperparameters (learning rate, batch size)
- Different data versions
- Different preprocessing techniques

Without tracking, you lose track of "what made model v5 perform 2% better than v4?" Experiment tracking records every run so you can compare and learn.

**Why Reproducibility Matters:**  
If your model performs well in development but poorly in production, you need to know:
- What exact data version was it trained on?
- What was the random seed?
- What Python package versions?
- What hardware?

Without this, debugging is impossible.

### Tooling & Implementation

#### 3.1 Experiment Tracking

**Tools:**
- **MLflow**: Open-source (Databricks), logs parameters, metrics, models
- **Weights & Biases**: Popular, great for hyperparameter sweeps
- **Neptune**: Lightweight experiment tracking
- **Comet**: Enterprise experiment management

**Beginner Example - MLflow:**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.start_run()

# Log parameters
mlflow.log_param("n_estimators", 100)
mlflow.log_param("max_depth", 10)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Log metrics
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

# Log model
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()

# Compare experiments in MLflow UI (http://localhost:5000)
```

#### 3.2 Model Versioning & Registry

**Concept**: Like Git for models. Keep previous versions so you can rollback if a new one performs worse.

**Tools:**
- **MLflow Model Registry**: Track model stage (Development → Staging → Production)
- **Hugging Face Model Hub**: For NLP/LLM models, easy sharing
- **Model Catalog**: Cloud provider registries (GCP Vertex, Azure ML)

**Beginner Example - MLflow Registry:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model
model_uri = "runs:/12345/model"
mv = client.create_model_version("recommendation-model", model_uri, "v1")

# Transition to production
client.transition_model_version_stage(
    name="recommendation-model",
    version=1,
    stage="Production"
)

# Retrieve production model
model = mlflow.pyfunc.load_model(
    "models:/recommendation-model/Production"
)
predictions = model.predict(X_test)
```

#### 3.3 Evaluation Metrics & Testing

**Concept**: How do you know if your model is good? You need metrics that match what you care about.

**Common Metrics:**
- Classification: Accuracy, Precision, Recall, F1, AUC
- Regression: MSE, RMSE, MAE, R²
- Ranking: NDCG, MAP, MRR
- Generation (LLM): BLEU, ROUGE, Perplexity

**Beginner Example - Evaluation:**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1: {f1_score(y_test, y_pred)}")
```

#### 3.4 Bias & Fairness

**Concept**: Machine learning models can perpetuate or amplify biases. A model trained on historical data where certain groups were denied loans might continue denying them unfairly.

**Common Fairness Metrics:**
- **Demographic Parity**: Model approval rate same across groups
- **Equalized Odds**: False positive/negative rates same across groups
- **Calibration**: Predictions equally reliable across groups

**Tools:**
- **Fairness Indicators**: TensorFlow's fairness library
- **AI Fairness 360**: IBM's bias detection toolkit
- **Responsible AI Dashboard**: Azure ML's fairness tools

---

## 4. Model Deployment & Serving

### Concepts & Explanation

**What is Model Serving?**  
Once you've trained a model, you need to make predictions on new data. There are two main approaches:

1. **Batch Serving**: Process large amounts of data offline, write results to database
   - Use case: Daily email recommendations (precompute for all 10M users)
   - Latency: Minutes to hours OK
   - Cost: Lower (amortize compute across many predictions)

2. **Online Serving**: Make predictions in real-time, within milliseconds
   - Use case: Show recommendation when user opens app
   - Latency: <100ms required
   - Cost: Higher (need to keep servers running 24/7)

**Why Architecture Matters:**  
A model that works in notebooks might:
- Be too slow for real-time (takes 5 seconds per prediction)
- Be too large to fit in memory
- Not handle failures gracefully
- Cost too much to run at scale

Serving architecture solves these problems.

### Tooling & Implementation

#### 4.1 Serving Architectures

**Batch Serving Architecture:**
```
Data Source (BigQuery, S3)
  ↓
Preprocessing (feature engineering)
  ↓
Batch Inference (GPU cluster processes 1M rows in 10 min)
  ↓
Results Storage (database, cache)
  ↓
Application reads results (low latency)
```

**Online Serving Architecture:**
```
User Request (API call)
  ↓
Load Balancer (route to healthy server)
  ↓
Preprocessing (feature lookup, <10ms)
  ↓
Model Inference (forward pass, <50ms)
  ↓
Post-processing (rank, filter, format)
  ↓
Response (JSON, <100ms total)
```

#### 4.2 Model Servers & Containerization

**Concept**: You can't serve a model with just Python code. You need a production server that handles:
- Concurrent requests
- Automatic retries
- Load balancing
- Graceful shutdown

**Tools:**
- **TensorFlow Serving**: For TensorFlow/Keras models
- **TorchServe**: For PyTorch models
- **vLLM**: Optimized for LLM serving
- **FastAPI + Gunicorn**: Custom Python serving (most flexible)
- **Flask**: Simple serving (development, not production-grade)

**Beginner Example - FastAPI:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    age: int
    income: float

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([[request.age, request.income]])[0]
    return {"prediction": float(prediction)}

# Run: uvicorn main:app --host 0.0.0.0 --port 8000
```

**Containerization with Docker:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.3 Deployment Strategies

**Blue-Green Deployment:**
- Keep two identical production environments (blue & green)
- Deploy new model to green, test it
- Switch traffic from blue to green instantly
- Pro: Zero downtime, easy rollback
- Con: Need 2x resources

**Canary Deployment:**
- Deploy new model to small % of traffic (5%)
- Monitor metrics (latency, error rate, accuracy)
- Gradually increase (10%, 50%, 100%) if metrics good
- Pro: Risk-limited, natural A/B test
- Con: Slow rollout

#### 4.4 Model Optimization

**Concept**: A model might be accurate but too slow or too large. Optimization techniques make it production-ready.

**Techniques:**
- **Quantization**: Use 8-bit integers instead of 32-bit floats (4x smaller, ~same accuracy)
- **Pruning**: Remove unimportant weights, sparse models
- **Knowledge Distillation**: Train small model to mimic large one
- **Batching**: Group requests together for GPU efficiency

**Beginner Example - Quantization (PyTorch):**
```python
import torch
import torch.quantization

# Prepare model for quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate on representative data
for batch in calibration_loader:
    model(batch)

# Convert to quantized
torch.quantization.convert(model, inplace=True)

# Result: 4x smaller, inference 2-3x faster
print(f"Model size: {os.path.getsize('model.pth') / 1e6:.1f} MB")
```

---

## 5. Monitoring, Observability & Alerting

### Concepts & Explanation

**What is Monitoring?**  
Once your model is live, you need continuous checks that it's still working well. Monitoring answers:
- "Is my model still accurate?"
- "Are users happy with predictions?"
- "Did something break?"
- "Are we spending too much?"

**Why Monitoring is Non-negotiable:**  
Models degrade silently. A model that worked great during development can:
- Fail on edge cases in production
- Drift as user behavior changes
- Consume resources unexpectedly
- Make unfair predictions for new demographic groups

Without monitoring, you won't know for weeks (until customers complain).

**Three Types of Metrics:**
1. **System Metrics**: Server health (CPU, memory, latency)
2. **Model Metrics**: Prediction quality (accuracy, calibration)
3. **Business Metrics**: Business impact (conversion, revenue)

### Tooling & Implementation

#### 5.1 Metrics to Track

**Key Metrics by Category:**

**Model Performance:**
```
Accuracy, Precision, Recall, F1, AUC (classification)
MSE, RMSE, MAE (regression)
Calibration error (are predicted probabilities correct?)
Prediction latency (p50, p95, p99)
```

**Data Quality:**
```
Feature distributions (input drift)
Missing values %
Out-of-range values
New values not seen in training
```

**System Health:**
```
CPU/GPU utilization
Memory usage
Request latency (p50, p95, p99)
Error rate, timeout rate
Throughput (requests/second)
```

**Business Metrics:**
```
Conversion rate
Revenue per user
Customer satisfaction
Cost per prediction
```

#### 5.2 Drift Detection

**Concept**: Data Drift = input distribution changed. Your model was trained on 2023 data, but now it's 2026 and user behavior has changed. You need to detect this.

**Statistical Tests:**
- **Kolmogorov-Smirnov (KS) Test**: Compare two distributions, returns p-value
- **Wasserstein Distance**: How different are two distributions?
- **Chi-square Test**: For categorical features

**Beginner Example - KS Test:**
```python
from scipy.stats import ks_2samp
import numpy as np

# Training data distribution
training_data = np.random.normal(100, 15, 10000)  # mean=100, std=15

# Current data (drifted)
current_data = np.random.normal(105, 15, 10000)  # mean shifted to 105

# KS test
statistic, p_value = ks_2samp(training_data, current_data)

if p_value < 0.05:
    print("ALERT: Data drift detected!")  # p-value < 0.05 = significant difference
else:
    print("Data looks normal")
```

#### 5.3 Observability & Logging

**Concept**: When something breaks, you need detailed logs to diagnose the problem.

**Structured Logging (vs. printf debugging):**
```python
# Bad: just prints, hard to search/filter
print(f"Prediction made: {prediction}")

# Good: structured JSON, easy to query in logs
import logging
import json

logger.info(json.dumps({
    "timestamp": "2026-06-02T10:30:45Z",
    "model_version": "v2.1.3",
    "request_id": "abc-123",
    "prediction": 0.95,
    "latency_ms": 45,
    "user_id": 12345
}))
```

**Tools:**
- **Prometheus + Grafana**: Metrics collection + visualization
- **ELK Stack**: Elasticsearch, Logstash, Kibana for log analysis
- **Datadog**: SaaS monitoring (easy, pricey)
- **CloudWatch**: AWS's monitoring service
- **Azure Monitor**: Microsoft's monitoring service

#### 5.4 Alerting Strategy

**Concept**: Define rules that trigger alerts when something breaks. Too many alerts = alert fatigue (people ignore them). Too few = miss problems.

**Beginner Example - Alert Rules:**
```
IF p99_latency > 500ms FOR 5 minutes THEN page oncall
IF error_rate > 5% FOR 2 minutes THEN page oncall
IF data_drift_p_value < 0.01 THEN email data_team (not emergency)
IF accuracy < baseline - 5% THEN email ml_team (daily digest)
```

**Tools:**
- **Prometheus Alertmanager**: Flexible alert routing
- **PagerDuty**: Incident management, on-call scheduling
- **Opsgenie**: Similar to PagerDuty

---

## 6. ML Infrastructure & Operations

### Concepts & Explanation

**What is ML Infrastructure?**  
Infrastructure is the foundation that enables:
- Training models on GPUs (expensive hardware)
- Running data pipelines daily
- Serving models at scale
- Monitoring 24/7

Without proper infrastructure, you'd be:
- Running training on your laptop (takes weeks)
- Manually running scripts (easy to forget)
- Serving from a single server (no failover)

**Key Components:**
1. **Training Infrastructure**: GPUs, distributed training, job scheduling
2. **Data Pipelines**: Workflow orchestration, scheduling
3. **Serving Infrastructure**: Load balancers, caching, auto-scaling
4. **Monitoring**: Continuous checks, alerting

### Tooling & Implementation

#### 6.1 Training Infrastructure

**GPU Clusters:**
- Single GPU training: good for small models
- Multi-GPU training (data parallelism): split batch across GPUs
- Distributed training (across multiple servers): for huge models

**Job Scheduling Tools:**
- **Kubernetes**: Container orchestration, complex setup
- **Ray Cluster**: Distributed computing, easier than Kubernetes
- **Spark**: Big data processing, integrates with most tools
- **Slurm**: HPC (high-performance computing) job scheduler

**Beginner Example - Ray for Distributed Training:**
```python
import ray
from ray import tune

ray.init()  # Start Ray cluster

# Define training function
def train_model(config):
    learning_rate = config["learning_rate"]
    # ... training code ...
    tune.report(accuracy=0.92)

# Hyperparameter sweep (runs in parallel)
analysis = tune.run(
    train_model,
    config={"learning_rate": tune.grid_search([0.001, 0.01, 0.1])},
    num_samples=1,
    resources_per_trial={"gpu": 1}
)
```

#### 6.2 Data Pipeline Orchestration

**Concept**: You need to automate "every day at 2am, run feature computation." This is workflow orchestration.

**Tools:**
- **Apache Airflow**: Workflow DAGs, most popular
- **Prefect**: Modern alternative to Airflow
- **Dagster**: Asset-oriented orchestration (emerging)
- **dbt**: Transform data in warehouse (SQL-focused)

**Beginner Example - Airflow DAG:**
```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml_team',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('daily_training', default_args=default_args, 
         schedule_interval='0 2 * * *') as dag:  # 2am daily
    
    fetch_data = BashOperator(
        task_id='fetch_data',
        bash_command='python scripts/fetch_data.py'
    )
    
    train_model = BashOperator(
        task_id='train_model',
        bash_command='python scripts/train.py'
    )
    
    deploy = BashOperator(
        task_id='deploy',
        bash_command='python scripts/deploy.py'
    )
    
    fetch_data >> train_model >> deploy  # Dependency: A → B → C
```

#### 6.3 Real-Time Serving Infrastructure

**Components:**
- **Load Balancer**: Route requests across servers
- **Cache Layer (Redis)**: Store computed features for fast lookup
- **Message Queue (Kafka)**: Handle traffic spikes
- **Request/Response Protocols**: gRPC (fast) vs REST (easy)

**Beginner Example - Redis Caching:**
```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def get_user_features(user_id):
    # Check cache first
    cached = redis_client.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss, compute features
    features = compute_features(user_id)
    
    # Store in cache, expire after 1 hour
    redis_client.setex(
        f"user:{user_id}",
        3600,
        json.dumps(features)
    )
    
    return features
```

#### 6.4 CI/CD for ML

**Concept**: Just like software, ML code should go through testing before production.

**ML-Specific CI/CD Steps:**
1. **Data Validation**: Check schema, distributions match training
2. **Model Testing**: Unit tests for preprocessing, inference
3. **Model Quality Check**: Compare new model vs. baseline
4. **Regression Testing**: Ensure existing features still work
5. **Deployment**: Deploy to staging, then production

**Beginner Example - GitHub Actions:**
```yaml
name: ML Pipeline

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Validate data
        run: python scripts/validate_data.py
      
      - name: Run model tests
        run: pytest tests/test_model.py
      
      - name: Train model
        run: python scripts/train.py
      
      - name: Compare vs baseline
        run: python scripts/evaluate.py
      
      - name: Deploy if metrics improve
        run: python scripts/deploy.py
```

---

# Part B: LLMOps & Specialized Topics

## 7. LLMOps Specific Patterns

### Concepts & Explanation

**What is LLMOps?**  
LLMs (Large Language Models like GPT-4, Claude) behave differently than traditional ML models. Instead of predicting numbers, they generate text. This requires different operational practices.

**Key Differences:**
- **No training data needed**: Use prompt engineering instead of fine-tuning
- **Output non-deterministic**: Same prompt gives slightly different responses
- **High latency acceptable**: 1-2 seconds is OK (vs. 100ms for recommendations)
- **Cost per token**: Billing based on input/output tokens, not compute time
- **Safety critical**: Model can generate harmful content, needs guardrails

### Tooling & Implementation

#### 7.1 Prompt Engineering & Management

**Concept**: The "code" for LLMs is the prompt. You iterate on prompts like you iterate on code.

**Prompt Components:**
```
System Prompt: "You are a helpful customer service agent"
Context: Relevant documents/history
Few-shot Examples: Example Q&A to guide behavior
User Question: What the user actually asks
```

**Beginner Example:**
```python
from anthropic import Anthropic

client = Anthropic()

# Version 1: Simple prompt
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "What is machine learning?"
        }
    ]
)
print(response.content[0].text)

# Version 2: Better prompt with context
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are an expert ML engineer explaining concepts to beginners.",
    messages=[
        {
            "role": "user",
            "content": "What is machine learning? Keep it to 3 sentences."
        }
    ]
)
print(response.content[0].text)
```

**Prompt Version Control:**
```python
# Store prompts in version-controlled files/database
SYSTEM_PROMPTS = {
    "v1": "You are a customer service agent.",
    "v2": "You are a helpful customer service agent. Always be polite and try to resolve issues.",
    "v3": "You are a professional customer service agent..."
}

def get_response(user_input, prompt_version="v3"):
    system_prompt = SYSTEM_PROMPTS[prompt_version]
    # ... call LLM ...
```

**Tools:**
- **Prompt Flow**: Azure tool for building LLM chains
- **LangChain**: Framework for building LLM apps
- **Llama Index**: RAG framework (retrieve documents + LLM)
- **Weights & Biases Prompts**: Prompt versioning & A/B testing

#### 7.2 Fine-Tuning & Adaptation

**Concept**: If prompt engineering isn't enough, fine-tune a model on your domain-specific data.

**Options:**
- **Full Fine-tuning**: Update all model weights (slow, expensive)
- **LoRA**: Update only a small adapter (fast, cheap, better)
- **QLoRA**: 4-bit quantized LoRA (even cheaper)

**Beginner Example - LoRA Fine-tuning:**
```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Configure LoRA (only tune 0.3% of parameters!)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Fine-tune on your data
# ... training code ...

# Result: Only 100MB adapter, can combine with base model
model.save_pretrained("adapter")
```

#### 7.3 RAG (Retrieval-Augmented Generation)

**Concept**: Instead of fine-tuning, retrieve relevant documents and include them in the prompt. This gives LLMs access to current information.

**Architecture:**
```
User Question
  ↓
Vector Search (embed question, find similar docs)
  ↓
Retrieve Top-K Documents
  ↓
Prompt Template (include docs as context)
  ↓
LLM generates response using documents
```

**Beginner Example - Simple RAG:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Embed documents once
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [
    "Machine learning is a type of AI",
    "Deep learning uses neural networks",
    "Transformers power modern LLMs"
]
embeddings = model.encode(documents)

# User question
query = "What is machine learning?"
query_embedding = model.encode([query])[0]

# Find similar documents
similarities = [np.dot(query_embedding, doc) for doc in embeddings]
top_doc = documents[np.argmax(similarities)]

# Create prompt with context
prompt = f"""Answer based on this context:
{top_doc}

Question: {query}"""

# Send to LLM
# response = llm.generate(prompt)
```

**Tools:**
- **LlamaIndex**: Document indexing + retrieval
- **LangChain**: LLM + retrieval framework
- **Pinecone**: Vector database for retrieval
- **Weaviate**: Open-source vector database

#### 7.4 Output Quality & Safety

**Concept**: LLMs can hallucinate (make up facts) or generate harmful content. You need guardrails.

**Safety Checks:**
- **Output Validation**: Parsed structured output matches schema
- **Fact Verification**: Check claims against knowledge base
- **Toxicity Detection**: Flag harmful language
- **PII Detection**: Hide sensitive data in outputs

**Beginner Example - Output Validation:**
```python
import json
from anthropic import Anthropic

client = Anthropic()

# Ask LLM for structured output
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": """Extract user info as JSON:
Name: John Doe
Age: 30
Email: john@example.com

Return only JSON, no other text."""
        }
    ]
)

# Validate output
try:
    output = json.loads(response.content[0].text)
    assert "name" in output
    assert "age" in output
    print("Output valid:", output)
except (json.JSONDecodeError, AssertionError) as e:
    print("Invalid output:", e)
```

**Tools:**
- **Guardrails AI**: Output validation & safety
- **LlamaGuard**: Meta's safety classifier
- **Azure Content Safety**: Content moderation API

#### 7.5 Agent Systems

**Concept**: LLMs can decide which tools to call (search, calculator, database) to answer questions. This is more powerful than static prompts.

**Agent Loop:**
```
1. User asks question
2. LLM decides which tool to use
3. Execute tool (API call, database query)
4. Return result to LLM
5. LLM decides if answer complete or needs more tools
6. Return final answer
```

**Beginner Example - Simple Agent:**
```python
from anthropic import Anthropic

client = Anthropic()

def calculator(expression):
    return str(eval(expression))

def web_search(query):
    return f"Search results for: {query}"

tools = [
    {
        "name": "calculator",
        "description": "Evaluate mathematical expressions",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}}
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
    }
]

# Agent loop
messages = [{"role": "user", "content": "What's 5 + 3? Then search for machine learning."}]

while True:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # Check if LLM wants to use tools
    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "calculator":
                    result = calculator(block.input["expression"])
                elif block.name == "web_search":
                    result = web_search(block.input["query"])
                
                # Add tool result back to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    }]
                })
    else:
        # Final answer
        print(response.content[0].text)
        break
```

**Tools:**
- **LangChain**: Agent framework
- **AutoGen**: Multi-agent conversations
- **Anthropic SDK**: Native tool use (Claude)

#### 7.6 Cost Optimization for LLMs

**Concept**: LLM APIs are expensive (dollars per 1M tokens). You need strategies to reduce costs.

**Strategies:**
1. **Prompt Caching**: Reuse expensive prompts (system prompt, context)
2. **Token Counting**: Compress prompts before sending
3. **Model Selection**: Use cheaper models for simple tasks
4. **Batch Processing**: Process multiple requests together for 50% discount
5. **Local Models**: Run Llama/Mistral on your hardware instead of API

**Beginner Example - Prompt Caching (Claude API):**
```python
from anthropic import Anthropic

client = Anthropic()

# Expensive context that won't change
system_prompt = """You are an expert on company documents. 
Answer questions based on the provided documents."""

large_context = "Document 1: ...\n" * 1000  # 10K tokens of context

# First request: context is cached
response1 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": system_prompt
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}  # Cache this!
        }
    ],
    messages=[
        {"role": "user", "content": "What does document 1 say?"}
    ]
)

# Second request: reuses cached context (90% cheaper for context!)
response2 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": system_prompt
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": "What about document 2?"}
    ]
)
```

**Cost Comparison Example:**
```
Task: Summarize 20 customer support tickets

Option 1: GPT-4 (expensive)
- Cost: $20 per request

Option 2: GPT-3.5 (cheaper)
- Cost: $0.50 per request
- Trade-off: Slightly lower quality

Option 3: Local Llama 2 (free after initial setup)
- Cost: Hardware cost (~$500/month GPU rental)
- Trade-off: Need to manage infrastructure

Decision: Use GPT-3.5 + local Llama for filtering
- Filter with Llama: "Is this ticket important?"
- Only send important ones to GPT-4
- Result: 80% cost savings
```

---

## 8. Governance & Compliance

### Concepts & Explanation

**What is ML Governance?**  
As ML systems make important decisions (loan approvals, medical diagnoses, hiring), companies need processes to ensure:
- Models are fair and unbiased
- Predictions are explainable
- Data is handled securely
- Systems comply with regulations

**Key Regulations:**
- **GDPR** (EU): Right to explanation, data deletion, consent
- **CCPA** (California): Data access rights, opt-out
- **Fair Lending Laws** (US): No discrimination in credit decisions
- **HIPAA** (Healthcare): Patient privacy protection

### Tooling & Implementation

#### 8.1 Model Governance

**Concept**: Document your models like you'd document code.

**Model Card Template:**
```
Model Name: Credit Risk Model v2.1

Intended Use:
- Predict probability of loan default
- Use in lending decisions for amounts > $10K

Performance:
- Accuracy: 92% on test set
- False Positive Rate: 8%
- False Negative Rate: 5%

Limitations:
- Only trained on borrowers aged 25-65
- May not work well for new credit products
- Needs retraining if interest rates change >5%

Ethical Considerations:
- Model slightly biased against certain demographics
- Mitigation: Manual review for borderline decisions

Data:
- Training data: 500K loans from 2020-2022
- Feature: Age, income, credit score, employment history
```

**Tools:**
- **Model Card Generator**: Auto-generate from experiment tracking
- **Datasheets for Datasets**: Document data sources
- **Responsible AI Dashboards**: Azure ML, Google Cloud

#### 8.2 Regulatory Compliance

**Concept**: Some regulations require specific technical implementations.

**GDPR Compliance Example:**
```
Right to Explanation:
→ Must explain why model made a decision
→ Use SHAP values to show feature importance
→ "Your loan was denied because: credit score too low (weight: 0.8)"

Right to Data Deletion:
→ Must remove user data from all systems
→ Retraining models after deletion
→ Log deletions for audit

Consent:
→ Document what data is used
→ Get explicit consent before collection
```

#### 8.3 Security

**Concept**: Protect model weights and predictions from theft/misuse.

**Security Measures:**
- **Encryption**: Data encrypted in transit (HTTPS) and at rest (AES-256)
- **Access Control**: Only authorized users can access models
- **Rate Limiting**: Limit predictions per user to prevent abuse
- **Audit Logging**: Log all predictions for compliance

**Beginner Example - Rate Limiting:**
```python
from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")  # Max 100 requests per minute
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {"prediction": prediction}
```

---

## 9. Cost Management & Economics

### Concepts & Explanation

**Why Cost Matters:**  
ML systems are expensive:
- GPUs for training: $10-100/hour
- Cloud storage: $23/TB/month
- Model serving: Always-on servers
- Data labeling: $1-10 per label

A careless ML team can easily spend $100K+/month. Careful teams spend $10K.

**Cost Categories:**
1. **Compute**: GPUs, CPUs for training & serving
2. **Storage**: Data, models, logs, backups
3. **Data**: Labeling, annotation, collection
4. **Human**: ML engineers, data scientists, annotation workers

### Tooling & Implementation

#### 9.1 Cost Attribution & Tracking

**Concept**: Know which ML system costs how much, so you can optimize.

**Cost Tracking Setup:**
```
Label all resources with: model_name, team, cost_center
Then query cloud billing to see:
- Cost per model
- Cost per team
- Cost per training job
```

**Beginner Example - GCP Cost Tracking:**
```python
from google.cloud import bigquery
from google.cloud import monitoring_v3

# Query GCP billing data
client = bigquery.Client()
query = """
SELECT
  labels.value as model_name,
  SUM(cost) as total_cost
FROM `project.billing_dataset.gcp_billing_export_v1`
WHERE DATE(usage_start_time) >= "2026-06-01"
GROUP BY model_name
ORDER BY total_cost DESC
"""
results = client.query(query).result()
for row in results:
    print(f"{row.model_name}: ${row.total_cost:.2f}")
```

#### 9.2 Cost Optimization Strategies

**Quick Wins:**
1. **Use Spot/Preemptible Instances**: 70% cheaper, OK for training
2. **Compress Models**: Quantization = 4x smaller, 50% faster
3. **Batch Requests**: Process 100 requests = lower per-request cost
4. **Cache Results**: Don't recompute same predictions
5. **Smaller Models**: GPT-3.5 vs GPT-4 = 10x cheaper

**Beginner Example - Spot Instance Training:**
```python
# On GCP: use preemptible VMs (70% cheaper, can be interrupted)
# On AWS: use spot instances

from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name="training_job",
    script_path="train.py",
    machine_type="n1-standard-4",
    use_preemptible=True,  # Use cheap, interruptible VMs
)

job.run()  # 70% cheaper than on-demand!
```

#### 9.3 ROI & Business Impact

**Concept**: Does your ML system make money? You need to measure impact.

**ROI Calculation:**
```
Annual Cost: $100K (compute + team time)
Annual Benefit: $500K (revenue from better recommendations)
ROI: ($500K - $100K) / $100K = 400%

This model is worth 4x its cost.
```

**Measurement Framework:**
```
Baseline: What would happen without the model?
- e.g., random recommendations, human decisions

With Model:
- Measure business metric (conversion, revenue, efficiency)
- Account for seasonality, other changes

Attribution:
- What % of improvement is due to the model?
- A/B testing can answer this (model vs. baseline)
```

---

## 10. Production Incident Management

### Concepts & Explanation

**What Happens When Things Break?**  
In production, something will go wrong:
- Data pipeline stops fetching data
- Model accuracy drops 10%
- Server overloads, returns errors
- New user demographic causes biased predictions

You need processes to:
1. Detect the problem quickly
2. Diagnose root cause
3. Fix and rollback fast
4. Prevent recurrence

### Tooling & Implementation

#### 10.1 Common Failure Modes

**Data Pipeline Failures:**
- Upstream data unavailable
- Schema changed unexpectedly
- Quality check failed

**Model Failures:**
- Accuracy dropped (data drift, concept drift)
- Latency increased (overloaded servers)
- Predictions became biased

**Infrastructure Failures:**
- GPU server down
- Out of memory
- Network partition
- Database connection failed

#### 10.2 Incident Response Process

**Incident Severity:**
```
P1 (Critical): Model completely down, users can't get predictions
P2 (High): Model degraded, 50% error rate, latency 5x normal
P3 (Medium): Minor issues, but not affecting users
P4 (Low): Non-urgent improvements
```

**Response Process:**
```
1. DETECT: Alert fires (SLO breach)
2. PAGE: Oncall engineer is notified
3. DIAGNOSE: Check logs, metrics, try to understand root cause
4. DECIDE: Rollback? Patch? Scale? Kill-switch?
5. REMEDIATE: Execute fix
6. VALIDATE: Confirm metrics returning to normal
7. POST-MORTEM: What went wrong? How prevent next time?
```

**Beginner Example - Incident Runbook:**
```markdown
# Model Accuracy Drop Runbook

## Detection
- Alert: accuracy < 0.85 for 30 minutes
- Metrics: P (0.92 → 0.81), R (0.88 → 0.75)

## Diagnosis Checklist
- [ ] Check data drift: `python scripts/check_drift.py`
- [ ] Review recent deployments: `git log --oneline -20`
- [ ] Check prediction logs: `tail -f /logs/predictions.log`
- [ ] Check feature freshness: Are features stale?
- [ ] Check for new user cohort: `SELECT distinct user_segment FROM predictions WHERE timestamp > NOW() - 1h`

## Decision Tree
- If recent deployment: ROLLBACK to previous model
- If data drift: RETRAIN on fresh data
- If feature stale: REFRESH feature cache
- If new cohort: INVESTIGATE why different behavior

## Rollback Steps
1. `kubectl rollout undo deployment/model-serving`
2. Wait 2 min, verify accuracy returning
3. Investigate offline (don't roll forward until understood)
```

#### 10.3 Reliability & SLOs

**Concept**: Service Level Objective = what performance you promise.

**ML-Specific SLOs:**
```
Latency SLO: p99 latency < 200ms, 99.9% of requests
Availability SLO: Model returns response 99.9% of the time
Accuracy SLO: Model accuracy > 0.90 (baseline)
Fairness SLO: <5% difference in error rate across demographic groups
Cost SLO: <$0.01 cost per prediction
```

**Error Budget:**
```
SLO: 99.9% availability
= 43 minutes of downtime allowed per month
= Your "error budget"

Once you've hit error budget, no risky deployments!
```

**Beginner Example - Monitoring SLOs:**
```python
import time
from prometheus_client import Counter, Histogram

# Track latency
request_latency = Histogram('request_latency_seconds', 'Request latency')
error_count = Counter('errors_total', 'Total errors')

@app.post("/predict")
def predict(request):
    start = time.time()
    try:
        result = model.predict(request.features)
        latency = time.time() - start
        request_latency.observe(latency)
        return result
    except Exception as e:
        error_count.inc()
        raise
```

---

# Part C: Organization & Tools

## 11. Team & Process

### Concepts & Explanation

**Why Team Structure Matters:**  
ML systems fail not because of bad algorithms, but because of bad processes:
- Unclear who owns the model
- Data scientist trains model, ops doesn't know how to deploy
- Everyone makes changes, nobody knows why model broke

Good team structure prevents these problems.

### Typical Team Structure

**Roles:**
- **ML Engineers**: Build models, experiment tracking
- **MLOps Engineers**: Deploy models, infrastructure, monitoring
- **Data Engineers**: Build data pipelines, feature stores
- **ML Platform Team**: Shared tools, self-serve infrastructure

**Responsibilities:**
```
ML Engineers own:
├── Model training code
├── Experiment tracking
├── Feature definition
└── Model evaluation

MLOps Engineers own:
├── Model serving infrastructure
├── Deployment pipelines
├── Monitoring & alerting
└── Incident response

Data Engineers own:
├── Data pipelines
├── Data warehouse
├── Feature computation
└── Data quality checks
```

### Documentation & Runbooks

**What to Document:**
- How to run training job
- How to deploy model
- How to monitor
- How to rollback
- What to do if accuracy drops

**Runbook Example:**
```markdown
# How to Deploy a Model

## Prerequisites
- Model registered in model registry
- Metrics reviewed and approved
- Canary traffic percentage decided

## Steps
1. Create endpoint: `python deploy.py --model-name recommendation-v3`
2. Route 5% traffic: `kubectl patch svc model-serving -p '{"spec":{"canary": 0.05}}'`
3. Monitor for 1 hour: Check latency, error rate, accuracy
4. Increase to 50%: `kubectl patch svc model-serving -p '{"spec":{"canary": 0.50}}'`
5. Monitor for 4 hours
6. Route 100%: `kubectl patch svc model-serving -p '{"spec":{"canary": 1.0}}'`

## Rollback
If metrics degrade: `kubectl rollout undo deployment/model-serving`
```

---

## 12. Tools Ecosystem

### Concepts & Explanation

**Why Tool Selection Matters:**  
Choosing the wrong tool wastes months:
- Tool not mature enough
- Doesn't integrate with existing systems
- Too expensive
- Poor community support

Good tool selection enables teams to move fast.

### Tooling & Implementation

**Complete MLOps Stack:**

#### Training & Experiment Tracking
```
MLflow: Open-source, tracks experiments, model registry
Weights & Biases: Paid, but great UX, hyperparameter sweeps
Neptune: Lightweight alternative
```

#### Data & Features
```
Feast: Open-source feature store
Tecton: Enterprise feature store
dbt: Transform data with SQL
Apache Spark: Big data processing
```

#### Model Serving
```
FastAPI + Gunicorn: Simple, flexible
TensorFlow Serving: For TensorFlow models
TorchServe: For PyTorch models
vLLM: For LLM serving (optimized)
```

#### Monitoring & Observability
```
Prometheus + Grafana: Metrics + dashboards
ELK Stack: Log aggregation
Datadog: SaaS monitoring (expensive but complete)
WhyLabs: ML monitoring (data drift, etc.)
```

#### Orchestration
```
Apache Airflow: Workflow DAGs (most popular)
Prefect: Modern Airflow alternative
Dagster: Asset-oriented
Kubernetes: Container orchestration
```

---

# Part D: Cloud Platforms

## 13. GCP MLOps & Vertex AI

### Concepts & Explanation

**What is Vertex AI?**  
Google Cloud's unified ML platform. Instead of piecing together tools, Vertex AI provides:
- Managed training
- Centralized model registry
- Managed serving endpoints
- Feature store
- Monitoring
- All integrated

**Why Use Vertex AI vs. DIY?**
- Managed infrastructure (don't run servers yourself)
- Built-in monitoring & drift detection
- Integrated with BigQuery (data warehouse)
- Auto-scaling out of the box

### Tooling & Implementation

#### 13.1 Vertex AI Training

**Concept**: Instead of running training on your laptop or a Kubernetes cluster, use Vertex AI's managed service.

**Training Types:**
- **Pre-built containers**: PyTorch, TensorFlow, scikit-learn
- **Custom training**: Your Docker container
- **AutoML**: No-code (for tabular, image, text)
- **Hyperparameter tuning**: Parallel trials

**Beginner Example - Training:**
```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name="train_recommendation_model",
    script_path="train.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-12",
    requirements=["tensorflow==2.12", "pandas"]
)

model = job.run(
    args=["--learning-rate=0.001"],
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1
)
```

#### 13.2 Feature Store Integration

**Concept**: Centralized features that training and serving both use.

**Beginner Example - Vertex Feature Store:**
```python
from google.cloud import aiplatform

# Create feature store
fs = aiplatform.FeatureStore(
    featurestore_name='my_feature_store',
    project='my-project',
    location='us-central1'
)

# Get features for training
training_data = fs.batch_serve_to_bq(
    serving_input_ids=pd.read_csv("user_ids.csv"),
    destination_uri="bq://my-project.dataset.training_table"
)

# Get features for serving (online)
online_features = fs.online_serve(
    request={"user_id": "123"}
)  # Returns in <100ms
```

#### 13.3 Model Serving

**Concept**: Deploy models to auto-scaling endpoints.

**Beginner Example - Vertex Endpoints:**
```python
from google.cloud import aiplatform

# Create endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="recommendation-endpoint"
)

# Deploy model
endpoint.deploy(
    model=model,
    traffic_split={"0": 100},  # 100% traffic to this model
    machine_type="n1-standard-2"
)

# Make predictions
predictions = endpoint.predict(
    instances=[
        {
            "age": 30,
            "income": 50000
        }
    ]
)
```

#### 13.4 Monitoring & Drift Detection

**Concept**: Automatic monitoring for data and prediction drift.

**Setup:**
```python
from google.cloud import aiplatform

# Enable monitoring on endpoint
endpoint.configure_drift_detection(
    drift_thresholds={
        "user_age": 0.05,  # Alert if distribution shifts >5%
        "income": 0.03
    },
    attribution_columns=["user_age", "income"]
)
```

#### 13.5 Cost Management

**GCP Cost Structure:**
- Training: $0.35-0.70 per GPU-hour (cheaper than AWS)
- Serving: $0.05-0.15 per replica-hour
- Storage: $0.02 per GB/month
- BigQuery: Free tier 1TB/month, then $7.25 per TB

**Cost Optimization:**
```
Use spot instances (70% cheaper) for training
Use smaller machines for light workloads
Cache predictions to avoid redundant queries
```

---

## 14. Azure MLOps & Azure ML

### Concepts & Explanation

**What is Azure ML?**  
Microsoft's managed ML platform. Similar to Vertex AI but integrated with:
- Azure Data Lake & Synapse (data warehouse)
- Azure OpenAI (GPT-4, GPT-3.5)
- Prompt Flow (LLM workflow tool)
- Application Insights (monitoring)

**When to Choose Azure:**
- You're already using Azure/Microsoft
- Need enterprise features (HIPAA, FedRAMP compliance)
- Want to use Azure OpenAI (GPT-4)
- Prefer SQL-first data approach (Synapse)

### Tooling & Implementation

#### 14.1 Azure ML Compute

**Concept**: Managed compute for training.

**Beginner Example:**
```python
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

# Connect to Azure ML
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group>",
    workspace_name="<workspace>"
)

# Define training job
job = command(
    code="./src",
    command="python train.py --learning_rate 0.001",
    environment="AzureML-sklearn-1.3",
    compute="cpu-cluster"
)

# Submit job
returned_job = ml_client.create_or_update(job)
```

#### 14.2 Model Registry & Deployment

**Concept**: Track models, deploy to endpoints.

**Beginner Example:**
```python
from azure.ai.ml.entities import Model

# Register model
model = Model(
    path="outputs/model.pkl",
    name="recommendation-model",
    version=1
)

ml_client.models.create_or_update(model)

# Create endpoint
endpoint = ManagedOnlineEndpoint(name="rec-endpoint")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy model
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="rec-endpoint",
    model="recommendation-model:1",
    instance_type="Standard_F2s_v2",
    instance_count=2
)
```

#### 14.3 Monitoring

**Concept**: Application Insights integration for metrics.

**Setup:**
```python
# Predictions automatically logged to Application Insights
# Query logs:
query = """
requests
| where name == "score_endpoint"
| summarize 
    Avg_Latency=avg(duration),
    Error_Count=count(~success)
| by bin(timestamp, 5m)
"""

logs_client.query_workspace(query)
```

#### 14.4 LLMOps with Azure OpenAI

**Concept**: Build LLM apps using managed Azure OpenAI service.

**Beginner Example:**
```python
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="...",
    api_version="2024-02-15-preview",
    azure_endpoint="https://my-resource.openai.azure.com/"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is machine learning?"}
    ]
)

print(response.choices[0].message.content)
```

#### 14.5 Prompt Flow

**Concept**: Visual tool for building LLM chains (prompts + logic + tool calls).

**Example Prompt Flow:**
```
1. Input: User question
2. Retrieve: Search documents for relevant context
3. Prompt: Send question + context to LLM
4. Output: LLM response
5. Validate: Check response matches expected schema
6. Return: Final answer
```

---

## 15. GCP vs. Azure: Decision Framework

### Concepts & Explanation

**How to Choose?**

**Choose GCP Vertex AI if:**
- You do heavy analytics (BigQuery is excellent)
- You need fast, cheap inference
- You're comfortable with Google ecosystem
- You want open-source-first tools

**Choose Azure ML if:**
- You already use Microsoft services
- You need enterprise compliance (HIPAA)
- You want Azure OpenAI integration
- You prefer SQL/BI tools

### Comparison Table

| Aspect | GCP Vertex AI | Azure ML |
|--------|---|---|
| **Data Integration** | BigQuery (excellent) | Synapse, ADLS Gen2 |
| **Feature Store** | Production-ready | In preview |
| **Serving** | gRPC endpoint, fast | REST endpoint |
| **Cost** | ~$0.35/GPU-hour | ~$0.50/GPU-hour |
| **LLM Support** | Gemini, open models | Azure OpenAI |
| **Compliance** | SOC2, GDPR | HIPAA, FedRAMP |
| **Preferred Language** | Python, SQL | Python, C#, PowerShell |

---

# Part E: Interview & Summary

## 16. Interview Q&A Examples

### Q1: Design an ML System for 100M Requests/Day

**Problem:** How would you serve 100M predictions daily with <100ms latency?

**Solution Approach:**
```
1. Batch precomputation for 80% (obvious, recurring patterns)
   - Daily: Compute recommendations for all users overnight
   - Store in cache (Redis/Memcached)
   - Latency: <10ms (cache lookup)

2. Online inference for 20% (personalized, real-time)
   - Model serving (quantized for speed)
   - Feature lookup from Feature Store (<10ms)
   - Model inference (<50ms)
   - Total: <100ms

3. Infrastructure:
   - Cache cluster: 1TB Redis (hot data)
   - Model servers: 10 GPU servers, auto-scale to 50
   - Load balancer: Distribute across servers
   - Multi-region: Replicate cache & models

4. Cost Optimization:
   - 80% from cache (cheapest)
   - Batch during off-peak (20% price discount)
   - Model quantization (2x faster, 4x smaller)
   - Estimated cost: $50K-100K/month
```

### Q2: Model Accuracy Dropped 5%

**Problem:** Your model's accuracy dropped from 0.92 to 0.87 in production. How do you diagnose?

**Investigation Checklist:**
```
1. DATA DRIFT CHECK
   ├── Compare training data distribution vs. current data
   ├── KS test on key features (age, income, location)
   └── If drifted: User behavior changed, retrain on new data

2. LABEL SHIFT CHECK
   ├── Compare class distribution (if classification)
   ├── Example: Was 60% positive in training, 40% now
   └── Adjust decision threshold or retrain

3. CONCEPT DRIFT CHECK
   ├── Ground truth changed (world changed, not just data)
   ├── Example: Loan default patterns changed due to economy
   └── May need domain expert review

4. FEEDBACK LOOP CONTAMINATION
   ├── Are training labels poisoned?
   ├── Example: Model's old predictions used as labels for retraining
   └── Check data pipeline for feedback loop

5. SERVING DISCREPANCY
   ├── Preprocessing in serving different from training?
   ├── Quantization artifacts changing predictions?
   ├── Feature computed differently?

6. RESOLUTION
   ├── If data drift: Retrain on recent data
   ├── If label shift: Adjust threshold or retrain
   ├── If serving issue: Fix preprocessing
   └── If unknown: Rollback, investigate, deploy fixed version
```

### Q3: Reduce LLM Inference Cost by 10x

**Problem:** LLM API calls cost $10K/day. How do you reduce?

**Solutions (in priority order):**
```
1. PROMPT CACHING (90% savings on context)
   - Reuse system prompt + context across queries
   - Example: Customer support with 10K company docs
   - Cost before: 10K docs * 1000 queries * $0.001 = $10K
   - Cost after: 10K docs cached + 1000 queries = $0.1K
   - Result: 100x cheaper!

2. MODEL SELECTION (10x cheaper)
   - Replace GPT-4 with GPT-3.5 for simple tasks
   - Simple task: Classify email category
   - Cost: $0.5 (3.5) vs $5 (4) = 10x savings
   - Quality trade-off: Still 95% accurate for this task

3. LOCAL MODELS (free after infra cost)
   - Run Llama 2 on your hardware
   - Setup cost: $500/month GPU rental
   - Per-query cost: $0 (already paying for compute)
   - Good for: High-volume, not requiring state-of-art quality

4. BATCH API (50% discount)
   - Process requests in batches, not real-time
   - Latency: 24 hours OK
   - Cost: 50% cheaper than real-time

5. FALLBACK LOGIC (rules before LLM)
   - Use simple rules for 70% of cases (free)
   - Use LLM only for complex cases (30%)
   - Example:
     - If "password reset" → send reset link (rule, free)
     - If complex question → ask LLM ($)
   - Result: 70% savings

COMBINED: Caching + Model Selection + Fallback = 100x+ savings!
```

### Q4: Compare Vertex AI vs Azure ML

**Scenario:** You're building a recommendation system for 50M users. Choose GCP or Azure.

**Analysis:**

**GCP Vertex AI Advantages:**
- BigQuery integration (5-minute SQL analysis)
- Feature Store production-ready
- Faster inference (gRPC)
- 30% cheaper compute

**Azure ML Advantages:**
- Enterprise compliance (HIPAA, FedRAMP)
- Synapse Analytics (SQL + Spark)
- Azure OpenAI integration
- Better for hybrid deployments

**Recommendation:**
```
If GCP-first company:
→ Use Vertex AI
→ BigQuery for analytics
→ Feature Store for serving
→ Estimated cost: $50K/month

If Azure-first company:
→ Use Azure ML
→ Synapse for analytics
→ ADLS for data lake
→ Estimated cost: $65K/month (25% premium for enterprise features)

If cost-sensitive, analytics-heavy:
→ Choose GCP Vertex AI

If compliance-required, enterprise-heavy:
→ Choose Azure ML
```

### Q5: Design Data Drift Monitoring

**Problem:** Design a system to detect when your model's input data has drifted.

**Solution:**
```
1. BASELINE COMPUTATION (during training)
   - Compute statistics on training data
   - Store: mean, std, quantiles for each feature
   - Example:
     age: mean=35, std=12, p50=34, p95=60

2. PERIODIC DRIFT CHECK (daily)
   - Compute same statistics on last 24h of data
   - Compare distributions:
     - age (new): mean=36.5, std=11
     - KS test p-value: 0.0001 (significant change!)
   
3. ALERT IF DRIFTED
   - If p-value < 0.05: Alert data team
   - Log which features drifted
   - Example: "age distribution shifted 2%" →  not critical
   
4. INVESTIGATION
   - If seasonal: Expected, no action
   - If new cohort: May need retraining
   - If data quality issue: Fix pipeline
   
5. RESOLUTION
   - Option A: Retrain on latest data
   - Option B: Adjust model for new distribution
   - Option C: Adjust SLA (accept slightly lower accuracy)

MONITORING SETUP:
- Vertex Model Monitoring (automated)
- OR manual Airflow job:
  ```python
  # Daily Airflow task
  def check_drift():
    current = get_feature_stats("last_24h")
    baseline = get_feature_stats("training_data")
    for feature, stats in current.items():
      p_value = ks_test(baseline[feature], stats)
      if p_value < 0.05:
        send_alert(f"Drift detected in {feature}")
  ```
```

---

## 17. Key Takeaways

### For Beginners

1. **MLOps is Infrastructure for ML**: Just like DevOps enables software teams, MLOps enables ML teams
2. **Data > Model**: 80% of effort is data management, not model development
3. **Monitoring is Critical**: Your model will break. Detect it before users notice
4. **Automation Saves**: Manual deployments/retraining leads to mistakes and burnout
5. **Cloud Platforms Help**: Managed services (Vertex, Azure ML) let you focus on ML, not infrastructure

### For Staff/Architects

1. **Architecture Matters**: Batch vs. online serving changes entire cost structure (10x difference)
2. **Platform Choice**: GCP for analytics, Azure for enterprise compliance (or hybrid)
3. **Cost Optimization is Strategic**: Prompt caching, model selection, local inference can reduce costs 10-100x
4. **Reproducibility is Non-negotiable**: Version everything (data, code, models, hyperparameters)
5. **Governance Scales**: Start simple, add compliance/fairness as needed
6. **Incident Response Process**: Detection → diagnosis → remediation → learning
7. **Team Structure**: Clear roles (ML eng, MLOps, data eng) prevent firefighting
8. **Tool Ecosystem**: Choose tools that integrate well, not just best-of-breed

### Decision Trees

**When to Batch vs. Online Serving?**
```
Latency requirement < 500ms? → Online serving
Latency requirement 1+ minutes? → Batch serving
Cost > business value? → Batch
User interactivity required? → Online
```

**When to Fine-tune vs. Prompt Engineer?**
```
Can you achieve goal with prompts? → Prompt engineer (faster, cheaper)
Prompts hitting accuracy ceiling? → Fine-tune (expensive)
Need custom domain knowledge? → Fine-tune
Constantly changing requirements? → Prompts
```

**Which Cloud Platform?**
```
Heavy analytics workload? → GCP Vertex AI
Enterprise compliance required? → Azure ML
Cost-sensitive? → GCP Vertex AI
Already on Azure? → Azure ML
Need LLM integration? → Azure (OpenAI) or GCP (Gemini)
```

---

**Last Updated:** June 2, 2026  
**Target Audience:** Beginners to Staff/Architects  
**Format:** Concepts first (beginner-friendly), then tooling details  
**Coverage:** MLOps fundamentals, LLMOps, GCP Vertex AI, Azure ML, interview questions, decision frameworks
