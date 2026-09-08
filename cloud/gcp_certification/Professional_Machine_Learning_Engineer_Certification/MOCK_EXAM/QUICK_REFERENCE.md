# GCP PMLE Exam - Quick Reference Cheat Sheet

## Essential GCP ML Services

### Vertex AI
```
Vertex AI Workbench
  └─ Managed Jupyter notebooks
  └─ Data exploration & prototyping

Vertex AI AutoML
  ├─ Image Classification/Detection
  ├─ Text Classification/Entity Extraction
  ├─ Tabular (Regression/Classification)
  └─ Video Classification/Action Recognition

Vertex AI Custom Training
  ├─ TensorFlow
  ├─ PyTorch
  ├─ Scikit-learn
  └─ XGBoost

Vertex AI Pipelines
  ├─ Kubeflow Pipelines SDK
  ├─ TensorFlow Extended (TFX)
  └─ Custom components

Vertex AI Feature Store
  ├─ Centralized feature management
  ├─ Online & batch serving
  └─ Feature sharing across teams

Vertex AI Model Monitoring
  ├─ Data drift detection
  ├─ Training/serving skew
  └─ Prediction drift monitoring

Vertex AI Endpoints
  ├─ Batch predictions
  ├─ Online predictions
  └─ Model versioning & routing
```

### BigQuery ML (BQML)
```
CREATE OR REPLACE MODEL project.dataset.model_name
OPTIONS(
  model_type='linear_reg',  -- linear_reg, logistic_reg, time_series_forecasting, clustering, etc.
  input_label_cols=['label']
) AS
SELECT * FROM project.dataset.table;

Model Types:
- linear_reg: Linear Regression
- logistic_reg: Binary/Multi-class Classification
- time_series_forecasting: Time series prediction
- clustering: K-means clustering
- matrix_factorization: Recommendations
- deep_neural_network: DNN Classification/Regression
- xgboost_classifier/regressor: XGBoost models
- arima_plus: Advanced time series
```

---

## Model Development Workflow

### 1. Data Preparation
```
1. Collect & Store Data
   └─ Cloud Storage / BigQuery

2. Explore Data
   └─ Vertex AI Workbench / SQL queries

3. Handle Data Quality
   ├─ Missing values (imputation/removal)
   ├─ Outliers (detection/removal)
   ├─ Class imbalance (SMOTE/weighting/oversampling)
   └─ Duplicates (detection/removal)

4. Feature Engineering
   ├─ Normalization (0-1)
   ├─ Standardization (mean=0, std=1)
   ├─ Encoding (one-hot for categorical)
   ├─ Feature scaling
   └─ Feature selection (correlation, importance)
```

### 2. Data Splitting
```
Train/Validation/Test Split
├─ Training set: 60-70% (model learning)
├─ Validation set: 15-20% (hyperparameter tuning)
└─ Test set: 15-20% (final evaluation)

Time Series:
├─ Train: Historical data
├─ Validation: Most recent past
└─ Test: Future period
```

### 3. Model Training
```
1. Choose Algorithm
   ├─ Regression: Linear, Tree-based, Neural Networks
   ├─ Classification: Logistic Regression, SVM, Tree-based
   ├─ Clustering: K-means, DBSCAN, Hierarchical
   └─ Deep Learning: CNN, RNN, Transformers

2. Hyperparameter Tuning
   ├─ Learning rate
   ├─ Batch size
   ├─ Number of epochs
   ├─ Regularization (L1/L2)
   └─ Tree depth (for tree-based)

3. Cross-Validation
   └─ k-fold (typically k=5 or k=10)
```

### 4. Model Evaluation
```
Classification Metrics:
├─ Accuracy: (TP+TN)/(TP+TN+FP+FN)
├─ Precision: TP/(TP+FP)  [Avoid false positives]
├─ Recall: TP/(TP+FN)      [Catch positives]
├─ F1-Score: 2*(Precision*Recall)/(Precision+Recall)
├─ ROC-AUC: Probability model correctly ranks examples
└─ Confusion Matrix: TP, TN, FP, FN

Regression Metrics:
├─ MAE: Mean Absolute Error
├─ RMSE: Root Mean Squared Error
├─ R²: Coefficient of determination (0-1)
└─ MAPE: Mean Absolute Percentage Error

Time Series:
├─ MAPE: Percentage error
├─ RMSE: Prediction error magnitude
└─ Directional accuracy
```

---

## Production ML Patterns

### Model Deployment
```
Option 1: Online Predictions (Vertex AI Endpoint)
├─ Use: Real-time predictions
├─ Latency: Low (milliseconds)
├─ Throughput: ~1000s predictions/second
└─ Cost: Container instance running continuously

Option 2: Batch Predictions (Vertex AI Batch)
├─ Use: Large-scale, non-urgent predictions
├─ Latency: High (hours/days)
├─ Throughput: Millions of predictions
└─ Cost: Pay per job, scales dynamically
```

### Model Monitoring & Retraining
```
Triggers for Retraining:
1. Scheduled: Fixed time intervals (risky - unnecessary waste)
2. Performance-based: Drop in validation metric
3. Data-driven: Data drift detected
4. Traffic-based: Model serving skew threshold exceeded

Best Practice:
1. Compare new evaluation to production model baseline
2. Deploy only if improved
3. Monitor for drift using Vertex AI Model Monitoring
4. Trigger retraining when thresholds exceeded
```

### Deployment Strategies
```
Canary Deployment:
├─ Route small percentage to new model
├─ Monitor metrics vs. old model
├─ Gradually increase traffic
└─ Rollback if issues detected

A/B Testing:
├─ Route 50% to model A, 50% to model B
├─ Compare metrics
├─ Winner becomes production model

Blue-Green:
├─ Blue: Current production
├─ Green: New model in staging
├─ Switch traffic to green
└─ Keep blue as rollback
```

---

## Common Algorithms & When to Use

### Classification
```
Logistic Regression
├─ When: Binary/multiclass, interpretability needed
├─ Pros: Fast, interpretable, good baseline
└─ Cons: Linear only

Decision Trees/Random Forest
├─ When: Non-linear, feature importance needed
├─ Pros: No scaling needed, handles mixed features
└─ Cons: Prone to overfitting

SVM (Support Vector Machine)
├─ When: Binary classification, high-dimensional
├─ Pros: Works well in high dimensions
└─ Cons: Slow with large datasets

Neural Networks
├─ When: Complex patterns, large datasets
├─ Pros: Can learn any function
└─ Cons: Needs lots of data, black-box

XGBoost
├─ When: Competitive ML, structured data
├─ Pros: Fast, accurate, feature importance
└─ Cons: Complex, memory intensive
```

### Regression
```
Linear Regression
├─ When: Linear relationship, simple baseline
├─ Pros: Fast, interpretable
└─ Cons: Only linear relationships

Polynomial Regression
├─ When: Curved relationships
├─ Pros: More flexible than linear
└─ Cons: Risk of overfitting

Decision Trees/Random Forest
├─ When: Non-linear, feature importance needed
└─ Same as classification

Neural Networks
├─ When: Complex patterns
└─ Same as classification

Time Series (ARIMA, Prophet)
├─ When: Time-dependent data
├─ Pros: Handles trends, seasonality
└─ Cons: Assumes temporal patterns continue
```

---

## Data Preprocessing Decision Tree

```
Missing Values?
├─ < 5%: Remove rows or forward fill (time series)
├─ 5-20%: Imputation (mean/median/mode)
├─ > 20%: Create separate category or feature engineering
└─ Consider: Why is it missing?

Categorical Variables?
├─ Ordinal (ordered): Encode as numbers (1,2,3...)
├─ Nominal (unordered): One-hot encoding
├─ High cardinality (>50 unique): Frequency encoding or target encoding
└─ Rare categories: Group into "Other"

Numerical Variables?
├─ Different scales: Normalize (0-1) or Standardize (mean=0, std=1)
├─ Outliers: Clip, remove, or robust scaling
├─ Skewed: Log transformation or Box-Cox
└─ Heavy tail: Quantile normalization

Class Imbalance?
├─ Ratio 1:10 or less: Oversampling, SMOTE, or class weights
├─ Ratio 1:100+: Stratified sampling, threshold adjustment
└─ Severe: Anomaly detection instead of classification

Feature Engineering?
├─ Continuous: Binning, interactions, polynomials
├─ Temporal: Day of week, seasonality, lags
├─ Domain knowledge: Business metrics, ratios
└─ Selection: Correlation analysis, feature importance
```

---

## GCP ML Services Comparison

| Aspect | AutoML | BigQuery ML | Custom Training | AutoML + Custom |
|--------|--------|-------------|-----------------|-----------------|
| **Expertise Required** | None | SQL | Python/TensorFlow | ML Engineer |
| **Training Time** | Hours | Minutes | Variable | Days |
| **Customization** | Limited | Some | Full | Full |
| **Cost** | High (per hour) | Low | Compute-based | Compute-based |
| **Scalability** | Limited | Unlimited | Unlimited | Unlimited |
| **Best For** | Quick baseline | Data warehouse | Production | Complex problems |

---

## Key Decision Patterns for PMLE Exam

### Choosing Data Storage
```
BigQuery?
├─ When: Large structured data, SQL queries, ML needs
├─ When: Need to share data across teams
└─ Avoid: Small datasets, unstructured

Cloud Storage?
├─ When: Raw/unstructured data, files
├─ When: Training data for custom models
└─ Avoid: Structured queries, small files

Cloud SQL/Firestore?
├─ When: Transactional apps, real-time writes
└─ Avoid: ML workloads, analytics

Vertex AI Feature Store?
├─ When: Shared features across models
├─ When: Real-time feature serving needed
└─ When: Feature version management needed
```

### Choosing Model Type
```
AutoML?
├─ When: No ML expertise, need quick MVP
├─ When: Limited data science team
└─ Cost: Monitor carefully (can be expensive)

BigQuery ML?
├─ When: Data already in BigQuery
├─ When: Simple models OK, SQL preference
└─ Cost: Included with BigQuery

Custom Training?
├─ When: Complex model needed
├─ When: Specific algorithm required
├─ When: Fine-grained control needed
└─ Cost: Pay for compute resources
```

### Choosing Training Strategy
```
Batch Training?
├─ When: Periodic model updates
├─ When: Cost sensitive
└─ When: No real-time requirements

Continuous Training?
├─ When: Data constantly changing
├─ When: Model performance critical
├─ When: Monitoring shows drift
└─ When: Budget allows

Transfer Learning?
├─ When: Similar pretrained model exists
├─ When: Limited training data
├─ When: Need fast results
└─ When: Budget/time constrained
```

---

## Vertex AI Pipelines (Common Components)

```
Data Ingestion
├─ Read from Cloud Storage
├─ Query BigQuery
└─ Stream from Pub/Sub

Data Processing
├─ Transform/Clean
├─ Feature engineering
└─ Validation checks

Model Training
├─ Custom training component
├─ AutoML training
└─ Transfer learning

Model Evaluation
├─ Metrics calculation
├─ Model comparison
└─ Threshold validation

Conditional Deployment
├─ If metrics pass: Deploy to Endpoint
├─ If failed: Alert/rerun
└─ Archive model artifacts

Monitoring & Retraining
├─ Monitor predictions
├─ Detect drift
└─ Trigger retraining on threshold
```

---

## Exam Strategy - Question Types

### Type 1: "What should you do?"
- Identify the constraint/requirement
- Eliminate obviously wrong answers
- Pick most complete/production-ready solution

### Type 2: Scenario-based
- Read entire scenario carefully
- Identify: Goal, Data, Scale, Cost, Timeline
- Think: Which GCP service fits best?

### Type 3: Best practice
- Production concerns matter (monitoring, cost, scale)
- GCP-specific solutions preferred over generic
- Think: What would Google recommend?

### Type 4: Cost optimization
- Managed services usually cheaper than infrastructure
- Batch predictions cheaper than online
- Serverless cheaper than always-on

### Type 5: Multiple select
- Read carefully: "Which of the following are true?"
- All selected answers must be correct
- No partial credit

---

## Last-Minute Facts to Memorize

1. **Vertex AI Endpoints** serve models with low latency (online predictions)
2. **Batch Predictions** for large-scale, cost-effective predictions
3. **Model Monitoring** detects data drift and training/serving skew
4. **Retraining** should be triggered by performance degradation, not schedule
5. **BigQuery ML** is best for data already in warehouse
6. **Vertex AI Pipelines** orchestrates ML workflows with Kubeflow/TFX
7. **AutoML** needs no ML expertise but is more expensive
8. **Feature Store** centralizes features for multiple models
9. **Cross-validation** prevents overfitting better than train/test split
10. **Canary deployment** is safer than full rollout for production models

---

Good luck on your GCP PMLE exam! You've got this! 🎯
