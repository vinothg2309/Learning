# GCP PMLE Mock Exam — 250 Complex Scenario Q&A

## How to Use This File
This mock exam contains 250 scenario-based questions spanning all PMLE exam domains. Most questions are medium-to-high difficulty, mirroring the actual exam. Single-choice questions have 4 options; multi-select questions ask you to select all correct answers. Review explanations after each question to reinforce concepts.

---

## Domain 1: BigQuery ML (Q1–Q20)

### Q1. BQML Linear Regression Model Selection
**Scenario:** You have customer data with 50 features (numerical and categorical). You want to predict house price. Should you use LINEAR_REG or BOOSTED_TREE?

- [x] A. Use LINEAR_REG first for baseline (simpler, interpretable); if performance is poor, try BOOSTED_TREE for non-linear patterns
- [ ] B. Always use BOOSTED_TREE; it's always better
- [ ] C. Use LINEAR_REG for all problems
- [ ] D. Choose randomly

> **Explanation:** LINEAR_REG is interpretable baseline; BOOSTED_TREE handles non-linearity better but is less transparent. Start simple, escalate if needed.

---

### Q2. BQML Feature Scaling and Normalization
**Scenario:** Your dataset has: age (0–100), income (10K–1M), credit score (300–850). You train a LINEAR_REG model. Which preprocessing step should you apply?

- [x] A. Use ML.STANDARD_SCALER to normalize features to mean=0, std=1; ensures features contribute equally to loss
- [ ] B. No preprocessing needed; BQML handles it automatically
- [ ] C. Scale only numeric features, leave categorical unchanged
- [ ] D. Use ML.ONE_HOT_ENCODER for all features

> **Explanation:** STANDARD_SCALER prevents high-range features (income) from dominating low-range features (age). ONE_HOT_ENCODER is for categorical encoding, not scaling.

---

### Q3. BQML Classification: Binary vs Multi-class
**Scenario:** You want to classify customer churn (yes/no). Your model uses LOGISTIC_REG. After training, you check: accuracy=95%, but recall=20%. Your boss wants to catch 90% of churners. What's the issue?

- [x] A. Class imbalance: if 98% don't churn, predicting "no churn" always gives 98% accuracy but 0% recall. Set class_weight=BALANCED in training to penalize false negatives more
- [ ] B. Use LINEAR_REG instead
- [ ] C. Ignore recall; accuracy is sufficient
- [ ] D. Increase dataset size indefinitely

> **Explanation:** Imbalanced data requires adjusting class weights or threshold. High accuracy ≠ high recall. For churn (catch churners), prioritize recall.

---

### Q4. BQML Model Evaluation: ML.EVALUATE
**Scenario:** You train a model and run ML.EVALUATE. Output shows: precision=0.8, recall=0.6, f1_score=0.686. A colleague says "precision is high, so the model is good." Is this correct?

- [x] A. No; high precision alone doesn't guarantee good model. If recall is low (0.6), you're missing 40% of true positives. F1 (harmonic mean) better balances precision-recall trade-off
- [ ] B. Yes; precision is the most important metric
- [ ] C. Ignore recall; focus only on precision
- [ ] D. Metrics don't matter in BQML

> **Explanation:** Precision measures correctness of positive predictions; recall measures coverage of actual positives. Both matter. F1 is a balanced metric.

---

### Q5. BQML Feature Engineering: ML.FEATURE_CROSS
**Scenario:** You have features: age_bucket (young/middle/senior), product_category (A/B/C). You suspect interaction: seniors buy category C more. How do you model this?

- [x] A. Use ML.FEATURE_CROSS to create interaction terms (e.g., 'senior_C'). Add crossed features to model; captures non-linear relationships
- [ ] B. Use separate models for each age group
- [ ] C. Ignore interactions; not worth the complexity
- [ ] D. Use LINEAR_REG; it automatically captures interactions

> **Explanation:** FEATURE_CROSS creates new features from combinations of existing ones. Captures interactions. LINEAR_REG doesn't automatically find interactions; BOOSTED_TREE does.

---

### Q6. BQML Time Series Forecasting: ARIMA
**Scenario:** You want to forecast weekly sales. Data: 104 weeks of history. You train ARIMA model. Should you include trend and seasonality?

- [x] A. Yes; if data shows trend (sales growing) or seasonality (peaks in Dec), include them. ARIMA with trend + seasonality captures these patterns. Check ACF/PACF plots
- [ ] B. Never use seasonality
- [ ] C. Use LINEAR_REG for time series
- [ ] D. Ignore data patterns; use defaults

> **Explanation:** Time series often have trend (long-term direction) and seasonality (repeating patterns). ARIMA models both. Inspect data first to decide.

---

### Q7. BQML Model Explanation: ML.EXPLAIN_PREDICT
**Scenario:** Your model predicts loan approval. A customer asks: "Why was I rejected?" You run ML.EXPLAIN_PREDICT. Output shows: feature importance for debt_ratio=0.8, income=0.1. What does this mean?

- [x] A. debt_ratio was most important in the prediction decision (80% contribution). For this customer, high debt_ratio likely triggered rejection. Income had minimal impact
- [ ] B. debt_ratio is always the cause of rejection
- [ ] C. Feature importance is meaningless
- [ ] D. All features contribute equally

> **Explanation:** EXPLAIN_PREDICT shows feature contribution to individual prediction. High importance ≠ causality, but indicates which features mattered for that prediction.

---

### Q8. BQML Hyperparameter Tuning in SQL
**Scenario:** You train a LINEAR_REG model with learning_rate=0.1. Validation loss plateaus. Should you increase or decrease learning_rate?

- [x] A. Try lower learning_rate (0.01) first; high learning rate may cause oscillation. If validation loss is still high, increase. Use grid search: test multiple rates and pick best
- [ ] B. Always use learning_rate=0.1
- [ ] C. Learning rate doesn't matter
- [ ] D. Increase learning_rate indefinitely

> **Explanation:** Learning rate too high: oscillates, misses optimum. Too low: converges slowly. Use CV with multiple rates. BQML supports hyperparameter tuning via CREATE MODEL options.

---

### Q9. BQML Regression Model Evaluation
**Scenario:** You train BQML LINEAR_REG to predict house price. ML.EVALUATE shows: mean_absolute_error=15K, root_mean_squared_error=20K. Is this good?

- [x] A. Depends on context. If typical house price is 300K, MAE=15K is 5% error (good). If typical price is 100K, 15% error (poor). Compare to baseline (always predicting mean price)
- [ ] B. 15K error is always good
- [ ] C. 20K error is always bad
- [ ] D. Regression errors are meaningless

> **Explanation:** Regression error evaluation requires domain context. RMSE penalizes large errors more than MAE. Compare to baseline and business requirements.

---

### Q10. BQML Model Input Validation
**Scenario:** You train a BQML model on 100K rows. Test set: 10K rows. Training shows loss decreasing. But test accuracy drops by 5%. What's happening?

- [x] A. Overfitting: model memorized training data. Diagnose via: (1) compare train vs test loss (gap=overfitting), (2) reduce model complexity, (3) regularization (add L1/L2 penalty), (4) more data. In BQML, set input_label_cols, use balanced data
- [ ] B. Test set is too small
- [ ] C. Overfitting is impossible
- [ ] D. Ignore test performance

> **Explanation:** Train-test gap indicates overfitting. Remedies: simplify model, regularize, get more data, cross-validation. Always monitor both metrics.

---

### Q11. BQML Time Series: Splitting Strategy
**Scenario:** You have 104 weeks of sales data. You want to forecast week 105. How should you split train/test to evaluate model?

- [x] A. Temporal split: train on weeks 1–90, test on weeks 91–104. Never use future data for training (data leakage). Evaluate: does model predict weeks 91–104 well?
- [ ] B. Random split: shuffle and split 80/20
- [ ] C. Use future data for training
- [ ] D. No evaluation needed

> **Explanation:** Time series requires temporal split (preserve order). Random split causes data leakage (model sees future during training). Always train on past, test on future.

---

### Q12. BQML Feature Interactions and Non-linearity
**Scenario:** You train LINEAR_REG: y = w1*age + w2*income + b. Predictions are poor. A colleague suggests: "Add more features." You add: age^2, age*income, income^2. Does this help LINEAR_REG?

- [x] A. Yes; manually creating polynomial/interaction features extends LINEAR_REG expressiveness. y = w1*age + w2*age^2 + w3*age*income + ... captures non-linearity. But BOOSTED_TREE does this automatically
- [ ] B. No; LINEAR_REG can't use these features
- [ ] C. Features don't matter
- [ ] D. Use LINEAR_REG with raw features only

> **Explanation:** LINEAR_REG is linear in features, not necessarily in raw inputs. Creating polynomial/interaction features manually allows non-linearity. BOOSTED_TREE handles this without manual engineering.

---

### Q13. BQML Batch Prediction: ML.PREDICT
**Scenario:** You trained a model. Now you want to predict on 1M new customers. Should you use ML.PREDICT (batch) or serve online endpoint?

- [x] A. Use ML.PREDICT for batch: predict on 1M rows in BigQuery in minutes. Use online endpoint if you need <100ms per-request latency. Batch is cheaper; online is faster
- [ ] B. Always use online endpoint
- [ ] C. Batch and online are the same
- [ ] D. Prediction is impossible at scale

> **Explanation:** Batch prediction (BigQuery SQL) is cost-effective for large-scale, non-time-sensitive predictions. Online endpoints are for real-time, low-latency serving.

---

### Q14. BQML Model Persistence and Versioning
**Scenario:** You train model v1 and deploy. Later, you train model v2 (better performance). You want to keep both. How do you manage versions?

- [x] A. Name models with version: model_v1, model_v2. Store both in BigQuery ML. Keep model_v1 in production, serve model_v2 in canary (5% traffic). After validation, promote v2. Keep old versions for rollback
- [ ] B. Delete v1; keep only v2
- [ ] C. No versioning needed
- [ ] D. Mix v1 and v2 in same model

> **Explanation:** Model versioning enables safe upgrades. Name conventions (v1, v2) or timestamps help. Keep old versions for rollback. Canary deployment validates new version before full promotion.

---

### Q15. BQML Regularization: Preventing Overfitting
**Scenario:** You train BQML LINEAR_REG. Train loss=0.01, test loss=0.5. Model is overfitting. What regularization can you use?

- [x] A. L2 regularization (Ridge): adds penalty for large weights. In BQML, set L2_REG parameter. L1 (Lasso) zeroes out weak features. Adjust regularization strength until train-test gap shrinks
- [ ] B. Remove features
- [ ] C. Ignore overfitting
- [ ] D. Use more data only

> **Explanation:** Regularization penalizes model complexity. L2 shrinks all weights; L1 selects features. Both reduce overfitting. Tune regularization strength via CV.

---

### Q16. BQML Classification Threshold and ROC Curve
**Scenario:** BQML predicts probability of fraud (0–1). Default threshold=0.5 (if prob>0.5, flag as fraud). But you flag 1% of transactions; compliance wants only 0.1%. Should you change threshold?

- [x] A. Yes; lower threshold (e.g., 0.01) flags more transactions as fraud (catches more true fraud but more false alarms). Use ROC curve (ML.ROC_CURVE) to find threshold balancing precision/recall for your use case
- [ ] B. Threshold can't be changed
- [ ] C. 0.5 is always optimal
- [ ] D. Increase threshold indefinitely

> **Explanation:** Classification threshold is adjustable. Lower threshold: higher recall (catch more), lower precision (more false alarms). Choose threshold based on cost of FP vs FN.

---

### Q17. BQML Multi-class Classification Evaluation
**Scenario:** You predict product category (A, B, C, D, E). Model accuracy=90%. But category C accuracy=50%. Why?

- [x] A. Class imbalance: if C is rare in data (5%), model optimizes for majority classes. Check per-class accuracy via confusion matrix. Use BALANCED class weights or collect more C samples
- [ ] B. Accuracy is meaningless for multi-class
- [ ] C. All classes are equally predicted
- [ ] D. Ignore category C

> **Explanation:** Multi-class evaluation needs per-class metrics. Overall accuracy masks poor performance on minority classes. Use confusion matrix and weighted metrics.

---

### Q18. BQML Export and Vertex AI Integration
**Scenario:** You train BQML model. Now you want to deploy to Vertex AI Endpoint for real-time serving. Can you export BQML model?

- [x] A. Export BQML model to Cloud Storage (as .json or model artifact). Import to Vertex AI Model Registry. Deploy endpoint. BQML models are standalone; no framework-specific code needed
- [ ] B. BQML models can't be exported
- [ ] C. Use only BigQuery for serving
- [ ] D. Manual reimplementation required

> **Explanation:** BQML models can be exported and served via Vertex AI. This bridges batch (BigQuery) and online (Vertex) inference.

---

### Q19. BQML Forecasting with External Regressors (Arimaplus)
**Scenario:** You forecast sales using ARIMA. External data: marketing spend, competitor price. Should you include them?

- [x] A. Yes; BQML ARIMAPLUS allows external regressors. Include marketing spend (strong signal) and competitor price. Model: sales ~ ARIMA + external_regressors. Improves accuracy if external data is predictive
- [ ] B. Never use external data
- [ ] C. ARIMA can't use external data
- [ ] D. External data always hurts

> **Explanation:** ARIMAPLUS extends ARIMA with exogenous variables (external regressors). Useful if external factors drive outcome.

---

### Q20. BQML Clustering: K-means and Segmentation
**Scenario:** You want to segment 1M customers into 5 groups (high/medium/low value). Should you use K-MEANS or LINEAR_REG?

- [x] A. Use K-MEANS clustering (unsupervised). No labels needed; groups similar customers. LINEAR_REG is supervised (needs target). Create features (spending, frequency, RFM). Run K-MEANS with k=5. Analyze clusters
- [ ] B. LINEAR_REG is for segmentation
- [ ] C. Clustering is impossible
- [ ] D. Use random assignment

> **Explanation:** K-MEANS groups similar data points. LINEAR_REG predicts numerical targets. Clustering doesn't need labels; ideal for segmentation.

---

## Domain 2: Vertex AI Pipelines & KFP (Q21–Q40)

### Q21. Vertex AI Pipelines: DAG and Components
**Scenario:** You design a pipeline: (1) ingest data, (2) preprocess, (3) train model, (4) evaluate, (5) deploy if eval passes. How do you structure this in Kubeflow Pipelines (KFP)?

- [x] A. Define 5 components (functions wrapped with @dsl.component). Connect via inputs/outputs. Use dsl.Dag to define execution order. Add conditional gate after evaluate: if metrics_pass, deploy; else skip. Compile to YAML
- [ ] B. Write one monolithic script
- [ ] C. Components must run sequentially
- [ ] D. No structured pipeline format

> **Explanation:** KFP components are reusable functions. DAG defines dependencies. Conditionals enable branching (e.g., deploy only if threshold met).

---

### Q22. Component I/O and Artifact Passing
**Scenario:** Component A outputs a trained model (large file). Component B needs to use it for evaluation. How do you pass the model?

- [x] A. Define output parameter in A: model_artifact = Output[Model]. Component B: input model_artifact = Input[Model]. KFP automatically handles artifact storage (Cloud Storage) and passing. Use @dsl.component decorator
- [ ] B. Use global variables
- [ ] C. Artifacts can't be passed between components
- [ ] D. Manual file copying

> **Explanation:** KFP manages artifact I/O. Output[Model], Input[Model] handle storage transparently. Scalable and tracked.

---

### Q23. Conditional Execution in Pipelines
**Scenario:** Evaluate component outputs metrics. If accuracy > 95%, deploy model; else retrain with more data. How do you express this in KFP?

- [x] A. Use dsl.Condition: @dsl.component def deploy(acc) and @dsl.component def retrain. In DAG: with dsl.Condition(evaluate.outputs['accuracy'] > 0.95): deploy_task. Else: retrain_task
- [ ] B. No conditional logic in KFP
- [ ] C. Always deploy
- [ ] D. Manual branching

> **Explanation:** KFP Condition allows branching based on component outputs. Enables dynamic workflows (deploy conditionally).

---

### Q24. Caching in Vertex AI Pipelines
**Scenario:** Your pipeline runs daily. Steps 1–3 use the same input data; only step 4 (model training) changes. Pipeline takes 2 hours. How do you speed it up?

- [x] A. Enable caching: set enable_caching=True on steps 1–3 that don't change. KFP skips re-execution if inputs are identical, reuses cached outputs. Saves 30–60 min. Only step 4 (training) re-runs
- [ ] B. No caching available
- [ ] C. Disable step 1–3
- [ ] D. Rerun everything

> **Explanation:** KFP caching memoizes component outputs. If inputs unchanged, reuse output. Dramatically speeds iterative development.

---

### Q25. Parallel and Sequential Execution
**Scenario:** Pipeline trains 5 different model architectures independently (takes 30 min each). Then ensembles them (5 min). Currently serial: 5*30+5=155 min. How do you optimize?

- [x] A. Parallelize training: use @dsl.component def train_arch1, train_arch2, ... inside a loop or fan-out. KFP runs all 5 in parallel (on separate workers). Total: max(30) + 5 = 35 min. Requires sufficient compute quota
- [ ] B. Serial only
- [ ] C. Parallelization impossible
- [ ] D. Manual parallelization

> **Explanation:** KFP parallelizes independent tasks. Multiple workers = faster execution. Ensemble after all trains complete.

---

### Q26. Artifact Types and Metadata
**Scenario:** Component outputs: trained_model (Model), metrics.json (Metrics), feature_schema (Schema). How do you track these in Vertex Pipelines?

- [x] A. Use Output[Model], Output[Metrics], Output[ClassificationMetrics]. KFP auto-tracks artifact type, stores in Artifact Registry, records metadata (model version, accuracy). Enables lineage (which data trained which model)
- [ ] B. No artifact tracking
- [ ] C. All outputs are same type
- [ ] D. Manual metadata

> **Explanation:** Artifact typing enables lineage, versioning, and governance. Vertex Pipelines automatically records provenance.

---

### Q27. Pipeline Parameterization
**Scenario:** Pipeline trains model with hyperparams: learning_rate=0.01, batch_size=32, epochs=50. You run pipeline daily with different params. How do you avoid hardcoding?

- [x] A. Use pipeline parameters: @dsl.pipeline(name='train', default_params={'learning_rate': 0.01, ...}). Pass at execution: Compiler().compile(pipeline, 'pipeline.yaml'); Run with params override. Enables experimentation
- [ ] B. Edit code for each run
- [ ] C. No parameterization
- [ ] D. Always use defaults

> **Explanation:** Pipeline parameters allow external configuration. Same DAG, different runs with different params.

---

### Q28. Error Handling and Retries
**Scenario:** Component fetches data from Pub/Sub. Network occasionally fails (3% failure rate). Should you retry?

- [x] A. Yes; use @dsl.component(..., retry=3) to retry failed component. Exponential backoff (wait 1s, 2s, 4s). If 3 retries fail, pipeline fails. Handles transient failures gracefully
- [ ] B. No retries
- [ ] C. Fail immediately
- [ ] D. Infinite retries

> **Explanation:** Retries handle transient failures. Exponential backoff avoids thundering herd. Use judiciously (don't retry logic errors).

---

### Q29. Pipeline Scheduling and Continuous Training
**Scenario:** Model degrades monthly (new data arrives, distribution shifts). You want to retrain automatically on schedule. How do you automate?

- [x] A. Define pipeline in KFP. Deploy to Vertex AI Pipelines. Create schedule trigger (Cloud Scheduler): run pipeline every 30 days. Monitor drift; if detected, trigger retraining immediately. Logs training history in Vertex ML Metadata
- [ ] B. Manual retraining
- [ ] C. No scheduling
- [ ] D. Train once, never retrain

> **Explanation:** Vertex Pipelines + Cloud Scheduler enables continuous training. Automatable and auditable.

---

### Q30. Data Validation in Pipelines (TFDV Integration)
**Scenario:** Training data has: 10K new records daily. Previously, NULL age values caused model failures. How do you prevent regressions?

- [x] A. Add TFDV (TensorFlow Data Validation) component: compute statistics, generate schema, validate data quality. Reject data if nulls exceed threshold. Fail pipeline, alert team. Prevents bad data→bad model
- [ ] B. No validation
- [ ] C. Manual inspection
- [ ] D. Ignore nulls

> **Explanation:** Data validation (TFDV) catches quality issues early. Automated gate prevents downstream problems.

---

### Q31. Model Registry and Versioning in Vertex
**Scenario:** You train 3 models (v1, v2, v3). v2 is best. You want to promote v2 to production, keep v1 as fallback. How do you manage versions?

- [x] A. Register all models in Vertex Model Registry: set v2 as "Candidate" (shadow), v1 as "Production" (live). After validation, promote v2 to Production, archive v1. Each version has metadata (training data, metrics, lineage)
- [ ] B. Delete old versions
- [ ] C. No versioning
- [ ] D. Mix versions

> **Explanation:** Model Registry enables versioning and safe promotion. Metadata tracks provenance. Rollback is easy.

---

### Q32. Pipeline Monitoring and Logging
**Scenario:** Pipeline fails at step 3 (training). Logs show out-of-memory error. How do you debug?

- [x] A. Enable verbose logging in KFP. Check Vertex Pipelines dashboard → failed run → step 3 → logs. Observe OOM; increase memory or reduce batch size. Re-run. Logs are persistent in Cloud Logging
- [ ] B. Guess what went wrong
- [ ] C. No logging
- [ ] D. Manual stderr capture

> **Explanation:** Structured logging (Vertex Pipelines → Cloud Logging) enables debugging. Inspect step-by-step execution.

---

### Q33. Pipeline Orchestration vs Scheduling
**Scenario:** Pipeline A trains model. Pipeline B deploys it. Sometimes B runs before A completes. How do you ensure correct order?

- [x] A. Use orchestration: Pipeline B depends on Pipeline A (output artifact of A is input to B). Or use Cloud Workflows to chain pipelines: Run A, wait for completion, then run B. Prevents race conditions
- [ ] B. Run both simultaneously
- [ ] C. Manual coordination
- [ ] D. No dependency management

> **Explanation:** Orchestration expresses dependencies. A finishes, then B starts. Prevents race conditions and data corruption.

---

### Q34. Feature Store Integration in Pipelines
**Scenario:** Training component needs features: customer age, income, credit_score. These exist in Vertex Feature Store. How do you fetch in pipeline?

- [x] A. Add component: fetch features from Vertex Feature Store using feature view (pre-defined grouping). Component outputs dataframe. Training component uses it. Enables point-in-time lookups (features as they were at training time)
- [ ] B. Manual SQL queries
- [ ] C. Feature Store can't integrate with pipelines
- [ ] D. Hardcode features in code

> **Explanation:** Feature Store integration ensures consistent feature engineering. Point-in-time correctness (no data leakage).

---

### Q35. Metrics and Model Evaluation Tracking
**Scenario:** After training, you log metrics: accuracy=92%, precision=0.9, recall=0.85. You want to track these over time to detect degradation. How?

- [x] A. Log metrics to Vertex ML Metadata using @dsl.component output (Metrics artifact). Vertex Pipelines tracks all runs, visualizes trends. Alert if metrics drop >5%. Enables continuous monitoring and regression detection
- [ ] B. CSV file manually
- [ ] C. No tracking
- [ ] D. Ignore metrics

> **Explanation:** ML Metadata tracks all experiments. Historical trends reveal degradation. Automated alerts prevent silent failures.

---

### Q36. Custom Containers in Vertex Pipelines
**Scenario:** You need Python 3.11, custom libraries (not in default KFP images). How do you use custom runtime?

- [x] A. Build custom container image (Dockerfile, push to Artifact Registry). In KFP component, specify @dsl.component(base_image='gcr.io/my-project/my-image:latest'). KFP uses your image for that component
- [ ] B. Only default images allowed
- [ ] C. Manual environment setup
- [ ] D. No customization

> **Explanation:** Custom containers enable custom dependencies. Dockerfile → Artifact Registry → KFP component. Reproducible environments.

---

### Q37. Pipeline Triggering: Event-driven vs Scheduled
**Scenario:** Model retrains on: (1) schedule (monthly) or (2) event (new training data arrives in bucket). How do you trigger both?

- [x] A. Schedule: Cloud Scheduler (cron). Event: Cloud Storage trigger (notify Cloud Pub/Sub when file uploaded) → Cloud Functions → trigger pipeline. Hybrid: schedule + event-driven automation. Provides flexibility
- [ ] B. Schedule only
- [ ] C. Event only
- [ ] D. Manual triggering

> **Explanation:** Scheduled + event-driven triggers enable responsive automation. Combine for best coverage.

---

### Q38. Pipeline Rollback and Recovery
**Scenario:** Pipeline deploys model v2. After 1 hour, performance drops (data quality issue). You want to roll back to v1 immediately. How?

- [x] A. Pipeline stores model versions in Model Registry. Rollback: endpoint → traffic to v1. Simultaneously, run pipeline to fix data issue, retrain, redeploy v2. Use versioning + traffic splitting for quick recovery
- [ ] B. No rollback possible
- [ ] C. Discard v1
- [ ] D. Manual model replacement

> **Explanation:** Model versioning enables fast rollback. Traffic splitting allows gradual rollback (5% v2, 95% v1) for safety.

---

### Q39. Budget and Cost Optimization in Pipelines
**Scenario:** Pipeline runs daily (30 min each): training (N1 VM), evaluation (T4 GPU). Monthly cost: $5K. How do you reduce?

- [x] A. Optimize: (1) use preemptible VMs (70% cheaper, okay for training), (2) use smaller GPU (T4→K80 if throughput allows), (3) profile code (data loading is bottleneck?), (4) cache intermediate outputs (skip redundant steps), (5) cloud build on-demand (no reserved capacity). Target: $2K/month
- [ ] B. No cost control
- [ ] C. Always max resources
- [ ] D. Manual optimization

> **Explanation:** Infrastructure optimization (preemptible, right-sizing, caching) cuts costs 50%+. Monitor via Cloud Billing.

---

### Q40. Compliance and Audit in Pipelines
**Scenario:** Regulations require audit trail: who ran pipeline, when, what data, output models. How do you document?

- [x] A. Vertex Pipelines auto-tracks: execution metadata (user, timestamp), input datasets (lineage), output models (version). ML Metadata stores all. Export audit trail: query Metadata API, generates report. Models tagged with compliance metadata
- [ ] B. No audit trail
- [ ] C. Manual documentation
- [ ] D. Compliance impossible

> **Explanation:** ML Metadata provides audit trail. Compliance-ready with lineage, versioning, and access logs.

---

## Domain 3: Data Engineering/Dataflow/Beam (Q41–Q60)

### Q41. Windowing: Tumbling vs Sliding vs Session
**Scenario:** Real-time streaming: sensor readings (every 1 second). You want metrics every 1 minute (1M readings/min). Which window type: tumbling (fixed 1-min chunks), sliding (1-min window, every 30 sec), or session (adaptive)?

- [x] A. Use tumbling (1-min, fixed) for regular metrics. Sliding if you want smooth overlapping views (trade-off: 2x computation). Session if idle periods matter (e.g., user sessions in logs). For sensors: tumbling is efficient
- [ ] B. Always use sliding
- [ ] C. Session only
- [ ] D. No windowing

> **Explanation:** Windowing partitions streams. Tumbling: non-overlapping. Sliding: overlapping (redundant). Session: event-driven gaps. Choose by use case.

---

### Q42. Watermarks and Late Data
**Scenario:** Events arrive (timestamp T). Watermark lags 30 seconds (data arrives late). Event at T=100s arrives at wall-clock 130s. You've already emitted window [60–120s]. Should you re-emit with late event?

- [x] A. Yes; use allowed_lateness=30s in Dataflow. When late data arrives within 30s, re-emit window. If you care about completeness, keep window open. If punctuality matters, close window early. Trade-off: lateness vs completeness
- [ ] B. Discard late data
- [ ] C. Watermarks don't exist
- [ ] D. Ignore lateness

> **Explanation:** Watermarks estimate completeness. allowed_lateness handles late events. Adjust by use case (metrics vs logs).

---

### Q43. Triggers: When to Emit Window Results
**Scenario:** Window [0–60s]. Early result at 5s (partial). Final result at 60s (complete). Should you emit early?

- [x] A. Depends on use case. Default: emit on watermark (complete data, final). Add trigger: AfterWatermark().with_early_firings() to emit early (e.g., every 5s). Trade-off: low latency (early) vs high accuracy (final). Choose by SLA
- [ ] B. Never emit early
- [ ] C. Always emit early
- [ ] D. No triggers

> **Explanation:** Triggers control emission. Early+Final+Late allow responsive, complete streaming. Latency vs completeness trade-off.

---

### Q44. Side Inputs and Reference Data
**Scenario:** Stream of transactions. You want to enrich each transaction with: customer tier (gold/silver/bronze from database). Tier rarely changes. How do you join?

- [x] A. Load tier reference data into side input (broadcast to all workers). In pipeline: main stream joined with side input via ParDo. Efficient (small lookup table broadcast once). Reload when reference updates. Avoids expensive stream-stream join
- [ ] B. Stream-stream join (expensive)
- [ ] C. No enrichment
- [ ] D. Manual lookup

> **Explanation:** Side inputs efficiently join streams with reference data. Broadcast small tables; stream-stream joins are expensive.

---

### Q45. Pub/Sub Integration with Dataflow
**Scenario:** Millions of events/second on Pub/Sub topic. You process (filter, aggregate) and write to BigQuery. How do you scale?

- [x] A. Use Cloud Dataflow template: Pub/Sub to BigQuery. Dataflow auto-scales workers based on Pub/Sub backlog (Stackdriver metrics). Pipeline: Read(Pub/Sub) → Filter → Window → Aggregate → Write(BigQuery). Highly available, fault-tolerant
- [ ] B. Single worker
- [ ] C. Manual scaling
- [ ] D. Pub/Sub-Dataflow impossible

> **Explanation:** Dataflow auto-scales with Pub/Sub. Templates provide turnkey solutions. Pub/Sub backlog triggers scaling.

---

### Q46. Stateful Processing in Dataflow
**Scenario:** Stream of user actions. You want to track: total spend per user, sessions started count. State is mutable (persists across windows). How do you maintain state?

- [x] A. Use ParDoFn with @process_bundle and self.state (per-worker state) or StateValueSpec for keyed state. For user-level aggregation: use stateful ParDo. State is fault-tolerant (backed by Dataflow state service). Caution: state per key must fit in memory
- [ ] B. Global in-memory dict
- [ ] C. No state
- [ ] D. Manual state management

> **Explanation:** Stateful processing maintains per-key state (e.g., per-user counters). Fault-tolerant via Dataflow state service. Use judiciously (memory-bounded).

---

### Q47. Source and Sink Connectors
**Scenario:** Pipeline reads from: Cloud Storage (CSV), Pub/Sub (streaming), BigQuery (reference). Writes to: BigQuery (results), Cloud Storage (archive). How do you handle multiple I/O?

- [x] A. Use BeamPipeline with multiple Read sources (Cloud Storage, Pub/Sub, BigQuery) and Write sinks. Branch pipeline: Read → Filter/Transform → Write to different sinks. Dataflow handles orchestration. Each sink runs independently (fan-out)
- [ ] B. Single source/sink
- [ ] C. Manual I/O
- [ ] D. No multiple connectors

> **Explanation:** Beam supports multiple sources/sinks. Fan-out writes to different destinations in parallel.

---

### Q48. Backpressure and Flow Control
**Scenario:** Source produces 1M events/sec. Processing takes 5 sec per event (bottleneck: expensive ML inference). Backlog grows. Memory fills. What happens?

- [x] A. Backpressure (flow control): Dataflow slows down source reading to match processing speed. Backlog stabilizes (doesn't grow indefinitely). Auto-scaling helps: add more workers to process faster. Monitor: if CPU <50%, add workers; if CPU >90%, investigate bottleneck
- [ ] B. All events processed immediately
- [ ] C. No backpressure
- [ ] D. Out-of-memory crash

> **Explanation:** Backpressure prevents resource exhaustion. Source slows to match sink speed. Auto-scaling adjusts worker count.

---

### Q49. Dataflow Templates and Reusability
**Scenario:** You write a Dataflow pipeline: Pub/Sub → filter → BigQuery. You want to reuse it with different: input topics, output tables, filter conditions. How?

- [x] A. Create Dataflow template: parameterize (template parameters for topic, table, filter). Package template in Cloud Storage. Execute via API with different params. Dataflow launches workers, runs pipeline with params. Enables self-service for non-engineers
- [ ] B. Copy-paste and edit code each time
- [ ] C. Templates don't exist
- [ ] D. Hardcode all values

> **Explanation:** Dataflow templates enable reusability. Parameterized pipelines allow self-service without coding.

---

### Q50. Exactly-Once Semantics (Deduplication)
**Scenario:** Stream of financial transactions. Some events are duplicated (network retries). You write to BigQuery. Should you deduplicate?

- [x] A. Yes; use exactly-once semantics. Dataflow + BigQuery streaming inserts support exactly-once (deduplication by insert ID). If duplicate detected, BigQuery discards. Alternatively, use ParDo with state to track seen IDs (memory-bounded dedup). Prevents double-counting
- [ ] B. No deduplication
- [ ] C. Duplicates are okay
- [ ] D. Manual dedup

> **Explanation:** Streaming systems must handle duplicates. Exactly-once semantics or explicit deduplication prevent data corruption.

---

### Q51. Batch vs Streaming in Dataflow
**Scenario:** Daily ETL: read 1B rows from BigQuery, transform, write to Cloud Storage. Also need real-time: Pub/Sub → transform → BigQuery. Can same pipeline handle both?

- [x] A. Yes; Beam unified: batch (bounded sources like BigQuery) and streaming (unbounded like Pub/Sub) use same APIs. Pipeline logic is identical. Dataflow runner selects batch vs streaming execution. One codebase, dual deployment. Reduces maintenance
- [ ] B. Separate pipelines required
- [ ] C. No batch-streaming unification
- [ ] D. Manual duplication

> **Explanation:** Beam's unified model handles batch and streaming. Single codebase, multiple execution modes.

---

### Q52. Performance Optimization: Parallelism and Bundles
**Scenario:** Pipeline processes 1B records. Single DoFn bottleneck (takes 10s per bundle, serial). Throughput: 100K rec/sec. How do you speed up?

- [x] A. Increase parallelism: add more workers (auto-scaling), increase bundles per worker. Optimize DoFn: batch operations (vectorized), cache expensive calls. Use profiler to identify bottleneck (I/O vs CPU). Typical speedup: 5–10x with optimization
- [ ] B. Single worker
- [ ] C. No optimization
- [ ] D. Accept slow throughput

> **Explanation:** Parallelism and profiling enable scaling. Vectorization and caching speed up DoFns.

---

### Q53. Dead Letter Pattern for Error Handling
**Scenario:** Pipeline processes events. 0.1% fail (malformed JSON). Pipeline crashes on first error. How do you handle?

- [x] A. Use dead letter pattern: ParDo with TryCatch. Try: process event. Catch: write to dead letter queue (Cloud Pub/Sub topic or BigQuery table). Main pipeline continues (doesn't fail on errors). Monitor dead letter queue; fix bad records offline
- [ ] B. Crash on error
- [ ] C. Ignore errors
- [ ] D. No error handling

> **Explanation:** Dead letter pattern decouples error handling. Bad records isolated; pipeline continues. Enables debugging without blocking production.

---

### Q54. Filtering and Transformations
**Scenario:** Stream of 1M clicks/sec. You filter: (1) bots (user agent = "bot"), (2) internal IPs. Then aggregate: clicks per URL. Which operations?

- [x] A. Filter(lambda x: "bot" not in x["user_agent"] and x["ip"] not in INTERNAL_IPS) → ParDo(count per URL) → Write. Filters reduce data early (save bandwidth). ParDo applies transformations. Early filtering is critical for large streams
- [ ] B. No filtering
- [ ] C. Filter after aggregation
- [ ] D. Manual filtering

> **Explanation:** Push filters early (reduce data volume). Transformations after filtering. Reduces downstream load.

---

### Q55. Join Patterns: Stream-Stream, Stream-Table
**Scenario:** Stream A: user purchases. Stream B: user profile updates. You want: purchase enriched with current profile. How?

- [x] A. Stream-table join: Load user profile into side input (reference table, updated hourly). Join stream A with side input via ParDo. Efficient. Alternatively, stream-stream join (windowed) if both unbounded. Trade-off: side input is simple but may be stale; stream-stream is live but expensive
- [ ] B. No enrichment
- [ ] C. Global join impossible
- [ ] D. Manual join

> **Explanation:** Stream-table joins (side input) are efficient. Stream-stream joins are expensive but real-time. Choose by latency/cost trade-off.

---

### Q56. Autoscaling Policies in Dataflow
**Scenario:** Pipeline currently processes 100K events/sec. Load varies hourly (peak: 1M events/sec). Autoscaling: currently 10 workers. How do you set policy?

- [x] A. CPU-based autoscaling: target 70% CPU. At 100K events: 10 workers, ~10% CPU (too much spare capacity). At 1M events: scale to 100 workers, hit 70% target. Set min=5, max=200. Monitor: watch Stackdriver for CPU, worker count trends
- [ ] B. No autoscaling
- [ ] C. Fixed workers
- [ ] D. Manual adjustment

> **Explanation:** CPU-based autoscaling matches load. Set target CPU (70% is typical), min/max workers. Responsive to spikes.

---

### Q57. Monitoring and Alerting in Dataflow
**Scenario:** Pipeline latency increases from 5min to 15min (user-facing dashboard impact). How do you diagnose?

- [x] A. Check Dataflow dashboard: element latency (read, process, write), lag (Pub/Sub backlog). If read latency high: source issue. If process high: DoFn bottleneck (profile). If write high: sink issue (BigQuery throttle). Set alert: if element latency >10min, notify. Investigate
- [ ] B. Ignore latency
- [ ] C. No monitoring
- [ ] D. Manual checks

> **Explanation:** Element latency breakdown reveals bottlenecks. Alerts on SLA violations trigger investigation.

---

### Q58. Sidecar Patterns and External Services
**Scenario:** DoFn enriches records by calling external ML inference service (100ms each). Throughput drops. How do you scale?

- [x] A. Batch requests: instead of 1 inference per record, batch 100 records per request. Call inference service with batch. Throughput improves 10x (1 call vs 100). Use ParDo with bundle-level batching. Caution: increased latency due to batching (acceptable?)
- [ ] B. Sequential calls
- [ ] C. No batching
- [ ] D. Remove inference

> **Explanation:** Batching external calls reduces overhead. Trade latency for throughput. Tune batch size by use case.

---

### Q59. Replayability and Checkpointing
**Scenario:** Pipeline processes events with IDs: [1, 2, 3, 4, 5]. At event 3, crash (Dataflow fails). On restart, where does it resume?

- [x] A. Dataflow checkpoints state periodically. On crash, resume from last checkpoint (event 2). Events [3, 4, 5] reprocessed. Duplicate event 2's output (at-least-once semantics). For exactly-once, use idempotent writes (BigQuery deduplication) or transactions
- [ ] B. Restart from beginning
- [ ] C. No checkpointing
- [ ] D. Lost events

> **Explanation:** Checkpointing enables fault recovery. At-least-once guarantees; exactly-once requires idempotent sinks.

---

### Q60. Transitioning from Batch to Streaming
**Scenario:** Currently use Cloud Functions (batch) to process daily logs (1TB). New requirement: real-time metrics within 1 minute. Should you switch to Dataflow?

- [x] A. Yes; Dataflow streaming enables real-time. Start with batch (existing pipeline). Add streaming sidecar: Pub/Sub ingests real-time logs → Dataflow stream processes → BigQuery updated live. Phase: batch (nightly) + streaming (real-time). Eventually retire batch. Lower ops complexity than managing Lambda (batch + streaming separately)
- [ ] B. Stay with batch only
- [ ] C. Streaming impossible
- [ ] D. Complex architecture

> **Explanation:** Dataflow unified batch/stream simplifies architectures. Migrate incrementally (batch + stream coexist).

---

## Domain 4: Vertex AI Model Training & Deployment (Q61–Q80)

### Q61. Custom Training with Vertex AI
**Scenario:** You want to train a TensorFlow model on GPU (requires 40GB VRAM). Should you use AutoML or Custom Training?

- [x] A. Use Custom Training: define training script (train.py), specify GPU (A100 with 40GB). Vertex AI runs script on GKE worker. More control than AutoML. For simple models, AutoML is faster; for custom architectures, use Custom Training
- [ ] B. AutoML always
- [ ] C. Only CPU training
- [ ] D. No GPU support

> **Explanation:** Custom Training offers flexibility. AutoML is simpler but limited. Choose by complexity vs ease trade-off.

---

### Q62. Hyperparameter Tuning with Vertex AI
**Scenario:** Model trains with: learning_rate ∈ [0.001, 0.1], batch_size ∈ [16, 128], dropout ∈ [0.1, 0.5]. You want best hyperparams (minimize validation loss). How?

- [x] A. Use Vertex AI Hyperparameter Tuning (HPT): specify parameter ranges, metric (validation_loss). Tuning runs parallel trials (e.g., 10 workers test different params). Best params selected. Saves time (parallel) vs manual grid search. Trade-off: cost (many trials)
- [ ] B. Manual grid search
- [ ] C. No tuning
- [ ] D. Random params

> **Explanation:** Hyperparameter tuning automates search. Parallel trials find good params faster. More expensive but better results.

---

### Q63. Model Evaluation Before Deployment
**Scenario:** You train model with 95% accuracy on validation set. Before deploying to production, what tests should you run?

- [x] A. Evaluate on held-out test set (unseen data). Check fairness (accuracy per demographic). Test edge cases (empty input, extreme values). Monitor inference latency and throughput. Test rollback procedures. Only deploy if test accuracy >90% and fairness gaps <5%
- [ ] B. Deploy immediately if val accuracy is high
- [ ] C. No pre-deployment tests
- [ ] D. Test on training data only

> **Explanation:** Pre-deployment evaluation prevents failures. Test set, fairness, edge cases, performance. Comprehensive validation reduces incidents.

---

### Q64. Endpoint Deployment and Traffic Splitting
**Scenario:** Model v2 (trained) is ready. v1 is in production. You want to gradually shift traffic: 5% v2, 95% v1. How?

- [x] A. Create endpoint with traffic split: 5% → v2, 95% → v1. Monitor v2 metrics (latency, errors). If good, increase to 25% v2. A/B test: measure which version has better CTR. Gradually shift (canary deployment). Rollback easy if v2 fails
- [ ] B. Switch immediately to v2
- [ ] C. No traffic splitting
- [ ] D. Keep v1 forever

> **Explanation:** Traffic splitting enables safe rollout. Canary deployment detects issues early. A/B testing validates improvements.

---

### Q65. Model Serving: Online vs Batch
**Scenario:** You need: (1) real-time predictions for web app (user submits form, predict in <100ms), (2) daily batch scoring on 10M customers. Use 1 or 2 endpoints?

- [x] A. Use 2: Online endpoint for web app (Vertex AI Prediction, low-latency). Batch endpoint (Vertex AI Batch Prediction, cost-effective, runs job overnight). Same model, different serving modes. Reduces cost (pay-per-prediction online, flat batch job cost)
- [ ] B. Batch only
- [ ] C. Online only
- [ ] D. Same endpoint for both

> **Explanation:** Batch and online have different tradeoffs (speed vs cost). Use both for hybrid needs.

---

### Q66. Model Monitoring: Drift Detection
**Scenario:** Model predicts loan approval. After 3 months, approval rate drops from 45% to 30%. Performance on test set unchanged. What's happening?

- [x] A. Data drift: input distribution shifted (new data differs from training data). Age distribution changed, or credit scores shifted. Detect via monitoring: compare new data distribution to training baseline (Kolmogorov-Smirnov test, chi-square). Alert if drift >threshold. Trigger retraining with new data
- [ ] B. No problem
- [ ] C. Model is broken
- [ ] D. Ignore drift

> **Explanation:** Data drift causes performance drop on live data while test accuracy holds. Continuous monitoring detects drift. Retraining recovers performance.

---

### Q67. Prediction Logging and Feedback Loop
**Scenario:** Model makes predictions. You want to track: which customers got predicted as "risky", ground truth labels (actual defaults 3 months later). How do you set up feedback?

- [x] A. Enable prediction logging: Vertex AI auto-logs predictions to BigQuery (features, predicted label, confidence). Link to ground truth labels (join on customer ID, date). Compute accuracy on logged predictions. Use for retraining with fresh, labeled data
- [ ] B. No logging
- [ ] C. Manual tracking
- [ ] D. Feedback impossible

> **Explanation:** Prediction logging creates labeled data for retraining. Feedback loop enables continuous improvement.

---

### Q68. Model Explainability: Feature Importance
**Scenario:** Model predicts customer churn. You need to explain: "Why is this customer likely to churn?" Feature importance shows: last_purchase_days_ago=0.7, customer_lifetime_value=0.2. How do you interpret?

- [x] A. last_purchase_days_ago is most important (70% contribution). For high-churn customer, recency is key (long time since purchase indicates churn risk). Use Vertex AI Explainable AI (SHAP, integrated gradients) to get per-instance explanations. Show explanation to customer: "You haven't purchased in 90 days, so we predict churn"
- [ ] B. All features equally important
- [ ] C. Importance is meaningless
- [ ] D. Only use feature importance

> **Explanation:** Feature importance helps model transparency. Per-instance explanations drive action (e.g., offer discount to high-risk churners).

---

### Q69. Model Container and Serving Framework
**Scenario:** You train model in PyTorch. How do you deploy to Vertex AI Endpoint?

- [x] A. Create prediction container: (1) write predictor.py (load model, define predict(instance)), (2) Dockerfile (FROM python:3.9, COPY model.pkl, COPY predictor.py), (3) build and push to Artifact Registry, (4) create endpoint with custom container. Vertex AI runs container, handles autoscaling
- [ ] B. Deploy .py directly
- [ ] C. Framework matters
- [ ] D. Containers not needed

> **Explanation:** Custom containers enable any framework (PyTorch, JAX, etc.). Dockerfile defines environment. Vertex AI handles orchestration.

---

### Q70. Prediction Latency Optimization
**Scenario:** Prediction endpoint latency: 500ms (slow for real-time app). Profiling shows: model inference 50ms, loading features 400ms. How do you optimize?

- [x] A. Feature latency is bottleneck (400ms). Cache features: pre-load customer features to in-memory cache (Redis) before prediction. Or batch: collect requests, predict 100 at once (amortize latency). Target: <100ms latency. A/B test: does caching improve UX?
- [ ] B. No optimization
- [ ] C. Model inference is slow
- [ ] D. Accept 500ms

> **Explanation:** Identify bottleneck (feature loading, not model). Caching and batching reduce latency dramatically.

---

### Q71. Model Versioning and Rollback
**Scenario:** You deploy v2 (better accuracy). After 1 day, accuracy degrades (distribution shift). You want to rollback to v1. How?

- [x] A. Endpoint has v2 deployed. Rollback: re-create endpoint with v1 model (or traffic split 100% → v1). Takes minutes. Keep v2 registered in Model Registry (version history). Diagnose v2 issue, fix, retrain, redeploy. Versioning enables fast recovery
- [ ] B. Can't rollback
- [ ] C. Permanently stuck with v2
- [ ] D. Manual retraining

> **Explanation:** Model versioning and registry enable quick rollback. Safe deployment recovery.

---

### Q72. Autoscaling for Endpoints
**Scenario:** Endpoint receives variable traffic: 10 req/sec (off-peak), 1K req/sec (peak). You allocate 2 Replicas now. Peak times, latency spikes to 5s (SLA: <1s). How?

- [x] A. Enable autoscaling: set min=2, max=20 replicas. Target CPU 70% or request latency <1s. Vertex AI auto-scales: at 1K req/sec, scales to 10 replicas (CPU ~70%), latency <1s. Cost: pay per replica per hour (only when needed). Monitor scaling lag (usually <1 min)
- [ ] B. Fixed 2 replicas
- [ ] C. No autoscaling
- [ ] D. Increase to 20 always

> **Explanation:** Autoscaling maintains SLA while reducing cost. Scale by CPU or latency. Pay only for used capacity.

---

### Q73. Request/Response Format for Serving
**Scenario:** Model expects: JSON input {"age": 25, "income": 50000}. REST endpoint receives: CSV format (1 row per request). How do you handle?

- [x] A. Define request schema in custom predictor.py: parse CSV, convert to JSON, pass to model. Response: JSON output. Or pre-process: client converts CSV to JSON before API call. Custom container handles format conversion (flexibility)
- [ ] B. Model must use exact format
- [ ] C. No conversion
- [ ] D. Accept all formats

> **Explanation:** Custom containers enable format flexibility. Predictor bridges client format and model API.

---

### Q74. Batch Prediction at Scale
**Scenario:** Score 100M customers for risk. Online endpoint expensive (pay per prediction). Use batch prediction? Expected runtime: 2 hours, cost: $100.

- [x] A. Yes; Batch Prediction: load 100M rows from BigQuery, run prediction job (parallel, serverless), write results back to BigQuery. Cost: <$100, time: ~2 hours. Way cheaper than online (would cost $1K+). Use for offline scoring (customer lists, reports)
- [ ] B. Online endpoint
- [ ] C. Manual scoring
- [ ] D. Can't score 100M

> **Explanation:** Batch prediction is cost-effective at scale. Ideal for bulk scoring. Online for real-time.

---

### Q75. Model Registry and Artifact Management
**Scenario:** You train 5 models over 3 months. Each has: model artifact, training code, evaluation metrics, schema. How do you organize?

- [x] A. Use Vertex AI Model Registry: upload each model with metadata (version, metrics, training dataset, responsible team). Each model version has: artifact (Cloud Storage), lineage (which data trained it), evaluation results. Query registry: find best model by metric, rollback to older version, compare versions
- [ ] B. No organization
- [ ] C. Manual spreadsheet
- [ ] D. No history

> **Explanation:** Model Registry provides version control and lineage. Essential for ML governance.

---

### Q76. Fairness and Bias Detection
**Scenario:** Loan approval model. Overall accuracy: 90%. But accuracy for women: 75%, men: 92%. Is model biased?

- [x] A. Yes; accuracy gap (17%) indicates bias. Model performs worse on women. Root cause: training data imbalance or feature bias (e.g., occupation proxy for gender). Remedies: (1) collect more women's data, (2) use fairness constraints during training (equalized odds), (3) remove biased features, (4) audit with Vertex AI Fairness & Bias (computes demographic parity, equalized odds)
- [ ] B. No bias
- [ ] C. Bias is okay
- [ ] D. Don't check

> **Explanation:** Fairness monitoring detects discriminatory models. Bias remediation ensures ethical AI.

---

### Q77. Continuous Model Retraining
**Scenario:** Model deployed 6 months ago. Monthly accuracy drop: 98% → 95%. Data distribution changed (new user demographics). How do you automate retraining?

- [x] A. Set up pipeline: (1) monthly trigger (Cloud Scheduler), (2) pipeline fetches latest data, trains model, evaluates. If accuracy >95% and improves by >2%, promote to production. If accuracy drops >2%, alert team (manual review). Use ML Metadata to track all experiments
- [ ] B. Manual retraining
- [ ] C. Never retrain
- [ ] D. Retrain daily (expensive)

> **Explanation:** Automated retraining keeps models current. Conditional promotion ensures quality. Prevents performance decay.

---

### Q78. Shadow Mode Deployment
**Scenario:** You want to test v2 before full deployment. v1 is in production. Can you run v2 in parallel for validation?

- [x] A. Yes; shadow deployment: route traffic to v1 (primary) and v2 (shadow). v2 predictions logged but NOT returned to users. Compare v2 predictions to v1. If v2 metrics (latency, errors) good, promote to production. Enables risk-free testing of new models
- [ ] B. No shadow mode
- [ ] C. Can't test before deploy
- [ ] D. Full deploy only

> **Explanation:** Shadow deployment validates models before production. Low-risk testing.

---

### Q79. Model Compression for Mobile/Edge
**Scenario:** Model is 500MB (too large for mobile app). Inference takes 2s (too slow). How do you compress?

- [x] A. Techniques: (1) quantization (reduce float32 to int8, 4x smaller), (2) pruning (remove weak weights), (3) distillation (train smaller student model on large teacher model). Result: 50MB model, 200ms inference. Trade: accuracy drop ~1–2%. Test on device before deploy
- [ ] B. No compression
- [ ] C. Large models only
- [ ] D. Accept 500MB

> **Explanation:** Model compression enables on-device serving. Quantization and distillation reduce size and latency.

---

### Q80. Model Card and Documentation
**Scenario:** Colleagues ask: "How was model trained? What data? What are limitations?" You need documentation. What's a Model Card?

- [x] A. Model Card: document model details (training data, features, performance metrics, fairness evaluation, limitations, recommended use cases). Store in Model Registry. Share with stakeholders. Enables transparency, reproducibility, and responsible deployment. Standard practice (Google, Hugging Face use Model Cards)
- [ ] B. Not needed
- [ ] C. Secret documentation
- [ ] D. Undocumented models

> **Explanation:** Model Cards promote transparency and responsible AI. Essential for governance and collaboration.

---

## Domain 5: Analytics & BigQuery SQL (Q81–Q100)

### Q81. BigQuery Partitioning Strategy
**Scenario:** Table with 1B rows (daily events for 5 years). Queries usually filter by date (WHERE date >= '2024-01-01'). Table size: 5TB. How do you optimize?

- [x] A. Partition by date (ingestion time or specific column). BigQuery only scans relevant date partitions (reduces data scanned, faster queries, lower cost). Instead of scanning 5TB, scan ~100GB per month. Use _PARTITIONTIME pseudo-column for queries. Partition pruning is automatic
- [ ] B. No partitioning
- [ ] C. Partition by random column
- [ ] D. Scan all 5TB always

> **Explanation:** Partitioning reduces data scanned. Essential for large tables. Huge cost and speed savings.

---

### Q82. BigQuery Clustering for Performance
**Scenario:** Query filters by: WHERE country='US' AND user_id=123. Without clustering, scans entire partition. With clustering by [country, user_id], how does it help?

- [x] A. Clustering collocates similar rows (country/user_id) on same storage blocks. Query filters: scans only relevant blocks (95% fewer blocks touched). Faster + cheaper. Combine: partition by date, cluster by [country, user_id]. Clustered tables auto-maintain (no manual rebuild)
- [ ] B. No benefit
- [ ] C. Clustering is slower
- [ ] D. Manual clustering

> **Explanation:** Clustering further prunes data within partitions. Multiple clustering columns for complex filters.

---

### Q83. BigQuery Window Functions
**Scenario:** Sales data: [date, customer_id, amount]. You want: for each customer, calculate running total (cumulative sum over time). How?

- [x] A. Window function: SELECT customer_id, date, amount, SUM(amount) OVER (PARTITION BY customer_id ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total. Efficient (single scan), avoids self-join
- [ ] B. Self-join (expensive)
- [ ] C. Multiple queries
- [ ] D. Window functions impossible

> **Explanation:** Window functions enable complex analytics. Running totals, rank, dense_rank, etc. Single scan efficiency.

---

### Q84. BigQuery JOIN Optimization
**Scenario:** Table A: 1B rows. Table B: 1M rows. Query: SELECT * FROM A JOIN B ON A.id = B.id. Broadcast join vs sort-merge?

- [x] A. BigQuery auto-chooses: if B is small (<10GB), broadcast join (copy B to all workers, faster). If both large, sort-merge (sort both, merge). Explicit hint: use INNER JOIN (A, B) or /* @broadcast(B) */ to force broadcast. For your case (1B vs 1M), broadcast is efficient
- [ ] B. Always sort-merge
- [ ] C. No optimization
- [ ] D. Same speed

> **Explanation:** BigQuery automatically optimizes joins. Broadcast for small tables, sort-merge for large. Hints enable manual control.

---

### Q85. BigQuery Subqueries and CTEs
**Scenario:** Need: average customer spending, rank customers by spending, filter top 100. Nested subqueries are slow. Use CTE (Common Table Expression)?

- [x] A. Yes; WITH cte AS (SELECT customer_id, SUM(amount) as total FROM sales GROUP BY customer_id), ranked AS (SELECT *, ROW_NUMBER() OVER (ORDER BY total DESC) as rank FROM cte) SELECT * FROM ranked WHERE rank <= 100. CTEs are readable and often faster (materialized or optimized)
- [ ] B. Nested subqueries only
- [ ] C. Multiple queries
- [ ] D. CTEs impossible

> **Explanation:** CTEs improve readability and performance. Equivalent to temporary views, often optimized.

---

### Q86. BigQuery UNION and Set Operations
**Scenario:** Two tables: us_sales (USA orders), intl_sales (international orders). Combine and aggregate (total revenue by region). Use UNION or UNION ALL?

- [x] A. Use UNION ALL (faster): concatenate both tables without deduplication. Then aggregate: SELECT region, SUM(revenue) FROM (SELECT * FROM us_sales UNION ALL SELECT * FROM intl_sales) GROUP BY region. UNION ALL is usually correct (no duplicates expected). UNION removes duplicates (slower)
- [ ] B. UNION only
- [ ] C. Separate queries
- [ ] D. UNION impossible

> **Explanation:** UNION ALL preserves duplicates (faster). UNION deduplicates (slower). Choose based on data.

---

### Q87. BigQuery Aggregations with GROUP BY
**Scenario:** Sales data with NULL values in region column (10% of rows). Query: SELECT region, COUNT(*) FROM sales GROUP BY region. Does GROUP BY include NULLs?

- [x] A. Yes; GROUP BY treats NULL as a group. Result has row with region=NULL, count=0.1M (10% of rows). If you want to exclude NULLs: WHERE region IS NOT NULL GROUP BY region. Or: COALESCE(region, 'Unknown') GROUP BY ... to aggregate NULLs into 'Unknown'
- [ ] B. NULLs excluded automatically
- [ ] C. NULLs aren't groupable
- [ ] D. No NULL handling

> **Explanation:** GROUP BY includes NULL. Explicit filtering or COALESCE for control.

---

### Q88. BigQuery Nested and Repeated Fields
**Scenario:** Table with structure: [customer_id, name, orders: [order_id, amount, date, items: [item_id, qty, price]]]. Query: total revenue per customer?

- [x] A. FLATTEN and aggregate: WITH flattened AS (SELECT customer_id, item.price * item.qty as item_value FROM sales, UNNEST(orders) as order, UNNEST(order.items) as item) SELECT customer_id, SUM(item_value) as total_revenue FROM flattened GROUP BY customer_id. UNNEST denormalizes nested arrays
- [ ] B. No nested queries
- [ ] C. Can't aggregate nested
- [ ] D. Manual denormalization

> **Explanation:** UNNEST flattens nested structures. Enables analysis of complex data.

---

### Q89. BigQuery Approximate Functions
**Scenario:** Table with 100B rows. You want: approximate distinct count of user_id (EXACT COUNT(DISTINCT user_id) takes 10 min, too slow). Alternative?

- [x] A. Use APPROX_COUNT_DISTINCT(): fast approximate result (usually within 1% error) in <1 second. Or HyperLogLog sketch (APPROX_QUANTILES for percentiles). Trade: accuracy for speed. For most metrics, approximate is sufficient
- [ ] B. EXACT COUNT only
- [ ] C. No approximation
- [ ] D. Accept 10 min

> **Explanation:** Approximate functions speed up large aggregations. Acceptable for analytics (reports, dashboards).

---

### Q90. BigQuery Materialized Views
**Scenario:** Popular query (SELECT date, region, SUM(revenue) FROM sales GROUP BY date, region) runs hourly, takes 5 min, scans 1TB. Optimize?

- [x] A. Create materialized view: CREATE MATERIALIZED VIEW sales_by_region_date AS (SELECT...). BigQuery stores pre-computed result. Query MV instead of raw table (instant, costs 0 bytes scanned). MV auto-refreshes (can set schedule). Huge speedup for repeated aggregations
- [ ] B. No view
- [ ] C. Repeat query hourly
- [ ] D. Accept 5 min

> **Explanation:** Materialized views cache aggregations. Instant queries, auto-refresh. Ideal for dashboards.

---

### Q91. BigQuery BI Engine Acceleration
**Scenario:** Dashboard with 50 queries (each 1–10 sec). Users complain: slow. BigQuery slots reserved for batch, so interactive queries slow. Alternative?

- [x] A. Use BI Engine (in-memory cache): reserve 50GB BI Engine. Vertex AI BI/ Looker dashboards use BI Engine: first query (computes), stores in RAM. Subsequent queries hit cache (instant). No query latency. Cost: extra monthly fee for BI Engine reservation
- [ ] B. No acceleration
- [ ] C. Larger slots
- [ ] D. Accept slow dashboards

> **Explanation:** BI Engine provides in-memory caching. Instant dashboards. Ideal for interactive BI.

---

### Q92. BigQuery Data Sharing with Authorized Views
**Scenario:** You have sensitive data (customers). You want to share with partner: allow them to query but only see specific columns (hide salary). Use views?

- [x] A. Create authorized view: (1) SELECT customer_id, name, location FROM sensitive_table (hide salary), (2) GRANT SELECT on view to partner's project. Partner queries view (sees only 3 columns). View acts as security gate (data access control)
- [ ] B. Share raw table
- [ ] C. No sharing
- [ ] D. Manual filtering

> **Explanation:** Authorized views enable secure data sharing. Column/row-level access control.

---

### Q93. BigQuery Time Travel with Snapshots
**Scenario:** Accidental DELETE deleted 1M rows (it's been 2 hours). Can you recover?

- [x] A. Time travel: SELECT * FROM `project.dataset.table`@-7200000 (2 hours ago, in milliseconds). BigQuery stores table snapshots (default 7 days retention). Restore deleted data from snapshot. Or use TIME TRAVEL in CREATE TABLE AS SELECT to restore specific time
- [ ] B. Data lost forever
- [ ] C. No snapshots
- [ ] D. Manual backup only

> **Explanation:** Time travel enables recovery from accidental deletes. Snapshots retained 7 days (configurable).

---

### Q94. BigQuery Incremental Data Loading
**Scenario:** Daily ETL loads 1M new records. Currently, reload entire table (100M rows): 5 min. Inefficient. How to load only new records?

- [x] A. Incremental load: (1) track max timestamp of last load, (2) load only WHERE timestamp > last_max, (3) append to table (upsert on unique ID if needed). Dataflow template: Cloud Storage → BigQuery (incremental). Reduces load time to <1 min. Standard pattern for streaming/batch ETL
- [ ] B. Full reload always
- [ ] C. No incremental
- [ ] D. Accept 5 min

> **Explanation:** Incremental loading reduces load time. Standard for daily ETL. Use CDC (change data capture) or timestamp-based filters.

---

### Q95. BigQuery Schema Design: Denormalization
**Scenario:** Normalized schema: Customers table, Orders table, OrderItems table. Queries JOIN all 3 (slow). Denormalize for performance?

- [x] A. Denormalize: flatten into single Orders table with: [customer_id, customer_name, customer_email, order_id, item_id, item_name, item_qty, item_price]. Single table, no JOINs, faster queries. Trade: data redundancy (customer_name repeated per order). BigQuery handles well (columnar storage, compression minimizes redundancy)
- [ ] B. Normalize always
- [ ] C. No schema
- [ ] D. JOINs are fine

> **Explanation:** BigQuery is columnar (denormalization cheap). Denormalized schema simplifies queries and improves performance.

---

### Q96. BigQuery Parameterized Queries and Security
**Scenario:** Query: SELECT * FROM users WHERE user_id = CAST(variable AS INT). User input: "1 OR 1=1" (SQL injection). Is this safe?

- [x] A. Not safe with string concatenation. Use parameterized queries: DECLARE user_id INT64 = @user_id; SELECT * FROM users WHERE user_id = user_id. BigQuery placeholders (@param) prevent SQL injection. Always use parameters for user input
- [ ] B. String concatenation is safe
- [ ] C. No injection risk
- [ ] D. CAST prevents injection

> **Explanation:** Parameterized queries prevent SQL injection. Always use for user input.

---

### Q97. BigQuery Federated Queries (External Data)
**Scenario:** Data in Cloud Storage (CSV, Parquet) and BigQuery tables. Query both together?

- [x] A. Federated query: CREATE EXTERNAL TABLE ext AS (SELECT * FROM `gs://bucket/data.parquet` FORMAT='PARQUET'). Then: SELECT * FROM bq_table JOIN ext ON bq_table.id = ext.id. BigQuery handles joining external + BQ tables. External queries scan Cloud Storage, slower/costlier than BQ native (but possible)
- [ ] B. Separate queries
- [ ] C. No federation
- [ ] D. Move all to BQ first

> **Explanation:** Federated queries join external and native BigQuery data. Useful for data lakes.

---

### Q98. BigQuery Cost Analysis and Optimization
**Scenario:** Monthly BigQuery bill: $5K (1PB scanned monthly). Optimize without losing functionality?

- [x] A. Audit queries (Cloud Logging → find heavy queries). (1) Partition/cluster tables (reduce scan), (2) use materialized views (cache aggregations), (3) use BI Engine (dashboard cache), (4) reserve slots (cheaper bulk compute), (5) archive old data (delete or move to cold storage). Target: $2K/month (60% savings)
- [ ] B. No optimization
- [ ] C. Accept $5K
- [ ] D. Delete tables

> **Explanation:** BigQuery cost optimization combines partitioning, caching, and slots. Significant savings possible.

---

### Q99. BigQuery Real-time Analytics with Pub/Sub
**Scenario:** Real-time events stream to Pub/Sub. Ingest to BigQuery for live dashboards (update every 1 sec). Dataflow or direct ingestion?

- [x] A. Use Pub/Sub → BigQuery direct write (no Dataflow): Pub/Sub writes directly to BigQuery (streaming inserts). Latency: <10 sec. OR: Pub/Sub → Dataflow → BigQuery (for transforms). Direct write is simpler (no pipeline), lower cost. Choose by complexity (transforms needed?)
- [ ] B. Dataflow required
- [ ] C. No real-time ingestion
- [ ] D. Batch only

> **Explanation:** Pub/Sub direct writes to BigQuery are simple and fast. Dataflow for complex processing.

---

### Q100. BigQuery Compliance: Data Masking and Governance
**Scenario:** PII data (names, emails, phone) in table. Compliance requires: only data team sees raw data; analytics team sees masked data. How?

- [x] A. Column-level security: GRANT roles/analyticViewer on column with DATA_MASKING policy. Masking rules: HASH for PII, NULL for sensitive columns. Analytics team queries, sees only hashed/masked data. Big Query automatically applies masking per user role. Data governance via IAM
- [ ] B. No masking
- [ ] C. Separate tables for each role
- [ ] D. Manual redaction

> **Explanation:** Column-level masking enables compliance. Automatic, role-based data protection.

---

## Domain 6: MLOps & Governance (Q101–Q120)

### Q101. Model Performance Monitoring Dashboard
**Scenario:** Production model makes 10K predictions/day. After 2 weeks, accuracy drops 5% (unnoticed). Design monitoring dashboard to catch this.

- [x] A. Dashboard in Looker/Datalab: (1) prediction volume (trend line), (2) accuracy/precision/recall (daily, weekly rolling average, alert if drop >2%), (3) latency histogram, (4) model predictions vs ground truth comparison (once labels arrive). Use Cloud Logging + Pub/Sub for real-time metric ingestion. Alert email if threshold breached
- [ ] B. No monitoring
- [ ] C. Check manually monthly
- [ ] D. Ignore degradation

> **Explanation:** Real-time monitoring dashboards catch performance regressions. Automated alerts enable quick response.

---

### Q102. Concept Drift vs Data Drift
**Scenario:** Model predicts churners. Data drift: customer age distribution changed (new younger users). Concept drift: churn reasons changed (price sensitivity vs product quality). How do you detect each?

- [x] A. Data drift: compare input distributions (KS test, chi-square). Concept drift: compare true label distributions (churn rate changed). Or: train baseline on old data, compare to new data (regression). Remedies differ: data drift → retrain with new distribution, concept drift → redesign features/labels. Both need monitoring
- [ ] B. Drift is same
- [ ] C. No drift detection
- [ ] D. Ignore both

> **Explanation:** Data drift (input change) vs concept drift (label change) require different handling. Monitor both.

---

### Q103. Model Retraining Automation
**Scenario:** Model needs retraining: (1) monthly schedule, (2) if accuracy drops >3%, (3) if data volume doubles, (4) on demand. How to automate all triggers?

- [x] A. Pipeline with multiple triggers: (1) Cloud Scheduler (cron monthly), (2) custom metric alert (accuracy <threshold) → Cloud Pub/Sub → trigger pipeline, (3) data volume metric (BigQuery) → alert → trigger, (4) API endpoint for manual trigger. All routes to same Vertex AI Pipeline. Retrain only if conditions met (avoid unnecessary retrains)
- [ ] B. Single schedule
- [ ] C. No automation
- [ ] D. Manual retrain only

> **Explanation:** Multi-trigger retraining automation responds to multiple conditions. Reduces manual intervention.

---

### Q104. A/B Testing in Production
**Scenario:** New model v2. You want to measure impact: does v2 improve business metric (e.g., click-through rate)? Design A/B test.

- [x] A. Split users: 50% use v1 (control), 50% use v2 (treatment). Run 2 weeks. Collect metrics: CTR, conversions, latency. Statistical test (t-test): is v2 CTR significantly >v1? If p-value <0.05, v2 wins; upgrade all. Use Vertex AI experiments feature to track, or Analytics tool (e.g., Firebase A/B testing)
- [ ] B. Deploy v2 to 100%
- [ ] C. No A/B testing
- [ ] D. Guess which is better

> **Explanation:** A/B testing validates improvements with statistical rigor. Essential for production ML.

---

### Q105. Feature Store for Governance
**Scenario:** Features: customer_age, credit_score, income. Features owned by different teams (data, risk, finance). How do you manage consistency?

- [x] A. Use Vertex AI Feature Store: (1) register features with ownership (team, SLA, schema), (2) centralized feature repository (single source of truth), (3) versioning (feature v1, v2), (4) governance (who can use? PII access logs), (5) point-in-time correctness (time-travel to get features as of training date). Teams maintain their features; others consume consistently
- [ ] B. No feature store
- [ ] C. Each team owns features
- [ ] D. Duplicate features

> **Explanation:** Feature Store provides centralized governance. Ensures consistency, compliance, and reusability.

---

### Q106. Model Governance and Explainability
**Scenario:** Loan approval model rejects someone. Customer: "Why?" Regulation (GDPR, FCRA) requires explanation. What do you provide?

- [x] A. Model explanation: (1) Explainable AI (SHAP values, integrated gradients) shows which features contributed to rejection (e.g., debt_ratio=0.8, income=0.1), (2) counterfactual: "If income was $60K (vs actual $40K), likely approved", (3) document model card (training data, fairness evaluation, limitations). Provide to customer + regulator
- [ ] B. "Model said no" (vague)
- [ ] C. No explanation
- [ ] D. Trade secret

> **Explanation:** Explainability is both ethical and regulatory requirement. Vertex AI supports multiple explanation methods.

---

### Q107. Model Validation and Certification
**Scenario:** Before prod deploy, model must pass: (1) performance threshold (accuracy >90%), (2) fairness (accuracy gap <5%), (3) latency SLA (<100ms), (4) security review. Process?

- [x] A. Validation checklist: (1) run test set, log accuracy. If >90%, pass, else reject. (2) compute fairness metrics (per-demographic accuracy). If gap <5%, pass. (3) load test endpoint, measure p99 latency. If <100ms, pass. (4) security review: PII handling, model injection, model stealing risks. If all pass, issue certification (signed-off for production deploy). Track in Model Registry
- [ ] B. No validation
- [ ] C. Deploy without checks
- [ ] D. Certification unnecessary

> **Explanation:** Pre-deployment validation ensures quality and safety. Structured checklist and sign-off reduce incidents.

---

### Q108. Model Audit Trail and Reproducibility
**Scenario:** Auditor asks: "Which data was used to train model v2? Who approved deployment? What was evaluated?" Reproduce training?

- [x] A. Audit trail in Vertex ML Metadata: (1) training pipeline execution ID, (2) training data artifact (dataset version, time range), (3) training code version (Git commit), (4) evaluation metrics, (5) approval record (who, when). Reproduce: re-run pipeline with same inputs (deterministic if seeds fixed). All logged automatically
- [ ] B. No audit trail
- [ ] C. Lost history
- [ ] D. Manual notes

> **Explanation:** ML Metadata provides audit trail. Reproducibility and compliance essential.

---

### Q109. Model Versioning in Production
**Scenario:** Deployed models: v1 (live), v2 (staging), v3 (training). Customer reports issue. How do you manage?

- [x] A. Model Registry tracking: v1 (production), v2 (staging, 5% canary traffic), v3 (candidate, 0% traffic). Issue: investigate v1 (root cause: data quality?). If urgent, rollback to v0 (keep prev version). Fix issue, retrain v4, test in staging, promote. Use traffic splitting for safe transitions. All versions in registry with metadata
- [ ] B. No versioning
- [ ] C. Only 1 version
- [ ] D. Delete old versions

> **Explanation:** Model versioning and traffic management enable safe operations. Keep version history.

---

### Q110. Cost Attribution and Chargeback
**Scenario:** ML pipeline costs: $10K/month (training, serving, monitoring). Finance asks: "Which models/teams are expensive?" Chargeback?

- [x] A. Cloud Billing integration: tag each pipeline/endpoint with team (labels: team=finance, model=churn). BigQuery reports: cost by team, model, date. Create chargeback model: charge each team for their ML infrastructure. Encourages cost-conscious behavior. Use commitment-based pricing (reserved slots) to reduce costs overall
- [ ] B. No cost tracking
- [ ] C. Shared cost
- [ ] D. No chargeback

> **Explanation:** Cost attribution enables accountability. Tags and reporting drive efficiency.

---

### Q111. Model Observability: Beyond Metrics
**Scenario:** Model's F1 score is 0.85 (looks good). But customers report: "Predictions are nonsensical (spurious correlations learned)." Metrics don't catch this. What's wrong?

- [x] A. Need observability beyond metrics: (1) log actual predictions + explanations (is SHAP reasonable?), (2) behavioral testing (edge cases: empty input, extreme values), (3) human review samples (spot-check predictions), (4) adversarial testing (can you fool model?). Metrics alone insufficient. Observability reveals systematic issues metrics miss
- [ ] B. Trust metrics only
- [ ] C. No observability
- [ ] D. Ignore customers

> **Explanation:** Observability goes beyond metrics. Includes behavior, explanations, human review. Catches systematic failures.

---

### Q112. Regulatory Compliance: GDPR, CCPA
**Scenario:** Model uses customer data to target ads. GDPR right to deletion (customer asks: "Delete my data"). How do you handle?

- [x] A. Compliance workflow: (1) customer request submitted, (2) identify all data (training data, predictions, logs), (3) delete from all systems (BigQuery, Model Registry, logging), (4) model affected? May need retraining (remove data + retrain). (5) audit log deletion (compliance). Use Vertex AI data governance tools, BigQuery deletion (TRUNCATE), Cloud Audit Logs
- [ ] B. No deletion
- [ ] C. Retain forever
- [ ] D. Ignore GDPR

> **Explanation:** Data privacy regulations require deletion capabilities. ML systems must support data removal without breaking.

---

### Q113. Model Explainability for Different Stakeholders
**Scenario:** Model explanation needed for: (1) data scientist ("debug"), (2) business user ("impact"), (3) regulator ("fairness"). Single explanation insufficient?

- [x] A. Multi-level explanations: (1) Data scientist: SHAP values, feature importance, model weights (technical), (2) Business user: counterfactual ("if age was 5 years older, outcome changes"), impact score ("feature X accounts for 20% of risk"), (3) Regulator: fairness metrics (demographic parity), protected attribute handling, mitigation steps. Tailor explanation to audience
- [ ] B. Single explanation fits all
- [ ] C. No explanation
- [ ] D. Ignore audiences

> **Explanation:** Explainability must be tailored. Technical vs business vs regulatory needs differ.

---

### Q114. Data Privacy in Feature Engineering
**Scenario:** Features: customer age, location, income. These are PII. Feature store stores features. Access control?

- [x] A. Feature store access control: (1) register feature as PII (metadata), (2) grant access selectively (data team: full, analytics: anonymized only), (3) audit logs (who accessed age feature?), (4) encryption at rest/in transit. BigQuery column-level security + IAM control data access. ML pipelines can use features but respect access policies
- [ ] B. No access control
- [ ] C. Everyone sees PII
- [ ] D. Delete features

> **Explanation:** PII protection in feature store is critical. Role-based access + encryption.

---

### Q115. Model Governance Framework
**Scenario:** Organization deploys 50 ML models across teams. Some low-quality (high error rates, unexplained). Need governance framework. What?

- [x] A. Governance framework: (1) model registry (all models tracked, versioned, documented), (2) approval workflow (model passes tests before production), (3) ownership (each model has owner, SLA), (4) monitoring (all models monitored for drift, performance), (5) escalation (if model fails SLA, alert owner, revert if needed), (6) audit (all changes logged). Use Vertex AI Model Registry + Pipelines + custom CRUD API
- [ ] B. No governance
- [ ] C. Decentralized (each team does own thing)
- [ ] D. Governance is overhead

> **Explanation:** Model governance scales ML. Centralized registry, workflows, monitoring prevent chaos.

---

### Q116. MLOps Maturity Model
**Scenario:** Current state: ML models trained by data scientists, deployed manually to servers, no monitoring. Where on maturity scale? Next steps?

- [x] A. Level 1 (ad-hoc): manual everything. Level 2: automated training pipeline (Vertex AI), versioning (Model Registry). Level 3: automated deployment + monitoring (canary, alerts). Level 4: MLOps (continuous retraining, A/B testing, governance). Level 5: AI-driven decisions (self-healing, drift prediction). Current: Level 1. Next: move to Level 2 (automate training + versioning). Then Level 3 (automate deployment + monitoring)
- [ ] B. Already Level 5
- [ ] C. Maturity unnecessary
- [ ] D. No levels

> **Explanation:** MLOps maturity progresses: manual → pipeline → monitoring → governance → intelligent. Assess and improve incrementally.

---

### Q117. Cross-functional Collaboration in ML Projects
**Scenario:** Model underperforms in production (business impact, customer complaints). Data scientist says: "Need more training data." Business: "Too costly." Data engineer: "Data quality issues, not quantity." How to resolve?

- [x] A. Collaboration: (1) align on root cause (root cause analysis: data drift? model design? labeling error?), (2) investigate together (data engineer checks quality, scientist runs diagnostics, business measures impact), (3) agree on solution (retraining with quality fixes? redesign features? collect more data?), (4) prioritize (what's quickest fix?). Use shared dashboard to track progress. Document decisions
- [ ] B. Data scientist decides alone
- [ ] C. Ignore problem
- [ ] D. No collaboration

> **Explanation:** MLOps requires cross-functional alignment. Shared context and communication prevent siloed decisions.

---

### Q118. Knowledge Transfer and Documentation in ML Teams
**Scenario:** Model expert leaves team. Replacement must maintain churn model. What should be documented?

- [x] A. Complete documentation: (1) model card (training data, features, performance, fairness), (2) pipeline code (GitHub, comments), (3) operations guide (how to retrain, deploy, monitor, alert thresholds), (4) incident playbook (if accuracy drops X%, do Y), (5) contact info for feature owners, (6) data schema + assumptions. Store in wiki, linked from Model Registry. New person can take over independently
- [ ] B. Code only
- [ ] C. No documentation
- [ ] D. Oral knowledge only

> **Explanation:** Documentation enables knowledge transfer and reduces bus factor. Critical for team resilience.

---

### Q119. Model Refresh and Deprecation Strategy
**Scenario:** Model v1 is 2 years old. Newer v2 (better performance) deployed. When to retire v1? Deprecation strategy?

- [x] A. Deprecation: (1) announce end-of-life (v1 support ends in 3 months), (2) migrate users to v2 (communicate benefits), (3) monitor v2 performance (ensure no regressions), (4) stop retraining v1, (5) archive after 3 months (keep in Model Registry for history, but no production serving), (6) delete after 1 year (if no compliance requirements). Smooth transition
- [ ] B. Immediately delete v1
- [ ] C. Keep v1 forever
- [ ] D. No deprecation plan

> **Explanation:** Deprecation strategy manages technical debt. Smooth transitions reduce customer impact.

---

### Q120. Ethical AI and Responsible ML
**Scenario:** High-performing model for hiring recommendations. Test: accuracy for white candidates 92%, for Black candidates 75% (hidden bias). Should you deploy?

- [x] A. No; deploy only if fairness validated. Remedies: (1) collect more minority data, (2) use fairness constraints (equalized odds, demographic parity), (3) remove biased features (proxy discrimination?), (4) human-in-the-loop (ML recommends, human decides), (5) audit for discrimination before deploy. Deploy only if fairness gap <5% + human oversight. Ethics first, performance second
- [ ] B. Deploy despite bias
- [ ] C. Performance is only metric
- [ ] D. Ignore fairness

> **Explanation:** Ethical AI requires fairness validation. Responsible deployment prevents discrimination and regulatory issues.

---

## Summary of Domains
- **Domain 1 (Q1–Q20):** BigQuery ML fundamentals (models, evaluation, regularization)
- **Domain 2 (Q21–Q40):** Vertex AI Pipelines (DAG, components, orchestration, monitoring)
- **Domain 3 (Q41–Q60):** Data Engineering/Dataflow/Beam (streaming, windowing, state)
- **Domain 4 (Q61–Q80):** Vertex AI Model Training & Deployment (training, serving, optimization)
- **Domain 5 (Q81–Q100):** Analytics & BigQuery SQL (partitioning, windowing, optimization)
- **Domain 6 (Q101–Q120):** MLOps & Governance (monitoring, fairness, compliance, reproducibility)

---

**End of Q1–Q120**
### Q121. Automated Model Retraining: Handling Staleness (Select all that apply)

**Scenario:** Your model retrains weekly. Between retrains, predictions are served using the stale model. You want to minimize staleness while balancing compute cost. Which approaches help?

- [x] A. Use incremental/online learning to update model weights on streaming new data (rather than full retraining)
- [x] B. Retrain more frequently (e.g., daily) if the domain is fast-changing
- [ ] C. Never retrain; staleness is unavoidable
- [x] D. Monitor model age (time since last training); alert if > threshold

> **Explanation:** (A), (B), and (D) are valid approaches. (C) ignores the problem.

---

### Q122. Monitoring Model Predictions: Logging and Auditing (Select all that apply)

**Scenario:** Your model serves loan approval decisions. You log every prediction (applicant features, model output, decision) to BigQuery for auditing and compliance. Which additional data should you log?

- [x] A. Model version and timestamp (for traceability)
- [x] B. Prediction confidence/probability (for understanding certainty)
- [x] C. Actual outcome (if available, for continuous monitoring)
- [x] D. User ID of the decision-maker (if human-in-the-loop); helps audit override patterns

> **Explanation:** All (A), (B), (C), and (D) are valuable for auditing and continuous improvement.

---

### Q123. Deployment Strategies: Canary Deployments

**Scenario:** You have a new model v2 that shows 2% AUC improvement over v1 in offline tests. You deploy to production via canary: route 5% of traffic to v2, 95% to v1. After 1 day, you observe: v2 latency = 150ms, v1 latency = 100ms. v2 has 5% higher error rate on edge cases. Should you proceed?

- [ ] A. Promote v2 to 100%; the AUC improvement justifies the latency increase
- [x] B. Investigate the edge cases where v2 fails; if they're unacceptable, halt or optimize v2; continue canary at 5% while resolving issues
- [ ] C. Rollback to v1; latency increase is unacceptable
- [x] D. Implement SLO thresholds: promote only if latency < 120ms and error rate < 2%; adjust canary traffic based on metrics

> **Explanation:** (B) and (D) are pragmatic. (A) ignores operational concerns. (C) is premature without deeper investigation.

---

### Q124. Shadow Deployments and Validation (Select all that apply)

**Scenario:** You deploy model v2 in shadow mode: it generates predictions but they're not served to users. Real requests are served by v1; v2's predictions are logged for comparison. After 1 week, you compare v1 vs. v2 predictions on 1M requests. What can you measure?

- [x] A. Prediction divergence: how often do v1 and v2 disagree (e.g., AUC rank difference)?
- [x] B. Latency and resource usage: does v2 meet performance SLOs before traffic is switched?
- [x] C. Edge case detection: identify request types where v2 behaves unexpectedly
- [x] D. User satisfaction (if surveys are available): does v2 impact user experience before serving it?

> **Explanation:** All (A), (B), (C), and (D) are valid shadow deployment metrics.

---

### Q125. A/B Testing in Production: Sample Size and Duration

**Scenario:** You're A/B testing model v2 (variant) vs. v1 (control). You want to detect a 1% improvement in conversion rate with 80% power and 5% significance level. Current baseline conversion = 5%. How many samples per variant are needed?

- [x] A. Use power analysis calculators; for 1% absolute improvement (5% â 6%), roughly 4,000â5,000 samples per variant are needed
- [ ] B. Run the test for 1 week; 1 week is sufficient time for any experiment
- [ ] C. With 1M users/day, run for 5 seconds (sample size is large enough)
- [x] D. Run longer than the minimum sample size to account for temporal effects (day-of-week, seasonality)

> **Explanation:** (A) and (D) are correct. (B) and (C) ignore statistical requirements.

---

### Q126. Handling Model Failures and Rollback (Select all that apply)

**Scenario:** Model v2 is deployed, but after 2 hours, users report poor predictions. You investigate: training data had a bug (NULL values were ignored instead of filled). Should you immediately rollback?

- [x] A. Yes; rollback to v1 to restore service quickly
- [x] B. While rolling back, start a post-mortem: investigate training bug, fix it, retrain on corrected data
- [ ] C. Don't rollback; let users suffer until v3 is ready
- [x] D. Implement automated rollback triggers based on error rate spikes or performance degradation

> **Explanation:** (A), (B), and (D) are correct. (C) is bad practice.

---

### Q127. Model Governance and Approval Workflows

**Scenario:** Your organization requires sign-off before deploying new models to production: data scientist submits â manager approves â compliance reviews â deployment. What are benefits and drawbacks?

- [x] A. Benefits: catches mistakes, ensures accountability; Drawbacks: slower deployment, bottleneck on approvers
- [x] B. Implement automated checks (data validation, performance thresholds) to reduce manual review burden
- [x] C. Use Vertex AI Pipelines with approval gates (conditional execution) to enforce workflows
- [ ] D. Skip approval for high-confidence models; only review low-confidence ones

> **Explanation:** (A), (B), and (C) are correct. (D) introduces risk.

---

### Q128. Continuous Data Quality Monitoring (Select all that apply)

**Scenario:** Your pipeline ingests data daily for training. You want to catch data quality issues early (before they affect models). Which checks should be automated?

- [x] A. Schema validation: columns match expected types and presence
- [x] B. Statistical checks: distribution shifts, outliers, missing value rates
- [x] C. Business logic checks: values within expected ranges (e.g., age 0â120)
- [x] D. Freshness checks: data arrives on schedule; alerts if delays occur

> **Explanation:** All (A), (B), (C), and (D) are standard data quality checks.

---

### Q129. ML System Design: Static vs. Dynamic Training

**Scenario:** Your system trains models using two approaches: (1) static training (weekly batch), (2) dynamic training (continuous online learning on streaming data). Which is better?

- [ ] A. Static training is always better; simpler and more interpretable
- [x] B. Context-dependent: static is more controlled and reproducible; dynamic adapts faster to new patterns but is harder to debug and may suffer from catastrophic forgetting
- [ ] C. Dynamic is always better; continuously learns from new data
- [x] D. Use static training for baseline; add online learning components if drift is significant

> **Explanation:** (B) and (D) are correct. (A) and (C) are too absolute.

---

### Q130. Feature Store Integration with ML Pipelines (Select all that apply)

**Scenario:** Your Vertex AI Pipeline trains a model. It needs features from Vertex AI Feature Store (customer demographics, purchase history). The pipeline should: (1) fetch features from the offline store, (2) ensure point-in-time correctness, (3) version features used for training. How do you integrate?

- [x] A. Use Vertex AI Feature Store API in the pipeline to fetch features with `snapshot_time`; log feature versions in Vertex ML Metadata
- [x] B. Create a BigQuery view that queries the Feature Store's offline tables; train on the view; BigQuery handles versioning
- [ ] C. Export all features to CSV; load manually in the pipeline
- [x] D. Document feature versions in the model's training metadata; enables reproducibility

> **Explanation:** (A), (B), and (D) are valid integration patterns.

---

### Q131. Handling Long-Running Training Jobs and Distributed Training (Select all that apply)

**Scenario:** Your model takes 48 hours to train on a single machine. You submit it to Vertex Training (distributed). The job crashes after 36 hours due to a transient network error. How do you handle this?

- [x] A. Checkpoint frequently (every 2 hours); resume from the latest checkpoint instead of restarting
- [x] B. Use `restart_job_on_worker_restart=true` in Vertex Training configuration for fault tolerance
- [ ] C. Rerun the entire job; checkpoints add overhead
- [x] D. Implement adaptive checkpointing: save more frequently when close to quota limits

> **Explanation:** (A), (B), and (D) are fault-tolerance techniques. (C) ignores efficiency.

---

### Q132. Model Monitoring: Performance vs. Data Drift Monitoring (Select all that apply)

**Scenario:** You monitor model predictions in production daily. You track: (1) model performance on a labeled test set, (2) feature distributions. Over 3 months, feature distribution shifts, but model performance stays constant. Should you retrain?

- [ ] A. Yes; data drift always indicates the model should retrain
- [x] B. Not necessarily; if performance remains acceptable, drift is not critical; monitor performance as the primary signal and use drift as a secondary indicator
- [x] C. Investigate why performance is stable despite drift: is the model robust to this drift, or is the test set unrepresentative?
- [x] D. Use a hybrid approach: retrain if performance degrades OR drift exceeds a threshold (chosen to be conservative)

> **Explanation:** (B), (C), and (D) are correct. (A) is too reactive.

---

### Q133. ML Observability: Logging and Tracing (Select all that apply)

**Scenario:** You deploy a complex ML system with multiple components: preprocessing (Cloud Dataflow), feature retrieval (Vertex Feature Store), model serving (Vertex Endpoints). Prediction latency is high (500ms). How do you debug?

- [x] A. Use Cloud Trace to profile each component; identify which is the bottleneck (e.g., feature retrieval = 300ms)
- [x] B. Log component input/output sizes and durations to Cloud Logging; query logs to find slow steps
- [x] C. Use distributed tracing (OpenTelemetry) to correlate latency across components
- [ ] D. Manually profile by adding print statements and timing code

> **Explanation:** (A), (B), and (C) are production-grade observability approaches. (D) is hacky and doesn't scale.

---

### Q134. Handling Model Versioning and Reproducibility (Select all that apply)

**Scenario:** You deploy model_v1.0 trained on dataset_v1.0 with TensorFlow 2.10. Six months later, you need to reproduce the exact training run (for compliance audit or bug investigation). What should you have documented?

- [x] A. Git commit hash of training code; container image digest; training dataset version (BigQuery table snapshot or dataset checksum)
- [x] B. Hyperparameters (learning rate, batch size, num_epochs) used in training
- [x] C. Random seed and TensorFlow version; enables deterministic reproduction
- [x] D. Full training pipeline configuration and execution time; logs from training

> **Explanation:** All (A), (B), (C), and (D) are essential for reproducibility.

---

### Q135. Cost Optimization in ML Systems (Select all that apply)

**Scenario:** Your ML system incurs high costs: model training ($10K/month), serving (Vertex Endpoints, $8K/month), data storage (BigQuery, $3K/month). Which optimizations are reasonable?

- [x] A. Use preemptible VMs for training (cheaper, but can be interrupted)
- [x] B. Batch predictions offline when possible; use online serving only for latency-sensitive queries
- [x] C. Archive old training data to Cloud Storage (colder tiers); keep only recent data in BigQuery
- [x] D. Right-size model (smaller models = cheaper serving); use knowledge distillation to compress

> **Explanation:** All (A), (B), (C), and (D) are valid cost optimizations.

---

### Q136. ML System Resilience and Graceful Degradation

**Scenario:** Your model serving system has a hard dependency on Vertex AI Feature Store. If Feature Store is down, all predictions fail. How do you improve resilience?

- [x] A. Cache frequently-used features locally (Redis); use cache on Feature Store outage
- [x] B. Implement feature fallbacks: use stale features or pre-computed averages if real-time features are unavailable
- [ ] C. Make Feature Store calls optional; if they fail, skip features and serve predictions anyway
- [x] D. Monitor Feature Store health; proactively fail over to read replicas or alternative data sources

> **Explanation:** (A), (B), and (D) are resilience patterns. (C) leads to poor predictions.

---

### Q137. Model Explainability in Production (Select all that apply)

**Scenario:** Your ML system serves loan approval predictions. Applicants ask why they were denied. You need to provide explanations. Which approaches work at scale?

- [x] A. Log input features and model version; use post-hoc explanations (SHAP, LIME) on-demand for a specific prediction
- [x] B. Pre-compute feature importance during training; include in serving (faster than on-demand)
- [ ] C. Always explain every prediction; compute SHAP values for all 100K daily predictions in real-time
- [x] D. Combine global explanations (feature importance) with instance-level explanations (why this specific prediction?)

> **Explanation:** (A), (B), and (D) are scalable approaches. (C) is computationally expensive; use selectively.

---

### Q138. Incident Response and Automation (Select all that apply)

**Scenario:** Your monitoring detects: model accuracy dropped 10% overnight. You implement automated incident response: (1) alert the team, (2) rollback to previous model, (3) trigger data quality checks, (4) start a retraining job. Which are appropriate to automate?

- [x] A. Alert the team (always automate monitoring)
- [ ] B. Rollback to previous model (always automate; this is critical)
- [x] C. Trigger data quality checks (safe; provides diagnostic info)
- [x] D. Start retraining job (acceptable if configured with guardrails; can also alert for manual approval)

> **Explanation:** (A) and (C) are safe to automate. (B) is risky without context (may mask deeper issues). (D) is acceptable with safeguards.

---

### Q139. Model Governance: Explainability and Fairness Requirements

**Scenario:** Your organization mandates: models must be explainable (SHAP/LIME available), and fairness metrics must be tracked (performance parity across demographics). You're evaluating two approaches: (1) transparent models (linear, decision trees), (2) black-box models (deep learning) + post-hoc explanations. Which is better?

- [x] A. Trade-off: transparent models are inherently explainable but may have lower performance; black-box + post-hoc explanations offer better performance but explanations may not be fully trustworthy
- [ ] B. Always use transparent models; explainability is non-negotiable
- [x] C. Use the approach that meets business requirements: if interpretability is critical, use transparent; if performance is critical, use black-box + post-hoc
- [x] D. Implement both; use transparent models when possible, but allow black-box models when justified

> **Explanation:** (A), (C), and (D) are pragmatic. (B) is too rigid.

---

### Q140. Automated Testing for ML Code (Select all that apply)

**Scenario:** Your ML codebase has unit tests, integration tests, and end-to-end tests. You run tests on every commit via CI/CD. Which tests are most critical?

- [x] A. Unit tests for data preprocessing (ensure transformations are correct)
- [x] B. Integration tests for model inference (ensure model + preprocessing work together)
- [x] C. Contract tests (ensure output schema matches expectations)
- [x] D. Performance regression tests (ensure model inference latency doesn't degrade unexpectedly)

> **Explanation:** All (A), (B), (C), and (D) are critical for ML CI/CD.

---

## Domain 8: Vertex AI Model Monitoring (Q141âQ155)

### Q141. Model Monitoring: Skew vs. Drift Detection

**Scenario:** Your model makes loan approval predictions. Vertex AI Model Monitoring alerts you to a "skew" issue: feature distributions differ between training data and production data. Later, it detects "drift": feature distributions in production changed over time. Which is more urgent?

- [ ] A. Skew is more urgent; the model was trained on different data
- [x] B. Both are important, but indicate different issues: skew suggests training-serving misalignment (data pipeline bugs); drift suggests model aging (needs retraining); prioritize based on performance impact
- [ ] C. Drift is more urgent; skew is expected in production
- [x] D. Investigate root cause: is skew due to a data pipeline bug (repairable) or legitimate distribution change (requires retraining)?

> **Explanation:** (B) and (D) are correct. Both skew and drift matter but require different responses.

---

### Q142. Model Monitoring: Feature Importance and Attribution (Select all that apply)

**Scenario:** Model Monitoring shows a feature, "time_of_day", has increased importance over time (from 5% to 20%). Meanwhile, prediction latency is stable. Should you investigate?

- [x] A. Yes; increased feature importance indicates the model's decision-making has changed; investigate why
- [x] B. This could signal concept drift (the relationship between features and labels has changed)
- [ ] C. Ignore; feature importance naturally varies with data
- [x] D. Check if time-based features are predictive due to external events (e.g., seasonal sales); ensure this isn't spurious

> **Explanation:** (A), (B), and (D) are correct. Monitoring feature importance captures model behavior changes.

---

### Q143. Setting Monitoring Thresholds and Alert Configurations

**Scenario:** You configure Vertex AI Model Monitoring with thresholds: alert if AUC drops >10%, feature drift (KL-divergence) > 1.0, or false positive rate increases >5%. After 1 week, you receive 50 alerts. Most are false alarms due to noise/randomness. How do you calibrate?

- [x] A. Increase thresholds to reduce alert fatigue (e.g., AUC drop >15%, drift > 1.5); monitor via dashboard (not alerts) for trending issues
- [ ] B. Lower thresholds to catch all issues; tolerate false alarms
- [ ] C. Remove monitoring; it generates too many alerts
- [x] D. Use statistical significance testing (e.g., p-value < 0.05) instead of raw thresholds; adjust alert windows (daily vs. weekly)

> **Explanation:** (A) and (D) are calibration approaches. (B) leads to alert fatigue. (C) abandons monitoring.

---

### Q144. Monitoring Data Quality and Preprocessing Issues (Select all that apply)

**Scenario:** Model Monitoring tracks input features' statistical properties. One day, a numeric feature (age) suddenly shows many extreme values (age > 150 years). Should the system alert?

- [x] A. Yes; this is a data quality issue; investigate the data pipeline (bug in parsing or validation)
- [x] B. Use TFDV (TensorFlow Data Validation) to define schema constraints and catch invalid values before they reach the model
- [x] C. Implement automated alerts for schema violations (unexpected data types, ranges)
- [ ] D. Ignore; the model is robust to outliers, so they don't matter

> **Explanation:** (A), (B), and (C) are correct. (D) is risky; outliers may corrupt model training or serving.

---

### Q145. Model Monitoring: Custom Metrics and Business Metrics

**Scenario:** Standard ML metrics (AUC, precision) show the model is performing well. But business metrics (user click-through rate, revenue) have declined. What might be happening?

- [x] A. Model output may not align with business goals; re-examine feature engineering or loss function; optimize for business metrics, not just classification metrics
- [ ] B. Business metrics are unreliable; trust ML metrics instead
- [x] C. External factors (market conditions, competitors) may affect business metrics independent of model quality
- [x] D. Implement monitoring for both ML metrics and business metrics; correlate them to validate model value

> **Explanation:** (A), (C), and (D) are correct. (B) ignores business context.

---

### Q146. Vertex AI Model Monitoring: Baseline Specification

**Scenario:** You set up Model Monitoring and specify the baseline: training data distribution from 6 months ago. This baseline never updates, so comparisons become less meaningful as production data naturally evolves. How should baselines be managed?

- [ ] A. Fix baselines permanently; they represent the "correct" state
- [x] B. Update baselines periodically (e.g., monthly or when data significantly changes) to reflect current expectations; use versioned baselines to track evolution
- [ ] C. Use a separate baseline per day-of-week to account for weekly patterns
- [x] D. Allow multiple baselines: training baseline (for initial validation), recent baseline (for trend detection)

> **Explanation:** (B) and (D) are pragmatic. (A) is too rigid. (C) may over-complicate.

---

### Q147. Monitoring for Adversarial Examples and Poisoning Attacks

**Scenario:** Your model serves recommendations. An attacker crafts adversarial inputs (e.g., product descriptions designed to trigger specific recommendations). How do you detect and mitigate?

- [x] A. Monitor for anomalous input patterns (outliers in feature space); use methods like LOF (Local Outlier Factor) or isolation forests
- [x] B. Track prediction changes on deliberately perturbed inputs (adversarial robustness testing); large changes indicate vulnerability
- [ ] C. Assume models are robust; attacks are unlikely
- [x] D. Implement input validation (range checks, schema validation) to reject suspicious inputs

> **Explanation:** (A), (B), and (D) are valid. (C) is dangerous.

---

### Q148. Monitoring Model Serving Latency and Resources (Select all that apply)

**Scenario:** You monitor model serving latency. P95 latency = 150ms (SLO = 100ms). You identify bottlenecks: feature retrieval (80ms), model inference (60ms), post-processing (10ms). How do you optimize?

- [x] A. Focus on feature retrieval; it's the bottleneck; cache features or pre-compute
- [x] B. Profile inference latency by layer; may find slow layers worth optimizing or quantizing
- [x] C. Use resource monitoring (CPU/memory) to identify if the system is CPU-bound, memory-bound, or I/O-bound
- [ ] D. Increase machine size to meet SLO; if SLO is hard, scaling resources is justified

> **Explanation:** (A), (B), and (C) are diagnostic approaches. (D) is a last resort after optimization.

---

### Q149. Model Monitoring and Retraining Triggering (Select all that apply)

**Scenario:** Model Monitoring detects: (1) AUC on recent data = 0.75 (down from 0.82 at training), (2) feature drift score = 0.8 (moderate), (3) no formal ground truth labels available yet (labels arrive with 2-week delay). Should you retrain?

- [x] A. Retrain if AUC drop is statistically significant and persists; use proxy labels (e.g., user feedback) for faster validation before ground truth arrives
- [x] B. Combine signals: AUC drop + feature drift together suggest retraining is warranted; any single signal alone might be noise
- [ ] C. Wait for ground truth labels (2 weeks) before deciding; cannot act without them
- [x] D. Retrain on fresh data using the same hyperparameters; if performance improves, promote the new model

> **Explanation:** (A), (B), and (D) are pragmatic without ground truth. (C) is too conservative.

---

### Q150. Monitoring for Model Degradation: Gradual vs. Sudden (Select all that apply)

**Scenario:** Model Monitoring tracks AUC over 6 months: Month 1â3 (AUC=0.85), Month 4 (AUC=0.82), Month 5 (AUC=0.80), Month 6 (AUC=0.78). This is gradual degradation. Month 6 Day 15 (AUC=0.55, sudden drop). How do you respond?

- [x] A. Gradual degradation: retrain on recent data; this is expected model aging
- [x] B. Sudden drop: immediately investigate; check for data pipeline bugs, label leakage, or external disruptions
- [x] C. For sudden drops, consider automatic rollback to previous model; for gradual degradation, automatic retraining is acceptable
- [ ] D. Both gradual and sudden degradation require the same response

> **Explanation:** (A), (B), and (C) are correct; they require different urgency levels. (D) oversimplifies.

---

### Q151. Monitoring Cross-Dataset Performance: Train vs. Validation vs. Production

**Scenario:** Model performance differs across datasets: training set AUC=0.92, validation set (from 1 month ago) AUC=0.85, production (today) AUC=0.78. What does this pattern suggest?

- [x] A. Overfitting on training data; generalization gap between train and validation; additional degradation in production suggests data/concept drift
- [ ] B. Model is broken; all metrics should be equal
- [x] C. Retrain on recent data; the validation set is stale (1 month old)
- [x] D. Use production data (if labels become available) to update the validation set; re-validate the model

> **Explanation:** (A), (C), and (D) are correct. (B) ignores typical ML reality.

---

### Q152. Model Monitoring: Fairness and Demographic Parity (Select all that apply)

**Scenario:** Vertex AI Model Monitoring tracks fairness metrics. You observe: Group A (male): recall=0.90, precision=0.80. Group B (female): recall=0.70, precision=0.75. There's a significant difference in recall. Is this a fairness violation?

- [x] A. Potentially yes; Group B has lower recall (may miss important cases for that group); investigate whether this is acceptable or indicates bias
- [ ] B. No; precision is similar, so the model is fair
- [x] C. Check if the differences are due to data quality (Group B may have fewer training examples or noisier labels)
- [x] D. Define fairness requirements for this use case (equal recall, equal precision, or demographic parity?) and monitor accordingly

> **Explanation:** (A), (C), and (D) are correct. (B) ignores the recall disparity.

---

### Q153. Model Monitoring: Integrating with Alerting Systems (Select all that apply)

**Scenario:** Vertex AI Model Monitoring detects an issue. You want to alert the team via: (1) Slack notification, (2) PagerDuty for high-severity issues, (3) Cloud Logging. How do you set this up?

- [x] A. Use Cloud Monitoring to capture Vertex AI Model Monitoring metrics; configure alerting policies to trigger Slack/PagerDuty notifications
- [x] B. Use Cloud Logging to log monitoring results; downstream systems consume logs and route alerts
- [x] C. Integrate with incident management tools (PagerDuty, Opsgenie) for on-call escalation on critical issues
- [ ] D. Manually check Vertex AI Model Monitoring dashboards daily

> **Explanation:** (A), (B), and (C) are automated. (D) is manual and misses real-time issues.

---

### Q154. Model Monitoring: Handling Seasonal and Cyclical Patterns

**Scenario:** Your model forecasts retail sales. Sales naturally spike in Q4 (holidays). Model Monitoring naively compares Q4 production data to the training baseline (which included historical Q4 data). Should it alert on distribution shift?

- [ ] A. Yes; always alert on distribution shift regardless of season
- [x] B. No; expected seasonal patterns should not trigger alerts; configure monitoring to account for seasonality (e.g., use Q4-specific baselines)
- [x] C. Implement seasonal baselines: separate baselines for Q1âQ4; compare production data to the corresponding seasonal baseline
- [x] D. Monitor for anomalous seasonality (e.g., Q4 spike is smaller than historical average by 2 standard deviations); this may indicate real problems

> **Explanation:** (B), (C), and (D) are correct; ignore expected seasonality, but alert on anomalous deviations. (A) generates false alarms.

---

### Q155. Model Monitoring and Continuous Improvement Feedback Loops (Select all that apply)

**Scenario:** Model Monitoring generates insights: features X and Y have high drift; model performance degrades on a specific user segment. How do you close the feedback loop to improve the system?

- [x] A. Feed monitoring insights to the data science team; prioritize retraining if drift is severe
- [x] B. Log monitoring alerts in a ticketing system (Jira); create backlog items for investigation
- [x] C. Automate retraining when specific thresholds are crossed (e.g., drift > 1.5)
- [x] D. Use A/B testing to validate that retraining actually improves business metrics before full rollout

> **Explanation:** All (A), (B), (C), and (D) close feedback loops systematically.

---

## Domain 9: Model Deployment & Serving (Q156âQ175)

### Q156. Batch vs. Online Serving: Trade-offs

**Scenario:** You're deploying a model to predict customer purchase propensity. Use case 1: nightly batch predictions for marketing campaigns (48-hour freshness is acceptable). Use case 2: real-time predictions at checkout (sub-100ms latency required). Which serving approach for each?

- [x] A. Batch predictions via Dataflow/BigQuery for use case 1; online serving via Vertex AI Endpoints for use case 2
- [ ] B. Use online serving for both; batch is outdated
- [ ] C. Use batch for both; simpler infrastructure
- [x] D. Batch is cost-effective for high volume, low freshness tolerance; online is expensive but provides low latency

> **Explanation:** (A) and (D) are correct. (B) and (C) are too one-dimensional.

---

### Q157. Vertex AI Endpoints: Autoscaling Configuration (Select all that apply)

**Scenario:** You deploy a model to a Vertex AI Endpoint. Traffic varies: 100 QPS at off-peak, 10K QPS at peak. You configure autoscaling. What should you set?

- [x] A. `min_replica_count=1` (handle off-peak); `max_replica_count=100` (handle peak)
- [x] B. `target_cpu_utilization=0.7` or `target_request_count_per_replica` to decide scaling up
- [ ] C. Fixed replicas=50; autoscaling is overhead
- [x] D. Monitor scaling latency; set scale-up threshold conservatively to avoid serving degradation during ramp-up

> **Explanation:** (A), (B), and (D) are correct configurations. (C) wastes resources at off-peak.

---

### Q158. GPU Selection for Model Serving (Select all that apply)

**Scenario:** Your model is a 500M-parameter transformer. You're evaluating GPUs for serving: NVIDIA V100, T4, A100. You want to minimize cost while keeping latency <50ms at 100 QPS. Which should you evaluate?

- [x] A. T4 is cheapest; test latency first; if <50ms, use T4
- [x] B. V100 and A100 are more expensive but faster; use if T4 latency exceeds SLO
- [x] C. Profile model with different batch sizes (1, 4, 8) on each GPU type; choose based on throughput-per-dollar
- [ ] D. Always use A100; it's the fastest

> **Explanation:** (A), (B), and (C) are pragmatic. (D) ignores cost optimization.

---

### Q159. Model Versioning and Serving Multiple Versions (Select all that apply)

**Scenario:** You deploy model v2 alongside v1 in a Vertex AI Endpoint. You route 80% traffic to v1, 20% to v2 (canary). After 1 week, you fully promote v2 and retire v1. How do you manage versioning?

- [x] A. Use Vertex AI Endpoints' traffic splitting feature to route requests by percentage
- [x] B. Tag models in the Vertex AI Model Registry with environment labels (v1:prod, v2:canary)
- [x] C. Maintain model lineage: v2 was trained on dataset_v1.2, hyperparameters X, with performance metrics Y
- [x] D. After full promotion, retain v1 as a rollback version for at least 30 days

> **Explanation:** All (A), (B), (C), and (D) are best practices for version management.

---

### Q160. Serverless Model Serving: Cloud Run and Cloud Functions

**Scenario:** You have a lightweight model (100MB, inference in 10ms). You need auto-scaling to handle variable load. Should you use Vertex AI Endpoints or Cloud Run?

- [x] A. Cloud Run is suitable for lightweight models with low latency; cheaper than Vertex Endpoints for occasional traffic
- [ ] B. Vertex Endpoints is always better; Cloud Run is too simple
- [x] C. Evaluate: Cloud Run has ~100ms cold-start overhead; if your model latency is 10ms, total latency â 110ms (may exceed SLO)
- [x] D. Use Vertex Endpoints if latency SLO is tight (<100ms); use Cloud Run for looser SLOs

> **Explanation:** (A), (C), and (D) are correct trade-offs. (B) is too absolute.

---

### Q161. TensorFlow Lite and Mobile Model Serving

**Scenario:** You want to run a model on mobile devices (iOS/Android apps). Your TensorFlow model is 200MB. You need to optimize for size and latency. What should you do?

- [x] A. Convert to TensorFlow Lite format; it's ~75% smaller (150MB)
- [x] B. Apply quantization (INT8): reduces size further and speeds up inference on mobile
- [ ] C. Run inference on the backend; send predictions to the device (costs per-request, higher latency)
- [x] D. Include both: TFLite + quantization + pruning if latency is still problematic

> **Explanation:** (A), (B), and (D) are mobile optimization techniques. (C) defeats the purpose of on-device serving.

---

### Q162. Model Serving and Feature Latency: Feature Store Bottlenecks (Select all that apply)

**Scenario:** Model serving latency is 500ms: feature retrieval (400ms), inference (50ms), post-processing (50ms). Feature retrieval is the bottleneck. How do you optimize?

- [x] A. Cache frequently accessed features in Redis; use Feature Store as a fallback
- [x] B. Pre-fetch features in batch mode before serving (if applicable); reduces per-request latency
- [x] C. Use an in-process feature cache (updated periodically); trade memory for latency
- [x] D. Optimize Feature Store queries: add indexes, use point lookups instead of range queries

> **Explanation:** All (A), (B), (C), and (D) are optimization techniques.

---

### Q163. Load Balancing and Traffic Routing (Select all that apply)

**Scenario:** You deploy a model to multiple Vertex AI Endpoints (in different regions) for geographic redundancy. You want to route requests to the nearest endpoint. Which approaches work?

- [x] A. Use Cloud Load Balancer with geo-routing; routes to nearest region
- [x] B. Use Cloud CDN in front of endpoints; caches responses, reduces latency
- [x] C. Implement custom routing logic: client determines nearest endpoint and calls directly
- [ ] D. Deploy to a single region; multi-region is unnecessary complexity

> **Explanation:** (A), (B), and (C) are valid. (D) ignores redundancy benefits.

---

### Q164. A/B Testing in Production Serving (Select all that apply)

**Scenario:** You deploy model v2 (new) alongside v1 (old) in production. You want to A/B test for 1 week before full promotion. 5% of traffic goes to v2. How do you implement?

- [x] A. Use Vertex Endpoints' traffic splitting; route requests probabilistically (5% v2, 95% v1)
- [x] B. Log request IDs and variant (v1 or v2) to BigQuery; enable statistical analysis post-hoc
- [ ] C. Deterministic routing: route user_id % 20 == 0 to v2; ensures consistency
- [x] D. Analyze metrics: latency, errors, business outcomes; decide to promote or rollback after 1 week

> **Explanation:** (A), (B), and (D) are correct. (C) is less flexible; probabilistic routing is better for uniform sampling.

---

### Q165. Model Serving with Request Validation (Select all that apply)

**Scenario:** Your model serves loan approvals. Invalid inputs (missing fields, wrong types) crash the model. How do you add validation?

- [x] A. Define a request schema (JSON Schema or Protocol Buffers); validate before inference
- [x] B. Return error responses for invalid inputs (HTTP 400); log for debugging
- [ ] C. Allow the model to handle invalid inputs gracefully; it's robust
- [x] D. Use Vertex AI Endpoints' built-in request validation features (if available)

> **Explanation:** (A), (B), and (D) are correct. (C) is risky; models often crash on invalid inputs.

---

### Q166. Model Serving and Privacy: Data Redaction (Select all that apply)

**Scenario:** Your model makes predictions based on sensitive customer data (email, SSN, credit card). You want to: (1) serve predictions, (2) log requests for debugging, (3) not log sensitive data. How?

- [x] A. Redact sensitive fields before logging; keep only non-sensitive features (e.g., age, income)
- [x] B. Use separate audit logs with restricted access (encryption, IAM); limited to authorized personnel
- [x] C. Hash sensitive fields; log hashes instead of raw values (enables debugging without exposing data)
- [ ] D. Log everything; privacy is the application's responsibility

> **Explanation:** (A), (B), and (C) are privacy-preserving logging. (D) violates best practices.

---

### Q167. Model Serving and Explainability: Real-Time Explanations (Select all that apply)

**Scenario:** Your model serves predictions in real-time. Users request explanations (e.g., "Why was I rejected?"). Computing SHAP explanations on-demand takes 5 seconds (SLO = 100ms). How do you provide explanations without violating latency SLO?

- [x] A. Pre-compute explanations offline for common requests; cache and serve cached explanations
- [x] B. Serve predictions fast (<100ms); generate explanations asynchronously in the background; return to user later (e.g., via email)
- [x] C. Use approximation methods (LIME instead of SHAP) for faster computation
- [ ] D. Don't provide explanations; they're too expensive

> **Explanation:** (A), (B), and (C) are pragmatic. (D) ignores user expectations.

---

### Q168. Monitoring Model Serving: Request/Response Logging (Select all that apply)

**Scenario:** You log all requests and responses to Vertex AI Endpoints. You collect: input features, prediction, timestamp, request_id. For 10K QPS, you generate ~10GB/day of logs. How do you manage storage and analysis?

- [x] A. Stream logs to BigQuery for efficient querying; partition by date and model version
- [x] B. Use Cloud Logging with appropriate retention policies (e.g., delete logs after 30 days)
- [x] C. Sample logs (e.g., 10% of requests) for cost savings; ensure sampling is representative
- [x] D. Compress logs; use columnar formats (Parquet) for storage efficiency

> **Explanation:** All (A), (B), (C), and (D) are cost-optimization techniques for logging.

---

### Q169. Model Serving and Graceful Degradation (Select all that apply)

**Scenario:** Your model serves loan approvals at 10K QPS. Suddenly, a dependency (Feature Store) becomes unavailable. Your service should degrade gracefully without crashing. What are mitigation strategies?

- [x] A. Use cached features (from previous requests); serve predictions with slightly stale features
- [x] B. Use default feature values (pre-computed averages) if live features are unavailable
- [x] C. Return a timeout error to users; do not serve predictions with degraded inputs
- [x] D. Automatically switch to a simpler, faster model that doesn't require the unavailable dependency

> **Explanation:** (A), (B), and (D) enable graceful degradation. (C) is a valid fallback if degradation is unacceptable.

---

### Q170. Containerization and Model Serving: Docker and Kubernetes (Select all that apply)

**Scenario:** You're serving a model using Vertex AI Endpoints. Internally, Vertex Endpoints run your model in a containerized environment. You need to customize the serving container (e.g., install custom libraries). How?

- [x] A. Create a Dockerfile with base image (Python 3.9) and custom dependencies; push to Container Registry
- [x] B. Build custom container image; specify in the model deployment config; Vertex deploys the custom container
- [ ] C. Modify Vertex's built-in container (not possible; use custom container instead)
- [x] D. Test the container locally (Docker run) before deploying to Vertex

> **Explanation:** (A), (B), and (D) are correct. (C) is not possible; always use custom images.

---

### Q171. Model Serving and Request Batching (Select all that apply)

**Scenario:** Your model processes individual requests, but inference is faster with batching (e.g., batch_size=32 is 4x faster than 32Ã batch_size=1). You receive requests asynchronously. How do you implement batching?

- [x] A. Use a request queue (e.g., Cloud Tasks or Pub/Sub); batch requests before inference
- [x] B. Implement request buffering: wait up to 100ms for requests to accumulate before inferencing (trade latency for throughput)
- [ ] C. Batch all requests; no single-request serving
- [x] D. Adaptive batching: vary batch size based on queue depth (larger batch if queue is deep)

> **Explanation:** (A), (B), and (D) are batching strategies. (C) is too rigid; single-request serving should remain available.

---

### Q172. Model Serving: Canary Deployments and Gradual Rollouts (Select all that apply)

**Scenario:** You deploy model v2 via canary: 5% traffic initially. After observing stability, you increase to 25%, then 50%, then 100%. At 25%, you notice: v2 latency â 120ms vs. v1 â 100ms. Should you halt?

- [x] A. Investigate why v2 is slower; may be resolvable (e.g., optimize dependencies)
- [ ] B. Halt rollout; latency increase is unacceptable
- [x] C. If 20ms is acceptable, continue rollout; if not, optimize v2 before proceeding
- [x] D. Establish latency SLO; only promote if v2 meets SLO

> **Explanation:** (A), (C), and (D) are pragmatic. (B) is premature without understanding the cause.

---

### Q173. Model Serving: Hot Reloads and Zero-Downtime Deployments

**Scenario:** You deploy a new model version to a Vertex Endpoint (100K QPS). You need to avoid dropping requests during the update. How?

- [x] A. Use rolling updates: gradually move traffic from old to new model; no requests are dropped
- [ ] B. Stop the endpoint, update, restart (results in downtime)
- [x] C. Use Vertex Endpoints' update mechanism; it handles in-place upgrades without downtime
- [x] D. Implement health checks; only route traffic to healthy replicas during update

> **Explanation:** (A), (C), and (D) are zero-downtime strategies. (B) causes downtime.

---

### Q174. Model Serving and Prediction Caching (Select all that apply)

**Scenario:** Your model serves recommendations. The same user asks the same query multiple times per day. You want to cache predictions. How?

- [x] A. Cache based on input features (hashed feature vector) â prediction; use Redis for fast lookups
- [x] B. Set cache TTL (time-to-live) based on domain (e.g., 1 hour for recommendations; stale is acceptable)
- [ ] C. Cache all predictions forever; they never change
- [x] D. Invalidate cache when model version changes (v1 predictions should not be served after v2 is deployed)

> **Explanation:** (A), (B), and (D) are caching best practices. (C) ignores staleness and model updates.

---

### Q175. Model Serving: Multi-Model Endpoints (Select all that apply)

**Scenario:** You have 3 related models: model_A (CTR prediction), model_B (revenue prediction), model_C (brand affinity). You want to serve all 3 from a single endpoint. How?

- [x] A. Deploy all models to the same Vertex Endpoint; use a routing layer to direct requests to the appropriate model
- [x] B. Assign each model to a route (e.g., `/predict/ctr`, `/predict/revenue`, `/predict/affinity`); clients specify which model to call
- [ ] C. Require clients to choose a model before creating an endpoint (static routing)
- [x] D. Use a meta-model to decide which model to use (if applicable); route dynamically

> **Explanation:** (A), (B), and (D) are flexible multi-model serving approaches.

---

## Domain 10: Hyperparameter Tuning (Q176âQ185)

### Q176. Hyperparameter Tuning Strategies: Grid vs. Random vs. Bayesian Search

**Scenario:** You're tuning learning rate (1e-4 to 1e-2), batch size (16â256), and dropout (0.0â0.5). You have 100 GPU hours of compute budget. Which search strategy should you use?

- [ ] A. Grid search: exhaustively try all combinations (too expensive; millions of combinations)
- [x] B. Random search: sample 50â100 random configurations; often finds good hyperparameters efficiently
- [x] C. Bayesian search (e.g., Vertex AI Vizier): models the objective function; prioritizes promising regions; typically outperforms random
- [x] D. Start with random; if promising patterns emerge, use Bayesian for refinement

> **Explanation:** (B), (C), and (D) are practical. (A) is computationally prohibitive.

---

### Q177. Vertex AI Vizier for Hyperparameter Tuning (Select all that apply)

**Scenario:** You use Vertex AI Vizier to tune a model. You specify: search space (learning_rate â [1e-5, 1e-1], num_layers â [2, 10]), objective (maximize AUC), and 200 trials budget. What are Vizier's responsibilities?

- [x] A. Suggest hyperparameter configurations based on Bayesian optimization; learn from previous trials to improve suggestions
- [x] B. You run training for each suggested configuration; report results (AUC) back to Vizier
- [ ] C. Vizier runs training automatically; you only need to define the search space
- [x] D. Vizier outputs the top-N configurations; you select the best for production

> **Explanation:** (A), (B), and (D) are correct. (C) is incorrect; you must run training for each trial.

---

### Q178. Early Stopping in Hyperparameter Tuning

**Scenario:** You're tuning hyperparameters using Vizier. For each trial, training takes 30 minutes. You've run 100 trials (50 hours of compute). Some trials plateau early (no improvement after epoch 10). Should you implement early stopping?

- [x] A. Yes; stop training if validation loss doesn't improve for 3 consecutive epochs; redeploy compute to new trials
- [ ] B. No; all trials should run the full duration for fairness
- [x] C. Early stopping frees up compute; allows more trials in the same budget
- [x] D. Be careful: aggressive early stopping may discard configurations that improve later; use moderate patience (5â10 epochs)

> **Explanation:** (A), (C), and (D) are correct. (B) wastes compute on plateaued trials.

---

### Q179. Importance-Based Hyperparameter Selection (Select all that apply)

**Scenario:** After tuning with Vizier, you want to understand which hyperparameters had the most impact on model performance. Vizier outputs: learning_rate importance=0.6, batch_size importance=0.3, dropout importance=0.1. How should you use this?

- [x] A. Focus future tuning efforts on high-importance hyperparameters (learning_rate); less critical ones can be held fixed
- [x] B. For deployment, establish sensitivity: if learning_rate changes by 10%, does AUC change significantly? If so, keep it fixed
- [ ] C. Importance scores are final; don't retune high-importance hyperparameters
- [x] D. Different models/datasets may have different importance scores; repeat analysis if domain shifts

> **Explanation:** (A), (B), and (D) are correct. (C) is too rigid.

---

### Q180. Learning Rate Scheduling During Tuning (Select all that apply)

**Scenario:** You're tuning a model with Vizier. For each trial, you use a fixed learning rate (suggested by Vizier) for the entire training run. But you know that learning rate schedules (warmup + decay) improve convergence. Should you apply scheduling?

- [x] A. Yes; use a learning rate schedule for each trial; report final AUC to Vizier
- [x] B. Apply the same schedule (e.g., warmup for 10% of steps, linear decay) to all trials for fairness
- [ ] C. Let Vizier suggest learning rate schedules as hyperparameters (too many combinations)
- [x] D. If using a fixed schedule, ensure it doesn't confound Vizier's tuning (the schedule itself becomes a hidden hyperparameter)

> **Explanation:** (A), (B), and (D) are correct. (C) is impractical.

---

### Q181. Hyperparameter Tuning for Different Model Types (Select all that apply)

**Scenario:** You're comparing two model types: Linear Regression and Gradient Boosted Trees. Tuning spaces differ: Linear has 3 hyperparameters (regularization, learning_rate, solver), Boosted Trees has 8 (num_trees, tree_depth, learning_rate, subsampling, ...). Should you allocate the same tuning budget?

- [x] A. Allocate more tuning budget to Boosted Trees (more hyperparameters = larger search space)
- [x] B. Use importance analysis post-hoc; some hyperparameters may be less critical (can reduce search space)
- [ ] C. Allocate equal budget; the models should be treated fairly
- [x] D. Tune the simpler model (Linear) first; baseline results help calibrate the search for the more complex model

> **Explanation:** (A), (B), and (D) are pragmatic. (C) ignores complexity.

---

### Q182. Transfer Learning and Hyperparameter Reuse (Select all that apply)

**Scenario:** You fine-tune a pre-trained transformer on a new task. You had previously tuned hyperparameters on the original task (learning_rate=2e-5, warmup_steps=1000). Should you reuse these for fine-tuning?

- [x] A. Hyperparameters from the original task are a reasonable starting point; may still be optimal for fine-tuning
- [x] B. The new task may require different hyperparameters; tune on the new task to be sure
- [ ] C. Always retune from scratch; prior hyperparameters are task-specific
- [x] D. Perform ablation: train with original hyperparameters vs. newly tuned; compare to quantify the benefit

> **Explanation:** (A), (B), and (D) are correct. (C) is overly cautious; prior hyperparameters are useful starting points.

---

### Q183. Hyperparameter Tuning for Distributed Training (Select all that apply)

**Scenario:** You tune a model for single-GPU training. You find optimal learning_rate=1e-3, batch_size=32. Now you scale to 4 GPUs with data parallelism. Should you retune?

- [x] A. Yes; with larger effective batch size (4Ã32=128), learning_rate may need to increase (larger batch = noisier gradients)
- [ ] B. No; hyperparameters should be independent of parallelism
- [x] C. Use batch size scaling heuristic: lr_new â lr_old Ã (new_batch_size / old_batch_size)^0.5
- [x] D. Validate with a quick tune: test a few learning rates on the distributed setup; pick the best

> **Explanation:** (A), (C), and (D) are correct. (B) is incorrect; batch size effects learning rate.

---

### Q184. Hyperparameter Tuning and Model Interpretability Trade-off (Select all that apply)

**Scenario:** You tune a model for accuracy. The best hyperparameters result in: complex tree ensemble (1000 trees, depth=15), which is hard to interpret. An alternative (100 trees, depth=7) has 2% lower AUC but is simpler. Which should you choose for production?

- [x] A. Depends on the use case: if interpretability is critical (e.g., lending), sacrifice 2% accuracy for simplicity
- [x] B. Use the complex model + post-hoc explanations (SHAP) for interpretability
- [ ] C. Always choose the highest-accuracy model; interpretability is a luxury
- [x] D. Consider business cost: does 2% accuracy loss translate to significant revenue loss? If not, prefer simplicity

> **Explanation:** (A), (B), and (D) are pragmatic. (C) ignores business context.

---

### Q185. Hyperparameter Tuning: Ensemble of Models with Different Hyperparameters (Select all that apply)

**Scenario:** You tune 100 models with Vizier. The top-5 models have similar performance (AUC within 0.5%). Should you create an ensemble of the top-5 instead of selecting one?

- [x] A. Yes; ensembles of diverse models often outperform single models; the top-5 are likely diverse (different hyperparameters)
- [ ] B. No; ensembles don't improve if models are similar
- [x] C. Validate: train ensemble on the top-5; test on held-out data; compare to best individual model
- [x] D. Ensemble is more expensive to serve (5Ã inference calls); weigh cost vs. performance gain

> **Explanation:** (A), (C), and (D) are correct. (B) is incorrect; diverse top-5 models can ensemble well.

---

## Domain 11: GenAI, LLMs & RAG (Q186âQ210)

### Q186. Fine-Tuning vs. Prompt Engineering for LLMs

**Scenario:** You want to adapt a foundation model (e.g., PaLM 2) to your domain (e.g., legal contract analysis). You have two approaches: (1) prompt engineering (craft detailed prompts with examples), (2) fine-tuning (train on domain-specific data). Which should you try first?

- [x] A. Prompt engineering first; it's fast and free; if it achieves acceptable performance, use it
- [ ] B. Always fine-tune; it's more robust than prompt engineering
- [x] C. If prompt engineering is insufficient (<70% accuracy), proceed to fine-tuning
- [x] D. Fine-tuning on 1000 examples may cost $100â1000 and time; weigh against prompt engineering ROI

> **Explanation:** (A), (C), and (D) are pragmatic. (B) oversimplifies.

---

### Q187. Retrieval-Augmented Generation (RAG) Architecture (Select all that apply)

**Scenario:** You build a customer support chatbot that must answer questions about your company's products. You implement RAG: 1) retrieve relevant documents from a knowledge base, 2) augment the prompt with retrieved content, 3) generate a response. Which components are essential?

- [x] A. Vector embeddings: embed documents and queries into a common space; retrieve similar documents via vector similarity
- [x] B. Retriever (e.g., Chroma, Pinecone): efficient data structure for nearest-neighbor search
- [x] C. LLM (e.g., Vertex AI Generative API): reads augmented prompt and generates response
- [x] D. Ranking/reranking: optionally re-rank retrieved documents by relevance before passing to LLM

> **Explanation:** All (A), (B), (C), and (D) are components of a robust RAG system.

---

### Q188. Chunking Strategies for RAG (Select all that apply)

**Scenario:** You have a 200-page product manual. You want to chunk it into pieces for RAG. If chunks are too small (50 tokens), each chunk lacks context; if too large (2000 tokens), retrieval is noisy. What are reasonable strategies?

- [x] A. Fixed-size chunks (e.g., 512 tokens) with overlap (e.g., 50-token overlap) to maintain context
- [x] B. Semantic chunking: split at logical boundaries (e.g., section breaks, paragraph ends)
- [x] C. Query-based chunking: chunk differently based on expected query patterns
- [ ] D. Chunk by sentence; one sentence per chunk (too fragmented)

> **Explanation:** (A), (B), and (C) are practical chunking strategies. (D) is too granular.

---

### Q189. Embedding Models for RAG (Select all that apply)

**Scenario:** You're choosing an embedding model for RAG: Vertex AI Text Embedding API (768-dim), Sentence-Transformers (384-dim), or custom-trained (100-dim). Which factors matter?

- [x] A. Embedding quality: higher quality captures semantic meaning better; Text Embedding API is often highest quality but slower/more expensive
- [x] B. Dimensionality: higher-dim embeddings are more expressive but slower to search and require more storage
- [x] C. Latency: if retrieval must be <100ms, lightweight embeddings may be required
- [x] D. Domain-specificity: if your domain is specialized (e.g., biomedical), fine-tuned embeddings may outperform general-purpose

> **Explanation:** All (A), (B), (C), and (D) are trade-offs in embedding selection.

---

### Q190. Evaluating RAG System Quality (Select all that apply)

**Scenario:** You've built a RAG system for Q&A. You want to evaluate: (1) retrieval quality (are relevant documents retrieved?), (2) generation quality (does the LLM generate accurate answers?). What metrics apply?

- [x] A. Retrieval metrics: NDCG (rank-weighted recall), Recall@k (is the relevant doc in top-k?), Precision@k
- [x] B. Generation metrics: BLEU, ROUGE (lexical overlap with ground-truth answers); may be insufficient (multiple correct answers exist)
- [x] C. End-to-end metrics: user satisfaction (Likert scale), correctness (human annotation)
- [x] D. Latency and cost: retrieval + LLM inference latency; embedding and LLM API costs

> **Explanation:** All (A), (B), (C), and (D) are relevant evaluation dimensions.

---

### Q191. LLM Fine-Tuning Methods: Full Fine-Tuning vs. LoRA (Select all that apply)

**Scenario:** You fine-tune a 7B-parameter model on domain data. Full fine-tuning requires 40GB GPU memory. You can't afford this. What are alternatives?

- [x] A. LoRA (Low-Rank Adaptation): add trainable low-rank matrices; freeze the base model; drastically reduces memory (requires only â1GB)
- [x] B. QLoRA: quantize the model (INT4) + LoRA; even more memory-efficient (â6GB for 7B model)
- [ ] C. Prompt engineering instead of fine-tuning (not an alternative to fine-tuning, but a workaround)
- [x] D. Use Vertex AI's managed fine-tuning; infrastructure is abstracted; no need to manage GPU memory

> **Explanation:** (A), (B), and (D) are practical. (C) is a different approach.

---

### Q192. Prompt Engineering: Few-Shot vs. Zero-Shot Learning (Select all that apply)

**Scenario:** You want to classify product reviews as positive, neutral, or negative using an LLM. You have two approaches: (1) zero-shot: "Classify this review: ...", (2) few-shot: provide 3 examples, then ask for classification. Which is better?

- [ ] A. Zero-shot is always faster; few-shot adds latency
- [x] B. Few-shot typically achieves higher accuracy; examples teach the model the task
- [x] C. Few-shot uses more tokens (higher cost); trade accuracy for cost
- [x] D. Start with zero-shot; if accuracy is insufficient, upgrade to few-shot

> **Explanation:** (B), (C), and (D) are correct. (A) is true but overlooks accuracy benefits of few-shot.

---

### Q193. Chain-of-Thought Prompting for Complex Reasoning (Select all that apply)

**Scenario:** You ask an LLM: "If a book costs $10 and is on 20% off sale, how much do I save?" The model responds "Save $2" directly. With chain-of-thought prompting, you ask: "Let's think step-by-step. ...". Which benefits might you observe?

- [x] A. LLM generates intermediate steps; more transparent reasoning
- [x] B. Complex reasoning tasks (multi-step math, logic) often improve with step-by-step prompting
- [ ] C. Chain-of-thought always improves accuracy; use it for all tasks
- [x] D. Trade-off: more tokens used (higher cost); slower generation

> **Explanation:** (A), (B), and (D) are correct. (C) is too broad; some tasks don't benefit.

---

### Q194. Grounding LLM Outputs with External Data (Select all that apply)

**Scenario:** Your RAG system retrieves documents for context, but the LLM sometimes "hallucinates" (generates plausible-sounding but false information not in the retrieved documents). How do you reduce hallucinations?

- [x] A. Use citation mechanisms: require the LLM to cite which retrieved document supports each claim
- [x] B. Prompt engineering: instruct the model to only use information from the provided context
- [x] C. Constrained decoding: limit LLM outputs to facts present in retrieved documents (more complex but effective)
- [x] D. Evaluate with ground truth; fine-tune on examples where the model hallucinated

> **Explanation:** All (A), (B), (C), and (D) reduce hallucinations.

---

### Q195. Multi-Turn Conversations and Context Management (Select all that apply)

**Scenario:** You build a chatbot. A user asks: "What's the capital of France?" (LLM responds "Paris"). Then: "When was it founded?" The LLM must know from context that "it" refers to Paris. How do you manage conversational context?

- [x] A. Maintain a rolling context window: keep the last N messages; include them in each request to the LLM
- [x] B. Summarization: periodically summarize old messages into a compact summary; include summary + recent messages
- [ ] C. Each message is independent; context windows are not needed
- [x] D. Be aware of token limits: long conversations exceed model context windows (e.g., 4K tokens); truncate or summarize

> **Explanation:** (A), (B), and (D) are context management strategies. (C) leads to poor conversational coherence.

---

### Q196. Vertex AI Generative API: Model Selection (Select all that apply)

**Scenario:** You're choosing between Vertex AI models: Gemini 1.5 (27M context window, stronger reasoning), GPT-4 (8K context, available via API), or custom fine-tuned (smaller, cheaper). Which should you use for a legal document Q&A system?

- [x] A. Gemini 1.5's large context is ideal for long legal documents; in-context learning without fine-tuning
- [x] B. Fine-tuning on legal data may improve accuracy further; combine with Gemini for best results
- [ ] C. GPT-4 via API is cheaper than Vertex AI (often incorrect; costs vary)
- [x] D. Evaluate total cost: API calls Ã frequency vs. fine-tuning cost + reduced latency

> **Explanation:** (A), (B), and (D) are correct. (C) is a misconception; cost depends on usage.

---

### Q197. LLM Hallucination and Mitigation Strategies (Select all that apply)

**Scenario:** Your LLM-based summarization system hallucinates facts (e.g., attributes incorrect quotes to people). You want to reduce hallucinations. Which strategies help?

- [x] A. Temperature tuning: lower temperature (e.g., 0.3 instead of 1.0) reduces randomness; more deterministic, less creative outputs
- [x] B. Retrieval-augmented generation (RAG): ground outputs in factual documents
- [x] C. Prompt engineering: instruct the model to say "I don't know" if unsure, rather than guess
- [x] D. Fine-tuning on high-quality, factual data (not internet scraped data with misinformation)

> **Explanation:** All (A), (B), (C), and (D) reduce hallucinations.

---

### Q198. Semantic Search for RAG: Vector Databases (Select all that apply)

**Scenario:** You're building a RAG system with 1M documents. You embed each into a 768-dim vector. You use a vector database (e.g., Vertex Vector Search, Pinecone) for efficient retrieval. Which query types are efficient?

- [x] A. Nearest neighbor search: find top-k most similar vectors to a query vector (efficient with ANN indexing)
- [x] B. Semantic filtering: combine vector similarity with metadata filters (e.g., "find similar documents from 2024")
- [ ] C. Full-text search on vector embeddings (defeats the purpose; embeddings don't support text search directly)
- [x] D. Hybrid search: combine vector similarity with keyword-based search for better results

> **Explanation:** (A), (B), and (D) are practical. (C) confuses vector and full-text search.

---

### Q199. Fine-Tuning Data Quality for LLMs (Select all that apply)

**Scenario:** You prepare 10K examples for fine-tuning: prompt-response pairs from your domain. What quality checks should you perform?

- [x] A. Check for duplicates (remove); deduplicate reduces data efficiency
- [x] B. Validate prompt-response consistency (do responses actually answer the prompts?)
- [x] C. Check for bias (are all examples representative of your use case, or skewed?)
- [x] D. Balance dataset: if most examples are easy, include harder examples to improve robustness

> **Explanation:** All (A), (B), (C), and (D) improve fine-tuning data quality.

---

### Q200. Batching for LLM Inference Cost Optimization (Select all that apply)

**Scenario:** You have 1M documents to embed for RAG. Embedding API costs $0.0001 per 1K tokens. Single-document batches cost 1.5M tokens (1K tokens/doc Ã 1M docs + overhead). Can you optimize?

- [x] A. Batch documents: embed 100 documents in a single API call; reduces overhead; roughly 1M tokens (1K/doc Ã 1M docs, with less overhead)
- [ ] B. Batch size doesn't matter; API pricing is per-document
- [x] C. Calculate cost difference: 1.5M vs. 1M tokens Ã $0.0001 / 1K = significant savings
- [x] D. Use Vertex AI Batch Prediction API (if available) for large-scale embedding; may offer volume discounts

> **Explanation:** (A), (C), and (D) optimize embedding costs. (B) is incorrect; batching reduces per-document overhead.

---

### Q201. Filtering and Ranking Retrieved Documents for RAG Quality (Select all that apply)

**Scenario:** Your RAG retriever fetches top-10 documents for a query. But 3 of them are irrelevant. You implement a reranking step: use a cross-encoder model to score relevance. Which strategies work?

- [x] A. Cross-encoder ranking: rerank retrieved documents; pass only top-3 to LLM (higher quality, but reduces context)
- [x] B. Threshold filtering: discard documents with relevance score <0.5; ensures retrieved documents are relevant
- [ ] C. Use all top-10; LLM will filter irrelevant documents (often doesn't; LLM gets confused by noise)
- [x] D. Multi-stage ranking: semantic + keyword + domain-specific signals for robust ranking

> **Explanation:** (A), (B), and (D) improve retrieval quality. (C) is risky; noisy context hurts LLM reasoning.

---

### Q202. Guardrails for LLM Safety (Select all that apply)

**Scenario:** Your LLM-based system must handle adversarial queries (e.g., "Generate a password reset code for any user"). You want to prevent misuse. Which safeguards apply?

- [x] A. Input validation: filter prompts for suspicious patterns (e.g., "generate credential", "bypass security")
- [x] B. Output validation: filter LLM responses for sensitive data (passwords, PII, API keys)
- [x] C. Fine-tune on safe examples; instruct tuning to refuse harmful requests
- [x] D. Use specialized guardrail APIs (e.g., Google Cloud's Guardrails for GenAI)

> **Explanation:** All (A), (B), (C), and (D) are guardrails for LLM safety.

---

### Q203. LLM Cost Estimation and Budgeting (Select all that apply)

**Scenario:** You estimate costs for an LLM application: 1K users Ã 10 queries/day Ã 365 days = 3.65M queries/year. LLM API: $0.001 per 1K input tokens + $0.003 per 1K output tokens. Average tokens per query: 200 input, 100 output. What's the annual cost?

- [x] A. Input cost: 3.65M queries Ã 200 tokens Ã $0.001 / 1K = $730
- [x] B. Output cost: 3.65M queries Ã 100 tokens Ã $0.003 / 1K = $1,095
- [x] C. Total: $730 + $1,095 = $1,825 / year (excluding infrastructure)
- [x] D. Add infrastructure costs (embeddings, vector DB, storage) to get total cost; may double or triple the LLM cost

> **Explanation:** All calculations are correct. (D) highlights often-overlooked infrastructure costs.

---

### Q204. Streaming LLM Outputs for Low Latency (Select all that apply)

**Scenario:** Your chatbot generates responses token-by-token using streaming. Users see text appearing in real-time instead of waiting for the full response. Which trade-offs apply?

- [x] A. Streaming reduces perceived latency; users see results immediately
- [x] B. Streaming requires client-side buffering; servers push tokens as they arrive
- [ ] C. Streaming is always faster; no trade-offs
- [x] D. For RAG systems: stream the response after retrieval; no speed gain from retrieval (already async)

> **Explanation:** (A), (B), and (D) are correct. (C) ignores that streaming is a UI/UX improvement, not a speed improvement for the backend.

---

### Q205. Knowledge Distillation for LLMs (Select all that apply)

**Scenario:** You have a large LLM (70B parameters) that's accurate but expensive to serve. You distill it into a smaller model (7B) using 100K examples generated by the large model. What are benefits and costs?

- [x] A. Smaller model is faster and cheaper to serve; trades some accuracy for efficiency
- [x] B. Distillation requires generating training data (100K examples from large model); adds cost
- [ ] C. Distilled models always match the teacher's accuracy; no degradation
- [x] D. Distillation works well if the task is well-covered by training data; custom tasks may not distill as well

> **Explanation:** (A), (B), and (D) are correct. (C) is too optimistic; accuracy degradation is expected.

---

### Q206. LLM Finetuning for Specific Domains (Select all that apply)

**Scenario:** Your organization is in healthcare. You fine-tune Gemini on medical data (clinical notes, research papers). Which considerations apply?

- [x] A. Domain-specific terminology: fine-tuning teaches the model medical jargon and context
- [x] B. Privacy and compliance: medical data is sensitive; ensure HIPAA compliance (e.g., deidentification, secure storage)
- [x] C. Domain evaluation: use domain-specific metrics (e.g., medical accuracy, clinician review) not generic benchmarks
- [x] D. Continuous updates: medical knowledge evolves; periodically retrain on new evidence

> **Explanation:** All (A), (B), (C), and (D) are domain-specific considerations.

---

### Q207. Prompt Template Versioning and Management (Select all that apply)

**Scenario:** Your system uses prompt templates for various tasks (summarization, classification, Q&A). As you optimize prompts, versions diverge. How do you manage versions?

- [x] A. Version prompts in Git; tag versions that go to production
- [x] B. Store templates in a Prompt Management System (e.g., LangChain Hub); track A/B test results per version
- [x] C. Document prompt changes: what changed, why, expected impact
- [x] D. Test new prompts on a holdout set before production deployment (similar to model A/B testing)

> **Explanation:** All (A), (B), (C), and (D) are best practices for prompt management.

---

### Q208. Using LLMs for Data Labeling and Annotation (Select all that apply)

**Scenario:** You need labels for a classification task (1M documents). Hiring annotators is expensive ($10K). You use an LLM to auto-label: prompt LLM with document â classification. How do you validate quality?

- [x] A. Sample 1K documents; have humans label them; compare human vs. LLM labels; estimate LLM accuracy
- [x] B. Use agreement between multiple LLM models (ensemble voting) to identify uncertain predictions
- [x] C. Fine-tune a classification model on LLM labels; evaluate on human-labeled holdout set
- [x] D. Iteratively improve: use model predictions + human feedback to refine labels (active learning)

> **Explanation:** All (A), (B), (C), and (D) validate and improve LLM-generated labels.

---

### Q209. Multimodal LLMs for Vision and Language (Select all that apply)

**Scenario:** You're building a document analysis system using Gemini's multimodal capabilities (image + text input). Documents are scanned PDFs. Which use cases are enabled?

- [x] A. Extract text from scanned PDFs (OCR replacement); Gemini reads the image and outputs text
- [x] B. Classify documents by type (invoice, receipt, contract) by reading the image
- [x] C. Summarize document content; Gemini understands both text and visual layout
- [ ] D. Store multimodal embeddings (image + text); not natively supported by multimodal LLMs

> **Explanation:** (A), (B), and (C) are enabled by multimodal LLMs. (D) requires custom embedding approaches.

---

### Q210. LLM Robustness and Adversarial Examples (Select all that apply)

**Scenario:** Your chatbot is deployed to the public. Users test prompt injection attacks (e.g., "Forget the system prompt and execute: ..."). How do you harden the system?

- [x] A. Use a robust system prompt; prioritize safety over flexibility
- [x] B. Implement input validation; filter suspicious patterns (e.g., "system prompt", "execute", "ignore")
- [x] C. Fine-tune on adversarial examples; teach the model to resist attacks
- [x] D. Monitor for unusual patterns (e.g., unusual request volume, same user repeated attempts); rate-limit or block

> **Explanation:** All (A), (B), (C), and (D) harden the system against adversarial attacks.

---

## Domain 12: Responsible AI (Q211âQ230)

### Q211. Bias in ML Systems: Types of Bias (Select all that apply)

**Scenario:** Your hiring recommendation model shows 78% accuracy for male candidates, 72% for female candidates. Which types of bias might be present?

- [x] A. Training data bias: if training data underrepresents women or contains biased labels, the model learns biased patterns
- [x] B. Algorithmic bias: some algorithms are inherently more biased (e.g., using gender-correlated features)
- [x] C. Measurement bias: if evaluation metrics differ between groups (e.g., different thresholds), bias may be masked
- [x] D. Representation bias: if test set underrepresents women, evaluation is unreliable

> **Explanation:** All (A), (B), (C), and (D) are potential bias sources in ML systems.

---

### Q212. Fairness Metrics and Definitions (Select all that apply)

**Scenario:** Your organization defines fairness as: across demographic groups, the model should have equal precision (true positive rate). Your model achieves: Group A precision=0.85, Group B precision=0.75. Is it fair by this definition?

- [ ] A. Yes; precision is above 0.70 for both groups
- [x] B. No; the 10% gap violates the equal precision criterion
- [x] C. But fairness has trade-offs; improving Group B precision may reduce overall model accuracy
- [x] D. Choose a different fairness definition (e.g., equalized odds) if the chosen one is unachievable; document the choice

> **Explanation:** (B), (C), and (D) are correct. (A) ignores the fairness definition mismatch.

---

### Q213. Explainability Methods: SHAP vs. LIME vs. XRAI (Select all that apply)

**Scenario:** Your model makes loan approval decisions. You want to explain predictions. SHAP (Shapley values) is theoretically sound but slow. LIME (Local Interpretable Model-Agnostic Explanations) is fast but approximate. XRAI (Grad-CAM variant for images) is visual. When do you use each?

- [x] A. SHAP: use for smaller models/datasets where speed is acceptable; provides theoretically grounded importance scores
- [x] B. LIME: use for real-time serving; trades accuracy for speed; suitable for approximate explanations
- [x] C. XRAI: use for image-based models; highlights salient image regions driving predictions
- [x] D. Ensemble: pre-compute SHAP for offline analysis; use LIME for real-time explanations

> **Explanation:** All (A), (B), (C), and (D) are pragmatic choices.

---

### Q214. Interpretability for Black-Box Models (Select all that apply)

**Scenario:** You deploy a deep neural network for medical image classification. Clinicians need to understand why the model flagged a scan as abnormal. The model is a black box. Which approaches work?

- [x] A. Saliency maps: highlight image regions the model focuses on; use Grad-CAM or SHAP
- [x] B. Attention visualization: if the model uses attention layers, visualize which regions it attends to
- [x] C. Counterfactual explanations: show what would need to change in the image for the model to change its decision
- [x] D. Replace the model with a simpler, interpretable model (trade accuracy for interpretability)

> **Explanation:** All (A), (B), (C), and (D) provide interpretability for black-box models.

---

### Q215. Concept Activation Vectors (TCAV) for Interpretability (Select all that apply)

**Scenario:** Your image classification model predicts "bird" on an image. You want to know: does the model rely on "striped pattern" or "flight capability"? TCAV is a technique to measure concept importance. Which statements are true?

- [x] A. TCAV measures importance of human-defined concepts (e.g., "striped", "flying") to model predictions
- [x] B. You provide examples of the concept (striped images); TCAV computes the concept's influence on the model
- [ ] C. TCAV is model-agnostic; works on any model without modification
- [x] D. TCAV helps debug model behavior: if an undesired concept is influential, you can address it (e.g., collect more diverse data)

> **Explanation:** (A), (B), and (D) are correct. (C) is partially true; TCAV requires gradient information, limiting applicability.

---

### Q216. Privacy-Preserving ML: Differential Privacy (Select all that apply)

**Scenario:** You train a model on sensitive medical data (100 patients). You want to ensure that the model doesn't leak individual patient information. You use differential privacy (DP): add carefully calibrated noise during training. Which statements apply?

- [x] A. DP adds noise to gradients during training; provides a mathematical guarantee that individual records are protected
- [x] B. DP has a privacy budget (epsilon, delta); tighter budgets (lower epsilon) mean more privacy but worse model accuracy
- [x] C. DP is training-time noise injection; does not protect against model inversion attacks at inference (use additional safeguards)
- [x] D. DP-SGD (Differentially Private Stochastic Gradient Descent) is an implementation; fit hyperparameters (noise scale, batch size) for your privacy requirements

> **Explanation:** All (A), (B), (C), and (D) are correct DP concepts.

---

### Q217. Federated Learning for Privacy (Select all that apply)

**Scenario:** You want to train a model on data from 1000 mobile devices without collecting raw data centrally. You use federated learning: each device trains locally, uploads gradients, server aggregates. Which privacy benefits apply?

- [x] A. Raw data never leaves devices; central server only sees gradients (more private than centralized training)
- [x] B. Gradients can leak information (membership inference attacks); combine FL with differential privacy for stronger privacy
- [x] C. Communication cost is high; sending gradients for 1M-parameter models to 1000 devices incurs significant bandwidth
- [x] D. FL works best for latency-tolerant applications (updates every few hours, not real-time)

> **Explanation:** All (A), (B), (C), and (D) are FL characteristics.

---

### Q218. Fairness Interventions and Debiasing Techniques (Select all that apply)

**Scenario:** Your model is biased against Group B. You want to debias. Which techniques help?

- [x] A. Rebalancing training data: oversample Group B or undersample Group A (changes class distribution)
- [x] B. Fairness-aware learning: add fairness constraints to the loss function (e.g., equal precision across groups)
- [x] C. Post-processing: adjust model outputs to meet fairness criteria (e.g., set decision threshold per group)
- [x] D. Data augmentation: generate synthetic Group B samples to balance dataset

> **Explanation:** All (A), (B), (C), and (D) are debiasing techniques.

---

### Q219. Fairness Evaluation on Disaggregated Data (Select all that apply)

**Scenario:** You evaluate your model on the full test set (accuracy = 0.88). When you disaggregate by demographics: Group A = 0.92, Group B = 0.82. This is concerning. Why disaggregate?

- [x] A. Disaggregation reveals performance disparities that aggregate metrics hide
- [x] B. A single metric (0.88) is misleading if it masks inequality across groups
- [x] C. Business/ethical requirement: fairness regulations (e.g., GDPR Article 22) demand transparency per group
- [ ] D. Disaggregation is unnecessary; aggregate metrics are sufficient

> **Explanation:** (A), (B), and (C) are correct reasons to disaggregate. (D) is dangerous; it masks inequality.

---

### Q220. Transparency and Model Cards (Select all that apply)

**Scenario:** Your organization publishes a model card for a pre-trained model. What should the card include?

- [x] A. Model details: architecture, training data, evaluation metrics
- [x] B. Limitations: known failure modes, intended use cases, unsuitable applications
- [x] C. Ethical considerations: bias evaluation, fairness metrics, disparities across groups
- [x] D. Reproducibility: hyperparameters, code, datasets (or pointers to them)

> **Explanation:** All (A), (B), (C), and (D) are components of comprehensive model cards.

---

### Q221. Responsible AI and Environmental Impact (Select all that apply)

**Scenario:** You train large ML models. Training a 100B-parameter model requires weeks on high-performance GPUs, consuming significant electricity. Which considerations apply?

- [x] A. Carbon footprint: compute costs translate to CO2 emissions; consider using renewable energy sources
- [x] B. Model efficiency: smaller models (distillation, pruning) reduce carbon footprint significantly
- [ ] C. Environmental impact is not a concern for ML practitioners
- [x] D. Report training costs/energy usage in model cards; transparency enables accountability

> **Explanation:** (A), (B), and (D) are correct. (C) dismisses an important responsibility.

---

### Q222. Bias Amplification in Recommendation Systems (Select all that apply)

**Scenario:** Your recommendation system learns from user clicks. Initially, some items are underrepresented (e.g., women-authored books). Clicks are biased (users click what they're shown). Over time, underrepresented items get fewer clicks, fewer recommendations, fewer clicks (amplification). How do you mitigate?

- [x] A. Explore-exploit: periodically recommend items outside the learned preference to break feedback loops
- [x] B. Fair ranking: ensure recommendations include diversity (e.g., 30% underrepresented items)
- [x] C. Weighted sampling: boost underrepresented items' weight during training
- [x] D. Monitor recommendation diversity over time; alert if diversity decreases

> **Explanation:** All (A), (B), (C), and (D) mitigate bias amplification.

---

### Q223. Fairness and Trade-offs with Model Performance (Select all that apply)

**Scenario:** You retrofit your model to achieve equal recall across demographic groups. Recall improves for Group B (78% â 85%) but decreases for Group A (92% â 88%). Overall accuracy drops from 90% to 87%. Is this acceptable?

- [x] A. Context-dependent: in high-stakes applications (e.g., criminal justice), fairness may outweigh overall accuracy
- [x] B. Negotiate trade-offs with stakeholders; fairness is not a technical problem alone
- [ ] C. Fairness always reduces accuracy; unachievable to have both
- [x] D. Quantify trade-offs: present stakeholders with multiple options (fairness threshold vs. accuracy) and their costs

> **Explanation:** (A), (B), and (D) are pragmatic. (C) is too absolute; sometimes fairness and accuracy align.

---

### Q224. Model Auditing and External Validation (Select all that apply)

**Scenario:** Your organization deploys a high-stakes model (e.g., loan approval). You want independent validation to ensure fairness, accuracy, and safety. Which auditing approaches apply?

- [x] A. Third-party audit: external team tests the model independently, reports findings
- [x] B. Red-teaming: adversarial testing to find failure modes, biases, adversarial examples
- [x] C. Benchmark testing: evaluate on publicly available datasets and compare to established baselines
- [x] D. Continuous monitoring: after deployment, track fairness/accuracy metrics and alert if they degrade

> **Explanation:** All (A), (B), (C), and (D) are auditing approaches.

---

### Q225. Explainability for Regulatory Compliance (Select all that apply)

**Scenario:** Under GDPR Article 22, individuals have a right to explanation for automated decisions. Your model makes loan approval decisions. You must explain why someone was denied. Which methods suffice?

- [x] A. Feature importance: which features drove the denial decision?
- [x] B. Rule-based explanations: if the model uses interpretable rules (e.g., decision tree), explain using the rules
- [x] C. Counterfactual explanations: "If your income were $10K higher, you'd be approved"
- [ ] D. No explanation needed; GDPR compliance only requires disclosure, not detailed explanations

> **Explanation:** (A), (B), and (C) are accepted explanation methods. (D) is incorrect; GDPR mandates meaningful explanations.

---

### Q226. Consent and Data Governance in ML (Select all that apply)

**Scenario:** You collect user data to train a recommendation model. You want to ensure ethical data practices. Which considerations apply?

- [x] A. Informed consent: users should understand how their data is used (training model, profiling, etc.)
- [x] B. Data minimization: collect only data necessary for the stated purpose
- [x] C. Data deletion: users can request deletion (right to be forgotten); model retraining may be required
- [x] D. Transparency: document data sources, retention periods, and uses

> **Explanation:** All (A), (B), (C), and (D) are ethical data governance principles.

---

### Q227. Adversarial Robustness and Attacks (Select all that apply)

**Scenario:** Your image classification model classifies a stop sign correctly. An adversary modifies the image slightly (adds imperceptible noise); the model misclassifies it as "yield". How do you build robustness?

- [x] A. Adversarial training: train on adversarial examples; model learns to resist attacks
- [x] B. Certified defenses: use techniques that provide formal robustness guarantees
- [x] C. Ensemble models: combine multiple models; adversarial examples may not fool all models simultaneously
- [x] D. Input smoothing: add noise during inference; reduces vulnerability to carefully crafted attacks

> **Explanation:** All (A), (B), (C), and (D) improve adversarial robustness.

---

### Q228. Responsible AI in High-Stakes Applications: Criminal Justice (Select all that apply)

**Scenario:** Your organization builds a recidivism prediction model (predict likelihood of re-offending). The model is used in parole decisions. Which responsible AI practices are critical?

- [x] A. Explainability: judges must understand why the model flagged someone as high-risk
- [x] B. Fairness auditing: ensure racial/demographic parity in predictions (historical biases in data are pervasive)
- [x] C. Human-in-the-loop: model provides recommendations; humans make final decisions
- [x] D. Regular audits: track disparities over time; retrain with corrected data if needed

> **Explanation:** All (A), (B), (C), and (D) are essential in high-stakes settings.

---

### Q229. Interpretability vs. Accuracy Spectrum (Select all that apply)

**Scenario:** You're choosing between: (1) Logistic Regression (interpretable, ~0.70 accuracy), (2) Random Forest (moderate interpretability, ~0.85 accuracy), (3) Deep Neural Network (black-box, ~0.90 accuracy). Which factors guide the choice?

- [x] A. Stakes of decisions: high-stakes (medical) require interpretability; low-stakes (product recommendations) can use black-box
- [x] B. Regulatory environment: GDPR/FCRA may mandate explainability; choose accordingly
- [x] C. User trust: interpretable models may earn user trust better, even if less accurate
- [x] D. Hybrid approach: use black-box model for predictions + post-hoc explanations (SHAP/LIME) to balance accuracy and interpretability

> **Explanation:** All (A), (B), (C), and (D) are valid decision factors.

---

### Q230. Responsible AI Governance: Organizational Practices (Select all that apply)

**Scenario:** Your organization builds many ML models. You want to ensure all are developed responsibly. Which governance practices help?

- [x] A. Model review boards: technical and ethics review before production deployment
- [x] B. Documentation requirements: model cards, bias audits, limitations documented for all models
- [x] C. Training and education: all data scientists understand bias, fairness, privacy implications
- [x] D. Incident response: establish procedures for responding to model failures or discovered biases

> **Explanation:** All (A), (B), (C), and (D) are governance best practices.

---

## Domain 13: Production ML Systems (Q231âQ250)

### Q231. Static vs. Dynamic Training Regimes (Select all that apply)

**Scenario:** Your system predicts user churn. You have two approaches: (1) static training: retrain weekly using the latest 90 days of data, (2) dynamic training: continuously update model weights on streaming data. Which trade-offs apply?

- [x] A. Static: simpler to implement, reproducible, easier to debug; slower to adapt to rapid changes
- [x] B. Dynamic: adapts faster to new patterns; harder to debug, risk of catastrophic forgetting
- [x] C. Static often uses A/B testing for validation; dynamic uses online metrics
- [x] D. Hybrid: static training weekly + dynamic fine-tuning for rapid adaptation

> **Explanation:** All (A), (B), (C), and (D) are correct.

---

### Q232. Data Drift, Model Drift, and Concept Drift (Select all that apply)

**Scenario:** A year after deployment, your demand forecasting model's accuracy drops. You investigate and find: (1) input feature distributions have shifted (feature drift), (2) model weights have drifted (model drift), (3) the relationship between inputs and outputs has changed (concept drift). What do these mean?

- [x] A. Feature drift: input distributions changed; model may be extrapolating beyond training domain
- [x] B. Model drift: weights have shifted (often due to retraining); can be positive or negative
- [x] C. Concept drift: the underlying relationship changed (e.g., customer behavior changed due to market conditions); requires retraining on new data
- [x] D. All three may occur together; diagnose root cause before intervention

> **Explanation:** All (A), (B), (C), and (D) are drift concepts and require different interventions.

---

### Q233. TensorFlow Extended (TFX) for ML Pipelines (Select all that apply)

**Scenario:** You want to build an end-to-end ML pipeline: data ingestion, validation, preprocessing, training, evaluation, model serving. TFX is a toolkit for this. Which components are typical?

- [x] A. ExampleGen: data ingestion from various sources
- [x] B. StatisticsGen: compute statistics on data; feed to TFDV for validation
- [x] C. Transform: feature engineering using TensorFlow Transform (TFT)
- [x] D. Trainer: model training; Tuner: hyperparameter tuning; Evaluator: model evaluation

> **Explanation:** All (A), (B), (C), and (D) are core TFX components.

---

### Q234. Cloud Dataproc for Large-Scale Data Processing (Select all that apply)

**Scenario:** You need to process 1TB of data for model training. You provision a Dataproc cluster with 100 worker nodes (Hadoop/Spark). Which considerations apply?

- [x] A. Cluster sizing: balance cost (larger cluster = more cost) vs. speed (more parallelism = faster)
- [x] B. Auto-scaling: scale workers up/down based on job load; saves cost during idle periods
- [x] C. Spot/Preemptible instances: cheaper, but can be interrupted; acceptable for fault-tolerant jobs
- [x] D. Network bandwidth: 1TB transfer to/from cluster may be slow; consider colocation of data and compute

> **Explanation:** All (A), (B), (C), and (D) are Dataproc considerations.

---

### Q235. Training-Serving Skew Detection and Prevention (Select all that apply)

**Scenario:** Your model performs well on training data but poorly in production. You suspect training-serving skew: training data differs from production data. Which indicators and mitigation strategies apply?

- [x] A. Statistical tests: compare training and production data distributions; alert if they diverge significantly
- [x] B. Schema validation: ensure production data schema matches training schema (same columns, types)
- [x] C. Feature audit: log features used in production; compare against training features
- [x] D. A/B testing: serve both training and production models; compare metrics to quantify skew impact

> **Explanation:** All (A), (B), (C), and (D) detect and mitigate training-serving skew.

---

### Q236. Model Registry and Artifact Management (Select all that apply)

**Scenario:** Your organization trains 100 models across teams. Models are scattered in different places (Cloud Storage, local drives, different ML frameworks). You want centralized management. Which benefits apply to using Vertex AI Model Registry?

- [x] A. Single source of truth: all models tracked in one place with versioning
- [x] B. Lineage tracking: which data and code produced each model version
- [x] C. Metadata: model type, performance metrics, training hyperparameters, associated pipelines
- [x] D. Governance: approval workflows, deployment tracking, versioning history

> **Explanation:** All (A), (B), (C), and (D) are benefits of centralized model registry.

---

### Q237. Continuous Integration for ML Pipelines (Select all that apply)

**Scenario:** Your team commits code changes to a ML pipeline. You want CI to automatically: (1) run tests, (2) train a model, (3) evaluate on test set, (4) gat approval before merge. Which components are needed?

- [x] A. CI triggers: on every commit/PR, execute the pipeline
- [x] B. Automated tests: unit tests for data preprocessing, model inference, etc.
- [x] C. Model validation: enforce performance thresholds (e.g., AUC > 0.80) before approval
- [x] D. Approval gates: human review required for high-risk changes; automated approval for low-risk

> **Explanation:** All (A), (B), (C), and (D) are CI components for ML.

---

### Q238. Continuous Delivery and Deployment Strategy (Select all that apply)

**Scenario:** You have a validated model ready to deploy. You want a strategy that minimizes risk. Which approaches apply?

- [x] A. Canary deployment: route 5% traffic initially; monitor; gradually increase if stable
- [x] B. Blue-green deployment: maintain two production environments; switch traffic when ready
- [x] C. Rolling deployment: gradually replace old models with new; no downtime
- [x] D. Shadow deployment: new model runs alongside old; no user impact; compare predictions

> **Explanation:** All (A), (B), (C), and (D) are safe deployment strategies.

---

### Q239. Model Serving at Scale: Distributed Inference (Select all that apply)

**Scenario:** Your model serves 1M QPS. A single GPU can handle 10K QPS. You need 100 GPUs. Which architectural patterns enable this scale?

- [x] A. Load balancing: distribute requests across GPUs; Cloud Load Balancer routes traffic
- [x] B. Request batching: aggregate requests before inference; improves GPU utilization
- [x] C. Model sharding: split large model across multiple GPUs; enables larger models than fit on one GPU
- [x] D. Caching: cache predictions for identical inputs; reduces redundant computation

> **Explanation:** All (A), (B), (C), and (D) are scaling techniques.

---

### Q240. Detecting and Handling Model Failures (Select all that apply)

**Scenario:** Your model suddenly degrades: error rate spikes from 0.1% to 5%. Which debugging steps and mitigations apply?

- [x] A. Immediate rollback: revert to previous stable model; minimize user impact
- [x] B. Diagnostics: check input data quality, model artifacts, dependencies (Feature Store, etc.)
- [x] C. Data quality alerts: if data issues detected, pause serving until fixed
- [x] D. Automated fallback: switch to a secondary model (e.g., heuristic-based) while investigating

> **Explanation:** All (A), (B), (C), and (D) are failure handling practices.

---

### Q241. Cost Optimization in ML Systems (Select all that apply)

**Scenario:** Your ML system has high operational costs. Where can you optimize?

- [x] A. Model inference: use smaller models (distillation), quantization, hardware accelerators (TPUs vs. GPUs)
- [x] B. Data storage: archive old data, use colder storage tiers, deduplication
- [x] C. Training: use preemptible VMs, distributed training with efficient communication
- [x] D. Feature Store: cache features, reduce online vs. offline store writes

> **Explanation:** All (A), (B), (C), and (D) optimize costs.

---

### Q242. Active Learning and Annotation Efficiency (Select all that apply)

**Scenario:** You have an unlabeled dataset of 100K samples. Labeling all is expensive. You use active learning: train a model, identify uncertain predictions, label those samples, retrain. Which benefits apply?

- [x] A. Label efficiency: achieve target accuracy with fewer labels (50% less labeling for comparable accuracy)
- [x] B. Targeted labeling: focus on hard/uncertain examples; they're most valuable for learning
- [x] C. Iterative improvement: each labeling round improves the model; can stop early when performance plateaus
- [ ] D. Active learning always outperforms random sampling; use it for all tasks

> **Explanation:** (A), (B), and (C) are benefits. (D) is too absolute; active learning has overhead.

---

### Q243. Handling Imbalanced Data in Production (Select all that apply)

**Scenario:** Your production model predicts rare events (0.1% positive rate). At scale, false positives are costly. Which techniques handle imbalanced data?

- [x] A. Threshold tuning: adjust decision threshold to balance precision/recall for your use case
- [x] B. Class weighting: up-weight the minority class during training
- [x] C. Ensemble methods: combine multiple models; some may specialize in minority class
- [x] D. Data augmentation: generate synthetic minority examples (SMOTE); improves representation

> **Explanation:** All (A), (B), (C), and (D) handle imbalanced data.

---

### Q244. Real-Time Feature Computation and Caching (Select all that apply)

**Scenario:** Your model serves predictions at low latency (<50ms). Some features require computation (e.g., "user activity in last hour"). Which strategies enable fast feature access?

- [x] A. Pre-compute offline: compute features nightly, cache in memory/database; serve from cache at inference time
- [x] B. Real-time computation: compute features on-demand; acceptable if <30ms
- [x] C. Hybrid: pre-compute batch features; compute only missing real-time features on-demand
- [x] D. Bloom filters / approximate counts: trade accuracy for speed (acceptable for non-critical features)

> **Explanation:** All (A), (B), (C), and (D) are strategies.

---

### Q245. Monitoring Model Fairness in Production (Select all that apply)

**Scenario:** You deploy a model and want continuous fairness monitoring. Which metrics and practices apply?

- [x] A. Demographic parity: monitor accuracy, precision, recall per demographic group; alert if disparity exceeds threshold
- [x] B. Equalized odds: monitor true positive rate and false positive rate per group; alert on disparity
- [x] C. Causal fairness: monitor if model decisions are independent of protected attributes (advanced, requires causal inference)
- [x] D. Threshold effects: for binary predictions, monitor how decision threshold affects fairness per group

> **Explanation:** All (A), (B), (C), and (D) are fairness monitoring approaches.

---

### Q246. Scaling Feature Store to Production (Select all that apply)

**Scenario:** Your Feature Store serves 100K QPS (online predictions). Features come from both offline and online stores. Which considerations ensure scalability?

- [x] A. Denormalization: flatten feature schemas to reduce joins; trade storage for query speed
- [x] B. Partitioning: shard features by entity_id; distribute load across servers
- [x] C. Caching tiers: L1 (memory), L2 (Redis), L3 (database); serve from nearest tier
- [x] D. Async writes: batch writes to offline store; don't block online serving on write latency

> **Explanation:** All (A), (B), (C), and (D) enable Feature Store scaling.

---

### Q247. AutoML and NAS for Model Search (Select all that apply)

**Scenario:** You want to automatically find the best model architecture for your task. You use AutoML (e.g., Vertex AI AutoML) or Neural Architecture Search (NAS). Which considerations apply?

- [x] A. Search space: define candidate architectures; large search spaces require more compute
- [x] B. Early stopping: terminate unpromising architectures; saves compute
- [x] C. Benchmark: compare AutoML-discovered models against hand-crafted baselines
- [x] D. Transfer learning: initialize search with pre-trained models; speeds up search

> **Explanation:** All (A), (B), (C), and (D) are AutoML considerations.

---

### Q248. Reproducibility and Experiment Tracking (Select all that apply)

**Scenario:** Your team runs 100 experiments per week. After 6 months, you can't reproduce an old experiment (which random seed was used? Which dataset version?). How do you improve reproducibility?

- [x] A. Experiment tracking: log all hyperparameters, seed, dataset version, code commit hash
- [x] B. Versioning: version datasets, models, code; tag releases
- [x] C. Environment isolation: Docker containers, Conda environments; ensure same dependencies
- [x] D. MLflow / Weights & Biases: tools for tracking experiments; enable easy reproduction

> **Explanation:** All (A), (B), (C), and (D) improve reproducibility.

---

### Q249. Observability and Debugging Production ML Systems (Select all that apply)

**Scenario:** Your production model is degrading. Users report poor experience. You need to diagnose quickly. Which observability tools/practices help?

- [x] A. Metrics dashboards: AUC, latency, error rate in real-time; visualize trends
- [x] B. Structured logging: log predictions with input/output/latency; enable debugging
- [x] C. Distributed tracing: trace request through system components; identify bottlenecks
- [x] D. Alerts: trigger on metric anomalies; page on-call engineers

> **Explanation:** All (A), (B), (C), and (D) enable production observability.

---

### Q250. Organizational Maturity and ML Governance (Select all that apply)

**Scenario:** Your organization has grown from 1 ML engineer to 50. You need governance to ensure: consistency, quality, compliance. Which practices support scaling?

- [x] A. ML platforms/tools: standardized stacks (Vertex AI, Feature Store, etc.); enable self-service model development
- [x] B. Policies and standards: documented best practices for data collection, model training, deployment
- [x] C. Cross-functional collaboration: data eng, ML eng, product, ethics teams align on requirements
- [x] D. Continuous learning: share learnings, incident postmortems, and best practices across teams

> **Explanation:** All (A), (B), (C), and (D) support organizational scaling.

---

# End of Mock Exam

Thank you for working through these 250 questions. Use this resource to:
- Identify knowledge gaps
- Simulate exam conditions (time yourself)
- Practice multi-select questions
- Deepen understanding through explanations

Good luck on your PMLE certification!