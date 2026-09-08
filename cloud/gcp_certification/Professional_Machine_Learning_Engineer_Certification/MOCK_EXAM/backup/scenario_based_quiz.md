# Vertex AI Scenario-Based Quiz

## Instructions
For each scenario, identify the **best Vertex AI service(s)** to use. Multiple services may be mentioned, but choose the **primary service** most suitable for the use case.

---

## Scenario 1: Real-Time Fraud Detection API
**Context:** Your company processes 10,000 credit card transactions per second. You've trained a Random Forest model that needs to score each transaction in <50ms and return a fraud probability. The model is already trained and ready for production.

**Scenario Question:**
You need to serve this model with guaranteed low latency and auto-scaling to handle traffic spikes. Which Vertex AI service should you use?

**Options:**
- [ ] A) Vertex AI Batch Prediction
- [ ] B) Vertex AI Endpoints
- [ ] C) Vertex AI Pipelines
- [ ] D) Vertex AI Custom Training

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Endpoints**: Provides REST API with <100ms latency, auto-scaling, traffic splitting. Perfect for real-time inference.
- ✗ A) Batch Prediction: Processes data asynchronously (hours), not real-time
- ✗ C) Pipelines: Orchestrates workflows, not a serving mechanism
- ✗ D) Custom Training: For model development, not serving

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Real-time inference with <100ms latency → **Endpoints**
- Need REST/gRPC API → **Endpoints**
- Auto-scaling required → **Endpoints**

</details>

---

## Scenario 2: Overnight Batch Scoring of 500M Customer Records
**Context:** Every night, you need to score 500 million customer records through a trained churn prediction model. Results should be written to BigQuery by 6 AM. The model is already trained and in the Model Registry.

**Scenario Question:**
Which Vertex AI service is most cost-effective and suitable for this use case?

**Options:**
- [ ] A) Vertex AI Endpoints with continuous scoring
- [ ] B) Vertex AI Batch Prediction
- [ ] C) Vertex AI Pipelines with custom training
- [ ] D) Vertex AI Feature Store

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Batch Prediction**: Asynchronous processing of large datasets, cost-effective, outputs directly to BigQuery. No need for real-time latency.
- ✗ A) Endpoints: Over-engineered for batch jobs, wasteful cost (real-time pricing)
- ✗ C) Pipelines: Good for ML workflows, but not specifically for batch prediction
- ✗ D) Feature Store: For feature management, not batch scoring

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Large-scale batch processing (>100M records) → **Batch Prediction**
- Output to BigQuery → **Batch Prediction**
- Non-real-time requirements → **Batch Prediction**
- Cost-conscious approach → **Batch Prediction**

</details>

---

## Scenario 3: Data Preparation for AutoML Training
**Context:** Your team has 50,000 customer support tickets with manual labels (urgent/not-urgent) scattered across multiple CSV files. You need to prepare this data for training a text classification model using Vertex AI AutoML.

**Scenario Question:**
Which Vertex AI service should you use to organize, validate, and prepare this labeled data?

**Options:**
- [ ] A) Vertex AI Workbench
- [ ] B) Vertex AI Datasets
- [ ] C) Vertex AI Data Labeling Service
- [ ] D) Vertex AI Feature Store

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Datasets**: Manages labeled training data, validates schema, versions data, integrates with AutoML. Consolidates multiple data sources.
- ✗ A) Workbench: IDE for development, not data management
- ✗ C) Data Labeling: For creating labels, but data is already labeled
- ✗ D) Feature Store: For pre-computed features for training/serving, not raw data

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Organize labeled training data → **Datasets**
- Data validation and versioning → **Datasets**
- Integration with AutoML → **Datasets**
- Multiple data sources consolidation → **Datasets**

</details>

---

## Scenario 4: Automated Hyperparameter Optimization
**Context:** You've built a custom TensorFlow model for image classification. Initial training gives 85% accuracy, but you suspect better hyperparameters could improve this. You need to systematically test different learning rates, batch sizes, and regularization values.

**Scenario Question:**
Which Vertex AI service should automate this hyperparameter search?

**Options:**
- [ ] A) Vertex AI Experiments
- [ ] B) Vertex AI Hyperparameter Tuning (HPO)
- [ ] C) Vertex AI TensorBoard
- [ ] D) Vertex AI Custom Training

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI HPO**: Automated hyperparameter optimization using Bayesian search, grid search, or random search. Runs multiple training jobs with different hyperparameters and finds optimal values.
- ✗ A) Experiments: Tracks different runs, but doesn't automate hyperparameter search
- ✗ C) TensorBoard: Visualization tool, not optimization
- ✗ D) Custom Training: Runs training jobs, but doesn't optimize hyperparameters automatically

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Systematic hyperparameter optimization → **HPO**
- Bayesian/Grid/Random search → **HPO**
- Improve model accuracy → **HPO**
- Avoid manual trial-and-error → **HPO**

</details>

---

## Scenario 5: Feature Consistency Between Training and Production
**Context:** Your team builds 50+ churn prediction models using customer features (age, purchase_frequency, lifetime_value). During development, each team recomputes these features differently. In production, served features sometimes differ from training features, causing model accuracy degradation.

**Scenario Question:**
Which Vertex AI service ensures features are consistent between training and serving?

**Options:**
- [ ] A) Vertex AI Datasets
- [ ] B) Vertex AI Model Registry
- [ ] C) Vertex AI Feature Store
- [ ] D) Vertex AI Pipelines

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Feature Store**: Centralized feature repository with offline (training) and online (serving) stores. Ensures same features train/serve. Prevents training-serving skew.
- ✗ A) Datasets: Stores raw data, not pre-computed features
- ✗ B) Model Registry: Stores model artifacts, not features
- ✗ D) Pipelines: Orchestrates workflows, doesn't manage features

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Prevent training-serving skew → **Feature Store**
- Centralized feature repository → **Feature Store**
- Multiple models sharing features → **Feature Store**
- Feature versioning & lineage → **Feature Store**

</details>

---

## Scenario 6: End-to-End ML Workflow Automation
**Context:** Your data science team manually performs: data preparation → feature engineering → model training → evaluation → deployment. This process takes 2 weeks for each model iteration. You need to automate this entire workflow to trigger daily.

**Scenario Question:**
Which Vertex AI service orchestrates this entire ML pipeline?

**Options:**
- [ ] A) Vertex AI Custom Training
- [ ] B) Vertex AI Pipelines
- [ ] C) Vertex AI Experiments
- [ ] D) Vertex AI AutoML

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Pipelines**: Orchestrates multi-step ML workflows using Kubeflow DAGs. Automates data prep → training → evaluation → deployment with scheduling.
- ✗ A) Custom Training: Runs training jobs, doesn't orchestrate workflows
- ✗ C) Experiments: Tracks runs, doesn't orchestrate pipelines
- ✗ D) AutoML: Only for training, not orchestration

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Multi-step workflow automation → **Pipelines**
- Scheduled recurring jobs → **Pipelines**
- Orchestrate components → **Pipelines**
- End-to-end ML automation → **Pipelines**

</details>

---

## Scenario 7: Model Explainability for Loan Approval Rejection
**Context:** Your bank deployed a model that automatically rejects loan applications. A customer challenges the decision and demands explanation. Your audit team needs to understand which features drove the rejection decision.

**Scenario Question:**
Which Vertex AI service provides feature attribution explanations?

**Options:**
- [ ] A) Vertex AI Model Monitoring
- [ ] B) Vertex AI Explainable AI (XAI)
- [ ] C) Vertex AI Experiments
- [ ] D) Vertex AI Model Registry

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI XAI**: Provides feature attribution (SHAP, Integrated Gradients) showing which features influenced predictions. Critical for compliance and debugging.
- ✗ A) Model Monitoring: Tracks performance drift, not explanations
- ✗ C) Experiments: Tracks runs and metrics, not feature importance
- ✗ D) Model Registry: Stores model versions, not explanations

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Feature attribution explanations → **XAI**
- Model interpretability → **XAI**
- Regulatory compliance (loan decisions, hiring) → **XAI**
- Debugging model behavior → **XAI**

</details>

---

## Scenario 8: Production Model Performance Monitoring & Drift Detection
**Context:** Your recommendation model has been in production for 3 months. Accuracy was 92% at launch, but today it's 78%. You suspect customer behavior changed (data drift) or model is overfitting to old patterns.

**Scenario Question:**
Which Vertex AI service detects and alerts on model performance degradation?

**Options:**
- [ ] A) Vertex AI Model Registry
- [ ] B) Vertex AI Experiments
- [ ] C) Vertex AI Model Monitoring
- [ ] D) Vertex AI TensorBoard

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Model Monitoring**: Continuous monitoring of deployed models. Detects performance degradation, data drift, prediction distribution changes. Triggers retraining alerts.
- ✗ A) Model Registry: Stores model versions, doesn't monitor performance
- ✗ B) Experiments: Tracks training runs, not production monitoring
- ✗ D) TensorBoard: Visualizes training metrics, not production monitoring

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Production model monitoring → **Model Monitoring**
- Data drift detection → **Model Monitoring**
- Performance degradation alerts → **Model Monitoring**
- Trigger retraining workflows → **Model Monitoring**

</details>

---

## Scenario 9: Quick Prototyping Without ML Expertise
**Context:** Your product manager wants a text classification model to categorize customer feedback (positive/negative/neutral). No ML experts available. Need to build and deploy in 2 days.

**Scenario Question:**
Which Vertex AI service enables non-experts to build models quickly?

**Options:**
- [ ] A) Vertex AI Custom Training
- [ ] B) Vertex AI AutoML
- [ ] C) Vertex AI Workbench
- [ ] D) Vertex AI Pipelines

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI AutoML**: No-code model training for text, images, tabular data. Automatic feature engineering, model selection, hyperparameter tuning. Perfect for quick prototyping by non-experts.
- ✗ A) Custom Training: Requires coding, ML expertise
- ✗ C) Workbench: Requires data science knowledge
- ✗ D) Pipelines: Requires ML workflow design knowledge

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- No-code model building → **AutoML**
- Rapid prototyping → **AutoML**
- Non-ML experts → **AutoML**
- Quick MVP development → **AutoML**

</details>

---

## Scenario 10: Tracking & Comparing Multiple Model Experiments
**Context:** Your data scientist is experimenting with different architectures (logistic regression, random forest, neural network) for churn prediction. Each run produces different metrics (accuracy, precision, recall, AUC). Team needs to track and compare all runs.

**Scenario Question:**
Which Vertex AI service tracks experiments and enables comparison?

**Options:**
- [ ] A) Vertex AI Model Registry
- [ ] B) Vertex AI Experiments
- [ ] C) Vertex AI Pipelines
- [ ] D) Vertex AI Batch Prediction

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Experiments**: Logs parameters, metrics, artifacts for each training run. Compares runs side-by-side in UI. Enables reproducibility and best-practice tracking.
- ✗ A) Model Registry: Stores final models, not experiment runs
- ✗ C) Pipelines: Orchestrates workflows, doesn't track runs
- ✗ D) Batch Prediction: Scores data, doesn't track experiments

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Experiment tracking & comparison → **Experiments**
- Parameter logging → **Experiments**
- Metric visualization → **Experiments**
- Run reproducibility → **Experiments**

</details>

---

## Scenario 11: Processing 100GB of Images for Object Detection
**Context:** You have 100GB of raw images that need preprocessing (resizing, normalization) and training a custom object detection model. Processing should run on GPUs in parallel.

**Scenario Question:**
Which Vertex AI service allows custom TensorFlow code with GPU distribution?

**Options:**
- [ ] A) Vertex AI AutoML
- [ ] B) Vertex AI Custom Training
- [ ] C) Vertex AI Pipelines
- [ ] D) Vertex AI Workbench

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Custom Training**: Full control over training code. Supports TensorFlow, PyTorch, scikit-learn. Distributed training on GPUs/TPUs. Perfect for complex workflows.
- ✗ A) AutoML: Limited to built-in architectures
- ✗ C) Pipelines: Orchestration tool, not training framework
- ✗ D) Workbench: Development IDE, not distributed training

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Custom ML code → **Custom Training**
- GPU/TPU distribution → **Custom Training**
- Complex preprocessing pipelines → **Custom Training**
- Framework flexibility → **Custom Training**

</details>

---

## Scenario 12: Creating Labeled Dataset from Raw Documents
**Context:** You have 10,000 invoices that need to be labeled (extract: invoice_id, amount, date, vendor_name). Your team manually extracts ~100 invoices/day, but at that pace, it will take 100 days to label all 10,000.

**Scenario Question:**
Which Vertex AI service accelerates data labeling?

**Options:**
- [ ] A) Vertex AI Datasets
- [ ] B) Vertex AI Workbench
- [ ] C) Vertex AI Data Labeling Service
- [ ] D) Vertex AI Document AI

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Data Labeling Service**: Managed labeling with human annotators. Integrates with active learning to prioritize hardest samples. Reduces labeling time by 50-70%.
- ✗ A) Datasets: Manages labeled data, doesn't label it
- ✗ B) Workbench: Development IDE, not labeling service
- ✗ D) Document AI: Extracts structured data from documents (different use case)

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Accelerate data labeling → **Data Labeling Service**
- Human annotation management → **Data Labeling Service**
- Active learning prioritization → **Data Labeling Service**
- Reduce labeling time & cost → **Data Labeling Service**

</details>

---

## Scenario 13: Extracting Data from Business Documents (Invoices, Contracts)
**Context:** Your company receives 1,000 invoices daily in PDF format. You need to extract structured data (amount, date, vendor, account_code) into a database automatically. Current manual process takes 4 hours/day.

**Scenario Question:**
Which Vertex AI service automates document data extraction?

**Options:**
- ( ) A) Vertex AI Workbench
- ( ) B) Vertex AI Document AI
- ( ) C) Vertex AI Data Labeling Service
- ( ) D) Vertex AI Datasets

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Document AI**: Pre-trained models for document parsing (invoices, receipts, contracts). Extracts structured fields automatically. OCR + table extraction built-in.
- ✗ A) Workbench: Development IDE
- ✗ C) Data Labeling Service: For labeling, not extraction
- ✗ D) Datasets: Data management, not extraction

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Document data extraction → **Document AI**
- OCR + structured data extraction → **Document AI**
- Invoice/receipt/contract processing → **Document AI**
- Automation of manual document processing → **Document AI**

</details>

---

## Scenario 14: Deploying to Edge Devices (Mobile, IoT)
**Context:** Your company builds a mobile app for offline object detection. Users need predictions even without internet. Model must run on-device with <100ms latency.

**Scenario Question:**
Which Vertex AI service supports edge deployment?

**Options:**
- ( ) A) Vertex AI Endpoints
- ( ) B) Vertex AI Batch Prediction
- ( ) C) Vertex AI Custom Training (with TFLite export)
- ( ) D) Vertex AI Workbench

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Custom Training**: Train model, export to TensorFlow Lite (TFLite) for mobile/edge deployment. On-device inference without server calls.
- ✗ A) Endpoints: Cloud-hosted, requires internet
- ✗ B) Batch Prediction: Offline batch processing, not edge
- ✗ D) Workbench: Development IDE only

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Edge device deployment (mobile, IoT) → **Custom Training + TFLite export**
- Offline inference → **TFLite on-device**
- Low-latency edge predictions → **Edge deployment**

</details>

---

## Scenario 15: Development Environment for Data Scientists
**Context:** Your data science team needs a Jupyter notebook environment pre-configured with TensorFlow, scikit-learn, and BigQuery access to start EDA immediately.

**Scenario Question:**
Which Vertex AI service provides managed Jupyter notebooks?

**Options:**
- ( ) A) Vertex AI Datasets
- ( ) B) Vertex AI Workbench
- ( ) C) Vertex AI Pipelines
- ( ) D) Vertex AI TensorBoard

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Workbench**: Managed Jupyter notebook environment. Pre-configured kernels, GCP service integration (BigQuery, GCS). Per-user or managed notebooks.
- ✗ A) Datasets: Data management, not development
- ✗ C) Pipelines: Workflow orchestration
- ✗ D) TensorBoard: Metrics visualization

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Jupyter notebook environment → **Workbench**
- Pre-configured ML libraries → **Workbench**
- GCP service integration → **Workbench**
- Rapid EDA setup → **Workbench**

</details>

---

## Scenario 16: Building Recommendation System with Vector Search
**Context:** Your e-commerce platform needs a recommendation engine based on semantic similarity of products. You have 1 million products, each represented as 768-dimensional embeddings. Need <100ms retrieval for top-10 recommendations.

**Scenario Question:**
Which Vertex AI service provides fast vector similarity search?

**Options:**
- ( ) A) Vertex AI Feature Store
- ( ) B) Vertex AI Matching Engine
- ( ) C) Vertex AI Batch Prediction
- ( ) D) Vertex AI Custom Training

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Matching Engine**: Vector database for approximate nearest neighbor search. Handles millions of vectors with <100ms retrieval. Perfect for semantic search & recommendations.
- ✗ A) Feature Store: For feature management, not vector search
- ✗ C) Batch Prediction: Asynchronous, not real-time
- ✗ D) Custom Training: For training, not serving

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Semantic/vector similarity search → **Matching Engine**
- Large-scale nearest neighbor search → **Matching Engine**
- Recommendation systems → **Matching Engine**
- Sub-100ms retrieval at scale → **Matching Engine**

</details>

---

## Scenario 17: Building Conversational Chatbot with LLMs
**Context:** Your company wants to build a customer support chatbot using Google's Gemini model. The chatbot should answer questions about your products using company documentation as context.

**Scenario Question:**
Which Vertex AI service enables RAG (Retrieval Augmented Generation) chatbots?

**Options:**
- ( ) A) Vertex AI Workbench
- ( ) B) Vertex AI Search & Conversation
- ( ) C) Vertex AI Custom Training
- ( ) D) Vertex AI Experiments

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Search & Conversation**: Build conversational search & RAG systems. Retrieves relevant docs, passes to LLM for grounded responses. No hallucinations from context.
- ✗ A) Workbench: Development IDE
- ✗ C) Custom Training: For model training
- ✗ D) Experiments: Experiment tracking

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Conversational AI with LLMs → **Search & Conversation**
- RAG (Retrieval Augmented Generation) → **Search & Conversation**
- Context-grounded responses → **Search & Conversation**
- Reduce LLM hallucinations → **Search & Conversation**

</details>

---

## Scenario 18: Fine-Tuning Gemini for Domain-Specific Tasks
**Context:** You want to improve Gemini model for your specific domain (healthcare/finance). Generic Gemini is too broad; you need domain-specific expertise.

**Scenario Question:**
Which Vertex AI service enables LLM fine-tuning?

**Options:**
- ( ) A) Vertex AI Generative AI with Prompt Tuning
- ( ) B) Vertex AI Prompt Engineering
- ( ) C) Vertex AI Custom Training
- ( ) D) Vertex AI AutoML

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Prompt Engineering**: Tune and optimize prompts for LLMs. Adjust parameters like temperature, top-K, top-P. Also supports instruction tuning and fine-tuning on domain data.
- ✗ A) Generative AI: Access to models, not fine-tuning
- ✗ C) Custom Training: More complex, for full model training
- ✗ D) AutoML: For non-LLM models

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Prompt optimization → **Prompt Engineering**
- Instruction tuning → **Prompt Engineering**
- Domain-specific LLM adaptation → **Prompt Engineering**
- Improve LLM output quality → **Prompt Engineering**

</details>

---

## Scenario 19: Time-Series Forecasting (Demand, Sales, Revenue)
**Context:** Your retail company needs to forecast daily sales for next 30 days to optimize inventory. You have 2 years of daily sales history with seasonality (weekly, yearly patterns).

**Scenario Question:**
Which Vertex AI service specializes in time-series forecasting?

**Options:**
- ( ) A) Vertex AI AutoML
- ( ) B) Vertex AI Forecasting
- ( ) C) Vertex AI Custom Training
- ( ) D) Vertex AI Pipelines

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Forecasting**: Dedicated service for time-series forecasting. Handles seasonality, trends, auto ARIMA, Prophet, neural networks. Outputs confidence intervals.
- ✗ A) AutoML: General-purpose, not time-series optimized
- ✗ C) Custom Training: More work, less specialized
- ✗ D) Pipelines: Orchestration, not forecasting

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Time-series forecasting → **Forecasting**
- Seasonality & trend handling → **Forecasting**
- Demand/sales/revenue prediction → **Forecasting**
- Auto model selection → **Forecasting**

</details>

---

## Scenario 20: Model Performance Analysis Before Deployment
**Context:** You've trained a model and now need to evaluate it thoroughly before production deployment. You need accuracy, precision, recall, confusion matrix, fairness metrics (disparate impact), and class distribution insights.

**Scenario Question:**
Which Vertex AI service provides comprehensive model evaluation?

**Options:**
- ( ) A) Vertex AI Experiments
- ( ) B) Vertex AI Model Evaluation
- ( ) C) Vertex AI TensorBoard
- ( ) D) Vertex AI Model Registry

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Model Evaluation**: Automated model evaluation with classification/regression/NLP metrics. Fairness checks, class imbalance analysis, evaluation reports.
- ✗ A) Experiments: Tracks runs, not comprehensive evaluation
- ✗ C) TensorBoard: Visualizes training metrics
- ✗ D) Model Registry: Stores models, doesn't evaluate

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Pre-deployment model evaluation → **Model Evaluation**
- Comprehensive metrics + fairness → **Model Evaluation**
- Quality gate before production → **Model Evaluation**
- Identify class imbalance issues → **Model Evaluation**

</details>

---

## Scenario 21: Multi-Step AI Reasoning for Complex Problems
**Context:** Your company needs to build a system that can solve complex, multi-step problems like "Recommend products based on user profile, inventory availability, historical purchases, and current trends." Requires reasoning beyond single inference.

**Scenario Question:**
Which Vertex AI service enables multi-step reasoning?

**Options:**
- ( ) A) Vertex AI Endpoints
- ( ) B) Vertex AI Reasoning Engine
- ( ) C) Vertex AI Feature Store
- ( ) D) Vertex AI Batch Prediction

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Reasoning Engine**: Build reasoning systems for complex LLM tasks. Multi-step planning, long-context processing, tool orchestration. Decompose hard problems into steps.
- ✗ A) Endpoints: Single inference only
- ✗ C) Feature Store: Feature management
- ✗ D) Batch Prediction: Batch scoring

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Multi-step reasoning → **Reasoning Engine**
- Complex problem decomposition → **Reasoning Engine**
- Tool orchestration → **Reasoning Engine**
- Long-context processing → **Reasoning Engine**

</details>

---

## Scenario 22: Visualizing Training Metrics in Real-Time
**Context:** Your custom TensorFlow training job is running. You need to monitor loss curves, accuracy progression, layer activations, and gradient distributions in real-time without connecting to the server via SSH.

**Scenario Question:**
Which Vertex AI service visualizes training metrics?

**Options:**
- ( ) A) Vertex AI Experiments
- ( ) B) Vertex AI TensorBoard
- ( ) C) Vertex AI Workbench
- ( ) D) Vertex AI Model Monitoring

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI TensorBoard**: Real-time visualization of training metrics. Loss curves, accuracy plots, histograms, gradient distributions, layer activations. Cloud-hosted access.
- ✗ A) Experiments: Logs metrics, doesn't real-time visualize
- ✗ C) Workbench: Development IDE
- ✗ D) Model Monitoring: Production monitoring only

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Real-time training visualization → **TensorBoard**
- Loss curves & accuracy plots → **TensorBoard**
- Debugging training progression → **TensorBoard**
- Distributed training visualization → **TensorBoard**

</details>

---

## Scenario 23: Centralizing Trained Model Artifacts
**Context:** Your team has 50+ trained models across different projects. Need a single registry to track versions, metadata (accuracy, training date, framework), and deployment history.

**Scenario Question:**
Which Vertex AI service manages model artifacts and versions?

**Options:**
- ( ) A) Vertex AI Datasets
- ( ) B) Vertex AI Model Registry
- ( ) C) Vertex AI Workbench
- ( ) D) Vertex AI Custom Training

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Model Registry**: Centralized repository for trained models. Version control, metadata tracking, lineage, deployment history. Integrates with Endpoints for serving.
- ✗ A) Datasets: Data management, not models
- ✗ C) Workbench: Development IDE
- ✗ D) Custom Training: Training service, not registry

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Model versioning & registry → **Model Registry**
- Metadata tracking → **Model Registry**
- Model governance → **Model Registry**
- Deployment lineage → **Model Registry**

</details>

---

## Scenario 24: Autonomous AI Agents for Workflow Automation
**Context:** Your company wants to automate customer support workflows: receive ticket → categorize → route to department → generate response → schedule follow-up. This requires multi-step reasoning, tool access, and autonomous decision-making.

**Scenario Question:**
Which Vertex AI service builds autonomous agents?

**Options:**
- ( ) A) Vertex AI Pipelines
- ( ) B) Vertex AI Agent Builder
- ( ) C) Vertex AI Search & Conversation
- ( ) D) Vertex AI Custom Training

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Agent Builder**: Create autonomous agents that can reason, plan, and execute multi-step tasks. Integrates with tools/APIs. Used for workflow automation, customer support, sales.
- ✗ A) Pipelines: Workflow orchestration, not autonomous agents
- ✗ C) Search & Conversation: Chatbots, not autonomous agents
- ✗ D) Custom Training: Model training only

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Autonomous agents → **Agent Builder**
- Multi-step workflow automation → **Agent Builder**
- Tool/API integration → **Agent Builder**
- Intelligent decision-making → **Agent Builder**

</details>

---

## Scenario 25: Computer Vision - Image Classification
**Context:** Your manufacturing company has 100,000 labeled images of defective vs. non-defective products. You need to build an image classifier to automatically catch defects on the production line.

**Scenario Question:**
Which Vertex AI service should you use?

**Options:**
- ( ) A) Vertex AI AutoML (Vision)
- ( ) B) Vertex AI Custom Training (Alternative for more control)
- ( ) C) Vertex AI Workbench
- ( ) D) Vertex AI Forecasting

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **AutoML (Vision)**: Pre-built for image classification. Auto feature extraction, model selection, hyperparameter tuning. With 100K labeled images, achieves high accuracy quickly.
- ✓ **Custom Training**: Also valid if you need model architecture control (CNNs, ResNet, EfficientNet)
- ✗ C) Workbench: Development IDE
- ✗ D) Forecasting: Time-series only

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Image classification with labeled data → **AutoML (Vision)**
- No-code image model → **AutoML**
- Production deployment on Edge → **Custom Training + TFLite**

</details>

---

## Scenario 26: NLP - Sentiment Analysis at Scale
**Context:** Your company receives 10 million customer reviews daily. You need to classify sentiment (positive/negative/neutral) and extract key phrases for each review.

**Scenario Question:**
Which approach best fits this scenario?

**Options:**
- ( ) A) Vertex AI AutoML (Text)
- ( ) B) Vertex AI Generative AI (Gemini) with Prompt Engineering
- ( ) C) Vertex AI Custom Training
- ( ) D) Vertex AI Workbench

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Generative AI**: Use Gemini with few-shot prompting for sentiment analysis and key phrase extraction. No retraining needed, flexible, handles complex NLP tasks.
- ✓ **AutoML (Text)**: Also valid if you want dedicated fine-tuning on your domain
- ✗ C) Custom Training: Over-engineering for this task
- ✗ D) Workbench: Development IDE

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Scale NLP with LLMs → **Generative AI + Prompt Engineering**
- Few-shot learning → **Generative AI**
- Sentiment analysis at scale → **Generative AI or AutoML**

</details>

---

## Scenario 27: Video Analysis - Action Recognition
**Context:** Your sports app needs to classify actions in short video clips (jump, run, kick, throw) to provide real-time coaching feedback. You have 50,000 labeled video clips.

**Scenario Question:**
Which Vertex AI service handles video classification?

**Options:**
- ( ) A) Vertex AI AutoML (Video)
- ( ) B) Vertex AI Custom Training
- ( ) C) Vertex AI Video Service
- ( ) D) Vertex AI Batch Prediction

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Vertex AI Video**: Specialized service for video analysis. Action recognition, scene detection, object tracking across frames. Pre-trained models for quick deployment.
- ✓ **AutoML (Video)**: Also good option with labeled video data
- ✗ C) Custom Training: More work for temporal modeling
- ✗ D) Batch Prediction: For scoring, not model building

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Video classification/action recognition → **Video or AutoML (Video)**
- Temporal pattern analysis → **Video**
- Sports/action detection → **Video**

</details>

---

## Scenario 28: Tabular Data Classification - Customer Segmentation
**Context:** You have customer data (age, income, purchase_history, location, device_type) and want to predict customer segment (high-value, regular, at-risk). Data is in CSV with 100K rows.

**Scenario Question:**
Which Vertex AI service is best for tabular data?

**Options:**
- ( ) A) Vertex AI AutoML (Tabular)
- ( ) B) Vertex AI Custom Training
- ( ) C) Vertex AI Forecasting
- ( ) D) Vertex AI Workbench

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **AutoML (Tabular)**: Designed for tabular/structured data. Auto feature engineering, handles categorical/numerical features, missing values. No coding needed.
- ✗ B) Custom Training: Overkill for simple tabular classification
- ✗ C) Forecasting: Time-series only
- ✗ D) Workbench: Development IDE

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Tabular data classification/regression → **AutoML (Tabular)**
- Quick model building from CSV → **AutoML**
- Feature engineering automation → **AutoML**

</details>

---

## Scenario 29: Multi-Modal Model - Image + Text Generation
**Context:** Your content generation platform needs to generate creative product descriptions given product images and category. Task requires understanding both image and text context.

**Scenario Question:**
Which Vertex AI service handles multi-modal tasks?

**Options:**
- ( ) A) Vertex AI AutoML
- ( ) B) Vertex AI Generative AI (Gemini)
- ( ) C) Vertex AI Custom Training
- ( ) D) Vertex AI Forecasting

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Generative AI (Gemini)**: Multi-modal model (image + text input). Generate creative content based on visual and textual context. Perfect for image→text generation.
- ✗ A) AutoML: Single modality only
- ✗ C) Custom Training: More complex for multi-modal
- ✗ D) Forecasting: Time-series only

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Multi-modal tasks (image + text) → **Generative AI**
- Image-to-text generation → **Generative AI**
- Creative content generation → **Generative AI**

</details>

---

## Scenario 30: Critical Production Issue - Model Accuracy Dropped 15%
**Context:** Your deployed recommendation model had 88% accuracy. Today, accuracy is 73%. Customer complaints increased. You need to quickly identify root cause (data drift, model degradation, feature quality).

**Scenario Question:**
Which Vertex AI service helps diagnose the issue?

**Options:**
- ( ) A) Vertex AI Model Monitoring (primary diagnosis)
- ( ) B) Vertex AI Explainable AI (secondary - understand features)
- ( ) C) Vertex AI Experiments (compare against baseline)
- ( ) D) All of above

<details>
<summary><strong>Explanation</strong></summary>

- ✓ **Model Monitoring**: Detects data drift, performance degradation. Shows feature distribution changes, prediction volume changes.
- ✓ **XAI**: Identifies which features changed most, causing predictions to shift.
- ✓ **Experiments**: Compare current model against previous baseline to understand regression.
- **Best Answer: A (Model Monitoring)** for immediate diagnosis, then B & C for deeper analysis

</details>

<details>
<summary><strong>Key Takeaways</strong></summary>

- Production issue diagnosis → **Model Monitoring**
- Data drift detection → **Model Monitoring**
- Root cause analysis → **Model Monitoring + XAI**
- Rollback decision → **Experiments**

</details>

---

# Answer Summary Table

| Scenario | Service | When to Use |
|----------|---------|------------|
| 1. Real-time API predictions | **Vertex AI Endpoints** | <100ms latency, auto-scaling, REST API |
| 2. Batch scoring 500M records | **Vertex AI Batch Prediction** | Large-scale asynchronous processing |
| 3. Labeled data organization | **Vertex AI Datasets** | Data preparation for training |
| 4. Hyperparameter optimization | **Vertex AI HPO** | Systematic parameter search |
| 5. Feature consistency | **Vertex AI Feature Store** | Prevent training-serving skew |
| 6. ML workflow automation | **Vertex AI Pipelines** | End-to-end pipeline orchestration |
| 7. Model explainability | **Vertex AI XAI** | Feature attribution, compliance |
| 8. Production monitoring | **Vertex AI Model Monitoring** | Drift detection, performance tracking |
| 9. Quick prototyping | **Vertex AI AutoML** | No-code model building |
| 10. Experiment tracking | **Vertex AI Experiments** | Compare runs and metrics |
| 11. Custom GPU training | **Vertex AI Custom Training** | Framework flexibility, distributed training |
| 12. Accelerate labeling | **Vertex AI Data Labeling** | Human annotation management |
| 13. Document extraction | **Vertex AI Document AI** | Invoice/contract data extraction |
| 14. Edge deployment | **Custom Training + TFLite** | On-device inference |
| 15. Jupyter notebooks | **Vertex AI Workbench** | Development environment |
| 16. Vector search | **Vertex AI Matching Engine** | Recommendation systems, semantic search |
| 17. Chatbots with context | **Vertex AI Search & Conversation** | RAG-based conversational AI |
| 18. LLM fine-tuning | **Vertex AI Prompt Engineering** | Domain-specific LLM adaptation |
| 19. Time-series forecasting | **Vertex AI Forecasting** | Demand/sales prediction |
| 20. Model evaluation | **Vertex AI Model Evaluation** | Pre-deployment QA, fairness checks |
| 21. Multi-step reasoning | **Vertex AI Reasoning Engine** | Complex problem decomposition |
| 22. Training visualization | **Vertex AI TensorBoard** | Real-time metrics monitoring |
| 23. Model versioning | **Vertex AI Model Registry** | Artifact management |
| 24. Autonomous agents | **Vertex AI Agent Builder** | Workflow automation, decision-making |
| 25. Image classification | **Vertex AI AutoML (Vision)** | No-code image models |
| 26. Sentiment analysis | **Vertex AI Generative AI** | Scale NLP with LLMs |
| 27. Video analysis | **Vertex AI Video** | Action recognition, scene detection |
| 28. Tabular data | **Vertex AI AutoML (Tabular)** | Structured data classification |
| 29. Multi-modal generation | **Vertex AI Generative AI** | Image + text tasks |
| 30. Production diagnosis | **Vertex AI Model Monitoring + XAI** | Root cause analysis |

---

# Quick Decision Guide

## Choose Based on Task Type:

| Task | Service(s) |
|------|-----------|
| **Data Preparation** | Datasets, Workbench, Data Labeling |
| **Model Building** | AutoML, Custom Training, Generative AI |
| **Optimization** | HPO, Experiments, Prompt Engineering |
| **Serving** | Endpoints, Batch Prediction, Search & Conversation |
| **Orchestration** | Pipelines, Agent Builder |
| **Feature Management** | Feature Store |
| **Monitoring & Maintenance** | Model Monitoring, Model Evaluation, XAI |
| **Specialized Domains** | Forecasting, Document AI, Video, Matching Engine |

## Choose Based on Expertise Level:

| Level | Service |
|-------|---------|
| **No ML experience** | AutoML, Generative AI, Agent Builder |
| **Some ML experience** | Custom Training, Pipelines, Feature Store |
| **ML expert** | Custom Training, Workbench, Reasoning Engine |

## Choose Based on Scale:

| Scale | Service |
|-------|---------|
| **<1GB data, <1 week** | AutoML, Generative AI |
| **1-100GB, weeks-months** | Custom Training, Pipelines |
| **>100GB, 24/7 production** | Endpoints, Feature Store, Model Monitoring |
