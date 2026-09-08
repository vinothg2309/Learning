# BigQuery ML — Capabilities & Complete Command Reference

## What BigQuery ML lets you do

- **Build ML models using plain SQL** — no need to move data out to Python/Vertex AI notebooks; train where your data already lives.
- **Broad model coverage** — regression, classification, clustering, time-series forecasting, recommendation, anomaly detection, dimensionality reduction (PCA), and deep learning (DNN, Boosted Trees, AutoML, Wide-and-Deep).
- **Generative AI, built in** — call Gemini models directly from SQL for text generation, embeddings, summarization, translation, transcription, and document/table understanding via **remote models**.
- **Bring your own model** — import externally-trained **TensorFlow, ONNX, or XGBoost** models and run inference on them inside BigQuery.
- **Built-in feature engineering** — scaling, bucketizing, one-hot encoding, imputation, and feature crosses via SQL functions, so preprocessing lives in the same place as training.
- **Explainability out of the box** — per-prediction and global feature attributions without extra tooling.
- **Hyperparameter tuning** — automatic trials via `NUM_TRIALS`, with functions to inspect trial results.
- **MLOps-ready** — models register in **BigQuery ML Model Registry**, and pipelines can be orchestrated via Cloud Build / Vertex AI Pipelines for CI/CD/CT.
- **Cost model** — you pay for the BigQuery compute used to train/query, plus storage for the model; external/remote model calls (e.g., Gemini) also incur separate charges.

*(Note: Google's docs are transitioning from "Vertex AI" naming to "Gemini Enterprise Agent Platform" for the model-serving side that BigQuery ML connects to — you may see both terms.)*

---

## 1. Model Lifecycle — DDL Statements

| Command | Description |
|---|---|
| `CREATE MODEL` | Trains a new model using a SQL query as the training data source. |
| `CREATE MODEL IF NOT EXISTS` | Same as above, but skips training if a model with that name already exists. |
| `CREATE OR REPLACE MODEL` | Trains a model and overwrites any existing model of the same name. |
| `ALTER MODEL SET OPTIONS` | Updates metadata/options (e.g., expiration time, labels) of an existing model without retraining. |
| `EXPORT MODEL` | Exports a trained BigQuery ML model (e.g., to Cloud Storage) for use outside BigQuery, such as serving via Vertex AI or TensorFlow Serving. |
| `DROP MODEL` | Deletes a model from a dataset. |
| `DROP MODEL IF EXISTS` | Deletes a model only if it exists (no error if missing). |

---

## 2. Training, Evaluation & Inspection Functions

| Function | Description |
|---|---|
| `ML.EVALUATE` | Computes evaluation metrics (precision, recall, RMSE, log loss, ROC AUC, etc.) for a trained model, either on the original training/eval split or new data. |
| `ML.TRAINING_INFO` | Returns training statistics per iteration — loss, duration, learning rate — useful for diagnosing under/overfitting. |
| `ML.FEATURE_INFO` | Returns summary statistics (min, max, mean, null counts) for each input feature used in training. |
| `ML.WEIGHTS` | Returns the trained model's learned weights/coefficients (for linear/logistic regression, matrix factorization, etc.). |
| `ML.ADVANCED_WEIGHTS` | Returns detailed weight information including category/vocabulary mappings for categorical features. |
| `ML.CONFUSION_MATRIX` | Returns a confusion matrix for classification models to inspect true/false positive and negative counts. |
| `ML.ROC_CURVE` | Returns precision/recall/true-positive-rate values across thresholds for binary classification models. |
| `ML.STUDY_INFO` | Returns hyperparameter-tuning trial summary information when `NUM_TRIALS` was used. |
| `ML.TRIAL_INFO` | Returns detailed results for each individual hyperparameter-tuning trial. |

---

## 3. Prediction / Inference Functions

| Function | Description |
|---|---|
| `ML.PREDICT` | Runs inference using a trained model on new input data; returns predicted labels/values. |
| `ML.FORECAST` | Generates future time-point forecasts for `ARIMA_PLUS` / `ARIMA_PLUS_XREG` time-series models. |
| `ML.DETECT_ANOMALIES` | Flags anomalous rows/time points using a trained model (supports several model types, including ARIMA and clustering models). |
| `ML.RECOMMEND` | Generates recommendations from a trained matrix factorization model. |
| `ML.CENTROIDS` | Returns cluster centroid information for k-means clustering models. |
| `ML.RECONSTRUCTION_LOSS` | Returns reconstruction loss values, used for anomaly detection with autoencoder/PCA models. |
| `ML.PRINCIPAL_COMPONENTS` | Returns the principal component vectors from a trained PCA model. |
| `ML.PRINCIPAL_COMPONENT_INFO` | Returns explained-variance information for each principal component in a PCA model. |
| `ML.ARIMA_COEFFICIENTS` | Returns the fitted ARIMA model coefficients. |
| `ML.ARIMA_EVALUATE` | Evaluates candidate ARIMA models generated during automatic model selection. |

---

## 4. Explainability Functions

| Function | Description |
|---|---|
| `ML.EXPLAIN_PREDICT` | Returns predictions along with per-feature attribution scores explaining each individual prediction. |
| `ML.EXPLAIN_FORECAST` | Returns forecasted values decomposed into trend, seasonality, and holiday-effect components. |
| `ML.GLOBAL_EXPLAIN` | Returns overall (dataset-level) feature importance for a trained model. |

---

## 5. Feature Engineering / Preprocessing Functions

| Function | Description |
|---|---|
| `ML.STANDARD_SCALER` | Standardizes a numeric column to zero mean and unit variance. |
| `ML.MIN_MAX_SCALER` | Scales a numeric column to a 0–1 range. |
| `ML.MAX_ABS_SCALER` | Scales a numeric column by its maximum absolute value. |
| `ML.BUCKETIZE` | Buckets a numeric column into discrete ranges you define. |
| `ML.QUANTILE_BUCKETIZE` | Buckets a numeric column into equally-sized quantile-based bins. |
| `ML.POLYNOMIAL_EXPAND` | Generates polynomial and interaction features from a set of numeric columns. |
| `ML.FEATURE_CROSS` | Creates a combined categorical feature by crossing two or more categorical columns. |
| `ML.NGRAMS` | Generates n-grams from an array of text tokens, for text feature engineering. |
| `ML.HASH_BUCKETIZE` | Hashes a categorical column into a fixed number of buckets. |
| `ML.LABEL_ENCODER` | Converts categorical values into integer-encoded labels. |
| `ML.ONE_HOT_ENCODER` | Converts a categorical column into one-hot encoded vectors. |
| `ML.IMPUTER` | Fills missing values in a column using mean, median, or most-frequent-value strategies. |
| `ML.NORMALIZER` | Normalizes a numeric vector to unit norm. |
| `ML.DECODE_IMAGE` | Converts image bytes (from an object table) into a numeric array usable as model input. |
| `ML.RESIZE_IMAGE` | Resizes decoded image data to specified dimensions. |
| `ML.CONVERT_COLOR_SPACE` | Converts image color space (e.g., RGB to grayscale) during preprocessing. |
| `ML.CONVERT_IMAGE_TYPE` | Converts image data between supported image encodings. |
| `ML.EXTRACT_PATCHES` | Extracts fixed-size patches from image data for patch-based model input. |

---

## 6. Generative AI / Remote Model Functions

| Function | Description |
|---|---|
| `ML.GENERATE_TEXT` | Generates text (summaries, responses, content) using a remote Gemini/LLM model referenced from BigQuery. |
| `ML.GENERATE_EMBEDDING` | Generates vector embeddings for text or multimodal data, useful for semantic search and vector similarity. |
| `ML.GENERATE_TABLE` | Generates structured, tabular output from an LLM based on a prompt/schema. |
| `ML.UNDERSTAND_TEXT` | Sends text to a Cloud Natural Language remote model for entity extraction, sentiment, and syntax analysis. |
| `ML.UNDERSTAND_TABLE` | Uses an LLM to analyze and describe structured table data. |
| `ML.TRANSLATE` | Translates text using a remote Cloud Translation model. |
| `ML.TRANSCRIBE` | Transcribes audio data using a remote Speech-to-Text model. |
| `ML.PROCESS_DOCUMENT` | Extracts structured information from documents (PDFs, forms) via Document AI remote models. |

---

## Quick example tying it together

```sql
-- 1. Train
CREATE OR REPLACE MODEL `mydataset.churn_model`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['churned']) AS
SELECT * FROM `mydataset.customer_features`;

-- 2. Evaluate
SELECT * FROM ML.EVALUATE(MODEL `mydataset.churn_model`);

-- 3. Predict
SELECT * FROM ML.PREDICT(MODEL `mydataset.churn_model`,
  (SELECT * FROM `mydataset.new_customers`));

-- 4. Explain
SELECT * FROM ML.EXPLAIN_PREDICT(MODEL `mydataset.churn_model`,
  (SELECT * FROM `mydataset.new_customers`), STRUCT(3 AS top_k_features));
```

Want me to build a hands-on notebook that walks through each of these (train → evaluate → predict → explain → a Gemini `ML.GENERATE_TEXT` example) the same way we've done for your other cert-prep material?

---

# BigQuery ML - Step-by-Step Guide (Based on Attached File Only)

---

# Step 1: Explore the Dataset

### Purpose

Prepare the dataset that will be used for model training.

### What this query does

* Creates **label** (1 = purchase, 0 = no purchase)
* Extracts important features:

  * Operating System
  * Mobile/Desktop
  * Country
  * Page Views
* Reads Google Analytics public dataset
* Uses data from **2016-08-01 to 2017-06-31**
* Limits output to **10,000 records**

### SQL

```sql
SELECT
  IF(totals.transactions IS NULL, 0, 1) AS label,
  IFNULL(device.operatingSystem, "") AS os,
  device.isMobile AS is_mobile,
  IFNULL(geoNetwork.country, "") AS country,
  IFNULL(totals.pageviews, 0) AS pageviews
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20160801' AND '20170631'
LIMIT 10000;
```

### Save Output

Save as View:

```
bqml_lab.training_data
```

---

# Step 2: Create Machine Learning Model

### Purpose

Train a Binary Classification model that predicts whether a visitor will purchase.

### Model Details

* Model Name: `sample_model`
* Algorithm: Logistic Regression
* Target Column: `label`
* Training Data: `training_data`

### SQL

```sql
CREATE MODEL
  `PROJECT_ID.bqml_lab.sample_model`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['label']
) AS

SELECT
  label,
  os,
  is_mobile,
  country,
  pageviews
FROM
  `PROJECT_ID.bqml_lab.training_data`;
```

> Replace `PROJECT_ID` with your Google Cloud Project ID.

---

# Step 3: Evaluate Model Performance

### Purpose

Measure how well the trained model performs using **ML.EVALUATE**.

### SQL

```sql
SELECT
  *
FROM
ML.EVALUATE(
  MODEL `PROJECT_ID.bqml_lab.sample_model`,
  TABLE `PROJECT_ID.bqml_lab.training_data`
);
```

### Output

Returns evaluation metrics such as:

* Precision
* Recall
* Accuracy
* F1 Score
* Log Loss
* ROC AUC

---

# Step 4: Prepare Prediction Dataset

### Purpose

Load unseen (July 2017) data for prediction.

### What changes?

* Same features as training data
* Adds **fullVisitorId**
* Uses new date range:

  * 2017-07-01
  * 2017-08-01

### SQL

```sql
SELECT
  IF(totals.transactions IS NULL, 0, 1) AS label,
  IFNULL(device.operatingSystem, "") AS os,
  device.isMobile AS is_mobile,
  IFNULL(geoNetwork.country, "") AS country,
  IFNULL(totals.pageviews, 0) AS pageviews,
  fullVisitorId
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20170701' AND '20170801';
```

### Save Output

Save as View:

```
bqml_lab.july_data
```

---

# Step 5: Predict Purchases

### Purpose

Use the trained model to predict visitor purchases.

### Correct SQL

```sql
SELECT
  country,
  SUM(predicted_label) AS total_predicted_purchases
FROM
ML.PREDICT(
  MODEL `bqml_lab.sample_model`,
  (
    SELECT *
    FROM `bqml_lab.july_data`
  )
)
GROUP BY
  country
ORDER BY
  total_predicted_purchases DESC
LIMIT 10;
```

### Output

Returns:

* Country
* Total predicted purchases
* Top 10 countries with highest predicted purchases

---

# Debugging Example (From Lab)

### Incorrect Query

```sql
SELECT
  country,
  TOTAL(predicted_label) AS total_predicted_purchases
FROM
ML.PREDICT(
  MODEL `bqml_lab.sample_model`,
  (
    SELECT *
    FROM `bqml_lab.july_data`
  )
)
GROUP BY country
ORDER BY total_predicted_purchases DESC
LIMIT 10;
```

### Error

```
Function not found: TOTAL
```

### Fix

Replace:

```sql
TOTAL(predicted_label)
```

with

```sql
SUM(predicted_label)
```

---

# BigQuery ML Workflow Summary

| Step | Objective                       | SQL Used                        |
| ---- | ------------------------------- | ------------------------------- |
| 1    | Prepare training dataset        | `SELECT`                        |
| 2    | Train Logistic Regression model | `CREATE MODEL`                  |
| 3    | Evaluate model                  | `ML.EVALUATE`                   |
| 4    | Prepare prediction dataset      | `SELECT`                        |
| 5    | Make predictions                | `ML.PREDICT`                    |
| 6    | Aggregate predictions           | `SUM()`, `GROUP BY`, `ORDER BY` |

---

# Invoking BigQueryML from App

## Option 1 (Recommended): Execute `ML.PREDICT` SQL from Python

This is the standard production approach. Your Python application submits a SQL query to BigQuery, and BigQuery executes the prediction inside the data warehouse.

### Step 1: Install SDK

```bash
pip install google-cloud-bigquery
```

---

### Step 2: Authenticate

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/service-account.json
```

or

```python
from google.cloud import bigquery

client = bigquery.Client(project="my-project")
```

---

### Step 3: Execute Prediction Query

```python
from google.cloud import bigquery

client = bigquery.Client()

query = """
SELECT
    *
FROM
ML.PREDICT(
    MODEL `my-project.bqml_lab.sample_model`,
    (
        SELECT
            "Android" AS os,
            TRUE AS is_mobile,
            "India" AS country,
            12 AS pageviews
    )
)
"""

job = client.query(query)

results = job.result()

for row in results:
    print(row)
```

Output

```
predicted_label = 1
predicted_label_probs = [...]
```

---

## Option 2: Predict Multiple Records

Suppose your application receives a list of users.

```python
query = """
SELECT *
FROM ML.PREDICT(
MODEL `my-project.bqml_lab.sample_model`,
(
SELECT * FROM UNNEST([
STRUCT(
"Android" AS os,
TRUE AS is_mobile,
"India" AS country,
10 AS pageviews
),
STRUCT(
"iOS",
FALSE,
"USA",
4
)
])
)
)
"""
```

BigQuery predicts all records in one request.

---

## Option 3: Predict from Existing BigQuery Table

If data already exists in BigQuery:

```python
query = """
SELECT *
FROM ML.PREDICT(
MODEL `my-project.bqml_lab.sample_model`,
TABLE `my-project.dataset.july_data`
)
"""
```

---

## Parameterized Query (Recommended)

Instead of string formatting SQL:

```python
query = """
SELECT *
FROM ML.PREDICT(
MODEL `my-project.bqml_lab.sample_model`,
(
SELECT
@os AS os,
@is_mobile AS is_mobile,
@country AS country,
@pageviews AS pageviews
)
)
"""

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("os", "STRING", "Android"),
        bigquery.ScalarQueryParameter("is_mobile", "BOOL", True),
        bigquery.ScalarQueryParameter("country", "STRING", "India"),
        bigquery.ScalarQueryParameter("pageviews", "INT64", 8),
    ]
)

result = client.query(query, job_config=job_config).result()

for row in result:
    print(row.predicted_label)
```

This avoids SQL injection and is the preferred approach.

---

## Calling from a FastAPI Endpoint

```python
from fastapi import FastAPI
from google.cloud import bigquery

app = FastAPI()
client = bigquery.Client()

@app.post("/predict")
def predict(data: dict):

    query = f"""
    SELECT *
    FROM ML.PREDICT(
    MODEL `my-project.bqml_lab.sample_model`,
    (
    SELECT
    '{data['os']}' AS os,
    {str(data['is_mobile']).upper()} AS is_mobile,
    '{data['country']}' AS country,
    {data['pageviews']} AS pageviews
    )
    )
    """

    rows = client.query(query).result()

    return [dict(row.items()) for row in rows]
```

In production, replace the f-string with the parameterized query shown earlier.

---

# Production Architecture

```text
Client
   │
   ▼
FastAPI / Flask
   │
   ▼
google-cloud-bigquery SDK
   │
   ▼
BigQuery
   │
   ├── ML.PREDICT()
   ├── Reads BQML model
   └── Returns predictions
   │
   ▼
Python API Response
```

## Best Practices

* Use a **service account** with the `BigQuery Job User` and `BigQuery Data Viewer` roles, plus permission to access the model.
* Use **parameterized queries** instead of string interpolation.
* For high throughput, send **multiple records** in one `ML.PREDICT` call rather than issuing one query per record.
* If predictions are latency-sensitive, consider batching requests or using a dedicated online prediction service (such as a model deployed on Vertex AI Prediction). BQML is excellent for batch and analytics-oriented inference but incurs BigQuery query startup overhead for each request.
---

