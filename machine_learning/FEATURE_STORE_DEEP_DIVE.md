# Feature Store Deep Dive: What Gets Stored & Prediction-Time Transformation

**Duration:** 20 minutes  
**Focus:** Feature Store architecture, artifacts, and real-world transformation handling

---

## Table of Contents

1. [What Gets Stored in Feature Store](#what-gets-stored-in-feature-store)
2. [Online vs. Offline Stores](#online-vs-offline-stores)
3. [Feature Definitions & Versioning](#feature-definitions--versioning)
4. [Prediction-Time Transformation](#prediction-time-transformation)
5. [Training-Time vs. Serving-Time Consistency](#training-time-vs-serving-time-consistency)
6. [Real-World Examples](#real-world-examples)
7. [Common Pitfalls](#common-pitfalls)

---

## What Gets Stored in Feature Store

### Architecture Overview

```
┌──────────────────────────────────────────────────┐
│         Feature Store (Central Repository)       │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │ Feature Definitions (Schemas)           │   │
│  ├─────────────────────────────────────────┤   │
│  │ user_id: String (entity key)            │   │
│  │ age: Int (categorical range: 0-150)     │   │
│  │ income: Float (continuous, USD)         │   │
│  │ purchase_count_7d: Int (count)          │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │ Computed Feature Values                 │   │
│  ├─────────────────────────────────────────┤   │
│  │ Online Store (Redis/Spanner):           │   │
│  │   user:123 → {age: 30, income: 80000}   │   │
│  │   user:456 → {age: 45, income: 120000}  │   │
│  │   [Latest values, <10ms lookup]         │   │
│  │                                         │   │
│  │ Offline Store (BigQuery/Snowflake):     │   │
│  │   user:123 → {age: 30, income: 80000}   │   │
│  │   user:456 → {age: 45, income: 120000}  │   │
│  │   [Historical values, bulk queries]     │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │ Transformation Code                     │   │
│  ├─────────────────────────────────────────┤   │
│  │ def compute_age():                      │   │
│  │   return TODAY - birth_date              │   │
│  │                                         │   │
│  │ def compute_purchase_count_7d():        │   │
│  │   return SUM(amount) FROM purchases     │   │
│  │          WHERE date > TODAY - 7 days    │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  ┌─────────────────────────────────────────┐   │
│  │ Metadata & Statistics                   │   │
│  ├─────────────────────────────────────────┤   │
│  │ version: v2.1                           │   │
│  │ last_updated: 2026-06-02 02:30 UTC      │   │
│  │ freshness_sla: <24 hours                │   │
│  │ data_type: numeric                      │   │
│  │ mean: 35.2                              │   │
│  │ std: 12.1                               │   │
│  │ p95: 60                                 │   │
│  │ missing_values_pct: 0.1%                │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 1. Feature Definitions (Schema)

**What it is:** Metadata describing each feature

**Stores:**

```yaml
Feature: age
├─ Data Type: Integer
├─ Entity: user_id (which user does this belong to?)
├─ Description: "Age of user in years"
├─ Valid Range: 0-150
├─ Freshness SLA: "Update daily"
├─ Source Table: raw_users.birth_date
├─ Computation: "TODAY - birth_date"
└─ Version: v1.2

Feature: recent_purchases_7d
├─ Data Type: Float
├─ Entity: user_id
├─ Description: "Total purchase amount in last 7 days (USD)"
├─ Valid Range: 0-1000000
├─ Freshness SLA: "Update daily"
├─ Source Table: transactions
├─ Computation: "SUM(amount) WHERE date > TODAY - 7 days"
└─ Version: v2.0
```

**Example (Python using Feast):**

```python
from feast import Feature, FeatureView, Entity
from feast.types import Int32, Float32

# Define entity
user = Entity(
    name="user_id",
    description="Unique user identifier"
)

# Define features
age_feature = Feature(
    name="age",
    dtype=Int32,
    description="User age in years",
    value_type="int32"
)

purchase_feature = Feature(
    name="recent_purchases_7d",
    dtype=Float32,
    description="Total purchases in last 7 days",
    value_type="float32"
)

# Group into feature view
user_features = FeatureView(
    name="user_features",
    entities=["user_id"],
    features=[age_feature, purchase_feature],
    ttl=timedelta(days=1),  # Freshness SLA
    source=source  # BigQuery table
)
```

### 2. Computed Feature Values

**What it is:** The actual numerical values for each entity

**Online Store (for serving):**

```
Type: Redis, Cloud Spanner, or similar
Access: <10ms
Content: Latest feature values only

Example:
  Key: "user:123"
  Value: {
    "age": 30,
    "recent_purchases_7d": 250.50,
    "avg_rating": 4.5,
    "purchase_count_lifetime": 45
  }

  Key: "user:456"
  Value: {
    "age": 45,
    "recent_purchases_7d": 1250.00,
    "avg_rating": 4.8,
    "purchase_count_lifetime": 350
  }
```

**Offline Store (for training):**

```
Type: BigQuery, Snowflake, Data Lake
Access: Bulk queries (seconds)
Content: Historical feature values with timestamps

Example BigQuery table:
┌─────────────┬────────────┬──────────────────────┬──────────────────────┐
│ user_id     │ timestamp  │ age                  │ recent_purchases_7d  │
├─────────────┼────────────┼──────────────────────┼──────────────────────┤
│ 123         │ 2026-06-02 │ 30                   │ 250.50               │
│ 123         │ 2026-06-01 │ 30                   │ 180.00 (different!)  │
│ 123         │ 2026-05-31 │ 30                   │ 350.00               │
│ 456         │ 2026-06-02 │ 45                   │ 1250.00              │
│ 456         │ 2026-06-01 │ 45                   │ 1100.00              │
└─────────────┴────────────┴──────────────────────┴──────────────────────┘

Use: Training can use historical data with timestamps
     "What did the features look like on 2026-05-01?"
```

### 3. Transformation Code

**What it is:** The SQL/Python that computes features from raw data

**Stored as:**

```python
# Feature transformation (stored in feature store)
class UserFeatures:
    
    @staticmethod
    def compute_age(birth_date: str) -> int:
        """Transform birth_date → age"""
        from datetime import date
        birth = date.fromisoformat(birth_date)
        today = date.today()
        return (today - birth).days // 365
    
    @staticmethod
    def compute_purchase_count_7d(user_id: str) -> int:
        """Query purchase count last 7 days"""
        query = f"""
        SELECT COUNT(*) as count
        FROM purchases
        WHERE user_id = '{user_id}'
        AND transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        """
        return run_query(query)
    
    @staticmethod
    def compute_avg_rating(user_id: str) -> float:
        """Query average rating from user reviews"""
        query = f"""
        SELECT AVG(rating) as avg_rating
        FROM reviews
        WHERE user_id = '{user_id}'
        """
        return run_query(query)
```

**SQL Version (for batch computation):**

```sql
-- Stored in feature store, run daily in data warehouse

-- Feature: age
SELECT
  user_id,
  CURRENT_DATE() as timestamp,
  EXTRACT(YEAR FROM AGE(CURRENT_DATE(), birth_date)) as age
FROM raw_users
;

-- Feature: recent_purchases_7d
SELECT
  user_id,
  CURRENT_DATE() as timestamp,
  SUM(purchase_amount) as recent_purchases_7d
FROM transactions
WHERE transaction_date >= CURRENT_DATE() - INTERVAL '7 days'
GROUP BY user_id
;

-- Feature: avg_rating
SELECT
  user_id,
  CURRENT_DATE() as timestamp,
  AVG(rating) as avg_rating
FROM reviews
GROUP BY user_id
;
```

### 4. Metadata & Statistics

**What it is:** Information about data quality and distributions

**Stores:**

```yaml
Feature: age
├─ Data Type: Integer
├─ Version: v1.2
├─ Last Updated: 2026-06-02 02:30 UTC
├─ Freshness SLA: <24 hours
├─ Description: "User age in years"
├─
├─ Statistics (from training data):
│  ├─ Count: 5,000,000
│  ├─ Missing %: 0.1%
│  ├─ Mean: 35.2
│  ├─ Std Dev: 12.1
│  ├─ Min: 18
│  ├─ Max: 95
│  ├─ P25: 25
│  ├─ P50: 34
│  ├─ P75: 47
│  ├─ P95: 60
│  └─ P99: 70
│
├─ Data Quality Checks:
│  ├─ No nulls allowed: YES
│  ├─ Valid range: 0-150 (check > 150 is error)
│  ├─ Expected distribution: Bell-shaped
│  └─ Alert if drift: p-value < 0.05
│
└─ Lineage:
   ├─ Source: raw_users.birth_date
   ├─ Transformation: TODAY - birth_date
   └─ Dependencies: [raw_users, calendar table]
```

---

## Online vs. Offline Stores

### Comparison

```
┌─────────────────────────────────────────────────────────────┐
│ Aspect              │ Online Store    │ Offline Store      │
├─────────────────────────────────────────────────────────────┤
│ Technology          │ Redis, Spanner  │ BigQuery, Snowflake│
├─────────────────────────────────────────────────────────────┤
│ Latency             │ <10ms           │ 5-30 seconds       │
├─────────────────────────────────────────────────────────────┤
│ Data Recency        │ Latest only     │ Historical         │
├─────────────────────────────────────────────────────────────┤
│ Use Case            │ Real-time pred. │ Training, analysis │
├─────────────────────────────────────────────────────────────┤
│ Query Pattern       │ By entity ID    │ Bulk/SQL queries   │
├─────────────────────────────────────────────────────────────┤
│ Size                │ ~100GB          │ ~10TB+             │
├─────────────────────────────────────────────────────────────┤
│ Cost                │ $$$ (fast)      │ $$ (cheap)         │
├─────────────────────────────────────────────────────────────┤
│ Consistency         │ Eventually      │ Strongly consistent│
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌──────────────────────────────────────┐
│ Raw Data Sources                     │
│ (Databases, APIs, Logs)              │
└─────────────────────┬────────────────┘
                      │
                      │ Daily Batch Job
                      │ (2am - 6am)
                      │
        ┌─────────────▼─────────────┐
        │ Feature Computation       │
        │ (Spark SQL on Big Data)   │
        └─────────────┬─────────────┘
                      │
              ┌───────┴────────┐
              │                │
    ┌─────────▼────────┐   ┌───▼────────────┐
    │ Offline Store    │   │ Online Store   │
    │ (BigQuery)       │   │ (Redis)        │
    │ Historical data  │   │ Latest values  │
    │ For training     │   │ For serving    │
    └────────┬─────────┘   └───┬────────────┘
             │                 │
    ┌────────▼─────────┐      │
    │ Training Pipeline│      │
    │ (ML model)       │      │
    │ Uses history     │      │
    └──────────────────┘      │
                              │
                    ┌─────────▼──────────┐
                    │ Prediction Request │
                    │ (User request)     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Fetch from Online  │
                    │ Store (<10ms)      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Run Model Inference│
                    └──────────────────┘
```

---

## Feature Definitions & Versioning

### Feature Definition Structure

```python
from feast import Feature, FeatureView, Entity
from feast.types import Float32, Int32
from datetime import timedelta

# 1. Define Entity (what are we computing features for?)
user_entity = Entity(
    name="user_id",
    description="Unique identifier for user",
    value_type="STRING"  # Type of the entity ID
)

# 2. Define Features (what values do we compute?)
features = [
    Feature(
        name="age",
        description="User age in years",
        dtype=Int32,
        value_type="int32",
        tags={
            "owner": "data_science_team",
            "version": "1.0",
            "data_type": "categorical",
        }
    ),
    Feature(
        name="recent_purchases_7d",
        description="Total purchase amount last 7 days",
        dtype=Float32,
        value_type="float32",
        tags={
            "owner": "analytics_team",
            "version": "2.0",
            "data_type": "continuous",
        }
    ),
    Feature(
        name="avg_rating",
        description="Average review rating",
        dtype=Float32,
        value_type="float32",
        tags={
            "owner": "data_science_team",
            "version": "1.5",
            "data_type": "continuous",
        }
    ),
]

# 3. Define FeatureView (group features + specify how to compute)
user_features_view = FeatureView(
    name="user_features",
    entities=[user_entity],
    features=features,
    ttl=timedelta(hours=24),  # How fresh should features be?
    source=bigquery_source,   # Where to read data from
    online=True,              # Materialize to online store?
    offline=True,             # Materialize to offline store?
    owner="ml_team@company.com",
    tags={"team": "ml", "critical": True},
)
```

### Versioning Example

**Scenario: You discover age calculation was wrong**

```yaml
Feature: age
├─ Version v1.0 (OLD, DEPRECATED)
│  ├─ Formula: YEAR(CURRENT_DATE) - YEAR(birth_date)
│  │  ← WRONG! Doesn't account for month/day
│  ├─ Created: 2025-01-01
│  ├─ Status: Deprecated
│  └─ Reason: Off by 1 year in some cases
│
├─ Version v1.1 (CURRENT PRODUCTION)
│  ├─ Formula: DATEDIFF(DAY, birth_date, CURRENT_DATE) / 365.25
│  │  ← CORRECT! Handles leap years, fractional
│  ├─ Created: 2026-01-15
│  ├─ Status: Active
│  ├─ Models using v1.1: recommendation_v2.1, churn_v3.0
│  └─ Metrics: No drift detected, matches training data distribution
│
└─ Version v1.2 (IN REVIEW)
   ├─ Formula: EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date))
   │  ← More efficient, same result as v1.1
   ├─ Created: 2026-05-20
   ├─ Status: Testing
   ├─ Validation: No difference from v1.1 on 1M samples
   └─ Plan: Deploy 2026-06-15 after approval
```

---

## Prediction-Time Transformation

### The Challenge

**Problem:** How do we ensure new data transformations match training transformations?

```
Training:
├─ Raw data (user birth_date = "1990-01-01")
├─ Transform: age = 2026 - 1990 = 36
├─ Model trained on: age=36
└─ Model learned: "30-40 age range → high purchase probability"

Prediction (New User):
├─ Raw data (user birth_date = "1985-06-15")
├─ Transform: age = 2026 - 1985 = 41
│  ↓
│  BUT: If transformation done different way:
│  age = DATEDIFF(birth_date, TODAY) / 365 = 40.92 ≈ 40
│  ↓
├─ Model predicts on: age=40 (slightly different!)
└─ Problem: Prediction slightly different due to transformation mismatch
```

### Solution 1: Fetch from Feature Store (Recommended)

**Flow:**

```
NEW PREDICTION REQUEST
├─ User arrives with: user_id = 123
│
├─ Query Feature Store (Online)
│  └─ Returns: {"age": 36, "recent_purchases": 250.5, ...}
│     [These were pre-computed, not re-transformed]
│
├─ Run Model Inference
│  ├─ Input: age=36, recent_purchases=250.5, ...
│  │ [Same format as training]
│  └─ Output: prediction=0.92
│
└─ Return prediction to user
   └─ Guaranteed: Features match training distribution
```

**Code:**

```python
from feast import FeatureStore

# Initialize feature store
fs = FeatureStore(repo_path=".")

# During prediction
def predict(user_id: str):
    # Fetch features (pre-computed, no transformation)
    features = fs.get_online_features(
        features=[
            "user_features:age",
            "user_features:recent_purchases_7d",
            "user_features:avg_rating"
        ],
        entity_rows=[{"user_id": user_id}]
    )
    
    # Convert to model input format
    input_data = {
        "age": features["age"].values[0],
        "recent_purchases_7d": features["recent_purchases_7d"].values[0],
        "avg_rating": features["avg_rating"].values[0]
    }
    
    # Run inference
    prediction = model.predict(input_data)
    
    return prediction
```

**Advantages:**

✅ Features guaranteed to match training (pre-computed)  
✅ Fast (<10ms, no computation)  
✅ Consistent (same transformation code used for all)  
✅ Simple (no transformation logic in serving code)  

**Disadvantages:**

✗ Data always 1 day old (stale)  
✗ Doesn't work for new users (not in feature store yet)  
✗ Infrastructure cost (maintain feature store)  

---

### Solution 2: Apply Same Transformation at Serving Time

**Problem:** What if data changed in last 24 hours?

```
Scenario: New user (not pre-computed in feature store)
├─ User: John (user_id = NEW_999)
├─ Feature store doesn't have features (new!)
├─ So: Compute on-the-fly using same code as training
└─ Ensure transformation matches exactly
```

**Flow:**

```
NEW PREDICTION REQUEST (USER NOT IN FEATURE STORE)
├─ User arrives with: user_id = NEW_999
│
├─ Transformation Service (reuses training code)
│  ├─ age = compute_age(birth_date="1992-05-10")
│  │       = 2026 - 1992 = 34
│  │
│  ├─ recent_purchases = query_purchases(user_id, days=7)
│  │                   = 450.00
│  │
│  └─ Returns: {"age": 34, "recent_purchases": 450.00, ...}
│
├─ Run Model Inference
│  └─ Input: age=34, recent_purchases=450.00
│
└─ Return prediction
   └─ Features computed on-the-fly, but using exact training code
```

**Code (Shared transformation library):**

```python
# shared_transformations.py
# Used by BOTH training pipeline and serving

class FeatureTransformations:
    """Feature transformations (version-controlled)"""
    
    @staticmethod
    def compute_age(birth_date: str, reference_date: str = None) -> int:
        """
        Compute age from birth date
        
        Used in: Training pipeline + Serving pipeline
        Important: Same code, same logic, guarantees consistency
        """
        from datetime import date
        
        if reference_date is None:
            reference_date = str(date.today())
        
        birth = date.fromisoformat(birth_date)
        reference = date.fromisoformat(reference_date)
        
        age = reference.year - birth.year
        # Account for birthday this year
        if (reference.month, reference.day) < (birth.month, birth.day):
            age -= 1
        
        return age
    
    @staticmethod
    def compute_recent_purchases(user_id: str, days: int = 7) -> float:
        """
        Sum purchase amount in last N days
        """
        from datetime import datetime, timedelta
        
        query = f"""
        SELECT SUM(amount) as total
        FROM purchases
        WHERE user_id = '{user_id}'
        AND transaction_date >= CURRENT_DATE - INTERVAL {days} DAY
        """
        result = run_query(query)
        return result['total'] or 0.0
    
    @staticmethod
    def compute_avg_rating(user_id: str) -> float:
        """
        Average review rating for user
        """
        query = f"""
        SELECT AVG(rating) as avg_rating
        FROM reviews
        WHERE user_id = '{user_id}'
        """
        result = run_query(query)
        return result['avg_rating'] or 0.0

# ========== TRAINING PIPELINE ==========
def train_model(training_data_df):
    """Training uses shared transformations"""
    from shared_transformations import FeatureTransformations as FT
    
    # Apply transformations to training data
    training_data_df['age'] = training_data_df['birth_date'].apply(
        FT.compute_age
    )
    training_data_df['recent_purchases'] = training_data_df['user_id'].apply(
        FT.compute_recent_purchases
    )
    # ... train model ...

# ========== SERVING PIPELINE ==========
def predict(user_id: str):
    """Serving uses SAME shared transformations"""
    from shared_transformations import FeatureTransformations as FT
    
    # Get raw user data
    user = query_user(user_id)
    
    # Apply SAME transformations
    age = FT.compute_age(user['birth_date'])
    recent_purchases = FT.compute_recent_purchases(user_id)
    avg_rating = FT.compute_avg_rating(user_id)
    
    # Run inference on transformed data
    input_data = {
        "age": age,
        "recent_purchases": recent_purchases,
        "avg_rating": avg_rating
    }
    
    prediction = model.predict(input_data)
    
    return prediction
```

**Advantages:**

✅ Works for new users (computes on-the-fly)  
✅ Features always fresh (computed at request time)  
✅ Guaranteed consistency (shared code)  

**Disadvantages:**

✗ Slower (computation takes 50-500ms)  
✗ More complex (requires transformation service)  
✗ Could fail (external APIs might be down)  

---

### Solution 3: Hybrid (Recommended in Practice)

**Best approach: Use both strategies**

```
PREDICTION REQUEST
├─ Check Feature Store (online) for user
│  ├─ IF user exists in store:
│  │  └─ Return pre-computed features (fast, <10ms)
│  │
│  └─ IF user NOT in store (new user):
│     └─ Compute on-the-fly (slower, but handles new users)
│
└─ Run inference
```

**Code:**

```python
def predict(user_id: str):
    """
    Hybrid approach:
    1. Try fetching from feature store (fast)
    2. Fall back to on-the-fly computation (handles new users)
    """
    from feast import FeatureStore
    from shared_transformations import FeatureTransformations as FT
    
    try:
        # FAST PATH: Fetch from feature store (pre-computed)
        fs = FeatureStore(repo_path=".")
        features = fs.get_online_features(
            features=[
                "user_features:age",
                "user_features:recent_purchases_7d",
                "user_features:avg_rating"
            ],
            entity_rows=[{"user_id": user_id}]
        )
        
        # Check if features exist
        if features["age"].values[0] is not None:
            # Use pre-computed features
            input_data = {
                "age": features["age"].values[0],
                "recent_purchases_7d": features["recent_purchases_7d"].values[0],
                "avg_rating": features["avg_rating"].values[0]
            }
            latency = "FAST (cached)"
        else:
            raise ValueError("User not in feature store")
    
    except Exception as e:
        # SLOW PATH: Compute on-the-fly for new users
        user = query_user(user_id)
        
        age = FT.compute_age(user['birth_date'])
        recent_purchases = FT.compute_recent_purchases(user_id)
        avg_rating = FT.compute_avg_rating(user_id)
        
        input_data = {
            "age": age,
            "recent_purchases_7d": recent_purchases,
            "avg_rating": avg_rating
        }
        latency = "SLOWER (on-the-fly)"
    
    # Run inference (same for both paths)
    prediction = model.predict(input_data)
    
    return {
        "prediction": prediction,
        "latency_type": latency
    }
```

---

## Training-Time vs. Serving-Time Consistency

### The Problem: Data Leakage & Distribution Mismatch

**Scenario 1: Different Preprocessing**

```
TRAINING:
├─ Raw: age_days = 13000 (days since birth)
├─ Transform: age = age_days / 365.25
├─ Model trained on: age ≈ 35.6 (float)
└─ Model learned: "age > 30 → high value customer"

SERVING (Bug #1):
├─ Raw: age = 35 (already in years, different source!)
├─ Transform: age = age / 365.25  ← WRONG! Divides twice!
├─ Model inference on: age ≈ 0.096 (not 35!)
└─ Model predicts: "age=0.1 → low value" (WRONG!)
```

**Scenario 2: Time-based Features**

```
TRAINING (June 2, 2025):
├─ Today's date: 2025-06-02
├─ Feature: "days_since_last_purchase" = 10 days ago - today
│         = 2025-05-23 - 2025-06-02 = -10 days
│
└─ Model trained on: -10 days

SERVING (June 2, 2026):
├─ Today's date: 2026-06-02
├─ Feature: "days_since_last_purchase" = 10 days ago - today
│         = 2026-05-23 - 2026-06-02 = -10 days
│
├─ BUT: Same relative value (-10), but user's behavior changed
│ (1 year passed, they are different now!)
└─ Model predicts: OLD prediction (1 year stale!)
```

### The Solution: Synchronization

#### Approach 1: Shared Feature Definitions

```python
# features_config.yaml (VERSION CONTROLLED)
# Single source of truth for feature transformations

features:
  age:
    definition: "EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date))"
    data_type: INTEGER
    range: [0, 150]
    description: "Age in years (correct handling of leap years)"
    version: v1.1
    approved_by: ml_lead@company.com
    approval_date: 2026-01-15

  days_since_last_purchase:
    definition: "DATEDIFF(DAY, MAX(purchase_date), CURRENT_DATE)"
    data_type: INTEGER
    range: [0, 3650]  # Max 10 years
    description: "Days since most recent purchase (relative time)"
    version: v2.0

  purchase_count_7d:
    definition: |
      SELECT COUNT(*)
      FROM purchases
      WHERE user_id = {user_id}
      AND transaction_date >= CURRENT_DATE - INTERVAL 7 DAY
    data_type: INTEGER
    range: [0, 1000]
    version: v1.0
```

#### Approach 2: Feature Version Tracking

**Keep track of which feature version was used in training:**

```
Model: recommendation_v2.1
└─ Training Features:
   ├─ age (v1.1) ✓ Current version
   ├─ days_since_last_purchase (v2.0) ✓ Current version
   ├─ purchase_count_7d (v1.0) ⚠️ OUTDATED (current is v1.2)
   │  ├─ v1.2 has bug fix
   │  ├─ Risk: Model may not work with v1.2
   │  └─ Action: Retrain model on v1.2 or pin to v1.0
   └─ avg_rating (v1.0) ✓ Current version

Action:
├─ During serving: Use feature versions from training
├─ If feature version mismatch detected: Alert
├─ Plan: Retrain model on latest features
```

---

## Real-World Examples

### Example 1: E-Commerce Recommendation System

**What Gets Stored:**

```
Feature Store Content:

Entity: product_id
├─ price (latest product price)
├─ avg_rating (average review rating)
├─ inventory_count (current stock)
├─ days_since_last_sold (staleness metric)
└─ purchase_velocity_7d (trend metric)

Entity: user_id
├─ age (years, from birth_date)
├─ country (shipping location)
├─ total_purchases (lifetime)
├─ avg_order_value (AVG purchase amount)
├─ days_since_last_purchase (recency)
├─ browsing_category (current interest)
└─ is_vip_customer (boolean)

Entity: user-product pair (interaction features)
├─ user_product:viewed_7d (did user view product in last 7 days?)
├─ user_product:clicked_7d (did user click product?)
├─ user_product:purchased_90d (did user buy similar product recently?)
└─ user_product:price_sensitivity (does user buy at full price?)
```

**Prediction-Time Transformation:**

```python
def predict_click_probability(user_id: str, product_id: str):
    """
    Predict: Will this user click on this product?
    """
    
    # TRY: Fetch from Feature Store (fast)
    try:
        fs = FeatureStore()
        
        # Get user features
        user_features = fs.get_online_features(
            features=[
                "user_features:age",
                "user_features:country",
                "user_features:avg_order_value",
                "user_features:days_since_last_purchase"
            ],
            entity_rows=[{"user_id": user_id}]
        )
        
        # Get product features
        product_features = fs.get_online_features(
            features=[
                "product_features:price",
                "product_features:avg_rating",
                "product_features:purchase_velocity_7d"
            ],
            entity_rows=[{"product_id": product_id}]
        )
        
        # Get interaction features
        interaction_features = fs.get_online_features(
            features=[
                "interaction_features:viewed_7d",
                "interaction_features:price_sensitivity"
            ],
            entity_rows=[{
                "user_id": user_id,
                "product_id": product_id
            }]
        )
        
        input_data = {
            "user_age": user_features["age"].values[0],
            "user_country": user_features["country"].values[0],
            "user_avg_order_value": user_features["avg_order_value"].values[0],
            "product_price": product_features["price"].values[0],
            "product_rating": product_features["avg_rating"].values[0],
            "user_viewed_7d": interaction_features["viewed_7d"].values[0],
        }
    
    except Exception as e:
        # FALLBACK: Compute features for new users
        from shared_transformations import compute_*
        
        user = query_user(user_id)
        product = query_product(product_id)
        
        input_data = {
            "user_age": compute_age(user["birth_date"]),
            "user_country": user["country"],
            "user_avg_order_value": compute_avg_order_value(user_id),
            "product_price": product["price"],
            "product_rating": compute_avg_rating(product_id),
            "user_viewed_7d": check_viewed_recently(user_id, product_id, days=7),
        }
    
    # Run inference
    click_probability = model.predict(input_data)
    
    return click_probability
```

---

### Example 2: Credit Risk (Finance)

**What Gets Stored:**

```
Feature Store Content:

Entity: customer_id
├─ age (years)
├─ income (annual, USD)
├─ employment_length (years with current employer)
├─ credit_score (FICO score: 300-850)
├─ total_debt (all outstanding debt)
├─ available_credit (total credit limit - used)
├─ missed_payments_12m (count)
├─ avg_payment_days_late (history)
├─ debt_to_income_ratio (calculated)
└─ industry (employment type)
```

**Critical: Exact Transformation for Regulatory Compliance**

```python
# MUST use exact same transformation for training & serving
# Regulators require reproducibility!

class CreditRiskFeatures:
    """Credit features - VERSION CONTROLLED"""
    
    @staticmethod
    def compute_debt_to_income_ratio(
        total_debt: float,
        annual_income: float
    ) -> float:
        """
        Debt-to-income ratio
        
        REGULATORY: This must be computed exactly same way
        during training and serving, for audit trail
        """
        if annual_income <= 0:
            return float('inf')
        
        return (total_debt * 12) / annual_income  # Multiply by 12 (monthly payment)
    
    @staticmethod
    def compute_credit_score_category(fico_score: int) -> str:
        """
        Categorize FICO score
        
        REGULATORY: Standard FICO categories
        """
        if fico_score >= 750:
            return "excellent"
        elif fico_score >= 700:
            return "good"
        elif fico_score >= 650:
            return "fair"
        else:
            return "poor"

# During serving
def predict_default_risk(customer_id: str):
    """Predict: Will customer default on loan?"""
    
    customer = query_customer(customer_id)
    
    # Use shared transformation (guaranteed consistency)
    debt_to_income = CreditRiskFeatures.compute_debt_to_income_ratio(
        total_debt=customer['total_debt'],
        annual_income=customer['annual_income']
    )
    
    credit_category = CreditRiskFeatures.compute_credit_score_category(
        fico_score=customer['fico_score']
    )
    
    input_data = {
        "age": customer["age"],
        "debt_to_income_ratio": debt_to_income,
        "credit_score_category": credit_category,
        "missed_payments_12m": customer["missed_payments_12m"],
        "income": customer["annual_income"]
    }
    
    default_risk = model.predict(input_data)
    
    # LOG FOR AUDIT TRAIL (regulatory requirement)
    log_prediction(
        customer_id=customer_id,
        features=input_data,
        prediction=default_risk,
        model_version="v2.1",
        feature_versions=["debt_to_income:v1.0", "credit_score_category:v1.0"],
        timestamp=datetime.now()
    )
    
    return default_risk
```

---

## Common Pitfalls

### ❌ Pitfall 1: Training-Serving Skew

**Problem:**

```
Training:
├─ Date: 2025-06-01
├─ Feature: "days_since_last_purchase"
│  └─ Computed as: 2025-06-01 - last_purchase_date
│
└─ Model sees: Customer 123 has 10 days since purchase

Serving (1 year later):
├─ Date: 2026-06-01
├─ Feature: "days_since_last_purchase"
│  └─ Computed as: 2026-06-01 - last_purchase_date
│     (same last_purchase_date as training!)
│
└─ Model sees: Customer 123 has 375 days since purchase
              (COMPLETELY DIFFERENT!)
   This will break the model!
```

**Solution:**

```python
# Use reference date (not current date) for consistency

def compute_days_since_last_purchase(
    user_id: str,
    reference_date: str = None  # If None, use today
) -> int:
    """
    Days since last purchase
    
    reference_date: For testing, allows consistent calculations
    """
    if reference_date is None:
        from datetime import date
        reference_date = str(date.today())
    
    query = f"""
    SELECT MAX(transaction_date) as last_purchase
    FROM purchases
    WHERE user_id = '{user_id}'
    """
    
    last_purchase = run_query(query)['last_purchase']
    
    return (reference_date - last_purchase).days
```

---

### ❌ Pitfall 2: Missing Value Handling

**Problem:**

```
Training (preprocessing with defaults):
├─ Feature: age
│  ├─ Value: 35 (most values present)
│  └─ If NULL: default to 30 (average age)
│
└─ Model trained on mix of 35s and 30s

Serving (different handling):
├─ Feature: age
│  ├─ Value: 35 (most values present)
│  └─ If NULL: crash! (no default)
│
└─ Crashes on some users (new user without age)
```

**Solution:**

```python
# Define handling in feature definition

age_feature = Feature(
    name="age",
    dtype=Int32,
    description="User age in years",
    
    # DEFINE HANDLING
    default_value=30,  # Use 30 if missing
    # OR
    fill_strategy="forward_fill",  # Use previous value
    # OR
    fill_strategy="mean",  # Use training set mean
)

# Use same strategy everywhere
def get_age_feature(user_id: str, missing_value_strategy="mean"):
    """
    Fetch/compute age feature
    
    If missing, apply same strategy as training
    """
    value = fetch_from_feature_store(user_id, "age")
    
    if value is None:
        if missing_value_strategy == "mean":
            return 30  # Training set mean
        elif missing_value_strategy == "forward_fill":
            return fetch_last_known_value(user_id, "age")
    
    return value
```

---

### ❌ Pitfall 3: Feature Type Mismatch

**Problem:**

```
Training (SQL):
├─ SELECT CAST(rating AS INTEGER)
├─ 4.8 → 4 (integer truncation)
└─ Model trained on integers: 1, 2, 3, 4, 5

Serving (Python):
├─ rating = 4.8
├─ No casting
└─ Model inference on float: 4.8 (different!)
   Model never saw 4.8 during training!
```

**Solution:**

```python
def cast_feature(value, feature_name: str, feature_type: str):
    """
    Cast feature to correct type
    
    Ensures training/serving consistency
    """
    if feature_type == "integer":
        return int(value)  # 4.8 → 4
    elif feature_type == "float":
        return float(value)
    elif feature_type == "string":
        return str(value)
    else:
        raise ValueError(f"Unknown type {feature_type}")

# In serving
rating = fetch_feature(user_id, "rating")
rating = cast_feature(rating, "rating", "integer")  # Ensure correct type
```

---

### ❌ Pitfall 4: Dependency on External State

**Problem:**

```
Training (May 2025):
├─ Feature: "is_holiday_today"
│  └─ Computed as: if today in HOLIDAYS_LIST
│  └─ Holidays: Memorial Day, Independence Day, ...
│  └─ Model learns: "Holiday → different behavior"
│
└─ Model trained to handle holidays

Serving (June 2026):
├─ Feature: "is_holiday_today"
│  └─ HOLIDAYS_LIST not updated with 2026 holidays!
│  └─ June 19, 2026 is Juneteenth (new federal holiday in 2021)
│  └─ But HOLIDAYS_LIST still has 2025 holidays
│
└─ Predicts wrong (doesn't recognize new holiday)
```

**Solution:**

```python
def is_holiday(date_str: str) -> bool:
    """
    Is this date a holiday?
    
    IMPORTANT: Use authoritative external data source
    Not a hardcoded list!
    """
    # Option 1: Query database (updated regularly)
    query = f"SELECT is_holiday FROM holidays WHERE date = '{date_str}'"
    result = run_query(query)
    return result['is_holiday']
    
    # Option 2: Use library that updates
    from datetime import date
    import holidays
    
    us_holidays = holidays.US()
    return date_str in us_holidays
```

---

### ❌ Pitfall 5: Feature Store Staleness

**Problem:**

```
Feature Store Update (Daily, 2am):
├─ Updates all 100M user features
├─ But: Takes 4 hours to complete
├─ Finishes at: 6am

User Request at 5am:
├─ Wants latest features
├─ Feature store still updating!
├─ Gets old features (from previous day)
├─ Model predicts based on 25-hour-old data
│
└─ If user behavior changed overnight: Inaccurate!
```

**Solution:**

```python
def get_features_with_staleness_check(user_id: str):
    """
    Fetch features, but check freshness
    """
    features = fetch_from_feature_store(user_id)
    last_updated = features['_last_updated_timestamp']
    
    # Check: Are features fresh?
    age_hours = (datetime.now() - last_updated).total_seconds() / 3600
    
    if age_hours > 24:  # Older than 24 hours?
        # Option 1: Fallback to on-the-fly computation
        features = compute_features_onthefly(user_id)
    elif age_hours > 12:  # Older than 12 hours?
        # Option 2: Use with caution, log warning
        logger.warning(f"Features are {age_hours:.1f} hours old")
    
    return features
```

---

## Summary: What to Remember

### What Gets Stored in Feature Store

```
┌─────────────────────────────────────────────┐
│ 1. Feature Definitions (Schema)             │
│    └─ How to compute each feature           │
│                                             │
│ 2. Computed Values                          │
│    ├─ Online: Latest (Redis, <10ms)        │
│    └─ Offline: Historical (BigQuery)       │
│                                             │
│ 3. Transformation Code                      │
│    └─ Shared between training & serving    │
│                                             │
│ 4. Metadata & Statistics                    │
│    └─ Data quality, distributions, SLAs    │
└─────────────────────────────────────────────┘
```

### How to Transform Features During Prediction

```
┌─────────────────────────────────────────────────┐
│ RECOMMENDATION: Hybrid Approach                │
├─────────────────────────────────────────────────┤
│                                                 │
│ 1. TRY: Fetch from Feature Store (fast)       │
│    ├─ Pre-computed values                      │
│    ├─ <10ms latency                            │
│    └─ Guaranteed consistency with training    │
│                                                 │
│ 2. FALLBACK: Compute on-the-fly (for new users)
│    ├─ Use shared transformation code           │
│    ├─ 50-500ms latency                         │
│    └─ Handles users not in store yet           │
│                                                 │
│ KEY: Same transformation logic everywhere     │
│      (training, online store, serving)         │
└─────────────────────────────────────────────────┘
```

---

**Key Insight:** The Feature Store is not just a database—it's a contract between training and serving that says: "These features will be computed this way, every time, guaranteed."

Consistency > Speed > Flexibility.
