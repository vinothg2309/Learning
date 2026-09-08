feature_hasher_explained.md
# FeatureHasher — How It Works with Examples

---

## Core Concept

**FeatureHasher** converts variable-length features (like categories or sparse data) into fixed-size feature vectors using a **hash function**. Instead of creating a new column for each unique value, it hashes the value into one of N buckets.

---

## The Problem It Solves

Imagine you're building a recommendation system and you have a feature "browser_type":

```
Original Data:
  Row 1: chrome
  Row 2: firefox
  Row 3: safari
  Row 4: chrome
  Row 5: edge
  Row 6: opera
  Row 7: seamonkey
  ...
  Row 10000: netscape  ← New browser we've never seen!
```

### Naive Approach (One-Hot Encoding):

```python
# Create a column for each unique browser
# browsers = [chrome, firefox, safari, edge, opera, seamonkey, ...]

# Sparse matrix:
#        chrome  firefox  safari  edge  opera  seamonkey  ...  (100s of columns)
# Row 1    1       0        0      0     0        0
# Row 2    0       1        0      0     0        0
# Row 3    0       0        1      0     0        0
# ...
# Row 10000 0      0        0      0     0        0      ← Unknown browser!
```

**Problems:**
- ❌ If you have 1M unique values → 1M columns (memory explosion)
- ❌ New values at inference time break the model (extra column not in training)
- ❌ Sparse (mostly zeros) → slow computation

### FeatureHasher Approach:

Instead of creating a column per value, **hash each value into one of N fixed buckets** (e.g., 1024 buckets regardless of how many unique values exist).

```
FeatureHasher with n_features=1024:

Row 1: chrome      → hash("chrome")      % 1024 = 372  → bucket 372 gets +1
Row 2: firefox     → hash("firefox")     % 1024 = 891  → bucket 891 gets +1
Row 3: safari      → hash("safari")      % 1024 = 156  → bucket 156 gets +1
Row 10000: netscape → hash("netscape")    % 1024 = 512  → bucket 512 gets +1 (handles new value!)

Result: Fixed 1024-dimensional vector (not 1M!)
```

---

## How The Hash Function Works

### Step 1: Hash the Value

Python's hash function converts string → integer:

```python
hash("chrome")     # → some large integer like -4534985234
hash("firefox")    # → some large integer like 8293742974
hash("netscape")   # → some large integer like 2384729342 (NEW, never seen!)
```

### Step 2: Modulo Into N Buckets

```python
n_features = 1024

bucket_id = hash("chrome") % 1024
          = (-4534985234) % 1024
          = 372

bucket_id = hash("firefox") % 1024
          = 8293742974 % 1024
          = 891

bucket_id = hash("netscape") % 1024
          = 2384729342 % 1024
          = 512
```

### Step 3: Increment Bucket Count

```python
feature_vector = [0] * 1024

feature_vector[372] += 1    # "chrome" goes here
feature_vector[891] += 1    # "firefox" goes here
feature_vector[156] += 1    # "safari" goes here
feature_vector[512] += 1    # "netscape" goes here (new value, no problem!)

# Result: sparse vector with 1024 dimensions
# Most buckets are 0, only 4 buckets have values
```

---

## Hash Collision

Sometimes two different values hash to the **same bucket** — this is called a **collision**.

```python
# (Hypothetical example)
hash("chrome") % 1024 = 372
hash("opera") % 1024 = 372  ← Same bucket! Collision!

# What happens?
feature_vector[372] += 1  # chrome
feature_vector[372] += 1  # opera (same bucket!)
feature_vector[372] = 2   # Can't distinguish chrome from opera

# Trade-off: Some information loss, but handles unlimited unique values
```

**Impact:**
- With 1024 buckets and millions of unique values → many collisions
- With 2^20 buckets (1M) and thousands of unique values → few collisions
- You choose `n_features` to balance memory vs accuracy

---

## Real Example: User Features

### Scenario

Building a fraud detection model with user features:

```
User 1:  browser=chrome, country=US, ip_first_octet=192
User 2:  browser=firefox, country=UK, ip_first_octet=10
User 3:  browser=safari, country=US, ip_first_octet=172
...
User 10000: browser=netscape (UNKNOWN!), country=Japan, ip_first_octet=203
```

### Code: FeatureHasher

```python
from sklearn.feature_extraction import FeatureHasher
import numpy as np

# Raw feature dictionaries
users = [
    {'browser': 'chrome', 'country': 'US', 'ip_first_octet': 192},
    {'browser': 'firefox', 'country': 'UK', 'ip_first_octet': 10},
    {'browser': 'safari', 'country': 'US', 'ip_first_octet': 172},
    {'browser': 'netscape', 'country': 'Japan', 'ip_first_octet': 203},  # NEW browser
]

# Create hasher: 1024 buckets (can handle unlimited unique values)
hasher = FeatureHasher(n_features=1024, input_type='dict')

# Transform
X_hashed = hasher.transform(users)

# Result: sparse matrix (1024 columns)
print(X_hashed.shape)  # (4, 1024)
print(X_hashed.toarray())  # Dense version for visualization

# Output (sparse):
# Row 1: bucket_372=1 (chrome), bucket_654=1 (US), bucket_789=192
# Row 2: bucket_891=1 (firefox), bucket_654=1 (UK), bucket_123=10
# Row 3: bucket_156=1 (safari), bucket_654=1 (US), bucket_456=172
# Row 4: bucket_512=1 (netscape), bucket_899=1 (Japan), bucket_203=203
#        ↑ New value handled!
```

### What Happened

```
User 1: browser=chrome
  hash("browser=chrome") % 1024 = 372
  X[0, 372] = 1

User 1: country=US
  hash("country=US") % 1024 = 654
  X[0, 654] = 1

User 1: ip_first_octet=192
  hash("ip_first_octet=192") % 1024 = 789
  X[0, 789] = 192  ← The numeric value itself!

User 4 (NEW browser):
  hash("browser=netscape") % 1024 = 512
  X[3, 512] = 1  ← Works! No error even though netscape never seen in training
```

---

## Step-by-Step Walkthrough

Let's trace one example completely:

```python
from sklearn.feature_extraction import FeatureHasher

# Single user
user = {'browser': 'chrome', 'country': 'US', 'age': 30}

hasher = FeatureHasher(n_features=8, input_type='dict')  # Small (8 buckets) for demo
X = hasher.transform([user])

print(X.toarray())
# [[1 0 0 0 1 0 0 30]]
#   ↑           ↑  ↑
#   bucket0   bucket4 age in bucket7

# What happened internally:
# hash("browser=chrome") % 8 = 0     → X[0, 0] = 1
# hash("country=US") % 8 = 4         → X[0, 4] = 1
# hash("age=30") % 8 = 7 (and value is 30) → X[0, 7] = 30
```

---

## Comparison: One-Hot vs FeatureHasher

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

browsers = ['chrome', 'firefox', 'safari', 'edge', 'opera']
training_data = [['chrome'], ['firefox'], ['safari'], ['chrome']]

encoder = OneHotEncoder(sparse=False)
X_onehot = encoder.fit_transform(training_data)

print(X_onehot)
#     chrome  firefox  safari  edge  opera
# [[1        0        0       0     0   ]
#  [0        1        0       0     0   ]
#  [0        0        1       0     0   ]
#  [1        0        0       0     0   ]]

# At inference: NEW browser "netscape"
test_data = [['netscape']]
X_test = encoder.transform(test_data)  # ❌ ERROR! 'netscape' not in training
```

**Problems:**
- ❌ New categories at inference crash the encoder
- ❌ Number of features = number of unique values (100K browsers = 100K columns)
- ❌ Memory explosion

### FeatureHasher

```python
from sklearn.feature_extraction import FeatureHasher

training_data = [
    {'browser': 'chrome'},
    {'browser': 'firefox'},
    {'browser': 'safari'},
    {'browser': 'chrome'},
]

hasher = FeatureHasher(n_features=1024, input_type='dict')
X_hash = hasher.transform(training_data)

print(X_hash.shape)  # (4, 1024) — Fixed size!

# At inference: NEW browser "netscape"
test_data = [{'browser': 'netscape'}]
X_test = hasher.transform(test_data)  # ✅ Works! No error
print(X_test.shape)  # (1, 1024) — Same size as training
```

**Advantages:**
- ✅ Handles new categories automatically
- ✅ Fixed number of features (always n_features)
- ✅ Memory-efficient (sparse matrix)

---

## Real-World Use Cases

### 1. Web Services (Unknown Features at Inference)

```python
# Training: 1000 unique browsers
# Inference: New browser "ChromeOS", "RoboBot", "Googlebot"
#            We didn't know about at training time!

hasher = FeatureHasher(n_features=2**16, input_type='dict')  # 65536 buckets
# Can handle thousands of unknown browsers without breaking
```

### 2. Text Processing (Document Terms)

```python
# Each document has different words
documents = [
    {'word_the': 5, 'word_quick': 2, 'word_fox': 1},
    {'word_the': 3, 'word_lazy': 1, 'word_dog': 4},
    {'word_jumping': 2, 'word_over': 1, 'word_moon': 1},  # Different words!
]

hasher = FeatureHasher(n_features=2**10, input_type='dict')
X = hasher.transform(documents)

print(X.shape)  # (3, 1024) — All documents same size!
```

### 3. Streaming Data (Don't Know All Categories Upfront)

```python
# Recommendation system: user_id values keep growing
# Instead of adding a new column each time, hash into fixed buckets

stream_users = [
    {'user_id': 12345, 'action': 'click'},
    {'user_id': 99999, 'action': 'view'},
    {'user_id': 500000000, 'action': 'purchase'},  # Huge ID, but same bucket size
]

hasher = FeatureHasher(n_features=1024)
X = hasher.transform(stream_users)  # Always 1024 columns
```

---

## Hash Function Choice

FeatureHasher uses **signed hashing** by default (can be positive or negative hash values):

```python
from sklearn.feature_extraction import FeatureHasher

# Default: signed=True
# hash("feature") can be positive or negative
# This matters for regularization in linear models

hasher_signed = FeatureHasher(n_features=1024, signed=True)
# Helps with L1/L2 regularization (prevents cancel-out)

hasher_unsigned = FeatureHasher(n_features=1024, signed=False)
# All bucket increments are positive
```

**Why signed matters:**

```python
# Signed (default):
# hash("feature_A") % 1024 = 372, signed hash = +372
# hash("feature_B") % 1024 = 372, signed hash = -372
# Bucket 372 = +372 + (-372) = 0  ← They cancel out!
# Helps regularization distinguish features

# Unsigned:
# Both increment same bucket, can't distinguish
```

---

## Pros and Cons

### Pros ✅

| Advantage | Why |
|---|---|
| **Fixed size** | Always n_features columns, regardless of unique values |
| **Handles unknowns** | New categories at inference = no problem |
| **Memory efficient** | Sparse matrix, not dense |
| **Fast** | Hash computation is O(1) |
| **Scalable** | Works with streaming, unbounded categories |
| **Works with text** | Great for NLP (billions of possible words) |

### Cons ❌

| Disadvantage | Why |
|---|---|
| **Hash collisions** | Different features map to same bucket → info loss |
| **Interpretability** | Can't tell which bucket corresponds to what feature |
| **No feature names** | Lose semantic meaning (unlike One-Hot) |
| **Approximation** | Less accurate than explicit columns (for known categories) |
| **Regularization quirks** | Need to use signed=True for L1/L2 to work well |

---

## When to Use FeatureHasher vs OneHotEncoder

```
Use OneHotEncoder when:
├─ Categorical features known and finite (< 100 unique values)
├─ Interpretability matters (need to see which category)
└─ You won't see new categories at inference

Use FeatureHasher when:
├─ Unbounded categories (millions of possible values)
├─ Unknown categories at inference (must handle gracefully)
├─ Streaming data (can't compute all categories upfront)
├─ Text features (NLP — billions of possible words)
└─ Memory constraints (sparse, fixed-size output)
```

---

## Complete Working Example

```python
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
import numpy as np

# Simulated user interaction data
training_data = [
    {'browser': 'chrome', 'os': 'windows', 'country': 'US', 'spent': 50},
    {'browser': 'firefox', 'os': 'mac', 'country': 'UK', 'spent': 75},
    {'browser': 'safari', 'os': 'ios', 'country': 'US', 'spent': 100},
    {'browser': 'chrome', 'os': 'linux', 'country': 'Germany', 'spent': 45},
]

test_data = [
    {'browser': 'netscape', 'os': 'windows', 'country': 'Japan', 'spent': 60},  # NEW browser!
    {'browser': 'opera', 'os': 'android', 'country': 'India', 'spent': 30},    # NEW os!
]

# Labels (purchased: yes=1, no=0)
y_train = [0, 1, 1, 0]

# Step 1: Hash features
hasher = FeatureHasher(n_features=256, input_type='dict')

X_train = hasher.transform(training_data)
X_test = hasher.transform(test_data)

print(f"Training shape: {X_train.shape}")  # (4, 256)
print(f"Test shape: {X_test.shape}")       # (2, 256)
print(f"Training sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.2%}")

# Step 2: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 3: Predict on NEW unseen features
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")  # Works even with new browser, os, country!

print("\n✅ FeatureHasher handled unknown features seamlessly!")
```

---

## Summary

**FeatureHasher** is a hashing trick that:

1. **Maps** any categorical/sparse feature to one of N fixed buckets
2. **Handles** unknown values at inference without error
3. **Scales** to unbounded categories (streaming, text, high-cardinality features)
4. **Trades off** some accuracy (hash collisions) for memory and speed

**Key formula:**
```
bucket_id = hash(feature) % n_features
X[sample, bucket_id] += feature_value
```

**Best for:** Text, streaming data, unknown categories, high-cardinality features  
**Avoid for:** Small finite categories where interpretability matters
