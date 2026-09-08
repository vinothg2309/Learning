feature_engineering_staff_scenarios.md
# Feature Engineering — Staff-Level Scenario Questions for AI71

---

## S1: Feature Explosion in Real-Time Recommendation System

**Scenario:**

You're the lead ML engineer at Falcon (AI71's recommendation engine) serving 50M users. Current setup:

- **User features:** 200 static features (age, location, subscription tier, etc.)
- **Item features:** 500 features per item (category, price, embeddings, metadata)
- **Interaction features:** Last 30 days of user behavior = 50 features (clicks, dwells, purchases, etc.)
- **Contextual features:** Time of day, device, network, location context = 20 features
- **Cross-features:** Manually engineered interactions = 100+ features

**Total: ~870 features**

**The Problem:**

1. **Latency requirement:** Recommendation must return in <100ms p99 (including feature computation)
2. **Feature freshness:** User behavior changes every hour; many features become stale
3. **Storage:** Storing 870 features for 50M users × 1M items = petabyte-scale data
4. **Model complexity:** XGBoost with 870 features trains in 24+ hours (too slow for daily updates)
5. **Feature drift:** 40% of features have seasonal patterns; models degrade without retraining
6. **Cost:** Feature storage + computation = $500K/month, limits profitability

**Interview Question:**

"How would you design a feature engineering strategy that reduces dimensionality while maintaining model performance, stays within the <100ms latency budget, handles feature freshness, and reduces operational costs? Walk me through your decision-making process, trade-offs, and what you'd measure."

---

## Expected Staff-Level Answer Structure

### Part 1: Problem Diagnosis (5 min)

**What you should identify:**

```
1. Not all 870 features are equally important
   └─ Pareto: ~80% of performance from ~15-20% of features
   
2. Feature freshness creates two problems:
   ├─ Stale features degrade model (e.g., yesterday's behavior vs current session)
   └─ Real-time computation is expensive
   
3. Storage bottleneck:
   └─ Storing all historical features for all users = unsustainable
   
4. Training velocity is the constraint:
   └─ 24hr training cycle = can't adapt quickly to new trends
   
5. Cost is driven by storage + inference computation
```

**Key questions to ask:**
- What's the current model performance baseline (NDCG@10, CTR, revenue lift)?
- How much latency budget do we have for feature computation itself (vs. model inference)?
- What's the acceptable drop in performance if we reduce features (1%, 5%, 10%)?
- Are all 870 features used in production, or are many dead weight?
- What's the revenue impact of each 0.1% improvement in CTR?

---

### Part 2: Feature Importance Analysis (10 min)

**What you should propose:**

```
Step 1: Rank features by importance
├─ Method 1: Permutation importance on validation set
│  └─ Remove each feature, see performance drop
│  └─ Identifies features with high predictive power
│
├─ Method 2: SHAP values
│  └─ Explains feature contribution per prediction
│  └─ Catches non-linear interactions
│
├─ Method 3: Information gain / mutual information
│  └─ For categorical features
│
└─ Method 4: Correlation analysis
   └─ Remove highly correlated features (keep the stronger one)

Step 2: Categorize features by freshness requirement
├─ Real-time features: Session behavior, current context (need <1 min updates)
├─ Near-real-time: User behavior (update hourly)
└─ Static features: Demographics, historical aggregations (update daily/weekly)

Step 3: Identify compute-expensive features
├─ Embedding lookups: Which embeddings? How many inference calls?
├─ Complex cross-features: Do they compound latency?
└─ Time-series aggregations: Sliding windows, expensive?
```

**Code Sketch:**
```python
import pandas as pd
from sklearn.inspection import permutation_importance
import shap

# Permutation importance
result = permutation_importance(
    model, X_val, y_val, n_repeats=10, random_state=42
)
feature_importance = pd.DataFrame({
    'feature': X_val.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

# Top 20% of features cover 80% of importance?
cumsum = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()
top_features = feature_importance[cumsum <= 0.80]
print(f"Top {len(top_features)} features ({len(top_features)/len(feature_importance)*100:.1f}%)")

# SHAP for complex interactions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)  # Visualize feature importance
```

---

### Part 3: Dimensionality Reduction Strategy (15 min)

**What you should propose (pick 2-3 approaches):**

#### Approach 1: Feature Selection (Aggressive)

```
Goal: Keep only top-K features that drive 90% of performance

Implementation:
  1. Run permutation importance on current model
  2. Iteratively train models with top-K features (K=50, 100, 150, ...)
  3. Plot performance vs K
  4. Choose K where performance drops <1% (e.g., K=150)
  
Result: 870 → 150 features (83% reduction)

Trade-off:
  ✅ Faster training (24hr → 4hr)
  ✅ Faster inference (100ms → 30ms)
  ✅ Smaller storage (petabyte → terabyte)
  ❌ Lose tail performance on edge cases
  ❌ Less interpretability on removed features

When to use: If <1% performance drop is acceptable
```

```python
# Progressive feature selection
from sklearn.feature_selection import SelectKBest, f_classif

for k in [50, 100, 150, 200, 300]:
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_train, y_train)
    
    model = xgb.XGBRanker(...)
    model.fit(X_selected, y_train)
    
    val_score = model.score(selector.transform(X_val), y_val)
    print(f"k={k}: NDCG@10={val_score:.4f}, latency reduction=~{(870-k)/870*100:.1f}%")
```

#### Approach 2: Feature Engineering (Smart Combinations)

```
Goal: Replace 870 raw features with 50-100 high-signal derived features

Principles:
  1. Domain-driven: What matters for recommendations?
     ├─ User affinity to item category (user × item interaction)
     ├─ Temporal patterns (time-of-day effects)
     ├─ Content similarity (embedding-based)
     └─ User engagement momentum (trending up or down?)
  
  2. Data-driven: What do SHAP + permutation importance say?
     └─ Build cross-features from top features
  
  3. Real-time feasible: Can we compute this in <20ms?
     └─ Avoid expensive embeddings, complex aggregations

Example feature engineering:
  Raw: [user_id, item_id, last_7_days_clicks, ...]
  
  Derived:
    ├─ user_affinity_to_category = (user_clicks_in_category) / (total_user_clicks)
    ├─ item_popularity_score = (item_clicks_this_week) / (total_clicks_this_week)
    ├─ user_item_similarity = cosine(user_embed, item_embed)  [precomputed]
    ├─ temporal_bias = (hour_of_day × user_click_pattern)
    ├─ recency_boost = exp(-time_since_last_interaction / 6_hours)
    ├─ user_engagement_trend = (clicks_this_week - clicks_last_week) / clicks_last_week
    └─ content_match_score = (user_interests ∩ item_topics) / len(user_interests)
  
  Result: 870 → 100 derived features (88% reduction)

Advantage:
  ✅ Still interpretable (each feature has clear meaning)
  ✅ Faster computation (no complex embeddings per inference)
  ✅ Handles cold start (formula-based, not lookup)
  ✅ More robust to distribution shift
```

#### Approach 3: Two-Stage Model Architecture

```
Goal: Keep important latency-sensitive features separate from compute-heavy ones

Architecture:
  Stage 1 (Fast Path): <30ms
    ├─ Input: User, item, context (9 features)
    ├─ Model: Lightweight neural net (2 layers)
    └─ Output: Quick recommendation score
  
  Stage 2 (Slow Path): Enrichment, only for top-K candidates
    ├─ Input: Top-100 items from Stage 1 + 150 rich features
    ├─ Model: XGBoost (full feature set)
    └─ Output: Final ranking
    └─ Latency budget: 70ms (plenty of time for 100 items)
  
  Total latency: 30ms + 70ms = 100ms ✓

Advantage:
  ✅ Most users see results in 30ms (low latency perception)
  ✅ Only re-rank top candidates (compute-efficient)
  ✅ Can use heavy features on small set
  ✅ A/B test Stage 2 independently
```

---

### Part 4: Feature Freshness Strategy (10 min)

**What you should address:**

```
Problem: Features become stale as user behavior changes

Solution by feature freshness category:

1. Static Features (update daily/weekly)
   └─ Demographics, preferences, subscription tier
   └─ Storage: Redis with daily batch updates
   └─ Lookup: <1ms cache hit

2. Near-Real-Time (update hourly)
   └─ Weekly aggregations: clicks, purchases, engagement
   └─ Storage: Time-series DB (ClickHouse, TimescaleDB)
   └─ Lookup: <5ms from cache

3. Real-Time (update per session/request)
   ├─ Current session behavior: clicks in this session
   ├─ Time-of-day context: current hour, day-of-week
   ├─ Device context: current device, network
   └─ Storage: In-memory session store (Redis Streams)
   └─ Lookup: <1ms

Feature Pipeline Architecture:
  
  Real-time features (online):
    User behavior in current session → Redis Streams
                        ↓
    Inference request → Feature server pulls from session store + cache
                        ↓
                        <10ms total

  Near-real-time (batch + cache):
    Hourly aggregation job → ClickHouse → Redis cache (with TTL)
                                              ↓
                                    Inference pulls from cache
                                              ↓
                                        <5ms lookup
  
  Static (batch):
    Daily/weekly batch job → database → Redis cache (24hr TTL)
                                            ↓
                                    Inference pulls
                                            ↓
                                        <1ms lookup
```

**Code sketch:**
```python
# Stratified feature freshness approach

class FeatureServer:
    def __init__(self):
        self.redis_static = Redis(db=0)      # Static features
        self.redis_session = Redis(db=1)     # Session features
        self.timeseries_db = ClickHouse()    # Time-series features
    
    def get_features(self, user_id, item_id, session_id):
        """Get all features with appropriate freshness."""
        start = time.time()
        
        # Real-time features (session)
        session_features = self.redis_session.hgetall(f"session:{session_id}")
        
        # Near-real-time features (cached hourly aggregates)
        agg_features = self.redis_static.hgetall(f"agg:user:{user_id}")
        if not agg_features:  # Cache miss, fetch from TimeSeries DB
            agg_features = self.timeseries_db.query(
                f"SELECT * FROM user_hourly WHERE user_id={user_id} LIMIT 1"
            )
            self.redis_static.expire(f"agg:user:{user_id}", 3600)  # 1hr TTL
        
        # Static features (cached daily)
        static_features = self.redis_static.hgetall(f"static:user:{user_id}")
        if not static_features:
            static_features = self.db.query(f"SELECT * FROM users WHERE id={user_id}")
            self.redis_static.expire(f"static:user:{user_id}", 86400)  # 24hr TTL
        
        features = {**static_features, **agg_features, **session_features}
        latency = time.time() - start
        
        assert latency < 20, f"Feature server took {latency}ms, over budget"
        return features
```

---

### Part 5: Measurement & Monitoring (10 min)

**What you should propose:**

```
Metrics to track after feature reduction:

1. Model Performance
   ├─ NDCG@10 (ranking quality)
   ├─ CTR (click-through rate)
   ├─ Revenue per user
   └─ Conversion rate

2. Feature Quality
   ├─ Feature importance distribution (Pareto check)
   ├─ Feature correlation (multicollinearity issues?)
   ├─ Feature coverage (% of predictions using each feature)
   └─ Missing feature rates (nulls, unavailable at inference)

3. Operational Metrics
   ├─ Inference latency (p50, p95, p99)
   ├─ Feature computation latency
   ├─ Storage size (disk, memory)
   ├─ Cost per 1M inferences
   └─ Feature staleness (max age of cached features)

4. Model Monitoring (Feature Drift)
   ├─ KL divergence of feature distributions (train vs prod)
   ├─ Model performance degradation rate
   ├─ Retraining cadence needed (daily? weekly?)
   └─ A/B test: Full features vs. reduced features

Dashboard:
  Real-time:
    ├─ Latency: p50, p95, p99 (alert if p99 > 100ms)
    ├─ Error rate: (missing features, compute failures)
    └─ Cache hit rate: % of requests hitting cache
  
  Daily:
    ├─ Performance: NDCG, CTR, revenue
    ├─ Feature importance: Top-10 features, distribution
    └─ Cost: $/per million inferences
  
  Weekly:
    ├─ Model degradation: NDCG trend
    ├─ Feature drift: KL divergence vs baseline
    └─ Retraining impact: Before/after model swap
```

---

### Part 6: Implementation Plan (5 min)

**What you should outline:**

```
Phase 1 (Week 1-2): Analysis
  ├─ Run permutation importance on current model
  ├─ Compute SHAP values
  ├─ Identify top-K features for 90% performance
  └─ Measure current latency breakdown

Phase 2 (Week 3): Baseline Feature Selection
  ├─ Train model with top-150 features (aggressive)
  ├─ A/B test vs. baseline (10% traffic)
  ├─ If <1% performance drop: proceed
  └─ Else: Reduce aggressiveness

Phase 3 (Week 4-5): Feature Engineering
  ├─ Build 100 domain-driven derived features
  ├─ Retrain model with selected + derived features
  ├─ A/B test (25% traffic)
  └─ Monitor latency improvement

Phase 4 (Week 6): Feature Freshness Pipeline
  ├─ Implement Redis session store
  ├─ Set up hourly aggregation job
  ├─ Deploy feature server
  └─ Measure staleness metrics

Phase 5 (Week 7-8): Monitoring & Optimization
  ├─ Deploy monitoring dashboard
  ├─ Tune caching strategies
  ├─ Auto-scaling for peak traffic
  └─ Document feature catalog

Success Criteria:
  ✓ Training latency: 24hr → 6hr (4x improvement)
  ✓ Inference latency: 100ms p99 (maintained)
  ✓ Storage: 50% reduction
  ✓ Model performance: <1% drop in NDCG@10
  ✓ Cost: 30% reduction in feature infrastructure
```

---

## Follow-Up Questions (Interviewer May Ask)

### 1. "What if we need to maintain feature backward compatibility?"

**Your answer:**
```
Problem: Removing features breaks existing feature definitions, 
         models trained on old feature set can't use new pipeline

Solution: Feature versioning + adapter layer

  Old model expects: [f1, f2, f3, ..., f870]
  New pipeline produces: [derived_f1, derived_f2, ..., derived_f100]
  
  Adapter layer: Maps new → old features
    └─ If old feature removed, compute from derived features
    └─ If exact match impossible, approximate from similar features
    └─ Version tracking: model_v1 uses legacy features, model_v2 uses new
  
  Gradual migration:
    Week 1-2: Run old + new in parallel, compare outputs
    Week 3-4: Route 10% traffic to new model
    Week 5-8: Gradually increase traffic to new model
    Week 9: Deprecate old model
```

### 2. "How would you handle features with high cardinality (e.g., item_id)?"

**Your answer:**
```
Problem: item_id has 1M+ unique values → embedding bloat

Solutions:
  1. Embedding lookup (memory-efficient)
     └─ Store only top-10K item embeddings
     └─ For tail items: use category average embedding
  
  2. Dimensionality reduction on embeddings
     └─ PCA: 768 dims → 50 dims (98% variance retained)
     └─ Quantization: float32 → int8 (4x storage reduction)
  
  3. Bucketing / hashing
     └─ Hash item_id into 1000 buckets
     └─ One-hot encode: 1000 binary features
     └─ Trade-off: Loss of granularity, but smaller feature space
  
  4. Target encoding
     └─ Replace item_id with (avg_clicks_for_this_item, avg_revenue_for_this_item)
     └─ 1 feature instead of 1M
     └─ Risk: Leakage if not careful with train/validation split
```

### 3. "What if features are correlated and removing one breaks model?"

**Your answer:**
```
Symptom: Permutation importance says feature X is important,
         but it's highly correlated with feature Y

Root cause: Model learned to use X as a proxy for Y

Solution:
  1. Check actual correlation
     └─ If corr(X, Y) > 0.9 → Keep only one
     └─ Choose the one that's cheaper to compute
  
  2. Look at SHAP values
     └─ SHAP can distinguish between X and Y contributions
     └─ If Y has higher SHAP magnitude → keep Y
  
  3. Test both in final model
     └─ Train with X only, Y only, both
     └─ See which performs best
  
  4. Consider interactions
     └─ Maybe X × Z interaction is important
     └─ Even if X alone is correlated with Y
```

---

## Expected Quality Indicators

**What makes a Staff-level answer:**

✅ **Holistic thinking:** Not just "remove features," but full system design  
✅ **Trade-off analysis:** Every decision has pros/cons, you articulate both  
✅ **Production awareness:** Latency, cost, monitoring, operational burden  
✅ **Measurement:** How you'll know if the solution works  
✅ **Feasibility:** Realistic timelines, phased rollout  
✅ **Data-driven:** Permutation importance, SHAP, A/B testing  
✅ **Scalability:** Handles 50M users, 1M items, petabyte data  
✅ **Risk mitigation:** Cold starts, feature drift, backward compatibility  

**Red flags (avoid):**

❌ "Just use PCA" without understanding what you lose  
❌ "Remove the bottom-50% features" without measurement  
❌ Ignoring latency/cost constraints  
❌ No plan for feature freshness  
❌ No monitoring strategy  
❌ Over-engineering (Spark, distributed features for 100K users)  
