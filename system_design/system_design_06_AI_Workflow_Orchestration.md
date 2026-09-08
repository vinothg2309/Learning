06_AI_Workflow_Orchestration.md
# AI Workflow Orchestration Interview Questions

## Q1: Design an End-to-End AI Workflow for Autonomous Merchant Risk Assessment

**Context**: Design a production workflow that automatically ingests merchant transactions, analyzes risk factors, flags suspicious patterns, and initiates remediation actions—all autonomously with human escalation when needed. Must scale to 10M+ daily transactions with sub-5s decision latency.

### Follow-up Questions:
- How would you orchestrate multiple dependent AI tasks?
- What's your approach to error handling and retries?
- How do you ensure data consistency across workflow steps?
- How would you monitor workflow health and identify bottlenecks?

### Architecture Diagram:
```
┌──────────────────────────────────────────────────────────┐
│      AI WORKFLOW ORCHESTRATION (Merchant Risk)           │
│              Apache Airflow + Vertex AI Workflows        │
└──────────────────────────────────────────────────────────┘

WORKFLOW: "Autonomous Merchant Risk Assessment"

┌─────────────────────────────────────────┐
│  Data Ingestion Layer                   │
│  ├─ Kafka: Real-time transaction stream│
│  ├─ BigQuery: Batch merchant profiles  │
│  ├─ API: Risk policy updates           │
│  └─ Schedule: Hourly batch processing  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Task 1: Data Preparation               │
│  ├─ Join: Transactions + Merchant data  │
│  ├─ Feature engineering: Risk signals   │
│  ├─ Deduplication: Duplicate removal    │
│  ├─ SLA: <30s for batch                │
│  └─ Output: Prepared dataset (GCS)      │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
┌────────▼────────┐   ┌───▼─────────────┐
│  Task 2a:       │   │  Task 2b:       │
│  Fraud Scoring  │   │ Risk Assessment │
│  ├─ Model: XGB  │   │ ├─ Model: LLM   │
│  ├─ Latency:    │   │ ├─ Latency:     │
│  │  <1s         │   │ │  <2s          │
│  └─ Output:     │   │ └─ Output:      │
│    fraud_score  │   │   risk_level    │
└────────┬────────┘   └────────┬────────┘
         │                     │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Task 3: Decision   │
         │  Making             │
         │  ├─ Combine scores  │
         │  ├─ Check policies  │
         │  ├─ Determine action│
         │  └─ Confidence: >90%│
         └──────────┬──────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
┌───────▼────────┐   ┌────────▼──────┐
│ Task 4a: LOW   │   │ Task 4b: HIGH │
│ Risk           │   │ Risk/ESCALATE │
│ ├─ Action: OK  │   │ ├─ Action:    │
│ ├─ Log result  │   │ │ Review      │
│ ├─ Monitor     │   │ ├─ Notify     │
│ └─ Next cycle  │   │ │ Analyst     │
└────────────────┘   │ └─ Wait for   │
                     │   Decision    │
                     └───────┬───────┘
                             │
                   ┌─────────▼─────────┐
                   │  Task 5: Action   │
                   │  Execution        │
                   │  ├─ If approved:  │
                   │  │ ├─ Blacklist   │
                   │  │ ├─ Limit       │
                   │  │ └─ Investigate │
                   │  ├─ SLA: <1s      │
                   │  └─ Idempotent    │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼──────────┐
                   │  Task 6: Monitoring│
                   │  & Reporting       │
                   │  ├─ Log all actions│
                   │  ├─ Metrics:       │
                   │  │ ├─ Blocked txns │
                   │  │ ├─ Escalations  │
                   │  │ └─ Accuracy     │
                   │  ├─ Alerts (>10%   │
                   │  │ escalation)     │
                   │  └─ Dashboard      │
                   └────────────────────┘
```

### Workflow Orchestration Details:

**Workflow Definition (YAML/Python)**:
```python
# Using Apache Airflow
@dag(dag_id='merchant_risk_assessment', schedule_interval='@hourly')
def workflow():
    # Task 1: Data Prep
    prep = prepare_data()
    
    # Task 2a, 2b: Parallel execution
    fraud_scores = fraud_scoring(prep)
    risk_scores = risk_assessment(prep)
    
    # Task 3: Combine results
    decision = make_decision([fraud_scores, risk_scores])
    
    # Task 4a/4b: Conditional branching
    low_risk = execute_action_low_risk(decision)
    high_risk = escalate_human(decision)
    
    # Task 6: Monitoring
    monitor = log_results([low_risk, high_risk])
    
    prep >> [fraud_scores, risk_scores] >> decision >> [low_risk, high_risk] >> monitor

workflow()
```

**Error Handling & Retries**:
```
Task Failure Strategy:

For idempotent tasks (Data Prep):
├─ Retries: 3
├─ Backoff: exponential (5s → 10s → 20s)
├─ Recovery: Restart from checkpoint
└─ Alert: If all retries fail

For non-idempotent tasks (Blacklist action):
├─ Retries: 0 (prevent duplicate action)
├─ Idempotency key: {merchant_id}_{timestamp}
├─ Recovery: Check if already executed
└─ Alert: Immediately on failure

Downstream Impact:
├─ If Task 1 fails: Entire workflow skips (no data)
├─ If Task 2a fails (fraud score): Use default score (0.5)
├─ If Task 3 fails: Escalate to human (conservative)
└─ If Task 5 fails: Retry with exponential backoff
```

**State Management**:
```
Workflow State Checkpoint:
├─ After Task 1: Save prepared dataset (GCS path)
├─ After Task 2: Save scores (BigQuery)
├─ After Task 3: Save decision (Firestore)
└─ After Task 5: Save execution log

Benefits:
├─ Resume from checkpoint (no re-computation)
├─ Audit trail (who did what, when)
├─ Debugging (replay failed task)
└─ Cost: Save ~30% on re-runs
```

**Monitoring & Alerting**:
```
Metrics Dashboard:
├─ Workflow execution time (target: <5s per batch)
├─ Task-level latency breakdown
├─ Success rate per task (target: >99.9%)
├─ Escalation rate (target: <5% of decisions)
├─ Error rate by type (timeout, OOM, etc.)
├─ Resource utilization (CPU, memory, GPU)
└─ Cost per workflow run

Alerts:
├─ Workflow SLA violation: Notify ops
├─ Task failure rate > 1%: Page on-call
├─ Escalation rate > 10%: Investigate model
├─ Data quality issue: Block processing
└─ Resource exhaustion: Scale up
```

---

## Q2: Design a Complex Multi-Step Workflow with Conditional Logic

**Context**: Design a workflow for "AI-Powered Dispute Resolution" that involves multiple AI models, tool calls, and human-in-the-loop stages. Example flow: Intake → Evidence Analysis → Policy Check → Decision → Communication.

### Workflow Steps:

```
Step 1: INTAKE (Validation)
├─ Extract dispute details from customer submission
├─ Validate required fields (merchant, amount, date)
├─ Classify dispute type (chargeback, item not received, etc.)
├─ SLA: <10s
└─ Escalation: Missing critical info → Request from customer

Step 2: EVIDENCE COLLECTION
├─ Retrieve: Customer evidence, merchant response, transaction details
├─ Multimodal: Text, images, receipts, shipping tracking
├─ Timeline: Reconstruct event sequence
├─ Completeness: Flag missing evidence
└─ Scoring: Evidence quality score (0-1)

Step 3: EVIDENCE ANALYSIS (LLM)
├─ Analyze: Does customer evidence support claim?
├─ Extract key facts (dates, amounts, descriptions)
├─ Assess: Strength of evidence (strong/moderate/weak)
├─ Reasoning: Provide chain-of-thought explanation
└─ Confidence: <0.7 → Escalate to human

Step 4: POLICY MATCHING (RAG)
├─ Query: "What policy applies to this dispute type?"
├─ Retrieve: Relevant policy sections
├─ Matching: Evidence vs policy requirements
├─ Precedent: Similar dispute outcomes
└─ Legal: Risk assessment (dispute win probability)

Step 5: DECISION (Multi-Agent)
├─ Agents involved:
│  ├─ Evidence Agent: "Evidence is strong?"
│  ├─ Policy Agent: "Policy supports customer?"
│  ├─ Risk Agent: "What if we lose at chargeback?"
│  └─ Merchant Agent: "Any flags on merchant?"
├─ Consensus: Voting-based decision
├─ Output: Decision (refund/deny/escalate) + confidence
└─ Threshold: High-confidence → Auto-decision, Low → Human

Step 6: HUMAN REVIEW (HITL)
├─ Cases: Low confidence (< 0.75) or high-value (>$5K)
├─ Presentation: Evidence summary, AI recommendation, policy refs
├─ Tools: Zoom into details, add notes, override decision
├─ SLA: 2-4 hours for human review
└─ Output: Final decision + human notes

Step 7: CUSTOMER COMMUNICATION
├─ Generate: Personalized letter (empathetic, policy-grounded)
├─ Reason: Explain decision in simple terms
├─ Recourse: "You can escalate within 30 days"
├─ Multi-channel: Email, SMS, portal notification
└─ Follow-up: Track acknowledgment, send reminders

Step 8: MERCHANT NOTIFICATION
├─ Alert: Dispute outcome (refund, chargeback cost)
├─ If refunded: "Item/fund returned to customer"
├─ If denied: "Customer's dispute claim not substantiated"
└─ Portal: Detailed evidence & decision rationale

Step 9: MONITORING & LEARNING
├─ Metrics:
│  ├─ Customer satisfaction (post-resolution survey)
│  ├─ Appeal rate (do customers escalate?)
│  ├─ Accuracy (vs chargeback actual outcome)
│  └─ Speed (days to resolution)
├─ Learning:
│  ├─ Difficult cases → Fine-tune model
│  ├─ Appeal patterns → Update policy interpretation
│  └─ Agent performance → Retrain agents
└─ Continuous improvement cycle
```

### Conditional Branching Logic:

```
if evidence_quality_score < 0.3:
    escalate_to_human("Missing critical evidence")
elif policy_confidence < 0.75:
    escalate_to_human("Low policy match confidence")
elif dispute_amount > $5000:
    escalate_to_human("High-value dispute")
else:
    if all_agents_agree_on_decision:
        auto_execute_decision()
        send_notification()
    else:
        escalate_to_human("Agent disagreement")
```

---

## Q3: Design a Distributed Workflow Engine for 1M+ Daily Tasks

**Context**: Build a workflow engine that can schedule, execute, and monitor 1M+ workflows daily with reliability, fault tolerance, and cost optimization. Must handle retries, backpressure, and ensure exactly-once semantics.

### Architecture:

```
DISTRIBUTED WORKFLOW ENGINE

Task Queue (Kafka):
├─ Topic: workflow_tasks
├─ Partitions: 100 (parallelism)
├─ Throughput: 10K tasks/sec
├─ Durability: Retention 7 days
└─ Consumer group: Worker pool

Worker Pool (Stateless):
├─ Count: 100-1000 (auto-scale by queue depth)
├─ Each worker: Poll tasks, execute, report result
├─ Heartbeat: 30s (detect dead workers)
├─ Graceful shutdown: Complete current task first
└─ Resource: CPU-optimized (not GPU)

Task Execution:

Exactly-Once Semantics:
├─ Idempotency key: Stored in task
├─ Before execution: Check if already done
├─ If yes: Return cached result
├─ If no: Execute + cache result
└─ Prevents duplicate actions (critical for payments)

Checkpointing:
├─ Save state after each task step
├─ Enables resume from failure
├─ Cost: Saves re-computation time
└─ Trade-off: I/O overhead vs speed

Backpressure Handling:
├─ If queue grows > 1M tasks: Pause ingestion
├─ Scale workers: If latency > SLA
├─ Batching: Group small tasks (amortize overhead)
└─ Priority: High-priority tasks processed first

Monitoring:
├─ Queue depth (alert if >500K)
├─ Worker health (% healthy)
├─ Task latency distribution (p50/p95/p99)
├─ Task failure rate by type
├─ Cost per task (compute + storage)
└─ Throughput: Tasks/sec processed
```

---

## Q4: Design a Real-Time Stream Processing Workflow

**Context**: Build a real-time workflow that processes payment transactions as they arrive, detects anomalies, and takes immediate action (block, flag, notify).

### Key Components:

```
Stream Processing Pipeline:

Data Source (Kafka):
├─ Topic: transactions
├─ Rate: 100K events/sec
├─ Latency: End-to-end <1s
└─ Data: {customer_id, merchant, amount, timestamp, ...}

Stream Operators:

1. Windowing:
├─ Tumbling window: 1-minute windows
├─ Purpose: Aggregate per-customer metrics
├─ Output: {customer_id, total_amount, num_txns}

2. Stateful Processing:
├─ Maintain: Per-customer state (running balance, velocity)
├─ Update: With each new transaction
├─ Query: Check against thresholds
└─ TTL: State expires after 7 days of inactivity

3. Enrichment:
├─ Join: Transaction + Customer profile
├─ Join: Transaction + Merchant risk score
├─ Join: Transaction + Geolocation data
└─ Latency: <100ms (using side input cache)

4. Anomaly Detection:
├─ Model: Isolation Forest (fast, outlier detection)
├─ Features: Amount, velocity, location, merchant_risk
├─ Output: Anomaly score (0-1)
├─ Threshold: >0.8 → Block or flag

5. Alerts & Actions:
├─ Block action: Reject transaction immediately
├─ Flag action: Store in investigation queue
├─ Notify: Send SMS/email to customer
└─ Logging: Store all decisions for audit

Output Sinks:

Blocking Service:
├─ Latency: <50ms
├─ Action: Immediately reject transaction
└─ Customer: Receives error "Transaction declined"

Investigation Queue:
├─ Flagged transactions: Stored for analyst review
├─ Priority: Risk score determines review order
└─ HITL: Analyst can override block decision

Analytics DB:
├─ Store: All transactions + decisions
├─ Purpose: Model training, pattern analysis
├─ Partitioning: By day, customer_id
└─ TTL: 2 years (compliance)

Monitoring:
├─ Latency: p95 < 500ms (end-to-end)
├─ False positive rate: <1% (user complaints)
├─ True positive rate: >95% (catch actual fraud)
├─ Throughput: 100K events/sec sustainable
└─ Cost: <$0.001 per transaction
```

---

## Q5: Design a Workflow for A/B Testing & Experimentation

**Context**: Build a workflow that automatically runs A/B tests on AI models, statistical analysis, and safe rollout of winning variants.

### A/B Test Workflow:

```
Stage 1: SETUP
├─ Define experiment: Control (old model) vs Treatment (new model)
├─ Split: Random 50/50 routing
├─ Duration: 7 days (1M transactions)
├─ Metrics: Primary (accuracy), Secondary (latency, cost)
├─ Power: 95% confidence, 5% significance

Stage 2: EXECUTION
├─ Route: 50% traffic to each variant
├─ Collect: Metrics for each variant
├─ Monitor: Real-time dashboard
├─ Guardrails: If error rate >1%, stop experiment
└─ Early stopping: If clear winner emerges

Stage 3: ANALYSIS
├─ Statistical test: Two-sample t-test on accuracy
├─ Calculate: P-value, confidence interval
├─ Check: Is improvement significant (p < 0.05)?
├─ Secondary metrics: Latency, cost impact
└─ Decision: Declare winner or inconclusive

Stage 4: ROLLOUT
├─ Winner validation: Human review
├─ Gradual rollout: 10% → 25% → 50% → 100% (with pauses)
├─ Monitor: Metrics in production
├─ Canary: If any metric degrades, rollback
└─ Final: After 24h at 100%, declare success

Cost Tracking:
├─ Control: $1000 (serving old model)
├─ Treatment: $800 (serving new model)
├─ Improvement: 20% cost reduction
└─ Total experiment cost: $1800 (1 week)
```

---

## Interview Tips for Top Companies:

1. **Distributed Systems Thinking**: Discuss CAP theorem, consistency models, failure modes
2. **Monitoring & Observability**: How you'd debug a slow workflow at 3am
3. **Cost Awareness**: Calculate cost per workflow, optimize resource allocation
4. **Safety First**: Always have a fallback—escalate to humans when uncertain
5. **Production Lessons**: Share real incident (what went wrong, how you fixed it)

---

## References to Your pp Experience:

- Multi-agent orchestration (A2A workflows via MCP protocol)
- Sub-2s p95 latency on complex workflows
- 99.9% uptime across 50+ AI services
- Autonomous customer support with HITL escalation (40% resolution improvement)
- Redis checkpoint memory for stateful workflows
- Automated 600+ hours annually (workflow automation)
