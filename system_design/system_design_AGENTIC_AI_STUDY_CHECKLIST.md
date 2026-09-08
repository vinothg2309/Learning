AGENTIC_AI_STUDY_CHECKLIST.md
# Enterprise Agentic AI - Study Checklist

## Before Deep-Diving

- [ ] Read the question title & context completely
- [ ] Identify key constraints (latency SLA, scale, regulations)
- [ ] Note what makes this question "hard" (distributed, adversarial, etc.)

## Q6: Distributed Multi-Domain Agent System

**Difficulty**: HARD | **Time to Study**: 45-60 min | **Interview Value**: ★★★★★

### Concepts to Understand
- [ ] CAP theorem (consistency, availability, partition tolerance)
- [ ] Distributed consensus problem
- [ ] Two-Phase Commit vs. Saga vs. Quorum-based consensus
- [ ] Circular dependency detection (graph algorithms)
- [ ] Network partition handling & fallback strategies
- [ ] A2A latency optimization techniques

### Key Challenges
- [ ] How do agents decide when they disagree?
- [ ] How do you prevent circular deadlocks?
- [ ] What happens when a service goes down?
- [ ] How do you optimize latency to 50ms p99?

### Practice Questions
1. "If payment agent approves but risk agent disagrees later, what happens?"
2. "How do you detect & prevent circular dependencies?"
3. "What's your fallback when risk agent is unreachable?"
4. "Can you achieve 50ms p99 latency with multiple agent calls?"

### Your Story Hook
"At pp, we coordinated agents across 4 domains (payments, disputes, risk, compliance)
and had to balance strong consistency with latency requirements..."

---

## Q7: Multi-Level Feedback & Self-Improving Agent Loop

**Difficulty**: HARD | **Time to Study**: 45-60 min | **Interview Value**: ★★★★★

### Concepts to Understand
- [ ] Selection bias in feedback (only hard cases escalate)
- [ ] Outcome feedback lag & credit assignment problem
- [ ] Inverse propensity weighting
- [ ] Temporal attribution windows
- [ ] Doubly robust estimation
- [ ] Safe deployment pipeline (canary, staged rollout, auto-rollback)

### Key Challenges
- [ ] Why does training on escalations make agent worse?
- [ ] How do you handle 30-day outcome lag?
- [ ] What's the safe way to deploy an improved agent?
- [ ] How do you prevent false positive rate from increasing?

### Practice Questions
1. "Why is feedback from escalations biased?"
2. "How would you detect if a new agent policy is performing worse?"
3. "What's your rollback strategy if metrics degrade?"
4. "How would you weight feedback from different sources?"

### Your Story Hook
"We built learning loops that improved our agents over time, but we had to be
careful about feedback bias. We used staged validation to prevent negative
incidents..."

---

## Q8: Agent Reasoning Transparency & Explainability

**Difficulty**: HARD | **Time to Study**: 45-60 min | **Interview Value**: ★★★★

### Concepts to Understand
- [ ] Interpretability vs. explainability trade-offs
- [ ] SHAP (feature importance)
- [ ] Counterfactual explanations
- [ ] Attention-based explanations
- [ ] Example-based explanations
- [ ] Audit trail requirements for compliance
- [ ] Regulatory requirements (GDPR, PCI-DSS)

### Key Challenges
- [ ] How do you explain a black-box LLM agent?
- [ ] What overhead is acceptable for explainability?
- [ ] How do you prove to regulators WHY a decision was made?
- [ ] How do you handle GDPR right to explanation?

### Practice Questions
1. "How would you explain an LLM agent's decision to an auditor?"
2. "Can you add SHAP without adding 100ms latency?"
3. "What should be in the audit trail for compliance?"
4. "How would you handle 'right to explanation' requests?"

### Your Story Hook
"Compliance demanded that we explain every decision. We used SHAP for feature
importance and built full audit trails so regulators could verify our systems..."

---

## Q9: Agent System Robustness Against Adversarial Behavior

**Difficulty**: HARD | **Time to Study**: 45-60 min | **Interview Value**: ★★★★★

### Concepts to Understand
- [ ] Real adversarial attacks (fraud rings, boundary probing, arbitrage)
- [ ] Statistical pattern detection (O(1) cost)
- [ ] Randomization & jitter for defense
- [ ] Decoy policies
- [ ] Formal verification of policies
- [ ] Adaptive defense mechanisms
- [ ] False positive rate vs. security trade-off

### Key Challenges
- [ ] How do you detect 1000 coordinated accounts in 24h?
- [ ] How do you stop attackers from finding exact thresholds?
- [ ] How do you prevent regulatory arbitrage?
- [ ] How do you keep false positive rate below 1%?

### Practice Questions
1. "How would you detect a dispute fraud ring?"
2. "How would you defend against boundary probing?"
3. "What's your false positive tolerance for security?"
4. "How would you prevent policy-regulation divergence?"

### Your Story Hook
"We had to think like attackers. We detected fraud patterns in real-time using
statistical tests, and when we found a ring, we'd randomize decisions to prevent
boundary probing..."

---

## Q10: Multi-Currency, Multi-Regulation Compliance at Global Scale

**Difficulty**: HARD | **Time to Study**: 45-60 min | **Interview Value**: ★★★★

### Concepts to Understand
- [ ] Jurisdiction-aware routing (50+ countries)
- [ ] Applicable regulations per jurisdiction combination
- [ ] Policy conflict resolution (intersection of rules)
- [ ] Multi-currency normalization (PPP)
- [ ] Exchange rate handling
- [ ] Micro-transaction aggregation
- [ ] Latency optimization for global queries
- [ ] Regulatory audit trails

### Key Challenges
- [ ] How do you handle 50×50 jurisdiction combinations?
- [ ] What's your policy when regulations conflict?
- [ ] How do you normalize currency thresholds globally?
- [ ] How do you maintain < 100ms latency globally?

### Practice Questions
1. "How would you route a decision for US customer, Indian merchant, EU processor?"
2. "What happens when US and India rules conflict?"
3. "How do you normalize a $500 threshold across all currencies?"
4. "Can you handle 1M decisions/day globally with < 100ms latency?"

### Your Story Hook
"pp operates globally across 50+ countries. Each has different regulations
(PCI-DSS, RBI, GDPR, etc.). We pre-computed a jurisdiction matrix offline
and achieved O(1) lookup with 65ms latency..."

---

## Study Schedule (8 Hours)

### Day 1 (2 hours)
- [ ] Read Q6 (45 min) → Practice on whiteboard (75 min)

### Day 2 (2 hours)
- [ ] Read Q7 (45 min) → Practice on whiteboard (75 min)

### Day 3 (2 hours)
- [ ] Read Q8 (45 min) → Practice on whiteboard (75 min)

### Day 4 (2 hours)
- [ ] Read Q9 + Q10 (60 min) → Choose one to practice (60 min)

## Mock Interview Prep

### Before Mock Interview
- [ ] Can you whiteboard Q6 from memory? (Distributed consensus)
- [ ] Can you explain Q7's learning loop? (Feedback bias)
- [ ] Can you describe Q8's audit trail? (Compliance)
- [ ] Can you defend against Q9's attacks? (Adversarial robustness)
- [ ] Can you route Q10's decision? (Global compliance)

### During Mock Interview
- [ ] Ask clarifying questions first (scale, latency, regulations)
- [ ] Think out loud (show reasoning)
- [ ] Discuss trade-offs explicitly
- [ ] Ground examples in pp experience
- [ ] Handle edge cases gracefully
- [ ] Show monitoring & observability mindset

### After Mock Interview
- [ ] What feedback did you get?
- [ ] Which questions felt strongest?
- [ ] Where did you struggle?
- [ ] What would you change next time?

---

## Company-Specific Focus

### If Careem Interview
- [ ] Focus: Q6 + Q10 (real-time decisions, multi-region)
- [ ] Mention: Q7 (learning from ride patterns)
- [ ] Practice: Latency optimization

### If AI71 Interview
- [ ] Focus: Q6 (A2A latency optimization)
- [ ] Focus: Q7 (safe learning with fast feedback)
- [ ] Mention: Q9 (robustness testing)

### If OpenAI/Anthropic Interview
- [ ] Focus: Q7 + Q8 (learning, safety, explainability)
- [ ] Focus: Q9 (adversarial robustness)
- [ ] Mention: Q5 from original file (governance)

### If DeepMind Interview
- [ ] Focus: Q6 (consensus algorithms)
- [ ] Focus: Q7 (feedback bias, causal inference)
- [ ] Mention: Q8 (interpretability)

### If Meta Interview
- [ ] Focus: Q6 (large-scale coordination)
- [ ] Focus: Q10 (global infrastructure)
- [ ] Mention: Q9 (security at scale)

---

## Quick Reference: Key Numbers to Remember

- Q6: 50ms p99 latency for A2A, quorum = 2/3 consensus
- Q7: 5% canary test, 24h minimum before next stage, 2-3 days total
- Q8: < 100ms overhead for explainability, ~200B per decision audit log
- Q9: < 1% false positive rate for security alerts, O(1) detection cost
- Q10: 65ms latency globally, 50×50 jurisdiction matrix, < $0.001 per decision cost

---

## Success Criteria

You know this content when you can:

- [ ] Explain Q6 without reading (consensus, partition handling, latency optimization)
- [ ] Explain Q7 without reading (feedback bias, safe deployment, metrics)
- [ ] Explain Q8 without reading (SHAP, counterfactuals, audit trails)
- [ ] Explain Q9 without reading (attack detection, randomization, adaptive defense)
- [ ] Explain Q10 without reading (jurisdiction routing, multi-currency, global latency)
- [ ] Handle follow-up questions on all 5 (interviewers WILL ask 2-3 follow-ups)
- [ ] Connect each to your pp experience naturally
- [ ] Discuss trade-offs without prompting

---

Good luck! You've got this. 💪

