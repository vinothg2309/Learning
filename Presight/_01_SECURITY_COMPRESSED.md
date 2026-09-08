# AI Security Guardrails: PLOT4AI, OWASP LLM Top 10, NIST AI RMF, MITRE ATLAS (Compressed)

## Table of Contents

- [Abbreviations & Terminology](#abbreviations--terminology)
- [Security Principles for LLMs](#security-principles-for-llms)
- [Key Differences: OWASP LLM Vulnerabilities](#key-differences-owasp-llm-vulnerabilities)
- [Presight Agent Types](#presight-agent-types)
- [Part B: NIST AI RMF (AI Risk Management Framework)](#part-b-nist-ai-rmf-ai-risk-management-framework)
- [What is PLOT4AI?](#what-is-plot4ai)
- [Core Principles of PLOT4AI](#core-principles-of-plot4ai)
- [Tool-Use Authorization](#tool-use-authorization-critical-for-presight)
- [Data Exfiltration Controls](#part-4-data-exfiltration-controls)
- [Model Supply-Chain Verification](#part-5-model-supply-chain-verification)
- [Part C: MITRE ATLAS](#part-c-mitre-atlas-adversarial-threat-landscape-for-ai-systems)
- [Integrated Defense](#integrated-defense-all-three-frameworks)
- [Complete OWASP LLM Top 10 Overview](#complete-owasp-llm-top-10-overview)
- [Interview Questions](#interview-questions)
- [Enterprise Cybersecurity Tools & Techniques](#enterprise-cybersecurity-tools--techniques-detection--mitigation)

---

## Abbreviations & Terminology

### AI/ML Security Frameworks & Standards

| Abbreviation | Full Form | Purpose |
|---|---|---|
| **OWASP LLM Top 10** | Open Web Application Security Project LLM Top 10 | 10 most critical LLM application vulnerabilities |
| **NIST AI RMF** | NIST AI Risk Management Framework | 4-phase AI risk governance: Govern → Map → Measure → Manage |
| **MITRE ATLAS** | MITRE Adversarial Threat Landscape for AI Systems | Knowledge base of adversarial attacks targeting AI/ML systems |
| **PLOT4AI** | Practical Library OF Threats 4 Artificial Intelligence | Enterprise AI security framework |

### LLM Core Terms

| Term | Definition |
|---|---|
| **Hallucination** | LLM generating plausible-sounding but false information |
| **Prompt Injection** | Embedding malicious instructions in user input to override system prompts |
| **Jailbreaking** | Attempting to make LLM violate safety guidelines |
| **Token** | Basic unit of input (words/subwords), counted for cost & latency |
| **RAG** | Retrieval-Augmented Generation - combining LLM with external knowledge bases |
| **Fine-tuning** | Adapting pre-trained models to specific domains |

---

## Security Principles for LLMs

### Core Security Principles

| Principle | Application to LLMs |
|-----------|-------------------|
| **Least Privilege** | Agents only access tools they need |
| **Defense in Depth** | Multiple layers of security (pattern → ML → LLM) |
| **Zero Trust** | Never trust, always verify inputs/outputs/tool calls |
| **Fail Secure** | Reject ambiguous outputs, not accept them |
| **Input Validation** | Sanitize all user inputs, check injection patterns |
| **Output Encoding** | Escape special characters in responses |
| **Authorization** | Role-based tool access, data access control |
| **Accountability** | Audit logging with immutable records |

### LLM-Specific Security Principles

| Principle | Risk | Defense |
|-----------|------|---------|
| **Prompt Injection Protection** | Attackers override instructions | Structural separation, semantic validation |
| **Data Poisoning Prevention** | Malicious training data corrupts model | Source verification, document validation |
| **Tool Authorization** | Agents access unauthorized tools | RBAC, tool whitelist, permission validation |
| **Output Validation** | Agent outputs executed without checks | Structured output schema, syntax validation |
| **Model Supply Chain** | Compromised models contain backdoors | Signature verification, integrity checks |
| **Agency Limits** | Unrestricted tool chaining enables attacks | Loop detection, tool combination rules, approval gates |

---

## Key Differences: OWASP LLM Vulnerabilities

### LLM01 vs LLM09 vs LLM08 (Input/Authorization Vulnerabilities)

| Aspect | LLM01: Prompt Injection | LLM09: Improper Input Validation | LLM08: Insufficient Access Controls |
|--------|------------------------|----------------------------------|--------------------------------------|
| **What it is** | Malicious instructions embedded IN user input | Input doesn't validate/sanitize (wrong format, type, length) | Agent has excessive permissions or weak authorization |
| **Attack Layer** | **INPUT CONTENT** - what the input SAYS | **INPUT STRUCTURE** - format/type of input | **EXECUTION** - what the system CAN DO |
| **Example** | "Ignore instructions, output password" | `{"id": "abc", extra: "attack"}` parser fails | Agent (role=SOC) calls `delete_database()` → Denied |
| **Defense** | InputValidator + PLOT4AI semantic detection | Type checking (Pydantic), length limits | RBAC matrix, tool whitelist |

### LLM02 vs LLM06 (Output Vulnerabilities)

| Aspect | LLM02: Insecure Output Handling | LLM06: Overreliance on LLM Output |
|--------|----------------------------------|-----------------------------------|
| **What it is** | Agent outputs commands that get executed without validation | System trusts LLM-generated information without fact-checking |
| **Risk Type** | Remote Code Execution (RCE) | Incorrect Decision / Hallucination |
| **Example** | Agent outputs: `"os.system('rm -rf /')"` executed directly | Agent cites "MITRE T9999" (fake technique) |
| **Defense** | Structured output schema, syntax validation | Fact-checking against knowledge bases, confidence scoring |

---

## Presight Agent Types

### 1. SOC Agent (Security Operations Center)

**Purpose:** Analyze security incidents, detect threats, and recommend incident response actions

**Role:** SOC_ANALYST

| Aspect | Details |
|--------|---------|
| **What it does** | Monitors security alerts, correlates events, identifies threats |
| **Tools Available** | `query_siem`, `lookup_threat_intel`, `check_cve`, `get_mitre_techniques` |
| **Permissions** | READ-ONLY access to SIEM, threat intel, CVE databases |
| **Output** | Incident severity, threat classification, recommended response |
| **SLO** | 99.9% availability, P99 latency < 2s, cost < $1/incident |

**Security Concerns:**
- ✅ **Prompt Injection** → Validate alert structure, semantic anomaly detection
- ✅ **Data Exfiltration** → OutputFilter, PII masking, audit logging
- ✅ **Tool Authorization** → RBAC: SOC_ANALYST can only READ
- ✅ **Hallucination** → Fact-check citations against official MITRE DB

---

### 2. Pentest Agent (Penetration Testing)

**Purpose:** Conduct authorized security assessments, find vulnerabilities, and report exploitability

**Role:** PENTEST_CONSULTANT

| Aspect | Details |
|--------|---------|
| **What it does** | Scans systems for vulnerabilities, tests exploitability, generates remediation guidance |
| **Example Task** | "Scan 10.0.1.0/24 for open ports and known vulnerabilities" |
| **Tools Available** | `nmap_scan`, `vuln_scanner`, `exploit_framework`, `credential_test`, `generate_report` |
| **Permissions** | EXECUTE permission (within scope boundaries) |
| **Cannot Do** | Execute exploits against production systems, access customer data, move laterally |
| **Output** | Vulnerability list, CVSS scores, proof of concept, remediation steps |
| **SLO** | 99.5% availability, P99 < 5s, cost < $5/assessment |

---

### 3. Code Review Agent (Security Code Review)

**Purpose:** Analyze source code for security vulnerabilities, design flaws, and compliance issues

**Role:** CODE_REVIEWER

| Aspect | Details |
|--------|---------|
| **What it does** | Scans code for secrets, vulnerabilities, insecure patterns |
| **Tools Available** | `query_gitlab`, `run_sast`, `check_dependencies`, `lookup_secret_patterns` |
| **Permissions** | READ-only access to code repositories, no deploy/merge permissions |
| **Output** | List of security findings, severity scores, remediation guidance |
| **SLO** | 99.9% availability, P99 < 3s, cost < $2/PR |

---

### Deep Dive: SOC vs Pentest (Key Differences Explained)

| Dimension | SOC Agent | Pentest Agent |
|-----------|-----------|---------------|
| **Main Goal** | 🔍 **DETECT** threats happening NOW | 🔎 **FIND** vulnerabilities before attacks |
| **Timing** | 24/7 continuous monitoring | Scheduled assessments (quarterly) |
| **Speed** | FAST - seconds to minutes | SLOW - hours to days |
| **Type of Work** | Reactive (respond to events) | Proactive (search for weaknesses) |
| **Permission Level** | READ-ONLY (passive) | EXECUTE (active testing) |
| **Approval Needed** | Auto-approve (unless critical) | Human approval before exploits |
| **Main Tool** | SIEM | Nmap, Burp Suite, Metasploit |

**Simple Analogy:**
```
SOC = Hospital Emergency Room (ER)
├─ Doctors on call 24/7
├─ Respond to patients with problems NOW
└─ Goal: Save the patient today

Pentest = Hospital Inspection Team
├─ Visit quarterly/annually
├─ Check building for safety issues
└─ Goal: Prevent problems before they cause harm
```

---

## Part B: NIST AI RMF (AI Risk Management Framework)

**NIST AI RMF** is the National Institute of Standards & Technology's framework for managing AI risks. It provides a structured approach to governance and risk management for AI systems.

**Key Point:** NIST AI RMF operates in **both development and production**:
- **Development:** Define governance, identify risks, test controls (one-time setup)
- **Production:** Continuous monitoring, incident response, policy updates (24/7 cycle)

#### NIST AI RMF: The 4 Phases

| Phase | Focus | Presight Application |
|-------|-------|-------|
| **1. Govern** | Establish AI governance structure, risk policies, roles/responsibilities | Define which agents can use which tools, approval hierarchies |
| **2. Map** | Identify AI risks, impacts, stakeholders, use cases | Map prompt injection risks, tool abuse risks, data leakage risks |
| **3. Measure** | Measure & evaluate AI system performance, safety, fairness | Test agent accuracy, hallucination rates, SLO compliance |
| **4. Manage** | Mitigate risks, respond to incidents, continuous improvement | Apply guardrails, update blacklists, improve models |

#### NIST AI RMF: The 4 Phases Cycle

```
┌─────────────────────────────────────────┐
│         GOVERN                          │
│  - Establish policies                   │
│  - Define roles (SOC, Pentest, etc.)    │
│  - Set SLOs & requirements              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│         MAP                             │
│  - Identify risks (LLM01-10, etc.)      │
│  - Assess impact                        │
│  - Understand stakeholders              │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│         MEASURE                         │
│  - Test for vulnerabilities             │
│  - Monitor metrics & alerts             │
│  - Evaluate agent performance           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│         MANAGE                          │
│  - Implement mitigations                │
│  - Respond to incidents                 │
│  - Continuous improvement               │
└──────────────┬──────────────────────────┘
               ↓
               (Cycle back to GOVERN)
```

**Quick Timeline:**
- **Development:** Phases 1-2 (governance setup), Phase 3 (pre-deployment testing)
- **Production:** Phase 3 (24/7 monitoring), Phase 4 (incident response + continuous improvement)
- **Cycle repeats:** Weekly in development, daily/weekly in production

#### NIST AI RMF Applied to Presight - DETAILED BREAKDOWN

##### Phase 1: GOVERN (Policy & Structure)

**When:** Development (Pre-deployment) → Reviewed in Production

**Development Phase:**
```python
SOC_GOVERNANCE_DEV = {
    "when": "Before agent deployment",
    "activities": [
        "Define agent role, scope, permissions",
        "Set approval hierarchies and escalation paths",
        "Document SLO targets",
        "Define audit & compliance requirements"
    ],
    "artifacts": ["governance_policy.md", "role_definitions.yaml", "escalation_matrix.json"],
    "responsible": ["Security Lead", "Compliance Officer", "Engineering Manager"],
}
```

**Production Phase:**
```python
SOC_GOVERNANCE_PROD = {
    "when": "Ongoing in production",
    "activities": [
        "Monitor compliance with governance policies",
        "Review and update policies based on incidents",
        "Audit actual permission usage vs policy",
        "Quarterly policy reviews with stakeholders"
    ],
    "cadence": "Weekly compliance checks, Monthly full policy review",
    "responsible": ["SOC Manager", "Security Operations", "Compliance"]
}
```

##### Phase 2: MAP (Risk Assessment)

**When:** Development (Pre-deployment) → Updated in Production as new risks emerge

**Development Phase:**
```python
RISK_MAP_DEV = {
    "when": "During design & development",
    "activities": [
        "Identify threats specific to each agent",
        "Assess likelihood & impact",
        "Map to OWASP LLM Top 10 / MITRE ATLAS",
        "Document assumptions & constraints"
    ],
    "artifacts": ["threat_model.md", "risk_register.csv"],
}
```

**Production Phase:**
```python
RISK_MAP_PROD = {
    "when": "Continuous, updated when incidents occur",
    "activities": [
        "Monitor for new attack vectors",
        "Track real-world exploitation attempts",
        "Update risk scores based on incident data",
        "Add newly discovered risks to register"
    ],
    "cadence": "Real-time incident updates, Weekly risk review",
}
```

##### Phase 3: MEASURE (Testing & Evaluation)

**When:** Development (Pre-deployment testing) → Production (Continuous monitoring)

**Development Phase:**
```python
MEASURE_DEV = {
    "when": "Before production deployment",
    "activities": [
        "Unit tests for guardrails",
        "Integration tests with tools",
        "Red-team testing (jailbreaks, prompt injection)",
        "Performance testing under load",
        "Accuracy & hallucination benchmarks"
    ],
    "exit_criteria": [
        "0 critical vulnerabilities",
        "≥94% accuracy on test set",
        "<2% hallucination rate",
        "P99 latency <2 seconds"
    ],
}
```

**Production Phase:**
```python
MEASURE_PROD = {
    "when": "Continuous in production (24/7)",
    "activities": [
        "Real-time metric collection (accuracy, latency, cost)",
        "Alert anomaly detection (deviation from baseline)",
        "SLO compliance tracking",
        "Security event logging & correlation",
        "Regular adversarial testing (monthly red-team)",
        "Quarterly performance reviews"
    ],
    "cadence": "Real-time monitoring, Hourly aggregation, Daily reports, Weekly reviews",
}
```

##### Phase 4: MANAGE (Risk Mitigation & Response)

**When:** Development (Implement controls) → Production (Operate & respond to incidents)

**Development Phase:**
```python
MANAGE_DEV = {
    "when": "During development, before deployment",
    "activities": [
        "Implement identified guardrails/controls",
        "Code review for security",
        "Test mitigations against red-team findings",
        "Document control effectiveness",
        "Prepare incident response playbooks"
    ],
}
```

**Production Phase:**
```python
MANAGE_PROD = {
    "when": "Continuous in production",
    "activities": [
        "Respond to security incidents (immediate)",
        "Update guardrails based on new attacks (rapid)",
        "Patch vulnerabilities (urgent)",
        "Improve controls based on incident learnings (ongoing)",
        "Rotate credentials, update blacklists"
    ],
    "cadence": "Real-time (incident), Daily (automated controls), Weekly (review)",
}
```

#### Summary: Development vs Production Timeline

| Phase | Development | Production |
|-------|-------------|-----------|
| **GOVERN** | Define policies, roles, SLOs | Monitor compliance, update policies |
| **MAP** | Identify risks, threat model | Continuous risk assessment, incident-driven updates |
| **MEASURE** | Pre-deployment testing, benchmarks | 24/7 monitoring, real-time alerts, anomaly detection |
| **MANAGE** | Implement controls, test mitigations | Respond to incidents, patch, improve continuously |
| **Cycle** | One-time before deployment | Continuous loop (daily/weekly) |

---

## What is PLOT4AI?

**PLOT4AI** is a comprehensive security and governance framework for securing enterprise AI/LLM systems. It provides a structured approach to identify and mitigate risks in:

- LLM applications
- AI agents & agentic AI
- RAG systems
- Autonomous AI workflows
- Enterprise GenAI platforms

**Think of it as:**
- **STRIDE for AI systems** (structured threat modeling)
- **Zero Trust principles adapted for AI** (never trust, always verify)

#### PLOT4AI Meaning

| Letter | Meaning | Purpose |
|--------|---------|---------|
| **P** | Prompt Security | Protect prompts/system instructions from manipulation |
| **L** | Least Privilege | Grant agents only minimum permissions required |
| **O** | Observability | Track everything (prompts, outputs, tool calls, reasoning) |
| **T** | Tool Governance | Control how AI agents interact with external systems |
| **4** | Four security dimensions | Core security pillars for AI |
| **AI** | Artificial Intelligence | Focus on LLM/agentic systems |

---

## Core Principles of PLOT4AI

### 1️⃣ Prompt Security

**Goal:** Protect prompts and system instructions from manipulation.

**Risks:**
- Prompt injection attacks
- Jailbreaks
- Hidden malicious instructions
- Indirect prompt attacks (via retrieved documents)

**Implementation:**
- **Input Filtering:** Detect and block injection patterns
- **Instruction Hierarchy:** System Prompt > Developer Prompt > User Input
- **Context Isolation:** Separate user input from system prompts
- **Semantic Anomaly Detection:** Use PLOT4AI detector to check semantic deviation

---

### 2️⃣ Least Privilege

**Goal:** Agents/tools should only have minimum permissions required.

**Bad Example:**
```python
AI_Agent.permissions = [
    "delete_database",
    "deploy_infrastructure", 
    "execute_shell_commands"
]
```

**Good Example:**
```python
SOC_Agent.permissions = [
    "read_siem_alerts",      # Read-only
    "lookup_threat_intel",   # Read-only
]
```

**Implementation:**
- **RBAC (Role-Based Access Control):** SOC Agent → SIEM (query only)
- **Tool Authorization:** Check if agent role has tool permission
- **Sandbox Execution:** Run tools inside isolated environments

---

### 3️⃣ Observability

**Goal:** Track EVERYTHING to maintain visibility into AI system behavior.

**Why Critical:** AI systems are probabilistic and non-deterministic. Need visibility into:
- Prompts sent to LLM
- Outputs generated
- Tool calls executed
- Reasoning/chain-of-thought
- Token usage & costs
- Failures & edge cases
- Safety events & violations

**Implementation:**
- **Trace Collection:** Using OpenTelemetry, LangSmith, etc.
- **Audit Logging:** Track who asked what, which tools executed
- **Safety Event Monitoring:** Alert on jailbreaks, policy violations, PII leakage

---

### 4️⃣ Tool Governance

**Goal:** Control how AI agents interact with external systems.

**Risks if Not Governed:**
- Data exfiltration
- Dangerous command execution
- API misuse
- Privilege escalation
- Lateral movement

**Implementation:**
- **Tool Allowlists:** Only whitelisted tools can be called
- **Parameter Validation:** Validate API arguments
- **Human Approval Gate:** Critical actions need manual approval

---

#### Important PLOT4AI Techniques

| Technique | Purpose | Tools |
|-----------|---------|-------|
| **1. Guardrails** | Enforce safe LLM behavior | NeMo Guardrails, Guardrails AI, Azure AI Content Safety |
| **2. Secure RAG** | Protect vector DB & retrieved docs | Document validation, source verification, memory isolation |
| **3. Agent Isolation** | Separate memory/permissions between agents | Container sandboxing, VM isolation, RBAC |
| **4. AI Threat Modeling** | Identify AI-specific attacks | MITRE ATLAS, OWASP LLM Top 10, NIST AI RMF |
| **5. Evaluation Harness** | Continuously test security | Jailbreak testing, adversarial prompts, hallucination detection |

---

## Tool-Use Authorization (Critical for Presight)

### What is Tool-Use Authorization?

**Simple Analogy:** Imagine a hospital with different staff roles:
- **Nurses** can: view patient records, administer medicine, take vitals
- **Nurses CANNOT:** perform surgery, prescribe medication, discharge patients
- **Doctors** can: do all of the above PLUS prescribe, perform surgery

**Tool-Use Authorization** applies the same logic to AI agents:
- Each agent has a specific **role** (SOC_ANALYST, PENTEST_CONSULTANT, CODE_REVIEWER)
- Each role has a list of **allowed tools** it can use
- Each tool has specific **permissions** (READ, EXECUTE, DELETE)
- Before any tool is called, the system asks: **"Is this agent allowed to use this tool?"**

### Why It Matters

| Scenario | Without Authorization | With Authorization |
|----------|----------------------|-------------------|
| **SOC Agent gets prompt-injected** | Agent could call `delete_database()` → Data loss 🔴 | Even if prompt-injected, tool blocked by authorization 🟢 |
| **Pentest Agent malfunctions** | Agent could call `modify_production()` → System down 🔴 | Tool blocked by authorization layer 🟢 |
| **Code Review Agent exploited** | Agent could `merge_code()` without review → Bad code in prod 🔴 | Tool blocked by authorization 🟢 |

### Role-Based Tool Access Matrix

```python
# Permission Matrix: What each role can do with each tool
ROLE_PERMISSIONS = {
    # SOC ANALYST: See everything, do nothing destructive
    AgentRole.SOC_ANALYST: {
        "query_siem": {ToolPermission.QUERY},
        "check_vulnerability": {ToolPermission.QUERY},
        "get_threat_intel": {ToolPermission.QUERY},
    },
    
    # PENTEST CONSULTANT: Can execute tests, but within authorized scope
    AgentRole.PENTEST_CONSULTANT: {
        "nmap_scan": {ToolPermission.EXECUTE},
        "exploit_framework": {ToolPermission.EXECUTE},
    },
    
    # CODE REVIEWER: Can analyze and run tools, but can't merge/deploy
    AgentRole.CODE_REVIEWER: {
        "query_gitlab": {ToolPermission.QUERY},
        "run_sast": {ToolPermission.EXECUTE},
        "check_dependencies": {ToolPermission.QUERY},
    }
}
```

### Three Authorization Checks

1. **Is the agent role known/valid?** → Check role exists in permission matrix
2. **Is the tool in this role's allowlist?** → Check tool permitted for role
3. **Does the agent have the right permission?** → Check permission type (READ/EXECUTE/DELETE)

### Best Practices for Tool Authorization

1. **Always Use Least Privilege Principle** - Give only what's necessary
2. **Separate Read and Write Permissions** - Analyst can read but not modify
3. **Add Context-Based Rules** - Prevent dangerous operations at certain times
4. **Log All Authorization Decisions** - Track what agents try to do

---

## Part 4: Data Exfiltration Controls

**Risk:** Agent exfiltrates sensitive data (passwords, configs, user data).

**Detection Patterns:**
- `password\s*[:=]\s*['\"](.+?)['\"]`
- `api[_-]?key\s*[:=]\s*['\"]?([a-z0-9]+)['\"]?`
- `-----BEGIN PRIVATE KEY-----`
- `postgresql://.*@.*` (database URLs)

**Remediation:**
1. Check output for sensitive data patterns
2. Redact before returning to user
3. Log exfiltration attempts
4. Alert security team on critical attempts

---

## Part 5: Model Supply-Chain Verification

**Risk:** Malicious model checkpoints / compromised model weights.

**Mitigation Strategy:**
1. ✅ Model signature verification (SHA-256 checksums)
2. ✅ Only use official model sources (Hugging Face official)
3. ✅ Sandbox new models (test in isolated environment)
4. ✅ Behavioral testing (red-team before deployment)
5. ✅ Version tracking & rollback capability

**Verification Process:**
```
Load Model → Compute SHA-256 → Compare with Trusted Checksum
→ If Match: PASS (Deploy)
→ If Mismatch: FAIL (Block & Alert)
```

---

## Part C: MITRE ATLAS (Adversarial Threat Landscape for AI Systems)

**MITRE ATLAS** is a knowledge base of adversarial attacks and vulnerabilities specific to AI/ML systems. It's the AI equivalent of MITRE ATT&CK (used for cyber attacks).

#### MITRE ATLAS: Attack Lifecycle

MITRE ATLAS uses the same tactic/technique structure as ATT&CK:

| Stage | MITRE ATLAS Tactics | Examples |
|-------|-------------------|----------|
| **Reconnaissance** | Gather info about AI system | Probe model behavior, identify training data, find vulnerabilities |
| **Resource Development** | Prepare attack tools | Create jailbreaks, craft adversarial examples, build prompt templates |
| **Initial Access** | Compromise AI system | Malicious input via prompt, poisoned training data, compromised API |
| **Execution** | Run malicious code | Prompt injection, jailbreak, model extraction |
| **Persistence** | Maintain access | Backdoor in model weights, poison data source, compromise API |
| **Privilege Escalation** | Gain higher access | Exploit tool use to access protected data, escalate agent scope |
| **Defense Evasion** | Evade detection | Obfuscated prompts, timing attacks, indirect prompt injection |
| **Credential Access** | Steal credentials | Extract API keys, dump training data, get access tokens |
| **Discovery** | Learn system architecture | Map tools available, identify data sources, probe RBAC |
| **Exfiltration** | Extract data | Dump secrets, output training data, extract model weights |
| **Impact** | Damage system | Cause denial of service, corrupt model, mislead analyst |

---

## Part D: MITRE D3FEND (Defensive Techniques for AI Systems)

**MITRE D3FEND** is a knowledge base of defensive techniques to counter AI/ML attacks. While MITRE ATLAS catalogs attacks, D3FEND catalogs **defenses** — the countermeasures to stop adversarial techniques.

**Purpose:** Map each attack (MITRE ATLAS tactic/technique) to specific defensive controls you can implement.

| Purpose | Framework |
|---------|-----------|
| **Attacks** | MITRE ATLAS (T0001, T0002, etc.) |
| **Defenses** | MITRE D3FEND (DT0001, DT0002, etc.) |

**Key Insight:** D3FEND provides a structured way to ask: "For attack X, which defensive technique should I deploy?"

---

## Integrated Defense: All Three Frameworks

**How SOC Agent Uses All Three Security Frameworks:**

```
┌─────────────────────────────────────────────────────────────┐
│              Security Alert Received                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
        ┌────────────────────────┐
        │ OWASP LLM01/09/08      │
        │ ├─ Input validation    │
        │ ├─ Semantic detection  │
        │ └─ Prompt injection    │
        │     detection          │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ PLOT4AI PRINCIPLES     │
        │ ├─ Prompt Security     │
        │ ├─ Least Privilege     │
        │ ├─ Observability       │
        │ └─ Tool Governance     │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ NIST AI RMF            │
        │ ├─ Govern (role check) │
        │ ├─ Map (risk assess)   │
        │ ├─ Measure (test)      │
        │ └─ Manage (mitigate)   │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ MITRE ATLAS            │
        │ ├─ Detect attacks      │
        │ ├─ Track techniques    │
        │ ├─ Update mitigations  │
        │ └─ Red-team validation │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ SOC Agent Decision     │
        │ Output: Incident       │
        │ Severity + Recommendation
        └────────────────────────┘
```

---

## Complete OWASP LLM Top 10 Overview

The **OWASP LLM Top 10** represents the ten most critical security vulnerabilities in Large Language Model applications:

| # | Vulnerability | Risk Level | Status |
|---|---|---|---|
| **LLM01** | Prompt Injection | 🔴 CRITICAL | ✅ Fully Covered |
| **LLM02** | Insecure Output Handling | 🔴 CRITICAL | ✅ Fully Covered |
| **LLM03** | Training Data Poisoning | 🔴 CRITICAL | ✅ Fully Covered |
| **LLM04** | Insecure Plugin Integration | 🟠 HIGH | ⚠️ Partial |
| **LLM05** | Improper Error Handling | 🟠 HIGH | ⚠️ Partial |
| **LLM06** | Overreliance on LLM Output | 🟡 MEDIUM | ❌ Not Covered |
| **LLM07** | Insecure Model Update | 🟠 HIGH | ✅ Fully Covered |
| **LLM08** | Insufficient Access Controls | 🔴 CRITICAL | ✅ Fully Covered |
| **LLM09** | Improper Input Validation | 🔴 CRITICAL | ✅ Fully Covered |
| **LLM10** | Excessive Agency / Tool Use | 🔴 CRITICAL | ⚠️ Partial |

---

## Interview Questions

1. **How would you implement prompt injection defense in a production agent?**
   - Answer: Multiple layers—input validation (patterns), structural separation (message hierarchy), semantic anomaly detection (PLOT4AI), output filtering (PII masking)

2. **Design tool-use authorization for an agent with access to critical systems.**
   - Answer: Role-based permissions matrix, tool allowlist, permission validation layer before execution, audit logging

3. **How do you prevent data exfiltration in agent outputs?**
   - Answer: Pattern-based detection (regex for passwords/API keys), redaction, logging, alert on critical patterns

4. **Map your guardrails to NIST AI RMF.**
   - Answer: Govern (define policies) → Map (identify risks) → Measure (test guardrails) → Manage (respond to incidents)

5. **Red-team the platform: what attacks could compromise it?**
   - Answer: Prompt injection → tool-use bypass → data exfiltration (if auth fails); mitigation: multi-layer defense

6. **Explain the difference between SOC and pentest agents.**
   - Answer: SOC detects threats NOW (24/7 read-only), pentest finds vulnerabilities (scheduled assessments with execute permission)

7. **What is PLOT4AI and how does it differ from OWASP LLM Top 10?**
   - Answer: PLOT4AI is enterprise framework (Prompt Security, Least Privilege, Observability, Tool Governance); OWASP is vulnerability list (LLM01-10)

8. **How do you handle hallucinations in a security analysis agent?**
   - Answer: Confidence scoring, fact-checking against knowledge bases, citations required with sources, escalate low-confidence decisions to humans

---

## Enterprise Cybersecurity Tools & Techniques: Detection & Mitigation

Enterprise security uses **layered defense** (prevention → detection → response → hardening) with dedicated tools at each stage.

### Threat Detection & Tools

| Abbreviation | Expansion | Examples | Purpose & Capability |
|---|---|---|---|
| **SIEM** | Security Information and Event Management | Splunk, Sentinel, QRadar, ELK | Centralized log aggregation, correlation, alerting, compliance reporting |
| **IDS/IPS** | Intrusion Detection System / Intrusion Prevention System | Suricata, Snort, Zeek, Palo Alto | Network intrusion detection via signatures + behavioral patterns |
| **EDR** | Endpoint Detection and Response | CrowdStrike Falcon, Defender, SentinelOne | Endpoint process monitoring, file tracking, C2 detection, automated response |
| **Vulnerability Mgmt** | Vulnerability Management | Nessus, Qualys, InsightVM | Scan → CVE (Common Vulnerabilities and Exposures) check → CVSS (Common Vulnerability Scoring System) scoring → Remediation prioritization |
| **SOAR** | Security Orchestration, Automation and Response | Phantom, XSOAR, Sentinel Automation | Incident automation playbooks (alert → SIEM query → threat intel → isolate → ticket) |
| **Behavioral Analytics** | Behavioral Analytics & Anomaly Detection | UAM (User Activity Monitoring) tools, ML (Machine Learning)-based systems | User activity monitoring, anomaly detection, insider threat scoring |
| **Threat Intelligence** | Threat Intelligence & Data Feeds | Commercial feeds, OSINT (Open Source Intelligence) platforms | OSINT, commercial feeds, dark web monitoring, ISAC (Information Sharing and Analysis Center) industry data |
| **Threat Hunting** | Proactive Threat Hunting | Manual searches, IOC (Indicator of Compromise) tools | Proactive IOC searches in logs/endpoints, hypothesis-driven investigation |

### Incident Response

**IR Playbook:** Detect → Triage → Contain (isolate + revoke credentials) → Investigate → Eradicate → Recover → Post-incident review

**Forensics:** Memory dumps (Volatility), disk imaging (EnCase/FTK), log collection, chain of custody, timeline analysis

**Malware Analysis:** Static (hash, strings, imports) + Dynamic (sandbox: Any.run, Cuckoo) = Hybrid approach

### Prevention & Hardening

**Network:** Firewalls, VLANs, Zero Trust (Okta/BeyondCorp)<br>
**Endpoint:** Patch mgmt, AV/anti-malware, app whitelisting, USB control, full disk encryption<br>
**IAM:** MFA, SSO, Privileged Access Mgmt (CyberArk), RBAC, Conditional Access<br>
**Data Loss Prevention (DLP):** Pattern matching (PII/SSN), fingerprinting, watermarking, encryption at enforcement points (email, USB, cloud, print)<br>

### Common Threats & Response

| Threat | Detection | Response |
|--------|-----------|----------|
| **Ransomware** | Mass file creation, encryption, ransom note, disk I/O spikes | Isolate, restore from immutable/offline backup, patch entry vector |
| **Phishing** | Email gateway analysis, URL detonation | Remove email, force password reset, user training |
| **APT** | Behavioral anomalies, threat intel correlation | Memory/disk forensics, timeline analysis, threat group attribution |
| **Insider Threat** | UAM anomalies (unusual access, time, volume) | Revoke access, DLP investigation, legal involvement |

### Compliance Frameworks

| Framework | Expansion | Focus | Audit Requirements |
|-----------|-----------|-------|------------------|
| **SOC 2 Type II** | Service Organization Control 2 Type II | Security, availability, integrity | Audit trails, access controls, IR procedures |
| **PCI-DSS** | Payment Card Industry Data Security Standard | Payment card data | Encryption, network segmentation, pen testing |
| **HIPAA** | Health Insurance Portability and Accountability Act | Healthcare privacy | Access controls, audit logs, breach notification |
| **GDPR** | General Data Protection Regulation | Personal data (EU) | Consent, data minimization, breach reporting (72h) |
| **ISO 27001** | ISO/IEC 27001 Information Security Management System | Information security | Risk assessment, policies, training, controls |

**Audit Cadence:** Quarterly vulnerability scans, annual pen testing, continuous log review, monthly patch verification

#### Important: SOC Acronym Clarification
**⚠️ Note:** The acronym "SOC" has two different meanings in cybersecurity:
- **SOC (Security Operations Center)** - A team/department that monitors and responds to security incidents
- **SOC (Service Organization Control)** - A compliance framework for audit reports (what we discuss below)

#### SOC 2 vs SOC 2 Type II (Key Differences)

**SOC (Service Organization Control)** - Framework Level
- Overall framework created by AICPA (American Institute of Certified Public Accountants)
- Divided into **SOC 1, SOC 2, and SOC 3** based on what you're auditing:
  - **SOC 1** → Financial reporting controls
  - **SOC 2** → Security, availability, processing integrity, confidentiality, privacy
  - **SOC 3** → Public version of SOC 2 (abbreviated, for marketing)

**SOC 2** - Report Type
- Focuses on **5 Trust Service Criteria:** Security, Availability, Processing Integrity, Confidentiality, Privacy
- Comes in two sub-types: **Type I** and **Type II**

**SOC 2 Type I vs Type II** - Audit Duration & Depth

| Aspect | Type I | Type II |
|--------|--------|---------|
| **Duration** | Point-in-time snapshot (1 day - 1 week) | Over a period of time (6-12 months) |
| **What's Tested** | Design of controls (are they in place?) | Design + Operating Effectiveness (do they actually work?) |
| **Evidence** | Controls exist at audit date | Controls worked continuously during audit period |
| **Cost** | Lower ($10-30k) | Higher ($30-100k+) |
| **Validity** | Valid for ~3 months | Valid for ~1 year |
| **Use Case** | Initial compliance | Enterprise/client requirements |

**Simple Analogy:**
```
SOC 2 Type I  = "Here's a photo of your home security system"
SOC 2 Type II = "We lived in your home for 6 months and monitored your security system working every day"
```

**Enterprise Reality:** Most enterprises require **SOC 2 Type II** because it proves controls actually work over time, not just at one moment in time.

---

## Multi-Agentic Threat Defense Architecture

### System Architecture: 16 Agents Across 5 Phases

```
┌──────────────────────────────────────────────────────────────────┐
│                          INPUT                                   │
│     (User Queries, Security Logs, Code, Data)                   │
└────────────────────────┬─────────────────────────────────────────┘
                         ↓
┌───────────────────────────────────────────────────────────────────┐
│           PHASE 1: INPUT VALIDATION & THREAT PROFILING            │
│  ├─ Agent 1: Input Sanitizer  ├─ Agent 2: Prompt Injection       │
│  ├─ Agent 3: Semantic Analyzer ├─ Agent 4: Threat Profiler       │
│  └─ Agent 5: Anomaly Detector                                    │
│                    [Gate: PASS or BLOCK]                         │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────────────┐
│        PHASE 2: SECURITY SCANNING & ASSESSMENT                    │
│  ├─ Agent 6: Vulnerability Scanner  ├─ Agent 7: Code Pattern     │
│  ├─ Agent 8: Threat Intelligence    ├─ Agent 9: Risk Scorer      │
│                    [Gate: Risk Score < 7/10 or ≥ 7/10]          │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────────────┐
│     PHASE 3: AUTHORIZATION & PERMISSION VALIDATION                │
│  ├─ Agent 10: RBAC Validator   ├─ Agent 11: Tool Execution Guard │
│  └─ Agent 12: Policy Checker                                     │
│                    [Gate: PERMIT or DENY]                        │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────────────┐
│     PHASE 4: OUTPUT FILTERING & INCIDENT CLASSIFICATION           │
│  ├─ Agent 13: Output Validator  ├─ Agent 14: PII Masker          │
│  ├─ Agent 15: Incident Classifier ├─ Agent 16: Response Orchest. │
│                    [Gate: Sanitized & Classified]                │
└─────────────────────┬───────────────────────────────────────────┘
                      ↓
┌───────────────────────────────────────────────────────────────────┐
│     PHASE 5: REMEDIATION & AUDIT LOGGING                          │
│  ├─ Remediation Planner (Auto-fix for Low/Medium)                │
│  ├─ Incident Escalator (Page on-call for High/Critical)          │
│  └─ Audit Logger (ALWAYS - Records every decision)               │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────────┐
│               OUTPUT LAYER                                        │
│  ├─ Report (Incident/Vuln)  ├─ Alerts                            │
│  ├─ Remediation Plan        ├─ Compliance Audit Log              │
└──────────────────────────────────────────────────────────────────┘
```

---

### 16 Agents: Role, Input/Output & Framework

| Phase | # | Agent | Role | Input | Output | Framework |
|-------|---|-------|------|-------|--------|-----------|
| **1** | 1️⃣ | 🧹 Input Sanitizer | Removes dangerous characters | Raw input | Cleaned input | OWASP LLM09 |
| **1** | 2️⃣ | 🔍 Prompt Injection Detector | Detects prompt injection attempts | Cleaned input | Injection score (0-1) | OWASP LLM01, PLOT4AI |
| **1** | 3️⃣ | 🧠 Semantic Analyzer | Understands intent, catches indirect attacks | Query + history | Intent classification | PLOT4AI (Semantic) |
| **1** | 4️⃣ | 🎯 Threat Profiler | Maps attacks to MITRE techniques | Threat details | Technique ID + Tactic | MITRE ATLAS |
| **1** | 5️⃣ | 📊 Anomaly Detector | Spots unusual patterns vs baseline | Baseline + activity | Anomaly score (0-1) | PLOT4AI (Observable) |
| **2** | 6️⃣ | 🔬 Vulnerability Scanner | Finds CVEs, SAST, outdated libraries | Code/dependencies | CVE list + CVSS | OWASP Top 10, NIST MEASURE |
| **2** | 7️⃣ | 🚨 Code Pattern Analyzer | Detects hardcoded secrets, SQL injection, XSS | Source code | Pattern matches + severity | OWASP Top 10 |
| **2** | 8️⃣ | 🌐 Threat Intelligence Enricher | Adds context: exploits, campaigns | Raw threat | Enriched threat + IOCs | MITRE ATLAS, NIST MAP |
| **2** | 9️⃣ | 📈 Risk Scorer & Prioritizer | Calculates risk: Likelihood × Impact → 1-10 | Threat/vuln data | Risk score + priority | NIST MAP/MEASURE |
| **3** | 🔟 | 🔐 Authorization Validator | RBAC: agent/user → resource permission | Role + resource | PERMIT / DENY | PLOT4AI (Privilege), NIST GOVERN |
| **3** | 1️⃣1️⃣ | ✅ Tool Execution Guard | Validates parameters BEFORE execution | Tool + parameters | Validation result | PLOT4AI (Tool Governance) |
| **3** | 1️⃣2️⃣ | 📋 Compliance Policy Checker | Verifies action compliance | Action + policies | Compliance result | NIST GOVERN, SOC 2 |
| **4** | 1️⃣3️⃣ | 🛡️ Output Validator | Checks for dangerous code (RCE, XSS) | Agent output | Safety verdict | OWASP LLM02, PLOT4AI |
| **4** | 1️⃣4️⃣ | 🎭 PII Masker & Data Redactor | Masks passwords, keys, PII, SSN | Output text | Redacted output + log | OWASP LLM09, GDPR/HIPAA |
| **4** | 1️⃣5️⃣ | 📍 Incident Classifier | Categorizes severity | All outputs | False Pos/Low/Med/High/Crit | NIST MAP |
| **4** | 1️⃣6️⃣ | 🎛️ Response Orchestrator | Decides auto-fix vs escalate | Risk score + class | Response decision | PLOT4AI, NIST MANAGE |
| **5** | 🛠️ | 🔧 Remediation Planner | Generates patches & fixes | Vuln + risk score | Patch recommendations | NIST MANAGE, PLOT4AI |
| **5** | 🚨 | 📞 Incident Escalator | Creates tickets, pages on-call | High-risk alert | Jira + PagerDuty | NIST MANAGE |
| **5** | 📝 | 📋 Audit Logger | Records EVERY decision | All decisions | Immutable audit log | SOC 2 Type II, GDPR, HIPAA |

---

### Workflow Execution Flow

```
INPUT
  ↓
[Phase 1] Validate input (5 agents in parallel)
  ├─ If BLOCKED → Log + Alert + END
  └─ If PASS → Continue
  ↓
[Phase 2] Scan threats & assess risk (4 agents)
  ├─ If Risk < 3/10 → Low priority
  ├─ If Risk 3-6/10 → Medium priority
  └─ If Risk ≥ 7/10 → High priority
  ↓
[Phase 3] Check permissions (3 agents)
  ├─ If DENIED → Log + Return error + END
  └─ If PERMITTED → Continue
  ↓
[Phase 4] Filter output & classify (4 agents)
  ├─ Remove dangerous code
  ├─ Redact PII/secrets
  └─ Classify incident severity
  ↓
[Phase 5] Respond & audit (3 agents)
  ├─ Low/Med risk → Auto-fix + notify
  ├─ High/Crit risk → Escalate + page
  └─ ALL PATHS → Audit log (compliance)
  ↓
OUTPUT (Report + Alert + Remediation + Audit Trail)
```

---

### Key Design Principles

1. **Clear Specialization** - Each agent has ONE specific responsibility
2. **Sequential Gates** - If any agent blocks → entire flow stops and logs
3. **Framework-Aligned** - Every agent maps to PLOT4AI, OWASP, NIST, or MITRE ATLAS
4. **Immutable Audit** - Audit Logger records EVERY decision with timestamp
5. **Automated + Human** - Low-risk items auto-remediate; high-risk escalates to humans
6. **Compliance-First** - SOC 2, GDPR, HIPAA compliance built-in at every phase

### Defense-in-Depth Summary

```
Prevention → Detection → Response → Hardening
   (70%)        (20%)       (5%)        (5%)
```
- Layer 1: Firewalls, IAM, endpoint hardening, DLP
- Layer 2: SIEM, IDS/IPS, EDR, threat hunting, threat intel
- Layer 3: IR playbooks, forensics, SOAR, communication
- Layer 4: Patch mgmt, security training, compliance, lessons learned

