# Context & Prompt Engineering Mastery Guide

## Table of Contents
- [Context \& Prompt Engineering Mastery Guide](#context--prompt-engineering-mastery-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Core Principles](#1-core-principles)
    - [The 6 Cs of Effective Prompting](#the-6-cs-of-effective-prompting)
  - [2. Prompt Structure Framework](#2-prompt-structure-framework)
    - [Example: Complete Prompt Structure](#example-complete-prompt-structure)
  - [3. Core Techniques \& Frameworks](#3-core-techniques--frameworks)
    - [3.1 Chain-of-Thought (CoT)](#31-chain-of-thought-cot)
      - [Real-World CoT Examples](#real-world-cot-examples)
    - [3.2 Zero-Shot, Few-Shot, Many-Shot Learning](#32-zero-shot-few-shot-many-shot-learning)
      - [Real-World Few-Shot Examples](#real-world-few-shot-examples)
    - [3.3 ReAct (Reasoning + Acting)](#33-react-reasoning--acting)
      - [Real-World ReAct Examples](#real-world-react-examples)
    - [CoT vs ReAct: Key Differences](#cot-vs-react-key-differences)
    - [3.4 Tree of Thoughts (ToT)](#34-tree-of-thoughts-tot)
      - [Real-World ToT Examples](#real-world-tot-examples)
  - [4. Context Management Strategies](#4-context-management-strategies)
    - [4.1 Context Compression Example](#41-context-compression-example)
    - [4.2 RAG Pattern](#42-rag-pattern)
      - [Real-World RAG Examples](#real-world-rag-examples)
  - [5. Advanced Techniques](#5-advanced-techniques)
    - [5.1 Self-Consistency](#51-self-consistency)
    - [5.2 Prompt Chaining](#52-prompt-chaining)
      - [Real-World Prompt Chaining Examples](#real-world-prompt-chaining-examples)
    - [5.3 Constitutional AI / Self-Critique](#53-constitutional-ai--self-critique)
      - [Real-World Self-Critique Examples](#real-world-self-critique-examples)
  - [6. Best Practices Checklist](#6-best-practices-checklist)
    - [✓ Clarity \& Specificity](#-clarity--specificity)
    - [✓ Provide Context](#-provide-context)
    - [✓ Specify Output Format](#-specify-output-format)
    - [✓ Use Delimiters](#-use-delimiters)
    - [✓ Request Step-by-Step Reasoning](#-request-step-by-step-reasoning)
  - [7. Common Pitfalls \& Solutions](#7-common-pitfalls--solutions)
  - [8. Prompt Template Library](#8-prompt-template-library)
    - [Code Review Template](#code-review-template)
  - [Issues Found](#issues-found)
  - [Recommendations](#recommendations)
  - [Rating](#rating)
    - [Creative Writing Template](#creative-writing-template)
  - [9. Evaluation \& Testing](#9-evaluation--testing)
    - [Evaluation Example](#evaluation-example)
  - [10. Quick Reference Card](#10-quick-reference-card)
  - [11. LangChain Integration Examples](#11-langchain-integration-examples)
    - [Basic Prompt Template](#basic-prompt-template)
    - [Chat Prompt with System Message](#chat-prompt-with-system-message)
    - [Few-Shot Template](#few-shot-template)
    - [Output Parser](#output-parser)
  - [12. Final Tips](#12-final-tips)
  - [Resources \& Further Reading](#resources--further-reading)

---

## 1. Core Principles

**Overview:** Foundational concepts that underpin effective prompt engineering. Master these principles to create clear, actionable prompts that consistently produce high-quality outputs from LLMs.

```
┌─────────────────────────────────────────────────────────────┐
│                 Prompt Engineering Pyramid                  │
├─────────────────────────────────────────────────────────────┤
│                    ▲ Evaluation & Iteration                 │
│                   ╱ ╲                                        │
│                  ╱   ╲ Testing & Refinement                 │
│                 ╱─────╲                                      │
│                ╱       ╲ Structured Prompts                 │
│               ╱─────────╲                                    │
│              ╱           ╲ Context Management               │
│             ╱─────────────╲                                  │
│            ╱               ╲ Clear Instructions             │
│           ╱─────────────────╲                                │
│          ╱    Foundation     ╲                               │
│         ╱═══════════════════════╲                            │
└─────────────────────────────────────────────────────────────┘
```

### The 6 Cs of Effective Prompting
1. **Clear** - Unambiguous instructions
2. **Contextual** - Relevant background information
3. **Concise** - No unnecessary verbosity
4. **Constrained** - Well-defined boundaries
5. **Complete** - All necessary information
6. **Checkable** - Verifiable outputs

---

## 2. Prompt Structure Framework

**Overview:** A template for structuring well-formed prompts. Following this 5-part anatomy (Role → Context → Task → Format → Examples) ensures your prompts contain all necessary information for optimal LLM performance.

```
┌──────────────────────────────────────────────────────────────┐
│                    OPTIMAL PROMPT ANATOMY                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. ROLE/PERSONA                                    │    │
│  │    "You are an expert data scientist..."          │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 2. CONTEXT                                         │    │
│  │    "Given this dataset with 10K rows..."          │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 3. TASK/INSTRUCTION                                │    │
│  │    "Analyze patterns and identify anomalies"       │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 4. FORMAT/CONSTRAINTS                              │    │
│  │    "Output as JSON with keys: pattern, anomaly"   │    │
│  └────────────────────────────────────────────────────┘    │
│           ↓                                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 5. EXAMPLES (Optional)                             │    │
│  │    "Example output: {...}"                         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Example: Complete Prompt Structure

```python
prompt = """
You are a senior Python developer reviewing code for production.

Context: This function processes user payments in an e-commerce system.

Task: Review this code for security vulnerabilities and performance issues:

def process_payment(card_number, amount):
    query = f"INSERT INTO payments VALUES ('{card_number}', {amount})"
    db.execute(query)

Output Format:
- List vulnerabilities as bullet points
- Provide fixed code
- Rate severity (Low/Medium/High)

Example Output:
- SQL Injection vulnerability [High]
- Missing input validation [Medium]
"""
```

---

## 3. Core Techniques & Frameworks

**Overview:** Battle-tested prompting patterns that dramatically improve LLM performance. These techniques (CoT, Few-Shot, ReAct, ToT) are the building blocks for creating sophisticated AI applications and should be in every prompt engineer's toolkit.

### 3.1 Chain-of-Thought (CoT)

**What it is:** A prompting technique that instructs the LLM to break down complex reasoning tasks into explicit intermediate steps, showing its "thinking process" before arriving at the final answer. This improves accuracy on tasks requiring logic, math, or multi-step reasoning.

```
┌─────────────────────────────────────────────────────┐
│            Without CoT         │      With CoT      │
├────────────────────────────────┼────────────────────┤
│                                │                    │
│  Q: What is 23 * 47?           │  Q: What is        │
│                                │     23 * 47?       │
│  A: 1081                       │                    │
│     (often wrong)              │  A: Let's break    │
│                                │     this down:     │
│                                │     23 * 40 = 920  │
│                                │     23 * 7 = 161   │
│                                │     920 + 161      │
│                                │     = 1081 ✓       │
│                                │                    │
└────────────────────────────────┴────────────────────┘
```

**Implementation:**
```python
# Standard Prompt
"What is the result of 23 * 47?"

# CoT Prompt
"Let's solve 23 * 47 step by step:
Step 1: Break down the multiplication
Step 2: Calculate each part
Step 3: Sum the results
What is the final answer?"
```

#### Real-World CoT Examples

**Example 1: Medical Diagnosis Assistant**
```python
"""
Analyze the following patient symptoms and provide a diagnosis.

Patient Information:
- Age: 45, Female
- Symptoms: Persistent cough for 3 weeks, fever (101°F), fatigue, chest pain
- Medical History: Non-smoker, no chronic conditions
- Recent Travel: None

Think through this step-by-step:

Step 1: Identify primary symptoms and their duration
Step 2: Consider common conditions that match these symptoms
Step 3: Evaluate severity and urgency indicators
Step 4: Rule out less likely conditions
Step 5: Provide top 3 differential diagnoses with reasoning
Step 6: Recommend next steps (tests, specialist referral)

Analysis:
"""
```

**Example 2: Financial Investment Analysis**
```python
"""
Should I invest $10,000 in Company XYZ stock?

Company Data:
- Current Price: $150/share
- P/E Ratio: 25
- Revenue Growth: 15% YoY
- Debt-to-Equity: 0.5
- Industry: Technology
- Market Cap: $50B

Analyze step-by-step:

Step 1: Evaluate current valuation (is P/E ratio reasonable for the industry?)
Step 2: Assess growth potential (revenue trends, market position)
Step 3: Analyze financial health (debt levels, profitability)
Step 4: Consider market conditions and sector trends
Step 5: Compare with industry peers
Step 6: Identify risks and opportunities
Step 7: Provide recommendation with reasoning

Investment Analysis:
"""
```

### 3.2 Zero-Shot, Few-Shot, Many-Shot Learning

**What it is:** Techniques that control how many examples you provide to guide the model's behavior. Zero-shot uses no examples (relies on instructions only), few-shot provides 1-5 examples to establish a pattern, and many-shot uses 10+ examples for complex pattern learning.

```
┌──────────────────────────────────────────────────────────┐
│               Learning Type Comparison                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ZERO-SHOT (No examples)                                │
│  ════════════════════════                                │
│  Prompt: "Classify sentiment: I love this product!"     │
│  → Relies purely on pre-training                         │
│                                                          │
│  ────────────────────────────────────────────────────   │
│                                                          │
│  FEW-SHOT (1-5 examples)                                │
│  ════════════════════════                                │
│  Examples:                                               │
│    "Great!" → Positive                                   │
│    "Terrible" → Negative                                 │
│  Query: "I love this product!" → ?                       │
│  → Learns pattern from examples                          │
│                                                          │
│  ────────────────────────────────────────────────────── │
│                                                          │
│  MANY-SHOT (10+ examples)                               │
│  ════════════════════════                                │
│  [10+ labeled examples]                                  │
│  Query: "I love this product!" → ?                       │
│  → Better pattern recognition                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Example:**
```python
# Few-Shot Prompt
prompt = """
Classify the sentiment of the following reviews:

Review: "This phone is amazing! Best purchase ever."
Sentiment: Positive

Review: "Worst experience. Broke after 2 days."
Sentiment: Negative

Review: "It's okay, nothing special."
Sentiment: Neutral

Review: "Absolutely love the camera quality and battery life!"
Sentiment: ?
"""
```

#### Real-World Few-Shot Examples

**Example 1: SQL Query Generator**
```python
"""
Generate SQL queries based on natural language questions.

Examples:

Question: "Show me all customers who made purchases in January"
SQL: SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id WHERE MONTH(o.order_date) = 1;

Question: "What's the total revenue by product category?"
SQL: SELECT category, SUM(revenue) as total_revenue FROM products GROUP BY category ORDER BY total_revenue DESC;

Question: "Find users who haven't logged in for 30 days"
SQL: SELECT * FROM users WHERE last_login < DATE_SUB(NOW(), INTERVAL 30 DAY);

Now generate SQL for this question:
Question: "List the top 5 customers by total spending"
SQL:
"""
```

**Example 2: Email Classification (Spam Detection)**
```python
"""
Classify emails as 'spam' or 'legitimate'.

Examples:

Email: "Congratulations! You've won $1,000,000! Click here NOW!!!"
Classification: spam

Email: "Hi John, can we reschedule tomorrow's meeting to 3pm? Thanks, Sarah"
Classification: legitimate

Email: "URGENT: Your account will be suspended unless you verify your credentials immediately"
Classification: spam

Email: "Team update: Q4 results are in. Please review the attached report before Friday's meeting."
Classification: legitimate

Email: "Get rich quick! Invest in cryptocurrency now! 500% returns guaranteed!"
Classification: spam

Now classify this email:
Email: "Your Amazon order #12345 has been shipped and will arrive by Thursday"
Classification:
"""
```

### 3.3 ReAct (Reasoning + Acting)

**What it is:** A framework that combines reasoning (thinking) with actions (using tools/APIs) in an interleaved loop. The model alternates between thinking about what to do next, taking an action, observing the result, and repeating until the task is complete. Essential for building agents that interact with external systems.

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct Framework                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│     Thought ──→ Action ──→ Observation                 │
│        ↑                        │                       │
│        │                        │                       │
│        └────────────────────────┘                       │
│              (Loop until done)                          │
│                                                         │
│  Example Flow:                                          │
│  ─────────────                                          │
│  Thought: "I need to find current weather"             │
│      ↓                                                  │
│  Action: search("New York weather today")              │
│      ↓                                                  │
│  Observation: "72°F, partly cloudy"                    │
│      ↓                                                  │
│  Thought: "I have the answer"                          │
│      ↓                                                  │
│  Action: finish("Weather is 72°F, partly cloudy")      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
prompt = """
Answer the question by using the following tools:
- search(query): Search the web
- calculate(expression): Perform calculations
- finish(answer): Provide final answer

Use this format:
Thought: [your reasoning]
Action: [tool and input]
Observation: [tool output]
... (repeat as needed)
Thought: I have the final answer
Action: finish([final answer])

Question: What is the population of the capital of France multiplied by 2?

Thought: I need to find the capital of France first
Action: search("capital of France")
Observation: Paris is the capital of France
Thought: Now I need Paris population
Action: search("Paris population 2026")
Observation: Paris population is approximately 2.2 million
Thought: Now I can calculate the result
Action: calculate(2.2 * 2)
Observation: 4.4
Thought: I have the final answer
Action: finish("4.4 million")
"""
```

#### Real-World ReAct Examples

**Example 1: Customer Service Agent with Tools**
```python
"""
You are a customer service agent with access to the following tools:
- check_order(order_id): Get order status
- check_inventory(product_id): Check product availability
- process_refund(order_id): Initiate refund
- send_email(recipient, message): Send email to customer
- finish(response): Provide final response to customer

Use the ReAct format to help customers.

Customer: "I ordered a laptop 2 weeks ago (Order #45678) but haven't received it yet. What's going on?"

Thought: I need to check the order status first to see what happened
Action: check_order("45678")
Observation: Order #45678 - Status: Shipped on Dec 1, Expected delivery: Dec 10, Carrier: FedEx, Tracking: FDX123456789

Thought: The order was shipped but delivery is delayed. I should check if there are any tracking updates
Action: search("FedEx tracking FDX123456789")
Observation: Package is delayed due to weather conditions, new estimated delivery: Dec 15

Thought: I have enough information to respond to the customer with tracking details and send them an email
Action: send_email("customer@email.com", "Your order #45678 is delayed due to weather. New delivery: Dec 15. Tracking: FDX123456789")
Observation: Email sent successfully

Thought: Now I can provide the final response
Action: finish("I apologize for the delay. Your laptop was shipped on Dec 1st but is experiencing delays due to weather. The new estimated delivery is Dec 15. I've sent you an email with tracking details (FDX123456789). Would you like me to expedite shipping or process a refund?")
"""
```

**Example 2: Research Assistant**
```python
"""
You are a research assistant with access to:
- search(query): Search the web
- read_paper(url): Read academic paper
- calculate(expression): Perform calculations
- create_summary(content): Summarize content
- finish(answer): Provide final answer

Question: "What's the latest research on transformer models' efficiency improvements in 2025?"

Thought: I need to search for recent papers on transformer efficiency
Action: search("transformer model efficiency improvements 2025")
Observation: Found papers: 1) "Flash Attention 3", 2) "Sparse Transformers", 3) "MoE Optimization"

Thought: Let me read the most relevant paper on Flash Attention 3
Action: read_paper("arxiv.org/flash-attention-3-2025")
Observation: Paper describes 5x speedup through kernel fusion and memory optimization. Key innovation: fused multi-head attention with reduced memory footprint.

Thought: I should get details on Sparse Transformers too
Action: read_paper("arxiv.org/sparse-transformers-2025")
Observation: Introduces learnable sparsity patterns reducing compute by 70% while maintaining 95% accuracy.

Thought: Now I can create a comprehensive summary
Action: create_summary("Flash Attention 3: 5x speedup via kernel fusion. Sparse Transformers: 70% compute reduction with learnable sparsity.")
Observation: Summary created with key findings

Thought: I have enough information to answer
Action: finish("Latest transformer efficiency improvements in 2025 include: 1) Flash Attention 3 achieving 5x speedup through kernel fusion, 2) Sparse Transformers reducing compute by 70% using learnable sparsity patterns while maintaining 95% accuracy, 3) Mixture-of-Experts optimizations for better scaling.")
"""
```

---

### CoT vs ReAct: Key Differences

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CoT vs ReAct Comparison                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Chain-of-Thought (CoT)          │  ReAct (Reasoning + Acting)     │
│  ═══════════════════════          │  ══════════════════════════     │
│                                   │                                 │
│  Purpose:                         │  Purpose:                       │
│  Internal reasoning only          │  Reasoning + External actions   │
│                                   │                                 │
│  Process:                         │  Process:                       │
│  Question → Think → Answer        │  Question → Think → Act →       │
│                                   │  Observe → Think → ... → Answer │
│                                   │                                 │
│  Tools/APIs:                      │  Tools/APIs:                    │
│  ❌ No external tool access       │  ✅ Can call APIs, search,      │
│                                   │     databases, calculators      │
│                                   │                                 │
│  Best For:                        │  Best For:                      │
│  • Math problems                  │  • Multi-step tasks with tools  │
│  • Logic puzzles                  │  • Information retrieval        │
│  • Analysis & reasoning           │  • Agent workflows              │
│  • Explaining decisions           │  • Dynamic problem solving      │
│                                   │                                 │
│  Example:                         │  Example:                       │
│  "Solve 25 * 48                   │  "Book a flight to NYC"         │
│   Step 1: 25 * 40 = 1000          │  Thought: Need to search        │
│   Step 2: 25 * 8 = 200            │  Action: search_flights()       │
│   Step 3: 1000 + 200 = 1200"      │  Observation: [flight options]  │
│                                   │  Thought: User wants cheapest   │
│                                   │  Action: book_flight(id=123)    │
│                                   │  Observation: Booking confirmed │
│                                   │                                 │
│  Limitations:                     │  Limitations:                   │
│  • Cannot access real-time data   │  • More complex to implement    │
│  • Cannot interact with systems   │  • Requires tool integration    │
│  • Static knowledge only          │  • May need error handling      │
│                                   │                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**When to Use CoT vs ReAct:**

| Scenario | Use CoT | Use ReAct |
|----------|---------|-----------|
| Solve a math word problem | ✅ | ❌ |
| Explain a complex concept | ✅ | ❌ |
| Search web + summarize findings | ❌ | ✅ |
| Check database + send email | ❌ | ✅ |
| Analyze data from external API | ❌ | ✅ |
| Compare pros/cons of options | ✅ | ❌ |
| Book appointment (check calendar, reserve slot) | ❌ | ✅ |
| Troubleshoot issue (check logs, run diagnostics) | ❌ | ✅ |

**Key Insight:** CoT is about *showing your work* when thinking. ReAct is about *doing things* while thinking. You can even combine them - use CoT reasoning within each ReAct thought step for maximum effectiveness!

**Combined Example (CoT + ReAct):**
```python
"""
Question: "What's the average temperature in cities with population > 1M in California?"

Thought: I need to find California cities with 1M+ population, get their temperatures,
         and calculate average. Let me break this down:
         Step 1: Get list of CA cities
         Step 2: Filter by population
         Step 3: Get temperature for each
         Step 4: Calculate average
Action: search("California cities population over 1 million")
Observation: Los Angeles (3.9M), San Diego (1.4M), San Jose (1.0M)

Thought: Now I have 3 cities. Let me get current temperature for each.
Action: get_weather("Los Angeles, CA")
Observation: 72°F

Thought: Got LA temperature. Continue with San Diego.
Action: get_weather("San Diego, CA")
Observation: 68°F

Thought: Got SD temperature. Finally, San Jose.
Action: get_weather("San Jose, CA")
Observation: 65°F

Thought: Now I can calculate the average:
         Step 1: Sum = 72 + 68 + 65 = 205
         Step 2: Count = 3 cities
         Step 3: Average = 205 / 3 = 68.33°F
Action: finish("The average temperature in California cities with population over 1M is 68.33°F (based on Los Angeles: 72°F, San Diego: 68°F, San Jose: 65°F)")
"""
```

---

### 3.4 Tree of Thoughts (ToT)

**What it is:** An advanced technique that explores multiple reasoning paths simultaneously (like a decision tree), evaluates each path, prunes weak ones, and selects the best solution. Useful for complex problems with multiple valid approaches where you need to compare trade-offs before deciding.

```
┌─────────────────────────────────────────────────────────┐
│                  Tree of Thoughts                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    Problem                              │
│                      │                                  │
│          ┌───────────┼───────────┐                     │
│          │           │           │                      │
│      Thought1    Thought2    Thought3                  │
│       ├──┤        ├──┤        ├──┤                     │
│      T1.1 T1.2  T2.1 T2.2  T3.1 T3.2                   │
│       │    ✓     │    ✗     ✓    │                     │
│       │          │               │                      │
│    Evaluate & Prune Best Paths                         │
│       │          │               │                      │
│       └──────────┴───────────────┘                     │
│                  │                                      │
│            Final Solution                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Example:**
```python
prompt = """
Solve this problem by exploring multiple reasoning paths:

Problem: Plan a 3-day trip to Japan with a $2000 budget.

Generate 3 different approaches:

Approach 1: Budget-focused
- Stay in hostels: $30/night
- Street food: $20/day
- Public transport: $10/day
Total: ~$270
Evaluation: ★★★★ (Affordable, authentic)

Approach 2: Comfort-focused
- Mid-range hotel: $100/night
- Restaurant meals: $50/day
- JR Pass: $280
Total: ~$730
Evaluation: ★★★★★ (Balanced)

Approach 3: Luxury-focused
- Luxury hotel: $300/night
- Fine dining: $150/day
- Private tours: $200/day
Total: ~$1950
Evaluation: ★★★ (Exceeds budget slightly)

Best Solution: Approach 2 (Comfort-focused)
"""
```

#### Real-World ToT Examples

**Example 1: System Architecture Design**
```python
"""
Design a scalable backend architecture for a real-time chat application with 1M users.

Explore multiple architectural approaches:

Approach 1: Monolithic Architecture
Components:
  - Single Node.js server with Socket.io
  - MongoDB for messages
  - Redis for caching
Pros: Simple deployment, easy development
Cons: Single point of failure, hard to scale
Scalability Score: ★★☆☆☆ (Poor - Won't handle 1M users)
Cost Estimate: $500/month
Verdict: ❌ Not suitable for requirements

Approach 2: Microservices with WebSockets
Components:
  - API Gateway (Kong)
  - Message Service (Node.js + Socket.io cluster)
  - User Service (Python FastAPI)
  - PostgreSQL (primary) + Cassandra (messages)
  - Redis Cluster (pub/sub)
  - Kubernetes for orchestration
Pros: Highly scalable, fault tolerant, independent services
Cons: Complex infrastructure, higher latency
Scalability Score: ★★★★★ (Excellent)
Cost Estimate: $3000/month
Verdict: ✅ Best for 1M+ users

Approach 3: Serverless Architecture
Components:
  - AWS API Gateway + Lambda
  - DynamoDB for data
  - AppSync for real-time (GraphQL subscriptions)
  - CloudFront CDN
Pros: Auto-scaling, pay-per-use, low maintenance
Cons: Cold starts, vendor lock-in, complex debugging
Scalability Score: ★★★★☆ (Very Good)
Cost Estimate: $2000/month (variable)
Verdict: ✅ Good alternative, better cost at lower usage

Final Recommendation: Approach 2 (Microservices)
Reasoning: For 1M concurrent users, microservices provide best scalability and control.
Start with core services, add monitoring (Prometheus/Grafana), use CDN for static content.
"""
```

**Example 2: Marketing Campaign Strategy**
```python
"""
Develop a marketing strategy to increase user signups by 50% in Q1.

Current State:
- Monthly signups: 10,000
- Budget: $50,000
- Target: 15,000/month (50% increase)
- Product: B2B SaaS tool

Strategy 1: Content Marketing + SEO
Tactics:
  - 20 blog posts/month targeting long-tail keywords
  - Guest posts on industry sites
  - YouTube tutorial series
  - Free tools/calculators
Investment: $15,000 (writers, SEO tools)
Expected Signups: +2,000/month
Timeline: 3-4 months to see results
ROI: Medium (long-term value)
Risk: ★★☆☆☆ (Low risk)
Evaluation: ★★★☆☆ - Slow but sustainable

Strategy 2: Paid Advertising (Google Ads + LinkedIn)
Tactics:
  - Google Search ads for high-intent keywords
  - LinkedIn sponsored content
  - Retargeting campaigns
  - A/B test landing pages
Investment: $40,000 (ad spend)
Expected Signups: +6,000/month
Timeline: Immediate results
ROI: High (if optimized)
Risk: ★★★★☆ (High - depends on CPA)
Evaluation: ★★★★☆ - Fast but expensive

Strategy 3: Hybrid Approach
Tactics:
  - $25,000 paid ads (Google + LinkedIn)
  - $15,000 content marketing
  - $10,000 referral program (incentivize users)
Investment: $50,000 total
Expected Signups: +5,000/month
Timeline: 1-2 months
ROI: High (balanced)
Risk: ★★★☆☆ (Medium)
Evaluation: ★★★★★ - Best balance

Recommended Strategy: Strategy 3 (Hybrid)
Rationale: Combines immediate results from paid ads with long-term value from content.
Referral program reduces acquisition cost over time. Diversified approach reduces risk.

Implementation Plan:
Week 1-2: Set up campaigns, create landing pages
Week 3-8: Run ads, publish content, launch referral program
Week 9-12: Optimize based on data, scale winners
"""
```

---

## 4. Context Management Strategies

**What it is:** Techniques for efficiently managing the limited context window (token limit) of LLMs. This includes strategies like summarization, sliding windows, and RAG (Retrieval Augmented Generation) to provide relevant information without exceeding token limits or degrading performance.

```
┌────────────────────────────────────────────────────────────┐
│              Context Window Management                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────┐        │
│  │ System Prompt (Role & Guidelines)            │        │
│  │ ┌──────────────────────────────────────────┐ │        │
│  │ │ Few-Shot Examples                        │ │        │
│  │ │ ┌──────────────────────────────────────┐ │ │        │
│  │ │ │ Retrieved Context (RAG)              │ │ │        │
│  │ │ │ ┌──────────────────────────────────┐ │ │ │        │
│  │ │ │ │ Conversation History             │ │ │ │        │
│  │ │ │ │ ┌──────────────────────────────┐ │ │ │ │        │
│  │ │ │ │ │ Current Query                │ │ │ │ │        │
│  │ │ │ │ └──────────────────────────────┘ │ │ │ │        │
│  │ │ │ └──────────────────────────────────┘ │ │ │        │
│  │ │ └──────────────────────────────────────┘ │ │        │
│  │ └──────────────────────────────────────────┘ │        │
│  └──────────────────────────────────────────────┘        │
│                                                            │
│  Strategies:                                               │
│  • Summarization: Compress old conversations              │
│  • Sliding Window: Keep recent N messages                 │
│  • RAG: Retrieve relevant context only                    │
│  • Hierarchical: Summarize → Details → Current            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.1 Context Compression Example

```python
# Bad: Sending entire conversation
context = all_messages  # 50,000 tokens

# Good: Summarize + Recent
context = f"""
Previous conversation summary: User is building a chatbot for customer service.
They use Python and LangChain. Previous issues: API rate limits.

Recent messages:
User: How do I implement streaming responses?
Assistant: You can use...
User: Show me code example
"""  # 200 tokens
```

### 4.2 RAG Pattern

```
┌─────────────────────────────────────────────────────────┐
│              RAG (Retrieval Augmented Generation)       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User Query                                             │
│       │                                                 │
│       ↓                                                 │
│  Embed Query ──→ Search Vector DB                       │
│                       │                                 │
│                       ↓                                 │
│                 Retrieve Top-K                          │
│                 Relevant Docs                           │
│                       │                                 │
│                       ↓                                 │
│         ┌─────────────────────────┐                     │
│         │  Prompt = Query +       │                     │
│         │  Retrieved Context      │                     │
│         └─────────────────────────┘                     │
│                       │                                 │
│                       ↓                                 │
│                   LLM Generate                          │
│                       │                                 │
│                       ↓                                 │
│                   Response                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Example:**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Retrieve relevant context
query = "How do I handle API rate limits?"
relevant_docs = vectorstore.similarity_search(query, k=3)

# Build prompt with context
prompt = f"""
Context from documentation:
{relevant_docs[0].page_content}
{relevant_docs[1].page_content}
{relevant_docs[2].page_content}

Question: {query}

Answer based on the context above:
"""
```

#### Real-World RAG Examples

**Example 1: Legal Document Q&A**
```python
"""
You are a legal research assistant. Answer questions based on retrieved case law.

Retrieved Documents:
---
Document 1 (Case: Smith v. Jones, 2020):
"The court held that employers must provide reasonable accommodations for
employees with disabilities under ADA Section 12112(b)(5)(A). Failure to
engage in interactive process constitutes discrimination."

Document 2 (Case: Brown v. Tech Corp, 2022):
"Reasonable accommodation includes modified work schedules, ergonomic equipment,
and remote work options. Employers must document the interactive process."

Document 3 (ADA Guidelines):
"Interactive process requires good faith dialogue between employer and employee
to identify effective accommodations that don't cause undue hardship."
---

Question: What are an employer's obligations when an employee requests
accommodation for a disability?

Instructions:
- Cite specific cases and statutes
- Provide actionable guidance
- Note any exceptions or limitations

Answer:
"""
```

**Example 2: Technical Support with Product Documentation**
```python
"""
Answer the customer's question using the product documentation below.

Retrieved Documentation:
---
Section 1: Installation
"CloudSync requires Python 3.8+ and 2GB RAM. Install via: pip install cloudsync"

Section 2: Authentication
"Use API keys for authentication. Generate keys in dashboard under Settings > API.
Set environment variable: export CLOUDSYNC_API_KEY='your-key'"

Section 3: Error Handling
"Error CS-401: Invalid API key. Verify key is active and has correct permissions.
Error CS-429: Rate limit exceeded. Default limit: 100 requests/minute."
---

Customer Question: "I'm getting error CS-401 when trying to sync files.
I just installed CloudSync yesterday."

Response Format:
1. Explain the error
2. Provide step-by-step solution
3. Include relevant commands/code
4. Suggest prevention tips

Answer:
"""
```

---

## 5. Advanced Techniques

**Overview:** Next-level strategies that combine multiple techniques or add quality-control layers. Use these when basic prompting isn't enough and you need maximum reliability, quality, or workflow automation.

### 5.1 Self-Consistency

**What it is:** Generate multiple independent responses to the same prompt (usually with CoT), then select the most consistent or majority answer. This reduces errors and improves reliability for reasoning tasks by leveraging the wisdom of multiple reasoning paths.

```
┌──────────────────────────────────────────────────────┐
│             Self-Consistency Approach                │
├──────────────────────────────────────────────────────┤
│                                                      │
│              Same Question                           │
│                    │                                 │
│         ┌──────────┼──────────┐                     │
│         ↓          ↓          ↓                      │
│     Sample 1   Sample 2   Sample 3                  │
│     (CoT)      (CoT)      (CoT)                     │
│         │          │          │                      │
│      Answer A  Answer B  Answer A                   │
│         │          │          │                      │
│         └──────────┼──────────┘                     │
│                    ↓                                 │
│            Majority Vote                             │
│                    │                                 │
│                    ↓                                 │
│              Final Answer: A                         │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 5.2 Prompt Chaining

**What it is:** Breaking down complex tasks into a sequence of simpler prompts where each prompt's output feeds into the next. This improves accuracy, maintainability, and allows specialization at each step (e.g., extract → analyze → summarize → format).

```
┌────────────────────────────────────────────────────┐
│              Prompt Chaining Pipeline              │
├────────────────────────────────────────────────────┤
│                                                    │
│  Input: "Analyze customer feedback"               │
│    │                                               │
│    ↓                                               │
│  ┌──────────────────────────────┐                │
│  │ Prompt 1: Extract Topics    │                 │
│  └──────────────────────────────┘                │
│    │                                               │
│    ↓ Output: ["pricing", "support", "features"]  │
│    │                                               │
│  ┌──────────────────────────────┐                │
│  │ Prompt 2: Classify Sentiment │                 │
│  └──────────────────────────────┘                │
│    │                                               │
│    ↓ Output: {"pricing": "negative", ...}        │
│    │                                               │
│  ┌──────────────────────────────┐                │
│  │ Prompt 3: Generate Summary   │                 │
│  └──────────────────────────────┘                │
│    │                                               │
│    ↓ Final Report                                 │
│                                                    │
└────────────────────────────────────────────────────┘
```

**Implementation:**
```python
from langchain.chains import LLMChain, SequentialChain

# Chain 1: Extract entities
extract_chain = LLMChain(
    llm=llm,
    prompt="Extract all product names from: {text}"
)

# Chain 2: Analyze sentiment
sentiment_chain = LLMChain(
    llm=llm,
    prompt="Analyze sentiment for these products: {products}"
)

# Chain 3: Generate report
report_chain = LLMChain(
    llm=llm,
    prompt="Create report from: {sentiment_data}"
)

# Combine
full_chain = SequentialChain(
    chains=[extract_chain, sentiment_chain, report_chain]
)
```

#### Real-World Prompt Chaining Examples

**Example 1: Content Creation Pipeline**
```python
"""
CHAIN 1: Topic Ideation
---
Task: Generate 5 blog topic ideas for a B2B SaaS company selling project management tools.
Target audience: Engineering managers
Focus: Productivity, team collaboration
Output format: JSON array of objects with 'title' and 'angle' keys
---

CHAIN 2: Outline Creation
---
Input: {selected_topic_from_chain_1}
Task: Create detailed blog post outline
Requirements:
- 5-7 main sections
- Include key points for each section
- Suggest data/statistics to include
- Recommend CTAs
Output format: Structured outline with H2, H3 headings
---

CHAIN 3: Content Writing
---
Input: {outline_from_chain_2}
Task: Write complete blog post (1500 words)
Style: Professional but conversational
Tone: Authoritative, helpful
SEO: Include keywords naturally: "project management", "team productivity"
---

CHAIN 4: SEO Optimization
---
Input: {draft_from_chain_3}
Task: Optimize for SEO
- Generate meta title (60 chars)
- Generate meta description (160 chars)
- Suggest 5 internal linking opportunities
- Add FAQ schema markup
---
"""
```

**Example 2: Data Analysis Workflow**
```python
"""
CHAIN 1: Data Cleaning
---
Input: Raw CSV data with customer transactions
Task:
1. Identify missing values
2. Detect outliers
3. Suggest data type corrections
4. Recommend filtering criteria
Output: Data quality report + cleaning recommendations
---

CHAIN 2: Exploratory Analysis
---
Input: {cleaned_data_from_chain_1}
Task:
1. Calculate summary statistics
2. Identify trends and patterns
3. Find correlations between variables
4. Segment customers by behavior
Output: Key insights with supporting statistics
---

CHAIN 3: Hypothesis Generation
---
Input: {insights_from_chain_2}
Task:
1. Generate 3 testable hypotheses
2. Suggest metrics to track
3. Recommend A/B test scenarios
Output: Hypothesis document with testing plan
---

CHAIN 4: Visualization Recommendations
---
Input: {hypotheses_from_chain_3}
Task:
1. Recommend chart types for each insight
2. Suggest dashboard layout
3. Provide Python code (matplotlib/seaborn) for key visualizations
Output: Visualization spec + code snippets
---
"""
```

### 5.3 Constitutional AI / Self-Critique

**What it is:** A two-step process where the model first generates a response, then critiques it against specified principles (accuracy, safety, bias, completeness), and finally revises the response if issues are found. Improves output quality and alignment with guidelines.

```
┌─────────────────────────────────────────────────┐
│          Self-Critique Pattern                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Initial Prompt                                 │
│       │                                         │
│       ↓                                         │
│  Generate Response                              │
│       │                                         │
│       ↓                                         │
│  Critique Against Principles:                  │
│    • Is it harmful?                            │
│    • Is it biased?                             │
│    • Is it accurate?                           │
│       │                                         │
│       ↓                                         │
│  If issues found ──→ Revise                    │
│       │                                         │
│       ↓                                         │
│  Final Response                                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Example:**
```python
critique_prompt = """
Review this response for potential issues:

Original Response: {response}

Check for:
1. Factual accuracy
2. Bias or unfairness
3. Harmful content
4. Completeness

If issues exist, provide a revised response.
"""
```

#### Real-World Self-Critique Examples

**Example 1: Code Review with Self-Critique**
```python
"""
STEP 1: Generate Code Solution
---
Task: Write a Python function to validate email addresses

Generated Code:
def validate_email(email):
    return '@' in email and '.' in email

---

STEP 2: Self-Critique
---
Review the code above against these criteria:

1. Correctness: Does it handle all valid email formats?
   Issue: ❌ Too simplistic. Accepts invalid emails like "@.", "a@b@c.com"

2. Security: Any security vulnerabilities?
   Issue: ❌ No input sanitization, could be used in injection attacks

3. Edge Cases: Does it handle edge cases?
   Issue: ❌ Doesn't check for None, empty string, multiple @, etc.

4. Performance: Is it efficient?
   Status: ✅ O(n) is acceptable for this use case

5. Best Practices: Does it follow Python conventions?
   Issue: ❌ No type hints, no docstring, no error handling

---

STEP 3: Revised Solution
---
import re
from typing import Optional

def validate_email(email: Optional[str]) -> bool:
    """
    Validate email address using RFC 5322 regex pattern.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_email("user@example.com")
        True
        >>> validate_email("invalid.email")
        False
    """
    if not email or not isinstance(email, str):
        return False

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
---
"""
```

**Example 2: Content Moderation with Self-Critique**
```python
"""
STEP 1: Generate Response
---
User Question: "What's the best way to deal with difficult coworkers?"

Initial Response:
"Just ignore them or report them to HR. Some people are impossible to work with
and you shouldn't waste your time trying to fix the relationship."

---

STEP 2: Critique Against Principles
---
Principle 1: Helpfulness
Issue: ❌ Response is overly simplistic and doesn't provide constructive advice

Principle 2: Harmlessness
Issue: ⚠️ Could encourage avoidance rather than conflict resolution
Could damage workplace relationships if applied broadly

Principle 3: Accuracy
Issue: ⚠️ Oversimplifies complex interpersonal situations
HR involvement isn't always appropriate first step

Principle 4: Bias
Issue: ⚠️ Assumes coworker is at fault without considering context
Doesn't account for cultural or communication style differences

Principle 5: Completeness
Issue: ❌ Missing specific strategies, examples, and nuance

---

STEP 3: Revised Response
---
"Dealing with difficult coworkers requires a thoughtful approach:

1. Self-Reflection First
   - Consider if communication styles differ
   - Identify specific behaviors (not personality) that are problematic

2. Direct Communication
   - Have a private, respectful conversation
   - Use 'I' statements: 'I feel concerned when...'
   - Listen to their perspective

3. Find Common Ground
   - Focus on shared goals
   - Look for win-win solutions
   - Be willing to compromise

4. Set Boundaries
   - Be clear about what behaviors are acceptable
   - Document interactions if issues persist

5. Escalate Appropriately
   - Try direct resolution first
   - Involve manager if behavior affects work
   - HR is for serious issues (harassment, discrimination)

Remember: Most workplace conflicts stem from miscommunication.
Approach with curiosity rather than judgment."

---
"""
```

---

## 6. Best Practices Checklist

### ✓ Clarity & Specificity
```python
# ✗ Bad
"Tell me about Python"

# ✓ Good
"Explain Python's GIL (Global Interpreter Lock) and its impact on
multi-threaded performance with a code example showing the difference
between threading and multiprocessing."
```

### ✓ Provide Context
```python
# ✗ Bad
"Fix this bug"

# ✓ Good
"""
I'm working on a Flask API (Python 3.11) that processes image uploads.
Users report 500 errors when uploading files > 5MB.

Error: "Connection reset by peer"

Code: [paste code]

Question: How do I fix this timeout issue?
"""
```

### ✓ Specify Output Format
```python
# ✗ Bad
"Analyze this data"

# ✓ Good
"""
Analyze this sales data and output JSON:
{
  "total_revenue": float,
  "top_products": [{"name": str, "revenue": float}],
  "insights": [str]
}
"""
```

### ✓ Use Delimiters
```python
prompt = """
Summarize the text between ### markers.

### Text ###
{user_input}
### End ###

Summary:
"""
```

### ✓ Request Step-by-Step Reasoning
```python
# ✗ Bad
"What's the answer?"

# ✓ Good
"""
Solve this step by step:
1. Identify the problem type
2. List known variables
3. Apply relevant formulas
4. Calculate result
5. Verify answer
"""
```

---

## 7. Common Pitfalls & Solutions

```
┌────────────────────────────────────────────────────────────┐
│                    Pitfall → Solution                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│ ✗ Vague instructions                                      │
│   → Use specific, measurable criteria                     │
│                                                            │
│ ✗ Too much context (hits token limit)                    │
│   → Summarize, use RAG, or chunk into multiple prompts   │
│                                                            │
│ ✗ No examples for complex tasks                          │
│   → Provide 2-3 few-shot examples                        │
│                                                            │
│ ✗ Asking multiple questions at once                      │
│   → Break into separate prompts or numbered list         │
│                                                            │
│ ✗ No output format specified                             │
│   → Use JSON schema, markdown template, or examples      │
│                                                            │
│ ✗ Ignoring model limitations                             │
│   → Understand context window, use appropriate model     │
│                                                            │
│ ✗ Not iterating on prompts                               │
│   → Test, measure, refine based on outputs               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 8. Prompt Template Library

### Code Review Template
```python
CODE_REVIEW = """
Role: You are a senior {language} developer with expertise in {domain}.

Task: Review the following code for:
- Security vulnerabilities
- Performance issues
- Code quality and maintainability
- Best practices adherence

Code:
```{language}
{code}
```

Output Format:
## Issues Found
- [Severity] Issue description
  - Location: line X
  - Fix: suggested solution

## Recommendations
- Improvement suggestions

## Rating
Overall code quality: [1-10]
"""
```

### Data Analysis Template
```python
DATA_ANALYSIS = """
You are a data analyst. Analyze the dataset below and provide insights.

Dataset:
{data}

Analysis Requirements:
1. Identify key trends
2. Detect anomalies
3. Provide 3 actionable recommendations

Output as JSON:
{
  "trends": [{"trend": str, "confidence": float}],
  "anomalies": [{"description": str, "severity": str}],
  "recommendations": [str]
}
"""
```

### Creative Writing Template
```python
CREATIVE_WRITING = """
Write a {format} about {topic} with the following specifications:

Tone: {tone}
Audience: {audience}
Length: {length} words
Style: {style}

Requirements:
- Include {requirement_1}
- Avoid {avoid_1}
- Focus on {focus_area}

Begin:
"""
```

---

## 9. Evaluation & Testing

```
┌─────────────────────────────────────────────────────┐
│            Prompt Testing Framework                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Define Success Metrics                         │
│     • Accuracy                                      │
│     • Consistency                                   │
│     • Latency                                       │
│     • Cost                                          │
│                                                     │
│  2. Create Test Set                                 │
│     • Edge cases                                    │
│     • Common cases                                  │
│     • Error cases                                   │
│                                                     │
│  3. Run A/B Tests                                   │
│     • Variant A vs Variant B                       │
│     • Measure metrics                               │
│                                                     │
│  4. Iterate                                         │
│     • Analyze failures                              │
│     • Refine prompts                                │
│     • Re-test                                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Evaluation Example
```python
import json

test_cases = [
    {"input": "2+2", "expected": "4"},
    {"input": "10*5", "expected": "50"},
    # Add edge cases
    {"input": "1/0", "expected": "Error: Division by zero"}
]

def evaluate_prompt(prompt_template, test_cases):
    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "failures": []
    }

    for case in test_cases:
        prompt = prompt_template.format(input=case["input"])
        response = llm(prompt)

        if response.strip() == case["expected"]:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "input": case["input"],
                "expected": case["expected"],
                "actual": response
            })

    return results
```

---

## 10. Quick Reference Card

```
╔══════════════════════════════════════════════════════════╗
║              PROMPT ENGINEERING CHEAT SHEET              ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║ Technique          When to Use                           ║
║ ─────────────────  ─────────────────────────────────     ║
║ Zero-shot          Simple, clear tasks                   ║
║ Few-shot           Pattern learning needed               ║
║ CoT                Reasoning/math problems               ║
║ ReAct              Multi-step + tool use                 ║
║ ToT                Multiple solution paths               ║
║ Self-consistency   Critical accuracy needed              ║
║ RAG                Knowledge-intensive tasks             ║
║ Chaining           Complex multi-stage workflows         ║
║                                                          ║
║ Key Elements:                                            ║
║ ✓ Role/Persona                                           ║
║ ✓ Clear Context                                          ║
║ ✓ Specific Task                                          ║
║ ✓ Output Format                                          ║
║ ✓ Constraints                                            ║
║ ✓ Examples (if needed)                                   ║
║                                                          ║
║ Token Optimization:                                      ║
║ • Summarize long context                                 ║
║ • Use references instead of duplication                  ║
║ • Cache system prompts                                   ║
║ • Implement sliding window for chat                      ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## 11. LangChain Integration Examples

### Basic Prompt Template
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Simple template
template = PromptTemplate(
    input_variables=["topic", "difficulty"],
    template="Explain {topic} at a {difficulty} level."
)

prompt = template.format(topic="quantum computing", difficulty="beginner")
```

### Chat Prompt with System Message
```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = "You are a {role} with expertise in {domain}."
human_template = "{task}\n\nContext: {context}"

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

messages = chat_prompt.format_messages(
    role="data scientist",
    domain="machine learning",
    task="Analyze this dataset",
    context="Customer churn prediction"
)
```

### Few-Shot Template
```python
from langchain.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_template = """
Input: {input}
Output: {output}
"""

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    ),
    prefix="Give the antonym of the word:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)
```

### Output Parser
```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schemas = [
    ResponseSchema(name="sentiment", description="positive, negative, or neutral"),
    ResponseSchema(name="confidence", description="confidence score 0-1"),
    ResponseSchema(name="reasoning", description="brief explanation")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

template = """
Analyze the sentiment of this text: {text}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)
```

---

## 12. Final Tips

1. **Start Simple, Then Optimize**
   - Begin with basic prompts
   - Add complexity only when needed
   - Measure impact of each change

2. **Version Control Your Prompts**
   - Track changes
   - A/B test variants
   - Document what works

3. **Understand Model Capabilities**
   - Different models excel at different tasks
   - GPT-4: Complex reasoning
   - Claude: Long context, code
   - Gemini: Multimodal

4. **Cost vs Quality Trade-off**
   ```
   Simple task → Use cheaper model (GPT-3.5, Haiku)
   Complex reasoning → Use advanced model (GPT-4, Opus)
   Long context → Use appropriate model (Claude, Gemini)
   ```

5. **Iterate Based on Failures**
   - Log failures
   - Analyze patterns
   - Refine prompts systematically

---

## Resources & Further Reading

- **Papers:**
  - "Chain-of-Thought Prompting" (Wei et al., 2022)
  - "ReAct: Reasoning and Acting" (Yao et al., 2023)
  - "Tree of Thoughts" (Yao et al., 2023)

- **Tools:**
  - LangChain: Framework for LLM applications
  - LangSmith: Prompt testing and monitoring
  - PromptLayer: Prompt management
  - Weights & Biases: Experiment tracking

- **Communities:**
  - LangChain Discord
  - r/PromptEngineering
  - OpenAI Developer Forum

---

**Last Updated:** 2026-02-08

