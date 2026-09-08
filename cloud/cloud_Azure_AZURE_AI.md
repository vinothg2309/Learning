AZURE_AI.md
# Azure AI/ML for Lead & Principal Roles - Beginner's Guide

**Interview-Ready Roadmap with Clear Explanations & Real Examples**

---

## Table of Contents

### [1. Azure AI Core (Must Master)](#1-azure-ai-core-must-master-)
- [Azure AI Studio (New GenAI Hub)](#azure-ai-studio-new-genai-hub)
  - [What is it?](#what-is-it)
  - [Key Components](#key-components)
- [Azure OpenAI Service](#azure-openai-service-)
  - [GPT Models](#gpt-models-gpt-4-gpt-4o-embeddings)
  - [Rate Limits & Quotas](#rate-limits--quotas)
  - [Function Calling (Tool Use)](#function-calling-tool-use)
  - [Streaming Responses](#streaming-responses)
  - [Embedding Models for RAG](#embedding-models-for-rag)
- [RAG on Azure (DEEP KNOWLEDGE REQUIRED)](#rag-on-azure-deep-knowledge-required-)
  - [Azure AI Search](#azure-ai-search)
  - [Embeddings](#embeddings-convert-text-to-numbers)
  - [Indexing & Retrieval Workflow](#indexing--retrieval-workflow)

### [2. Azure ML (MLOps Platform)](#2-azure-ml-mlops-platform-)
- [Azure Machine Learning (AML) - Core Service](#azure-machine-learning-aml---core-service)
  - [Workspaces (Environment Isolation)](#workspaces-environment-isolation)
  - [Training Jobs](#training-jobs-compute-allocation--monitoring)
  - [Model Registry](#model-registry-versioning--lineage)
  - [Endpoints](#endpoints-batch--real-time-inference)
  - [Environment Management](#environment-management-dependencies)
- [Pipelines & CI/CD](#pipelines--cicd)
  - [AML Pipelines](#aml-pipelines-component-based)
  - [Integration with GitHub/Azure DevOps](#integration-with-githubazure-devops-cicd)
- [Feature Store (Basic Understanding)](#feature-store-basic-understanding)

### [3. LLMOps / GenAI Ops (VERY IMPORTANT)](#3-llmops--genai-ops-very-important-)
- [Prompt Versioning](#prompt-versioning)
- [Evaluation Frameworks](#evaluation-frameworks)
  - [RAGAS (RAG Assessment)](#ragas-rag-assessment)
  - [LangSmith → Azure Translation](#langsmith--azure-translation)
- [Guardrails & Safety](#guardrails--safety)
  - [Content Moderation](#content-moderation)
  - [Output Validation](#output-validation)
  - [Cost Control (Token Limits)](#cost-control-token-limits)
  - [Latency Monitoring](#latency-monitoring)
- [Mapping Your Experience](#mapping-your-experience)
  - [LangSmith → Azure](#langsmith--azure)
  - [RAGAS → Azure](#ragas--azure)

### [4. Azure Infrastructure (Critical for Architect Roles)](#4-azure-infrastructure-critical-for-architect-roles-)
- [Compute (Where your code runs)](#compute-where-your-code-runs)
  - [AKS (Azure Kubernetes Service)](#aks-azure-kubernetes-service-)
  - [Azure Container Apps](#azure-container-apps)
  - [VMs (GPU Nodes)](#vms-gpu-nodes)
- [Storage (Where your data lives)](#storage-where-your-data-lives)
  - [Blob Storage](#blob-storage)
  - [Azure Data Lake (ADLS)](#azure-data-lake-adls)
  - [Cosmos DB (NoSQL Database)](#cosmos-db-nosql-database)
- [Networking](#networking)
  - [VNet (Virtual Networks)](#vnet-virtual-networks)
  - [Private Endpoints (VERY IMPORTANT)](#private-endpoints-very-important)
  - [API Gateway (Azure API Management)](#api-gateway-azure-api-management)

### [5. Identity & Security (VERY IMPORTANT IN AZURE)](#5-identity--security-very-important-in-azure-)
- [Azure Entra ID (Azure AD)](#azure-entra-id-azure-ad)
  - [RBAC (Role-Based Access Control)](#rbac-role-based-access-control)
  - [Managed Identity](#managed-identity)
  - [Service Principals](#service-principals)
  - [OIDC Integration](#oidc-integration)
- [Key Vault](#key-vault)

### [6. Data Layer (You WILL be asked)](#6-data-layer-you-will-be-asked-)
- [Azure AI Search (Deep Knowledge)](#azure-ai-search-deep-knowledge)
- [Cosmos DB vs Azure SQL](#cosmos-db-vs-azure-sql)

### [7. Deployment & Serving](#7-deployment--serving-)
- [When to use which service?](#when-to-use-which-service)

### [8. Observability & Monitoring](#8-observability--monitoring-)
- [Azure Monitor](#azure-monitor)
- [Application Insights](#application-insights)

### [9. System Design (MOST IMPORTANT)](#9-system-design-most-important-)
- [Design Pattern 1: Enterprise RAG System](#design-pattern-1-enterprise-rag-system)

### [10. Key Interview Topics & Questions](#10-key-interview-topics--questions-)
- [Question 1: Design Enterprise RAG](#question-1-design-enterprise-rag)
- [Question 2: Cost Optimization](#question-2-cost-optimization)
- [Question 3: Security & Compliance](#question-3-security--compliance)

### [Quick Reference: Services & Their Purpose](#quick-reference-services--their-purpose)
### [Final Tips for Interviews](#final-tips-for-interviews)

---

## 1. Azure AI Core (Must Master) 🧠

### Azure AI Studio (New GenAI Hub)

#### What is it?
Think of Azure AI Studio like a **single creative workspace** where you can:
- Write and test prompts (like a document editor for AI)
- Run AI models without writing code
- Test outputs automatically
- Deploy models to production

#### Why does it matter?
- **For you**: Instead of jumping between 5 different tools, everything is in one place
- **For companies**: Teams can collaborate better; less confusion
- **For interviews**: Shows you understand modern AI workflows

#### Clear Example
Imagine you work at a bank building a customer service chatbot.

**Without Azure AI Studio** (the old way):
```
Day 1: Write prompt in Notepad
Day 2: Test it in Python script
Day 3: Share feedback via email
Day 4: Someone updates it differently
Day 5: Chaos - nobody knows which version works
```

**With Azure AI Studio** (the new way):
```
1. Open Azure AI Studio
2. Write prompt in editor: "You are a helpful bank assistant..."
3. Click "Test" → see how it responds to sample questions
4. Teammates can see changes in real-time
5. Run automated tests (does it answer correctly?)
6. Click "Deploy" → it's live
7. Everyone can track changes (like Google Docs)
```

#### Key Components

**Prompt Flow**
- **What**: Visual tool to build chatbot conversations
- **Simple explanation**: Connect blocks (user input → AI → validator → response)
- **Example**: 
  ```
  User asks: "Can I get a refund?"
    ↓
  [Send to GPT-4]
    ↓
  [Check if response is safe]
    ↓
  [If safe, show response; else block it]
  ```

**Model Deployments (GPT, OSS)**
- **What**: Use different AI models for different tasks
- **Simple explanation**: 
  - GPT-4: Very smart, but expensive (use for complex questions)
  - GPT-4o: Good and cheaper (use for most things)
  - Open-source models: Free but less smart (use for simple tasks)
- **Example at a bank**:
  ```
  Complex question: "Help me understand mortgage terms" → Use GPT-4 (best quality)
  Simple question: "What's your hours?" → Use GPT-4o (fast & cheap)
  Internal task: "Summarize email" → Use free open-source model (costs $0)
  ```

**Evaluation Pipelines**
- **What**: Automatic tests that check if model outputs are good
- **Simple explanation**: Like a quality inspector at a factory
- **Example**:
  ```
  Test 100 customer questions:
  ✅ 87% answered correctly
  ✅ 92% responses are safe
  ✅ Average response time: 2 seconds
  Result: Model passes quality check, can deploy
  ```

**Safety Filters & Content Moderation**
- **What**: Blocks bad responses automatically
- **Simple explanation**: Like a bouncer at a club (good stuff in, bad stuff blocked)
- **Example**:
  ```
  Model tries to say: "I can help you commit fraud"
  Safety filter blocks it
  Customer gets: "I can't help with that"
  ```

---

### Azure OpenAI Service ⭐⭐⭐

#### What is it?
A service that lets you use powerful AI models (GPT-4, GPT-4o) through Azure's secure, enterprise-grade infrastructure.

#### Why does it matter?
- **For companies**: Can't just use ChatGPT.com (no data privacy, no SLA, not enterprise-ready)
- **For interviews**: This is the #1 most asked service in Azure interviews
- **For your role**: Every company building AI uses this

#### Simple Analogy
```
ChatGPT.com = eating at a food truck (cheap, but anyone can see what you order)
Azure OpenAI = private kitchen in a restaurant (secure, professional, guaranteed quality)
```

#### Key Components

**GPT Models (gpt-4, gpt-4o, embeddings)**

What each model does:

1. **GPT-4**: The smartest, most capable
   - **What**: Complex reasoning, programming, analysis
   - **Cost**: ~$0.03 per 1K tokens (expensive)
   - **Speed**: 1-2 seconds per response
   - **Example use**: Analyzing legal contracts, strategic planning

2. **GPT-4o** (4 optimized): Almost as smart, much faster & cheaper
   - **What**: Most general tasks
   - **Cost**: ~$0.006 per 1K tokens (4x cheaper than GPT-4)
   - **Speed**: 0.5 seconds per response
   - **Example use**: Customer service, content writing, summarization

3. **Embeddings**: Convert text to numbers
   - **What**: Find similar documents, group by meaning
   - **Cost**: ~$0.00002 per 1K tokens (very cheap!)
   - **Speed**: Instant
   - **Example use**: Search documents, find duplicates

**Interview tip**: Know when to use which model
```
Question: "Should I use GPT-4 or GPT-4o?"
Good answer: "It depends on the task. For complex analysis, GPT-4. 
For general customer service, GPT-4o (4x cheaper, 95% as smart, faster)."
```

**Rate Limits & Quotas**
- **What**: Restrictions on how fast/much you can use the service
- **Simple explanation**: Like speed limits on a highway
- **Example**:
  ```
  You get 40,000 tokens per minute
  Each customer question ≈ 500 tokens (input) + 500 tokens (response) = 1,000 tokens
  Max: 40 concurrent customers
  
  If 50 customers ask at same time → some requests wait in queue
  ```
- **Interview question**: "Your app suddenly gets 10x more users. What happens?"
  - Answer: "Request queue grows, latency increases. Solution: Add more rate limit quota, or implement batching."

**Function Calling (Tool Use)**
- **What**: Let the model call external tools/APIs
- **Simple explanation**: Model doesn't just generate text; it can trigger actions
- **Example**:
  ```
  User: "What's my account balance?"
  
  Model thinks: "I need to call the bank's API"
  Model calls: bank_api.get_account_balance(customer_id=123)
  API returns: $5,000
  Model responds: "Your balance is $5,000"
  ```
- **Real-world example**:
  ```
  Agent system:
  User: "Book a flight to New York tomorrow"
    ↓
  Model thinks: I need 3 tools
    1. Search flights: search_flights(from="SF", to="NY", date="tomorrow")
    2. Check price: check_price(flight_id=456)
    3. Book: book_flight(flight_id=456, customer_id=789)
    ↓
  System executes all 3 tools in sequence
    ↓
  Model response: "Booked flight XYZ, departs 10am tomorrow, costs $200"
  ```

**Streaming Responses**
- **What**: Get response word-by-word instead of waiting for complete response
- **Why**: Better user experience (they see response appearing in real-time)
- **Example**:
  ```
  Without streaming:
  User clicks "Send"
  (Wait 5 seconds...)
  Full response appears all at once
  
  With streaming:
  User clicks "Send"
  Response starts appearing immediately:
  "Hello, I can help..."
  "You can return items within..."
  "Please contact customer service at..."
  (More natural, feels faster)
  ```

**Embedding Models for RAG**
- **What**: Convert documents to numbers so you can search by meaning
- **Why**: Foundation of RAG (finding relevant documents)
- **Simple example**:
  ```
  Bank documents:
  - "Money back policy" → converted to vector [0.2, -0.5, 0.8, ...]
  - "Refund process" → converted to vector [0.21, -0.48, 0.79, ...]
  - "Account closure" → converted to vector [0.1, 0.3, -0.2, ...]
  
  User searches: "Can I get my money back?"
  - Gets converted to vector [0.19, -0.52, 0.82, ...]
  - System finds closest vectors = "Money back policy" & "Refund process"
  - Shows these to GPT to answer the question
  ```

---

### RAG on Azure (DEEP KNOWLEDGE REQUIRED) 🔥

#### What is RAG?
**RAG = Retrieval-Augmented Generation**

**Simple explanation**: 
Instead of asking a model to generate answers from its training data (which can be wrong), you:
1. Find relevant documents
2. Give those documents to the model
3. Ask it to answer based on those documents

**Why it's better**:
- ✅ Answers are backed by actual documents (less hallucination)
- ✅ Works with company's private data (not in training data)
- ✅ Easy to update (add new documents, no retraining)
- ✅ Cheaper than fine-tuning

**Real-world analogy**:
```
Question: "What's our return policy?"

Without RAG (dangerous):
Model generates from memory: "I think you can return within 30 days"
(But what if it's actually 60 days? Model is guessing)

With RAG (safe):
1. Search company documents
2. Find: "Return Policy: 60 days with receipt"
3. Show to model
4. Model says: "According to our policy, you can return within 60 days"
(Now it's guaranteed correct)
```

---

#### Azure AI Search

**What is it?**
A search engine that finds documents by meaning (not just keywords).

**Simple explanation**:
Like Google, but for your company's documents, and it understands meaning.

**Example**:
```
Company documents:
1. "Refund Policy: You can get money back within 60 days"
2. "Return Process: Complete form and ship item back"
3. "Warranty Terms: Covers manufacturing defects for 1 year"

User search: "Can I get my money back?"

Keyword search (dumb):
  - Finds doc #1 (has word "money")
  - Maybe misses doc #2 (no word "money" but about returns)

Semantic search (smart):
  - Understands "get money back" = "refund" = "return"
  - Finds docs #1 and #2 (both relevant)
  - Doesn't return doc #3 (warranty ≠ refund)
```

**Key concepts**:

**Vector Search**
- **What**: Search based on meaning, not keywords
- **How**: Convert documents to numbers → compare numbers → find similar
- **Real example**:
  ```
  Doc A: "How do I return a damaged item?"
  Doc B: "What's the procedure for returning products?"
  
  These mean the same thing (return process)
  Vector search finds them as similar
  Keyword search might miss one (different words)
  ```

**Hybrid Search (BM25 + Vector)**
- **What**: Use both keyword matching AND meaning matching
- **Why**: Catches everything
- **Example**:
  ```
  Search: "GPT"
  
  Keyword match: Finds docs with word "GPT"
  Vector match: Finds docs about "generative AI" (means similar)
  Together: Don't miss anything
  ```

**Indexing Strategies**
- **What**: How to organize documents for fast searching
- **Simple explanation**: Like a library's card catalog (organized so books are easy to find)
- **Decision 1**: Update index immediately or in batches?
  ```
  Real-time: Search results always current, but costs more
  Batch: Slightly old results, but cheaper
  
  Example: Customer service (real-time) vs analytics report (batch)
  ```
- **Decision 2**: Which fields to search?
  ```
  Search all fields = slower & more expensive
  Search important fields only (title, content) = faster & cheaper
  ```

**Query Expansion & Reranking**
- **Query expansion**: Generate alternative searches
  - User searches: "car accident"
  - System also searches: "vehicle collision", "traffic incident", "crash"
  - Returns best results from all
  
- **Reranking**: Get 100 results, use AI to pick best 10
  - Example: Azure Bing Reranker (more accurate than raw similarity)
  - Usually worth the cost (improves quality)

---

#### Embeddings (Convert Text to Numbers)

**What is it?**
Convert words/documents into lists of numbers that represent meaning.

**Simple visualization**:
```
Document: "The bank approved my loan"
  ↓
Embedding model processes it
  ↓
Vector: [0.23, -0.45, 0.89, 0.12, -0.33, ..., 0.78]
         (1536 numbers for large model)

Document: "I got my loan"
  ↓
Vector: [0.24, -0.44, 0.91, 0.11, -0.32, ..., 0.79]
         (Very similar numbers! So search finds it)
```

**Why embeddings matter**:
- Numbers are comparable (find similar documents)
- Numbers are compact (easy to store in database)
- Numbers preserve meaning (synonym documents have similar numbers)

**Embedding models in Azure**:

1. **text-embedding-3-small**
   - Cost: Cheapest
   - Speed: Fastest
   - Quality: Good enough for most tasks
   - Size: 512 dimensions (smaller vector)
   - Example use: E-commerce product search, FAQ matching

2. **text-embedding-3-large**
   - Cost: 2x more expensive
   - Speed: Slightly slower
   - Quality: Best quality, catches nuances
   - Size: 1536 dimensions (bigger vector)
   - Example use: Legal document search, medical literature

**Interview question**: "Which embedding model would you choose?"
- Answer: "Depends on use case. For 99% of applications, use small (fast & cheap). For high-precision domains (law, medicine), use large."

**OSS (Open-Source) Alternatives**:
- **What**: Free embedding models you can run yourself
- **Examples**: Sentence-BERT, E5, UAE Embeddings
- **Trade-off**: 
  - ✅ No cost (vs OpenAI's embedding costs)
  - ✅ Your data stays private (vs sending to OpenAI)
  - ❌ Slightly lower quality
- **When to use**: Cost-sensitive projects, privacy-critical applications

**Chunking Strategies** (Breaking documents into pieces)
- **What**: Large documents → split into smaller pieces → embed each piece
- **Why**: LLMs have limited memory (can't read entire 100-page document)

**Example**:
```
50-page employee handbook
  ↓
Split into chunks (strategies below)
  ↓
Each chunk embedded separately
  ↓
When user asks question, search finds relevant chunks
  ↓
Show relevant chunks to GPT → GPT answers based on them
```

**3 chunking strategies**:

1. **Fixed-size chunking** (simple)
   ```
   Split every 500 words
   Chunk 1: words 1-500
   Chunk 2: words 501-1000
   Chunk 3: words 1001-1500
   
   ✅ Pro: Simple to implement
   ❌ Con: Might cut mid-sentence, lose context
   ```

2. **Semantic chunking** (smart)
   ```
   Split at natural boundaries (paragraphs, sections)
   Chunk 1: Introduction section
   Chunk 2: Policy section
   Chunk 3: FAQ section
   
   ✅ Pro: Preserves meaning
   ❌ Con: Harder to implement
   ```

3. **Sliding window chunking** (balanced)
   ```
   Chunk 1: words 1-500
   Chunk 2: words 400-900 (20% overlap)
   Chunk 3: words 800-1300
   
   ✅ Pro: Context preserved, not too complex
   ❌ Con: Creates duplicate work
   ```

**Batch Processing**
- **What**: Process thousands of documents offline
- **When**: Initial setup of your RAG system
- **Why**: Cheaper than real-time API calls
- **Example**:
  ```
  Company has 10,000 documents
  Cost to embed all at once: $20
  Cost to embed on-demand as needed: $50-100/month
  
  → Better to batch embed everything upfront
  ```

---

#### Indexing & Retrieval Workflow

**Full RAG flow**:
```
1. Document arrives
2. Split into chunks (semantic chunking)
3. Create embedding for each chunk
4. Store in Azure AI Search
5. User asks question
6. Embed user question
7. Search for similar chunks (vector + keyword)
8. Rerank top 10 results
9. Pass to GPT-4
10. GPT answers based on chunks
11. Return answer to user
```

**Key Topics**:

**Document Chunking (Semantic vs Fixed-Size)**
- Already explained above

**Metadata Filtering**
- **What**: Filter search results before returning
- **Simple example**:
  ```
  Documents have metadata tags:
  - Document type: "Policy", "FAQ", "News"
  - Date: "2024-01", "2024-02"
  - Department: "HR", "Finance", "IT"
  
  Search for: "Leave policy from Finance, 2024"
  → Filter: type="Policy" AND dept="Finance" AND date>"2024-01-01"
  → Returns only matching documents
  → Faster & cheaper search
  ```

**Ranking & Relevance Tuning**
- **What**: Adjust how much each factor matters in search
- **Example**:
  ```
  Default: 60% keyword + 40% semantic
  
  For legal docs: 80% keyword + 20% semantic
  (Exact clause matching more important)
  
  For medical docs: 30% keyword + 70% semantic
  (Understanding meaning more important than exact terms)
  ```

**Re-ranking Strategies**
- **What**: Get 100 results, use better AI to pick top 10
- **Why**: Raw similarity might miss context
- **Example**:
  ```
  Initial search returns:
  1. Doc A (score: 0.85)
  2. Doc B (score: 0.84)
  3. Doc C (score: 0.83)
  ...
  100. Doc Z (score: 0.50)
  
  Reranker (more sophisticated):
  1. Doc B (better answers the question)
  2. Doc A
  3. Doc C
  (Ranking is better now)
  
  Cost: ~$0.02 to rerank 100 docs (very cheap for quality improvement)
  ```

**Expected Interview Question**:
> "Design an enterprise RAG system using Azure for a financial services company with 50,000 documents."

**Answer structure**:
1. **Architecture**: 
   - Document ingestion (from Data Lake)
   - Semantic chunking (preserve context)
   - OpenAI embeddings (use large model for accuracy)
   - Azure AI Search (hybrid search)
   - GPT-4 for answer generation
   
2. **Key decisions**:
   - Why semantic chunking? "Regulations are complex; preserving sentence boundaries is critical"
   - Why hybrid search? "Need both exact clause matches (keyword) and meaning matches (vector)"
   - Why reranking? "Financial accuracy matters; reranker ensures best results"
   
3. **Scale handling**:
   - 50K documents = batch processing upfront
   - Index updates daily (documents change)
   - Regional replicas (latency)
   
4. **Quality assurance**:
   - Evaluation: RAGAS metrics (faithfulness, relevance, precision)
   - Pass/fail gates before production
   - Monitor hallucinations in production
   
5. **Cost optimization**:
   - Cache embeddings (don't recompute)
   - Batch queries when possible
   - Use smaller embedding model if quality allows

---

## 2. Azure ML (MLOps Platform) ⚙️

#### What is it?
The platform for managing the complete machine learning lifecycle:
Data → Training → Testing → Deployment → Monitoring

**Simple analogy**: Like a factory management system (intake → process → QA → ship → track)

#### Why does it matter?
- **For companies**: Without MLOps, ML projects fail (no reproducibility, no monitoring)
- **For interviews**: "How do you handle ML in production?" → This is the answer
- **For your role**: As a lead, you architect this infrastructure

#### Core Problems it Solves

**Problem 1: Data Chaos**
```
Without MLOps:
Engineer 1: Trains on 2024 data
Engineer 2: Trains on 2023 data
Both get different results → confusion

With MLOps:
All training tracked (which data? which version? which date?)
Can reproduce exactly
```

**Problem 2: Model Management Nightmare**
```
Without MLOps:
- Models scattered (Google Drive, someone's laptop, etc.)
- Nobody knows which model is in production
- Old model has bug; deploy new one; new one is worse; can't rollback

With MLOps:
- Central registry (all models versioned)
- Know exactly which model is live
- Easy rollback
```

**Problem 3: Compliance Issues**
```
Auditor asks: "How was this model trained? What data? By whom? When?"

Without MLOps: "Uh... I think John trained it... last month?"
With MLOps: "Model v2.1, trained by John on 2024-01-15, using dataset XYZ (v3.2), with code commit abc123"
```

---

### Azure Machine Learning (AML) - Core Service

#### Workspaces (Environment Isolation)

**What is it?**
A container that holds all your ML assets (data, code, models, training jobs).

**Simple analogy**: Like having separate project folders (dev folder ≠ production folder)

**Why it matters**:
- Keep dev & production separate (mistakes don't affect live systems)
- Different teams can work independently
- Cost tracking per team/project

**Real-world example**:
```
Company structure:
├── NLP Team
│   └── Workspace: nlp-prod
│       ├── Datasets
│       ├── Models (sentiment analysis, intent detection, etc.)
│       ├── Training jobs
│       └── Deployed endpoints
│
└── Computer Vision Team
    └── Workspace: vision-prod
        ├── Datasets
        ├── Models (object detection, image classification, etc.)
        ├── Training jobs
        └── Deployed endpoints

Benefits:
- NLP team doesn't accidentally use Vision team's resources
- Billing separate
- Access control separate
```

#### Training Jobs (Compute Allocation & Monitoring)

**What is it?**
Submit training code → Azure allocates GPU → code runs → metrics tracked → results saved

**Simple workflow**:
```
You: "Train my model on this data using GPUs"
Azure: "Starting job XYZ on GPU cluster"
(Your code runs...)
Job finishes → Metrics saved → Model saved
You: "Great! Can I see the metrics?"
Azure: Shows loss curves, accuracy, training time, resource usage
```

**Why it matters**:
- **Reproducibility**: Same code + same data = same result (every time)
- **Tracking**: All experiments logged (vs scattered notebooks)
- **Cost**: Auto-cleanup (don't forget to turn off GPUs)

**Real example at a company**:
```
Training job run 1:
- Engineer: Sarah
- Date: Jan 15, 2024
- Model: BERT
- Learning rate: 0.001
- Epochs: 10
- Results: Accuracy 92.1%

Training job run 2:
- Same setup as run 1
- Results: Accuracy 92.1% (reproducible! ✅)

Training job run 3:
- Changed learning rate to 0.0001
- Results: Accuracy 94.2% (better!)
- Compare: run 3 is 2.1% better than run 1
```

#### Model Registry (Versioning & Lineage)

**What is it?**
A database of all your models with full history (who trained it, when, with what data, what performance).

**Simple analogy**: Like a museum's artifact registry (catalog everything, know its history)

**Why it matters**:
- Audit trail (who deployed what when?)
- Easy rollback (bad deployment → restore previous version in 1 click)
- Performance comparison (model v1.0 vs v1.1 - which is better?)

**Real-world scenario**:
```
Timeline:
Mon: Deploy model v1.0 → users happy
Wed: Deploy model v2.0 (improvements) → users report bugs
Fri: Deploy model v1.5 (fixed bugs)

Your boss asks: "What happened Monday?"
Without registry: "Uh... we deployed something?"
With registry: 
- v1.0 deployed Mon by John, accuracy 92%, latency 100ms
- v2.0 deployed Wed by Sarah, accuracy 95%, latency 150ms (slower!)
- Issue: Better accuracy but slower response
- v1.5 deployed Fri by John, accuracy 94%, latency 90ms (best of both)
```

#### Endpoints (Batch & Real-time Inference)

**What is it?**
Expose trained model as an API that applications can call.

**Two types**:

**Real-time Endpoint**
```
Application calls endpoint:
  App: "Classify this email"
  Endpoint: (immediately) "Spam" [100ms]
  App continues

Used for: Interactive, immediate feedback needed
Examples:
- Email classification (show result to user now)
- Recommendation system (show product now)
- Chatbot response (user waiting for answer)
```

**Batch Endpoint**
```
Application submits job:
  App: "Score 1 million customers tonight"
  Endpoint: "Job queued, will finish by 6am"
  App: (next morning) "Get results"
  Endpoint: Returns scores for all 1M customers

Used for: Large-scale processing, not time-sensitive
Examples:
- Nightly customer scoring
- Monthly report generation
- Bulk email campaign targeting
```

**Interview question**: "When would you use batch vs real-time?"
```
Good answer:
"Batch for back-office tasks (nightly scoring, monthly reports).
Real-time for user-facing features (chatbot, recommendations).
Batch is 10x cheaper per prediction but has latency.
Real-time is faster but more expensive."
```

#### Environment Management (Dependencies)

**What is it?**
Specify exact Python version, library versions → reproducibility guaranteed

**Why it matters**:
```
Problem: "Works in my environment but not in production"

Without environment management:
Your machine: Python 3.9, scikit-learn 0.24
Production: Python 3.11, scikit-learn 1.0
Results different (small changes in libraries can affect ML)

With environment management (AML):
Define once: Python 3.9, scikit-learn 0.24
Every run uses exact same environment
Always reproducible
```

**Real example**:
```
requirements.txt
torch==2.0.1
transformers==4.30.0
numpy==1.24.3

AML creates exact environment from this file
6 months later, you retrain model
Same code + same environment → same results
No surprises
```

---

### Pipelines & CI/CD

#### What is it?
Automated workflows that connect multiple steps (no manual intervention).

**Simple example**:
```
Old way (manual):
Mon: Download data
Tue: Clean data
Wed: Train model
Thu: Test model
Fri: Deploy if good
(5 days, error-prone)

New way (automated pipeline):
Monday 9am: Trigger pipeline
Mon 9:05: Data downloaded
Mon 9:10: Data cleaned
Mon 9:20: Model trained
Mon 9:30: Tests run
Mon 9:35: If good → Deploy
(26 minutes, no manual steps, no errors)
```

#### AML Pipelines (Component-based)

**What is it?**
Chain multiple steps together, each reusable.

**Visual example**:
```
Step 1: Data Ingestion
   ↓ (data passes to step 2)
Step 2: Data Validation
   ↓ (validated data passes to step 3)
Step 3: Feature Engineering
   ↓ (features pass to step 4)
Step 4: Model Training
   ↓ (trained model passes to step 5)
Step 5: Evaluation
   ↓ (if metrics pass:)
Step 6: Deploy to Production
```

**Real-world example at a company**:
```
Bank's loan approval model pipeline:

Step 1: Data Ingestion
  - Pull customer applications from database
  - Output: 10,000 applications

Step 2: Data Validation
  - Check for missing values
  - Check data types
  - Output: 9,950 valid applications (50 had errors)

Step 3: Feature Engineering
  - Calculate credit score from data
  - Calculate income stability
  - Calculate debt ratio
  - Output: Applications with 50 features each

Step 4: Model Training
  - Train logistic regression on features
  - Split: 80% train, 20% test
  - Output: Trained model

Step 5: Evaluation
  - Test on 20% held-out data
  - Check: Accuracy, False positive rate, False negative rate
  - Check: Does accuracy > 90%? Is false positive rate < 5%?
  - Output: Metrics report

Step 6: Decision
  If metrics pass: Deploy to production
  If metrics fail: Notify data scientist, stop pipeline
```

**Key Concepts**:

**Component-based Architecture**
- Each step is self-contained (can use separately)
- Reusable across projects
- Example:
  ```
  "StandardScaler component" used by:
  - Loan approval model
  - Credit card fraud detection
  - Customer churn prediction
  (All three teams reuse same component)
  ```

**Data Passing**
- Output of step 1 → Input of step 2
- Automatic serialization (Azure handles it)

**Versioning & Reproducibility**
- Each run of pipeline is versioned
- Can compare: pipeline run v1 vs v2
- Can re-run old pipeline with new parameters

#### Integration with GitHub/Azure DevOps (CI/CD)

**What is it?**
Automatically trigger pipeline when code changes.

**Workflow**:
```
1. Engineer commits new training code to GitHub
   (Commit message: "Improve feature engineering")

2. GitHub webhook triggers Azure DevOps

3. Azure pipeline:
   a) Runs linting checks (code quality)
   b) Runs unit tests
   c) Triggers AML training pipeline
   d) Evaluates: Does new model improve accuracy?
   e) If yes → Deploy to staging
   f) If no → Reject merge, notify engineer

4. Engineer gets feedback in 5 minutes (not next week)
```

**Why it matters**:
- Speed: Deploy improvements in minutes (not weeks)
- Safety: Bad models can't reach production
- Auditability: Every deployment linked to code commit

**Real-world example**:
```
Monday:
Engineer Sarah commits: "Add new feature: customer_account_age"

Automated:
- 9:00am: Code committed
- 9:05am: Tests run (pass ✅)
- 9:10am: AML trains new model
- 9:25am: Evaluation: Accuracy 93.2% (vs 92.5% before)
- 9:30am: Auto-deploy to production
- 9:35am: Sarah gets Slack notification: "Model deployed!"

No manual work, no bottlenecks, high quality guaranteed
```

---

### Feature Store (Basic Understanding)

#### What is it?
A database of pre-computed features (derived data) that both training and serving use.

**Simple analogy**:
```
Without Feature Store:
Training: Calculate "customer_spend_last_30_days" from raw data
Serving: Calculate "customer_spend_last_30_days" from raw data
(Same calculation twice, might be slightly different)

With Feature Store:
Calculate "customer_spend_last_30_days" once
Both training and serving use that exact number
(Guaranteed consistency)
```

#### Real-world example

**Company**: E-commerce

**Feature**: "Has customer purchased in last 30 days?"

```
Feature Store stores:
customer_id: 123 → feature_value: true
customer_id: 456 → feature_value: false
customer_id: 789 → feature_value: true
(... millions of customers)

Training pipeline:
- Build recommendation model
- Uses: Has purchased (last 30d), Purchase frequency, Last purchase date
- All features from Feature Store
- Train on 2024-01-01 data

Serving (recommendation):
- User visits website
- Look up: Has user purchased (last 30d)?
- Feature Store returns: true
- Use to generate recommendation
- Show "Complete your recent items" section

Same feature definition, guaranteed consistency
```

#### Key Concepts

**Feature Engineering Automation**
- **What**: Automatically compute features on schedule
- **Why**: No manual data prep, reduce mistakes
- **Example**: Every day, Feature Store computes "spend_last_30d" for all customers

**Feature Versioning**
- **What**: Track which features were used in which model version
- **Example**:
  ```
  Model v1.0 (2024-01-01):
  - Features: customer_spend, purchase_count, days_since_purchase
  - Performance: 85% accuracy
  
  Model v2.0 (2024-02-01):
  - Added feature: customer_segment
  - Performance: 87% accuracy (improvement!)
  
  Why did v2.0 improve? → Because of new feature!
  ```

**Online vs Offline Serving**
- **Offline** (for training):
  ```
  Daily job at 2am:
  Compute all features for all customers
  Store in database
  Training pipeline reads from database
  ```
  
- **Online** (for real-time serving):
  ```
  User visits website → 10ms deadline
  Look up: "What features does this user have?"
  Feature Store returns in <5ms
  ```

**Point-in-Time Correctness**
- **What**: Get feature value as it was at specific time
- **Why**: Avoid data leakage
- **Example**:
  ```
  Training on 2024-01-15 data
  Don't use feature values from 2024-02-01 (that's the future!)
  
  Feature Store enforces: training date = 2024-01-15
                       feature date = 2024-01-15
  (same day)
  ```

---

## 3. LLMOps / GenAI Ops (VERY IMPORTANT) 🔥

#### What is it?
The discipline of managing LLM applications in production (prompts, evaluation, deployment, monitoring).

**Simple analogy**:
```
MLOps: Manage traditional ML models
LLMOps: Manage large language models

Both need:
- Version control
- Testing
- Deployment
- Monitoring

But LLMs have unique challenges:
- Prompts instead of code
- Hallucinations instead of wrong predictions
- Token cost instead of compute cost
```

#### Why does it matter?
- Interview: "How do you ensure quality LLM responses?" → This is the answer
- Production: Without LLMOps, quality degrades, costs spiral
- Your role: Define LLMOps processes for your team

---

### Prompt Versioning

#### What is it?
Version control for prompts (like Git but for text).

#### Why it matters?
- **Test before deploying**: Try different prompts, measure quality
- **Rollback**: Bad prompt broke production → revert to good version
- **Collaboration**: Multiple people safely work on same prompt

#### Real-world workflow

```
Prompt v1:
"Answer the question"
Test on 100 samples: Success rate 85%

Prompt v2:
"Answer the question in 2-3 sentences"
Test on 100 samples: Success rate 92% ✅ (better!)

Prompt v3:
"Answer the question in 2-3 sentences. Be concise."
Test on 100 samples: Success rate 91% (worse than v2)

Decision: Deploy v2 to production
```

#### Where to version prompts?

**Option 1: Git repository**
```
/prompts
  /customer-service
    /v1.txt
    /v2.txt
    /v3.txt
  /loan-approval
    /v1.txt
    /v2.txt
```
- Pro: Integrated with code
- Con: Not great UI for non-engineers

**Option 2: LangSmith (specialized tool)**
- Pro: Great UI for experiments, feedback loops
- Con: Another tool to learn

**Option 3: Azure AI Studio**
- Pro: Integrated with Azure ecosystem
- Con: Specific to Azure

---

### Evaluation Frameworks

#### What is it?
Automated testing to measure model output quality.

#### Why it matters?
```
Without evaluation:
- Deploy new prompt
- Get customer complaints after 3 days
- Revert and debug

With evaluation:
- Test new prompt on 100 samples before deployment
- Catch issues immediately
- Deploy only if metrics improve
```

#### Simple evaluation example

```
Test case: "What's the return policy?"
Expected answer: "You can return within 60 days with receipt"

Automated checks:
✅ Does answer mention timeframe? (60 days)
✅ Does answer mention condition? (with receipt)
✅ Is answer accurate? (matches policy document)
✅ Is answer safe? (no harmful content)
✅ Is answer concise? (<100 words)

Success rate: If passes 5/5 checks, it passes
```

#### RAGAS (RAG Assessment)

**What is it?**
Framework to measure RAG system quality automatically.

**Why it matters**:
```
Without RAGAS:
- Hard to measure if RAG is working
- Change indexing strategy, don't know if it helped
- Deploy bad changes, don't realize

With RAGAS:
- Metric score: 0.78 before, 0.85 after
- Clear improvement (deploy!)
```

**4 Key RAGAS Metrics**:

1. **Faithfulness** (Is answer grounded in documents?)
   ```
   Context document: "Company offers 30 days return, needs receipt"
   
   Answer 1: "You can return within 30 days" ✅ (faithful)
   Answer 2: "You can return within 60 days" ❌ (faithfulness fail)
   Answer 3: "You can return without receipt" ❌ (hallucinated info)
   
   Metric: 1.0 (perfect), 0.5 (partial), 0.0 (not faithful)
   ```

2. **Answer Relevance** (Does answer address the question?)
   ```
   Question: "Can I return my purchase?"
   
   Answer 1: "Yes, you can return items within 30 days" ✅ (relevant)
   Answer 2: "Company was founded in 2010" ❌ (not relevant)
   Answer 3: "Return policy: [full policy text]" ✅ (relevant but verbose)
   
   Metric: Measures if answer directly addresses question
   ```

3. **Context Precision** (Are retrieved documents relevant?)
   ```
   Question: "What's the return policy?"
   Retrieved documents:
   1. "Return Policy" ✅ (relevant)
   2. "Shipping Info" ❌ (not relevant to return policy)
   3. "Our History" ❌ (not relevant)
   
   Metric: 1/3 = 0.33 (could be better)
   ```

4. **Context Recall** (Did you retrieve all relevant documents?)
   ```
   Question: "What's the return policy?"
   Relevant documents in system: 3 docs
   - "Return Policy"
   - "Return Form"
   - "Return Shipping"
   
   You retrieved: "Return Policy" + "Return Form"
   You missed: "Return Shipping"
   
   Metric: 2/3 = 0.67 (missed 33% of relevant docs)
   ```

#### LangSmith to Azure Translation

**LangSmith** (LangChain's tool):
- Track LLM calls
- A/B test prompts
- Collect user feedback

**Azure AI Studio** (Microsoft's tool):
- Track LLM calls ✅
- A/B test prompts ✅
- Evaluate automatically ✅
- Integrated with MLOps ✅

**Migration example**:
```
In LangSmith:
- Create dataset (100 test questions)
- Run prompt v1 on dataset
- Score: 85%

In Azure AI Studio:
- Create evaluation dataset (same 100 questions)
- Run prompt v1 on dataset
- Metric: 85% ✅ (same!)
```

---

### Guardrails & Safety

#### What is it?
Mechanisms to ensure LLM outputs are safe, correct, and compliant.

#### Why it matters?
```
Without guardrails:
- Model gives wrong financial advice → Customer loses money → Company liable
- Model leaks customer data (PII) → Regulatory fine
- Model generates hate speech → Reputation damage

With guardrails:
- Wrong advice blocked
- PII detected and masked
- Offensive content blocked
```

#### Content Moderation

**What is it?**
Automatic filtering that blocks/flags unsafe content.

**Real-world examples**:

```
1. Hate speech detection
   Model tries to generate: "Group X is bad"
   Safety filter: Blocks it
   Customer sees: "I can't help with that"

2. PII (Personal information) detection
   Model tries to generate: "Your SSN is 123-45-6789"
   Safety filter: Masks it
   Customer sees: "Your SSN is [REDACTED]"

3. Medical/Financial advice detection
   Model tries to generate: "Take aspirin for your pain"
   Safety filter: Blocks it (requires licensed professional)
   Customer sees: "Please consult a doctor"
```

#### Output Validation

**What is it?**
Structured checks that output meets requirements.

**Real-world example** (Loan approval system):
```
Customer: "Should I get approved for a loan?"

System generates: "You should get a loan"

Validation checks:
❌ Missing: Approval amount
❌ Missing: Interest rate
❌ Missing: Justification

System: "Output invalid. Asking model again..."

Model retries: "You qualify for $10,000 at 5.2% annual rate. Reason: Good credit score."

Validation checks:
✅ Has amount
✅ Has rate
✅ Has justification

✅ Output valid → Show to customer
```

#### Cost Control (Token Limits)

**What is it?**
Budgets that prevent runaway spending.

**Real-world scenario**:
```
Without cost controls:
- Feature goes viral
- 10,000 users suddenly use it
- Each user makes 10 API calls
- 100,000 API calls in 1 hour
- Bill: $10,000 (oops!)

With cost controls:
- Set daily limit: 50,000 API calls
- Divide by users: 50 calls per user per day
- Excess requests queued until next day
- Max daily bill: $500 (controlled)
```

#### Latency Monitoring

**What is it?**
Track response times; alert if slow.

**Why it matters**:
```
Slow response = bad user experience = users leave

Monitor:
- Average response time
- 99th percentile (is 1% of users waiting 30 seconds?)
- Spike alerts (suddenly slow = something wrong)

Example alerts:
- Average response > 2 seconds → Investigate
- Response times increasing → Might need more resources
- 99th percentile > 10 seconds → Fix bad queries
```

---

### Mapping Your Experience

#### LangSmith → Azure

**What you've done in LangSmith**:
- Tracked prompt experiments
- Collected user feedback
- Measured performance

**Azure equivalent**:
```
LangSmith Dataset       → Azure Evaluation Dataset
LangSmith Run          → Azure AI Studio Evaluation Run
User Feedback          → Azure feedback collection
Performance tracking   → Application Insights
```

**Interview answer**:
> "I've used LangSmith to manage prompt experiments and collect feedback. 
> On Azure, I'd replicate this using AI Studio Evaluations, which integrates 
> directly into MLOps pipelines. The core workflow is the same: 
> version → test → measure → iterate → deploy."

#### RAGAS → Azure

**What you've done with RAGAS**:
- Measured RAG system quality
- Caught evaluation regressions
- Tracked metrics over time

**Azure equivalent**:
```
RAGAS Metrics          → Azure Evaluation Metrics
RAGAS Pipeline         → AML Pipeline with evaluation step
Metric tracking        → Application Insights + Monitor
```

**Interview answer**:
> "I've implemented RAGAS to measure RAG quality (faithfulness, relevance, precision).
> On Azure, I'd structure this as an evaluation component in AML Pipelines:
> 1. Retrieve documents
> 2. Generate answer
> 3. Score with RAGAS metrics
> 4. Only deploy if metrics improve
> This ensures quality gates automatically before production."

---

## 4. Azure Infrastructure (Critical for Architect Roles) ☁️

#### What is it?
The compute, storage, and networking services that run your AI applications.

#### Why it matters?
- Interview: "Design a scalable LLM serving system" → Tests infrastructure knowledge
- Production: Wrong infrastructure = slow/expensive
- Your role: You choose the right tool for the job

---

### Compute (Where your code runs)

#### AKS (Azure Kubernetes Service) ⭐⭐⭐

**What is it?**
A system for running containerized applications at scale.

**Simple analogy**:
```
Without Kubernetes:
- You manage 1 server
- Server breaks → app offline
- 100 users need 100 servers? (impossible to manage)

With Kubernetes (AKS):
- Define: "I want 10 copies of my app"
- AKS: Creates 10 copies automatically
- Server breaks? AKS replaces it automatically
- 100 users coming? AKS creates 100 copies automatically
```

**Real-world example** (LLM serving):
```
Your app:
- Serves LLM predictions
- Uses 1 GPU per copy
- Each copy handles 100 concurrent users
- Peak traffic: 10,000 concurrent users

Without AKS:
- Buy 100 GPUs upfront ($100K)
- Manage 100 servers yourself (nightmare)
- 70% of GPUs idle (waste of money)

With AKS:
- Define: "I want 100 copies of LLM server"
- AKS creates 100 copies
- If traffic drops to 5,000 users: AKS removes 50 copies (save money)
- If traffic spikes to 15,000: AKS adds 50 copies (handle load)
```

**Key concepts**:

**GPU Node Pools**
```
Nodes (VMs) in cluster:
- CPU nodes: 100 copies (cheap, for non-intensive work)
- GPU nodes: 20 copies (expensive, for model serving)

Kubernetes scheduler:
- LLM inference → GPU nodes
- Data processing → CPU nodes
- (Automatic routing based on requirements)
```

**Auto-scaling Policies**
```
Rule: If CPU > 80% for 2 minutes → add more nodes
Rule: If CPU < 20% for 5 minutes → remove nodes

Benefit: Always have right amount of resources
```

**Cost Optimization**
```
Spot VMs: Use Azure's unused capacity (60% discount)
Risk: Azure can reclaim anytime
Use case: Batch processing, training (not production serving)

Reserved instances: Commit 1 year upfront (30% discount)
Use case: Production baseline capacity
```

#### Azure Container Apps

**What is it?**
Serverless containers (you don't manage servers).

**Simple comparison**:
```
AKS: You manage servers (more control, more complexity)
Container Apps: Azure manages servers (less control, simpler)

Like:
- AKS = renting apartment (you fix problems)
- Container Apps = hotel (staff handles everything)
```

**When to use**:
```
Use Container Apps if:
- Simple app (stateless, no special needs)
- Variable traffic (don't want to manage scaling)
- Low traffic (don't need full Kubernetes)

Use AKS if:
- Complex workloads (multiple interdependent services)
- Need fine-grained control
- High traffic (cost-effective at scale)
```

#### VMs (GPU Nodes)

**What is it?**
Raw compute (single machine or cluster).

**When to use**:
```
Use VMs if:
- Running batch training jobs
- Simple single-model serving
- Don't need Kubernetes complexity

Don't use if:
- Need high availability (VM breaks = offline)
- Need auto-scaling
```

---

### Storage (Where your data lives)

#### Blob Storage

**What is it?**
Cloud storage for files (documents, images, models, etc.).

**Simple analogy**: AWS S3 equivalent

**Real-world use**:
```
Machine learning pipeline:
1. Raw documents → Blob Storage
2. Processed documents → Blob Storage
3. Models → Blob Storage
4. Logs → Blob Storage

Everything is stored in Blob Storage
```

**Key features**:
- **Lifecycle policies**: Auto-delete old data (save cost)
- **Tiering**: Hot (expensive but fast) → Cool (cheap, slow) → Archive (very cheap, very slow)
- **Example**:
  ```
  Day 1-7: Store training data in "Hot" tier (fast access)
  Day 8-30: Move to "Cool" tier (don't access often, cheaper)
  Day 31+: Move to "Archive" tier (rarely needed, very cheap)
  ```

#### Azure Data Lake (ADLS)

**What is it?**
Storage optimized for big data (like Blob but with hierarchical structure).

**Real-world example**:
```
Folder structure:
/datalake
  /raw
    /2024-01
    /2024-02
  /processed
    /features
    /analytics
  /models
```

**When to use**:
- Large data pipelines (1TB+)
- Data lake for organization
- Integration with Spark/Synapse

#### Cosmos DB (NoSQL Database)

**What is it?**
Database for flexible, unstructured data.

**When to use**:
```
Structured data (relational): Use Azure SQL
Unstructured data (documents): Use Cosmos DB

Example:
- Customer profiles: Cosmos DB (each customer has different fields)
- Transactions: Azure SQL (strictly structured)
```

**Real-world example** (LLM app):
```
Store conversation history:

Conversation 1:
{
  user_id: 123,
  messages: [
    { role: "user", content: "..." },
    { role: "assistant", content: "..." }
  ],
  created_at: "2024-01-15"
}

Conversation 2:
{
  user_id: 456,
  messages: [...],
  created_at: "2024-01-16",
  tags: ["urgent", "finance"]  // Extra field!
}

(Each conversation can have different structure)
```

---

### Networking

#### VNet (Virtual Networks)

**What is it?**
Private network for your Azure resources.

**Simple analogy**: Like creating your own private internet

**Why it matters**:
```
Without VNet:
- All resources publicly accessible
- Anyone on the internet can try to access them
- Security risk

With VNet:
- Private network (only your resources can talk)
- External access only through controlled entry points
- Much more secure
```

#### Private Endpoints (VERY IMPORTANT)

**What is it?**
Access Azure services privately (no public internet).

**Real-world example** (LLM application):
```
Without private endpoints:
Your App → [Internet] → Azure OpenAI
(Your API key travels over internet, exposed to snooping)

With private endpoints:
Your App → [Private VNet] → Azure OpenAI
(Never touches public internet, secure)
```

**Why it matters for compliance**:
```
Bank requirements:
"AI models must be accessed privately (not over internet)"

Solution: Private endpoint
- Connect to Azure OpenAI through VNet
- Meets compliance requirement
```

#### API Gateway (Azure API Management)

**What is it?**
Control point for all API traffic (rate limiting, authentication, logging).

**Real-world example**:
```
Multiple clients call your LLM API:
Client 1 (bad): Makes 10,000 requests in 1 minute
Client 2 (good): Makes 10 requests in 1 minute

Without API Gateway:
- Both hammering your backend
- Backend overloaded, slow for everyone

With API Gateway:
- Rate limit: 100 requests per minute per client
- Client 1: Requests queued after limit
- Client 2: Gets through immediately
- Everyone happy
```

**Key features**:
- Rate limiting (prevent abuse)
- Authentication (who are you?)
- Logging (track API usage)
- Versioning (support multiple API versions)

---

## 5. Identity & Security (VERY IMPORTANT IN AZURE) 🔐

#### What is it?
Controls for: who can access what, when, how.

#### Why it matters?
```
Without proper security:
- Anyone can access customer data
- Regulatory fine: $100M+
- Reputation damage

With proper security:
- Only authorized people access data
- Compliance: ✅
- Reputation: ✅
```

---

### Azure Entra ID (Azure AD)

#### RBAC (Role-Based Access Control)

**What is it?**
Define roles (jobs) and assign permissions (what they can do).

**Simple example**:

```
Roles:
- Data Scientist: Can train models, see training logs
- DevOps Engineer: Can deploy, manage infrastructure
- Manager: Can view reports, not modify data
- Viewer: Can only view dashboards, read-only

Resource: Azure Machine Learning workspace

Permissions:
- Data Scientist → Can use workspace (train jobs, access data)
- DevOps Engineer → Can deploy models
- Manager → Can view metrics (read-only)
- Viewer → Can view dashboards (read-only)
```

**How to assign**:
```
Workspace → Access Control (IAM)
  Add: Sarah
  Role: "Data Scientist"
  Scope: This workspace
  
Result: Sarah can train models in this workspace
        (automatically, no manual setup)
```

**Custom roles**:
```
Built-in roles cover 90% of cases
But sometimes need custom:

Custom role: "ML Engineer"
Permissions:
- Can create training jobs ✅
- Can register models ✅
- Can deploy to staging ✅
- Cannot deploy to production ❌ (requires approval)
- Cannot delete models ❌ (too risky)
```

#### Managed Identity

**What is it?**
Azure automatically manages credentials (no passwords!).

**Problem it solves**:
```
Without managed identity:
- Store password in code/config: RISKY (can leak)
- Store password in environment variable: RISKY (visible in logs)
- Manage password rotation: TEDIOUS (remember when to change?)

With managed identity:
- Azure handles credentials automatically
- They're never exposed in your code
- Azure auto-rotates them
```

**Real-world example**:
```
Your Python code needs to read data from Azure Storage:

Without managed identity:
connection_string = "DefaultEndpointProtocol=https;..."
(Password in code - bad!)

With managed identity:
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
(Automatic, no password in code - good!)
```

#### Service Principals

**What is it?**
Identity for automated processes (not humans).

**Real-world example**:
```
Humans: Alice (alice@company.com)
Machines: GitHub Actions job (needs to deploy model)

Service Principal: "model-deployer"
- Used by GitHub Actions job
- Has permission to deploy to AKS
- Can't be used by humans
```

#### OIDC Integration

**What is it?**
Federated identity (link external identities to Azure).

**Real-world example**:
```
Your company uses GitHub Actions for CI/CD

Without OIDC:
- Store personal token in GitHub secrets
- Problem: If token leaks, attacker can access Azure
- Token needs rotation

With OIDC:
- GitHub → Azure trusts GitHub
- GitHub Actions job gets temporary token (valid 5 minutes)
- No token stored anywhere
- Much more secure
```

---

### Key Vault

**What is it?**
Secure storage for secrets (passwords, API keys, certificates).

**Real-world example**:
```
Your LLM app needs:
- Azure OpenAI API key
- Database password
- Slack webhook URL

Where to store?
❌ In code: Anyone who reads code gets secrets
❌ In config file: File gets checked into Git, exposed
✅ In Key Vault: Encrypted, only authorized people can access

How to use:
1. App runs
2. App asks Key Vault: "Give me the Azure OpenAI API key"
3. Key Vault checks: Is this app authorized?
4. If yes: Returns encrypted key
5. App uses key to call Azure OpenAI
```

**Features**:
- **Encryption**: Secrets encrypted at rest
- **Access logs**: Track who accessed what secret when
- **Auto-rotation**: Automatically rotate passwords (optional)

---

## 6. Data Layer (You WILL be asked) 📊

This section covers where/how to store data for AI applications.

### Azure AI Search (Deep Knowledge)

**Already covered in section 1 (RAG)**

Key points to remember:
- Vector search: Find by meaning
- Hybrid search: Keyword + vector
- Reranking: Improve quality of results

### Cosmos DB vs Azure SQL

**Decision tree**:
```
"Is my data structured (rows & columns)?"
  Yes → Use Azure SQL
  No → Use Cosmos DB

"Do I need complex joins between tables?"
  Yes → Use Azure SQL
  No → Use Cosmos DB

"Do I need global distribution (multiple regions)?"
  Yes → Use Cosmos DB
  No → Azure SQL is fine
```

**Real-world comparison**:

```
Azure SQL:
- Tables: Customers, Orders, Payments
- Structure: Strict (each customer has same fields)
- Use case: Traditional business data

Cosmos DB:
- Documents: Each user profile is different
- Structure: Flexible
- Use case: Content storage, conversation history, user preferences
```

---

## 7. Deployment & Serving 🚀

#### When to use which service?

```
                          Low complexity              High complexity
                         /              \            /              \
                    AKS          Container Apps   AML Endpoints    (Custom)

Best for:
AKS → Production LLM serving (high scale)
AML Endpoints → Managed serving (easy monitoring)
Container Apps → Simple stateless services
```

---

## 8. Observability & Monitoring 📈

### Simple Explanation

**What is it?**
Track: Is my app working? Is it fast? Is it costing too much?

### Azure Monitor

**What is it?**
Centralized place to see all metrics.

**Real-world dashboard**:
```
LLM Service Metrics:
- Requests per minute: 5,000 (healthy)
- Latency (average): 1.2 seconds (good)
- Latency (99th percentile): 3.5 seconds (okay)
- Errors: 0.1% (acceptable)
- Token usage: 10M tokens/day (cost on track)
- GPU utilization: 65% (good balance)
```

### Application Insights

**What is it?**
Logs and traces for your application.

**Real-world use**:
```
Alert: Error rate suddenly jumped to 5%

Investigate:
1. Open Application Insights
2. Filter: errors in last 10 minutes
3. See: All errors are "RAG search timeout"
4. Check: Azure AI Search overloaded
5. Fix: Add more replicas to search service
```

---

## 9. System Design (MOST IMPORTANT) 🔥

### Design Pattern 1: Enterprise RAG System

**Requirements**:
- 100K documents
- 1,000 concurrent users
- Sub-2 second response time
- 99.9% uptime

**Architecture**:

```
Users (Web interface)
  ↓
API Gateway (rate limiting, auth)
  ↓
AKS Cluster (10 pods, auto-scaling)
  ├─ Pod 1: LLM inference
  ├─ Pod 2: LLM inference
  ├─ ...
  └─ Pod 10: LLM inference
  ↓
Azure AI Search (vector + keyword)
  ├─ 100K documents indexed
  ├─ 10 search replicas (scale queries)
  └─ Reranker (improve relevance)
  ↓
Redis Cache (store frequent queries)
  ↓
Azure OpenAI Service (GPT-4)
  ↓
Monitoring (Azure Monitor, Application Insights)
```

**Key Decisions**:

1. **Chunking**: Semantic chunking (respect document boundaries)
2. **Embedding**: text-embedding-3-large (high quality for finance)
3. **Search**: Hybrid (catch both keywords and meaning)
4. **Reranking**: Yes (ensure quality at scale)
5. **Caching**: Redis (same query = instant response)
6. **Deployment**: AKS (handles 1000 concurrent users)

---

## 10. Key Interview Topics & Questions 🎯

### Question 1: Design Enterprise RAG
**"Design a RAG system for a financial services company with 50,000 policy documents."**

**Structure your answer**:
1. **Architecture diagram** (document flow → search → LLM → response)
2. **Key technical decisions** (why semantic chunking, why hybrid search, why reranking)
3. **Scaling strategy** (handle 1000 concurrent users)
4. **Quality assurance** (how do you measure RAG quality?)
5. **Cost optimization** (how to keep costs low?)

---

### Question 2: Cost Optimization
**"How would you reduce Azure OpenAI costs while maintaining quality?"**

**Answer points**:
- Prompt caching: Reuse system prompts (significant savings)
- Model selection: gpt-4o vs gpt-4 (4x cheaper)
- Batch processing: For non-urgent queries
- Fine-tuning: Better quality with fewer tokens
- Request caching: Same query = use cached response

---

### Question 3: Security & Compliance
**"Design a secure GenAI platform that meets regulatory requirements."**

**Answer points**:
- Private endpoints (no public internet)
- Managed Identity (no passwords in code)
- Data encryption (in-transit, at-rest)
- Audit logging (track who did what when)
- PII detection (mask sensitive data)

---

## Quick Reference: Services & Their Purpose

| Service | Purpose | Example |
|---------|---------|---------|
| Azure AI Studio | Build/test/evaluate GenAI apps | Prompt flow, safety filters |
| Azure OpenAI | LLM API access | GPT-4, embeddings |
| Azure AI Search | Vector + keyword search | RAG document retrieval |
| AML | ML model lifecycle | Training, evaluation, deployment |
| AKS | Run applications at scale | LLM serving infrastructure |
| Key Vault | Store secrets securely | API keys, passwords |
| Azure Monitor | Track system health | Metrics, alerts, logs |

---

## Final Tips for Interviews

✅ **Do**:
- Draw architecture diagrams
- Explain trade-offs (why this choice vs that choice?)
- Map your experience (I've done X, similar to Azure's Y)
- Think about scale (what if 10x users?)
- Discuss cost (mention optimization)

❌ **Don't**:
- Say "I don't know" without trying to reason through it
- Forget about monitoring (production needs visibility)
- Ignore security (regulatory requirements matter)
- Choose service just because it exists (use what makes sense)

---

**Last Updated**: 2026-04-18
**Interview Date**: 2026-03-27 (Mastercard AI Managing Consultant)

