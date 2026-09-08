chapter_7_agentic_llm.md
# Resources

| **Type** | **Link** |
|----------|----------|
| **Course** | https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9 |
| **Material** | https://cme295.stanford.edu/syllabus/ |

---

# Chapter 7: Agentic LLM - RAG (Retrieval-Augmented Generation)

## Table of Contents

### 1. Introduction
- [1.1 Introduction to RAG](#introduction-to-rag)
- [1.2 RAG Architecture Overview](#rag-architecture-overview)
- [1.3 Key Takeaways](#key-takeaways)

### 2. RAG Pipeline Components
- [2.1 Document Processing and Embedding](#document-processing-and-embedding)
- [2.2 Query Processing and Retrieval](#query-processing-and-retrieval)
- [2.3 Generation with Retrieved Context](#generation-with-retrieved-context)
- [2.4 RAG System Flow Diagram](#rag-system-flow-diagram)

### 3. Knowledge Base Creation
- [3.1 Document Chunking Strategies](#document-chunking-strategies)
- [3.2 Embeddings and Vector Databases](#embeddings-and-vector-databases)
- [3.3 Key Concepts](#key-concepts-embeddings)

### 4. Retrieval Methods
- [4.1 Candidate Retrieval](#candidate-retrieval)
- [4.2 BM25 (Keyword-Based Search)](#bm25-best-matching-25)
- [4.3 Embedding Search (Semantic Search)](#bm25-vs-embedding-search)
- [4.4 Hybrid Search](#hybrid-search)

### 5. Advanced Techniques
- [5.1 Contextual Chunking](#chunkingcontextchunk)
- [5.2 Prompt Caching](#prompt-caching)

### 6. Re-Ranking
- [6.1 Why Re-Ranking?](#why-re-ranking)
- [6.2 Re-Ranking Architecture](#re-ranking-architecture)
- [6.3 Re-Ranking in RAG Pipeline](#re-ranking-in-rag-pipeline)
- [6.4 Bi-Encoder vs Cross-Encoder](#bi-encoder-vs-cross-encoder-deep-dive)
  - [6.4.1 The Core Problem](#the-core-problem)
  - [6.4.2 Bi-Encoder (Stage 1: Fast Retrieval)](#bi-encoder-stage-1-fast-retrieval)
  - [6.4.3 Cross-Encoder (Stage 2: Precise Re-Ranking)](#cross-encoder-stage-2-precise-re-ranking)
  - [6.4.4 Comparison Table](#comparison-table)
  - [6.4.5 The Two-Stage Workflow](#the-two-stage-workflow)
  - [6.4.6 Real-World Example](#real-world-example)
  - [6.4.7 Technical Architecture](#technical-architecture)
  - [6.4.8 When to Use What?](#when-to-use-what)
  - [6.4.9 Popular Models](#popular-models)
  - [6.4.10 Key Takeaways](#key-takeaways-encoders)

### 7. Performance Evaluation
- [7.1 Quantify Performance of Retrieval](#quantify-performance-of-retrieval)
- [7.2 Metrics Summary](#metrics-summary)

### 8. Tool Calling
- [8.1 Tool Calling Overview](#tool-calling)

---

# 1. Introduction

## Introduction to RAG
RAG is a powerful technique that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval. Instead of relying solely on the knowledge encoded in the model's parameters during training, RAG allows the model to access and use relevant information from external databases or documents in real-time.

## RAG Architecture Overview
<img src="images/introduction_rag_architecture_overview_01.png" alt="RAG Architecture" width="600"/>

**What this shows:** High-level RAG system architecture.

**Key Points:**
- **Problem:** LLMs only know what they were trained on and get outdated
- **Solution:** RAG retrieves current information from external knowledge bases
- **Components:** User Query → Retrieval System → Knowledge Base → LLM → Answer
- **How it works:** Query triggers retrieval of relevant docs, then LLM generates answer using that context

**Purpose:** Shows how information flows through a RAG system.

---

## Key Takeaways

**RAG = Retrieval + Generation:** Combines smart search with LLM text generation

**Three Main Phases:**
- **Index:** Process and store documents as embeddings
- **Retrieve:** Find relevant information for queries
- **Generate:** Create answers using retrieved context

**Benefits:**
- Access to current, specific information
- Reduced hallucinations
- No need to retrain for new info
- Works with private data

**Applications:** Customer support bots, research assistants, legal case search, company knowledge bases

---

# 2. RAG Pipeline Components

## Document Processing and Embedding
<img src="images/rag_pipeline_components_document_processing_and_embedding_01.png" alt="Document Processing" width="600"/>

**What this shows:** How documents are processed into searchable format.

**Key Points:**
- **Chunking:** Large docs split into smaller pieces (200-500 tokens) due to LLM token limits
- **Embeddings:** Each chunk → numerical vector that captures meaning
- **Vector Database:** Stores embeddings for fast similarity search
- **Why it matters:** Enables semantic search (meaning-based) vs keyword matching

**Purpose:** The "indexing" phase before queries can be answered.

---

## Query Processing and Retrieval
<img src="images/rag_pipeline_components_query_processing_and_retrieval_01.png" alt="diagram" width="600"/>

**What this shows:** How user queries are processed to find relevant information.

**Key Points:**
- **Query Embedding:** Your question → same vector format as documents
- **Similarity Search:** Compares query vector with all doc vectors (cosine similarity)
- **Top-k Results:** Returns most similar chunks (e.g., top 3-5)
- **Key insight:** Finds semantically similar content, not just keyword matches

**Purpose:** The "retrieval" phase that finds relevant info for your query.

---

## Generation with Retrieved Context
<img src="images/rag_pipeline_components_generation_with_retrieved_context_01.png" alt="diagram" width="600"/>

**What this shows:** How the LLM generates answers using retrieved context.

**Key Points:**
- **Prompt Construction:** Combines your question + retrieved docs + instructions
- **LLM Processing:** Reads both query and context together
- **Answer Generation:** Response is grounded in retrieved information, reducing hallucinations
- **Why powerful:** Can answer questions about info it was never trained on

**Purpose:** The final "generation" phase producing accurate, context-aware answers.

---

## RAG System Flow Diagram
<img src="images/rag_pipeline_components_rag_system_flow_diagram_01.png" alt="diagram" width="600"/>

**What this shows:** Complete query flow through the RAG pipeline.

**Key Points:**
- Shows sequential process: Query → Embedding → Search → Retrieve → Generate → Answer
- Each component has a specific role in the pipeline
- Demonstrates timing and dependencies between steps

**Purpose:** Bird's-eye view of how all pieces work together in sequence.

---

# 3. Knowledge Base Creation

The retrieval phase finds the most relevant information from your knowledge base. Think of it as an intelligent librarian pulling the right books from shelves.

## Document Chunking Strategies

Before retrieval, you must build a knowledge base (one-time setup, but updatable).

<img src="images/knowledge_base_creation_document_chunking_strategies_01.png" alt="diagram" width="600"/>

**What this shows:** Different strategies for breaking documents into chunks.

**Key Points:**
- **Why chunk:** LLMs have context limits; can't process entire books
- **Strategies:**
  - Fixed-size: Every N tokens (simple but may break sentences)
  - Sentence/paragraph-based: Preserves meaning
  - Semantic: AI identifies natural topic boundaries
- **Trade-offs:** Smaller chunks = precise but less context; Larger = more context but less precise
- **Overlap:** Chunks overlap (e.g., 50 tokens) to preserve boundary information

**Purpose:** Shows that chunking strategy significantly impacts retrieval quality.

---

## Embeddings and Vector Databases
<img src="images/knowledge_base_creation_embeddings_and_vector_databases_01.png" alt="diagram" width="600"/>

**What this shows:** How chunks become embeddings stored in vector databases.

**Key Points:**
- **Process:** Text Chunk → Embedding Model → Vector (e.g., 1536 dimensions) → Vector DB
- **Embedding models:** OpenAI ada-002, Sentence-BERT, etc.
- **Vector DB stores:** Original text + vector + metadata (source, page, date)
- **Why special:** Regular DBs find exact matches; Vector DBs find similar items
- **Popular options:** Pinecone, Weaviate, Chroma, Qdrant, FAISS

**Purpose:** Shows how text becomes searchable data enabling semantic search.

---

## Key Concepts: Embeddings

**Embeddings:** Like coordinates on a map - similar meanings are "close together"
- Example: "king" near "queen", far from "pizza"

**Chunking Best Practices:**
- Preserve context, use overlap, test different sizes

**Vector DB Selection:**
- Consider: scale, speed, features, cost, ease of use

---

# 4. Retrieval Methods

## Candidate Retrieval

**How It Works**

```
**Similarity Search:** Compares query vector with all doc vectors (cosine similarity)
**Top-k Results:** Returns most similar chunks (e.g., top 3-5)
```

**The Search Process**
User query → converted to embedding vector
Comparison → query vector compared with all document chunk vectors in the database
Ranking → chunks ranked by similarity score (cosine similarity)
Selection → top-k most similar chunks returned (typically 3-5)

**Why Not Pre-Grouped?**
While chunks aren't pre-grouped for search, vector databases use optimization techniques:

- `Approximate Nearest Neighbor (ANN) algorithms (like HNSW, IVF) to speed up search without comparing against every single vector`
- These create efficient index structures, but logically the search considers all chunks
- The vector database handles this optimization internally

<img src="images/retrieval_methods_candidate_retrieval_01.png" alt="diagram" width="600"/>

---

## BM25 (Best Matching 25)

<img src="images/retrieval_methods_bm25_best_matching_25_01.png" alt="diagram" width="600"/>

<img src="images/retrieval_methods_bm25_best_matching_25_02.png" alt="diagram" width="600"/>

### BM25 vs Embedding Search

**BM25 (Keyword-Based)**
- **How it works:** Statistical algorithm using term frequency (TF) and inverse document frequency (IDF)
- **Strengths:** Fast, exact keyword matching, works for specific terms/IDs, no training needed
- **Weaknesses:** No semantic understanding, misses synonyms ("car" ≠ "automobile")
- **Best for:** Product catalogs, legal docs, code search, exact term matching

**Embedding Search (Semantic-Based)**
- **How it works:** Neural networks convert text to vectors, finds similar meanings using cosine similarity
- **Strengths:** Understands context and synonyms, finds conceptually related content
- **Weaknesses:** Slower, more storage, may miss exact terms, computationally expensive
- **Best for:** Question answering, customer support, research, natural language queries

**Key Difference:**
- **BM25:** Finds what you **asked for** (exact words)
- **Embeddings:** Finds what you **meant** (semantic meaning)

**Hybrid Approach:**
Modern systems combine both for best results:
```
Final Score = α × BM25_score + (1-α) × Embedding_score
```
- BM25 catches exact matches and rare terms
- Embeddings provide semantic understanding
- Together: precision + recall

---

## Hybrid Search

<img src="images/retrieval_methods_hybrid_search_01.png" alt="diagram" width="600"/>

**What this shows:** Combining BM25 and embedding search for optimal retrieval.

**Key Points:**
- **Dual Retrieval:** Query processed by both BM25 (keyword) and embedding (semantic) systems in parallel
- **Score Fusion:** Results merged using weighted combination: `α × BM25 + (1-α) × Embedding`
- **Best of Both:** Catches exact term matches AND semantically similar content
- **Result:** Higher accuracy - doesn't miss important documents due to different wording

**Purpose:** Shows how hybrid search leverages strengths of both retrieval methods.

---

# 5. Advanced Techniques

## Chunking(Context/chunk)

<img src="images/advanced_techniques_chunkingcontextchunk_01.png" alt="diagram" width="600"/>

**What this shows:** Advanced chunking strategy where LLM adds context to each chunk.

**Key Points:**
- **Problem:** Isolated chunks lack surrounding context (e.g., "it increased by 50%" - what increased?)
- **Solution:** LLM reads each chunk and adds brief contextual summary
- **Enhanced Chunk:** Original text + LLM-generated context = better retrieval accuracy
- **Example:** "Q2 sales: $2M" → "Q2 2024 sales for Product X in North America: $2M"

**Purpose:** Improves chunk quality by making each piece self-contained and understandable.

`LLM will provide context for each chunks`

<img src="images/advanced_techniques_chunkingcontextchunk_02.png" alt="diagram" width="600"/>

**What this shows:** Visual example of how chunks are enriched with context.

**Key Points:**
- **Before:** Raw chunk with unclear references
- **After:** Same chunk with added context (dates, entities, relationships)
- **Process:** LLM analyzes surrounding text and prepends/appends relevant context
- **Benefit:** Retrieving this chunk gives searcher complete information without reading entire document

**Purpose:** Demonstrates the before/after transformation of contextual chunk enrichment.

---

## Prompt Caching

<img src="images/advanced_techniques_prompt_caching_01.png" alt="diagram" width="600"/>

**What this shows:** How prompt caching reduces costs and speeds up repetitive LLM calls.

**Key Points:**
- **Problem:** Processing same context repeatedly (e.g., same document chunks) is expensive
- **Solution:** Cache processed prompts - reuse instead of recomputing
- **How it works:** First call processes full prompt; subsequent calls with same prefix use cache
- **Savings:** Up to 90% cost reduction and 80% latency reduction for cached portions

**Purpose:** Shows optimization technique for RAG systems making many LLM calls with shared context.

---

<img src="images/advanced_techniques_prompt_caching_02.png" alt="diagram" width="600"/>

**What this shows:** Detailed breakdown of prompt caching mechanics and cost savings.

**Key Points:**
- **Cached Prefix:** Static parts (system instructions, document context) cached
- **Dynamic Suffix:** Only user query changes and needs processing
- **Cost Structure:** 
  - Cache write: Regular price (one-time)
  - Cache read: 10% of regular price (repeated use)
- **Use Case:** Perfect for RAG where same documents queried multiple times

**Purpose:** Explains the technical implementation and economic benefits of caching in RAG pipelines.

---

# 6. Re-Ranking

## Why Re-Ranking?

After initial retrieval (embedding or BM25), you have candidate chunks. Re-ranking refines this list to put the MOST relevant chunks at the top before sending to the LLM.

---

## Re-Ranking Architecture

`Encoder represents Transformer Encoder`

<img src="images/reranking_reranking_architecture_01.png" alt="diagram" width="600"/>

**What this shows:** How re-ranking models work to improve initial retrieval results.

**Key Points:**
- **Input:** Query + Document pair fed together into a specialized model
- **Cross-Encoder:** Unlike embeddings (bi-encoder), this processes query and doc jointly for better relevance scoring
- **Output:** Relevance score (0-1) indicating how well document answers query
- **Process:** Takes top-k candidates from retrieval (e.g., 50 chunks) and re-scores them more accurately

**Purpose:** Shows the architecture of re-ranking models that improve upon initial retrieval by deeper analysis.

---

## Re-Ranking in RAG Pipeline

<img src="images/reranking_reranking_in_rag_pipeline_01.png" alt="diagram" width="600"/>

**What this shows:** Where re-ranking fits in the complete RAG workflow.

**Key Points:**
- **Step 1:** Initial retrieval gets 50-100 candidates (fast but approximate)
- **Step 2:** Re-ranker scores each candidate more carefully (slower but accurate)
- **Step 3:** Top 3-5 re-ranked chunks sent to LLM for answer generation
- **Trade-off:** Re-ranking is expensive, so only applied to pre-filtered candidates
- **Result:** Better quality context for LLM = better answers

**Purpose:** Demonstrates the two-stage retrieval strategy: broad retrieval → precise re-ranking → generation.

---

## Key Concepts: Re-Ranking

**Why Two Stages?**
- **Retrieval:** Fast, checks millions of docs, approximate matching
- **Re-Ranking:** Slow, checks top candidates, precise relevance scoring

**Popular Re-Rankers:**
- Cohere Rerank
- Cross-Encoder models (BERT-based)
- LLM-based re-rankers

---

## Bi-Encoder vs Cross-Encoder: Deep Dive

### The Core Problem

In RAG, we need to find the most relevant documents for a user's question. But we face a dilemma:
- **Speed:** Need to search through millions of documents quickly
- **Accuracy:** Need to find the BEST matches, not just good ones

**Solution:** Use BOTH bi-encoders and cross-encoders in a two-stage process!

---

### Bi-Encoder (Stage 1: Fast Retrieval)

**What It Does:** Encodes queries and documents **separately** and **independently**.

**How It Works:**
```
Documents (done once):
Doc 1 → Bi-Encoder → Vector [0.2, 0.8, 0.3, ...] → Store in DB
Doc 2 → Bi-Encoder → Vector [0.5, 0.1, 0.9, ...] → Store in DB
... (millions of docs pre-computed)

Query (done at search time):
"How to fix car engine?" → Bi-Encoder → Vector [0.3, 0.7, 0.5, ...]
Compare with all stored vectors → Get top 50 matches
```

**Restaurant Analogy:** Sorting restaurants by distance - quick to find "restaurants within 5 miles"

**Characteristics:**
- ✅ **Super Fast:** Pre-compute document vectors once
- ✅ **Scalable:** Search millions of documents in milliseconds
- ❌ **Less Accurate:** Doesn't understand query-doc interaction

**Purpose:** Cast a wide net - narrow millions down to top 50-100 candidates

---

### Cross-Encoder (Stage 2: Precise Re-Ranking)

**What It Does:** Encodes query and document **together** as a **single input**.

**How It Works:**
```
Take top 50 candidates from bi-encoder:
Query + Doc 1 → Cross-Encoder → Relevance Score: 0.92
Query + Doc 2 → Cross-Encoder → Relevance Score: 0.45
Query + Doc 3 → Cross-Encoder → Relevance Score: 0.88
Re-sort by these precise scores → Pick top 3-5 for LLM
```

**Restaurant Analogy:** Reading reviews for each restaurant considering YOUR specific needs - slower but much more accurate

**Characteristics:**
- ✅ **Highly Accurate:** Query and doc processed together
- ✅ **Context-Aware:** Understands if doc actually answers the question
- ❌ **Very Slow:** Must process each query+doc pair individually
- ❌ **Not Scalable:** Can't check millions of docs

**Purpose:** Refine results - carefully score 50-100 candidates to find best 3-5

---

### Comparison Table

| Feature | Bi-Encoder | Cross-Encoder |
|---------|------------|---------------|
| **Input** | Query and Doc separately | Query + Doc together |
| **Speed** | Very fast (ms) | Slow (seconds) |
| **Pre-computation** | ✅ Yes | ❌ No |
| **Scalability** | Millions of docs | Hundreds of docs |
| **Accuracy** | Good (80-85%) | Excellent (90-95%) |
| **Use Case** | Initial retrieval | Final re-ranking |

---

### The Two-Stage Workflow

```
User Query: "How do I reset my password?"

STAGE 1 (Bi-Encoder):
→ Search 1 million articles → Get top 100 candidates (50ms)

STAGE 2 (Cross-Encoder):
→ Re-rank 100 candidates → Get top 5 best matches (2 seconds)

STAGE 3 (LLM):
→ Generate answer using top 5 contexts
```

**Why Not Just Cross-Encoder?** Would take hours to score 1 million docs (1M × 0.02s = 5,555 hours!)

**Why Not Just Bi-Encoder?** Less accurate, might miss perfect answer, worse LLM output

**Sweet Spot:** Bi-encoder eliminates 99.99% irrelevant docs (fast) → Cross-encoder finds absolute best from remaining 0.01% (accurate)

---

### Real-World Example

**User Question:** "My laptop won't charge even though it's plugged in"

**Bi-Encoder (50ms):** Searches 100,000 articles → Returns 100 candidates

**Cross-Encoder (2s):** Re-ranks those 100:
1. "Laptop charging issues" - Score: 0.94 ✅
2. "Power adapter troubleshooting" - Score: 0.89 ✅
3. "Battery replacement guide" - Score: 0.82 ✅
4. "Laptop won't turn on" - Score: 0.65 ❌ (similar but not right)

**Result:** LLM gets the 3 most relevant articles and generates accurate solution!

---

### Technical Architecture

**Bi-Encoder:**
```
Query: "fix car" → BERT → [768-dim vector]
Doc: "auto repair guide" → BERT → [768-dim vector]
Similarity = cosine(query_vector, doc_vector)
```

**Cross-Encoder:**
```
Combined: "[CLS] fix car [SEP] auto repair guide [SEP]" → BERT → Score: 0.92
```

**Key Difference:** 
- Bi-encoder: Two separate BERT passes
- Cross-encoder: One BERT pass with both inputs

---

### When to Use What?

**Use Only Bi-Encoder:** Speed critical (< 100ms), millions of docs, limited resources

**Use Both (Recommended):** Production RAG, accuracy matters, can afford 1-3s latency

**Use Only Cross-Encoder:** Small doc set (< 1000), maximum accuracy, speed not important

---

### Popular Models

**Bi-Encoders:** Sentence-BERT (all-MiniLM-L6-v2), OpenAI ada-002, Cohere embed

**Cross-Encoders:** ms-marco-MiniLM-L-12-v2, Cohere Rerank, cross-encoder models

---

## Key Takeaways: Encoders

- **Bi-Encoder:** 🚀 Fast but approximate - narrows millions to hundreds
- **Cross-Encoder:** 🎯 Slow but precise - refines hundreds to top 3-5
- **Together:** Speed + Accuracy = Best practice for production RAG
- **Remember:** Bi-encoder casts the net, cross-encoder picks the best fish! 🎣

---

# 7. Performance Evaluation

## Quantify Performance of Retrieval

How do we measure if our retrieval system is working well? We need metrics to evaluate and improve performance.

<img src="images/performance_evaluation_quantify_performance_of_retrieval_01.png" alt="diagram" width="600"/>

<img src="images/performance_evaluation_quantify_performance_of_retrieval_02.png" alt="diagram" width="600"/>

<img src="images/performance_evaluation_quantify_performance_of_retrieval_03.png" alt="diagram" width="600"/>

<img src="images/performance_evaluation_quantify_performance_of_retrieval_04.png" alt="diagram" width="600"/>

<img src="images/performance_evaluation_quantify_performance_of_retrieval_05.png" alt="diagram" width="600"/>

---

## Metrics Summary

| Metric | What It Measures | Best For |
|--------|------------------|----------|
| **Precision** | Quality of returned results | When false positives are costly |
| **Recall** | Coverage of all relevant docs | When missing results is costly |
| **MRR** | Position of first relevant result | When users want quick answers |
| **MAP** | Overall ranking quality | Balanced retrieval performance |
| **NDCG** | Graded relevance ranking | Real-world nuanced relevance |
| **Hit@k** | Any relevant result in top-k | Simple pass/fail evaluation |

**Key Insight:** Use multiple metrics together - no single metric tells the whole story!

**Practical Tip:** 
- Start with Hit@k and MRR (easy to understand)
- Use MAP for balanced evaluation
- Use NDCG when you have graded relevance labels

---

# 8. Tool Calling

## What is Tool Calling?

Tool calling allows LLMs to interact with external functions, APIs, and systems to perform actions beyond text generation. Instead of just answering with text, the LLM can call tools to get real-time data, perform calculations, or execute tasks.

---

<img src="images/tool_calling_what_is_tool_calling_01.png" alt="diagram" width="600"/>

**What this shows:** Basic tool calling workflow.

**Key Points:**
- User asks a question requiring external data (e.g., "What's the weather?")
- LLM recognizes it needs a tool and generates a structured function call
- External system executes the tool and returns results
- LLM uses the result to formulate a natural language response

**Purpose:** Demonstrates how LLMs bridge the gap between language understanding and real-world actions.

---

<img src="images/tool_calling_what_is_tool_calling_02.png" alt="diagram" width="600"/>

**What this shows:** The tool calling process flow with decision points.

**Key Points:**
- LLM first analyzes if the query needs external tools or can be answered directly
- Decision tree: Can I answer this myself? → Yes: respond directly, No: call tool
- After tool execution, LLM synthesizes the result into a user-friendly answer
- Multiple tools can be called in sequence if needed

**Purpose:** Shows the decision-making logic behind when and how to use tools.

---

<img src="images/tool_calling_what_is_tool_calling_03.png" alt="diagram" width="600"/>

**What this shows:** Anatomy of a tool definition/schema.

**Key Points:**
- Tools are defined with: name, description, parameters, and expected output format
- **Function signature:** Specifies what inputs the tool needs (e.g., location for weather)
- **Type definitions:** Each parameter has a data type (string, number, etc.)
- LLM uses these schemas to understand which tool to call and how

**Purpose:** Explains how tools are formally defined so LLMs can use them correctly.

---

<img src="images/tool_calling_what_is_tool_calling_04.png" alt="diagram" width="600"/>

**What this shows:** Example of LLM generating a structured tool call request.

**Key Points:**
- Input: User query in natural language
- Output: Structured JSON with function name and arguments
- Example: `{"function": "get_weather", "arguments": {"city": "London"}}`
- This JSON is machine-readable and can be executed by external systems

**Purpose:** Shows the transformation from human language to executable code.

---

<img src="images/tool_calling_what_is_tool_calling_05.png" alt="diagram" width="600"/>

**What this shows:** Multi-step tool calling scenario.

**Key Points:**
- Complex queries may require multiple tool calls in sequence
- Example: "Book a flight to Paris" → Check availability → Get prices → Make booking
- LLM orchestrates the sequence, using output from one tool as input to the next
- Maintains context across multiple steps

**Purpose:** Demonstrates how LLMs can handle complex, multi-step workflows.

---

<img src="images/tool_calling_what_is_tool_calling_06.png" alt="diagram" width="600"/>

**What this shows:** Tool call result handling and error management.

**Key Points:**
- Tool execution can succeed or fail
- LLM must handle various scenarios: success, partial data, errors, timeouts
- Error handling: LLM can retry, ask for clarification, or use alternative tools
- Result validation: Check if the returned data makes sense

**Purpose:** Shows that tool calling is robust and handles real-world complications.

---

<img src="images/tool_calling_what_is_tool_calling_07.png" alt="diagram" width="600"/>

**What this shows:** Types of tools available to LLMs.

**Key Points:**
- **Data retrieval:** APIs for weather, news, databases, search engines
- **Computation:** Calculator, code interpreter, data analysis tools
- **Actions:** Send email, book appointments, update records
- **Custom functions:** Company-specific or domain-specific tools

**Purpose:** Overview of the diverse capabilities tools provide to LLMs.

---

## Tool Calling via Training

Training-based approach: The LLM is fine-tuned specifically to generate tool calls during its training phase.

<img src="images/tool_calling_tool_calling_via_training_01.png" alt="diagram" width="600"/>

**What this shows:** How LLMs are trained to use tools through fine-tuning.

**Key Points:**
- Training data includes examples of queries paired with correct tool calls
- Model learns patterns: "weather question" → call `get_weather()` function
- **Advantage:** More reliable, faster, built-in understanding
- **Used by:** GPT-4, Claude, Gemini (have native tool calling capabilities)

**Purpose:** Explains the training approach where tool-calling ability is baked into the model.

---

## Tool Calling via Prompt

Prompt-based approach: Tool definitions and usage instructions are provided in the system prompt.

<img src="images/tool_calling_tool_calling_via_prompt_01.png" alt="diagram" width="600"/>

**What this shows:** Using prompts to teach an LLM how to call tools.

**Key Points:**
- System prompt includes: list of available tools, their descriptions, usage examples
- LLM learns tool calling from instructions in the prompt, not from training
- **Advantage:** Works with any LLM, no special training needed
- **Disadvantage:** Less reliable, takes up context space, may make mistakes

**Purpose:** Shows the alternative approach using prompt engineering for tool calling.

---

<img src="images/tool_calling_tool_calling_via_prompt_02.png" alt="diagram" width="600"/>

**What this shows:** Example of a detailed tool specification in a prompt.

**Key Points:**
- Prompt contains full JSON schema for each tool
- Clear instructions on when and how to use each tool
- Examples of correct tool call format
- The more detailed the prompt, the better the LLM performs

**Purpose:** Demonstrates what a prompt-based tool definition looks like in practice.

---

`Too many tool results in conflicting which LLM can't determine correct tool always`

<img src="images/tool_calling_tool_calling_via_prompt_03.png" alt="diagram" width="600"/>

**What this shows:** The challenge of tool selection when many tools are available.

**Key Points:**
- **Problem:** With 50+ tools, LLM struggles to pick the right one
- Similar-sounding tools cause confusion (e.g., `search_web` vs `search_docs`)
- More tools = higher chance of errors, slower decision-making
- Context window gets crowded with tool definitions

**Purpose:** Highlights a key limitation—too many tools can hurt performance.

---

## Tool Summary

<img src="images/tool_calling_tool_summary_01.png" alt="diagram" width="600"/>

**What this shows:** Overview comparing training-based vs prompt-based tool calling.

**Key Points:**
- **Training-based:** Native support, reliable, fast, but requires special models
- **Prompt-based:** Universal, flexible, but less accurate and uses context space
- **Best practice:** Use training-based (GPT-4, Claude) when available
- **Fallback:** Use prompt-based for models without native tool support

**Purpose:** Summarizes the two approaches and when to use each.

---

## Tool Selection

Strategies for helping LLMs choose the right tool from a large set.

<img src="images/tool_calling_tool_selection_01.png" alt="diagram" width="600"/>

**What this shows:** Tool selection strategies to manage large tool sets.

**Key Points:**
- **Categorization:** Group tools by domain (finance, weather, database, etc.)
- **Two-stage selection:** First pick category, then specific tool within that category
- **Tool indexing:** Use embeddings to find semantically similar tools to the query
- **Reduces confusion:** Narrows down choices, improving accuracy

**Purpose:** Shows techniques to solve the "too many tools" problem.

---

<img src="images/tool_calling_tool_selection_02.png" alt="diagram" width="600"/>

**What this shows:** The two-stage tool selection process in detail.

**Key Points:**
- **Stage 1:** LLM identifies the domain/category (e.g., "This is a weather question")
- **Stage 2:** LLM chooses from only the tools in that category
- Example: Query → Identify "Weather" category → Choose `get_weather()` from 5 weather tools
- **Benefit:** Instead of choosing from 100 tools, choose from 5-10 relevant ones

**Purpose:** Demonstrates how hierarchical selection improves tool-calling accuracy.

---

# MCP (Model Context Protocol)

MCP is a standardized protocol for connecting LLMs to external data sources and tools, created by Anthropic.

<img src="images/tool_calling_tool_selection_03.png" alt="diagram" width="600"/>

**What this shows:** The MCP architecture and its role in tool calling.

**Key Points:**
- **Standardized interface:** Uniform way for LLMs to connect to any tool or data source
- **Components:** LLM ↔ MCP Server ↔ External Systems (databases, APIs, files)
- **Benefits:** One protocol for all integrations, no custom code for each tool
- Like USB for LLMs—one standard connection for everything

**Purpose:** Introduces MCP as a universal standard for LLM-tool integration.

---

<img src="images/tool_calling_tool_selection_04.png" alt="diagram" width="600"/>

**What this shows:** How MCP servers work as intermediaries.

**Key Points:**
- **MCP Server:** Acts as a bridge between LLM and external resources
- Handles authentication, data formatting, error handling
- Multiple MCP servers can run simultaneously for different resources
- LLM just needs to know the MCP protocol, not specific APIs

**Purpose:** Explains the server architecture that enables standardized tool access.

---

<img src="images/tool_calling_tool_selection_05.png" alt="diagram" width="600"/>

**What this shows:** Example MCP implementation connecting to various resources.

**Key Points:**
- **Use cases:** Connect to filesystems, databases, cloud services, local apps
- **Example flow:** LLM → MCP request → Server fetches data → Returns formatted result
- Developers can build MCP servers for any resource type
- Growing ecosystem of pre-built MCP servers

**Purpose:** Shows practical applications and flexibility of the MCP standard.

---

# Agents

Agents are autonomous LLM systems that can plan, use tools, and take actions to achieve goals.

<img src="images/tool_calling_tool_selection_06.png" alt="diagram" width="600"/>

**What this shows:** The concept of an AI agent vs a basic LLM.

**Key Points:**
- **Basic LLM:** Input → Output (one-shot response)
- **Agent:** Can plan, execute multiple steps, use tools, adapt based on results
- **Autonomous:** Can make decisions about next actions without constant user input
- **Goal-oriented:** Works toward completing a task, not just answering a question

**Purpose:** Distinguishes between simple LLM responses and agentic behavior.

---

<img src="images/tool_calling_tool_selection_07.png" alt="diagram" width="600"/>

**What this shows:** Core components of an AI agent system.

**Key Points:**
- **Planning module:** Breaks down complex tasks into steps
- **Memory:** Stores conversation history and intermediate results
- **Tool use:** Can call external functions and APIs
- **Reasoning:** Evaluates results and decides next actions
- All components work together in a loop until goal is achieved

**Purpose:** Shows the architecture that enables autonomous agent behavior.

---

## ReAct (Reason + Action)

ReAct is a popular agent framework that alternates between reasoning (thinking) and acting (tool use).

<img src="images/tool_calling_react_reason__action_01.png" alt="diagram" width="600"/>

**What this shows:** The ReAct framework's core loop.

**Key Points:**
- **Thought:** LLM reasons about what to do next
- **Action:** LLM calls a tool or takes an action
- **Observation:** System returns the result
- **Loop:** Repeat Thought → Action → Observation until goal achieved

**Purpose:** Introduces the foundational pattern for most modern agents.

---

<img src="images/tool_calling_react_reason__action_02.png" alt="diagram" width="600"/>

**What this shows:** ReAct prompt structure and format.

**Key Points:**
- System prompt defines the Thought/Action/Observation format
- LLM explicitly writes out its reasoning in "Thought" sections
- Actions are structured tool calls
- Observations are factual results from tools

**Purpose:** Shows the actual prompt template that implements ReAct.

---

<img src="images/tool_calling_react_reason__action_03.png" alt="diagram" width="600"/>

**What this shows:** A concrete ReAct example solving a real problem.

**Key Points:**
- **Question:** "What's the capital of the country where the Eiffel Tower is located?"
- **Thought 1:** "I need to find which country has the Eiffel Tower"
- **Action 1:** Search("Eiffel Tower location")
- **Observation 1:** "France"
- **Thought 2:** "Now I need the capital of France"
- **Action 2:** Search("capital of France")
- **Observation 2:** "Paris"
- **Final Answer:** "Paris"

**Purpose:** Demonstrates the step-by-step reasoning process in action.

---

<img src="images/tool_calling_react_reason__action_04.png" alt="diagram" width="600"/>

**What this shows:** How ReAct handles errors and adapts.

**Key Points:**
- Agent can recover from failed tool calls
- If one approach fails, agent reasons about alternatives
- Example: Search fails → Try different keywords → Or use different tool
- Self-correction is a key advantage

**Purpose:** Shows ReAct's robustness and ability to handle real-world complications.

---

<img src="images/tool_calling_react_reason__action_05.png" alt="diagram" width="600"/>

**What this shows:** ReAct with multiple tools available.

**Key Points:**
- Agent has access to: Search, Calculator, Database Query, API calls
- For each step, decides which tool is most appropriate
- Can chain different tools together
- Example: Search for data → Calculator to analyze → Database to store

**Purpose:** Demonstrates ReAct's flexibility with diverse tool sets.

---

<img src="images/tool_calling_react_reason__action_06.png" alt="diagram" width="600"/>

**What this shows:** ReAct performance metrics and evaluation.

**Key Points:**
- Success rate: How often agent completes tasks correctly
- Efficiency: Number of steps needed
- Tool selection accuracy: Choosing the right tool
- Comparison: ReAct vs Chain-of-Thought vs Direct prompting

**Purpose:** Provides empirical evidence of ReAct's effectiveness.

---

<img src="images/tool_calling_react_reason__action_07.png" alt="diagram" width="600"/>

**What this shows:** Limitations and failure modes of ReAct.

**Key Points:**
- **Infinite loops:** Agent gets stuck repeating same actions
- **Hallucinated reasoning:** Makes up logical-sounding but wrong steps
- **Tool dependency:** Fails if required tool is unavailable
- **Token limits:** Long reasoning chains can exceed context window

**Purpose:** Realistic view of challenges when deploying ReAct agents.

---

<img src="images/tool_calling_react_reason__action_08.png" alt="diagram" width="600"/>

**What this shows:** Optimizations and improvements to ReAct.

**Key Points:**
- **Reflection:** Agent reviews its own reasoning for errors
- **Self-consistency:** Run multiple reasoning paths, pick most common answer
- **Few-shot examples:** Provide example reasoning traces in prompt
- **Max iterations:** Set limit to prevent infinite loops

**Purpose:** Shows techniques to make ReAct more reliable in production.

---

<img src="images/tool_calling_react_reason__action_09.png" alt="diagram" width="600"/>

**What this shows:** ReAct agent architecture diagram.

**Key Points:**
- User query enters the system
- ReAct controller manages the Thought/Action/Observation loop
- Tool executor runs the actual function calls
- Memory stores context and intermediate results
- Response generator formats final answer

**Purpose:** System-level view of how ReAct is implemented.

---

<img src="images/tool_calling_react_reason__action_10.png" alt="diagram" width="600"/>

**What this shows:** ReAct vs other agent frameworks comparison.

**Key Points:**
- **ReAct:** Alternating thought and action
- **Plan-and-Execute:** Plan all steps first, then execute
- **Reflexion:** ReAct + self-evaluation and refinement
- Each has trade-offs in reliability, speed, and complexity

**Purpose:** Positions ReAct within the broader landscape of agent architectures.

---

<img src="images/tool_calling_react_reason__action_11.png" alt="diagram" width="600"/>

**What this shows:** Real-world ReAct use cases.

**Key Points:**
- **Customer support:** Research issue → Query knowledge base → Generate solution
- **Data analysis:** Load data → Clean → Analyze → Visualize → Report
- **Research assistant:** Search papers → Summarize → Compare → Synthesize
- **Task automation:** Understand request → Plan steps → Execute → Verify

**Purpose:** Concrete examples of where ReAct agents provide value.

---

<img src="images/tool_calling_react_reason__action_12.png" alt="diagram" width="600"/>

**What this shows:** Building blocks for implementing ReAct.

**Key Points:**
- **LLM:** GPT-4, Claude, or similar with good reasoning
- **Tool library:** Collection of callable functions
- **Orchestration layer:** Manages the loop and state
- **Prompt templates:** Define the Thought/Action/Observation format
- **Frameworks:** LangChain, LlamaIndex, AutoGPT provide pre-built components

**Purpose:** Practical guide to the technology stack needed.

---

<img src="images/tool_calling_react_reason__action_13.png" alt="diagram" width="600"/>

**What this shows:** ReAct prompt engineering best practices.

**Key Points:**
- Clear tool descriptions with examples
- Explicit format requirements for Thought/Action/Observation
- Guidelines for when to stop (e.g., "When you have final answer, return it")
- Examples of good reasoning traces
- Error handling instructions

**Purpose:** Shows how prompt quality directly impacts agent performance.

---

<img src="images/tool_calling_react_reason__action_14.png" alt="diagram" width="600"/>

**What this shows:** Future directions and advanced ReAct variants.

**Key Points:**
- **Multi-agent ReAct:** Multiple agents collaborating
- **Hierarchical ReAct:** Agents with sub-agents for complex tasks
- **Learned planning:** Training models specifically for better planning
- **Tool learning:** Agents that learn to use new tools from examples

**Purpose:** Preview of ongoing research and improvements to agent systems.

---

# A2A (Agent-to-Agent Communication)

A2A enables multiple AI agents to communicate and collaborate to solve complex tasks.

<img src="images/tool_calling_react_reason__action_15.png" alt="diagram" width="600"/>

**What this shows:** Multi-agent systems where agents collaborate.

**Key Points:**
- **Multiple specialized agents:** Each agent has specific expertise (e.g., researcher, coder, writer)
- **Communication protocol:** Agents share information and coordinate actions
- **Divide and conquer:** Complex tasks split among agents based on their strengths
- **Example:** Research agent finds info → Analysis agent processes → Writer agent summarizes

**Purpose:** Introduces the concept of agent collaboration for complex problem-solving.

---

<img src="images/tool_calling_react_reason__action_16.png" alt="diagram" width="600"/>

**What this shows:** A2A communication patterns and architectures.

**Key Points:**
- **Sequential:** Agents work in pipeline (output of one → input to next)
- **Parallel:** Multiple agents work simultaneously, results aggregated
- **Hierarchical:** Manager agent coordinates worker agents
- **Collaborative:** Agents negotiate and debate to reach consensus
- **Benefits:** Specialization, parallelization, robustness through redundancy

**Purpose:** Shows different ways agents can be organized and communicate to achieve goals.

---

# Safety

![alt text](images/safety_1.png)

![alt text](images/safety_2.png)
