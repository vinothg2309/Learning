RAG_Playbook.md
# RAG Playbook

## Table of Contents
- [RAG Playbook](#rag-playbook)
  - [Table of Contents](#table-of-contents)
- [Chunk Metadata — Schema, Strategy \& Effective Filtering](#chunk-metadata--schema-strategy--effective-filtering)
  - [Where Is Metadata Saved? Is It in VectorDB?](#where-is-metadata-saved-is-it-in-vectordb)
  - [Why Metadata Matters in Production RAG](#why-metadata-matters-in-production-rag)
  - [Top Metadata Fields to Include](#top-metadata-fields-to-include)
    - [Tier 1 — Must Have (zero cost, always available)](#tier-1--must-have-zero-cost-always-available)
    - [Tier 2 — High Value (cheap, structured)](#tier-2--high-value-cheap-structured)
    - [Tier 3 — Add When Needed (LLM cost)](#tier-3--add-when-needed-llm-cost)
    - [Minimal starting payload](#minimal-starting-payload)
  - [Complete Metadata Schema](#complete-metadata-schema)
  - [1. Identity \& Provenance](#1-identity--provenance)
  - [2. ACL — Access Control](#2-acl--access-control)
  - [3. Document Structure](#3-document-structure)
  - [4. Semantic Enrichment — Keywords](#4-semantic-enrichment--keywords)
  - [5. Semantic Enrichment — Summary](#5-semantic-enrichment--summary)
  - [6. Hypothetical Query Generation](#6-hypothetical-query-generation)
  - [7. Temporal \& Lifecycle](#7-temporal--lifecycle)
  - [8. Content Quality Signals](#8-content-quality-signals)
  - [9. Retrieval Tuning Signals](#9-retrieval-tuning-signals)
  - [10. Effective Metadata Filtering at Retrieval](#10-effective-metadata-filtering-at-retrieval)
    - [Filter-First Architecture (Pre-filter before ANN)](#filter-first-architecture-pre-filter-before-ann)
    - [Filter Decision Matrix](#filter-decision-matrix)
  - [11. Full Ingestion Pipeline](#11-full-ingestion-pipeline)
  - [Metadata Field Reference Table](#metadata-field-reference-table)
- [Document Management in VectorDB](#document-management-in-vectordb)
  - [1. Check if Content Already Exists](#1-check-if-content-already-exists)
    - [Problem](#problem)
    - [Solution: Content-Based Hashing](#solution-content-based-hashing)
    - [Alternative: Query by Metadata](#alternative-query-by-metadata)
  - [2. Upsert Operations](#2-upsert-operations)
    - [What is Upsert?](#what-is-upsert)
    - [ChromaDB Upsert](#chromadb-upsert)
    - [Pinecone Upsert](#pinecone-upsert)
    - [FAISS (No Native Upsert)](#faiss-no-native-upsert)
    - [Comparison](#comparison)
  - [3. Prevent Duplicate Ingestion](#3-prevent-duplicate-ingestion)
    - [Strategy 1: Content-Based Deduplication](#strategy-1-content-based-deduplication)
    - [Strategy 2: Metadata-Based Tracking](#strategy-2-metadata-based-tracking)
    - [Strategy 3: Batch Deduplication with Similarity](#strategy-3-batch-deduplication-with-similarity)
  - [4. Document Version Management](#4-document-version-management)
    - [Problem](#problem-1)
    - [Strategy 1: Replace Old Version (Simple)](#strategy-1-replace-old-version-simple)
    - [Strategy 2: Maintain Version History](#strategy-2-maintain-version-history)
    - [Strategy 3: Soft Delete with Versioning](#strategy-3-soft-delete-with-versioning)
    - [Best Practices](#best-practices)
    - [Document Management Workflow](#document-management-workflow)
- [RAG Architectures](#rag-architectures)
  - [1. Naive RAG](#1-naive-rag)
    - [Definition](#definition)
    - [How It Works](#how-it-works)
    - [Example](#example)
  - [2. Self-RAG (Self-Reflective RAG)](#2-self-rag-self-reflective-rag)
    - [Definition](#definition-1)
    - [How It Works](#how-it-works-1)
  - [3. Corrective RAG (CRAG)](#3-corrective-rag-crag)
    - [Definition](#definition-2)
    - [How It Works](#how-it-works-2)
  - [4. Agentic RAG](#4-agentic-rag)
    - [Definition](#definition-3)
    - [How It Works](#how-it-works-3)
    - [Example](#example-1)
  - [5. Graph RAG](#5-graph-rag)
    - [Definition](#definition-4)
    - [How It Works](#how-it-works-4)
    - [Example](#example-2)
  - [6. Multi-Modal RAG](#6-multi-modal-rag)
    - [Definition](#definition-5)
    - [How It Works](#how-it-works-5)
    - [Example](#example-3)
  - [7. Adaptive RAG](#7-adaptive-rag)
    - [Definition](#definition-6)
    - [How It Works](#how-it-works-6)
    - [Example](#example-4)
  - [RAG Architectures Comparison](#rag-architectures-comparison)
    - [When to Use Which](#when-to-use-which)
- [Chunking Strategies](#chunking-strategies)
  - [1. Fixed-Size Chunking](#1-fixed-size-chunking)
    - [Definition](#definition-7)
    - [How It Works](#how-it-works-7)
    - [Example](#example-5)
  - [2. Recursive Character Splitting](#2-recursive-character-splitting)
    - [Definition](#definition-8)
    - [How It Works](#how-it-works-8)
    - [Example](#example-6)
  - [3. Document-Specific Chunking](#3-document-specific-chunking)
    - [Definition](#definition-9)
    - [How It Works](#how-it-works-9)
    - [Example](#example-7)
  - [4. Semantic Chunking](#4-semantic-chunking)
    - [Definition](#definition-10)
    - [How It Works](#how-it-works-10)
    - [Example](#example-8)
  - [Semantic Chunking for Large PDFs (1000+ pages)](#semantic-chunking-for-large-pdfs-1000-pages)
    - [Problem](#problem-2)
    - [Solution: Hybrid 2-Stage Approach](#solution-hybrid-2-stage-approach)
    - [Code Example](#code-example)
  - [When to Use What](#when-to-use-what)
  - [5. Agentic Chunking](#5-agentic-chunking)
    - [Definition](#definition-11)
    - [How It Works](#how-it-works-11)
    - [Example](#example-9)
  - [6. Hierarchical Chunking (Large Documents)](#6-hierarchical-chunking-large-documents)
    - [Definition](#definition-12)
    - [How It Works](#how-it-works-12)
    - [Example](#example-10)
  - [ParentDocumentRetriever — Injection \& Retrieval](#parentdocumentretriever--injection--retrieval)
    - [Part 1 — Injection Pipeline (Run Once / On Schedule)](#part-1--injection-pipeline-run-once--on-schedule)
    - [Part 2 — Retrieval Pipeline (Run Per Query)](#part-2--retrieval-pipeline-run-per-query)
    - [What `ParentDocumentRetriever` Does in Each Part](#what-parentdocumentretriever-does-in-each-part)
    - [Production — Two Services, Shared Stores](#production--two-services-shared-stores)
    - [Production Docstore by Scale](#production-docstore-by-scale)
  - [7. Tabular Data Chunking](#7-tabular-data-chunking)
    - [The Problem with Tables](#the-problem-with-tables)
    - [Tools and Techniques for Table Extraction](#tools-and-techniques-for-table-extraction)
    - [How Each Tool Keeps the Table as One Chunk](#how-each-tool-keeps-the-table-as-one-chunk)
    - [Which Tool to Pick](#which-tool-to-pick)
  - [8. Image Chunking (Multi-modal Documents)](#8-image-chunking-multi-modal-documents)
    - [The Problem](#the-problem)
    - [Strategy 1 — Extract + Caption with Vision LLM](#strategy-1--extract--caption-with-vision-llm)
    - [Strategy 2 — Multi-modal Embeddings (CLIP / ColPali)](#strategy-2--multi-modal-embeddings-clip--colpali)
    - [Strategy 3 — Anchor Image to Surrounding Text](#strategy-3--anchor-image-to-surrounding-text)
    - [What to Store for Image Chunks](#what-to-store-for-image-chunks)
    - [Full Pipeline: PDF with Mixed Content](#full-pipeline-pdf-with-mixed-content)
    - [Decision Matrix for Images](#decision-matrix-for-images)
  - [9. Chunk Size Impact on Context Precision \& Recall](#9-chunk-size-impact-on-context-precision--recall)
    - [Chunk Too Small](#chunk-too-small)
    - [Chunk Too Large](#chunk-too-large)
    - [The Tradeoff](#the-tradeoff)
    - [Production Fix — Parent-Child Chunking](#production-fix--parent-child-chunking)
    - [Optimal Chunk Size (Rule of Thumb)](#optimal-chunk-size-rule-of-thumb)
  - [Chunking Strategies Comparison](#chunking-strategies-comparison)
    - [Choosing a Strategy](#choosing-a-strategy)
- [Retrieval Search Methods](#retrieval-search-methods)
  - [1. Cosine Similarity](#1-cosine-similarity)
    - [Definition](#definition-13)
    - [How It Works](#how-it-works-13)
    - [Example](#example-11)
  - [2. Euclidean Distance (L2)](#2-euclidean-distance-l2)
    - [Definition](#definition-14)
    - [How It Works](#how-it-works-14)
    - [Example](#example-12)
  - [3. Maximum Marginal Relevance (MMR)](#3-maximum-marginal-relevance-mmr)
    - [Definition](#definition-15)
    - [How It Works](#how-it-works-15)
    - [Example](#example-13)
  - [4. Similarity Score Threshold](#4-similarity-score-threshold)
    - [Definition](#definition-16)
    - [How It Works](#how-it-works-16)
    - [Example](#example-14)
  - [5. BM25 (Keyword-Based)](#5-bm25-keyword-based)
    - [Definition](#definition-17)
    - [How It Works](#how-it-works-17)
    - [Code Example](#code-example-1)
  - [6. Hybrid Search](#6-hybrid-search)
    - [Definition](#definition-18)
    - [What You Store Per Chunk](#what-you-store-per-chunk)
    - [What is an Inverted Index?](#what-is-an-inverted-index)
    - [Where to Store Chunks](#where-to-store-chunks)
    - [How Weaviate Stores Both (Vector + Keyword)](#how-weaviate-stores-both-vector--keyword)
    - [How Hybrid Search Works](#how-hybrid-search-works)
    - [Why Keyword Search Is Critical in Production](#why-keyword-search-is-critical-in-production)
    - [Mental Model](#mental-model)
    - [Example (LangChain EnsembleRetriever)](#example-langchain-ensembleretriever)
  - [Retrieval Methods Comparison](#retrieval-methods-comparison)
- [ContextualCompressionRetriever](#contextualcompressionretriever)
  - [What is ContextualCompressionRetriever?](#what-is-contextualcompressionretriever)
  - [Why Compress Retrieved Documents?](#why-compress-retrieved-documents)
  - [How It Works](#how-it-works-18)
    - [Architecture Flow:](#architecture-flow)
  - [Compression Methods](#compression-methods)
    - [1. LLMChainExtractor (Most Common)](#1-llmchainextractor-most-common)
    - [2. EmbeddingsFilter (Semantic-Based)](#2-embeddingsfilter-semantic-based)
    - [3. DocumentCompressorPipeline (Chained Compression)](#3-documentcompressorpipeline-chained-compression)
  - [Compression in Practice](#compression-in-practice)
    - [Before Compression](#before-compression)
    - [After LLMChainExtractor Compression](#after-llmchainextractor-compression)
  - [Complete Example](#complete-example)
    - [Full RAG Pipeline with Compression](#full-rag-pipeline-with-compression)
  - [When to Use Compression](#when-to-use-compression)
    - [✅ Use Compression When:](#-use-compression-when)
    - [Compression Method Comparison](#compression-method-comparison)
  - [Key Takeaways](#key-takeaways)
- [Vector Database Indexing](#vector-database-indexing)
  - [What is Vector Indexing?](#what-is-vector-indexing)
    - [Saving Embeddings vs Pre-Indexed](#saving-embeddings-vs-pre-indexed)
    - [When Pre-Indexing Is NOT Done](#when-pre-indexing-is-not-done)
  - [HNSW (Hierarchical Navigable Small World)](#hnsw-hierarchical-navigable-small-world)
    - [Definition](#definition-19)
    - [How It Works](#how-it-works-19)
    - [Performance](#performance)
    - [When to Use](#when-to-use)
    - [HNSW in Code](#hnsw-in-code)
  - [IVF (Inverted File Index)](#ivf-inverted-file-index)
    - [Definition](#definition-20)
    - [How It Works](#how-it-works-20)
    - [Performance](#performance-1)
    - [When to Use](#when-to-use-1)
    - [IVF in Code](#ivf-in-code)
  - [HNSW vs IVF Comparison](#hnsw-vs-ivf-comparison)
  - [Quick Reference](#quick-reference)
    - [Library Support](#library-support)
    - [Configuration Cheat Sheet](#configuration-cheat-sheet)
- [HyDE (Hypothetical Document Embeddings)](#hyde-hypothetical-document-embeddings)
  - [What is HyDE?](#what-is-hyde)
  - [How It Works](#how-it-works-21)
    - [Traditional RAG:](#traditional-rag)
    - [HyDE RAG:](#hyde-rag)
  - [When to Use HyDE](#when-to-use-hyde)
- [RAG Document Versioning Fingerprint](#rag-document-versioning-fingerprint)
  - [Problem Overview](#problem-overview)
  - [How Fingerprinting Works](#how-fingerprinting-works)
  - [Duplicate Detection: Skip Same Document](#duplicate-detection-skip-same-document)
  - [Version Update: v1 → v2 Management](#version-update-v1--v2-management)
    - [Decision Flow](#decision-flow)
    - [Complete Implementation](#complete-implementation)
    - [Usage Example](#usage-example)
  - [Metadata Schema Reference](#metadata-schema-reference)
  - [Strategy Comparison](#strategy-comparison)
  - [Best Practices](#best-practices-1)
- [Embedding](#embedding)
  - [What are Embeddings?](#what-are-embeddings)
    - [Key Properties](#key-properties)
  - [Understanding Dimensions in Embeddings](#understanding-dimensions-in-embeddings)
    - [What Does "Dimension" Mean?](#what-does-dimension-mean)
    - [Real-World: 768 Dimensions](#real-world-768-dimensions)
    - [Does Higher Dimension = Better Accuracy?](#does-higher-dimension--better-accuracy)
    - [Practical Comparison](#practical-comparison)
    - [Visual: How Dimensions Capture Meaning](#visual-how-dimensions-capture-meaning)
    - [Rule of Thumb](#rule-of-thumb)
  - [Storage \& Cost Impact of High Dimensions](#storage--cost-impact-of-high-dimensions)
    - [Yes! High dimensions directly impact storage, speed, and cost](#yes-high-dimensions-directly-impact-storage-speed-and-cost)
    - [Storage Size Calculation](#storage-size-calculation)
    - [Real-World Example: How Data is Stored](#real-world-example-how-data-is-stored)
      - [Understanding Vector Database Storage](#understanding-vector-database-storage)
      - [Two Approaches to Metadata](#two-approaches-to-metadata)
      - [Storage Breakdown: 10M Documents](#storage-breakdown-10m-documents)
      - [Metadata Storage Formats](#metadata-storage-formats)
      - [When to Include Metadata in Embeddings](#when-to-include-metadata-in-embeddings)
      - [Key Takeaways](#key-takeaways-1)
    - [VectorDB Cost Comparison (1M Vectors)](#vectordb-cost-comparison-1m-vectors)
    - [Retrieval Speed Impact](#retrieval-speed-impact)
    - [The "Curse of Dimensionality"](#the-curse-of-dimensionality)
    - [Cost Optimization Strategies](#cost-optimization-strategies)
      - [1. **Dimension Reduction (Matryoshka Embeddings)**](#1-dimension-reduction-matryoshka-embeddings)
      - [2. **Quantization (Reduce Precision)**](#2-quantization-reduce-precision)
      - [How to Implement Quantization](#how-to-implement-quantization)
      - [Quantization Types Comparison](#quantization-types-comparison)
      - [Best Practices](#best-practices-2)
      - [3. **Product Quantization (PQ)**](#3-product-quantization-pq)
    - [Practical Recommendations](#practical-recommendations)
    - [Cost-Benefit Analysis Table](#cost-benefit-analysis-table)
    - [Decision Framework](#decision-framework)
    - [Key Takeaways](#key-takeaways-2)
  - [How Words are Converted to Embeddings](#how-words-are-converted-to-embeddings)
    - [Traditional Approach (Word2Vec/GloVe)](#traditional-approach-word2vecglove)
    - [Modern Approach (Transformers: BERT, Sentence-BERT)](#modern-approach-transformers-bert-sentence-bert)
    - [How Embeddings Are Applied (Step 3 Detailed)](#how-embeddings-are-applied-step-3-detailed)
    - [Visual: Complete Embedding Process](#visual-complete-embedding-process)
    - [Key Differences from Word2Vec](#key-differences-from-word2vec)
    - [Beginner Summary](#beginner-summary)
  - [Visual: Semantic Space](#visual-semantic-space)
  - [How Embedding Models Work](#how-embedding-models-work)
    - [Architecture Overview](#architecture-overview)
    - [Top Performing Embedding Models in Market (2026)](#top-performing-embedding-models-in-market-2026)
    - [Model Category Breakdown](#model-category-breakdown)
      - [🏆 Best for Production (Paid)](#-best-for-production-paid)
      - [🆓 Best Free/Open-Source](#-best-freeopen-source)
    - [Use Case Recommendations](#use-case-recommendations)
    - [Cost Comparison (1 Billion Tokens)](#cost-comparison-1-billion-tokens)
    - [Quick Selection Guide](#quick-selection-guide)
  - [Cost Reduction Strategies in Production](#cost-reduction-strategies-in-production)
    - [1. **Caching Strategy**](#1-caching-strategy)
    - [2. **Dimension Reduction**](#2-dimension-reduction)
    - [3. **Batch Processing**](#3-batch-processing)
    - [4. **Model Selection Based on Use Case**](#4-model-selection-based-on-use-case)
    - [5. **Self-Hosted Models**](#5-self-hosted-models)
  - [Best Practices for RAG Embeddings](#best-practices-for-rag-embeddings)
    - [1. **Chunk Size Optimization**](#1-chunk-size-optimization)
    - [2. **Add Metadata to Chunks**](#2-add-metadata-to-chunks)
    - [3. **Hybrid Search Strategy**](#3-hybrid-search-strategy)
    - [4. **Query Enhancement**](#4-query-enhancement)
    - [5. **Fine-tune Embeddings for Domain**](#5-fine-tune-embeddings-for-domain)
  - [RAG-Specific Optimization Techniques](#rag-specific-optimization-techniques)
    - [1. **Parent-Child Chunking**](#1-parent-child-chunking)
    - [2. **Multi-Vector Retrieval**](#2-multi-vector-retrieval)
    - [3. **Embedding Quality Metrics**](#3-embedding-quality-metrics)
  - [Cost Comparison Table](#cost-comparison-table)
  - [Quick Decision Framework](#quick-decision-framework)
  - [Key Takeaways](#key-takeaways-3)
  - [Production Checklist](#production-checklist)
  - [VectorDB Re-Indexing at Scale (100 to 50,000+ Documents)](#vectordb-re-indexing-at-scale-100-to-50000-documents)
    - [What is Re-Indexing in VectorDB?](#what-is-re-indexing-in-vectordb)
    - [Why Re-Indexing is Needed at Scale](#why-re-indexing-is-needed-at-scale)
    - [How Re-Indexing is Performed](#how-re-indexing-is-performed)
      - [Step 1 — Trigger Detection](#step-1--trigger-detection)
      - [Step 2 — Choose Re-Index Strategy](#step-2--choose-re-index-strategy)
      - [Step 3 — Full Re-Index Pipeline](#step-3--full-re-index-pipeline)
      - [Step 4 — Incremental Re-Index Pipeline](#step-4--incremental-re-index-pipeline)
      - [Step 5 — Zero-Downtime Blue/Green Re-Index](#step-5--zero-downtime-bluegreen-re-index)
    - [Impact of Re-Indexing](#impact-of-re-indexing)
      - [Performance Impact](#performance-impact)
      - [Cost Impact](#cost-impact)
      - [Quality Impact](#quality-impact)
    - [Re-Index Decision Matrix](#re-index-decision-matrix)
    - [Production Re-Index Checklist](#production-re-index-checklist)
  - [Pre-Filtering vs Post-Filtering in VectorDB](#pre-filtering-vs-post-filtering-in-vectordb)
    - [What is Filtering in RAG Retrieval?](#what-is-filtering-in-rag-retrieval)
    - [Pre-Filtering — How It Works](#pre-filtering--how-it-works)
      - [How to Apply Pre-Filtering](#how-to-apply-pre-filtering)
      - [Pre-Filter on Multiple Fields](#pre-filter-on-multiple-fields)
      - [Dynamic Pre-Filter from User Context](#dynamic-pre-filter-from-user-context)
      - [Pre-Filtering with Self-Query Retriever](#pre-filtering-with-self-query-retriever)
    - [Post-Filtering — How It Works](#post-filtering--how-it-works)
      - [How to Apply Post-Filtering](#how-to-apply-post-filtering)
    - [Pre-Filter vs Post-Filter — Decision Guide](#pre-filter-vs-post-filter--decision-guide)
    - [Hybrid: Pre-Filter + Post-Filter Together](#hybrid-pre-filter--post-filter-together)
    - [Common Pitfalls](#common-pitfalls)
  - [RAG Evaluation in Production](#rag-evaluation-in-production)
    - [Best Practice: Evaluating Retrieval Without Ground Truth](#best-practice-evaluating-retrieval-without-ground-truth)
    - [Chunk Size vs Precision / Recall](#chunk-size-vs-precision--recall)
    - [Evaluation Metrics Reference](#evaluation-metrics-reference)
  - [Token Pricing — Top Embedding Models](#token-pricing--top-embedding-models)
    - [Hosted / API (pay per token)](#hosted--api-pay-per-token)
    - [Open Source (self-host — free inference, pay only for compute)](#open-source-self-host--free-inference-pay-only-for-compute)
    - [How to use open source (HuggingFace)](#how-to-use-open-source-huggingface)
    - [Hosted vs Open Source — when to pick which](#hosted-vs-open-source--when-to-pick-which)
    - [Cost comparison at scale](#cost-comparison-at-scale)
- [Vector DB at Scale: Managing Exponential Growth \& Retrieval Quality](#vector-db-at-scale-managing-exponential-growth--retrieval-quality)
  - [🎯 Main Question: Vector DB Management with Exponential Growth](#-main-question-vector-db-management-with-exponential-growth)
    - [Answer (Beginner-Friendly):](#answer-beginner-friendly)
  - [❓ Follow-Up Questions: Deep Dives](#-follow-up-questions-deep-dives)
    - [Q1: Parent-Child (Hierarchical) Chunking — Storage, Latency \& Architecture](#q1-parent-child-hierarchical-chunking--storage-latency--architecture)
    - [Q2: How HNSW and IVF Work — Indexing Explained For Beginners](#q2-how-hnsw-and-ivf-work--indexing-explained-for-beginners)
      - [HNSW (Hierarchical Navigable Small World) — GPS with Zoom Levels](#hnsw-hierarchical-navigable-small-world--gps-with-zoom-levels)
      - [IVF (Inverted File Index) — Library Sections](#ivf-inverted-file-index--library-sections)
      - [HNSW vs IVF — Decision Guide](#hnsw-vs-ivf--decision-guide)
    - [Q3: Optimization Techniques for HNSW \& IVF (Parameter Tuning Guide)](#q3-optimization-techniques-for-hnsw--ivf-parameter-tuning-guide)
      - [HNSW Optimizations](#hnsw-optimizations)
      - [IVF Optimizations](#ivf-optimizations)
      - [Decision Framework for Beginners](#decision-framework-for-beginners)
    - [Q4: Hybrid Search — Does It Increase Storage \& Latency?](#q4-hybrid-search--does-it-increase-storage--latency)
    - [Q5: What Are "relevant\_doc\_ids" and How Are They Generated?](#q5-what-are-relevant_doc_ids-and-how-are-they-generated)
      - [Method 1: **Manual Annotation (Gold Standard)**](#method-1-manual-annotation-gold-standard)
      - [Method 2: **LLM-as-Judge (No Ground Truth Needed)**](#method-2-llm-as-judge-no-ground-truth-needed)
      - [Method 3: **Synthetic Generation (Semi-Automated)**](#method-3-synthetic-generation-semi-automated)
      - [Method 4: **User Behavior (Implicit Feedback)**](#method-4-user-behavior-implicit-feedback)
  - [🏁 Summary Table: Managing Large Vector DBs](#-summary-table-managing-large-vector-dbs)

---

# Chunk Metadata — Schema, Strategy & Effective Filtering

---

## Where Is Metadata Saved? Is It in VectorDB?

**Yes — metadata is stored as payload directly inside the VectorDB alongside the vector.**

```
VectorDB Point (one per chunk)
├── id          →  unique UUID
├── vector      →  [0.12, -0.45, 0.88, ...]  ← embedding (used for ANN search)
└── payload     →  { "source": "...", "tenant_id": "...", "is_active": true, ... }
                                               ← metadata (used for filtering)
```

| Storage | What | Used for |
|---|---|---|
| **VectorDB payload** | All metadata fields | Pre-filter, post-filter, display |
| **Vector** | Embedding of chunk text | ANN similarity search |
| **Relational DB (optional)** | Audit logs, user feedback | Analytics, feedback tracking |
| **Object Store (optional)** | Raw original documents | Re-ingestion, reprocessing |

> Metadata is **NOT** part of similarity scoring — it only restricts/ranks results via filters.

---

## Why Metadata Matters in Production RAG

In naive RAG, every query searches **all chunks** equally — no access control, no recency, no relevance boosting. At scale this breaks:

```
10M chunks in prod → wrong user gets confidential HR policy
                  → outdated 2022 drug dosage returned alongside 2025 version
                  → irrelevant cross-department docs pollute context window
```

**Metadata solves this.** It enables:
- **Pre-filter** — restrict search space before ANN runs (faster + cheaper)
- **ACL enforcement** — user only retrieves what they are allowed to see
- **Recency ranking** — prefer newer chunks over stale ones
- **Deduplication** — hash-based skip at ingestion
- **Explainability** — LLM can cite source, section, author in its answer

---

## Top Metadata Fields to Include

Start lean. Add LLM-generated fields only after measuring retrieval quality.

### Tier 1 — Must Have (zero cost, always available)

| Field | Type | Purpose |
|---|---|---|
| `chunk_id` | string | Deduplication + upsert by ID |
| `doc_id` | string | Group all chunks of a document |
| `source` | string | Citation in LLM answer |
| `fingerprint` | string | Skip re-ingestion of unchanged docs |
| `is_active` | bool | Filter out retired / stale chunks |
| `ingested_at` | datetime | Recency sorting |
| `page` / `chunk_index` | int | Provenance + ordering |

### Tier 2 — High Value (cheap, structured)

| Field | Type | Purpose |
|---|---|---|
| `tenant_id` | string | Multi-tenant isolation (pre-filter) |
| `allowed_roles` | string[] | RBAC enforcement (pre-filter) |
| `doc_type` | string | Filter by type — policy / faq / contract |
| `h1` / `h2` | string | Section context shown to LLM |
| `language` | string | Route to correct embedding model |
| `char_count` | int | Skip empty / too-short chunks |

### Tier 3 — Add When Needed (LLM cost)

| Field | Type | When to add |
|---|---|---|
| `keywords` | string[] | BM25 / hybrid search is needed |
| `chunk_summary` | string | Queries are vague or high-level |
| `hypothetical_queries` | string[] | Recall is low |
| `quality_score` | float | Noisy chunks degrade answers |
| `feedback_score` | float | After go-live, real user feedback available |

### Minimal starting payload

```python
{
    "chunk_id":      "doc-abc#chunk_3",
    "doc_id":        "doc-abc",
    "source":        "s3://bucket/policy.pdf",
    "fingerprint":   "a3f8c2d1...",
    "is_active":     True,
    "ingested_at":   "2025-11-01T08:00:00Z",
    "page":          14,
    "chunk_index":   3,
    "tenant_id":     "acme-corp",
    "allowed_roles": ["employee"],
    "doc_type":      "policy",
    "h1":            "Leave Policy",
    "h2":            "Parental Leave",
}
```

```
Without metadata:  query → ANN search over ALL chunks → top-k
With metadata:     query → filter(tenant, acl, is_active) → ANN search over SUBSET → top-k
                   Result: 10x faster, access-safe, no stale data
```

---

## Complete Metadata Schema

```python
{
    # ── 1. IDENTITY & PROVENANCE ────────────────────────────────
    "source":          "s3://corp-docs/hr/leave_policy_v3.pdf",
    "doc_id":          "hr-leave-policy",           # stable logical ID
    "chunk_id":        "hr-leave-policy#v3#chunk_7",
    "version":         "3.0",
    "fingerprint":     "a3f8c2d1...",               # SHA-256 of chunk text
    "is_active":       True,                        # False = retired/superseded
    "ingested_at":     "2025-11-01T08:00:00Z",
    "retired_at":      None,

    # ── 2. ACCESS CONTROL (ACL) ─────────────────────────────────
    "tenant_id":       "-hr",                 # multi-tenant isolation
    "allowed_roles":   ["hr-manager", "employee"],  # RBAC
    "allowed_users":   [],                          # ABAC (specific users)
    "confidentiality": "internal",                  # public/internal/confidential/restricted
    "data_region":     "us-east-1",                 # data residency compliance

    # ── 3. DOCUMENT STRUCTURE ───────────────────────────────────
    "doc_type":        "policy",                    # policy/contract/faq/report/manual
    "file_type":       "pdf",
    "h1":              "Leave & Time Off",          # section hierarchy from DocLing
    "h2":              "Parental Leave",
    "h3":              "Eligibility Criteria",
    "page":            14,
    "chunk_index":     7,                           # position in document
    "total_chunks":    22,
    "char_count":      842,
    "language":        "en",

    # ── 4. SEMANTIC ENRICHMENT — KEYWORDS ───────────────────────
    "keywords":        ["parental leave", "FMLA", "eligibility", "12 weeks"],
    "entities":        ["FMLA", "California SB 1383"],  # NER extracted entities
    "topics":          ["hr-policy", "benefits", "compliance"],

    # ── 5. SEMANTIC ENRICHMENT — SUMMARY ────────────────────────
    "chunk_summary":   "Employees with 12+ months tenure qualify for 12 weeks paid parental leave under FMLA.",

    # ── 6. HYPOTHETICAL QUERY GENERATION ────────────────────────
    "hypothetical_queries": [
        "How many weeks of parental leave am I entitled to?",
        "What is the parental leave eligibility requirement?",
        "Can a new employee take parental leave?"
    ],

    # ── 7. TEMPORAL & LIFECYCLE ─────────────────────────────────
    "doc_created_at":  "2023-06-01",
    "doc_updated_at":  "2025-10-15",
    "effective_from":  "2025-11-01",               # policy effective date
    "expires_at":      "2026-12-31",               # content expiry for auto-retire
    "review_cycle":    "annual",

    # ── 8. CONTENT QUALITY SIGNALS ──────────────────────────────
    "quality_score":   0.92,                       # human/LLM quality rating 0-1
    "verified":        True,                        # SME-reviewed flag
    "verified_by":     "jane.doe@corp.com",
    "verified_at":     "2025-10-20",
    "has_table":       False,                       # chunk contains a table
    "has_image":       False,                       # chunk references an image
    "chunk_type":      "narrative",                 # narrative/table/list/code/header

    # ── 9. RETRIEVAL TUNING SIGNALS ─────────────────────────────
    "retrieval_boost": 1.2,                         # manually boost important chunks
    "feedback_score":  0.88,                        # avg user thumbs-up score
    "retrieval_count": 142,                         # how many times retrieved
    "last_retrieved":  "2026-03-01T14:23:00Z",
    "avg_rank":        1.4,                         # avg rank when retrieved
}
```

---

## 1. Identity & Provenance

Track exactly where a chunk came from and its lifecycle state.

```python
import hashlib, uuid
from datetime import datetime, timezone

def build_identity_metadata(
    text: str,
    source: str,
    doc_id: str,
    version: str,
    chunk_index: int
) -> dict:
    fingerprint   = hashlib.sha256(text.encode()).hexdigest()
    chunk_id      = f"{doc_id}#v{version}#chunk_{chunk_index}"

    return {
        "source":      source,
        "doc_id":      doc_id,
        "chunk_id":    chunk_id,
        "version":     version,
        "fingerprint": fingerprint,
        "is_active":   True,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "retired_at":  None,
        "chunk_index": chunk_index,
    }
```

**Why `is_active` matters:** When a policy is updated, old chunks are marked `is_active=False` — they stay in the DB for audit but are excluded from all retrieval queries. No hard deletes needed.

---

## 2. ACL — Access Control

**Most critical field for multi-tenant / enterprise RAG.** Prevents data leakage across tenants, roles, or data residency boundaries.

```python
from typing import Optional

def build_acl_metadata(
    tenant_id:       str,
    allowed_roles:   list[str],                   # e.g. ["hr-manager", "employee"]
    confidentiality: str,                          # public/internal/confidential/restricted
    allowed_users:   Optional[list[str]] = None,  # for ABAC — specific user emails
    data_region:     str = "us-east-1"
) -> dict:
    return {
        "tenant_id":       tenant_id,
        "allowed_roles":   allowed_roles,
        "allowed_users":   allowed_users or [],
        "confidentiality": confidentiality,
        "data_region":     data_region,
    }

# Retrieval — enforce ACL as pre-filter BEFORE vector search
def build_acl_filter(user_role: str, tenant_id: str, user_id: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

    return Filter(must=[
        FieldCondition(key="tenant_id",     match=MatchValue(value=tenant_id)),
        FieldCondition(key="is_active",     match=MatchValue(value=True)),
        FieldCondition(key="allowed_roles", match=MatchAny(any=[user_role, "public"])),
    ])

# Usage at query time
acl_filter = build_acl_filter(
    user_role="hr-manager",
    tenant_id="-hr",
    user_id="abc@abc.com"
)
results = vector_store.similarity_search(query, k=5, filter=acl_filter)
```

**ACL enforcement layers:**

| Layer | Mechanism | Where enforced |
|---|---|---|
| Tenant isolation | `tenant_id` pre-filter | Vector DB query filter |
| Role-based (RBAC) | `allowed_roles` pre-filter | Vector DB query filter |
| User-specific (ABAC) | `allowed_users` pre-filter | Vector DB query filter |
| Data residency | `data_region` filter | Deployment-level routing |
| Confidentiality | `confidentiality` tier | Application layer check |

---

## 3. Document Structure

Preserves the document hierarchy so retrieved chunks carry context about *where* in the document they belong.

```python
from docling.document_converter import DocumentConverter

def build_structure_metadata(doc, chunk, chunk_index: int, total_chunks: int) -> dict:
    return {
        "doc_type":    doc.metadata.get("doc_type", "unknown"),
        "file_type":   doc.metadata.get("file_type", "pdf"),
        "h1":          chunk.metadata.get("h1", ""),
        "h2":          chunk.metadata.get("h2", ""),
        "h3":          chunk.metadata.get("h3", ""),
        "page":        chunk.metadata.get("page", 0),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "char_count":  len(chunk.page_content),
        "language":    detect_language(chunk.page_content),
        "chunk_type":  classify_chunk_type(chunk.page_content),
        "has_table":   "---" in chunk.page_content and "|" in chunk.page_content,
        "has_image":   "[IMAGE:" in chunk.page_content,
    }

def classify_chunk_type(text: str) -> str:
    if "|" in text and "---" in text:  return "table"
    if text.strip().startswith(("```", "    ")): return "code"
    if text.strip().startswith(("-", "*", "1.")): return "list"
    if len(text.split()) < 15:         return "header"
    return "narrative"
```

**Why `h1/h2/h3` in metadata matters for RAG:**

```python
# LLM gets richer context about each retrieved chunk
context = "\n\n".join([
    f"[{doc.metadata['h1']} > {doc.metadata['h2']} > {doc.metadata['h3']}]\n"
    f"(Page {doc.metadata['page']}, Source: {doc.metadata['source']})\n"
    f"{doc.page_content}"
    for doc in retrieved_docs
])
# LLM now knows: "this is from Chapter 3 > Parental Leave > Eligibility Criteria, page 14"
```

---

## 4. Semantic Enrichment — Keywords

Extracted keywords serve **two purposes**: BM25 hybrid search boost AND metadata pre-filtering.

```python
import yake
from langchain_openai import ChatOpenAI
import json

# Option A — Fast, local, no LLM cost (YAKE)
def extract_keywords_yake(text: str, max_keywords: int = 8) -> list[str]:
    extractor = yake.KeywordExtractor(lan="en", n=2, top=max_keywords)
    keywords  = extractor.extract_keywords(text)
    return [kw for kw, score in keywords]

# Option B — LLM-extracted, higher quality
def extract_keywords_llm(text: str, llm: ChatOpenAI) -> list[str]:
    resp = llm.invoke(
        f"Extract 5-8 precise keyword phrases from this text for search indexing.\n"
        f"Return ONLY a JSON array of strings.\n\nText:\n{text[:1500]}"
    )
    return json.loads(resp.content)

# Option C — NER entities (proper nouns, regulations, product names)
def extract_entities_spacy(text: str) -> list[str]:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return list({ent.text for ent in doc.ents
                 if ent.label_ in ("ORG", "LAW", "PRODUCT", "GPE", "PERSON")})

def build_keyword_metadata(text: str, llm=None) -> dict:
    keywords = extract_keywords_yake(text)
    entities = extract_entities_spacy(text)
    return {
        "keywords": keywords,
        "entities": entities,
        "topics":   [],   # populated separately via topic classifier
    }
```

**Filter by keyword at retrieval — find chunks mentioning specific regulation:**

```python
from qdrant_client.models import Filter, FieldCondition, MatchAny

kw_filter = Filter(must=[
    FieldCondition(key="keywords", match=MatchAny(any=["FMLA", "parental leave"])),
    FieldCondition(key="is_active", match=MatchValue(value=True)),
])
results = vector_store.similarity_search(query, k=5, filter=kw_filter)
```

---

## 5. Semantic Enrichment — Summary

A per-chunk LLM summary embedded alongside the chunk. Dramatically improves retrieval for vague or short queries that wouldn't match the raw chunk text.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_chunk_summary(text: str, doc_context: str = "") -> str:
    """Generate a concise summary of the chunk for indexing."""
    prompt = f"""Summarize this document chunk in 1-2 sentences for search indexing.
Be specific — include key facts, numbers, names, or regulations mentioned.
{f'Document context: {doc_context}' if doc_context else ''}

Chunk:
{text[:2000]}

Summary:"""
    return llm.invoke(prompt).content.strip()

# Two embedding strategies with summary:

# Strategy A — Embed summary instead of raw chunk (better for vague queries)
def ingest_with_summary_embedding(chunk, vector_store):
    summary = generate_chunk_summary(chunk.page_content)
    vector_store.add_texts(
        texts=[summary],                          # EMBED the summary
        metadatas=[{**chunk.metadata,
                    "chunk_summary": summary,
                    "page_content":  chunk.page_content}]  # store raw in metadata
    )

# Strategy B — Embed both (parent-child index)
# summary vector → retrieves chunk → returns raw text to LLM
```

**When summary embedding wins:**

| Query type | Raw chunk embedding | Summary embedding |
|---|---|---|
| "What is the FMLA eligibility?" | Good — exact keyword | Good |
| "Can I take time off for a new baby?" | Poor — no "parental leave" keyword | Good — summary captures intent |
| "12 weeks" | Good | Good |
| Vague/conversational queries | Poor | Significantly better |

---

## 6. Hypothetical Query Generation

Generate 3-5 questions that this chunk would answer. Embed these questions — queries match question embeddings better than raw text embeddings.

```python
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def generate_hypothetical_queries(text: str, n: int = 5) -> list[str]:
    """
    HyQ (Hypothetical Questions) — inverse of HyDE.
    HyDE: embed hypothetical answer to match chunk.
    HyQ:  embed hypothetical questions in chunk metadata to match user queries.
    """
    prompt = f"""Generate {n} realistic questions that a user would ask
whose answer is contained in this chunk.
Vary phrasing — include formal, casual, and specific versions.
Return ONLY a JSON array of question strings.

Chunk:
{text[:2000]}"""

    resp = llm.invoke(prompt)
    return json.loads(resp.content)

# Two approaches for using hypothetical queries:

# Approach A — Store as metadata, use in keyword/BM25 search
def build_hyq_metadata(text: str) -> dict:
    queries = generate_hypothetical_queries(text)
    return {"hypothetical_queries": queries}

# Approach B — Create SEPARATE embedding entries per question (Multi-vector index)
# Each question gets its own vector pointing to the same chunk
def ingest_with_hyq_embeddings(chunk, vector_store):
    queries  = generate_hypothetical_queries(chunk.page_content)
    base_id  = chunk.metadata["chunk_id"]

    # Ingest the original chunk
    vector_store.add_texts(
        texts=[chunk.page_content],
        metadatas=[{**chunk.metadata, "entry_type": "chunk"}],
        ids=[base_id]
    )

    # Ingest each hypothetical question as a separate vector → same chunk_id
    for i, q in enumerate(queries):
        vector_store.add_texts(
            texts=[q],
            metadatas=[{**chunk.metadata,
                        "entry_type": "hyq",
                        "hyq_index":  i,
                        "parent_chunk_id": base_id}],
            ids=[f"{base_id}#hyq_{i}"]
        )
    # At retrieval: fetch by HyQ vector → look up parent_chunk_id → return raw chunk text
```

---

## 7. Temporal & Lifecycle

Critical for policy-heavy domains (healthcare, legal, finance) where outdated content causes serious errors.

```python
from datetime import datetime, timezone, date

def build_temporal_metadata(
    doc_created_at:  str,         # ISO date string
    doc_updated_at:  str,
    effective_from:  str | None = None,   # when policy takes effect
    expires_at:      str | None = None,   # auto-retire after this date
    review_cycle:    str = "annual"       # annual/quarterly/ad-hoc
) -> dict:
    return {
        "doc_created_at": doc_created_at,
        "doc_updated_at": doc_updated_at,
        "effective_from": effective_from,
        "expires_at":     expires_at,
        "review_cycle":   review_cycle,
    }

# Auto-retire expired chunks via scheduled job
def retire_expired_chunks(collection: str, client):
    from qdrant_client.models import Filter, FieldCondition, Range

    today = date.today().isoformat()
    expired, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="is_active", match=MatchValue(value=True)),
            FieldCondition(key="expires_at", range=Range(lt=today)),
        ]),
        limit=1000,
        with_payload=True
    )

    for point in expired:
        client.set_payload(
            collection_name=collection,
            payload={"is_active": False,
                     "retired_at": datetime.now(timezone.utc).isoformat()},
            points=[point.id]
        )
    print(f"Auto-retired {len(expired)} expired chunks")

# Retrieval — only return currently effective content
def build_temporal_filter(as_of_date: str | None = None):
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    today = as_of_date or date.today().isoformat()
    return Filter(must=[
        FieldCondition(key="is_active", match=MatchValue(value=True)),
        # Only return content that has taken effect
        FieldCondition(key="effective_from", range=Range(lte=today)),
    ])
```

---

## 8. Content Quality Signals

Score and flag chunks by quality — suppress low-quality, surface verified content.

```python
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def score_chunk_quality(text: str) -> dict:
    """LLM-scored quality signal — run at ingestion time, not at query time."""
    prompt = f"""Rate this document chunk on 3 dimensions (0.0–1.0 each):
1. completeness: Does it contain a complete thought/answer?
2. specificity:  Does it contain specific facts (not vague)?
3. clarity:      Is it clearly written?

Return ONLY JSON: {{"completeness": float, "specificity": float, "clarity": float}}

Chunk:
{text[:1000]}"""

    result = json.loads(llm.invoke(prompt).content)
    quality_score = (result["completeness"] + result["specificity"] + result["clarity"]) / 3

    return {
        "quality_score":    round(quality_score, 3),
        "quality_details":  result,
        "verified":         False,
        "verified_by":      None,
        "verified_at":      None,
        "chunk_type":       classify_chunk_type(text),
        "has_table":        "|" in text and "---" in text,
        "has_image":        "[IMAGE:" in text,
    }

# At retrieval — boost high-quality, suppress low-quality
def quality_filter(min_quality: float = 0.5):
    from qdrant_client.models import Filter, FieldCondition, Range
    return Filter(must=[
        FieldCondition(key="quality_score", range=Range(gte=min_quality))
    ])
```

---

## 9. Retrieval Tuning Signals

Feedback-loop metadata — updated at runtime, used to re-rank retrieved chunks.

```python
from datetime import datetime, timezone
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

def record_retrieval(collection: str, chunk_id: str, rank: int):
    """Increment retrieval_count and update avg_rank — called after each retrieval."""
    point = client.retrieve(collection, ids=[chunk_id], with_payload=True)[0]
    payload = point.payload

    old_count = payload.get("retrieval_count", 0)
    old_rank  = payload.get("avg_rank", rank)
    new_count = old_count + 1
    new_rank  = (old_rank * old_count + rank) / new_count

    client.set_payload(
        collection_name=collection,
        payload={
            "retrieval_count": new_count,
            "avg_rank":        round(new_rank, 2),
            "last_retrieved":  datetime.now(timezone.utc).isoformat(),
        },
        points=[chunk_id]
    )

def record_user_feedback(collection: str, chunk_id: str, thumbs_up: bool):
    """Update feedback_score from user thumbs up/down."""
    point   = client.retrieve(collection, ids=[chunk_id], with_payload=True)[0]
    payload = point.payload

    old_score = payload.get("feedback_score", 0.5)
    new_vote  = 1.0 if thumbs_up else 0.0
    # Exponential moving average — recent feedback weighted more
    new_score = 0.8 * old_score + 0.2 * new_vote

    client.set_payload(
        collection_name=collection,
        payload={"feedback_score": round(new_score, 3)},
        points=[chunk_id]
    )

def boost_important_chunk(collection: str, chunk_id: str, boost: float = 1.5):
    """Manually boost a chunk — SME marks it as authoritative."""
    client.set_payload(
        collection_name=collection,
        payload={"retrieval_boost": boost},
        points=[chunk_id]
    )
```

**Re-ranking with retrieval signals (post-retrieval):**

```python
def rerank_with_signals(docs_with_scores: list[tuple]) -> list:
    """Apply feedback_score and retrieval_boost to re-rank retrieved docs."""
    def adjusted_score(doc, vector_score):
        boost    = doc.metadata.get("retrieval_boost", 1.0)
        feedback = doc.metadata.get("feedback_score", 0.5)
        quality  = doc.metadata.get("quality_score", 0.5)
        # Weighted combination
        return vector_score * boost * (0.6 + 0.2*feedback + 0.2*quality)

    reranked = sorted(
        docs_with_scores,
        key=lambda x: adjusted_score(x[0], x[1]),
        reverse=True
    )
    return [doc for doc, score in reranked]
```

---

## 10. Effective Metadata Filtering at Retrieval

### Filter-First Architecture (Pre-filter before ANN)

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
from datetime import date

def build_production_filter(
    tenant_id:    str,
    user_role:    str,
    user_id:      str,
    doc_types:    list[str] | None  = None,
    topic_filter: list[str] | None  = None,
    min_quality:  float             = 0.5,
    as_of_date:   str | None        = None,
) -> Filter:
    """Compose all metadata filters for a production retrieval query."""

    today = as_of_date or date.today().isoformat()
    must_conditions = [
        # Identity
        FieldCondition(key="tenant_id",     match=MatchValue(value=tenant_id)),
        FieldCondition(key="is_active",     match=MatchValue(value=True)),
        # ACL
        FieldCondition(key="allowed_roles", match=MatchAny(any=[user_role, "public"])),
        # Temporal
        FieldCondition(key="effective_from", range=Range(lte=today)),
        # Quality gate
        FieldCondition(key="quality_score", range=Range(gte=min_quality)),
    ]

    if doc_types:
        must_conditions.append(
            FieldCondition(key="doc_type", match=MatchAny(any=doc_types))
        )
    if topic_filter:
        must_conditions.append(
            FieldCondition(key="topics", match=MatchAny(any=topic_filter))
        )

    return Filter(must=must_conditions)


# Full retrieval pipeline
def retrieve(
    query:      str,
    vector_store,
    tenant_id:  str,
    user_role:  str,
    user_id:    str,
    doc_types:  list[str] | None = None,
    k:          int = 5,
) -> list:
    pre_filter = build_production_filter(
        tenant_id=tenant_id,
        user_role=user_role,
        user_id=user_id,
        doc_types=doc_types
    )

    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        k=k * 3,          # over-fetch to allow re-ranking
        filter=pre_filter
    )

    # Re-rank with feedback + quality signals
    reranked = rerank_with_signals(docs_with_scores)
    return reranked[:k]
```

### Filter Decision Matrix

| Filter | When to apply | Stage |
|---|---|---|
| `tenant_id` | Always — multi-tenant | Pre-filter (must) |
| `is_active=True` | Always — exclude retired | Pre-filter (must) |
| `allowed_roles` | Always — access control | Pre-filter (must) |
| `effective_from <= today` | Policy/legal/medical | Pre-filter (must) |
| `quality_score >= 0.5` | Production queries | Pre-filter (should) |
| `doc_type` | Domain-specific queries | Pre-filter (optional) |
| `topics` | Scoped queries | Pre-filter (optional) |
| `keywords` | Keyword-boosted hybrid | Pre-filter (optional) |
| `language` | Multilingual apps | Pre-filter (optional) |
| `feedback_score` | Re-ranking | Post-retrieval sort |
| `retrieval_boost` | Re-ranking | Post-retrieval sort |
| `chunk_type=table` | Table QA queries | Pre-filter (optional) |
| `expires_at > today` | Time-bound content | Pre-filter (optional) |

---

## 11. Full Ingestion Pipeline

```python
import hashlib, uuid
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def build_full_metadata(
    text:         str,
    source:       str,
    doc_id:       str,
    version:      str,
    chunk_index:  int,
    total_chunks: int,
    tenant_id:    str,
    allowed_roles: list[str],
    doc_type:     str,
    effective_from: str | None = None,
    expires_at:   str | None   = None,
    h1: str = "", h2: str = "", h3: str = "", page: int = 0,
) -> dict:

    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    chunk_id    = f"{doc_id}#v{version}#chunk_{chunk_index}"
    now         = datetime.now(timezone.utc).isoformat()

    # Enrichment — LLM calls (batch these in production)
    summary     = generate_chunk_summary(text)
    keywords    = extract_keywords_yake(text)
    entities    = extract_entities_spacy(text)
    hyq         = generate_hypothetical_queries(text, n=3)
    quality     = score_chunk_quality(text)

    return {
        # Identity
        "source": source, "doc_id": doc_id, "chunk_id": chunk_id,
        "version": version, "fingerprint": fingerprint,
        "is_active": True, "ingested_at": now, "retired_at": None,
        # ACL
        "tenant_id": tenant_id, "allowed_roles": allowed_roles,
        "allowed_users": [], "confidentiality": "internal",
        # Structure
        "doc_type": doc_type, "h1": h1, "h2": h2, "h3": h3,
        "page": page, "chunk_index": chunk_index, "total_chunks": total_chunks,
        "char_count": len(text), "chunk_type": classify_chunk_type(text),
        "has_table": "|" in text and "---" in text,
        # Semantic
        "keywords": keywords, "entities": entities,
        "chunk_summary": summary, "hypothetical_queries": hyq,
        # Temporal
        "effective_from": effective_from or now[:10],
        "expires_at": expires_at,
        # Quality
        **quality,
        # Retrieval signals (initialised)
        "retrieval_boost": 1.0, "feedback_score": 0.5,
        "retrieval_count": 0,   "avg_rank": 0.0,
        "last_retrieved": None,
    }


def ingest_chunks_production(
    chunks:     list[Document],
    vector_store: QdrantVectorStore,
    tenant_id:  str,
    allowed_roles: list[str],
    doc_type:   str,
    **kwargs
):
    new_texts, new_meta, new_ids = [], [], []

    for i, chunk in enumerate(chunks):
        meta = build_full_metadata(
            text=chunk.page_content,
            source=chunk.metadata.get("source", ""),
            doc_id=chunk.metadata.get("doc_id", "unknown"),
            version=chunk.metadata.get("version", "1.0"),
            chunk_index=i,
            total_chunks=len(chunks),
            tenant_id=tenant_id,
            allowed_roles=allowed_roles,
            doc_type=doc_type,
            h1=chunk.metadata.get("h1", ""),
            h2=chunk.metadata.get("h2", ""),
            h3=chunk.metadata.get("h3", ""),
            page=chunk.metadata.get("page", 0),
            **kwargs
        )
        chunk_id = meta["chunk_id"]
        new_texts.append(chunk.page_content)
        new_meta.append(meta)
        new_ids.append(str(uuid.UUID(bytes=bytes.fromhex(meta["fingerprint"][:32]))))

    vector_store.add_texts(texts=new_texts, metadatas=new_meta, ids=new_ids)
    print(f"Ingested {len(new_texts)} chunks with full metadata")
```

---

## Metadata Field Reference Table

| Field | Category | Type | Index? | Filter? | Notes |
|---|---|---|---|---|---|
| `source` | Identity | string | Yes | Optional | File path / S3 URI |
| `doc_id` | Identity | string | Yes | Yes | Stable logical doc ID |
| `chunk_id` | Identity | string | Yes | — | Unique per chunk+version |
| `fingerprint` | Identity | string | Yes | Yes | SHA-256, dedup check |
| `version` | Identity | string | Yes | Optional | Semantic version |
| `is_active` | Identity | bool | Yes | **Always** | Soft-delete flag |
| `ingested_at` | Identity | datetime | — | Optional | Audit trail |
| `tenant_id` | ACL | string | Yes | **Always** | Multi-tenant isolation |
| `allowed_roles` | ACL | string[] | Yes | **Always** | RBAC enforcement |
| `allowed_users` | ACL | string[] | Yes | Optional | ABAC specific users |
| `confidentiality` | ACL | string | Yes | Optional | Tier: public→restricted |
| `data_region` | ACL | string | Yes | Optional | Residency compliance |
| `doc_type` | Structure | string | Yes | Often | policy/faq/contract |
| `h1/h2/h3` | Structure | string | Yes | Optional | Section hierarchy |
| `page` | Structure | int | — | Optional | Source page number |
| `chunk_index` | Structure | int | — | — | Position in document |
| `chunk_type` | Structure | string | Yes | Optional | narrative/table/code |
| `has_table` | Structure | bool | Yes | Optional | Table QA routing |
| `language` | Structure | string | Yes | Optional | Multilingual filter |
| `keywords` | Semantic | string[] | Yes | Optional | BM25 + hybrid boost |
| `entities` | Semantic | string[] | Yes | Optional | Regulation/NER filter |
| `topics` | Semantic | string[] | Yes | Optional | Domain scoping |
| `chunk_summary` | Semantic | string | — | — | Used in LLM context |
| `hypothetical_queries` | Semantic | string[] | — | — | HyQ multi-vector |
| `effective_from` | Temporal | date | Yes | **Always** | Policy effective date |
| `expires_at` | Temporal | date | Yes | Optional | Auto-retire trigger |
| `doc_updated_at` | Temporal | date | Yes | Optional | Recency filter |
| `quality_score` | Quality | float | Yes | Yes | Gate low-quality |
| `verified` | Quality | bool | Yes | Optional | SME-reviewed |
| `feedback_score` | Retrieval | float | Yes | Post-rank | User thumbs up avg |
| `retrieval_count` | Retrieval | int | — | — | Popularity signal |
| `retrieval_boost` | Retrieval | float | — | Post-rank | Manual boost |
| `avg_rank` | Retrieval | float | — | — | Avg position when retrieved |

**Index strategy for Qdrant payload indexes** (create at collection setup, not query time):

```python
from qdrant_client.models import PayloadSchemaType

# Create indexes on fields used in pre-filters
for field, schema_type in [
    ("tenant_id",     PayloadSchemaType.KEYWORD),
    ("is_active",     PayloadSchemaType.BOOL),
    ("allowed_roles", PayloadSchemaType.KEYWORD),
    ("doc_type",      PayloadSchemaType.KEYWORD),
    ("topics",        PayloadSchemaType.KEYWORD),
    ("keywords",      PayloadSchemaType.KEYWORD),
    ("quality_score", PayloadSchemaType.FLOAT),
    ("effective_from",PayloadSchemaType.KEYWORD),
    ("expires_at",    PayloadSchemaType.KEYWORD),
    ("chunk_type",    PayloadSchemaType.KEYWORD),
    ("language",      PayloadSchemaType.KEYWORD),
]:
    client.create_payload_index(
        collection_name="rag-prod",
        field_name=field,
        field_schema=schema_type
    )
```

---

# Document Management in VectorDB

## 1. Check if Content Already Exists

### Problem
Avoid ingesting duplicate documents that waste storage and cause retrieval issues.

### Solution: Content-Based Hashing

`hashlib.sha256` converts any text into a **fixed 256-bit fingerprint**.
Same content → same hash. One character change → completely different hash.
Use it to detect duplicates before ingestion.

`SHA - Secure Hash Algorithm`

```python
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def hash_document(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

# "hello world" → b94d27b9...  (always the same)
# "hello World" → 64ec88ca...  (capital W = totally different hash)

def check_document_exists(vectorstore, doc_hash: str) -> bool:
    results = vectorstore.get(where={"doc_hash": doc_hash})
    return len(results['ids']) > 0

# Usage
doc_content = "Your document content..."
doc_hash = hash_document(doc_content)

if check_document_exists(vectorstore, doc_hash):
    print("❌ Document already exists, skipping")
else:
    vectorstore.add_documents([
        Document(
            page_content=doc_content,
            metadata={"doc_hash": doc_hash, "source": "file.pdf"}
        )
    ])
    print("✅ Document added")
```

### Alternative: Query by Metadata

```python
def document_exists_by_source(vectorstore, source: str) -> bool:
    """Check if document from this source exists"""
    results = vectorstore.get(where={"source": source})
    return len(results['ids']) > 0

# Check before ingestion
if document_exists_by_source(vectorstore, "report_2024.pdf"):
    print("Document already indexed")
```

---

## 2. Upsert Operations

### What is Upsert?

**Upsert** = Update if exists, Insert if new

Most vector databases support upsert through **IDs**.

### ChromaDB Upsert

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embeddings
)

# Upsert by ID
doc_id = "doc_123"
new_content = "Updated content"

vectorstore.add_documents(
    documents=[Document(page_content=new_content)],
    ids=[doc_id]  # Same ID = updates existing
)

# If doc_123 exists → Updates it
# If doc_123 doesn't exist → Creates it
```

### Pinecone Upsert

```python
from langchain_pinecone import PineconeVectorStore
import pinecone

vectorstore = PineconeVectorStore.from_documents(
    documents,
    embeddings,
    index_name="my-index"
)

# Upsert with IDs
vectorstore.add_documents(
    documents=[Document(page_content="New content")],
    ids=["doc_123"]  # Automatically upserts
)
```

### FAISS (No Native Upsert)

FAISS doesn't support upsert - you must delete and re-add:

```python
from langchain_community.vectorstores import FAISS

# Workaround: Delete by ID, then add
def upsert_faiss(vectorstore, doc_id: str, new_doc: Document):
    # FAISS doesn't have delete by ID
    # Option 1: Track IDs separately and rebuild
    # Option 2: Use Chroma/Pinecone for upsert support
    pass

# Better: Use a VectorDB with upsert support
```

### Comparison

| VectorDB | Upsert Support | Method |
|----------|----------------|--------|
| **ChromaDB** | ✅ Yes | Same ID = update |
| **Pinecone** | ✅ Yes | Automatic |
| **Qdrant** | ✅ Yes | PUT endpoint |
| **Weaviate** | ✅ Yes | Automatic |
| **FAISS** | ❌ No | Manual delete + add |
| **Milvus** | ✅ Yes | Primary key |

---

## 3. Prevent Duplicate Ingestion

### Strategy 1: Content-Based Deduplication

```python
import hashlib
from typing import List
from langchain.schema import Document

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Remove duplicate documents based on content hash"""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        # Generate hash from content
        doc_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            # Store hash in metadata
            doc.metadata["doc_hash"] = doc_hash
            unique_docs.append(doc)

    print(f"Original: {len(documents)}, Unique: {len(unique_docs)}")
    return unique_docs

# Usage
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./docs", glob="**/*.pdf")
all_docs = loader.load()

# Remove duplicates before ingestion
unique_docs = deduplicate_documents(all_docs)

vectorstore.add_documents(unique_docs)
```

### Strategy 2: Metadata-Based Tracking

```python
from datetime import datetime

class DocumentIngestionTracker:
    """Track ingested documents to prevent duplicates"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def is_ingested(self, source: str, last_modified: datetime) -> bool:
        """Check if document version is already ingested"""
        results = self.vectorstore.get(
            where={
                "source": source,
                "last_modified": last_modified.isoformat()
            }
        )
        return len(results['ids']) > 0

    def ingest_if_new(self, doc: Document, source: str, last_modified: datetime):
        """Ingest only if document is new or modified"""
        if self.is_ingested(source, last_modified):
            print(f"⏭️  Skipping {source} (already ingested)")
            return False

        # Add with tracking metadata
        doc.metadata.update({
            "source": source,
            "last_modified": last_modified.isoformat(),
            "ingested_at": datetime.now().isoformat()
        })

        self.vectorstore.add_documents([doc])
        print(f"✅ Ingested {source}")
        return True

# Usage
import os

tracker = DocumentIngestionTracker(vectorstore)

for file_path in glob.glob("./docs/**/*.pdf"):
    last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

    # Only ingest if new or updated
    tracker.ingest_if_new(
        doc=Document(page_content=load_pdf(file_path)),
        source=file_path,
        last_modified=last_modified
    )
```

### Strategy 3: Batch Deduplication with Similarity

```python
from langchain_openai import OpenAIEmbeddings

def remove_near_duplicates(
    documents: List[Document],
    embeddings: OpenAIEmbeddings,
    similarity_threshold: float = 0.95
) -> List[Document]:
    """Remove documents that are too similar (near-duplicates)"""
    if not documents:
        return []

    # Embed all documents
    doc_embeddings = embeddings.embed_documents(
        [doc.page_content for doc in documents]
    )

    unique_docs = [documents[0]]  # Keep first doc
    unique_embeddings = [doc_embeddings[0]]

    for i in range(1, len(documents)):
        current_embedding = doc_embeddings[i]

        # Check similarity with all unique docs
        is_duplicate = False
        for unique_emb in unique_embeddings:
            # Cosine similarity
            similarity = cosine_similarity([current_embedding], [unique_emb])[0][0]

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(documents[i])
            unique_embeddings.append(current_embedding)

    print(f"Removed {len(documents) - len(unique_docs)} near-duplicates")
    return unique_docs
```

---

## 4. Document Version Management

### Problem
When document v2 replaces v1, you need to:
1. Remove old version
2. Add new version
3. Maintain version history (optional)

### Strategy 1: Replace Old Version (Simple)

```python
def update_document_version(
    vectorstore,
    doc_id: str,
    new_content: str,
    version: str
):
    """Replace old document version with new one"""

    # Step 1: Delete old version (if using ChromaDB)
    vectorstore.delete(ids=[doc_id])

    # Step 2: Add new version with same ID
    vectorstore.add_documents(
        documents=[Document(
            page_content=new_content,
            metadata={
                "version": version,
                "updated_at": datetime.now().isoformat()
            }
        )],
        ids=[doc_id]
    )

    print(f"✅ Updated {doc_id} to version {version}")

# Usage
update_document_version(
    vectorstore,
    doc_id="report_2024",
    new_content="Updated Q2 report...",
    version="2.0"
)
```

### Strategy 2: Maintain Version History

```python
def add_document_version(
    vectorstore,
    source: str,
    content: str,
    version: str
):
    """Add new version while keeping old versions"""

    # Unique ID with version suffix
    doc_id = f"{source}_v{version}"

    # Mark old versions as deprecated
    old_versions = vectorstore.get(
        where={
            "source": source,
            "is_latest": True
        }
    )

    for old_id in old_versions['ids']:
        # Update metadata to mark as old
        vectorstore.update_document(
            id=old_id,
            metadata={"is_latest": False}
        )

    # Add new version
    vectorstore.add_documents(
        documents=[Document(
            page_content=content,
            metadata={
                "source": source,
                "version": version,
                "is_latest": True,
                "created_at": datetime.now().isoformat()
            }
        )],
        ids=[doc_id]
    )

# Retrieve only latest versions
def get_latest_documents(vectorstore, query: str):
    """Retrieve only latest document versions"""
    return vectorstore.similarity_search(
        query,
        filter={"is_latest": True}  # Only latest versions
    )
```

### Strategy 3: Soft Delete with Versioning

```python
class VersionedVectorStore:
    """Wrapper for version-aware document management"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def add_or_update(self, source: str, content: str, version: str):
        """Add new version and deprecate old ones"""

        # Step 1: Mark old versions as deleted
        self._soft_delete_old_versions(source)

        # Step 2: Add new version
        doc_id = f"{source}#{version}"
        self.vectorstore.add_documents(
            documents=[Document(
                page_content=content,
                metadata={
                    "source": source,
                    "version": version,
                    "is_deleted": False,
                    "is_latest": True,
                    "created_at": datetime.now().isoformat()
                }
            )],
            ids=[doc_id]
        )

        print(f"✅ Added {source} v{version}")

    def _soft_delete_old_versions(self, source: str):
        """Mark old versions as deleted without removing"""
        old_docs = self.vectorstore.get(
            where={
                "source": source,
                "is_deleted": False
            }
        )

        for doc_id in old_docs['ids']:
            # Soft delete: just update metadata
            self.vectorstore.update_document(
                id=doc_id,
                metadata={
                    "is_deleted": True,
                    "is_latest": False,
                    "deleted_at": datetime.now().isoformat()
                }
            )

    def search(self, query: str, k: int = 5):
        """Search excluding soft-deleted documents"""
        return self.vectorstore.similarity_search(
            query,
            k=k,
            filter={
                "is_deleted": False,
                "is_latest": True
            }
        )

# Usage
vs = VersionedVectorStore(vectorstore)

# Upload v1
vs.add_or_update("policy.pdf", "Old policy...", "1.0")

# Upload v2 (automatically deprecates v1)
vs.add_or_update("policy.pdf", "New policy...", "2.0")

# Search returns only v2
results = vs.search("What is the policy?")
```

### Best Practices

| Scenario | Strategy | Approach |
|----------|----------|----------|
| **Simple replacement** | Replace | Delete old, add new with same ID |
| **Need audit trail** | Version history | Keep all versions, mark latest |
| **Compliance/legal** | Soft delete | Never delete, just mark inactive |
| **Frequent updates** | Replace | Avoid version bloat |
| **Critical documents** | Version history | Full change tracking |

### Document Management Workflow

```python
from typing import Optional

class DocumentManager:
    """Complete document lifecycle management"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def ingest(self, file_path: str, version: Optional[str] = None):
        """Smart ingestion with duplicate detection"""

        # 1. Load document
        content = self._load_file(file_path)

        # 2. Generate hash
        doc_hash = hashlib.sha256(content.encode()).hexdigest()

        # 3. Check if exists
        existing = self.vectorstore.get(
            where={"doc_hash": doc_hash}
        )

        if existing['ids']:
            print(f"⏭️  Duplicate detected: {file_path}")
            return "duplicate"

        # 4. Check for previous versions
        source_name = os.path.basename(file_path)
        old_versions = self.vectorstore.get(
            where={"source": source_name}
        )

        if old_versions['ids']:
            # Soft delete old versions
            for old_id in old_versions['ids']:
                self.vectorstore.delete(ids=[old_id])
            print(f"🔄 Replaced old version of {source_name}")

        # 5. Add new document
        doc_id = f"{source_name}#{version or '1.0'}"
        self.vectorstore.add_documents(
            documents=[Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "version": version or "1.0",
                    "doc_hash": doc_hash,
                    "file_path": file_path,
                    "ingested_at": datetime.now().isoformat()
                }
            )],
            ids=[doc_id]
        )

        print(f"✅ Ingested {source_name} v{version or '1.0'}")
        return "success"

# Usage
manager = DocumentManager(vectorstore)

# Initial upload
manager.ingest("report.pdf", version="1.0")

# Update (automatically replaces v1.0)
manager.ingest("report.pdf", version="2.0")

# Duplicate upload (skipped)
manager.ingest("report.pdf", version="2.0")  # Detects duplicate
```

---

# RAG Architectures

## 1. Naive RAG

### Definition
Basic RAG pattern: Retrieve documents based on query similarity, then generate answer using retrieved context. No quality checks or refinement.

### How It Works
1. User query → Embed query
2. Vector search → Retrieve top-k documents
3. Concatenate query + documents → Send to LLM
4. Generate answer

### Example
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
llm = ChatOpenAI(model="gpt-4")

# Naive RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type="stuff"  # Stuff all docs into prompt
)

# Query
answer = qa_chain.invoke("What is machine learning?")
print(answer)
```

**Limitations**: No quality control, retrieves irrelevant docs, no verification

---

## 2. Self-RAG (Self-Reflective RAG)

### Definition
RAG with self-reflection: LLM critiques its own retrieval relevance and answer quality, then decides whether to retrieve more documents or refine the answer.

### How It Works
1. Retrieve documents
2. **Reflection tokens**: LLM evaluates retrieval quality
   - Is retrieval relevant? (yes/no)
   - Is answer supported by context? (yes/no)
3. If low quality → Retrieve again or generate without retrieval
4. Generate final answer with confidence scores


**Benefits**: Self-correcting, higher quality answers, reduces hallucination

**LangGraph Implementation**:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

class SelfRAGState(TypedDict):
    query: str
    documents: List[str]
    context: str
    is_relevant: bool
    answer: str
    is_supported: bool

def retrieve(state: SelfRAGState):
    """Retrieve documents"""
    vectorstore = Chroma.from_documents(documents, embeddings)
    docs = vectorstore.similarity_search(state["query"], k=5)
    return {
        "documents": [doc.page_content for doc in docs],
        "context": "\n".join([doc.page_content for doc in docs])
    }

def check_relevance(state: SelfRAGState):
    """Check if retrieved documents are relevant"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    Query: {state['query']}
    Context: {state['context']}

    Is this context relevant? Answer ONLY 'yes' or 'no'
    """
    result = llm.invoke(prompt).content.strip().lower()
    return {"is_relevant": result == "yes"}

def generate_answer(state: SelfRAGState):
    """Generate answer from context"""
    llm = ChatOpenAI(model="gpt-4")
    if state["is_relevant"]:
        prompt = f"Context: {state['context']}\n\nQuestion: {state['query']}\n\nAnswer:"
    else:
        prompt = f"Answer based on your knowledge: {state['query']}"

    answer = llm.invoke(prompt).content
    return {"answer": answer}

def check_support(state: SelfRAGState):
    """Verify if answer is supported by context"""
    if not state["is_relevant"]:
        return {"is_supported": False}

    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    Context: {state['context']}
    Answer: {state['answer']}

    Is the answer supported by context? Answer 'yes' or 'no'
    """
    result = llm.invoke(prompt).content.strip().lower()
    return {"is_supported": result == "yes"}

# Build graph
workflow = StateGraph(SelfRAGState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("generate", generate_answer)
workflow.add_node("verify", check_support)

# Add edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "check_relevance")
workflow.add_edge("check_relevance", "generate")
workflow.add_edge("generate", "verify")
workflow.add_edge("verify", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"query": "What is quantum computing?"})
print(f"Answer: {result['answer']}")
print(f"Supported: {result['is_supported']}")
```

---

## 3. Corrective RAG (CRAG)

### Definition
RAG with external knowledge correction: Evaluates retrieved documents, filters irrelevant ones, and uses web search as fallback for missing information.

### How It Works
1. Retrieve documents
2. **Relevance grading**: Score each document (relevant/irrelevant)
3. **Decision**:
   - All relevant → Generate answer
   - Some irrelevant → Filter + use relevant ones
   - All irrelevant → Fallback to web search
4. Generate answer with corrected knowledge


**Benefits**: Reduces irrelevant context, adaptive knowledge sources, higher accuracy

**LangGraph Implementation**:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

class CRAGState(TypedDict):
    query: str
    documents: List[str]
    relevant_docs: List[str]
    answer: str
    source: Literal["retrieval", "web", "filtered"]

def retrieve_documents(state: CRAGState):
    """Retrieve documents from vector store"""
    vectorstore = Chroma.from_documents(documents, embeddings)
    docs = vectorstore.similarity_search(state["query"], k=5)
    return {"documents": [doc.page_content for doc in docs]}

def grade_documents(state: CRAGState):
    """Grade each document for relevance"""
    llm = ChatOpenAI(model="gpt-4")
    relevant_docs = []

    for doc in state["documents"]:
        prompt = f"""
        Query: {state['query']}
        Document: {doc}

        Is this relevant? Answer 'yes' or 'no'
        """
        result = llm.invoke(prompt).content.strip().lower()
        if result == "yes":
            relevant_docs.append(doc)

    return {"relevant_docs": relevant_docs}

def decide_source(state: CRAGState) -> Literal["web_search", "generate"]:
    """Decide whether to use web search or generate from docs"""
    if len(state["relevant_docs"]) == 0:
        return "web_search"
    return "generate"

def web_search(state: CRAGState):
    """Fallback to web search"""
    search = DuckDuckGoSearchRun()
    web_results = search.run(state["query"])
    return {
        "relevant_docs": [web_results],
        "source": "web"
    }

def generate_answer(state: CRAGState):
    """Generate answer from relevant documents"""
    llm = ChatOpenAI(model="gpt-4")
    context = "\n".join(state["relevant_docs"])

    prompt = f"""
    Context: {context}
    Question: {state['query']}

    Provide a clear answer.
    """
    answer = llm.invoke(prompt).content

    # Determine source
    if state.get("source") == "web":
        source = "web"
    elif len(state["relevant_docs"]) < len(state["documents"]):
        source = "filtered"
    else:
        source = "retrieval"

    return {"answer": answer, "source": source}

# Build graph
workflow = StateGraph(CRAGState)

# Add nodes
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate_answer)

# Add edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")

# Conditional edge: web search or generate
workflow.add_conditional_edges(
    "grade",
    decide_source,
    {
        "web_search": "web_search",
        "generate": "generate"
    }
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"query": "Latest AI news 2024"})
print(f"Answer: {result['answer']}")
print(f"Source: {result['source']}")
```

---

## 4. Agentic RAG

### Definition
RAG with autonomous agents: LLM acts as an agent with tools (retrieval, search, calculator) and decides which tools to use, when to retrieve, and how to solve complex queries.

### How It Works
1. Query → Agent analyzes task
2. **Tool selection**: Agent chooses tools (vector DB, web search, code execution)
3. **Multi-step reasoning**: Iterative retrieval and synthesis
4. **Self-correction**: Re-retrieve if needed
5. Generate final answer

### Example
```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
vectorstore = Chroma.from_documents(documents, embeddings)

# Create retriever tool
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(),
    name="knowledge_base",
    description="Search internal company knowledge base for information about products, policies, and procedures"
)

# Web search tool
web_search_tool = DuckDuckGoSearchRun()

# Create agent with multiple tools
tools = [retriever_tool, web_search_tool]
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Query - Agent decides which tools to use
query = "Compare our product pricing with industry standards in 2024"

result = agent_executor.invoke({"input": query})
print(result["output"])

# Agent automatically:
# 1. Uses knowledge_base to get company pricing
# 2. Uses web_search to find industry standards
# 3. Synthesizes both sources
# 4. Provides comprehensive comparison
```

**Advanced: Multi-Query Agentic RAG**
```python
from langchain.agents import Tool
from langchain.prompts import PromptTemplate

def agentic_rag_multi_query(query, vectorstore, llm):
    # Agent generates multiple search queries
    multi_query_prompt = f"""
    Generate 3 different search queries to thoroughly answer: {query}
    Return as comma-separated list.
    """
    queries = llm.invoke(multi_query_prompt).content.split(',')

    # Agent retrieves for each query
    all_docs = []
    for q in queries:
        docs = vectorstore.similarity_search(q.strip(), k=3)
        all_docs.extend(docs)

    # Agent deduplicates and synthesizes
    unique_content = list(set([doc.page_content for doc in all_docs]))
    context = "\n".join(unique_content)

    # Generate comprehensive answer
    answer = llm.invoke(f"Context: {context}\n\nQuestion: {query}\n\nAnswer:").content
    return answer
```

**Benefits**: Autonomous decision-making, handles complex queries, multi-tool orchestration

---

## 5. Graph RAG

### Definition
RAG using knowledge graphs: Stores information as entities and relationships in a graph database, enabling structured reasoning and relationship-based retrieval.

### How It Works
1. **Build graph**: Extract entities and relationships from documents
2. **Query understanding**: Identify entities in query
3. **Graph traversal**: Navigate relationships to find relevant subgraphs
4. **Context assembly**: Combine graph paths + original text
5. Generate answer with structured knowledge

### Example
```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Setup Neo4j graph database
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Build knowledge graph from documents
def build_knowledge_graph(documents, graph):
    for doc in documents:
        # Extract entities and relationships (simplified)
        extract_prompt = f"""
        Extract entities and relationships from this text:
        {doc.page_content}

        Format:
        ENTITY: [name, type]
        RELATIONSHIP: [entity1, relationship, entity2]
        """
        llm = ChatOpenAI(model="gpt-4")
        entities_rels = llm.invoke(extract_prompt).content

        # Create graph nodes and edges
        # (Simplified - real implementation would parse and create)
        graph.query(f"CREATE (n:Document {{content: '{doc.page_content}'}})")

# Graph RAG query
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# Query translates to Cypher (graph query language)
result = chain.invoke({"query": "Who is the CEO of Company X?"})
print(result["result"])

# Behind the scenes:
# 1. LLM converts question → Cypher query
# 2. MATCH (p:Person)-[:CEO_OF]->(c:Company {name: 'Company X'})
# 3. Returns structured result
# 4. LLM formats natural language answer
```

**Simplified Graph RAG without Neo4j**
```python
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

# Build simple graph structure
llm = ChatOpenAI(model="gpt-4")
graph_transformer = LLMGraphTransformer(llm=llm)

# Convert documents to graph
graph_documents = graph_transformer.convert_to_graph_documents(documents)

# Store both graph relationships + vector embeddings
for graph_doc in graph_documents:
    print(f"Entities: {graph_doc.nodes}")
    print(f"Relationships: {graph_doc.relationships}")

# Query uses both structure and semantics
def graph_rag_query(query, graph_docs, vectorstore):
    # 1. Semantic search
    similar_docs = vectorstore.similarity_search(query, k=3)

    # 2. Find related entities in graph
    related_entities = []
    for doc in similar_docs:
        for graph_doc in graph_docs:
            if doc.page_content in graph_doc.source.page_content:
                related_entities.extend(graph_doc.nodes)

    # 3. Combine semantic + structural context
    context = f"Documents: {similar_docs}\nRelated Entities: {related_entities}"

    return llm.invoke(f"Context: {context}\n\nQuestion: {query}").content
```

**Benefits**: Structured reasoning, relationship traversal, better for complex queries with dependencies

---

## 6. Multi-Modal RAG

### Definition
RAG supporting multiple data types: text, images, tables, charts. Retrieves and processes different modalities to answer queries.

### How It Works
1. Index multiple modalities (text, images, tables)
2. Query → Identify needed modality
3. Retrieve relevant items (text chunks, images, tables)
4. **Multi-modal LLM** processes all modalities
5. Generate comprehensive answer

### Example
```python
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import base64

# Multi-modal LLM (GPT-4V)
llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)

# Index text + image metadata
def index_multimodal_docs(documents, images):
    # Text embeddings
    text_vectorstore = Chroma.from_documents(
        documents,
        OpenAIEmbeddings()
    )

    # Image descriptions (for retrieval)
    image_docs = []
    for img_path, description in images:
        image_docs.append(Document(
            page_content=description,
            metadata={"type": "image", "path": img_path}
        ))

    image_vectorstore = Chroma.from_documents(
        image_docs,
        OpenAIEmbeddings()
    )

    return text_vectorstore, image_vectorstore

def multimodal_rag(query, text_vectorstore, image_vectorstore, llm):
    # Retrieve text
    text_docs = text_vectorstore.similarity_search(query, k=3)
    text_context = "\n".join([doc.page_content for doc in text_docs])

    # Retrieve images
    image_docs = image_vectorstore.similarity_search(query, k=2)

    # Encode images to base64
    image_data = []
    for doc in image_docs:
        if doc.metadata.get("type") == "image":
            with open(doc.metadata["path"], "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
                image_data.append(img_base64)

    # Multi-modal prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context: {text_context}\n\nQuestion: {query}"}
            ] + [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in image_data
            ]
        }
    ]

    response = llm.invoke(messages)
    return response.content

# Usage
images = [
    ("chart1.png", "Sales chart showing Q1-Q4 revenue"),
    ("diagram.png", "System architecture diagram")
]

text_vs, image_vs = index_multimodal_docs(documents, images)
answer = multimodal_rag("What were Q2 sales?", text_vs, image_vs, llm)
print(answer)
```

**Benefits**: Handles rich content (PDFs with images, charts), comprehensive context, visual understanding

---

## 7. Adaptive RAG

### Definition
RAG that dynamically chooses strategy based on query complexity: routes simple queries to basic retrieval, complex queries to agentic/multi-step RAG.

### How It Works
1. **Query classification**: Analyze query complexity
2. **Routing decision**:
   - Simple factual → Naive RAG
   - Needs verification → Self-RAG
   - Needs external knowledge → CRAG
   - Multi-step reasoning → Agentic RAG
3. Execute appropriate strategy
4. Generate answer

### Example
```python
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from enum import Enum

class QueryType(Enum):
    SIMPLE = "simple"
    VERIFICATION_NEEDED = "verification"
    EXTERNAL_KNOWLEDGE = "external"
    COMPLEX = "complex"

def classify_query(query, llm):
    """Classify query complexity"""
    prompt = f"""
    Classify this query complexity:
    Query: {query}

    Categories:
    - SIMPLE: Direct factual question from knowledge base
    - VERIFICATION: Needs fact-checking or source verification
    - EXTERNAL: May need external/current information
    - COMPLEX: Multi-step reasoning or comparison needed

    Answer with ONE category only.
    """
    result = llm.invoke(prompt).content.strip().upper()
    return QueryType[result]

def adaptive_rag(query, vectorstore, llm):
    # Step 1: Classify query
    query_type = classify_query(query, llm)
    print(f"Query Type: {query_type.value}")

    # Step 2: Route to appropriate strategy
    if query_type == QueryType.SIMPLE:
        # Naive RAG
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
        answer = llm.invoke(f"Context: {context}\n\nQuestion: {query}").content
        strategy = "Naive RAG"

    elif query_type == QueryType.VERIFICATION_NEEDED:
        # Self-RAG with verification
        answer = self_rag(query, vectorstore, llm)["answer"]
        strategy = "Self-RAG"

    elif query_type == QueryType.EXTERNAL_KNOWLEDGE:
        # Corrective RAG with web search
        from langchain_community.tools import DuckDuckGoSearchRun
        web_search = DuckDuckGoSearchRun()
        answer = corrective_rag(query, vectorstore, llm, web_search)["answer"]
        strategy = "Corrective RAG"

    else:  # COMPLEX
        # Agentic RAG
        answer = agentic_rag_multi_query(query, vectorstore, llm)
        strategy = "Agentic RAG"

    return {
        "answer": answer,
        "strategy_used": strategy,
        "query_type": query_type.value
    }

# Usage
llm = ChatOpenAI(model="gpt-4")
vectorstore = Chroma.from_documents(documents, embeddings)

# Simple query → Naive RAG
result1 = adaptive_rag("What is the capital of France?", vectorstore, llm)
print(f"Strategy: {result1['strategy_used']}")

# Complex query → Agentic RAG
result2 = adaptive_rag(
    "Compare our Q1 performance with competitors and predict Q2 trends",
    vectorstore,
    llm
)
print(f"Strategy: {result2['strategy_used']}")
```

**Benefits**: Efficient resource usage, optimal strategy per query, cost-effective

**LangGraph Implementation**:
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

class AdaptiveRAGState(TypedDict):
    query: str
    query_type: Literal["simple", "verification", "external", "complex"]
    documents: List[str]
    context: str
    answer: str
    strategy: str

def classify_query(state: AdaptiveRAGState):
    """Classify query complexity"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"""
    Classify this query:
    {state['query']}

    Categories:
    - simple: Direct factual question
    - verification: Needs fact-checking
    - external: Needs current/external info
    - complex: Multi-step reasoning

    Answer with ONE word only.
    """
    result = llm.invoke(prompt).content.strip().lower()
    return {"query_type": result}

def route_query(state: AdaptiveRAGState) -> str:
    """Route to appropriate RAG strategy"""
    return state["query_type"]

def naive_rag(state: AdaptiveRAGState):
    """Simple retrieval + generation"""
    vectorstore = Chroma.from_documents(documents, embeddings)
    docs = vectorstore.similarity_search(state["query"], k=3)
    context = "\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4")
    answer = llm.invoke(f"Context: {context}\n\nQ: {state['query']}\n\nA:").content

    return {
        "answer": answer,
        "strategy": "Naive RAG",
        "context": context
    }

def self_rag_strategy(state: AdaptiveRAGState):
    """Self-RAG with verification"""
    vectorstore = Chroma.from_documents(documents, embeddings)
    docs = vectorstore.similarity_search(state["query"], k=5)
    context = "\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4")

    # Check relevance
    relevance_check = llm.invoke(
        f"Is this context relevant to '{state['query']}'? Answer yes/no\n\n{context}"
    ).content.strip().lower()

    if relevance_check == "yes":
        answer = llm.invoke(f"Context: {context}\n\nQ: {state['query']}\n\nA:").content
    else:
        answer = llm.invoke(f"Answer: {state['query']}").content

    return {
        "answer": answer,
        "strategy": "Self-RAG",
        "context": context
    }

def corrective_rag_strategy(state: AdaptiveRAGState):
    """CRAG with web search fallback"""
    from langchain_community.tools import DuckDuckGoSearchRun

    vectorstore = Chroma.from_documents(documents, embeddings)
    docs = vectorstore.similarity_search(state["query"], k=5)

    llm = ChatOpenAI(model="gpt-4")
    relevant_docs = []

    # Grade documents
    for doc in docs:
        grade = llm.invoke(
            f"Is this relevant to '{state['query']}'?\n\n{doc.page_content}\n\nAnswer yes/no"
        ).content.strip().lower()
        if grade == "yes":
            relevant_docs.append(doc.page_content)

    # Fallback to web if no relevant docs
    if len(relevant_docs) == 0:
        search = DuckDuckGoSearchRun()
        web_results = search.run(state["query"])
        context = web_results
    else:
        context = "\n".join(relevant_docs)

    answer = llm.invoke(f"Context: {context}\n\nQ: {state['query']}\n\nA:").content

    return {
        "answer": answer,
        "strategy": "Corrective RAG",
        "context": context
    }

def agentic_rag_strategy(state: AdaptiveRAGState):
    """Agentic RAG with multi-query"""
    llm = ChatOpenAI(model="gpt-4")

    # Generate multiple queries
    multi_query = llm.invoke(
        f"Generate 3 search queries for: {state['query']}\nComma-separated:"
    ).content.split(',')

    # Retrieve for each
    vectorstore = Chroma.from_documents(documents, embeddings)
    all_docs = []
    for q in multi_query:
        docs = vectorstore.similarity_search(q.strip(), k=2)
        all_docs.extend([d.page_content for d in docs])

    # Deduplicate
    unique_docs = list(set(all_docs))
    context = "\n".join(unique_docs)

    answer = llm.invoke(f"Context: {context}\n\nQ: {state['query']}\n\nA:").content

    return {
        "answer": answer,
        "strategy": "Agentic RAG",
        "context": context
    }

# Build graph
workflow = StateGraph(AdaptiveRAGState)

# Add nodes
workflow.add_node("classify", classify_query)
workflow.add_node("simple", naive_rag)
workflow.add_node("verification", self_rag_strategy)
workflow.add_node("external", corrective_rag_strategy)
workflow.add_node("complex", agentic_rag_strategy)

# Add edges
workflow.set_entry_point("classify")

# Route based on query type
workflow.add_conditional_edges(
    "classify",
    route_query,
    {
        "simple": "simple",
        "verification": "verification",
        "external": "external",
        "complex": "complex"
    }
)

# All strategies end
workflow.add_edge("simple", END)
workflow.add_edge("verification", END)
workflow.add_edge("external", END)
workflow.add_edge("complex", END)

# Compile
app = workflow.compile()

# Run examples
result1 = app.invoke({"query": "What is the capital of France?"})
print(f"Strategy: {result1['strategy']}")  # → Naive RAG

result2 = app.invoke({"query": "Compare Q1 vs Q2 performance with trends"})
print(f"Strategy: {result2['strategy']}")  # → Agentic RAG
```

**Visualization**:
```python
from IPython.display import Image, display

# Visualize the graph
display(Image(app.get_graph().draw_mermaid_png()))
```

---

## RAG Architectures Comparison

| Architecture | Complexity | Accuracy | Cost | Speed | Best For |
|--------------|------------|----------|------|-------|----------|
| **Naive RAG** | Low | Medium | Low | Fast | Simple Q&A, prototypes |
| **Self-RAG** | Medium | High | Medium | Medium | Quality-critical apps |
| **Corrective RAG** | Medium | High | Medium | Medium | Dynamic knowledge, fact-checking |
| **Agentic RAG** | High | Very High | High | Slow | Complex queries, multi-step reasoning |
| **Graph RAG** | High | Very High | High | Medium | Relationship-heavy queries |
| **Multi-Modal** | High | High | High | Slow | Documents with images/charts |
| **Adaptive RAG** | Medium | High | Medium | Variable | Production systems (handles all) |

### When to Use Which

```
Simple factual Q&A                    → Naive RAG
Need high accuracy + verification     → Self-RAG
External/current knowledge needed     → Corrective RAG
Complex multi-step reasoning          → Agentic RAG
Relationship-based queries            → Graph RAG
PDFs with images/charts               → Multi-Modal RAG
Production (varied query types)       → Adaptive RAG
```

---

# Chunking Strategies

## 1. Fixed-Size Chunking

### Definition
Splits documents into equal-sized chunks with a fixed character/token count and optional overlap between chunks.

### How It Works
- Set chunk size (e.g., 500 characters)
- Set overlap size (e.g., 50 characters for context continuity)
- Split text into fixed-size segments
- Simple, fast, no intelligence needed

### Example
```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=500,        # 500 characters per chunk
    chunk_overlap=50,      # 50 character overlap
    separator="\n"
)

chunks = text_splitter.split_text(document)
# Result: ["chunk1 (500 chars)", "chunk2 (500 chars)", ...]
```

**Use Case**: Simple documents, quick implementation, when semantic boundaries don't matter

---

## 2. Recursive Character Splitting

### Definition
Intelligently splits text by trying separators in order (paragraphs → sentences → words) to respect natural boundaries.

### How It Works
- Try splitting by paragraph first (`\n\n`)
- If chunks too large, split by sentence (`. `)
- If still too large, split by word (` `)
- Preserves document structure better than fixed-size

### Example
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Order matters
)

chunks = text_splitter.split_text(document)
# Respects paragraphs and sentences
```

**Use Case**: General-purpose chunking, maintains context, works for most documents

---

## 3. Document-Specific Chunking

### Definition
Uses document structure (headers, code blocks, markdown) to create meaningful chunks based on content type.

### How It Works
- **Markdown**: Split by headers (`#`, `##`, `###`)
- **Code**: Split by functions, classes
- **HTML**: Split by tags (`<p>`, `<div>`, `<section>`)
- Preserves logical document structure

### Example
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Markdown chunking
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

chunks = markdown_splitter.split_text(markdown_doc)
# Each chunk = content under a header

# Code chunking
from langchain.text_splitter import PythonCodeTextSplitter

python_splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

code_chunks = python_splitter.split_text(python_code)
# Splits by functions/classes
```

**Use Case**: Structured documents (markdown, code, HTML), technical documentation

---

## 4. Semantic Chunking

### Definition
Groups text based on semantic meaning using embeddings - chunks contain semantically similar content.

### How It Works
- Embed sentences/paragraphs
- Calculate semantic similarity between adjacent segments
- Split when similarity drops below threshold
- Creates contextually coherent chunks

### Example
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"  # or "standard_deviation"
)

chunks = text_splitter.split_text(document)
# Each chunk has coherent semantic meaning

# Example:
# Chunk 1: All sentences about "Python basics"
# Chunk 2: All sentences about "Advanced features"
# (even if interleaved in original text)
```

**Use Case**: Complex documents where context matters, research papers, narrative text



## Semantic Chunking for Large PDFs (1000+ pages)

### Problem

**SemanticChunker on huge PDFs** → Crashes or takes hours (calculates embeddings for every split point)


### Solution: Hybrid 2-Stage Approach

**Stage 1: Pre-chunk with RecursiveCharacterTextSplitter (~3000 chars)**
**Stage 2: Semantic chunking within each pre-chunk**

---

### Code Example

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Step 1: Pre-chunk large document
pre_chunker = RecursiveCharacterTextSplitter(
    chunk_size=3000,      # Manageable size for semantic analysis
    chunk_overlap=200
)
pre_chunks = pre_chunker.split_text(pdf_text)

# Step 2: Semantic chunking on each pre-chunk
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
semantic_chunker = SemanticChunker(embedding_function)

final_chunks = []
for pre_chunk in pre_chunks:
    semantic_docs = semantic_chunker.create_documents([pre_chunk])
    final_chunks.extend(semantic_docs)

print(f"Final chunks: {len(final_chunks)}")
```

---

## When to Use What

| Document Size | Strategy | Pre-chunk Size | Semantic? |
|---------------|----------|----------------|-----------|
| **< 50 pages** | Direct semantic | N/A | ✅ Yes |
| **50-500 pages** | Hybrid | 3000 chars | ✅ Yes |
| **500+ pages** | Character only | 1000 chars | ❌ Skip |

**Trade-off**: Hybrid = Better quality boundaries vs Character-only = Faster processing

---

## 5. Agentic Chunking

### Definition
Uses LLMs to intelligently determine chunk boundaries based on content understanding and context.

### How It Works
- LLM analyzes document content
- Identifies logical breakpoints (topic changes, sections)
- Creates chunks based on semantic coherence
- Most intelligent but slowest and most expensive

### Example
```python
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI

# LLM-based chunking with propositions
llm = ChatOpenAI(model="gpt-4")

# Custom agentic chunker
def agentic_chunk(document, llm):
    prompt = f"""
    Split this document into logical chunks.
    Each chunk should cover one main topic/idea.
    Return chunks separated by '---SPLIT---'

    Document: {document}
    """

    response = llm.invoke(prompt)
    chunks = response.content.split('---SPLIT---')
    return [chunk.strip() for chunk in chunks]

chunks = agentic_chunk(document, llm)
# Intelligent, context-aware splits
```

**Use Case**: High-value documents requiring precision, complex multi-topic content, when cost is not a constraint

---

## 6. Hierarchical Chunking (Large Documents)

### Definition
Multi-level chunking strategy for very large documents (1000+ pages) that combines coarse-to-fine splitting to handle size constraints.

### How It Works
**Problem**: Semantic/Agentic chunking can't process 1000-page documents in one LLM call due to context limits.

**Solution**: Hierarchical approach in 3 steps:
1. **Level 1 (Coarse)**: Split large document into manageable sections (chapters, parts)
2. **Level 2 (Medium)**: Apply semantic/recursive chunking within each section
3. **Level 3 (Fine)**: Create parent-child chunk relationships for retrieval

### Example
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# Step 1: Coarse chunking (split 1000-page doc into chapters)
coarse_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50000,      # ~50 pages per chunk
    chunk_overlap=5000,
    separators=["\n\n# ", "\n\n## "]  # Split by headers
)

large_doc = load_1000_page_document()
sections = coarse_splitter.split_text(large_doc)  # ~20 sections

# Step 2: Fine chunking within each section
fine_splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)

all_chunks = []
for i, section in enumerate(sections):
    # Now each section is small enough for semantic chunking
    section_chunks = fine_splitter.split_text(section)

    # Add metadata to track hierarchy
    for chunk in section_chunks:
        all_chunks.append({
            'content': chunk,
            'section_id': i,
            'parent_section': section[:200]  # Store parent context
        })

# Result: Intelligent chunks with hierarchical context
print(f"Total chunks: {len(all_chunks)}")
# ~200-500 semantically coherent chunks from 1000 pages
```

---

## ParentDocumentRetriever — Injection & Retrieval

`ParentDocumentRetriever` solves the Precision vs Recall tradeoff by using **two separate stores** — child chunks in vectorstore for precise semantic search, parent chunks in docstore for full context returned to the LLM.

```
VECTORSTORE (Chroma / Pinecone)          DOCSTORE (Redis / InMemoryStore)
─────────────────────────────────        ────────────────────────────────
child chunks → embedded, searchable      parent chunks → raw text, key-value lookup

child_id  │ embedding  │ parent_id        key       │ value (full text)
──────────┼────────────┼──────────        ──────────┼────────────────────
ch-001    │ [0.2,-0.4] │ abc-001          abc-001   │ full 512-token text
ch-002    │ [0.1, 0.9] │ abc-001          abc-002   │ full 512-token text
ch-003    │ [-0.3,0.7] │ abc-001          abc-003   │ full 512-token text
ch-004    │ [0.5,-0.2] │ abc-002
ch-005    │ [0.8, 0.1] │ abc-002
```

---

### Part 1 — Injection Pipeline (Run Once / On Schedule)

```python
# injection_pipeline.py

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ── Stores (must persist between injection and retrieval) ──────────────────────
embeddings  = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="fraud_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",    # saved to disk — survives restart
)
docstore = InMemoryStore()              # swap to RedisStore in production

# ── Splitters ──────────────────────────────────────────────────────────────────
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=20)

# ── ParentDocumentRetriever — wires both stores together ──────────────────────
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      # child chunks → embedded here
    docstore=docstore,            # parent chunks → raw text here
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ── Raw documents ──────────────────────────────────────────────────────────────
raw_docs = [
    Document(page_content="FinTech Corp uses AI to detect fraud using 400 signals including device fingerprint, transaction history and geolocation.", metadata={"source": "fraud_policy.pdf"}),
    Document(page_content="FinTech Corp checkout flow routes payments through multiple processors. Each transaction is scored in real-time within 200ms.", metadata={"source": "checkout_doc.pdf"}),
]

# ── add_documents() triggers the full injection pipeline ──────────────────────
# Internally:
#   1. Split each doc → parent chunks (512 tokens), assign parent_id
#   2. Save parents → docstore  (raw text, key-value, no embedding)
#   3. Split each parent → child chunks (128 tokens, parent_id in metadata)
#   4. Embed children → save to vectorstore
retriever.add_documents(raw_docs)

print(f"Docstore   — parent chunks: {len(list(docstore.yield_keys()))}")
print(f"Vectorstore — child chunks: {vectorstore._collection.count()}")
# Docstore   — parent chunks: 4
# Vectorstore — child chunks: 16
```

---

### Part 2 — Retrieval Pipeline (Run Per Query)

```python
# retrieval_pipeline.py

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Reconnect to SAME stores used during injection ────────────────────────────
embeddings  = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="fraud_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",    # same path as injection
)
docstore = InMemoryStore()              # production: RedisStore(redis_url=...)

# ── Splitters must match injection config ─────────────────────────────────────
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=20)

# ── Same retriever config — add_documents NOT called here ─────────────────────
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ── retriever.invoke() does the full 2-step lookup automatically ──────────────
#   Step 1: embed query → similarity search vectorstore → matched child chunks
#   Step 2: extract parent_ids from child metadata → deduplicate
#   Step 3: fetch full parent text from docstore by parent_id (key-value, no embedding)
def retrieve_with_debug(query: str):
    print(f"Query: '{query}'\n")

    # Step 1 — vectorstore: semantic search on child chunks only
    children = vectorstore.similarity_search_with_score(query, k=4)
    print("Step 1 — Child chunks matched in vectorstore:")
    for doc, score in children:
        print(f"  score={score:.3f} | parent_id={doc.metadata['doc_id']} | '{doc.page_content[:60]}...'")

    # Step 2 — deduplicate parent_ids
    # multiple children from same parent → fetch parent only ONCE
    parent_ids = list({doc.metadata["doc_id"] for doc, _ in children})
    print(f"\nStep 2 — Unique parent_ids (deduplicated): {parent_ids}")

    # Step 3 — docstore: pure key-value fetch, NO embedding involved
    parents = [p for p in docstore.mget(parent_ids) if p]
    print(f"\nStep 3 — Parent docs fetched from docstore: {len(parents)}")
    for p in parents:
        print(f"  '{p.page_content[:120]}...'")

    return parents

# ── RAG Chain ─────────────────────────────────────────────────────────────────
llm    = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {
        "context":  retriever | (lambda docs: "\n\n---\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

query  = "How does FinTech Corp detect fraud?"
retrieve_with_debug(query)          # debug: see every step
answer = rag_chain.invoke(query)    # full RAG: retrieval + LLM answer
print(f"\nLLM Answer:\n{answer}")
```

---

### What `ParentDocumentRetriever` Does in Each Part

```
INJECTION (add_documents)          RETRIEVAL (invoke)
─────────────────────────          ──────────────────
Uses child_splitter                Does NOT split anything
Uses parent_splitter               Does NOT split anything
Writes to docstore                 Reads from docstore
Writes to vectorstore              Reads from vectorstore
Runs ONCE per document             Runs on EVERY query
```

---

### Production — Two Services, Shared Stores

```
Injection Service (batch job / cron)    Retrieval Service (FastAPI)
────────────────────────────────────    ──────────────────────────
runs nightly or on doc upload           runs per user query

retriever.add_documents(docs)           retriever.invoke(query)
      │                                       │
      ├─ writes → Chroma (disk/cloud)  ───────┤ reads → Chroma
      └─ writes → Redis docstore       ───────┘ reads → Redis

Both services connect to the SAME Chroma + Redis
Injection writes, Retrieval reads — no conflict
```

### Production Docstore by Scale

| Scale | Docstore | Why |
|---|---|---|
| Dev / small | `InMemoryStore` | Simple, no setup — lost on restart |
| Medium (10K–100K docs) | Redis | Fast, persistent, TTL support |
| Large (100K+ docs) | MongoDB / PostgreSQL | Scalable, queryable, durable |

---

**Best Practices for 1000-Page Documents**:
1. **Level 1**: Use document structure (chapters) → 20-50 large parent sections
2. **Level 2**: Split each section into children → 10-20 child chunks per parent
3. **Store metadata**: Section number, page range, source in both stores
4. **Use parent-child retrieval**: Child for precision, parent for full context

---

## 7. Tabular Data Chunking

### The Problem with Tables

Standard text splitters break tables mid-row, destroying structure and making retrieval useless:

```
# What a naive splitter produces:
chunk_1: "Product | Price | St"
chunk_2: "ock\nWidget A | $10 | 5"
chunk_3: "00\nWidget B | $20 | 200"
```

A vector search for "Widget A price" gets meaningless fragments.

The solution: use a tool that **detects table boundaries**, extracts the full table, converts it to markdown, and stores it as a **single chunk**.

---

### Tools and Techniques for Table Extraction

| Tool / Library | Input Format | Table Detection Method | Output Format | Notes |
|---|---|---|---|---|
| **pdfplumber** | PDF | Heuristic (line detection) | DataFrame → `.to_markdown()` | Best for native PDFs with ruled lines |
| **camelot** | PDF | Lattice (grid lines) or Stream (whitespace) | DataFrame → `.to_markdown()` | High accuracy, two modes |
| **tabula-py** | PDF | Java-based line detection | DataFrame → `.to_markdown()` | Wraps Tabula-Java, good for dense tables |
| **pymupdf (fitz)** | PDF | Block-level layout analysis | Raw dict → manual markdown | Fastest; needs custom markdown builder |
| **Unstructured.io** | PDF, DOCX, HTML, PPTX | ML-based layout detection | `Table` element with `.text` | Handles mixed documents; single pipeline |
| **LlamaParse** | PDF, DOCX | Vision LLM + layout model | Markdown natively | Best quality; handles complex/merged cells |
| **Azure Document Intelligence** | PDF, image, DOCX | ML OCR + table detection | JSON with cell coords → markdown | Cloud; handles scanned docs |
| **AWS Textract** | PDF, image | ML OCR + table detection | JSON → markdown | Cloud; strong on scanned/handwritten |
| **pandas `read_html()`** | HTML | HTML `<table>` tag parsing | DataFrame → `.to_markdown()` | Only for HTML source |
| **Beautiful Soup** | HTML | CSS selector on `<table>` | DataFrame → `.to_markdown()` | Custom HTML parsing |

---

### How Each Tool Keeps the Table as One Chunk

Every tool above extracts the table as a **complete unit** (DataFrame or element). You then convert to markdown and store as a single chunk:

```python
# pdfplumber / camelot / tabula — all return a DataFrame
markdown_table = df.to_markdown(index=False)

chunk = {
    "chunk_id":   f"{doc_id}_table_{page}_{idx}",
    "chunk_text": f"Table on page {page}:\n\n{markdown_table}",
    "chunk_type": "table",
    "source":     doc_id,
    "page":       page,
    "columns":    df.columns.tolist(),
    "row_count":  len(df),
}
```

For Unstructured.io, `Table` elements are already isolated — call `.text` or serialize directly:

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("doc.pdf", strategy="hi_res")
for el in elements:
    if el.category == "Table":
        chunk = {
            "chunk_id":   f"{doc_id}_table_{el.id}",
            "chunk_text": el.text,          # already clean text/markdown
            "chunk_type": "table",
            "source":     doc_id,
            "page":       el.metadata.page_number,
        }
```

---

### Which Tool to Pick

| Scenario | Recommended Tool |
|---|---|
| Native PDF (digital, with grid lines) | **camelot** (lattice mode) or **pdfplumber** |
| Native PDF (no grid lines, whitespace-only) | **camelot** (stream mode) or **tabula-py** |
| Scanned PDF / image-based | **Azure Document Intelligence** or **AWS Textract** |
| Mixed document (text + tables + images) | **Unstructured.io** or **LlamaParse** |
| HTML page | **pandas `read_html()`** or **Beautiful Soup** |
| DOCX / PPTX | **Unstructured.io** or **python-docx** |
| Highest accuracy (complex/merged cells) | **LlamaParse** (vision LLM-based) |


---

## 8. Image Chunking (Multi-modal Documents)

### The Problem

Standard text splitters ignore images entirely. A PDF with diagrams, charts, or screenshots loses all visual information when only text is extracted.

---

### Strategy 1 — Extract + Caption with Vision LLM

Use a vision model (GPT-4o, Claude, Gemini) to generate a text description of each image, then store that description as a text chunk.

```python
import base64
from pathlib import Path

def image_to_caption(image_path: str, llm_vision) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Describe this image in detail for a search index. "
        "Include: what it shows, key labels, data values if a chart, "
        "any text visible, and what question it answers."
    )
    response = llm_vision.invoke([
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": prompt},
    ])
    return response.content


def chunk_image(image_path: str, doc_id: str, page: int, llm_vision) -> dict:
    caption = image_to_caption(image_path, llm_vision)
    return {
        "chunk_id":    f"{doc_id}_img_p{page}",
        "chunk_text":  caption,
        "chunk_type":  "image",
        "image_path":  image_path,       # store path for retrieval display
        "source":      doc_id,
        "page":        page,
    }
```

**What gets embedded:** The caption text → searchable like any other chunk.
**What gets returned to LLM:** Caption text + optionally the image itself (if multi-modal LLM).

---

### Strategy 2 — Multi-modal Embeddings (CLIP / ColPali)

Embed images directly as vectors using a multi-modal embedding model, without converting to text first.

```python
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path: str) -> list[float]:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding[0].tolist()


def chunk_image_multimodal(image_path: str, doc_id: str, page: int) -> dict:
    return {
        "chunk_id":   f"{doc_id}_img_p{page}",
        "embedding":  embed_image(image_path),   # image vector
        "chunk_type": "image",
        "image_path": image_path,
        "source":     doc_id,
        "page":       page,
    }
```

> CLIP embeds both text and images into the **same vector space**, so a text query can retrieve images directly.

---

### Strategy 3 — Anchor Image to Surrounding Text

Keep the image description as part of the text chunk that surrounds it in the document, preserving context.

```python
def extract_page_with_images(page_text: str, image_captions: list[str]) -> str:
    """
    Merge surrounding text + image captions into one chunk.
    Keeps image semantically anchored to its context.
    """
    image_block = "\n".join(
        f"[IMAGE: {caption}]" for caption in image_captions
    )
    return f"{page_text}\n\n{image_block}"
```

**Best for:** Diagrams that only make sense with the surrounding paragraph (architecture diagrams, annotated screenshots).

---

### What to Store for Image Chunks

```python
{
    "chunk_id":    "report_img_p5",
    "chunk_text":  "Bar chart showing Q1 revenue by region...",  # caption
    "chunk_type":  "image",
    "image_path":  "s3://bucket/doc/page_5_img_1.png",  # original for display
    "source":      "annual_report_2024",
    "page":        5,
    "image_index": 1,     # nth image on this page
    "alt_text":    "Q1 Revenue Chart",   # from PDF metadata if available
}
```

---

### Full Pipeline: PDF with Mixed Content

```
PDF Page
  ├── Text block → RecursiveCharacterTextSplitter → text chunks
  ├── Table → Keep whole → table chunk (markdown or NL)
  └── Image → Vision LLM caption → image chunk

All chunks → embed → store in vector DB
```

```python
def process_mixed_page(page, doc_id: str, page_num: int, llm_vision, llm) -> list[dict]:
    chunks = []

    # 1. Text chunks
    if page.text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for i, text in enumerate(splitter.split_text(page.text)):
            chunks.append({
                "chunk_id":   f"{doc_id}_p{page_num}_text_{i}",
                "chunk_text": text,
                "chunk_type": "text",
                "source":     doc_id,
                "page":       page_num,
            })

    # 2. Table chunks
    for i, table_df in enumerate(page.tables):
        table_id = f"{doc_id}_p{page_num}_table_{i}"
        chunks.append(table_to_chunk(table_df, table_id, doc_id))

    # 3. Image chunks
    for i, image_path in enumerate(page.images):
        chunks.append(chunk_image(image_path, doc_id, page_num, llm_vision))

    return chunks
```

---

### Decision Matrix for Images

| Scenario | Strategy |
|---|---|
| Chart / diagram with labels | Vision LLM caption |
| Screenshot with text | Vision LLM caption (OCR-aware) |
| Product photo / figure | CLIP multi-modal embedding |
| Diagram tied to surrounding text | Anchor to text chunk |
| Need to return image to user | Store `image_path` in metadata |
| No vision LLM available | Extract alt-text / PDF metadata |

---

## 9. Chunk Size Impact on Context Precision & Recall

Chunk size directly controls the tradeoff between **Context Precision** (did we retrieve junk?) and **Context Recall** (did we miss anything?).

```
Context Precision  =  of all chunks retrieved, how many are actually relevant?
Context Recall     =  of all relevant info that EXISTS, how much did we retrieve?
```

---

### Chunk Too Small

```
Document: "FinTech Corp uses AI to detect fraud. The system analyzes 400 signals
           including device fingerprint, transaction history, and geolocation."

Small chunks:
  chunk_1: "FinTech Corp uses AI to detect fraud."
  chunk_2: "The system analyzes 400 signals"
  chunk_3: "including device fingerprint,"
  chunk_4: "transaction history, and geolocation."

Query: "How does FinTech Corp detect fraud?"
Retrieved: chunk_1, chunk_2   ← top-2 by similarity
Missed:    chunk_3, chunk_4   ← relevant but scored low in isolation
```

| Metric | Result | Why |
|---|---|---|
| Context Precision | ✅ High | Retrieved chunks are relevant, no junk |
| Context Recall | ❌ Low | Answer split across chunks — retriever missed half |

**Other problems:** one idea split across many chunks → retriever must get ALL or answer is incomplete. More chunks in index → more noise in top-K results → LLM gets fragmented context.

---

### Chunk Too Large

```
Document: "FinTech Corp fraud detection... [500 words]
           FinTech Corp checkout flow... [500 words]
           FinTech Corp merchant fees... [500 words]"

Large chunk: all 1500 words merged

Query: "How does FinTech Corp detect fraud?"
Retrieved: chunk_1 ← contains the answer but also 1000 words of irrelevant content
```

| Metric | Result | Why |
|---|---|---|
| Context Precision | ❌ Low | Chunk retrieved but mostly irrelevant content inside |
| Context Recall | ✅ High | Answer is fully contained |

**Other problems:** embedding of 1500-word chunk is a blurry average — semantic signal diluted. LLM suffers "lost in the middle" — relevant part buried in noise. Token cost spikes.

---

### The Tradeoff

| | Small Chunks | Large Chunks |
|---|---|---|
| Context Precision | ✅ High | ❌ Low |
| Context Recall | ❌ Low | ✅ High |
| Embedding quality | ✅ Focused signal | ❌ Diluted/blurry |
| LLM focus | ✅ Tight context | ❌ Lost in the middle |
| Token cost | ✅ Low | ❌ High |
| Complete answer | ❌ Answer split | ✅ Fully contained |

---

### Production Fix — Parent-Child Chunking

Index small chunks for **precise retrieval**, return large parent for **complete context**:

```
Parent chunk (512 tokens) — stored, returned to LLM
  ├── Child chunk 1 (128 tokens) — embedded, used for retrieval
  ├── Child chunk 2 (128 tokens) — embedded, used for retrieval
  └── Child chunk 3 (128 tokens) — embedded, used for retrieval

Query hits child chunk 2 (high precision match)
       ↓
Retrieve its parent (complete context, high recall)
       ↓
LLM gets focused but complete answer

Context Precision → HIGH  (child embedding is focused)
Context Recall    → HIGH  (parent contains the full idea)
```

This is why `ParentDocumentRetriever` in LangChain exists — it solves exactly this tradeoff.

---

### Optimal Chunk Size (Rule of Thumb)

| Content type | Chunk size |
|---|---|
| Dense technical docs, legal, policy | 256–512 tokens |
| General knowledge, articles | 512–1024 tokens |
| Code | Per function / class boundary |
| Tables | One table = one chunk |
| Parent-child setup | Child: 128–256 / Parent: 512–1024 |

> No universal best chunk size — evaluate with Context Precision + Recall on your actual data using DeepEval or RAGAS.

---

## Chunking Strategies Comparison

| Strategy | Intelligence | Speed | Cost | Best For | Doc Size Limit |
|----------|-------------|-------|------|----------|----------------|
| **Fixed-Size** | Low | Very Fast | Free | Simple docs, quick setup | Unlimited |
| **Recursive** | Medium | Fast | Free | General-purpose, most documents | Unlimited |
| **Document-Specific** | Medium | Fast | Free | Structured content (code, markdown) | Unlimited |
| **Semantic** | High | Medium | Low (embeddings) | Context-critical content | <10MB/doc |
| **Agentic** | Very High | Slow | High (LLM calls) | Complex, high-value documents | <100KB/doc |
| **Hierarchical** | High | Medium | Low-Medium | Very large docs (100+ pages) | Unlimited |
| **Tabular** | Medium | Fast | Free–Low | Tables, spreadsheets, structured data | Unlimited |
| **Image** | High | Medium | Medium (vision LLM) | PDFs with diagrams, charts, screenshots | Per image |

### Choosing a Strategy

```
START: Need to chunk documents for RAG
  |
  ├─ Document > 100 pages (or > 10MB)?
  |    └─ Yes → Hierarchical Chunking (coarse → fine)
  |
  ├─ Document contains tables?
  |    └─ Yes → Tabular Chunking (keep table as unit, row-level for large tables)
  |
  ├─ Document contains images / charts / diagrams?
  |    └─ Yes → Image Chunking (vision LLM caption or CLIP embedding)
  |
  ├─ Structured format (code, markdown, HTML)?
  |    └─ Yes → Document-Specific Chunking
  |
  ├─ Budget/speed critical?
  |    └─ Yes → Fixed-Size or Recursive
  |
  ├─ Context and meaning critical?
  |    └─ Yes → Semantic Chunking
  |
  ├─ Complex multi-topic documents?
  |    └─ Yes → Agentic Chunking (if < 100KB)
  |
  └─ Default → Recursive Character Splitting
```

**For 1000-Page Documents**: Use **Hierarchical Chunking**
- Avoids context window limits (LLMs can't process 1000 pages at once)
- Combines structural splitting (chapters) + semantic chunking (within sections)
- Best of both worlds: Intelligence + scalability

**For Mixed Documents (text + tables + images)**: Apply all three strategies per element type — text chunks, table chunks, and image caption chunks all co-exist in the same vector store, differentiated by `chunk_type` metadata.

---

# Retrieval Search Methods

## 1. Cosine Similarity

### Definition
Measures the cosine of the angle between two vectors, representing semantic similarity regardless of magnitude.

### How It Works
- Converts query and documents to embeddings (vectors)
- Calculates cosine angle: `similarity = (A·B) / (||A|| × ||B||)`
- Score range: -1 to 1 (higher = more similar)
- Focuses on direction, not magnitude

### Example
```python
from langchain_community.vectorstores import Chroma

# Default search method in most vector stores
docs = vectorstore.similarity_search(
    query="What is RAG?",
    k=5  # Return top 5 most similar
)
# Returns documents with highest cosine similarity
```

**Use Case**: General-purpose semantic search (most common default method)

---

## 2. Euclidean Distance (L2)

### Definition
Measures straight-line distance between two vector points in embedding space.

### How It Works
- Calculates distance: `d = sqrt(Σ(Ai - Bi)²)`
- Lower distance = more similar
- Sensitive to vector magnitude
- Common in spatial/geometric applications

### Example
```python
from langchain_community.vectorstores import FAISS

# Create FAISS index with L2 distance
index = FAISS.from_documents(docs, embeddings, distance_strategy="EUCLIDEAN_DISTANCE")

results = index.similarity_search(query, k=5)
# Returns documents with smallest Euclidean distance
```

**Use Case**: When magnitude matters (e.g., image similarity, certain scientific applications)

---

## 3. Maximum Marginal Relevance (MMR)

### Definition
Balances relevance to query with diversity among results to reduce redundancy.

### How It Works
- First, retrieves candidates based on similarity to query
- Then, iteratively selects documents that are:
  - Relevant to query (high similarity)
  - Different from already selected docs (low similarity to each other)
- Parameter `lambda_mult`: 0 (max diversity) to 1 (max relevance)

![alt text](image.png)

### Example
```python
from langchain_community.vectorstores import Chroma

# MMR retrieval
docs = vectorstore.max_marginal_relevance_search(
    query="What is machine learning?",
    k=5,                    # Return 5 documents
    fetch_k=20,             # Fetch 20 candidates first
    lambda_mult=0.5         # Balance relevance vs diversity
)
# Returns diverse yet relevant documents
```

**Use Case**: Avoid redundant results, get diverse perspectives on a topic

---

## 4. Similarity Score Threshold

### Definition
Filters results to only return documents above a minimum similarity score.

### How It Works
- Performs similarity search (cosine/L2)
- Returns only documents with score ≥ threshold
- Prevents low-quality matches
- Dynamic result count (0 to k documents)

### Example
```python
from langchain_community.vectorstores import Chroma

# Only return docs with similarity >= 0.8
docs = vectorstore.similarity_search_with_relevance_scores(
    query="RAG implementation",
    k=10,
    score_threshold=0.8  # Minimum similarity threshold
)

# Filter results
relevant_docs = [doc for doc, score in docs if score >= 0.8]
```

**Use Case**: Ensure minimum quality, prevent irrelevant results in production

---

## 5. BM25 (Keyword-Based)

### Definition
Keyword-based ranking algorithm that scores documents based on term frequency and inverse document frequency.

### How It Works
- Analyzes exact keyword matches (not semantic)
- Scores based on:
  - **Term Frequency (TF)**: How often a word appears in a document
    - If "python" appears 3 times → higher score than appearing 1 time
    - Logic: More mentions = more relevant to that term

  - **Inverse Document Frequency (IDF)**: How rare/unique a word is across all documents
    - Word in 1/100 docs → high IDF (very informative)
    - Word in 99/100 docs → low IDF (common, less informative)
    - Logic: Rare words are more meaningful (e.g., "TensorFlow" vs "the")

  - Document length normalization (prevents bias toward long docs)
- No embeddings required

**Simple Example**:
```
Query: "python tutorial"

Document Collection (5 docs total):
  Doc1: "Python tutorial for beginners" (python:1, tutorial:1)
  Doc2: "Advanced Python, Python, Python guide" (python:3, tutorial:0)
  Doc3: "JavaScript tutorial" (python:0, tutorial:1)
  Doc4: "Tutorial on web development" (tutorial appears)
  Doc5: "Python programming basics" (python appears)

Term Analysis (IDF Calculation):
  "python" → appears in 3/5 docs = 60% of docs
    IDF = log(5/3) = 0.51 (moderate importance)

  "tutorial" → appears in 4/5 docs = 80% of docs
    IDF = log(5/4) = 0.22 (lower importance, very common)

BM25 Scoring Logic:
  Doc1: HIGH SCORE
    - TF: "python" (1x) + "tutorial" (1x)
    - IDF: python(0.51) + tutorial(0.22) = 0.73
    - ✅ BEST: Both terms present, balanced score

  Doc2: MEDIUM SCORE
    - TF: "python" (3x) = boosted TF score
    - IDF: python(0.51) × 3 occurrences = ~1.2
    - ❌ Missing "tutorial" = loses 0.22 points
    - Result: High TF, but missing a query term

  Doc3: LOW SCORE
    - TF: "tutorial" (1x)
    - IDF: tutorial(0.22) only
    - ❌ Missing "python" = loses 0.51 points
    - Result: Very low total score

Final Ranking: Doc1 > Doc2 > Doc3
(Having both terms beats high frequency of one term)
```

### Code Example
```python
from langchain.retrievers import BM25Retriever

# Create BM25 retriever
retriever = BM25Retriever.from_documents(documents)
retriever.k = 5  # Return top 5

# Keyword-based search
docs = retriever.invoke("password reset authentication")
# Ranks by keyword matches, not semantic meaning
```

**Use Case**: Exact keyword matching, technical terms, proper nouns, IDs

---

## 6. Hybrid Search

### Definition
Combines semantic search (embeddings) with keyword/BM25 search (inverted index) for best of both worlds. This is the **production standard** for RAG retrieval systems.

---

### What You Store Per Chunk

In production hybrid retrieval, every chunk requires three things stored together:

| Field | Purpose |
|---|---|
| `embedding` | Semantic similarity via vector search |
| `chunk_text` | BM25 / keyword search + snippet generation + reranking |
| `metadata` | doc_id, chunk_id, source, page, section, timestamps, ACLs |

> **Key insight**: Do not store only embeddings. Raw chunk text is required for keyword search, debugging, and reranking.

---

### What is an Inverted Index?

An inverted index is the data structure used by search engines for fast keyword lookup. Instead of storing `document → words`, it stores `word → documents`.

**Forward index (normal storage):**
```
chunk_1: "AI detects payment fraud"
chunk_2: "payment routing optimization"
```

**Inverted index (built automatically by the search engine):**
```
AI          → chunk_1
detects     → chunk_1
payment     → chunk_1, chunk_2
fraud       → chunk_1
routing     → chunk_2
optimization → chunk_2
```

When you query `payment fraud`, the engine intersects the lists instantly — this is why keyword search is extremely fast.

> You do **not** build the inverted index manually. The search engine builds it automatically when you store text.

---

### Where to Store Chunks

**Option A — Single engine supporting hybrid natively (preferred):**

| Engine | Hybrid Support |
|---|---|
| OpenSearch / Elasticsearch | ✅ Yes |
| Weaviate | ✅ Yes |
| Azure AI Search | ✅ Yes |
| Vespa | ✅ Yes |

**Option B — Separate systems:**
- Vector DB (Pinecone, Qdrant) → stores embeddings
- Search engine (Elasticsearch) → stores text + inverted index
- Join results by `chunk_id`

> Best practice: use a single engine if it supports both, to simplify sync and ranking.

---

### How Weaviate Stores Both (Vector + Keyword)

Each object in Weaviate stores:

```json
{
  "chunk_id": "chunk_101",
  "text": "AI detects payment fraud",
  "embedding": [0.23, -0.45, "..."],
  "source": "fraud_policy_doc",
  "page": 5
}
```

Internally, Weaviate builds:
- **HNSW vector index** — for semantic search over embeddings
- **Inverted index** — for BM25 keyword search over text

Both are queried simultaneously on a hybrid search call.

---

### How Hybrid Search Works

```
                 Query
                   │
           ┌───────┴────────┐
           │                │
     Vector Search      BM25 Search
       (Embeddings)   (Inverted Index)
           │                │
           └───────┬────────┘
                   │
             Score Fusion
           (weighted or RRF)
                   │
              Top-K Results
                   │
           Optional Reranker
           (cross-encoder)
                   │
                   LLM
```

**Score fusion options:**
- **Weighted:** `score = α × vector_score + (1−α) × bm25_score`
- **Reciprocal Rank Fusion (RRF):** rank-based fusion, robust to score scale differences

---

### Why Keyword Search Is Critical in Production

Embeddings alone fail for:

| Case | Example |
|---|---|
| Product/error codes | `ERR_PAYMENT_4593` |
| Exact IDs | `TXN-00291-XZ` |
| Acronyms | `FMLA`, `SOC2`, `PCI-DSS` |
| Rare terms | domain-specific jargon |
| Fresh vocabulary | newly released product names |
| Compliance phrases | exact policy wording |

> A vector search for `ERR_PAYMENT_4593` may return semantically similar but wrong results. Keyword search matches it exactly via the inverted index.

---

### Mental Model

| Component | Answers |
|---|---|
| Embedding + vector index | *meaning* — "what does this text mean?" |
| Chunk text + inverted index | *exact words* — "does this text contain this term?" |
| Hybrid fusion | best of both |
| Reranker | final precision pass |

---

### Example (LangChain EnsembleRetriever)

```python
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.vectorstores import Chroma

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 30% keyword, 70% semantic
)

docs = ensemble_retriever.invoke("reset user password")
```

**Use Case**: Production RAG systems requiring both semantic understanding and exact term matching.

---

## Retrieval Methods Comparison

| Method | Type | Speed | Accuracy | Best For |
|--------|------|-------|----------|----------|
| **Cosine Similarity** | Semantic | Fast | High | General semantic search |
| **Euclidean (L2)** | Semantic | Fast | Medium | Magnitude-sensitive tasks |
| **MMR** | Semantic + Diversity | Medium | High | Diverse results |
| **Score Threshold** | Filter | Fast | Variable | Quality control |
| **BM25** | Keyword | Very Fast | Medium | Exact term matching |
| **Hybrid** | Combined | Medium | Highest | Production systems |

---

# ContextualCompressionRetriever

## What is ContextualCompressionRetriever?

A **ContextualCompressionRetriever** in LangChain is a retriever that filters and compresses retrieved documents to keep only the relevant parts, discarding noise and reducing context window usage.

---

## Why Compress Retrieved Documents?

**Problems Solved:**
- Reduces token costs (60-80% savings)
- Fits more context within LLM limits
- Removes noise that confuses models
- Improves response quality

**Impact:** 2,000 tokens → 200 tokens (10% of original) with same relevance

---

## How It Works

### Architecture Flow:

```
Query
  ↓
Base Retriever (vector search, BM25, etc.)
  ↓
[Full Documents: 2000 tokens each]
  ↓
Compressor (LLM or specialized filter)
  ↓
[Compressed Excerpts: 100-300 tokens each]
  ↓
LLM Generation
```

---

## Compression Methods

### 1. LLMChainExtractor (Most Common)

**How it works:**
- Uses an LLM to extract relevant quotes
- Asks: "What parts of this document answer the query?"
- Returns only matching sentences/passages

**Example:**
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)

# Retrieve and compress
compressed_docs = compression_retriever.invoke("What is RAG?")
```

The LLM internally extracts only the RAG-related parts, removing noise.

---

### 2. EmbeddingsFilter (Semantic-Based)

**How it works:**
- Embeds query and documents
- Keeps only docs with high semantic similarity to query
- Faster and cheaper than LLM-based compression

**Example:**
```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
similarity_threshold = 0.76

compressor = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=similarity_threshold
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)
```

---

### 3. DocumentCompressorPipeline (Chained Compression)

**How it works:**
- Combine multiple compressors in sequence
- Example: EmbeddingsFilter → LLMChainExtractor

**Example:**
```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# First filter by semantic similarity, then extract relevant quotes with LLM
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsFilter(embeddings=embeddings, threshold=0.76),
        LLMChainExtractor.from_llm(llm)
    ]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=vector_store.as_retriever()
)
```

---

## Compression in Practice

### Before Compression

```
Query: "How does RAG work?"

Retrieved Document (2000 tokens):
"RAG stands for Retrieval-Augmented Generation. It was introduced in 2020...
[100 lines of history]...
The basic workflow is: 1) Retrieve documents 2) Augment prompt 3) Generate...
[Detailed explanation]...
[Unrelated section about transformers]...
[Another unrelated section about fine-tuning]..."
```

### After LLMChainExtractor Compression

```
Extracted Relevant Passages (300 tokens):
"RAG stands for Retrieval-Augmented Generation.
The basic workflow is: 1) Retrieve documents 2) Augment prompt 3) Generate.
This allows LLMs to access external knowledge without fine-tuning."
```

**Result:** The LLM intelligently extracted only the parts answering the query.

---

## Complete Example

### Full RAG Pipeline with Compression

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA

# 1. Setup base retriever
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 2. Setup compressor
llm = ChatOpenAI(model="gpt-4")
compressor = LLMChainExtractor.from_llm(llm)

# 3. Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 4. Use in RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,  # Uses compressed docs
    chain_type="stuff"
)

# 5. Generate answer with compressed context
answer = qa_chain.invoke("What is RAG?")
print(answer)
```

---

## When to Use Compression

### ✅ Use Compression When:

| Scenario | Reason |
|----------|--------|
| **Latency matters** | API calls must be fast |
| **Budget-conscious** | You pay per token (OpenAI, etc.) |
| **Long documents** | Retrieved docs are very long (>2K tokens each) |
| **Many retrieved chunks** | You retrieve 50+ documents |
| **Quality over quantity** | You'd rather have fewer, cleaner docs |

### Compression Method Comparison

| Method | Speed | Cost | Quality | Use Case |
|--------|-------|------|---------|----------|
| **LLMChainExtractor** | Slow | High | Best | Maximum accuracy needed |
| **EmbeddingsFilter** | Fast | Low | Good | Budget/speed priority |
| **Pipeline** | Medium | Medium | Excellent | Balanced approach |

---

## Key Takeaways

1. **Compression saves 60-80% tokens** - Significantly reduces API costs
2. **Three main methods** - LLM extraction, embeddings filter, or pipeline
3. **Trade-off**: Speed/cost vs quality - Choose based on your needs
4. **Best practice**: Use pipeline (filter → extract) for balanced results
5. **Critical for production**: Essential when dealing with large document sets

---

# Vector Database Indexing

## What is Vector Indexing?

Vector indexes enable fast similarity search by creating smart data structures that check only a fraction of vectors instead of comparing against all vectors.

**Performance Impact**:
```
Without indexing: 500-1000ms (slow)
With HNSW:       2-5ms (200x faster)
With IVF:        8-20ms (50x faster)
```

### Saving Embeddings vs Pre-Indexed

These are two separate steps — both happen when you call `add_texts()` or `add_documents()`.

| | Saves vector | Builds HNSW index | Query speed |
|--|--|--|--|
| Raw DB column | ✅ | ❌ | O(n) — full scan |
| ChromaDB / Pinecone | ✅ | ✅ | O(log n) — ANN |

- **Saving embedding** = storing raw float vector `[0.23, -0.45, 0.78, ...]` in storage
- **Pre-indexed** = HNSW graph built over those vectors at ingestion time, so query is O(log n) — not a brute-force scan

```
Without index:  query vector vs ALL stored vectors  → O(n) scan    → slow
With HNSW:      navigates a graph of vectors        → O(log n) ANN → fast
```

`add_texts()` and `add_documents()` both embed → store → index in one call. `add_documents()` is just a wrapper over `add_texts()` for `Document` objects (from loaders/splitters):

```python
# add_texts — pass raw strings
vectorstore.add_texts(texts=["30-day return policy"], metadatas=[{"source": "policy"}])

# add_documents — pass Document objects (from loaders/splitters)
vectorstore.add_documents([Document(page_content="30-day return policy", metadata={"source": "policy"})])

# add_documents internally calls add_texts — same indexing, same result
def add_documents(self, documents):
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    return self.add_texts(texts, metadatas)
```

> **"Pre-indexed"** = embedding computed + HNSW index built at ingestion, so query time is purely a fast index lookup — documents are never re-embedded.

### When Pre-Indexing Is NOT Done

You encounter this with **FAISS Flat** or raw numpy/list storage — vectors stored but no index built, so every query scans all vectors.

```python
import faiss, numpy as np

# ❌ No index — brute force scan on every query
index = faiss.IndexFlatL2(384)   # "Flat" = no HNSW graph
index.add(vectors)               # stored, not indexed

distances, ids = index.search(query, k=5)  # O(n) — scans all vectors

# ✅ With HNSW — O(log n) ANN search
index_hnsw = faiss.IndexHNSWFlat(384, 32)
index_hnsw.add(vectors)          # stored + index built
distances, ids = index_hnsw.search(query, k=5)  # ~2ms vs ~500ms
```

**When flat (no-index) is acceptable:**

| Scenario | Why |
|----------|-----|
| < 10k docs | Flat scan fast enough |
| Exact search needed | HNSW is approximate — can miss true nearest |
| Prototyping | No setup overhead |

> ChromaDB / Pinecone always pre-index automatically — you only hit "no index" with raw FAISS Flat or numpy.

---

## HNSW (Hierarchical Navigable Small World)

### Definition
Graph-based index organizing vectors in hierarchical layers for fast approximate nearest neighbor search.

### How It Works
- Creates multi-level graph structure (highways → main roads → local streets)
- **Layer 0**: All vectors fully connected
- **Upper layers**: Sparse "express" nodes for fast navigation
- Search starts at top layer, progressively zooms to bottom
- Nodes connect to M nearest neighbors by similarity

### Performance
- **Speed**: 1-5ms query time, 95-99% recall
- **Memory**: 1.5-2× base vector size
- **Best for**: Real-time apps, high accuracy needs, frequent updates

### When to Use
✅ Real-time applications, high accuracy (>95%), frequent updates, low latency (<10ms)
❌ Memory constrained, >100M vectors, lower accuracy acceptable

### HNSW in Code

```python
import chromadb
from chromadb.config import Settings

# Create ChromaDB client with HNSW
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = client.create_collection(
    name="my_documents",
    metadata={
        "hnsw:space": "cosine",           # Distance metric
        "hnsw:construction_ef": 200,      # Build quality (higher = better)
        "hnsw:search_ef": 100,            # Search quality (higher = slower but better)
        "hnsw:M": 16                      # Connections per node (higher = better quality)
    }
)

# Add documents
collection.add(
    documents=["doc1", "doc2", ...],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    ids=["id1", "id2", ...]
)

# Query (uses HNSW automatically)
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],
    n_results=10
)
# Returns top 10 in ~2-5ms
```

---

## IVF (Inverted File Index)

### Definition
Clustering-based index that groups similar vectors into clusters, searching only relevant clusters instead of all vectors.

### How It Works
- **Training**: Groups vectors into clusters (e.g., 1,000 clusters with centroids)
- **Search**: Finds nprobe nearest clusters to query, searches only those
- **Key params**:
  - `nlist`: Number of clusters
  - `nprobe`: Clusters to search (higher = better accuracy, slower)
- Like library sections: search only relevant sections, not entire library

### Performance
- **Speed**: 5-20ms query time, 85-95% recall (nprobe=10-20)
- **Memory**: 1.1-1.3× base size (or 0.25× with Product Quantization)
- **Best for**: Large datasets (>10M), memory optimization, GPU acceleration

### When to Use
✅ Large datasets (>10M), memory constrained, GPU available, moderate accuracy OK
❌ Small datasets (<1M), highest accuracy needed (>97%), frequent updates

### IVF in Code

```python
import faiss
import numpy as np

# Create IVF index
dimension = 384  # Embedding dimension
nlist = 1000     # Number of clusters
nprobe = 10      # Number of clusters to search

# 1. Create quantizer (for clustering)
quantizer = faiss.IndexFlatL2(dimension)

# 2. Create IVF index
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 3. Train the index (create clusters)
vectors = np.random.random((100000, dimension)).astype('float32')
index.train(vectors)

# 4. Add vectors to index
index.add(vectors)

# 5. Set search parameters
index.nprobe = nprobe  # Search 10 clusters

# 6. Search
query = np.random.random((1, dimension)).astype('float32')
k = 10  # Return top 10
distances, indices = index.search(query, k)
# Returns top 10 in ~8-15ms
```

**With Product Quantization (Memory Optimization)**:

```python
import faiss

dimension = 768
nlist = 4096      # More clusters for larger dataset
m = 96            # Number of sub-quantizers
bits = 8          # Bits per sub-quantizer

# Create IVF + PQ index (compressed)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)

# Train and add
index.train(vectors)
index.add(vectors)

# Search
index.nprobe = 20
distances, indices = index.search(query, k)

# Memory savings: 90% reduction!
# 5M vectors: 15GB → 1.5GB
```

---

## HNSW vs IVF Comparison

| Feature | HNSW | IVF |
|---------|------|-----|
| **Speed** | 1-5ms | 5-20ms |
| **Accuracy** | 95-99% | 85-95% |
| **Memory** | 1.5-2x base | 1.1-1.3x base |
| **Scalability** | Up to 50M | Billions+ |
| **Updates** | Fast online | Slow, needs retrain |
| **GPU** | Limited | Excellent |

**Choose HNSW**: <1M vectors, <5ms latency, high accuracy, frequent updates
**Choose IVF**: >50M vectors, memory/cost critical, GPU available

---

## Quick Reference

### Library Support

| Library | HNSW | IVF | Notes |
|---------|------|-----|-------|
| **ChromaDB** | ✅ | ❌ | Small-medium datasets |
| **Faiss** | ✅ | ✅ | Large scale, GPU |
| **Pinecone** | ✅ | ✅ | Managed service |
| **Qdrant** | ✅ | ❌ | Fast HNSW |
| **Milvus** | ✅ | ✅ | Enterprise-grade |

### Configuration Cheat Sheet

**HNSW**:
```python
M=16, construction_ef=200, search_ef=100  # Balanced
M=8, construction_ef=100, search_ef=50    # Fast
M=32, construction_ef=400, search_ef=200  # Accurate
```

**IVF**:
```python
1M vectors:   nlist=1000, nprobe=10
10M vectors:  nlist=4096, nprobe=20
100M vectors: nlist=16384, nprobe=40
```

---

# HyDE (Hypothetical Document Embeddings)

## What is HyDE?

HyDE generates a **hypothetical answer** to the query first, then uses that answer (instead of the query) to retrieve documents.

**Core Purpose**: Bridges the semantic gap between questions and answers for better retrieval.

---

## How It Works

### Traditional RAG:
```
Query → Embed Query → Search → Retrieve Docs → LLM Answer
```

### HyDE RAG:
```
Query → LLM (Generate Hypothetical Answer) → Embed Answer → Search → Retrieve Docs → LLM (Final Answer)
```

**Example**:

**Query**: "What causes inflation?"

**Traditional**: Embeds "What causes inflation?" → searches for question-like text

**HyDE**:
1. LLM generates hypothetical answer: "Inflation is caused by increased money supply, rising production costs, strong demand, and supply chain disruptions."
2. Embeds this answer → searches for similar answers
3. Finds better matching documents

**Result**: 10-15% better retrieval accuracy

---

## When to Use HyDE

**Use HyDE when:**
- Short/vague queries ("password reset" → detailed explanation)
- Technical questions (bridges format gap)
- Poor retrieval quality with traditional search

**Avoid HyDE when:**
- Simple keyword search (overhead not worth it)
- Latency-critical (adds +500ms for hypothesis generation)
- Budget-constrained (2x LLM calls vs 1x)

**Trade-off**: Better accuracy (+10-15%) at cost of 2x LLM calls and +500ms latency

**Code Example**:
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Generate hypothetical answer
llm = ChatOpenAI(model="gpt-4")
hyde_prompt = PromptTemplate(
    template="Write a detailed answer to: {query}"
)

query = "What is HyDE?"
hypothetical_answer = llm.invoke(hyde_prompt.format(query=query))

# Embed hypothetical answer (not query)
hyde_embedding = embeddings.embed_query(hypothetical_answer.content)

# Search with hypothetical answer embedding
docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=5)
```

---

# RAG Document Versioning Fingerprint

## Problem Overview

Two critical ingestion scenarios require explicit handling:

| Scenario | Description | Expected Behaviour |
|----------|-------------|-------------------|
| **Exact Duplicate** | Same document uploaded again (identical content) | Detect and **skip** |
| **Version Update** | A revised document (v2) uploaded when v1 is already ingested | **Retire v1**, ingest v2 |

Without fingerprinting, re-uploading the same document silently doubles its representation in the vector store, degrading retrieval quality.

---

## How Fingerprinting Works

A **content fingerprint** (SHA-256 hash) is a fixed-length string derived deterministically from document content.

```
Same content   →  Same fingerprint   (exact duplicate)
Modified content →  Different fingerprint  (new version or new document)
```

```python
import hashlib

def fingerprint(content: str) -> str:
    """Generate SHA-256 fingerprint of document content."""
    return hashlib.sha256(content.encode()).hexdigest()

fingerprint("Refund Policy v1")  # → "a3f9c2..."  (always identical)
fingerprint("Refund Policy v1")  # → "a3f9c2..."  (same)
fingerprint("Refund Policy v2")  # → "d71bc8..."  (one word changed → entirely different)
```

**Why SHA-256?**
- Deterministic: identical input always produces identical output
- Collision-resistant: two different documents never produce the same hash in practice
- Fast: hashing a 10MB document takes milliseconds

---

## Duplicate Detection: Skip Same Document

When the **exact same document** is re-uploaded:
1. Compute SHA-256 fingerprint of new content
2. Query vector store for any document with matching `fingerprint` metadata
3. If found → skip ingestion entirely

```python
import hashlib
from datetime import datetime
from langchain.schema import Document

def compute_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def is_duplicate(vectorstore, fp: str) -> bool:
    """Return True if a document with this fingerprint already exists."""
    results = vectorstore.get(where={"fingerprint": fp})
    return len(results["ids"]) > 0

def ingest_if_not_duplicate(vectorstore, content: str, source: str) -> str:
    fp = compute_fingerprint(content)

    if is_duplicate(vectorstore, fp):
        print(f"⏭️  Exact duplicate: '{source}' already ingested — skipping.")
        return "skipped"

    vectorstore.add_documents([
        Document(
            page_content=content,
            metadata={
                "source": source,
                "fingerprint": fp,
                "ingested_at": datetime.now().isoformat()
            }
        )
    ])
    print(f"✅ Ingested '{source}'")
    return "ingested"
```

---

## Version Update: v1 → v2 Management

### Decision Flow

When a document is uploaded, apply this three-step check in order:

```
Upload document
       │
       ▼
Compute SHA-256 fingerprint
       │
       ├── fingerprint in DB? ──▶ YES ──▶ Skip (Exact Duplicate)
       │
       ▼ NO
       │
       ├── source name active in DB? ──▶ YES ──▶ Version Update:
       │                                          retire old → ingest new
       ▼ NO
       │
       └─────────────────────────────────▶ Fresh document: ingest directly
```

**Key insight**: same `source` + different `fingerprint` = the document was updated.

### Complete Implementation

```python
import hashlib
from datetime import datetime
from typing import Literal
from langchain.schema import Document

class DocumentVersionManager:
    """
    Manages document ingestion with fingerprint-based duplicate detection
    and version lifecycle (v1 → v2 retirement).
    """

    def __init__(self, vectorstore):
        self.vs = vectorstore

    # ── Helpers ────────────────────────────────────────────────────────────

    def _fingerprint(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _find_by_fingerprint(self, fp: str) -> list[str]:
        """Return IDs of documents with this exact content hash."""
        return self.vs.get(where={"fingerprint": fp})["ids"]

    def _find_active_by_source(self, source: str) -> dict:
        """Return all active (non-retired) versions of a source document."""
        return self.vs.get(where={"source": source, "is_active": True})

    def _retire(self, existing: dict):
        """Soft-delete old versions by flipping is_active to False."""
        for idx, old_id in enumerate(existing["ids"]):
            old_meta = existing["metadatas"][idx]
            old_content = existing["documents"][idx]
            self.vs.update_document(
                document_id=old_id,
                document=Document(
                    page_content=old_content,
                    metadata={
                        **old_meta,
                        "is_active": False,
                        "retired_at": datetime.now().isoformat(),
                    }
                )
            )
            print(f"   🗃️  Retired: {old_id} (was v{old_meta.get('version', '?')})")

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest(
        self,
        content: str,
        source: str,
        version: str = "1.0"
    ) -> Literal["ingested", "duplicate", "version_update"]:
        """
        Ingest a document with full duplicate and version awareness.

        Args:
            content:  Raw text content of the document.
            source:   Stable identifier, e.g. 'refund_policy.pdf'.
            version:  Caller-supplied version tag, e.g. '2.0'.

        Returns:
            'duplicate'      – identical content already ingested; skipped.
            'version_update' – new version ingested; old version retired.
            'ingested'       – fresh document ingested for the first time.
        """
        fp = self._fingerprint(content)

        # Step 1: Exact duplicate check (fingerprint collision)
        if self._find_by_fingerprint(fp):
            print(f"⏭️  Duplicate: '{source}' v{version} already exists — skipping.")
            return "duplicate"

        # Step 2: Check for an existing active version of the same source
        existing = self._find_active_by_source(source)
        result: Literal["ingested", "version_update"] = "ingested"

        if existing["ids"]:
            old_version = existing["metadatas"][0].get("version", "?")
            print(f"🔄 Version update: '{source}' v{old_version} → v{version}")
            self._retire(existing)
            result = "version_update"

        # Step 3: Ingest the new (or first) version
        doc_id = f"{source}#v{version}"
        self.vs.add_documents(
            documents=[Document(
                page_content=content,
                metadata={
                    "source":       source,
                    "version":      version,
                    "fingerprint":  fp,
                    "is_active":    True,
                    "ingested_at":  datetime.now().isoformat(),
                }
            )],
            ids=[doc_id]
        )
        print(f"✅ Ingested '{source}' v{version} [id: {doc_id}]")
        return result

    def search(self, query: str, k: int = 5):
        """Retrieve only active (latest) document versions."""
        return self.vs.similarity_search(
            query, k=k,
            filter={"is_active": True}
        )

    def history(self, source: str) -> list[dict]:
        """Return full version history (active + retired) for a source."""
        return self.vs.get(where={"source": source})["metadatas"]
```

### Usage Example

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=OpenAIEmbeddings()
)

manager = DocumentVersionManager(vectorstore)

# ── 1. First upload (v1) ──────────────────────────────────────────────────
manager.ingest(
    content="Refund allowed within 30 days with receipt.",
    source="refund_policy.pdf",
    version="1.0"
)
# ✅ Ingested 'refund_policy.pdf' v1.0 [id: refund_policy.pdf#v1.0]

# ── 2. Re-upload identical document (exact duplicate) ─────────────────────
manager.ingest(
    content="Refund allowed within 30 days with receipt.",
    source="refund_policy.pdf",
    version="1.0"
)
# ⏭️  Duplicate: 'refund_policy.pdf' v1.0 already exists — skipping.

# ── 3. Upload revised document (v2) ──────────────────────────────────────
manager.ingest(
    content="Refund allowed within 60 days. Receipt optional from 2025.",
    source="refund_policy.pdf",
    version="2.0"
)
# 🔄 Version update: 'refund_policy.pdf' v1.0 → v2.0
#    🗃️  Retired: refund_policy.pdf#v1.0 (was v1.0)
# ✅ Ingested 'refund_policy.pdf' v2.0 [id: refund_policy.pdf#v2.0]

# ── 4. Query returns only active (v2) ─────────────────────────────────────
results = manager.search("What is the refund window?")
# Returns v2 chunks only; v1 is retired and filtered out

# ── 5. Inspect full history ───────────────────────────────────────────────
for meta in manager.history("refund_policy.pdf"):
    print(meta["version"], "active:", meta["is_active"])
# 1.0  active: False
# 2.0  active: True
```

---

## Metadata Schema Reference

Every ingested document chunk should carry these metadata fields:

| Field | Type | Purpose |
|-------|------|---------|
| `source` | `string` | Stable document identifier (e.g. filename) |
| `version` | `string` | Caller-supplied semantic version (`"1.0"`, `"2.0"`) |
| `fingerprint` | `string` | SHA-256 hash of full document content |
| `is_active` | `bool` | `True` = current version; `False` = retired |
| `ingested_at` | ISO 8601 | Timestamp when this version was ingested |
| `retired_at` | ISO 8601 | Timestamp when this version was retired (optional) |

Always filter by `is_active: True` at query time to ensure only current content is retrieved.

---

## Strategy Comparison

| Scenario | Fingerprint Match | Source in DB | Action |
|----------|:-----------------:|:------------:|--------|
| Brand new document | No | No | Ingest fresh |
| Exact duplicate re-upload | Yes | Yes | Skip entirely |
| Updated version (v2 of v1) | No | Yes | Retire v1 → Ingest v2 |
| Different document, same content | Yes | No | Skip (content-level dedup) |

---

## Best Practices

| Practice | Why |
|----------|-----|
| Store `fingerprint` in every chunk's metadata | O(1) duplicate detection via metadata filter — no embedding comparison needed |
| Use `is_active` flag (soft-delete) instead of hard-delete | Preserves audit trail; supports rollback to previous version if needed |
| Append version to document ID (`source#v2.0`) | Avoids ID collisions across versions in the same collection |
| Record `ingested_at` and `retired_at` timestamps | Enables time-based queries and compliance auditing |
| Filter `is_active: True` in every production query | Prevents retired/outdated content from leaking into responses |
| Hash the full raw content before chunking | Ensures the fingerprint represents the whole document, not individual chunks |


---

# Embedding


## What are Embeddings?

Embeddings are **numerical representations** of text (or other data) as dense vectors in a high-dimensional space. They capture semantic meaning, allowing computers to understand relationships between words, sentences, or documents.

```
Text: "cat"  →  [0.2, -0.5, 0.8, ..., 0.3]  (768 dimensions)
Text: "dog"  →  [0.3, -0.4, 0.7, ..., 0.2]  (768 dimensions)
                  ↑ Similar vectors = Similar meaning
```

### Key Properties
- **Semantic similarity**: Similar concepts have similar vectors
- **Dense vectors**: Every dimension contains a value (vs sparse like one-hot)
- **Fixed dimensions**: Typically 384, 768, 1024, or 1536 dimensions
- **Contextual**: Modern embeddings capture context, not just words

---

## Understanding Dimensions in Embeddings

### What Does "Dimension" Mean?

Think of dimensions as **features or characteristics** that describe a word. Each dimension captures a different aspect of meaning.

**Simple Example: 2D Embedding (Beginner)**

```
Imagine only 2 dimensions:
- Dimension 1: "Is it alive?" (0 = no, 1 = yes)
- Dimension 2: "Is it big?" (0 = small, 1 = large)

Word        Dim-1   Dim-2   Vector
"ant"       1.0     0.1     [1.0, 0.1]  (alive, tiny)
"elephant"  1.0     0.9     [1.0, 0.9]  (alive, huge)
"car"       0.0     0.7     [0.0, 0.7]  (not alive, big)
"pebble"    0.0     0.1     [0.0, 0.1]  (not alive, tiny)

Visualized:
    Big (1.0)
        │
        │  elephant
        │
        │  car
        │
        │  ant, pebble
        └─────────────── Alive (1.0)
       Not Alive (0.0)
```

**Problem**: 2 dimensions can't capture complex meaning!
- "King" vs "Queen" (both alive, both important, but different gender)
- "Happy" vs "Joyful" (subtle emotion differences)

---

### Real-World: 768 Dimensions

```
In practice, models use hundreds of dimensions:

Word "king" = [0.23, -0.51, 0.82, 0.12, -0.34, ..., 0.45]
               ↑      ↑      ↑      ↑      ↑         ↑
              Dim1   Dim2   Dim3   Dim4   Dim5   ... Dim768

Each dimension might capture:
- Dim1-50:   Grammar (noun, verb, adjective)
- Dim51-150: Semantics (animal, object, emotion)
- Dim151-300: Context (formal, informal, technical)
- Dim301-500: Relationships (king-queen, man-woman)
- Dim501-768: Subtle nuances (cultural, historical)

Note: These are learned automatically, not predefined!
```

---

### Does Higher Dimension = Better Accuracy?

**Short Answer**: Not always! There's a **sweet spot**.

```
┌─────────────────────────────────────────────┐
│ Dimension vs Accuracy Trade-off            │
└─────────────────────────────────────────────┘

Accuracy
   ↑
   │                  ╱─────── Plateau
   │               ╱
   │            ╱
   │         ╱        Sweet Spot
   │      ╱           (768-1024)
   │   ╱
   │╱
   └────────────────────────────────────────→
   50   384   768   1024  1536  3072  Dimensions

Issues with TOO MANY dimensions:
❌ More storage/memory
❌ Slower computation
❌ Overfitting (learns noise)
❌ Diminishing returns

Issues with TOO FEW dimensions:
❌ Can't capture complexity
❌ Words collide (similar vectors)
❌ Poor accuracy
```

---

### Practical Comparison

**Example: Finding similar words to "doctor"**

```
50-Dimension Model:
"doctor" similar to: physician, nurse, teacher, scientist
                                       ↑ Wrong! (not specific enough)

384-Dimension Model (Sentence-BERT):
"doctor" similar to: physician, surgeon, medical practitioner
                                       ✓ Good!

768-Dimension Model (BERT):
"doctor" similar to: physician, surgeon, medical practitioner, clinician
                                       ✓ Better nuance

3072-Dimension Model (OpenAI large):
"doctor" similar to: physician, surgeon, medical practitioner, clinician
                                       ✓ Slightly better, but 4x cost

Verdict: 384-768 is usually optimal!
```

---

### Visual: How Dimensions Capture Meaning

```
1D: Only one axis (useless for meaning)
───────────────────────────
cat  dog  car  happy  king
(All on a line - no relationships!)


2D: Can show basic categories
        Emotions
            │
      happy │ sad
            │
────────────┼────────────
            │
      cat   │   car
      dog   │
        Animals
(Better, but still limited)


768D: Captures complex relationships
(Can't visualize, but imagine 768 axes!)

In this space:
- king - man + woman ≈ queen
- Paris - France + Germany ≈ Berlin
- Good - Bad + Terrible ≈ Awful

Each dimension adds another "axis" of meaning!
```

---

### Rule of Thumb

| Dimensions | Use Case | Example Models |
|------------|----------|----------------|
| **50-100** | Simple classification | Word2Vec mini |
| **384** | Fast, good enough for most tasks | all-MiniLM-L6-v2 |
| **768** | Standard quality | BERT, most RAG systems |
| **1024** | High quality | BGE-large |
| **1536+** | Maximum quality, expensive | OpenAI text-embedding-3 |

**For RAG applications**: Start with **384-768 dimensions** - best balance of cost, speed, and accuracy!

---

## Storage & Cost Impact of High Dimensions

### Yes! High dimensions directly impact storage, speed, and cost

```
┌────────────────────────────────────────────────┐
│ The Hidden Cost of Dimensions                 │
└────────────────────────────────────────────────┘

384-dim   →  1.5 KB per vector
768-dim   →  3.0 KB per vector  (2x storage!)
1536-dim  →  6.0 KB per vector  (4x storage!)
3072-dim  →  12 KB per vector   (8x storage!)
```

---

### Storage Size Calculation

**Formula**: `Size per vector = Dimensions × 4 bytes`
(Each dimension is a 32-bit float = 4 bytes)

**Example: 1 Million Documents**

```
┌─────────────────────────────────────────────────────┐
│ Storage Requirements for 1M Embeddings             │
├─────────────────────────────────────────────────────┤
│ Dimensions │ Per Vector │ 1M Vectors │ 100M Vectors │
│────────────│────────────│────────────│──────────────│
│    384     │   1.5 KB   │   1.5 GB   │    150 GB    │
│    768     │   3.0 KB   │   3.0 GB   │    300 GB    │
│   1024     │   4.0 KB   │   4.0 GB   │    400 GB    │
│   1536     │   6.0 KB   │   6.0 GB   │    600 GB    │
│   3072     │  12.0 KB   │  12.0 GB   │   1200 GB    │
└─────────────────────────────────────────────────────┘

⚠️ This is JUST embeddings, not including:
   - Vector database index (HNSW/IVF adds 20-50% overhead)
   - Metadata storage
   - Backup copies
```

---

### Real-World Example: How Data is Stored

```
Scenario: RAG system with 10 million documents
```

#### Understanding Vector Database Storage

Each document in a VectorDB is stored as a **record** with multiple components:

```
┌─────────────────────────────────────────────────────────┐
│ Single Record in Vector Database                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. ID: "doc_12345"                                     │
│    └─ Unique identifier (8-16 bytes)                   │
│                                                         │
│ 2. VECTOR (EMBEDDED):                                  │
│    └─ [0.23, -0.51, 0.82, ..., 0.34]                  │
│       └─ 768-dim = 3 KB                                │
│       └─ 3072-dim = 12 KB                              │
│                                                         │
│ 3. METADATA (RAW TEXT/JSON - NOT EMBEDDED):            │
│    ├─ "title": "Introduction to Neural Networks"      │
│    ├─ "author": "John Doe"                             │
│    ├─ "date": "2024-01-15"                             │
│    ├─ "category": "AI/ML"                              │
│    ├─ "tags": ["deep-learning", "AI"]                 │
│    └─ "chunk_id": "chunk_5"                            │
│       └─ Stored as JSON/key-value (~200-500 bytes)    │
│                                                         │
│ 4. ORIGINAL TEXT (OPTIONAL):                           │
│    └─ "Neural networks are computational models..."    │
│       └─ Raw text (~1-5 KB per chunk)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

Total per record: ~4-18 KB (depending on dimensions + metadata)
```

---

#### Two Approaches to Metadata

```
┌──────────────────────────────────────────────────────────┐
│ Approach 1: Metadata Separate (for filtering)           │
├──────────────────────────────────────────────────────────┤
│ Text: "Neural networks use layers..."                   │
│   → Embed: [0.2, -0.5, ...]                             │
│   → Store metadata separately as JSON                    │
│                                                          │
│ ✅ Filter exact values: author="John", date>2024        │
│ ✅ Update metadata without re-embedding                 │
│ ❌ Metadata not in semantic search                      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Approach 2: Metadata Included (for semantic search)     │
├──────────────────────────────────────────────────────────┤
│ Text: "Title: Neural Networks | AI/ML                   │
│        Neural networks use layers..."                    │
│   → Embed: [0.3, -0.4, ...] (includes metadata!)        │
│                                                          │
│ ✅ "AI article by John" matches semantically            │
│ ✅ Better context for retrieval                         │
│ ❌ Must re-embed if metadata changes                    │
└──────────────────────────────────────────────────────────┘

BEST PRACTICE: Hybrid (Combine Both)
┌──────────────────────────────────────────────────────────┐
│ Embed: "Title: Neural Networks in Healthcare            │
│         Category: Medical AI | 2024                      │
│         Content: Deep learning models..."                │
│                                                          │
│ + Store: {"category": "Medical AI", "date": "2024"}     │
│                                                          │
│ = Semantic search + exact filtering                     │
└──────────────────────────────────────────────────────────┘

Example Query: "Find AI papers about diagnosis from 2024"
→ Vector search matches "AI", "diagnosis" (semantic)
→ Filter matches category="Medical AI", date>="2024" (exact)
```

---

#### Storage Breakdown: 10M Documents

```
384-dim embeddings:
├─ Vectors (embedded): 15 GB
├─ HNSW index: +7 GB (50% overhead)
├─ Metadata (raw JSON): +2 GB
│  └─ ~200 bytes × 10M records
└─ Total: ~24 GB
   Cost on Pinecone: ~$70/month
   Retrieval: ~10ms

3072-dim embeddings:
├─ Vectors (embedded): 120 GB
├─ HNSW index: +60 GB (50% overhead)
├─ Metadata (raw JSON): +2 GB (same size!)
│  └─ Metadata size doesn't change with dimensions
└─ Total: ~182 GB
   Cost on Pinecone: ~$500/month
   Retrieval: ~30-40ms

💰 8x dimensions = 7x cost increase!
⏱️  8x dimensions = 3-4x slower retrieval!

Key Insight:
- Metadata is NOT embedded (stays same size)
- Only the vector dimensions increase storage
```

---

#### Metadata Storage Formats

```
Different VectorDBs store metadata differently:

Pinecone:
{
  "id": "doc_123",
  "values": [0.2, -0.5, ...],
  "metadata": {"title": "...", "author": "..."}
}

Weaviate:
{
  "class": "Document",
  "vector": [0.2, -0.5, ...],
  "properties": {"title": "...", "author": "..."}
}

Qdrant:
{
  "id": 123,
  "vector": [0.2, -0.5, ...],
  "payload": {"title": "...", "author": "..."}
}

All store metadata as raw text/JSON, NOT embedded!
```

---

#### When to Include Metadata in Embeddings

```
✅ DO include in embedded text:
├─ Document title (very relevant to search)
├─ Section headers (context for chunk)
├─ Category/topic (helps semantic matching)
├─ Date (if time-relevant: "2024 AI trends")
└─ Key descriptors (e.g., "beginner-friendly")

❌ DON'T include in embedded text:
├─ Internal IDs (not semantic)
├─ File paths (technical, not meaningful)
├─ User permissions (security concern)
├─ Update timestamps (not search-relevant)
└─ Large redundant data

Example:

Good (include):
"""
Title: Python Debugging Guide
Level: Beginner | Topic: Programming

Learn to debug Python code effectively...
"""

Bad (don't include):
"""
ID: 47382-ABC-DEF
Path: /srv/docs/v2/backup_2024/file.md
Created_by_user_id: 829374
Last_modified: 2024-01-15T14:23:11Z

Learn to debug Python code effectively...
"""
```

---

#### Key Takeaways

✅ **Metadata is stored as RAW JSON** - not embedded into vectors

✅ **Metadata size is fixed** - doesn't change with embedding dimensions

✅ **Two storage locations**:
   - Embedded: Title, category, key context (for semantic search)
   - Raw metadata: All fields (for filtering)

✅ **Best practice**: Include important metadata in text to embed + store all metadata separately

✅ **Filtering**: Use raw metadata for exact matching (author="John")

✅ **Semantic search**: Use embedded metadata for fuzzy matching

---

### VectorDB Cost Comparison (1M Vectors)

```
┌──────────────────────────────────────────────────────────┐
│ Popular Vector Databases - Monthly Cost                  │
├──────────────────────────────────────────────────────────┤
│ Provider    │ 384-dim │ 768-dim │ 1536-dim │ 3072-dim  │
│─────────────│─────────│─────────│──────────│───────────│
│ Pinecone    │   $7    │  $14    │   $28    │   $56     │
│ (p1 pods)   │         │         │          │           │
│─────────────│─────────│─────────│──────────│───────────│
│ Weaviate    │   $5    │  $10    │   $20    │   $40     │
│ (cloud)     │         │         │          │           │
│─────────────│─────────│─────────│──────────│───────────│
│ Qdrant      │   $4    │   $8    │   $16    │   $32     │
│ (cloud)     │         │         │          │           │
│─────────────│─────────│─────────│──────────│───────────│
│ Milvus      │   $6    │  $12    │   $24    │   $48     │
│ (Zilliz)    │         │         │          │           │
│─────────────│─────────│─────────│──────────│───────────│
│ Self-hosted │ Storage │ Storage │ Storage  │ Storage   │
│ (any)       │ cost    │ cost    │ cost     │ cost only │
└──────────────────────────────────────────────────────────┘

Note: Prices scale linearly with storage
      (10M vectors = 10x cost)
```

---

### Retrieval Speed Impact

```
┌────────────────────────────────────────────────┐
│ Why Higher Dimensions = Slower Search         │
└────────────────────────────────────────────────┘

To find similar vectors, VectorDB calculates distance:

384 dimensions:
├─ Compare 384 numbers per candidate
├─ Fast! (1-2ms for 1M vectors)
└─ Can check more candidates quickly

3072 dimensions:
├─ Compare 3072 numbers per candidate (8x work!)
├─ Slower (8-12ms for 1M vectors)
└─ Must sacrifice accuracy or speed

┌──────────────────────────────────────┐
│ Latency Comparison (1M vectors)     │
├──────────────────────────────────────┤
│ Dimensions │ Search Time │ Index Size│
│────────────│─────────────│───────────│
│    384     │   ~10ms    │    2 GB   │
│    768     │   ~15ms    │    4 GB   │
│   1536     │   ~25ms    │    8 GB   │
│   3072     │   ~40ms    │   16 GB   │
└──────────────────────────────────────┘
```

---

### The "Curse of Dimensionality"

```
As dimensions increase:

1. Distance becomes less meaningful
   ┌─────────────────────────────────────┐
   │ In very high dimensions (3072+),   │
   │ ALL points become "far apart"      │
   │ and roughly equidistant!           │
   │                                     │
   │ This makes similarity search       │
   │ harder and less accurate!          │
   └─────────────────────────────────────┘

2. More data needed
   ┌─────────────────────────────────────┐
   │ High-dim spaces are SPARSE          │
   │ Need exponentially more data        │
   │ to fill the space meaningfully      │
   └─────────────────────────────────────┘

Visual:
2D: Need 100 points to cover space
3D: Need 1,000 points
10D: Need 10 billion points!
3072D: Need... astronomical amounts!
```

---

### Cost Optimization Strategies

#### 1. **Dimension Reduction (Matryoshka Embeddings)**

```
Use OpenAI's flexible dimensions:

Generate: text-embedding-3-large (3072-dim)
          ↓
Store only first 1024 dimensions
          ↓
Result: 1/3 storage, 95% accuracy!

Example:
Full:     [0.2, -0.5, 0.8, ..., 0.3]  (3072 numbers)
Truncate: [0.2, -0.5, 0.8, ..., 0.1]  (1024 numbers)
                                       ↑ Drop last 2048!
Accuracy loss: Only 2-5%
Storage saved: 67%
Cost saved: 67%
```

#### 2. **Quantization (Reduce Precision)**

**Concept**: Convert high-precision floats to lower-precision integers

```
Each dimension: 32-bit float (4 bytes)
                ↓
Convert to:     8-bit integer (1 byte)
                ↓
Storage: 75% reduction!

Example (768-dim):
Original: 768 × 4 bytes = 3 KB
Quantized: 768 × 1 byte = 0.75 KB

Accuracy loss: ~1-3% for most tasks
Providers: Pinecone, Qdrant support this
```

---

#### How to Implement Quantization

**3 Implementation Approaches:**

**1. Let VectorDB Handle It (Recommended)**
```python
# Pinecone (automatic)
index = pc.create_index(name="my-index", dimension=768, metric="cosine")
# Auto-quantizes on serverless - 75% storage reduction

# Qdrant (configurable)
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantization(type="int8", quantile=0.99)
    )
)

# Weaviate (binary - 32x compression!)
schema = {
    "vectorIndexConfig": {
        "quantizer": {"type": "bq", "enabled": True}
    }
}
```

**2. Manual Quantization (DIY)**
```python
import numpy as np

# Float32 → Int8 (75% reduction)
embedding = np.array([0.234, -0.512, 0.821, ...], dtype=np.float32)
min_val, max_val = embedding.min(), embedding.max()
quantized = ((embedding - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Result: 3072 bytes → 768 bytes
# Accuracy loss: ~1-3%
```

**3. Using FAISS Library**
```python
import faiss

# Scalar Quantization (Int8)
sq_index = faiss.IndexIVFScalarQuantizer(
    quantizer, dimension=768, nlist=100,
    faiss.ScalarQuantizer.QT_8bit
)

# Product Quantization (8-16x compression)
pq_index = faiss.IndexPQ(dimension=768, m=96, nbits=8)
```

---

#### Quantization Types Comparison

| Type | Compression | Accuracy Loss | Use When |
|------|-------------|---------------|----------|
| **Float16** | 2x | ~0.001% | Need high accuracy |
| **Int8 (Scalar)** | 4x | ~1-3% | Standard production (BEST) |
| **Product Quantization** | 8-16x | ~5-10% | Large-scale (>10M vectors) |
| **Binary** | 32x | ~20-30% | Initial filtering only |

**Quick Decision:**
- **< 1M vectors**: No quantization needed
- **1-10M vectors**: Use Int8
- **> 10M vectors**: Use Product Quantization

---

#### Best Practices

```
✅ DO:
├─ Let VectorDB handle quantization (Pinecone/Qdrant)
├─ Use Int8 for 75% storage reduction with <3% accuracy loss
├─ Test accuracy on YOUR data before production
└─ Use binary quantization only for coarse filtering

❌ DON'T:
├─ Quantize embeddings before storing (DB does it better)
├─ Use binary for final ranking (too much accuracy loss)
└─ Mix quantized and non-quantized vectors in same index
```

#### 3. **Product Quantization (PQ)**

```
Split 768-dim into chunks:
[dim 1-8] [dim 9-16] ... [dim 761-768]
    ↓         ↓              ↓
  Code 1   Code 2  ...   Code 96

Instead of 768 floats, store 96 codes!
Compression: 16x smaller!

Used by: Faiss, Milvus
Accuracy: 90-95% retained
```

---

### Practical Recommendations

```
┌────────────────────────────────────────────────┐
│ Choose Dimensions Based on Scale              │
└────────────────────────────────────────────────┘

Small Scale (< 100K docs):
├─ Use 1536-dim (OpenAI small)
├─ Storage: < 1 GB
├─ Cost: $5-10/month
└─ Don't worry about optimization

Medium Scale (100K - 10M docs):
├─ Use 768-dim (BERT, BGE)
├─ Storage: 3-30 GB
├─ Cost: $10-100/month
└─ Consider quantization

Large Scale (10M+ docs):
├─ Use 384-dim (MiniLM)
├─ OR 768-dim with quantization
├─ Storage: 30+ GB
├─ Cost: $100+/month
└─ MUST optimize!

Massive Scale (100M+ docs):
├─ Use 384-dim + quantization + PQ
├─ OR self-host with compression
├─ Storage: 300+ GB → 50 GB (compressed)
├─ Cost: $500+ cloud OR self-host
└─ Hire experts or use managed service
```

---

### Cost-Benefit Analysis Table

| Dimensions | Accuracy | Storage (1M) | Search Speed | Use When |
|------------|----------|--------------|--------------|----------|
| **384** | ⭐⭐⭐ | 1.5 GB | ⚡⚡⚡ Fast | Budget tight, scale large |
| **768** | ⭐⭐⭐⭐ | 3.0 GB | ⚡⚡ Medium | Best balance (recommended) |
| **1024** | ⭐⭐⭐⭐ | 4.0 GB | ⚡⚡ Medium | Need better accuracy |
| **1536** | ⭐⭐⭐⭐⭐ | 6.0 GB | ⚡ Slower | Quality > cost |
| **3072** | ⭐⭐⭐⭐⭐ | 12.0 GB | 🐌 Slow | Only if critical need |

---

### Decision Framework

```
┌───────────────────────────────────────────┐
│ How many documents?                       │
└───────────────────────────────────────────┘
         │
         ├─ < 1M docs
         │  └─→ Use 768 or 1536-dim (don't optimize yet)
         │
         ├─ 1M - 10M docs
         │  └─→ Use 768-dim + quantization
         │
         ├─ 10M - 100M docs
         │  └─→ Use 384-768-dim + PQ compression
         │
         └─ > 100M docs
            └─→ Use 384-dim + all optimizations
                OR consider self-hosting
```

---

### Key Takeaways

✅ **Storage Impact**: Dimensions scale linearly (2x dims = 2x storage)

✅ **Cost Impact**: VectorDB pricing based on storage (2x storage ≈ 2x cost)

✅ **Speed Impact**: Higher dims = slower search (diminishing returns after 768)

✅ **Sweet Spot**: 768 dimensions for most production use cases

✅ **Optimization**: Use quantization, dimension reduction, or compression at scale

✅ **Start Simple**: Don't over-optimize early - scale first, optimize later

⚠️ **Warning**: Going from 768 → 3072 dims gives only 3% accuracy boost but costs 4x more!

---

## How Words are Converted to Embeddings

### Traditional Approach (Word2Vec/GloVe)

```
Step 1: Tokenization
"The cat sat" → ["The", "cat", "sat"]

Step 2: Context Window
    [The] cat sat  →  Context: ["cat"]
    The [cat] sat  →  Context: ["The", "sat"]
    The cat [sat]  →  Context: ["cat"]

Step 3: Neural Network Training
Input Word → Hidden Layer → Predict Context Words
                  ↓
            Extract weights as embeddings
```

**Limitation**: Same word = same embedding (ignores context)

---

### Modern Approach (Transformers: BERT, Sentence-BERT)

```
┌─────────────────────────────────────────────────┐
│ Input: "The bank river is beautiful"            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 1: Tokenization (text → tokens)           │
│ ["The", "bank", "river", "is", "beautiful"]    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 2: Token IDs (tokens → numbers)           │
│ [2053, 2924, 2314, 2003, 3376]                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 3: EMBEDDING LAYER (Token ID → Vector)    │
│ This is where embeddings are applied!          │
│                                                 │
│ 2053 → [0.1, -0.3, 0.7, ..., 0.2] (768-dim)   │
│ 2924 → [0.4, -0.1, 0.5, ..., 0.8] (768-dim)   │
│ 2314 → [0.2, 0.6, -0.2, ..., 0.3] (768-dim)   │
│ ...                                             │
│                                                 │
│ See detailed breakdown below ↓                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 4: Transformer Encoder Layers             │
│                                                 │
│  Layer 1: Self-Attention + Feed Forward        │
│  Layer 2: Self-Attention + Feed Forward        │
│  ...                                            │
│  Layer 12: Self-Attention + Feed Forward       │
│                                                 │
│  "bank" sees "river" → water bank (not money)  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 5: Pooling Strategy                        │
│ • CLS token: [CLS] representation              │
│ • Mean pooling: Average all token vectors      │
│ • Max pooling: Take max values                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Final Sentence Embedding Vector                 │
│ [0.23, -0.51, 0.82, ..., 0.34] (768-dim)      │
└─────────────────────────────────────────────────┘
```

---

### How Embeddings Are Applied (Step 3 Detailed)

Transformers use **THREE types of embeddings** that get added together:

```
Token ID: 2924 (word "bank")
         ↓
┌────────────────────────────────────────────────┐
│ 1. TOKEN EMBEDDING (What word?)               │
│    Lookup Table: 30,000 words × 768 dims      │
│    ID 2924 → [0.4, -0.1, 0.5, ..., 0.8]      │
│              (learned meaning of "bank")       │
└────────────────────────────────────────────────┘
         +
┌────────────────────────────────────────────────┐
│ 2. POSITION EMBEDDING (Where in sentence?)    │
│    Position 2 → [0.1, 0.2, -0.3, ..., 0.1]   │
│              (2nd position in sequence)        │
└────────────────────────────────────────────────┘
         +
┌────────────────────────────────────────────────┐
│ 3. SEGMENT EMBEDDING (Which sentence?)        │
│    Sentence A → [0.0, 0.0, 0.0, ..., 0.0]    │
│              (first sentence - optional)       │
└────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────┐
│ FINAL INPUT EMBEDDING (Sum of all 3)          │
│ [0.5, 0.1, 0.2, ..., 0.9] → Transformer      │
└────────────────────────────────────────────────┘
```

---

### Visual: Complete Embedding Process

```
Input: "I love cats"
        ↓
┌─────────────────────────────────────────────────┐
│ Tokenization → ["I", "love", "cats"]           │
└─────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│ Token IDs → [100, 250, 450]                    │
└─────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────┐
│ EMBEDDING TABLE (Vocabulary × Dimensions)                 │
│                                                            │
│ Token ID | Token Embedding (768 dimensions)               │
│ ---------|-----------------------------------------------  │
│   100    | [0.2, -0.5, 0.8, 0.1, ..., 0.3]  ("I")        │
│   250    | [0.6, 0.2, -0.3, 0.9, ..., 0.7]  ("love")     │
│   450    | [0.1, 0.8, 0.4, -0.2, ..., 0.5]  ("cats")     │
│   ...    | ...                                             │
│  30000   | [...]                                           │
│                                                            │
│ This table is LEARNED during training!                    │
└────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│ Add Position & Segment Embeddings              │
│                                                 │
│ Token 1: "I"    → Emb + Pos[0] + Seg[A]       │
│ Token 2: "love" → Emb + Pos[1] + Seg[A]       │
│ Token 3: "cats" → Emb + Pos[2] + Seg[A]       │
└─────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│ Feed to Transformer Layers                      │
│ (Self-attention makes it context-aware!)       │
└─────────────────────────────────────────────────┘
```

---

### Key Differences from Word2Vec

| Aspect | Word2Vec | Transformer (BERT) |
|--------|----------|-------------------|
| **Embedding Type** | Static (1 embedding per word) | Dynamic (context-aware) |
| **How Applied?** | Direct lookup | Lookup + Position + Processing |
| **"bank" meaning** | Always same vector | Changes based on context |
| **Example** | bank = [0.5, 0.2, ...] always | bank (river) ≠ bank (money) |

---

### Beginner Summary

**Q: Where are embeddings used in transformers?**

**A: In the Embedding Layer (Step 3)** - It converts token IDs into vectors using three components:

1. **Token Embedding** = What word? (from learned lookup table)
2. **Position Embedding** = Where in sentence? (position 1, 2, 3...)
3. **Segment Embedding** = Which sentence? (sentence A or B)

All three are **added together** and fed into the transformer layers.

**Q: What makes it better than Word2Vec?**

**A:** After the embedding layer, **self-attention** in transformer layers looks at surrounding words and adjusts the meaning. So "bank" near "river" gets different meaning than "bank" near "money" - this happens AFTER the initial embedding!

**Advantage**: Context-aware embeddings (bank = river bank vs financial bank)

---

## Visual: Semantic Space

```
                 High-dimensional space (768D)
                 Projected to 2D for visualization

                    pets
                     │
        ┌────────────┼────────────┐
        │            │            │
      dog  ←─────  cat  ─────→  kitten
        │            │            │
      puppy      animals       feline
        │                         │
    mammals                    meow


    finance                  water
        │                      │
      bank ←──────────────→  river
        │                      │
     money                   stream
        │                      │
     loan                     flow
```

---

## How Embedding Models Work

### Architecture Overview

```
┌──────────────────────────────────────────────────┐
│ Training Data: Large corpus (Wikipedia, books)   │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Training Objective (Contrastive Learning)        │
│                                                  │
│  Anchor: "Paris is the capital of France"       │
│  Positive: "France's capital city is Paris"     │
│  Negative: "Tokyo is in Japan"                  │
│                                                  │
│  Goal: Anchor ≈ Positive, Anchor ≠ Negative     │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Trained Embedding Model                          │
│ • Encodes semantic meaning                       │
│ • Similarity via cosine distance                 │
└──────────────────────────────────────────────────┘
```

### Top Performing Embedding Models in Market (2026)

**Ranked by MTEB (Massive Text Embedding Benchmark) Score**

| Rank | Model | MTEB Score | Dimensions | Max Tokens | Type | Cost | Key Benefits |
|------|-------|------------|------------|------------|------|------|--------------|
| 🥇 | **Voyage-3** | 72.3 | 1024 | 32000 | Paid | $0.06/1M tokens | Best overall, long context, domain-adaptive |
| 🥈 | **Cohere embed-v3** | 71.8 | 1024 | 512 | Paid | $0.10/1M tokens | Multilingual (100+ langs), excellent retrieval |
| 🥉 | **OpenAI text-embedding-3-large** | 71.5 | 3072 | 8191 | Paid | $0.13/1M tokens | High accuracy, flexible dimensions |
| 4 | **Nvidia NV-Embed-v2** | 71.0 | 4096 | 32768 | Free | Self-host | Long context, instruction-aware |
| 5 | **BGE-M3** | 70.2 | 1024 | 8192 | Free | Self-host | Multilingual, multi-granularity |
| 6 | **OpenAI text-embedding-3-small** | 68.9 | 1536 | 8191 | Paid | $0.02/1M tokens | Best price/performance ratio |
| 7 | **Jina AI v3** | 68.5 | 1024 | 8192 | Free/Paid | API/Self-host | Long context, task-specific modes |
| 8 | **BGE-large-en-v1.5** | 67.8 | 1024 | 512 | Free | Self-host | Best open-source for English |
| 9 | **GTE-large-v1.5** | 67.2 | 1024 | 512 | Free | Self-host | Fast, instruction-following |
| 10 | **E5-Mistral-7B-Instruct** | 66.9 | 4096 | 32768 | Free | Self-host | Long context, instruction-based |
| 11 | **all-MiniLM-L6-v2** | 58.5 | 384 | 256 | Free | Self-host | Fastest, lightweight, edge devices |

---

### Model Category Breakdown

#### 🏆 Best for Production (Paid)

```
┌─────────────────────────────────────────────────────┐
│ Voyage-3 - WINNER                                   │
├─────────────────────────────────────────────────────┤
│ ✓ Highest MTEB score (72.3)                        │
│ ✓ 32K token context (great for large docs)        │
│ ✓ Domain customization available                   │
│ ✓ Best for: Legal, Medical, Finance RAG           │
│ ✗ Cost: $0.06/1M tokens                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ Cohere embed-v3                                     │
├─────────────────────────────────────────────────────┤
│ ✓ Multilingual champion (100+ languages)          │
│ ✓ Compression mode (reduce storage 4x)            │
│ ✓ Best for: Global apps, cross-lingual search     │
│ ✗ Higher cost: $0.10/1M tokens                    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ OpenAI text-embedding-3-small                       │
├─────────────────────────────────────────────────────┤
│ ✓ Best price/performance (68.9 score)             │
│ ✓ Only $0.02/1M tokens                             │
│ ✓ Flexible dimension truncation                    │
│ ✓ Best for: Startups, MVP, cost-sensitive apps    │
└─────────────────────────────────────────────────────┘
```

#### 🆓 Best Free/Open-Source

```
┌─────────────────────────────────────────────────────┐
│ Nvidia NV-Embed-v2 - FREE WINNER                    │
├─────────────────────────────────────────────────────┤
│ ✓ Near-SOTA performance (71.0 score)              │
│ ✓ 32K context window                               │
│ ✓ Instruction-aware (follows query hints)         │
│ ✓ Best for: Self-hosted production                │
│ ✗ Requires GPU (A100/H100 recommended)            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ BGE-M3                                              │
├─────────────────────────────────────────────────────┤
│ ✓ Multi-lingual, multi-granularity                │
│ ✓ Works on consumer GPU (RTX 3090+)               │
│ ✓ Best for: Multilingual self-hosted RAG          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ all-MiniLM-L6-v2                                    │
├─────────────────────────────────────────────────────┤
│ ✓ Runs on CPU (no GPU needed!)                    │
│ ✓ 20MB model size (edge/mobile deployment)        │
│ ✓ Fastest inference (100x faster than large)      │
│ ✓ Best for: Prototyping, edge devices, demos      │
│ ✗ Lower accuracy (58.5 score)                     │
└─────────────────────────────────────────────────────┘
```

---

### Use Case Recommendations

| Your Scenario | Recommended Model | Why? |
|---------------|-------------------|------|
| **Startup MVP** | OpenAI text-embedding-3-small | Low cost, good quality, easy to start |
| **Enterprise RAG** | Voyage-3 or Cohere v3 | Best accuracy, support, SLA |
| **Multilingual App** | Cohere embed-v3 or BGE-M3 | Native multilingual support |
| **Self-hosted** | NV-Embed-v2 or BGE-large | Free, competitive quality |
| **Long Documents** | Voyage-3 or E5-Mistral | 32K+ context window |
| **Mobile/Edge** | all-MiniLM-L6-v2 | Tiny size, CPU-friendly |
| **Medical/Legal** | Voyage-3 (domain-adapted) | Domain customization |
| **Budget-constrained** | BGE-large-en-v1.5 | Free, solid performance |
| **Fastest Inference** | all-MiniLM-L6-v2 | Blazing fast |
| **Maximum Accuracy** | Voyage-3 | Highest MTEB score |

---

### Cost Comparison (1 Billion Tokens)

```
Model                          Cost for 1B tokens    GPU Cost (Self-host)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OpenAI text-embedding-3-small  $20,000              N/A
OpenAI text-embedding-3-large  $130,000             N/A
Cohere embed-v3                $100,000             N/A
Voyage-3                       $60,000              N/A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NV-Embed-v2 (self-host)        ~$1,000 (electricity) $5,000-15,000 (one-time)
BGE-large-en (self-host)       ~$500 (electricity)   $2,000-5,000 (one-time)
all-MiniLM (CPU)               ~$200 (electricity)   $0 (runs on CPU)

Break-even point: ~3-10M tokens/month for self-hosting
```

---

### Quick Selection Guide

```
┌────────────────────────────────────────────────┐
│ Decision Tree: Choose Your Embedding Model    │
└────────────────────────────────────────────────┘
                    ↓
        Can you self-host with GPU?
                ↓           ↓
              Yes           No
                ↓           ↓
        NV-Embed-v2    Need multilingual?
        or BGE-M3           ↓           ↓
                          Yes           No
                           ↓            ↓
                     Cohere v3    Budget < $20/M tokens?
                                       ↓           ↓
                                     Yes           No
                                      ↓            ↓
                              OpenAI-small   Voyage-3 or
                                            OpenAI-large
```

---

## Cost Reduction Strategies in Production

### 1. **Caching Strategy**

```
┌─────────────────────────────────────────────┐
│ User Query: "What is machine learning?"     │
└─────────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Check Cache (Redis)  │
         └──────────────────────┘
            ↓              ↓
         Found          Not Found
            ↓              ↓
    Return cached    Generate new
     embedding       embedding
            ↓              ↓
         Done       Store in cache
                           ↓
                        Return
```

**Savings**: 70-90% cost reduction for repeated queries

### 2. **Dimension Reduction**

```python
# Instead of 3072 dimensions (expensive)
OpenAI text-embedding-3-large: [0.23, -0.51, ..., 0.34] (3072-dim)
                                        ↓
                              Truncate to 1024-dim
                                        ↓
Reduced embedding: [0.23, -0.51, ..., 0.19] (1024-dim)

# Result: 67% storage reduction, minimal accuracy loss
```

### 3. **Batch Processing**

```
❌ Sequential (Expensive)
Doc 1 → API Call → Embedding 1
Doc 2 → API Call → Embedding 2
Doc 3 → API Call → Embedding 3
Total: 3 API calls

✅ Batch (Cost-effective)
[Doc 1, Doc 2, Doc 3] → Single API Call → [Emb 1, Emb 2, Emb 3]
Total: 1 API call

# Cost savings: Up to 3x cheaper
```

### 4. **Model Selection Based on Use Case**

```
┌─────────────────────────────────────────────────┐
│ Decision Tree: Which Embedding Model?           │
└─────────────────────────────────────────────────┘
                    ↓
        Need highest accuracy? ────No───→ Use smaller model
                    │                     (text-embedding-3-small)
                   Yes
                    ↓
        Can run locally? ────Yes──→ Use open-source
                    │                (BGE, Sentence-BERT)
                    No
                    ↓
        Use text-embedding-3-large
        with dimension reduction
```

### 5. **Self-Hosted Models**

```
Cloud API Cost (OpenAI):
├─ $0.13 per 1M tokens (small)
├─ $0.62 per 1M tokens (large)
└─ Ongoing per-request costs

Self-Hosted Cost:
├─ One-time GPU setup ($500-2000)
├─ ~$0.001 per 1M tokens (electricity)
└─ Fixed cost regardless of volume

Break-even: ~100M tokens/month
```

---

## Best Practices for RAG Embeddings

### 1. **Chunk Size Optimization**

```
┌─────────────────────────────────────────────────┐
│ Document: 10,000 words                          │
└─────────────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Chunking Strategy     │
         └──────────────────────┘
                    ↓
    ┌───────────────┼───────────────┐
    │               │               │
Too Small       Optimal        Too Large
(50 words)    (200-500 words)  (2000 words)
    │               │               │
Loss of       Good balance    Loss of
context       of context      precision
              and precision
```

**Recommendation**: 200-500 words with 50-word overlap

### 2. **Add Metadata to Chunks**

```
❌ Basic Chunk:
"The transformer architecture uses self-attention..."

✅ Enhanced Chunk:
"""
Document: Neural Networks Guide
Section: Transformer Architecture
Date: 2024
Tags: deep-learning, NLP

The transformer architecture uses self-attention...
"""

# Better retrieval accuracy due to richer context
```

### 3. **Hybrid Search Strategy**

```
┌──────────────────────────────────────────┐
│ User Query: "GPU memory optimization"    │
└──────────────────────────────────────────┘
            ↓               ↓
    ┌───────────┐    ┌──────────┐
    │ Vector    │    │ Keyword  │
    │ Search    │    │ Search   │
    │ (Semantic)│    │ (BM25)   │
    └───────────┘    └──────────┘
            ↓               ↓
    Results (0.85)   Results (0.92)
            ↓               ↓
        ┌─────────────────────┐
        │ Rerank & Combine    │
        │ (RRF or learned)    │
        └─────────────────────┘
                ↓
        Final Top-K Results

# Hybrid = Better than either alone
```

### 4. **Query Enhancement**

```
Original Query: "gpu fast"
        ↓
┌──────────────────────┐
│ Query Expansion      │
└──────────────────────┘
        ↓
Enhanced: "How to make GPU processing faster?
          Optimize CUDA kernels for speed.
          Reduce memory transfer overhead."
        ↓
Better Embeddings → Better Retrieval
```

### 5. **Fine-tune Embeddings for Domain**

```
┌─────────────────────────────────────────┐
│ Pre-trained Model (General knowledge)   │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Domain-specific Data                    │
│ • Medical records                       │
│ • Legal documents                       │
│ • Technical manuals                     │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Fine-tuning (Contrastive learning)      │
│ Similar pairs from your domain          │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ Custom Embedding Model                  │
│ 10-30% better accuracy on domain tasks  │
└─────────────────────────────────────────┘
```

---

## RAG-Specific Optimization Techniques

### 1. **Parent-Child Chunking**

```
┌────────────────────────────────────────┐
│ Parent Chunk (Context)                 │
│ "Chapter 5: Neural Networks"           │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ Child 1: "Feed-forward layers..." │ │
│  └──────────────────────────────────┘ │
│  ┌──────────────────────────────────┐ │
│  │ Child 2: "Activation functions..."│ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘

Process:
1. Embed child chunks (precise matching)
2. Retrieve matching child
3. Return full parent (rich context)
```

### 2. **Multi-Vector Retrieval**

```
Document Chunk
      ↓
┌─────────────┐
│ Generate:   │
│ • Summary   │ ─→ Embedding 1 (high-level)
│ • Questions │ ─→ Embedding 2 (query-focused)
│ • Keywords  │ ─→ Embedding 3 (precise terms)
└─────────────┘

Query matches any vector → Retrieve document
# Improves recall significantly
```

### 3. **Embedding Quality Metrics**

```
Evaluation Pipeline:

Test Set: Query-Document pairs
         ↓
┌────────────────────────────┐
│ Metrics:                   │
│ • Recall@K (top-K hits)   │
│ • MRR (Mean Reciprocal    │
│   Rank)                   │
│ • NDCG (ranking quality)  │
└────────────────────────────┘
         ↓
Iterate: Improve chunking,
         model selection,
         or fine-tuning
```

---

## Cost Comparison Table

| Strategy | Implementation | Cost Reduction | Complexity |
|----------|----------------|----------------|------------|
| **Caching** | Redis/Memcached | 70-90% | Low |
| **Batch Processing** | API batching | 50-70% | Low |
| **Dimension Reduction** | Truncate vectors | 30-67% | Low |
| **Self-hosting** | Open-source models | 90-99% | High |
| **Smaller Models** | Switch to mini models | 80% | Low |
| **Lazy Embedding** | Embed on-demand | Varies | Medium |

---

## Quick Decision Framework

```
┌─────────────────────────────────────────┐
│ Are you processing < 1M tokens/month?   │
└─────────────────────────────────────────┘
         Yes │              │ No
             ↓              ↓
    Use OpenAI API    Self-host or
    with caching      batch heavily
             │              │
             ↓              ↓
┌─────────────────────────────────────────┐
│ Need multilingual support?              │
└─────────────────────────────────────────┘
         Yes │              │ No
             ↓              ↓
    Cohere/OpenAI    BGE or SBERT
             │              │
             ↓              ↓
┌─────────────────────────────────────────┐
│ Is accuracy critical? (medical, legal)  │
└─────────────────────────────────────────┘
         Yes │              │ No
             ↓              ↓
    Fine-tune model   Use pre-trained
    on domain data    general model
```

---

## Key Takeaways

1. **Embeddings capture semantic meaning** in numerical form
2. **Modern transformers** provide context-aware embeddings
3. **Cache aggressively** - biggest cost saver (70-90%)
4. **Batch processing** - reduces API calls by 50-70%
5. **Right-size your model** - don't use 3072-dim if 384-dim works
6. **Chunk size matters** - 200-500 words with overlap
7. **Hybrid search** - combine vector + keyword search
8. **Fine-tune for domain** - 10-30% accuracy improvement
9. **Self-host at scale** - break-even at ~100M tokens/month
10. **Monitor and iterate** - track retrieval metrics continuously

---

## Production Checklist

- [ ] Implement caching layer (Redis)
- [ ] Use batch API calls where possible
- [ ] Choose right model size for use case
- [ ] Optimize chunk size (test 200-500 words)
- [ ] Add metadata to chunks
- [ ] Implement hybrid search
- [ ] Set up monitoring (recall, latency, cost)
- [ ] Consider self-hosting at scale
- [ ] Test with domain-specific queries
- [ ] Implement fallback mechanisms

---

## VectorDB Re-Indexing at Scale (100 to 50,000+ Documents)

### What is Re-Indexing in VectorDB?

Re-indexing is the process of **rebuilding the vector search index** from scratch (or partially) when the existing index no longer reflects the current state of your document corpus or embedding model.

A VectorDB index is a data structure (e.g., HNSW graph, IVF clusters) built on top of raw vectors to enable fast approximate nearest-neighbor (ANN) search. When the corpus grows significantly or the embedding model changes, the index becomes **stale, unbalanced, or misaligned** — and re-indexing restores correctness and performance.

```
Documents → Embed → Vectors → [INDEX] → Fast ANN Search
                                   ↑
                         Re-indexing rebuilds this
```

---

### Why Re-Indexing is Needed at Scale

| Trigger | Symptoms | Scale Threshold |
|---|---|---|
| Corpus grows from 100 → 50,000 docs | HNSW graph becomes unbalanced; recall drops | > 10x growth |
| Embedding model changed (e.g., text-ada-002 → text-embedding-3-large) | Old vectors are in a different semantic space | Any model swap |
| Chunk strategy changed | New chunks are a different size/shape than existing | Any chunking change |
| Metadata schema evolved | Filters return wrong results | Schema drift |
| Retrieval quality degrades | Relevant docs not returned in top-k | Ongoing monitoring |
| Index parameters suboptimal | HNSW `ef_construction` too low for new scale | Parameter tuning |

**Key insight:** HNSW is built incrementally — early-inserted vectors form the graph backbone. When your corpus was 100 documents, the graph was optimized for that scale. At 50,000 documents, the graph's connectivity and layer structure may be poorly suited, causing recall degradation even though all vectors exist.

---

### How Re-Indexing is Performed

#### Step 1 — Trigger Detection

Monitor these signals before deciding to re-index:

```python
# Retrieval quality monitor
def check_reindex_needed(vectordb, eval_queries, expected_docs):
    recall_scores = []
    for query, expected in zip(eval_queries, expected_docs):
        results = vectordb.similarity_search(query, k=10)
        retrieved_ids = {r.metadata["doc_id"] for r in results}
        hit = expected in retrieved_ids
        recall_scores.append(int(hit))

    recall_at_10 = sum(recall_scores) / len(recall_scores)

    # Trigger re-index if recall drops below threshold
    if recall_at_10 < 0.75:
        print(f"Recall@10 = {recall_at_10:.2f} — RE-INDEX RECOMMENDED")
    return recall_at_10
```

---

#### Step 2 — Choose Re-Index Strategy

```
Document count growth?
├── < 2x growth      → Incremental upsert (no re-index needed)
├── 2x–10x growth    → Incremental re-index (rebuild index params only)
└── > 10x growth     → Full re-index (rebuild everything)

Embedding model changed?
└── Always → Full re-index (vectors are incompatible across models)

Chunk strategy changed?
└── Always → Full re-index (new chunks = new vectors)

Zero downtime required?
└── Always → Blue/Green re-index
```

---

#### Step 3 — Full Re-Index Pipeline

Use when corpus grows > 10x, embedding model changes, or chunk strategy changes.

```python
import hashlib
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def full_reindex(documents: list, collection_name: str, new_collection: str):
    """
    Full re-index:
    1. Re-chunk all documents
    2. Re-embed all chunks
    3. Write to a NEW collection
    4. Swap the pointer
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Step 1 — Re-chunk
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    chunks = splitter.split_documents(documents)

    print(f"Re-chunked into {len(chunks)} chunks from {len(documents)} docs")

    # Step 2 — Re-embed and write to NEW collection (not old one)
    new_vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=new_collection,  # new_collection = "rag_v2"
        persist_directory="./chroma_db"
    )

    print(f"New index '{new_collection}' built with {len(chunks)} vectors")

    # Step 3 — Validate new index before swap
    test_query = "test retrieval quality"
    results = new_vectordb.similarity_search(test_query, k=5)
    assert len(results) > 0, "New index returned no results — abort swap"

    # Step 4 — Swap pointer (update config/env to point to new collection)
    return new_vectordb  # caller updates the live retriever reference

# Usage
new_db = full_reindex(all_documents, "rag_v1", "rag_v2")
# Update retriever to use new_db
```

---

#### Step 4 — Incremental Re-Index Pipeline

Use when only new documents are added and the embedding model is unchanged.

```python
def incremental_reindex(new_documents: list, vectordb: Chroma):
    """
    Incremental re-index:
    - Only process documents not already in the index
    - Uses fingerprint (hash) to skip unchanged docs
    - Deletes stale chunks for updated docs, re-embeds new version
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)

    added, skipped, updated = 0, 0, 0

    for doc in new_documents:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        doc_id = doc.metadata.get("source", "unknown")

        # Check if this doc+version already indexed
        existing = vectordb.get(where={"content_hash": content_hash})
        if existing["ids"]:
            skipped += 1
            continue

        # Check if an older version exists — delete stale chunks
        old_chunks = vectordb.get(where={"source": doc_id})
        if old_chunks["ids"]:
            vectordb.delete(ids=old_chunks["ids"])
            updated += 1
        else:
            added += 1

        # Chunk and embed the new/updated document
        doc.metadata["content_hash"] = content_hash
        chunks = splitter.split_documents([doc])
        vectordb.add_documents(chunks)

    print(f"Incremental re-index: {added} added, {updated} updated, {skipped} skipped")
    return vectordb
```

---

#### Step 5 — Zero-Downtime Blue/Green Re-Index

Use in production where the retriever must stay live during re-indexing.

```
BLUE  (current live)    GREEN (being re-indexed)
   rag_v1 collection  →      rag_v2 collection
        ↑                          ↑
   Live traffic            Background job

        After validation:
        Traffic → rag_v2 (GREEN becomes BLUE)
        rag_v1 deleted after grace period
```

```python
import threading

class BlueGreenReindexer:
    def __init__(self, blue_collection: str, green_collection: str):
        self.blue = blue_collection   # currently live
        self.green = green_collection # being built
        self.active = "blue"

    def build_green(self, documents: list):
        """Run in background — does not affect live traffic."""
        def _build():
            print(f"Building GREEN index: {self.green}")
            full_reindex(documents, self.blue, self.green)
            print(f"GREEN index ready: {self.green}")

        thread = threading.Thread(target=_build)
        thread.start()
        return thread

    def promote_green(self, retriever):
        """Swap live traffic to GREEN after validation."""
        # Validate GREEN before swap
        green_db = Chroma(
            collection_name=self.green,
            embedding_function=OpenAIEmbeddings()
        )
        results = green_db.similarity_search("validation query", k=5)
        assert len(results) > 0, "GREEN validation failed — keeping BLUE live"

        # Atomic swap — update retriever to point to green
        retriever.vectorstore = green_db
        self.active = "green"
        print(f"Swapped: BLUE={self.blue} → GREEN={self.green} is now live")

# Usage
reindexer = BlueGreenReindexer("rag_v1", "rag_v2")
build_thread = reindexer.build_green(all_documents)
build_thread.join()                     # wait for completion
reindexer.promote_green(live_retriever) # atomic traffic swap
```

---

### Impact of Re-Indexing

#### Performance Impact

| Metric | Before Re-Index (stale) | After Re-Index |
|---|---|---|
| Query latency | Increases as index grows unbalanced | Resets to baseline |
| Recall@10 | Can degrade to 60–70% | Restores to 85–95% |
| Throughput (QPS) | Drops with large unbalanced HNSW | Improves with tuned `ef_construction` |
| Index build time | N/A | 50K docs ≈ 15–45 min (depends on embed API speed) |

**HNSW-specific:** HNSW graph traversal degrades when the graph has poor connectivity from incremental insertions. A full re-index rebuilds the graph with all vectors present, producing optimal layer structure.

---

#### Cost Impact

| Cost Driver | Incremental Re-Index | Full Re-Index |
|---|---|---|
| Embedding API calls | Only new/changed docs | ALL docs re-embedded |
| Example: 50K docs × 512 tokens | ~$0 (reuse cached) | ~$2–$10 (OpenAI text-embedding-3-large) |
| Compute (self-hosted embed) | Low | High (GPU hours) |
| Downtime cost | Zero | Zero (if Blue/Green) |
| Storage during transition | 1x | 2x (Blue + Green coexist) |

**Cost optimization:** Cache embeddings externally (Redis / S3) keyed by `content_hash`. On re-index, only re-embed docs whose hash changed — skips 80–95% of API calls on incremental updates.

```python
import json, redis

cache = redis.Redis()

def get_or_embed(text: str, model) -> list:
    key = f"embed:{hashlib.md5(text.encode()).hexdigest()}"
    cached = cache.get(key)
    if cached:
        return json.loads(cached)

    vector = model.embed_query(text)
    cache.set(key, json.dumps(vector), ex=86400 * 30)  # 30-day TTL
    return vector
```

---

#### Quality Impact

| Scenario | Quality Change After Re-Index |
|---|---|
| Same embedding model, more docs | Neutral to slight improvement (better graph) |
| Better embedding model | Significant improvement (10–30% recall gain) |
| Improved chunking strategy | Moderate improvement (better semantic boundaries) |
| No model/chunk change, just re-build | Recall restored to original baseline |

**Important:** Re-indexing with the same embedding model and same chunks produces **identical vectors** — quality is restored, not improved. Quality improvement requires a model or chunking upgrade.

---

### Re-Index Decision Matrix

| Situation | Strategy | Downtime | Cost |
|---|---|---|---|
| New docs added, model unchanged | Incremental upsert | None | Low |
| Corpus grew 10x, model unchanged | Full re-index + Blue/Green | None | Medium |
| Embedding model upgraded | Full re-index + Blue/Green | None | High |
| Chunk size changed | Full re-index + Blue/Green | None | High |
| Recall degraded, no model change | Rebuild HNSW index params | Maintenance window | Low |
| Metadata schema changed | Incremental update (metadata only) | None | Very Low |

---

### Production Re-Index Checklist

- [ ] Monitor Recall@10 weekly — trigger re-index if below 0.80
- [ ] Cache embeddings by `content_hash` — avoid redundant API calls
- [ ] Always re-index to a NEW collection (Blue/Green) — never rebuild in-place on live
- [ ] Validate the new index with golden query set before traffic swap
- [ ] Keep the old collection alive for 24h after swap — fast rollback if issues arise
- [ ] Log re-index events: timestamp, strategy, docs processed, recall before/after
- [ ] Set HNSW `ef_construction` proportional to corpus size (e.g., 200 for 50K docs)
- [ ] Budget 2x storage temporarily during Blue/Green transition
- [ ] Alert on embed API errors during re-index — partial re-index is worse than no re-index
- [ ] Document which embedding model version was used for each index version


---

## Pre-Filtering vs Post-Filtering in VectorDB

### What is Filtering in RAG Retrieval?

When you have 50,000 documents, a raw ANN search scans **all vectors**. Filtering restricts which vectors participate in that search — either **before** (pre-filter) or **after** (post-filter) the ANN step.

```
Without filter:  Query → ANN over 50K vectors → top-k results
Pre-filter:      Query → [narrow to 5K via metadata] → ANN over 5K → top-k results
Post-filter:     Query → ANN over 50K → top-k*N results → [apply filter] → top-k results
```

Filtering uses **metadata fields** stored alongside each vector (e.g., `department`, `doc_type`, `access_level`, `date`).

---

### Pre-Filtering — How It Works

Pre-filtering restricts the candidate vector pool **before** the ANN search runs. The ANN engine only searches within vectors that satisfy the filter condition.

```
All Vectors (50K)
      │
   [WHERE department = "finance" AND doc_type = "policy"]
      │
   Candidate Pool (e.g., 3K vectors)
      │
   ANN Search (cosine similarity)
      │
   Top-k Results (e.g., k=5)
```

**Key properties:**
- ANN only touches the filtered subset — faster and cheaper at scale
- Results are **guaranteed** to satisfy the filter
- Risk: if the filtered subset is too small, ANN quality degrades (too few neighbors)

---

#### How to Apply Pre-Filtering

**Chroma (LangChain)**

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectordb = Chroma(
    collection_name="rag_docs",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Pre-filter: only search within finance + policy documents
results = vectordb.similarity_search(
    query="What is the expense reimbursement limit?",
    k=5,
    filter={
        "department": "finance",
        "doc_type": "policy"
    }
)

for doc in results:
    print(doc.metadata["source"], doc.page_content[:200])
```

**Pinecone**

```python
import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = PineconeVectorStore(
    index_name="rag-index",
    embedding=OpenAIEmbeddings()
)

# Pre-filter via Pinecone metadata filter syntax
results = vectorstore.similarity_search(
    query="expense reimbursement policy",
    k=5,
    filter={
        "department": {"$eq": "finance"},
        "doc_type": {"$in": ["policy", "guideline"]},
        "year": {"$gte": 2023}
    }
)
```

**Weaviate**

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="RagDocs",
    text_key="content",
    embedding=OpenAIEmbeddings()
)

# Pre-filter using Weaviate where clause
from weaviate.classes.query import Filter

results = vectorstore.similarity_search(
    query="expense reimbursement policy",
    k=5,
    filters=Filter.by_property("department").equal("finance")
)
```

---

#### Pre-Filter on Multiple Fields

```python
# Chroma compound filter
results = vectordb.similarity_search(
    query="data retention compliance",
    k=5,
    filter={
        "$and": [
            {"department": {"$eq": "legal"}},
            {"doc_type": {"$in": ["policy", "regulation"]}},
            {"effective_date": {"$gte": "2024-01-01"}},
            {"access_level": {"$eq": "internal"}}
        ]
    }
)
```

**Supported filter operators (Chroma / Pinecone):**

| Operator | Meaning | Example |
|---|---|---|
| `$eq` | Equal | `{"status": {"$eq": "active"}}` |
| `$ne` | Not equal | `{"status": {"$ne": "archived"}}` |
| `$in` | In list | `{"type": {"$in": ["pdf","doc"]}}` |
| `$nin` | Not in list | `{"type": {"$nin": ["image"]}}` |
| `$gte` / `$lte` | Range | `{"year": {"$gte": 2023}}` |
| `$and` | All conditions | `{"$and": [{...}, {...}]}` |
| `$or` | Any condition | `{"$or": [{...}, {...}]}` |

---

#### Dynamic Pre-Filter from User Context

In production, filters are built dynamically from the user's session context — not hardcoded.

```python
def build_filter(user_context: dict) -> dict:
    """
    Construct metadata pre-filter from user session context.
    user_context = {
        "department": "engineering",
        "access_level": "confidential",
        "language": "en"
    }
    """
    conditions = []

    if dept := user_context.get("department"):
        conditions.append({"department": {"$eq": dept}})

    if level := user_context.get("access_level"):
        # User can see their level and below
        level_map = {"public": 0, "internal": 1, "confidential": 2}
        allowed = [k for k, v in level_map.items() if v <= level_map.get(level, 0)]
        conditions.append({"access_level": {"$in": allowed}})

    if lang := user_context.get("language"):
        conditions.append({"language": {"$eq": lang}})

    return {"$and": conditions} if len(conditions) > 1 else conditions[0] if conditions else {}


def retrieve_with_user_filter(query: str, user_context: dict, vectordb, k: int = 5):
    metadata_filter = build_filter(user_context)
    return vectordb.similarity_search(query, k=k, filter=metadata_filter)


# Usage
user_ctx = {"department": "engineering", "access_level": "internal", "language": "en"}
results = retrieve_with_user_filter(
    "How do I deploy to production?",
    user_ctx,
    vectordb
)
```

---

#### Pre-Filtering with Self-Query Retriever

LangChain's `SelfQueryRetriever` does **both** vector search and metadata pre-filtering — it uses the LLM to split the natural language query into two parts: a **semantic search query** (for ANN vector search) and a **structured filter** (applied as a pre-filter on metadata). No manual filter construction needed.

**Internal flow:**
```
User query: "Show me finance policies from 2023 or later about expense limits"
                        │
               LLM (query constructor)
                        │
          ┌─────────────┴─────────────┐
     semantic_query              structured_filter
    "expense limits"         dept=finance, year>=2023
          │                          │
    embed + ANN search          pre-filter on metadata
    over filtered subset              │
          └─────────────┬─────────────┘
                   Top-k results
```

The LLM produces a structured object with **two fields**:

```
{
  "query": "expense limits",        ← embedded and used for ANN vector search
  "filter": {
    "department": {"$eq": "finance"},
    "doc_type":   {"$eq": "policy"},
    "year":       {"$gte": 2023}
  }
}
```

`metadata_field_info` only tells the LLM **which fields exist and what they mean** — it has no effect on the vector search itself. If the query has no filterable intent, the LLM returns `filter: None` and the full query goes to vector search.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

# Describe your metadata fields to the LLM
metadata_field_info = [
    AttributeInfo(name="department", description="Department that owns the document", type="string"),
    AttributeInfo(name="doc_type", description="Type: policy, guideline, report, contract", type="string"),
    AttributeInfo(name="year", description="Year the document was published", type="integer"),
    AttributeInfo(name="access_level", description="Access level: public, internal, confidential", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectordb,
    document_contents="Company policies and internal guidelines",
    metadata_field_info=metadata_field_info,
    verbose=True
)

# Query with filterable intent — LLM splits into semantic_query + filter
results = retriever.invoke("Show me finance policies from 2023 or later about expense limits")
# semantic_query → "expense limits"  (ANN search)
# filter         → {department: finance, doc_type: policy, year: {$gte: 2023}}

# Query with no filterable intent — full query goes to vector search, no filter applied
results = retriever.invoke("What is the expense policy?")
# semantic_query → "expense policy"  (ANN search over all docs)
# filter         → None
```

---

### Post-Filtering — How It Works

Post-filtering runs the ANN search over **all vectors first**, then discards results that don't meet the filter condition from the returned set.

```
All Vectors (50K)
      │
   ANN Search (cosine similarity) — no restriction
      │
   Top k*N results (e.g., fetch 50 instead of 5)
      │
   [Apply filter: department = "finance"]
      │
   Keep only matching results → final top-k
```

**Key properties:**
- ANN sees the full corpus — best recall for semantic similarity
- Results may be fewer than k if most top-N don't pass the filter
- Higher cost at scale (scans all 50K even if only 1K are relevant)

---

#### How to Apply Post-Filtering

```python
def retrieve_with_post_filter(
    query: str,
    vectordb,
    filter_fn,          # callable: doc -> bool
    k: int = 5,
    fetch_multiplier: int = 10   # fetch k*10 to compensate for filter loss
) -> list:
    """
    Post-filter pattern:
    1. Fetch k * multiplier candidates from ANN
    2. Apply filter_fn to narrow down
    3. Return top-k from passing results
    """
    candidates = vectordb.similarity_search(query, k=k * fetch_multiplier)
    passing = [doc for doc in candidates if filter_fn(doc)]
    return passing[:k]


# Example filter function
def is_finance_active(doc) -> bool:
    return (
        doc.metadata.get("department") == "finance" and
        doc.metadata.get("status") == "active"
    )


results = retrieve_with_post_filter(
    query="expense reimbursement limit",
    vectordb=vectordb,
    filter_fn=is_finance_active,
    k=5
)
```

**Post-filter with score threshold:**

```python
def retrieve_with_score_post_filter(query: str, vectordb, min_score: float = 0.75, k: int = 5):
    """Post-filter: only return results above a similarity score threshold."""
    docs_with_scores = vectordb.similarity_search_with_score(query, k=k * 5)

    filtered = [
        doc for doc, score in docs_with_scores
        if score >= min_score  # cosine similarity: higher = more similar
    ]
    return filtered[:k]
```

---

### Pre-Filter vs Post-Filter — Decision Guide

| Factor | Use Pre-Filter | Use Post-Filter |
|---|---|---|
| **Filter selectivity** | Filter removes > 70% of corpus | Filter removes < 30% of corpus |
| **Corpus size** | Large (10K–1M vectors) | Small (< 5K vectors) |
| **Result guarantee** | Must guarantee results satisfy filter | Approximate match acceptable |
| **Latency requirement** | Strict (< 100ms) | Relaxed (200–500ms ok) |
| **Filter fields** | Indexed metadata fields (exact match) | Complex logic / computed conditions |
| **ACL enforcement** | Always — security-critical | Never for security — use pre-filter |
| **Semantic purity** | Can accept slight recall trade-off | Maximize semantic relevance first |
| **Filter known at query time?** | Yes (from user session / request) | No (derived from result inspection) |

**Rules of thumb:**

```
Is the filter a hard access/security constraint?
└── YES → Always pre-filter (never post-filter security rules)

Does the filter narrow the corpus by > 50%?
└── YES → Pre-filter (saves ANN cost dramatically)
└── NO  → Post-filter (filter is loose; full ANN preserves recall better)

Is the filter condition computable from metadata alone?
└── YES → Pre-filter
└── NO  → Post-filter (e.g., "only results where LLM judges relevance > 0.8")

Are you getting too few results after filtering?
└── Pre-filter: increase k, or relax filter
└── Post-filter: increase fetch_multiplier (k*N)
```

---

### Hybrid: Pre-Filter + Post-Filter Together

In production, use both layers — pre-filter for hard constraints (security, tenant), post-filter for soft quality constraints (score threshold, LLM re-ranking).

```python
from langchain_openai import ChatOpenAI
from langchain.schema import Document

def hybrid_retrieval(
    query: str,
    user_context: dict,
    vectordb,
    k: int = 5,
    min_score: float = 0.70
) -> list[Document]:
    """
    Layer 1 — Pre-filter: hard constraints from user context (access, dept, lang)
    Layer 2 — ANN search within filtered subset
    Layer 3 — Post-filter: score threshold + LLM relevance check
    """

    # Layer 1: Pre-filter (hard constraints — enforced before ANN)
    pre_filter = build_filter(user_context)  # from earlier example

    # Layer 2: ANN search within pre-filtered pool, fetch extra for post-filter loss
    candidates = vectordb.similarity_search_with_score(
        query,
        k=k * 4,          # fetch 4x to compensate post-filter drop
        filter=pre_filter
    )

    # Layer 3a: Post-filter — score threshold
    score_filtered = [
        (doc, score) for doc, score in candidates
        if score >= min_score
    ]

    # Layer 3b: Post-filter — LLM relevance re-rank (optional, high-precision use cases)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def llm_is_relevant(doc: Document, query: str) -> bool:
        prompt = f"""Is the following document relevant to the query?
Query: {query}
Document: {doc.page_content[:500]}
Answer only YES or NO."""
        response = llm.invoke(prompt).content.strip().upper()
        return response == "YES"

    final = [
        doc for doc, _ in score_filtered
        if llm_is_relevant(doc, query)
    ]

    return final[:k]


# Usage
user_ctx = {"department": "finance", "access_level": "internal"}
results = hybrid_retrieval(
    "What is the travel expense policy for international trips?",
    user_ctx,
    vectordb,
    k=5,
    min_score=0.72
)
```

**Pipeline visualization:**

```
User Query + Context
       │
  [Pre-filter]  ← department=finance, access_level=internal
       │
  50K → 4K candidate vectors
       │
  [ANN Search]  ← cosine similarity
       │
  Top 20 results (k=5, fetch_multiplier=4)
       │
  [Post-filter: score >= 0.72]
       │
  12 results pass
       │
  [Post-filter: LLM relevance check]
       │
  Final 5 results → LLM context
```

---

### Common Pitfalls

| Pitfall | Problem | Fix |
|---|---|---|
| Pre-filter subset too small (< 50 docs) | ANN has too few candidates — recall collapses | Relax filter or increase k |
| Post-filter with no fetch multiplier | Fetch k=5, filter removes 4 → only 1 result | Always fetch k × 5–10 before post-filtering |
| Security rules in post-filter | ANN returns unauthorized docs; post-filter is too late | Security ACL must always be pre-filter |
| Filtering on un-indexed metadata field | Full scan instead of index lookup — slow | Only filter on fields declared as indexed |
| Self-query LLM extracts wrong filter | Bad filter silently returns no results | Add fallback: if results == 0, retry without filter |
| Filtering by date string instead of timestamp | String comparison ("2024-01-01") breaks with some DBs | Store dates as Unix timestamps (integers) |


---

## RAG Evaluation in Production

### Best Practice: Evaluating Retrieval Without Ground Truth

In production, you rarely have labelled ground truth. Use **LLM-as-judge** instead.

```
Query → Retriever → Chunks → LLM Judge → Relevance Score
                                ↑
                    "Does this chunk help answer the query?"
                    Score 0 or 1 per chunk — no ground truth needed
```

**Practical approaches:**

| Approach | How | Ground Truth? |
|---|---|---|
| **LLM-as-judge** | LLM scores chunk relevance per query | ✗ Not needed |
| **Implicit feedback** | Re-ask rate, thumbs down, session abandonment | ✗ Not needed |
| **Synthetic QA** | LLM generates Q&A from docs → eval test set | Self-generated |
| **Human spot-check** | Sample 50–100 queries weekly | ✗ Small set |

```python
# LLM-as-judge — no ground truth required
prompt = f"""
Query: {query}
Retrieved chunk: {chunk}
Does this chunk contain information relevant to answering the query?
Score 1 (relevant) or 0 (not relevant). Reply with just the number.
"""
score = llm.invoke(prompt)
precision = sum(scores) / len(scores)   # % relevant chunks retrieved
```

---

### Chunk Size vs Precision / Recall

```
SMALL CHUNKS:                         LARGE CHUNKS:
  Embedding focused → Precision ↑↑     Full answer fits → Recall ↑↑
  Answer split across chunks → Recall↓ Irrelevant content → Precision ↓↓
  Best for: retrieval step             Best for: LLM context step

  FIX: Parent-Child chunking
  Small child → embed + retrieve (Precision ↑)
  Large parent → send to LLM (Recall ↑)
  Both high — best of both worlds
```

---

### Evaluation Metrics Reference

| Metric | What | Ground Truth? | Tool |
|---|---|---|---|
| **Context Precision** | % retrieved chunks that are relevant | ✓ / LLM-judge | RAGAS, DeepEval |
| **Context Recall** | % relevant chunks actually retrieved | ✓ / LLM-judge | RAGAS, DeepEval |
| **Answer Relevance** | Does answer address the question? | ✗ LLM-judge | RAGAS |
| **Faithfulness** | Answer grounded in chunks (no hallucination)? | ✗ LLM-judge | RAGAS |
| **MRR** | Rank of first relevant chunk | ✓ | Custom |
| **NDCG** | Full ranking quality | ✓ | Custom |

**No ground truth → use:** Answer Relevance + Faithfulness + Context Relevance (all LLM-judge)
**With ground truth → use:** Context Precision + Context Recall + MRR

```
Context Precision = relevant retrieved / total retrieved   (are results clean?)
Context Recall    = relevant retrieved / total relevant    (did we miss any?)
```

---

## Token Pricing — Top Embedding Models

### Hosted / API (pay per token)

| Model | Provider | Dimensions | Max Tokens | Price (per 1M tokens) | Best For |
|---|---|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 | 8,191 | **$0.02** | Cost-efficient, general use |
| `text-embedding-3-large` | OpenAI | 3072 | 8,191 | **$0.13** | Higher accuracy, production RAG |
| `text-embedding-ada-002` | OpenAI | 1536 | 8,191 | **$0.10** | Legacy, replaced by v3 |
| `embed-english-v3.0` | Cohere | 1024 | 512 | **$0.10** | English-only, reranking |
| `embed-multilingual-v3.0` | Cohere | 1024 | 512 | **$0.10** | 100+ languages |
| `textembedding-gecko@003` | Google (Vertex AI) | 768 | 3,072 | **$0.025** | GCP-native stack |
| `amazon.titan-embed-text-v2` | AWS Bedrock | 1024 | 8,192 | **$0.02** | AWS-native stack |

---

### Open Source (self-host — free inference, pay only for compute)

| Model | Dimensions | Max Tokens | Size | Best For |
|---|---|---|---|---|
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | 335M | Best open-source English, MTEB top rank |
| `BAAI/bge-m3` | 1024 | 8,192 | 570M | Multilingual, long context, hybrid search |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | 22M | Lightweight, fast, low memory |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 384 | 110M | Balanced accuracy + speed |
| `intfloat/e5-large-v2` | 1024 | 512 | 335M | Strong for asymmetric search (query ≠ doc) |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 8,192 | 137M | Long context, Apache 2.0 license |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 512 | 335M | MTEB top performer, MIT license |

---

### How to use open source (HuggingFace)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Embed documents
doc_embeddings = model.encode(["Refunds processed in 7 days", "Store hours 9-5"])

# Embed query (bge models need query prefix for asymmetric search)
query_embedding = model.encode(["Represent this query: What is the refund policy?"])
```

---

### Hosted vs Open Source — when to pick which

```
HOSTED API:
  ✓ No GPU needed — zero infra
  ✓ Always latest model
  ✓ Pay per use (good for low volume)
  ✗ Data leaves your infrastructure
  ✗ Cost grows linearly with volume
  ✗ Latency depends on API

OPEN SOURCE (self-hosted):
  ✓ Data stays on-premise (GDPR, HIPAA, financial data)
  ✓ Free inference at scale — fixed compute cost
  ✓ Full control over model version
  ✗ GPU/CPU infra required
  ✗ You manage updates, scaling

RULE OF THUMB:
  < 10M tokens/month   → Hosted API (cheaper than GPU cost)
  > 10M tokens/month   → Self-hosted (compute cheaper than API fees)
  Regulated industry   → Self-hosted always (data sovereignty)
```

---

### Cost comparison at scale

```
10M tokens/month:
  text-embedding-3-small (OpenAI):  10M × $0.02/1M = $0.20/month
  text-embedding-3-large (OpenAI):  10M × $0.13/1M = $1.30/month

100M tokens/month:
  text-embedding-3-large:  $13/month  ← still cheap
  1B tokens/month:         $130/month ← consider self-hosting

Self-hosted (bge-large on 1× A10G GPU):
  ~$0.80/hour × 720hrs = ~$576/month
  Handles ~500M–1B tokens/month depending on batch size
  Break-even vs text-embedding-3-large: ~4.4B tokens/month
```

---

# Vector DB at Scale: Managing Exponential Growth & Retrieval Quality

## 🎯 Main Question: Vector DB Management with Exponential Growth

**Question:** Vector DB data ingested with 10 docs, it grows exponentially. How to manage the retrieval accuracies? Provide the best practices to validate retrieval quality. Is it possible to have golden dataset to ensure the retrieval quality?

### Answer (Beginner-Friendly):

**Problem Overview:**
When your vector database grows from 10 documents to millions, retrieval becomes harder because:
- More candidates to search through → Slower queries
- More variety in content → Harder to rank relevance
- Quality degrades without active management

**Best Practices for Managing Retrieval Accuracy at Scale:**

| Challenge | Solution | Why it Works |
|-----------|----------|-------------|
| **Lost relevance as DB grows** | Use metadata filtering + pre-ranking | Reduces search space from 1M to 100K docs first |
| **Embedding drift over time** | Periodically re-embed with latest model | Better semantic understanding as models improve |
| **Poor chunk boundaries** | Implement Parent-Child chunking | Small chunks for retrieval, large context for LLM |
| **Garbage in → Garbage out** | Quality gate: remove low-quality docs | 100 good docs > 1000 mediocre docs |
| **No visibility into quality** | Instrument with logging + metrics | Catch problems before users do |

**Can You Have a Golden Dataset?** ✅ **YES — Absolutely**

A **golden dataset** is a small, curated set of queries with known-good retrieved chunks and expected answers. Think of it as your "test suite" for RAG.

```
Golden Dataset Example:
┌─────────────────────────────────────────────────────────┐
│ Query: "What is our refund policy?"                     │
│ Expected Relevant Chunks: [chunk_5, chunk_12]           │
│ Expected Answer: "30 days with original receipt"        │
│ Metrics to Validate:                                    │
│   ✓ Recall@10 = 2/2 (both chunks in top-10)            │
│   ✓ Precision@10 = 2/10 (80% of results relevant)      │
│   ✓ LLM response matches expected answer                │
└─────────────────────────────────────────────────────────┘
```

**How to Build & Use Golden Dataset:**

1. **Start small** (50–100 queries): Covers ~80% of user patterns
2. **Quarterly review**: Update as docs/policies change
3. **Run against it weekly**: Detect regressions before production
4. **Use LLM-as-judge** (no ground truth needed):
   ```python
   # Score each retrieved chunk without labeled data
   prompt = f"Query: {query}\nChunk: {chunk}\nRelevant? (Yes/No)"
   relevance_score = llm(prompt)
   retrieval_precision = sum(scores) / len(scores)
   ```

---

## ❓ Follow-Up Questions: Deep Dives

### Q1: Parent-Child (Hierarchical) Chunking — Storage, Latency & Architecture

**Does it use 2 different DBs?** Not necessarily — it uses 2 different **STORES** within the same system:

```
┌────────────────────────────────────────────────────────┐
│             Your Application                           │
│                                                        │
│  ┌──────────────────────────┐  ┌──────────────────┐   │
│  │    Vector Store           │  │   Doc Store      │   │
│  │  (Qdrant/Pinecone/        │  │  (Redis/Mongo/   │   │
│  │   Weaviate/Milvus)        │  │   InMemory/S3)   │   │
│  │                           │  │                  │   │
│  │ ✓ Child chunks            │  │ ✓ Parent chunks  │   │
│  │ ✓ Embeddings (searchable) │  │ ✓ Raw text only  │   │
│  │ ✓ HNSW/IVF index          │  │ ✓ Key-value GET  │   │
│  └──────────┬────────────────┘  └──────────┬───────┘   │
│             │ vector_search(Q)             │ fetch_id  │
│             └────────────────┬─────────────┘           │
│                              ▼                        │
│                       Final Context                   │
└────────────────────────────────────────────────────────┘
```

**The Doc Store doesn't need vector capabilities** — it's just a simple key-value lookup (Redis, MongoDB, dict). This separation is crucial: **search** (vectors) lives in Vector Store, **storage** (text) lives in Doc Store.

**Storage Cost — Reality Check:**

```
WITHOUT Parent-Child (Naive):
  1000 chunks × (1536 dims × 4 bytes) = 6 GB embeddings
  + Text: 1000 × 512 tokens × 1 byte = 512 KB (negligible)
  Total: ~6 GB

WITH Parent-Child (Actual numbers):
  Child chunks:  4000 smaller chunks (from fragmentation)
                 4000 × 6 GB / 1000 = 24 GB embeddings
  
  Parent chunks: 1000 chunks plain text in docstore
                 1000 × 800 tokens × 1 byte = 800 KB
  
  This LOOKS like 4× worse! But context matters:

CORRECT INTERPRETATION:
  The 4 extra vectors are GOOD (small chunks match queries better)
  Plain text storage is free compared to vectors (100x cheaper)
  
  Realistic breakdown for SAME amount of content:
    Naive (1000 big chunks):      6 GB vectors
    Parent-Child (4000 small):   ~7.5 GB vectors (25% more)
    Doc store text:              negligible
    Mapping overhead:            negligible
    
  Total overhead: ~25% MORE storage (acceptable!)
```

**Real Latency Impact:**

```
WITHOUT Parent-Child:
  Query → Vector search → Return chunks
  Time: ~20 ms

WITH Parent-Child:
  Query → Vector search (child chunks) → Lookup parent ID → Fetch from docstore
  
  Breakdown:
    Vector search:    ~18 ms (more vectors, but well-indexed)
    Mapping lookup:   ~0.5 ms (in-memory hash table)
    Doc fetch:        ~2 ms (key-value lookup, O(1))
    ───────────────────────────
    Total:           ~20.5 ms
  
  Overhead: +0.5 ms (imperceptible!)
  
  The docstore is NOT a vector search — it's just GET chunk_id → return text.
  This is 100× faster than vector operations.
```

**Verdict: Always Use Parent-Child**

```
┌────────────────────────────────────────────────────┐
│ Cost:    +25% storage, +0.5 ms latency             │
│ Benefit: +15-25% retrieval precision               │
│ Benefit: Full context to LLM (no fragmentation)    │
└────────────────────────────────────────────────────┘

Exceptions (use naive only if):
  • Extreme memory constraints (< 2GB RAM)
  • Ultra-low latency (< 10ms requirement)
```

---

### Q2: How HNSW and IVF Work — Indexing Explained For Beginners

**The Problem They Solve:**
If you have 1M vectors and receive a query, comparing it to all 1M vectors is too slow. Indexes are smart shortcuts that reduce comparisons from 1M to ~100 (100,000× faster!).

#### HNSW (Hierarchical Navigable Small World) — GPS with Zoom Levels

**Mental Model: Navigate a map with zoom levels**

```
CONCEPT: Multi-layer graph where higher layers = sparse long-range jumps,
         lower layers = dense local neighborhoods

LAYER 2 (SPARSE, LONG-RANGE):
  A ─────────────────────── E
  │                         │
  └─────── C ──────────────┘

LAYER 1 (MEDIUM):
  A ── B ── C ── D ── E ── F
  │              │         │
  └──────────────┘─────────┘

LAYER 0 (DENSE, LOCAL):
  A─B─C─D─E─F─G─H─I─J─K─L─M─N─O─P
  (all 1M vectors here, heavily connected)

SEARCH TIME: Start layer 2 → zoom in → reach layer 0 ≈ O(log N) jumps!
```

**How Indexing Works (Step-by-Step):**

```python
# When a new vector arrives:

Step 1: Assign random max_layer (most vectors → layer 0, few → higher)
Step 2: Starting at highest layer, greedily navigate toward query
        (move to neighbor that's closer to query)
Step 3: When can't improve → drop one layer
Step 4: Repeat until reaching layer 0
Step 5: Connect this node to M nearest neighbors at each layer
Step 6: If any neighbor now has > Mmax connections → prune weakest ones

Result: Graph where neighbors ARE nearby in embedding space
```

**Real Numbers:**

```
1M vectors, 768-dim embeddings
Brute force: 1M × 768 = 768M ops = 5 seconds ❌

HNSW search:
  Start layer 2: Check ~5 entry points
  Layer 1: Check ~20 neighbors
  Layer 0: Check ~50 neighbors in final cluster
  Total: ~75 comparisons = 50 ms ✅ (100× faster!)
```

**Storage per vector:**

```python
node = {
    "id": "chunk_142",
    "vector": [0.12, -0.34, ...],        # 1536 × 4 bytes = 6144 bytes
    "neighbors": {
        2: ["chunk_001", "chunk_089"],              # 2 pointers
        1: ["chunk_001", "chunk_089", ...],         # 5 pointers
        0: ["chunk_001", "chunk_003", ...],         # 16 pointers (M=16)
    }
}

Memory for neighbors: (2 + 5 + 16) × 8 bytes = 184 bytes per vector
Overhead: 184 / 6144 = 3% extra memory

For 1M vectors: 6 GB + 6 GB × 3% = 6.18 GB
```

---

#### IVF (Inverted File Index) — Library Sections

**Mental Model: Find books by subject section, then search within**

```
CONCEPT: Divide vector space into K clusters (K=1000), each with a centroid.
To search, only check nearby clusters, not all vectors.

TRAINING PHASE (happens once):
  Run K-Means: Group 1M vectors into 1000 clusters
  Each cluster has centroid (center point)

        Centroid C1              Centroid C2
        (billing docs)           (refund docs)
           ●                         ●
          /|\                       /|\
         / | \                     / | \
       doc1 doc2 doc3           doc4 doc5 doc6


INDEXING PHASE:
  For each new vector:
    → Find nearest centroid
    → Add to that centroid's bucket

    Structure:
    {
      "C1": [doc_1, doc_4, doc_9, ...],        # 1000 docs
      "C2": [doc_2, doc_5, doc_12, ...],       # 950 docs
      ...
      "C1000": [doc_3, doc_7, ...]             # 1050 docs
    }


SEARCH PHASE:
  Query Q comes in
  → Find nprobe nearest centroids (e.g., 5 closest)
  → Search ONLY those 5 buckets (5000 vectors) 
  → Ignore other 995 buckets

  Without IVF: Search 1M vectors = 5 seconds
  With IVF:    Search 5K vectors = 10 ms ✅ (500× faster!)
```

**Real Numbers:**

```
1M vectors, IVF with K=1000 clusters
Each cluster: ~1000 vectors

Search strategy: nprobe=5 (search 5 nearest clusters)

Comparisons:
  Find nearest centroids: 1000 comparisons = 1 ms
  Search 5 buckets:       5000 comparisons = 5 ms
  Total: 6 ms ✅

vs Brute force: 1M comparisons = 5 seconds ❌

Speedup: 833×
```

**Storage for IVF:**

```
Centroids:       1000 × 1536 × 4 bytes = 6 MB
Inverted file:   1M × 2 bytes (cluster_id + frequency) = 2 MB
Vectors:         (same as stored) = 6 GB
Total overhead:  8 MB / 6 GB ≈ 0.1% (minimal!) ✅
```

---

#### HNSW vs IVF — Decision Guide

| Metric | HNSW | IVF |
|--------|------|-----|
| **Build time** | Incremental, ~1hr for 1M vecs | Batch K-Means, ~5 min for 1M vecs |
| **Query time** | 50 ms | 10 ms (faster!) |
| **Recall** | 99%+ (thorough) | 92% (may miss clusters) |
| **Memory overhead** | 3-5% | 0.1% (minimal) |
| **Add new vectors** | Fast (update graph) | Slow (rebuild clusters) |
| **Best for** | Online (continuous updates) | Offline (static corpus) |
| **Worst case** | Fully explore all layers | Query lands in wrong cluster |

**Rule of Thumb for Beginners:**

```
< 1M vectors    → HNSW (simpler, better recall)
1M–10M vectors  → HNSW with tuned parameters
> 10M vectors   → IVF + Product Quantization
Frequent updates → HNSW
Static corpus   → IVF
```

---

### Q3: Optimization Techniques for HNSW & IVF (Parameter Tuning Guide)

#### HNSW Optimizations

**Parameter 1: M (Maximum Degree — How many neighbors?)**

```
Think: "How many roads should connect each city?"

M=4   → Few connections → fast search, low recall (80%)
M=16  → DEFAULT, good balance (95% recall)
M=32  → Dense graph, slower, high recall (97%)
M=64  → Max quality, slowest build (98%+ recall)

Visual (Recall vs M):
│ Recall ▲
│ 98% │       M=64
│     │     ╱
│ 96% │   ╱  M=32
│     │ ╱  M=16
│ 94% │ M=4
│     └──────────────────────► Memory / Build Time

Rule of Thumb:
  M = 2 × (dim / 100)
  For 768-dim: M ≈ 2 × 7.68 = 15–16

Tuning Decision:
  Real-time API (< 50ms)   → M=16
  Search engine (accuracy) → M=32–48
  Mobile app (memory tight) → M=8–12
```

**Parameter 2: EF (Expansion Factor — How thoroughly to search?)**

```
Two uses (different meanings!):

EF_CONSTRUCT (during indexing):
  How many candidates to check when ADDING a new node
  Default: 200
  High (400): Slow to build, better graph quality
  Low (40):   Fast to build, lower quality graph
  
  ✓ Only affects BUILD time, not query time
  Invest in high EF_build on first index

EF_SEARCH (during queries):
  How wide to search at query time
  Default: 10
  Tuning (find sweet spot):
    EF=5:   Speed 40ms, Recall 85%
    EF=20:  Speed 65ms, Recall 95%  ← often best
    EF=50:  Speed 120ms, Recall 98%
  
  ✓ This is a RUNTIME parameter — tune per query
    Low EF: "I need sub-10ms, some misses OK"
    High EF: "Accuracy > speed"
```

**Parameter 3: mL (Layer Probability — Skip for beginners)**

```
Default mL = 1/ln(2) ≈ 1.44

This controls how many layers you build. Don't worry about this.
Default works fine. Only adjust if you're expert.
```

**Best Practice for HNSW:**

```python
# First index
vector_store = init_hnsw(
    M=16,           # balanced
    ef_construct=200,  # thorough
    ef_search=20    # balanced query time
)

# Then periodically:
if time_to_reindex():
    # Rebuild with high EF for best quality
    vector_store = init_hnsw(
        M=32,           # higher quality
        ef_construct=400,  # very thorough
        ef_search=20
    )
```

---

#### IVF Optimizations

**Parameter 1: n_list (Number of Clusters)**

```
Think: "How many library sections should we create?"

n_list = 100:    Each bucket has 10k vectors, search slow
n_list = 1000:   Balanced (each bucket has 1k vectors)
n_list = 10000:  Each bucket has 100 vectors, search fast but might miss

Rule of thumb:
  n_list = 4 × sqrt(total_vectors)
  1M vectors    → n_list ≈ 4 × 1000 = 4000
  10M vectors   → n_list ≈ 4 × 3162 = 12,648

Visual:
n_list         Cluster Size  Search Speed  Recall
100            10K vectors   Slow          95%
1000           1K vectors    Fast          92%  ← balanced
10000          100 vectors   Very fast     87%
```

**Parameter 2: nprobe (How many clusters to search?)**

```
Think: "How many library sections should we check?"

nprobe=1    → Check only 1 cluster ≈ 80% recall, < 1ms
nprobe=5    → Check 5 clusters ≈ 90% recall, 5ms
nprobe=20   → Check 20 clusters ≈ 96% recall, 20ms
nprobe=100  → Check many, ≈ 99% recall, 100ms

THIS IS THE MAIN QUALITY KNOB at query time!

Tuning Strategy:
  Step 1: Set nprobe=1, measure recall
  Step 2: Increase nprobe by 5 increments
  Step 3: Stop when recall ≥ target (usually 95%)

Example tuning:
  nprobe=1: Recall 80% ← Too low
  nprobe=5: Recall 88% ← Better
  nprobe=10: Recall 93% ← Close
  nprobe=15: Recall 96% ← ✓ GOOD, use this
  nprobe=20: Recall 97% ← Overkill
```

**Parameter 3: PQ (Product Quantization — Advanced)**

```
For massive scale (> 100M vectors), compress embeddings

PQ m=96, bits=8:  96 bytes/vector, 5% accuracy loss
PQ m=48, bits=8:  48 bytes/vector, 10% accuracy loss
PQ m=96, bits=4:  48 bytes/vector, 12% accuracy loss

Without PQ: 1M × 1536 × 4 = 6 GB
With PQ:    1M × 96 × 1 = 96 MB (60× compression!)

Tradeoff: compression vs accuracy. Usually worth it at extreme scale.
```

---

#### Decision Framework for Beginners

```
┌────────────────────────────────────────────┐
│ My dataset size...  Use this config        │
├────────────────────────────────────────────┤
│ < 100K vectors      HNSW, M=16, EF=100    │
│                     (default, just works)  │
│                                            │
│ 100K–1M vectors     HNSW, M=16, EF=150    │
│                     (tune EF for recall)  │
│                                            │
│ 1M–10M vectors      HNSW, M=32, EF=200    │
│                     (higher quality)      │
│                                            │
│ 10M+ vectors        IVF+PQ                 │
│                     n_list = 4×sqrt(N)    │
│                     nprobe = 10–20        │
│                                            │
│ Frequent updates    Always HNSW           │
│ Static corpus       IVF (rebuild 1×/week) │
│ Extreme memory      IVF+PQ aggressive     │
└────────────────────────────────────────────┘
```

**3. Cluster Re-balancing**
```
If clusters become imbalanced (some have 10k vectors, others have 10),
search becomes uneven → some queries slow.

Solution: Periodic re-clustering (weekly/monthly)
```

**4. Combined IVF + HNSW**
```
Modern approach: Use IVF to narrow to top clusters,
then use HNSW within each cluster for fine-grained search.

Speed: 100× faster than pure brute force
Recall: Nearly 99%+ (nearly as good as exact search)
```

---

### Q4: Hybrid Search — Does It Increase Storage & Latency?

**Short Answer:** Yes to both, but the retrieval quality improvement outweighs the costs.

**Storage Breakdown:**

```
PURE DENSE (Vector-only):
  1M vectors × 768 dims × 4 bytes = 3.1 GB

PURE BM25 (Keyword-only):
  1M chunks × avg 100 words × avg 6 bytes/word (inverted index)
  = ~600 MB (highly compressible)

HYBRID (Dense + BM25):
  Dense: 3.1 GB
  BM25 inverted index: 600 MB
  Mapping (chunk_id → vector_id): ~10 MB
  ─────────────────────────────
  Total: ~3.8 GB (≈ 22% more than pure dense)
```

**Latency Breakdown:**

```
PURE DENSE SEARCH:
  Vector search: 50 ms
  Total: 50 ms

PURE BM25 SEARCH:
  Inverted index lookup: 10 ms
  Total: 10 ms (faster!)

HYBRID SEARCH (Parallel):
  Vector search: 50 ms  ┐
  BM25 search: 10 ms   ├─ Run in parallel
  ────────────────────  │
  Max(50, 10) = 50 ms  ┘
  Fusion/Reranking: 20 ms
  ─────────────────────────
  Total: ~70 ms (40% slower than pure dense)

HYBRID SEARCH (Sequential):
  Vector search: 50 ms
  BM25 search: 10 ms
  Fusion: 20 ms
  Total: 80 ms (60% slower)

BEST PRACTICE: Parallel execution → ~70 ms
```

**Quality Improvement (Why Worth It):**

```
Query: "iPhone 15 release date"

PURE DENSE:
  ✓ Good: Matches semantic similarity
  ✗ Bad: "phone release announcement" ranked higher than exact match

PURE BM25:
  ✓ Good: "iPhone" and "15" exact keywords found
  ✗ Bad: "Why iPhones are good" (has keywords but irrelevant)

HYBRID (Fusion):
  ✓ Combines strength of both
  ✓ "iPhone 15 release" gets top rank
  ✓ Reduces noise, high precision

METRIC IMPROVEMENT:
  BM25 only:       Precision@5 = 60%
  Dense only:      Precision@5 = 75%
  Hybrid fusion:   Precision@5 = 92%  ⬅️ Best
```

**Storage & Latency Verdict:**
- 22% extra storage is acceptable for 17-point precision gain
- 70ms latency is acceptable (imperceptible to users)
- **Net benefit: Hybrid search is THE production standard**

---

### Q5: What Are "relevant_doc_ids" and How Are They Generated?

**Definition:**
`relevant_doc_ids` is a list of document/chunk IDs that **actually contain the answer** to a given query. It's used to evaluate retrieval quality.

```
EXAMPLE:
Query:  "What is our refund policy?"
Corpus: 1000 documents

Relevant docs:
  Document IDs: [doc_5, doc_12, doc_18]
  ↓
  All 3 documents contain refund policy info

During RETRIEVAL EVALUATION:
  Retriever returns top-5: [doc_2, doc_5, doc_7, doc_12, doc_999]
  ↓
  Compare against relevant_doc_ids: [doc_5, doc_12, doc_18]
  ↓
  Match: doc_5 ✓, doc_12 ✓  (doc_18 was missed)
  ↓
  Recall@5 = 2/3 = 66.7%
```

**How to Generate relevant_doc_ids:**

#### Method 1: **Manual Annotation (Gold Standard)**
```
Process:
1. Domain expert reads each document
2. For every query in test set, marks which docs are relevant
3. Creates ground truth

Pros: Highest quality, unambiguous
Cons: Slow, expensive, doesn't scale beyond 1000 docs

Cost: ~2 min per query × 100 queries = 3–5 hours labor
Result: 100 curated Q&A pairs with ground truth
```

#### Method 2: **LLM-as-Judge (No Ground Truth Needed)**
```
Process:
1. For each document, ask LLM: "Does this answer the query?"

prompt = f"""
Query: {query}
Document: {doc}

Does this document help answer the query?
Respond with RELEVANT or NOT_RELEVANT.
"""
relevant = llm(prompt)

2. Collect all documents marked RELEVANT

Pros: Fast, scales to millions of docs, no manual work
Cons: Depends on LLM quality, can have biases

Cost: ~1 second per doc (parallel) = automated
Result: Relevance scores for entire corpus
```

#### Method 3: **Synthetic Generation (Semi-Automated)**
```
Process:
1. Take existing documents
2. Use LLM to generate Q&A pairs FROM each document

for doc in corpus:
    questions = llm.generate_questions(doc)  # 2-3 questions per doc
    for q in questions:
        relevant_doc_ids[q] = [doc]

3. Now you have synthetic queries with known answers

Pros: Automatically creates test set, scales to billions
Cons: Quality depends on generated questions (may not reflect real users)

Cost: ~1 sec per doc to generate questions
Result: 3K test queries from 1K documents
```

#### Method 4: **User Behavior (Implicit Feedback)**
```
Process:
1. In production, track which docs users click/upvote after retrieval
2. If user clicks doc_5 after query Q → mark [doc_5] as relevant

Pros: Real user signal, no annotation needed
Cons: Sparse (not all relevant docs are clicked), biased (UI affects clicks)

Data point:
  User query: "refund policy"
  Retrieved chunks: [chunk_2, chunk_5, chunk_7]
  User action: Clicks chunk_5 → chunk_5 marked relevant
```

---

**Complete Example: Building relevant_doc_ids for Your Golden Dataset**

```python
# Mix of methods for robustness

# Method 1: Manually annotate 50 critical queries
manual_queries = {
    "What is the refund policy?": [doc_5, doc_12],
    "How do I create an account?": [doc_8, doc_15],
    # ...
}

# Method 2: Use LLM to generate 50 more from remaining docs
synthetic_queries = llm.generate_qna_from_docs(corpus, num_queries=50)

# Method 3: Monitor real user queries for 2 weeks
user_queries = production_system.get_user_feedback(days=14)

# Combine all
relevant_doc_ids = {
    **manual_queries,
    **synthetic_queries,
    **user_queries
}

# Evaluate on this mixed set weekly
for query, relevant_ids in relevant_doc_ids.items():
    retrieved = retriever.search(query, top_k=10)
    matches = len(set(retrieved) & set(relevant_ids))
    recall = matches / len(relevant_ids)
    print(f"Query: {query} → Recall@10: {recall}")
```

---

## 🏁 Summary Table: Managing Large Vector DBs

| Aspect | Best Practice | Why |
|--------|---------------|-----|
| **Retrieval Quality** | Golden dataset + weekly evals | Catch regressions early |
| **Chunking** | Parent-Child hierarchical | Precision + recall balance |
| **Indexing** | HNSW for production (IVF for batch) | Faster, better recall |
| **Search** | Hybrid (Dense + BM25) | 92% precision vs 75% pure dense |
| **Storage** | Accept 30% overhead | Quality worth the cost |
| **Latency** | 70–100ms acceptable | User imperceptible |
| **Growth** | Quarterly re-evaluation | Ensure quality as DB grows |