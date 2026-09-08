05_Multimodal_AI_System_Design.md
# Multimodal AI System Design Interview Questions

## Q1: Design a Multimodal RAG System for Document Understanding

**Context**: Build a multimodal RAG system that processes PDFs, images, tables, and text to answer complex questions like "Show me all transactions with high-risk merchants from Q4 reports." Required: Support 100K+ documents with mixed modalities, sub-2s latency, high accuracy.

### Follow-up Questions:
- How would you handle different modalities (text vs images vs tables)?
- What's your chunk strategy for multimodal content?
- How do you embed different modalities into the same vector space?
- What's your approach to table/structured data retrieval?

### Architecture Diagram:
```
┌──────────────────────────────────────────────────────────┐
│        MULTIMODAL RAG SYSTEM (PDF/Image/Table)          │
│                  Document Intelligence                   │
└──────────────────────────────────────────────────────────┘

INGESTION LAYER
┌──────────────────────────────────────────────────────────┐
│  Document Parsing & Preprocessing                        │
│                                                           │
│  PDF Input:                                              │
│  ├─ Extract text (PyPDF2, pdfplumber)                  │
│  ├─ Detect images (scans, charts, embedded images)     │
│  ├─ Identify tables (layout analysis)                   │
│  ├─ OCR for handwritten/scanned content                │
│  └─ Preserve layout information                         │
│                                                           │
│  Image/Chart Processing:                                │
│  ├─ Detect type: Graph, chart, diagram, photo          │
│  ├─ Extract text via OCR (Tesseract, GCP Vision)       │
│  ├─ Generate description (LLM/Vision model)            │
│  ├─ Extract data points (chart parser)                 │
│  └─ Spatial relationships (objects in image)           │
│                                                           │
│  Table Processing:                                       │
│  ├─ Identify table structure (rows, columns)           │
│  ├─ Extract row/column headers                         │
│  ├─ Preserve cell contents (text/numbers)              │
│  ├─ Convert to structured format (JSON/CSV)            │
│  └─ Index for SQL-like queries                         │
│                                                           │
│  Metadata Extraction:                                    │
│  ├─ Document type: report, policy, contract            │
│  ├─ Date, author, classification                       │
│  ├─ Section hierarchy (chapter > section > subsection) │
│  └─ Referenced entities (merchant, customer, transaction)│
└──────────────────┬───────────────────────────────────────┘
                   │
CHUNKING STRATEGY
┌──────────────────▼───────────────────────────────────────┐
│  Multimodal Chunk Types                                  │
│                                                           │
│  Type 1: Text Chunks                                    │
│  ├─ Size: 512 tokens (sliding window)                  │
│  ├─ Overlap: 50% (for context continuity)              │
│  ├─ Boundary: Sentence-aware (no mid-sentence cuts)   │
│  └─ Metadata: Page number, section, document ID        │
│                                                           │
│  Type 2: Image Chunks                                  │
│  ├─ Original image + extracted text + description      │
│  ├─ Image metadata: dimensions, type, location         │
│  ├─ Extracted entities: labels, numbers, trends        │
│  └─ Context: Surrounding paragraph (50 tokens)         │
│                                                           │
│  Type 3: Table Chunks                                  │
│  ├─ Full table as structured data (JSON)               │
│  ├─ Row-level chunks (for large tables)                │
│  ├─ Headers: Column names for SQL querying             │
│  └─ Summary: Key statistics computed                   │
│                                                           │
│  Type 4: Hybrid Chunks (Complex)                       │
│  ├─ Chart + caption + surrounding text                 │
│  ├─ Table + interpretation in text                     │
│  └─ Multi-modal coherence maintained                   │
└──────────────────┬───────────────────────────────────────┘

EMBEDDING & INDEXING
┌──────────────────▼───────────────────────────────────────┐
│  Unified Vector Space (Multi-Modal)                      │
│                                                           │
│  Text Embeddings:                                        │
│  ├─ Model: text-embedding-3-large (3072 dims)          │
│  ├─ Chunks embedded individually                        │
│  ├─ Batch size: 128 chunks/call                         │
│  └─ Cost: $0.02 per 1M tokens                           │
│                                                           │
│  Image Embeddings:                                       │
│  ├─ Model: CLIP (vision-language, 512 dims)            │
│  ├─ Images embedded to same space as text              │
│  ├─ Technique: Projection layer (aligns 512→3072 dims) │
│  └─ Cost: $0.01 per image                              │
│                                                           │
│  Table Embeddings:                                      │
│  ├─ Strategy 1: Embed as text (SQL + headers)          │
│  ├─ Strategy 2: Embed row-level summaries              │
│  ├─ Strategy 3: Dense retrieval + sparse (BM25)        │
│  └─ Hybrid: Text embedding + structured index           │
│                                                           │
│  Cross-Modal Alignment (Optional):                      │
│  ├─ Fine-tune alignment layer on pp data           │
│  ├─ Goal: Image of transaction ≈ Text description      │
│  ├─ Loss: Triplet loss (image, text, hard negatives)   │
│  └─ Benefit: Better multimodal understanding           │
│                                                           │
│  Vector Index:                                           │
│  ├─ Database: Milvus (1M+ vectors)                      │
│  ├─ Index type: HNSW (hierarchical navigable)          │
│  ├─ Metadata: Document ID, page, chunk type            │
│  └─ Query speed: <50ms for k=20                        │
└──────────────────┬───────────────────────────────────────┘

RETRIEVAL
┌──────────────────▼───────────────────────────────────────┐
│  Multi-Modal Query Understanding                         │
│                                                           │
│  Query Processing:                                       │
│  ├─ Intent classification:                              │
│  │  ├─ Asking for text info? (→ text retrieval)        │
│  │  ├─ Asking for image/chart? (→ image retrieval)     │
│  │  ├─ Asking for numbers/data? (→ table retrieval)    │
│  │  └─ Asking for synthesis? (→ all modalities)        │
│  ├─ Entity extraction: Dates, merchant names, amounts  │
│  ├─ Query expansion: HyDE synthetic docs               │
│  └─ Query embedding: Same space as indexed chunks      │
│                                                           │
│  Retrieval Strategy (Hybrid):                           │
│  ├─ Step 1: Semantic search (vector similarity)        │
│  │  └─ Top-20 chunks from all modalities               │
│  ├─ Step 2: Keyword search (BM25)                      │
│  │  └─ Top-10 based on exact term matching             │
│  ├─ Step 3: Table queries (SQL-like)                   │
│  │  └─ If entities found: SELECT ... WHERE ...        │
│  ├─ Step 4: Fusion (RRF - Reciprocal Rank Fusion)     │
│  │  └─ Combine results, de-duplicate, rank            │
│  └─ Output: Top-5 diverse chunks (multimodal mix)      │
│                                                           │
│  Caching:                                                │
│  ├─ Redis semantic cache (L1): 30% hit rate           │
│  ├─ Common queries cached (date ranges, merchants)     │
│  ├─ Cache key: query embedding + entity filters        │
│  └─ TTL: 24h                                            │
└──────────────────┬───────────────────────────────────────┘

RANKING & REFINEMENT
┌──────────────────▼───────────────────────────────────────┐
│  Multimodal Reranking                                    │
│                                                           │
│  Stage 1: Fast Reranking                               │
│  ├─ Relevance scoring per chunk type                   │
│  ├─ Penalize text-only for image-specific questions    │
│  ├─ Boost tables for numeric queries                   │
│  └─ Diversity: Don't return 5 text chunks (mix types)  │
│                                                           │
│  Stage 2: Semantic Reranking (LLM)                     │
│  ├─ Cross-encoder: "Is chunk relevant to query?"       │
│  ├─ Model: miniLM (fast, 50ms per batch of 5)         │
│  ├─ Threshold: Score > 0.6 for inclusion              │
│  └─ Result: Final top-3 chunks                         │
│                                                           │
│  Stage 3: Modality-Specific Ranking                    │
│  ├─ Image chunks: Ranked by visual similarity          │
│  ├─ Table chunks: Ranked by cell relevance             │
│  └─ Text chunks: Ranked by semantic similarity         │
└──────────────────┬───────────────────────────────────────┘

GENERATION
┌──────────────────▼───────────────────────────────────────┐
│  Multimodal Answer Synthesis                             │
│                                                           │
│  Prompt Construction:                                    │
│  ├─ System prompt: Role as pp domain expert        │
│  ├─ Retrieved context: Top-3 chunks (text + images)    │
│  ├─ Instructions: Format answer, cite sources          │
│  ├─ User query: Original question                      │
│  └─ Special handling:                                   │
│  │  ├─ If image chunk: "Image shows [description]"    │
│  │  ├─ If table chunk: "From the table: ..."          │
│  │  └─ If text chunk: "According to documentation..."  │
│                                                           │
│  Generation (Vertex AI):                                │
│  ├─ Model: Llama 3.1 (8B, fine-tuned)                 │
│  ├─ Temperature: 0.3 (deterministic, factual)         │
│  ├─ Max tokens: 500 (concise answers)                 │
│  ├─ Stop sequences: ["\n\n", "###"]                   │
│  └─ Latency: 50-100ms                                 │
│                                                           │
│  Multi-Modal Response:                                  │
│  ├─ If image retrieved: Embed in response             │
│  ├─ If table retrieved: Format as markdown table      │
│  ├─ Citations: Link to source document + page         │
│  ├─ Confidence: "High confidence (3 supporting sources)"│
│  └─ Format: Text + embedded images + tables           │
└──────────────────┬───────────────────────────────────────┘

EVALUATION & MONITORING
┌──────────────────▼───────────────────────────────────────┐
│  Multimodal Quality Metrics                              │
│                                                           │
│  RAGAS Metrics (Extended):                             │
│  ├─ Faithfulness: Does answer follow docs? (>0.85)    │
│  ├─ Image accuracy: Correct image retrieved? (>0.9)   │
│  ├─ Table correctness: Data extracted correctly? (>0.95)│
│  ├─ Modality match: Query type matches retrieved mode? │
│  └─ Citation accuracy: Links point to right content?  │
│                                                           │
│  User Feedback:                                         │
│  ├─ Thumbs up/down on response                         │
│  ├─ "Image was/wasn't helpful"                         │
│  ├─ "Answer matched/didn't match reality"              │
│  ├─ Time to resolution (support use case)              │
│  └─ Used for continuous improvement (reranking tuning) │
│                                                           │
│  Monitoring Dashboard:                                   │
│  ├─ Query success rate: 95%+                           │
│  ├─ Latency: p95 < 2s (including image embedding)     │
│  ├─ Modality distribution: Text 40%, Image 30%, Table 30%│
│  ├─ Reranking improvement: (before → after ranking)   │
│  └─ Cost per query: <$0.02 (with caching)             │
└──────────────────────────────────────────────────────────┘
```

### Key Implementation Details:

**Cross-Modal Embedding Alignment**:
```
Without alignment:
├─ "Chart showing merchant risk" (text) ≠ Chart image
├─ Different embedding spaces
└─ Poor retrieval for multimodal questions

With alignment (Fine-tuning):
├─ Text description embedding ≈ Chart image embedding
├─ Same vector space
└─ Multimodal questions retrieve relevant content

Training Data:
├─ 5K pairs: {image, descriptive text}
├─ Triplet loss: anchor, positive, hard negative
├─ Fine-tune projection layer (512 dims → 3072 dims)
└─ Result: +15% multimodal retrieval accuracy
```

**Table Handling Strategy**:
```
Option 1: Text Representation (Simple)
├─ Convert table to text: "Row 1: Name=ABC, Risk=High"
├─ Embed as text
├─ Pros: Simple, works with text embeddings
└─ Cons: Loses tabular structure

Option 2: SQL + Text (Recommended)
├─ Store table as SQL-queryable database
├─ Embed headers + summary statistics
├─ Retrieve via keyword match
├─ Execute query on retrieved table
├─ Pros: Precise, handles complex queries
└─ Cons: Requires SQL generation

Option 3: Hybrid (Best for pp)
├─ Dense retrieval: Find relevant tables (semantic)
├─ Structured retrieval: Column/row filtering (explicit)
├─ Execute query on filtered subset
└─ Result: Precise + semantic
```

---

## Q2: Design a Video Understanding System for Compliance & Training

**Context**: Build a system that automatically analyzes training videos, compliance recordings, or meeting recordings to extract insights, generate summaries, and answer questions about content.

### Key Components:

```
Video Processing Pipeline:

Frame Extraction:
├─ Sample every 1 second (30 fps → 1 per second)
├─ Scene detection (significant changes)
├─ Key frame identification (high information content)
└─ Result: 100-500 frames per minute of video

Audio Processing:
├─ Speech-to-text (Whisper, high accuracy)
├─ Speaker diarization (who spoke when?)
├─ Timestamp synchronization
├─ Transcript: Time-aligned with frames
└─ Sentiment/tone analysis

Vision Understanding:
├─ Object detection: Identify relevant entities
├─ OCR: Extract text from slides/graphics
├─ Scene understanding: What's happening?
├─ Action recognition: Person typing, presenting, etc.

Multimodal Fusion:
├─ Combine: Audio transcript + key frames + OCR
├─ Temporal alignment: Match speech with visuals
├─ Generate comprehensive embeddings
└─ Index for semantic search

QA System:
├─ User question: "When did we discuss merchant risk?"
├─ Retrieve relevant video segments (clips)
├─ Generate answer with timestamps + transcript excerpt
├─ Show relevant frame screenshots
└─ Accuracy: >90% on test set
```

---

## Q3: Design a Document Intelligence System (Invoices, Receipts, Forms)

**Context**: Automatically extract structured information from unstructured documents (invoices, receipts, forms, contracts) with high accuracy and compliance.

### Key Features:

```
Document Classification:
├─ CNN classifier: Invoice, Receipt, Contract, Report
├─ Accuracy: 97%+
└─ Use: Route to appropriate processing pipeline

Field Extraction:
├─ Template-free (works on any invoice format)
├─ Vision model: Detect field locations
├─ OCR: Extract field values
├─ LLM: Interpret extracted text (normalize currencies, etc.)
└─ Structured output: JSON with {field: value} pairs

PII Handling:
├─ Detect: Credit card, SSN, account numbers
├─ Mask: Before storing or processing
├─ Compliance: Audit trail of PII access
└─ Retention: Auto-deletion after TTL

Validation & Consistency:
├─ Cross-field validation (total = sum of items)
├─ Date format normalization
├─ Currency conversion if multi-currency
├─ Confidence scores per field
└─ Escalate low-confidence fields to human review
```

---

## Q4: Design a Multi-Modality Recommendation System

**Context**: Build a recommendation engine that considers text, images, and user behavior to recommend products or services.

### Architecture:

```
Input Modalities:

User Profile:
├─ Behavior: Purchase history, clicks, time-on-page
├─ Preferences: Ratings, feedback, wishlist
├─ Metadata: Location, device, customer segment
└─ Embeddings: Behavioral model (MLP)

Product Data:
├─ Text: Title, description, category, reviews
├─ Images: Product photos, lifestyle imagery
├─ Metadata: Price, rating, in-stock status
└─ Embeddings: Multimodal (text + image CLIP)

Recommendation Model:

Collaborative Filtering:
├─ User-product matrix
├─ Similar users/products (cosine similarity)
├─ Matrix factorization

Content-Based:
├─ User embeddings (behavioral) × Product embeddings
├─ Similarity score
├─ Rank by score

Hybrid (Best):
├─ Combine CF + CB with learned weights
├─ Contextual bandits: Which strategy works best?
├─ Online learning: Update as user gives feedback
└─ Real-time personalization per user

Ranking & Personalization:
├─ Candidate generation (1M → 1K products)
├─ Reranking (1K → 100 products)
├─ Personalization (100 → 10 recommendations)
├─ Diversity: Avoid similar items
└─ Cold-start: New users/products

Serving:
├─ Low-latency (user waiting for page load)
├─ Real-time freshness (new products visible)
├─ Online caching (popular recommendations)
└─ A/B testing: Measure impact on CTR, conversion
```

---

## Interview Tips for Top Companies:

1. **Modality-Specific Handling**: Show you understand nuances of each modality
2. **Cost Optimization**: Vision APIs are expensive—discuss batching, caching
3. **Quality Control**: PII handling, validation, human-in-the-loop
4. **Scalability**: How you'd handle 1M images/day ingestion
5. **Real-Time**: Some modalities expensive—discuss when to compute vs. pre-compute

---

## References to Your pp Experience:

- Multimodal RAG with A/B testing and HITL approval workflow
- Automated 600+ hours annually via multimodal RAG system
- Advanced reranking and HITL approval pipelines
- Compliance-focused document processing (PCI-DSS)

