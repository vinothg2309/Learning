# Efficient RAG (Retrieval Augmented Generation) in Azure

## Comprehensive Guide: Document Processing → Vector Storage → Multi-Tenant Retrieval

---

## Scenario Overview

**Challenge**: Build a production-ready RAG system that:
1. ✅ Handles diverse document types (PDF, Excel, Word, PPT) while preserving layout
2. ✅ Chunks intelligently with metadata preservation
3. ✅ Supports multi-tenant access with RBAC
4. ✅ Retrieves with high precision & recall using reranking
5. ✅ Filters by metadata and tenant isolation

---

## Architecture: End-to-End RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EFFICIENT RAG PIPELINE IN AZURE                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PHASE 1: DOCUMENT INGESTION & CHUNKING                                        │
│  ─────────────────────────────────────────                                     │
│  User uploads: PDF, Excel, Word, PPT                                           │
│     ↓                                                                           │
│  Azure Blob Storage (Raw documents)                                            │
│     ↓                                                                           │
│  Azure Document Intelligence (Layout preservation)                             │
│     ↓                                                                           │
│  Intelligent Chunking (with metadata)                                          │
│                                                                                 │
│  PHASE 2: VECTOR STORAGE & INDEXING                                            │
│  ──────────────────────────────────                                            │
│  Chunks with Metadata                                                          │
│     ↓                                                                           │
│  Azure OpenAI (Embeddings API)                                                 │
│     ↓                                                                           │
│  Vector DB (Azure AI Search)                                                   │
│     ↓                                                                           │
│  Indexed vectors + metadata                                                    │
│                                                                                 │
│  PHASE 3: RETRIEVAL WITH FILTERING & RERANKING                                 │
│  ─────────────────────────────────────────────────                            │
│  User Query                                                                     │
│     ↓                                                                           │
│  Tenant Isolation + RBAC Check                                                 │
│     ↓                                                                           │
│  Query Embedding (OpenAI)                                                      │
│     ↓                                                                           │
│  Azure AI Search (Vector similarity search + metadata filtering)                │
│     ↓                                                                           │
│  Candidate documents (top-50)                                                  │
│     ↓                                                                           │
│  Reranker (LLM-based, precision mode)                                          │
│     ↓                                                                           │
│  Final top-K results (high quality)                                            │
│     ↓                                                                           │
│  LLM (GPT-4, Azure OpenAI)                                                     │
│     ↓                                                                           │
│  Generated Response                                                             │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Document Ingestion & Intelligent Chunking

### 1.1 Document Upload Flow

```
User Upload
   ├─ PDF files
   ├─ Excel spreadsheets
   ├─ Word documents
   ├─ PowerPoint presentations
   └─ Other formats
        ↓
   Azure Blob Storage (Container structure)
        ├─ /tenant-1/documents/
        ├─ /tenant-2/documents/
        └─ /tenant-3/documents/
        
        (Organized by tenant for isolation)
```

### 1.2 Azure Document Intelligence: What It Does

**Azure Document Intelligence** is Microsoft's AI service that:

1. **Extracts text with layout preservation** (not just raw text)
2. **Identifies document structure** (headers, tables, paragraphs, form fields)
3. **Preserves spatial relationships** (position of text on page)
4. **Handles multiple document types** (forms, receipts, invoices, PDFs, images)
5. **Outputs structured JSON** with coordinates and semantic meaning

---

### 1.3 Document Intelligence Output Example

**Input**: PDF with table and text

```
┌─────────────────────────────┐
│  Sales Report Q4 2024       │  ← Header
├─────────────────────────────┤
│                             │
│  Sales by Region:           │  ← Paragraph
│                             │
│  Region    Sales    Growth  │  ← Table
│  ─────────────────────────  │
│  North     $500M    +12%    │
│  South     $300M    +8%     │
│  East      $400M    +15%    │
│  West      $350M    -2%     │
│                             │
└─────────────────────────────┘
```

**Document Intelligence Output** (Structured JSON):

```json
{
  "pages": [
    {
      "page_number": 1,
      "content": [
        {
          "type": "heading",
          "text": "Sales Report Q4 2024",
          "bounding_box": {"x": 50, "y": 20, "width": 200, "height": 30},
          "confidence": 0.99
        },
        {
          "type": "paragraph",
          "text": "Sales by Region:",
          "bounding_box": {"x": 50, "y": 80, "width": 150, "height": 20},
          "confidence": 0.98
        },
        {
          "type": "table",
          "rows": [
            {
              "cells": [
                {"text": "Region", "coordinates": [50, 120]},
                {"text": "Sales", "coordinates": [120, 120]},
                {"text": "Growth", "coordinates": [180, 120]}
              ]
            },
            {
              "cells": [
                {"text": "North", "coordinates": [50, 140]},
                {"text": "$500M", "coordinates": [120, 140]},
                {"text": "+12%", "coordinates": [180, 140]}
              ]
            }
            // ... more rows
          ],
          "bounding_box": {"x": 50, "y": 120, "width": 200, "height": 80},
          "confidence": 0.97
        }
      ]
    }
  ],
  "document_type": "invoice/receipt/report", // Detected type
  "language": "en",
  "confidence": 0.95
}
```

**Key Points**:
- ✅ Preserves layout with `bounding_box` coordinates
- ✅ Identifies content type (heading, table, paragraph, form field)
- ✅ Provides confidence scores
- ✅ Maintains semantic structure

---

### 1.4 Intelligent Chunking Strategy

**Problem**: How do we chunk this while preserving layout and meaning?

**Solution**: Hybrid chunking approach

```
CHUNKING STRATEGY:
─────────────────

1. SEMANTIC UNITS FIRST
   └─ Don't split in middle of table rows
   └─ Keep paragraphs + their headers together
   └─ Preserve list items with context

2. SIZE-AWARE CHUNKING
   └─ Small chunks: ~300-500 tokens (good for retrieval)
   └─ Respect document boundaries (no cross-page artificial splits)
   └─ Consider token limits for embedding models

3. METADATA ENRICHMENT
   └─ Add source context: page number, section, doc type
   └─ Add structural info: is_table, is_header, section_name
   └─ Add temporal info: upload_date, document_date
   └─ Add tenant/user info: tenant_id, uploaded_by, RBAC_tags

4. PRESERVE LAYOUT IN CHUNK
   └─ Keep bounding box info
   └─ Maintain table structure (don't flatten into text)
   └─ Store table as structured data (JSON) not text
```

---

### 1.5 Chunking Example

**Original Document** (from above):

```
Sales Report Q4 2024

Sales by Region:

| Region | Sales  | Growth |
|--------|--------|--------|
| North  | $500M  | +12%   |
| South  | $300M  | +8%    |
| East   | $400M  | +15%   |
| West   | $350M  | -2%    |
```

**After Intelligent Chunking**:

```
CHUNK 1:
─────────
Type: Heading + Paragraph
Content: "Sales Report Q4 2024. Sales by Region:"
Metadata: {
  "page": 1,
  "doc_id": "doc-xyz-123",
  "chunk_id": "chunk-001",
  "doc_type": "report",
  "section": "executive_summary",
  "tenant_id": "tenant-1",
  "uploaded_by": "user-42",
  "upload_date": "2024-01-15",
  "rbac_tags": ["team:sales", "dept:business"],
  "type": "heading+paragraph",
  "confidence": 0.98,
  "source_page": 1
}
Bounding Box: {x: 50, y: 20, width: 300, height: 100}

CHUNK 2:
─────────
Type: Table
Content: {
  "headers": ["Region", "Sales", "Growth"],
  "rows": [
    {"Region": "North", "Sales": "$500M", "Growth": "+12%"},
    {"Region": "South", "Sales": "$300M", "Growth": "+8%"},
    {"Region": "East", "Sales": "$400M", "Growth": "+15%"},
    {"Region": "West", "Sales": "$350M", "Growth": "-2%"}
  ]
}
Metadata: {
  "page": 1,
  "doc_id": "doc-xyz-123",
  "chunk_id": "chunk-002",
  "doc_type": "report",
  "section": "sales_data",
  "tenant_id": "tenant-1",
  "uploaded_by": "user-42",
  "upload_date": "2024-01-15",
  "rbac_tags": ["team:sales", "dept:business"],
  "type": "table",
  "confidence": 0.97,
  "source_page": 1,
  "is_structured": true
}
Bounding Box: {x: 50, y: 120, width: 200, height: 80}
```

**Key Benefits**:
- ✅ Chunks respect semantic boundaries
- ✅ Metadata enables filtering (tenant, RBAC, section)
- ✅ Table stored as JSON (searchable structure)
- ✅ Layout preserved for rendering if needed

---

## Phase 2: Vector Storage & Persistence in Vector DB

### 2.1 How Chunks Are Persisted

**Pipeline**:

```
Chunks (from Phase 1)
   ├─ Text content
   └─ Metadata
        ↓
   Generate Embeddings (Azure OpenAI Embedding API)
   └─ text-embedding-3-small or text-embedding-3-large
   └─ Output: 384-3072 dimensional vector
        ↓
   Store in Vector DB (Azure AI Search)
   ├─ Vector index (optimized for similarity search)
   ├─ Metadata fields (filterable, searchable)
   ├─ BM25 full-text index (keyword search)
   └─ Stored documents (for retrieval)
```

### 2.2 Azure AI Search: Vector Storage Structure

**Configuration in Azure AI Search**:

```json
{
  "name": "rag-index",
  "fields": [
    {
      "name": "chunk_id",
      "type": "Edm.String",
      "key": true,
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": true,
      "retrievable": true
    },
    {
      "name": "content_vector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "vector": {
        "dimensions": 1536,
        "similarity_metric": "cosine"
      }
    },
    {
      "name": "doc_id",
      "type": "Edm.String",
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "page_number",
      "type": "Edm.Int32",
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "section",
      "type": "Edm.String",
      "filterable": true,
      "facetable": true,
      "retrievable": true
    },
    {
      "name": "tenant_id",
      "type": "Edm.String",
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "rbac_tags",
      "type": "Collection(Edm.String)",
      "filterable": true,
      "facetable": true,
      "retrievable": true
    },
    {
      "name": "uploaded_by",
      "type": "Edm.String",
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "upload_date",
      "type": "Edm.DateTimeOffset",
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "doc_type",
      "type": "Edm.String",
      "filterable": true,
      "facetable": true,
      "retrievable": true
    },
    {
      "name": "is_structured",
      "type": "Edm.Boolean",
      "filterable": true,
      "retrievable": true
    },
    {
      "name": "confidence",
      "type": "Edm.Double",
      "filterable": true,
      "retrievable": true,
      "sortable": true
    }
  ],
  "vectorSearch": {
    "algorithms": [
      {
        "name": "hnsw-config",
        "kind": "hnsw",
        "parameters": {
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        }
      }
    ],
    "profiles": [
      {
        "name": "vector-profile",
        "algorithm": "hnsw-config"
      }
    ]
  }
}
```

### 2.3 Data Persistence Example

**Chunk stored in Azure AI Search**:

```json
{
  "chunk_id": "doc-xyz-123-chunk-002",
  "content": "Region | Sales | Growth\nNorth | $500M | +12%\nSouth | $300M | +8%\nEast | $400M | +15%\nWest | $350M | -2%",
  "content_vector": [0.041, -0.032, 0.118, ..., 0.204],  // 1536 dimensions
  "doc_id": "doc-xyz-123",
  "page_number": 1,
  "section": "sales_data",
  "tenant_id": "tenant-1",
  "rbac_tags": ["team:sales", "dept:business", "role:analyst"],
  "uploaded_by": "user-42",
  "upload_date": "2024-01-15T10:30:00Z",
  "doc_type": "report",
  "is_structured": true,
  "confidence": 0.97,
  "bounding_box": "{'x': 50, 'y': 120, 'width': 200, 'height': 80}"
}
```

**Storage Benefits**:
- ✅ Vector field optimized for similarity search (HNSW algorithm)
- ✅ Metadata fields enable multi-dimensional filtering
- ✅ Full-text index (BM25) for keyword search
- ✅ Tenant isolation via `tenant_id` filter
- ✅ RBAC via `rbac_tags` filter

---

## Phase 3: Retrieval with Multi-Tenant, RBAC, and Reranking

### 3.1 Retrieval Pipeline

```
USER QUERY: "Show me regional sales data for Q4"
└─ Current User: user-42
└─ Tenant: tenant-1
└─ User Roles: ["team:sales", "role:analyst"]

        ↓

STEP 1: TENANT ISOLATION
├─ Filter: tenant_id == "tenant-1"
└─ Prevents cross-tenant data leakage

        ↓

STEP 2: RBAC VALIDATION
├─ Check: user-42's RBAC tags include any rbac_tags in document
├─ Filter: rbac_tags intersects ["team:sales", "role:analyst"]
└─ Only documents this user can access

        ↓

STEP 3: GENERATE QUERY EMBEDDING
├─ Query text: "Show me regional sales data for Q4"
├─ API: Azure OpenAI Embedding (text-embedding-3-small)
└─ Output: [0.051, -0.041, 0.129, ..., 0.215] (1536 dimensions)

        ↓

STEP 4: VECTOR SEARCH (Azure AI Search)
├─ Find top-50 similar vectors (cosine similarity)
├─ Apply filters:
│  ├─ tenant_id == "tenant-1"
│  ├─ rbac_tags intersects user's roles
│  └─ doc_type in ["report", "summary"]
└─ Return: Candidate documents with scores

        ↓

STEP 5: METADATA FILTERING
├─ Filter by upload_date (recent documents preferred)
├─ Filter by confidence (high-confidence extractions only)
├─ Filter by section (prefer "sales_data" section)
└─ Return: Refined candidates

        ↓

STEP 6: RERANKING (LLM-based, high precision)
├─ Take top-50 candidates
├─ Query: "Which documents best answer: Show me regional sales data?"
├─ LLM scores each for relevance (0.0-1.0)
└─ Return: Top-5 reranked results

        ↓

STEP 7: FINAL RESULTS → LLM
├─ Top-5 chunks (high quality)
├─ Inject into prompt: "Based on these documents..."
└─ Generate response
```

### 3.2 Detailed Retrieval Code Example

**Retrieval Query Structure**:

```python
# PSEUDO CODE for Azure AI Search retrieval
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential

search_client = SearchClient(
    endpoint="https://myai-search.search.windows.net",
    index_name="rag-index",
    credential=DefaultAzureCredential()
)

user_query = "Show me regional sales data for Q4"
user_id = "user-42"
tenant_id = "tenant-1"
user_roles = ["team:sales", "role:analyst"]

# Step 1: Generate query embedding
query_embedding = openai_client.embeddings.create(
    input=user_query,
    model="text-embedding-3-small"
).data[0].embedding

# Step 2: Build filter string (Tenant + RBAC)
filter_string = f"""
tenant_id eq '{tenant_id}' 
and 
(rbac_tags/any(t: t eq 'team:sales') or rbac_tags/any(t: t eq 'role:analyst'))
"""

# Step 3: Vector search with filters
results = search_client.search(
    vector=query_embedding,
    k=50,  # Get top-50 candidates
    filter=filter_string,
    select=["chunk_id", "content", "doc_id", "page_number", "section", "rbac_tags"],
    query_type="semantic"  # Use semantic search
)

candidates = list(results)
print(f"Found {len(candidates)} candidates after filtering")

# Step 4: Reranking with LLM
reranked = rerank_with_llm(candidates, user_query)

# Step 5: Return top-5
final_docs = reranked[:5]

# Step 6: Inject into LLM prompt
context = "\n".join([doc["content"] for doc in final_docs])
prompt = f"""
Based on these documents:
{context}

Answer: {user_query}
"""

response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

### 3.3 Multi-Tenant Isolation

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-TENANT ISOLATION (Security)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SCENARIO: user-42 (tenant-1) searches                          │
│                                                                 │
│  ❌ BEFORE FILTERING:                                           │
│  └─ Documents from tenant-1: 50 results                         │
│  └─ Documents from tenant-2: 30 results (VISIBLE!)             │
│  └─ Documents from tenant-3: 20 results (VISIBLE!)             │
│  └─ SECURITY BREACH! Cross-tenant data leak                    │
│                                                                 │
│  ✅ AFTER FILTERING:                                            │
│  └─ Filter: tenant_id eq 'tenant-1'                            │
│  └─ Documents from tenant-1: 50 results                         │
│  └─ Documents from tenant-2: BLOCKED                           │
│  └─ Documents from tenant-3: BLOCKED                           │
│  └─ SECURE! Only tenant-1 data visible                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 RBAC-Based Access Control

```
┌─────────────────────────────────────────────────────────────────┐
│                  RBAC FILTERING (Access Control)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  USER: user-42                                                  │
│  Roles: ["team:sales", "role:analyst"]                         │
│                                                                 │
│  DOCUMENT 1: Sales Report Q4                                   │
│  rbac_tags: ["team:sales", "dept:business"]                   │
│  Result: ✅ ACCESSIBLE (user has "team:sales")                 │
│                                                                 │
│  DOCUMENT 2: Executive Summary                                 │
│  rbac_tags: ["role:director", "team:leadership"]              │
│  Result: ❌ BLOCKED (user doesn't have director role)          │
│                                                                 │
│  DOCUMENT 3: Employee Data                                     │
│  rbac_tags: ["role:hr", "dept:human_resources"]               │
│  Result: ❌ BLOCKED (user doesn't have HR role)                │
│                                                                 │
│  Filter Logic:                                                 │
│  └─ User can access if: user_roles ∩ doc.rbac_tags ≠ ∅       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Metadata Filtering Examples

```
EXAMPLE 1: Filter by Document Type
─────────────────────────────────
Query: "Show me reports from sales team"
Filter: doc_type eq 'report' and rbac_tags/any(t: t eq 'team:sales')
Result: Only sales reports, no PDFs or images

EXAMPLE 2: Filter by Date Range
─────────────────────────────────
Query: "Recent Q4 data"
Filter: upload_date ge 2024-10-01 and upload_date le 2024-12-31
Result: Only documents uploaded in Q4

EXAMPLE 3: Filter by Confidence Score
──────────────────────────────────────
Query: "Get high-quality extracted data"
Filter: confidence ge 0.95 and is_structured eq true
Result: Only high-confidence, structured data

EXAMPLE 4: Combined Multi-Filter
────────────────────────────────
Filter: 
  tenant_id eq 'tenant-1' 
  and (rbac_tags/any(t: t eq 'team:sales') or rbac_tags/any(t: t eq 'role:analyst'))
  and doc_type eq 'report'
  and upload_date ge 2024-10-01
  and confidence ge 0.95
Result: Highly specific, secure results
```

### 3.6 Reranking for High Precision & Recall

```
┌─────────────────────────────────────────────────────────────────┐
│              RERANKING PIPELINE (Precision & Recall)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Top-50 candidates from vector search                   │
│                                                                 │
│  RERANKING STAGE 1: Fast Filter                               │
│  ├─ Remove duplicates                                          │
│  ├─ Remove very short chunks                                   │
│  ├─ Remove low confidence                                      │
│  └─ Output: ~40 candidates                                     │
│                                                                 │
│  RERANKING STAGE 2: LLM Scoring                               │
│  ├─ Query: "How relevant is this to: {user_query}"            │
│  ├─ Model: gpt-4 (high quality)                                │
│  ├─ Output: Relevance score 0.0-1.0 for each chunk            │
│  ├─ Cost: Moderate (LLM calls × 40)                            │
│  └─ Quality: Very High (semantic understanding)                │
│                                                                 │
│  RERANKING STAGE 3: Sort & Select                             │
│  ├─ Sort by LLM relevance score (descending)                   │
│  ├─ Pick top-5 highest scoring documents                       │
│  ├─ Output: Final 5 chunks (high precision)                    │
│  └─ Metrics: High precision (right answer), High recall (found │
│                                                                 │
│  PRECISION vs RECALL:                                          │
│  ├─ Precision: Of top-5 results, how many are relevant?       │
│  │  Goal: 90%+ (only 1 or fewer irrelevant)                   │
│  ├─ Recall: Of all relevant documents, how many are in top-5? │
│  │  Goal: 80%+ (found most relevant docs)                     │
│  └─ Trade-off: Reranking balances both                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Reranking Example**:

```
BEFORE RERANKING:
─────────────────
1. "Sales in North region increased 12% YoY"       (similarity: 0.85)
2. "Northern hemisphere weather patterns"           (similarity: 0.84)
3. "The northern lights are visible in winter"      (similarity: 0.83)
4. "East region sales grew 15%"                     (similarity: 0.82)
5. "West region declined 2%"                        (similarity: 0.81)

AFTER LLM RERANKING (for "Show me regional sales for Q4"):
────────────────────────────────────────────────────────
1. "East region sales grew 15%"                     (LLM score: 0.98)  ✅
2. "Sales in North region increased 12% YoY"       (LLM score: 0.96)  ✅
3. "West region declined 2%"                        (LLM score: 0.94)  ✅
4. "Northern hemisphere weather patterns"           (LLM score: 0.15)  ❌
5. "The northern lights are visible in winter"      (LLM score: 0.08)  ❌

Result: Top-3 are HIGHLY RELEVANT. Precision improved from 60% → 100%
```

---

## Efficiency Metrics & Performance

### 4.1 End-to-End Latency

```
LATENCY BREAKDOWN:
──────────────────

User sends query
        ↓ (10ms)
Query embedding (OpenAI API)
        ↓ (500ms)
Vector search (Azure AI Search) with filters
        ↓ (200ms)
Top-50 candidates returned
        ↓ (1000ms - reranking with LLM)
Rerank: LLM scores top-50
        ↓ (10ms)
Select top-5
        ↓ (3000ms - LLM generates response)
Generate response (GPT-4)
        ↓ (50ms)
Return to user

TOTAL: ~4.77 seconds (optimized)

BREAKDOWN:
├─ Embedding: 500ms (non-negotiable)
├─ Vector search: 200ms (very fast with HNSW)
├─ Reranking: 1000ms (can batch)
├─ LLM generation: 3000ms (depends on response length)
└─ Overhead: 70ms

OPTIMIZATION:
├─ Cache embeddings for repeated queries
├─ Batch reranking (score 50 documents at once vs sequential)
├─ Use smaller embedding model (text-embedding-3-small)
├─ Parallel: While LLM generates, fetch next batch
```

### 4.2 Precision & Recall Metrics

```
MEASUREMENT:
─────────────

Precision = (Relevant results in top-5) / 5
Recall = (Relevant results in top-5) / (Total relevant documents)

BEFORE RERANKING:
─────────────────
Vector search without reranking
├─ Precision: 65% (3-4 out of 5 relevant)
├─ Recall: 45% (found 45% of all relevant docs)
└─ Issue: Keyword similarity doesn't equal semantic meaning

AFTER RERANKING:
─────────────────
Vector search + LLM reranking
├─ Precision: 92% (4.6 out of 5 relevant on average)
├─ Recall: 78% (found 78% of all relevant docs)
└─ Improvement: +27pp precision, +33pp recall

COST TRADE-OFF:
─────────────────
├─ Reranking adds 1 second latency
├─ Reranking costs ~$0.002 per query (LLM calls)
├─ But improves answer quality significantly
└─ ROI: Worth it for production systems
```

### 4.3 Scalability & Cost

```
STORAGE:
─────────
1M documents × 200 chunks/doc = 200M chunks
200M chunks × 1536 dimensions = 307B floats (~1.2TB vectors)
+ Metadata: ~500GB
Total: ~1.7TB (manageable with Azure AI Search)

Cost: $1000-2000/month (depends on tier)

QUERY THROUGHPUT:
──────────────────
Azure AI Search can handle:
├─ 100+ QPS (queries per second) for vector search
├─ Bottleneck: Reranking LLM (limited by token rate)
├─ Solution: Queue reranking requests, batch process

COST PER QUERY:
────────────────
├─ Embedding: $0.00001 (very cheap)
├─ Vector search: $0 (included in tier)
├─ Reranking: $0.002 per query (LLM calls)
├─ LLM generation: $0.05 per query (GPT-4)
├─ Total: ~$0.052 per query
└─ For 1M queries/month: ~$52,000/month
```

---

## Best Practices Summary

### 5.1 Chunking Strategy

✅ **DO**:
- Preserve document structure (don't flatten tables)
- Respect semantic boundaries (full paragraphs, complete tables)
- Add rich metadata (tenant, RBAC, section, confidence)
- Keep chunks 300-500 tokens
- Store structured data (tables, lists) as JSON

❌ **DON'T**:
- Split in middle of sentences
- Lose layout information
- Chunk by fixed size without semantic awareness
- Store all metadata in content (use separate fields)
- Lose confidence scores from Document Intelligence

### 5.2 Vector DB Schema

✅ **DO**:
- Make `tenant_id` filterable (mandatory for multi-tenant)
- Make `rbac_tags` filterable (mandatory for access control)
- Create composite filters (tenant + RBAC + metadata)
- Store confidence scores (enable quality filtering)
- Use HNSW algorithm (faster than alternatives)

❌ **DON'T**:
- Forget tenant isolation (security risk!)
- Skip RBAC fields (compliance issue!)
- Store sensitive data in vector embeddings
- Use exact vector matching (use similarity search)

### 5.3 Retrieval Pipeline

✅ **DO**:
- Always filter by tenant_id first
- Always check RBAC before returning results
- Use reranking for precision-critical queries
- Cache embeddings for repeated queries
- Monitor precision & recall metrics

❌ **DON'T**:
- Return vector search results without RBAC check
- Skip reranking (precision will suffer)
- Allow cross-tenant result leakage
- Cache without user context
- Ignore confidence thresholds

---

## Integration with Azure Services

```
┌───────────────────────────────────────────────────────────────────┐
│          COMPLETE AZURE RAG ARCHITECTURE                          │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INGESTION:                                                       │
│  ├─ User uploads document                                        │
│  ├─ Azure Blob Storage (stores raw document)                     │
│  └─ Event Grid (triggers)                                        │
│                                                                   │
│  PROCESSING:                                                      │
│  ├─ Azure Function (triggered by Event Grid)                     │
│  ├─ Calls Document Intelligence API                              │
│  ├─ Intelligent chunking logic (your code)                       │
│  └─ Generates embeddings (Azure OpenAI)                          │
│                                                                   │
│  STORAGE:                                                         │
│  ├─ Azure AI Search (vector DB + metadata)                       │
│  ├─ Azure Cosmos DB (document metadata backup)                   │
│  └─ Azure SQL (RBAC and tenant mappings)                         │
│                                                                   │
│  RETRIEVAL:                                                       │
│  ├─ User query via API                                           │
│  ├─ Azure OpenAI Embedding (query)                               │
│  ├─ Azure AI Search (vector search + filters)                    │
│  ├─ LLM Reranking (Azure OpenAI)                                 │
│  ├─ Azure OpenAI (generate response)                             │
│  └─ Return to user                                               │
│                                                                   │
│  SECURITY:                                                        │
│  ├─ Managed Identity (no secrets in code)                        │
│  ├─ Azure RBAC (access control)                                  │
│  ├─ Key Vault (API keys)                                         │
│  └─ Private endpoints (network isolation)                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Summary: Why This Approach Is Efficient

| Aspect | Why It Works |
|--------|-------------|
| **Layout Preservation** | Document Intelligence maintains structure + coordinates → tables queryable, not flattened |
| **Metadata** | Rich metadata enables multi-dimensional filtering (tenant, RBAC, section, date, confidence) |
| **Multi-Tenant Isolation** | `tenant_id` filter prevents cross-tenant data leaks at search layer |
| **RBAC** | `rbac_tags` enable user-specific access control without per-query authorization server |
| **Vector Search** | HNSW algorithm finds semantically relevant documents in milliseconds |
| **Reranking** | LLM scores improve precision (90%+) and recall (80%+) vs pure vector similarity |
| **Metadata Filtering** | Confidence + date filters eliminate low-quality or outdated documents early |
| **Scalability** | Azure AI Search handles millions of chunks; can filter before expensive reranking |
| **Cost-Effective** | Most queries stop after cheap vector search; only rerank when needed |

---

## Next Steps

1. **Build**: Implement chunking logic that preserves layout
2. **Test**: Measure precision & recall on your domain-specific queries
3. **Deploy**: Set up Azure AI Search with proper tenant/RBAC schema
4. **Monitor**: Track latency, cost, and accuracy metrics
5. **Optimize**: Cache, batch reranking, tune rerank thresholds

