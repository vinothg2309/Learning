## RAG, NVIDIA NeMo, Fine-Tuning, Quantization, PEFT

---

# 1. Retrieval-Augmented Generation (RAG)

## 📚 Conceptual Questions (20)

### Q1. What problem does Retrieval-Augmented Generation (RAG) solve compared to vanilla LLMs?

**Answer:**
- **Core benefit**: Consults external knowledge store at query time vs. relying only on trained parameters
- **Problems solved**:
  - ✅ Reduces hallucinations by grounding answers in retrieved documents
  - ✅ Overcomes outdated knowledge (cutoff date limitation)
  - ✅ Enables domain-specific accuracy using private/proprietary data
  - ✅ Provides source attribution for verifiable answers
  - ✅ Cost-efficient - updates knowledge without retraining entire model

---

### Q2. Describe the high-level architecture of a typical RAG system.

**Answer:**
**Core components**:
1. **Retriever**: Finds relevant documents from corpus
   - Vector search (dense embeddings) or BM25 (sparse/keyword)
2. **Reader/Generator (LLM)**: Consumes retrieved docs + query → generates answer
3. **Controller/Orchestrator**: Coordinates retrieval → ranking → prompt construction

**Additional components**:
- Vector database (Pinecone, Weaviate, Chroma) - stores embeddings
- Reranker (cross-encoder) - improves relevance of top results
- Metadata filters - enables targeted retrieval (by date, source, type)

---

### Q3. How does dense retrieval differ from sparse retrieval in the context of RAG?

**Answer:**

| Aspect | Sparse Retrieval | Dense Retrieval |
|--------|-----------------|-----------------|
| **Method** | Keyword statistics (BM25, TF-IDF) | Neural embeddings (768-1536 dims) |
| **Matching** | Exact words/stems | Semantic similarity (cosine) |
| **Strengths** | Fast, rare terms, proper nouns | Synonyms, paraphrases, context |
| **Example** | "car" ≠ "vehicle" | "car" ≈ "vehicle" |

**Best practice**: Hybrid (combine both) - Dense for semantic, Sparse for keywords

---

### Q4. Why is chunking necessary when building a document store for RAG?

**Answer:**
**Why chunk?** Splits long documents into smaller segments

**Key reasons**:
1. **Embedding model limits**: Most models max at 512-8192 tokens
2. **Retrieval precision**: Smaller chunks match specific concepts better
   - Example: Find paragraph about "refund policy" vs. entire 100-page manual
3. **LLM context efficiency**: Reduces irrelevant text in prompt
   - Sending full document wastes context window

**Without chunking**: 100-page PDF = 1 unit → impossible to find specific info

---

### Q5. What trade-offs are involved in choosing chunk size for RAG?

**Answer:**

| Chunk Size | Pros | Cons |
|------------|------|------|
| **Small (128-256 tokens)** | ✅ Precise retrieval | ❌ Lost context, incomplete info |
| **Large (512-1024 tokens)** | ✅ Preserve context | ❌ Noisy retrieval, waste tokens |

**Optimal**: 200-500 tokens with 10-20% overlap

**Example scenario** - "What's the refund policy?":
- 100-token chunk: Returns only policy statement (incomplete)
- 500-token chunk: Includes eligibility + process steps (complete)

**Best solution**: Parent-child chunking (index small, retrieve large parent)

---

### Q6. What is hybrid retrieval and why is it often better than using only dense or only sparse retrieval?

**Answer:**
**What it is**: Combines BM25 (sparse) + Dense embeddings in one retrieval pipeline

**Why better**:
- BM25: Strong at exact keywords, rare terms, proper nouns (e.g., "iPhone 15")
- Dense: Strong at semantic meaning, paraphrases (e.g., "phone" → "mobile device")
- **Together**: Higher recall + robustness across varied query types

**Example**:
- Query: "Steve Jobs announcement 2007"
  - BM25: Matches exact name + year
  - Dense: Understands "announcement" context
  - Hybrid: Retrieves both keyword-heavy docs + semantically related content

---

### Q7. What is a reranker in RAG, and how does it differ from the initial retriever?

**Answer:**

| Component | Retriever | Reranker |
|-----------|-----------|----------|
| **Purpose** | Fast candidate selection (top 100-1000) | Precise re-ordering (top 10-20) |
| **Method** | Approximate similarity (vector search) | Cross-encoder or LLM scoring |
| **Speed** | Very fast (milliseconds) | Slower (reads query + doc together) |
| **Accuracy** | Good recall | Better precision |

**Why both?**
- Retriever: Narrows 1M docs → 100 candidates (fast)
- Reranker: Re-orders 100 → top 10 (accurate)

**Example**: Query "best Italian restaurant NYC"
- Retriever: Returns 100 restaurant docs (some irrelevant)
- Reranker: Re-scores based on query-doc match → top 10 most relevant

---

### Q8. Explain the "retrieve-then-read" pipeline in RAG.

**Answer:**
**Two-stage pipeline**:

1. **Retrieve** (Information Gathering):
   - Input: User query
   - Process: Vector search → top-K chunks (e.g., K=5)
   - Output: Most relevant document chunks

2. **Read** (Answer Generation):
   - Input: Query + retrieved chunks
   - Process: Construct prompt → LLM generates answer
   - Output: Grounded answer citing retrieved context

**Example flow**:
```
Query: "What is the refund policy?"
↓
Retrieve: [Chunk 1: "Refunds within 30 days...", Chunk 2: "Original receipt required..."]
↓
Prompt: "Context: [chunks]. Question: What is the refund policy?"
↓
LLM: "Based on the policy, refunds are accepted within 30 days with original receipt."
```

---

### Q9. What is conversational RAG and what extra challenge does it introduce?

**Answer:**
**What it is**: RAG extended to multi-turn dialogue (chat interface)

**Extra challenges**:
1. **Context dependency**: Follow-up questions reference previous turns
   - User: "Who is the CEO?"
   - System: "John Smith"
   - User: "What is his background?" ← Needs "John Smith" context

2. **Solution - Query Rewriting**:
   - Original: "What is his background?"
   - Rewritten: "What is John Smith's background?" ← Self-contained

**Implementation**:
- Maintain conversation history
- Use LLM to rewrite ambiguous queries
- Retrieve using rewritten query (not original)

---

### Q10. How does query rewriting help RAG quality?

**Answer:**
**Purpose**: Reformulates ambiguous/incomplete queries → standalone, explicit queries

**Techniques**:
1. **Resolve pronouns**: "his background" → "John Smith's background"
2. **Expand abbreviations**: "ML model" → "machine learning model"
3. **Add context**: "refund process" → "product refund process for electronics"
4. **Decompose complex**: "Compare X and Y pricing" → Two queries for X and Y

**Example**:
| Original Query | Problem | Rewritten Query |
|----------------|---------|-----------------|
| "How much does it cost?" | Missing subject | "How much does the Pro plan cost?" |
| "What about pricing?" | Vague | "What is the pricing for enterprise tier?" |

**Result**: Better retrieval accuracy (20-40% improvement in relevance)

---

### Q11. What is semantic chunking and how is it implemented?

**Answer:**
**What it is**: Splits text at topic/semantic boundaries (not fixed lengths)

**How it works**:
1. Embed each sentence
2. Calculate similarity between adjacent sentences
3. Split when similarity drops (topic change detected)

**Example**:
```
Sent1: "Tesla released new EV" ──0.95→ Sent2: "EV sales increased" (same topic, no split)
Sent2: "EV sales increased" ──0.32→ Sent3: "Apple announced iPhone" (topic change, SPLIT!)
```

**Benefits**:
- ✅ Chunks align with natural topics
- ✅ Better coherence than fixed-size
- ❌ Slower (requires embedding each sentence)

---

### Q12. Why is it important to control for context window limits in RAG?

**Answer:**
**Problem**: LLMs have finite context windows (4K-128K tokens)

**Risks of exceeding limits**:
- ❌ Truncation: Important info cut off
- ❌ Dilution: Relevant info lost in noise
- ❌ Higher cost: More tokens = higher API costs
- ❌ Slower response: Processing time increases

**Solutions**:
1. **Retrieve top-K only** (e.g., top 5 chunks, not 50)
2. **Compression**: Summarize retrieved chunks before sending to LLM
3. **Token budgeting**: Reserve tokens for query (500) + chunks (3000) + answer (500)

**Example**: GPT-4 (8K context) - Don't send 10 chunks of 1K tokens each!

---

### Q13. Define "groundedness" in RAG and how it differs from generic answer quality.

**Answer:**
**Groundedness**: Answer supported by retrieved documents (factual basis)

**Comparison**:
| Metric | Groundedness | Answer Quality |
|--------|--------------|----------------|
| **Measures** | Supported by context? | Fluent? Relevant? Complete? |
| **Focus** | Factual accuracy | User satisfaction |

**Example**:
```
Context: "Tesla was founded in 2003"
Question: "When was Tesla founded?"

Answer 1: "Tesla was founded in 2003" ✅ Grounded ✅ High quality
Answer 2: "Tesla is a leading EV company" ❌ Not grounded ⚠️ Fluent but irrelevant
Answer 3: "Tesla was founded in 2000" ❌ Not grounded ❌ Hallucinated
```

**Why it matters**: Fluent hallucinations are dangerous - look correct but are wrong!

---

### Q14. What is the role of a vector database in a RAG system?

**Answer:**
**Purpose**: Stores + searches embeddings at scale

**Key capabilities**:
1. **Storage**: Millions/billions of embeddings (768-1536 dims each)
2. **Fast search**: ANN (Approximate Nearest Neighbor) - millisecond retrieval
3. **Similarity search**: Returns top-K most similar vectors to query
4. **Metadata filtering**: Filter by date, category, source before similarity search

**Popular options**:
- **Pinecone**: Managed, serverless, easy to use
- **Weaviate**: Open-source, supports hybrid search
- **Chroma**: Lightweight, good for local dev
- **Milvus**: High-performance, self-hosted

**Example**: Query "machine learning" → Find docs with closest embeddings from 10M documents in <100ms

---

### Q15. How do you use metadata in retrieval to improve RAG results?

**Answer:**
**What is metadata**: Additional fields stored with each chunk

**Common metadata fields**:
- `source`: File name, URL
- `date`: Creation/modified date
- `type`: PDF, webpage, email
- `category`: Product docs, legal, HR
- `language`: en, es, fr
- `user_id`: For multi-tenant systems

**Use cases**:
1. **Pre-filtering**: "Only search documents from 2023"
2. **Boosting**: "Prioritize official docs over user comments"
3. **Access control**: "Show only docs user has permission for"
4. **Multi-tenancy**: "Retrieve only customer X's data"

**Example**:
```python
# Without metadata: Returns all docs about "pricing"
retriever.search("pricing")

# With metadata: Returns only 2024 pricing docs
retriever.search("pricing", filters={"date": "2024", "type": "official"})

---

### Q16. What is "parent–child" retrieval and why is it useful?

**Answer:**
**Strategy**: Index small chunks, retrieve large parent contexts

**How it works**:
1. **Child chunks** (small, 128 tokens): Indexed for precise retrieval
2. **Parent chunks** (large, 512 tokens): Sent to LLM with full context
3. **Mapping**: Each child → 1 parent

**Why useful**:
- ✅ Precise retrieval (small chunks match specific concepts)
- ✅ Rich context (LLM gets full paragraph/section)
- ✅ Best of both worlds

**Example**:
```
Parent: "Our refund policy allows returns within 30 days with receipt.
         Refunds processed in 5-7 business days to original payment."

Child 1: "refund policy allows returns within 30 days with receipt"
Child 2: "Refunds processed in 5-7 business days"

Query: "refund timeline"
→ Matches Child 2
→ Retrieves entire Parent (gives full policy + timeline)

---

### Q17. Why might you use multiple retrievers or indexes in a single RAG system?

**Answer:**
**Reason**: Different content types need different retrieval strategies

**Use cases**:
1. **Multiple data types**:
   - Index 1: Code files (optimized for exact matching)
   - Index 2: Documentation (semantic search)
   - Index 3: FAQs (keyword matching)

2. **Domain separation**:
   - Index 1: Legal documents
   - Index 2: Product manuals
   - Index 3: Customer support tickets

**Strategies**:
- **Routing**: Classify query → route to appropriate index
- **Fusion**: Search all indexes → merge + rerank results

**Example**:
```
Query: "How to authenticate API?"
→ Route to Code index (likely code snippet needed)

Query: "What is our privacy policy?"
→ Route to Legal index
```

---

### Q18. What is "agentic RAG" as opposed to single-step RAG?

**Answer:**

| Type | Single-Step RAG | Agentic RAG |
|------|-----------------|-------------|
| **Flow** | Query → Retrieve once → Answer | Query → Plan → Multiple retrievals → Synthesize |
| **Queries** | 1 retrieval call | Multiple iterative calls |
| **Complexity** | Simple questions | Multi-hop reasoning |

**Agentic RAG capabilities**:
1. **Query decomposition**: Break complex query into sub-questions
2. **Iterative retrieval**: Retrieve → Analyze → Retrieve more if needed
3. **Self-correction**: Detect gaps and fetch missing info
4. **Tool use**: Can call different retrievers, APIs, databases

**Example**:
```
Question: "Compare Tesla's 2022 revenue to Ford's and explain the difference"

Single-step: Retrieve "Tesla Ford revenue 2022" → May miss context

Agentic:
  Step 1: "What was Tesla's 2022 revenue?" → Retrieve → $81.5B
  Step 2: "What was Ford's 2022 revenue?" → Retrieve → $158B
  Step 3: "Why is Ford higher?" → Retrieve industry analysis
  Step 4: Synthesize complete comparison
```

---

### Q19. How would you evaluate the retrieval component of a RAG pipeline separately from generation?

**Answer:**
**Goal**: Test if retriever finds relevant docs (before LLM even sees them)

**Setup**:
1. Create test set: Query + labeled relevant documents
2. Run retriever only (no LLM)
3. Check if relevant docs in top-K results

**Key metrics**:

| Metric | What it measures | Formula |
|--------|------------------|---------|
| **Recall@K** | % of relevant docs in top-K | Relevant found / Total relevant |
| **Precision@K** | % of top-K that are relevant | Relevant in top-K / K |
| **MRR** | How high first relevant doc ranks | 1 / rank of 1st relevant |
| **nDCG@K** | Quality of ranking order | Weighted by position |

**Example**:
```
Query: "refund policy"
Relevant docs: [doc_5, doc_12, doc_23]
Retrieved top-10: [doc_2, doc_5, doc_7, doc_12, ...]

Recall@10 = 2/3 = 0.67  (found doc_5 and doc_12, missed doc_23)
Precision@10 = 2/10 = 0.20
```

---

### Q20. Why might RAG still hallucinate even if retrieval is working well?

**Answer:**
**Even with perfect retrieval, LLM can hallucinate**

**Common causes**:
1. **Ignoring context**: LLM relies on training data instead of retrieved docs
2. **Overgeneralization**: Extrapolates beyond what's in context
3. **Gap filling**: Creates plausible-sounding info for missing details
4. **Conflicting info**: Chooses wrong source when docs contradict
5. **Prompt issues**: Instructions don't emphasize "use only provided context"

**Solutions**:

| Problem | Solution |
|---------|----------|
| Ignores context | Add "Answer ONLY from context. If not found, say 'I don't know'" |
| Overgeneralizes | Require citations: "Quote the source for each claim" |
| Fills gaps | Use stricter system prompts limiting creativity |
| Conflicting info | Instruct to note contradictions instead of choosing |

**Example prompt fix**:
```
❌ "Answer the question using the context"
✅ "Answer ONLY using the context below. Quote sources. If info not in context, respond 'Not found in documents'"
```

---

## 📋 Scenario-Based Questions (10)

### Q1. Your RAG system often returns relevant documents, but answers still contain unsupported claims. How would you systematically debug and reduce hallucinations?

**Answer:**
**Diagnosis**:
1. Check if retrieved docs actually included in prompt
2. Test if LLM ignores context (send dummy context)
3. Review prompt instructions

**Solutions**:
- ✅ **Improve prompts**: "Answer ONLY from context. Quote sources. If not found, say 'I don't know'"
- ✅ **Reduce noise**: Limit to top 3-5 chunks (not 10+)
- ✅ **Require citations**: Force model to quote specific snippets
- ✅ **Post-verification**: Add validation step checking claims vs. retrieved docs
- ✅ **Better reranking**: Filter low-relevance chunks before LLM

---

### Q2. Users complain that your RAG chatbot over an internal wiki frequently responds with "I don't know" even when answers exist. What could be going wrong and how would you address it?

**Answer:**
**Root causes** (Low recall):
- ❌ Poor chunking (relevant info split across chunks)
- ❌ Incorrect embeddings (semantic mismatch)
- ❌ Missing metadata filters
- ❌ Too small K (retrieving top-3 when answer in position 10)
- ❌ Corpus not fully indexed

**Solutions**:
1. **Verify indexing**: Check all wiki pages ingested
2. **Experiment with chunking**: Try 200, 400, 800 token sizes
3. **Test embedding models**: Try different models (OpenAI, Cohere, BGE)
4. **Increase K**: Retrieve top-10 or top-20 instead of top-5
5. **Add hybrid search**: Combine BM25 + dense for better coverage
6. **Query expansion**: Rewrite query with synonyms/variations

### Q3. You notice that for short keyword queries, BM25 outperforms dense retrieval, but for long natural language questions, dense retrieval does better. How would you design a retriever strategy?

**Answer:**
**Observation**:
- Short queries ("iPhone 15") → BM25 wins (exact keyword match)
- Long queries ("What are the benefits of...") → Dense wins (semantic understanding)

**Strategy - Adaptive Hybrid**:

| Query Type | Detection | Retriever Strategy |
|------------|-----------|-------------------|
| **Short (< 5 words)** | Word count, no question words | BM25 weight: 0.7, Dense: 0.3 |
| **Long (> 5 words)** | Natural language, question words | BM25 weight: 0.3, Dense: 0.7 |

**Implementation**:
```python
def adaptive_retrieval(query):
    word_count = len(query.split())
    is_question = query.startswith(("what", "how", "why", "when"))

    if word_count <= 5 and not is_question:
        # Short keyword query
        return weighted_fusion(bm25_results, dense_results, alpha=0.7)
    else:
        # Long natural language query
        return weighted_fusion(bm25_results, dense_results, alpha=0.3)
```

---

### Q4. A legal RAG system is mixing documents from different jurisdictions leading to wrong advice. How would you redesign the retrieval pipeline?

**Answer:**
**Problem**: Retrieving California law when user needs New York law

**Solution - Mandatory Metadata Filtering**:

1. **Extract & store metadata**:
```python
document_metadata = {
    "jurisdiction": "New York",
    "court_level": "Supreme Court",
    "year": "2023",
    "case_type": "Criminal"
}
```

2. **Enforce filters at retrieval**:
```python
retriever.search(
    query="sentencing guidelines",
    filters={
        "jurisdiction": user_jurisdiction,  # REQUIRED
        "court_level": applicable_level
    }
)
```

3. **Architecture changes**:
- ✅ Separate indexes per jurisdiction (best isolation)
- ✅ Tag extraction during ingestion (automated metadata)
- ✅ User context: Capture jurisdiction from user profile/session
- ✅ Validation: Pre-retrieval check that required filters present

### Q5. Your RAG over PDF documents gives incomplete answers because important information is in tables and diagrams. What changes would you make?

**Answer:**
**Problem**: Standard PDF extraction misses tables/diagrams → incomplete answers

**Solutions**:

1. **Table extraction**:
```python
# Use specialized tools
- PyMuPDF / pdfplumber: Extract tables as structured data
- Camelot / Tabula: Table-specific extraction
- LlamaParse: AI-powered table understanding
```

2. **Convert tables to text**:
```
Table: Product Pricing
| Plan | Price | Users |
|------|-------|-------|
| Pro  | $99   | 10    |
| Enterprise | $299 | Unlimited |

→ "The Pro plan costs $99 per month and supports 10 users.
   The Enterprise plan costs $299 and supports unlimited users."
```

3. **Handle diagrams**:
- Extract figure captions (embed these)
- Use multimodal models (GPT-4V) to describe images
- Generate textual summaries of flowcharts/diagrams

4. **Chunk & index**:
- Treat tables as separate chunks
- Link to source page in metadata

**Q6. In a multi-tenant SaaS product, each customer’s documents must be kept separate. How do you architect RAG to ensure strict data isolation?**

**A.** I would partition the vector index logically or physically per tenant, tag each chunk with tenant ID, enforce tenant-specific filters at retrieval, and ensure that any caching or logging is also tenant-scoped so that queries never cross data boundaries.

**Q7. The RAG system becomes very slow after adding millions of documents. What optimizations would you consider?**

**A.** I would move from brute-force search to ANN indexes, tune index parameters for speed/recall trade-offs, shard the index across machines, precompute and cache retrievals for frequent queries, and reduce context size and reranking cost.

**Q8. A product team wants RAG to support both English and Hindi documents and queries. What design decisions are needed?**

**A.** I’d choose a multilingual embedding model, ensure tokenization and chunking handle both scripts well, store language metadata, route queries to language-specific or shared indexes, and decide whether to translate queries or documents or rely on cross-lingual embeddings.

**Q9. You’re asked to build a RAG system that can answer “why” questions requiring multi-hop reasoning across several documents. How would you extend a simple retrieve-then-read design?**

**A.** I’d add query decomposition so the model breaks questions into sub-questions, run retrieval separately for each, and use an agent that can iteratively retrieve, reason, and fetch more evidence, potentially using an intermediate chain-of-thought to orchestrate multi-step retrieval.

**Q10. In production, you observe that new documents are not influencing answers for many hours. How would you make the RAG system more real-time?**

**A.** I would build an incremental ingestion pipeline that embeds and upserts new or updated documents continuously (or on a short schedule), avoid full re-index, and ensure retriever queries the latest index plus optionally a small “fresh” index for very recent documents.

2. NVIDIA NeMo
2.1 Conceptual Questions (20)

**Q1. What is NVIDIA NeMo and where does it sit in the generative AI stack?**

**A.** NVIDIA NeMo is a framework and set of tools for building, training, fine-tuning, aligning, and deploying large language and multimodal models, sitting above low-level GPU libraries and below application-level orchestration.

**Q2. How does NeMo leverage Megatron-style model parallelism?**

**A.** NeMo uses Megatron techniques like tensor parallelism (splitting large matrix multiplications across GPUs) and pipeline parallelism (splitting layers across stages) to train very large transformer models across many GPUs efficiently.

**Q3. Explain data parallelism, tensor parallelism, and pipeline parallelism as supported in NeMo.**

**A.** Data parallelism replicates the model across GPU workers and splits batches; tensor parallelism shards individual layers’ weights across GPUs; pipeline parallelism splits the model depth into stages so microbatches flow through them like an assembly line.

**Q4. What is 3D parallelism in the context of NeMo?**

**A.** 3D parallelism combines data parallelism, tensor parallelism, and pipeline parallelism simultaneously, assigning each GPU to a specific combination of data shard, tensor shard, and pipeline stage to scale to very large models.

**Q5. How does NeMo support mixed-precision training (e.g., FP16, BF16, FP8)?**

**A.** NeMo integrates with Transformer Engine and GPU tensor cores to use reduced-precision formats like FP16 or BF16 for compute and FP32 for master weights, optionally FP8 on newer GPUs, improving throughput and memory efficiency while controlling numerical stability.

**Q6. What is NeMo Curator used for?**

**A.** NeMo Curator is used for large-scale data processing and cleaning: deduplication, quality filtering, language detection, PII removal, and other preprocessing needed before pre-training or fine-tuning large models.

**Q7. What is the purpose of NeMo-Aligner?**

**A.** NeMo-Aligner provides infrastructure to perform alignment methods like supervised fine-tuning, RLHF, and preference-based optimization so that models follow instructions, reflect human preferences, and avoid unsafe behavior.

**Q8. How does NeMo handle large-context or long-sequence training?**

**A.** NeMo can use sequence parallelism and memory-saving techniques such as activation checkpointing and selective recomputation, along with efficient attention implementations, to support long context windows.

**Q9. What is the benefit of using NeMo’s pre-built model checkpoints versus starting from scratch?**

**A.** Pre-built checkpoints give you strong base capabilities trained on large corpora, which you can adapt via fine-tuning or PEFT, saving massive compute and time compared to pre-training a model from random initialization.

**Q10. How does NeMo integrate with distributed training environments like multi-node GPU clusters?**

**A.** NeMo uses PyTorch distributed primitives, NCCL, and Megatron-Core to orchestrate communication across GPUs and nodes, with configuration-driven parallelism layouts so that training can scale across large GPU clusters.

**Q11. How does NeMo support parameter-efficient fine-tuning methods?**

**A.** NeMo allows you to attach modules such as LoRA layers or adapters to base models, training only these small modules while freezing most weights, reducing GPU memory and compute requirements for fine-tuning.

**Q12. What is NeMo Guardrails and how is it related to NeMo models?**

**A.** NeMo Guardrails is a separate but related toolkit that allows you to specify safety and behavior rules for LLM applications; it can be used alongside NeMo-trained models to intercept, validate, and adjust user inputs and model outputs according to those rules.

**Q13. Why is configuration management important in large NeMo training jobs?**

**A.** Large NeMo runs have many hyperparameters and parallelism settings; configuration files (often YAML) keep these structured, reproducible, and version-controlled, reducing human error and simplifying experiments.

**Q14. How does NeMo support Mixture-of-Experts (MoE) models?**

**A.** NeMo uses expert parallelism where different experts are placed on different GPUs, and a router sends tokens to experts based on learned routing weights, while handling gradient routing and load balancing for sparse models.

**Q15. What is activation checkpointing and why is it commonly used with NeMo?**

**A.** Activation checkpointing saves memory by discarding some intermediate activations during forward pass and recomputing them during backward pass; NeMo uses it to fit larger models or longer sequences on a fixed GPU memory budget.

**Q16. How do you monitor and debug a NeMo training job at scale?**

**A.** You typically use logging of loss and metrics, GPU utilization monitoring, distributed traces, gradient and weight statistics, and checkpoint inspection, often integrated with tools like TensorBoard or custom dashboards.

**Q17. What does it mean to “shard” optimizer states in NeMo?**

**A.** Sharding optimizer states means splitting large optimizer-related tensors (like moment estimates) across data-parallel workers to reduce memory overhead, often using techniques similar to ZeRO.

**Q18. Why is careful batch size and microbatch size tuning important in NeMo?**

**A.** On large distributed setups, global batch size affects optimization behavior, while microbatch size and gradient accumulation steps must be set to respect per-GPU memory limits while keeping GPUs fully utilized.

**Q19. How does NeMo handle loading and saving large checkpoints efficiently?**

**A.** NeMo writes sharded checkpoints corresponding to tensor and pipeline parallel partitions and may provide tools to consolidate or convert these, so that save/load operations are distributed and do not become a bottleneck.

**Q20. What kinds of models besides text-only LLMs can be trained with NeMo?**

**A.** NeMo also supports ASR (speech), TTS, multimodal vision–language models, and domain-specific models, sharing common training infrastructure across modalities where possible.

2.2 Scenario-Based Questions (10)

**Q1. You need to train a 70B-parameter model using NeMo on a cluster with 32 GPUs. How would you think about configuring data, tensor, and pipeline parallelism?**

**A.** I would size tensor parallelism to fit layer dimensions across multiple GPUs, use pipeline parallelism to split depth into a few stages, and use data parallelism for replication, making sure that each combination still fits in GPU memory while achieving good utilization.

**Q2. A NeMo training job is frequently running out of memory when you increase sequence length. What are your options?**

**A.** I can enable or increase activation checkpointing, use sequence parallelism, reduce microbatch size and increase gradient accumulation, or reduce model size; if hardware permits, I might also switch to more memory-efficient attention implementations.

**Q3. Your NeMo fine-tuned model has significantly worse perplexity than the base model. What steps would you take to diagnose the issue?**

**A.** I would verify data formatting and tokenization, check that the right tokenizer and vocab are used, inspect learning rate and schedule for being too aggressive, compare training and validation losses for overfitting, and re-run on a small subset to see if the problem reproduces.

**Q4. During multi-node NeMo training, you experience communication bottlenecks. How could you alleviate them?**

**A.** I’d examine network topology and NCCL settings, adjust parallelism strategy to reduce cross-node tensor parallel communication, enable gradient compression if available, and potentially increase pipeline depth to reduce cross-node tensor traffic.

**Q5. A customer wants to minimize cloud costs by reusing the same base model for multiple tasks with NeMo. What design would you propose?**

**A.** I’d propose a single shared base checkpoint with separate adapter or LoRA heads per task trained via PEFT, so that only small task-specific modules change while the base remains common, reducing both training and storage costs.

**Q6. You’re aligning a NeMo model with RLHF and notice training instability and reward hacking. What adjustments might help?**

**A.** I could tune the KL penalty coefficient, reduce learning rate, improve the quality and diversity of preference data, regularize the reward model, and monitor reward vs actual human quality alignment to detect and correct for reward gaming.

**Q7. Logs show that your NeMo pre-training run has a high variance in loss between workers. What might cause that and how would you investigate?**

**A.** Possible causes include inconsistent data sharding, bad or repeated samples on some workers, or gradient anomalies; I’d inspect data loaders, verify random seeds, compare sample distributions across workers, and check for numerical issues on specific GPUs.

**Q8. You have a NeMo model that works well in offline evaluation, but latency in an online setting is too high. How can you use NeMo outputs effectively in deployment?**

**A.** I’d export or convert the model to an optimized inference runtime (e.g., via ONNX/TensorRT or a serving stack), consider quantization and KV-cache optimizations, and tune batch sizes and sequence limits for lower latency.

**Q9. A team wants to switch from full fine-tuning to PEFT in NeMo to cut costs. What practical migration steps would you recommend?**

**A.** I’d freeze the base weights, identify which layers to adapt with LoRA or adapters, pick ranks and learning rates, re-run fine-tuning on the same datasets, and compare quality and resource usage before fully committing to PEFT.

**Q10. When resuming a large NeMo training run from a checkpoint, metrics diverge from the original trajectory. What could be going wrong?**

**A.** The optimizer state or scheduler state may not have been restored correctly, or random seeds and data shuffling could differ; I’d ensure that both model and optimizer states are saved and reloaded and that data order is reproducible across runs.

3. Fine-Tuning
3.1 Conceptual Questions (20)

**Q1. What is the difference between pre-training and fine-tuning for LLMs?**

**A.** Pre-training learns broad language representations from huge unlabeled corpora with self-supervised objectives, while fine-tuning adapts this base model on smaller labeled or curated datasets for particular tasks, domains, or behaviors.

**Q2. Why is full fine-tuning often impractical for very large models?**

**A.** Updating all parameters of a multi-billion-parameter model requires substantial GPU memory, compute time, and storage for multiple checkpoints, making full fine-tuning expensive and slow, especially across many tasks.

**Q3. What is supervised fine-tuning (SFT) in the context of instruction-tuned LLMs?**

**A.** SFT trains the model on instruction–response pairs with a standard supervised loss, teaching it to map prompts to desired outputs and to follow instruction style and formatting.

**Q4. How does catastrophic forgetting manifest during fine-tuning?**

**A.** After fine-tuning on a narrow dataset, the model may lose some of its general abilities or prior knowledge and perform worse on tasks it previously handled well.

**Q5. What strategies help mitigate catastrophic forgetting?**

**A.** Using smaller learning rates, mixing in some general or original data, adopting PEFT to limit how much of the model changes, and using regularization techniques or replay buffers can all help preserve prior knowledge.

**Q6. Why is learning rate scheduling important in fine-tuning?**

**A.** A well-chosen schedule like warm-up followed by decay can stabilize early training, prevent overshooting minima, and improve convergence, whereas a poor schedule may cause divergence or suboptimal results.

**Q7. How does instruction-tuning differ from task-specific fine-tuning?**

**A.** Instruction-tuning uses a diverse set of tasks and instructions to teach the model to follow natural language instructions in general, while task-specific fine-tuning focuses on maximizing performance for one particular objective.

**Q8. What is the role of the system prompt when training chat-style models?**

**A.** The system prompt defines global behavior (persona, safety, style); including it consistently during fine-tuning teaches the model to respect that “role” at inference time.

**Q9. Why can too many fine-tuning epochs hurt performance on real-world data?**

**A.** Too many epochs overfit the training set, making the model brittle and less able to generalize to slightly different or noisier user inputs seen in production.

**Q10. How does batch size impact fine-tuning dynamics?**

**A.** Larger batches provide smoother gradient estimates and can be more stable but may hurt generalization; smaller batches introduce noise that can help escape local minima but can also cause instability if too small.

**Q11. What is the purpose of gradient accumulation in LLM fine-tuning?**

**A.** Gradient accumulation simulates a larger batch size by summing gradients across several microbatches before updating weights, allowing training with effectively large batches on limited GPU memory.

**Q12. When might continual pre-training be preferable before fine-tuning?**

**A.** When the target domain’s unlabeled text distribution is very different from generic web data (e.g., biomedical or legal text), continual pre-training can adapt the representations before you apply task-specific supervised fine-tuning.

**Q13. What is multi-task fine-tuning and what are its benefits?**

**A.** Multi-task fine-tuning trains on multiple tasks simultaneously, often using special prompts or instructions to identify each task; it can improve generalization and reduce the need for separate models for each task.

**Q14. How do you decide between few-shot prompting and actual fine-tuning for a new task?**

**A.** If a strong base model performs acceptably with well-crafted few-shot prompts and the task is not critical, prompting may suffice; for more demanding accuracy, consistency, or latency requirements, fine-tuning is usually preferable.

**Q15. Why is data quality often more important than data quantity in fine-tuning?**

**A.** Noisy or poorly formatted data can teach bad behavior and confuse the model; a small, clean, diverse set of high-quality examples can yield better performance than a much larger but messy dataset.

**Q16. What is knowledge distillation and how does it relate to fine-tuning?**

**A.** Knowledge distillation trains a smaller “student” model to imitate a larger “teacher” model’s outputs; fine-tuning is used to adjust the student to match the teacher on selected tasks or datasets.

**Q17. How does model size influence the benefits you get from fine-tuning?**

**A.** Larger models have stronger priors and can often be adapted with smaller datasets and milder parameter changes, while smaller models may require more targeted or specialized fine-tuning to reach comparable performance.

**Q18. Why is it important to match tokenization between base model and fine-tuning data?**

**A.** Using a mismatched tokenizer can break the mapping between tokens and learned embeddings, causing degraded performance or even training instability.

**Q19. How can evaluation setups mislead you about the success of fine-tuning?**

**A.** If the validation set is too similar to training data, if prompts differ from real usage, or if evaluation only measures superficial metrics, you may overestimate real-world performance.

**Q20. Why might you fine-tune on formatting or style even if the base model already “knows” the task?**

**A.** Fine-tuning can enforce consistent output structure, domain-specific phrasing, or compliance rules, which may be as important as raw task accuracy in production applications.

3.2 Scenario-Based Questions (10)

**Q1. After SFT on your company’s support transcripts, the model answers questions accurately but in an overly casual tone. How would you adjust fine-tuning?**

**A.** I’d curate additional examples with the desired professional tone, update the system prompt to explicitly specify style, and run another fine-tuning pass emphasizing style and phrasing, possibly with preference-based optimization.

**Q2. A fine-tuned model performs well on your in-house test set but fails on real user queries. What could be the root cause?**

**A.** The test set likely doesn’t reflect real query distribution; it may be too clean, too narrow, or not diverse; I’d collect representative production queries, build a more realistic evaluation set, and fine-tune using data closer to actual usage.

**Q3. You only have 300 labeled examples for a specialized classification task. How would you approach fine-tuning?**

**A.** I’d start with a strong instruction-tuned base model, use PEFT with a small learning rate, possibly augment the dataset with synthetic examples or paraphrases, and carefully monitor validation performance to avoid overfitting.

**Q4. During fine-tuning, you see training loss going down but validation loss going up. How do you respond?**

**A.** That indicates overfitting; I would lower the number of epochs, reduce learning rate, try early stopping, and possibly augment or regularize data rather than continuing training.

**Q5. You want a single model to do summarization, Q&A, and rewriting for customer documents. How would you design the fine-tuning dataset?**

**A.** I’d mix tasks with explicit instructions indicating each task type, ensure balanced coverage, and create task-specific prompts and labels so that the model learns to follow task cues within one unified training run.

**Q6. A legal compliance team requests that the model explicitly state uncertainty instead of guessing. How can fine-tuning help?**

**A.** I’d include many examples where the model responds with “I don’t know” or defers when information is missing, and possibly penalize hallucinated answers via preference-based tuning so the model learns to show uncertainty appropriately.

**Q7. You notice that fine-tuning on a narrow task caused the model to forget how to write in multiple languages. How might you fix this?**

**A.** I’d include a small portion of multilingual data in the fine-tuning mix, reduce learning rate, possibly use PEFT instead of full fine-tuning, and validate on multilingual benchmarks while training.

**Q8. Your fine-tuned code assistant sometimes outputs syntactically invalid code. What training data or objectives could improve this?**

**A.** I would add more examples labeled with correct code outputs, include tests or error messages, and possibly incorporate a reinforcement step where passing unit tests is rewarded and failing code is penalized.

**Q9. After deploying a fine-tuned model, you find out many users are using it for a different purpose than originally intended. What is a safe strategy to adapt?**

**A.** I’d collect anonymized logs, filter them with privacy and safety constraints, curate new fine-tuning or preference data that reflect the new use case, and re-run fine-tuning or alignment while preserving previously validated behaviors.

**Q10. Your organization has strict data privacy constraints and won’t allow any external API calls. How does this affect your fine-tuning approach?**

**A.** I must run all training on-prem or within a private cloud, use only locally stored datasets, and ensure the base model and training pipeline do not send telemetry, possibly preferring open-weight models and self-managed infrastructure.

4. Quantization
4.1 Conceptual Questions (20)

**Q1. What is quantization in the context of neural networks?**

**A.** Quantization reduces the numerical precision of weights and/or activations (e.g., from 32-bit floating point to 8-bit integer) to save memory and computation at the cost of some approximation error.

**Q2. How does post-training quantization (PTQ) differ from quantization-aware training (QAT)?**

**A.** PTQ applies quantization to a pre-trained model using calibration data without further training, while QAT simulates quantization during training so the model learns to compensate for quantization errors.

**Q3. What is the typical trade-off when moving from FP32 to INT8 weights?**

**A.** You generally gain significant memory and speed improvements while incurring only a modest loss in accuracy or perplexity, assuming a good quantization scheme and calibration.

**Q4. Why is INT4 quantization more challenging than INT8?**

**A.** With only 16 discrete levels, INT4 has far fewer representable values, making it more sensitive to quantization error and requiring more sophisticated schemes to preserve accuracy.

**Q5. What is symmetric versus asymmetric quantization?**

**A.** Symmetric quantization uses a scale centered around zero (same magnitude for positive and negative ranges), while asymmetric quantization allows distinct zero-points and ranges to better fit skewed activation distributions.

**Q6. What is per-channel quantization and why is it beneficial?**

**A.** Per-channel quantization uses separate scales for each channel or row/column of a weight matrix, better matching per-channel statistics and reducing error compared to a single shared scale.

**Q7. Why do we need calibration data in PTQ?**

**A.** Calibration data approximates real activation distributions so the quantizer can choose appropriate scales and zero-points; poor calibration can cause severe accuracy degradation.

**Q8. What is weight-only quantization and when is it sufficient?**

**A.** Weight-only quantization encodes weights at low precision while leaving activations in higher precision; it reduces memory and model size and can provide good speedups on memory-bound workloads without changing activation arithmetic.

**Q9. How does quantization impact the KV cache in transformer inference?**

**A.** Quantizing the KV cache reduces the memory and bandwidth cost of storing and reading past key–value activations, which grow linearly with sequence length and can dominate memory usage for long contexts.

**Q10. Why might you choose mixed-precision quantization?**

**A.** Different layers or tensors have different sensitivity to precision; mixed precision allows you to keep sensitive parts (e.g., embeddings or output layers) at higher precision while aggressively quantizing others for better overall trade-offs.

**Q11. How does quantization affect numerical stability in deep networks?**

**A.** Reduced precision can increase rounding error and quantization noise, potentially amplifying through layers, which is why careful scaling, clipping, and sometimes retraining (QAT) are used to maintain stability.

**Q12. Why are large models sometimes more robust to quantization than smaller ones?**

**A.** Large models have redundancy and overparameterization, so the effect of small perturbations in weights is often absorbed without catastrophic degradation, whereas small models can be more brittle.

**Q13. What is the role of zero-point in integer quantization?**

**A.** The zero-point represents which integer value corresponds to floating-point zero, enabling integer representation of asymmetric ranges shifted away from zero.

**Q14. How do hardware capabilities influence quantization strategy?**

**A.** Available data types and instructions (e.g., INT8 dot products, FP16/FP8 tensor cores) constrain which quantization formats are efficient, so you pick schemes that match the hardware’s optimized paths.

**Q15. What is “fake quantization” used in QAT?**

**A.** Fake quantization inserts simulated quantize–dequantize operations in the forward pass during training while still storing parameters as floats; gradients flow through approximations of the quantization step.

**Q16. Why might you avoid quantizing the final output layer in some applications?**

**A.** The output layer often directly impacts logits and probabilities; small errors there can disproportionately affect final predictions, so leaving it at higher precision preserves accuracy.

**Q17. How does quantization interact with activation functions like ReLU or GELU?**

**A.** These activations shape the distribution of activations; for example, ReLU clips negatives to zero, producing skewed distributions that asymmetric quantization may model better than symmetric.

**Q18. What kinds of tasks tend to be most sensitive to aggressive quantization?**

**A.** Tasks requiring precise numeric reasoning, multi-step logic, or exact string outputs (like code generation) can degrade more with low-bit quantization than simpler classification or coarse summarization.

**Q19. How can you detect that quantization has degraded your model too much?**

**A.** You compare metrics (e.g., accuracy, BLEU, perplexity) before and after quantization and also perform qualitative checks on representative prompts to catch subtle degradations that metrics may miss.

**Q20. Why is quantization particularly attractive for edge and mobile deployment?**

**A.** Lower-precision models fit into smaller memory footprints, reduce energy consumption, and can run efficiently on limited hardware with integer arithmetic units, making them suitable for on-device inference.

4.2 Scenario-Based Questions (10)

**Q1. You have a 13B-parameter model that doesn’t fit on your 16 GB GPU in FP16. How would quantization help, and what scheme would you start with?**

**A.** I’d start with INT8 or a proven 4-bit scheme for weight-only quantization, reducing memory enough for the model to fit and then check if quality remains acceptable for the target task.

**Q2. After applying naive INT4 quantization, your model’s perplexity doubles. What steps would you take to improve results?**

**A.** I’d use better calibration, consider per-channel scales, selectively keep sensitive layers at higher precision, or switch to a more advanced 4-bit method; if that still fails, I’d back off to INT8.

**Q3. Your quantized model works fine for classification but fails badly on a code generation benchmark. What does this suggest and how do you respond?**

**A.** It suggests code generation is more sensitive to quantization error; I’d try INT8 instead of INT4 for critical layers, use mixed precision for embeddings and output projection, and re-evaluate.

**Q4. During deployment tests, you find that quantization only slightly improved latency but significantly reduced memory usage. Why might that be?**

**A.** The workload may be compute-bound rather than memory-bound, or the inference runtime may not fully exploit integer kernels; memory savings help capacity but not necessarily latency without optimized compute paths.

**Q5. Your team wants to quantize both weights and activations aggressively to hit a tight latency target. What risks and mitigations do you consider?**

**A.** The main risk is large accuracy loss; I’d use carefully calibrated QAT, keep some layers at higher precision, and set strict acceptance thresholds on evaluation metrics before deploying.

**Q6. You quantized a multilingual model and observed that performance dropped much more for low-resource languages than for English. Why could this happen?**

**A.** Representations for low-resource languages may be more fragile and concentrated in fewer parameters, so quantization noise impacts them more; I would keep embeddings or some layers in higher precision or use milder quantization for those parts.

**Q7. A hardware team asks you to target a specific integer format that your usual framework doesn’t support natively. How do you approach this?**

**A.** I’d check if custom kernels or conversion tools exist, possibly write a small adapter layer that converts to the hardware format, and run careful calibration and validation to ensure compatibility and accuracy.

**Q8. You suspect your calibration dataset doesn’t represent real traffic well. What effect could this have on quantization and how do you fix it?**

**A.** Misrepresentative calibration leads to bad scale/zero-point choices and poor accuracy; I’d collect calibration traces that match typical input distributions and re-run calibration and evaluation.

**Q9. After quantization, you see inconsistent results between two different inference libraries using the same quantized weights. Why might that be?**

**A.** Libraries might implement quantization differently (e.g., rounding rules, scale application, integer accumulation precision), so you need to align schemes, verify assumptions, and possibly export in a standardized format.

**Q10. Your edge deployment requires strict real-time constraints but allows some offline computation. How can quantization be combined with other techniques to meet requirements?**

**A.** I’d quantize the model and KV cache, precompute or cache common intermediate results, limit maximum sequence lengths, and possibly use speculative decoding or smaller draft models to reduce per-token compute.

5. Parameter-Efficient Fine-Tuning (PEFT)
5.1 Conceptual Questions (20)

**Q1. What is parameter-efficient fine-tuning (PEFT) and why is it useful for large models?**

**A.** PEFT fine-tunes only a small subset of parameters (or adds small new modules) while keeping most base weights frozen, reducing memory, compute, and storage compared to full fine-tuning.

**Q2. How does LoRA (Low-Rank Adaptation) modify a pre-trained weight matrix?**

**A.** LoRA keeps the original weight matrix frozen and adds a low-rank decomposition (A and B matrices) whose product is added to the original weight during forward pass, with only A and B being trained.

**Q3. What is the intuition behind using low-rank updates in LoRA?**

**A.** Many useful adaptations can be expressed as low-rank changes to existing weights, so you can approximate the needed update with far fewer parameters than a full matrix.

**Q4. How do you choose the rank for a LoRA adapter and what are the trade-offs?**

**A.** Higher rank gives more expressive capacity and potentially better performance but increases memory and compute; you typically start with a modest rank (e.g., 8–64) and tune based on task difficulty and resources.

**Q5. What is the role of the LoRA scaling factor (often called alpha)?**

**A.** Alpha scales the LoRA update relative to the frozen weight, helping control the strength of adaptation and stabilizing training by keeping the effective change within a reasonable range.

**Q6. How does PEFT enable storing many specialized models cheaply?**

**A.** You store one large base model and many tiny adapter parameter sets for different tasks; each adapter is orders of magnitude smaller than a full model, so you can keep many variants without multiplying storage.

**Q7. What are adapter layers and how do they differ from LoRA?**

**A.** Adapters are small bottleneck feed-forward modules inserted within the network layers, whereas LoRA modifies existing linear projections via low-rank updates; both train only a small number of new parameters.

**Q8. What is prompt tuning or soft prompting in the PEFT context?**

**A.** Prompt tuning learns additional continuous embedding vectors prepended to inputs (soft prompts) while keeping the model weights frozen, effectively steering behavior through learned prompts.

**Q9. When might soft prompting be insufficient compared to LoRA or adapters?**

**A.** For complex tasks or major domain shifts, the limited capacity of soft prompts may not capture necessary transformations, so methods that modify internal representations like LoRA or adapters are more effective.

**Q10. How does PEFT reduce communication overhead in distributed training?**

**A.** Since only a small set of adapter parameters have gradients, the amount of data that needs to be synchronized across data-parallel workers is much smaller than for full-model gradient updates.

**Q11. Why is PEFT particularly attractive for organizations with many downstream tasks?**

**A.** They can maintain a single base model and cheaply spin up task-specific variants by training small adapters, avoiding repeated full fine-tunes and reducing both engineering and infrastructure costs.

**Q12. Can multiple PEFT adapters be combined or composed at inference time?**

**A.** In principle, yes—you can load multiple adapters and either select one by task or design mechanisms to blend their contributions, though careful design is needed to avoid interference.

**Q13. How does PEFT interact with quantized base models?**

**A.** Often the base model is quantized while adapters remain in higher precision; during inference, quantized weights plus small high-precision updates are combined, enabling fine-tuning on limited hardware.

**Q14. What is the trade-off between many small task-specific adapters vs a single multi-task adapter?**

**A.** Many small adapters isolate tasks and reduce interference, while a single multi-task adapter can exploit shared structure but risks negative transfer and requires careful multi-task training.

**Q15. How does PEFT help with rapid experimentation?**

**A.** Because training adapters is relatively fast and cheap, teams can iterate over datasets, objectives, and hyperparameters quickly without needing to rerun expensive full-model fine-tunes.

**Q16. Why might you still choose full fine-tuning for some use cases even if PEFT is available?**

**A.** For mission-critical tasks demanding maximal performance, extreme domain shift, or where infrastructure is ample and maintenance complexity is low, full fine-tuning may yield slightly better results.

**Q17. How does PEFT affect model loading and startup times in production?**

**A.** The base model is loaded once; switching tasks typically involves only loading or swapping small adapter weight files, which is faster and lighter than reloading entire models.

**Q18. What considerations go into deciding which layers to adapt with LoRA?**

**A.** You target layers where small changes have strong downstream effects, typically attention projections and/or feed-forward layers; empirically, some layers are more influential and worth adapting.

**Q19. Why do PEFT methods often generalize surprisingly well despite their small parameter counts?**

**A.** They exploit the rich representational capacity learned during pre-training and only need to steer this capacity slightly, so small changes can suffice to align behavior with new tasks.

**Q20. What limitations should you keep in mind when relying heavily on PEFT?**

**A.** PEFT can sometimes underperform full fine-tuning on very complex or highly specialized tasks; adapter interactions can be tricky; and you must manage adapter proliferation and configuration carefully.

5.2 Scenario-Based Questions (10)

**Q1. You need to adapt a 70B model to a new domain using a single 48 GB GPU. How would PEFT help?**

**A.** I’d load the base model in as efficient a format as possible and add LoRA adapters, training only the adapters with small batch sizes and gradient accumulation so the GPU memory budget is not exceeded.

**Q2. A company wants separate chatbots for HR, legal, and engineering, all based on the same LLM. How would you structure the solution with PEFT?**

**A.** I’d maintain one base model and train three separate adapters, one per domain, each fine-tuned on domain-specific data; at runtime, I’d select the appropriate adapter based on which chatbot is being used.

**Q3. You trained a LoRA adapter but see little difference in model behavior. What might be wrong?**

**A.** The rank or scaling might be too small, the learning rate might be too low, or the adapter may be attached to ineffective layers; I’d inspect logs, increase rank or alpha, and ensure gradients are flowing where expected.

**Q4. Your team wants to experiment with a new task without risking interference with existing adapters. How do you proceed?**

**A.** I would create a new adapter for the new task, train it independently while leaving others untouched, and then evaluate task performance and any unintended interactions before integrating.

**Q5. You applied PEFT for a complex reasoning task but performance plateaued far below full fine-tuning baselines. What options remain?**

**A.** I could increase adapter rank, adapt more layers, refine data, or selectively fine-tune a subset of base layers (partial full fine-tuning) in addition to adapters.

**Q6. In deployment, you need to serve many tenants each with their own adapter. How do you manage memory and throughput?**

**A.** I’d load the base model once, implement efficient adapter loading and caching, batch requests across tenants where possible, and evict least-used adapters when memory is tight.

**Q7. You quantized the base model and noticed that adapter fine-tuning is unstable. What might you change?**

**A.** I’d ensure adapters are kept in higher precision, adjust learning rate and scaling, verify that the quantization scheme is compatible, and possibly use a slightly less aggressive quantization format.

**Q8. Users report that adapter A and adapter B conflict when both are active. How do you debug and address this?**

**A.** I’d test them separately to confirm they work in isolation, inspect their layer targets for overlap, and consider switching to explicit routing (only one active at a time) or retraining a combined adapter for overlapping tasks.

**Q9. New regulations require that models be easily updateable to reflect policy changes. How can PEFT help meet this requirement?**

**A.** I can train updated adapters on policy changes while keeping the base fixed; swapping to new adapters is fast and doesn’t require re-validating the entire base model.

**Q10. An experiment shows that full fine-tuning gives only a small performance boost over a well-designed adapter. How do you decide whether the extra complexity is worth it?**

**A.** I would weigh the marginal accuracy gain against the increased training cost, infrastructure needs, and maintenance burden; for many applications, a small benefit is not worth the significant added complexity, so PEFT remains the practical choic

---

## RAG — Chunking & Evaluation

**Q1. What happens to Context Precision and Context Recall when chunks are too small?**

**A.** Context Precision goes HIGH — retrieved chunks are focused and relevant, little junk. Context Recall goes LOW — the full answer is split across many small chunks and the retriever misses some of them. One idea spanning 4 chunks means the retriever must rank all 4 in top-K or the answer is incomplete.

**Q2. What happens to Context Precision and Context Recall when chunks are too large?**

**A.** Context Recall goes HIGH — the answer is fully contained in the chunk. Context Precision goes LOW — the chunk carries a lot of irrelevant content alongside the answer. The embedding also becomes a blurry average of multiple topics, weakening semantic search. The LLM suffers "lost in the middle" — the relevant part is buried.

**Q3. How does Parent-Child chunking solve the Precision vs Recall tradeoff?**

**A.** Small child chunks (128–256 tokens) are embedded and used for retrieval — giving high Precision because the embedding is focused. When a child chunk is matched, its parent chunk (512–1024 tokens) is returned to the LLM — giving high Recall because the full context is intact. LangChain's ParentDocumentRetriever implements this pattern.

**Q4. What is the "lost in the middle" problem in RAG?**

**A.** When a large chunk is passed to the LLM, the relevant answer may be buried in the middle of hundreds of irrelevant tokens. Research shows LLMs attend most to content at the beginning and end of context — content in the middle is often ignored, leading to incorrect or incomplete answers despite the information being present.

**Q5. How do you decide the optimal chunk size for a production RAG system?**

**A.** There is no universal best size — it depends on content type. Dense technical/legal docs: 256–512 tokens. General articles: 512–1024 tokens. Code: per function/class boundary. Tables: one table per chunk. Always evaluate using Context Precision and Context Recall metrics on your actual data with DeepEval or RAGAS before committing to a chunk size.

---

## RAG — Production Evaluation

**Q6. What is the best practice to evaluate RAG retrieval accuracy in PROD since we won't have ground truth?**

**A.** Use **LLM-as-judge** — let the LLM score whether the retrieved chunks actually answer the query. No ground truth needed.

Three practical approaches:

| Approach | How | Ground Truth? |
|---|---|---|
| **LLM-as-judge** | LLM scores relevance of each retrieved chunk (0–1) | ✗ Not needed |
| **Implicit feedback** | Track user thumbs down, follow-up "that didn't help" signals | ✗ Not needed |
| **Synthetic ground truth** | LLM generates Q&A pairs from your docs — use as test set | Self-generated |
| **Human spot-check** | Sample 50–100 queries weekly, humans rate retrieval | ✗ Small set |

**LLM-as-judge pattern (no ground truth):**
```python
# For each retrieved chunk, ask LLM: "Does this chunk help answer the query?"
prompt = f"""
Query: {query}
Retrieved chunk: {chunk}

Does this chunk contain information relevant to answering the query?
Score 1 (relevant) or 0 (not relevant). Reply with just the number.
"""
score = llm.invoke(prompt)   # 0 or 1 per chunk
precision = sum(scores) / len(scores)   # % of retrieved chunks that were relevant
```

**Implicit signals in production:**
```
User asks → RAG responds → user immediately asks same question differently → BAD retrieval
User asks → RAG responds → user says "thanks" / continues → GOOD retrieval
Track: re-ask rate, session abandonment, thumbs down → proxy for retrieval quality
```

---

**Q7. If chunk is large → high recall, low precision. If chunk is small → high precision, low recall. Is this right?**

**A.** Yes — correct understanding. Here's why:

```
LARGE CHUNKS:
  ✓ Full answer fits inside one chunk → Recall HIGH (answer not missed)
  ✗ Chunk has lots of irrelevant content too → Precision LOW
  ✗ Embedding = blurry average of many topics → weaker semantic match

SMALL CHUNKS:
  ✓ Embedding is focused on one idea → strong semantic match → Precision HIGH
  ✗ Answer split across many chunks → retriever must rank ALL of them in top-K
    → if any are missed → Recall LOW
```

```
Chunk size  ──────────────────────────────────────────►
Small        Medium (sweet spot)                  Large
  │                  │                              │
Precision ↑↑        Precision ↑, Recall ↑         Recall ↑↑
Recall ↓↓           (Parent-Child achieves this)   Precision ↓↓
```

**Fix:** Parent-Child chunking — small chunks for retrieval (precision), large parent returned to LLM (recall).

---

**Q8. What are the evaluation metrics for RAG retrieval quality? Do they require Ground Truth?**

**A.**

| Metric | What it measures | Ground Truth needed? |
|---|---|---|
| **Context Precision** | % of retrieved chunks that are actually relevant | ✓ Yes (or LLM-as-judge) |
| **Context Recall** | % of relevant chunks that were retrieved (none missed) | ✓ Yes |
| **Answer Relevance** | Does the final answer address the question? | ✗ No (LLM-as-judge) |
| **Faithfulness** | Is the answer grounded in retrieved chunks (no hallucination)? | ✗ No (LLM-as-judge) |
| **MRR** (Mean Reciprocal Rank) | How high does the first relevant chunk rank? | ✓ Yes |
| **NDCG** | Quality of full ranking order of retrieved chunks | ✓ Yes |

**Without ground truth — use these (LLM-as-judge):**
```
Answer Relevance  → LLM scores: does response answer the question?
Faithfulness      → LLM scores: is every claim in the answer supported by chunks?
Context Relevance → LLM scores: is each retrieved chunk relevant to the query?
```

**With ground truth (offline eval / synthetic):**
```
Context Precision = relevant chunks retrieved / total chunks retrieved
Context Recall    = relevant chunks retrieved / total relevant chunks in corpus
```

**Tools:**
- **RAGAS** — computes all above metrics, supports LLM-as-judge mode (no ground truth)
- **DeepEval** — similar, production-ready, integrates with CI/CD
- **LangSmith** — traces + evaluates in one platform