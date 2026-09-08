07_RAG_PIPELINE.md
# RAG Pipeline: Vector Stores, Embeddings, Memory, and Knowledge Corpora

## Table of Contents
- [RAG Pipeline: Vector Stores, Embeddings, Memory, and Knowledge Corpora](#rag-pipeline-vector-stores-embeddings-memory-and-knowledge-corpora)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Part 1: RAG Architecture](#part-1-rag-architecture)
    - [Concept: What is RAG](#concept-what-is-rag)
    - [RAG vs Fine-Tuning vs Prompting](#rag-vs-fine-tuning-vs-prompting)
    - [RAG Pipeline Components](#rag-pipeline-components)
    - [Kubernetes Deployment](#kubernetes-deployment)
  - [Part 2: Vector Databases](#part-2-vector-databases)
    - [Concept: Why Vector Databases](#concept-why-vector-databases)
    - [pgvector (PostgreSQL)](#pgvector-postgresql)
    - [Qdrant (Standalone)](#qdrant-standalone)
    - [Weaviate (Distributed)](#weaviate-distributed)
    - [Comparison \& Decision Tree](#comparison--decision-tree)
  - [Part 3: Embedding Pipeline](#part-3-embedding-pipeline)
    - [Concept: Converting Text to Vectors](#concept-converting-text-to-vectors)
    - [Embedding Models](#embedding-models)
    - [Chunking Strategies](#chunking-strategies)
    - [Batch Ingestion](#batch-ingestion)
  - [Part 4: Knowledge Corpora for Presight](#part-4-knowledge-corpora-for-presight)
    - [Security Knowledge Sources](#security-knowledge-sources)
    - [MITRE ATT\&CK Knowledge Base](#mitre-attck-knowledge-base)
    - [Custom Security Playbooks](#custom-security-playbooks)
  - [Part 5: Long-Term Memory for Agents](#part-5-long-term-memory-for-agents)
    - [Concept: Why Agents Need Memory](#concept-why-agents-need-memory)
    - [Conversation Memory](#conversation-memory)
    - [Incident Memory](#incident-memory)
    - [Learning from Feedback](#learning-from-feedback)
  - [Part 6: RAG in Agent Loops](#part-6-rag-in-agent-loops)
    - [Planner + RAG](#planner--rag)
    - [Executor + RAG](#executor--rag)
    - [Critic + RAG](#critic--rag)
  - [Production: Evaluation \& Safety](#production-evaluation--safety)
    - [RAG-Specific Evaluation Metrics](#rag-specific-evaluation-metrics)
  - [Interview Questions](#interview-questions)

---

## Overview

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the prompt with those documents
3. **Generating** a response grounded in the retrieved information

For Presight's security agent platform:
- **Knowledge base** = MITRE ATT&CK, vulnerability databases, playbooks, incident history
- **Retrieval** = Fast semantic search via vector embeddings
- **Augmentation** = "Here's what happened in similar incidents before"
- **Generation** = Agent reason over historical context + real-time data

---

## Part 1: RAG Architecture

### Concept: What is RAG

**Problem Without RAG:**
```
User: "What's the remediation for CVE-2024-12345?"
     ↓
LLM: "I don't know. My knowledge cutoff is April 2024."
```

**Solution With RAG:**
```
User: "What's the remediation for CVE-2024-12345?"
     ↓
Retriever: Search knowledge base → Find similar CVEs and remediations
     ↓
Augment: "Here are similar CVEs: CVE-2024-11111 (remediation: patch X), CVE-2024-11112 (remediation: patch Y)"
     ↓
LLM: "Based on similar CVEs, remediation is likely patch X or Y, combined with..."
```

### RAG vs Fine-Tuning vs Prompting

| Approach | Cost | Speed | Accuracy | Use Case |
|----------|------|-------|----------|----------|
| **Prompting** | ✓ Low | ✓ Fast | ✗ Generic | Simple Q&A |
| **RAG** | ✓ Medium | ✓ Fast | ✓✓ Accurate | Domain-specific (security) |
| **Fine-Tuning** | ✗ High | ✗ Slow | ✓✓✓ Precise | Highly specialized |

**For Presight**: Use **RAG** because:
- Security knowledge changes rapidly (new CVEs, exploits)
- Fine-tuning is expensive and slow to update
- RAG lets us update knowledge base daily

### RAG Pipeline Components

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INGEST (Offline, batch)                                      │
│   ├─ Source: MITRE ATT&CK, CVE feeds, playbooks, incident logs  │
│   ├─ Process: Chunk → Embed → Index                              │
│   └─ Store: Vector DB (Qdrant, pgvector, Weaviate)              │
├─────────────────────────────────────────────────────────────────┤
│ 2. RETRIEVE (Online, per query)                                 │
│   ├─ User Query: "Suspicious PowerShell execution"              │
│   ├─ Embedding: Convert to vector (768-dim)                     │
│   ├─ Search: Find top-K similar documents (k=5-10)              │
│   └─ Filter: By severity, recency, relevance score              │
├─────────────────────────────────────────────────────────────────┤
│ 3. AUGMENT (Prompt building)                                    │
│   ├─ Template: "Similar incidents:"                             │
│   ├─ Insert: Retrieved documents                                │
│   └─ Result: Extended context for LLM                           │
├─────────────────────────────────────────────────────────────────┤
│ 4. GENERATE (LLM reasoning)                                     │
│   ├─ Input: Original query + retrieved context                  │
│   ├─ Process: LLM reasons over all information                  │
│   └─ Output: Grounded, contextual response                      │
├─────────────────────────────────────────────────────────────────┤
│ 5. EVALUATION (Safety & Quality)                                │
│   ├─ Groundedness: Is response based on retrieved docs?         │
│   ├─ Relevance: Did we retrieve relevant docs?                  │
│   ├─ Hallucination: Did LLM make up facts?                      │
│   └─ Safety: No data exfiltration, injection attempts?          │
└─────────────────────────────────────────────────────────────────┘
```

### Kubernetes Deployment

```yaml
# rag-infrastructure.yaml

---
# 1. Vector Database (Qdrant)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: presight-agents
spec:
  serviceName: qdrant
  replicas: 3
  selector:
    matchLabels:
      component: vector-db
  template:
    metadata:
      labels:
        component: vector-db
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - name: http
          containerPort: 6333
        - name: grpc
          containerPort: 6334
        
        env:
        - name: QDRANT_API_KEY
          valueFrom:
            secretKeyRef:
              name: qdrant-secrets
              key: api-key
        
        volumeMounts:
        - name: qdrant-data
          mountPath: /qdrant/storage
        
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 6333
          initialDelaySeconds: 30
          periodSeconds: 10
  
  volumeClaimTemplates:
  - metadata:
      name: qdrant-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 500Gi

---
# 2. Service for Qdrant
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: presight-agents
spec:
  clusterIP: None  # Headless service for StatefulSet
  selector:
    component: vector-db
  ports:
  - name: http
    port: 6333
    targetPort: 6333
  - name: grpc
    port: 6334
    targetPort: 6334

---
# 3. Embedding Service (Uses sentence-transformers)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-service
  namespace: presight-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-service
  template:
    metadata:
      labels:
        app: embedding-service
    spec:
      containers:
      - name: embeddings
        image: presight/embedding-service:v0.1
        ports:
        - name: http
          containerPort: 8000
        
        env:
        - name: MODEL_NAME
          value: "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
        
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
# 4. Ingestion Pipeline (Batch job)
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rag-ingest-daily
  namespace: presight-agents
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: ingest
            image: presight/rag-ingest:v0.1
            env:
            - name: QDRANT_URL
              value: "http://qdrant:6333"
            - name: EMBEDDING_SERVICE_URL
              value: "http://embedding-service:8000"
            - name: KNOWLEDGE_SOURCE_URL
              value: "https://attack.mitre.org/api"
            volumeMounts:
            - name: ingest-config
              mountPath: /etc/rag
          
          volumes:
          - name: ingest-config
            configMap:
              name: rag-ingest-config
          
          restartPolicy: OnFailure
          backoffLimit: 3
```

---

## Part 2: Vector Databases

### Concept: Why Vector Databases

**Traditional Database:**
```sql
SELECT * FROM documents WHERE title LIKE '%PowerShell%'
-- Finds literal string matches only
```

**Vector Database:**
```python
# Find semantically similar incidents
results = vector_db.search(
    query_vector=[0.23, 0.45, -0.12, ...],  # Query embedding
    top_k=10,
    threshold=0.85  # Similarity score
)
# Returns: "PowerShell execution", "script-based attack", "command execution"
# All semantically related, not just string matches!
```

### pgvector (PostgreSQL)

**Best for:** Integrated stack, small-medium corpora, existing PostgreSQL users

```python
# Installation
# pip install pgvector psycopg2

from pgvector.psycopg2 import register_vector
import psycopg2

conn = psycopg2.connect("postgresql://user:password@localhost/presight")
register_vector(conn)

# Create table with vector column
conn.execute("""
    CREATE TABLE security_knowledge (
        id SERIAL PRIMARY KEY,
        title VARCHAR(255),
        content TEXT,
        embedding vector(384),  -- 384-dim embeddings
        source VARCHAR(100),    -- MITRE, CVE, custom
        created_at TIMESTAMP DEFAULT NOW(),
        
        -- Index for fast search
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    );
""")

# Insert with embedding
embedding = embed_model.encode("PowerShell execution detected")

conn.execute(
    "INSERT INTO security_knowledge (title, content, embedding, source) VALUES (%s, %s, %s, %s)",
    ("PowerShell Execution", "Description...", embedding, "mitre")
)

# Search (semantic)
results = conn.execute("""
    SELECT title, content, 1 - (embedding <=> %s::vector) AS similarity
    FROM security_knowledge
    WHERE source = 'mitre'
    ORDER BY embedding <=> %s::vector
    LIMIT 10
""", (embedding, embedding))
```

**Pros:**
- ✓ Integrated with PostgreSQL (same DB, backups, RBAC)
- ✓ Cost-effective (no separate service)
- ✓ SQL queries on structured + vector data

**Cons:**
- ✗ Slower for large-scale (>10M documents)
- ✗ No built-in replication at scale

### Qdrant (Standalone)

**Best for:** Large corpora, dedicated vector search, production scale

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("http://qdrant:6333")

# Create collection
client.create_collection(
    collection_name="security-knowledge",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Upsert documents with vectors
points = [
    PointStruct(
        id=1,
        vector=[0.23, 0.45, -0.12, ...],  # 384-dim embedding
        payload={
            "title": "PowerShell Execution Detection",
            "content": "Monitor for suspicious PowerShell activity...",
            "source": "mitre",
            "severity": "high",
            "created_at": "2024-05-31"
        }
    ),
    # ... more points
]

client.upsert(
    collection_name="security-knowledge",
    points=points
)

# Semantic search
results = client.search(
    collection_name="security-knowledge",
    query_vector=[0.21, 0.44, -0.11, ...],  # Query embedding
    limit=10,
    query_filter={
        "must": [
            {
                "key": "severity",
                "range": {"gte": "high"}  # Filter by severity
            }
        ]
    }
)

for result in results:
    print(f"Score: {result.score}, Title: {result.payload['title']}")
```

**Pros:**
- ✓ Fast semantic search (millions of vectors)
- ✓ Filtering + search combined (no post-filtering)
- ✓ Auto-replication and HA
- ✓ gRPC interface (fast)

**Cons:**
- ✗ Separate service to manage
- ✗ More memory overhead

### Weaviate (Distributed)

**Best for:** Multi-tenant SaaS, distributed teams, complex queries

```python
import weaviate

client = weaviate.Client("http://weaviate:8080")

# Define schema
security_class = {
    "class": "SecurityIncident",
    "description": "Security incidents and knowledge",
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Incident title"
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Incident details"
        },
        {
            "name": "severity",
            "dataType": ["string"],
            "enum": ["critical", "high", "medium", "low"]
        },
        {
            "name": "source",
            "dataType": ["string"]
        }
    ]
}

client.schema.create_class(security_class)

# Add data
client.batch.add_data_object(
    data_object={
        "title": "PowerShell Execution Detection",
        "content": "Monitor for suspicious PowerShell activity...",
        "severity": "high",
        "source": "mitre"
    },
    class_name="SecurityIncident"
)

# Semantic search with GraphQL
query = client.query.get("SecurityIncident").with_near_text({
    "concepts": ["PowerShell execution"],
    "distance": 0.7  # Similarity threshold
}).with_where({
    "path": ["severity"],
    "operator": "Equal",
    "valueString": "high"
}).with_limit(10).do()

print(query["data"]["Get"]["SecurityIncident"])
```

**Pros:**
- ✓ Multi-tenant architecture
- ✓ GraphQL API (powerful queries)
- ✓ Built-in authentication

**Cons:**
- ✗ More complex setup
- ✗ Overkill for single-tenant

### Comparison & Decision Tree

| Feature | pgvector | Qdrant | Weaviate |
|---------|----------|--------|----------|
| **Scale** | <10M docs | >10M docs | >100M docs |
| **Speed** | Medium | Fast | Fast |
| **Filtering** | SQL | Native | GraphQL |
| **Setup** | Simple | Medium | Complex |
| **Cost** | ✓ Low | Medium | High |
| **HA/Replication** | Postgres HA | Built-in | Built-in |

**For Presight:**
- **Start with Qdrant** (good balance of speed, features, simplicity)
- **Scale to multi-region with Weaviate** if needed later
- **Use pgvector** only if already running PostgreSQL cluster

---

## Part 3: Embedding Pipeline

### Concept: Converting Text to Vectors

```
"PowerShell execution detected"
            ↓
[Embedding Model]
            ↓
[0.234, -0.123, 0.456, ..., 0.089]  ← 384 numbers (384-dim vector)
            ↓
Store in Vector DB
            ↓
Search: Find other texts with similar vectors
```

### Embedding Models

```python
from sentence_transformers import SentenceTransformer

# Free, fast, good for security domain
model = SentenceTransformer('all-MiniLM-L6-v2')
# 384 dimensions, 22M parameters
# Speed: ~5000 docs/sec on single GPU
# Cost: Free (open-weight)

embeddings = model.encode([
    "PowerShell execution detected",
    "Script-based attack from Windows server",
    "Command execution via shell"
])

# embeddings.shape = (3, 384)
```

**Embedding Model Comparison:**

| Model | Dims | Size | Speed | Cost | Domain |
|-------|------|------|-------|------|--------|
| **all-MiniLM-L6-v2** | 384 | 22M | 5000 docs/s | Free | General |
| **all-mpnet-base-v2** | 768 | 110M | 1000 docs/s | Free | General (better) |
| **nomic-embed-text-v1** | 768 | 137M | 1500 docs/s | Free | Long docs |
| **SecurityBERT** | 768 | Custom | TBD | Free | Security |

**For Presight**: Use **all-MiniLM-L6-v2** (fast, small, good enough for security)

### Chunking Strategies

**Problem:** Documents are large. How to split?

```
Document: "PowerShell execution detected on server-01 at 2024-05-31 10:23 UTC.
           Forensics showed: command history, registry changes, DLL injection..."

Option 1: Fixed size (512 tokens)
Option 2: Semantic chunks
Option 3: Sliding window
Option 4: Document structure (sections)
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split by sentences, then paragraphs, then fixed size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,           # Max tokens per chunk
    chunk_overlap=50,         # Context overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""]  # Priority order
)

chunks = splitter.split_text(document)

# Each chunk:
# - Is ~512 tokens
# - Overlaps previous chunk by 50 tokens (context)
# - Respects sentence/paragraph boundaries
```

**For Security Knowledge:**
- **Chunk on logical boundaries** (procedures, techniques, remediations)
- **Min 100 tokens** (enough context)
- **Max 512 tokens** (fits in context window)
- **Overlap 10-20%** (maintain context between chunks)

### Batch Ingestion

```python
# presight/rag/ingest_pipeline.py

import asyncio
from typing import List
from datetime import datetime
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class RAGIngestionPipeline:
    def __init__(self, vector_db_url: str, embedding_model: str):
        self.vector_client = QdrantClient(vector_db_url)
        self.embedding_model = SentenceTransformer(embedding_model)
    
    async def ingest_mitre_attack(self) -> dict:
        """
        Daily ingestion of MITRE ATT&CK data
        1. Fetch latest MITRE techniques
        2. Chunk documents
        3. Generate embeddings
        4. Upsert to Qdrant
        """
        
        # Fetch from MITRE
        techniques = await self._fetch_mitre_techniques()
        
        # Process in batches
        batch_size = 100
        total_ingested = 0
        
        for i in range(0, len(techniques), batch_size):
            batch = techniques[i:i+batch_size]
            
            # Chunk
            chunks = []
            for technique in batch:
                chunks.extend(self._chunk_technique(technique))
            
            # Embed
            texts = [c['text'] for c in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            # Prepare points for Qdrant
            points = [
                PointStruct(
                    id=hash(chunk['id']),
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'],
                        'technique_id': chunk['technique_id'],
                        'severity': chunk['severity'],
                        'source': 'mitre',
                        'ingested_at': datetime.utcnow().isoformat()
                    }
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            # Upsert to vector DB
            self.vector_client.upsert(
                collection_name="security-knowledge",
                points=points
            )
            
            total_ingested += len(points)
            print(f"Ingested {total_ingested} chunks...")
        
        return {
            'status': 'success',
            'chunks_ingested': total_ingested,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _chunk_technique(self, technique: dict) -> List[dict]:
        """Split MITRE technique into semantic chunks"""
        
        chunks = []
        
        # Chunk 1: Technique overview
        chunks.append({
            'id': f"{technique['id']}_overview",
            'technique_id': technique['id'],
            'text': f"Technique: {technique['name']}. Description: {technique['description']}",
            'severity': technique.get('severity', 'medium')
        })
        
        # Chunk 2: Detection
        if technique.get('detection'):
            chunks.append({
                'id': f"{technique['id']}_detection",
                'technique_id': technique['id'],
                'text': f"Detection for {technique['name']}: {technique['detection']}",
                'severity': technique.get('severity', 'medium')
            })
        
        # Chunk 3: Mitigation
        if technique.get('mitigation'):
            chunks.append({
                'id': f"{technique['id']}_mitigation",
                'technique_id': technique['id'],
                'text': f"Mitigation for {technique['name']}: {technique['mitigation']}",
                'severity': technique.get('severity', 'medium')
            })
        
        return chunks
    
    async def _fetch_mitre_techniques(self) -> List[dict]:
        """Fetch latest MITRE ATT&CK techniques"""
        # Implementation would call MITRE API
        pass
```

---

## Part 4: Knowledge Corpora for Presight

### Security Knowledge Sources

```python
# Configuration for what to ingest
KNOWLEDGE_SOURCES = {
    'mitre_attack': {
        'url': 'https://raw.githubusercontent.com/mitre-attack/attack-website/master/docs/_data/matrix.json',
        'update_frequency': 'weekly',
        'priority': 'critical',
        'chunk_strategy': 'by_technique'
    },
    'cve_feed': {
        'url': 'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-modified.json',
        'update_frequency': 'daily',
        'priority': 'critical',
        'chunk_strategy': 'by_severity'
    },
    'custom_playbooks': {
        'source': 'S3://presight-playbooks/',
        'update_frequency': 'on_change',
        'priority': 'high',
        'chunk_strategy': 'by_section'
    },
    'incident_history': {
        'source': 'PostgreSQL incident_db',
        'update_frequency': 'daily',
        'priority': 'medium',
        'chunk_strategy': 'by_incident'
    }
}
```

### MITRE ATT&CK Knowledge Base

```python
# Indexed in RAG:
# - Techniques (attack methods)
# - Tactics (objectives)
# - Procedures (specific implementations)
# - Mitigations
# - Detection methods
# - Data sources

# Example query
def find_related_techniques(observed_behavior: str) -> List[dict]:
    """
    Given observed behavior, find related MITRE techniques
    
    Example: observed_behavior = "Suspicious PowerShell command history"
    Returns: [PowerShell execution technique, Command & Script Interpreter, ...]
    """
    
    # Embed the observation
    query_vector = embedding_model.encode(observed_behavior)
    
    # Search vector DB
    results = vector_db.search(
        query_vector=query_vector,
        collection="security-knowledge",
        filter={"source": "mitre"},
        top_k=5
    )
    
    return results
```

### Custom Security Playbooks

```python
# Example: Ransomware Response Playbook
PLAYBOOK_RANSOMWARE_RESPONSE = """
# Ransomware Incident Response Playbook

## Detection Phase
- Monitor for: FileEncryption, RegistryModification, ProcessExecution
- Alert on: Unusual file activity patterns, bulk encryption attempts

## Containment Phase
1. Isolate affected systems from network (immediately)
2. Disable lateral movement (block C2 servers)
3. Kill suspicious processes
4. Disable external backup access

## Eradication Phase
1. Identify ransomware family (using YARA rules)
2. Find all affected systems (scan network)
3. Remove malware (antivirus scan + manual cleanup)
4. Check for persistence mechanisms

## Recovery Phase
1. Restore from clean backups (verify no malware in backups!)
2. Rebuild affected systems from baseline
3. Implement hardening measures

## Follow-up
1. Post-mortem analysis (how did they get in?)
2. Update detection rules for this strain
3. Implement long-term mitigations
"""

# Chunked and indexed in RAG
# So when agent detects ransomware, it can retrieve relevant playbook sections
```

---

## Part 5: Long-Term Memory for Agents

### Concept: Why Agents Need Memory

**Without Memory:**
```
Incident #1: "Admin account compromised. Fixed by resetting password and enabling MFA."
Incident #2 (next week, same admin): "Admin account compromised again."
Agent: "Uh... let's reset password?"

No learning! Same mistake repeats.
```

**With Long-Term Memory:**
```
Agent: "This admin account was compromised before. Let me check the playbook..."
Agent: "I see—password reset alone didn't work. The issue was..."
Agent: Applies lessons from incident #1 to incident #2
```

### Conversation Memory

```python
# Per-session conversation history

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Agent remembers conversation
memory.chat_memory.add_user_message("User: Suspicious PowerShell detected")
memory.chat_memory.add_ai_message("Agent: Checking MITRE ATT&CK for PowerShell techniques...")

# On next message, agent knows context
history = memory.buffer
# "User: What should we do next?"
# Agent has context from previous messages
```

### Incident Memory

```python
# Long-term: Store incident learnings

class IncidentMemory:
    """Learn from past incidents"""
    
    def store_incident(self, incident_id: str, incident_data: dict):
        """
        Store incident learnings for future reference
        
        incident_data = {
            'description': 'Admin account compromised',
            'root_cause': 'Weak password',
            'detection_method': 'Failed login attempts',
            'remediation': 'Password reset + MFA',
            'time_to_detect': 300,  # seconds
            'time_to_respond': 120,
            'lessons_learned': 'MFA is critical for admin accounts'
        }
        """
        
        # Embed key learnings
        learning_text = f"{incident_data['description']}. Root cause: {incident_data['root_cause']}"
        embedding = embedding_model.encode(learning_text)
        
        # Store in vector DB
        vector_db.upsert({
            'vector': embedding,
            'payload': {
                'incident_id': incident_id,
                'type': 'incident_learning',
                'description': incident_data['description'],
                'root_cause': incident_data['root_cause'],
                'remediation': incident_data['remediation'],
                'timestamp': datetime.utcnow().isoformat()
            }
        })
    
    def retrieve_similar_incidents(self, current_incident: str) -> List[dict]:
        """Find past incidents similar to current one"""
        
        query_vector = embedding_model.encode(current_incident)
        results = vector_db.search(
            query_vector=query_vector,
            filter={"type": "incident_learning"},
            top_k=5
        )
        
        return results
```

### Learning from Feedback

```python
# Agent learns from human feedback

class AgentFeedbackLoop:
    """Improve agent with human feedback"""
    
    def record_feedback(self, response_id: str, feedback: dict):
        """
        feedback = {
            'quality': 'good'|'bad',
            'reason': 'Correctly identified attack pattern',
            'improvement': 'Could have mentioned related MITRE technique'
        }
        """
        
        # Store feedback
        self.db.insert('agent_feedback', {
            'response_id': response_id,
            'feedback': feedback,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Update agent weights/retrieval strategy
        if feedback['quality'] == 'bad':
            # Adjust similarity threshold
            # Or adjust retrieval filters
            # Or fine-tune embedding model
            pass
        
        if feedback['improvement']:
            # Add to training data for future improvement
            pass
```

---

## Part 6: RAG in Agent Loops

### Planner + RAG

```python
# Agent Planner Loop with RAG

class SecurityPlannerWithRAG:
    """Plan security response using RAG for historical context"""
    
    async def plan_response(self, incident: dict) -> dict:
        """
        1. Retrieve similar past incidents
        2. Use as inspiration for response plan
        3. Generate detailed steps
        """
        
        # RAG: Find similar incidents
        similar = await self.rag_retriever.find_similar_incidents(incident['description'])
        
        context = f"""
        Similar past incidents:
        {json.dumps(similar, indent=2)}
        
        Current incident: {incident['description']}
        """
        
        # Planner with context
        plan = await self.llm.generate(
            prompt=f"""
            {context}
            
            Based on similar past incidents, create a step-by-step response plan for the current incident.
            """,
            temperature=0.1  # Deterministic planning
        )
        
        return {
            'plan': plan,
            'retrieved_incidents': similar,
            'confidence': 0.95
        }
```

### Executor + RAG

```python
# Agent Executor executes plan, uses RAG for tool details

class ExecutorWithRAG:
    """Execute response plan using RAG for tool guidance"""
    
    async def execute_step(self, step: str, context: dict) -> dict:
        """
        1. Retrieve tool documentation from RAG
        2. Retrieve examples from playbooks
        3. Execute with confidence
        """
        
        # RAG: Tool documentation
        tool_docs = await self.rag_retriever.find_tool_documentation(step)
        
        # RAG: Examples
        examples = await self.rag_retriever.find_playbook_examples(step)
        
        context_text = f"""
        Tool documentation:
        {tool_docs}
        
        Example from playbook:
        {examples}
        """
        
        # Execute with guided context
        result = await self.tool_executor.execute(
            step=step,
            context=context_text,
            guidance=tool_docs
        )
        
        return result
```

### Critic + RAG

```python
# Agent Critic reviews response using RAG for safety checks

class CriticWithRAG:
    """Critique response using RAG for safety/compliance"""
    
    async def critique_response(self, response: dict) -> dict:
        """
        1. Retrieve safety guidelines from RAG
        2. Retrieve compliance requirements
        3. Check response against guidelines
        """
        
        # RAG: Safety guidelines
        safety_guidelines = await self.rag_retriever.find_safety_guidelines(response['type'])
        
        # RAG: Compliance requirements
        compliance = await self.rag_retriever.find_compliance_requirements(response['sensitivity'])
        
        context = f"""
        Safety guidelines:
        {safety_guidelines}
        
        Compliance requirements:
        {compliance}
        
        Response to review:
        {response['text']}
        """
        
        # Critique with context
        critique = await self.llm.generate(
            prompt=f"""
            {context}
            
            Does this response comply with safety guidelines and compliance requirements?
            List any violations.
            """,
            temperature=0.1
        )
        
        return {
            'critique': critique,
            'passes_safety': 'violation' not in critique.lower(),
            'guidelines_applied': safety_guidelines
        }
```

---

## Production: Evaluation & Safety

### RAG-Specific Evaluation Metrics

```python
# presight/evaluation/rag_metrics.py

class RAGEvaluationMetrics:
    """Measure RAG quality"""
    
    @staticmethod
    def measure_retrieval_relevance(query: str, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """
        Precision: % of retrieved docs that are relevant
        Recall: % of relevant docs that were retrieved
        """
        
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        precision = len(retrieved_set & relevant_set) / len(retrieved_set) if retrieved_set else 0
        recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
    
    @staticmethod
    def measure_groundedness(generated_text: str, retrieved_docs: List[str]) -> float:
        """
        Is the generated text based on retrieved docs?
        (Not hallucinating new facts)
        """
        
        # Check if generated claims are supported by retrieved docs
        # Using semantic similarity
        pass
    
    @staticmethod
    def measure_latency(retrieval_time_ms: float) -> dict:
        """Track retrieval speed"""
        
        return {
            'retrieval_latency_ms': retrieval_time_ms,
            'within_slo': retrieval_time_ms < 500,  # SLO: <500ms
            'percentile': 'P99'  # Track P50, P95, P99
        }
```

---

## Interview Questions

1. **Design a RAG pipeline for security knowledge. What are the components?**
   - Ingest (MITRE, CVE feeds, playbooks) → Chunk → Embed → Index
   - Retrieve (semantic search) → Augment (prompt building) → Generate (LLM) → Evaluate

2. **How would you choose between pgvector, Qdrant, and Weaviate?**
   - pgvector: <10M docs, integrated with PostgreSQL
   - Qdrant: 10M-100M docs, standalone, fast search
   - Weaviate: >100M docs, multi-tenant, distributed

3. **How do you ensure RAG doesn't hallucinate or exfiltrate data?**
   - Groundedness evaluation (is response based on retrieved docs?)
   - Content filtering (check retrieved docs for sensitive data)
   - Critic agent (review response against guidelines)

4. **How would you update the RAG knowledge base in production?**
   - Daily batch ingestion (MITRE, CVE feeds)
   - Real-time ingestion of incidents
   - Version control for playbooks

5. **How do you handle RAG latency in real-time agent responses?**
   - Cache popular queries
   - Use approximate search (trade accuracy for speed)
   - Parallel retrieval from multiple sources
