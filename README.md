# ⚖️ Legal RAG QA — Document Intelligence Platform

A production-grade, enterprise-ready **Retrieval-Augmented Generation (RAG)** platform engineered specifically for legal, compliance, human resources, and insurance document intelligence.

Legal RAG QA answers natural-language questions with **provable citations** (`[Source: filename, Page X]`) grounded strictly in authorized source documents, enforcing **zero tolerance for hallucinations**.

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
  - [High-Level Architecture](#high-level-architecture)
  - [Document Ingestion Pipeline](#document-ingestion-pipeline)
  - [Query & Answering Lifecycle](#query--answering-lifecycle)
- [In-Depth Design Decisions](#-in-depth-design-decisions)
  - [1. Legal-Aware Chunking & Deduplication](#1-legal-aware-chunking--deduplication)
  - [2. Hybrid Retrieval: PGVector + BM25 via RRF](#2-hybrid-retrieval-pgvector--bm25-via-rrf)
  - [3. Zero-Hallucination Guard & Citation Enforcement](#3-zero-hallucination-guard--citation-enforcement)
  - [4. Multi-Provider LLM Factory](#4-multi-provider-llm-factory)
  - [5. Clean API Contract & Schema Segregation](#5-clean-api-contract--schema-segregation)
  - [6. Resilient Database Dialect Normalization](#6-resilient-database-dialect-normalization)
  - [7. Native PostgreSQL GIN Full-Text vs. In-Memory BM25 at Scale](#7-native-postgresql-gin-full-text-vs-in-memory-bm25-at-scale)
  - [8. FlashRank Local Neural Cross-Encoder Reranking](#8-flashrank-local-neural-cross-encoder-reranking)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [1. Installation](#1-installation)
  - [2. Environment Configuration](#2-environment-configuration)
  - [3. Database Schema & Index Setup](#3-database-schema--index-setup)
  - [4. Ingest Sample Documents](#4-ingest-sample-documents)
  - [5. Run the Server](#5-run-the-server)
- [API Reference](#-api-reference)
  - [Endpoints](#endpoints)
  - [Example Requests & Responses](#example-requests--responses)
- [Postman Collection](#-postman-collection)
- [🚀 1M Scale Architecture (FOSS Stack)](#-1m-scale-architecture-foss-stack)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Docker Deployment](#-docker-deployment)
- [☁️ Render Deployment](#️-render-deployment)
- [🚀 AWS EC2 & Coolify Self-Hosted Deployment](#-aws-ec2--coolify-self-hosted-deployment)
  - [1. EC2 Instance & Firewall Prerequisites](#1-ec2-instance--firewall-prerequisites)
  - [2. Swap File Configuration (Critical for t2.micro)](#2-swap-file-configuration-critical-for-t2micro)
  - [3. Coolify Installation](#3-coolify-installation)
  - [4. Deploying Legal RAG QA via Coolify Dashboard](#4-deploying-legal-rag-qa-via-coolify-dashboard)
  - [5. Environment Variables Setup](#5-environment-variables-setup)
  - [6. Healthcheck & Domain Configuration](#6-healthcheck--domain-configuration)
  - [7. Verification & Production Testing](#7-verification--production-testing)
- [License](#-license)

---

## 🌟 Key Highlights

- **1M+ Document Scale Architecture**: Engineered to handle **1,000,000+ legal documents (~20,000,000 chunks)** using a 100% Free and Open-Source Software (FOSS) stack: PostgreSQL 16 + pgvector HNSW, MinIO S3 storage, Celery + Redis distributed workers, FlashRank neural reranking, Redis semantic cache, and Langfuse.
- **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Combines dense vector semantics (Neon PGVector HNSW) with database-native keyword matching (Postgres GIN `tsvector`) or BM25 to prevent legal term misses with zero RAM bloat.
- **Sub-10ms Redis Semantic Caching**: Caches query embeddings with cosine similarity matching (>0.96 threshold) to return cached answers instantly for recurring legal inquiries without incurring LLM inference costs.
- **Local CPU Cross-Encoder Reranking**: Integrates FlashRank (`ms-marco-MiniLM-L-12-v2`) via ONNX on CPU (<100MB RAM) for high-precision clause re-ranking with $0 API fees.
- **Zero Hallucination Tolerance**: Strict system prompts compel the model to refuse queries when context is insufficient rather than fabricate clauses, policies, or statutory numbers.
- **Auditable Citations**: Every answer includes traceable document references with page numbers and exact excerpt snippets.
- **Clean End-User API Contract**: Request bodies take only `{"question": "..."}`—backend infrastructure details (provider, model, internal tokens) are fully decoupled.
- **Multi-LLM Provider Support**: Pluggable support for Groq (Qwen / LLaMA), Google Gemini, OpenAI, Anthropic, and Mistral with lazy importing.
- **Serverless PostgreSQL + PGVector**: Cloud-native, scalable vector database hosted on Neon (or local Docker) with automated `postgresql+psycopg://` v3 driver dialect handling.
- **Streaming & REST**: Supports both standard synchronous JSON responses and real-time Server-Sent Events (SSE) token streaming.

---

## 🛠 Technology Stack

| Layer | Technologies | Purpose | Cost |
|---|---|---|---|
| **Language & Runtime** | **Python 3.12+**, **Astral `uv`** | High-performance Python runtime and lightning-fast package management | **$0** (FOSS) |
| **API Framework** | **FastAPI 0.115+**, **Uvicorn 0.34+** | Asynchronous web framework, OpenAPI 3.1 docs, CORS, and GZip compression | **$0** (FOSS) |
| **Configuration** | **Pydantic Settings v2.7+**, **Pydantic v2** | Strictly typed, environment-driven configuration with `.env` overrides | **$0** (FOSS) |
| **Vector Database** | **PostgreSQL 16**, **`pgvector`**, **`langchain-postgres` 0.0.13** | Serverless/containerized PostgreSQL vector store with HNSW indexing (`m=16, ef_construction=64`) | **$0** (FOSS) |
| **Database Driver** | **`psycopg` 3 (`psycopg[binary]>=3.2`)** | Modern async-compatible DBAPI driver with auto-dialect normalization | **$0** (FOSS) |
| **Dense Embeddings** | **Google Gemini Embeddings** (`gemini-embedding-001`, 768-dim) | High-fidelity dense semantic representations | Free Tier |
| **Full-Text Keyword Search** | **PostgreSQL GIN `tsvector`** + **`rank-bm25`** | Native database tsvector search (`to_tsvector`/`plainto_tsquery`) for 1M docs + BM25 fallback | **$0** (FOSS) |
| **Neural Reranker** | **FlashRank** (`ms-marco-MiniLM-L-12-v2`) | Ultra-fast local cross-encoder running via ONNX on CPU (<100MB RAM, <15ms latency) | **$0** (FOSS) |
| **Semantic Response Cache** | **Redis 7** (In-Memory Vector Cache) | Caches query embeddings with cosine similarity matching for <10ms responses | **$0** (FOSS) |
| **Async Task Queue** | **Celery 5.6+**, **Redis 7** | Distributed background worker queue for asynchronous parsing, chunking, and embedding | **$0** (FOSS) |
| **Raw Object Storage** | **MinIO S3**, **`boto3`** | S3-compatible distributed object storage for raw document blobs; local disk fallback | **$0** (FOSS) |
| **Observability & Tracing** | **Langfuse 2** (Self-Hosted) | Open-source LLM tracing, latency monitoring, token usage analytics, and prompt tracking | **$0** (FOSS) |
| **LLM Orchestration** | **`langchain-core` 1.6+** | Lightweight, modular runnables and prompt chaining (no `langchain-community` bloat) | **$0** (FOSS) |
| **LLM Providers** | **Groq**, **Google Gemini**, **OpenAI**, **Anthropic**, **Mistral** | Pluggable multi-model inference with lazy loading | Mixed / Free Tier |
| **Document Parsers** | **`pypdf` 5+**, **`python-docx` 1.1+** | Native extraction for PDF, DOCX, and raw legal TXT files | **$0** (FOSS) |
| **Testing & Quality** | **Pytest 8.4+**, **Pytest-Asyncio**, **HTTPX**, **Ruff 0.8+** | 40 automated unit/integration tests, sub-second linting, and formatting | **$0** (FOSS) |
| **Containerization** | **Docker Compose** | 6-service production stack (`api`, `worker`, `postgres`, `redis`, `minio`, `langfuse`) | **$0** (FOSS) |

---

## 🏗 System Architecture

### High-Level Architecture

```mermaid
flowchart TD
    User([Client / Postman / Frontend]) -->|HTTP REST / SSE| API[FastAPI Gateway /api/v1]
    
    subgraph "Application Layer (src/app)"
        API --> Endpoints["Endpoints: /health, /documents, /ask"]
        Endpoints --> RAG[RAGChain Engine]
        Endpoints --> AsyncIngest[POST /documents/upload-async]
        Endpoints --> SyncIngest[POST /documents/ingest]
        
        subgraph "1M-Scale Distributed Services"
            AsyncIngest -->|Push Blob| S3[(MinIO S3 Storage)]
            AsyncIngest -->|Dispatch Task| Celery[Celery Worker Queue]
            Celery -->|Redis Broker| Redis[(Redis 7)]
            Celery -->|Chunk & Embed| PGV[(PostgreSQL 16 + pgvector)]
        end

        subgraph "Semantic Cache Layer"
            RAG -->|Cosine Match > 0.96| Cache[(Redis Semantic Cache)]
            Cache -.->|Cache Hit <10ms| User
        end
        
        subgraph "Hybrid Retrieval & Reranking Subsystem"
            RAG -->|Cache Miss| Hybrid[HybridRetriever]
            Hybrid -->|Dense Semantic Query| PGV
            Hybrid -->|Native GIN tsvector Search| PGV
            Hybrid -->|RRF Fusion Algorithm| Candidates[Top 25 Candidates]
            Candidates --> FlashRank[FlashRank CPU Neural Reranker\nms-marco-MiniLM-L-12-v2]
            FlashRank --> RankedDocs[Top 5 Precision Chunks]
        end

        subgraph "Generation & Observability Subsystem"
            RankedDocs --> Guard{Hallucination Guard}
            Guard -->|Context Missing| Refusal["Refusal Response: 'I cannot find...'"]
            Guard -->|Context Found| Prompt[Legal Grounding Prompt Template]
            Prompt --> Factory[LLMFactory]
            Factory --> Provider[Groq / Gemini / OpenAI / Anthropic]
            Provider -.->|Trace Latency & Tokens| Langfuse[(Langfuse Observability)]
        end
    end

    Provider --> Response["Cited RAGResponse {answer, sources}"]
    Refusal --> Response
    Response --> User
```

---

### Document Ingestion Pipeline

The platform supports both synchronous immediate ingestion and **asynchronous distributed ingestion** for large legal volumes:

```mermaid
sequenceDiagram
    autonumber
    actor Admin as User / Frontend / Seed Script
    participant API as FastAPI Gateway
    participant MinIO as MinIO S3 Storage
    participant Celery as Celery Worker
    participant Parser as Loaders (PDF/DOCX/TXT)
    participant Chunker as Legal Chunker
    participant Embedder as Gemini Embeddings (768d)
    participant PG as PostgreSQL 16 (HNSW + GIN)

    alt Async Ingestion (1M Scale: POST /api/v1/documents/upload-async)
        Admin->>API: Upload contract (multipart/form-data)
        API->>MinIO: Store raw blob in S3 bucket (legal-documents)
        API->>Celery: queue ingest_document_task(file_key, filename)
        API-->>Admin: 202 Accepted {task_id, status: "PENDING"}
        Celery->>MinIO: Fetch document bytes
        Celery->>Parser: Parse text & page markers
        Celery->>Chunker: Split preserving legal clauses & structure
        Celery->>Embedder: Generate dense embeddings
        Celery->>PG: Bulk insert into langchain_pg_embedding (HNSW + GIN tsv)
        Admin->>API: GET /api/v1/documents/tasks/{task_id}
        API-->>Admin: 200 OK {status: "SUCCESS", chunks_created: 142}
    else Synchronous Ingestion (POST /api/v1/documents/ingest)
        Admin->>API: Upload file
        API->>Chunker: Parse & Chunk
        API->>Embedder: Embed Chunks
        API->>PG: Store in PGVector
        API-->>Admin: 201 Created {chunks_created: 10}
    end
```

---

### Query & Answering Lifecycle

When a user submits a natural-language legal question:

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Client App
    participant API as POST /api/v1/ask
    participant Cache as Redis Semantic Cache
    participant Retriever as HybridRetriever
    participant PG as PostgreSQL (HNSW + GIN)
    participant Reranker as FlashRank Reranker (CPU)
    participant Chain as RAGChain
    participant LLM as Active LLM (Groq / Gemini)
    participant Langfuse as Langfuse Tracing

    Client->>API: {"question": "What is the probation period?"}
    API->>Chain: ask(QuestionRequest)
    
    opt Semantic Cache Check
        Chain->>Cache: check_cache(question_embedding, threshold=0.96)
        Cache-->>Chain: Cache HIT (<10ms)
        Chain-->>Client: Cached RAGResponse with provable citations
    end

    opt Cache Miss
        par Hybrid Search (pgvector HNSW + GIN tsvector)
            Chain->>Retriever: retrieve(query, top_k=25)
            Retriever->>PG: Dense HNSW ANN cosine search
            Retriever->>PG: Full-text plainto_tsquery ts_rank_cd
        end
        PG-->>Retriever: Dense & Keyword Candidates
        Retriever->>Retriever: Reciprocal Rank Fusion (RRF) Merge
        Retriever->>Reranker: rerank(query, top_candidates=25)
        Reranker-->>Retriever: Top 5 Neural Ranked Chunks
        Retriever-->>Chain: Top 5 Relevant Clauses
        
        alt Insufficient Context
            Chain-->>Client: Refusal: "I cannot find sufficient information..."
        else Sufficient Legal Context
            Chain->>LLM: QA_PROMPT(context, question)
            LLM-->>Chain: Answer with [Source: filename, Page X]
            Chain->>Cache: set_cache(query, answer, sources)
            Chain->>Langfuse: log_trace(latency, tokens, prompt)
            Chain-->>API: Cited RAGResponse
        end
    end
    API-->>Client: 200 OK JSON Response
```

---

## 🧠 In-Depth Design Decisions

### 1. Legal-Aware Chunking & Deduplication
Legal contracts, insurance policies, and employee handbooks feature dense structural hierarchies (e.g., *Section 1.3*, *Clause 4(a)*, *Grade 1–3*). Standard character-count splitters break mid-sentence or tear clauses apart.
- **Regex Boundary Priority**: The [`DocumentChunker`](src/app/document_processing/chunker.py) uses a tiered hierarchy of separators: double newlines (`\n\n`), section breaks (`(?m)^(?=[A-Z0-9\.\s]{3,}:)`), numbered clauses (`(?m)^(?=\d+\.\d+)`), and sentence terminators.
- **SHA-256 Deduplication**: [`IngestionService`](src/app/document_processing/ingestion.py) hashes file contents to prevent re-embedding identical files, saving API tokens and database footprint.

### 2. Hybrid Retrieval: PGVector + BM25 via RRF
Pure vector search routinely fails on legal texts because:
1. Exact terms like "Section 1.3" or "Grade 7" lack semantic distinctiveness from "Section 1.4" in vector space.
2. BM25 catches exact keyword identifiers with high BM25 scores.
3. Vector search catches semantic paraphrasing (e.g., *"quitting my job"* matches *"resignation notice"*).

**Reciprocal Rank Fusion (RRF)**:
Instead of trying to normalize incommensurable distance metrics (cosine similarity vs. unbounded BM25 scores), RRF combines results based strictly on their **ordinal ranks**:

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where:
- $M = \{\text{vector}, \text{bm25}\}$
- $k = 60$ (standard smoothing constant to prevent top ranks from dominating)
- $w_{\text{vector}} = 0.7$, $w_{\text{bm25}} = 0.3$ (configurable in `.env`)

### 3. Zero-Hallucination Guard & Citation Enforcement
The system prompt in [`src/app/llm/prompts.py`](src/app/llm/prompts.py) explicitly restricts the model:
```text
You are a precise, helpful legal assistant. Answer the user's question
based STRICTLY on the provided context below.

Rules:
1. Answer ONLY using the facts directly stated in the context.
2. If the context does not contain the answer, say:
   "I cannot find sufficient information in the provided documents to answer this question."
3. Every factual claim MUST be followed by its citation: [Source: <filename>, Page <X>]
```
If retrieval returns zero documents above the relevance threshold, the generation step is bypassed entirely, instantly returning a factual refusal without incurring LLM cost.

### 4. Multi-Provider LLM Factory
The application is not tied to a single AI vendor. The [`LLMFactory`](src/app/llm/factory.py) uses lazy imports so dependencies are only loaded when requested:
- **Groq**: Ultra-low latency inference using models like `qwen/qwen3.8-27b` and `openai/gpt-oss-120b`.
- **Google Gemini**: Large-context reasoning using `gemini-2.5-flash` or `gemini-2.5-pro`.
- **OpenAI**: GPT-4o / GPT-4o-mini.
- **Anthropic**: Claude 3.5 Sonnet / Claude 3 Haiku.
- **Mistral**: Mistral Large / Mistral Small.

### 5. Clean API Contract & Schema Segregation
Following enterprise FastAPI standards, all Pydantic models are decoupled from internal pipelines into [`src/app/schemas/`](src/app/schemas/):
- **Request Decoupling**: End-users do not send `provider`, `model`, or internal `top_k` hyperparameters. The request payload is purely:
  ```json
  { "question": "What is the notice period for junior staff?" }
  ```
- **Response Privacy**: Internal telemetry (`latency_ms`, `provider`, `model`) is omitted from client responses to avoid infrastructure leakage. Metrics are recorded in server logs and APM headers.

### 6. Resilient Database Dialect Normalization
Standard PostgreSQL URLs (from Neon, Supabase, AWS RDS) begin with `postgresql://` or `postgres://`. In SQLAlchemy, these schemes default to the legacy `psycopg2` driver. Because this project runs on modern `psycopg` v3, [`src/app/config/settings.py`](src/app/config/settings.py) includes an automated pre-validator:
```python
@field_validator("database_url", mode="before")
@classmethod
def normalize_database_url(cls, value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value[len("postgresql://"):]
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value[len("postgres://"):]
    return value
```
Users can paste standard connection strings into `.env` without encountering `ModuleNotFoundError: No module named 'psycopg2'`.

### 7. Native PostgreSQL GIN Full-Text vs. In-Memory BM25 at Scale
At a few hundred documents, an in-memory Okapi BM25 index (built using `rank-bm25`) is fast and requires minimal overhead. However, at **1,000,000 documents (~20,000,000 chunks)**:
- Storing inverted index Python dictionaries in application RAM demands **8–16 GB of memory**, leading to Out-Of-Memory (OOM) crashes and preventing efficient horizontal scaling across multiple stateless API worker instances.
- Rebuilding the in-memory index on server restarts or new document uploads becomes prohibitively slow.

**The Solution:**
[`src/app/retrieval/postgres_fulltext.py`](src/app/retrieval/postgres_fulltext.py) implements `PostgresFullTextRetriever`, delegating keyword search directly to PostgreSQL's native `tsvector` engine with a GIN index:
```sql
SELECT document, cmetadata,
       ts_rank_cd(to_tsvector('english', document), plainto_tsquery('english', :query)) AS rank
FROM langchain_pg_embedding
WHERE collection_id = :collection_id
  AND to_tsvector('english', document) @@ plainto_tsquery('english', :query)
ORDER BY rank DESC
LIMIT :limit;
```
- **0 MB Python RAM Footprint**: The database handles lexeme parsing, stemming, stop-word elimination, and inverted index lookups on disk.
- **Immediate Indexing**: As soon as a document chunk is inserted into PostgreSQL, the generated `tsvector` column is stored and indexed in the GIN tree instantly.

### 8. FlashRank Local Neural Cross-Encoder Reranking
Standard bi-encoder embeddings project queries and chunks into independent vectors and measure cosine similarity. While fast, bi-encoders lose subtle cross-attention details between query constraints (e.g. *"after probation period but before 1 year"*) and complex legal exceptions.

- Hosted cross-encoder APIs (e.g. Cohere Rerank) cost $1.00–$2.00 per 1,000 searches and add 100–300ms network latency.
- [`src/app/retrieval/reranker.py`](src/app/retrieval/reranker.py) implements `FlashRankReranker`, running the `ms-marco-MiniLM-L-12-v2` cross-encoder locally on CPU via ONNX Runtime.
- Evaluates the top 25 candidates from Hybrid RRF and ranks the top 5 highest-confidence clauses in **<15ms**, with **<100MB RAM**, requiring **$0 external API spend**.

---

## 📁 Project Directory Structure

```text
legal-rag-qa/
├── data/
│   ├── samples/                    # Sample legal & HR policies (employee_handbook.txt, etc.)
│   └── uploads/                    # Uploaded documents directory (local dev fallback)
├── scripts/
│   └── seed.py                     # CLI ingestion utility for initial data loading
├── src/
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory (create_app), lifespan, & middlewares
│   │   ├── api/
│   │   │   ├── deps.py             # Dependency injection providers (settings, vector_store, rag)
│   │   │   └── api_v1/
│   │   │       ├── api.py          # /api/v1 Router aggregator
│   │   │       └── endpoints/
│   │   │           ├── health.py   # Liveness & database readiness probes
│   │   │           ├── documents.py# Sync/async upload, tasks, listing, & deletion
│   │   │           └── question.py # Sync (/ask) & streaming (/ask/stream) QA endpoints
│   │   ├── cache/
│   │   │   ├── __init__.py         # Cache service exports
│   │   │   └── semantic_cache.py   # Redis semantic vector cache (>0.96 cosine threshold)
│   │   ├── config/
│   │   │   ├── __init__.py         # Config exports
│   │   │   └── settings.py         # Pydantic v2 Settings (DEV/PROD/LOCAL, CORS, URLs, FOSS flags)
│   │   ├── document_processing/
│   │   │   ├── loaders.py          # PDF, DOCX, and TXT loaders
│   │   │   ├── chunker.py          # Legal structure-preserving text chunker
│   │   │   └── ingestion.py        # Pipeline orchestrator with SHA-256 deduplication
│   │   ├── llm/
│   │   │   ├── factory.py          # Multi-LLM provider instantiation (Groq, Gemini, OpenAI, etc.)
│   │   │   ├── observability.py    # Langfuse callback tracing handler
│   │   │   └── prompts.py          # Grounding prompt templates with strict citation rules
│   │   ├── rag/
│   │   │   ├── __init__.py         # RAG package exports
│   │   │   └── chain.py            # Hybrid retrieval, cache check, hallucination guard & generation
│   │   ├── retrieval/
│   │   │   ├── vector_store.py     # Neon / PostgreSQL PGVector manager (HNSW indexing)
│   │   │   ├── postgres_fulltext.py# PostgreSQL GIN tsvector full-text search (1M docs, 0 RAM)
│   │   │   ├── bm25_retriever.py   # In-memory Okapi BM25 keyword search service (fallback)
│   │   │   ├── reranker.py         # FlashRank CPU neural cross-encoder (ms-marco-MiniLM-L-12-v2)
│   │   │   └── hybrid.py           # Reciprocal Rank Fusion (RRF) + FlashRank rank merger
│   │   ├── schemas/
│   │   │   ├── __init__.py         # Centralized schema exports
│   │   │   ├── question.py         # QuestionRequest, RAGResponse, SourceReference
│   │   │   ├── document.py         # IngestionResponse, DocumentInfo, DocumentListResponse
│   │   │   └── health.py           # HealthResponse
│   │   ├── storage/
│   │   │   ├── __init__.py         # Storage service exports
│   │   │   └── s3_storage.py       # MinIO / S3 object storage service with local fallback
│   │   └── tasks/
│   │       ├── __init__.py         # Task exports
│   │       ├── celery_app.py       # Celery distributed worker application & config
│   │       └── ingestion_tasks.py  # Asynchronous background document parsing & embedding tasks
│   └── tests/
│       ├── conftest.py             # Shared fixtures and mock configurations
│       ├── test_api.py             # FastAPI TestClient endpoint integration tests
│       ├── test_config.py          # Settings, env parsing, and dialect normalization tests
│       ├── test_document_processing.py # Chunker, loader, and ingestion tests
│       ├── test_rag_chain.py       # Chain execution, citation, and refusal tests
│       ├── test_retrieval.py       # BM25 and Hybrid RRF fusion tests
│       └── test_scalability.py     # MinIO, Celery, Postgres GIN, FlashRank, & Redis cache tests
├── .env.example                    # Environment template with secure defaults
├── data.sql                        # PostgreSQL schema, HNSW & GIN index initialization script
├── Dockerfile                      # Multi-stage container definition
├── docker-compose.yml              # 6-service compose stack (api, worker, postgres, redis, minio, langfuse)
├── postman.json                    # Postman Collection v2.1.0 with 12 pre-configured requests
├── pyproject.toml                  # UV dependencies, scripts, build-system, & tool configs
└── start.sh                        # Production/Development executable startup script
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **[UV](https://docs.astral.sh/uv/)** package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A **Neon PostgreSQL** database with the `pgvector` extension enabled.
- An API key for at least one LLM provider (e.g. Groq, Google Gemini, or OpenAI).

### 1. Installation

```bash
git clone <repo-url>
cd legal-rag-qa

# Install all dependencies and development tools
uv sync --dev
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and configure your credentials:

```bash
cp .env.example .env
```

Key variables in `.env`:
```dotenv
ENV=DEV
DEBUG=true
LOG_LEVEL=INFO

# Neon PostgreSQL + pgvector connection string
DATABASE_URL=postgresql://user:password@ep-xyz.us-east-2.aws.neon.tech/vector_db?sslmode=require

# Default LLM Provider & Model
DEFAULT_LLM_PROVIDER=groq
DEFAULT_LLM_MODEL=qwen/qwen3.8-27b
LLM_MAX_TOKENS=1000

# API Keys (set the ones you use)
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...

# Embedding Configuration
EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=models/gemini-embedding-001
EMBEDDING_DIMENSIONS=768
```

### 3. Database Schema & Index Setup

Run the included [`data.sql`](data.sql) script to initialize extensions (`vector`, `uuid-ossp`), tables, HNSW index, and GIN full-text index in your PostgreSQL database:

```bash
psql "$DATABASE_URL" -f data.sql
```

### 4. Ingest Sample Documents

Populate your vector database with the sample employee handbook and insurance policy:

```bash
uv run python scripts/seed.py
```
*Output:*
```text
📂  Found 2 sample file(s) in .../data/samples
🔌  Connecting to vector store...
📄  Ingesting: employee_handbook.txt
    Status: success | Chunks: 10
📄  Ingesting: insurance_policy.txt
    Status: success | Chunks: 11
✅  Seeding complete! Total chunks ingested: 21
```

### 5. Run the Server

```bash
# Standard startup using ./start.sh
./start.sh

# Or start with auto-reload on a custom port
./start.sh --reload --port 8000
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness check (version, environment, active providers) |
| `GET` | `/api/v1/health/ready` | Readiness probe (verifies Neon database connectivity) |
| `POST` | `/api/v1/documents/ingest` | Upload and ingest a file synchronously (`multipart/form-data`) |
| `POST` | `/api/v1/documents/upload-async` | Upload file to S3/MinIO and queue async background ingestion (`202 Accepted`) |
| `GET` | `/api/v1/documents/tasks/{task_id}` | Query Celery ingestion task status and progress |
| `GET` | `/api/v1/documents` | List all ingested documents and their chunk counts |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its chunks by ID |
| `POST` | `/api/v1/ask` | Submit a question and receive a cited answer (JSON) |
| `POST` | `/api/v1/ask/stream` | Ask a question with Server-Sent Events (SSE) token streaming |
| `GET` | `/docs` | Interactive Swagger UI documentation |
| `GET` | `/openapi.json` | Full OpenAPI 3.1 schema specification |

---

### Example Requests & Responses

#### 1. Ask a Legal Question (`POST /api/v1/ask`)

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period for junior staff?"}'
```

**Response (`200 OK`):**
```json
{
  "answer": "Junior staff (Grade 1–3) must provide thirty (30) calendar days of written notice when resigning. [Source: employee_handbook.txt, Page 1]",
  "sources": [
    {
      "document": "employee_handbook.txt",
      "page": 0,
      "chunk_text": "1.3 Notice Period\nEmployees wishing to resign must provide written notice as follows:\n  - Junior staff (Grade 1–3): Thirty (30) calendar days\n  - Mid-level staff (Grade 4–6): Sixty (60) calendar days...",
      "relevance_score": 0.0164
    }
  ],
  "sufficient_context": true
}
```

---

#### 2. Hallucination Refusal (`POST /api/v1/ask`)

**Request (Out-of-domain query):**
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

**Response (`200 OK`):**
```json
{
  "answer": "I cannot find sufficient information in the provided documents to answer this question.",
  "sources": [],
  "sufficient_context": false
}
```

---

#### 3. Upload Document (`POST /api/v1/documents/ingest`)

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@data/samples/employee_handbook.txt"
```

**Response (`201 Created`):**
```json
{
  "document_id": "05a118c0c5be396f2bef68a6b32abda89c0e0132526e901a5f48dd7b8f62e286",
  "filename": "employee_handbook.txt",
  "chunks_created": 10,
  "status": "success",
  "message": "Ingested in 3450 ms."
}
```

---

#### 4. Upload Document Asynchronously (`POST /api/v1/documents/upload-async`)

For large multi-hundred-page contracts and high-volume pipelines, upload directly to MinIO S3 and queue Celery background workers:

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload-async \
  -F "file=@data/samples/insurance_policy.txt"
```

**Response (`202 Accepted`):**
```json
{
  "task_id": "9b12a84d-2e91-4db4-a957-3f309ff935e4",
  "filename": "insurance_policy.txt",
  "status": "PENDING",
  "message": "File uploaded to storage and queued for background ingestion."
}
```

---

#### 5. Check Ingestion Task Status (`GET /api/v1/documents/tasks/{task_id}`)

Poll or query background Celery ingestion task progress and chunk statistics:

**Request:**
```bash
curl -X GET http://localhost:8000/api/v1/documents/tasks/9b12a84d-2e91-4db4-a957-3f309ff935e4
```

**Response (`200 OK`):**
```json
{
  "task_id": "9b12a84d-2e91-4db4-a957-3f309ff935e4",
  "status": "SUCCESS",
  "filename": "insurance_policy.txt",
  "chunks_created": 11,
  "error": null
}
```

---

## 📮 Postman Collection

The project includes a ready-to-import Postman collection at the repository root: [`postman.json`](postman.json).

### How to Import:
1. Open **Postman**.
2. Click **Import** (top left).
3. Select [`postman.json`](postman.json).
4. The collection will import with **12 pre-configured requests** grouped under:
   - `Health` (Health Check, Readiness Probe)
   - `Documents` (List Documents, Ingest File Sync, Upload Async Celery, Check Task Status, Delete File)
   - `Question Answering (RAG)` (Ask Question, Probation Rules, SSE Streaming, Refusal Test)
   - `System & OpenAPI` (OpenAPI Schema)
5. Start your server with `./start.sh` or `docker compose up -d` and test immediately!

---

## 🚀 1M Scale Architecture (FOSS Stack)

To scale Legal RAG QA from hundreds of documents to **1,000,000+ legal documents (~20,000,000 chunks)** without high SaaS costs, the application integrates six battle-tested **Free and Open-Source Software (FOSS)** technologies:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 FastAPI Gateway (/api/v1)              │
                  └────────────┬─────────────────────────────┬──────────────┘
                               │                             │
                     Async Ingestion                   Query / Ask
                               │                             │
                  ┌────────────▼────────────┐       ┌────────▼────────┐
                  │  MinIO (S3 Compatible)  │       │  Redis Semantic │
                  │     Raw Document Blob   │       │   Vector Cache  │
                  └────────────┬────────────┘       └────────┬────────┘
                               │                             │ (Cache Miss)
                  ┌────────────▼────────────┐                │
                  │   Celery Worker Queue   │       ┌────────▼───────────────────────┐
                  │    (Redis Broker)       │       │       Hybrid Retrieval         │
                  └────────────┬────────────┘       │                                │
                               │ Chunk & Embed      │  1. PGVector HNSW (Semantic)   │
                               │                    │  2. PostgreSQL GIN (tsvector)  │
                  ┌────────────▼────────────┐       └────────┬───────────────────────┘
                  │ PostgreSQL 16 + pgvector│                │ Top 25 Candidates
                  │  • HNSW Vector Index    │                ▼
                  │  • Native GIN tsvector  │       ┌────────────────────────────────┐
                  └─────────────────────────┘       │ FlashRank CPU Neural Reranker  │
                                                    │   (ms-marco-MiniLM-L-12-v2)    │
                                                    └────────┬───────────────────────┘
                                                             │ Top 5 Reranked Chunks
                                                             ▼
                                                    ┌────────────────────────────────┐
                                                    │ LLM Response + Langfuse Traces │
                                                    └────────────────────────────────┘
```

### Key Scale Components:

| Component | Technology | Role in 1M Document Scale | Cost |
|---|---|---|---|
| **Raw Storage** | **MinIO** | S3-compatible, distributed object store. Keeps multi-gigabyte raw PDFs/DOCX out of the vector database. Local filesystem fallback for dev. | **$0** (FOSS) |
| **Worker Queue** | **Celery + Redis** | Offloads document parsing, token chunking, and embedding generation to asynchronous background workers. Prevents API gateway timeouts. | **$0** (FOSS) |
| **Vector Search** | **PostgreSQL 16 + pgvector** | HNSW index (`m=16, ef_construction=64`) for sub-50ms approximate nearest neighbor search over 20M+ rows. | **$0** (FOSS) |
| **Keyword Search** | **PostgreSQL GIN `tsvector`** | Replaces RAM-heavy in-memory BM25 with native database-backed full-text search (`to_tsvector` / `plainto_tsquery`). Zero RAM footprint. | **$0** (FOSS) |
| **Neural Reranker** | **FlashRank** | Ultra-fast local cross-encoder (`ms-marco-MiniLM-L-12-v2`) running via ONNX on CPU (<100MB RAM). Re-ranks top 25 candidates without external API fees. | **$0** (FOSS) |
| **Response Cache** | **Redis Semantic Cache** | Computes cosine similarity of query embeddings against cached questions. Returns cached legal answers in <10ms for repeated queries (threshold > 0.96). | **$0** (FOSS) |
| **Observability** | **Langfuse** | Self-hosted LLM tracing, latency monitoring, token usage analytics, and prompt tracking. | **$0** (FOSS) |

### Running the Full Scale Stack:

```bash
# 1. Start all 6 services with Docker Compose
docker compose up -d

# 2. Or run Celery worker locally alongside ./start.sh:
uv run celery -A app.tasks.celery_app.celery_app worker --loglevel=info --concurrency=4
```

### Graceful Degradation:
The application is designed to be fully backward-compatible:
- If MinIO is disabled or unreachable, files are automatically stored on the local disk (`data/uploads`).
- If Redis is offline, semantic caching and Celery degrade gracefully without crashing queries or sync uploads.
- If FlashRank is disabled, standard Reciprocal Rank Fusion (RRF) is used.
- If PostgreSQL GIN is not configured, in-memory BM25 acts as fallback.

---

## 🧪 Testing & Quality Assurance

The test suite runs against real and mock configurations using `pytest` and `pytest-asyncio`:

```bash
# Run all unit and integration tests
uv run pytest

# Run with test coverage summary
uv run pytest --cov=app --cov-report=term-missing

# Run code style and lint checks with Ruff
uv run ruff check src scripts

# Format source code
uv run ruff format src scripts
```

### Test Coverage Summary (40 Automated Tests):
- **`test_api.py`** — Validates all endpoint routes, sync ingestion, async upload queuing, validation errors, and health checks.
- **`test_config.py`** — Verifies Pydantic v2 settings, environment overrides, CORS parsing, and database URL dialect auto-normalization.
- **`test_document_processing.py`** — Checks file loaders, legal chunking logic, and SHA-256 deduplication hashing.
- **`test_rag_chain.py`** — Asserts hallucination refusal, grounding prompt construction, and citation formatting.
- **`test_retrieval.py`** — Verifies BM25 index rebuilding and Reciprocal Rank Fusion (RRF) re-ranking.
- **`test_scalability.py`** — Validates MinIO S3 upload & fallback, Celery task queue dispatch, PostgreSQL native tsvector full-text search, FlashRank neural cross-encoder reranking, and Redis semantic cache cosine matching.

---

## 🐳 Docker Deployment

The project provides a multi-container **Docker Compose** stack running all 6 production services:

```bash
# Build and start all services in the background
docker compose up -d

# View live container logs
docker compose logs -f api worker

# Check service health
docker compose ps
```

### Stack Service Map:

| Container / Service | Image | Local Port | Access / Credentials |
|---|---|---|---|
| **FastAPI Gateway (`api`)** | `legal-rag-qa-api` | `http://localhost:8000` | Interactive docs at [`/docs`](http://localhost:8000/docs) |
| **Celery Worker (`worker`)** | `legal-rag-qa-worker` | Internal | Background task execution (concurrency=2) |
| **PostgreSQL 16 (`postgres`)** | `pgvector/pgvector:pg16` | `localhost:5432` | User: `postgres`, DB: `legal_rag`, pgvector enabled |
| **Redis Broker (`redis`)** | `redis:7-alpine` | `localhost:6379` | Celery broker + Vector Semantic Cache |
| **MinIO S3 (`minio`)** | `minio/minio:latest` | `http://localhost:9001` (UI)<br>`localhost:9000` (API) | User: `${S3_ACCESS_KEY}` (`minioadmin`)<br>Pass: `${S3_SECRET_KEY}` |
| **Langfuse Dashboard (`langfuse`)** | `ghcr.io/langfuse/langfuse:2` | `http://localhost:3000` | Web UI for LLM traces, latency, and prompt tracking |

### Container Security:
- Non-root user (`appuser`, UID 10001) in application containers.
- Automatic Docker entrypoint seeding via [`data.sql`](data.sql).
- Automated healthchecks across API, PostgreSQL, and Redis.

---

## ☁️ Render Deployment

The repository includes a production-ready Render Blueprint specification ([`render.yaml`](render.yaml)) supporting both automated Infrastructure-as-Code (IaC) and manual Docker deployments.

### Method 1: Automated Blueprint Deployment (Recommended)

1. Navigate to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** (top right) ➔ **Blueprint**.
3. Connect your GitHub repository: `https://github.com/PPC2001/legal-rag-intelligence`.
4. Render will parse [`render.yaml`](render.yaml) and automatically configure:
   - **`legal-rag-api`** — FastAPI Docker Web Service (health check: `/api/v1/health`)
   - **`legal-rag-worker`** — Celery Background Ingestion Worker
   - **`legal-rag-redis`** — Free Managed Redis instance (used for semantic caching and task queue)
5. Fill in the required secrets under **Environment Variables**:
   - `DATABASE_URL`: Your Neon PostgreSQL connection string (`postgresql://user:pass@ep-...neon.tech/vector_db?sslmode=require`)
   - `GROQ_API_KEY`: Your Groq API key (`gsk_...`)
   - `GOOGLE_API_KEY`: Your Google Gemini API key (`AIza...`)
6. Click **Apply**. Render will build the container and deploy your live public URL (e.g. `https://legal-rag-api.onrender.com`).

---

### Method 2: Manual Web Service Deployment

If you want to deploy just the standalone API web service:

1. Click **New +** ➔ **Web Service**.
2. Select your repository `legal-rag-intelligence`.
3. Configure the runtime settings:
   - **Name**: `legal-rag-api`
   - **Region**: Oregon (US West) or Frankfurt (EU)
   - **Branch**: `master`
   - **Runtime**: **Docker**
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.`
4. In **Advanced Settings**:
   - **Health Check Path**: `/api/v1/health`
5. Under **Environment Variables**, add:
   ```env
   ENV=PROD
   DEBUG=false
   LOG_LEVEL=INFO
   CORS_ORIGINS=["*"]
   DATABASE_URL=postgresql://user:password@ep-...neon.tech/vector_db?sslmode=require
   DEFAULT_LLM_PROVIDER=groq
   DEFAULT_LLM_MODEL=qwen/qwen3.8-27b
   GROQ_API_KEY=gsk_...
   GOOGLE_API_KEY=AIza...
   EMBEDDING_PROVIDER=google
   EMBEDDING_MODEL=models/gemini-embedding-001
   EMBEDDING_DIMENSIONS=768
   SPARSE_SEARCH_BACKEND=postgres
   ENABLE_RERANKER=true
   ENABLE_SEMANTIC_CACHE=false
   ```
   *(Set `ENABLE_SEMANTIC_CACHE=true` if you attach a Render Redis instance).*
6. Click **Create Web Service**.
7. Once deployed, open your live API docs at `https://<your-render-app>.onrender.com/docs`.

---

## 🚀 AWS EC2 & Coolify Self-Hosted Deployment

This guide covers deploying the entire Legal RAG QA platform on a **100% Free-Tier AWS EC2 instance (`t2.micro`)** managed with **Coolify** (open-source PaaS) connected to your serverless **Neon PostgreSQL (pgvector)** database.

```mermaid
flowchart TD
    Client([Client / Browser / Postman]) -->|Port 80 / 443| Traefik[Coolify Traefik Proxy]
    Traefik -->|Internal Route: 8000| API[FastAPI Web Service]
    API -->|Port 6379| Redis[(Redis 7 Cache & Broker)]
    API -->|Port 9000| MinIO[(MinIO S3 Blob Storage)]
    Worker[Celery Ingestion Worker] -->|Poll Queue| Redis
    Worker -->|Store Chunks| Neon[(Neon Serverless pgvector)]
    API -->|Query Vectors| Neon
```

---

### 1. EC2 Instance & Firewall Prerequisites

1. Launch an AWS EC2 instance with the following specifications:
   * **AMI**: Ubuntu Server 24.04 LTS (x86_64)
   * **Instance Type**: `t2.micro` (AWS Free Tier eligible: 1 vCPU, 1 GB RAM)
   * **Storage**: 30 GiB gp3 (maximum allowed within AWS Free Tier)
   * **Key Pair**: Download your `.pem` key (e.g., `coolify-key.pem`) and restrict permissions:
     ```bash
     chmod 400 coolify-key.pem
     ```

2. Configure your EC2 **Security Group** with the following **Inbound Rules**:

| Type | Port Range | Protocol | Source | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | `22` | TCP | `My IP` (or `0.0.0.0/0`) | Secure terminal access |
| **HTTP** | `80` | TCP | `0.0.0.0/0` | Coolify Traefik reverse proxy & SSL challenges |
| **HTTPS** | `443` | TCP | `0.0.0.0/0` | Secure public API traffic (Let's Encrypt SSL) |
| **Custom TCP** | `8000` | TCP | `0.0.0.0/0` | Coolify Management Web Dashboard |
| **Custom TCP** | `8001` | TCP | `0.0.0.0/0` | Direct FastAPI host port access (optional) |

---

### 2. Swap File Configuration (Critical for t2.micro)

> [!IMPORTANT]
> The `t2.micro` instance has only 1 GB of physical RAM. Running Docker builds and multi-container workloads without swap will trigger Linux Out-Of-Memory (OOM) kills. A **4 GB swap file** provides the necessary memory buffer for smooth builds and zero crashes.

SSH into your EC2 server and configure swap:

```bash
ssh -i coolify-key.pem ubuntu@<YOUR_EC2_PUBLIC_IP>

# Allocate 4GB swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Persist swap across server reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active (Swap total should report 4.0Gi)
free -h
```

---

### 3. Coolify Installation

Install Coolify with the official automated installation script:

```bash
curl -fsSL https://cdn.coollabs.io/coolify/install.sh | bash
```

Once installation finishes (approx. 2 minutes), you will see:
```text
Coolify is ready!
You can access Coolify through your Public IP: http://<YOUR_EC2_PUBLIC_IP>:8000
```

1. Open your browser and navigate to `http://<YOUR_EC2_PUBLIC_IP>:8000`.
2. Register your administrative account (Name, Email, Password).
3. On the onboarding screen, select **"This machine"** (localhost) as the target deployment server.

---

### 4. Deploying Legal RAG QA via Coolify Dashboard

1. In Coolify, navigate to **Projects** (left sidebar) ➔ Click **`default`** ➔ Click **`production`**.
2. Click **`+ Add Resource`** (or **`+ New`**).
3. Select **Public Repository** (or GitHub).
4. Fill in your repository details:
   * **Repository URL**: `https://github.com/PPC2001/legal-rag-intelligence.git`
   * **Branch**: `master`
   * **Build Pack**: **`Docker Compose`**
5. Click **Check repository** / **Continue**. Coolify will detect [`docker-compose.yaml`](docker-compose.yaml).

---

### 5. Environment Variables Setup

In your application's settings in Coolify, open the **Environment Variables** tab and add the following variables:

| Variable | Recommended Production Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://neondb_owner:...@ep-...neon.tech/vector_db?sslmode=require&channel_binding=require` | Your Neon serverless PostgreSQL connection string |
| `DEFAULT_LLM_PROVIDER` | `groq` | Primary LLM provider (`groq`, `gemini`, `openai`) |
| `DEFAULT_LLM_MODEL` | `qwen/qwen3.8-27b` *(or `llama-3.3-70b-versatile`)* | Primary generation model name |
| `GROQ_API_KEY` | `gsk_...` | Groq API Key |
| `GOOGLE_API_KEY` | `AIza...` | Google Gemini API Key (used for embeddings) |
| `EMBEDDING_PROVIDER` | `google` | Embedding engine |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | 768-dimensional dense embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Matches Neon vector table dimensions |
| `SPARSE_SEARCH_BACKEND` | `postgres` | Native GIN tsvector keyword search (0 MB Python RAM) |
| `ENABLE_RERANKER` | `true` | FlashRank neural cross-encoder reranking on CPU |
| `ENABLE_SEMANTIC_CACHE`| `true` | Redis cosine similarity cache (<10ms repeat responses) |
| `S3_ACCESS_KEY` | `minioadmin` | MinIO object storage root access key |
| `S3_SECRET_KEY` | `minioadmin123` | MinIO object storage root secret key |
| `S3_BUCKET_NAME` | `legal-documents` | Default bucket name for document uploads |

Click **Save**, then click the **Deploy** button in the top-right corner.

---

### 6. Healthcheck & Domain Configuration

#### A. Turn Coolify Healthcheck Indicator Green:
1. In your application dashboard, click the **Healthcheck** tab.
2. Toggle **Enabled** to **ON**.
3. Set the parameters:
   * **Path**: `/api/v1/health`
   * **Port**: `8000` *(internal container port)*
   * **Method**: `GET`
   * **Expected Status Code**: `200`
4. Click **Save**. The status indicator will turn **`● Healthcheck Healthy` (Green)**!

#### B. Configure Public Domain Routing (Traefik):
1. In your application dashboard, click the **General** tab.
2. Under **Domains**, provide a public domain or wildcard IP domain:
   ```text
   http://<YOUR_EC2_PUBLIC_IP>.sslip.io
   ```
   *(For example: `http://15.252.12.249.sslip.io`)*
3. Click **Save**. Coolify's Traefik proxy on port 80 will instantly route public traffic to your FastAPI app!

---

### 7. Verification & Production Testing

#### 1. System Healthcheck
```bash
curl -i http://<YOUR_EC2_PUBLIC_IP>:8001/api/v1/health
```
**Expected Response (`200 OK`)**:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "DEV",
  "available_providers": ["gemini", "groq"],
  "database_connected": true
}
```

#### 2. Interactive Swagger UI
Open your browser and navigate to:
```text
http://<YOUR_EC2_PUBLIC_IP>.sslip.io/docs
# or via direct host port:
http://<YOUR_EC2_PUBLIC_IP>:8001/docs
```

#### 3. Ask a Grounded Question (`POST /api/v1/ask`)
```bash
curl -X POST "http://<YOUR_EC2_PUBLIC_IP>:8001/api/v1/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the probation period policy?"}'
```
**Live Output with Strict Grounding & Citations**:
```json
{
  "answer": "Based on the provided documents, the probationary period policy is as follows:\n\n* Duration: New employees are subject to a probationary period of ninety (90) calendar days from the date of hire [Source: employee_handbook.txt, Page 1].\n* Evaluation and Termination: During this period, performance is evaluated, and employment may be terminated without the standard notice period [Source: employee_handbook.txt, Page 1].\n* Benefits Eligibility: Eligibility for benefits begins on the first day following the successful completion of the probationary period [Source: employee_handbook.txt, Page 1].",
  "sources": [
    {
      "document": "employee_handbook.txt",
      "page": 0,
      "relevance_score": 0.0018
    }
  ],
  "sufficient_context": true
}
```

#### 4. Asynchronous Document Ingestion via Celery (`POST /api/v1/documents/async`)
Upload a PDF or DOCX file to be processed asynchronously in the background:
```bash
curl -X POST "http://<YOUR_EC2_PUBLIC_IP>:8001/api/v1/documents/async" \
     -F "file=@sample_policy.pdf"
```
**Response**:
```json
{
  "task_id": "c6a1e389-9b43-4c91-9e23-389d41209b55",
  "filename": "sample_policy.pdf",
  "storage_key": "data/uploads/sample_policy.pdf",
  "status": "PENDING",
  "message": "Document uploaded and queued for background ingestion."
}
```

Check ingestion progress:
```bash
curl http://<YOUR_EC2_PUBLIC_IP>:8001/api/v1/documents/tasks/c6a1e389-9b43-4c91-9e23-389d41209b55
```

---

## 📄 License

This project is licensed under the **MIT License**.

