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
- [License](#-license)

---

## 🌟 Key Highlights

- **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Combines dense vector semantics (Neon PGVector) with sparse keyword matching (BM25) to prevent legal term misses.
- **Zero Hallucination Tolerance**: Strict system prompts compel the model to refuse queries when context is insufficient rather than fabricate clauses or policies.
- **Auditable Citations**: Every answer includes traceable document references with page numbers and exact excerpt snippets.
- **Clean End-User API Contract**: Request bodies take only `{"question": "..."}`—backend infrastructure details (provider, model, internal tokens) are fully decoupled.
- **Multi-LLM Provider Support**: Pluggable support for Groq (Qwen / LLaMA), Google Gemini, OpenAI, Anthropic, and Mistral with lazy importing.
- **Serverless PostgreSQL + PGVector**: Cloud-native, scalable vector database hosted on Neon with automated `postgresql+psycopg://` v3 driver dialect handling.
- **Streaming & REST**: Supports both standard synchronous JSON responses and real-time Server-Sent Events (SSE) token streaming.

---

## 🛠 Technology Stack

| Layer | Technologies | Purpose |
|---|---|---|
| **Language & Runtime** | **Python 3.12+**, **Astral `uv`** | High-performance Python runtime and lightning-fast package management |
| **API Framework** | **FastAPI 0.115+**, **Uvicorn 0.34+** | Asynchronous web framework, OpenAPI 3.1 docs, CORS, and GZip compression |
| **Configuration** | **Pydantic Settings v2.7+**, **Pydantic v2** | Strictly typed, environment-driven configuration with `.env` overrides |
| **Vector Store** | **Neon PostgreSQL**, **`pgvector`**, **`langchain-postgres` 0.0.13** | Cloud-native serverless PostgreSQL vector store with HNSW/IVFFlat indexing |
| **Postgres Driver** | **`psycopg` 3 (`psycopg[binary]>=3.2`)** | Modern async-compatible DBAPI driver with auto-dialect normalization |
| **Dense Embeddings** | **Google Gemini Embeddings** (`gemini-embedding-001`, 768-dim) | High-fidelity dense semantic representations |
| **Sparse Keyword Search** | **`rank-bm25` (Okapi BM25)** | Lexical exact-match retrieval for policy numbers, statutory references, and clauses |
| **LLM Orchestration** | **`langchain-core` 1.6+** | Lightweight, modular runnables and prompt chaining (no `langchain-community` bloat) |
| **LLM Providers** | **Groq**, **Google Gemini**, **OpenAI**, **Anthropic**, **Mistral** | Pluggable multi-model inference with lazy loading |
| **Document Parsers** | **`pypdf` 5+**, **`python-docx` 1.1+** | Native extraction for PDF, DOCX, and raw legal TXT files |
| **Testing & Quality** | **Pytest 8.4+**, **Pytest-Asyncio**, **HTTPX**, **Ruff 0.8+** | 35 automated tests, sub-second linting, and formatting |
| **Containerization** | **Docker**, **Docker Compose** | Multi-stage distroless-style build running under non-root unprivileged user |

---

## 🏗 System Architecture

### High-Level Architecture

```mermaid
flowchart TD
    User([Client / Postman / Frontend]) -->|HTTP REST / SSE| API[FastAPI Gateway /api/v1]
    
    subgraph "Application Layer (src/app)"
        API --> Endpoints["Endpoints: /health, /documents, /ask"]
        Endpoints --> RAG[RAGChain Engine]
        Endpoints --> Ingest[IngestionService]
        
        subgraph "Retrieval Subsystem"
            RAG --> Hybrid[HybridRetriever]
            Hybrid -->|Dense Semantic Query| PGV[(Neon PostgreSQL + pgvector)]
            Hybrid -->|Sparse Lexical Query| BM25[In-Memory Okapi BM25 Index]
            Hybrid -->|RRF Fusion Algorithm| RankedDocs[Fused Top-K Chunks]
        end

        subgraph "Generation Subsystem"
            RAG --> Guard{Hallucination Guard}
            Guard -->|Context Missing| Refusal["Refusal Response: 'I cannot find...'"]
            Guard -->|Context Found| Prompt[Legal Grounding Prompt Template]
            Prompt --> Factory[LLMFactory]
            Factory --> Provider[Groq / Gemini / OpenAI / Anthropic]
        end
    end

    Provider --> Response["Cited RAGResponse {answer, sources}"]
    Refusal --> Response
    Response --> User
```

---

### Document Ingestion Pipeline

When documents are uploaded or seeded via `scripts/seed.py`:

```mermaid
sequenceDiagram
    autonumber
    participant Admin as User / Seed Script
    participant Ingestion as IngestionService
    participant Parser as Document Loaders (PDF/DOCX/TXT)
    participant Chunker as DocumentChunker
    participant Embedder as Embedding Model (Google GenAI)
    participant Neon as Neon PGVector
    participant BM25 as BM25 Index

    Admin->>Ingestion: Upload file (e.g. employee_handbook.txt)
    Ingestion->>Ingestion: Compute SHA-256 Hash (Deduplication check)
    Ingestion->>Parser: Extract text and page metadata
    Parser-->>Ingestion: Raw text pages
    Ingestion->>Chunker: Split text into legal chunks (preserve clauses)
    Chunker-->>Ingestion: Chunks with source, page, & chunk_index
    Ingestion->>Embedder: Generate 768-dim embeddings
    Embedder-->>Ingestion: Vectors
    Ingestion->>Neon: Store chunks & vectors in langchain_pg_embedding
    Ingestion->>Neon: Query all collection documents via direct SQL
    Neon-->>Ingestion: Document corpus
    Ingestion->>BM25: Rebuild BM25 in-memory index
    Ingestion-->>Admin: IngestionResponse (status, chunks_created)
```

---

### Query & Answering Lifecycle

When a user submits a natural-language question:

```mermaid
sequenceDiagram
    autonumber
    participant Client as Client Application
    participant API as POST /api/v1/ask
    participant Retriever as HybridRetriever
    participant PGVector as Neon PGVector
    participant BM25 as Okapi BM25
    participant Chain as RAGChain
    participant LLM as Active LLM (e.g. Groq Qwen 3.8)

    Client->>API: {"question": "What is the notice period for junior staff?"}
    API->>Chain: ask(QuestionRequest)
    par Concurrent Dense & Sparse Search
        Chain->>Retriever: retrieve(question, top_k=5)
        Retriever->>PGVector: similarity_search_with_score(query, k=5)
        Retriever->>BM25: invoke(query, k=5)
    end
    PGVector-->>Retriever: Top dense chunks
    BM25-->>Retriever: Top sparse chunks
    Retriever->>Retriever: Reciprocal Rank Fusion (RRF) Re-ranking
    Retriever-->>Chain: Top 5 fused document chunks
    alt No Relevant Chunks Found
        Chain-->>API: Refusal: "I cannot find sufficient information..."
    else Relevant Chunks Available
        Chain->>Chain: Format context blocks [Source: file, Page X]
        Chain->>LLM: QA_PROMPT(context, question)
        LLM-->>Chain: Grounded Answer with exact citations
        Chain-->>API: RAGResponse(answer, sources, sufficient_context=true)
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

---

## 📁 Project Directory Structure

```text
legal-rag-qa/
├── data/
│   ├── samples/                    # Sample legal & HR policies (employee_handbook.txt, etc.)
│   └── uploads/                    # Uploaded documents directory
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
│   │   │           ├── documents.py# Document upload, listing, & deletion
│   │   │           └── question.py # Sync (/ask) & streaming (/ask/stream) QA endpoints
│   │   ├── config/
│   │   │   ├── __init__.py         # Config exports
│   │   │   └── settings.py         # Pydantic v2 Settings (DEV/PROD/LOCAL, CORS, URLs)
│   │   ├── document_processing/
│   │   │   ├── loaders.py          # PDF, DOCX, and TXT loaders
│   │   │   ├── chunker.py          # Legal structure-preserving text chunker
│   │   │   └── ingestion.py        # Pipeline orchestrator with SHA-256 deduplication
│   │   ├── llm/
│   │   │   ├── factory.py          # Multi-LLM provider instantiation
│   │   │   └── prompts.py          # Grounding prompt templates with strict citation rules
│   │   ├── rag/
│   │   │   ├── __init__.py         # RAG package exports
│   │   │   └── chain.py            # Hybrid retrieval, hallucination guard, & generation pipeline
│   │   ├── retrieval/
│   │   │   ├── vector_store.py     # Neon PGVector manager with direct SQL indexing
│   │   │   ├── bm25_retriever.py   # In-memory Okapi BM25 keyword search service
│   │   │   └── hybrid.py           # Reciprocal Rank Fusion (RRF) rank merger
│   │   └── schemas/
│   │       ├── __init__.py         # Centralized schema exports
│   │       ├── question.py         # QuestionRequest, RAGResponse, SourceReference
│   │       ├── document.py         # IngestionResponse, DocumentInfo, DocumentListResponse
│   │       └── health.py           # HealthResponse
│   └── tests/
│       ├── conftest.py             # Shared fixtures and mock configurations
│       ├── test_api.py             # FastAPI TestClient endpoint integration tests
│       ├── test_config.py          # Settings, env parsing, and dialect normalization tests
│       ├── test_document_processing.py # Chunker, loader, and ingestion tests
│       ├── test_rag_chain.py       # Chain execution, citation, and refusal tests
│       └── test_retrieval.py       # BM25 and Hybrid RRF fusion tests
├── .env.example                    # Environment template
├── Dockerfile                      # Multi-stage container definition
├── docker-compose.yml              # Single compose file for containerized execution
├── postman.json                    # Postman Collection v2.1.0 with 10 pre-configured requests
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

## 📮 Postman Collection

The project includes a ready-to-import Postman collection at the repository root: [`postman.json`](postman.json).

### How to Import:
1. Open **Postman**.
2. Click **Import** (top left).
3. Select [`postman.json`](postman.json).
4. The collection will import with 10 pre-configured requests grouped under:
   - `Health` (Health Check, Readiness Probe)
   - `Documents` (List Documents, Ingest File, Delete File)
   - `Question Answering (RAG)` (Ask Question, Probation Rules, SSE Streaming, Refusal Test)
   - `System & OpenAPI` (OpenAPI Schema)
5. Start your server with `./start.sh` and test immediately!

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

### Test Coverage Summary:
- **`test_api.py`** — Validates all endpoint routes, validation errors, and health checks.
- **`test_config.py`** — Verifies Pydantic v2 settings, environment overrides, CORS parsing, and database URL dialect auto-normalization.
- **`test_document_processing.py`** — Checks file loaders, chunking logic, and ingestion hashing.
- **`test_rag_chain.py`** — Asserts hallucination refusal, prompt construction, and citation generation.
- **`test_retrieval.py`** — Verifies BM25 index rebuilding and Reciprocal Rank Fusion re-ranking.

---

## 🐳 Docker Deployment

The [`Dockerfile`](Dockerfile) uses a lightweight, secure multi-stage build:

```bash
# Build and run the containerized application
docker compose up --build
```

The container runs with:
- Non-root user (`appuser`, UID 10001).
- Healthcheck calling `/api/v1/health` every 30s.
- Bound to `http://0.0.0.0:8000`.

---

## 📄 License

This project is licensed under the **MIT License**.
