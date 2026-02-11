# EcoloGRAPH — Development Log

> Step-by-step development history with justifications for each decision.

---

## Phase 0–2: Foundation (PDF → Chunks → Entities)

### What was built
- `pdf_parser.py` — Docling-based PDF extraction
- `chunker.py` — Section-aware document chunking
- `entity_extractor.py` — LLM-powered entity extraction
- `schemas.py` — Pydantic data models with traceability

### Key Decisions

**Why Docling for PDF parsing?**
Docling preserves document structure (sections, tables, figures) and position metadata. Unlike PyPDF2 or pdfplumber, it understands academic paper layouts: multi-column text, captions, table structures. This is critical for traceability — knowing which page and section a species mention came from.

**Why section-aware chunking?**
Naive fixed-size chunking breaks context at arbitrary points. Section-aware chunking respects paragraph and section boundaries, keeping related content together. This improves both classification accuracy and LLM extraction quality.

**Why Pydantic for schemas?**
Every extracted entity (`SpeciesMention`, `Measurement`, `Location`, `EcologicalRelation`) carries a `SourceReference` with `doc_id`, `page`, `position`. Pydantic enforces this at the type level — if a developer creates a `SpeciesMention` without a source, it fails at construction, not silently at runtime.

---

## Phase 2.5–3: Domain Classification

### What was built
- `domain_registry.py` — 43 scientific domains with weighted keywords
- `domain_classifier.py` — Multi-label classification with keyword + LLM hybrid

### Key Decisions

**Why 43 domains instead of a generic taxonomy?**
Ecological research spans marine ecology, genetics, conservation, toxicology, climate change ecology, soundscape ecology, ethology, biogeography, and dozens more. A generic "biology" label is useless for cross-domain analysis. The 43 domains enable precise affinity scoring (e.g., marine ecology ↔ fisheries = high affinity, marine ecology ↔ machine learning = low).

**Why weighted keywords?**
Not all keywords are equally diagnostic. "coral" strongly indicates coral reef ecology (weight 3.0), while "species" is generic (weight 0.5). Weighted scoring prevents common ecological terms from dominating classification.

**Why lower the confidence threshold from 0.5 to 0.15?**
Weighted scores produce lower absolute numbers than unweighted counts. A weighted score of 0.18 with domain-specific terms is actually a confident classification. Testing showed 0.5 was rejecting valid classifications.

**Why add study type detection?**
Knowing whether a paper is experimental, observational, modeling, or a meta-analysis is critical metadata for synthesis. A meta-analysis about fish populations has different evidentiary weight than a single field observation.

---

## Phase 4: External API Scrapers

### What was built
- `fishbase_client.py` — FishBase REST API client
- `gbif_occurrence_client.py` — GBIF occurrence data
- `iucn_client.py` — IUCN Red List conservation status

### Key Decisions

**Why these three APIs?**
They cover the three pillars of species data: **biology** (FishBase: morphology, habitat, trophic level), **distribution** (GBIF: occurrence records, countries, coordinates), and **conservation** (IUCN: threat status, population trends). Together they provide a comprehensive species profile without needing the papers themselves.

**Why context managers (`with` pattern)?**
HTTP sessions should be opened and closed cleanly. Context managers guarantee cleanup even if exceptions occur. This prevents connection leaks during batch processing.

**Why rate limiting?**
All three APIs have usage policies. FishBase especially requires 2s between requests. Built-in rate limiting prevents IP bans during batch enrichment.

---

## Phase 5: Knowledge Graph (Neo4j)

### What was built
- `graph_builder.py` — Neo4j schema initialization and data loading
- `queries.py` — 14 reusable Cypher query templates

### Key Decisions

**Why Neo4j over a simple relational DB?**
Ecological data is inherently networked: species eat other species, live in locations, are studied in papers, have measurements. Graph databases express these relationships natively. Finding "all species connected to *Gadus morhua* by shared habitat within 2 hops" is a single Cypher traversal but would require complex recursive SQL.

**Why pre-built Cypher templates?**
Common queries (species profile, food web, co-occurrence, ecological network) are used repeatedly. Templates ensure consistent, optimized queries and make them available to the agent as tools.

---

## Phase 6: Vector Store & Hybrid Retrieval

### What was built
- `vector_store.py` — Qdrant vector store with sentence embeddings
- `hybrid_retriever.py` — Combined vector + graph retrieval

### Key Decisions

**Why Qdrant over FAISS?**
Qdrant runs as a service with persistent storage, metadata filtering, and a REST API. FAISS is in-memory only. For a knowledge base that grows over time, persistence is essential.

**Why all-MiniLM-L6-v2 for embeddings?**
It runs locally (no API calls), is fast (384-dim vectors), and has strong performance on semantic similarity benchmarks. The priority is local-first operation, not maximum accuracy.

**Why hybrid retrieval?**
Keyword search (BM25) finds exact term matches that embeddings might miss. Semantic search finds conceptually related content that keywords miss. Combining both yields the best recall.

---

## Phase 7–7.5: Cross-Domain Inference & Search

### What was built
- `cross_domain_linker.py` — Domain affinity matrix + link discovery
- `inference_proposer.py` — Rule-based and LLM-powered hypothesis generation
- `paper_index.py` — SQLite FTS5 full-text index
- `ranked_search.py` — Two-stage BM25 + semantic reranking

### Key Decisions

**Why a domain affinity matrix?**
Knowing that marine ecology and fisheries are highly related (0.85) while marine ecology and machine learning are not (0.3) enables intelligent cross-domain suggestions. The matrix is hand-tuned based on ecological domain knowledge.

**Why rule-based hypothesis generation?**
LLMs aren't always available (local server might be off). Rule-based templates from `CrossDomainLink` types (shared species, ecological cascade, complementary data) provide useful hypotheses even without an LLM.

**Why SQLite FTS5 specifically?**
FTS5 is built into Python's sqlite3 module — zero dependencies, zero server setup. It provides BM25 ranking out of the box. For a tool that must run anywhere with minimal setup, this is ideal.

---

## Phase 8: Query Agent

### What was built
- `query_agent.py` — LangGraph two-tier agent (fast router + full ReAct)
- `tool_registry.py` — 7 LangChain-compatible tools
- `chat_demo.py` — Interactive terminal demo
- `ingest.py` — Full ingestion pipeline

### Key Decisions

**Why LangGraph over LangChain AgentExecutor?**
LangGraph provides explicit state management via `StateGraph`. The agent's behavior is a directed graph with named nodes (router, chat, agent, tools), making it debuggable and extensible. AgentExecutor is a black box.

**Why a two-tier architecture?**
Most user queries don't need the full tool pipeline. "Hello" or "What tools do you have?" can be answered instantly without invoking search or APIs. The fast router (regex + minimal LLM) classifies intent as meta/chat/research, and only research queries trigger the full agent. This cuts response time in half for simple queries.

**Why 7 tools and not more?**
Each tool the LLM must reason about increases cognitive load and latency. Seven tools cover all use cases (search, classify, species info, graph queries, cross-domain, hypotheses) without overwhelming the model. Local 7B-14B models perform well with 5–8 tools but degrade beyond 10.

**Why auto-detect the loaded LLM?**
Users switch models frequently in LM Studio. Auto-detection queries the `/v1/models` endpoint to find whatever's loaded, eliminating manual model name configuration.

---

## Audit & Integration Fix

### What was found
A codebase audit of all 38 files revealed **7 critical bugs** in `tool_registry.py`:
- Wrong method names for all 3 scraper clients
- Wrong import path for GBIF client
- `classify_text` tool feeding empty text to classifier
- `generate_hypotheses` calling a nonexistent method
- Wrong attribute names on `Hypothesis` dataclass

### Root cause
The tools were written in a session where the actual scraper APIs were assumed rather than verified. Session interruptions prevented testing.

### Fix
Complete rewrite of `tool_registry.py` with verified API calls. Added `query_graph` tool exposing Neo4j. Created 33-test integration suite to prevent regressions.

---

## Phase 9: Streamlit UI

### What was built
- `theme.py` — Dark glassmorphism CSS design system
- 4 pages: Dashboard, Search, Species Explorer, Domain Lab
- `app.py` — Multi-page entry point

### Key Decisions

**Why Streamlit over Flask/FastAPI?**
Streamlit enables rapid prototyping of data-centric UIs without writing HTML/JS. For a research tool, development speed trumps frontend flexibility.

**Why glassmorphism dark theme?**
Premium appearance builds confidence in the tool. Dark themes reduce eye strain during long research sessions. Glassmorphism (frosted glass cards) provides visual depth without clutter.

**Why 4 pages?**
Each page maps to a distinct user workflow: overview (Dashboard), finding papers (Search), investigating species (Species Explorer), and analytical work (Domain Lab). More pages would fragment the experience.

---

## API Key Management

### What was built
- `config.py` → `_load_api_key_file()` reads `config/api-key` and sets `OPENAI_API_KEY` env var
- `llm_client.py` → Falls back to `OPENAI_API_KEY` env var if no explicit key passed
- `config/api-key.example` → Template for new users

### Key Decision

**Why a file instead of `.env`?**
The user explicitly requested not hardcoding the API key. A dedicated `config/api-key` file is simpler than `.env` — no parsing library needed, no variable name confusion, and the auto-loading chain (`config/api-key` → `os.environ["OPENAI_API_KEY"]` → `LLMClient`) is transparent.

---

## Infrastructure: Docker Services

### Key Decision

**Why Docker for Qdrant and Neo4j?**
Both services require their own runtimes. Docker provides one-command setup:
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```
No manual installation, no version conflicts, reproducible across machines.

**Why Docker Desktop must be started first (Windows)?**
The Docker daemon runs inside Docker Desktop on Windows. The `npipe:////./pipe/dockerDesktopLinuxEngine` error means the daemon isn't running — opening Docker Desktop from the Start menu resolves this.

---

## Dashboard Bug Fix

### Issue
Dashboard showed "100 Papers Indexed" but "No papers indexed yet" simultaneously.

### Root cause
`_get_index_stats()` called `get_all_papers()` which doesn't exist on `PaperIndex`. The `hasattr` guard silently returned `[]`, so `domain_stats` was always empty.

### Fix
Replaced with `idx.get_domains()` which was already implemented and queries SQL directly:
```python
domain_list = idx.get_domains()  # returns [(domain, count), ...]
domain_stats = {domain: cnt for domain, cnt in domain_list}
```
