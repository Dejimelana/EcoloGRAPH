# EcoloGRAPH Architecture

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Storage Layer](#storage-layer)
- [Agent System](#agent-system)
- [UI Layer](#ui-layer)
- [Performance Optimizations](#performance-optimizations)
- [Deployment Considerations](#deployment-considerations)

---

## System Overview

EcoloGRAPH is a **multi-layer Graph RAG system** designed for ecological research. The architecture follows a modular design with clear separation between ingestion, storage, retrieval, and presentation layers.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│   Streamlit Dashboard (9 pages) + Theme System (Dark/Light)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                     AGENT LAYER                                  │
│  LangGraph Router + 8 Specialized Tools + Query Orchestration   │
└───────────┬──────────────────────────────────┬──────────────────┘
            │                                  │
┌───────────▼─────────────┐       ┌───────────▼────────────────┐
│   RETRIEVAL LAYER       │       │   ANALYSIS LAYER           │
│  - Hybrid Search        │       │  - Cross-Domain Linker     │
│  - BM25 + Semantic      │       │  - Hypothesis Generator    │
│  - Reranking            │       │  - Network Analytics       │
└───────────┬─────────────┘       └───────────┬────────────────┘
            │                                  │
┌───────────▼─────────────┬────────────────────▼────────────────┐
│              STORAGE LAYER                                     │
│  SQLite FTS5  │  Qdrant Vectors  │  Neo4j Graph               │
└───────────┬──────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                            │
│  PDF Parser → Chunker → Classifier → LLM Extractor → Indexer   │
└──────────────────────────────────────────────────────────────────┘
```

### Recent Updates (v1.4.0 - February 2026)

**Paper-Based Entity Extraction (Major Architecture Change)**
- Redesigned extraction from chunk-by-chunk to **paper-as-functional-unit** approach
- Added `token_utils.py` for context window validation and safe batch size estimation
- Created paper-level extraction prompt (`config/prompts/extraction_paper.txt`)
- Implemented automatic batching fallback when papers exceed context limits
- **Performance**: ~70% reduction in LLM calls (25 vs 80 for typical paper)
- **Quality**: Cross-chunk entity resolution and automatic deduplication

**LM Studio Auto-Restart System**
- Created `lm_studio_manager.py` for process health monitoring and restart
- Detects corrupted state after context overflow and automatically restarts
- Pauses ingestion until LM Studio is ready (~30-60s recovery time)
- Prevents cascade failures from single problematic chunk

**Metadata Extraction Improvements**
- Added `year` field to `ParsedDocument` schema
- Enhanced PyMuPDF fallback to detect bad titles (<!-- image -->, etc.)
- Automatic year extraction from PDF metadata and text patterns (©2024, (2024), Published: 2024)
- Improved logging of metadata enhancement

**Data Querying Tools**
- Created `query_chunks_lite.py` for lightweight chunk exploration without heavy dependencies
- Supports listing all papers, viewing chunks by doc_id, and searching by title/source
- Added comprehensive querying documentation to README and project docs

**Vector Store Improvements**
- Fixed `.tolist()` errors in `vector_store.py` by checking embedding type before conversion
- Improved compatibility with various embedding libraries (numpy arrays vs lists)
- Batch processing now handles both formats seamlessly

**Graph Explorer Consolidation** (v1.3.0)
- Merged Graph V2 into main Graph Explorer with tabbed interface
- Domain-based tab uses SQLite (always available, fast)
- Entity-based tab uses Neo4j with Species/Locations (requires ingestion with entity extraction)
- Unified sidebar for all node types with metadata display
- Document chunks viewer with section grouping


---

## Core Components

### 1. Ingestion Pipeline

**Location**: `src/ingestion/`, `src/extraction/`

#### PDF Parser (`parser.py`)
```python
class PDFParser:
    """
    Two-tier parsing strategy:
    1. Docling (primary) - Advanced table/figure extraction
    2. PyMuPDF (fallback) - Fast text extraction
    """
```

**Features**:
- Table structure preservation
- Figure/image detection
- Metadata extraction (title, authors, abstract)
- Automatic abstract cleaning (removes artifacts)

#### Chunker (`chunker.py`)
```python
class SectionAwareChunker:
    """
    Section-aware chunking with adaptive sizes:
    - Abstract: 500 tokens
    - Methods: 800 tokens  
    - Results: 1000 tokens
    - Standard: 700 tokens
    """
```

**Strategy**:
- Respects section boundaries (doesn't split mid-section)
- Preserves metadata (section name, page number)
- Optimized chunk sizes per section type

#### Domain Classifier (`domain_classifier.py`)
```python
class DomainClassifier:
    """
    Multi-label classifier using TF-IDF + domain-specific keywords.
    Supports 43 scientific domains with weighted scoring.
    """
```

**Approach**:
- TF-IDF vectorization of text
- Keyword matching with domain-specific dictionaries
- Weighted scoring (primary + secondary domains)
- Threshold-based multi-label assignment

#### Entity Extractor (`llm_extractor.py`)
```python
class LLMExtractor:
    """
    LLM-powered entity extraction with structured output.
    Extracts: species, measurements, locations, relationships.
    """
```

**Process**:
1. Send chunk + schema to LLM
2. Parse JSON response into Pydantic models
3. Validate and normalize entities
4. Link entities to source (doc_id, chunk_id, page)

---

### 2. Storage Layer

#### SQLite FTS5 (Full-Text Search)

**Location**: `src/search/paper_index.py`

```python
class PaperIndex:
    """
    SQLite with FTS5 virtual table for keyword search.
    BM25 ranking algorithm built-in.
    """
```

**Schema**:
```sql
CREATE TABLE papers (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,  -- JSON array
    year INTEGER,
    journal TEXT,
    abstract TEXT,
    keywords TEXT,  -- JSON array
    primary_domain TEXT,
    domains TEXT,  -- JSON dict with scores
    study_type TEXT,
    source_path TEXT,
    indexed_at TEXT
);

CREATE VIRTUAL TABLE papers_fts USING fts5(
    doc_id, title, abstract, keywords, authors,
    content='papers', content_rowid='rowid'
);
```

**Why FTS5?**
- Fast keyword search with BM25 ranking
- Lightweight (no external service)
- Supports phrase queries, boolean operators
- Automatic stemming and tokenization

#### Qdrant (Vector Store)

**Location**: `src/retrieval/vector_store.py`

```python
class VectorStore:
    """
    Qdrant vector database for semantic similarity search.
    Uses all-MiniLM-L6-v2 embeddings (384 dimensions).
    """
```

**Collection Schema**:
```python
{
    "chunk_id": str,  # Unique identifier
    "doc_id": str,    # Parent paper ID
    "text": str,      # Chunk text
    "section": str,   # Section name
    "page": int,      # Page number
    "domain": str,    # Classified domain
    "vector": [384]   # Embedding
}
```

**Why Qdrant?**
- Fast HNSW index for million-scale vectors
- Rich filtering (by doc_id, domain, etc.)
- Low memory footprint
- Easy Docker deployment

#### Neo4j (Knowledge Graph)

**Location**: `src/graph/graph_builder.py`

```python
class GraphBuilder:
    """
    Neo4j graph database for entity relationships.
    Stores species, measurements, locations, and ecological relations.
    """
```

**Graph Schema**:
```cypher
// Nodes
(:Paper {doc_id, title, year, abstract, doi})
(:Species {scientific_name, common_name, aphia_id, gbif_key})
(:Location {location_id, name, coordinates, country})
(:Measurement {measurement_id, parameter, value, unit})
(:Author {name})

// Relationships
(Paper)-[:AUTHORED_BY]->(Author)
(Paper)-[:MENTIONS]->(Species)
(Paper)-[:REFERENCES_LOCATION]->(Location)
(Paper)-[:CONTAINS]->(Measurement)
(Species)-[:HAS_MEASUREMENT]->(Measurement)
(Species)-[:RELATES_TO {type}]->(Species)
```

**Why Neo4j?**
- Native graph traversal (find connected papers)
- Powerful Cypher query language
- Community detection algorithms built-in
- Scales to millions of nodes/relationships

---

### 3. Retrieval System

#### Hybrid Search

**Location**: `src/retrieval/hybrid_retriever.py`

**Algorithm**:
```python
def hybrid_search(query: str, limit: int = 10):
    # 1. BM25 keyword search (SQLite FTS5)
    bm25_results = paper_index.search(query, limit=50)
    
    # 2. Semantic search (Qdrant)
    semantic_results = vector_store.search(query, limit=50)
    
    # 3. Merge and rerank
    combined = merge_results(bm25_results, semantic_results)
    reranked = rerank_by_score(combined, weights={
        "bm25": 0.4,
        "semantic": 0.6
    })
    
    return reranked[:limit]
```

**Rationale**:
- BM25 good for exact keyword matches
- Semantic search good for conceptual similarity
- Hybrid combines best of both

---

### 4. Agent System

#### LangGraph Router

**Location**: `src/agent/query_router.py`

```python
class QueryRouter:
    """
    Routes queries to appropriate handler:
    1. Regex-based intent detection (80% of queries)
    2. LLM classification (20% complex queries)
    """
```

**Routing Logic**:
```python
if matches_regex(query, PAPER_SEARCH_PATTERNS):
    return DirectAnswerAgent()
elif matches_regex(query, SPECIES_INFO_PATTERNS):
    return SpeciesScraperAgent()
else:
    return LangGraphAgent()  # LLM-based routing
```

**Why Router?**
- Avoids LLM call for simple queries (faster + cheaper)
- Provides instant responses for common patterns
- Fallback to LLM for complex multi-step queries

#### LangGraph Agent

**Location**: `src/agent/graph_agent.py`

**Tool Set**:
1. **search_papers** - Hybrid search across corpus
2. **search_by_domain** - Domain-filtered search
3. **search_related_papers** - Graph traversal for related papers
4. **query_graph** - Neo4j Cypher queries
5. **classify_text** - Domain classification
6. **get_species_info** - External API calls (FishBase, GBIF, IUCN)
7. **find_cross_domain_links** - Domain co-occurrence analysis
8. **generate_hypotheses** - Cross-domain inference

**Agentic Flow**:
```
User Query
    ↓
Router Decision
    ↓
LangGraph Agent
    ↓
Tool Selection (LLM)
    ↓
Tool Execution
    ↓
Result Synthesis (LLM)
    ↓
Response
```

---

### 5. UI Layer

#### Streamlit Dashboard

**Location**: `src/ui/pages/`

**Pages**:
1. **dashboard.py** - System overview and metrics
2. **chat.py** - Interactive agent interface
3. **papers.py** - PDF browser with viewer
4. **graph_explorer.py** - **Unified Graph Explorer** (Domain-based + Entity-based tabs)
   - Domain-based tab: Papers connected via shared scientific domains (SQLite)
   - Entity-based tab: Papers connected via Species/Locations (Neo4j)
   - Interactive node selection with unified sidebar
   - Document chunks viewer with section grouping
5. **search.py** - Search interface
6. **species.py** - Species data explorer
7. **validation.py** - GBIF taxonomic validation
8. **classifier.py** - Domain classifier demo

> **Note**: Graph V2 (graph_explorer_v2.py) has been deprecated and merged into graph_explorer.py as the "Interactive Explorer" mode with tabbed visualization.

#### Recent UI Improvements (v1.3.0)

**Unified Graph Explorer** - February 2026
- Consolidated Graph V2 functionality into main Graph Explorer
- Added tabbed interface: Domain-based (always available) and Entity-based (requires Neo4j entities)
- Implemented shared node click handler with session state management
- Enhanced sidebar with paper metadata display (title, year, abstract, top domains)
- Added functional "View Chunks" button with section-grouped expandable display
- Fixed Neo4j Cypher query syntax for entity extraction
- Improved chunks visibility with prominent toggle buttons

#### Theme System

**Location**: `src/ui/theme.py`, `src/ui/theme_light.py`

**Features**:
- Dark mode (default) - Glassmorphism design
- Light mode - Clean professional theme
- Toggle button in sidebar
- Persists via session state

---

## Data Flow

### Ingestion Flow

```
PDF File
    ↓
[1] PDFParser.parse()
    → Extract text, metadata, tables
    ↓
[2] SectionAwareChunker.chunk()
    → Split into section-aware chunks
    ↓
[3] DomainClassifier.classify()
    → Assign domains with scores
    ↓
[4] LLMExtractor.extract()
    → Parse entities (species, measurements, locations)
    ↓
[5] Parallel Indexing:
    ├─ PaperIndex.add_paper()      → SQLite FTS5
    ├─ VectorStore.add_chunks()    → Qdrant vectors
    └─ GraphBuilder.add_paper()    → Neo4j graph
```

### Query Flow

```
User Question
    ↓
[1] QueryRouter.route()
    → Regex match or LLM classification
    ↓
[2] Agent Selection:
    ├─ DirectAnswerAgent (80%)
    │   → search_papers() → SQLite + Qdrant → Response
    │
    └─ LangGraphAgent (20%)
        ↓
    [3] Tool Selection (LLM)
        → Decides which tools to call
        ↓
    [4] Tool Execution
        ├─ search_papers()
        ├─ query_graph()
        ├─ get_species_info()
        └─ ... (can chain multiple tools)
        ↓
    [5] Synthesis (LLM)
        → Combine tool results into natural language
        ↓
Response
```

---

## Performance Optimizations

### Query Caching (v1.2.0)

**Implementation**:
```python
from diskcache import Cache

_query_cache = Cache(".cache/neo4j_queries", size_limit=100 * 1024 * 1024)

@cached_query(ttl=300)  # 5 minutes
def get_paper_metadata(doc_id: str):
    # First call: query Neo4j
    # Subsequent calls: return from cache
    ...
```

**Impact**: 5-10x speedup on repeated queries

### Lazy Loading (v1.2.0)

**Implementation**:
```python
def _build_graph_data(max_nodes=100):
    query = """
    MATCH (p:Paper)-[r:MENTIONS]->(s:Species)
    ...
    ORDER BY mention_count DESC
    LIMIT $max_nodes  -- User-configurable (50-500)
    """
```

**Impact**: Smooth performance with 500+ papers

### Batch Processing (v1.2.0)

**Implementation**:
```python
def add_chunks(chunks, batch_size=200):  # Increased from 100
    for batch in chunks[::batch_size]:
        embeddings = encoder.encode(batch, 
                                    show_progress_bar=False,
                                    convert_to_numpy=False)
        client.upsert(points, wait=False)  # Non-blocking
```

**Impact**: 2x faster vector indexing

---

## Deployment Considerations

### Resource Requirements

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Python process | 2 cores | 4 GB | - |
| Qdrant | 1 core | 2 GB | 1 GB per 100k chunks |
| Neo4j | 2 cores | 2 GB | 500 MB per 10k papers |
| LLM (7B local) | - | 8 GB VRAM | 4 GB |

### Scaling

**Horizontal Scaling**:
- Qdrant supports clustering (shard across nodes)
- Neo4j supports read replicas
- Streamlit can run behind load balancer

**Vertical Scaling**:
- Increase Qdrant memory for more vectors
- Increase Neo4j heap for larger graphs
- Use faster LLM (e.g., Groq API) for agent

### Data Persistence

```bash
# Volumes to persist
data/
  ├── papers.db          # SQLite FTS5
  ├── qdrant_storage/    # Qdrant data
  └── raw/               # Original PDFs

# Neo4j data (Docker volume)
neo4j_data/

# Model cache
.cache/
  ├── sentence_transformers/  # Embedding models
  └── neo4j_queries/          # Query cache
```

---

## Security Considerations

### Data Privacy
- All processing can be done locally (no cloud APIs required)
- PDFs never leave your machine
- LLM runs locally via LM Studio/Ollama

### API Keys
- External APIs optional (FishBase, GBIF, IUCN)
- Keys stored in `config/api-key` (gitignored)
- Never hardcoded in source

### Access Control
- Neo4j requires authentication
- Streamlit has no built-in auth (use reverse proxy if needed)
- Docker containers isolated by default

---

## Future Architecture Improvements

### Planned Enhancements
1. **Distributed ingestion** - Celery task queue
2. **Real-time updates** - Watch folder for new PDFs
3. **Advanced caching** - Redis for distributed cache
4. **Multi-tenancy** - User-specific data isolation
5. **API layer** - REST API for programmatic access
6. **Enhanced monitoring** - Prometheus + Grafana metrics

---

## References

- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
