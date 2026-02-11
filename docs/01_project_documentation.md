# EcoloGRAPH — Project Documentation

> **Graph RAG for Ecological Research with Full Traceability**

## 1. Overview

EcoloGRAPH is a **Retrieval-Augmented Generation (RAG)** system designed for ecological and environmental research. It processes scientific publications (PDFs), extracts structured ecological data (species, measurements, locations, relationships), and organizes it into a searchable knowledge base spanning three complementary stores:

- **SQLite FTS5** — Full-text keyword search with BM25 ranking
- **Qdrant** — Semantic vector search with sentence embeddings
- **Neo4j** — Knowledge graph with typed relationships

An **interactive agent** (LangGraph) orchestrates queries across all stores, enriches results with external APIs (FishBase, GBIF, IUCN), and generates cross-domain scientific hypotheses.

## 2. Problem Statement

Ecological research spans dozens of interconnected domains (marine ecology, genetics, conservation, toxicology, etc.). Key challenges:

1. **Data fragmentation**: Relevant data is scattered across thousands of PDFs with no unified structure.
2. **Cross-domain blindness**: Researchers in one domain miss relevant findings from adjacent fields.
3. **Traceability**: AI-generated answers must cite specific papers, pages, and paragraphs.
4. **Local-first**: Many researchers need tools that run on local hardware without sending data to cloud APIs.

EcoloGRAPH addresses all four by building a locally-run Graph RAG pipeline with full source tracking.

## 3. Target Users

- **Ecological researchers** seeking to synthesize knowledge across domains
- **Conservation biologists** needing species profiles enriched with external data
- **Graduate students** exploring cross-domain connections in their thesis topics
- **Research groups** wanting to build searchable knowledge bases from their paper collections

## 4. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11+ | Core implementation |
| PDF Parsing | Docling | Extract text, tables, and metadata from PDFs |
| LLM | Local (LM Studio / Ollama) | Entity extraction, classification, agent reasoning |
| Vector DB | Qdrant | Semantic similarity search |
| Graph DB | Neo4j | Knowledge graph with Cypher queries |
| Search Index | SQLite FTS5 | BM25 keyword search |
| Agent Framework | LangGraph | Stateful, graph-based agent workflows |
| UI | Streamlit 1.54 | Interactive web dashboard |
| Embeddings | all-MiniLM-L6-v2 | Local sentence embeddings |
| Validation | Pydantic | Schema enforcement and data quality |
| External APIs | FishBase, GBIF, IUCN | Species enrichment |

## 5. Key Design Principles

1. **Full Traceability**: Every extracted entity carries a `SourceReference` (doc_id, page, position, chunk_id). Answers always cite sources.
2. **Graceful Degradation**: Each service (Qdrant, Neo4j, LLM) is optional. The system works with just SQLite if needed.
3. **Local-First**: All core functionality runs on local hardware. No cloud APIs required for base operation.
4. **43 Scientific Domains**: Rich domain taxonomy beyond generic "ecology" — from marine ecology to bioinformatics to ethology.
5. **Hybrid Search**: Combines keyword (BM25) and semantic (vector) search with learned reranking.

## 6. Project Structure

```
EcoloGRAPH/
├── config/
│   └── api-key              # LM Studio API key (auto-loaded)
├── data/
│   └── raw/                 # Input PDFs
├── docs/                    # This documentation
├── scripts/
│   ├── app.py               # Streamlit UI entry point
│   ├── chat_demo.py         # Terminal chat demo
│   ├── demo_pipeline.py     # Paper selection demo
│   └── ingest.py            # Ingestion pipeline
├── src/
│   ├── core/                # Config, schemas, LLM client, domain registry
│   ├── ingestion/           # PDF parser, document chunker
│   ├── extraction/          # Domain classifier, entity extractor
│   ├── search/              # SQLite FTS5 index, ranked search
│   ├── retrieval/           # Qdrant vector store, hybrid retriever
│   ├── graph/               # Neo4j graph builder, Cypher queries
│   ├── inference/           # Cross-domain linker, hypothesis generator
│   ├── scrapers/            # FishBase, GBIF, IUCN API clients
│   ├── enrichment/          # Crossref, Semantic Scholar, taxonomy
│   ├── agent/               # LangGraph query agent, tool registry
│   └── ui/                  # Streamlit pages and theme
└── tests/
    └── test_integration.py  # 33 integration tests
```

## 7. Environment Setup

```bash
# Create conda environment
conda create -n ecolograph python=3.11
conda activate ecolograph

# Install dependencies
pip install -r requirements.txt
```

### API Key

Place your LM Studio API key in `config/api-key` (auto-loaded, no `.env` needed).

### Docker Services

Start Docker Desktop, then:

```bash
# Qdrant (semantic vector search)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Neo4j (knowledge graph)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

### Ingest Papers

```bash
# Full pipeline (requires LLM + Docker services)
python scripts/ingest.py data/raw/

# Minimal (SQLite only, no Docker/LLM needed)
python scripts/ingest.py data/raw/ --skip-extract --skip-graph --skip-vectors
```

### Launch UI

```bash
streamlit run scripts/app.py
```

### Run Tests

```bash
python -m pytest tests/test_integration.py -v
# Expected: 33/33 passing
```

## 8. Querying Your Data

After ingestion, you can query and inspect your data in multiple ways:

### A. Command-Line Query Tool

```bash
# List all ingested papers
python scripts/query_chunks.py list

# View all 95 chunks from a specific paper
python scripts/query_chunks.py paper doc_abc123

# Semantic search across all chunks
python scripts/query_chunks.py search "carabid beetles" -n 10
```

### B. Python API

```python
# Query vector store (Qdrant)
from src.retrieval.vector_store import VectorStore

vs = VectorStore()
results = vs.search("species in wetlands", limit=5)

# Query paper metadata (SQLite)
from src.storage.paper_index import PaperIndex

idx = PaperIndex()
papers = idx.get_all_papers()
paper = idx.get_paper("doc_abc123")
```

### C. Streamlit UI

1. **Graph Explorer** → Select paper → "View Chunks" button
2. **Papers Page** → Click on paper → View chunks
3. **Search Page** → Enter query → See matching chunks

### D. Neo4j Browser

Access graph directly at `http://localhost:7474`:

```cypher
// Find all species mentioned in papers
MATCH (s:Species)<-[:MENTIONS]-(p:Paper)
RETURN s.name, count(p) as paper_count
ORDER BY paper_count DESC
LIMIT 20

// Find papers about specific species
MATCH (p:Paper)-[:MENTIONS]->(s:Species {name: "Pogonus minutus"})
RETURN p.title, p.doc_id
```

## 9. Current Status

All phases (0–9) are complete:
- ✅ 38 Python files across 10 modules
- ✅ 43 scientific domains with weighted keyword classification
- ✅ 7 agent tools (search, classify, species, graph, hypotheses)
- ✅ 33/33 integration tests passing
- ✅ Streamlit UI with 4 pages

## 9. License & Credits

Developed as a Bootcamp project. Uses open-source scientific APIs (FishBase, GBIF, IUCN Red List) and local LLMs for data processing.
