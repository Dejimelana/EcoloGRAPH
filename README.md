# EcoloGRAPH

**Graph RAG system for ecological research — process PDFs, build a knowledge graph, and query it with an LLM agent.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.4.0-green.svg)]()

---

## ✨ What is EcoloGRAPH?

EcoloGRAPH is a **Graph RAG (Retrieval-Augmented Generation) system** designed specifically for ecological and environmental research. It transforms PDF scientific publications into a structured, queryable knowledge base where every entity is traceable to its exact source (paper, page, paragraph).

### 🎯 Key Features

- **🌍 43 Scientific Domains** — Automatic multi-label classification with weighted scoring (ecology, marine biology, climate science, machine learning, and more)
- **🕸️ Knowledge Graph** — Neo4j graph database for species, measurements, locations, and ecological relationships
- **🔍 Hybrid Search** — SQLite BM25 + Qdrant semantic embeddings + intelligent reranking
- **📊 Graph Analytics** — NetworkX community detection, centrality analysis, interactive concept maps
- **🧠 Cross-Domain Inference** — Automatic hypothesis generation from multi-domain connections
- **🤖 Interactive Agent** — LangGraph-powered agent with 8 specialized tools for intelligent querying
- **🐟 Species Enrichment** — Integration with FishBase, GBIF, and IUCN Red List APIs
- **💻 100% Local** — Run entirely on your machine with LM Studio or Ollama (no API costs)
- **🎨 Modern UI** — Beautiful Streamlit dashboard with dark/light themes and glassmorphism design

### 🆕 Recent Improvements (v1.4.0)

- **Paper-Based Entity Extraction** — Redesigned from chunk-by-chunk to paper-level processing with 70% fewer LLM calls
- **LM Studio Auto-Restart** — Automatic recovery from context overflow crashes
- **Improved Metadata** — Enhanced PyMuPDF extraction for titles, abstracts, and publication years
- **Query Tools** — New `query_chunks_lite.py` for exploring ingested data without UI
- **Token Validation** — Prevents context overflow with automatic batching for large papers
- **Cross-Chunk Resolution** — Resolves entity references across chunks ("the species" → actual name)

---

## 🏗️ Architecture

```
PDFs → Parser → Chunker → Domain Classifier → Entity Extractor (LLM)
                                  ↓
                  ┌───────────────┼───────────────┐
                  ↓               ↓               ↓
            SQLite FTS5      Neo4j Graph     Qdrant Vectors
            (BM25 search)    (entities)      (embeddings)
                  ↓               ↓               ↓
                  └───────────────┼───────────────┘
                                  ↓
          RankedSearch (hybrid BM25 + semantic reranking)
                                  ↓
          ┌───────────────────────┼───────────────────────┐
          ↓                       ↓                       ↓
  CrossDomainLinker     InferenceProposer     NetworkX Analytics
                                  ↓                  (Pyvis viz)
                          Query Agent (LangGraph)
                              8 tools
                                  ↓
                          Streamlit Dashboard
                              9 pages
```

---

## 🚀 Quick Start

### Prerequisites

| Component | Required? | Purpose |
|-----------|-----------|---------|
| Python 3.11+ | **Yes** | Runtime |
| Docker Desktop | For Qdrant/Neo4j | Vector store + graph database |
| LM Studio or Ollama | For LLM features | Entity extraction + chat agent |
| GPU 8GB+ VRAM | Recommended | Local LLM inference speed |

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/EcoloGRAPH.git
cd EcoloGRAPH

# Create and activate environment
conda create -n ecolograph python=3.11
conda activate ecolograph

# Install dependencies
pip install -r requirements.txt

# Install Graph Explorer V2 dependencies
pip install streamlit-agraph
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your settings (see LLM Connection Guide below)
```

### 3. Start Backend Services

EcoloGRAPH supports 3 installation tiers depending on what you need:

#### Tier 1: Lite (SQLite only — no Docker needed)

Provides keyword search, domain classification, and basic analytics.

```bash
# No additional services needed — just ingest papers:
python scripts/ingest.py data/raw/ --skip-extract --skip-graph --skip-vectors
streamlit run scripts/app.py
```

#### Tier 2: Standard (+ Qdrant for semantic search)

Adds vector embeddings for semantic similarity search and hybrid reranking.

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

python scripts/ingest.py data/raw/ --skip-extract --skip-graph
streamlit run scripts/app.py
```

#### Tier 3: Full (+ Neo4j + LLM — all features)

Full knowledge graph extraction, entity relationships, and LLM-powered chat.

```bash
# Start services
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password neo4j

# Start your LLM server (see LLM Connection Guide below)

# Full pipeline
python scripts/ingest.py data/raw/
streamlit run scripts/app.py
```

### 4. Add Your Papers

Place PDF files in `data/raw/` and run the ingestion pipeline:

```bash
python scripts/ingest.py data/raw/
```

The pipeline will:
1. ✅ Parse PDFs (Docling + PyMuPDF fallback)
2. ✅ Chunk text into sections
3. ✅ Classify into 43 scientific domains
4. ✅ Extract entities with LLM (species, measurements, locations, relationships)
5. ✅ Index in SQLite FTS5 (keyword search)
6. ✅ Store embeddings in Qdrant (semantic search)
7. ✅ Build knowledge graph in Neo4j (entity relationships)

### 5. Launch the UI

```bash
streamlit run scripts/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🤖 Connecting an LLM

EcoloGRAPH uses a local LLM for two functions:

1. **Entity extraction** during ingestion (parsing species, measurements, locations from text)
2. **Chat agent** for interactive querying (the LangGraph agent with 8 tools)

### Option A: LM Studio (Recommended for Windows)

1. Download [LM Studio](https://lmstudio.ai/)
2. Load a model (recommended: **Qwen2.5 7B** or **Mistral 7B Instruct**)
3. Start the local server (default: `http://localhost:1234/v1`)
4. Configure `.env`:

```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=auto
LOCAL_LLM_BASE_URL=http://localhost:1234/v1
```

> **Note:** `auto` will auto-detect whatever model is loaded in LM Studio.

### Option B: Ollama (Recommended for Linux/Mac)

1. Install [Ollama](https://ollama.ai/)
2. Pull a model:

```bash
ollama pull qwen2.5:7b
# or
ollama pull mistral:7b-instruct
```

3. Ollama starts automatically on `http://localhost:11434/v1`
4. Configure `.env`:

```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=qwen2.5:7b
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
```

### Option C: OpenAI-compatible API (Cloud)

Any OpenAI-compatible endpoint works (e.g., Together AI, Groq, OpenRouter):

```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=your-model-name
LOCAL_LLM_BASE_URL=https://api.your-provider.com/v1
```

Place your API key in `config/api-key`:

```
sk-your-api-key-here
```

### Verifying the LLM Connection

```bash
# Terminal chat demo (quick test)
python scripts/chat_demo.py
```

If the LLM is connected, you'll see:
- Model name detected and displayed
- Response to a test query

If **not** connected, you'll get a clear error: `No model loaded in LM Studio/Ollama`.

---

## 🔍 Querying Chunks and Data

After ingestion, you can query and browse your document chunks in several ways:

### Option 1: Command-Line Query Tool (Recommended)

Use the provided script for quick data exploration:

```bash
# List all papers in database
python scripts/query_chunks.py list

# View all chunks from a specific paper
python scripts/query_chunks.py paper doc_abc123

# Search chunks semantically
python scripts/query_chunks.py search "Pogonus minutus" -n 10
```

**Example output:**
```
📚 Found 100 papers in database:

1. doc_abc123
   Title: Wildfire response of forest species...
   Chunks: 95

2. doc_def456
   Title: Ecological indicators in wetlands...
   Chunks: 72
```

### Option 2: Python API

Query chunks programmatically:

```python
from src.retrieval.vector_store import VectorStore

vector_store = VectorStore()

# Semantic search
results = vector_store.search(
    query_text="beetles in wetlands",
    limit=5
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...")
```

### Option 3: SQLite Metadata

Query paper metadata from SQLite:

```python
from src.storage.paper_index import PaperIndex

idx = PaperIndex()

# Get all papers
papers = idx.get_all_papers()
for p in papers:
    print(f"{p['doc_id']}: {p['title']} ({p['chunk_count']} chunks)")

# Get specific paper
paper = idx.get_paper("doc_abc123")
print(f"Domain: {paper['domain']}, Confidence: {paper['confidence']}")
```

### Option 4: Streamlit UI

Browse chunks visually in the web interface:

1. **Graph Explorer** → Select paper → Click "View Chunks" button
2. **Papers page** → Search paper → View chunks directly
3. **Search page** → Semantic search → See matching chunks

---

## 📁 Project Structure

```
EcoloGRAPH/
├── config/
│   ├── api-key.example        # API key template
│   └── prompts/               # LLM prompt templates
├── data/
│   └── raw/                   # Your PDFs go here
├── docs/                      # Full documentation (5 files)
│   ├── 01_project_documentation.md
│   ├── 02_development_log.md
│   ├── 03_tutorial.md
│   ├── 04_architecture_diagrams.md
│   └── 05_module_summary.md
├── scripts/
│   ├── app.py                 # Streamlit UI entry point
│   ├── ingest.py              # Ingestion pipeline CLI
│   ├── chat_demo.py           # Terminal chat agent
│   ├── fix_fts5.py            # FTS5 index repair tool
│   └── demo_pipeline.py       # Paper selection demo
├── src/
│   ├── core/                  # Config, schemas, LLM client, 43 domains
│   ├── ingestion/             # PDF parser (Docling), section-aware chunker
│   ├── extraction/            # Domain classifier, LLM entity extractor
│   ├── search/                # SQLite FTS5 index, BM25 ranked search
│   ├── retrieval/             # Qdrant vector store, hybrid retriever
│   ├── graph/                 # Neo4j builder, Cypher queries, NetworkX analytics
│   ├── inference/             # Cross-domain linker, hypothesis generator
│   ├── scrapers/              # FishBase, GBIF, IUCN API clients
│   ├── enrichment/            # Crossref, Semantic Scholar, taxonomy
│   ├── agent/                 # LangGraph query agent, 8 tools
│   └── ui/                    # Streamlit pages (9) + theme system
└── tests/
    ├── test_integration.py    # Integration tests
    └── unit/                  # Unit tests
```

---

## 🎨 Streamlit Dashboard

9-page dashboard with dark/light themes and glassmorphism design:

| Page | Features |
|------|----------|
| 📊 Dashboard | Metrics, service status, domain distribution charts |
| 💬 Chat | Interactive LangGraph agent with tool call visualization |
| 📄 Papers | Paper browser with PDF.js viewer, metadata, and abstracts |
| 🕸️ Graph | 4 interactive views: Paper Network, Domain Map, Community, Concept Map |
| 🔗 Graph V2 | **NEW** Connected Papers-style explorer with click-to-explore nodes, sidebar metadata, chunk viewer with entity highlighting |
| 🔍 Search | Hybrid search with result cards and domain filters |
| 🧬 Species | FishBase/GBIF/IUCN tabbed species explorer |
| ✅ Validation | Species validation and data quality checks |
| 🔬 Classifier | Text classification demo, cross-domain links, hypothesis generation |

---

## 🛠️ Agent Tools

| Tool | Description | Data Source |
|------|-------------|-------------|
| `search_papers` | Hybrid BM25 + semantic search | SQLite + Qdrant |
| `search_by_domain` | Domain-filtered paper search | SQLite |
| `search_related_papers` | Graph-based related paper discovery | NetworkX |
| `query_graph` | Species profiles, networks, measurements | Neo4j |
| `classify_text` | Classify text into 43 scientific domains | DomainClassifier |
| `get_species_info` | Taxonomic and conservation data | FishBase + GBIF + IUCN |
| `find_cross_domain_links` | Domain co-occurrence analysis | CrossDomainLinker |
| `generate_hypotheses` | Cross-domain hypothesis generation | InferenceProposer |

---

## 🌍 Scientific Domains (43)

| Category | Domains |
|----------| --------|
| Aquatic | marine_ecology, coral_reef_ecology, freshwater_ecology, deep_sea_ecology, oceanography |
| Terrestrial | forest_ecology, soil_ecology, urban_ecology, landscape_ecology, agroecology |
| Organisms | entomology, ornithology, herpetology, mammalogy, microbiology, phycology, mycology |
| Processes | population_ecology, community_ecology, ecosystem_ecology, molecular_ecology, chemical_ecology |
| Applied | conservation, restoration_ecology, wildlife_management, environmental_monitoring |
| Climate | climate_change_ecology, paleoecology |
| Technology | bioinformatics, remote_sensing, machine_learning, soundscape_ecology |
| Biology | genetics, evolution, taxonomy, physiology, ethology, biotic_interactions |
| Earth | geology, biogeography, limnology |
| General | general_ecology, invasive_species, parasitology, toxicology, fisheries |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `fts5: missing row N` | Run `python scripts/fix_fts5.py` to rebuild the FTS5 index |
| `No model loaded` | Start LM Studio/Ollama and load a model before launching |
| Qdrant connection error | Ensure Docker is running: `docker start qdrant` |
| Neo4j connection error | Ensure Docker is running: `docker start neo4j` |
| Slow chat responses | Router handles ~80% of queries without LLM; for the rest, use a faster model (7B vs 14B) |
| PDF viewer blank | PDF.js renders on canvas — if blank, check browser console for errors |
| Graph V2 not loading | Make sure `streamlit-agraph` is installed: `pip install streamlit-agraph` |

---

## 🧪 Running Tests

```bash
python -m pytest tests/test_integration.py -v
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Project Documentation](docs/01_project_documentation.md) | Overview, tech stack, design principles |
| [Development Log](docs/02_development_log.md) | Phase-by-phase decisions and justifications |
| [Tutorial](docs/03_tutorial.md) | Step-by-step user guide |
| [Architecture Diagrams](docs/04_architecture_diagrams.md) | 7 Mermaid system diagrams |
| [Module Summary](docs/05_module_summary.md) | All modules, classes, and functionalities |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute to the project |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture deep-dive |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up your development environment
- Code style and conventions
- Submitting pull requests
- Reporting bugs and requesting features

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Docling** for advanced PDF parsing
- **LangChain/LangGraph** for agent framework
- **Neo4j** for graph database
- **Qdrant** for vector search
- **Streamlit** for rapid UI development
- **FishBase, GBIF, IUCN** for species data APIs

---

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue on GitHub
- Email: diego.jimenez@mncn.csic.es

---

**Built with ❤️ for the ecological research community**
