# EcoloGRAPH

**Graph RAG system for ecological research — process PDFs, build a knowledge graph, and query it with an LLM agent.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.5.0-green.svg)]()

---

## ✨ What is EcoloGRAPH?

EcoloGRAPH is a **Graph RAG (Retrieval-Augmented Generation) system** designed specifically for ecological and environmental research. It transforms PDF scientific publications into a structured, queryable knowledge base where every entity is traceable to its exact source (paper, page, paragraph).

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/9ca0a607-23a4-4ac0-878c-73f45ac46e75" />

---

- **🧠 Multi-Domain Knowledge Graph**: Classifies papers across **43 scientific domains** and builds a rich Neo4j graph.
- **🧬 Dual-Prompt Extraction**: Ecological + generic scientific prompts with intelligent fallback.
- **📖 Citation Network**: Extracts references and builds `Paper` → `CITES` → `Citation` relationships.
- **🛡️ Robust Ingestion Pipeline**:
  - **Configurable LLM**: Choose model, thinking mode, tokens, and timeout via CLI flags.
  - **Smart Chunking**: Section-aware hybrid chunking for optimal retrieval.
  - **Auto-Restart**: Built-in mitigation for LLM memory leaks.
- **🔍 Hybrid Search Engine**: Combines **Qdrant** (semantic) + **SQLite FTS5** (keyword) + reranking.
- **🕸️ Interactive Graph Explorer**: Full-width graph with node-click details, species co-occurrence edges, and chunk viewer.
- **🧠 Cross-Domain Inference**: Automatic hypothesis generation from multi-domain connections.
- **🤖 Interactive Agent**: LangGraph-powered agent with 8 specialized tools.
- **🐟 Species Enrichment**: FishBase, GBIF, and IUCN Red List API integration with common name resolution.
- **🧬 Taxonomy Explorer**: GBIF-powered name resolver, batch validation, and taxonomic statistics.
- **💻 100% Local**: Runs on Ollama (recommended) or any OpenAI-compatible API — no cloud costs.
- **🎨 Modern UI**: Streamlit dashboard with dark/light themes and glassmorphism design.

### 🆕 Recent Improvements (v1.5.0)

- **Ollama Migration** — Primary LLM backend, replacing LM Studio. Full Qwen3 thinking model support.
- **Dual-Model Architecture** — Separate models for ingestion (fast, `/no_think`) and reasoning (deep, with thinking).
- **CLI Ingestion Params** — `--model`, `--thinking`, `--max-tokens`, `--timeout` for full control.
- **Qwen3 Compatibility** — Handles Ollama's `reasoning` field, strips `<think>` tags, configurable thinking mode.
- **Diagnostic Tools** — `diagnose_agent.py`, `test_ollama_models.py` for troubleshooting.
- **Testing Guide** — Component-by-component verification documentation (`docs/06_testing_guide.md`).

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
                              8 pages
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

### 💻 System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Disk Space** | ~5 GB (code + models + Docker images) | ~15 GB (with ingested data + embeddings) |
| **RAM** | 8 GB | 16 GB+ (LLM + Neo4j + Qdrant in parallel) |
| **GPU VRAM** | Not required (CPU mode) | 8 GB+ (RTX 3060 or higher for fast ingestion) |
| **CPU** | 4 cores | 8+ cores (parallel ingestion + services) |
| **OS** | Windows 10/11, Linux, macOS | Windows 11 / Ubuntu 22.04+ |
| **Docker** | Required for Qdrant + Neo4j | Docker Desktop with WSL2 backend |
| **LLM Backend** | Ollama (primary) or LM Studio | Ollama with `qwen3:8b` (~5 GB model) |

> **Note**: EcoloGRAPH runs 100% locally. No cloud API keys or internet required after initial setup.
> Both **Ollama** (recommended, port `11434`) and **LM Studio** (port `1234`) are supported —
> just set `LOCAL_LLM_BASE_URL` in `.env` to your preferred backend.

### 1. Clone and Install

```bash
git clone https://github.com/Dejimelana/EcoloGRAPH.git
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

# Start Ollama (see LLM Connection Guide below)
ollama pull qwen3:8b

# Full pipeline
python scripts/ingest.py data/raw/
streamlit run scripts/app.py
```

### 4. Add Your Papers

Place PDF files in `data/raw/` and run the ingestion pipeline:

```bash
# Default: uses model from config, thinking OFF, 2048 max tokens
python scripts/ingest.py data/raw/

# Custom: specific model, with thinking, higher timeout
python scripts/ingest.py data/raw/ --model qwen3:14b --thinking --max-tokens 4096 --timeout 180
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | config (qwen3:8b) | Ollama model for extraction |
| `--thinking` | OFF | Enable thinking mode (slower, better quality) |
| `--max-tokens` | 2048 | Max tokens per LLM response |
| `--timeout` | 120s | LLM request timeout |
| `--skip-extract` | OFF | Skip LLM extraction |
| `--skip-graph` | OFF | Skip Neo4j |
| `--skip-vectors` | OFF | Skip Qdrant |

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

EcoloGRAPH uses [Ollama](https://ollama.ai/) as its local LLM backend with a **dual-model architecture**:

| Role | Purpose | When | Config Variable |
|------|---------|------|----------------|
| **Ingestion** | Entity extraction from PDFs | `python scripts/ingest.py` | `INGESTION_LLM_MODEL` |
| **Reasoning** | Chat agent, graph queries | Streamlit chat | `REASONING_LLM_MODEL` |

### Installing Ollama

1. Download and install from [ollama.ai](https://ollama.ai/) (Windows, Linux, Mac)
2. Pull the recommended model for your GPU:

### Choosing a Model

| GPU VRAM | Recommended Model | Pull Command | Notes |
|----------|-------------------|-------------|-------|
| **6 GB** | `qwen3:1.7b` | `ollama pull qwen3:1.7b` | Fastest, basic extraction quality |
| **8 GB** | `qwen3:8b` ⭐ | `ollama pull qwen3:8b` | **Best balance** of speed and quality |
| **12 GB** | `qwen3:8b` | `ollama pull qwen3:8b` | Same model, more headroom for context |
| **16 GB+** | `qwen3:14b` | `ollama pull qwen3:14b` | Best quality, slower extraction |
| **CPU only** | `qwen3:1.7b` | `ollama pull qwen3:1.7b` | Works but significantly slower |

> **Recommended**: `qwen3:8b` for most setups. It provides excellent scientific entity extraction at reasonable speed (~5-15s per batch). Use the same model for both ingestion and reasoning unless you have 16GB+ VRAM.

3. Ollama starts automatically on `http://localhost:11434/v1`
4. Configure `.env`:

```env
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
INGESTION_LLM_MODEL=qwen3:8b
REASONING_LLM_MODEL=qwen3:8b
```

### Qwen3 Thinking Mode

Qwen3 models feature a **thinking mode** that improves reasoning at the cost of speed:

| Phase | Thinking | Why |
|-------|----------|-----|
| Ingestion (default) | **OFF** (`/no_think`) | Speed: process 100 papers overnight |
| Ingestion + `--thinking` | **ON** | Quality: ambiguous/complex entities |
| Chat / Agent | **ON** (auto) | Quality: complex multi-tool reasoning |

### Alternative: Cloud API

Any OpenAI-compatible endpoint works (Together AI, Groq, OpenRouter):

```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=your-model-name
LOCAL_LLM_BASE_URL=https://api.your-provider.com/v1
```

Place your API key in `config/api-key`.

### Verifying the Connection

```bash
# Quick diagnostic (tests raw speed, tool calling, full agent)
python scripts/diagnose_agent.py

# Compare models on extraction quality
python scripts/test_ollama_models.py

# Terminal chat
python scripts/chat_demo.py
```

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
│   └── prompts/               # LLM prompt templates (with /no_think support)
├── data/
│   └── raw/                   # Your PDFs go here
├── docs/                      # Full documentation (6 files)
│   ├── 01_project_documentation.md
│   ├── 02_development_log.md
│   ├── 03_tutorial.md
│   ├── 04_architecture_diagrams.md
│   ├── 05_module_summary.md
│   └── 06_testing_guide.md    # Component verification guide
├── scripts/
│   ├── app.py                 # Streamlit UI entry point
│   ├── ingest.py              # Ingestion pipeline CLI (with --model, --thinking)
│   ├── chat_demo.py           # Terminal chat agent
│   ├── diagnose_agent.py      # Ollama + agent diagnostic
│   ├── test_ollama_models.py  # Model comparison benchmark
│   ├── rebuild_fts.py         # FTS5 index rebuild tool
│   ├── verify_setup.py        # Installation verification
│   └── demo_pipeline.py       # Paper selection demo
├── src/
│   ├── core/                  # Config (dual-model), schemas, LLM client, 43 domains
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

8-page dashboard with dark/light themes and glassmorphism design:

| Page | Features |
|------|----------|
| 📊 Dashboard | Metrics, service status, domain distribution charts |
| 💬 Chat | Interactive LangGraph agent with tool call visualization |
| 📄 Papers | Paper browser with PDF.js viewer, metadata, and abstracts |
| 🕸️ Graph | Full-width interactive graph with node-click details panel, species co-occurrence edges, chunk viewer with entity highlighting |
| 🔍 Search | Hybrid search with clickable results and species common-name search |
| 🧬 Species | GBIF-powered species explorer with common name resolution, distribution maps, and occurrence records |
| ✅ Taxonomy Explorer | GBIF name resolver, batch validation, taxonomic statistics, Neo4j species browser |
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
| Aquatic | marine ecology, coral reef ecology, freshwater ecology, deep sea ecology, oceanography |
| Terrestrial | forest_ecology, soil ecology, urban ecology, landscape ecology, agroecology |
| Organisms | entomology, ornithology, herpetology, mammalogy, microbiology, phycology, mycology |
| Processes | population ecology, community ecology, ecosystem ecology, molecular ecology, chemical ecology |
| Applied | conservation, restoration ecology, wildlife management, environmental monitoring |
| Climate | climate change ecology, paleoecology |
| Technology | bioinformatics, remote sensing, machine learning, soundscape ecology |
| Biology | genetics, evolution, taxonomy, physiology, ethology, biotic interactions |
| Earth | geology, biogeography, limnology |
| General | general ecology, invasive species, parasitology, toxicology, fisheries |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `fts5: missing row N` | Run `python scripts/rebuild_fts.py` to rebuild the FTS5 index |
| `No model loaded` | Start Ollama: `ollama pull qwen3:8b` and ensure it's running |
| Qwen3 empty responses | Check `reasoning` field — use `python scripts/diagnose_agent.py` |
| Qwen3 `<think>` in JSON | Prompts include `/no_think`; add `--thinking` flag if you need it |
| Qdrant connection error | Ensure Docker is running: `docker start qdrant` |
| Neo4j connection error | Ensure Docker is running: `docker start neo4j` |
| Slow chat responses | Try smaller ingestion model: `--model qwen3:1.7b` |
| PDF viewer blank | PDF.js renders on canvas — check browser console for errors |
| Graph not loading | Install: `pip install streamlit-agraph` |

---

## 🧪 Running Tests

```bash
# Verify installation
python scripts/verify_setup.py

# Run all 58 automated tests
python -m pytest tests/ -v

# Test LLM connection
python scripts/diagnose_agent.py

# Run full component-by-component verification
# See docs/06_testing_guide.md for details
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
| [Testing Guide](docs/06_testing_guide.md) | Component-by-component verification |
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

**Built for the ecological research community**
