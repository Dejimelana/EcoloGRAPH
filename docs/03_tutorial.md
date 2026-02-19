# EcoloGRAPH — User Tutorial (v1.5.0)

> From zero to running the full system: ingestion, search, species lookup, graph exploration, and hypothesis generation.

---

## Prerequisites

- Python 3.11+ with the `ecolograph` conda environment
- **Ollama** running locally (for entity extraction and agent) — [ollama.ai](https://ollama.ai/)
- Docker Desktop (for Qdrant and Neo4j)
- PDFs in `data/raw/` directory

### Starting Docker services

1. Open **Docker Desktop** from Windows Start menu
2. Wait until the tray icon shows "Docker is running"
3. Run:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
```

> To stop/restart later: `docker start qdrant neo4j`

### Starting Ollama

```bash
# Pull the recommended model
ollama pull qwen3:8b

# Ollama starts automatically on http://localhost:11434/v1
```

---

## 1. Setting Up

### 1.1 Environment Configuration

Configure `.env` with your Ollama settings:

```env
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
INGESTION_LLM_MODEL=qwen3:8b
REASONING_LLM_MODEL=qwen3:8b
```

### 1.2 Verify Installation

```bash
conda activate ecolograph
python -m pytest tests/ -v
```

Expected: **58/58 tests passing**.

### 1.3 Diagnostics

```bash
# Quick test of Ollama connection + agent pipeline
python scripts/diagnose_agent.py

# Compare different models on extraction quality
python scripts/test_ollama_models.py
```

---

## 2. Ingesting Papers

### 2.1 Basic Ingestion (SQLite only — no LLM/Qdrant/Neo4j needed)

```bash
python scripts/ingest.py data/raw/ --skip-extract --skip-graph --skip-vectors
```

This parses all PDFs, classifies them into domains, and indexes them in SQLite FTS5.

### 2.2 Full Ingestion (with LLM entity extraction)

```bash
python scripts/ingest.py data/raw/ --skip-graph --skip-vectors
```

This adds LLM-powered extraction of species, measurements, locations, and relationships.

### 2.3 With Qdrant (semantic search)

```bash
# Start Qdrant first
docker run -p 6333:6333 qdrant/qdrant

python scripts/ingest.py data/raw/ --skip-graph
```

### 2.4 Full Pipeline (everything)

```bash
python scripts/ingest.py data/raw/

# Custom: specific model, with thinking, higher timeout
python scripts/ingest.py data/raw/ --model qwen3:14b --thinking --max-tokens 4096 --timeout 180
```

### 2.5 Pipeline Output

```
📄 Found 15 PDF(s) to process
[1/15] Processing: coral-reef-fish-diversity.pdf
  📖 Parsed: 12 pages, 8 sections, 3 tables
  🏷️  Domain: coral_reef_ecology (78%)
  📦 Chunks: 42
  💾 Indexed in SQLite
  🧬 Extracted 23 entities from 42 chunks
  ⏱️  Completed in 45.2s

📊 INGESTION COMPLETE
  Processed: 15/15 PDFs
  Chunks:    580
  SQLite:    15 papers indexed
```

---

## 3. Using the Streamlit UI

### 3.1 Launch

```bash
streamlit run scripts/app.py
```

Opens at `http://localhost:8501`.

### 3.2 Dashboard (📊)

The landing page shows:
- **Metric cards**: papers indexed, active domains, tools available
- **Service status**: green/red dots for SQLite, Qdrant, Neo4j
- **Domain distribution**: bar chart of papers per domain
- **Architecture overview**: all system modules

### 3.3 Graph Explorer (🕸️)

The interactive knowledge graph visualizer shows papers, species, and domains as nodes.

1. Select a **layout** (Barnes-Hut, Force Atlas, Hierarchical, etc.)
2. Choose **data source**: Neo4j or PaperIndex
3. Adjust **max nodes** with the slider

**Node interaction**:
- Click any node → **scroll down** to see the details panel
- **Paper nodes** show: title, year, abstract, species, locations, and source chunks with entity highlighting
- **Species nodes** show: papers mentioning the species, co-occurring species, and evidence chunks where the species appears (highlighted in green)
- **Domain nodes** show: papers classified in that domain

**Species co-occurrence**:
- Yellow `co-occurs (N)` edges link species that appear in the same papers
- Edge thickness is proportional to the number of shared papers

### 3.4 Search (🔍)

Two tabs: **Papers** and **Species**.

**Papers tab**:
1. Type a query: *"microplastics impact on marine organisms"*
2. Optionally filter by domain (dropdown)
3. Results show as cards with title, score, domain badge, and snippet
4. Click 📖 to navigate to the paper's detail page

**Species tab**:
1. Search by **scientific name** or **common name**: *"cod"*, *"wolf"*, *"Quercus"*
2. Results show matching species with paper count and family badge
3. Click any **📄 paper title** to navigate to that paper

### 3.5 Species Explorer (🧬)

1. Enter a scientific or common name: *"Gadus morhua"*, *"cod"*, *"springtail"*
2. The search triggers automatically on Enter
3. Common names resolve automatically: *"cod" → Gadus morhua*
4. Three tabs appear:
   - **Overview & Taxonomy**: full taxonomy hierarchy, common names, conservation status
   - **Distribution Map**: GBIF occurrence records on a world map
   - **Occurrence Records**: downloadable table of georeferenced records

### 3.6 Taxonomy Explorer (✅)

Three sub-tabs:

**Database Species**: Browse all species extracted from your papers (Neo4j):
- Filter by minimum paper count or name search
- Download as CSV for external analysis

**Name Resolver**: Validate any species or common name against GBIF:
1. Enter a name: *"springtail"*, *"Atlantic cod"*, *"Quercus robur"*
2. See resolved canonical name, family, kingdom, rank, and confidence score
3. One-click link to GBIF species page

**Taxonomy Stats**: Visualize your corpus:
- Family distribution bar chart
- Validation status breakdown
- Paper count per species

### 3.7 Domain Lab (🔬)

Three sub-tabs:

**Classify Text**: Paste any scientific text → get domain scores as horizontal bars

**Cross-Domain Links**: Select two domains → see affinity score (0–100%)

**Generate Hypotheses**:
1. Enter topic: *"Effect of ocean acidification on coral reef fish"*
2. Select focus domains: `coral_reef_ecology`, `conservation`
3. Click **Generate** → hypothesis cards with rationale and suggested experiments

---

## 4. Using the Chat Agent (Terminal)

```bash
python scripts/chat_demo.py
```

```
🌿 EcoloGRAPH Chat (beta)
🤖 Model: qwen3:8b (auto-detected)
🔧 Tools: 8 available

You: What species are most studied in coral reef ecology?
🧭 Route: research
🔧 Using: search_papers("coral reef species most studied")
📊 Found 8 results
...
EcoloGRAPH: Based on the indexed papers, the most frequently studied species...

You: /help
You: /info
You: /quit
```

---

## 5. Common Workflows

### 5.1 Build a Species Profile

1. Ingest papers mentioning the species
2. Open Species Explorer → enter name (scientific or common)
3. GBIF tab for distribution, Overview for taxonomy and conservation
4. Use Graph Explorer to see co-occurring species
5. Use Taxonomy Explorer to validate taxonomic information

### 5.2 Cross-Domain Literature Review

1. Ingest papers from two related domains
2. Domain Lab → Cross-Domain Links → check affinity
3. Domain Lab → Generate Hypotheses → get research ideas
4. Search → find papers bridging both domains

### 5.3 Explore Species Interactions

1. Open Graph Explorer → look for yellow co-occurrence edges
2. Click a species node → scroll down to see evidence chunks
3. Use Taxonomy Explorer → Name Resolver to validate species names

### 5.4 Classify Unknown Abstracts

1. Domain Lab → Classify Text
2. Paste abstract
3. View domain distribution bars
4. Primary domain + confidence tells you the fit

---

## 6. CLI Reference

### Ingestion Pipeline

```
python scripts/ingest.py <path> [OPTIONS]

Arguments:
  path              PDF file or directory

Options:
  --model <name>    Ollama model for extraction (default: from config)
  --thinking        Enable thinking mode (slower, better quality)
  --max-tokens <n>  Max tokens per LLM response (default: 2048)
  --timeout <s>     LLM request timeout (default: 120s)
  --skip-extract    Skip LLM entity extraction
  --skip-graph      Skip Neo4j graph building
  --skip-vectors    Skip Qdrant vector indexing
```

### Chat Demo

```
python scripts/chat_demo.py [OPTIONS]

Options:
  --model MODEL     LLM model name (default: auto-detect from config)
  --base-url URL    LLM API URL (default: http://localhost:11434/v1)
  --temperature T   Sampling temperature (default: 0.1)
```

### Diagnostics

```
python scripts/diagnose_agent.py    # Test Ollama + agent pipeline
python scripts/test_ollama_models.py # Compare models
python scripts/rebuild_fts.py       # Rebuild corrupted FTS5 index
python scripts/verify_setup.py      # Check installation
```

### Streamlit UI

```
streamlit run scripts/app.py
```

### Tests

```
python -m pytest tests/ -v                       # All 58 tests
python -m pytest tests/test_integration.py -v     # Integration tests only
python -m pytest tests/ -k "classify"             # Filter by name
```
