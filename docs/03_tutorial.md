# EcoloGRAPH — User Tutorial

> From zero to running the full system: ingestion, search, species lookup, and hypothesis generation.

---

## Prerequisites

- Python 3.11+ with the `ecolograph` conda environment
- LM Studio or Ollama running locally (for entity extraction and agent)
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

---

## 1. Setting Up

### 1.1 API Key

Place your LM Studio API key in `config/api-key`:

```
sk-lm-your-key-here
```

This is auto-loaded by the system — no `.env` file needed.

### 1.2 Verify Installation

```bash
conda activate ecolograph
python -m pytest tests/test_integration.py -v
```

Expected: **33/33 tests passing**.

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

### 3.3 Search (🔍)

1. Type a query: *"microplastics impact on marine organisms"*
2. Optionally filter by domain (dropdown)
3. Results show as cards with:
   - Title, score (BM25 + semantic combined)
   - Domain badge
   - Text snippet

### 3.4 Species Explorer (🐟)

1. Enter a scientific name: *"Gadus morhua"*
2. Click **Look up**
3. Three tabs appear:
   - **FishBase**: biology (max length, weight, trophic level, habitat)
   - **GBIF**: distribution (countries, year range, occurrence map)
   - **IUCN**: conservation status (threat category, population trend, threats)

### 3.5 Domain Lab (🔬)

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
🤖 Model: qwen3:14b (auto-detected)
🔧 Tools: 7 available

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
2. Open Species Explorer → enter name
3. FishBase tab for biology, GBIF for distribution, IUCN for conservation
4. Use Domain Lab to find which domains study this species

### 5.2 Cross-Domain Literature Review

1. Ingest papers from two related domains
2. Domain Lab → Cross-Domain Links → check affinity
3. Domain Lab → Generate Hypotheses → get research ideas
4. Search → find papers bridging both domains

### 5.3 Classify Unknown Abstracts

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
  --skip-extract    Skip LLM entity extraction
  --skip-graph      Skip Neo4j graph building
  --skip-vectors    Skip Qdrant vector indexing
  --chunk-size N    Chunk size in chars (default: 1000)
  --chunk-overlap N Overlap between chunks (default: 200)
```

### Chat Demo

```
python scripts/chat_demo.py [OPTIONS]

Options:
  --model MODEL     LLM model name (default: auto-detect)
  --base-url URL    LLM API URL (default: http://localhost:1234/v1)
  --temperature T   Sampling temperature (default: 0.1)
```

### Streamlit UI

```
streamlit run scripts/app.py
```

### Tests

```
python -m pytest tests/test_integration.py -v    # All 33 tests
python -m pytest tests/ -k "classify"             # Filter by name
```
