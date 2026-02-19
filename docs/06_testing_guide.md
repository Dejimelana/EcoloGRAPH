# EcoloGRAPH Testing Guide

Component-by-component verification for EcoloGRAPH v1.5.0.

---

## 1. Setup Verification

Verifies Python imports, dependencies, configuration, and directory structure.

```bash
python scripts/verify_setup.py
```

**Expected output:**
```
✅ src.core
✅ src.ingestion
✅ src.extraction
✅ src.graph
...
🎉 All checks passed!
```

**Troubleshooting:**
| Issue | Fix |
|-------|-----|
| `❌ src.xxx: ModuleNotFoundError` | `pip install -r requirements.txt` |
| `❌ Config error` | Copy `.env.example` to `.env` and edit credentials |
| `⚠️ streamlit (optional)` | `pip install streamlit streamlit-agraph` |

---

## 2. LLM Connection (Ollama)

Tests raw Ollama speed, tool calling support, and the full QueryAgent pipeline.

```bash
# Ensure Ollama is running and model is loaded
ollama list        # Should show qwen3:8b or your model
ollama ps          # Should show model in memory

# Run diagnostic
python scripts/diagnose_agent.py
```

**Expected output:**
```
TEST 1: Raw Ollama (qwen3:8b)
  Time: 3.2s
  Tokens: 45
  Response: Coral reefs host...

TEST 2: Tool Calling (qwen3:8b)
  Time: 5.1s
  Tool calls: 1
    -> search_papers({"query": "coral reef bleaching"})

TEST 3: Full QueryAgent Pipeline
  Model: qwen3:8b
  Tools: 8
  [2.1s] Routing: llm_agent
  [8.5s] Tool call: search_papers
  [12.3s] ANSWER (450 chars): ...

SUMMARY
  Raw Ollama:     3.2s
  Tool calling:   5.1s
  Full agent:     12.3s
  Agent overhead: 3.8x
```

**Troubleshooting:**
| Issue | Fix |
|-------|-----|
| `ERROR: Connection refused` | Start Ollama: `ollama serve` (Windows service may auto-start) |
| `ERROR: 404 Not Found` | Pull model: `ollama pull qwen3:8b` |
| No tool calls emitted | Model may not support function calling — try `qwen3:8b` or `qwen2.5:7b` |
| Agent overhead > 10x | Check Neo4j/Qdrant connectivity; slow tool execution inflates time |

---

## 3. Model Comparison

Compares Ollama models on scientific entity extraction quality, speed, and JSON compliance.

```bash
python scripts/test_ollama_models.py
```

**What it tests:**
- Entity extraction from a sample ecological text (species, measurements, locations)
- JSON format compliance (valid JSON output from LLM)
- Speed (tokens/second)
- Scientific accuracy (correct species binomials, units, values)

**Expected output (per model):**
```
Model: qwen3:8b
  ⏱️  Time: 4.2s (38 tok/s)
  ✅ Valid JSON
  📊 Species: 4, Measurements: 6, Locations: 3
  🎯 Accuracy score: 85%
```

**When to run:** Before switching models, to compare extraction quality.

---

## 4. Extraction Quality

Tests the entity extractor on sample text with known entities.

```bash
python scripts/test_extraction.py
```

**What it tests:**
- `EntityExtractor` initialization with LLM client
- Prompt template loading (`extraction_system.txt`, `extraction_paper.txt`)
- JSON response parsing (including `<think>` tag stripping)
- Entity type detection: species, measurements, locations, ecological relationships

**Troubleshooting:**
| Issue | Fix |
|-------|-----|
| `JSON parse error` | Check that extraction prompts include `/no_think` |
| Empty entity list | Increase `--max-tokens` (default: 2048) |
| `<think>` tags in output | LLM client should strip them automatically (v1.5.0+) |

---

## 5. Full Pipeline (End-to-End)

Tests the complete ingestion pipeline on a sample PDF.

```bash
# Quick test with 1 paper, no graph/vectors (fastest)
python scripts/ingest.py data/raw/ --skip-graph --skip-vectors --max-papers 1

# Full pipeline test (requires Qdrant + Neo4j running)
python scripts/test_full_pipeline.py
```

**What it tests:**
1. PDF parsing (Docling → PyMuPDF fallback)
2. Text chunking (section-aware, size-limited)
3. Domain classification (43 domains)
4. Entity extraction via LLM
5. SQLite FTS5 indexing
6. Qdrant vector storage
7. Neo4j graph building

**Expected output:**
```
📄 Processing: example_paper.pdf
  📑 Parsed: 15 pages, "Effects of warming on coral reefs"
  📦 Chunked: 42 chunks (avg 380 tokens)
  🏷️  Classified: marine_ecology (0.92), coral_reef_ecology (0.88)
  🧬 Extracted: 18 entities (5 species, 8 measurements, 5 locations)
  📇 Indexed in SQLite FTS5
  🔢 Stored 42 vector embeddings
  🕸️  Added to Neo4j graph
  ⏱️  Completed in 125.3s
```

---

## 6. Search & FTS5

Tests SQLite FTS5 full-text search and hybrid retrieval.

```bash
# Query chunks from the command line
python scripts/query_chunks.py list                          # List all papers
python scripts/query_chunks.py search "coral bleaching" -n 5  # Search

# If FTS5 is corrupted:
python scripts/rebuild_fts.py
```

**Manual verification (Python):**
```python
import sqlite3
conn = sqlite3.connect("data/paper_index.db")

# Check paper count
count = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
print(f"Papers: {count}")

# Test FTS5 search
results = conn.execute(
    "SELECT doc_id, title FROM papers_fts WHERE papers_fts MATCH 'species' LIMIT 5"
).fetchall()
print(f"FTS results: {len(results)}")
```

**Troubleshooting:**
| Issue | Fix |
|-------|-----|
| `fts5: missing row N` | Run `python scripts/rebuild_fts.py` |
| Papers table empty | Re-run ingestion: `python scripts/ingest.py data/raw/` |
| FTS returns 0 results | Check that data was indexed (not `--skip-extract`) |

---

## 7. Graph Database (Neo4j)

Verifies Neo4j knowledge graph population.

```bash
# Ensure Neo4j is running
docker start neo4j
```

**Browser verification:**
1. Open Neo4j Browser: `http://localhost:7474`
2. Login: `neo4j` / `password` (or as set in `.env`)

**Cypher verification queries:**
```cypher
// Count all nodes
MATCH (n) RETURN labels(n) AS type, count(n) AS count

// View species
MATCH (s:Species) RETURN s.name, s.common_name LIMIT 10

// View measurements
MATCH (m:Measurement) RETURN m.parameter, m.value, m.unit LIMIT 10

// View paper-species relationships
MATCH (p:Paper)-[:MENTIONS]->(s:Species)
RETURN p.title, collect(s.name) AS species LIMIT 5

// View citation network
MATCH (p:Paper)-[:CITES]->(c:Citation)
RETURN p.title, count(c) AS citations
ORDER BY citations DESC LIMIT 5
```

**Expected:** At least `Paper`, `Species`, `Measurement`, `Location` node types, with `MENTIONS`, `MEASURED_IN`, `LOCATED_IN` relationships.

---

## 8. Agent & Chat

Tests the interactive query agent.

```bash
# Terminal chat (quick)
python scripts/chat_demo.py

# Full web UI
streamlit run scripts/app.py
# Then open http://localhost:8501 → Chat page
```

**Terminal test queries:**
```
> What species are in the database?
> Find papers about coral reefs
> What measurements exist for temperature?
```

**Expected behavior:**
1. Agent routes query (regex or LLM-based)
2. Tools are called (search_papers, query_graph, etc.)
3. Response synthesized from tool results
4. Sources cited with paper IDs

---

## 9. Automated Tests (pytest)

Runs unit and integration test suites.

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests (requires services)
python -m pytest tests/test_integration.py -v

# Specific test
python -m pytest tests/test_token_utils.py -v
```

**Test coverage:**

| Test File | What it covers |
|-----------|---------------|
| `tests/unit/` | Schema validation, token utils, domain classifier |
| `tests/test_token_utils.py` | Context window calculation, batching logic |
| `tests/test_paper_extraction.py` | Paper-level entity extraction |
| `tests/test_integration.py` | Full pipeline integration (PDF → Graph) |

---

## Quick Reference: Complete Verification Sequence

Run all checks in order:

```bash
# 1. Setup
python scripts/verify_setup.py

# 2. Services running?
docker ps                    # Qdrant + Neo4j
ollama list                  # Models available
ollama ps                    # Model loaded

# 3. LLM works?
python scripts/diagnose_agent.py

# 4. Automated tests
python -m pytest tests/ -v

# 5. Ingestion pipeline
python scripts/ingest.py data/raw/ --max-papers 1

# 6. Search works?
python scripts/query_chunks.py list

# 7. Chat works?
python scripts/chat_demo.py

# 8. UI works?
streamlit run scripts/app.py
```

If all 8 steps pass, EcoloGRAPH is fully operational. ✅
