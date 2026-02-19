# EcoloGRAPH Development Log - Recent Updates

> Latest improvements and architectural changes (February 2026)

---

## v1.5.0 - Ollama Migration & Dual-Model Architecture (February 19, 2026)

### 🎯 Major Architecture Change

**Problem Identified:**
- LM Studio suffers memory leaks after 3000+ requests (~90GB leak)
- No native thinking model support (Qwen3's `reasoning` field not handled)
- Manual model management (load/unload via GUI)
- Auto-restart workaround was fragile

**Solution Implemented:**
Migrated to **Ollama** as primary LLM backend with **dual-model architecture**.

### Changes Made

#### 1. Dual-Model Configuration

**File**: `src/core/config.py`

```python
class LLMSettings(BaseSettings):
    ingestion_model: str = "qwen3:8b"   # Fast, for entity extraction
    reasoning_model: str = "qwen3:8b"   # Deep, for agent/chat
```

**Environment variables**:
```env
INGESTION_LLM_MODEL=qwen3:8b    # Entity extraction (with /no_think)
REASONING_LLM_MODEL=qwen3:8b    # Chat agent (thinking enabled)
```

#### 2. Qwen3 `reasoning` Field Support

**File**: `src/core/llm_client.py`

**Problem**: Qwen3 via Ollama puts thinking output in a `reasoning` JSON field instead of `content`.

**Fix**: LLM client now reads both fields:
```python
content = choice["message"].get("content", "")
if not content:
    # Qwen3 via Ollama: output is in reasoning field
    content = choice["message"].get("reasoning", "")

# Strip <think> tags if present
content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
```

#### 3. Thinking Mode Control

**Files**: `config/prompts/extraction_system.txt`, `config/prompts/extract_citations.txt`

Added `/no_think` directive to extraction prompts — disables Qwen3 thinking mode during high-volume extraction, saving ~400 tokens per request.

**Strategy**:
| Phase | Thinking Mode | Reason |
|-------|--------------|--------|
| Ingestion (extraction) | OFF (`/no_think`) | Speed: process 100 papers overnight |
| Ingestion (with `--thinking`) | ON | Quality: ambiguous entities |
| Agent/Chat | ON | Quality: complex reasoning |

#### 4. CLI Ingestion Parameters

**File**: `scripts/ingest.py`

New command-line flags for full control:
```bash
python scripts/ingest.py data/raw/ \
  --model qwen3:8b \
  --max-tokens 2048 \
  --timeout 120 \
  --thinking          # Enable thinking mode
```

#### 5. Graph Builder Schema Fixes

**File**: `src/graph/graph_builder.py`

Fixed 4 schema mismatches that caused `AttributeError` during graph building:

| Bug | Fix |
|-----|-----|
| `species.common_names` | → `species.common_name` (singular) |
| `measurement.min_value` | → `measurement.value_min` |
| `measurement.max_value` | → `measurement.value_max` |
| `measurement.std_error` | → `measurement.std_dev` |

#### 6. Diagnostic Tools

**New scripts**:
- `scripts/diagnose_agent.py` — Tests raw Ollama speed, tool calling, full agent
- `scripts/test_ollama_models.py` — Compare models: extraction quality, speed, JSON compliance
- `scripts/rebuild_fts.py` — Safe FTS5 index rebuild

### Performance Impact

| Metric | LM Studio (v1.4.0) | Ollama (v1.5.0) | Change |
|--------|--------------------|--------------------|--------|
| **Memory leaks** | Every ~3000 requests | None | ✅ Fixed |
| **Thinking overhead** | N/A | Eliminated with `/no_think` | **-40% time** |
| **Schema errors** | 4 AttributeErrors | 0 | ✅ Fixed |
| **Model switching** | Manual (GUI) | CLI flag (`--model`) | ✅ Easy |

---

## v1.4.0 - Paper-Based Entity Extraction (February 11, 2026)

### 🎯 Major Architecture Change

**Problem Identified:**
- Entity extraction was processing chunks individually (chunk-by-chunk)
- Lost cross-chunk context ("the species" couldn't be resolved)
- Context overflow crash when chunks exceeded model limits (~2048 tokens)
- Inefficient: 80 LLM calls for typical paper with 80 chunks

**Solution Implemented:**
Redesigned extraction to treat **paper as functional unit**, not individual chunks.

### Changes Made

#### 1. Token Estimation Utilities

**File**: `src/core/token_utils.py` (NEW)

**Purpose**: Validate context window before LLM calls

```python
estimate_tokens(text)  # ~1 token per 4 chars + 10% safety margin
fits_in_context(text, context_window=2048)  # Boolean check
estimate_safe_batch_size(chunks, context_window=2048)  # Auto-calculate batches
```

#### 2. Paper-Level Extraction Prompt

**File**: `config/prompts/extraction_paper.txt` (NEW)

**Features**:
- Includes paper metadata (title, authors, year) for context
- Processes multiple sequential chunks together
- Instructions for cross-reference resolution
- Entity consolidation and deduplication

#### 3. EntityExtractor Enhancement

**File**: `src/extraction/entity_extractor.py`

**New Method**: `extract_from_paper()` ⭐ RECOMMENDED

**Strategy**:
1. Estimate total tokens for entire paper
2. If fits in context → Process all chunks in ONE LLM call
3. If exceeds context → Automatically batch into multiple calls
4. Consolidate results

**Helper Methods**:
- `_format_chunks_for_paper_extraction()` - Formats all chunks into paper prompt
- `_extract_from_full_paper()` - Single-call extraction
- `_extract_from_paper_batched()` - Auto-batching with size calculation

#### 4. Ingestion Pipeline Update

**File**: `scripts/ingest.py`

Changed from:
```python
extraction_results = entity_extractor.extract_from_chunks(chunks)  # 80 calls
for result in extraction_results:
    graph_builder.add_extraction_result(doc_id, result)
```

To:
```python
extraction_result = entity_extractor.extract_from_paper(  # 1 result, auto-batched
    chunks,
    paper_metadata={'title': doc.title, 'authors': doc.authors, 'year': doc.year}
)
graph_builder.add_extraction_result(doc_id, extraction_result)
```

### Performance Impact

| Metric | Old (Chunk-by-Chunk) | New (Paper-Based) | Improvement |
|--------|---------------------|-------------------|-------------|
| **LLM Calls** | 750 | 220 | **-70%** |
| **Total Time** | 125 min | 45 min | **-64%** |
| **Context Overflows** | 15 crashes | 0 crashes | **-100%** |
| **Cross-References** | 0 resolved | 127 resolved | ✅ NEW |

*(Test dataset: 10 scientific papers, avg 75 chunks each)*

---

## v1.4.0 - LM Studio Auto-Restart System (February 11, 2026)

### Problem

LM Studio enters corrupted state after context overflow:
1. Chunk exceeds context → LLM hangs → Timeout (120s)
2. GPU state corrupted → All subsequent requests fail (HTTP 400)
3. Manual restart required

### Solution

**File**: `src/core/lm_studio_manager.py` (NEW)

**Features**:
- Process detection (finds LM Studio by name)
- Health check (pings `/v1/models` endpoint)
- Graceful stop → Force kill fallback
- Auto-restart with readiness wait

**Integration**: `src/extraction/entity_extractor.py`

```python
# In extract_from_chunks():
consecutive_failures = 0
for chunk in chunks:
    try:
        result = extract(chunk)
        consecutive_failures = 0
    except Exception:
        consecutive_failures += 1
        if consecutive_failures >= 3:
            lm_studio_manager.restart_and_wait(timeout=60)
            consecutive_failures = 0
```

**Expected Logs**:
```
❌ Chunk 46: Failed (1/3)
❌ Chunk 47: Failed (2/3)
❌ Chunk 48: Failed (3/3)
⚠️  Detected 3 consecutive failures!
🔄 Restarting LM Studio...
✅ Ready in 28s
✅ Chunk 49: Extracted 5 entities
```

---

## v1.4.0 - Metadata Extraction Improvements (February 11, 2026)

### Issue

Docling sometimes fails to extract title/abstract, returns:
- Title: `<!-- image -->`
- Abstract: None
- Year: Missing (caused crash)

### Solution

#### 1. Added `year` Field

**File**: `src/ingestion/pdf_parser.py`

```python
@dataclass
class ParsedDocument:
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str | None = None
    year: int | None = None  # ← NEW
```

#### 2. Enhanced PyMuPDF Fallback

**File**: `scripts/ingest.py` - `enhance_metadata_with_pymupdf()`

**Improvements**:
- Detects bad titles: `<!-- image -->`, `"image"`, too short
- Extracts year from:
  - PDF metadata (`creationDate`)
  - Text patterns: `(2024)`, `©2024`, `Published: 2024`, `2024`
- Better logging (INFO level instead of DEBUG)

**Example Output**:
```
📝 Title: <!-- image -->
✅ Enhanced title via PyMuPDF: Wildfire response of forest species...
✅ Enhanced year via PyMuPDF: 2024
```

---

## v1.4.0 - Data Querying Tools (February 11, 2026)

### Problem

Users need to explore ingested data without UI:
- What papers are in the database?
- What chunks does a specific paper have?
- How to search by title when doc_id is a hash?

### Solution

**File**: `scripts/query_chunks_lite.py` (NEW)

Lightweight tool without heavy dependencies (no transformers/torch).

**Commands**:
```bash
# List all papers with doc_ids
python scripts/query_chunks_lite.py list

# View chunks from specific paper
python scripts/query_chunks_lite.py paper doc_abc123

# Search papers by title/source filename
python scripts/query_chunks_lite.py search Wildfire
```

**Features**:
- Direct Qdrant access (bypasses VectorStore class)
- Shows doc_id, title, source path, chunk count
- Displays first 5 chunks with preview
- No emoji characters (Windows encoding compatibility)

### Documentation Updates

**README.md**: Added complete "🔍 Querying Chunks and Data" section

**docs/01_project_documentation.md**: Added "8. Querying Your Data" section

Both include:
- Command-line tool usage
- Python API examples
- Streamlit UI guide
- Neo4j Cypher queries

---

## Key Insights

### Why Paper-Based Extraction Works Better

**Semantic Understanding**:
- LLM sees entire paper context
- Resolves pronouns ("the species" → actual species name)
- Identifies relationships across sections

**Efficiency**:
- Batch processing reduces overhead
- Fewer API calls = faster ingestion
- Better resource utilization

**Reliability**:
- Token validation prevents crashes
- Graceful degradation with auto-batching
- Auto-restart as backup safety net

### Context Overflow: Saturation, Not Time

**Common Misconception**: "Give the LLM more time"

**Reality**:
```
Chunk: 1500 chars
+ System prompt: 800 chars
+ Template: 300 chars
+ Output buffer: 500 tokens
= ~3000 tokens

Model context: 2048 tokens
3000 > 2048 → OVERFLOW (not timeout)
```

LLM doesn't say "this will take a while" — it **hangs** trying to process impossible input.

**Analogy**: Asking someone to memorize a phone book isn't a time problem, it's a capacity problem.

---

## Migration Notes

### Updating from v1.3.0

**Breaking Changes**: None (backwards compatible)

**New Features Available**:
1. Paper-based extraction (automatic via `ingest.py`)
2. LM Studio auto-restart (enabled by default)
3. Improved metadata extraction (automatic)
4. Query tools (`query_chunks_lite.py`)

**Recommended Actions**:
```bash
# Re-ingest papers to get improvements
python scripts/ingest.py data/raw/

# Old data is updated (not deleted)
# Same doc_id → overwrites chunk
# New doc_id → adds new paper
```

### Database Behavior

| Database | Behavior on Re-ingestion |
|----------|-------------------------|
| **SQLite** | `INSERT OR REPLACE` - Updates existing, adds new |
| **Qdrant** | `upsert()` - Same chunk_id overwrites, new adds |
| **Neo4j** | `MERGE` - Same doc_id updates, new creates |

**Safe to re-ingest**: Data is additive with overwrite by ID, never deleted.

---

## Performance Benchmarks

### Entity Extraction (10 papers, 750 chunks total)

**Before (v1.3.0)**:
- Method: Chunk-by-chunk
- LLM calls: 750
- Time: 125 minutes
- Failures: 15 context overflows
- Cross-references: 0

**After (v1.4.0)**:
- Method: Paper-based with auto-batching
- LLM calls: 220 (avg 22 per paper)
- Time: 45 minutes
- Failures: 0 (token validation + auto-restart)
- Cross-references: 127 resolved

**Result**: **~3x faster**, **zero crashes**, **better quality**

---

## Next Steps

Potential future improvements:
1. **Entity deduplication**: Improve consolidation across batches
2. **Adaptive tokenization**: Use tiktoken for exact counts
3. **Retry logic**: Retry individual failed batches
4. **Streaming extraction**: Process very large papers in streaming mode

---

## Contributors

Development: EcoloGRAPH Team
Testing: Bootcamp participants
APIs: FishBase, GBIF, IUCN Red List

## License

Open source tools used under their respective licenses.
LLM processing: Local models only (no data sent to cloud).
