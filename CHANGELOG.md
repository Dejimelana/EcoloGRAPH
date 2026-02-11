# Changelog

All notable changes to EcoloGRAPH will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-02-11

### 🚀 Major Features

- **Paper-Based Entity Extraction**: Complete redesign from chunk-by-chunk to paper-as-functional-unit
  - Added `src/core/token_utils.py` for context window validation
  - Created `config/prompts/extraction_paper.txt` for paper-level extraction
  - Implemented `extract_from_paper()` method with automatic batching
  - **70% reduction in LLM calls** (25 vs 80 for typical paper)
  - **Cross-chunk entity resolution** (resolves "the species" references)
  - **Zero context overflow crashes** with token validation

### 🎉 Added

- **LM Studio Auto-Restart System**:
  - New `src/core/lm_studio_manager.py` for process health monitoring
  - Detects corrupted state after context overflow
  - Automatically restarts LM Studio and waits for readiness
  - Integrated into entity extractor with 3-failure threshold
  
- **Data Querying Tools**:
  - New `scripts/query_chunks_lite.py` for lightweight chunk exploration
  - Commands: `list`, `paper <doc_id>`, `search <title>`
  - No heavy dependencies (direct Qdrant access)
  - Added comprehensive querying section to README and docs

- **Metadata Extraction Improvements**:
  - Added `year` field to `ParsedDocument` schema
  - Enhanced PyMuPDF fallback to detect bad titles (<!-- image -->)
  - Automatic year extraction from PDF metadata and text patterns
  - Better logging for metadata enhancement

### 📚 Documentation

- Updated README.md with "Querying Chunks and Data" section
- Created `docs/03_recent_updates.md` with v1.4.0 deep-dive
- Updated `docs/01_project_documentation.md` with querying guide
- Updated ARCHITECTURE.md with recent changes

### ⚡ Performance

- **Entity Extraction**: 3x faster (45min vs 125min for 10 papers)
- **LLM Efficiency**: 70% fewer API calls
- **Reliability**: 0 crashes vs 15 context overflows in testing

### 🐛 Fixed

- Fixed `'ParsedDocument' object has no attribute 'year'` crash
- Fixed Docling title extraction failures
- Fixed emoji encoding issues in Windows console tools
- Fixed vector store `.tolist()` compatibility

---

## [1.2.0] - 2026-02-09

### 🎉 Added

- **Graph Explorer V2**: Connected Papers-style interactive graph viewer
  - Click-to-explore nodes with sidebar metadata panel
  - Real-time chunk viewer with document text
  - Entity highlighting (species in red, locations in teal, methods in blue)
  - Lazy loading support with configurable node limits (50-500)
  - Color-coded entity legend
  
- **Theme Toggle System**:
  - Dark/light mode toggle button in sidebar
  - Professional light theme with clean aesthetics
  - Theme persists across pages via session state
  - Glassmorphism design in dark mode
  
- **Export Features**:
  - Export graph visualizations as PNG
  - Export chunks as JSON with metadata
  - Export papers as BibTeX for citations
  - Unified export button row component
  
- **Entity Highlighter Component**:
  - Color-coded highlighting for species, locations, and methods
  - HTML-based highlighting with customizable colors
  - Integrated into chunk viewer

### ⚡ Performance

- **Query Caching**: 5-10x speedup for repeated Neo4j queries using diskcache
  - 5-minute TTL for `get_paper_metadata()`
  - 10-minute TTL for `get_species_papers()`
  
- **Optimized Vector Indexing**: 2x faster chunk processing
  - Increased batch size from 100 to 200 chunks
  - Non-blocking upserts with `wait=False`
  - Detailed per-batch timing and throughput metrics
  
- **Lazy Graph Loading**: Configurable node limits prevent slowdowns
  - Default 100 nodes, adjustable 50-500 via slider
  - Prioritizes most-connected papers (ORDER BY mention_count DESC)

### 🐛 Fixed

- **AttributeError in Graph Building**: Fixed enum/string handling
  - `measurement.unit` now handles both enum and string values
  - `relation.relation_type` now handles both enum and string values
  - Prevents crashes during graph construction

### 📚 Documentation

- Updated README.md with v1.2.0 features and emoji sections
- Created CONTRIBUTING.md with development guidelines
- Created ARCHITECTURE.md with technical deep-dive
- Added comprehensive walkthrough for all optimizations

---

## [1.1.0] - 2026-02-08

### Added

- **Species Validation UI**: Dedicated page for species data quality checks
- **43 Scientific Domains**: Expanded from original 8 to 43 domains including:
  - AI/ML, Bioinformatics, Soundscape Ecology
  - Physiology, Ethology, Biotic Interactions
  - Geology, Biogeography
  
- **External Data Scrapers**:
  - FishBase API integration
  - GBIF species lookup
  - IUCN Red List status
  
- **Cross-Domain Inference Engine**:
  - Automatic hypothesis generation
  - Domain affinity scoring
  - Co-occurrence analysis

### Changed

- Improved PDF abstract parsing with artifact removal
- Enhanced domain classification with weighted multi-label support
- Optimized chunk viewer rendering

---

## [1.0.0] - 2026-01-22

### Added

- **Core System**:
  - PDF ingestion pipeline (Docling + PyMuPDF fallback)
  - Section-aware chunking
  - SQLite FTS5 full-text search (BM25)
  - Qdrant vector store (semantic search)
  - Neo4j knowledge graph
  - LangGraph query agent with 8 tools
  
- **Streamlit Dashboard**:
  - 📊 Dashboard page
  - 💬 Chat page with agent visualization
  - 📄 Papers page with PDF.js viewer
  - 🕸️ Graph Explorer (Pyvis)
  - 🔍 Search page
  - 🧬 Species explorer
  - 🔬 Classifier demo
  
- **Agent Tools**:
  - search_papers (hybrid search)
  - search_by_domain
  - search_related_papers
  - query_graph
  - classify_text
  - get_species_info
  
- **Hierarchical Query Router**:
  - Regex-based intent detection (80% of queries)
  - LLM fallback for complex queries
  - Direct answer agent for simple requests

---

## [Unreleased]

### Planned

- Distributed ingestion with Celery
- Real-time PDF watch folder
- REST API layer
- Multi-user authentication
- Prometheus + Grafana monitoring
- CI/CD with GitHub Actions
- Docker Compose deployment

---

## Migration Guide

### Upgrading to 1.2.0

1. **Install new dependency**:
   ```bash
   pip install streamlit-agraph
   ```

2. **Clear cache** (optional, for query cache):
   ```bash
   rm -rf .cache/neo4j_queries
   ```

3. **No data migration required** - All existing data is compatible

---

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full list of contributors.

---

## Links

- [GitHub Repository](https://github.com/yourusername/EcoloGRAPH)
- [Documentation](docs/)
- [Issue Tracker](https://github.com/yourusername/EcoloGRAPH/issues)
