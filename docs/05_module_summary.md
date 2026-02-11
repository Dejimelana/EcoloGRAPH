# EcoloGRAPH — Module Summary

> Complete reference of all modules, their categories, files, classes, and functionalities.

---

## Module Overview

| # | Module | Category | Files | Key Classes |
|---|--------|----------|-------|-------------|
| 1 | `core/` | Foundation | 4 | `Settings`, `LLMClient`, `DomainType`, Pydantic schemas |
| 2 | `ingestion/` | Data Input | 2 | `PDFParser`, `DocumentChunker` |
| 3 | `extraction/` | Data Processing | 2 | `DomainClassifier`, `EntityExtractor` |
| 4 | `search/` | Information Retrieval | 3 | `PaperIndex`, `RankedSearch`, `QueryLogger` |
| 5 | `retrieval/` | Information Retrieval | 2 | `VectorStore`, `HybridRetriever` |
| 6 | `graph/` | Knowledge Graph | 3 | `GraphBuilder`, Cypher templates, NetworkX analytics |
| 7 | `inference/` | Analysis | 2 | `CrossDomainLinker`, `InferenceProposer` |
| 8 | `scrapers/` | External Data | 3 | `FishBaseClient`, `GBIFOccurrenceClient`, `IUCNClient` |
| 9 | `enrichment/` | External Data | 4 | `CrossrefClient`, `SemanticScholarClient`, `TaxonomyResolver` |
| 10 | `agent/` | Orchestration | 3 | `QueryAgent`, `ToolRegistry` (8 tools) |
| 11 | `ui/` | Presentation | 9 | 7 Streamlit pages + theme |

---

## 1. Core (`src/core/`)

**Category**: Foundation — configuration, data models, LLM interface

| File | Class/Function | Description |
|------|---------------|-------------|
| `config.py` | `Settings` | Pydantic-based config loading from env vars + `config/api-key` |
| | `LLMSettings` | LLM provider, model, base_url, temperature |
| | `QdrantSettings` | Vector store connection parameters |
| | `Neo4jSettings` | Graph database connection parameters |
| | `_load_api_key_file()` | Auto-loads API key from `config/api-key` into `OPENAI_API_KEY` env var |
| `schemas.py` | `SourceReference` | Traceability: doc_id, page, position, chunk_id |
| | `SpeciesMention` | Species with scientific name, source reference |
| | `Measurement` | Quantitative data with units and source |
| | `Location` | Geographic location with coordinates |
| | `EcologicalRelation` | Typed relationship between entities |
| | `ExtractionResult` | Container for all extracted entities from a chunk |
| `domain_registry.py` | `DomainType` | Enum with 43 scientific domains |
| | `DomainRegistry` | Domain configs, keywords, weights |
| `llm_client.py` | `LLMClient` | Unified interface for local LLMs |
| | `LLMResponse` | Structured response with token counts |

### 43 Scientific Domains

| Category | Domains |
|----------|---------|
| Aquatic | marine_ecology, coral_reef_ecology, freshwater_ecology, deep_sea_ecology, oceanography |
| Terrestrial | forest_ecology, soil_ecology, urban_ecology, landscape_ecology, agroecology |
| Organisms | entomology, ornithology, herpetology, mammalogy, microbiology, phycology, mycology |
| Processes | population_ecology, community_ecology, ecosystem_ecology, molecular_ecology, chemical_ecology |
| Applied | conservation, restoration_ecology, wildlife_management, environmental_monitoring |
| Climate | climate_change_ecology, paleoecology |
| Technology | bioinformatics, remote_sensing, machine_learning, soundscape_ecology |
| Biology | genetics, evolution, taxonomy, physiology, ethology, biotic_interactions |
| Earth | geology, biogeography, limnology |
| General | general_ecology, invasive_species, parasitology, toxicology, fisheries, unknown |

---

## 2. Ingestion (`src/ingestion/`)

**Category**: Data Input — PDF processing and document splitting

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `pdf_parser.py` | `PDFParser` | `parse(path)` | Docling-based PDF extraction |
| | `ParsedDocument` | — | Container: sections, tables, figures, metadata |
| | `ParsedSection` | — | Title, text, page range |
| | `TableData` | — | Extracted table with headers and rows |
| `chunker.py` | `DocumentChunker` | `chunk_document(doc)` | Section-aware splitting |
| | `DocumentChunk` | — | Text chunk with metadata and source tracking |

---

## 3. Extraction (`src/extraction/`)

**Category**: Data Processing — classification and entity extraction

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `domain_classifier.py` | `DomainClassifier` | `classify_text(text)` | Keyword + LLM hybrid classification |
| | | `classify_document(doc)` | Classify a `ParsedDocument` |
| | | `get_top_domains(result)` | Get domains above threshold |
| | `ClassificationResult` | — | Primary domain, scores, study type |
| `entity_extractor.py` | `EntityExtractor` | `extract_from_chunk(chunk)` | LLM-powered entity extraction |
| | | `extract_from_chunks(chunks)` | Batch extraction |

---

## 4. Search (`src/search/`)

**Category**: Information Retrieval — keyword and hybrid search

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `paper_index.py` | `PaperIndex` | `add_paper(paper)` | SQLite FTS5 full-text index |
| | | `search(query, limit)` | BM25 keyword search |
| | | `get_paper(doc_id)` | Retrieve by ID |
| | `IndexedPaper` | — | Paper metadata for indexing |
| `ranked_search.py` | `RankedSearch` | `search(query)` | Hybrid BM25 + semantic reranking |
| | | `search_by_domain(query, domain)` | Domain-filtered search |
| | `RankedResult` | — | Combined score result |
| `query_logger.py` | `QueryLogger` | `log_query(query)` | Search analytics |

---

## 5. Retrieval (`src/retrieval/`)

**Category**: Information Retrieval — vector search and multi-source retrieval

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `vector_store.py` | `VectorStore` | `add_chunks(chunks)` | Qdrant vector indexing |
| | | `search(query, limit)` | Semantic similarity search |
| | | `hybrid_search(query)` | Combined search modes |
| `hybrid_retriever.py` | `HybridRetriever` | `retrieve(query)` | Combine Qdrant + Neo4j results |
| | `RetrievalContext` | — | Unified context for LLM |

---

## 6. Graph (`src/graph/`)

**Category**: Knowledge Graph — Neo4j storage, queries, and NetworkX analytics

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `graph_builder.py` | `GraphBuilder` | `initialize_schema()` | Create Neo4j schema |
| | | `add_paper(...)` | Create Paper node |
| | | `add_extraction_result(...)` | Add entities + relationships |
| | | `get_ecological_network(species)` | Network traversal |
| | | `get_species_measurements(species)` | Measurement aggregation |
| | `GraphStats` | — | Node/relationship counts |
| `queries.py` | — | 14 Cypher templates | Pre-built query strings |
| `network_analysis.py` | — | `build_paper_graph(papers)` | Jaccard similarity paper graph |
| | | `build_concept_graph(papers)` | Domain + title word co-occurrence |
| | | `build_domain_graph(papers)` | Domain → paper bipartite graph |
| | | `detect_communities(G)` | Greedy modularity communities |
| | | `compute_centrality(G)` | Degree centrality for node sizing |

---

## 7. Inference (`src/inference/`)

**Category**: Analysis — cross-domain discovery and hypothesis generation

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `cross_domain_linker.py` | `CrossDomainLinker` | `get_domain_affinity(d1, d2)` | Domain pair affinity (0–1) |
| | | `find_shared_species_links(species)` | Cross-domain species links |
| | | `find_semantic_bridges(d1, d2)` | Semantic similarity bridges |
| | | `discover_all_links()` | Run all discovery methods |
| | `CrossDomainLink` | — | Link with type, confidence, evidence |
| | `LinkType` | — | shared_species, ecological_cascade, etc. |
| `inference_proposer.py` | `InferenceProposer` | `generate_hypothesis_from_link(link)` | LLM hypothesis |
| | | `_generate_rule_based_hypothesis(link)` | Template hypothesis |
| | | `identify_knowledge_gaps(domain, species)` | Missing data analysis |
| | `Hypothesis` | — | Statement, rationale, experiments |

---

## 8. Scrapers (`src/scrapers/`)

**Category**: External Data — species enrichment APIs

| File | Class | Key Methods | Output |
|------|-------|-------------|--------|
| `fishbase_client.py` | `FishBaseClient` | `get_species(name)` | `FishData` (biology, habitat, trophic level) |
| `gbif_occurrence_client.py` | `GBIFOccurrenceClient` | `get_distribution(name)` | `SpeciesDistribution` (countries, coords) |
| | | `get_occurrences(name, limit)` | `list[OccurrenceRecord]` |
| `iucn_client.py` | `IUCNClient` | `get_species(name)` | `ConservationStatus` (category, threats) |

---

## 9. Agent (`src/agent/`)

**Category**: Orchestration — LangGraph query agent

| File | Class | Key Methods | Description |
|------|-------|-------------|-------------|
| `query_agent.py` | `QueryAgent` | `ask_streaming(question)` | Full query pipeline |
| | | `get_info()` | System info (model, tools) |
| | `AgentState` | — | LangGraph state (messages, intent) |
| `tool_registry.py` | — | 8 `@tool` functions | LangChain-compatible tools |

### Agent Tools

| Tool | Backend Module | Purpose |
|------|---------------|---------|
| `search_papers` | `RankedSearch` | Hybrid paper search |
| `search_by_domain` | `RankedSearch` | Domain-filtered search |
| `search_related_papers` | NetworkX graph | Graph-based related paper discovery |
| `classify_text` | `DomainClassifier` | Text classification |
| `get_species_info` | FishBase + GBIF + IUCN | Species data |
| `find_cross_domain_links` | `CrossDomainLinker` | Domain affinity |
| `generate_hypotheses` | `InferenceProposer` | Hypothesis generation |
| `query_graph` | `GraphBuilder` | Neo4j Cypher queries |

---

## 10. UI (`src/ui/`)

**Category**: Presentation — Streamlit web interface (7 pages)

| File | Purpose |
|------|---------|
| `theme.py` | Dark glassmorphism CSS + reusable card components |
| `pages/dashboard.py` | Metrics, service status, domain distribution charts |
| `pages/chat.py` | Interactive LangGraph agent with tool call visualization |
| `pages/papers.py` | Paper browser with PDF.js viewer, metadata, abstracts |
| `pages/graph_explorer.py` | 4 interactive graph views (Pyvis): Paper, Domain, Community, Concept |
| `pages/search.py` | Hybrid search with result cards and domain filters |
| `pages/species.py` | Species explorer (FishBase/GBIF/IUCN tabs) |
| `pages/domain_lab.py` | Text classification, cross-domain links, hypotheses |

---

## 11. Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `app.py` | Streamlit entry point (`streamlit run scripts/app.py`) |
| `ingest.py` | Full ingestion pipeline (PDF → all stores) |
| `chat_demo.py` | Interactive terminal chat with agent |
| `demo_pipeline.py` | Paper selection demo (parse + classify) |
| `fix_fts5.py` | FTS5 index repair (fixes `missing row` corruption) |
| `verify_setup.py` | Environment verification script |

---

## 12. Tests (`tests/`)

| File | Tests | Coverage |
|------|-------|----------|
| `test_integration.py` | 33 | All 10 modules + scripts |

### Test Groups

| Group | Tests | Validates |
|-------|-------|-----------|
| CoreImports | 4 | schemas, config, domain_registry, llm_client |
| DomainClassifier | 4 | classification, empty text, top domains |
| CrossDomainLinker | 3 | affinity scoring (same, high, low) |
| InferenceProposer | 2 | hypothesis generation, dataclass structure |
| IngestionModules | 3 | parser, chunker, chunk_document |
| SearchModules | 2 | PaperIndex CRUD, RankedResult fields |
| ScraperClients | 3 | client construction + field validation |
| ToolRegistry | 8 | all 8 tools callable, edge cases |
| AgentModule | 2 | imports, class structure |
| GraphModule | 2 | builder import, Cypher templates, network_analysis |
| IngestScript | 1 | script importability |

---

## 13. Infrastructure Requirements

| Service | Purpose | Setup | Port(s) |
|---------|---------|-------|----------|
| **LM Studio / Ollama** | Local LLM for extraction + agent | Native app / `ollama pull model` | 1234 / 11434 |
| **Docker Desktop** | Container runtime for Qdrant + Neo4j | Install from docker.com | — |
| **Qdrant** | Semantic vector search | `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant` | 6333 |
| **Neo4j** | Knowledge graph | `docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j` | 7474, 7687 |
| **SQLite** | BM25 keyword search | Built-in (no setup) | — |

### Startup Order

1. Start **Docker Desktop** (Windows: Start menu → wait for green icon)
2. Start **Qdrant** and **Neo4j** containers (`docker start qdrant neo4j`)
3. Start **LM Studio** and load a model
4. Run `python scripts/ingest.py data/raw/`
5. Run `streamlit run scripts/app.py`
