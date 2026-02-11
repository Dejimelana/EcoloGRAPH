"""
EcoloGRAPH - Graph RAG for Ecological Data Extraction

Multi-domain scientific document processing with:
- 43 scientific domains with weighted keyword classification
- Knowledge Graph (Neo4j) + Vector Store (Qdrant) + SQLite FTS5
- NetworkX graph analytics (community detection, centrality, co-occurrence)
- Cross-domain inference and hypothesis generation
- Hybrid search: BM25 keyword → semantic reranking
- Interactive Pyvis graph visualization

Modules:
    core        - Configuration, schemas, LLM client, domain registry
    ingestion   - PDF parsing (Docling), chunking
    enrichment  - CrossRef, Semantic Scholar, taxonomy resolution
    extraction  - Entity extraction, domain classification
    scrapers    - FishBase, GBIF, IUCN API clients
    graph       - Neo4j graph builder, Cypher queries, NetworkX analytics
    retrieval   - Qdrant vector store + hybrid retriever
    search      - SQLite FTS5 index, ranked search, query logging
    inference   - Cross-domain linking, hypothesis generation
    agent       - LangGraph query agent with 8 tools
    ui          - Streamlit dashboard (7 pages) + glassmorphism theme
"""

__version__ = "1.2.0"
