"""
EcoloGRAPH — Comprehensive Integration Tests

Tests all modules to verify:
1. Core module imports and class instantiation
2. Domain classification (keyword-based, no LLM needed)
3. Cross-domain linking and hypothesis generation
4. Tool registry API compatibility
5. Search infrastructure (SQLite FTS5)
6. Scraper client construction
7. Agent module imports

Run:
    python tests/test_integration.py           # All tests
    python tests/test_integration.py -v        # Verbose
    python tests/test_integration.py -k graph  # Only graph-related tests
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 1. Core Module Tests
# ============================================================

class TestCoreImports(unittest.TestCase):
    """Test that all core modules import without errors."""

    def test_import_schemas(self):
        from src.core.schemas import (
            SourceReference, SpeciesMention, Measurement,
            Location, EcologicalRelation, ExtractionResult,
        )
        # Verify constructors
        sr = SourceReference(doc_id="test_doc")
        self.assertEqual(sr.doc_id, "test_doc")

    def test_import_config(self):
        from src.core.config import Settings, get_settings
        s = Settings()
        self.assertEqual(s.llm.provider, "local")
        self.assertEqual(s.embedding.model, "all-MiniLM-L6-v2")

    def test_import_domain_registry(self):
        from src.core.domain_registry import DomainType, DomainRegistry
        # Verify we have 43+ domains
        all_domains = list(DomainType)
        self.assertGreaterEqual(len(all_domains), 43)
        # Verify specific domains exist
        self.assertIn(DomainType.MARINE_ECOLOGY, all_domains)
        self.assertIn(DomainType.CONSERVATION, all_domains)
        self.assertIn(DomainType.ETHOLOGY, all_domains)
        self.assertIn(DomainType.BIOGEOGRAPHY, all_domains)

    def test_import_llm_client(self):
        from src.core.llm_client import LLMClient, LLMResponse
        # LLMClient should be importable (won't connect without server)
        self.assertTrue(callable(LLMClient))


# ============================================================
# 2. Domain Classification Tests
# ============================================================

class TestDomainClassifier(unittest.TestCase):
    """Test domain classification (keyword-only, no LLM needed)."""

    def setUp(self):
        from src.extraction.domain_classifier import DomainClassifier
        self.classifier = DomainClassifier()

    def test_marine_text(self):
        from src.core.domain_registry import DomainType
        text = "Coral reef fish populations decline due to ocean acidification and rising sea temperatures."
        result = self.classifier.classify_text(text, use_llm=False)
        self.assertIsNotNone(result.primary_domain)
        self.assertGreater(result.confidence, 0.0)
        # Should recognize marine-related domain
        self.assertIn(result.primary_domain.value, [
            "marine_ecology", "coral_reef_ecology", "fisheries",
            "oceanography", "conservation", "climate_change_ecology"
        ])

    def test_genetics_text(self):
        from src.core.domain_registry import DomainType
        text = "DNA barcoding reveals cryptic species diversity in freshwater invertebrates using mitochondrial genes."
        result = self.classifier.classify_text(text, use_llm=False)
        self.assertGreater(result.confidence, 0.0)

    def test_get_top_domains(self):
        text = "The impact of deforestation on bird species richness and conservation priority areas."
        result = self.classifier.classify_text(text, use_llm=False)
        top = self.classifier.get_top_domains(result, threshold=0.05)
        self.assertIsInstance(top, list)
        self.assertGreater(len(top), 0)
        # Each entry is (DomainType, float)
        dt, score = top[0]
        self.assertIsInstance(score, float)

    def test_empty_text(self):
        from src.core.domain_registry import DomainType
        result = self.classifier.classify_text("", use_llm=False)
        self.assertEqual(result.primary_domain, DomainType.UNKNOWN)


# ============================================================
# 3. Cross-Domain Linker Tests
# ============================================================

class TestCrossDomainLinker(unittest.TestCase):
    """Test domain affinity and link construction."""

    def setUp(self):
        from src.inference.cross_domain_linker import CrossDomainLinker
        self.linker = CrossDomainLinker()

    def test_same_domain_affinity(self):
        from src.core.domain_registry import DomainType
        score = self.linker.get_domain_affinity(
            DomainType.MARINE_ECOLOGY, DomainType.MARINE_ECOLOGY
        )
        self.assertEqual(score, 1.0)

    def test_high_affinity_pair(self):
        from src.core.domain_registry import DomainType
        score = self.linker.get_domain_affinity(
            DomainType.MARINE_ECOLOGY, DomainType.CONSERVATION
        )
        self.assertGreaterEqual(score, 0.3)

    def test_low_affinity_pair(self):
        from src.core.domain_registry import DomainType
        score = self.linker.get_domain_affinity(
            DomainType.MARINE_ECOLOGY, DomainType.MACHINE_LEARNING
        )
        self.assertLessEqual(score, 0.5)


# ============================================================
# 4. Hypothesis Generation Tests
# ============================================================

class TestInferenceProposer(unittest.TestCase):
    """Test rule-based hypothesis generation."""

    def test_generate_from_link(self):
        from src.inference.inference_proposer import InferenceProposer
        from src.inference.cross_domain_linker import CrossDomainLink, LinkType
        from src.core.domain_registry import DomainType

        proposer = InferenceProposer()

        link = CrossDomainLink(
            link_id="test_link_001",
            link_type=LinkType.SHARED_SPECIES,
            source_domain=DomainType.MARINE_ECOLOGY,
            source_entity="Gadus morhua",
            target_domain=DomainType.CONSERVATION,
            target_entity="Gadus morhua",
            confidence=0.8,
            description="Test link",
        )

        h = proposer._generate_rule_based_hypothesis(link)

        self.assertIsNotNone(h)
        self.assertIsInstance(h.statement, str)
        self.assertGreater(len(h.statement), 10)
        self.assertEqual(h.confidence_score, 0.8)
        self.assertIn("Gadus morhua", h.statement)

    def test_hypothesis_attributes(self):
        """Verify Hypothesis dataclass has the fields we expect."""
        from src.inference.inference_proposer import Hypothesis, HypothesisType, ConfidenceLevel

        h = Hypothesis(
            hypothesis_id="test",
            hypothesis_type=HypothesisType.ECOLOGICAL_EFFECT,
            statement="Test statement",
            rationale="Test rationale",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.7,
        )
        self.assertEqual(h.statement, "Test statement")
        self.assertEqual(h.hypothesis_type.value, "ecological_effect")
        self.assertEqual(h.confidence.value, "medium")
        # Verify to_dict works
        d = h.to_dict()
        self.assertEqual(d["statement"], "Test statement")


# ============================================================
# 5. Ingestion Module Tests
# ============================================================

class TestIngestionModules(unittest.TestCase):
    """Test ingestion module imports and basic functionality."""

    def test_import_pdf_parser(self):
        from src.ingestion.pdf_parser import PDFParser, ParsedDocument
        # Verify ParsedDocument can be constructed
        doc = ParsedDocument(
            doc_id="test_doc",
            source_path=Path("test.pdf"),
            full_text="Hello world",
            title="Test Paper",
        )
        self.assertEqual(doc.doc_id, "test_doc")
        self.assertEqual(doc.title, "Test Paper")

    def test_import_chunker(self):
        from src.ingestion.chunker import DocumentChunker, DocumentChunk
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        self.assertEqual(chunker.chunk_size, 500)

    def test_chunker_with_doc(self):
        from src.ingestion.pdf_parser import ParsedDocument, ParsedSection
        from src.ingestion.chunker import DocumentChunker

        doc = ParsedDocument(
            doc_id="test_doc",
            source_path=Path("test.pdf"),
            full_text="This is a long text about marine ecology and fish populations. " * 50,
            sections=[
                ParsedSection(
                    title="Introduction",
                    level=1,
                    text="This is the introduction about marine ecology. " * 30,
                    page_start=1,
                    page_end=1,
                )
            ],
        )

        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.doc_id == "test_doc" for c in chunks))


# ============================================================
# 6. Search Module Tests
# ============================================================

class TestSearchModules(unittest.TestCase):
    """Test search infrastructure (SQLite FTS5)."""

    def test_paper_index_crud(self):
        """Test PaperIndex create/search/delete cycle."""
        import tempfile, os
        from src.search.paper_index import PaperIndex, IndexedPaper
        from datetime import datetime

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        idx = PaperIndex(db_path=db_path)
        try:
            paper = IndexedPaper(
                doc_id="test_001",
                title="Effects of microplastics on marine fish populations",
                authors=["Smith J", "Doe A"],
                year=2023,
                journal="Marine Pollution Bulletin",
                abstract="This study examines the impact of microplastics on fish.",
                keywords=["microplastics", "fish", "marine"],
                primary_domain="marine_ecology",
                domains={"marine_ecology": 0.8, "toxicology": 0.3},
                study_type="experimental",
                source_path="test.pdf",
                indexed_at=datetime.now(),
            )

            idx.add_paper(paper)
            self.assertEqual(idx.count(), 1)

            # Search
            results = idx.search("microplastics fish")
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0].doc_id, "test_001")

            # Retrieve
            retrieved = idx.get_paper("test_001")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.title, paper.title)

            # Cleanup
            idx.remove_paper("test_001")
            self.assertEqual(idx.count(), 0)

        finally:
            # Close SQLite connection before deleting file (Windows lock)
            if hasattr(idx, 'conn'):
                idx.conn.close()
            try:
                os.unlink(db_path)
            except PermissionError:
                pass  # Windows may still hold lock briefly

    def test_ranked_search_import(self):
        from src.search.ranked_search import RankedSearch, RankedResult
        # Dataclass non-default fields don't appear via hasattr on class
        self.assertIn('combined_score', RankedResult.__dataclass_fields__)


# ============================================================
# 7. Scraper Client Tests
# ============================================================

class TestScraperClients(unittest.TestCase):
    """Test that scraper clients can be instantiated."""

    def test_fishbase_client(self):
        from src.scrapers.fishbase_client import FishBaseClient, FishData
        client = FishBaseClient()
        self.assertIsNotNone(client)
        # Dataclass fields checked via __dataclass_fields__
        fields = FishData.__dataclass_fields__
        for expected in ['scientific_name', 'habitat', 'max_length_cm', 'common_names', 'fb_name']:
            self.assertIn(expected, fields, f"FishData missing field: {expected}")
        client.close()

    def test_gbif_client(self):
        from src.scrapers.gbif_occurrence_client import GBIFOccurrenceClient, SpeciesDistribution
        client = GBIFOccurrenceClient()
        self.assertIsNotNone(client)
        # Dataclass fields checked via __dataclass_fields__
        fields = SpeciesDistribution.__dataclass_fields__
        for expected in ['total_occurrences', 'countries', 'scientific_name']:
            self.assertIn(expected, fields, f"SpeciesDistribution missing field: {expected}")
        client.close()

    def test_iucn_client(self):
        from src.scrapers.iucn_client import IUCNClient, ConservationStatus
        client = IUCNClient()
        self.assertIsNotNone(client)
        # Dataclass fields checked via __dataclass_fields__
        fields = ConservationStatus.__dataclass_fields__
        for expected in ['category', 'population_trend', 'category_name']:
            self.assertIn(expected, fields, f"ConservationStatus missing field: {expected}")
        client.close()


# ============================================================
# 8. Tool Registry Tests (API Compatibility)
# ============================================================

class TestToolRegistry(unittest.TestCase):
    """Test that tool registry tools are properly defined and API-compatible."""

    def test_all_tools_importable(self):
        from src.agent.tool_registry import ALL_TOOLS, get_tool_descriptions
        self.assertEqual(len(ALL_TOOLS), 7)
        desc = get_tool_descriptions()
        self.assertIn("search_papers", desc)
        self.assertIn("query_graph", desc)

    def test_tool_names(self):
        from src.agent.tool_registry import ALL_TOOLS
        expected = {
            "search_papers", "search_by_domain", "classify_text",
            "get_species_info", "find_cross_domain_links",
            "generate_hypotheses", "query_graph",
        }
        actual = {t.name for t in ALL_TOOLS}
        self.assertEqual(actual, expected)

    def test_classify_text_tool_no_crash(self):
        """classify_text should NOT crash on a valid text input."""
        from src.agent.tool_registry import classify_text
        result = classify_text.invoke(
            "Ocean acidification affects coral reef fish populations"
        )
        self.assertIsInstance(result, str)
        self.assertIn("Primary domain:", result)
        self.assertNotIn("error", result.lower())

    def test_find_cross_domain_links_tool(self):
        from src.agent.tool_registry import find_cross_domain_links
        result = find_cross_domain_links.invoke({
            "domain1": "marine_ecology",
            "domain2": "conservation"
        })
        self.assertIsInstance(result, str)
        self.assertIn("Affinity score:", result)

    def test_generate_hypotheses_tool(self):
        from src.agent.tool_registry import generate_hypotheses
        result = generate_hypotheses.invoke({
            "topic": "microplastics impact on marine organisms",
            "domains": "marine_ecology,toxicology"
        })
        self.assertIsInstance(result, str)
        self.assertIn("Hypotheses for", result)
        self.assertNotIn("Error", result)

    def test_generate_hypotheses_no_domains(self):
        from src.agent.tool_registry import generate_hypotheses
        result = generate_hypotheses.invoke({
            "topic": "biodiversity loss in tropical forests",
            "domains": ""
        })
        self.assertIsInstance(result, str)
        self.assertIn("Hypotheses for", result)

    def test_invalid_domain(self):
        from src.agent.tool_registry import find_cross_domain_links
        result = find_cross_domain_links.invoke({
            "domain1": "fake_domain",
            "domain2": "marine_ecology"
        })
        self.assertIn("Invalid domain", result)


# ============================================================
# 9. Agent Module Tests
# ============================================================

class TestAgentModule(unittest.TestCase):
    """Test agent module imports and structure."""

    def test_agent_imports(self):
        from src.agent import QueryAgent, ALL_TOOLS, get_tool_descriptions
        self.assertTrue(callable(QueryAgent))
        self.assertEqual(len(ALL_TOOLS), 7)

    def test_query_agent_class(self):
        from src.agent.query_agent import QueryAgent, AgentState
        # Verify QueryAgent has expected methods
        self.assertTrue(hasattr(QueryAgent, 'ask_streaming'))
        self.assertTrue(hasattr(QueryAgent, 'get_info'))


# ============================================================
# 10. Graph Module Tests
# ============================================================

class TestGraphModule(unittest.TestCase):
    """Test graph module imports (no Neo4j connection needed)."""

    def test_import_graph_builder(self):
        from src.graph.graph_builder import GraphBuilder, GraphStats
        self.assertTrue(callable(GraphBuilder))
        stats = GraphStats()
        self.assertEqual(stats.paper_count, 0)

    def test_import_queries(self):
        from src.graph import queries
        # Verify key query strings exist
        self.assertTrue(len(queries.SPECIES_PROFILE) > 0)
        self.assertTrue(len(queries.ECOLOGICAL_NETWORK) > 0)
        self.assertTrue(len(queries.FOOD_WEB) > 0)
        self.assertTrue(len(queries.SPECIES_CO_OCCURRENCE) > 0)
        self.assertTrue(len(queries.MEASUREMENT_SYNTHESIS) > 0)


# ============================================================
# 11. Ingestion Pipeline Script Test
# ============================================================

class TestIngestScript(unittest.TestCase):
    """Test that the ingest script imports and parses args."""

    def test_import_ingest(self):
        # This tests that the script can be imported without error
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ingest",
            Path(__file__).parent.parent / "scripts" / "ingest.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertTrue(hasattr(mod, 'ingest'))
        self.assertTrue(hasattr(mod, 'main'))


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EcoloGRAPH Integration Tests")
    print("=" * 60)
    print()
    unittest.main(verbosity=2)
