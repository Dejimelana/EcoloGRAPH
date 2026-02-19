"""
Script to verify EcoloGRAPH installation.

Run: python scripts/verify_setup.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check that all modules can be imported."""
    print("🔍 Checking module imports...")
    
    modules = [
        "src",
        "src.core",
        "src.ingestion",
        "src.enrichment",
        "src.extraction",
        "src.scrapers",
        "src.graph",
        "src.retrieval",
        "src.agent",
        "src.ui",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Check that key dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    dependencies = [
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]
    
    optional_deps = [
        ("docling", "docling"),
        ("qdrant_client", "qdrant-client"),
        ("neo4j", "neo4j"),
        ("streamlit", "streamlit"),
        ("sentence_transformers", "sentence-transformers"),
    ]
    
    all_ok = True
    
    print("  Required:")
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"    ✅ {package}")
        except ImportError:
            print(f"    ❌ {package} (run: pip install {package})")
            all_ok = False
    
    print("  Optional (will be needed later):")
    for module, package in optional_deps:
        try:
            __import__(module)
            print(f"    ✅ {package}")
        except ImportError:
            print(f"    ⚠️  {package} (install when needed: pip install {package})")
    
    return all_ok


def check_config():
    """Check configuration loading."""
    print("\n🔍 Checking configuration...")
    
    try:
        from src.core.config import get_settings
        settings = get_settings()
        print(f"  ✅ Config loaded successfully")
        print(f"     Base URL: {settings.llm.base_url}")
        print(f"     Ingestion Model: {settings.llm.ingestion_model}")
        print(f"     Reasoning Model: {settings.llm.reasoning_model}")
        print(f"     Embedding Model: {settings.embedding.model}")
        return True
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def check_directories():
    """Check that all directories exist."""
    print("\n🔍 Checking directory structure...")
    
    base = Path(__file__).parent.parent
    dirs = [
        "src/core",
        "src/ingestion",
        "src/enrichment",
        "src/extraction",
        "src/scrapers",
        "src/graph",
        "src/retrieval",
        "src/agent",
        "src/ui",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "config/prompts",
        "config/schemas",
        "data/raw",
        "data/processed",
        "data/cache",
        "scripts",
        "docs",
    ]
    
    all_ok = True
    for d in dirs:
        path = base / d
        if path.exists():
            print(f"  ✅ {d}")
        else:
            print(f"  ❌ {d} (missing)")
            all_ok = False
    
    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("EcoloGRAPH Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Directories", check_directories()))
    results.append(("Imports", check_imports()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Configuration", check_config()))
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All checks passed! Ready for Phase 1.")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
