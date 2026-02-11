"""
Query chunks from Qdrant vector store.

Examples:
- Get all chunks from a specific paper
- Search chunks by text similarity
- Browse all chunks in database
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore


def get_chunks_by_paper(doc_id: str):
    """Get all chunks from a specific paper."""
    print(f"Fetching chunks for paper: {doc_id}\n")
    
    vector_store = VectorStore()
    
    # Get collection stats
    stats = vector_store.client.get_collection("ecograph_chunks")
    print(f"Total chunks in database: {stats.points_count}\n")
    
    # Search chunks by doc_id using filter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    results = vector_store.client.scroll(
        collection_name="ecograph_chunks",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        ),
        limit=1000,  # Max chunks per paper
        with_payload=True,
        with_vectors=False
    )
    
    chunks = results[0]  # First element is the list of points
    
    if not chunks:
        print(f"❌ No chunks found for doc_id: {doc_id}")
        return
    
    print(f"✅ Found {len(chunks)} chunks for this paper\n")
    print("="*80)
    
    for i, point in enumerate(chunks, 1):
        payload = point.payload
        
        print(f"\n📄 Chunk {i}/{len(chunks)}")
        print(f"  ID: {point.id}")
        print(f"  Chunk ID: {payload.get('chunk_id', 'N/A')}")
        print(f"  Page: {payload.get('page', 'N/A')}")
        print(f"  Section: {payload.get('section', 'N/A')}")
        print(f"  Chars: {payload.get('char_count', 'N/A')}")
        print(f"  Words: {payload.get('word_count', 'N/A')}")
        print(f"\n  Text preview:")
        text = payload.get('text', '')
        print(f"  {text[:200]}...")
        print("  " + "-"*76)


def list_all_papers():
    """List all unique papers in the vector store."""
    print("Fetching all papers in database...\n")
    
    vector_store = VectorStore()
    
    # Scroll through all chunks
    results = vector_store.client.scroll(
        collection_name="ecograph_chunks",
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = results[0]
    
    # Extract unique doc_ids
    doc_ids = set()
    doc_info = {}
    
    for point in chunks:
        doc_id = point.payload.get('doc_id')
        if doc_id:
            doc_ids.add(doc_id)
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'title': point.payload.get('doc_title', 'Unknown'),
                    'chunk_count': 0
                }
            doc_info[doc_id]['chunk_count'] += 1
    
    print(f"📚 Found {len(doc_ids)} papers in database:\n")
    print("="*80)
    
    for i, (doc_id, info) in enumerate(sorted(doc_info.items()), 1):
        print(f"\n{i}. {doc_id}")
        print(f"   Title: {info['title'][:70]}...")
        print(f"   Chunks: {info['chunk_count']}")


def search_chunks(query: str, limit: int = 5):
    """Search chunks by semantic similarity."""
    print(f"Searching for: '{query}'\n")
    
    vector_store = VectorStore()
    
    results = vector_store.search(
        query_text=query,
        limit=limit,
        filter_domain=None
    )
    
    print(f"✅ Found {len(results)} matching chunks\n")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n📄 Result {i}/{len(results)}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Doc ID: {result['doc_id']}")
        print(f"  Title: {result.get('doc_title', 'N/A')[:60]}...")
        print(f"  Page: {result.get('page', 'N/A')}")
        print(f"  Section: {result.get('section', 'N/A')}")
        print(f"\n  Text:")
        print(f"  {result['text'][:300]}...")
        print("  " + "-"*76)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query chunks from Qdrant")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List papers
    subparsers.add_parser('list', help='List all papers in database')
    
    # Get chunks by paper
    paper_parser = subparsers.add_parser('paper', help='Get chunks from specific paper')
    paper_parser.add_argument('doc_id', help='Document ID (e.g., doc_abc123)')
    
    # Search chunks
    search_parser = subparsers.add_parser('search', help='Search chunks by text')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-n', '--limit', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_all_papers()
    elif args.command == 'paper':
        get_chunks_by_paper(args.doc_id)
    elif args.command == 'search':
        search_chunks(args.query, args.limit)
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  python scripts/query_chunks.py list")
        print("  python scripts/query_chunks.py paper doc_abc123")
        print("  python scripts/query_chunks.py search 'Pogonus minutus' -n 10")
