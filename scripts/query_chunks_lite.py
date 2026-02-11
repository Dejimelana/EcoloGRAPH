"""
Lightweight chunk query tool - no heavy dependencies.

Query chunks from Qdrant without importing full VectorStore.
"""
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


def get_client():
    """Get Qdrant client."""
    return QdrantClient(host="localhost", port=6333)


def list_all_papers():
    """List all unique papers in the vector store."""
    print("Fetching all papers in database...\n")
    
    client = get_client()
    
    # Scroll through all chunks
    results = client.scroll(
        collection_name="ecograph_chunks",
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = results[0]
    
    # Extract unique doc_ids
    doc_info = {}
    
    for point in chunks:
        doc_id = point.payload.get('doc_id')
        if doc_id:
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'title': point.payload.get('doc_title', 'Unknown'),
                    'source_path': point.payload.get('source_path', 'Unknown'),
                    'chunk_count': 0
                }
            doc_info[doc_id]['chunk_count'] += 1
    
    print(f"Found {len(doc_info)} papers in database:\n")
    print("="*80)
    
    for i, (doc_id, info) in enumerate(sorted(doc_info.items()), 1):
        title = info['title']
        source = info['source_path'].split('\\')[-1] if '\\' in info['source_path'] else info['source_path']
        
        print(f"\n{i}. doc_id: {doc_id}")
        print(f"   Title: {title[:70]}...")
        print(f"   Source: {source}")
        print(f"   Chunks: {info['chunk_count']}")


def get_chunks_by_paper(doc_id: str):
    """Get all chunks from a specific paper."""
    print(f"Fetching chunks for paper: {doc_id}\n")
    
    client = get_client()
    
    # Get collection stats
    stats = client.get_collection("ecograph_chunks")
    print(f"Total chunks in database: {stats.points_count}\n")
    
    # Search chunks by doc_id using filter
    results = client.scroll(
        collection_name="ecograph_chunks",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = results[0]
    
    if not chunks:
        print(f"No chunks found for doc_id: {doc_id}")
        print("\nTip: Use 'list' command to see all available doc_ids")
        return
    
    print(f"Found {len(chunks)} chunks for this paper\n")
    print("="*80)
    
    # Show first 5 chunks
    for i, point in enumerate(chunks[:5], 1):
        payload = point.payload
        
        print(f"\n📄 Chunk {i}/{len(chunks)}")
        print(f"  Chunk ID: {payload.get('chunk_id', 'N/A')}")
        print(f"  Page: {payload.get('page', 'N/A')}")
        print(f"  Section: {payload.get('section', 'N/A')}")
        print(f"  Chars: {payload.get('char_count', 'N/A')}")
        
        text = payload.get('text', '')
        print(f"\n  Text preview:")
        print(f"  {text[:200]}...")
        print("  " + "-"*76)
    
    if len(chunks) > 5:
        print(f"\n  ... and {len(chunks) - 5} more chunks")
        print("  (showing first 5 only)")


def search_by_title(query: str):
    """Search papers by title keyword."""
    print(f"Searching for papers with title containing: '{query}'\n")
    
    client = get_client()
    
    # Get all papers
    results = client.scroll(
        collection_name="ecograph_chunks",
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    
    chunks = results[0]
    
    # Find papers matching title
    matching_papers = {}
    for point in chunks:
        doc_id = point.payload.get('doc_id')
        title = point.payload.get('doc_title', '')
        source = point.payload.get('source_path', '')
        
        if query.lower() in title.lower() or query.lower() in source.lower():
            if doc_id not in matching_papers:
                matching_papers[doc_id] = {
                    'title': title,
                    'source': source,
                    'chunk_count': 0
                }
            matching_papers[doc_id]['chunk_count'] += 1
    
    if not matching_papers:
        print(f"No papers found matching: {query}")
        return
    
    print(f"Found {len(matching_papers)} matching papers:\n")
    print("="*80)
    
    for i, (doc_id, info) in enumerate(matching_papers.items(), 1):
        source = info['source'].split('\\')[-1] if '\\' in info['source'] else info['source']
        print(f"\n{i}. doc_id: {doc_id}")
        print(f"   Title: {info['title'][:70]}...")
        print(f"   Source: {source}")
        print(f"   Chunks: {info['chunk_count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query chunks from Qdrant (lightweight version)"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List papers
    subparsers.add_parser('list', help='List all papers in database')
    
    # Get chunks by paper
    paper_parser = subparsers.add_parser('paper', help='Get chunks from specific paper')
    paper_parser.add_argument('doc_id', help='Document ID (e.g., doc_abc123)')
    
    # Search by title
    search_parser = subparsers.add_parser('search', help='Search papers by title')
    search_parser.add_argument('query', help='Title search query')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_all_papers()
    elif args.command == 'paper':
        get_chunks_by_paper(args.doc_id)
    elif args.command == 'search':
        search_by_title(args.query)
    else:
        parser.print_help()
        print("\n\n📖 Examples:")
        print("  python scripts/query_chunks_lite.py list")
        print("  python scripts/query_chunks_lite.py paper doc_abc123")
        print("  python scripts/query_chunks_lite.py search Wildfire")
