"""
Ingestion status report - CLI tool.

Shows which papers are indexed in which stores.

Usage:
    python scripts/ingestion_report.py
    python scripts/ingestion_report.py --partial
    python scripts/ingestion_report.py --failed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.search.ingestion_ledger import IngestionLedger


def main():
    parser = argparse.ArgumentParser(description="Check ingestion ledger status")
    parser.add_argument("--partial", action="store_true", help="Show partially indexed papers")
    parser.add_argument("--failed", action="store_true", help="Show failed ingestions")
    parser.add_argument("--store", choices=["sqlite", "qdrant", "neo4j"], help="Filter by store")
    args = parser.parse_args()
    
    ledger = IngestionLedger("data/paper_index.db")
    
    if args.partial:
        print("\n📊 Partially Indexed Papers\n" + "="*60)
        partial = ledger.get_partial_ingestions()
        
        if not partial:
            print("✅ No partial ingestions found - all papers fully indexed!")
        else:
            for doc_id, status in partial:
                print(f"\n{doc_id}:")
                for store, info in status.items():
                    icon = "✅" if info["status"] == "success" else "❌" if info["status"] == "failed" else "⏳"
                    print(f"  {icon} {store:8s}: {info['status']:8s} @ {info['timestamp'][:19]}")
                    if info["error"]:
                        print(f"           Error: {info['error']}")
    
    elif args.failed:
        print("\n❌ Failed Ingestions\n" + "="*60)
        failed = ledger.get_failed_ingestions(args.store)
        
        if not failed:
            print("✅ No failed ingestions!")
        else:
            for doc_id, store, error in failed:
                print(f"\n{doc_id} → {store}")
                print(f"  Error: {error}")
    
    else:
        # Show stats
        print("\n📈 Ingestion Statistics\n" + "="*60)
        stats = ledger.get_stats()
        
        print(f"\n📦 Summary:")
        print(f"  Total documents: {stats['summary']['total_docs']}")
        print(f"  Fully indexed:   {stats['summary']['fully_indexed']}")
        print(f"  Partial:         {stats['summary']['partially_indexed']}")
        
        print(f"\n🗄️  By Store:")
        for store in ["sqlite", "qdrant", "neo4j"]:
            store_stats = stats[store]
            success = store_stats.get("success", 0)
            failed = store_stats.get("failed", 0)
            pending = store_stats.get("pending", 0)
            total = success + failed + pending
            
            if total > 0:
                success_rate = (success / total) * 100
                print(f"\n  {store.upper()}:")
                print(f"    Success: {success:4d} ({success_rate:5.1f}%)")
                print(f"    Failed:  {failed:4d}")
                print(f"    Pending: {pending:4d}")


if __name__ == "__main__":
    main()
