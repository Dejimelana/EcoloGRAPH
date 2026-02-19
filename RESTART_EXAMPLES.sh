# Test LM Studio Auto-Restart Feature
#
# This demonstrates the new --restart-interval feature that prevents
# LM Studio memory leaks during long ingestion runs.

# Example 1: Use default (restart every 20 papers)
python scripts/ingest.py data/papers/*.pdf

# Example 2: More aggressive restart (every 10 papers)
python scripts/ingest.py data/papers/*.pdf --restart-interval 10

# Example 3: Less aggressive restart (every 50 papers)
python scripts/ingest.py data/papers/*.pdf --restart-interval 50

# Example 4: Disable auto-restart completely
python scripts/ingest.py data/papers/*.pdf --restart-interval 0

# Example 5: Combined with other flags
python scripts/ingest.py data/papers/*.pdf \
  --restart-interval 15 \
  --skip-graph \
  --chunk-size 800

# Expected Output:
# ============================================================
# [1/100] Processing: paper1.pdf
#   📖 Parsed: 12 pages
#   🧬 Extracted 42 entities
#   ⏱️  Completed in 15.2s
# ...
# [20/100] Processing: paper20.pdf
#   ⏱️  Completed in 14.8s
#
# ============================================================
# 🔄 RESTARTING LM STUDIO (processed 20 papers)
#    Reason: Prevent memory leak accumulation
# ============================================================
# ✅ LM Studio restarted successfully
#
# [21/100] Processing: paper21.pdf
#   ...
