"""Pre-build the FAISS index over course_data.json.

Run this once after `pip install -r requirements.txt` so the Streamlit app
can load embeddings from disk instantly instead of paying ~30s of embedding
cost the first time a user confirms their trip.

Usage:
    # via env var
    GOOGLE_API_KEY=AIza... python build_index.py

    # via argv
    python build_index.py AIza...

    # force a rebuild (e.g. after editing course_data.json)
    python build_index.py --rebuild
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make sure we can import sibling modules when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import COURSE_DATA_PATH, VECTORSTORE_DIR, build_or_load_vectorstore


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-build the course FAISS index.")
    parser.add_argument(
        "api_key",
        nargs="?",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Gemini API key (or set GOOGLE_API_KEY env var).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-embedding even if vectorstore/ already exists.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "❌ No API key provided. Pass it as the first argument or set "
            "GOOGLE_API_KEY.",
            file=sys.stderr,
        )
        return 1

    if not COURSE_DATA_PATH.exists():
        print(f"❌ course_data.json not found at {COURSE_DATA_PATH}", file=sys.stderr)
        return 1

    print(f"📂 Course data : {COURSE_DATA_PATH}")
    print(f"💾 Vectorstore : {VECTORSTORE_DIR}")
    print(f"🔁 Rebuild     : {args.rebuild}")
    print("⏳ Embedding courses with text-embedding-004 ...")

    t0 = time.time()
    store = build_or_load_vectorstore(args.api_key, rebuild=args.rebuild)
    elapsed = time.time() - t0

    n_docs = store.index.ntotal if hasattr(store, "index") else "?"
    print(f"✅ Done. Indexed {n_docs} courses in {elapsed:.1f}s.")
    print(f"   Streamlit will now load embeddings from {VECTORSTORE_DIR} on startup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
