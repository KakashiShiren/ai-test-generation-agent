"""Build the persistent ChromaDB vector store from ingested code chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_test_gen.vector_store import build_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/requests_chunks.json"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="code_chunks")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_vector_store(
        chunks_path=args.chunks,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"Stored {count} embedded chunks in {args.persist_dir} / {args.collection}")


if __name__ == "__main__":
    main()
