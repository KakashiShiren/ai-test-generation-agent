"""Retrieve similar code chunks for a function signature query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_test_gen.vector_store import retrieve_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("function_signature", help="Function signature to search for.")
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="code_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve_context(
        function_signature=args.function_signature,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        top_k=args.top_k,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
