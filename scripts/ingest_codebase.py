"""Extract AST chunks from a Python repository and save them as JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_test_gen.ingest import write_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("source_repos/requests/src"),
        help="Path to the Python source tree to ingest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks/requests_chunks.json"),
        help="Path to write extracted chunk JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = write_chunks(repo_root=args.repo_root, output_path=args.output)
    print(f"Wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
