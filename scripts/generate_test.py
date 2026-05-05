"""Generate one pytest file from a target Python function signature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_test_gen.generator import DEFAULT_GROQ_MODEL, write_generated_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("function_signature", help="Python function signature to test.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_tests"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    parser.add_argument("--collection", default="code_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--source-package-root", default="requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = write_generated_test(
            function_signature=args.function_signature,
            output_dir=args.output_dir,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            top_k=args.top_k,
            model=args.model,
            source_package_root=args.source_package_root,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(f"Wrote generated pytest file to {output_path}")


if __name__ == "__main__":
    main()
