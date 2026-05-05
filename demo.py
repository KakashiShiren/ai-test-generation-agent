"""Command-line demo for generating a pytest file from a function signature."""

from __future__ import annotations

from pathlib import Path

from ai_test_gen.generator import write_generated_test


def main() -> None:
    signature = input("Paste a Python function signature: ").strip()
    if not signature:
        raise SystemExit("A function signature is required.")
    output_path = write_generated_test(
        function_signature=signature,
        output_dir=Path("generated_tests"),
    )
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
