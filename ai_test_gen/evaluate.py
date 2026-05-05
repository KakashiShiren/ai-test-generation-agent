"""Automated evaluation harness for generated pytest files."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_test_gen.generator import write_generated_test
from ai_test_gen.vector_store import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME


REPO_SRC = Path("source_repos/requests/src")
CHUNKS_PATH = Path("data/chunks/requests_chunks.json")
EVAL_DIR = Path("evaluation_runs")

EXCLUDED_NAME_PATTERNS = (
    "proxy",
    "netrc",
    "atomic_open",
    "extract_zipped_paths",
    "get_environ",
    "resolve_proxies",
    "should_bypass",
    "rewind_body",
    "stream_",
    "get_unicode_from_response",
    "extract_cookies_to_jar",
    "get_cookie_header",
    "main",
    "_init",
)

EXCLUDED_SOURCE_PATTERNS = (
    "open(",
    "os.environ",
    "os.getenv",
    "os.path.exists",
    "socket",
    "HTTPAdapter",
    "Session(",
    "raise NotImplementedError",
    "pass",
)


@dataclass(frozen=True)
class TargetFunction:
    """One function selected for generation and evaluation."""

    index: int
    name: str
    qualified_name: str
    signature: str
    file_path: str
    start_line: int
    end_line: int


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load previously ingested chunks."""

    return json.loads(chunks_path.read_text(encoding="utf-8"))


def signature_from_source(source_code: str) -> str:
    """Return a compact function signature from function source code."""

    node = ast.parse(source_code).body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Chunk is not a function.")
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    signature = f"{prefix} {node.name}({ast.unparse(node.args)})"
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"
    return signature


def is_reasonable_target(chunk: dict[str, Any]) -> bool:
    """Heuristic filter for deterministic, testable top-level functions."""

    if chunk["kind"] not in {"function", "async_function"}:
        return False
    if str(chunk["qualified_name"]).count(".") > 1:
        return False
    if not str(chunk["file_path"]).startswith("requests"):
        return False
    lowered_name = str(chunk["qualified_name"]).lower()
    if any(pattern in lowered_name for pattern in EXCLUDED_NAME_PATTERNS):
        return False
    source = str(chunk["source_code"])
    if "..." in source:
        return False
    if any(pattern in source for pattern in EXCLUDED_SOURCE_PATTERNS):
        return False
    try:
        signature_from_source(source)
    except (SyntaxError, ValueError):
        return False
    return True


def select_targets(limit: int = 50, chunks_path: Path = CHUNKS_PATH) -> list[TargetFunction]:
    """Select deterministic-looking functions from the ingested repository."""

    chunks = load_chunks(chunks_path)
    candidates = [chunk for chunk in chunks if is_reasonable_target(chunk)]
    candidates.sort(
        key=lambda chunk: (
            str(chunk["qualified_name"]).count("."),
            len(chunk["source_code"].splitlines()),
            chunk["file_path"],
            chunk["qualified_name"],
        )
    )
    seen: set[tuple[str, int]] = set()
    targets: list[TargetFunction] = []
    for chunk in candidates:
        identity = (chunk["file_path"], int(chunk["start_line"]))
        if identity in seen:
            continue
        seen.add(identity)
        targets.append(
            TargetFunction(
                index=len(targets) + 1,
                name=chunk["name"],
                qualified_name=chunk["qualified_name"],
                signature=signature_from_source(chunk["source_code"]),
                file_path=chunk["file_path"],
                start_line=int(chunk["start_line"]),
                end_line=int(chunk["end_line"]),
            )
        )
        if len(targets) >= limit:
            break
    if len(targets) < limit:
        raise RuntimeError(f"Only selected {len(targets)} targets; need {limit}.")
    return targets


def classify_failure(output: str) -> str:
    """Classify a pytest or generation failure for CSV aggregation."""

    lowered = output.lower()
    if "syntaxerror" in lowered:
        return "syntax error"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "import error"
    if "assertionerror" in lowered or "assert " in lowered:
        return "assertion error"
    if "timeout" in lowered:
        return "timeout"
    if "runtimeerror" in lowered or "api_key" in lowered:
        return "generation error"
    return "other"


def module_name_from_file_path(file_path: str) -> str:
    """Convert requests/utils.py to requests.utils."""

    return str(Path(file_path).with_suffix("")).replace("\\", ".").replace("/", ".")


def target_coverage_percent(coverage_json: Path, target: TargetFunction) -> float:
    """Compute covered executable lines inside the target function line range."""

    if not coverage_json.exists():
        return 0.0
    data = json.loads(coverage_json.read_text(encoding="utf-8"))
    wanted_suffix = target.file_path.replace("\\", "/")
    file_data = None
    for path, entry in data.get("files", {}).items():
        if path.replace("\\", "/").endswith(wanted_suffix):
            file_data = entry
            break
    if not file_data:
        return 0.0
    covered = set(file_data.get("executed_lines", []))
    missing = set(file_data.get("missing_lines", []))
    target_lines = set(range(target.start_line, target.end_line + 1))
    executable = (covered | missing) & target_lines
    if not executable:
        return 0.0
    return round((len(covered & target_lines) / len(executable)) * 100, 2)


def run_pytest_for_target(
    target: TargetFunction,
    test_path: Path,
    coverage_json: Path,
    repo_src: Path = REPO_SRC,
    timeout_seconds: int = 90,
) -> tuple[bool, float, str]:
    """Run pytest and return pass status, target-line coverage, and error text."""

    env = os.environ.copy()
    repo_src_abs = str(repo_src.resolve())
    env["PYTHONPATH"] = repo_src_abs + os.pathsep + env.get("PYTHONPATH", "")
    coverage_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(test_path),
        "-q",
        f"--cov={module_name_from_file_path(target.file_path)}",
        f"--cov-report=json:{coverage_json}",
        "--cov-report=term",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return False, 0.0, f"timeout: {error}"

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    coverage = target_coverage_percent(coverage_json, target)
    return completed.returncode == 0, coverage, output.strip()


def truncate_error(text: str, limit: int = 1000) -> str:
    """Keep CSV error cells readable."""

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def safe_path_part(value: str) -> str:
    """Return a filesystem-safe identifier."""

    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "target"


def evaluate_targets(
    limit: int = 50,
    output_dir: Path = EVAL_DIR,
    model: str = "llama-3.1-8b-instant",
    smoke_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Generate tests, run pytest coverage, and write a CSV report."""

    targets = select_targets(limit=limit)
    tests_dir = output_dir / "generated_tests"
    coverage_dir = output_dir / "coverage"
    csv_path = output_dir / "results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for target in targets:
        print(f"[{target.index}/{limit}] {target.signature} ({target.file_path}:{target.start_line})")
        try:
            test_path = write_generated_test(
                function_signature=target.signature,
                output_dir=tests_dir / f"{target.index:02d}_{safe_path_part(target.qualified_name)}",
                persist_dir=DEFAULT_CHROMA_DIR,
                collection_name=DEFAULT_COLLECTION_NAME,
                top_k=5,
                model=model,
                source_package_root="requests",
            )
            coverage_json = coverage_dir / f"{target.index:02d}_{target.name}.json"
            passed, coverage, output = run_pytest_for_target(target, test_path, coverage_json)
        except Exception as error:  # noqa: BLE001 - harness must continue across failures.
            test_path = Path("")
            passed = False
            coverage = 0.0
            output = f"{type(error).__name__}: {error}"

        error_type = "" if passed else classify_failure(output)
        rows.append(
            {
                "function_name": target.qualified_name,
                "signature": target.signature,
                "file_path": target.file_path,
                "line_range": f"{target.start_line}-{target.end_line}",
                "test_file": str(test_path),
                "pass_fail": "pass" if passed else "fail",
                "coverage_percent": coverage,
                "error_type": error_type,
                "error_message": "" if passed else truncate_error(output),
            }
        )
        if smoke_only:
            break

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pass_count = sum(1 for row in rows if row["pass_fail"] == "pass")
    coverages = [float(row["coverage_percent"]) for row in rows]
    failure_breakdown = Counter(row["error_type"] for row in rows if row["error_type"])
    summary = {
        "total": len(rows),
        "pass_count": pass_count,
        "pass_rate": round((pass_count / len(rows)) * 100, 2),
        "mean_coverage": round(statistics.mean(coverages), 2) if coverages else 0.0,
        "failure_breakdown": dict(failure_breakdown),
    }
    return csv_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path, summary = evaluate_targets(
        limit=args.limit,
        output_dir=args.output_dir,
        model=args.model,
        smoke_only=args.smoke_only,
    )
    print(f"CSV: {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
