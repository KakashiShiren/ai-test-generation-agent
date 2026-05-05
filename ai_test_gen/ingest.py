"""AST-based code ingestion for function-level retrieval chunks."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CodeChunk:
    """Serializable chunk produced from one Python function, method, or class."""

    name: str
    qualified_name: str
    kind: str
    source_code: str
    docstring: str | None
    type_hints: dict[str, str | None]
    file_path: str
    start_line: int
    end_line: int
    embedding_input: str


class DefinitionCollector(ast.NodeVisitor):
    """Collect function and class definitions from one Python module."""

    def __init__(self, source: str, file_path: Path, repo_root: Path) -> None:
        self.source = source
        self.file_path = file_path
        self.repo_root = repo_root
        self.chunks: list[CodeChunk] = []
        self._qualifier_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._add_chunk(node, kind="class")
        self._qualifier_stack.append(node.name)
        self.generic_visit(node)
        self._qualifier_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._add_chunk(node, kind="function")
        self._qualifier_stack.append(node.name)
        self.generic_visit(node)
        self._qualifier_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._add_chunk(node, kind="async_function")
        self._qualifier_stack.append(node.name)
        self.generic_visit(node)
        self._qualifier_stack.pop()

    def _add_chunk(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        source_code = ast.get_source_segment(self.source, node) or ""
        docstring = ast.get_docstring(node)
        type_hints = extract_type_hints(node)
        qualified_name = ".".join([*self._qualifier_stack, node.name])
        embedding_input = build_embedding_input(
            name=qualified_name,
            docstring=docstring,
            type_hints=type_hints,
        )

        self.chunks.append(
            CodeChunk(
                name=node.name,
                qualified_name=qualified_name,
                kind=kind,
                source_code=source_code,
                docstring=docstring,
                type_hints=type_hints,
                file_path=str(self.file_path.relative_to(self.repo_root)),
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                embedding_input=embedding_input,
            )
        )


def extract_type_hints(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str | None]:
    """Return type hint strings from a function or class definition."""

    if isinstance(node, ast.ClassDef):
        hints: dict[str, str | None] = {}
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                hints[statement.target.id] = ast.unparse(statement.annotation)
        return hints

    hints = {}
    for arg in [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]:
        hints[arg.arg] = ast.unparse(arg.annotation) if arg.annotation else None
    if node.args.vararg:
        hints[f"*{node.args.vararg.arg}"] = (
            ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else None
        )
    if node.args.kwarg:
        hints[f"**{node.args.kwarg.arg}"] = (
            ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else None
        )
    hints["return"] = ast.unparse(node.returns) if node.returns else None
    return hints


def build_embedding_input(
    name: str,
    docstring: str | None,
    type_hints: dict[str, str | None],
) -> str:
    """Build the compact text intended for embedding, excluding full source."""

    hint_text = ", ".join(
        f"{key}: {value}" if value else f"{key}: unknown"
        for key, value in type_hints.items()
    )
    parts = [f"name: {name}"]
    if docstring:
        parts.append(f"docstring: {docstring}")
    if hint_text:
        parts.append(f"type_hints: {hint_text}")
    return "\n".join(parts)


def iter_python_files(repo_root: Path) -> list[Path]:
    """Return Python files under a repo, excluding virtual env and VCS folders."""

    excluded = {".git", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache"}
    files: list[Path] = []
    for path in repo_root.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def extract_chunks(repo_root: Path) -> list[dict[str, Any]]:
    """Extract serializable function and class chunks from a Python repository."""

    repo_root = repo_root.resolve()
    chunks: list[CodeChunk] = []
    for file_path in iter_python_files(repo_root):
        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            continue
        collector = DefinitionCollector(source=source, file_path=file_path, repo_root=repo_root)
        collector.visit(tree)
        chunks.extend(collector.chunks)
    return [asdict(chunk) for chunk in chunks]


def write_chunks(repo_root: Path, output_path: Path) -> list[dict[str, Any]]:
    """Extract chunks and persist them as formatted JSON."""

    chunks = extract_chunks(repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return chunks
