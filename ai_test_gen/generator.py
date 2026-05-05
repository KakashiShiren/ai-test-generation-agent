"""LangChain test generation agent."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from ai_test_gen.vector_store import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME
from ai_test_gen.vector_store import retrieve_context as chroma_retrieve_context


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an expert Python test-generation agent.

You must retrieve relevant context before writing tests.
Generate a complete pytest test file for the target function.
Use the retrieved source, docstrings, and type hints to infer valid inputs.
Include at least 3 test cases:
1. happy path
2. edge case
3. error case

Prefer deterministic tests. Avoid network, filesystem, and live service calls.
Patch or monkeypatch collaborators when needed.
Do not invent exceptions for valid calls.
If a target has no invalid input, make the error case deterministic by monkeypatching
one of its collaborators to raise and asserting that error propagates.
Import the target from the source package using the file path in retrieved context.
Return only valid Python code. Do not include Markdown fences or explanations."""

USER_PROMPT = """Target function signature:
{function_signature}

Retrieved context from retrieve_context(function_signature):
{retrieved_context}

Source package root for imports:
{source_package_root}

Write the pytest file now."""


def retrieve_context_tool(
    function_signature: str,
    persist_dir: str = str(DEFAULT_CHROMA_DIR),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top semantically similar code chunks for a Python function signature."""

    return chroma_retrieve_context(
        function_signature=function_signature,
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        top_k=top_k,
    )


RETRIEVE_CONTEXT_TOOL = StructuredTool.from_function(
    func=retrieve_context_tool,
    name="retrieve_context",
    description="Retrieve top matching function/class chunks with full source code for a Python function signature.",
)


def get_generation_model(model: str = DEFAULT_GROQ_MODEL) -> ChatGroq:
    """Create the Groq chat model used for test generation."""

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required for Groq test generation.")
    return ChatGroq(model=model, temperature=0)


@retry(wait=wait_exponential(multiplier=1, min=1, max=45), stop=stop_after_attempt(6))
def invoke_generation_chain(chain: Any, payload: dict[str, Any]) -> str:
    """Invoke the LLM generation chain with retry and exponential backoff."""

    return chain.invoke(payload)


def generate_pytest_code(
    function_signature: str,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    model: str = DEFAULT_GROQ_MODEL,
    source_package_root: str = "requests",
) -> str:
    """Retrieve context and generate a complete pytest file as Python code."""

    contexts = RETRIEVE_CONTEXT_TOOL.invoke(
        {
            "function_signature": function_signature,
            "persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "top_k": top_k,
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )
    chain = prompt | get_generation_model(model=model) | StrOutputParser()
    generated = invoke_generation_chain(
        chain,
        {
            "function_signature": function_signature,
            "retrieved_context": json.dumps(contexts, indent=2),
            "source_package_root": source_package_root,
        },
    )
    return normalize_python_code(generated)


def normalize_python_code(text: str) -> str:
    """Strip common chat wrappers while preserving the Python program."""

    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    ast.parse(stripped)
    return stripped + "\n"


def infer_function_name(function_signature: str) -> str:
    """Infer a function name from a Python signature-like string."""

    try:
        tree = ast.parse(function_signature.strip() + "\n    pass\n")
        node = tree.body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    except SyntaxError:
        pass
    match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", function_signature)
    if match:
        return match.group(1)
    return "generated"


def safe_test_filename(function_signature: str) -> str:
    """Build a stable pytest filename for a function signature."""

    name = infer_function_name(function_signature)
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "generated"
    return f"test_{safe_name}.py"


def write_generated_test(
    function_signature: str,
    output_dir: Path = Path("generated_tests"),
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    model: str = DEFAULT_GROQ_MODEL,
    source_package_root: str = "requests",
) -> Path:
    """Generate and save a pytest file for one target function signature."""

    code = generate_pytest_code(
        function_signature=function_signature,
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_k=top_k,
        model=model,
        source_package_root=source_package_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / safe_test_filename(function_signature)
    output_path.write_text(code, encoding="utf-8")
    return output_path
