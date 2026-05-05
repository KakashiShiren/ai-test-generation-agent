# AI Test Generation Agent

LangChain-powered agent that retrieves Python code context from a local vector store and generates pytest files for target function signatures.

## Overview

This project ingests a real Python codebase, chunks it at the function/class level with `ast`, embeds compact semantic metadata into a persistent local ChromaDB store, and uses a LangChain generation chain to create pytest test files. The evaluation harness automatically generates tests for 50 target functions, runs them with pytest coverage, and writes a CSV report.

## Architecture

```text
Python repo
  |
  v
AST ingestion
  |  function/class chunks:
  |  name, source, docstring, type hints, path, lines
  v
JSON chunks ------------------------------+
  |                                       |
  v                                       |
OpenAI text-embedding-3-small             |
  |                                       |
  v                                       |
Persistent ChromaDB                       |
  |                                       |
  +--> retrieve_context(signature) -------+
          |
          v
LangChain + Groq LLM
          |
          v
Generated pytest file
          |
          v
pytest + pytest-cov evaluation harness
          |
          v
CSV results and coverage summary
```

## Tech Stack

- Python 3.11+
- LangChain
- ChromaDB, local persistent mode
- OpenAI `text-embedding-3-small` for embeddings
- Groq `llama-3.1-8b-instant` for generation
- pytest and pytest-cov
- Python `ast`
- Gradio demo app

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set API keys:

```powershell
$env:OPENAI_API_KEY="..."
$env:GROQ_API_KEY="..."
```

Clone and ingest the target repo:

```powershell
git clone https://github.com/psf/requests.git source_repos\requests
python scripts\ingest_codebase.py --repo-root source_repos\requests\src
```

Build the persistent vector store:

```powershell
python scripts\build_vector_store.py --reset
```

Generate one pytest file:

```powershell
python scripts\generate_test.py "def default_headers() -> CaseInsensitiveDict"
```

Run the evaluation harness:

```powershell
python scripts\evaluate_generated_tests.py --limit 50 --output-dir evaluation_runs\full
```

Run the demo app:

```powershell
python app.py
```

## Example

Input:

```python
def default_headers() -> CaseInsensitiveDict
```

Output:

```python
import pytest
from requests import utils
from requests.structures import CaseInsensitiveDict
from unittest.mock import patch

def test_default_headers_happy_path():
    headers = utils.default_headers()
    assert isinstance(headers, CaseInsensitiveDict)
    assert headers["Accept"] == "*/*"
    assert headers["Connection"] == "keep-alive"

def test_default_headers_edge_case():
    headers = utils.default_headers()
    assert headers["User-Agent"] != ""

def test_default_headers_error_case():
    with patch("requests.utils.default_user_agent", side_effect=Exception):
        with pytest.raises(Exception):
            utils.default_headers()
```

## Evaluation Results

The Phase 4 harness was run against 50 selected functions from `psf/requests`.

| Metric | Value |
| --- | ---: |
| Total functions | 50 |
| Passed generated test files | 4 |
| Pass rate | 8.0% |
| Mean target-line coverage | 48.45% |
| Functions with >=80% target coverage | 21 |
| Import errors | 5 |
| Assertion errors | 14 |
| Syntax errors | 15 |
| Other failures | 12 |

The low pass rate is intentional signal from a fully automated first-pass generator. The next improvement would be a repair loop that feeds pytest failures back into the agent and regenerates tests.

See `docs/evaluation_summary.csv` for the compact results snapshot.
