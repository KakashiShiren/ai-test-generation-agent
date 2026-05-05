"""Gradio demo for the AI Test Generation Agent."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ai_test_gen.generator import generate_pytest_code


def generate_tests(function_signature: str) -> str:
    """Generate pytest code for the Gradio interface."""

    signature = function_signature.strip()
    if not signature:
        return "Paste a Python function signature first."
    try:
        return generate_pytest_code(
            function_signature=signature,
            persist_dir=Path("data/chroma"),
            collection_name="code_chunks",
        )
    except Exception as error:  # noqa: BLE001 - surface demo failures in the UI.
        return f"{type(error).__name__}: {error}"


with gr.Blocks(title="AI Test Generation Agent") as demo:
    gr.Markdown("# AI Test Generation Agent")
    gr.Markdown("Generate pytest code from a Python function signature using retrieved code context.")
    signature_input = gr.Textbox(
        label="Function signature",
        lines=3,
        value="def default_headers() -> CaseInsensitiveDict",
    )
    generate_button = gr.Button("Generate tests", variant="primary")
    output = gr.Code(label="Generated pytest", language="python", lines=22)
    generate_button.click(
        fn=generate_tests,
        inputs=signature_input,
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
