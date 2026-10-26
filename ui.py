"""Browser UI to test the assistant. Each message runs main.handle(); the tool calls are shown next to the reply.
Start with `uv run ui.py` and open http://127.0.0.1:7860."""

import contextlib
import io

import gradio as gr

from main import handle

EXAMPLES = [
    "I lost my card yesterday, please block it",
    "How long does a card take to arrive?",
    "Can you cancel the transfer I just made?",
    "A payment I made was declined, why?",
    "What is the exchange rate you use?",
]


def respond(message, history):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):  # main.call_tool prints one "[tool] ..." line per call
        reply = handle(message)
    trace = out.getvalue().strip().replace(" -> ", "\n  -> ").replace("  [tool]", "\n[tool]")
    return reply, trace


with gr.Blocks(title="Banking assistant") as demo:
    gr.Markdown("# Banking assistant\nEach message is handled on its own (no conversation memory).")
    with gr.Row():
        with gr.Column(scale=2):
            trace = gr.Code(label="Tool calls (last message)", language=None, lines=14, wrap_lines=True, show_line_numbers=False, render=False)
            gr.ChatInterface(respond, examples=EXAMPLES, additional_outputs=[trace])
        with gr.Column(scale=1):
            trace.render()

if __name__ == "__main__":
    demo.launch()
