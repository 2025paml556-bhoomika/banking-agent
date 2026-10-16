"""E7: final banking assistant. One Hugging Face model does three jobs:
tool selection, intent classification (retrieval few-shot), and the customer reply."""

import json

from jsonschema import ValidationError, validate

from baseline import INTENTS, parse_intent
from llm import generate
from retrieval import build_prompt

INTENT_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": INTENTS}},
    "required": ["intent"],
    "additionalProperties": False,
}


def read_intent(message: str) -> dict:
    """Classify with the LLM using the 5 most similar labelled examples, then validate."""
    output = generate(build_prompt(message), max_new_tokens=32)
    result = {"intent": parse_intent(output)}
    try:
        validate(result, INTENT_SCHEMA)
    except ValidationError:
        return {"error": f"Could not map the message to a known intent (model said {output!r})."}
    return result


def execute_intent(intent: str) -> dict:
    if intent not in INTENTS:
        return {"error": f"Unknown intent '{intent}'. Call read_intent first."}
    # Stub. Replace with the call to the real banking backend.
    return {"intent": intent, "status": "done", "reference": f"REF-{abs(hash(intent)) % 10000:04d}"}


TOOL_FUNCS = {"read_intent": read_intent, "execute_intent": execute_intent}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_intent",
            "description": "Classify the customer message into one of the 77 BANKING77 intents.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_intent",
            "description": "Execute the banking action for the intent returned by read_intent.",
            "parameters": {
                "type": "object",
                "properties": {"intent": {"type": "string"}},
                "required": ["intent"],
            },
        },
    },
]

SYSTEM = """You are a customer support assistant for an online bank.
Reply to the customer in two or three short sentences based only on the tool results.
Mention the reference number. Do not claim any action that the tool results do not show."""


def call_tool(messages: list[dict], name: str, **arguments) -> dict:
    """Run one tool and record the call and its result in the chat history (Qwen tool-call format)."""
    output = TOOL_FUNCS[name](**arguments)
    print(f"  [tool] {name}({arguments}) -> {output}")
    messages.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]})
    messages.append({"role": "tool", "content": json.dumps(output)})
    return output


def handle(message: str) -> str:
    # The code fixes the tool order. The 1.5B model did not do this reliably on its own (see PROGRESS.md, step 9).
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message}]
    intent = call_tool(messages, "read_intent", message=message)
    if "error" in intent:
        return "Sorry, I could not understand your request. Could you say it in a different way?"
    call_tool(messages, "execute_intent", intent=intent["intent"])
    # Sampling is used only here, for the free-text reply (E3 conclusion).
    return generate(messages, max_new_tokens=80, tools=TOOLS, do_sample=True, temperature=0.7).strip()


if __name__ == "__main__":
    print("Banking assistant ready. Press Ctrl-D to exit.")
    while True:
        try:
            message = input("\nCustomer: ").strip()
        except EOFError:
            break
        if message:
            print("Assistant:", handle(message))
