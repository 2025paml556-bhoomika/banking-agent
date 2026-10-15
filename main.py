import json

import ollama
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

MODEL = "qwen2.5:3b"
DATA_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"

# Intent classifier trained on the BANKING77 train split (77 intents).
train = pd.read_csv(DATA_URL)
classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
classifier.fit(train["text"], train["category"])
INTENTS = sorted(train["category"].unique())


def read_intent(message: str) -> dict:
    return {"intent": classifier.predict([message])[0]}


def execute_intent(intent: str) -> dict:
    if intent not in INTENTS:
        return {"error": f"Unknown intent '{intent}'. Call read_intent first."}
    # Stub. Replace with the call to the real banking backend.
    return {"intent": intent, "status": "done", "reference": f"REF-{abs(hash(intent)) % 10000:04d}"}


TOOL_FUNCS = {"read_intent": read_intent, "execute_intent": execute_intent}

STOP_WORD = "DONE"

DESCRIPTION = f"""You are a customer support assistant for an online bank.
Use the available tools to handle the customer's request.
When the request is fully handled, reply to the customer in two or three short sentences
and then write {STOP_WORD} on a new line."""

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


def handle(message: str) -> str:
    messages = [{"role": "system", "content": DESCRIPTION}, {"role": "user", "content": message}]
    for _ in range(10):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(response.message)
        content = response.message.content or ""
        if STOP_WORD in content or not response.message.tool_calls:
            return content.replace(STOP_WORD, "").strip()
        for call in response.message.tool_calls:
            try:
                output = TOOL_FUNCS[call.function.name](**call.function.arguments)
            except TypeError as e:
                output = {"error": str(e)}
            print(f"  [tool] {call.function.name}({call.function.arguments}) -> {output}")
            messages.append(
                {"role": "tool", "tool_name": call.function.name, "content": json.dumps(output)}
            )
    return "Sorry, I could not complete your request. Please contact support."


if __name__ == "__main__":
    print("Banking assistant ready. Press Ctrl-D to exit.")
    while True:
        try:
            message = input("\nCustomer: ").strip()
        except EOFError:
            break
        if message:
            print("Assistant:", handle(message))
