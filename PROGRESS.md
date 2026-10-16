# Project Progress Log

Simple record of what we did, why we did it, and what comes next.
Updated after each step.

## The goal in one paragraph

The assignment asks for a banking assistant that uses an LLM to understand
customer intents (BANKING77 dataset, 77 intents). The marks come from the
**experiments and the report** around the assistant, not from the assistant
alone: prompt comparison, decoding comparison, structured output, tool
calling, evaluation, and failure analysis. The original `main.py` had the
assistant but almost no experiments.

## Plan

```
[done] 1. fixed test set
[done] 2. controllable model
[done] 3. E1 baseline           accuracy 0.41
[done] 4. E2 prompt comparison  best 0.43 (few-shot), format fixed by constraint
[done] 5. E3 decoding comparison  greedy 0.43, sampling 0.38 to 0.44, no gain
[done] 6. E4 structured output  100/100 valid JSON, 86/100 pass schema
[done] 7. Adaptation: retrieval few-shot  accuracy 0.84 (from 0.43)
[next] 8. E6 failure analysis
       9. E7 final system + notebook
      10. report + presentation
```

---

## Step 1: Fixed test set

**Files:** `make_eval_subset.py`, `eval_subset.csv`

**What we did.** Loaded the BANKING77 test split (3080 rows), took 100 rows
with a fixed random seed, and saved them to `eval_subset.csv`. Each row has
the customer message (`text`) and the correct intent (`category`).

**Why.** Every experiment must run on the same 100 messages. If prompt A
scores 60 and prompt B scores 70, the comparison is valid only when both
used the same inputs. This file is the one exam paper for all experiments.

**Details that matter.**

- Test split, not train split. The TF-IDF classifier in `main.py` learned
  from the train split, so train rows would give results that look better
  than they are.
- 100 rows. Small enough to run in minutes on a laptop.
- Fixed seed (`random_state=0`). Anyone can run the script and get the same
  rows. The report can say "evaluated on a fixed 100-example subset".

**Run:** `uv run make_eval_subset.py`

---

## Step 2: Controllable model

**Files:** `llm.py`, `pyproject.toml` (added `transformers`, `torch`)

**What we did.** Loaded `Qwen/Qwen2.5-1.5B-Instruct` through Hugging Face
`transformers`. One function does all generation:

```python
generate(messages, max_new_tokens=64, **decoding) -> str
```

Default is greedy. Pass `do_sample=True, temperature=0.7, top_k=50,
top_p=0.9` for sampling.

**Why.** The old code used Ollama. Ollama hides the tokenizer, the logits,
and the decoding settings. The assignment requires us to explain the
tokenizer (Req 4), use a Hugging Face causal LM (Req 5), and compare
decoding settings (Req 7). Hugging Face gives direct access to all three.

**Details that matter.**

- Same model family as before (Qwen 2.5), smaller size (1.5B instead of
  3B) so 100 generations run in a few minutes on a Mac.
- Runs on `mps` (Apple GPU) with `bfloat16`.
- `main.py` still uses Ollama for the tool-calling agent. Experiments
  import `llm.generate` instead. Tool calling through Ollama already
  satisfies E5.

**Run:** `uv run llm.py` (one test generation)

---

## Step 3: E1 baseline

**Files:** `baseline.py`, `results_baseline.csv`

**What we did.** One simple prompt, greedy decoding, 100 rows:

```
Classify the customer message into one of these intents:
activate_my_card, age_limit, ... (all 77)

Message: How do I track my card?
Intent:
```

No role, no examples, no output rules. The model's text output is parsed
with a loose match: the longest intent name that appears in the output
counts as the prediction. The CSV stores `text, category, output,
predicted, correct`.

**Result: accuracy 0.41** (41 of 100). 4 outputs had no valid intent.
For reference, the old TF-IDF classifier scores 0.79 on the same rows.

**Why.** This is the starting line. E2, E3, and E6 compare against this
number and reuse this loop. It also fixes design gap 1: the LLM now
identifies the intent, not the TF-IDF classifier.

**What we saw (useful for E2 and E6).**

- The model often writes `The intent for the message "..." is:` before
  the answer. With 16 output tokens, 34 answers were cut off and accuracy
  was 0.27. With 32 tokens, accuracy was 0.41. A constraint prompt
  ("answer with the intent name only") should remove this waste.
- The model sometimes invents intent names that are not in the list, for
  example `top_up_verification` and `Top_up_by_cheque_or_cash_deposit`
  (real name: `top_up_by_cash_or_cheque`).
- The model sometimes refuses: "does not match any of the provided
  intents".
- Many errors are between near intents: `card_arrival` vs
  `card_delivery_estimate`, `exchange_via_app` vs
  `supported_cards_and_currencies`. Few-shot examples should help here.
- One BANKING77 intent has a capital letter (`Refund_not_showing_up`).
  The parser must compare in lowercase.

**Failure groups (59 wrong answers).** Input for E6.

| Group | Count | Example (message -> model / correct) | Likely cause | Fix to test |
|---|---|---|---|---|
| Near-synonym intent | ~35 | "How do I track my card?" -> `card_delivery_estimate` / `card_arrival` | Model sees only 77 names, not the dataset boundary between them | Few-shot (E2) |
| Default to a favourite intent | ~10 | "Why would I have a pending payment?" -> `pending_transfer` / `pending_card_payment` | Short generic names attract a small model; `pending_transfer`, `top_up_failed`, `get_physical_card`, `activate_my_card` absorb many messages | Few-shot, larger model |
| Wrong focus in message | ~6 | "My top-up for Google Pay isn't working" -> `top_up_failed` / `apple_pay_or_google_pay` | Model latches on one word ("top-up") and ignores the rest | Few-shot, role prompt |
| Invented intent name | 3 | "How is the top-up card verified?" -> `top_up_verification` / `verify_top_up` | Model generates text, does not select from a list | Constraint prompt (E2), JSON schema (E4) |
| Refusal | 2 | "How many miles away is the ATM?" -> "does not match any of the provided intents" / `atm_support` | Model is honest when unsure; a classifier must pick the closest | Constraint prompt ("always choose one") |

Also: 57 of 100 outputs start with "The intent for the message ... is:".
Not wrong, but it wastes tokens and needs a loose parser.

**Run:** `uv run baseline.py` (about 2 minutes on MPS)

---

## Step 4: E2 prompt comparison

**Files:** `prompts.py`, `results_<style>.csv` (one per prompt)

**What we did.** Five prompt styles, same 100 rows, greedy decoding,
same parser. `build_prompt(message, style)` builds each one. This
function is also the "programmatic prompting" requirement (Req 11).

| Style | What is added to the E1 prompt |
|---|---|
| `zero_shot` | Nothing. Same as E1. |
| `role` | System message: "You are a customer support assistant for an online bank." |
| `constraint` | "Reply with the exact intent name from the list and nothing else. Always choose exactly one intent." |
| `few_shot` | 5 solved examples from the train split (fixed seed) before the message. |
| `few_shot_constraint` | Both of the above. Added after we saw the results of the first four. |

**Result.**

| Prompt | Accuracy | No valid intent | Outputs with preamble |
|---|---|---|---|
| zero_shot | 0.41 | 4 | 57 |
| role | 0.40 | 4 | 30 |
| constraint | 0.40 | 2 | 0 |
| few_shot | **0.43** | 12 | 1 |
| few_shot_constraint | **0.43** | 8 | 0 |

**What the table says, in simple words.**

- Accuracy barely moves: 0.40 to 0.43. The prompt is not the main
  problem. The 1.5B model cannot separate 77 near-synonym intents from
  the names alone.
- Format changes a lot. The constraint removed all preamble text (57 to
  0) and halved the invented names. Few-shot also removed the preamble
  (57 to 1) because the examples show the expected shape.
- Few-shot had a side effect: the model copied the snake_case style and
  invented more names that look real but are not in the list
  (`get_virtual_card`, `top_up_by_cheque`, `refund_not_shown_up`). The
  constraint reduced this from 12 to 8 but did not remove it.
- The role prompt did nothing useful. It made the model more talkative
  in some rows ("The customer is asking about...") and did not change
  accuracy.
- 40 of 100 rows were wrong under all five prompts. 22 were right under
  all five. The hard rows are hard for every prompt. This supports the
  adaptation decision (Req 8): prompting alone is not enough; this is a
  case for PEFT or for retrieval of similar labelled examples (RAG).

**Choice for E3.** `few_shot_constraint`. Same accuracy as `few_shot`,
cleaner output, fewer invented names.

**Run:** `uv run prompts.py` (about 8 minutes for all five on MPS)

---

## Decision: stay on the 1.5B model

The Mac has 8 GB RAM. Hugging Face loads full bf16 weights, so
Qwen2.5-3B (6 GB) would swap or crash, and 7B does not fit. The old
Ollama 3B worked because Ollama uses 4-bit weights. We finish all
experiments on 1.5B. A bigger model can be one extra experiment later on
Colab, where a free T4 GPU has 16 GB.

---

## Step 5: E3 decoding comparison

**Files:** `decoding.py`, `results_decoding_<setting>.csv`

**What we did.** Same prompt (`few_shot_constraint`), same 100 rows,
five decoding settings. Sampling runs use `torch.manual_seed(0)` so they
repeat.

| Setting | Parameters |
|---|---|
| greedy | `do_sample=False`. Always the most likely next token. |
| temperature_0.7 | `temperature=0.7`. Slightly random, favours likely tokens. |
| temperature_1.5 | `temperature=1.5`. Very random, flattens the choice. |
| top_k_50 | `top_k=50`. Random pick among the 50 most likely tokens. |
| top_p_0.9 | `top_p=0.9`. Random pick among the tokens that cover 90% of probability. |

**Result.**

| Decoding | Accuracy | No valid intent | Outputs different from greedy | Right/wrong flipped vs greedy |
|---|---|---|---|---|
| greedy | 0.43 | 8 | 0 | 0 |
| temperature_0.7 | 0.44 | 10 | 20 | 5 |
| temperature_1.5 | 0.40 | 7 | 42 | 13 |
| top_k_50 | 0.44 | 7 | 29 | 7 |
| top_p_0.9 | 0.38 | 11 | 28 | 11 |

**What the table says, in simple words.**

- Decoding does not help a classification task. The answer is one short
  label. There is one right token sequence. Randomness can only move
  away from the model's best guess. Gains of +0.01 are noise: the
  flipped rows go both ways (some wrong become right, some right become
  wrong).
- Higher randomness means more invented labels. At temperature 1.5 the
  model produced `unable_to_verify_identity`, `transaction_charged_twice`,
  and `intent unknown`, none of which fit the message.
- Greedy is the right choice for the final system: deterministic,
  reproducible, and as accurate as anything else.
- Sampling has a place in the customer-facing reply (the two or three
  sentences the agent writes at the end), where variety reads better than
  a fixed template. It has no place in the intent step.

**Choice for E4 and E7.** Greedy.

**Run:** `uv run decoding.py` (about 8 minutes on MPS)

---

## Step 6: E4 structured output

**Files:** `structured.py`, `results_structured.csv`

**What we did.** Asked the model to answer in JSON instead of plain
text, then checked every answer against a schema with the `jsonschema`
library. Same 100 rows, greedy, 5 few-shot examples written in JSON.

The schema:

```json
{
  "type": "object",
  "properties": {
    "intent":     {"type": "string", "enum": ["<the 77 intent names>"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["intent", "confidence"],
  "additionalProperties": false
}
```

The `enum` is the important part. It turns "is this a real intent?"
from a text-matching guess into a hard check.

**Result.**

| Check | Count |
|---|---|
| Output is valid JSON | 100 / 100 |
| Output passes the schema | 86 / 100 |
| Accuracy, all rows | 0.43 |
| Accuracy, schema-valid rows only | 0.50 |

All 14 schema failures were the same kind: an intent name that is not in
the list. `get_virtual_card` (4), `top_up_by_card` (4),
`refund_not_showing_up` (3, lowercase; the real label has a capital R),
`top_up_by_cheque` (2), `exchange_currency` (1).

**What this says, in simple words.**

- The model can follow a JSON format. 100 of 100 outputs parsed. 39
  were wrapped in ```` ```json ```` fences, which the parser strips.
- The schema catches the invented names that the loose text parser in
  E1 and E2 could not. In the final system these 14 rows become "ask the
  customer to rephrase" instead of a wrong action.
- Accuracy among valid outputs is 0.50. Validation did not make the
  model smarter; it filtered out a group that was always wrong.
- The confidence field is not useful. The model copied the example
  value: 84 rows said 0.9, 16 said 0.95. Mean confidence was 0.91 for
  right answers and 0.91 for wrong answers. A small model does not
  estimate its own certainty from a prompt. Real confidence needs the
  logits, which Hugging Face exposes; this is a possible extension.

**Sample validated outputs for the report.**

```json
{"intent": "card_arrival", "confidence": 0.9}         <- "How long should my new card take to arrive?"  valid, correct
{"intent": "exchange_rate", "confidence": 0.9}        <- "What foreign exchange rate will you use?"      valid, correct
{"intent": "card_delivery_estimate", "confidence": 0.9} <- "How do I track my card?"                    valid, wrong
{"intent": "get_virtual_card", "confidence": 0.9}     <- "Can I order a virtual card?"                   rejected by schema
```

**Run:** `uv run structured.py` (about 3 minutes on MPS)

---

## Step 7: Adaptation experiment, retrieval few-shot (Req 8)

**Files:** `retrieval.py`, `results_retrieval.csv`

**What we did.** Same prompt shape as `few_shot_constraint`, but the 5
examples are no longer random. For each customer message we find the 5
train messages with the highest TF-IDF cosine similarity and use those
as the examples. This is retrieval-augmented prompting (RAG) with the
labelled train split as the knowledge base. Greedy, 100 rows, same
parser.

**Result.**

| System | Accuracy |
|---|---|
| LLM, 5 random examples (E2 best) | 0.43 |
| 5-NN vote on the retrieved examples, no LLM | 0.77 |
| TF-IDF + Logistic Regression (old `main.py`) | 0.79 |
| **LLM, 5 retrieved examples** | **0.84** |

The correct label was present among the 5 retrieved examples in 94 of
100 rows. "No valid intent" dropped to 3.

**What this says, in simple words.**

- The 1.5B model was never the problem. It was missing the knowledge of
  where the label boundaries are. Five relevant examples supply that
  knowledge, and accuracy doubles.
- The LLM adds value above the retrieval. A plain majority vote on the
  same 5 examples gives 0.77. The LLM reads the examples and the message
  and picks better: it beat the vote in 12 rows and lost in 5.
- It also beats the TF-IDF classifier (0.79), the previous best, without
  any training.
- This is the adaptation decision for the report: prompting alone
  (E2, E3) stayed at 0.40 to 0.44; retrieval raised it to 0.84; fine-tuning
  (PEFT) was not needed to pass the classical baseline and is too heavy
  for an 8 GB machine. Choose RAG.

**Remaining failures (16).** Most are the same near-synonym pairs as
before (`card_arrival` / `card_delivery_estimate`, `exchange_via_app` /
`supported_cards_and_currencies`, `verify_my_identity` /
`why_verify_identity`). In 11 rows the kNN vote was also wrong, which
means the retrieved examples pointed the wrong way. Three rows got a
refusal ("intent not specified"). These feed E6.

**Run:** `uv run retrieval.py` (about 2 minutes on MPS)

---

## Steps 8 to 10 (planned)

| Step | What | Output for the report |
|---|---|---|
| E6 failure analysis | Pick 3+ wrong answers from the result CSVs. Explain why. | Failure table |
| E7 final system | Best prompt inside `main.py`. Notebook that runs end to end. | Demo |
| Report | Write up all of the above. 10 to 15 pages. | Report + slides |
