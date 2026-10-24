---
marp: true
paginate: true
---

# Banking Intent Assistant on BANKING77

PCAM ZG516 Applied Generative AI, Group Major Project, Assignment 1

Group _[number]_: _[name, ID]_ · _[name, ID]_ · _[name, ID]_ · _[name, ID]_

Code: https://github.com/2025paml556-bhoomika/banking-agent

<!-- 0:30. One sentence: a customer writes one free-text message, the assistant finds the intent, calls the right tool, and replies. -->

---

# 1. Problem and dataset

- Customer of an online bank writes one message in free text: "I lost my card, please block it"
- The bank must route it to one of **77 intents**, trigger the action, and answer
- Dataset D1, BANKING77: 13,083 messages, 77 labels, fixed train/test split
- Train (10,003 rows): source of examples only. No training
- Two fixed 100-row test samples: **development** (every experiment) and **held-out** (final system, once)
- Why an LLM: paraphrase and typos in, fluent grounded reply out, no training needed, one model for intent + tool call + reply

<!-- 1:00. Point out that the held-out set was never touched until the end. -->

---

# 2. Model and architecture

- `Qwen/Qwen2.5-1.5B-Instruct`, Hugging Face `transformers`, bfloat16, Apple M3 with 8 GB, MPS
- Autoregressive LM chosen over VAE and GAN: text in, discrete tokens out, one conditioning mechanism (the prefix), steerable at run time
- Four stages shown on a real message, "How do I track my card?"
  - Tokenization: 7 tokens
  - Embeddings: shape 1 × 7 × 1536
  - LM head: top-5 next-token probabilities
  - Greedy generation
- Four run-time levers, one experiment each: **prompt, decoding, output schema, tools**

<!-- 1:00. The notebook section 2 prints these four stages live. -->

---

# 3. E1 Baseline and E2 Prompts

| Prompt | Accuracy | No valid intent | Preamble |
|---|---|---|---|
| `zero_shot` (baseline) | 0.41 | 4 | 57 |
| `role` | 0.40 | 4 | 30 |
| `constraint` | 0.40 | 2 | 0 |
| `few_shot` (5 random) | 0.43 | 12 | 1 |
| `few_shot_constraint` | 0.43 | 8 | 0 |

- Classical reference on the same rows: TF-IDF + logistic regression **0.79**
- Instructions fix **format** (preamble 57 → 0), not **knowledge** (accuracy flat at 0.40 to 0.43)
- Failures: near-synonym labels (35), favourite labels (10), invented names (3), refusals (2)

<!-- 1:00. The one message to quote: "track my card" → card_delivery_estimate instead of card_arrival. The names alone do not say where the boundary is. -->

---

# 4. E3 Decoding

| Setting | Accuracy | No valid intent | Flips vs greedy |
|---|---|---|---|
| `greedy` | 0.43 | 8 | - |
| `temperature_0.7` | 0.44 | 10 | 5 |
| `temperature_1.5` | 0.40 | 7 | 13 |
| `top_k_50` | 0.44 | 7 | 7 |
| `top_p_0.9` | 0.38 | 11 | 11 |

- A one-label answer has one right token sequence; sampling can only move away from the best guess
- Flips went both ways; at the strongest settings more right→wrong than wrong→right
- Decision: **greedy for the intent, sampling only for the free-text reply**

<!-- 0:45. -->

---

# 5. E4 Structured output

```json
{"type": "object", "required": ["intent", "confidence"],
 "properties": {"intent": {"type": "string", "enum": ["<77 names>"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}
```

| Check | Count |
|---|---|
| Parses as JSON | 100 / 100 |
| Passes the schema | 86 / 100 |
| Accuracy, all rows / schema-valid rows | 0.43 / 0.50 |

- All 14 rejections were **invented intent names**: `get_virtual_card`, `top_up_by_card`, `refund_not_showing_up` (lowercase r)
- The `enum` turns "is this a real intent?" into a hard check. `confidence` was not informative (model copied 0.9)

<!-- 0:45. Hallucination in the classification setting: the model generates a value that looks right instead of selecting one. -->

---

# 6. Adaptation decision: retrieval, not fine-tuning

| System | Uses LLM | Trained | Examples per message | Accuracy |
|---|---|---|---|---|
| `few_shot_constraint` (before) | yes | no | 5, same for all | 0.43 |
| 5-NN vote on retrieved rows | no | no | 5, per message | 0.77 |
| TF-IDF + logistic regression | no | yes | - | 0.79 |
| **Retrieval few-shot + constraint** | yes | no | 5, per message | **0.84** |

- The errors were missing knowledge, so the fix must add knowledge: RAG over the train file (TF-IDF cosine, 5 nearest rows as examples)
- Correct label present among the 5 retrieved rows in 94 of 100 cases
- +41 rows net against random examples; +7 net against the vote: the LLM adds on top of the examples
- Fine-tuning not needed: the classical bar is passed with no training, on a laptop

<!-- 1:15. This is the key decision slide. -->

---

# 7. Final system: tools and the agent loop

![width:900px](images/sequence.png)

- Two tools with JSON schemas: `classify_intent(message)` → validated intent; `handle_intent(intent)` → `done` / `information` / `unsupported` with a reference number
- **The code fixes the tool order.** The 1.5B model driving the loop itself gave 0 of 8 correct sequences (4 messages, two setups): skipped tools, invented `BLOCK_CARD_LOST`, claimed actions not taken
- Reply is sampled from the real tool transcript; "Do not claim any action that the tool results do not show"

<!-- 1:00. Live demo here if time: `uv run main.py`, type "I lost my card, please block it". -->

---

# 8. Evaluation and failure analysis

| Subset | Accuracy | No valid intent |
|---|---|---|
| Development (chose the approach) | 0.84 | 3 |
| **Held-out (used once)** | **0.77** | 4 |

Seven failure types, eleven worked cases in `docs/FAILURES.md`:

| Type | Example | Fixed by |
|---|---|---|
| A. Near-synonym label | card_arrival vs card_delivery_estimate | Retrieval, mostly |
| C. Invented label | `top_up_verification` | Schema enum |
| D. Refusal | "none of the intents match" | Constraint line |
| F. Retrieval misleads | nearest rows share words, not meaning | Sentence embeddings (future) |
| G. Random decoding | right answer sampled away | Greedy |

- Honest claim: **matches a trained classifier with no training**, and also writes the reply and drives the tool call

<!-- 1:00. 0.77 is the number to quote. Level with the classical reference, not clearly above. -->

---

# 9. Responsible use and limits

- Wrong intent → wrong action: schema enum, "rephrase" path, and a confirmation step before irreversible actions in production
- Hallucinated claims: control in code, language in the model; reply generated from the tool transcript
- Prompt injection: the message enters only as data; tool order and arguments are set by code
- Privacy: model runs locally; PII masking before retrieval and logs in production
- Limits: 1.5B model on 8 GB; single turn; `handle_intent` is a simulation; TF-IDF retrieval matches words, not meaning

**Reproduce:** `notebooks/banking_agent.ipynb` runs every experiment; about 21 minutes on an M3 Mac

<!-- 0:45. Close with the three principles: control in code, validate at the boundary, fail towards a human. -->

---

# Contributions

| Member | Work |
|---|---|
| _[name, ID]_ | _[...]_ |
| _[name, ID]_ | _[...]_ |
| _[name, ID]_ | _[...]_ |
| _[name, ID]_ | _[...]_ |

Questions

<!-- Total: about 9 minutes. -->
