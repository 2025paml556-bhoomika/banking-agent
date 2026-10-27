# Fine-tune plan

## Goal

Teach `Qwen/Qwen2.5-1.5B-Instruct` to name the BANKING77 intent for a customer message. The current assistant already writes the reply from tool results. The train file has no reply text, so this run trains the intent step only.

## Data

Train on `data/train.csv` only.

- 10,003 rows.
- Two columns: `text` (customer message), `category` (one of 77 intent names).
- Same file the assistant already loads for the intent list and for retrieval examples.

Hold out 500 rows from `data/train.csv` as a validation set (`random_state=0`). Train on the other 9,503 rows. The 500 rows are for early stop and for a loss check during training. They are not the project test sets.

Do not train on these files:

- `data/test.csv` (3,080 rows)
- `evaluation/eval_subset.csv` (100 development rows)
- `evaluation/heldout_subset.csv` (100 held-out rows)

Those three files come from the BANKING77 test split. They stay unseen until scoring.

## Example format

Each train row becomes one chat turn, in the same shape as `src/baseline.py`:

```
User: Classify the customer message into one of these intents:
<77 names>

Message: <text>
Intent:

Assistant: <category>
```

## Method

Use LoRA on the 1.5B model. Full training of all weights needs more memory than this project targets.

- Base model: `Qwen/Qwen2.5-1.5B-Instruct` (the `MODEL` environment variable can point at another id).
- Adapter: LoRA on the attention projections.
- Objective: next-token loss on the assistant label only.
- Decode at check time with greedy decoding and `parse_intent` from `src/baseline.py`.
- Save the adapter under `models/intent-lora/`.

## Score

After training, run the adapter on the two existing subsets. Compare accuracy with the saved retrieval result (0.84 on the development subset, 0.77 on the held-out subset for the 1.5B model).

| Set | File | Use |
|---|---|---|
| Development | `evaluation/eval_subset.csv` | Compare with retrieval |
| Held-out | `evaluation/heldout_subset.csv` | One final score |

## Steps

1. Build the train and validation tables from `data/train.csv`.
2. Train the LoRA adapter with `uv`.
3. Load the adapter in `generate` when `ADAPTER` is set.
4. Score the development subset, then the held-out subset once.
5. Write the two scores next to the retrieval scores.

## Out of scope

- No change to tool actions or to the reply text.
- No training on test rows.
- No second model size in this pass.
