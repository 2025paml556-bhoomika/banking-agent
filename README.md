# Banking Intent Agent

A customer support assistant for an online bank. A local Hugging Face LLM
(`Qwen/Qwen2.5-1.5B-Instruct`) reads a customer message, finds the intent
(one of the 77 BANKING77 intents), runs a tool for that intent, and writes
a short reply.

The project has two parts:

1. **Experiments** (E1 to E7) that compare prompts, decoding settings,
   structured output, and retrieval. Each experiment is one script and
   writes one or more `results_*.csv` files to `evaluation/results/`.
2. **The final assistant** (`main.py`) that uses the best settings from the
   experiments.

A later LoRA follow-up trains an intent adapter. The commands are in
`docs/FINETUNE.md`. Those scores are for the intent name only. They are
not a new score for `main.py`.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) (installs Python and all packages)
- About 4 GB of free disk space for the model
- 8 GB RAM. The scripts run on Apple GPU (`mps`), NVIDIA GPU (`cuda`), or CPU.

## Setup

```bash
git clone https://github.com/2025paml556-bhoomika/banking-agent.git
cd banking-agent
uv sync
```

`uv sync` creates `.venv` and installs the packages from `uv.lock`.
The model downloads automatically the first time you run a script
(about 3 GB, cached by Hugging Face in `~/.cache/huggingface`).

## Quick start

Run the final assistant:

```bash
uv run main.py
```

Type a message, for example `I lost my card, please block it`.
The assistant prints each tool call and then the reply. Actions that change
the account (block a card, cancel a transfer, ...) are confirmed first: the
assistant says what it understood and acts only when you answer `yes`.
Press Ctrl-D to exit.

Or test it in the browser:

```bash
uv run ui.py
```

Open http://127.0.0.1:7860. The page shows the chat on the left and the
tool calls for the last message on the right. The first start takes about
30 seconds to load the model.

Run one test generation to check that the model loads:

```bash
uv run -m src.llm
```

## Run the experiments

Run the scripts in this order from the project root. Each one reads
`evaluation/eval_subset.csv` (the same 100 test messages) and writes its
results as CSV to `evaluation/results/`.
Times are for an M3 Mac with 8 GB RAM.

| Step | Command | Writes | Time |
|---|---|---|---|
| Subsets | `uv run -m evaluation.make_eval_subset` | `eval_subset.csv`, `heldout_subset.csv` (already in the repo) | seconds |
| E1 baseline | `uv run -m src.baseline` | `results_baseline.csv` | 2 min |
| E2 prompts | `uv run -m src.prompts` | `results_<style>.csv` (5 files) | 8 min |
| E3 decoding | `uv run -m src.decoding` | `results_decoding_<setting>.csv` (5 files) | 8 min |
| E4 structured output | `uv run -m src.structured` | `results_structured.csv` | 3 min |
| Adaptation (retrieval) | `uv run -m src.retrieval` | `results_retrieval.csv` | 2 min |
| Reference classifier | `uv run -m evaluation.reference_classifier` | `results_tfidf_logreg.csv` | seconds |
| Final, held-out | `uv run -m evaluation.final_eval` | `results_heldout.csv` | 2 min |
| LoRA train | `uv run -m src.finetune` | `models/intent-lora-retrieval/` | 2 h 53 min, RTX 3060 12 GB |
| LoRA score | `uv run -m evaluation.finetune_eval` | `results_finetune_retrieval_dev.csv`, `results_finetune_retrieval_heldout.csv` | minutes on that GPU |

The experiments use `evaluation/eval_subset.csv` (development set) to compare
approaches. `evaluation/final_eval.py` runs the chosen system once on
`evaluation/heldout_subset.csv`, 100 different test rows. The LoRA scorer
reads that file too. The earlier baseline-prompt adapter is already in
`models/intent-lora/` (0.93 on both subsets). The current train command
does not write that folder.

You do not have to rerun the experiments to read the results. All
`results_*.csv` files are in `evaluation/results/`.

To compare a larger model, set the `MODEL` variable before any command:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct uv run -m src.baseline
```

The 3B model needs about 6 GB RAM and does not fit on an 8 GB Mac.
Use Google Colab for it.

`evaluation/results/` holds the results of the default model. Every script
writes there, so after a run with another model, copy the files to a
subfolder named after the model before the next default run overwrites
them. `evaluation/results/phi4/` holds the same 15 files from a run with
`MODEL=microsoft/Phi-4-mini-instruct` on a Colab T4.

## Notebooks

`notebooks/banking_agent.ipynb` runs the experiment commands in the table
above, except the two LoRA rows, shows the result tables, and runs the
live demo. "Run all" takes about 21 minutes on an M3 Mac. The outputs are saved in the file, so you
can read the results without running it. To run it in about one minute,
skip the `%run -m <module>` cells; the result cells read the saved CSV
files.

The notebook imports the code from `main.py` and the `src/` modules. On
Colab, the first cell clones the repository into the session, installs the
dependencies, and links `evaluation/results/` to
`MyDrive/banking-agent-results/<model name>/` in Google Drive, so the
result files stay in Drive and nothing else is copied there. To run a
different model on Colab, set `os.environ["MODEL"]` in that cell; the
notebook clones the GitHub repository, so push your changes first.

To open a notebook locally, open it in VS Code or Cursor and select the
`.venv` Python as the kernel. The `ipykernel` package is already installed
by `uv sync`.

## Files

| File | Purpose |
|---|---|
| `main.py` | E7. The final assistant with `classify_intent` and `handle_intent` tools. |
| `ui.py` | Browser chat page (Gradio) that runs `main.handle()` and shows the tool calls. |
| `src/llm.py` | Loads the model. `generate(messages, **decoding)` is the one function all scripts use. |
| `src/baseline.py` | E1. Loads the 77 intent names, the simplest prompt, and `parse_intent`. |
| `src/prompts.py` | E2. Five prompt styles: zero-shot, role, constraint, few-shot, few-shot + constraint. |
| `src/decoding.py` | E3. Greedy, temperature 0.7, temperature 1.5, top-k 50, top-p 0.9. |
| `src/structured.py` | E4. JSON output validated with `jsonschema`. |
| `src/retrieval.py` | Adaptation. Five examples chosen by TF-IDF similarity. |
| `src/finetune.py` | LoRA training on the retrieval prompt. Saves `models/intent-lora-retrieval/`. |
| `evaluation/finetune_eval.py` | Scores that adapter on the development subset, then the held-out subset. |
| `data/train.csv`, `data/test.csv` | BANKING77 train and test splits (local copies from the PolyAI GitHub repository). |
| `evaluation/make_eval_subset.py` | Creates `eval_subset.csv` and `heldout_subset.csv` from `data/test.csv`. |
| `evaluation/reference_classifier.py` | TF-IDF + logistic regression, trained on the train split, scored on the development subset. The classical bar to beat. |
| `evaluation/final_eval.py` | Runs the final system once on the held-out subset. |
| `evaluation/eval_subset.csv` | Development set: the 100 test messages used by every experiment. |
| `evaluation/heldout_subset.csv` | Held-out set: 100 other test messages, used by `final_eval.py` and `finetune_eval.py`. |
| `evaluation/results/results_*.csv` | Output of each experiment with the default model (Qwen2.5-1.5B). |
| `evaluation/results/phi4/results_*.csv` | The same experiments with `microsoft/Phi-4-mini-instruct`. |
| `docs/PROGRESS.md` | Step-by-step log: what we did, why, and what we found. |
| `docs/FAILURES.md` | E6. Failure types and worked failure cases. |
| `docs/GAPS.md` | Gap analysis of the first version against the assignment. |
| `docs/FINETUNE.md` | Commands to train, score, and load a LoRA intent adapter. |
| `models/intent-lora/` | Earlier LoRA adapter, trained on the baseline prompt (0.93 / 0.93). |
| `models/intent-lora-retrieval/` | LoRA adapter trained on the retrieval prompt (0.94 / 0.91). |
| `docs/FINAL_REPORT.md` | The technical report. Convert with `pandoc docs/FINAL_REPORT.md -f gfm --resource-path=docs -o FINAL_REPORT.docx`. |
| `docs/SLIDES.md`, `docs/SLIDES.pptx`, `docs/SCRIPT.md` | Presentation slides (Marp source and PPTX) and the per-member speaking script. |

## Results in short

| System | Qwen2.5-1.5B | Phi-4-mini 3.8B |
|---|---|---|
| E1 baseline, zero-shot | 0.41 | 0.53 |
| E2 best instruction prompt | 0.43 | 0.60 |
| E3 best decoding (greedy) | 0.43 | 0.49 |
| E4 JSON output, schema-valid rows | 0.50 | 0.57 |
| TF-IDF + Logistic Regression (classic baseline) | 0.79 | 0.79 |
| Retrieval few-shot (used in `main.py`) | 0.84 | 0.92 |
| LoRA, baseline prompt (intent only) | 0.93 | — |
| LoRA, retrieval prompt (intent only) | 0.94 | — |

Accuracy on the 100 development messages.

| Final system, `main.py` classifier run once on 100 held-out messages | Qwen2.5-1.5B | Phi-4-mini 3.8B |
|---|---|---|
| Accuracy | **0.77** | **0.87** |
| LoRA, baseline prompt, same 100 messages (intent only) | 0.93 | — |
| LoRA, retrieval prompt, same 100 messages (intent only) | 0.91 | — |

Read `docs/PROGRESS.md` for the reasons behind each number.

## Common problems

- **`uv: command not found`**: install uv with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` and open a new terminal.
- **The first run is slow**: the model downloads once. Later runs load it
  from the cache.
- **Out of memory**: close other applications or use the default 1.5B model.
  Do not set `MODEL` to a larger model on an 8 GB machine.
- **No internet**: the BANKING77 CSVs are in `data/`, so the scripts do not
  download them. The model needs internet for the first download only.
