# Banking Intent Agent

A customer support assistant for an online bank. A local Hugging Face LLM
(`Qwen/Qwen2.5-1.5B-Instruct`) reads a customer message, finds the intent
(one of the 77 BANKING77 intents), runs a tool for that intent, and writes
a short reply.

The project has two parts:

1. **Experiments** (E1 to E7) that compare prompts, decoding settings,
   structured output, and retrieval. Each experiment is one script and
   writes one or more `results_*.csv` files.
2. **The final assistant** (`main.py`) that uses the best settings from the
   experiments.

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
The assistant prints each tool call and then the reply. Press Ctrl-D to exit.

Run one test generation to check that the model loads:

```bash
uv run llm.py
```

## Run the experiments

Run the scripts in this order. Each one reads `eval_subset.csv`
(the same 100 test messages) and writes its results as CSV.
Times are for an M3 Mac with 8 GB RAM.

| Step | Command | Writes | Time |
|---|---|---|---|
| Test set | `uv run make_eval_subset.py` | `eval_subset.csv` (already in the repo) | seconds |
| E1 baseline | `uv run baseline.py` | `results_baseline.csv` | 2 min |
| E2 prompts | `uv run prompts.py` | `results_<style>.csv` (5 files) | 8 min |
| E3 decoding | `uv run decoding.py` | `results_decoding_<setting>.csv` (5 files) | 8 min |
| E4 structured output | `uv run structured.py` | `results_structured.csv` | 3 min |
| Adaptation (retrieval) | `uv run retrieval.py` | `results_retrieval.csv` | 2 min |

You do not have to rerun the experiments to read the results. All
`results_*.csv` files are in the repository.

To compare a larger model, set the `MODEL` variable before any command:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct uv run baseline.py
```

The 3B model needs about 6 GB RAM and does not fit on an 8 GB Mac.
Use Google Colab for it.

## Notebooks

| Notebook | What it does | Time |
|---|---|---|
| `banking_agent.ipynb` | Loads the saved CSV results, shows the tables, runs the live demo | 1 min |
| `banking_agent_full.ipynb` | Recomputes every experiment from scratch | 21 min |

Both notebooks import the code from the `.py` files. The first cell
installs the dependencies when it runs on Colab.

To open a notebook locally, open it in VS Code or Cursor and select the
`.venv` Python as the kernel. The `ipykernel` package is already installed
by `uv sync`.

## Files

| File | Purpose |
|---|---|
| `llm.py` | Loads the model. `generate(messages, **decoding)` is the one function all scripts use. |
| `baseline.py` | E1. Loads the 77 intent names, the simplest prompt, and `parse_intent`. |
| `prompts.py` | E2. Five prompt styles: zero-shot, role, constraint, few-shot, few-shot + constraint. |
| `decoding.py` | E3. Greedy, temperature 0.7, temperature 1.5, top-k 50, top-p 0.9. |
| `structured.py` | E4. JSON output validated with `jsonschema`. |
| `retrieval.py` | Adaptation. Five examples chosen by TF-IDF similarity. |
| `main.py` | E7. The final assistant with `read_intent` and `execute_intent` tools. |
| `make_eval_subset.py` | Creates `eval_subset.csv` from the BANKING77 test split. |
| `eval_subset.csv` | The 100 test messages used by every experiment. |
| `results_*.csv` | Output of each experiment. |
| `PROGRESS.md` | Step-by-step log: what we did, why, and what we found. |
| `FAILURES.md` | E6. Failure types and worked failure cases. |
| `GAPS.md` | Gap analysis of the first version against the assignment. |
| `REPORT.md` | The technical report. Convert with `pandoc REPORT.md -o REPORT.docx`. |

## Results in short

| System | Accuracy on 100 test messages |
|---|---|
| E1 baseline, zero-shot | 0.41 |
| E2 best prompt (few-shot + constraint) | 0.43 |
| E3 best decoding (greedy) | 0.43 |
| E4 JSON output, schema-valid rows | 0.50 |
| TF-IDF + Logistic Regression (classic baseline) | 0.79 |
| **Retrieval few-shot (used in `main.py`)** | **0.84** |

Read `PROGRESS.md` for the reasons behind each number.

## Common problems

- **`uv: command not found`**: install uv with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` and open a new terminal.
- **The first run is slow**: the model downloads once. Later runs load it
  from the cache.
- **Out of memory**: close other applications or use the default 1.5B model.
  Do not set `MODEL` to a larger model on an 8 GB machine.
- **No internet**: `baseline.py` and `prompts.py` download the BANKING77
  train and test CSV from GitHub at import time. The model also needs
  internet for the first download.
