# Fine-tune the intent adapter

Train a LoRA adapter on `Qwen/Qwen2.5-1.5B-Instruct` so it names the
BANKING77 intent. The base weights stay frozen. Training updates four
attention layers: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

The current script trains on the retrieval prompt and saves
`models/intent-lora-retrieval/`. An earlier run trained on the baseline
prompt and saved `models/intent-lora/`. The current script does not
write that older folder.

Run every command from the project root.

## Requirements

- The setup in the main `README.md` (`uv sync`)
- On Windows, `uv sync` installs the CUDA build of PyTorch
- An NVIDIA GPU. The retrieval-prompt run took about 2 hours 53 minutes
  on an RTX 3060 with 12 GB. The earlier baseline-prompt run took about
  2 hours 11 minutes on the same GPU
- `data/train.csv` (already in the repository)

## Train

```bash
uv sync
uv run -m src.finetune
```

`src/finetune.py` reads `data/train.csv` (10,003 rows). It writes:

- `data/finetune_train.csv` — 9,503 training rows
- `data/finetune_val.csv` — 500 validation rows (`random_state=0`)

Each training prompt is the retrieval prompt. It has the task line, the
77 intent names, the reply rule, and the 5 most similar other training
rows (TF-IDF cosine). The current row is left out of those five.
Validation rows take their five examples from the 9,503 training rows
only. The assistant answer is the intent name, and the loss is on those
answer tokens only.

The script trains for up to 3 epochs and saves the adapter to
`models/intent-lora-retrieval/`. The two files the program loads are:

- `models/intent-lora-retrieval/adapter_config.json`
- `models/intent-lora-retrieval/adapter_model.safetensors`

The kept weights are epoch 2 (step 2376). Validation loss was 0.0709
after epoch 1, 0.0663 after epoch 2, and 0.0746 after epoch 3.

To name the base model explicitly:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct uv run -m src.finetune
```

A different `MODEL` value needs its own training run. The adapter in
`models/intent-lora-retrieval/` fits `Qwen/Qwen2.5-1.5B-Instruct` only.

## Score

```bash
uv run -m evaluation.finetune_eval
```

This loads the base model plus `models/intent-lora-retrieval/` and scores
the intent name with the retrieval prompt and greedy decoding. It writes:

- `evaluation/results/results_finetune_retrieval_dev.csv`
- `evaluation/results/results_finetune_retrieval_heldout.csv`
- `evaluation/results/finetune_retrieval_scores.md`

The development file is `evaluation/eval_subset.csv` (100 rows).
The held-out file is `evaluation/heldout_subset.csv` (100 other rows).
Both come from `data/test.csv`. Training does not read them.

The earlier baseline-prompt scores stay in
`results_finetune_dev.csv`, `results_finetune_heldout.csv`, and
`finetune_scores.md`.

## Use the adapter

```bash
echo ADAPTER=models/intent-lora-retrieval > .env
uv run main.py
```

Or open the browser page:

```bash
uv run ui.py
```

Open http://127.0.0.1:7860. `.env` is not committed. `src/llm.py` reads
`ADAPTER` from that file and attaches the adapter. `main.py` sends the
retrieval prompt, so this adapter is the one trained for that prompt.

Run `main.py` or `ui.py`. Each process loads the model. One model fits
on a 12 GB GPU.
