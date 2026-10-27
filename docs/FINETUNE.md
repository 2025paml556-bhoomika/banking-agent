# Fine-tune the intent adapter

Train a LoRA adapter on `Qwen/Qwen2.5-1.5B-Instruct` so it names the
BANKING77 intent. The base weights stay frozen. Training updates four
attention layers: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

Run every command from the project root.

## Requirements

- The setup in the main `README.md` (`uv sync`)
- On Windows, `uv sync` installs the CUDA build of PyTorch
- An NVIDIA GPU. This run took about 2 hours on an RTX 3060 with 12 GB
- `data/train.csv` (already in the repository)

## Train

```bash
uv sync
uv run -m src.finetune
```

`src/finetune.py` reads `data/train.csv` (10,003 rows). It writes:

- `data/finetune_train.csv` — 9,503 training rows
- `data/finetune_val.csv` — 500 validation rows (`random_state=0`)

It then trains for up to 3 epochs and saves the adapter to
`models/intent-lora/`. The two files the program loads are:

- `models/intent-lora/adapter_config.json`
- `models/intent-lora/adapter_model.safetensors`

To name the base model explicitly:

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct uv run -m src.finetune
```

A different `MODEL` value needs its own training run. The adapter in
`models/intent-lora/` fits `Qwen/Qwen2.5-1.5B-Instruct` only.

## Score

```bash
uv run -m evaluation.finetune_eval
```

This loads the base model plus `models/intent-lora/` and scores the
intent name with greedy decoding. It writes:

- `evaluation/results/results_finetune_dev.csv`
- `evaluation/results/results_finetune_heldout.csv`
- `evaluation/results/finetune_scores.md`

The development file is `evaluation/eval_subset.csv` (100 rows).
The held-out file is `evaluation/heldout_subset.csv` (100 other rows).
Both come from `data/test.csv`. Training does not read them.

## Use the adapter

```bash
echo ADAPTER=models/intent-lora > .env
uv run main.py
```

Or open the browser page:

```bash
uv run ui.py
```

Open http://127.0.0.1:7860. `.env` is not committed. `src/llm.py` reads
`ADAPTER` from that file and attaches the adapter.
