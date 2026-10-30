# Demo script, 10 minutes, 4 speakers

Order: Shravan → Girish → Mohit → Bhoomika. Bhoomika closes with the live demo.
Each block lists: what to say, what to show on screen, and which metrics to point at.

## Before the session (any member, 5 minutes before)

- Start `uv run ui.py` and send one test message, so the model is loaded and warm.
- Open these in editor tabs, in order: `data/train.csv`, `src/baseline.py`, `src/prompts.py`,
  `main.py`, `src/retrieval.py`, `docs/FINAL_REPORT.md`, `README.md`.
- Open http://127.0.0.1:7860 in the browser.

---

## Shravan, 0:00 – 2:00. Problem, dataset, baseline

**Say.** "A customer of an online bank writes one free-text message. The system must find which
of 77 intents it is, run the right backend action, and reply. The dataset is BANKING77: 13,083
labelled messages. Many intents are near-synonyms, and that one property causes most errors."

**Show 1: `data/train.csv`.** Scroll to show real rows. Point at `card_arrival` against
`card_delivery_estimate`: three similar messages, two labels.

**Say.** "We evaluate on two fixed subsets of 100 test rows. The development set compares
approaches. The held-out set is touched once, at the end, for the honest number."

**Show 2: `src/baseline.py`.** Point at `TASK` (line 13) and `build_prompt` (line 17): the
simplest prompt is the task line, the 77 names, the message.

**Metrics to state** (README "Results in short" table, or report Section 5):

- Baseline accuracy **0.41**; 57 of 100 outputs had a preamble sentence.
- The bar to beat: TF-IDF + logistic regression, trained on all 10,003 rows: **0.79**.
- 45 of the 59 errors are near-synonym or favourite-label confusions: missing knowledge,
  not a missing instruction.

**Report back-up:** Sections 1, 2, 5; Appendix B (what data each experiment used).

Handover: "The first idea was better instructions. Mohit shows what instructions can and
cannot fix — after Girish shows the model we steer."

---

## Girish, 2:00 – 4:00. Model choice, architecture, VAE and GAN

**Say.** "We need text in, text out, and run-time control without training. Only an
autoregressive language model gives all three. We did not take that on faith: we trained the
two other families ourselves and measured them."

**Show 1: report Section 3** (`docs/FINAL_REPORT.md`). Point at the family comparison table:
VAE and GAN need extra encoders, give weak discrete text, and have no pretrained checkpoints.

**Show 2: report Section 4.** The five stages on a real message: 7 tokens, embedding tensor
(1, 7, 1536), 28 decoder layers, the LM head over 151,936 tokens. The prompt's whole job is to
move probability onto the first token of the right intent name.

**Show 3: report Section 12, "VAE and GAN follow-up"** (or `evaluation/results/samples_gan.csv`
for real synthetic messages).

**Metrics to state** (`evaluation/results/vae_gan_results.json`, `gan_ablation.json`,
`guard_eval.json`):

- Synthetic text intent fidelity: VAE **56.8%**, GAN **63.2%**. Far below the LLM: the
  Section 3 choice, now measured.
- GAN ablation: latent Fréchet distance falls from **35.4** (500 steps) to **4.0** (8,000 steps).
- Augmentation: within noise at full data; **+0.05** accuracy in the low-resource setting.
- Out-of-scope guard: catches **90%** of off-topic messages at a **2%** false-reject rate.

**Report back-up:** Sections 3, 4, and the Section 12 follow-up.

Handover: "The model is fixed. Everything else is steering. Mohit starts with the prompt."

---

## Mohit, 4:00 – 6:00. Prompt engineering, programmatic prompting, tool calling

**Say.** "No prompt in this project is a hand-written string. Every prompt is built by a
function. One parameter switches the style, so one loop compares five styles fairly."

**Show 1: `src/prompts.py`**, `build_prompt` (lines 16–28): five styles from the same parts.

**Metrics to state** (report Section 6 table):

- Five prompt styles: accuracy **0.40 to 0.43**. Flat.
- But the constraint line cut preambles from **57 to 0**. Instructions fix format, not knowledge.

**Show 2: `main.py`.** Point at `TOOLS` (line 106), the `SYSTEM` prompt (line 133), and
`handle` (line 173): the code fixes the tool order, the model only writes the reply.

**Say.** "We tested letting the model drive the tools, the usual agent pattern: 0 of 8 correct
sequences. It invented the argument BLOCK_CARD_LOST and claimed it had blocked a card when it
had not. With a small model, control belongs in code."

**Metrics to state** (report Section 12, second-model table):

- Same pipeline, one environment variable changed, Phi-4-mini 3.8B: every lever behaved the
  same; held-out **0.87**, above the classical 0.79. The method is not tied to one model.

**Report back-up:** Sections 6, 10, 11; Appendix E (four real runs with tool transcripts).

Handover: "Instructions could not add knowledge. Bhoomika shows what did."

---

## Bhoomika, 6:00 – 10:00. Decoding, structured output, RAG, LoRA, and the live demo

**Say (6:00 – 7:45).** "Three more levers, then the one that worked."

**Show 1: report Section 12, the all-systems table.** Walk it top to bottom:

- Decoding (Section 7): greedy against four sampling settings, **0.38 to 0.44**. Randomness
  only moves away from the best guess. Decision: greedy for the label, sampling for the reply.
- Structured output (Section 9): 100 of 100 outputs parsed as JSON, **14 rejected by the
  schema** — all invented intent names like `get_virtual_card`. The enum turns hallucination
  into a handled error.
- Retrieval (Section 8): five examples chosen by TF-IDF similarity per message. **0.84** on
  development against 0.79 for the trained classical classifier, with no training. The correct
  label was among the five examples 94 times of 100.
- Held-out, run once: **0.77**. The honest number. Level with the classical bar.
- LoRA (Section 12): rank-8 adapter, 0.14% of the weights, RTX 3060. Retrieval prompt:
  **0.94 development, 0.91 held-out**. Baseline prompt: 0.93 both.

**Live demo (7:45 – 10:00).** Switch to the browser (`ui.py`, already running). Point at the
tool-call panel on the right after every message.

1. Type `How long does a transfer take?` — information intent, two tool calls, the reply uses
   only the tool result.
2. Type `I lost my card, please block it` — the assistant asks for confirmation first, because
   a wrong block is irreversible. Reply `yes` — the reference number appears.
3. Type `My card payment was declined` — unsupported intent, routed to a human agent by design.

**Close.** "Every number you saw is a CSV or JSON file in the repository, and the notebook
reproduces all of them in about 21 minutes. Thank you — questions welcome."

---

## If a question comes

- **Why did held-out drop from 0.84 to 0.77?** The development set chose the approach, so its
  score is optimistic by design. 0.77 is the honest number (Bhoomika or Shravan).
- **Why not let the model pick the tools?** 0 of 8 correct sequences; it claimed actions it
  did not do (Mohit).
- **Why not use the VAE/GAN text directly?** 57 to 63% intent fidelity against 0.84 from the
  LLM; it is support data, not a replacement (Girish).
- **Is 0.87 against 0.77 significant?** On 100 rows the 95% interval is about ±0.08, so it is
  at the edge, but the direction is the same on both subsets (Mohit).

## If time runs short

Cut in this order: demo message 3, the GAN ablation numbers, the Phi-4 detail. Never cut the
confirmation flow (demo message 2) or the all-systems table. Those two carry the grade.
