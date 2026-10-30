# Demo script, 10 minutes, 4 speakers

Order: Shravan → Girish → Mohit → Bhoomika. Bhoomika ends with the live demo.
**SHOW** = put this file on the screen. **SAY** = speak these lines.

## Before the session (5 minutes before)

- Start `uv run ui.py` and send one test message. This loads the model so the demo is fast.
- Open these tabs in the editor, in this order: `data/train.csv`, `src/baseline.py`,
  `src/prompts.py`, `main.py`, `src/decoding.py`, `src/structured.py`, `src/retrieval.py`,
  `docs/FINAL_REPORT.md`.
- Open http://127.0.0.1:7860 in the browser.

---

## Shravan, 0:00 – 2:00. Problem, dataset, baseline

**SHOW: `data/train.csv`.** Scroll a little so people see real rows.

**SAY:**
"Good morning. Our project is a banking support assistant. A customer types one message in
their own words, like 'I lost my card, please block it'. The system must find which of 77
request types it is, run the right backend action, and write a reply.

The dataset is BANKING77. It has 13,083 real customer messages, each with a label. Look at
these rows. 'How do I track my card' and 'I need my card quickly' look the same, but they have
two different labels: card arrival and card delivery estimate. Many labels are this close.
This one fact causes most of our errors.

We test on two fixed sets of 100 messages. The development set is for comparing ideas. The
held-out set is kept untouched and is used only once, at the end, for the honest final number."

**SHOW: `src/baseline.py`.** Point at `TASK` (line 13) and `build_prompt` (line 17).

**SAY:**
"We started with the simplest possible prompt: the task line, the 77 label names, and the
message. Nothing else. The result was 0.41. Forty-one correct out of one hundred. And 57 of
the outputs had an extra sentence before the answer.

For comparison, a classic machine learning classifier, TF-IDF with logistic regression,
trained on all ten thousand rows, scores 0.79 on the same messages. That is the bar our
language model has to beat.

Most of the 59 errors were the near-twin labels you just saw. The model knows banking words.
It does not know where this dataset draws the line between two similar labels. That is missing
knowledge, not a missing instruction. Keep that thought — it explains the whole project.

Girish will now show you the model we used, and two other models we built ourselves."

---

## Girish, 2:00 – 4:00. Model choice, architecture, VAE and GAN

**SHOW: `docs/FINAL_REPORT.md`, Section 3,** the family comparison table.

**SAY:**
"Why a language model and not a VAE or a GAN? Our task is text in, text out, with no training.
A language model does this natively. A VAE or a GAN would need extra parts for text, and no
good pretrained one exists.

We did not just claim this. We built both and measured them. We trained a text VAE on the
banking messages, and a GAN on top of it. The GAN cannot learn on text directly, because text
is made of discrete tokens, so it learns inside the VAE's latent space instead."

**SHOW: `docs/FINAL_REPORT.md`, Section 4,** the five-stage table.

**SAY:**
"Our model is Qwen 2.5, 1.5 billion parameters. It runs on a normal laptop. Here is what one
call looks like. The message becomes 7 tokens. Each token becomes a vector of 1,536 numbers.
28 transformer layers read them. The output is a probability for every one of 151,936 possible
next tokens. The whole project is about one thing: moving that probability onto the first
token of the correct label."

**SHOW: `evaluation/results/samples_gan.csv`** — real synthetic messages.

**SAY:**
"And here is what our own generators produce. The messages look real, but only 57 percent of
the VAE messages and 63 percent of the GAN messages actually express the intent we asked for.
The language model classifies at 84 percent. So the choice was right.

But the small models are still useful at the edges. When we cut the training data to ten
percent, adding synthetic messages raised accuracy by 5 points. And the VAE gives us a cheap
guard: it catches 90 percent of off-topic messages, like cooking questions, while rejecting
only 2 percent of real banking messages.

The model is fixed. Everything else is steering. Mohit starts with the prompt."

---

## Mohit, 4:00 – 6:00. Prompts, programmatic prompting, tool calling

**SHOW: `src/prompts.py`,** `build_prompt` (lines 16–28).

**SAY:**
"No prompt in this project is written by hand. Every prompt is built by a function like this
one. One parameter switches the style. So one loop tests five prompt styles on the same 100
messages, and the comparison is fair.

The result surprised us. Five styles, and accuracy stayed between 0.40 and 0.43. Flat. But the
format changed completely: one constraint line, 'reply with the exact intent name and nothing
else', cut the extra text from 57 outputs to zero.

So here is lesson one of this project: instructions fix the format. They do not add knowledge."

**SHOW: `main.py`.** Point at `TOOLS` (line 106) and `handle` (line 173).

**SAY:**
"The assistant has two tools: classify the message, then run the banking action. We first let
the model decide when to call the tools. That is the usual agent pattern. It failed completely:
zero correct sequences out of eight tries. It invented a fake argument, BLOCK_CARD_LOST, and
once it said 'I have blocked your card' without calling any tool at all. In a bank, that is
the worst possible failure.

So in the final system, the code decides the tool order. The model only writes the reply.
Lesson two: with a small model, control belongs in the code.

One more check: we ran the whole pipeline again with a different model from a different
company, Phi-4-mini from Microsoft. One environment variable changed, nothing else. Every
lesson held, and the final score on the held-out set was 0.87 — above the classic classifier.

Instructions could not add knowledge. Bhoomika shows what did."

---

## Bhoomika, 6:00 – 10:00. Decoding, schema, RAG, LoRA, and the live demo

**SHOW: `src/decoding.py`,** the `SETTINGS` dict (line 11).

**SAY:**
"We tested five decoding settings: greedy, two temperatures, top-k, and top-p. Accuracy moved
between 0.38 and 0.44. Randomness cannot help a one-label answer — it can only move away from
the model's best guess. So: greedy for the label, sampling only for the reply text."

**SHOW: `src/structured.py`,** the JSON schema with the 77-name `enum`.

**SAY:**
"Next, we made the model answer in JSON and validated it against this schema. All 100 outputs
were valid JSON, but 14 were rejected: all of them invented label names, like get_virtual_card
instead of getting_virtual_card. They look real but they are not in the list. The schema turns
this hallucination into a safe, handled error."

**SHOW: `src/retrieval.py`,** `retrieve` and `build_prompt` (lines 23–30).

**SAY:**
"And here is the step that worked. Instead of five random examples, we give the model the five
training messages most similar to the customer's message. That is all this code does — a small
TF-IDF search, no training at all.

Accuracy jumped from 0.43 to 0.84. The trained classic classifier scores 0.79. We passed it
with no training. The correct label was inside the five examples 94 times out of 100. And the
model is not just copying its examples: against a plain majority vote on the same five
examples, it wins 12 rows and loses 5.

On the untouched held-out set, run once, the score is 0.77. That is the honest number.

Finally, we fine-tuned the same model with LoRA — a small adapter, 0.14 percent of the
weights, on one desktop GPU. With the retrieval prompt it reaches 0.94 on development and 0.91
on held-out. The label knowledge now lives in the weights.

Now let me show you the real system."

**SHOW: the browser (`ui.py`).** Point at the tool-call panel on the right after every message.

**Demo steps and what to SAY:**

1. Type `How long does a transfer take?`
   "You see both tool calls on the right. The reply uses only the facts from the tool result.
   The model cannot invent a fee or a delay."
2. Type `I lost my card, please block it`
   "The assistant does not block the card yet. Our classifier is right about 8 times in 10,
   and a wrong block cannot be undone. So the code asks first."
   Type `yes`.
   "Now it acts, and the reply carries a reference number from the backend."
3. Type `My card payment was declined`
   "Disputes need a person. 26 of the 77 intents route to a human agent by design. When we are
   not sure, we fail towards a human."

**Closing SAY:**
"Every number you heard today is a file in our repository, and one notebook reproduces all of
them in about 21 minutes. Thank you. We are happy to take questions."

---

## If a question comes

- **Why did the score drop from 0.84 to 0.77 on held-out?** The development set was used to
  choose the approach, so its score is a little optimistic. 0.77 is the honest number.
  (Bhoomika or Shravan)
- **Why not let the model pick the tools?** We tried. Zero correct out of eight, and it claimed
  actions it never did. (Mohit)
- **Why not use the VAE or GAN text directly?** Only 57 to 63 percent of it expresses the right
  intent. The LLM is at 84. The small models are support, not a replacement. (Girish)
- **Is 0.87 versus 0.77 a real difference?** On 100 rows the error margin is about ±0.08, so it
  is at the edge. But the direction is the same on both test sets. (Mohit)

## If time runs short

Cut in this order: demo message 3, the GAN numbers, the Phi-4 detail. Never cut demo message 2
(the confirmation) or the retrieval numbers. Those two carry the grade.
