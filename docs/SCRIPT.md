# Presentation script

Slides: `SLIDES.md` / `SLIDES.pptx`. Target: 9 minutes, 4 speakers. Times are per slide.

| Member | Slides | Time |
|---|---|---|
| 1 | 1, 2, 3: title, problem and dataset, model and architecture | 2:15 |
| 2 | 4, 5: baseline and prompts, decoding | 2:00 |
| 3 | 6, 7: structured output, adaptation decision | 2:15 |
| 4 | 8, 9, 10, 11: final system, evaluation, responsible use, close | 2:40 |

---

## Member 1

### Slide 1, Title (0:20)

Good morning. We are group _[number]_. Our project is a banking intent assistant built on the BANKING77 dataset.

A customer writes one message in free text. The assistant finds the intent, calls the right tool, and replies. We built it with one small language model, with no training. Everything we show is in the GitHub repository on the slide.

### Slide 2, Problem and dataset (1:00)

The user is a customer of an online bank, writing in a support chat. A message like "I lost my card, please block it". The bank must route that message to one of 77 intents, trigger the matching action, and answer the customer.

Today this is done with menus, keyword rules, or a human. Menus frustrate, rules break on paraphrase and typos, humans are slow.

The dataset is BANKING77: 13,083 messages, 77 labels, with a fixed train and test split. We used the train split, about 10,000 rows, only as a source of examples. We never trained on it.

For evaluation we took two fixed samples of 100 rows from the test split. The development set: every experiment runs on it, and it chose the final system. And a held-out set, 100 different rows, that no experiment touched. The final system runs on it once, at the end. We will come back to that number.

Why a language model and not a classifier? Three reasons. Free text with paraphrase and typos comes in. A fluent reply that is grounded in the tool result goes out. And one model does all three jobs: intent, tool call, and reply.

### Slide 3, Model and architecture (0:55)

The model is Qwen 2.5, 1.5 billion parameters, instruction tuned. It runs on a laptop with 8 gigabytes of memory through Hugging Face transformers.

We chose an autoregressive language model over a VAE or a GAN because the task is text in, discrete tokens out. Everything we want to condition on, instructions, examples, retrieved context, tool results, is text, so it all goes into one prefix. And we can steer it at run time without training.

The notebook shows the four stages on a real message, "How do I track my card?". Tokenization gives 7 tokens. The embedding tensor has shape 1 by 7 by 1536. The LM head gives the top five next-token probabilities. Then greedy generation produces the answer.

Those run-time levers are the structure of the project. Four levers, four experiments: the prompt, the decoding settings, the output schema, and tool calling. Member 2 starts with the prompt.

---

## Member 2

### Slide 4, E1 Baseline and E2 Prompts (1:10)

The baseline is the simplest prompt: the task line, the 77 intent names, and the message. Greedy decoding. Accuracy was 0.41.

For comparison, a classical TF-IDF plus logistic regression classifier, trained on the full train file, scores 0.79 on the same 100 rows. That is the bar the LLM must beat.

In experiment 2 we built a parameterised prompt function and tested five styles. A system role. A constraint line: "reply with the exact intent name and nothing else". Five random solved examples. And examples plus constraint.

Look at the two columns on the right. The constraint line removed all preamble text, 57 outputs down to 0, and cut the invented names. But accuracy did not move: 0.40 to 0.43 across all five.

The lesson: instructions fix format, not knowledge. The failures were near-synonym labels. "Track my card" went to card delivery estimate instead of card arrival. The model has only the two names. Nothing tells it where the dataset draws the line between them. No instruction can supply that.

### Slide 5, E3 Decoding (0:50)

Experiment 3 keeps the best prompt and changes the decoding: greedy, temperature 0.7, temperature 1.5, top-k 50, top-p 0.9. The sampling runs are seeded, so they are repeatable.

No setting beat greedy in a meaningful way, 0.38 to 0.44. The last column shows how many answers each setting changed against greedy, 5 to 13 rows. The flips went both ways, and at the two strongest settings more went right to wrong than wrong to right.

The reason is simple. A one-label answer has one correct token sequence. Randomness can only move away from the model's best guess.

So the decision: greedy for the intent, and sampling only for the free-text reply, where variety is welcome. Member 3 takes the output schema.

---

## Member 3

### Slide 6, E4 Structured output (0:55)

In experiment 4 the model answers in JSON: an intent and a confidence. Every output is validated with jsonschema. The key part of the schema is on the slide: the intent field is an enum of the 77 names.

All 100 outputs parsed as JSON. 86 passed the schema. All 14 rejections were invented intent names. "get virtual card" instead of "getting virtual card". "top up by card" instead of "topping up by card". The model builds a name that looks right instead of selecting a real one. This is hallucination in a classification setting.

The enum turns "is this a real intent?" into a hard check. Accuracy on the 86 valid rows was 0.50, against 0.43 overall. The confidence field was not useful: the model copied the 0.9 from the example for almost every row.

### Slide 7, Adaptation decision (1:20)

This is the key decision of the project. After experiments 2 and 3 we were at 0.43, and the errors were missing knowledge. So the fix had to add knowledge. Three options: more prompting, retrieval, or fine-tuning.

Prompting was already shown flat. Fine-tuning needs a GPU we did not have, and it gives up the "no training" property. Retrieval needs only a TF-IDF index over the train file, built in seconds. So we tested retrieval. For each message, the five most similar train rows become the few-shot examples.

The table. Random examples: 0.43. The retrieved examples alone, a 5-nearest-neighbour vote with no LLM: 0.77. The classical classifier: 0.79. The LLM reading the five retrieved examples: 0.84.

Two more facts. The correct label was among the five retrieved rows in 94 of 100 cases. And against the vote, the LLM was right where the vote was wrong 12 times, and wrong where the vote was right 5 times. So the examples give the floor, and the model adds on top.

The classical bar is passed with no training, on a laptop. The adaptation decision is retrieval. Member 4 shows the final system.

---

## Member 4

### Slide 8, Final system (0:50)

`main.py` puts the conclusions together. One function, `handle`, takes a message and runs four steps.

One: `classify_intent`. Retrieve five examples, build the prompt, greedy decoding, parse, and validate against the 77-name enum. Two: `handle_intent`. A simulated backend that returns "done" with a reference number for 17 action intents, "information" for 34 question intents, or "unsupported" for 26 intents that need a human. Three: the reply, sampled, written from the tool transcript only. Four: if there is no valid intent, the assistant asks the customer to rephrase and calls no tool.

One design point. The code fixes the tool order, not the model. We tested letting the model drive the tool loop: 4 messages, two setups, 0 of 8 correct sequences. It skipped tools, invented an argument, "BLOCK CARD LOST", and claimed it had blocked a card when it had not. With a small model, control belongs in code.

### Slide 9, Evaluation and failure analysis (1:00)

Now the number we promised. On the held-out set, 100 rows no decision was made on, the final system with Qwen scores 0.77. The 0.84 was a development score. That is level with the classical classifier at 0.79, not above it.

So we asked: do our conclusions depend on this one model? We ran the whole pipeline again, same code, same prompts, same seeds, with one environment variable changed: Phi-4-mini from Microsoft, 3.8 billion parameters, on a Colab GPU. The second column. Every lever behaved the same way. Retrieval was again the step that mattered, plus 0.41 for Qwen and plus 0.32 for Phi. Sampling never beat greedy on either model. The schema caught the same invented names. And the larger model lifted the whole curve: 0.92 on development, 0.87 on held-out, which is above the classical reference.

The honest claim, then: the method holds across two models from two vendors. With the small model it matches a trained classifier with no training; with the larger one it beats it. Both also write the reply and drive the tool call.

The failure analysis has seven types and eleven worked cases in the report. The main ones are on the slide. Near-synonym labels, which retrieval mostly fixed; 10 held-out rows are still wrong for both models, and they are all pairs like card arrival against card delivery estimate. Invented labels, which the schema catches. Retrieval that misleads, when the nearest rows share words but not meaning; that needs a retriever that matches meaning, and is future work. And random decoding, fixed by greedy.

### Slide 10, Responsible use and limits (0:35)

This assistant can trigger actions on a bank account from free text. Three principles guided the controls. Put control in code and leave language to the model: the model never chooses which tool runs. Validate at the boundary: every value that crosses from model text to code passes a schema. Fail towards a human: when unsure, say so and point to an agent.

Limits: small models, single turn, a simulated backend, and word-level retrieval.

Everything is reproducible. The notebook runs every experiment in about 21 minutes and reproduced every number in the report. The Phi run used the same notebook with one variable changed.

### Slide 11, Close (0:15)

The contribution table is on the slide. Thank you. We are happy to take questions.

---

## Handover lines

- 1 → 2: "Four levers, four experiments. Member 2 starts with the prompt."
- 2 → 3: "Greedy for the intent, sampling for the reply. Member 3 takes the output schema."
- 3 → 4: "The adaptation decision is retrieval. Member 4 shows the final system."

## If asked

- **Why not fine-tune?** No GPU for a 1.5B model on 8 GB; and the goal was to beat the classical classifier, which retrieval did with no training. It stays as future work if retrieval reaches a ceiling.
- **Why did held-out drop from 0.84 to 0.77?** The development set also chose the approach, so 0.84 is optimistic by construction. 0.77 is the honest number. Both are 100 rows, so one row is one point.
- **Is the Phi run the "three model families" comparison?** No. That comparison is autoregressive LM against VAE and GAN, in report Section 3. The Phi run is a second pretrained LLM from another vendor, used to check that the method conclusions do not depend on one model.
- **Why did few-shot hurt Phi?** Five random examples took Phi from 0.60 to 0.49; for Qwen they were neutral. Unrelated examples pull a stronger model towards their labels. The statement that holds for both: relevant examples help, random ones do not.
- **Is 0.87 against 0.77 significant?** On 100 rows the 95% interval is about ±0.08, so the gap is at the edge. But it is in the same direction on both subsets, and 13 held-out rows are right for Phi only against 3 for Qwen only.
- **Is the confidence field useful?** No. The model copied the example value. Retrieval similarity would be a better signal for a human fallback.
