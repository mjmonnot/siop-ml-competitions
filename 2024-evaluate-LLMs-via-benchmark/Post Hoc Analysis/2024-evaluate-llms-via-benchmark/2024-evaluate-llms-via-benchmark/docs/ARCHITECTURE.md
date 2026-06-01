# Architecture

The premise of this repo is *one shared harness for four very different I-O tasks*. This document explains why that's possible at all, where the unification breaks down, and what each shared component is doing.

## The harness, in one diagram

    ┌────────────────────────────────────────────────────────────┐
    │  run.py                                                    │
    │    ─ load labels                                           │
    │    ─ load task-specific train/test inputs                  │
    │    ─ for each row:                                         │
    │         pick few-shot examples (random or by similarity)   │
    │         adapter.build_messages(row, examples)              │
    │         harness.call(spec)  ── or call_consistent(...)     │
    │         adapter.parse(text, row)                           │
    │    ─ score predictions, write submission CSV               │
    └────────────────────────────────────────────────────────────┘
                       │             ▲
                       ▼             │
    ┌────────────────────────────┐ ┌────────────────────────────┐
    │ adapters.py                │ │ harness.py                 │
    │   EmpathyAdapter           │ │   OpenAI client            │
    │   FairnessAdapter          │ │   Retries + backoff        │
    │   ClarityAdapter           │ │   On-disk response cache   │
    │   InterviewAdapter         │ │   Self-consistency wrapper │
    └────────────────────────────┘ └────────────────────────────┘
                       ▲             ▲
                       │             │
    ┌────────────────────────────┐ ┌────────────────────────────┐
    │ examples.py                │ │ scoring.py                 │
    │   pick_random(...)         │ │   accuracy(...)            │
    │   pick_similar(...)        │ │   pearson_r(...)           │
    │                            │ │   avg_cosine_similarity(...)│
    │   (uses MiniLM embeddings) │ │   composite(...)           │
    └────────────────────────────┘ └────────────────────────────┘

Four files in `src/`, plus a CLI. The adapters are the only thing that knows about individual tasks. Everything else is shared.

## Why one harness works for four different tasks

The four 2024 tasks look very different on the surface — binary classification, regression, text generation, paired classification — but at the OpenAI chat-completion API level they're all "messages in, text out." The differences live in three places:

1. **What goes in the messages.** Empathy gets a feedback email; interview gets four Q/R pairs and a fifth question; clarity gets a personality item string; fairness gets two policy descriptions. The adapter `build_messages` method handles this.

2. **What comes back.** Empathy and fairness want a JSON object with a constrained field; clarity wants a number; interview wants free-form text. The adapter `response_format` method declares the structured-output schema (if any), and `parse` does post-processing.

3. **How to score.** Accuracy for the classification tasks, Pearson correlation for clarity, average cosine for interview. The scoring module has one function per metric.

That's *all* the variation. The retry logic, the response cache, the few-shot example selection, the self-consistency wrapper, the CLI plumbing, the submission file writer — none of it changes across tasks. Building the four winners' approaches separately, every team duplicated all of that across their four notebooks. The argument for sharing it is just: don't.

## Where the unification breaks down

Three places.

### 1. Clarity probably shouldn't be in this harness at all

PAID Team's clarity win came from fine-tuning DeBERTa-V3-base, which is a different framework entirely (PyTorch + transformers, not the OpenAI API). The LLM-only clarity adapter in this repo is real and runnable, but capped at around r=.70 because no amount of prompt engineering produces calibrated continuous outputs from a chat model. The reasonable architecture for production would be: harness for empathy/interview/fairness; separate DeBERTa fine-tuning script for clarity; merge step combines the outputs.

The reason clarity stays in the harness here is pedagogical — the notebook walks through *why* it caps out where it does. See `notebooks/03_clarity.ipynb` and `KNOWN_LANDMINES.md` Landmine 1.

### 2. Interview's "few-shot examples" aren't few-shot

For empathy and fairness, the few-shot pool is the training set: K-of-N examples from the labeled training data. For interview, the "examples" are the candidate's own four prior responses — they're part of the input, not external context. The adapter handles this by ignoring the `examples` argument entirely and using the row's `Q1..Q5` / `R1..R4` fields. Cleaner alternatives (a separate `build_messages` signature for interview, or a "context-style" enum on the adapter) make the code more flexible but the comments harder to follow. Ignoring an argument is the cheaper move.

### 3. Self-consistency means different things for different tasks

For empathy and fairness (classification), self-consistency = N calls + mode. For clarity (regression), N calls + mean. For interview (generation), it's not obvious what to do — taking the textual mode is silly, taking the mean of embeddings then decoding back is intractable, and the right move (Akben's: generate N candidates, pick the one closest to the input) needs the cosine similarity machinery from scoring.py. The harness exposes a generic `reduce` callable, and each task's CLI invocation picks the right one. The harness deliberately doesn't try to be opinionated here.

## Component-by-component

### harness.py — the OpenAI client wrapper

The two non-obvious decisions:

**Caching by request hash.** Every CallSpec hashes to a 16-character key that combines model, messages, temperature, response_format, seed, and max_tokens. The cached response is just plain text under `.harness_cache/{key}.txt`. This means iterating on a single notebook cell doesn't re-bill the API for the same call, which matters when the few-shot context is large (empathy and fairness requests can be 4-8K tokens of input).

The cache is also what makes self-consistency cheap: if you re-run with `--self-consistency 5` then `--self-consistency 3`, you've already paid for 3 of the 5 calls.

**Retries cover transient errors only.** Rate limits and connection errors get exponential backoff and up to 4 retries. Bad model outputs (parse errors) are NOT retried at this level — that's the adapter's job to handle gracefully, and silently retrying would hide a real problem. The downside is that a particularly unlucky parse error becomes an immediate 5.4 default (clarity) or a "first" default (fairness). I'd rather see that in the output than have it papered over.

### examples.py — few-shot selection

Two strategies: `pick_random` and `pick_similar`. The 2024 winners universally used "all training examples" (a degenerate case of random with K=N). With 2026 hindsight, similarity-based selection helps on tasks where the training set is large enough to be picky — empathy (200 train rows, K=16) and fairness (25 train rows, K=24 effectively means "all").

Important caveat: similarity-based selection on clarity *hurts*. See KNOWN_LANDMINES.md Landmine 6. The CLI's `--similarity-examples` flag is opt-in per-task to make this hazard visible.

### adapters.py — the only place tasks differ

Four classes, each a TaskAdapter with the same three methods. The class structure is deliberately bare — no inheritance hierarchy, no plugin registry beyond a flat `ADAPTERS` dict. Two reasons:

1. The adapters genuinely don't share much code. Empathy/fairness share a *shape* (system prompt + few-shot turns + final user turn + JSON-schema output), but the system prompts and the response schemas are different enough that abstracting the shape would obscure rather than clarify.

2. Adding a fifth task should be a single new file or class, not "implement these eight abstract methods." See `ADAPTING_TO_NEW_TASKS.md`.

### scoring.py — three metrics, one composite

The metric functions match the competition's official scoring:
- accuracy (empathy, fairness)
- Pearson correlation (clarity)
- average cosine similarity using all-MiniLM-L6-v2 (interview)

The composite is exactly `0.25 * (empathy_acc + interview_cos + clarity_r + fairness_acc)`. The selftest in scoring.py validates this against the PAID Team's published composite of .666 — plug in their decoded per-task scores, the formula returns .666 to three decimal places.

## What you'd add for production

This is a competition harness, not a deployable system. Two things are missing that would matter in production:

1. **Calibration.** GPT-4 classifications are well-calibrated in their probability outputs, but we throw them away here in favor of the hard label. For a production fairness or empathy classifier, you'd want the log-probabilities (via logprobs=True in the OpenAI request) and a sigmoid-isotonic calibration step.

2. **Drift detection.** I-O text data drifts: HR vocabulary changes, organizational policy language shifts, new constructs emerge. None of the 2024 approaches addressed this; for a real system, you'd want a monitor on the few-shot pool's similarity to incoming queries, and an alarm when the gap widens.

Neither of these would help the leaderboard score, which is why none of the 2024 winners built them.

## What you'd remove for a real submission

If this were a competition entry rather than a teaching repo, three things would go:

1. The `--row-id` flag (it exists for iterating prompts; a final submission runs the full split)
2. The detailed parse-error fallback in every adapter (in a final run you'd want it to fail loudly so you can catch it)
3. The on-disk cache (a final submission is one-shot)

The harness as shipped is roughly 600 lines of source code. A pure-submission version would be closer to 350.
