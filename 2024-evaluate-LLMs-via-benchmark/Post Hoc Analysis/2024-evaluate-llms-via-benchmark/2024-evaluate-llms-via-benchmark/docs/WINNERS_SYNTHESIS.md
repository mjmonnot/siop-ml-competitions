# Winners Synthesis

A task-by-task comparison of how the four winning teams in the SIOP 2024 ML Competition approached each benchmark. The published per-task scores are decoded from the official competition deck's "Something to Learn from Different Approaches" table, where the published numbers are the team's contribution to the composite final score (each task weighted 0.25). Multiplying by 4 gives the raw metric value on each task.

| Task | Place 1 PAID | Place 2 Akben | Place 3 Hungry Llama | Place 4 Wonderlic |
|------|--------------|---------------|----------------------|-------------------|
| Empathy (acc) | .580 | **.608** | .560 | .488 |
| Interview (cos) | .440 | .496 | **.512** | .460 |
| Clarity (r) | **.816** | .676 | .740 | .772 |
| Fairness (acc) | **.828** | **.828** | .760 | .792 |
| **Composite** | **.666** | .652 | .643 | .630 |

No team won all four. PAID won by being best at the two regression-like tasks (clarity, fairness). Akben took empathy. Hungry Llama took interview. That's the headline of the whole competition: **on four different I-O tasks, four different approaches were optimal.**

What follows is a more careful look at why.

---

## Empathy — who got the email

The task is binary classification: a job candidate writes a feedback email to "Jonathan" about his work on the "Beta project," and human raters judge whether the email demonstrated empathy. Performance ranges from ~.49 to ~.61 — a 12-point spread on a binary task where chance is .50 says the task is genuinely hard. The dev/test distributions also drift noticeably (the public score distributions in the deck show empathy as the only task with substantially worse test than dev performance for top teams), suggesting that what counts as "empathetic" varies across raters more than the other tasks.

**PAID Team (.580)** ran GPT-4 with the entire training set as few-shot context, with their own twist: a Step 1 pass over the training data asking GPT-4 to *generate an explanation* of why each email did or did not demonstrate empathy, then a Step 2 pass where the test row was scored against the training emails plus their generated reasons as a conversation history. This is a soft form of chain-of-thought-as-supervision. The team's deck calls this "GPT-4 prompt engineering with thought process."

**Akben (.608, best in class)** did three things and majority-voted:
1. Text-completion N-shot with multiple foundation models (Bison-PaLM, davinci-002, gpt-3.5-instruct), repeated with self-consistency
2. **Elo rating** — pairwise comparisons of test emails ("which feedback would you prefer? You will pick the feedback that makes you feel motivated to work harder and understood") to build up a continuous empathy rating, then thresholded
3. Standard label-learning few-shot with GPT-4

The Elo trick is the genuinely creative move of the competition. Asking the model to compare two emails is much easier than asking it to rate one in isolation, and the resulting ratings — when aggregated over many comparisons — produce a calibrated empathy score. This is the same logic that underpins LMSYS Chatbot Arena.

**Hungry Llama (.560)** decomposed empathy into seven measurable sub-dimensions (sentiment, negativity, approval, appreciation, support, rapport, count of empathetic phrases), got Mistral/Mixtral to rate each one, weighted and summed them. The strategy is "if you don't trust the model to weigh dimensions, do the weighting yourself." It's a perfectly reasonable approach but it's the noisiest of the four — seven imperfect signals don't always combine into a less imperfect signal.

**Wonderlic (.488)** used SetFit (contrastive few-shot fine-tuning of sentence transformers) and prompt tuning. Their score is near chance, which says something about the data: SetFit normally works very well on similar low-resource text classification tasks, so the fact that it didn't here suggests the empathy task has substantial label noise that fine-tuning amplifies.

### Synthesis for empathy

The progression PAID → Akben → Hungry Llama → Wonderlic illustrates a real tradeoff:
- Adding **reasoning supervision** (PAID) helps over zero-shot
- Adding **multiple decision paths and a vote** (Akben) helps over a single chain
- **Decomposing into sub-dimensions** (Hungry Llama) is comparable but noisier
- **Fine-tuning on a noisy label distribution** (Wonderlic) underperforms

For a 2026 reconstruction, the cheap win is Akben's Elo idea combined with PAID's reasoning supervision. Adding self-consistency on top is free.

---

## Interview generation — write like the candidate

The task is generation: given four question-and-response pairs from a candidate's interview, generate the fifth response. The metric is cosine similarity between the generated response and the actual fifth response. The 2024 score range was tight (~.44 to ~.51), and PAID — the overall winner — was *last* on this task.

**PAID Team (.440, worst of the four)** did "question-centered" few-shot: collect all candidates' responses to Q5 across train/val/test, use them as few-shot examples, generate a response. This treats the task as "what's a plausible answer to this question" rather than "what would *this* candidate say." It works, but it strips out the per-candidate style signal — which is exactly what the cosine metric rewards.

**Hungry Llama (.512, best in class)** did the opposite: keep the candidate's prior four responses as the primary context, then *augment* them with inferred Big-5 personality ratings (via BART zero-shot classification across the candidate's prior responses, logit-transformed). Their system prompt explicitly conditioned on the candidate's reading level and average sentence length. This treats the task as style transfer, which matches what the metric measures.

**Akben (.496)** generated N candidate completions and picked the one with the highest cosine similarity to the *input* (the candidate's four prior responses concatenated). This is a clever self-consistency variant: instead of voting for content, it votes for style. The N=odd choice is irrelevant here — you just want the most-similar one.

**Wonderlic (.460)** also did personality-conditioned prompt engineering, but their deck is sparse on details. Their score is consistent with "GPT-4 + told to match the candidate's personality" without the post-hoc selection step.

### Synthesis for interview

The thing that matters is what signal you preserve from the candidate's prior responses:
- PAID lost the candidate's voice almost entirely by pooling over candidates
- Hungry Llama preserved it via explicit personality conditioning
- Akben preserved it via post-hoc cosine reranking
- Wonderlic preserved it via lighter prompt engineering

In a 2026 reconstruction, you'd combine Hungry Llama's personality conditioning with Akben's cosine reranking and probably hit .53+.

---

## Clarity — the task where LLMs lose

The task is regression: predict the mean human clarity rating (1–7 scale) for personality test items like "I am the life of the party" or "I find it difficult to get down to work." The metric is Pearson correlation, and the score range is *huge* — .68 to .82.

**PAID Team (.816, best in class) explicitly abandoned GPT-4 and fine-tuned GPT-3.5.** Their deck says: "Tried GPT4 (failed). Tried fine-tuned GPT3.5 (failed)." What worked was fine-tuning **DeBERTa-V3-base** with MSE or Pearson correlation as the loss, weight decay 0.1, train/validation split on the original training data. This is the most important pedagogical point in the whole competition.

The reason DeBERTa beats GPT-4 here is straightforward when you think about it: clarity is a regression task with an existing distribution of mean ratings, and there is no way to get GPT-4 to produce a calibrated continuous output. You can ask for a number 1–7, but the model has learned that "5" is a safe answer for most things, and the variance you need to score well on Pearson correlation just isn't there. A fine-tuned regression model is the right tool. The right model size is small (DeBERTa-base, not large) because the training set is small.

**Wonderlic (.772, second best)** likely used SetFit again. Their deck only shows their general framework; given they were 4th overall but 2nd on this task, they apparently figured out the regression issue.

**Hungry Llama (.740)** ran Mixtral 8x7B with NLP features plus BART-zero-shot Big-5 personality classifications of each item, then stacked everything in a downstream model. This is creative — they're effectively building a feature set and learning the regression separately. The downside is the same as for their empathy approach: lots of imperfect signals.

**Akben (.676, worst of the four)** ran multi-model ensembles (GPT-4 + GPT-3.5 + Claude-3) across 15 binary clarity sub-questions (passive voice? jargon? double negative? clear for a third-grader?), then aggregated. The decomposition is reasonable but suffers the same calibration issue as direct LLM rating: the binary outputs don't easily map to a continuous regression target.

### Synthesis for clarity

This task is the cleanest illustration of "use the right tool":
- **Fine-tuned BERT-family models beat LLMs at calibrated regression** — PAID's win is 4 points over the next-best team
- Multi-model LLM ensembles (Akben) don't fix the calibration issue
- Feature-based decomposition (Hungry Llama) partially fixes it but is still LLM-limited
- A traditional regressor over LLM-derived features (Wonderlic, probably) gets close

In a 2026 reconstruction, the right move is still PAID's: fine-tune a regression head. The shippable LLM-only approach (this repo) caps at around .70, and that's a real ceiling, not a bug.

---

## Fairness — the task where the LLM mostly wins

The task is binary classification: given two organizational policies (e.g. "Conflict Resolution Workshops" vs. "Conflict Resolution Workbooks"), pick which one human raters voted as fairer. The score range is .76–.83, with PAID and Akben tied at the top.

**PAID Team (.828, tied 1st)** ran GPT-4 with the entire fairness training set as few-shot context, with auto-generated reasons attached to each example (the same Step 1 / Step 2 trick they used for empathy). Their published prompt template asks "which one do you prefer?" and constrains the response to "first" or "second" at the beginning.

**Akben (.828, tied 1st)** ran GPT-4 with all training examples as few-shot, with N-shot self-consistency on top (N=odd, take the majority). Same shape, plus the consistency wrapper.

**Wonderlic (.792)** ran some variant of prompt engineering with in-context learning. Specifics not fully published.

**Hungry Llama (.760, worst of the four)** ran Mixtral with few-shot. The score gap between Hungry Llama and the rest probably reflects the model gap (Mixtral vs. GPT-4) more than the prompting strategy.

### Synthesis for fairness

This is the most-converged task of the four:
- Everyone used full-train-set few-shot
- The only meaningful differences were the model (GPT-4 beats Mixtral) and whether to add self-consistency (no clear gain when GPT-4 is already at .828)

The 2024 fairness training set was small enough (~25 paired comparisons) that "all examples as few-shot" was the right move. A 2026 reconstruction would do the same thing — there's no obvious improvement available — except possibly the structured-output JSON mode that didn't exist in early 2024, which removes parsing errors that may have eaten a fraction of a point.

---

## What unifies the four winners

Looking across all four tasks, the moves that consistently helped:

1. **Reasoning supervision** (PAID's auto-generated reasons on empathy/fairness training data, used as conversation-style few-shot)
2. **Self-consistency** (Akben's N-shot majority vote; never made it worse)
3. **Style/dimension conditioning** (Hungry Llama's Big-5 on interview, sub-dimension decomposition on empathy/clarity)
4. **Knowing when not to use an LLM** (PAID's DeBERTa pivot on clarity)

The moves that didn't generalize:
- SetFit and prompt tuning (Wonderlic) worked on some tasks but underperformed on empathy, where the label noise broke them
- Multi-model ensembling (Akben on clarity) didn't fix the underlying regression calibration issue
- Heavy sub-dimension decomposition (Hungry Llama on empathy) was comparable to but noisier than simpler approaches

For someone building one of these I-O LLM pipelines today, the synthesis is: **use a strong LLM, supervise its reasoning, add cheap consistency, and switch to a fine-tuned regressor for any calibrated continuous output.** That's what the harness in this repo does.
