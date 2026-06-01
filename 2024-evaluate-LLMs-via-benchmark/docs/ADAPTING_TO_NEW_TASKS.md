# Adapting to New Tasks

How to add a fifth I-O task to the harness. The intent is that you write one new file (the adapter) and a few lines elsewhere; everything else stays the same.

## The contract

A new task needs three things:

1. **A label file row** with `benchmark="your_task_name"` in `data/dev.csv` and `data/test.csv`. (For an actual new SIOP competition task, the organizers will provide this.)

2. **Input files** at `data/{your_task_name}_train.csv`, `data/{your_task_name}_dev_inputs.csv`, and `data/{your_task_name}_test_inputs.csv`. The schema (which columns are present) is yours to decide; it just has to be loadable by `csv.DictReader`.

3. **A TaskAdapter subclass** in `src/adapters.py` that knows how to build messages and parse responses for your task.

Plus optionally: a new metric in `src/scoring.py` if the existing four don't fit, and a notebook in `notebooks/` walking through the approach.

## Writing the adapter

Look at the existing four adapters as templates. The class needs three methods:

```python
@dataclass
class YourTaskAdapter(TaskAdapter):
    task_name: str = "your_task_name"
    k_examples: int = 16   # how many few-shot examples to use

    def build_messages(self, row: dict, examples: list[dict]) -> list[dict]:
        """Construct the chat-completion messages for one test row."""
        system = "You are an expert at [task]. ..."
        messages = [{"role": "system", "content": system}]
        for ex in examples:
            messages.append({"role": "user", "content": f"Input: {ex['input_text']}"})
            messages.append({"role": "assistant", "content": json.dumps({"label": ex["label"]})})
        messages.append({"role": "user", "content": f"Input: {row['input_text']}"})
        return messages

    def parse(self, text: str, row: dict) -> Any:
        """Take the model's text reply, return the typed prediction."""
        try:
            return json.loads(text)["label"]
        except (json.JSONDecodeError, KeyError):
            return DEFAULT_VALUE  # see Landmine 7

    def response_format(self) -> dict | None:
        """Optional: a JSON-schema structured-output spec."""
        return {"type": "json_schema", "json_schema": { ... }}
```

Then register it:

```python
ADAPTERS = {
    "empathy": EmpathyAdapter,
    ...
    "your_task_name": YourTaskAdapter,
}
```

That's it. The CLI, the harness, the example picker, the caching, and the self-consistency wrapper all work without further changes.

## Adding a new metric

If your task needs a metric the existing four don't cover (e.g., F1 instead of accuracy, RMSE instead of Pearson r), add it to `src/scoring.py`. The pattern is:

```python
def your_metric(predictions, truths) -> float:
    if len(predictions) != len(truths):
        raise ValueError(...)
    # compute and return
```

Then add a case in `run.py`'s scoring section to dispatch your task to your metric.

## What you should NOT change

If you find yourself wanting to modify these things, stop and think about whether your task really fits the harness:

- **The shared OpenAI client.** If you need a different model provider (Anthropic, Google), add a parallel harness for it; don't extend this one with conditional branches.

- **The cache key.** The cache is keyed on `model, messages, temperature, response_format, seed, max_tokens`. If your task needs cache invalidation on some other parameter, you've probably found a parameter the harness should expose globally; raise it instead of working around the cache.

- **The CLI structure.** `--task X --split Y --output Z` is the pattern. New flags for new tasks belong on the adapter, not on the CLI.

## Anti-patterns

Three things that look reasonable but will hurt you:

1. **Sharing prompts across adapters.** Each task has its own framing; the system prompts are short and writing them per-task is cheaper than the abstraction tax of a shared template.

2. **Building a "preprocessing" stage that touches all adapters.** If the same preprocessing applies to all tasks (e.g., stripping whitespace from inputs), put it in `load_csv` in `run.py`. If it's task-specific, it belongs in the adapter's `build_messages`. The middle ground — a shared preprocessing module — turns into a dumping ground.

3. **Optimizing the adapter for benchmark performance.** The adapter is supposed to be readable and obvious. If you find yourself adding clever caching, parallel async dispatch, or fancy parsing inside an adapter, that logic probably belongs in the harness.

## When the harness doesn't fit

There are real cases where a new task won't fit cleanly:

- **The task needs a non-LLM model in the loop.** Clarity is the canonical example: PAID's winning approach used fine-tuned DeBERTa, not an LLM. The harness has no good way to handle this except to route the entire task outside of it. A future version might add a "model adapter" layer that abstracts "thing that takes input, returns output" without committing to OpenAI; for now, do it as a separate script and merge the outputs at submission time.

- **The task needs multimodal inputs.** Images, audio. The harness assumes text-only chat completions. You'd need a parallel multimodal harness; OpenAI's vision endpoint has the same shape but different request structure.

- **The task is interactive / multi-turn.** The harness assumes one request, one response. A task that needs a follow-up question or a clarifying turn doesn't fit and shouldn't be shoehorned in.

If your task hits one of these, build a separate pipeline for it and use this repo's merge step (`python -m src.run --merge ...`) to combine the outputs at the end.
