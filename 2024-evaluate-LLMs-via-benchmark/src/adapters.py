"""Task-specific adapters for the four 2024 SIOP benchmarks.

Each adapter is a small class with three methods:

  build_messages(row, examples) -> list[dict]
      Construct the OpenAI chat-completion messages for a single test
      row, given selected few-shot examples.

  parse(text, row) -> Any
      Take the model's raw text reply and return the typed prediction
      (int for empathy, str for fairness, float for clarity, str for
      interview).

  response_format() -> dict | None
      Optional structured-output schema. If returned, the harness
      passes it as `response_format`, which dramatically reduces parse
      errors. Empathy and fairness use this; clarity and interview do
      not (clarity because a single float is overkill for a schema,
      interview because we want a free-text completion).

The four adapters intentionally do not share much code. Each task has a
different shape (the empathy emails are written-to-Jonathan-Beta-project
templates; interview is question/response chains; clarity is short item
text; fairness is a pair of policy descriptions). Trying to unify the
prompt construction across all four would obscure rather than clarify,
which is the same reason the winners built four separate notebooks.

Where this departs from each winner's approach is noted in the
docstring of each adapter.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# -------------------------------------------------------------------------
# Base adapter
# -------------------------------------------------------------------------


@dataclass
class TaskAdapter(ABC):
    """Common shape. Subclasses override the three abstract methods."""

    task_name: str = ""

    @abstractmethod
    def build_messages(self, row: dict, examples: list[dict]) -> list[dict]:
        ...

    @abstractmethod
    def parse(self, text: str, row: dict) -> Any:
        ...

    def response_format(self) -> dict | None:
        return None


# -------------------------------------------------------------------------
# Empathy: binary classification of feedback emails
# -------------------------------------------------------------------------


@dataclass
class EmpathyAdapter(TaskAdapter):
    """Classify whether a feedback email demonstrates empathy.

    The 2024 setup: a job candidate writes feedback to a colleague named
    Jonathan about his work on the "Beta project." Human raters then
    judged whether the email showed empathy (1) or not (0).

    Approach in this repo: GPT-4o with similarity-selected few-shot.
    This differs from the 2024 winners as follows:

    - PAID Team used ALL training examples as few-shot, with their own
      auto-generated reasoning chains. We use a smaller K (default 16),
      picked by similarity rather than including everything. Empirically
      this is comparable or better when K is well-chosen.

    - Akben ensembled three approaches (N-shot, Elo pairwise, label
      learning) with majority vote. The Elo approach is the genuinely
      novel one — see notebooks/01_empathy.ipynb for the reconstruction.

    - Hungry Llama decomposed empathy into 7 sub-dimensions (sentiment,
      negativity, approval, appreciation, support, rapport, phrase
      count) and weighted them. That's a great strategy if you don't
      trust the model to weigh the dimensions itself; with GPT-4o in
      2026 you mostly do.
    """

    task_name: str = "empathy"
    k_examples: int = 16

    def build_messages(self, row: dict, examples: list[dict]) -> list[dict]:
        system = (
            "You are an expert rater of empathy in workplace feedback emails. "
            "You will be shown an email written to a colleague named Jonathan about "
            "his work on the 'Beta project'. Decide whether the email demonstrates "
            "empathy (1) or does not demonstrate empathy (0). Reply with strict JSON "
            'of the form {"label": 0} or {"label": 1}.'
        )
        # Build the few-shot block as a flat list of user/assistant turns,
        # mirroring the structure PAID Team used. This format lets the
        # model see the labels as completed reasoning rather than as raw
        # examples in a list — the difference shows up in calibration.
        messages: list[dict] = [{"role": "system", "content": system}]
        for ex in examples:
            messages.append({
                "role": "user",
                "content": f"Email:\n###\n{ex['text']}\n###",
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({"label": int(ex["empathy"])}),
            })
        messages.append({
            "role": "user",
            "content": f"Email:\n###\n{row['text']}\n###",
        })
        return messages

    def parse(self, text: str, row: dict) -> int:
        # Even with structured outputs, defensive parsing is cheap.
        try:
            obj = json.loads(text)
            label = obj.get("label")
            if label in (0, 1, "0", "1"):
                return int(label)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        # Fallback: look for a 0 or 1 anywhere in the text.
        m = re.search(r"\b([01])\b", text)
        if m:
            return int(m.group(1))
        # Last-resort default. If we got here, the response is malformed
        # enough that we should log it; downstream code can replace this
        # with a class prior if you'd rather not bias toward 0.
        return 0

    def response_format(self) -> dict | None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "empathy_label",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        }


# -------------------------------------------------------------------------
# Fairness: binary classification of policy preference
# -------------------------------------------------------------------------


@dataclass
class FairnessAdapter(TaskAdapter):
    """Identify which of two organizational policies received the
    majority vote as fairer.

    Approach: GPT-4o with all training examples as few-shot. This task
    has the smallest training set of the four (the "fairness_train.csv"
    in the original release was ~25 paired policy comparisons), so
    "all" is feasible and is what every winning team did. We default to
    K=24 (one less than the typical training set size to leave a
    held-out check for prompt iteration), but the practical effect is
    the same as PAID Team's approach.

    - PAID Team: GPT-4 with auto-generated reasons for each training
      example, used as conversation-style few-shot. Achieved .207
      (tied for 1st on this task).

    - Akben: same shape (GPT-4 + full training set as few-shot) but
      added N-shot self-consistency: N=odd-number runs at non-zero
      temperature, take the majority. Also achieved .207 (tied 1st).

    - Hungry Llama: Mixtral 8x7B with few-shot. Lower score (.190).

    The self-consistency wrapper is available as call_consistent in the
    harness; whether to use it for this task is a judgment call given
    the tied top score without it.
    """

    task_name: str = "fairness"
    k_examples: int = 24

    def build_messages(self, row: dict, examples: list[dict]) -> list[dict]:
        system = (
            "You are an expert rater of organizational policy fairness. "
            "You will be shown two policies. Decide which one received the "
            "majority vote as fairer from human raters. Reply with strict JSON "
            'of the form {"choice": "first"} or {"choice": "second"}.'
        )
        messages: list[dict] = [{"role": "system", "content": system}]
        for ex in examples:
            messages.append({
                "role": "user",
                "content": (
                    f"First policy:\n###\n{ex['first_option']}\n###\n"
                    f"Second policy:\n###\n{ex['second_option']}\n###"
                ),
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({"choice": ex["majority_vote"]}),
            })
        messages.append({
            "role": "user",
            "content": (
                f"First policy:\n###\n{row['first_option']}\n###\n"
                f"Second policy:\n###\n{row['second_option']}\n###"
            ),
        })
        return messages

    def parse(self, text: str, row: dict) -> str:
        try:
            obj = json.loads(text)
            c = obj.get("choice", "").strip().lower()
            if c in ("first", "second"):
                return c
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        # Fallback: look for the word in the text.
        low = text.lower()
        if "first" in low and "second" not in low:
            return "first"
        if "second" in low and "first" not in low:
            return "second"
        # Mentions both or neither — guess by majority class. With no
        # better signal, "first" is the majority class in the training
        # set (~57% of cases), so it's the better default.
        return "first"

    def response_format(self) -> dict | None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "fairness_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "choice": {"type": "string", "enum": ["first", "second"]},
                    },
                    "required": ["choice"],
                    "additionalProperties": False,
                },
            },
        }


# -------------------------------------------------------------------------
# Clarity: regression of average human clarity rating
# -------------------------------------------------------------------------


@dataclass
class ClarityAdapter(TaskAdapter):
    """Predict the mean human clarity rating (1-7) for a personality
    test item.

    This is the task where the strong move is NOT to use an LLM — PAID
    Team's clarity score of .816 came from a fine-tuned DeBERTa-v3-base,
    after they explicitly tried and failed with GPT-4 and fine-tuned
    GPT-3.5. The DeBERTa approach is reconstructed in notebooks/
    03_clarity.ipynb; the LLM-only adapter here is the fallback for
    when you don't want to set up a fine-tuning stack and are willing
    to leave correlation points on the table.

    Practical expectation: this adapter will land around r=.65-.75 on
    test, comparable to Wonderlic (.772) but below PAID (.816) and
    Hungry Llama (.740) — those teams added either model fine-tuning
    or multi-model ensembling, neither of which is in this minimal
    harness.
    """

    task_name: str = "clarity"
    k_examples: int = 24

    def build_messages(self, row: dict, examples: list[dict]) -> list[dict]:
        system = (
            "You are an expert rater of item clarity for personality tests. "
            "Respondents rated the clarity of each item on a 7-point scale, "
            "where 1 = extremely unclear and 7 = extremely clear. Predict the "
            "MEAN clarity rating across respondents for the item shown. The "
            "answer is a single number between 1.0 and 7.0; reply with just "
            "the number, to two decimal places."
        )
        messages: list[dict] = [{"role": "system", "content": system}]
        for ex in examples:
            messages.append({
                "role": "user",
                "content": f"Item: {ex['item']}",
            })
            messages.append({
                "role": "assistant",
                "content": f"{float(ex['mean_clarity']):.2f}",
            })
        messages.append({
            "role": "user",
            "content": f"Item: {row['item']}",
        })
        return messages

    def parse(self, text: str, row: dict) -> float:
        m = re.search(r"[-+]?\d*\.?\d+", text)
        if m:
            try:
                v = float(m.group(0))
                # Clip to the valid range.
                return max(1.0, min(7.0, v))
            except ValueError:
                pass
        # No number at all — default to the train-set grand mean, which
        # is around 5.4 from the publicly-released clarity training
        # data. Predicting the grand mean is the right default for
        # regression: it's the minimum-MSE constant prediction.
        return 5.4

    # No structured-output schema; we want a freeform number with
    # whatever decimal precision the model wants to give us. Adding a
    # schema here would only constrain to a JSON object wrapper.
    def response_format(self) -> dict | None:
        return None


# -------------------------------------------------------------------------
# Interview: text generation given prior responses
# -------------------------------------------------------------------------


@dataclass
class InterviewAdapter(TaskAdapter):
    """Generate the candidate's 5th interview response given the
    previous 4 question/response pairs.

    The metric is cosine similarity between the generated response and
    the actual response. Two design choices matter:

    1. Length: the 2024 winners universally capped output around 100-150
       words. Going longer doesn't help (the cosine metric is bounded)
       and going shorter hurts. We use 120, matching PAID Team.

    2. Style conditioning: Hungry Llama's win (.512 vs. PAID's .440)
       came from explicitly conditioning on Big-5 personality inferred
       from the prior responses. This adapter takes a lighter approach:
       a system-prompt instruction to match the style/tone/personality
       of the four prior responses. That gets you most of Hungry
       Llama's gain without needing a separate personality classifier.

    A different approach worth knowing about (Akben): generate N
    candidates, then pick the one with the highest cosine similarity to
    the *input* (the four prior responses concatenated). This is a
    self-consistency variant that picks for stylistic continuity. It's
    available via harness.call_consistent with a custom reducer; see
    notebooks/02_interview.ipynb for the implementation.
    """

    task_name: str = "interview"
    max_words: int = 120

    def build_messages(self, row: dict, examples: list[dict]) -> list[dict]:
        # No few-shot for interview generation: each test row is its own
        # context (the four prior Q/R pairs), and adding external
        # examples would dilute the style signal. This matches what
        # every winning team did.
        del examples  # unused; we keep it for adapter signature uniformity

        system = (
            "You are a job candidate completing an interview. You will be "
            "shown your four previous question-and-response pairs. Generate "
            f"your response to the fifth question in no more than {self.max_words} "
            "words. Match the style, tone, vocabulary, and personality reflected "
            "in your previous responses — including any small grammatical quirks "
            "or word choices. Do not introduce experiences inconsistent with what "
            "you have already described."
        )

        # Build the Q/R chain. The adapter expects the row to have
        # Q1..Q5 and R1..R4 fields (the test row is missing R5; that's
        # what we're generating).
        user_parts = []
        for i in range(1, 5):
            user_parts.append(f"Question: {row[f'Q{i}']}\nResponse: {row[f'R{i}']}")
        user_parts.append(f"Question: {row['Q5']}")
        user = "\n\n".join(user_parts)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse(self, text: str, row: dict) -> str:
        # The model occasionally prefixes "Response:" or wraps the
        # output in quotes. Strip both.
        cleaned = text.strip()
        if cleaned.lower().startswith("response:"):
            cleaned = cleaned[len("response:"):].strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def response_format(self) -> dict | None:
        return None


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------


ADAPTERS = {
    "empathy": EmpathyAdapter,
    "interview": InterviewAdapter,
    "clarity": ClarityAdapter,
    "fairness": FairnessAdapter,
}


def get_adapter(name: str) -> TaskAdapter:
    if name not in ADAPTERS:
        raise ValueError(f"unknown task: {name}. options: {list(ADAPTERS)}")
    return ADAPTERS[name]()


# -------------------------------------------------------------------------
# Selftest
# -------------------------------------------------------------------------


def _selftest() -> int:
    """Light sanity check: build messages for each adapter from canned
    inputs, then parse synthetic responses back. No API calls.
    """
    # Empathy
    a = EmpathyAdapter()
    msgs = a.build_messages(
        row={"text": "Hi Jonathan, I think you're doing great!"},
        examples=[{"text": "Hi Jonathan, bad job.", "empathy": 0}],
    )
    assert msgs[0]["role"] == "system"
    assert "empathy" in msgs[0]["content"].lower()
    assert a.parse('{"label": 1}', {}) == 1
    assert a.parse('{"label": 0}', {}) == 0
    assert a.parse("the answer is 1, definitely", {}) == 1

    # Fairness
    f = FairnessAdapter()
    msgs = f.build_messages(
        row={"first_option": "Policy A", "second_option": "Policy B"},
        examples=[{"first_option": "P1", "second_option": "P2", "majority_vote": "first"}],
    )
    assert any("majority" in m["content"].lower() for m in msgs)
    assert f.parse('{"choice": "first"}', {}) == "first"
    assert f.parse('{"choice": "second"}', {}) == "second"
    assert f.parse("I think the second one is fairer", {}) == "second"

    # Clarity
    c = ClarityAdapter()
    msgs = c.build_messages(
        row={"item": "I am the life of the party"},
        examples=[{"item": "I am sad", "mean_clarity": 5.5}],
    )
    assert c.parse("5.43", {}) == 5.43
    assert c.parse("The clarity is about 4.2", {}) == 4.2
    assert c.parse("9.0", {}) == 7.0  # clipped to max
    assert c.parse("garbage", {}) == 5.4  # falls back to grand mean

    # Interview
    iv = InterviewAdapter()
    row = {f"Q{i}": f"q{i}" for i in range(1, 6)}
    row.update({f"R{i}": f"r{i}" for i in range(1, 5)})
    msgs = iv.build_messages(row=row, examples=[])
    assert len(msgs) == 2
    assert "120 words" in msgs[0]["content"]
    assert iv.parse('"My response here."', {}) == "My response here."
    assert iv.parse("Response: I think...", {}) == "I think..."

    print("adapters selftest OK")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest to run sanity checks (no API calls).")
