"""LLM-as-extractor: zero-shot Big Five scoring via Anthropic (or Gemini).

Frozen pretrained inference only — no labels used, so scores can be cached and
precomputed on all splits without leakage.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

from .data import PROJECT_ROOT, TEXT_COLS, TRAITS

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "llm_cache")

PROMPTS = {
    "A": "A colleague requested vacation for the same week as you; one of you must move it and they refuse. What would you do and why?",
    "C": "You have a project due in two weeks with a light workload, but your boss sometimes adds last-minute tasks. How would you handle the project and why?",
    "E": "After a long day you're invited to a networking meeting with a big client; your colleague may not go and then you'd know no one. What would you do and why?",
    "N": "Your manager gave you negative feedback you disagree with, but it could cost your bonus. How do you feel and what would you do?",
    "O": "The company needs a volunteer to work with a client from Norway, learning about the country/culture (no travel needed). Enjoyable or boring? Why?",
}
PROMPT_ORDER = ["A", "C", "E", "N", "O"]

SYSTEM = (
    "You are an expert organizational psychologist scoring the Big Five personality "
    "traits from a person's written answers to situational judgment questions. "
    "Rate each trait on a continuous 1.00-5.00 scale (higher = more of the trait), "
    "using subtle cues in tone, content, and style. Spread your scores; avoid "
    "defaulting to the middle. Respond with ONLY a JSON object."
)

RUBRIC = (
    "Trait cues:\n"
    "- Openness (O): curiosity, enthusiasm for learning/new cultures, imaginative, abstract.\n"
    "- Conscientiousness (C): planning, diligence, organization, follow-through.\n"
    "- Extraversion (E): sociability, energy, initiative in social settings.\n"
    "- Agreeableness (A): cooperation, empathy, compromise, warmth.\n"
    "- Neuroticism (N): anxiety, worry, emotional reactivity, defensiveness.\n"
)

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

VARIANT_INSTRUCTIONS = {
    "general": "",
    "evidence": (
        "Before assigning scores, privately weigh concrete behavioral evidence in "
        "the answers. Reward specific plans, emotional regulation, and explicit "
        "curiosity; penalize avoidance, hostility, and vague passivity. Output only JSON."
    ),
    "ranked": (
        "Use the full 1.00-5.00 range and make the scores useful for rank ordering "
        "people relative to one another. Avoid compressed middle scores. Output only JSON."
    ),
    "trait_focus": (
        "Score each trait using its eliciting scenario most heavily: vacation conflict "
        "for A, deadline planning for C, networking for E, negative feedback for N, "
        "Norway/culture learning for O. Use the other responses as secondary evidence. "
        "Output only JSON."
    ),
}

SUBFEATURES = [
    "compromise",
    "warmth",
    "conflict_assertiveness",
    "planning_specificity",
    "proactivity",
    "task_diligence",
    "social_approach",
    "social_confidence",
    "emotional_reactivity",
    "defensiveness",
    "emotional_regulation",
    "curiosity",
    "learning_orientation",
    "novelty_enjoyment",
    "agency",
    "specificity",
]

# ---------------------------------------------------------------------------
# Role-play + questionnaire (item-level BFI-2 simulation).
#
# Literature SOTA (Yang et al. 2024, "Predicting the Big Five ... Counselling
# Dialogues"): having the LLM role-play AS the person and answer individual
# personality-inventory items (then aggregate) roughly DOUBLES trait-level
# correlation vs. direct trait scoring. Each item: id, trait, keyed sign.
# 30 items, balanced 3 positive / 3 negative (reverse) per trait.
# ---------------------------------------------------------------------------
QUESTIONNAIRE_VERSION = "bfi2v1"

BFI_ITEMS = [
    # Extraversion
    ("E1", "E", +1, "am outgoing, sociable"),
    ("E2", "E", +1, "am full of energy and enthusiasm"),
    ("E3", "E", +1, "take charge and speak up in a group"),
    ("E4", "E", -1, "tend to be quiet"),
    ("E5", "E", -1, "am sometimes shy or reserved around others"),
    ("E6", "E", -1, "find it hard to get others to notice me"),
    # Agreeableness
    ("A1", "A", +1, "am compassionate and have a soft heart"),
    ("A2", "A", +1, "am respectful and treat others with respect"),
    ("A3", "A", +1, "am helpful and willing to compromise for others"),
    ("A4", "A", -1, "tend to find fault with others"),
    ("A5", "A", -1, "can be cold and uncaring"),
    ("A6", "A", -1, "am sometimes rude or short with people"),
    # Conscientiousness
    ("C1", "C", +1, "like to keep things organized and in order"),
    ("C2", "C", +1, "am dependable and follow through on commitments"),
    ("C3", "C", +1, "keep working until a task is completely finished"),
    ("C4", "C", -1, "tend to be disorganized"),
    ("C5", "C", -1, "have difficulty getting started on tasks"),
    ("C6", "C", -1, "can be careless or leave things half-done"),
    # Negative Emotionality (Neuroticism)
    ("N1", "N", +1, "worry a lot"),
    ("N2", "N", +1, "tend to feel down, blue, or discouraged"),
    ("N3", "N", +1, "get emotional or upset easily"),
    ("N4", "N", -1, "stay calm and optimistic after a setback"),
    ("N5", "N", -1, "am emotionally stable and not easily rattled"),
    ("N6", "N", -1, "rarely feel anxious or afraid"),
    # Openness
    ("O1", "O", +1, "am curious about many different things"),
    ("O2", "O", +1, "am inventive and find clever new ways to do things"),
    ("O3", "O", +1, "enjoy learning about other cultures, art, or ideas"),
    ("O4", "O", -1, "have little interest in abstract or theoretical ideas"),
    ("O5", "O", -1, "prefer routine and dislike new experiences"),
    ("O6", "O", -1, "avoid deep or philosophical discussions"),
]
QUESTIONNAIRE_KEYS = [it[0] for it in BFI_ITEMS]

QUESTIONNAIRE_SYSTEM = (
    "You will read five short answers a person wrote to workplace situational "
    "questions. Step into that person's shoes: infer their personality from the "
    "content, tone, and style of what they wrote, then answer a personality "
    "questionnaire AS IF YOU WERE THEM, honestly reflecting who they appear to be. "
    "For each statement rate agreement on a 1-5 scale where 1=strongly disagree, "
    "2=disagree, 3=neutral, 4=agree, 5=strongly agree. Use the full range and let "
    "different people get different answers. Respond with ONLY a JSON object."
)


def _build_questionnaire_user(row) -> str:
    parts = ["Here are the five answers the person wrote:\n"]
    for trait, col in zip(PROMPT_ORDER, TEXT_COLS):
        parts.append(f"Q ({trait}): {PROMPTS[trait]}\nA: {row[col]}\n")
    parts.append(
        "\nNow answer as this person. Rate how much THEY would agree with each "
        'statement beginning "I ...". Return ONLY a JSON object mapping each item '
        "id to an integer 1-5:\n"
    )
    for iid, _trait, _sign, stem in BFI_ITEMS:
        parts.append(f'  "{iid}": (I {stem})')
    parts.append(
        '\nExample format: {"E1": 4, "E2": 3, ... , "O6": 2}. '
        "Include every item id exactly once."
    )
    return "\n".join(parts)


def aggregate_questionnaire(item_matrix: np.ndarray) -> np.ndarray:
    """(n, 30) raw item responses -> (n, 5) reverse-scored trait means in TRAITS order."""
    trait_cols = {t: [] for t in TRAITS}
    for j, (_iid, trait, sign, _stem) in enumerate(BFI_ITEMS):
        trait_cols[trait].append((j, sign))
    out = np.full((item_matrix.shape[0], len(TRAITS)), np.nan)
    for ti, t in enumerate(TRAITS):
        acc = np.zeros(item_matrix.shape[0])
        cnt = 0
        for j, sign in trait_cols[t]:
            col = item_matrix[:, j]
            acc = acc + (col if sign > 0 else (6.0 - col))
            cnt += 1
        out[:, ti] = acc / max(cnt, 1)
    return out


def _build_user(row, variant: str = "general") -> str:
    parts = ["Here are the person's answers:\n"]
    for trait, col in zip(PROMPT_ORDER, TEXT_COLS):
        parts.append(f"Q ({trait}): {PROMPTS[trait]}\nA: {row[col]}\n")
    parts.append("\n" + RUBRIC)
    extra = VARIANT_INSTRUCTIONS.get(variant, "")
    if extra:
        parts.append("\nAdditional scoring instruction: " + extra)
    parts.append('\nReturn JSON exactly like {"O": 3.4, "C": 4.1, "E": 2.7, "A": 3.9, "N": 2.2}.')
    return "\n".join(parts)


def _build_subfeature_user(row) -> str:
    parts = ["Here are the person's answers:\n"]
    for trait, col in zip(PROMPT_ORDER, TEXT_COLS):
        parts.append(f"Q ({trait}): {PROMPTS[trait]}\nA: {row[col]}\n")
    parts.append(
        "\nRate each behavioral dimension on a continuous 1.00-5.00 scale. "
        "Use concrete evidence from the responses and spread scores for rank ordering. "
        "Return ONLY a JSON object with these exact keys:\n"
        + ", ".join(SUBFEATURES)
    )
    return "\n".join(parts)


_json_re = re.compile(r"\{[^{}]*\}")


def _parse(text: str) -> dict:
    m = _json_re.search(text or "")
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return {k: float(v) for k, v in d.items() if k in TRAITS}
    except Exception:
        return {}


def _anthropic_score_sys(
    user_prompt: str,
    model_name: str,
    system: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in msg.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)


def _gemini_score_sys(
    user_prompt: str,
    model_name: str,
    system: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    import google.generativeai as genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name, system_instruction=system)
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text


def _anthropic_score(
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in msg.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)


def _gemini_score(
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    import google.generativeai as genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name, system_instruction=SYSTEM)
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text


def score_rows(
    df,
    model_name: str | None = None,
    provider: str = "anthropic",
    temperature: float = 0.0,
    sleep_s: float = 0.25,
    variant: str = "general",
):
    """Return (n, 5) LLM trait scores in TRAITS order; cached per respondent."""
    if model_name is None:
        model_name = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else "gemini-2.0-flash"

    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = f"{provider}__{model_name}__scores__{variant}".replace("/", "__")
    out = np.full((len(df), len(TRAITS)), np.nan)

    to_run = []
    for pos, (_, row) in enumerate(df.iterrows()):
        rid = str(row["Respondent_ID"])
        cpath = os.path.join(CACHE_DIR, f"{safe}__{rid}.json")
        if os.path.exists(cpath):
            d = json.load(open(cpath, encoding="utf-8"))
            for i, t in enumerate(TRAITS):
                out[pos, i] = d.get(t, np.nan)
        else:
            to_run.append((pos, row, cpath))

    score_fn = _anthropic_score if provider == "anthropic" else _gemini_score

    for k, (pos, row, cpath) in enumerate(to_run):
        user_prompt = _build_user(row, variant=variant)
        for attempt in range(5):
            try:
                text = score_fn(user_prompt, model_name, temperature, 128)
                d = _parse(text)
                if len(d) < 5:
                    raise ValueError(f"incomplete JSON: {d}")
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(d, f)
                for i, t in enumerate(TRAITS):
                    out[pos, i] = d[t]
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err or "overloaded" in err:
                    wait = min(60, 2 ** attempt * 5)
                    print(f"  rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                elif attempt == 4:
                    print(f"  failed row {k} (id={row['Respondent_ID']}): {e}")
                else:
                    time.sleep(1)
        if (k + 1) % 25 == 0:
            print(f"  llm scored {k + 1}/{len(to_run)}")
        time.sleep(sleep_s)

    col_means = np.nanmean(out, axis=0)
    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return out


def score_subfeatures(
    df,
    model_name: str | None = None,
    provider: str = "anthropic",
    temperature: float = 0.0,
    sleep_s: float = 0.25,
):
    """Return (n, len(SUBFEATURES)) behavioral LLM features; cached per respondent."""
    if model_name is None:
        model_name = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else "gemini-2.0-flash"

    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = f"{provider}__{model_name}__subfeatures".replace("/", "__")
    out = np.full((len(df), len(SUBFEATURES)), np.nan)

    to_run = []
    for pos, (_, row) in enumerate(df.iterrows()):
        rid = str(row["Respondent_ID"])
        cpath = os.path.join(CACHE_DIR, f"{safe}__{rid}.json")
        if os.path.exists(cpath):
            d = json.load(open(cpath, encoding="utf-8"))
            for i, key in enumerate(SUBFEATURES):
                out[pos, i] = d.get(key, np.nan)
        else:
            to_run.append((pos, row, cpath))

    score_fn = _anthropic_score if provider == "anthropic" else _gemini_score

    for k, (pos, row, cpath) in enumerate(to_run):
        user_prompt = _build_subfeature_user(row)
        for attempt in range(5):
            try:
                text = score_fn(user_prompt, model_name, temperature, 512)
                parsed = _json_re.search(text or "")
                if not parsed:
                    raise ValueError("no JSON object")
                d = json.loads(parsed.group(0))
                vals = {key: float(d[key]) for key in SUBFEATURES if key in d}
                if len(vals) < len(SUBFEATURES):
                    raise ValueError(f"incomplete JSON: {vals}")
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(vals, f)
                for i, key in enumerate(SUBFEATURES):
                    out[pos, i] = vals[key]
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err or "overloaded" in err:
                    wait = min(60, 2 ** attempt * 5)
                    print(f"  rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                elif attempt == 4:
                    print(f"  failed subfeature row {k} (id={row['Respondent_ID']}): {e}")
                else:
                    time.sleep(1)
        if (k + 1) % 25 == 0:
            print(f"  llm subfeatures scored {k + 1}/{len(to_run)}")
        time.sleep(sleep_s)

    col_means = np.nanmean(out, axis=0)
    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return out


def score_questionnaire(
    df,
    model_name=None,
    provider="anthropic",
    temperature=0.0,
    sleep_s=0.25,
):
    """Return (n, len(BFI_ITEMS)) raw item responses; cached per respondent.

    Role-play + questionnaire simulation (literature SOTA). No labels used.
    """
    if model_name is None:
        model_name = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else "gemini-2.0-flash"

    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = f"{provider}__{model_name}__questionnaire__{QUESTIONNAIRE_VERSION}".replace("/", "__")
    out = np.full((len(df), len(BFI_ITEMS)), np.nan)

    to_run = []
    for pos, (_, row) in enumerate(df.iterrows()):
        rid = str(row["Respondent_ID"])
        cpath = os.path.join(CACHE_DIR, f"{safe}__{rid}.json")
        if os.path.exists(cpath):
            d = json.load(open(cpath, encoding="utf-8"))
            for i, key in enumerate(QUESTIONNAIRE_KEYS):
                out[pos, i] = d.get(key, np.nan)
        else:
            to_run.append((pos, row, cpath))

    def _q_call(user_prompt):
        if provider == "anthropic":
            return _anthropic_score_sys(user_prompt, model_name, QUESTIONNAIRE_SYSTEM,
                                        temperature, 512)
        return _gemini_score_sys(user_prompt, model_name, QUESTIONNAIRE_SYSTEM,
                                 temperature, 512)

    for k, (pos, row, cpath) in enumerate(to_run):
        user_prompt = _build_questionnaire_user(row)
        for attempt in range(5):
            try:
                text = _q_call(user_prompt)
                parsed = _json_re.search(text or "")
                if not parsed:
                    raise ValueError("no JSON object")
                d = json.loads(parsed.group(0))
                vals = {key: float(d[key]) for key in QUESTIONNAIRE_KEYS if key in d}
                if len(vals) < len(QUESTIONNAIRE_KEYS):
                    raise ValueError(f"incomplete JSON ({len(vals)}/{len(QUESTIONNAIRE_KEYS)})")
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(vals, f)
                for i, key in enumerate(QUESTIONNAIRE_KEYS):
                    out[pos, i] = vals[key]
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err or "overloaded" in err:
                    wait = min(60, 2 ** attempt * 5)
                    print(f"  rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                elif attempt == 4:
                    print(f"  failed questionnaire row {k} (id={row['Respondent_ID']}): {e}")
                else:
                    time.sleep(1)
        if (k + 1) % 25 == 0:
            print(f"  llm questionnaire scored {k + 1}/{len(to_run)}")
        time.sleep(sleep_s)

    col_means = np.nanmean(out, axis=0)
    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return out


# ---------------------------------------------------------------------------
# Two-stage persona-summary questionnaire (Liu et al. 2025, "SummaryAdded").
# Stage 1: compress the 5 SJI answers into a natural-language persona summary.
# Stage 2: answer the BFI-2 item battery conditioned on answers + summary.
# The summary encodes synergistic second-order trait info and empirically
# improves item-level role-play prediction. No labels used -> cacheable.
# ---------------------------------------------------------------------------
SUMMARY_VERSION = "v1"

SUMMARY_SYSTEM = (
    "You are an expert personality psychologist. You will read five short answers a "
    "person wrote to workplace situational questions. Write a concise, concrete "
    "personality profile of this person: what their tone, choices, and reasoning "
    "reveal about how they typically think, feel, and behave. Focus on behavioral "
    "evidence relevant to the Big Five (openness, conscientiousness, extraversion, "
    "agreeableness, emotional stability). Be specific and avoid hedging or "
    "boilerplate. Write 3-5 sentences of prose only."
)


def _build_summary_user(row) -> str:
    parts = ["Here are the five answers the person wrote:\n"]
    for trait, col in zip(PROMPT_ORDER, TEXT_COLS):
        parts.append(f"Q ({trait}): {PROMPTS[trait]}\nA: {row[col]}\n")
    parts.append("\nWrite the personality profile now (prose only, 3-5 sentences).")
    return "\n".join(parts)


def score_persona_summaries(
    df,
    model_name=None,
    provider="anthropic",
    temperature=0.3,
    sleep_s=0.25,
):
    """Return a list of persona-summary strings (one per row); cached per respondent."""
    if model_name is None:
        model_name = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else "gemini-2.0-flash"

    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = f"{provider}__{model_name}__persona_summary__{SUMMARY_VERSION}".replace("/", "__")
    out = [None] * len(df)

    to_run = []
    for pos, (_, row) in enumerate(df.iterrows()):
        rid = str(row["Respondent_ID"])
        cpath = os.path.join(CACHE_DIR, f"{safe}__{rid}.json")
        if os.path.exists(cpath):
            out[pos] = json.load(open(cpath, encoding="utf-8")).get("summary", "")
        else:
            to_run.append((pos, row, cpath))

    for k, (pos, row, cpath) in enumerate(to_run):
        user_prompt = _build_summary_user(row)
        for attempt in range(5):
            try:
                if provider == "anthropic":
                    text = _anthropic_score_sys(user_prompt, model_name, SUMMARY_SYSTEM,
                                                temperature, 400)
                else:
                    text = _gemini_score_sys(user_prompt, model_name, SUMMARY_SYSTEM,
                                             temperature, 400)
                text = (text or "").strip()
                if len(text) < 20:
                    raise ValueError("summary too short")
                with open(cpath, "w", encoding="utf-8") as fh:
                    json.dump({"summary": text}, fh)
                out[pos] = text
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err or "overloaded" in err:
                    wait = min(60, 2 ** attempt * 5)
                    print(f"  rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                elif attempt == 4:
                    print(f"  failed summary row {k} (id={row['Respondent_ID']}): {e}")
                    out[pos] = ""
                else:
                    time.sleep(1)
        if (k + 1) % 25 == 0:
            print(f"  persona summaries scored {k + 1}/{len(to_run)}")
        time.sleep(sleep_s)

    return out


def _build_questionnaire_user_with_summary(row, summary: str) -> str:
    parts = ["Here are the five answers the person wrote:\n"]
    for trait, col in zip(PROMPT_ORDER, TEXT_COLS):
        parts.append(f"Q ({trait}): {PROMPTS[trait]}\nA: {row[col]}\n")
    if summary:
        parts.append(
            "\nThe following is a supplementary personality profile of this person, "
            "which you may refer to as you see fit:\n" + summary + "\n"
        )
    parts.append(
        "\nNow answer as this person. Rate how much THEY would agree with each "
        'statement beginning "I ...". Return ONLY a JSON object mapping each item '
        "id to an integer 1-5:\n"
    )
    for iid, _trait, _sign, stem in BFI_ITEMS:
        parts.append(f'  "{iid}": (I {stem})')
    parts.append(
        '\nExample format: {"E1": 4, "E2": 3, ... , "O6": 2}. '
        "Include every item id exactly once."
    )
    return "\n".join(parts)


def score_questionnaire_summary(
    df,
    model_name=None,
    provider="anthropic",
    temperature=0.0,
    sleep_s=0.25,
):
    """Two-stage: persona summary -> questionnaire. Returns (n, len(BFI_ITEMS))."""
    if model_name is None:
        model_name = DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else "gemini-2.0-flash"

    summaries = score_persona_summaries(df, model_name=model_name, provider=provider)

    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = f"{provider}__{model_name}__questionnaire__{QUESTIONNAIRE_VERSION}_sum".replace("/", "__")
    out = np.full((len(df), len(BFI_ITEMS)), np.nan)

    to_run = []
    for pos, (_, row) in enumerate(df.iterrows()):
        rid = str(row["Respondent_ID"])
        cpath = os.path.join(CACHE_DIR, f"{safe}__{rid}.json")
        if os.path.exists(cpath):
            d = json.load(open(cpath, encoding="utf-8"))
            for i, key in enumerate(QUESTIONNAIRE_KEYS):
                out[pos, i] = d.get(key, np.nan)
        else:
            to_run.append((pos, row, cpath))

    for k, (pos, row, cpath) in enumerate(to_run):
        user_prompt = _build_questionnaire_user_with_summary(row, summaries[pos])
        for attempt in range(5):
            try:
                if provider == "anthropic":
                    text = _anthropic_score_sys(user_prompt, model_name, QUESTIONNAIRE_SYSTEM,
                                                temperature, 512)
                else:
                    text = _gemini_score_sys(user_prompt, model_name, QUESTIONNAIRE_SYSTEM,
                                             temperature, 512)
                parsed = _json_re.search(text or "")
                if not parsed:
                    raise ValueError("no JSON object")
                d = json.loads(parsed.group(0))
                vals = {key: float(d[key]) for key in QUESTIONNAIRE_KEYS if key in d}
                if len(vals) < len(QUESTIONNAIRE_KEYS):
                    raise ValueError(f"incomplete JSON ({len(vals)}/{len(QUESTIONNAIRE_KEYS)})")
                with open(cpath, "w", encoding="utf-8") as fh:
                    json.dump(vals, fh)
                for i, key in enumerate(QUESTIONNAIRE_KEYS):
                    out[pos, i] = vals[key]
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate" in err or "overloaded" in err:
                    wait = min(60, 2 ** attempt * 5)
                    print(f"  rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                elif attempt == 4:
                    print(f"  failed q-summary row {k} (id={row['Respondent_ID']}): {e}")
                else:
                    time.sleep(1)
        if (k + 1) % 25 == 0:
            print(f"  llm q-summary scored {k + 1}/{len(to_run)}")
        time.sleep(sleep_s)

    col_means = np.nanmean(out, axis=0)
    nan_mask = np.isnan(out)
    if np.any(nan_mask):
        out[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return out
