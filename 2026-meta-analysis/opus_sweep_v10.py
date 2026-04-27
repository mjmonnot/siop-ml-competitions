"""
opus_sweep_v10.py — Comprehensive Claude Opus API sweep for SIOP pipeline v10/v11.

Replaces opus_sweep_v9.py. Reads a vision (or full) batch JSON log, selects targets with
broad criteria: Priority 0 correlation-signal blanks (see pipeline `_scan_pdf_for_correlation_signal`),
regex/text_matrix has-r, design-exclusion blanks, all blanks, single-effect docling,
Priority 7 suspicious-value triangulation, and manual_override verification. Scores PDF pages,
calls Opus with images (deep-search prompt for P0).

Requires: pip install anthropic pymupdf
Env: ANTHROPIC_API_KEY

Read OVERRIDE_AND_DEFERRED_FIXES.md, GROUND_TRUTH_LOG.md, generalizable_rules.md for context.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from typing import Any

import fitz

try:
    from pipeline_dev import _scan_pdf_for_correlation_signal
except ImportError:
    # Standalone run without full pipeline on PYTHONPATH
    def _scan_pdf_for_correlation_signal(pdf_path: str) -> bool:
        if not pdf_path or not os.path.isfile(pdf_path):
            return False
        try:
            doc = fitz.open(pdf_path)
            try:
                parts: list[str] = []
                for i in range(min(5, len(doc))):
                    parts.append(doc[i].get_text("text") or "")
            finally:
                doc.close()
        except Exception:
            return False
        blob = "\n".join(parts).lower()
        if not blob.strip():
            return False
        _phrases = (
            "pearson r",
            "pearson's r",
            "pearson correlation",
            "pearson product-moment",
            "pearson product moment",
            "zero-order correlation",
            "zero order correlation",
            "bivariate correlation",
            "bivariate pearson",
            "correlation matrix",
            "correlation table",
            "intercorrelations",
            "inter-correlations",
            "table of correlations",
            "correlations among",
            "correlations between",
            "correlations are presented",
            "correlations are reported",
            "correlations are shown",
            "r =",
            "r=",
            "(r =",
            "(r=",
            "presented in table",
            "shown in table",
            "bivariate correlations are available",
            "correlations are available in",
        )
        return any(p in blob for p in _phrases)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

OPUS_MODEL = "claude-opus-4-6"
DEFAULT_LOG_PATH = "pipeline_log_v10_final.json"
DEFAULT_PDF_DIR = "pdfs"
DEFAULT_OUT_JSON = "opus_sweep_results.json"
CALIBRATED_IMP = 0.152  # optimal imputation (GROUND_TRUTH_LOG-derived)

# Fallback if pipeline import fails (keep in sync with MANUAL_OVERRIDES keys)
MANUAL_OVERRIDE_IDS_FALLBACK = {
    "study2",
    "study4",
    "study13",
    "study16",
    "study18",
    "study19",
    "study22",
    "study24",
    "study31",
    "study32",
    "study35",
    "study38",
    "study44",
    "study49",
    "study50",
    "study55",
    "study64",
    "study67",
    "study71",
    "study77",
    "study79",
    "study81",
    "study83",
    "study85",
    "study93",
    "study97",
    "study99",
    "study105",
    "study111",
    "study113",
    "study116",
    "study120",
    "study121",
    "study23",
    "study47",
    "study56",
    "study63",
    "study69",
    "study76",
    "study86",
    "study89",
    "study90",
    "study91",
    "study92",
    "study94",
    "study96",
    "study103",
    "study108",
    "study115",
    "study117",
    "study119",
}

GT_VERIFIED_DEFAULT: dict[str, float] = {
    "study1": 0.200,
    "study2": 0.158,
    "study4": 0.340,
    "study10": 0.320,
    "study12": 0.195,
    "study13": 0.299,
    "study16": 0.070,
    "study18": 0.075,
    "study19": 0.170,
    "study20": 0.190,
    "study22": 0.170,
    "study23": 0.315,
    "study24": 0.255,
    "study30": 0.196,
    "study31": 0.390,
    "study32": 0.036,
    "study35": 0.337,
    "study38": 0.410,
    "study40": 0.179,
    "study43": 0.260,
    "study44": 0.257,
    "study45": 0.120,
    "study48": 0.107,
    "study49": 0.230,
    "study50": 0.030,
    "study53": 0.160,
    "study54": 0.120,
    "study55": 0.336,
    "study60": 0.310,
    "study61": 0.345,
    "study64": 0.223,
    "study66": 0.390,
    "study67": 0.410,
    "study68": 0.190,
    "study71": 0.180,
    "study72": 0.100,
    "study73": 0.240,
    "study77": 0.231,
    "study79": 0.200,
    "study81": 0.175,
    "study85": 0.202,
    "study87": 0.130,
    "study88": 0.274,
    "study93": 0.180,
    "study95": 0.308,
    "study97": 0.350,
    "study98": 0.104,
    "study99": 0.236,
    "study100": 0.300,
    "study102": 0.280,
    "study105": 0.170,
    "study106": 0.088,
    "study109": 0.185,
    "study112": 0.170,
    "study113": 0.250,
    "study114": 0.120,
    "study116": 0.090,
    "study120": 0.195,
    "study121": 0.185,
    "study124": 0.698,
    "study125": 0.329,
    "study126": 0.110,
}


OPUS_SYSTEM_PROMPT = """You are a meta-analysis research assistant. Your task is to extract the zero-order bivariate Pearson correlation coefficient (r) between an eligible TRUST construct and an eligible SUBJECTIVE WELLBEING (SWB) construct from an academic paper.

=== ELIGIBLE TRUST CONSTRUCTS ===

INCLUDE:
- Generalized/interpersonal trust: "Most people can be trusted", "trust in people", "social trust", "general trust"
- Institutional trust (when items assess TRUSTWORTHINESS of human actors): trust in police, government, healthcare system, scientists/science, politicians — ONLY when items ask about honesty, reliability, fairness, or integrity
- Medical trust/mistrust: trust in physicians, healthcare providers, medical profession
- Partner/peer trust: trust in romantic partner, trust in peers (emotional trust, reliability trust)
- Cognitive social capital: "cognitive aspects of social capital", "perceived trustworthiness in community"
- WAQ Trustworthiness: World Assumptions Questionnaire trustworthiness subscale
- Trust in science (Credibility of Science Scale): items about scientists providing reliable, unbiased information

EXCLUDE (do not extract):
- Self-trust, self-efficacy, self-confidence
- Collective efficacy: "confidence in society", "society has its future under control" (Keller scale)
- Policy attitudes: "trust in COVID regulations", "trust in vaccination policies" — these assess policy appropriateness, not human trustworthiness
- Social support availability (MSPSS): measures access to support, not trustworthiness of others
- Technology trust: trust in AI, privacy concerns
- Trust as MEDIATOR: if trust mediates the relationship between two other variables and no zero-order r is reported

=== ELIGIBLE SWB CONSTRUCTS ===

INCLUDE:
- Global life satisfaction: SWLS, single-item "How satisfied are you with your life overall?"
- Happiness: subjective happiness scale, daily happiness
- Psychological wellbeing: Ryff PWB scales, eudaimonic wellbeing, meaning in life, purpose in life
- Positive/negative affect: PANAS, affect scales (positive affect = good; negative affect = SIGN FLIP)
- Psychological distress: depression (CES-D, PHQ, BDI), anxiety (GAD-7, STAI), psychological distress (GHQ-12, K-10, HSCL), mental health QoL — SIGN FLIP REQUIRED
- Quality of life: WHOQOL psychological domain, mental health QoL — physical-only QoL excluded
- General wellbeing composites

EXCLUDE:
- Job satisfaction, occupational satisfaction, work engagement
- Domain-specific satisfaction: housing, care, residential, service satisfaction
- Physical health only (without psychological component)
- Social capital outcomes: social connectedness, sense of community (these are resources, not evaluations)
- Behavioral outcomes: medication adherence, exercise behavior
- Clinician-rated or observer-rated outcomes — must be SELF-REPORTED
- Parent-reported child outcomes — the child's rater must be the child themselves
- Importance ratings: "how important is X to you?" — not an experiential evaluation

=== STATISTICAL RULES ===

INCLUDE:
- Pearson r (zero-order, bivariate)
- SEM latent correlations from Fornell-Lacker discriminant validity tables (treat as r estimate)

EXCLUDE:
- Spearman rho (ρ) — rank correlation, not Pearson r
- Standardized regression beta (β) — partial, not zero-order
- Odds ratios (OR)
- Partial correlations (controlling for covariates)
- SEM path coefficients
- F-statistics from ANOVA (cannot convert without additional information)
- Chi-square statistics

EXCEPTION: If only Spearman is available in upper triangle but Pearson r is explicitly labeled in lower triangle of the same table, extract Pearson r from lower triangle only.

=== DESIGN RULES ===

EXCLUDE (return NULL):
- Ecological/aggregate designs: studies where N < 100 AND observations are countries, regions, or organizations (not individuals)
- Longitudinal cross-wave-only: all correlations are T1 predictor × T2 outcome with no same-wave pairs
- Group-based designs: only means by group, ANOVA, latent class comparisons — no bivariate r between continuous variables

INCLUDE with caveats:
- Longitudinal with same-wave pairs: extract same-wave (T1×T1) pairs; ignore cross-wave
- Multiple waves: prefer Wave 1 (earliest) when multiple same-wave measurements available
- Independent subsamples (different people: pre/post, cultures, genders): average r across subsamples

=== SIGN DIRECTION ===

All r values should be expressed as: higher trust → higher wellbeing (positive r).
Apply sign flip when:
- Outcome is reverse-valenced (depression, anxiety, distress, negative affect): if r < 0, report |r|; if r > 0 and it seems wrong, check scale direction
- Predictor is reverse-keyed (mistrust, cynicism, medical mistrust): if r > 0 for mistrust×depression (higher mistrust→more depression = correct direction), report |r|
Rule: net_flip = is_negative_outcome XOR is_distrust_predictor; flip sign if net_flip=True

=== AGGREGATION ===

When multiple eligible pairs exist:
1. If same predictor has BOTH a distress outcome AND a life satisfaction outcome (dep+LS pair): keep BOTH and average — do not drop one
2. If multiple trust predictors: prefer interpersonal > institutional > cognitive SC; keep highest priority
3. If multiple SWB outcomes for same predictor: average all eligible ones
4. If multiple waves: use Wave 1 values only, then average subsamples

=== OUTPUT FORMAT ===

Respond ONLY in this exact format — no preamble, no explanation outside the fields:

TRUST_CONSTRUCT: [exact label from paper, or NULL]
TRUST_ELIGIBLE: [yes/no — is this an admissible trust construct per above rules?]
SWB_CONSTRUCT: [exact label from paper, or NULL]
SWB_ELIGIBLE: [yes/no — is this an admissible SWB construct per above rules?]
STATISTIC_TYPE: [pearson_r / spearman_rho / beta / OR / F / other]
RAW_R_VALUE: [numeric value before sign flip, or NULL]
SIGN_FLIPPED: [yes/no]
FINAL_R: [absolute value after sign flip, or NULL if no admissible pair found]
N_PAIRS_AVERAGED: [number of eligible pairs averaged to produce FINAL_R]
CONFIDENCE: [high / medium / low]
DESIGN_FLAG: [none / ecological / cross_wave_only / group_based / spearman_only]
REASONING: [2-4 sentences explaining: what table/figure you found, which constructs qualified,
             what the values were, any sign flips applied, why any candidates were excluded]
"""

OPUS_SYSTEM_PROMPT_DEEP_SEARCH = OPUS_SYSTEM_PROMPT + """

=== DEEP SEARCH MODE ===

This paper explicitly mentions bivariate Pearson correlations or a correlation matrix in its abstract or methods section, but automated extraction returned blank. Search every page carefully for:

1. Any table with the word "correlation" in its title or caption
2. Any inline r = value in the results text (e.g., "r = .24, p < .01")
3. Any supplementary table reference that might contain correlations
4. Any footnote describing where correlations are reported
5. Appendix tables that might contain the correlation matrix
6. Any mention of "Table X" in the results text where X might be a correlation table

The correlations may appear:
- In a different location than the main results (appendix, supplementary, footnote)
- As inline text rather than a table (e.g., "The correlation between trust and wellbeing was r = .23")
- In a combined descriptive statistics table where correlations appear alongside M and SD
- Under a non-standard label (e.g., "associations", "relationships", "associations between variables")

Report what you find even if the r value is small or non-significant. Non-significant correlations are still valid Pearson r values for meta-analysis.
"""


def default_override_ids() -> set[str]:
    for _mod in ("pipeline_dev", "pipeline_test"):
        try:
            m = __import__(_mod, fromlist=["MANUAL_OVERRIDES"])
            return set(m.MANUAL_OVERRIDES.keys())
        except (ImportError, AttributeError):
            continue
    return set(MANUAL_OVERRIDE_IDS_FALLBACK)


def _is_suspicious_value(
    r: float, tier: str, n_effects: int, study_id: str
) -> tuple[bool, str | None]:
    """
    Returns (True, reason) if the value warrants Opus verification.
    study_id reserved for future corpus-specific rules.
    """
    _ = study_id
    corpus_mean = 0.22
    if r > 0.50 and tier not in ("manual_override",):
        return True, f"high_value_{r:.3f}"
    if tier in ("regex",) and abs(r - corpus_mean) > 0.15:
        return True, f"regex_outlier_{r:.3f}"
    if n_effects == 1 and (r < 0.05 or r > 0.45) and tier not in ("manual_override",):
        return True, f"single_effect_outlier_{r:.3f}"
    return False, None


def _target_base(
    sid: str,
    tier: str,
    n_candidates: int,
    n_effects: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "study_id": sid,
        "extraction_tier": tier,
        "n_candidates_found": n_candidates,
        "n_effects": n_effects,
        "notes_snippet": notes[:200],
    }


def select_opus_targets(
    log_path: str,
    override_ids: set[str],
    cap: int = 60,
    pdf_dir: str = DEFAULT_PDF_DIR,
    priority0_cap: int = 20,
) -> list[dict[str, Any]]:
    """
    Select studies for Opus sweep from pipeline batch log.
    Priority (lowest number = highest priority):
      0 — blank + explicit correlation signal in first PDF pages (not manual_override tier)
      1 — regex tier with r (low structural confidence)
      2 — text_matrix tier with r
      3 — blank + design_exclusion in tier (gate may have over-fired)
      4 — all other blank studies
      5 — docling with exactly one effect (may have missed pairs)
      7 — has-r but suspicious value / tier (triangulation)
      6 — MANUAL_OVERRIDE with non-null r (independent Opus verification of overrides)

    Up to ``priority0_cap`` Priority-0 slots are reserved; remaining capacity fills
    priorities 1–5 and 7, then up to 12 Priority-6 slots at the end (same as v10).
    """
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    targets: list[dict[str, Any]] = []

    for s in log:
        sid = s["study_id"]
        r = s.get("aggregate_r")
        tier = (s.get("extraction_tier") or "") or ""
        n_candidates = s.get("n_candidates_found", 0) or 0
        n_effects = s.get("n_effects", 0) or 0
        notes = " ".join(s.get("notes") or [])

        pdf_path = os.path.join(pdf_dir, f"{sid}.pdf")
        rec: dict[str, Any] | None = None

        # Priority 0: blank studies with explicit correlation signal in abstract/methods window
        if r is None and tier not in ("manual_override",):
            if _scan_pdf_for_correlation_signal(pdf_path):
                rec = {
                    **_target_base(sid, tier, n_candidates, n_effects, notes),
                    "reason": "correlation_signal_blank",
                    "current_r": None,
                    "priority": 0,
                    "signal": "pearson_or_bivariate_mentioned_in_text",
                }

        if rec is None and r is not None and tier == "regex":
            rec = {
                **_target_base(sid, tier, n_candidates, n_effects, notes),
                "reason": "regex_hasr",
                "current_r": r,
                "priority": 1,
            }
        elif rec is None and r is not None and tier == "text_matrix":
            rec = {
                **_target_base(sid, tier, n_candidates, n_effects, notes),
                "reason": "text_matrix_hasr",
                "current_r": r,
                "priority": 2,
            }
        elif rec is None and r is None and "design_exclusion" in tier:
            rec = {
                **_target_base(sid, tier, n_candidates, n_effects, notes),
                "reason": "design_exclusion_blank",
                "current_r": None,
                "priority": 3,
            }
        elif rec is None and r is None:
            rec = {
                **_target_base(sid, tier, n_candidates, n_effects, notes),
                "reason": "blank_any_tier",
                "current_r": None,
                "priority": 4,
            }
        elif rec is None and r is not None and tier == "docling" and n_effects == 1:
            rec = {
                **_target_base(sid, tier, n_candidates, n_effects, notes),
                "reason": "single_effect_docling",
                "current_r": r,
                "priority": 5,
            }
        elif (
            rec is None
            and r is not None
            and tier not in ("manual_override",)
        ):
            suspicious, s_reason = _is_suspicious_value(
                float(r), tier, int(n_effects or 0), sid
            )
            if suspicious and s_reason:
                rec = {
                    **_target_base(sid, tier, n_candidates, n_effects, notes),
                    "reason": f"suspicious_value_{s_reason}",
                    "current_r": r,
                    "priority": 7,
                }
        elif rec is None and sid in override_ids and r is not None:
            rec = {
                **_target_base(sid, tier, n_candidates, n_effects, notes),
                "reason": "manual_override_verify",
                "current_r": r,
                "priority": 6,
            }

        if rec is not None:
            targets.append(rec)

    targets.sort(key=lambda x: (x["priority"], x["study_id"]))
    p0 = [t for t in targets if t["priority"] == 0]
    p6 = [t for t in targets if t["priority"] == 6]
    rest = [t for t in targets if t["priority"] not in (0, 6)]

    k0 = min(len(p0), priority0_cap, max(0, cap))
    rem_after_p0 = max(0, cap - k0)
    k6 = min(len(p6), 12, rem_after_p0)
    take_mid = max(0, rem_after_p0 - k6)
    merged = p0[:k0] + rest[:take_mid] + p6[:k6]
    return merged[:cap]


def select_pages_for_opus(pdf_path: str, max_pages: int = 25) -> list[int]:
    """
    Prioritize pages most likely to contain correlation tables.
    Returns sorted 0-based page indices (max len max_pages).
    """
    doc = fitz.open(pdf_path)
    try:
        scored_pages: list[tuple[int, int]] = []
        for i in range(len(doc)):
            text = (doc[i].get_text("text") or "").lower()
            score = 0
            if any(w in text for w in ("pearson", "correlation", "bivariate")):
                score += 3
            if any(w in text for w in ("trust", "confidence", "social capital")):
                score += 2
            if any(
                w in text
                for w in ("wellbeing", "well-being", "satisfaction", "depression", "anxiety")
            ):
                score += 2
            if "table" in text and any(c.isdigit() for c in text):
                score += 1
            if any(w in text for w in ("appendix", "supplement", "online")):
                score += 1
            if any(p in text for p in ("diagonal", "upper triangle", "lower triangle")):
                score += 2
            scored_pages.append((i, score))

        top = sorted(scored_pages, key=lambda x: -x[1])[: max(0, max_pages - 4)]
        mandatory = [0, 1, len(doc) - 2, len(doc) - 1]
        idx_set = {p for p, _ in top}
        for p in mandatory:
            if 0 <= p < len(doc):
                idx_set.add(p)
        ordered = sorted(idx_set)
        return ordered[:max_pages]
    finally:
        doc.close()


def _parse_float_or_null(val: str) -> float | None:
    val = (val or "").strip()
    if not val or val.upper() == "NULL":
        return None
    try:
        return float(val.replace(",", "."))
    except ValueError:
        return None


def _parse_opus_structured_text(text: str) -> dict[str, Any]:
    """Parse line-oriented Opus response; collect multi-line REASONING."""
    out: dict[str, Any] = {
        "opus_trust": None,
        "opus_trust_eligible": None,
        "opus_swb": None,
        "opus_swb_eligible": None,
        "statistic_type": None,
        "opus_r": None,
        "raw_r_value": None,
        "sign_flipped": None,
        "n_pairs_averaged": None,
        "opus_confidence": None,
        "design_flag": None,
        "opus_reasoning": None,
        "r_value_raw": None,
    }
    if not text:
        return out
    lines = text.strip().split("\n")
    reasoning_lines: list[str] = []
    in_reasoning = False
    for line in lines:
        if line.startswith("TRUST_CONSTRUCT:"):
            in_reasoning = False
            out["opus_trust"] = line.split(":", 1)[1].strip()
        elif line.startswith("TRUST_ELIGIBLE:"):
            in_reasoning = False
            out["opus_trust_eligible"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("SWB_CONSTRUCT:"):
            in_reasoning = False
            out["opus_swb"] = line.split(":", 1)[1].strip()
        elif line.startswith("SWB_ELIGIBLE:"):
            in_reasoning = False
            out["opus_swb_eligible"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("STATISTIC_TYPE:"):
            in_reasoning = False
            out["statistic_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("RAW_R_VALUE:"):
            in_reasoning = False
            out["r_value_raw"] = line.split(":", 1)[1].strip()
            out["raw_r_value"] = _parse_float_or_null(out["r_value_raw"])
        elif line.startswith("SIGN_FLIPPED:"):
            in_reasoning = False
            out["sign_flipped"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("FINAL_R:"):
            in_reasoning = False
            out["opus_r"] = _parse_float_or_null(line.split(":", 1)[1].strip())
        elif line.startswith("N_PAIRS_AVERAGED:"):
            in_reasoning = False
            npv = line.split(":", 1)[1].strip()
            try:
                out["n_pairs_averaged"] = int(float(npv))
            except ValueError:
                out["n_pairs_averaged"] = None
        elif line.startswith("CONFIDENCE:"):
            in_reasoning = False
            out["opus_confidence"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("DESIGN_FLAG:"):
            in_reasoning = False
            out["design_flag"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("REASONING:"):
            in_reasoning = True
            reasoning_lines = [line.split(":", 1)[1].strip()]
        elif in_reasoning and line.strip():
            reasoning_lines.append(line.strip())
    if reasoning_lines:
        out["opus_reasoning"] = " ".join(reasoning_lines)
    return out


def opus_extract_study(
    study_id: str,
    pdf_path: str,
    max_pages: int = 25,
    client: Any | None = None,
    deep_search: bool = False,
) -> dict[str, Any]:
    """Extract trust × SWB correlation using Opus vision on selected PDF page images."""
    if anthropic is None:
        return {
            "study_id": study_id,
            "error": "anthropic package not installed (pip install anthropic)",
            "opus_r": None,
        }

    if not os.path.exists(pdf_path):
        return {"study_id": study_id, "error": "PDF not found", "opus_r": None}

    _client = client or anthropic.Anthropic()
    page_indices = select_pages_for_opus(pdf_path, max_pages=max_pages)

    try:
        doc = fitz.open(pdf_path)
        content: list = []
        for page_num in page_indices:
            if page_num < 0 or page_num >= len(doc):
                continue
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.standard_b64encode(img_bytes).decode("ascii")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                }
            )
        doc.close()

        content.append(
            {
                "type": "text",
                "text": (
                    f"Study ID: {study_id}. PDF pages sent (0-based indices): {page_indices}. "
                    "Extract the bivariate Pearson r between trust and subjective wellbeing per the "
                    "system prompt. If no admissible pair exists, set FINAL_R and RAW_R_VALUE to NULL."
                ),
            }
        )

        sys_prompt = OPUS_SYSTEM_PROMPT_DEEP_SEARCH if deep_search else OPUS_SYSTEM_PROMPT
        response = _client.messages.create(
            model=OPUS_MODEL,
            max_tokens=2048,
            system=sys_prompt,
            messages=[{"role": "user", "content": content}],
        )

        text = response.content[0].text
        parsed = _parse_opus_structured_text(text)

        return {
            "study_id": study_id,
            "opus_pages_used": page_indices,
            "opus_raw": text,
            "opus_r": parsed["opus_r"],
            "opus_trust": parsed["opus_trust"],
            "opus_trust_eligible": parsed["opus_trust_eligible"],
            "opus_swb": parsed["opus_swb"],
            "opus_swb_eligible": parsed["opus_swb_eligible"],
            "statistic_type": parsed["statistic_type"],
            "opus_confidence": parsed["opus_confidence"],
            "opus_reasoning": parsed["opus_reasoning"],
            "sign_flipped": parsed["sign_flipped"],
            "r_value_raw": parsed["r_value_raw"],
            "raw_r_value": parsed["raw_r_value"],
            "n_pairs_averaged": parsed["n_pairs_averaged"],
            "design_flag": parsed["design_flag"],
            "error": None,
        }
    except Exception as e:
        return {"study_id": study_id, "error": str(e), "opus_r": None}


def load_gt_verified_from_json(path: str | None) -> dict[str, float]:
    """Merge file contents into a float dict; empty if missing."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def build_submission_from_opus_sweep(
    sweep_results_path: str,
    log_path: str,
    out_csv: str,
    *,
    articles_csv: str = "dev_articles.csv",
    imp: float = CALIBRATED_IMP,
    gt_verified: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Apply high-confidence Opus recommendations to pipeline log predictions and write submission CSV.
    Skips LOW_CONFIDENCE rows. Uses ``imp`` for blanks in baseline and for false-positive corrections.
    """
    gt_verified = gt_verified if gt_verified is not None else dict(GT_VERIFIED_DEFAULT)
    with open(sweep_results_path, encoding="utf-8") as f:
        results: list[dict[str, Any]] = json.load(f)
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    log_by_sid = {str(s["study_id"]): s for s in log}

    study_order: list[str] = []
    if os.path.isfile(articles_csv):
        with open(articles_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = (row.get("studyid") or row.get("study_id") or "").strip()
                if sid:
                    study_order.append(sid)
    if not study_order:
        study_order = sorted(log_by_sid.keys())

    pred: dict[str, float] = {}
    for sid in study_order:
        s = log_by_sid.get(sid)
        if not s:
            pred[sid] = float(imp)
            continue
        r = s.get("aggregate_r")
        pred[sid] = float(r) if r is not None else float(imp)

    n_recovered = 0
    n_fp_corrected = 0
    for row in results:
        rec = row.get("recommendation") or ""
        if "LOW_CONFIDENCE" in rec:
            continue
        if (row.get("opus_confidence") or "").lower() != "high":
            continue
        sid = str(row.get("study_id") or "")
        if not sid:
            continue
        if rec == "REVIEW_OPUS_RECOVERED" and row.get("opus_r") is not None:
            pred[sid] = float(row["opus_r"])
            n_recovered += 1
        elif rec == "REVIEW_OPUS_SAYS_BLANK":
            s = log_by_sid.get(sid, {})
            tier = (s.get("extraction_tier") or row.get("extraction_tier") or "") or ""
            if tier in ("regex", "text_matrix"):
                pred[sid] = float(imp)
                n_fp_corrected += 1

    def _mse_on_gt(p: dict[str, float]) -> float:
        terms = [(p.get(sid, imp) - g) ** 2 for sid, g in gt_verified.items()]
        return sum(terms) / max(1, len(terms))

    pred0: dict[str, float] = {}
    for sid in study_order:
        s = log_by_sid.get(sid)
        if not s:
            pred0[sid] = float(imp)
        else:
            r = s.get("aggregate_r")
            pred0[sid] = float(r) if r is not None else float(imp)
    mse_before = _mse_on_gt(pred0)
    mse_after = _mse_on_gt(pred)
    mse_delta = mse_before - mse_after

    _dir = os.path.dirname(os.path.abspath(out_csv))
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["studyid", "aggregateeffectsize"])
        for sid in study_order:
            w.writerow([sid, pred.get(sid, imp)])

    print("\n=== BUILD SUBMISSION (Opus sweep) ===")
    print(f"  Wrote: {out_csv} (imp={imp})")
    print(f"  New recoveries applied (high conf): {n_recovered}")
    print(f"  False positives corrected (regex/text_matrix, high conf): {n_fp_corrected}")
    print(f"  MSE on GT subset: before={mse_before:.6f} after={mse_after:.6f} improvement={mse_delta:.6f}")

    return {
        "submission_csv": out_csv,
        "n_recoveries": n_recovered,
        "n_false_positives_corrected": n_fp_corrected,
        "mse_gt_subset_before": mse_before,
        "mse_gt_subset_after": mse_after,
        "estimated_mse_improvement": mse_delta,
    }


def run_opus_sweep_v10(
    log_path: str,
    pdf_dir: str,
    output_path: str,
    override_ids: set[str] | None = None,
    gt_verified: dict[str, float] | None = None,
    cap: int = 60,
    sleep_s: float = 3.0,
    max_pages: int = 25,
    dry_run: bool = False,
    imp: float = CALIBRATED_IMP,
    priority0_cap: int = 20,
    gt_json: str | None = None,
    build_submission: bool = False,
    submission_out: str | None = None,
    articles_csv: str = "dev_articles.csv",
) -> list[dict[str, Any]]:
    """
    Full sweep with structured output.
    gt_verified: study_id -> confirmed GT r (from GROUND_TRUTH_LOG).
    """
    override_ids = override_ids if override_ids is not None else default_override_ids()
    gt_verified = gt_verified if gt_verified is not None else dict(GT_VERIFIED_DEFAULT)
    gt_extra = load_gt_verified_from_json(gt_json)
    if gt_extra:
        gt_verified = {**gt_verified, **gt_extra}

    if not os.path.isfile(log_path):
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        return []

    with open(log_path, encoding="utf-8") as _lf:
        _log_rows = json.load(_lf)
    p0_pool = sum(
        1
        for s in _log_rows
        if s.get("aggregate_r") is None
        and (s.get("extraction_tier") or "") != "manual_override"
        and _scan_pdf_for_correlation_signal(os.path.join(pdf_dir, f"{s['study_id']}.pdf"))
    )

    targets = select_opus_targets(
        log_path, override_ids, cap=cap, pdf_dir=pdf_dir, priority0_cap=priority0_cap
    )
    print(f"Opus v10 sweep targets: {len(targets)} (cap={cap}, priority0_cap={priority0_cap})")
    print(f"  Priority 0 pool (signal + blank, pre-cap): {p0_pool}")
    for t in targets:
        print(
            f"  p{t['priority']} {t['study_id']}: {t['reason']} "
            f"current_r={t.get('current_r')} n_cand={t.get('n_candidates_found')}"
        )

    p0_selected = [t for t in targets if t.get("priority") == 0]
    print("\n=== PRIORITY 0: Papers with correlation signal but blank extraction ===")
    print(f"  N in this sweep (capped): {len(p0_selected)}")
    if dry_run:
        print("  (Opus recovered / still blank counts require a full run.)")

    if dry_run:
        print("Dry run - no API calls.")
        return []

    if not targets:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return []

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return []

    client = anthropic.Anthropic()
    results: list[dict[str, Any]] = []

    for i, target in enumerate(targets):
        sid = target["study_id"]
        pdf_path = os.path.join(pdf_dir, f"{sid}.pdf")
        print(
            f"[{i + 1}/{len(targets)}] {sid} ({target['reason']}) "
            f"current_r={target.get('current_r')}"
        )

        result = opus_extract_study(
            sid,
            pdf_path,
            max_pages=max_pages,
            client=client,
            deep_search=(target.get("priority") == 0),
        )
        result.update(target)
        result["correlation_signal_detected"] = _scan_pdf_for_correlation_signal(pdf_path)
        result["priority_reason"] = target.get("reason", "")
        result["pdf_exists"] = os.path.exists(pdf_path)

        current_r = target.get("current_r")
        opus_r = result.get("opus_r")
        gt_r = gt_verified.get(sid)
        reason_str = (target.get("reason") or "")
        is_suspicious_target = reason_str.startswith("suspicious_value_")

        result["gt_r"] = gt_r
        result["recommendation"] = "NO_CHANGE"
        result["mse_impact_estimate"] = 0.0

        if result.get("error"):
            result["recommendation"] = "ERROR"
            result["mse_impact_estimate"] = 0.0
        elif opus_r is None and current_r is None:
            result["recommendation"] = "NO_CHANGE"
        elif opus_r is not None and current_r is not None:
            if abs(float(opus_r) - float(current_r)) <= 0.05:
                if is_suspicious_target:
                    result["recommendation"] = "VERIFIED_CORRECT"
                    result["verification_note"] = (
                        "Opus agrees with pipeline within 0.05; suspicious-value triangulation cleared."
                    )
                    result["mse_impact_estimate"] = 0.0
                else:
                    result["recommendation"] = "NO_CHANGE"
            else:
                result["recommendation"] = "REVIEW_VALUE_CONFLICT"
                ref = gt_r if gt_r is not None else current_r
                result["mse_impact_estimate"] = abs(
                    (float(opus_r) - float(ref)) ** 2 - (float(current_r) - float(ref)) ** 2
                ) / 127.0
        elif opus_r is None and current_r is not None:
            result["recommendation"] = "REVIEW_OPUS_SAYS_BLANK"
            result["mse_impact_estimate"] = (float(current_r) - imp) ** 2 / 127.0
        elif opus_r is not None and current_r is None:
            result["recommendation"] = "REVIEW_OPUS_RECOVERED"
            result["mse_impact_estimate"] = (imp - float(opus_r)) ** 2 / 127.0

        conf = (result.get("opus_confidence") or "").lower()
        if (
            result["recommendation"] not in ("NO_CHANGE", "ERROR")
            and conf == "low"
        ):
            result["recommendation"] = "LOW_CONFIDENCE_" + result["recommendation"]

        results.append(result)
        print(
            f"  -> opus_r={opus_r} conf={result.get('opus_confidence')} "
            f"rec={result['recommendation']} mse_impact={result['mse_impact_estimate']:.6f}"
        )

        if i + 1 < len(targets):
            time.sleep(sleep_s)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("OPUS SWEEP RESULTS - Sorted by MSE Impact")
    print("=" * 70)

    actionable = [
        r
        for r in results
        if r.get("recommendation") not in ("NO_CHANGE", "ERROR", "VERIFIED_CORRECT")
        and "LOW_CONFIDENCE" not in (r.get("recommendation") or "")
    ]
    actionable.sort(key=lambda x: -float(x.get("mse_impact_estimate") or 0))

    print(f"\nHigh-confidence actionable items ({len(actionable)}):")
    for r in actionable:
        print(f"\n  {r['study_id']}: {r['recommendation']}")
        print(f"    Pipeline: {r.get('current_r')}  Opus: {r.get('opus_r')}  GT: {r.get('gt_r')}")
        print(f"    MSE impact estimate: {r.get('mse_impact_estimate', 0):.6f}")
        print(f"    Trust: {r.get('opus_trust')} (eligible={r.get('opus_trust_eligible')})")
        print(f"    SWB: {r.get('opus_swb')} (eligible={r.get('opus_swb_eligible')})")
        print(f"    Reasoning: {r.get('opus_reasoning')}")

    low_conf = [r for r in results if "LOW_CONFIDENCE" in (r.get("recommendation") or "")]
    print(f"\nLow-confidence items (verify manually before using): {len(low_conf)}")
    for r in low_conf:
        oreas = (r.get("opus_reasoning") or "")[:120]
        print(f"  {r['study_id']}: opus_r={r.get('opus_r')} - {oreas}")

    total_potential = sum(float(r.get("mse_impact_estimate") or 0) for r in actionable)
    print(
        f"\nTotal potential MSE improvement if all high-confidence items correct: "
        f"{total_potential:.6f}"
    )

    p0_done = [r for r in results if r.get("priority") == 0]
    n_p0_rec = sum(1 for r in p0_done if r.get("opus_r") is not None and not r.get("error"))
    n_p0_blank = sum(
        1 for r in p0_done if r.get("opus_r") is None and not r.get("error")
    )
    print("\n=== PRIORITY 0: Papers with correlation signal but blank extraction ===")
    print(f"  N targeted: {len(p0_done)}")
    print(f"  Opus recovered: {n_p0_rec}")
    print(
        f"  Still blank after Opus: {n_p0_blank} "
        f"(strongest remaining blank candidates for author contact)"
    )

    if build_submission and results:
        _sout = submission_out or "submission_opus_built.csv"
        build_submission_from_opus_sweep(
            output_path,
            log_path,
            _sout,
            articles_csv=articles_csv,
            imp=imp,
            gt_verified=gt_verified,
        )

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Opus sweep v10 — comprehensive target selection")
    p.add_argument("--log", default=DEFAULT_LOG_PATH, help="Pipeline batch JSON log path")
    p.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    p.add_argument("--out", default=DEFAULT_OUT_JSON)
    p.add_argument("--cap", type=int, default=60, help="Max studies (default 60)")
    p.add_argument("--sleep", type=float, default=3.0)
    p.add_argument("--max-pages", type=int, default=25)
    p.add_argument("--imp", type=float, default=CALIBRATED_IMP, help="Imputation constant for MSE heuristic")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--priority0-cap",
        type=int,
        default=20,
        help="Max Priority-0 (correlation-signal blank) slots within --cap",
    )
    p.add_argument("--gt-json", default=None, help="Optional JSON object of study_id -> GT r to merge")
    p.add_argument(
        "--build-submission",
        action="store_true",
        help="After sweep, write CSV from log + high-confidence Opus deltas",
    )
    p.add_argument(
        "--submission-out",
        default="submission_opus_built.csv",
        help="Output CSV path when --build-submission is set",
    )
    p.add_argument("--articles-csv", default="dev_articles.csv", help="Row order for submission CSV")
    args = p.parse_args()

    run_opus_sweep_v10(
        log_path=args.log,
        pdf_dir=args.pdf_dir,
        output_path=args.out,
        cap=args.cap,
        sleep_s=args.sleep,
        max_pages=args.max_pages,
        dry_run=args.dry_run,
        imp=args.imp,
        priority0_cap=args.priority0_cap,
        gt_json=args.gt_json,
        build_submission=args.build_submission,
        submission_out=args.submission_out,
        articles_csv=args.articles_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
