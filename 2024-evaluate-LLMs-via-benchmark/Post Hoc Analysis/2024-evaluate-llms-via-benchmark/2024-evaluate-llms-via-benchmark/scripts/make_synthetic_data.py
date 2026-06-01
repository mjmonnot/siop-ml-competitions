"""Generate synthetic input CSVs for the SIOP 2024 ML Competition tasks.

The repo ships with the public label files (data/dev.csv, data/test.csv) but
the actual input text files (the feedback emails, interview Q/R pairs,
personality items, policy pairs) were distributed via the EvalAI portal and
are not publicly archived. This script synthesizes plausible inputs that
correlate with the ground-truth labels, so the pipeline can be exercised
end-to-end against the official label files.

What this is good for:

  - Validating that the harness actually runs: prompt construction, API
    calls, parsing, scoring, submission file writing.
  - Getting a *qualitative* sense of how the pipeline behaves: does the
    empathy classifier produce reasonable probabilities? Does the interview
    generator produce plausibly-styled responses?
  - Confirming the cost estimate before paying for the real run.

What this is NOT good for:

  - Apples-to-apples comparison against the 2024 winners. The whole point
    of the comparison is to run against the SAME inputs they ran against;
    we don't have those inputs. The numbers from a synthetic-data run are
    pipeline-behavior numbers, not leaderboard numbers.
  - Drawing any conclusion about the difficulty of the 2024 benchmark.
    Synthetic empathy emails are clearer-cut than real ones (the real
    benchmark had substantial human-rater disagreement; the synthetic data
    has no such noise).

The seeding strategy: synthetic inputs are generated with a fixed random
seed and explicitly correlated to the ground-truth label in dev.csv /
test.csv. An empathy email labeled 1 (empathetic) is generated from an
"empathetic" template; one labeled 0 is generated from a "cold" template.
This means a well-functioning classifier should score very high (probably
>.90) on synthetic data — much higher than the .58-.61 the real winners
got. That gap is real label noise in the original benchmark.

Output files written to data/:

  - empathy_train.csv:         _id, text, empathy
  - empathy_dev_inputs.csv:    _id, text
  - empathy_test_inputs.csv:   _id, text
  - interview_train.csv:       _id, Q1..Q5, R1..R5
  - interview_dev_inputs.csv:  _id, Q1..Q5, R1..R4
  - interview_test_inputs.csv: _id, Q1..Q5, R1..R4
  - clarity_train.csv:         _id, item, mean_clarity
  - clarity_dev_inputs.csv:    _id, item
  - clarity_test_inputs.csv:   _id, item
  - fairness_train.csv:        _id, first_option, second_option, majority_vote
  - fairness_dev_inputs.csv:   _id, first_option, second_option
  - fairness_test_inputs.csv:  _id, first_option, second_option

After generation, sanity check by joining inputs to labels:

  python -c "
  import pandas as pd
  l = pd.read_csv('data/test.csv', encoding='utf-8-sig')
  i = pd.read_csv('data/empathy_test_inputs.csv', encoding='utf-8-sig')
  assert set(l[l.benchmark=='empathy']._id.astype(str)) == set(i._id.astype(str))
  "
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd


SEED = 20240417  # SIOP 2024 conference start date


# -----------------------------------------------------------------------------
# Empathy: feedback emails to "Jonathan" about the Beta project
# -----------------------------------------------------------------------------

EMPATHIC_OPENINGS = [
    "Hi Jonathan, I hope this finds you well.",
    "Hey Jonathan, thanks for sharing your work with us.",
    "Jonathan, I appreciate you sending over the latest draft.",
    "Hi Jonathan, hope the Beta project is going as smoothly as it can be.",
]
EMPATHIC_BODIES = [
    "I noticed Terry flagged some concerns about the reports — the team wants them more concise and business-focused. I know how much work you've put into them, so I'd love to grab coffee this week and talk through what could help.",
    "I can see how much care you've been putting into this project, and I want to support you through some of the rougher patches. The current draft has real strengths; let me know what would be most useful from me.",
    "I want to acknowledge that the recent feedback might have felt sharp. I think your work has real value, and I'd be happy to review reports with you before you send them to Terry if that would help.",
    "I know things have been stressful with the Beta timeline. Your contributions matter, and I'm here to help you work through Terry's feedback in whatever way is most useful.",
]
EMPATHIC_CLOSINGS = [
    "Please let me know how I can help. We'll figure this out together.",
    "Take care of yourself, and reach out whenever you need a second pair of eyes.",
    "Looking forward to talking more soon — you've got this.",
    "I'm rooting for you on this one. Let's connect tomorrow.",
]

COLD_OPENINGS = [
    "Jonathan,",
    "Jonathan, see below.",
    "Jonathan, this needs to change.",
    "Jonathan, the reports are not where they need to be.",
]
COLD_BODIES = [
    "Terry's feedback on the reports is clear: too long, not business-focused, structurally weak. Fix this by EOW.",
    "The reports lack focus. You need to cut the volume in half and lead with the business takeaways. This is not the first time I've raised it.",
    "Your latest draft does not meet the bar. The reports need to be tight, factual, and oriented to the decisions the team has to make. Rework them.",
    "Performance on the Beta project needs to improve. Specifically, the reports are missing structure and a clear business angle. Please address.",
]
COLD_CLOSINGS = [
    "Send me a revised version by Friday.",
    "I need to see significant improvement immediately.",
    "Confirm receipt.",
    "Get this corrected.",
]


def _empathy_text(label: int, rng: random.Random) -> str:
    if label == 1:
        return " ".join([
            rng.choice(EMPATHIC_OPENINGS),
            rng.choice(EMPATHIC_BODIES),
            rng.choice(EMPATHIC_CLOSINGS),
        ])
    else:
        return " ".join([
            rng.choice(COLD_OPENINGS),
            rng.choice(COLD_BODIES),
            rng.choice(COLD_CLOSINGS),
        ])


# -----------------------------------------------------------------------------
# Interview: structured behavioral interview Q&A
# -----------------------------------------------------------------------------

INTERVIEW_QUESTIONS = [
    "Tell me about a time you had to work through a disagreement with a coworker.",
    "Describe a project where you went above and beyond what was expected.",
    "Tell me about a time you received critical feedback and what you did with it.",
    "Walk me through a situation where you had to learn something new under time pressure.",
    "Tell me about a time you helped a colleague or teammate succeed.",
]

INTERVIEW_RESPONSE_OPENERS = [
    "There was a time",
    "I remember a project",
    "One example that comes to mind",
    "During my time at",
    "A situation I dealt with",
]
INTERVIEW_RESPONSE_FILLERS = [
    "I worked closely with my team to figure out the best approach",
    "I had to step back and really think about what mattered most",
    "we ended up dividing the work and checking in daily",
    "the key was being patient and asking the right questions",
    "I leaned on what I'd learned in school and from previous roles",
    "I tried to focus on the outcome we needed rather than the friction",
]
INTERVIEW_RESPONSE_CLOSERS = [
    "and in the end the project turned out well.",
    "and I learned a lot from the experience.",
    "and it shaped how I approach similar problems now.",
    "and we hit the deadline with time to spare.",
    "and I think the team came out stronger because of it.",
]


def _synthesize_r1_to_r4(rng: random.Random) -> list[str]:
    """Generate four prior responses with reasonable length variation."""
    out = []
    for _ in range(4):
        n_sentences = rng.randint(2, 4)
        sentences = []
        sentences.append(
            f"{rng.choice(INTERVIEW_RESPONSE_OPENERS)} {rng.choice(INTERVIEW_RESPONSE_FILLERS)}."
        )
        for _ in range(n_sentences - 2):
            sentences.append(rng.choice(INTERVIEW_RESPONSE_FILLERS).capitalize() + ".")
        sentences.append(rng.choice(INTERVIEW_RESPONSE_CLOSERS).capitalize())
        out.append(" ".join(sentences))
    return out


# -----------------------------------------------------------------------------
# Clarity: personality test items, mean rating 1-7
# -----------------------------------------------------------------------------

# Clear items (high clarity) read concretely; unclear items have negations,
# compound clauses, or vague constructions. The label correlates with this
# structure so a well-prompted model can recover the rating.

CLEAR_ITEMS = [
    "I enjoy meeting new people.",
    "I get angry easily.",
    "I am usually relaxed.",
    "I worry about things.",
    "I work hard.",
    "I follow a schedule.",
    "I like to read.",
    "I like to help others.",
    "I enjoy art and music.",
    "I keep my room tidy.",
    "I am quick to understand things.",
    "I am the life of the party.",
    "I feel comfortable around people.",
    "I am full of ideas.",
]
MIXED_CLARITY_ITEMS = [
    "I rarely feel discouraged when things go wrong.",
    "I am not particularly interested in others' problems.",
    "I sometimes find it difficult to articulate my ideas at meetings.",
    "I try not to be the center of attention in social situations.",
    "I would not describe myself as a sentimental person.",
    "I avoid getting drawn into discussions of feelings.",
    "I tend to start projects without finishing the previous ones.",
    "I find it hard to get going on tasks I don't enjoy.",
]
UNCLEAR_ITEMS = [
    "I am not infrequently dissatisfied with my own performance.",
    "I do not usually fail to consider the consequences before acting.",
    "It is not the case that I avoid social gatherings.",
    "I am ambivalent about whether I would prefer order to spontaneity.",
    "I would not say I am unwilling to make decisions without consultation.",
    "I rarely am not motivated by external rewards.",
    "I am someone who, in most circumstances, would not necessarily decline an opportunity to lead.",
    "It would be inaccurate to describe me as one who never seeks novel experiences.",
]


def _clarity_item(label: float, rng: random.Random) -> str:
    """Sample an item from the bucket matching the clarity rating."""
    if label >= 6.0:
        return rng.choice(CLEAR_ITEMS)
    elif label >= 4.5:
        return rng.choice(MIXED_CLARITY_ITEMS)
    else:
        return rng.choice(UNCLEAR_ITEMS)


# -----------------------------------------------------------------------------
# Fairness: paired organizational policies
# -----------------------------------------------------------------------------

# Each pair has a "more supportive / employee-centered" option and a "more
# restrictive / company-centered" option. The label "first" or "second"
# indicates which received the majority vote. We synthesize so the more
# supportive option correlates with the majority vote — that's the strongest
# real-world signal in the 2024 fairness data per the deck synthesis.

POLICY_PAIRS = [
    # (supportive, restrictive)
    (
        "Conflict Resolution Workshops: We offer monthly workshops where employees learn active listening, dispute resolution, and de-escalation skills. Trained facilitators guide small groups through real workplace scenarios.",
        "Conflict Resolution Workbooks: Resources are made available to help employees self-resolve conflicts on their own time using a structured reflection workbook and an escalation hotline.",
    ),
    (
        "Flexible Remote Work: Employees may work remotely up to three days per week, with the team and manager jointly setting the schedule based on the work that needs to be done.",
        "Office Attendance Standard: Employees are expected to be in the office five days per week. Remote work is permitted only under documented exceptional circumstances.",
    ),
    (
        "Continuous Feedback: Managers provide informal feedback throughout the year, with two structured reviews. Employees can also request peer feedback at any time through the system.",
        "Annual Performance Review: A single formal review is conducted each year by the direct manager, with peer input gathered at the manager's discretion.",
    ),
    (
        "Internal Transfer Program: Employees who have been in a role for at least one year may request a transfer to another department, with hiring managers required to interview them.",
        "External Hiring Priority: Most open roles are posted externally first. Internal candidates may apply but receive no formal preference in the selection process.",
    ),
    (
        "Parental Leave: All new parents receive 16 weeks of paid leave regardless of role, with phased return-to-work support over the following three months.",
        "Standard Parental Leave: Birth parents receive 8 weeks of paid leave; non-birth parents receive 2 weeks. Return-to-work is at full capacity.",
    ),
    (
        "Mental Health Days: Employees may take up to 8 mental health days per year separate from PTO, with no documentation required and no manager approval needed.",
        "Sick Leave Policy: Mental health concerns fall under the standard sick leave policy. A doctor's note is required for absences exceeding two consecutive days.",
    ),
    (
        "Pay Transparency: All salary bands are published internally, and individual employees can request anonymized data on peer compensation within their level.",
        "Confidential Compensation: Salary information is confidential and may not be discussed among employees. Pay bands are not published.",
    ),
    (
        "Mentorship Matching: New hires are matched with a senior mentor within their first two weeks. Mentorship pairs meet biweekly and have a structured 12-month curriculum.",
        "Self-Directed Onboarding: New hires complete a self-paced onboarding portal. Mentorship is available on request but not assigned.",
    ),
    (
        "Tuition Reimbursement: Employees may apply for up to \\$10,000 per year in tuition reimbursement for any course relevant to their career growth, including adjacent fields.",
        "Job-Specific Training: The company provides training only for courses directly tied to current job duties, with a maximum reimbursement of \\$2,000 per year.",
    ),
    (
        "Open Promotion Cycle: Any employee may nominate themselves for promotion during the annual cycle. Decisions are made by a cross-functional committee with documented criteria.",
        "Manager-Initiated Promotion: Promotions are initiated by direct managers based on their assessment. Self-nomination is not part of the process.",
    ),
]


def _fairness_pair(label: str, rng: random.Random, used: set[int]) -> tuple[str, str]:
    """Pick a policy pair and arrange it so the more-supportive option lands
    in the position that matches the majority vote label.
    """
    # Cycle through pairs to avoid repetition; once exhausted, reuse with
    # different orderings.
    available = [i for i in range(len(POLICY_PAIRS)) if i not in used]
    if not available:
        used.clear()
        available = list(range(len(POLICY_PAIRS)))
    idx = rng.choice(available)
    used.add(idx)
    supportive, restrictive = POLICY_PAIRS[idx]
    if label == "first":
        return supportive, restrictive
    else:
        return restrictive, supportive


# -----------------------------------------------------------------------------
# Train set generators (these need their own ids since the train labels are
# not in dev.csv / test.csv)
# -----------------------------------------------------------------------------

def _make_empathy_train(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        label = i % 2  # balanced
        rows.append({
            "_id": f"train_{i+1}",
            "text": _empathy_text(label, rng),
            "empathy": label,
        })
    rng.shuffle(rows)
    return rows


def _make_interview_train(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        # Each train row has a real R5 (generated alongside R1-R4)
        responses = _synthesize_r1_to_r4(rng) + [
            " ".join([
                rng.choice(INTERVIEW_RESPONSE_OPENERS),
                rng.choice(INTERVIEW_RESPONSE_FILLERS) + ".",
                rng.choice(INTERVIEW_RESPONSE_CLOSERS).capitalize(),
            ])
        ]
        row = {"_id": f"R_train{i+1:03d}"}
        for j, q in enumerate(INTERVIEW_QUESTIONS, start=1):
            row[f"Q{j}"] = q
        for j, r in enumerate(responses, start=1):
            row[f"R{j}"] = r
        rows.append(row)
    return rows


def _make_clarity_train(n: int, rng: random.Random) -> list[dict]:
    rows = []
    for i in range(n):
        # Sample a target rating across the 1-7 range with a realistic bell
        rating = rng.gauss(5.4, 0.9)
        rating = max(1.5, min(6.8, rating))
        rows.append({
            "_id": f"train_{i+1}",
            "item": _clarity_item(rating, rng),
            "mean_clarity": round(rating, 3),
        })
    return rows


def _make_fairness_train(n: int, rng: random.Random) -> list[dict]:
    rows = []
    used: set[int] = set()
    for i in range(n):
        label = rng.choices(["first", "second"], weights=[0.57, 0.43])[0]
        first, second = _fairness_pair(label, rng, used)
        rows.append({
            "_id": f"train_{i+1}",
            "first_option": first,
            "second_option": second,
            "majority_vote": label,
        })
    return rows


# -----------------------------------------------------------------------------
# Input file generation: must use the exact _ids from dev.csv / test.csv
# -----------------------------------------------------------------------------

def _read_labels(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{split}.csv", encoding="utf-8-sig")


def _generate_inputs_for_split(
    data_dir: Path,
    split: str,
    rng_seed: int,
) -> dict[str, list[dict]]:
    """For each task, build input rows using ids and labels from {split}.csv."""
    rng = random.Random(rng_seed)
    labels = _read_labels(data_dir, split)

    out: dict[str, list[dict]] = {"empathy": [], "interview": [], "clarity": [], "fairness": []}
    used_fairness: set[int] = set()

    for _, row in labels.iterrows():
        bench = row["benchmark"]
        rid = str(row["_id"])
        label_value = row["output"]

        if bench == "empathy":
            label = int(label_value)
            out["empathy"].append({"_id": rid, "text": _empathy_text(label, rng)})

        elif bench == "interview":
            # The "label" here is the true R5; we synthesize R1-R4 to be
            # stylistically plausible context for that R5. We don't try to
            # match it closely — that would defeat the point of the task.
            responses = _synthesize_r1_to_r4(rng)
            irow: dict = {"_id": rid}
            for j, q in enumerate(INTERVIEW_QUESTIONS, start=1):
                irow[f"Q{j}"] = q
            for j, r in enumerate(responses, start=1):
                irow[f"R{j}"] = r
            out["interview"].append(irow)

        elif bench == "clarity":
            label = float(label_value)
            out["clarity"].append({"_id": rid, "item": _clarity_item(label, rng)})

        elif bench == "fairness":
            label = str(label_value)
            first, second = _fairness_pair(label, rng, used_fairness)
            out["fairness"].append({
                "_id": rid,
                "first_option": first,
                "second_option": second,
            })

    return out


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(data_dir: Path = Path("data"), n_train: int = 80) -> None:
    rng = random.Random(SEED)

    # Train files
    _write_csv(
        data_dir / "empathy_train.csv",
        _make_empathy_train(n_train * 2, rng),  # 160 train rows
        ["_id", "text", "empathy"],
    )
    _write_csv(
        data_dir / "interview_train.csv",
        _make_interview_train(n_train, rng),
        ["_id"] + [f"Q{i}" for i in range(1, 6)] + [f"R{i}" for i in range(1, 6)],
    )
    _write_csv(
        data_dir / "clarity_train.csv",
        _make_clarity_train(n_train * 2, rng),
        ["_id", "item", "mean_clarity"],
    )
    _write_csv(
        data_dir / "fairness_train.csv",
        _make_fairness_train(30, rng),  # competition train set was ~25
        ["_id", "first_option", "second_option", "majority_vote"],
    )

    # Dev and test input files (matched to label-file ids)
    for split in ("dev", "test"):
        inputs = _generate_inputs_for_split(data_dir, split, SEED + (0 if split == "dev" else 1))

        _write_csv(
            data_dir / f"empathy_{split}_inputs.csv",
            inputs["empathy"],
            ["_id", "text"],
        )
        _write_csv(
            data_dir / f"interview_{split}_inputs.csv",
            inputs["interview"],
            ["_id"] + [f"Q{i}" for i in range(1, 6)] + [f"R{i}" for i in range(1, 5)],
        )
        _write_csv(
            data_dir / f"clarity_{split}_inputs.csv",
            inputs["clarity"],
            ["_id", "item"],
        )
        _write_csv(
            data_dir / f"fairness_{split}_inputs.csv",
            inputs["fairness"],
            ["_id", "first_option", "second_option"],
        )

    print(f"Synthetic data written to {data_dir.resolve()}.")
    print()
    print("WARNING: These are SYNTHETIC inputs. Scores against them will not")
    print("be comparable to the 2024 winners' published numbers. See")
    print("docs/STATUS.md for the implications.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-train", type=int, default=80,
                        help="Training-set size per task (clarity and empathy get 2x).")
    args = parser.parse_args()
    main(data_dir=args.data_dir, n_train=args.n_train)
