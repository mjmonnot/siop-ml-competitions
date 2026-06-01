"""CLI for running a task end-to-end.

Examples:

    # Single row from dev set (good for prompt iteration):
    python -m src.run --task empathy --row-id 95 --split dev

    # Full task on test set (real submission):
    python -m src.run --task empathy --split test --output submissions/empathy_test.csv

    # Akben-style self-consistency on top of the default:
    python -m src.run --task empathy --split test --self-consistency 5 --output ...

    # Switch model to match what the 2024 winners used:
    python -m src.run --task fairness --split test --model gpt-4-0125-preview ...

    # Merge per-task submissions into the final combined CSV:
    python -m src.run --merge submissions/*.csv --output submissions/final.csv

Inputs:
- data/dev.csv and data/test.csv: label files (shipped with this repo).
- data/{task}_train.csv, data/{task}_dev_inputs.csv, data/{task}_test_inputs.csv:
  the actual text inputs for each task. These are NOT shipped with this
  repo — see data/README.md for how to obtain them. If they're missing,
  the CLI exits with a clear message rather than failing partway.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .examples import pick_random, pick_similar
from .harness import CallSpec, Harness, mean_reducer, mode_reducer
from .scoring import accuracy, avg_cosine_similarity, composite, pearson_r


DATA_DIR = Path(os.environ.get("SIOP_DATA_DIR", "data"))


def load_csv(path: Path) -> list[dict]:
    """Load a CSV into a list of dicts. Returns [] if the file does not
    exist (the caller is expected to handle missing inputs gracefully).

    Uses utf-8-sig to transparently strip the BOM that the EvalAI portal
    exports with — Excel adds one if you ever round-trip the file
    through a spreadsheet app, which is easy to do by accident.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_task_inputs(task: str, split: str) -> list[dict]:
    """Load the input text for a task and split.

    Filenames follow the convention used by the original PAID Team
    notebooks and the Hungry Llama documentation: {task}_train.csv,
    {task}_dev_inputs.csv (their "val_public"), {task}_test_inputs.csv
    (their "test_public"). See data/README.md.
    """
    p = DATA_DIR / f"{task}_{split}_inputs.csv"
    if not p.exists():
        p = DATA_DIR / f"{task}_{'val_public' if split == 'dev' else 'test_public'}.csv"
    return load_csv(p)


def _load_task_train(task: str) -> list[dict]:
    p = DATA_DIR / f"{task}_train.csv"
    return load_csv(p)


def _load_labels(split: str) -> dict[str, dict[str, str]]:
    """Load the label file (dev.csv or test.csv) into a nested dict:
    labels[task][id] = ground-truth string.
    """
    rows = load_csv(DATA_DIR / f"{split}.csv")
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        out.setdefault(r["benchmark"], {})[str(r["_id"])] = r["output"]
    return out


def run_task(
    task: str,
    split: str,
    *,
    model: str,
    self_consistency: int,
    output_path: Path | None,
    row_id: str | None,
    similarity_examples: bool,
) -> dict[str, Any]:
    """Run a single task end-to-end. Returns a dict with predictions
    and (if labels available) the score.
    """
    inputs = _load_task_inputs(task, split)
    if not inputs:
        return {
            "status": "no_inputs",
            "message": (
                f"No input file found for {task}/{split}. "
                f"Expected data/{task}_{split}_inputs.csv "
                f"or the original {task}_{'val_public' if split == 'dev' else 'test_public'}.csv. "
                "See data/README.md."
            ),
        }
    train = _load_task_train(task) if task != "interview" else []
    labels_by_task = _load_labels(split)
    labels = labels_by_task.get(task, {})

    if row_id is not None:
        inputs = [r for r in inputs if str(r.get("_id", r.get("id"))) == str(row_id)]
        if not inputs:
            return {"status": "row_not_found", "row_id": row_id}

    adapter = get_adapter(task)
    harness = Harness()

    predictions: list[Any] = []
    raw_outputs: list[str] = []
    ids: list[str] = []

    for row in inputs:
        rid = str(row.get("_id", row.get("id", "")))
        ids.append(rid)

        # Few-shot example selection.
        if task == "interview" or not train:
            examples: list[dict] = []
        elif similarity_examples and task in ("empathy", "fairness"):
            # Empathy uses 'text' as the input field; fairness uses both
            # 'first_option' and 'second_option' — for similarity we
            # concatenate them.
            if task == "empathy":
                target = row["text"]
            else:
                target = f"{row['first_option']} || {row['second_option']}"
                # Also augment train rows for the similarity model:
                for tr in train:
                    tr.setdefault("text", f"{tr['first_option']} || {tr['second_option']}")
            chosen, _ = pick_similar(train, target, k=adapter.k_examples)
            examples = chosen
        else:
            examples = pick_random(train, k=adapter.k_examples)

        messages = adapter.build_messages(row, examples)
        spec = CallSpec(
            messages=messages,
            model=model,
            response_format=adapter.response_format(),
            temperature=0.7 if self_consistency > 1 else 0.0,
        )

        if self_consistency > 1:
            reducer = mean_reducer if task == "clarity" else mode_reducer
            text = harness.call_consistent(spec, n=self_consistency, reduce=reducer)
        else:
            text = harness.call(spec)

        raw_outputs.append(text)
        predictions.append(adapter.parse(text, row))

    result: dict[str, Any] = {
        "status": "ok",
        "task": task,
        "split": split,
        "n": len(inputs),
        "ids": ids,
        "predictions": predictions,
        "raw_outputs": raw_outputs,
    }

    # Score if labels are available.
    if labels:
        truths_aligned = [labels.get(rid, "") for rid in ids]
        if task == "empathy":
            preds_str = [str(p) for p in predictions]
            result["score"] = accuracy(preds_str, truths_aligned)
            result["metric"] = "accuracy"
        elif task == "fairness":
            result["score"] = accuracy(predictions, truths_aligned)
            result["metric"] = "accuracy"
        elif task == "clarity":
            try:
                truths_f = [float(t) for t in truths_aligned if t]
                preds_f = [float(p) for p, t in zip(predictions, truths_aligned) if t]
                result["score"] = pearson_r(preds_f, truths_f)
                result["metric"] = "pearson_r"
            except ValueError:
                result["score"] = None
                result["metric"] = "pearson_r (failed: non-numeric truths)"
        elif task == "interview":
            result["score"] = avg_cosine_similarity(predictions, truths_aligned)
            result["metric"] = "avg_cosine_similarity"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["benchmark", "_id", "output"])
            for rid, p in zip(ids, predictions):
                w.writerow([task, rid, p])

    return result


def merge_submissions(input_paths: list[Path], output: Path) -> None:
    """Concatenate per-task submission CSVs into the merged format
    expected by the EvalAI portal (`benchmark,_id,output`).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "_id", "output"])
        for p in input_paths:
            with p.open(encoding="utf-8") as g:
                r = csv.DictReader(g)
                for row in r:
                    w.writerow([row["benchmark"], row["_id"], row["output"]])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--task", choices=["empathy", "interview", "clarity", "fairness"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--row-id", default=None, help="Run only this single row id.")
    parser.add_argument("--model", default="gpt-4o-2024-08-06")
    parser.add_argument(
        "--self-consistency",
        type=int,
        default=1,
        help="N independent calls reduced by mode/mean. Akben used 5.",
    )
    parser.add_argument(
        "--similarity-examples",
        action="store_true",
        help="Pick few-shot examples by cosine similarity instead of random.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="Merge per-task submissions into one CSV.")
    args = parser.parse_args(argv)

    if args.merge:
        if not args.output:
            print("--merge requires --output", file=sys.stderr)
            return 2
        merge_submissions(args.merge, args.output)
        print(f"merged {len(args.merge)} files into {args.output}")
        return 0

    if not args.task:
        parser.error("--task is required unless --merge is specified")

    result = run_task(
        args.task,
        args.split,
        model=args.model,
        self_consistency=args.self_consistency,
        output_path=args.output,
        row_id=args.row_id,
        similarity_examples=args.similarity_examples,
    )
    if result["status"] != "ok":
        print(result.get("message", result["status"]), file=sys.stderr)
        return 1
    score = result.get("score")
    metric = result.get("metric", "")
    if score is not None:
        print(f"{args.task}/{args.split}: n={result['n']}  {metric}={score:.4f}")
    else:
        print(f"{args.task}/{args.split}: n={result['n']} (no score — labels missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
