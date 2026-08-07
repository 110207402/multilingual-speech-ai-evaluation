#!/usr/bin/env python3
"""Evaluate the repository's synthetic JSONL example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_eval import character_error_rate, pii_scores, role_accuracy


def evaluate(path: Path) -> dict[str, float]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    mean_cer = sum(character_error_rate(row["reference"], row["hypothesis"]) for row in records) / len(records)
    pii = pii_scores(
        (item for row in records for item in row["reference_pii"]),
        (item for row in records for item in row["predicted_pii"]),
    )
    roles = role_accuracy(
        [row["reference_role"] for row in records],
        [row["predicted_role"] for row in records],
    )
    return {"cer": mean_cer, "pii_precision": pii["precision"], "pii_recall": pii["recall"], "role_accuracy": roles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Synthetic JSONL predictions")
    args = parser.parse_args()
    for metric, value in evaluate(args.input).items():
        print(f"{metric}: {value:.3%}")


if __name__ == "__main__":
    main()
