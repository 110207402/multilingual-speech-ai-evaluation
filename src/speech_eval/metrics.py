"""Dependency-free metrics used by the public evaluation example."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return the minimum insertion, deletion, and substitution count."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (ref_item != hyp_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate CER after removing whitespace from both strings."""
    ref_chars = list("".join(reference.split()))
    hyp_chars = list("".join(hypothesis.split()))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _levenshtein(ref_chars, hyp_chars) / len(ref_chars)


def pii_scores(reference: Iterable[str], prediction: Iterable[str]) -> dict[str, float]:
    """Calculate exact-match PII precision, recall, and F1."""
    ref = set(reference)
    pred = set(prediction)
    true_positive = len(ref & pred)
    precision = true_positive / len(pred) if pred else float(not ref)
    recall = true_positive / len(ref) if ref else float(not pred)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def diarization_error_rate(
    missed_speech_seconds: float,
    false_alarm_seconds: float,
    speaker_confusion_seconds: float,
    reference_speech_seconds: float,
) -> float:
    """Calculate DER from pre-aligned duration components."""
    if reference_speech_seconds <= 0:
        raise ValueError("reference_speech_seconds must be positive")
    error = missed_speech_seconds + false_alarm_seconds + speaker_confusion_seconds
    return error / reference_speech_seconds


def role_accuracy(reference: Sequence[str], prediction: Sequence[str]) -> float:
    """Calculate exact agent/customer role accuracy."""
    if len(reference) != len(prediction):
        raise ValueError("reference and prediction must have the same length")
    if not reference:
        return 1.0
    correct = sum(expected == actual for expected, actual in zip(reference, prediction))
    return correct / len(reference)
