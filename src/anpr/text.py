"""Text utilities kept dependency-free so they can be unit-tested in CI."""
import re


def normalize_plate(text: str) -> str:
    """Uppercase OCR output and remove spaces/punctuation."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def levenshtein_distance(a: str, b: str) -> int:
    a, b = normalize_plate(a), normalize_plate(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def character_accuracy(predicted: str, expected: str) -> float:
    """Normalized edit-distance character accuracy in [0, 1]."""
    expected_n = normalize_plate(expected)
    predicted_n = normalize_plate(predicted)
    denominator = max(len(expected_n), len(predicted_n), 1)
    return max(0.0, 1.0 - levenshtein_distance(predicted_n, expected_n) / denominator)
