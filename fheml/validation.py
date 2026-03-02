from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ValidationReport:
    top5_accuracy_plain: float
    top5_accuracy_fhe: float
    relative_error_percent: float
    passed: bool


class AccuracyValidator:
    def topk_accuracy(self, logits: List[List[float]], labels: List[int], k: int = 5) -> float:
        hits = 0
        for row, label in zip(logits, labels):
            topk_idx = sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k]
            if label in topk_idx:
                hits += 1
        return hits / len(labels) if labels else 0.0

    def compare(self, plain_logits: List[List[float]], fhe_logits: List[List[float]], labels: List[int]) -> ValidationReport:
        p = self.topk_accuracy(plain_logits, labels, k=5)
        f = self.topk_accuracy(fhe_logits, labels, k=5)
        rel = 0.0 if p == 0 else abs(f - p) / p * 100
        return ValidationReport(p, f, rel, rel < 0.1)
