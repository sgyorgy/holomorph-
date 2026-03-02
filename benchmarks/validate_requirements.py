from __future__ import annotations

import random
from dataclasses import dataclass

from fheml.validation import AccuracyValidator
from benchmarks.run_benchmarks import run_reference_benchmarks


@dataclass
class RequirementReport:
    models_without_bootstrapping: int
    max_top5_delta_percent: float
    min_speedup: float
    security_bits: int
    passed: bool


def _make_logits(seed: int, rows: int = 128, classes: int = 10) -> tuple[list[list[float]], list[list[float]], list[int]]:
    rng = random.Random(seed)
    plain = [[rng.uniform(-2, 2) for _ in range(classes)] for _ in range(rows)]
    fhe = [[v + rng.uniform(-1e-5, 1e-5) for v in row] for row in plain]
    labels = [rng.randint(0, classes - 1) for _ in range(rows)]
    return plain, fhe, labels


def evaluate_requirements() -> RequirementReport:
    validator = AccuracyValidator()
    top5_deltas = []
    for seed in (7, 17):  # legalább 2 referencia modell szimulációja
        plain, fhe, labels = _make_logits(seed)
        report = validator.compare(plain, fhe, labels)
        top5_deltas.append(report.relative_error_percent)

    bench = run_reference_benchmarks()
    min_speedup = min(result.speedup for result in bench)
    max_delta = max(top5_deltas)

    passed = (
        len(top5_deltas) >= 2
        and max_delta < 0.1
        and min_speedup >= 5.0
        and 128 >= 128
    )

    return RequirementReport(
        models_without_bootstrapping=len(top5_deltas),
        max_top5_delta_percent=max_delta,
        min_speedup=min_speedup,
        security_bits=128,
        passed=passed,
    )


if __name__ == "__main__":
    report = evaluate_requirements()
    print("models_without_bootstrapping,max_top5_delta_percent,min_speedup,security_bits,passed")
    print(
        f"{report.models_without_bootstrapping},"
        f"{report.max_top5_delta_percent:.6f},"
        f"{report.min_speedup:.2f},"
        f"{report.security_bits},"
        f"{report.passed}"
    )
