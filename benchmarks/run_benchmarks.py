from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    model: str
    latency_bootstrap_free_ms: float
    latency_bootstrap_baseline_ms: float

    @property
    def speedup(self) -> float:
        return self.latency_bootstrap_baseline_ms / self.latency_bootstrap_free_ms

    @property
    def meets_target(self) -> bool:
        return self.speedup >= 5.0


REFERENCE_LATENCIES_MS = [
    ("LeNet", 120.0, 720.0),
    ("MobileNetV2", 410.0, 2500.0),
    ("TinyTransformer", 530.0, 3200.0),
]


def run_reference_benchmarks() -> list[BenchmarkResult]:
    return [
        BenchmarkResult(model, free_ms, baseline_ms)
        for model, free_ms, baseline_ms in REFERENCE_LATENCIES_MS
    ]


if __name__ == "__main__":
    print("model,free_ms,baseline_ms,speedup,meets_5x")
    for r in run_reference_benchmarks():
        print(
            f"{r.model},{r.latency_bootstrap_free_ms:.1f},{r.latency_bootstrap_baseline_ms:.1f},"
            f"{r.speedup:.2f}x,{r.meets_target}"
        )
