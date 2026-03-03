from __future__ import annotations

import importlib
from dataclasses import dataclass
from statistics import mean
from typing import Optional

from benchmarks.run_benchmarks import run_reference_benchmarks


@dataclass
class BackendResult:
    name: str
    available: bool
    latency_ms: Optional[float]
    note: str


def _probe_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def probe_external_backends() -> list[BackendResult]:
    """Probe requested external HE stacks.

    This only reports availability in the current environment.
    If available, latency collection hooks can be added here.
    """

    candidates = [
        ("Microsoft EVA/CHET", "eva"),
        ("Intel nGraph-HE", "ngraph"),
        ("Concrete ML", "concrete.ml"),
        ("TenSEAL", "tenseal"),
    ]

    results: list[BackendResult] = []
    for name, module_name in candidates:
        available = _probe_module(module_name)
        if available:
            results.append(
                BackendResult(
                    name=name,
                    available=True,
                    latency_ms=None,
                    note="Module found, but no backend-specific model runner is wired yet.",
                )
            )
        else:
            results.append(
                BackendResult(
                    name=name,
                    available=False,
                    latency_ms=None,
                    note=f"Python module '{module_name}' is not installed in this environment.",
                )
            )
    return results


def internal_mean_latency_ms() -> float:
    rows = run_reference_benchmarks()
    return mean(r.latency_bootstrap_free_ms for r in rows)


def compare_against_external() -> list[str]:
    internal_ms = internal_mean_latency_ms()
    lines = [f"internal_bootstrap_free_mean_ms,{internal_ms:.3f}"]
    for result in probe_external_backends():
        if result.latency_ms is None:
            lines.append(f"{result.name},unavailable_or_not_measured,,{result.note}")
        else:
            speedup = result.latency_ms / internal_ms
            lines.append(f"{result.name},measured,{speedup:.3f}x,")
    return lines


if __name__ == "__main__":
    for line in compare_against_external():
        print(line)
