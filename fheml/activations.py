from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, List


@dataclass
class ActivationSpec:
    name: str
    interval: tuple[float, float]
    degree: int
    coeffs: List[float]
    mult_depth: int


class ChebyshevApproximator:
    """ReLU/Sigmoid/GELU közelítés monom bázisban, Chebyshev mintavétellel."""

    def __init__(self, sample_points: int = 128):
        self.sample_points = sample_points

    def _fn(self, activation: str) -> Callable[[float], float]:
        if activation == "relu":
            return lambda x: max(0.0, x)
        if activation == "sigmoid":
            return lambda x: 1.0 / (1.0 + math.exp(-x))
        if activation == "gelu":
            return lambda x: 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
        raise ValueError(f"Ismeretlen aktiváció: {activation}")

    def approximate(self, activation: str, degree: int, interval: tuple[float, float]) -> ActivationSpec:
        f = self._fn(activation)
        lo, hi = interval
        xs = [
            0.5 * (lo + hi) + 0.5 * (hi - lo) * math.cos((2 * i + 1) * math.pi / (2 * self.sample_points))
            for i in range(self.sample_points)
        ]
        ys = [f(x) for x in xs]
        coeffs = self._least_squares_polyfit(xs, ys, degree)
        mult_depth = max(0, (degree - 1).bit_length())
        return ActivationSpec(activation, interval, degree, coeffs, mult_depth)

    def _least_squares_polyfit(self, xs: List[float], ys: List[float], degree: int) -> List[float]:
        n = degree + 1
        ata = [[0.0] * n for _ in range(n)]
        aty = [0.0] * n
        for x, y in zip(xs, ys):
            powers = [1.0]
            for _ in range(1, n):
                powers.append(powers[-1] * x)
            for i in range(n):
                aty[i] += powers[i] * y
                for j in range(n):
                    ata[i][j] += powers[i] * powers[j]
        return self._solve_linear(ata, aty)

    def _solve_linear(self, a: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)
        for i in range(n):
            pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
            div = a[i][i] if abs(a[i][i]) > 1e-15 else 1e-15
            for j in range(i, n):
                a[i][j] /= div
            b[i] /= div
            for r in range(n):
                if r == i:
                    continue
                factor = a[r][i]
                for c in range(i, n):
                    a[r][c] -= factor * a[i][c]
                b[r] -= factor * b[i]
        return b

    @staticmethod
    def evaluate(spec: ActivationSpec, x_values: List[float]) -> List[float]:
        out = []
        for x in x_values:
            s = 0.0
            p = 1.0
            for c in spec.coeffs:
                s += c * p
                p *= x
            out.append(s)
        return out
