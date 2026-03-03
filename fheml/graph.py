from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class GraphNode:
    """Egyszerű számítási gráf csomópont.

    op támogatás: input, add, mul, matmul, conv, square, poly_eval, activation
    """

    name: str
    op: str
    inputs: List[str] = field(default_factory=list)
    metadata: Dict[str, int | float | str] = field(default_factory=dict)


class DepthEstimator:
    """Statikus CKKS multiplikációs mélységbecslő."""

    MULTIPLICATIVE_OPS = {"mul", "matmul", "conv", "square"}

    def estimate(self, nodes: Iterable[GraphNode]) -> int:
        by_name = {node.name: node for node in nodes}
        memo: Dict[str, int] = {}

        def visit(node_name: str) -> int:
            if node_name in memo:
                return memo[node_name]
            node = by_name[node_name]
            if not node.inputs:
                depth = 0
            else:
                parent_depth = max(visit(inp) for inp in node.inputs)
                depth = parent_depth

            if node.op in self.MULTIPLICATIVE_OPS:
                depth += 1
            elif node.op == "poly_eval":
                # metadata.degree = polynomial fokszám
                degree = int(node.metadata.get("degree", 1))
                depth += max(0, (degree - 1).bit_length())
            elif node.op == "activation":
                # metadata.mult_depth: előre kiszámolt közelítés mélység
                depth += int(node.metadata.get("mult_depth", 0))

            memo[node_name] = depth
            return depth

        return max(visit(name) for name in by_name)
