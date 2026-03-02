from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .activations import ActivationSpec
from .graph import DepthEstimator, GraphNode
from .params import CKKSParameterSelector, CKKSParameters


@dataclass
class CompiledPlan:
    depth: int
    parameters: CKKSParameters
    activation_specs: dict[str, ActivationSpec]


class FHECompiler:
    """Bootstrapping-mentes CKKS compiler planning réteg."""

    def __init__(self, param_selector: CKKSParameterSelector | None = None):
        self.estimator = DepthEstimator()
        self.param_selector = param_selector or CKKSParameterSelector()

    def compile(self, graph: Iterable[GraphNode], activation_specs: dict[str, ActivationSpec] | None = None) -> CompiledPlan:
        activation_specs = activation_specs or {}
        normalized_graph = []
        for node in graph:
            if node.op == "activation" and node.name in activation_specs:
                node.metadata["mult_depth"] = activation_specs[node.name].mult_depth
            normalized_graph.append(node)

        depth = self.estimator.estimate(normalized_graph)
        params = self.param_selector.select(depth)
        return CompiledPlan(depth=depth, parameters=params, activation_specs=activation_specs)
