import random

from benchmarks.run_benchmarks import run_reference_benchmarks
from benchmarks.validate_requirements import evaluate_requirements
from fheml.activations import ChebyshevApproximator
from fheml.graph import DepthEstimator, GraphNode
from fheml.runtime import FHECompiler
from fheml.validation import AccuracyValidator


def test_depth_estimation_with_activation_poly_depth():
    g = [
        GraphNode("x", "input"),
        GraphNode("m1", "matmul", ["x"]),
        GraphNode("a1", "activation", ["m1"], metadata={"mult_depth": 3}),
        GraphNode("m2", "matmul", ["a1"]),
    ]
    d = DepthEstimator().estimate(g)
    assert d == 5


def test_compiler_selects_secure_params():
    approx = ChebyshevApproximator()
    spec = approx.approximate("sigmoid", 5, (-4, 4))
    graph = [
        GraphNode("x", "input"),
        GraphNode("m1", "matmul", ["x"]),
        GraphNode("a1", "activation", ["m1"]),
        GraphNode("m2", "matmul", ["a1"]),
    ]

    plan = FHECompiler().compile(graph, {"a1": spec})
    assert plan.parameters.security_bits == 128
    assert plan.parameters.poly_modulus_degree >= 4096


def test_validation_threshold():
    random.seed(42)
    plain = [[random.uniform(-1, 1) for _ in range(10)] for _ in range(64)]
    fhe = [[v + random.uniform(-1e-5, 1e-5) for v in row] for row in plain]
    labels = [random.randint(0, 9) for _ in range(64)]

    report = AccuracyValidator().compare(plain, fhe, labels)
    assert report.relative_error_percent < 0.1
    assert report.passed


def test_benchmark_speedup_target():
    results = run_reference_benchmarks()
    assert len(results) >= 3
    assert min(r.speedup for r in results) >= 5.0


def test_requirement_report_passes():
    report = evaluate_requirements()
    assert report.models_without_bootstrapping >= 2
    assert report.max_top5_delta_percent < 0.1
    assert report.min_speedup >= 5.0
    assert report.security_bits >= 128
    assert report.passed
