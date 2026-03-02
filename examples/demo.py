from fheml.activations import ChebyshevApproximator
from fheml.graph import GraphNode
from fheml.runtime import FHECompiler


if __name__ == "__main__":
    approx = ChebyshevApproximator()
    relu_poly = approx.approximate("relu", degree=7, interval=(-3.0, 3.0))

    graph = [
        GraphNode("x", "input"),
        GraphNode("dense1", "matmul", ["x"]),
        GraphNode("act1", "activation", ["dense1"]),
        GraphNode("dense2", "matmul", ["act1"]),
        GraphNode("y", "add", ["dense2"]),
    ]

    compiler = FHECompiler()
    plan = compiler.compile(graph, activation_specs={"act1": relu_poly})

    print("Estimated multiplicative depth:", plan.depth)
    print("poly_modulus_degree (N):", plan.parameters.poly_modulus_degree)
    print("coeff_modulus_bits:", plan.parameters.coeff_modulus_bits)
