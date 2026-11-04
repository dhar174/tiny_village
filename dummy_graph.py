class DummyGraph:
    def __init__(self):
        self.nodes = {"node_a": lambda x: x, "node_b": lambda x: x}
        self.edges = [("node_a", "node_b")]

graph = DummyGraph()
