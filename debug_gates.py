
import tensorcircuit as tc
import logging

class GateManager:
    def __init__(self, layers, rows, cols):
        self.layers = layers
        self.rows = rows
        self.cols = cols

    def get_bond_gate(self, bond_idx, layer_idx):
        if bond_idx < 0 or bond_idx >= self.cols - 1:
            return None
        layer = self.layers[layer_idx]
        for g in layer:
            cols = [x // self.rows for x in g["index"]]
            if min(cols) == bond_idx and max(cols) == bond_idx + 1:
                return g
        return None

def generate_sycamore_circuit(rows, cols, depth):
    layers = []
    l0 = []
    for i in range(rows * cols):
        l0.append({"gatef": tc.gates.h, "index": [i], "parameters": {}})
    layers.append(l0)

    import random
    random.seed(42)
    for _ in range(depth):
        l = []
        for i in range(rows * cols):
            l.append({"gatef": tc.gates.rz, "index": [i], "parameters": {"theta": random.random()}})
        for r in range(rows):
            for c_ in range(cols):
                idx = c_ * rows + r
                if c_ < cols - 1:
                    idx_next = (c_ + 1) * rows + r
                    l.append({"gatef": tc.gates.cz, "index": [idx, idx_next], "parameters": {}})
                if r < rows - 1:
                    idx_next = c_ * rows + (r + 1)
                    l.append({"gatef": tc.gates.cz, "index": [idx, idx_next], "parameters": {}})
        layers.append(l)
    return layers

def main():
    rows = 4
    cols = 4
    depth = 6
    layers = generate_sycamore_circuit(rows, cols, depth)
    gm = GateManager(layers, rows, cols)

    # Check Layer 0, Bond 0
    g = gm.get_bond_gate(0, 0)
    print(f"Layer 0, Bond 0: {g}")

    # Check Layer 1, Bond 0
    g = gm.get_bond_gate(0, 1)
    print(f"Layer 1, Bond 0: {g}")

if __name__ == "__main__":
    main()
