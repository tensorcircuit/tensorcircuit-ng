# Contraction Infrastructure & Hyperedges

1.  **Algebraic Contraction Path for Hyperedges**:
    - Legacy `tensornetwork` contraction loops (`tn.contract_between`) do not natively support hyperedges (shared edges between >2 tensors) without creating intermediate diagonal `CopyNode` tensors, which frequently leads to Out-Of-Memory (OOM) errors.
    - **Protocol**: For networks with hyperedges, use the "algebraic" contraction path (e.g., `use_primitives=True`). This maps the entire node collection to a list of bare tensors and a single einsum expression, allowing `cotengra` to optimize the contraction tree globally without materializing massive diagonal tensors.

2.  **Deterministic Topology for JIT Cache**:
    - Backends like JAX and optimizers like `cotengra` rely on consistent string/topology representations for caching.
    - **Protocol**: Before extracting einsum symbols or topology, **always sort the input nodes** (e.g., using `_stable_id_`). This ensures that structurally identical circuits always yield the same einsum string, maximizing path-search cache hits and JIT compilation reuse.

3.  **Node Consistency in Partial Contraction**:
    - **Pitfall**: An algebraic contraction result tensor is just a bare array. If the initial nodes were part of a larger network (partial contraction), simply wrapping the array in a new `tn.Node` breaks the connectivity of the remaining dangling edges.
    - **Protocol**: After an algebraic contraction, manually re-attach the subgraph's original dangling edges to the new `final_node`:
      1.  Iterate through the dangling edges in the deterministic order used by the topology extractor.
      2.  Update each `edge.node1/2` and `edge.axis1/2` to point to the new `final_node`.
      3.  Assign the list of edges to `final_node.edges`.
      4.  Finally, call `final_node.reorder_edges(output_edge_order)` to ensure the tensor's axes match the expected edge sequence.

4.  **Cotengra Performance Tuning**:
    - **Protocol**: For large-scale tensor network contractions, use `tc.set_contractor("cotengra")`.
    - **Advanced Tuning**: For complex circuits, manually set a `ctg.ReusableHyperOptimizer` with specific `max_time` and `max_repeats` to find better paths. Use `preprocessing=True` to cache the path.
