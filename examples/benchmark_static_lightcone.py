"""
Compare dynamic and static light-cone evaluation for one observable.

Example::

    python examples/benchmark_static_lightcone.py

The static paths compare a reduced ``SymbolCircuit`` with the same reduced
circuit written out explicitly.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import sympy

import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

NQUBITS = 32
NLAYERS = 8
OBSERVABLE_QUBITS = [0]
REPEATS = 5
REDUCED_NQUBITS = 8
REDUCED_PARAMETER_INDICES = (
    0,
    1,
    2,
    3,
    16,
    17,
    18,
    31,
    32,
    33,
    47,
    48,
    62,
    63,
    78,
    93,
)


def ansatz(nqubits: int, nlayers: int) -> Tuple[tc.SymbolCircuit, List[sympy.Symbol]]:
    circuit = tc.SymbolCircuit(nqubits)
    circuit.h(0)
    symbols = []
    for layer in range(nlayers):
        for qubit in range(layer % 2, nqubits - 1, 2):
            symbol = sympy.Symbol(f"theta_{len(symbols)}", real=True)
            circuit.rzz(qubit, qubit + 1, theta=symbol)
            symbols.append(symbol)
    circuit.h(0)
    return circuit, symbols


def expectation(
    circuit: tc.Circuit, observable_qubits: Sequence[int], lightcone: bool
) -> Any:
    return tc.backend.real(
        circuit.expectation_ps(z=observable_qubits, enable_lightcone=lightcone)
    )


def dynamic_loss(nqubits: int, nlayers: int, observable_qubits: Sequence[int]):
    """Build the full circuit and reduce its light cone inside JAX."""

    def loss(params: Any) -> Any:
        circuit = tc.Circuit(nqubits)
        circuit.h(0)
        parameter = 0
        for layer in range(nlayers):
            for qubit in range(layer % 2, nqubits - 1, 2):
                circuit.rzz(qubit, qubit + 1, theta=params[parameter])
                parameter += 1
        circuit.h(0)
        return expectation(circuit, observable_qubits, lightcone=True)

    return loss


def symbolcircuit_loss(
    circuit: tc.SymbolCircuit,
    observable_qubits: Sequence[int],
    symbols: Sequence[sympy.Symbol],
):
    """Use the reduced SymbolCircuit and its backend-aware ``to_circuit``."""

    def loss(params: Any) -> Any:
        bindings = {symbol: params[index] for index, symbol in enumerate(symbols)}
        numerical_circuit = circuit.to_circuit(bindings)
        return expectation(numerical_circuit, observable_qubits, lightcone=False)

    return loss


def direct_loss(params: Any) -> Any:
    """Evaluate the explicitly written reduced circuit."""
    circuit = tc.Circuit(REDUCED_NQUBITS)
    circuit.h(0)
    circuit.rzz(0, 1, theta=params[0])
    circuit.rzz(2, 3, theta=params[1])
    circuit.rzz(4, 5, theta=params[2])
    circuit.rzz(6, 7, theta=params[3])
    circuit.rzz(1, 2, theta=params[4])
    circuit.rzz(3, 4, theta=params[5])
    circuit.rzz(5, 6, theta=params[6])
    circuit.rzz(0, 1, theta=params[7])
    circuit.rzz(2, 3, theta=params[8])
    circuit.rzz(4, 5, theta=params[9])
    circuit.rzz(1, 2, theta=params[10])
    circuit.rzz(3, 4, theta=params[11])
    circuit.rzz(0, 1, theta=params[12])
    circuit.rzz(2, 3, theta=params[13])
    circuit.rzz(1, 2, theta=params[14])
    circuit.rzz(0, 1, theta=params[15])
    circuit.h(0)
    return expectation(circuit, OBSERVABLE_QUBITS, lightcone=False)


def sync(value: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        value,
    )


def timed(function: Any, params: Any) -> Tuple[float, Any]:
    start = time.perf_counter()
    result = sync(function(params))
    return time.perf_counter() - start, result


def benchmark(loss: Any, params: Any, repeats: int) -> Dict[str, Any]:
    """Measure the first call including JIT compilation, then steady calls."""
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    first, (value, gradient) = timed(value_and_grad, params)
    steady = statistics.median(timed(value_and_grad, params)[0] for _ in range(repeats))
    return {
        "first": first,
        "steady": steady,
        "value": np.asarray(value),
        "gradient": np.asarray(gradient),
    }


def main() -> None:
    start = time.perf_counter()
    full, symbols = ansatz(NQUBITS, NLAYERS)
    reduced, info = tc.compiler.lightcone_compile(full, OBSERVABLE_QUBITS)
    observable_qubits = info["observable_qubits"]
    used = reduced.free_symbols()
    parameters = REDUCED_PARAMETER_INDICES
    assert used == {symbols[index] for index in parameters}
    setup = time.perf_counter() - start
    full_params = jnp.linspace(-0.7, 0.8, len(symbols), dtype=jnp.float32)
    reduced_params = full_params[jnp.asarray(parameters)]

    losses = {
        "dynamic": dynamic_loss(NQUBITS, NLAYERS, OBSERVABLE_QUBITS),
        "SymbolCircuit": symbolcircuit_loss(reduced, observable_qubits, symbols),
        "Handwritten": direct_loss,
    }
    results = {
        name: benchmark(
            loss, reduced_params if name == "Handwritten" else full_params, REPEATS
        )
        for name, loss in losses.items()
    }

    for name, result in results.items():
        if name == "Handwritten":
            expanded = np.zeros(len(symbols), dtype=np.float32)
            expanded[list(parameters)] = result["gradient"]
        else:
            expanded = result["gradient"]
        np.testing.assert_allclose(
            results["dynamic"]["value"], result["value"], atol=2e-5
        )
        np.testing.assert_allclose(results["dynamic"]["gradient"], expanded, atol=2e-5)

    assert np.linalg.norm(results["dynamic"]["gradient"]) > 1e-6

    print(
        f"static setup={setup:.6f}s, observable_qubits={OBSERVABLE_QUBITS}, "
        f"qubits={NQUBITS}->{REDUCED_NQUBITS}, "
        f"parameters={len(symbols)}->{len(parameters)}"
    )
    print("path              first value+grad  steady value+grad")
    for name, result in results.items():
        print(f"{name:<18} {result['first']:.6f}          " f"{result['steady']:.6f}")


if __name__ == "__main__":
    main()
