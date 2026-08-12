TenCirPauli Plugin
===================

`TenCirPauli <https://github.com/tensorcircuit/TenCirPauli>`_ is a Python-first, Rust-native companion for TensorCircuit-NG. Its two main entry points are approximate circuit simulation with Pauli propagation and backpropagation, and fast fixed-particle-number circuit simulation.

.. card-carousel:: 2

   .. card:: TenCirPauli on GitHub
      :link: https://github.com/tensorcircuit/TenCirPauli
      :shadow: md

      Read the full benchmarks and implementation details.

   .. card:: TenCirPauli Documentation
      :link: https://tensorcircuit.github.io/TenCirPauli/
      :shadow: md

      Read the user guide and API reference.

Installation
------------

Install the companion package alongside TensorCircuit-NG:

.. code-block:: bash

   pip install tencirpauli

The package complements TensorCircuit-NG rather than replacing its general-purpose circuit and tensor-network simulators. Use TensorCircuit-NG to construct circuits and work with its differentiable backend interface, then hand off suitable structured workloads to TenCirPauli.

Pauli propagation
------------------

TenCirPauli provides approximate circuit simulation based on Pauli propagation. It can import a regular TensorCircuit circuit, propagate Pauli operators through the circuit, and return observable values together with gate-parameter gradients:

.. code-block:: python

   import tencirpauli as tcp
   import tensorcircuit as tc

   circuit = tc.Circuit(4)
   circuit.h(0)
   circuit.ry(1, theta=0.21)
   circuit.cnot(0, 1)
   circuit.rz(2, theta=-0.17)
   circuit.cnot(2, 3)

   native_circuit = tcp.PropagationCircuit.from_circuit(circuit)
   observable = tcp.PauliOperator.from_terms(
       4, [("ZZII", 1.0), ("IIZZ", 0.5)]
   )
   result = native_circuit.value_and_grad(observable)

   print(result.value)
   print(result.gradient)

TenCirPauli is substantially faster than `PauliPropagation.jl <https://github.com/SparqleSim/PauliPropagation.jl>`_. See the `TenCirPauli performance comparisons <https://tensorcircuit.github.io/TenCirPauli/performance/>`_ for the benchmark details.

Fixed-particle-number circuits
------------------------------

TensorCircuit-NG already provides :py:class:`tensorcircuit.U1Circuit` for particle-number-conserving simulations. TenCirPauli provides a faster Rust implementation for this same class of circuits. It avoids JAX JIT compilation's cold-start overhead:

.. code-block:: python

   import tencirpauli as tcp
   import tensorcircuit as tc

   circuit = tc.U1Circuit(60, k=2, filled=[0, 1])
   for qubit in range(59):
       circuit.iswap(qubit, qubit + 1, theta=0.08)

   native_circuit = tcp.U1Circuit.from_circuit(circuit)
   hamiltonian = tcp.PauliOperator.from_terms(
       60, [("X" + "I" * 58 + "X", 0.5)]
   )
   result = native_circuit.value_and_grad(hamiltonian)


Structured operators and MVPs
-----------------------------

TenCirPauli offers backend-compatible matrix-vector-product (MVP) interfaces for structured fermionic, bosonic, Majorana, qudit, and hybrid operators. It also provides fermion-to-qubit mappings and import paths from quantum-chemistry libraries, so these operators can be used directly in TensorCircuit workflows without materializing dense matrices.

For example, a molecular Hamiltonian from PySCF can be mapped to a Pauli-string MVP:

.. code-block:: python

   from pyscf import gto
   from tencirpauli.integrations.pyscf import from_molecule
   from tencirpauli.integrations.tensorcircuit import backend_mvp
   import tensorcircuit as tc

   molecule = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
   fermion_hamiltonian = from_molecule(molecule)
   pauli_hamiltonian = fermion_hamiltonian.map_fermions("jordan_wigner")
   apply_hamiltonian = backend_mvp(
       pauli_hamiltonian.backend_mvp_plan(), backend=tc.backend
   )
   state = tc.Circuit(pauli_hamiltonian.nqubits).state()
   hamiltonian_times_state = apply_hamiltonian(state)

