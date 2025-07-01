================
Advanced Usage
================

MPS Simulator
----------------

TensorCircuit-NG provides Matrix Product State (MPS) simulation as an efficient alternative to exact simulation for quantum circuits. MPS simulation can handle larger quantum systems by trading off accuracy for computational efficiency.

MPS simulator is very straightforward to use, we provide the same set of API for ``MPSCircuit`` as ``Circuit``, 
the only new line is to set the bond dimension for the new simulator.

.. code-block:: python

    c = tc.MPSCircuit(n)
    c.set_split_rules({"max_singular_values": 50})

The larger bond dimension we set, the better approximation ratio (of course the more computational cost we pay).


Stacked gates syntax
------------------------

Stacked-gate is a simple syntactic sugar rendering circuit construction easily when multiple gate of the same type are applied on different qubits, namely, the index for gate call can accept list of ints instead of one integer.

.. code-block:: python

    >>> import tensorcircuit as tc
    >>> c = tc.Circuit(4)
    >>> c.h(range(3))
    >>> c.draw()
         ┌───┐
    q_0: ┤ H ├
         ├───┤
    q_1: ┤ H ├
         ├───┤
    q_2: ┤ H ├
         └───┘
    q_3: ─────


    >>> c = tc.Circuit(4)
    >>> c.cnot([0, 1], [2, 3])
    >>> c.draw()

    q_0: ──■───────
           │
    q_1: ──┼────■──
         ┌─┴─┐  │
    q_2: ┤ X ├──┼──
         └───┘┌─┴─┐
    q_3: ─────┤ X ├
              └───┘

    >>> c = tc.Circuit(4)
    >>> c.rx(range(4), theta=tc.backend.convert_to_tensor([0.1, 0.2, 0.3, 0.4]))
    >>> c.draw()
         ┌─────────┐
    q_0: ┤ Rx(0.1) ├
         ├─────────┤
    q_1: ┤ Rx(0.2) ├
         ├─────────┤
    q_2: ┤ Rx(0.3) ├
         ├─────────┤
    q_3: ┤ Rx(0.4) ├
         └─────────┘



Split Two-qubit Gates
-------------------------

The two-qubit gates applied on the circuit can be decomposed via SVD, which may further improve the optimality of the contraction pathfinding.

``split`` configuration can be set at circuit-level or gate-level.

.. code-block:: python

    split_conf = {
        "max_singular_values": 2,  # how many singular values are kept
        "fixed_choice": 1, # 1 for normal one, 2 for swapped one
    }

    c = tc.Circuit(nwires, split=split_conf)

    # or

    c.exp1(
            i,
            (i + 1) % nwires,
            theta=paramc[2 * j, i],
            unitary=tc.gates._zz_matrix,
            split=split_conf
        )

Note ``max_singular_values`` must be specified to make the whole procedure static and thus jittable.

Analog circuit simulation
-----------------------------

TensorCircuit-NG support digital-analog hybrid simulation (say cases in Rydberg atom arrays), where the analog part is simulated by the neural differential equation solver given the API to specify a time dependent Hamiltonian.
The simulation is still differentiable and jittable. Only jax backend is supported for analog simulation as the neural ode engine is built on top of jax. 

This utility is super helpful for optimizing quantum control or investigating digital-analog hybrid variational quantum schemes.

We support two modes of analog simulation, where :py:meth:`tensorcircuit.experimental.evol_global` evolve the state via a Hamiltonian define on the whole system, and :py:meth:`tensorcircuit.experimental.evol_local` evolve the state via a Hamiltonian define on a local subsystem.

.. Note::

    ``evol_global`` uses sparse Hamiltonian while ``evol_local`` uses dense Hamiltonian.


.. code-block:: python

    # in this demo, we build a jittable and differentiable simulation function `hybrid_evol` 
    # with both digital gates and local/global analog Hamiltonian evolutions

    import optax
    import tensorcircuit as tc
    from tensorcircuit.experimental import evol_global, evol_local

    K = tc.set_backend("jax")


    def h_fun(t, b):
        return b * tc.gates.x().tensor


    hy = tc.quantum.PauliStringSum2COO([[2, 0]])


    def h_fun2(t, b):
        return b[2] * K.cos(b[0] * t + b[1]) * hy


    @K.jit
    @K.value_and_grad
    def hybrid_evol(params):
        c = tc.Circuit(2)
        c.x([0, 1])
        c = evol_local(c, [1], h_fun, 1.0, params[0])
        c.cx(1, 0)
        c.h(0)
        c = evol_global(c, h_fun2, 1.0, params[1:])
        return K.real(c.expectation_ps(z=[0, 1]))


    b = K.implicit_randn([4])
    v, gs = hybrid_evol(b)



Jitted Function Save/Load
-----------------------------

To reuse the jitted function, we can save it on the disk via support from the TensorFlow `SavedModel <https://www.tensorflow.org/guide/saved_model>`_. That is to say, only jitted quantum function on the TensorFlow backend can be saved on the disk. 

We wrap the tf-backend ``SavedModel`` as very easy-to-use function :py:meth:`tensorcircuit.keras.save_func` and :py:meth:`tensorcircuit.keras.load_func`.

For the JAX-backend quantum function, one can first transform them into the tf-backend function via JAX experimental support: `jax2tf <https://github.com/google/jax/tree/main/jax/experimental/jax2tf>`_.

**Updates**: jax now also support jitted function save/load via ``export`` module, see `jax documentation <https://jax.readthedocs.io/en/latest/export/export.html>`_.

We wrap the jax function export capability in ``experimental`` module and can be used as follows

.. code-block:: python

    from tensorcircuit import experimental

    K = tc.set_backend("jax")

    @K.jit
    def f(weights):
        c = tc.Circuit(3)
        c.rx(range(3), theta=weights)
        return K.real(c.expectation_ps(z=[0]))

    print(f(K.ones([3])))

    experimental.jax_jitted_function_save("temp.bin", f, K.ones([3]))

    f_load = tc.experimental.jax_jitted_function_load("temp.bin")
    f_load(K.ones([3]))



Parameterized Measurements
-----------------------------

For plain measurements API on a ``tc.Circuit``, eg. ``c = tc.Circuit(3)``, if we want to evaluate the expectation :math:`<Z_1Z_2>`, we need to call the API as ``c.expectation((tc.gates.z(), [1]), (tc.gates.z(), [2]))``. 

In some cases, we may want to tell the software what to measure but in a tensor fashion. For example, if we want to get the above expectation, we can use the following API: :py:meth:`tensorcircuit.templates.measurements.parameterized_measurements`.

.. code-block:: python

    c = tc.Circuit(3)
    z1z2 = tc.templates.measurements.parameterized_measurements(c, tc.array_to_tensor([0, 3, 3, 0]), onehot=True) # 1

This API corresponds to measure :math:`I_0Z_1Z_2I_3` where 0, 1, 2, 3 are for local I, X, Y, and Z operators respectively.

Sparse Matrix
----------------

We support COO format sparse matrix as most backends only support this format, and some common backend methods for sparse matrices are listed below:

.. code-block:: python

    def sparse_test():
        m = tc.backend.coo_sparse_matrix(indices=np.array([[0, 1],[1, 0]]), values=np.array([1.0, 1.0]), shape=[2, 2])
        n = tc.backend.convert_to_tensor(np.array([[1.0], [0.0]]))
        print("is sparse: ", tc.backend.is_sparse(m), tc.backend.is_sparse(n))
        print("sparse matmul: ", tc.backend.sparse_dense_matmul(m, n))

    for K in ["tensorflow", "jax", "numpy"]:
        with tc.runtime_backend(K):
            print("using backend: ", K)
            sparse_test()

The sparse matrix is specifically useful to evaluate Hamiltonian expectation on the circuit, where sparse matrix representation has a good tradeoff between space and time.
Please refer to :py:meth:`tensorcircuit.templates.measurements.sparse_expectation` for more detail.

For different representations to evaluate Hamiltonian expectation in tensorcircuit, please refer to :doc:`tutorials/tfim_vqe_diffreph`.


Hamiltonian Matrix Building
----------------------------

TensorCircuit-NG provides multiple ways to build Hamiltonian matrices, especially for sparse Hamiltonians constructed from Pauli strings. This is crucial for quantum many-body physics simulations and variational quantum algorithms.

**Pauli String Based Construction:**

The most flexible way to build Hamiltonians is through Pauli strings:

.. code-block:: python

    import tensorcircuit as tc
    
    # Define Pauli strings and their weights
    # Each Pauli string is represented by a list of integers:
    # 0: Identity, 1: X, 2: Y, 3: Z
    pauli_strings = [
        [1, 1, 0],  # X₁X₂I₃
        [3, 3, 0],  # Z₁Z₂I₃
        [0, 0, 1],  # I₁I₂X₃
    ]
    weights = [0.5, 1.0, -0.2]
    
    # Build sparse Hamiltonian
    h_sparse = tc.quantum.PauliStringSum2COO(pauli_strings, weights)
    
    # Or dense Hamiltonian if preferred
    h_dense = tc.quantum.PauliStringSum2Dense(pauli_strings, weights)


**High-Level Hamiltonian Construction:**

For common Hamiltonians like Heisenberg model:

.. code-block:: python

    # Create a 1D chain with 10 sites
    g = tc.templates.graphs.Line1D(10, pbc=True)  # periodic boundary condition
    
    # XXZ model
    h = tc.quantum.heisenberg_hamiltonian(
        g,
        hxx=1.0,  # XX coupling
        hyy=1.0,  # YY coupling
        hzz=1.2,  # ZZ coupling
        hx=0.5,   # X field
        sparse=True
    )


**Advanced Usage:**

1. Converting between xyz and Pauli string representations:

.. code-block:: python

    # Convert Pauli string to xyz format
    xyz_dict = tc.quantum.ps2xyz([1, 2, 2, 0])  # X₁Y₂Y₃I₄
    print(xyz_dict)  # {'x': [0], 'y': [1, 2], 'z': []}
    
    # Convert back to Pauli string
    ps = tc.quantum.xyz2ps(xyz_dict, n=4)
    print(ps)  # [1, 2, 2, 0]


2. Working with MPO format:

TensorCircuit-NG supports conversion from different MPO (Matrix Product Operator) formats, particularly from TensorNetwork and Quimb libraries. This is useful when you want to leverage existing MPO implementations or convert between different frameworks.

**TensorNetwork MPO:**

For TensorNetwork MPOs, you can convert predefined models like the Transverse Field Ising (TFI) model:

.. code-block:: python

    import tensorcircuit as tc
    import tensornetwork as tn
    
    # Create TFI Hamiltonian MPO from TensorNetwork
    nwires = 6
    Jx = np.array([1.0] * (nwires - 1))  # XX coupling strength
    Bz = np.array([-1.0] * nwires)       # Transverse field strength
    
    # Create TensorNetwork MPO
    tn_mpo = tn.matrixproductstates.mpo.FiniteTFI(
        Jx, Bz, 
        dtype=np.complex64
    )
    
    # Convert to TensorCircuit format
    tc_mpo = tc.quantum.tn2qop(tn_mpo)
    
    # Get dense matrix representation
    h_matrix = tc_mpo.eval_matrix()

Note: TensorNetwork MPO currently only supports open boundary conditions.

**Quimb MPO:**

Quimb provides more flexible MPO construction options:

.. code-block:: python

    import tensorcircuit as tc
    import quimb.tensor as qtn
    
    # Create Ising Hamiltonian MPO using Quimb
    nwires = 6
    J = 4.0    # ZZ coupling
    h = 2.0    # X field
    qb_mpo = qtn.MPO_ham_ising(
        nwires, 
        J, h,
        cyclic=True  # Periodic boundary conditions
    )
    
    # Convert to TensorCircuit format
    tc_mpo = tc.quantum.quimb2qop(qb_mpo)
    
    # Custom Hamiltonian construction
    builder = qtn.SpinHam1D()
    builder += 1.0, "Y"  # Add Y term with strength 1.0
    builder += 0.5, "X"  # Add X term with strength 0.5
    H = builder.build_mpo(3)  # Build for 3 sites
    
    # Convert to TensorCircuit MPO
    h_tc = tc.quantum.quimb2qop(H)



Stabilizer Circuit Simulator
-----------------------------

TensorCircuit-NG provides a Stabilizer Circuit simulator for efficient simulation of Clifford circuits. 
This simulator is particularly useful for quantum error correction, measurement induced phase transition, etc.

The stabilizer simulation is backend by Python package `Stim <https://github.com/quantumlib/Stim>`_, please ensure you have ``pip install stim`` first.


.. code-block:: python

    import tensorcircuit as tc
    
    # Create a stabilizer circuit
    c = tc.StabilizerCircuit(2)
    
    # Apply Clifford gates
    c.h(0)
    c.cnot(0, 1)
    
    # Measure qubits
    results = c.measure(0, 1)  # Returns measurement outcomes
    
    # Sample multiple shots
    samples = c.sample(batch=1000)  # Returns array of shape (1000, 2)

**Supported Operations**

The simulator supports common Clifford gates and operations:

- Single-qubit gates: H, X, Y, Z, S, SDG (S dagger)
- Two-qubit gates: CNOT, CZ, SWAP
- Measurements: projective measurements (``c.measurement`` doesn't affect the state while ``c.cond_measure`` collpases the state)
- Post-selection (``c.post_select``)
- Random Clifford gates (``c.random_gate``)
- Gates defined by tableau (``c.tableau_gate``)
- Entanglement calculation (``c.entanglement_entropy``)
- Pauli string operator expectation (``c.expectation_ps``)
- Openqasm and qir transformation as usual circuits
- Initialization state provided by Pauli string stabilizer (``tc.StabCircuit(inputs=...)``) or inverse tableau (``tc.StabCircuit(tableau_inputs=)``)
- Probabilistic noise (``c.depolarizing``)


Example: Quantum Teleportation

.. code-block:: python

    c = tc.StabilizerCircuit(3)
    
    # Prepare Bell pair between qubits 1 and 2
    c.h(1)
    c.cnot(1, 2)
    
    # State to teleport on qubit 0 (must be Clifford)
    c.x(0)
    
    # Teleportation circuit
    c.cnot(0, 1)
    c.h(0)
    
    # Measure and apply corrections
    r0 = c.cond_measure(0)
    r1 = c.cond_measure(1)
    if r0 == 1:
        c.z(2)
    if r1 == 1:
        c.x(2)




Fermion Gaussian State Simulator
--------------------------------

TensorCircuit-NG provides a powerful Fermion Gaussian State (FGS) simulator for efficient simulation of non-interacting fermionic systems (with or without U(1) symmtery). The simulator is particularly useful for studying quantum many-body physics and entanglement properties.


.. code-block:: python

    import tensorcircuit as tc
    import numpy as np

    # Initialize a 4-site system with sites 0 and 2 occupied
    sim = tc.FGSSimulator(L=4, filled=[0, 2])
    
    # Evolve with hopping terms
    sim.evol_hp(i=0, j=1, chi=1.0)  # hopping between sites 0 and 1
    
    # Calculate entanglement entropy for subsystem of sites 0, 1
    entropy = sim.entropy([2, 3])


The simulator supports various operations including:

1. State initialization from quadratic Hamiltonians ground states
2. Time evolution (real and imaginary)
3. Entanglement measures (von Neumann, Renyi entropies and entanglement asymmetry)
4. Correlation matrix calculations
5. Measurements


Here's an example studying entanglement asymmetry in tilted ferromagnet states:

.. code-block:: python

    def xy_hamiltonian(theta, L):
        # XY model with tilted field
        gamma = 2 / (np.cos(theta) ** 2 + 1) - 1
        mu = 4 * np.sqrt(1 - gamma**2) * np.ones([L])
        
        # Construct Hamiltonian terms
        h = (generate_hopping_h(2.0, L) + 
             generate_pairing_h(gamma * 2, L) + 
             generate_chemical_h(mu))
        return h

    def get_saq_sa(theta, l, L, k, batch=1024):
        # Calculate entanglement asymmetry in the middle subsystem with size l
        traceout = [i for i in range(0, L//2 - l//2)] + \
                  [i for i in range(L//2 + l//2, L)]
        
        # Get Hamiltonian ground state which is within FGS
        hi = xy_hamiltonian(theta, L)
        sim = tc.FGSSimulator(L, hc=hi)
        
        # Get both symmetry-resolved and standard entanglement
        return (np.real(sim.renyi_entanglement_asymmetry(k, traceout, batch=batch)),
                sim.renyi_entropy(k, traceout))


Randoms, Jit, Backend Agnostic, and Their Interplay
--------------------------------------------------------

.. code-block:: python

    import tensorcircuit as tc
    K = tc.set_backend("tensorflow")
    K.set_random_state(42)

    @K.jit
    def r():
        return K.implicit_randn()

    print(r(), r()) # different, correct

.. code-block:: python

    import tensorcircuit as tc
    K = tc.set_backend("jax")
    K.set_random_state(42)

    @K.jit
    def r():
        return K.implicit_randn()

    print(r(), r()) # the same, wrong


.. code-block:: python

    import tensorcircuit as tc
    import jax
    K = tc.set_backend("jax")
    key = K.set_random_state(42)

    @K.jit
    def r(key):
        K.set_random_state(key)
        return K.implicit_randn()

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2)) # different, correct

Therefore, a unified jittable random infrastructure with backend agnostic can be formulated as 

.. code-block:: python

    import tensorcircuit as tc
    import jax
    K = tc.set_backend("tensorflow")

    def ba_key(key):
        if tc.backend.name == "tensorflow":
            return None
        if tc.backend.name == "jax":
            return jax.random.PRNGKey(key)
        raise ValueError("unsupported backend %s"%tc.backend.name)

        
    @K.jit
    def r(key=None):
        if key is not None:
            K.set_random_state(key)
        return K.implicit_randn()

    key = ba_key(42)

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2))

And a more neat approach to achieve this is as follows:

.. code-block:: python

    key = K.get_random_state(42)

    @K.jit
    def r(key):
        K.set_random_state(key)
        return K.implicit_randn()

    key1, key2 = K.random_split(key)

    print(r(key1), r(key2))

It is worth noting that since ``Circuit.unitary_kraus`` and ``Circuit.general_kraus`` call ``implicit_rand*`` API, the correct usage of these APIs is the same as above.

One may wonder why random numbers are dealt in such a complicated way, please refer to the `Jax design note <https://jax.readthedocs.io/en/latest/jep/263-prng.html>`_ for some hints.

If vmap is also involved apart from jit, I currently find no way to maintain the backend agnosticity as TensorFlow seems to have no support of vmap over random keys (ping me on GitHub if you think you have a way to do this). I strongly recommend the users using Jax backend in the vmap+random setup.
