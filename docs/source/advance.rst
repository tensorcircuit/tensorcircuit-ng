================
Advanced Usage
================

MPS Simulator
----------------

Very straightforward to use, we provide the same set of API for ``MPSCircuit`` as ``Circuit``, 
the only new line is to set the bond dimension for the new simulator.

.. code-block:: python

    c = tc.MPSCircuit(n)
    c.set_split_rules({"max_singular_values": 50})

The larger bond dimension we set, the better approximation ratio (of course the more computational cost we pay).


Stacked gates
----------------

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

`split` configuration can be set at circuit-level or gate-level.

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

We wrap the tf-backend `SavedModel` as very easy-to-use function :py:meth:`tensorcircuit.keras.save_func` and :py:meth:`tensorcircuit.keras.load_func`.

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

For plain measurements API on a ``tc.Circuit``, eg. `c = tc.Circuit(n=3)`, if we want to evaluate the expectation :math:`<Z_1Z_2>`, we need to call the API as ``c.expectation((tc.gates.z(), [1]), (tc.gates.z(), [2]))``. 

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

One may wonder why random numbers are dealt in such a complicated way, please refer to the `Jax design note <https://github.com/google/jax/blob/main/docs/design_notes/prng.md>`_ for some hints.

If vmap is also involved apart from jit, I currently find no way to maintain the backend agnosticity as TensorFlow seems to have no support of vmap over random keys (ping me on GitHub if you think you have a way to do this). I strongly recommend the users using Jax backend in the vmap+random setup.