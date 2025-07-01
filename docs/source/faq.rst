Frequently Asked Questions
============================

What is the relation between TensorCircuit and TensorCircuit-NG?
-------------------------------------------------------------------

Both packages are created by `Shi-Xin Zhang <https://www.iop.cas.cn/rcjy/tpyjy/?id=6789>`_ (`@refraction-ray <https://github.com/refraction-ray>`_). For the history of the evolution of tensorcircuit, please refer to `history <https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/HISTORY.md>`_.

From users' perspective, TensorCircuit-NG maintains full compatibility with the TensorCircuit API, enhancing it with additional features and critical bug fixes. Only TensorCircuit-NG is kept up-to-date with the fast evolving scientific computing and machine learning ecosystem in Python.

TensorCircuit-NG is intended as a drop-in replacement for TensorCircuit, namely, by simply ``pip uninstall tensorcircuit`` and ``pip install tensorcircuit-ng``, your existing applications should continue to function seamlessly without requiring any modification to the codebase (``import tensorcircuit`` still works).



How can I run TensorCircuit-NG on GPU?
-----------------------------------------

This is done directly through the ML backend. GPU support is determined by whether ML libraries are can run on GPU, we don't handle this within tensorcircuit-ng.
It is the users' responsibility to configure a GPU-compatible environment for these ML packages. Please refer to the installation documentation for these ML packages and directly use the official dockerfiles provided by TensorCircuit-NG.

- TensorFlow: ``pip install "tensorflow[and-cuda]"``

- Jax: ``pip install -U "jax[cuda12]"``

With GPU compatible environment, we can switch the use of GPU or CPU by a backend agnostic environment variable ``CUDA_VISIBLE_DEVICES``.


When should I use GPU?
----------------------------------------------------

In general, for a circuit with qubit count larger than 16 or for circuit simulation with large batch dimension more than 16, GPU simulation will be faster than CPU simulation.
That is to say, for very small circuits and the very small batch dimensions of vectorization, GPU may show worse performance than CPU.
But one have to carry out detailed benchmarks on the hardware choice, since the performance is determined by the hardware and task details.

For tensor network tasks of more regular shape, such as MPS-MPO contraction, GPU can be much more favored and efficient than CPU.


How can I use multiple GPUs?
----------------------------------------------------

For different observables evaluation on different cards, see `example <https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/examples/vqe_parallel_pmap.py>`_.

For distributed simulation of one circuit on multiple cards, see `example for expectation <https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/examples/slicing_auto_pmap_vqa.py>`_ and `example for MPO <https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/examples/slicing_auto_pmap_mpo.py>`_.


When should I jit the function?
----------------------------------------------------

For a function with "tensor in and tensor out", wrapping it with jit will greatly accelerate the evaluation. Since the first time of evaluation takes longer time (staging time), jit is only good for functions which have to be evaluated frequently.


.. Warning::

    Be caution that jit can be easily misused if the users are not familiar with jit mechanism, which may lead to:
    
        1. very slow performance due to recompiling/staging for each run, 
        2. error when run function with jit, 
        3. or wrong results without any warning.

    The most possible reasons for each problem are:
    
        1. function input are not all in the tensor form,
        2. the output shape of all ops in the function may require the knowledge of the input value more than the input shape, or use mixed ops from numpy and ML framework
        3. subtle interplay between random number generation and jit (see :ref:`advance:Randoms, Jit, Backend Agnostic, and Their Interplay` for the correct solution), respectively.


Which ML framework backend should I use?
--------------------------------------------

Since the Numpy backend has no support for AD, if you want to evaluate the circuit gradient, you must set the backend as one of the ML frameworks beyond Numpy.

Since PyTorch has very limited support for vectorization and jit while our package strongly depends on these features, it is not recommended to use. Though one can always wrap a quantum function on another backend using a PyTorch interface, say :py:meth:`tensorcircuit.interfaces.torch_interface`.

In terms of the choice between TensorFlow and Jax backend, the better one may depend on the use cases and one may want to benchmark both to pick the better one. There is no one-for-all recommendation and this is why we maintain the backend agnostic form of our software.

Some general rules of thumb:

* On both CPU and GPU, the running time of a jitted function is faster for jax backend.

* But on GPU, jit staging time is usually much longer for jax backend.

* For hybrid machine learning tasks, TensorFlow has a better ML ecosystem and reusable classical ML models.

* Jax has some built-in advanced features that are lacking in TensorFlow, such as checkpoint in AD and pmap for distributed computing.

* Jax is much insensitive to dtype where type promotion is handled automatically which means easier debugging.

* TensorFlow can cache the jitted function on the disk via SavedModel, which further amortizes the staging time.


What is the counterpart of ``QuantumLayer`` for PyTorch and Jax backend?
----------------------------------------------------------------------------

Since PyTorch doesn't have mature vmap and jit support and Jax doesn't have native classical ML layers, we highly recommend TensorFlow as the backend for quantum-classical hybrid machine learning tasks, where ``QuantumLayer`` plays an important role.
For PyTorch, we can in principle wrap the corresponding quantum function into a PyTorch module, we currently have the built-in support for this wrapper as ``tc.TorchLayer``.
In terms of the Jax backend, we highly suggested keeping the functional programming paradigm for such machine learning tasks.
Besides, it is worth noting that, jit and vmap are automatically taken care of in ``QuantumLayer``.

When do I need to customize the contractor and how?
------------------------------------------------------

As a rule of thumb, for the circuit with qubit counts larger than 16 and circuit depth larger than 8, customized contraction may outperform the default built-in greedy contraction strategy.

To set up or not set up the customized contractor is about a trade-off between the time on contraction pathfinding and the time on the real contraction via matmul.

The customized contractor costs much more time than the default contractor in terms of contraction path searching, and via the path it finds, the real contraction can take less time and space.

If the circuit simulation time is the bottleneck of the whole workflow, one can always try customized contractors to see whether there is some performance improvement.

We recommend to using `cotengra library <https://cotengra.readthedocs.io/en/latest/index.html>`_ to set up the contractor, since there are lots of interesting hyperparameters to tune, we can achieve a better trade-off between the time on contraction path search and the time on the real tensor network contraction.

It is also worth noting that for jitted function which we usually use, the contraction path search is only called at the first run of the function, which further amortizes the time and favors the use of a highly customized contractor.

In terms of how-to on contractor setup, please refer to :ref:`quickstart:Setup the Contractor`.

Is there some API less cumbersome than ``expectation`` for Pauli string?
----------------------------------------------------------------------------

Say we want to measure something like :math:`\langle X_0Z_1Y_2Z_4 \rangle` for a six-qubit system, the general ``expectation`` API may seem to be cumbersome.
So one can try one of the following options:

* ``c.expectation_ps(x=[0], y=[2], z=[1, 4])`` 

* ``tc.templates.measurements.parameterized_measurements(c, np.array([1, 3, 2, 0, 3, 0]), onehot=True)``

Can I apply quantum operation based on previous classical measurement results?
----------------------------------------------------------------------------------------------------

Try the following: (the pipeline is even fully jittable!)

.. code-block:: python

    c = tc.Circuit(2)
    c.H(0)
    r = c.cond_measurement(0)
    c.conditional_gate(r, [tc.gates.i(), tc.gates.x()], 1)

``cond_measurement`` will return 0 or 1 based on the measurement result on z-basis, and ``conditional_gate`` applies gate_list[r] on the circuit.

How to understand the difference between different measurement methods for ``Circuit``?
----------------------------------------------------------------------------------------------------

* :py:meth:`tensorcircuit.circuit.Circuit.measure` : used at the end of the circuit execution, return bitstring based on quantum amplitude probability (can also with the probability), the circuit and the output state are unaffected (no collapse). The jittable version is ``measure_jit``.

* :py:meth:`tensorcircuit.circuit.Circuit.cond_measure`: also with alias ``cond_measurement``, usually used in the middle of the circuit execution. Apply a POVM on z basis on the given qubit, the state is collapsed and nomarlized based on the measurement projection. The method returns an integer Tensor indicating the measurement result 0 or 1 based on the quantum amplitude probability. 

* :py:meth:`tensorcircuit.circuit.Circuit.post_select`: also with alia ``mid_measurement``, usually used in the middle of the circuit execution. The measurement result is fixed as given from ``keep`` arg of this method. The state is collapsed but unnormalized based on the given measurement projection.

Please refer to the following demos:

.. code-block:: python

    c = tc.Circuit(2)
    c.H(0)
    c.H(1)
    print(c.measure(0, 1))
    # ('01', -1.0)
    print(c.measure(0, with_prob=True))
    # ('0', (0.4999999657714588+0j))
    print(c.state()) # unaffected
    # [0.49999998+0.j 0.49999998+0.j 0.49999998+0.j 0.49999998+0.j]

    c = tc.Circuit(2)
    c.H(0)
    c.H(1)
    print(c.cond_measure(0))  # measure the first qubit return +z
    # 0
    print(c.state())  # collapsed and normalized
    # [0.70710678+0.j 0.70710678+0.j 0.        +0.j 0.        +0.j]

    c = tc.Circuit(2)
    c.H(0)
    c.H(1)
    print(c.post_select(0, keep=1))  # measure the first qubit and it is guranteed to return -z
    # 1
    print(c.state())  # collapsed but unnormalized
    # [0.        +0.j 0.        +0.j 0.49999998+0.j 0.49999998+0.j]


How to understand difference between ``tc.array_to_tensor`` and ``tc.backend.convert_to_tensor``?
------------------------------------------------------------------------------------------------------

``tc.array_to_tensor`` convert array to tensor as well as automatically cast the type to the default dtype of TensorCircuit-NG,
i.e. ``tc.dtypestr`` and it also support to specify dtype as ``tc.array_to_tensor( , dtype="complex128")``.
Instead, ``tc.backend.convert_to_tensor`` keeps the dtype of the input array, and to cast it as complex dtype, we have to
explicitly call ``tc.backend.cast`` after conversion. Besides, ``tc.array_to_tensor`` also accepts multiple inputs as
``a_tensor, b_tensor = tc.array_to_tensor(a_array, b_array)``.


How to arrange the circuit gate placement in the visualization from ``c.tex()``?
----------------------------------------------------------------------------------------------------

Try ``lcompress=True`` or ``rcompress=True`` option in :py:meth:`tensorcircuit.circuit.Circuit.tex` API to make the circuit align from the left or from the right.

Or try ``c.unitary(0, unitary=tc.backend.eye(2), name="invisible")`` to add placeholder on the circuit which is invisible for circuit visualization.


How many different formats for the circuit sample results?
--------------------------------------------------------------------------

When performing measurements or sampling in TensorCircuit-NG, there are six different formats available for the results:

1. ``"sample_int"``
    Returns measurement results as integer array.

    .. code-block:: python

        >>> c = tc.Circuit(2)
        >>> c.h(0)
        >>> c.sample(batch=3, format="sample_int")
        array([0, 2, 0])  # Each number represents a measurement outcome

2. ``"sample_bin"``
    Returns measurement results as a list of binary arrays.

    .. code-block:: python

        >>> c.sample(batch=3, format="sample_bin")
        Array([[0, 0],
                [1, 0],
                [1, 0]], dtype=int32)  # Each sub array represents a binary string

3. ``"count_vector"``
    Returns counts as a vector where index represents the state.

    .. code-block:: python

        >>> c.sample(batch=3, format="count_vector")
        Array([1, 0, 2, 0], dtype=int32)  # [#|00⟩, #|01⟩, #|10⟩, #|11⟩]

4. ``"count_tuple"``
    Returns counts as a tuple of indices and their frequencies.

    .. code-block:: python

        >>> c.sample(batch=4, format="count_tuple", jittable=False)
        (Array([0, 2], dtype=int32), Array([2, 1], dtype=int32))  # (int_states, frequencies)

5. ``"count_dict_bin"``
    Returns counts as a dictionary with binary strings as keys.

    .. code-block:: python

        >>> c.sample(batch=4, format="count_dict_bin")
        {"00": 2, "01": 0, "10": 2, "11": 0}

6. ``"count_dict_int"``
    Returns counts as a dictionary with integers as keys.

    .. code-block:: python

        >>> c.sample(batch=4, format="count_dict_int")
        {0: 2, 1: 0, 2: 2, 3: 0}  # {state_integer: frequency}


For more input parameters, see API doc :py:meth:`tensorcircuit.circuit.Circuit.sample`.


How to get the entanglement entropy from the circuit output?
--------------------------------------------------------------------

Try the following:

.. code-block:: python

    c = tc.Circuit(4)
    # omit circuit construction

    rho = tc.quantum.reduced_density_matrix(s, cut=[0, 1, 2])
    # get the redueced density matrix, where cut list is the index to be traced out

    rho.shape
    # (2, 2)

    ee = tc.quantum.entropy(rho)
    # get the entanglement entropy

    renyi_ee = tc.quantum.renyi_entropy(rho, k=2)
    # get the k-th order renyi entropy