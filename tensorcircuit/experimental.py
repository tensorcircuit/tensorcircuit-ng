"""
Experimental features
"""

# pylint: disable=unused-import

from functools import partial
import logging
from typing import Any, Callable, Dict, Optional, Tuple, List, Sequence, Union

import numpy as np

from .cons import backend, dtypestr, rdtypestr, get_tn_info
from .gates import Gate
from .timeevol import hamiltonian_evol, evol_global, evol_local


# for backward compatibility

Tensor = Any
Circuit = Any

logger = logging.getLogger(__name__)


def adaptive_vmap(
    f: Callable[..., Any],
    vectorized_argnums: Union[int, Sequence[int]] = 0,
    static_argnums: Optional[Union[int, Sequence[int]]] = None,
    chunk_size: Optional[int] = None,
) -> Callable[..., Any]:
    """
    Vectorized map with adaptive chunking for memory efficiency.

    :param f: Function to be vectorized
    :param vectorized_argnums: Arguments to be vectorized over
    :param static_argnums: Arguments that remain static during vectorization
    :param chunk_size: Size of chunks for batch processing, None means no chunking
        (naive vmap)
    :return: Vectorized function
    """
    if chunk_size is None:
        return backend.vmap(f, vectorized_argnums)  # type: ignore

    if isinstance(vectorized_argnums, int):
        vectorized_argnums = (vectorized_argnums,)

    def wrapper(*args: Any, **kws: Any) -> Tensor:
        # only support `f` outputs a tensor
        s1, s2 = divmod(args[vectorized_argnums[0]].shape[0], chunk_size)  # type: ignore
        # repetition, rest
        reshape_args = []
        rest_args = []
        for i, arg in enumerate(args):
            if i in vectorized_argnums:  # type: ignore
                if s2 != 0:
                    arg_rest = arg[-s2:]
                    arg = arg[:-s2]
                arg = backend.reshape(
                    arg,
                    [s1, chunk_size] + list(backend.shape_tuple(arg))[1:],
                )

            else:
                arg_rest = arg
            reshape_args.append(arg)
            if s2 != 0:
                rest_args.append(arg_rest)
        _vmap = backend.jit(
            backend.vmap(f, vectorized_argnums=vectorized_argnums),
            static_argnums=static_argnums,
        )
        r = []
        for i in range(s1):
            # currently using naive python loop for simplicity
            nreshape_args = [
                a[i] if j in vectorized_argnums else a  # type: ignore
                for j, a in enumerate(reshape_args)
            ]
            r.append(_vmap(*nreshape_args, **kws))
        r = backend.tree_map(lambda *x: backend.concat(x), *r)
        # rshape = list(backend.shape_tuple(r))
        # if len(rshape) == 2:
        #     nshape = [rshape[0] * rshape[1]]
        # else:
        #     nshape = [rshape[0] * rshape[1], -1]
        # r = backend.reshape(r, nshape)
        if s2 != 0:
            rest_r = _vmap(*rest_args, **kws)
            return backend.tree_map(lambda *x: backend.concat(x), r, rest_r)
        return r

    return wrapper


def _qng_post_process(t: Tensor, eps: float = 1e-4) -> Tensor:
    t += eps * backend.eye(t.shape[0], dtype=t.dtype)
    t = backend.real(t)
    return t


def _id(a: Any) -> Any:
    return a


def _vdot(i: Tensor, j: Tensor) -> Tensor:
    return backend.tensordot(backend.conj(i), j, 1)


def qng(
    f: Callable[..., Tensor],
    kernel: str = "qng",
    postprocess: Optional[str] = "qng",
    mode: str = "fwd",
) -> Callable[..., Tensor]:
    """
    Compute quantum natural gradient for quantum circuit optimization.

    :param f: Function that takes parameters and returns quantum state
    :param kernel: Type of kernel to use ("qng" or "dynamics"), the former has the second term
    :param postprocess: Post-processing method ("qng" or None)
    :param mode: Mode of differentiation ("fwd" or "rev")
    :return: Function computing QNG matrix

    :Example:

    >>> import tensorcircuit as tc
    >>> def ansatz(params):
    ...     c = tc.Circuit(2)
    ...     c.rx(0, theta=params[0])
    ...     c.ry(1, theta=params[1])
    ...     return c.state()
    >>> qng_fn = tc.experimental.qng(ansatz)
    >>> params = tc.array_to_tensor([0.5, 0.8])
    >>> qng_matrix = qng_fn(params)
    >>> print(qng_matrix.shape)  # (2, 2)
    """

    # for both qng and qng2 calculation, we highly recommended complex-dtype but real valued inputs
    def wrapper(params: Tensor, **kws: Any) -> Tensor:
        params = backend.cast(params, dtype=dtypestr)  # R->C protection
        psi = f(params)
        if mode == "fwd":
            jac = backend.jacfwd(f)(params)
        else:  # "rev"
            jac = backend.jacrev(f)(params)
            jac = backend.cast(jac, dtypestr)  # incase input is real
            # may have R->C issue for rev mode, which we obtain a real Jacobian
        jac = backend.transpose(jac)
        if kernel == "qng":

            def ij(i: Tensor, j: Tensor) -> Tensor:
                return _vdot(i, j) - _vdot(i, psi) * _vdot(psi, j)

        elif kernel == "dynamics":

            def ij(i: Tensor, j: Tensor) -> Tensor:
                return _vdot(i, j)

        vij = backend.vmap(ij, vectorized_argnums=0)
        vvij = backend.vmap(vij, vectorized_argnums=1)

        fim = vvij(jac, jac)
        # TODO(@refraction-ray): investigate more on
        # suitable hyperparameters and methods for regularization?
        if isinstance(postprocess, str):
            if postprocess == "qng":
                _post_process = _qng_post_process
            else:
                raise ValueError("Unsupported postprocess option")

        elif postprocess is None:
            _post_process = _id  # type: ignore
        else:
            _post_process = postprocess  # callable
        fim = _post_process(fim)
        return fim

    return wrapper


dynamics_matrix = partial(qng, kernel="dynamics", postprocess=None)


def qng2(
    f: Callable[..., Tensor],
    kernel: str = "qng",
    postprocess: Optional[str] = "qng",
    mode: str = "rev",
) -> Callable[..., Tensor]:
    # reverse mode has a slightly better running time
    # wan's approach for qng
    def wrapper(params: Tensor, **kws: Any) -> Tensor:
        params2 = backend.copy(params)
        params2 = backend.stop_gradient(params2)

        def outer_loop(params2: Tensor) -> Tensor:
            def inner_product(params: Tensor, params2: Tensor) -> Tensor:
                s = f(params)
                s2 = f(params2)
                fid = _vdot(s2, s)
                if kernel == "qng":
                    fid -= _vdot(s2, backend.stop_gradient(s)) * _vdot(
                        backend.stop_gradient(s2), s
                    )
                return fid

            _, grad = backend.vjp(
                partial(inner_product, params2=params2), params, backend.ones([])
            )
            return grad

        if mode == "fwd":
            fim = backend.jacfwd(outer_loop)(params2)
        else:
            fim = backend.jacrev(outer_loop)(params2)
        # directly real if params is real, then where is the imaginary part?
        if isinstance(postprocess, str):
            if postprocess == "qng":
                _post_process = _qng_post_process
            else:
                raise ValueError("Unsupported postprocess option")

        elif postprocess is None:
            _post_process = _id  # type: ignore
        else:
            _post_process = postprocess  # callable
        fim = _post_process(fim)
        return fim

    # on jax backend, qng and qng2 output is different by a conj
    # on tf backend, the outputs are the same
    return wrapper


def dynamics_rhs(f: Callable[..., Any], h: Tensor) -> Callable[..., Any]:
    # compute :math:`\langle \psi \vert H \vert \partial \psi \rangle`
    def wrapper(params: Tensor, **kws: Any) -> Tensor:
        def energy(params: Tensor) -> Tensor:
            w = f(params, **kws)
            wr = backend.stop_gradient(w)
            wl = backend.conj(w)
            wl = backend.reshape(wl, [1, -1])
            wr = backend.reshape(wr, [-1, 1])
            if not backend.is_sparse(h):
                e = wl @ h @ wr
            else:
                tmp = backend.sparse_dense_matmul(h, wr)
                e = wl @ tmp
            return backend.real(e)[0, 0]

        return backend.grad(energy)(params)

    return wrapper


def parameter_shift_grad(
    f: Callable[..., Tensor],
    argnums: Union[int, Sequence[int]] = 0,
    jit: bool = False,
    shifts: Tuple[float, float] = (np.pi / 2, 2),
) -> Callable[..., Tensor]:
    """
    similar to `grad` function but using parameter shift internally instead of AD,
    vmap is utilized for evaluation, so the speed is still ok

    :param f: quantum function with weights in and expectation out
    :type f: Callable[..., Tensor]
    :param argnums: label which args should be differentiated,
        defaults to 0
    :type argnums: Union[int, Sequence[int]], optional
    :param jit: whether jit the original function `f` at the beginning,
        defaults to False
    :type jit: bool, optional
    :param shifts: two floats for the delta shift on the numerator and dominator,
        defaults to (pi/2, 2) for parameter shift
    :type shifts: Tuple[float, float]
    :return: the grad function
    :rtype: Callable[..., Tensor]
    """
    if jit is True:
        f = backend.jit(f)

    if isinstance(argnums, int):
        argnums = [argnums]

    vfs = [backend.vmap(f, vectorized_argnums=i) for i in argnums]

    def grad_f(*args: Any, **kws: Any) -> Any:
        grad_values = []
        for i in argnums:  # type: ignore
            shape = backend.shape_tuple(args[i])
            size = backend.sizen(args[i])
            onehot = backend.eye(size)
            onehot = backend.cast(onehot, args[i].dtype)
            onehot = backend.reshape(onehot, [size] + list(shape))
            onehot = shifts[0] * onehot
            nargs = list(args)
            arg = backend.reshape(args[i], [1] + list(shape))
            batched_arg = backend.tile(arg, [size] + [1 for _ in shape])
            nargs[i] = batched_arg + onehot
            nargs2 = list(args)
            nargs2[i] = batched_arg - onehot
            r = (vfs[i](*nargs, **kws) - vfs[i](*nargs2, **kws)) / shifts[1]
            r = backend.reshape(r, shape)
            grad_values.append(r)
        if len(argnums) > 1:  # type: ignore
            return tuple(grad_values)
        return grad_values[0]

    return grad_f


def parameter_shift_grad_v2(
    f: Callable[..., Tensor],
    argnums: Union[int, Sequence[int]] = 0,
    jit: bool = False,
    random_argnums: Optional[Sequence[int]] = None,
    shifts: Tuple[float, float] = (np.pi / 2, 2),
) -> Callable[..., Tensor]:
    """
    similar to `grad` function but using parameter shift internally instead of AD,
    vmap is utilized for evaluation, v2 also supports random generator for finite
    measurememt shot, only jax backend is supported, since no vmap randomness is
    available in tensorflow

    :param f: quantum function with weights in and expectation out
    :type f: Callable[..., Tensor]
    :param argnums: label which args should be differentiated,
        defaults to 0
    :type argnums: Union[int, Sequence[int]], optional
    :param jit: whether jit the original function `f` at the beginning,
        defaults to False
    :type jit: bool, optional
    :param shifts: two floats for the delta shift on the numerator and dominator,
        defaults to (pi/2, 2) for parameter shift
    :type shifts: Tuple[float, float]
    :return: the grad function
    :rtype: Callable[..., Tensor]
    """
    # TODO(@refraction-ray): replace with new status support for the sample API
    if jit is True:
        f = backend.jit(f)

    if isinstance(argnums, int):
        argnums = [argnums]

    if random_argnums is None:
        vfs = [backend.vmap(f, vectorized_argnums=i) for i in argnums]
    else:
        if isinstance(random_argnums, int):
            random_argnums = [random_argnums]
        vfs = [
            backend.vmap(f, vectorized_argnums=[i] + random_argnums) for i in argnums  # type: ignore
        ]

    def grad_f(*args: Any, **kws: Any) -> Any:
        grad_values = []
        for i in argnums:  # type: ignore
            shape = backend.shape_tuple(args[i])
            size = backend.sizen(args[i])
            onehot = backend.eye(size)
            onehot = backend.cast(onehot, args[i].dtype)
            onehot = backend.reshape(onehot, [size] + list(shape))
            onehot = shifts[0] * onehot
            nargs = list(args)
            arg = backend.reshape(args[i], [1] + list(shape))
            batched_arg = backend.tile(arg, [size] + [1 for _ in shape])
            nargs[i] = batched_arg + onehot
            nargs2 = list(args)
            nargs2[i] = batched_arg - onehot
            if random_argnums is not None:
                for j in random_argnums:
                    keys = []
                    key = args[j]
                    for _ in range(size):
                        key, subkey = backend.random_split(key)
                        keys.append(subkey)
                    nargs[j] = backend.stack(keys)
                    keys = []
                    for _ in range(size):
                        key, subkey = backend.random_split(key)
                        keys.append(subkey)
                    nargs2[j] = backend.stack(keys)
            r = (vfs[i](*nargs, **kws) - vfs[i](*nargs2, **kws)) / shifts[1]
            r = backend.reshape(r, shape)
            grad_values.append(r)
        if len(argnums) > 1:  # type: ignore
            return tuple(grad_values)
        return grad_values[0]

    return grad_f


def finite_difference_differentiator(
    f: Callable[..., Any],
    argnums: Tuple[int, ...] = (0,),
    shifts: Tuple[float, float] = (0.001, 0.002),
) -> Callable[..., Any]:
    # \bar{x}_j = \sum_i \bar{y}_i \frac{\Delta y_i}{\Delta x_j}
    # tf only now and designed for hardware, since we dont do batch evaluation
    import tensorflow as tf

    @tf.custom_gradient  # type: ignore
    def tf_function(*args: Any, **kwargs: Any) -> Any:
        y = f(*args, **kwargs)

        def grad(ybar: Any) -> Any:
            # only support one output
            delta_ms = []
            for argnum in argnums:
                delta_m = []
                xi = tf.reshape(args[argnum], [-1])
                xi_size = xi.shape[0]
                onehot = tf.one_hot(tf.range(xi_size), xi_size)
                for j in range(xi_size):
                    xi_plus = xi + tf.cast(shifts[0] * onehot[j], xi.dtype)
                    xi_minus = xi - tf.cast(shifts[0] * onehot[j], xi.dtype)
                    args_plus = list(args)
                    args_plus[argnum] = tf.reshape(xi_plus, args[argnum].shape)
                    args_minus = list(args)
                    args_minus[argnum] = tf.reshape(xi_minus, args[argnum].shape)
                    dy = f(*args_plus, **kwargs) - f(*args_minus, **kwargs)
                    dy /= shifts[-1]
                    delta_m.append(tf.reshape(dy, [-1]))
                delta_m = tf.stack(delta_m)
                delta_m = tf.transpose(delta_m)
                delta_ms.append(delta_m)
            dxs = [tf.zeros_like(arg) for arg in args]
            ybar_flatten = tf.reshape(ybar, [1, -1])
            for i, argnum in enumerate(argnums):
                dxs[argnum] = tf.cast(
                    tf.reshape(ybar_flatten @ delta_ms[i], args[argnum].shape),
                    args[argnum].dtype,
                )

            return tuple(dxs)

        return y, grad

    return tf_function  # type: ignore


def jax_jitted_function_save(filename: str, f: Callable[..., Any], *args: Any) -> None:
    """
    save a jitted jax function as a file

    :param filename: _description_
    :type filename: str
    :param f: the jitted function
    :type f: Callable[..., Any]
    :param args: example function arguments for ``f``
    """

    from jax import export  # type: ignore

    f_export = export.export(f)(*args)  # type: ignore
    barray = f_export.serialize()

    with open(filename, "wb") as file:
        file.write(barray)


jax_func_save = jax_jitted_function_save


def jax_jitted_function_load(filename: str) -> Callable[..., Any]:
    """
    load a jitted function from file

    :param filename: _description_
    :type filename: str
    :return: the loaded function
    :rtype: _type_
    """
    from jax import export  # type: ignore

    with open(filename, "rb") as f:
        barray = f.read()

    f_load = export.deserialize(barray)  # type: ignore

    return f_load.call  # type: ignore


jax_func_load = jax_jitted_function_load


PADDING_VALUE = -1
jaxlib: Any
ctg: Any


class DistributedContractor:
    """
    A distributed tensor network contractor that parallelizes computations across multiple devices.

    This class uses cotengra to find optimal contraction paths and distributes the computational
    load across multiple devices (e.g., GPUs) for efficient tensor network calculations.
    Particularly useful for large-scale quantum circuit simulations and variational quantum algorithms.

    Example:
        >>> def nodes_fn(params):
        ...     c = tc.Circuit(4)
        ...     c.rx(0, theta=params[0])
        ...     return c.expectation_before([tc.gates.z(), [0]], reuse=False)
        >>> dc = DistributedContractor(nodes_fn, params)
        >>> value, grad = dc.value_and_grad(params)

    :param nodes_fn: Function that takes parameters and returns a list of tensor network nodes
    :type nodes_fn: Callable[[Tensor], List[Gate]]
    :param params: Initial parameters used to determine the tensor network structure
    :type params: Tensor
    :param cotengra_options: Configuration options passed to the cotengra optimizer. Defaults to None
    :type cotengra_options: Optional[Dict[str, Any]], optional
    :param devices: List of devices to use. If None, uses all available local devices
    :type devices: Optional[List[Any]], optional
    """

    def __init__(
        self,
        nodes_fn: Callable[[Tensor], List[Gate]],
        params: Tensor,
        cotengra_options: Optional[Dict[str, Any]] = None,
        devices: Optional[List[Any]] = None,
    ) -> None:
        global jaxlib
        global ctg

        logger.info("Initializing DistributedContractor...")
        import cotengra as ctg
        import jax as jaxlib

        self.nodes_fn = nodes_fn
        if devices is None:
            self.num_devices = jaxlib.local_device_count()
            self.devices = jaxlib.local_devices()
            # TODO(@refraction-ray): multi host support
        else:
            self.devices = devices
            self.num_devices = len(devices)

        if self.num_devices <= 1:
            logger.info("DistributedContractor is running on a single device.")

        self._params_template = params
        self._backend = "jax"
        self._compiled_v_fns: Dict[
            Tuple[Callable[[Tensor], Tensor], str],
            Callable[[Any, Tensor, Tensor], Tensor],
        ] = {}
        self._compiled_vg_fns: Dict[
            Tuple[Callable[[Tensor], Tensor], str],
            Callable[[Any, Tensor, Tensor], Tensor],
        ] = {}

        logger.info("Running cotengra pathfinder... (This may take a while)")
        nodes = self.nodes_fn(self._params_template)
        tn_info, _ = get_tn_info(nodes)
        default_cotengra_options = {
            "slicing_reconf_opts": {"target_size": 2**28},
            "max_repeats": 128,
            "progbar": True,
            "minimize": "write",
            "parallel": "auto",
        }
        if cotengra_options:
            default_cotengra_options = cotengra_options

        opt = ctg.ReusableHyperOptimizer(**default_cotengra_options)
        self.tree = opt.search(*tn_info)
        actual_num_slices = self.tree.nslices

        print("\n--- Contraction Path Info ---")
        stats = self.tree.contract_stats()
        print(f"Path found with {actual_num_slices} slices.")
        print(
            f"Arithmetic Intensity (higher is better): {self.tree.arithmetic_intensity():.2f}"
        )
        print("flops (TFlops):", stats["flops"] / 2**40 / self.num_devices)
        print("write (GB):", stats["write"] / 2**27 / actual_num_slices)
        print("size (GB):", stats["size"] / 2**27)
        print("-----------------------------\n")

        slices_per_device = int(np.ceil(actual_num_slices / self.num_devices))
        padded_size = slices_per_device * self.num_devices
        slice_indices = np.arange(actual_num_slices)
        padded_slice_indices = np.full(padded_size, PADDING_VALUE, dtype=np.int32)
        padded_slice_indices[:actual_num_slices] = slice_indices
        self.batched_slice_indices = backend.convert_to_tensor(
            padded_slice_indices.reshape(self.num_devices, slices_per_device)
        )
        print(
            f"Distributing across {self.num_devices} devices. Each device will sequentially process "
            f"up to {slices_per_device} slices."
        )

        self._compiled_vg_fn = None
        self._compiled_v_fn = None

        logger.info("Initialization complete.")

    def _get_single_slice_contraction_fn(
        self, op: Optional[Callable[[Tensor], Tensor]] = None
    ) -> Callable[[Any, Tensor, int], Tensor]:
        if op is None:
            op = backend.sum

        def single_slice_contraction(
            tree: ctg.ContractionTree, params: Tensor, slice_idx: int
        ) -> Tensor:
            nodes = self.nodes_fn(params)
            _, standardized_nodes = get_tn_info(nodes)
            input_arrays = [node.tensor for node in standardized_nodes]
            sliced_arrays = tree.slice_arrays(input_arrays, slice_idx)
            result = tree.contract_core(sliced_arrays, backend=self._backend)
            return op(result)

        return single_slice_contraction

    def _get_device_sum_vg_fn(
        self,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Callable[[Any, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        post_processing = lambda x: backend.real(backend.sum(x))
        if op is None:
            op = post_processing
        base_fn = self._get_single_slice_contraction_fn(op=op)
        # to ensure the output is real so that can be differentiated
        single_slice_vg_fn = jaxlib.value_and_grad(base_fn, argnums=1)

        if output_dtype is None:
            output_dtype = rdtypestr

        def device_sum_fn(
            tree: ctg.ContractionTree, params: Tensor, slice_indices_for_device: Tensor
        ) -> Tuple[Tensor, Tensor]:
            def scan_body(
                carry: Tuple[Tensor, Tensor], slice_idx: Tensor
            ) -> Tuple[Tuple[Tensor, Tensor], None]:
                acc_value, acc_grads = carry

                def compute_and_add() -> Tuple[Tensor, Tensor]:
                    value_slice, grads_slice = single_slice_vg_fn(
                        tree, params, slice_idx
                    )
                    new_value = acc_value + value_slice
                    new_grads = jaxlib.tree_util.tree_map(
                        jaxlib.numpy.add, acc_grads, grads_slice
                    )
                    return new_value, new_grads

                def do_nothing() -> Tuple[Tensor, Tensor]:
                    return acc_value, acc_grads

                return (
                    jaxlib.lax.cond(
                        slice_idx == PADDING_VALUE, do_nothing, compute_and_add
                    ),
                    None,
                )

            initial_carry = (
                backend.cast(backend.convert_to_tensor(0.0), dtype=output_dtype),
                jaxlib.tree_util.tree_map(lambda x: jaxlib.numpy.zeros_like(x), params),
            )
            (final_value, final_grads), _ = jaxlib.lax.scan(
                scan_body, initial_carry, slice_indices_for_device
            )
            return final_value, final_grads

        return device_sum_fn

    def _get_device_sum_v_fn(
        self,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Callable[[Any, Tensor, Tensor], Tensor]:
        base_fn = self._get_single_slice_contraction_fn(op=op)
        if output_dtype is None:
            output_dtype = dtypestr

        def device_sum_fn(
            tree: ctg.ContractionTree, params: Tensor, slice_indices_for_device: Tensor
        ) -> Tensor:
            def scan_body(
                carry_value: Tensor, slice_idx: Tensor
            ) -> Tuple[Tensor, None]:
                def compute_and_add() -> Tensor:
                    return carry_value + base_fn(tree, params, slice_idx)

                return (
                    jaxlib.lax.cond(
                        slice_idx == PADDING_VALUE, lambda: carry_value, compute_and_add
                    ),
                    None,
                )

            initial_carry = backend.cast(
                backend.convert_to_tensor(0.0), dtype=output_dtype
            )
            final_value, _ = jaxlib.lax.scan(
                scan_body, initial_carry, slice_indices_for_device
            )
            return final_value

        return device_sum_fn

    def _get_or_compile_fn(
        self,
        cache: Dict[
            Tuple[Callable[[Tensor], Tensor], str],
            Callable[[Any, Tensor, Tensor], Tensor],
        ],
        fn_getter: Callable[..., Any],
        op: Optional[Callable[[Tensor], Tensor]],
        output_dtype: Optional[str],
    ) -> Callable[[Any, Tensor, Tensor], Tensor]:
        """
        Gets a compiled pmap-ed function from cache or compiles and caches it.

        The cache key is a tuple of (op, output_dtype). Caution on lambda function!

        Returns:
            The compiled, pmap-ed JAX function.
        """
        cache_key = (op, output_dtype)
        if cache_key not in cache:
            device_fn = fn_getter(op=op, output_dtype=output_dtype)
            compiled_fn = jaxlib.pmap(
                device_fn,
                in_axes=(
                    None,
                    None,
                    0,
                ),  # tree: broadcast, params: broadcast, indices: map
                static_broadcasted_argnums=(0,),  # arg 0 (tree) is a static argument
                devices=self.devices,
            )
            cache[cache_key] = compiled_fn  # type: ignore
        return cache[cache_key]  # type: ignore

    def value_and_grad(
        self,
        params: Tensor,
        aggregate: bool = True,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculates the value and gradient, compiling the pmap function if needed for the first call.

        :param params: Parameters for the `nodes_fn` input
        :type params: Tensor
        :param aggregate: Whether to aggregate (sum) the results across devices, defaults to True
        :type aggregate: bool, optional
        :param op: Optional post-processing function for the output, defaults to None (corresponding to `backend.real`)
            op is a cache key, so dont directly pass lambda function for op
        :type op: Optional[Callable[[Tensor], Tensor]], optional
        :param output_dtype: dtype str for the output of `nodes_fn`, defaults to None (corresponding to `rdtypestr`)
        :type output_dtype: Optional[str], optional
        """
        compiled_vg_fn = self._get_or_compile_fn(
            cache=self._compiled_vg_fns,
            fn_getter=self._get_device_sum_vg_fn,
            op=op,
            output_dtype=output_dtype,
        )

        device_values, device_grads = compiled_vg_fn(
            self.tree, params, self.batched_slice_indices
        )

        if aggregate:
            total_value = backend.sum(device_values)
            total_grad = jaxlib.tree_util.tree_map(
                lambda x: backend.sum(x, axis=0), device_grads
            )
            return total_value, total_grad
        return device_values, device_grads

    def value(
        self,
        params: Tensor,
        aggregate: bool = True,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Tensor:
        """
        Calculates the value, compiling the pmap function for the first call.

        :param params: Parameters for the `nodes_fn` input
        :type params: Tensor
        :param aggregate: Whether to aggregate (sum) the results across devices, defaults to True
        :type aggregate: bool, optional
        :param op: Optional post-processing function for the output, defaults to None (corresponding to identity)
            op is a cache key, so dont directly pass lambda function for op
        :type op: Optional[Callable[[Tensor], Tensor]], optional
        :param output_dtype: dtype str for the output of `nodes_fn`, defaults to None (corresponding to `dtypestr`)
        :type output_dtype: Optional[str], optional
        """
        compiled_v_fn = self._get_or_compile_fn(
            cache=self._compiled_v_fns,
            fn_getter=self._get_device_sum_v_fn,
            op=op,
            output_dtype=output_dtype,
        )

        device_values = compiled_v_fn(self.tree, params, self.batched_slice_indices)

        if aggregate:
            return backend.sum(device_values)
        return device_values

    def grad(
        self,
        params: Tensor,
        aggregate: bool = True,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Tensor:
        _, grad = self.value_and_grad(
            params, aggregate=aggregate, op=op, output_dtype=output_dtype
        )
        return grad
