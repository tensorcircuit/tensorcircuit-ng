"""
Experimental features
"""

# pylint: disable=unused-import

from functools import partial
import logging
from typing import Any, Callable, Dict, Optional, Tuple, List, Sequence, Union
import pickle
import uuid
import time
import os

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
Mesh: Any
NamedSharding: Any
P: Any


def broadcast_py_object(obj: Any, shared_dir: Optional[str] = None) -> Any:
    """
    Broadcast a picklable Python object from process 0 to all other processes,
    with fallback mechanism from gRPC to file system based approach.

    This function first attempts to use gRPC-based broadcast. If that fails due to
    pickling issues, it falls back to a file system based approach that is more robust.

    :param obj: The Python object to broadcast. It must be picklable.
                This object should exist on process 0 and can be None on others.
    :type obj: Any
    :param shared_dir: Directory path for shared file system broadcast fallback.
                       If None, uses current directory. Only used in fallback mode.
    :type shared_dir: Optional[str], optional
    :return: The broadcasted object, now present on all processes.
    :rtype: Any
    """
    import jax
    from jax.experimental import multihost_utils

    try:
        result = broadcast_py_object_jax(obj)
        return result

    except pickle.UnpicklingError as e:
        # This block is executed if any process fails during the gRPC attempt.

        multihost_utils.sync_global_devices("grpc_broadcast_failed_fallback_sync")

        if jax.process_index() == 0:
            border = "=" * 80
            logger.warning(
                "\n%s\nJAX gRPC broadcast failed with error: %s\n"
                "--> Falling back to robust Shared File System broadcast method.\n%s",
                border,
                e,
                border,
            )

        return broadcast_py_object_fs(obj, shared_dir)


def broadcast_py_object_jax(obj: Any) -> Any:
    """
    Broadcast a picklable Python object from process 0 to all other processes
    within jax ditribution system.

    This function uses a two-step broadcast: first the size, then the data.
    This is necessary because `broadcast_one_to_all` requires the same
    shaped array on all hosts.

    :param obj: The Python object to broadcast. It must be picklable.
             This object should exist on process 0 and can be None on others.

    :return: The broadcasted object, now present on all processes.
    """
    import jax as jaxlib
    import pickle
    from jax.experimental import multihost_utils

    # Serialize to bytes on process 0, empty bytes on others
    if jaxlib.process_index() == 0:
        if obj is None:
            raise ValueError("Object to broadcast from process 0 cannot be None.")
        data = pickle.dumps(obj)
        logger.info(
            f"--- Size of object to be broadcast: {len(data) / 1024**2:.3f} MB ---"
        )

    else:
        data = b""

    # Step 1: Broadcast the length of the serialized data.
    # We send a single-element int32 array.
    length = np.array([len(data)], dtype=np.int32)
    length = multihost_utils.broadcast_one_to_all(length)

    length = int(length[0])  # type: ignore

    # Step 2: Broadcast the actual data.
    # Convert byte string to a uint8 array for broadcasting.
    send_arr_uint8 = np.frombuffer(data, dtype=np.uint8)
    padded_length = (length + 3) // 4 * 4
    if send_arr_uint8.size < padded_length:
        send_arr_uint8 = np.pad(  #  type: ignore
            send_arr_uint8, (0, padded_length - send_arr_uint8.size), mode="constant"
        )
    send_arr_int32 = send_arr_uint8.astype(np.int32)
    # send_arr_int32 = jaxlib.numpy.array(send_arr_int32, dtype=np.int32)
    send_arr_int32 = jaxlib.device_put(send_arr_int32)

    jaxlib.experimental.multihost_utils.sync_global_devices("bulk_before")

    received_arr = multihost_utils.broadcast_one_to_all(send_arr_int32)

    received_arr = np.array(received_arr)
    received_arr_uint8 = received_arr.astype(np.uint8)

    # Step 3: Reconstruct the object from the received bytes.
    # Convert the NumPy array back to bytes, truncate any padding, and unpickle.
    received_data = received_arr_uint8[:length].tobytes()
    # if jaxlib.process_index() == 0:
    #     logger.info(f"Broadcasted object {obj}")
    return pickle.loads(received_data)


def broadcast_py_object_fs(
    obj: Any, shared_dir: Optional[str] = None, timeout_seconds: int = 300
) -> Any:
    """
    Broadcast a picklable Python object from process 0 to all other processes
    using a shared file system approach.

    This is a fallback method when gRPC-based broadcast fails. It uses UUID-based
    file communication to share objects between processes through a shared file system.

    :param obj: The Python object to broadcast. Must be picklable.
                Should exist on process 0, can be None on others.
    :type obj: Any
    :param shared_dir: Directory path for shared file system communication.
                       If None, uses current directory.
    :type shared_dir: Optional[str], optional
    :param timeout_seconds: Maximum time to wait for file operations before timing out.
                            Defaults to 300 seconds.
    :type timeout_seconds: int, optional
    :return: The broadcasted object, now present on all processes.
    :rtype: Any
    """
    # to_avoid very subtle bugs for broadcast tree_data on A800 clusters
    import jax
    from jax.experimental import multihost_utils

    if shared_dir is None:
        shared_dir = "."
    if jax.process_index() == 0:
        os.makedirs(shared_dir, exist_ok=True)

    id_comm_path = os.path.join(shared_dir, f".broadcast_temp_12318")
    transfer_id = ""

    if jax.process_index() == 0:
        transfer_id = str(uuid.uuid4())
        # print(f"[Process 0] Generated unique transfer ID: {transfer_id}", flush=True)
        with open(id_comm_path, "w") as f:
            f.write(transfer_id)

    multihost_utils.sync_global_devices("fs_broadcast_id_written")

    if jax.process_index() != 0:
        start_time = time.time()
        while not os.path.exists(id_comm_path):
            time.sleep(0.1)
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Process {jax.process_index()} timed out waiting for ID file: {id_comm_path}"
                )
        with open(id_comm_path, "r") as f:
            transfer_id = f.read()

    multihost_utils.sync_global_devices("fs_broadcast_id_read")
    if jax.process_index() == 0:
        try:
            os.remove(id_comm_path)
        except OSError:
            pass  # 如果文件已被其他进程快速清理，忽略错误

    # 定义本次传输使用的数据文件和标志文件路径
    data_path = os.path.join(shared_dir, f"{transfer_id}.data")
    done_path = os.path.join(shared_dir, f"{transfer_id}.done")

    result_obj = None

    if jax.process_index() == 0:
        if obj is None:
            raise ValueError("None cannot be broadcasted.")

        # print(f"[Process 0] Pickling object...", flush=True)
        pickled_data = pickle.dumps(obj)
        logger.info(
            f"[Process 0] Writing {len(pickled_data) / 1024**2:.3f} MB to {data_path}"
        )
        with open(data_path, "wb") as f:
            f.write(pickled_data)

        with open(done_path, "w") as f:
            pass
        logger.info(f"[Process 0] Write complete.")
        result_obj = obj
    else:
        # print(f"[Process {jax.process_index()}] Waiting for done file: {done_path}", flush=True)
        start_time = time.time()
        while not os.path.exists(done_path):
            time.sleep(0.1)
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Process {jax.process_index()} timed out waiting for done file: {done_path}"
                )

        # print(f"[Process {jax.process_index()}] Done file found. Reading data from {data_path}", flush=True)
        with open(data_path, "rb") as f:
            pickled_data = f.read()

        result_obj = pickle.loads(pickled_data)
        logger.info(f"[Process {jax.process_index()}] Object successfully loaded.")

    multihost_utils.sync_global_devices("fs_broadcast_read_complete")

    if jax.process_index() == 0:
        try:
            os.remove(data_path)
            os.remove(done_path)
            # print(f"[Process 0] Cleaned up temporary files for transfer {transfer_id}.", flush=True)
        except OSError as e:
            logger.info(
                f"[Process 0]: Failed to clean up temporary files: {e}",
            )

    return result_obj


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
    :param devices: List of devices to use. If None, uses all available devices
    :type devices: Optional[List[Any]], optional
    :param mesh: Mesh object to use for distributed computation. If None, uses all available devices
    :type mesh: Optional[Any], optional
    """

    def __init__(
        self,
        nodes_fn: Callable[[Tensor], List[Gate]],
        params: Tensor,
        cotengra_options: Optional[Dict[str, Any]] = None,
        devices: Optional[List[Any]] = None,  # backward compatibility
        mesh: Optional[Any] = None,
        tree_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        global jaxlib
        global ctg
        global Mesh
        global NamedSharding
        global P

        logger.info("Initializing DistributedContractor...")
        import cotengra as ctg
        from cotengra import ContractionTree
        import jax as jaxlib
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        self.nodes_fn = nodes_fn
        if mesh is not None:
            self.mesh = mesh
        elif devices is not None:
            self.mesh = Mesh(devices, axis_names=("devices",))
        else:
            self.mesh = Mesh(jaxlib.devices(), axis_names=("devices",))
        self.num_devices = len(self.mesh.devices)

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
        if tree_data is None:
            if params is None:
                raise ValueError("Please provide specific circuit parameters array.")
            if jaxlib.process_index() == 0:
                logger.info("Process 0: Running cotengra pathfinder...")
                tree_data = self._get_tree_data(
                    self.nodes_fn, self._params_template, cotengra_options  # type: ignore
                )

            # Step 2: Use the robust helper function to broadcast the tree object.
            # Process 0 sends its computed `tree_object`.
            # Other processes send `None`, but receive the object from process 0.

            if jaxlib.process_count() > 1:
                # self.tree = broadcast_py_object(tree_object)
                jaxlib.experimental.multihost_utils.sync_global_devices("tree_before")
                logger.info(
                    f"Process {jaxlib.process_index()}: Synchronizing contraction path..."
                )
                tree_data = broadcast_py_object(tree_data)
                jaxlib.experimental.multihost_utils.sync_global_devices("tree_after")
        else:
            logger.info("Using pre-computed contraction path.")
        if tree_data is None:
            raise ValueError("Contraction path data is missing.")

        self.tree = ContractionTree.from_path(
            inputs=tree_data["inputs"],
            output=tree_data["output"],
            size_dict=tree_data["size_dict"],
            path=tree_data["path"],
        )

        # Restore slicing information
        for ind, _ in tree_data["sliced_inds"].items():
            self.tree.remove_ind_(ind)

        logger.info(
            f"Process {jaxlib.process_index()}: Contraction path successfully synchronized."
        )
        actual_num_slices = self.tree.nslices

        self._report_tree_info()

        slices_per_device = int(np.ceil(actual_num_slices / self.num_devices))
        padded_size = slices_per_device * self.num_devices
        slice_indices = np.arange(actual_num_slices)
        padded_slice_indices = np.full(padded_size, PADDING_VALUE, dtype=np.int32)
        padded_slice_indices[:actual_num_slices] = slice_indices

        # Reshape for distribution and define the sharding rule
        batched_indices = padded_slice_indices.reshape(
            self.num_devices, slices_per_device
        )
        # Sharding rule: split the first axis (the one for devices) across the 'devices' mesh axis
        self.sharding = NamedSharding(self.mesh, P("devices", None))
        # Place the tensor on devices according to the rule
        self.batched_slice_indices = jaxlib.device_put(batched_indices, self.sharding)

        # self.batched_slice_indices = backend.convert_to_tensor(
        #     padded_slice_indices.reshape(self.num_devices, slices_per_device)
        # )
        print(
            f"Distributing across {self.num_devices} devices. Each device will sequentially process "
            f"up to {slices_per_device} slices."
        )

        self._compiled_vg_fn = None
        self._compiled_v_fn = None

        logger.info("Initialization complete.")

    def _report_tree_info(self) -> None:
        print("\n--- Contraction Path Info ---")
        actual_num_slices = self.tree.nslices
        stats = self.tree.contract_stats()
        print(f"Path found with {actual_num_slices} slices.")
        print(
            f"Arithmetic Intensity (higher is better): {self.tree.arithmetic_intensity():.2f}"
        )
        print("flops (TFlops):", stats["flops"] / 2**40 / self.num_devices)
        print("write (GB):", stats["write"] / 2**27 / actual_num_slices)
        print("size (GB):", stats["size"] / 2**27)
        print("-----------------------------\n")

    @staticmethod
    def _get_tree_data(
        nodes_fn: Callable[[Tensor], List[Gate]],
        params: Tensor,
        cotengra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        global ctg

        import cotengra as ctg

        local_cotengra_options = (cotengra_options or {}).copy()

        nodes = nodes_fn(params)
        tn_info, _ = get_tn_info(nodes)
        default_cotengra_options = {
            "slicing_reconf_opts": {"target_size": 2**28},
            "max_repeats": 128,
            "minimize": "write",
            "parallel": "auto",
            "progbar": True,
        }
        default_cotengra_options.update(local_cotengra_options)

        opt = ctg.ReusableHyperOptimizer(**default_cotengra_options)
        tree_object = opt.search(*tn_info)
        tree_data = {
            "inputs": tree_object.inputs,
            "output": tree_object.output,
            "size_dict": tree_object.size_dict,
            "path": tree_object.get_path(),
            "sliced_inds": tree_object.sliced_inds,
        }
        return tree_data

    @staticmethod
    def find_path(
        nodes_fn: Callable[[Tensor], Tensor],
        params: Tensor,
        cotengra_options: Optional[Dict[str, Any]] = None,
        filepath: Optional[str] = None,
    ) -> None:
        tree_data = DistributedContractor._get_tree_data(
            nodes_fn, params, cotengra_options
        )
        if filepath is not None:
            with open(filepath, "wb") as f:
                pickle.dump(tree_data, f)
            logger.info(f"Contraction path data successfully saved to '{filepath}'.")

    @classmethod
    def from_path(
        cls,
        filepath: str,
        nodes_fn: Callable[[Tensor], List[Gate]],
        devices: Optional[List[Any]] = None,  # backward compatibility
        mesh: Optional[Any] = None,
    ) -> "DistributedContractor":
        with open(filepath, "rb") as f:
            tree_data = pickle.load(f)

        # Each process loads the file independently. No broadcast is needed.
        # We pass the loaded `tree_data` directly to __init__ to trigger the second workflow.
        return cls(
            nodes_fn=nodes_fn,
            params=None,
            mesh=mesh,
            devices=devices,
            tree_data=tree_data,
        )

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
        is_grad_fn: bool,
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

            def global_aggregated_fn(
                tree: Any, params: Any, batched_slice_indices: Tensor
            ) -> Any:
                # Use jax.vmap to apply the per-device function across the sharded data.
                # vmap maps `device_fn` over the first axis (0) of `batched_slice_indices`.
                # `tree` and `params` are broadcasted (in_axes=None) to each call.
                vmapped_device_fn = jaxlib.vmap(
                    device_fn, in_axes=(None, None, 0), out_axes=0
                )
                device_results = vmapped_device_fn(tree, params, batched_slice_indices)

                # Now, `device_results` is a sharded PyTree (one result per device).
                # We aggregate them using jnp.sum, which JAX automatically compiles
                # into a cross-device AllReduce operation.

                if is_grad_fn:
                    # `device_results` is a (value, grad) tuple of sharded arrays
                    device_values, device_grads = device_results

                    # Replace psum with jnp.sum
                    global_value = jaxlib.numpy.sum(device_values, axis=0)
                    global_grad = jaxlib.tree_util.tree_map(
                        lambda g: jaxlib.numpy.sum(g, axis=0), device_grads
                    )
                    return global_value, global_grad
                else:
                    # `device_results` is just the sharded values
                    return jaxlib.numpy.sum(device_results, axis=0)

            #  Compile the global function with jax.jit and specify shardings.
            # `params` are replicated (available everywhere).
            params_sharding = jaxlib.tree_util.tree_map(
                lambda x: NamedSharding(self.mesh, P(*((None,) * x.ndim))),
                self._params_template,
            )

            in_shardings = (params_sharding, self.sharding)

            if is_grad_fn:
                # Returns (value, grad), so out_sharding must be a 2-tuple.
                # `value` is a replicated scalar -> P()
                sharding_for_value = NamedSharding(self.mesh, P())
                # `grad` is a replicated PyTree with the same structure as params.
                sharding_for_grad = params_sharding
                out_shardings = (sharding_for_value, sharding_for_grad)
            else:
                # Returns a single scalar value -> P()
                out_shardings = NamedSharding(self.mesh, P())

            compiled_fn = jaxlib.jit(
                global_aggregated_fn,
                # `tree` is a static argument, its value is compiled into the function.
                static_argnums=(0,),
                # Specify how inputs are sharded.
                in_shardings=in_shardings,
                # Specify how the output should be sharded.
                out_shardings=out_shardings,
            )
            cache[cache_key] = compiled_fn  # type: ignore
        return cache[cache_key]  # type: ignore

    def value_and_grad(
        self,
        params: Tensor,
        # aggregate: bool = True,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculates the value and gradient, compiling the pmap function if needed for the first call.

        :param params: Parameters for the `nodes_fn` input
        :type params: Tensor
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
            is_grad_fn=True,
        )

        total_value, total_grad = compiled_vg_fn(
            self.tree, params, self.batched_slice_indices
        )
        return total_value, total_grad

    def value(
        self,
        params: Tensor,
        # aggregate: bool = True,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Tensor:
        """
        Calculates the value, compiling the pmap function for the first call.

        :param params: Parameters for the `nodes_fn` input
        :type params: Tensor
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
            is_grad_fn=False,
        )

        total_value = compiled_v_fn(self.tree, params, self.batched_slice_indices)
        return total_value

    def grad(
        self,
        params: Tensor,
        op: Optional[Callable[[Tensor], Tensor]] = None,
        output_dtype: Optional[str] = None,
    ) -> Tensor:
        _, grad = self.value_and_grad(params, op=op, output_dtype=output_dtype)
        return grad
