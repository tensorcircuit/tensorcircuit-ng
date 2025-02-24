"""
Interface wraps quantum function as a jax function
"""

from typing import Any, Callable, Tuple, Optional, Union, Sequence
from functools import wraps, partial

from ..cons import backend
from .tensortrans import general_args_to_backend

Tensor = Any


def jax_wrapper(
    fun: Callable[..., Any],
    enable_dlpack: bool = False,
    output_shape: Optional[
        Union[Tuple[int, ...], Tuple[int, ...], Sequence[Tuple[int, ...]]]
    ] = None,
    output_dtype: Optional[Union[Any, Sequence[Any]]] = None,
) -> Callable[..., Any]:
    import jax

    @wraps(fun)
    def fun_jax(*x: Any) -> Any:
        def wrapped_fun(*args: Any) -> Any:
            args = general_args_to_backend(args, enable_dlpack=enable_dlpack)
            y = fun(*args)
            y = general_args_to_backend(
                y, target_backend="jax", enable_dlpack=enable_dlpack
            )
            return y

        # Use provided shape and dtype if available, otherwise run test
        if output_shape is not None and output_dtype is not None:
            if isinstance(output_shape, Sequence) and not isinstance(
                output_shape[0], int
            ):
                # Multiple outputs case
                out_shape = tuple(
                    jax.ShapeDtypeStruct(s, d)
                    for s, d in zip(output_shape, output_dtype)
                )
            else:
                # Single output case
                out_shape = jax.ShapeDtypeStruct(output_shape, output_dtype)  # type: ignore
        else:
            # Get expected output shape by running function once
            test_out = wrapped_fun(*x)
            if isinstance(test_out, tuple):
                # Multiple outputs case
                out_shape = tuple(
                    jax.ShapeDtypeStruct(
                        t.shape if hasattr(t, "shape") else (),
                        t.dtype if hasattr(t, "dtype") else x[0].dtype,
                    )
                    for t in test_out
                )
            else:
                # Single output case
                out_shape = jax.ShapeDtypeStruct(  # type: ignore
                    test_out.shape if hasattr(test_out, "shape") else (),
                    test_out.dtype if hasattr(test_out, "dtype") else x[0].dtype,
                )

        # Use pure_callback with correct output shape
        result = jax.pure_callback(wrapped_fun, out_shape, *x)
        return result

    return fun_jax


def jax_interface(
    fun: Callable[..., Any],
    jit: bool = False,
    enable_dlpack: bool = False,
    output_shape: Optional[Union[Tuple[int, ...], Tuple[()]]] = None,
    output_dtype: Optional[Any] = None,
) -> Callable[..., Any]:
    """
    Wrap a function on different ML backend with a jax interface.

    :Example:

    .. code-block:: python

        tc.set_backend("tensorflow")

        def f(params):
            c = tc.Circuit(1)
            c.rx(0, theta=params[0])
            c.ry(0, theta=params[1])
            return tc.backend.real(c.expectation([tc.gates.z(), [0]]))

        f = tc.interfaces.jax_interface(f, jit=True)

        params = jnp.ones(2)
        value, grad = jax.value_and_grad(f)(params)

    :param fun: The quantum function with tensor in and tensor out
    :type fun: Callable[..., Any]
    :param jit: whether to jit ``fun``, defaults to False
    :type jit: bool, optional
    :param enable_dlpack: whether transform tensor backend via dlpack, defaults to False
    :type enable_dlpack: bool, optional
    :param output_shape: Optional shape of the function output, defaults to None
    :type output_shape: Optional[Union[Tuple[int, ...], Tuple[()]]], optional
    :param output_dtype: Optional dtype of the function output, defaults to None
    :type output_dtype: Optional[Any], optional
    :return: The same quantum function but now with jax array in and jax array out
        while AD is also supported
    :rtype: Callable[..., Any]
    """
    jax_fun = create_jax_function(
        fun,
        enable_dlpack=enable_dlpack,
        jit=jit,
        output_shape=output_shape,
        output_dtype=output_dtype,
    )
    return jax_fun


def create_jax_function(
    fun: Callable[..., Any],
    enable_dlpack: bool = False,
    jit: bool = False,
    output_shape: Optional[Union[Tuple[int, ...], Tuple[()]]] = None,
    output_dtype: Optional[Any] = None,
) -> Callable[..., Any]:
    import jax
    from jax import custom_vjp

    if jit:
        fun = backend.jit(fun)

    wrapped = jax_wrapper(
        fun,
        enable_dlpack=enable_dlpack,
        output_shape=output_shape,
        output_dtype=output_dtype,
    )

    @custom_vjp
    def f(*x: Any) -> Any:
        return wrapped(*x)

    def f_fwd(*x: Any) -> Tuple[Any, Tuple[Any, ...]]:
        y = wrapped(*x)
        return y, x

    def f_bwd(res: Tuple[Any, ...], g: Any) -> Tuple[Any, ...]:
        x = res

        if len(x) == 1:
            x = x[0]

        vjp_fun = partial(backend.vjp, fun)
        if jit:
            vjp_fun = backend.jit(vjp_fun)  # type: ignore

        def vjp_wrapped(args: Any) -> Any:
            args = general_args_to_backend(args, enable_dlpack=enable_dlpack)
            gb = general_args_to_backend(g, enable_dlpack=enable_dlpack)
            r = vjp_fun(args, gb)[1]
            r = general_args_to_backend(
                r, target_backend="jax", enable_dlpack=enable_dlpack
            )
            return r

        # Handle gradient shape for both single input and tuple inputs
        if isinstance(x, tuple):
            # Create a tuple of ShapeDtypeStruct for each input
            grad_shape = tuple(jax.ShapeDtypeStruct(xi.shape, xi.dtype) for xi in x)
        else:
            grad_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

        dx = jax.pure_callback(
            vjp_wrapped,
            grad_shape,
            x,
        )

        if not isinstance(dx, tuple):
            dx = (dx,)
        return dx  # type: ignore

    f.defvjp(f_fwd, f_bwd)
    return f
