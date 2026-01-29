"""
Comprehensive Demonstration of TensorCircuit Interfaces with Dynamic Backend Switching
==================================================================================

This script comprehensively demonstrates how to use TensorCircuit's interface functions
(`torch_interface`, `jax_interface`, `tf_interface`) to integrate TensorCircuit's quantum
functions into different Deep Learning frameworks (PyTorch, JAX, TensorFlow).

We define a single, backend-agnostic quantum function `universal_circuit` and reuse it
across different interface-backend combinations utilizing `tc.runtime_backend()`.
"""

import numpy as np
import torch
import tensorflow as tf
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def separation_line(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------
# 1. Universal Quantum Function Definition
# -----------------------------------------------------------------------
# This function is backend-agnostic. It assumes `tc.backend` is set correctly
# in the context where it is executed (or traced).


def universal_circuit(params):
    """
    A unified quantum circuit function.

    Args:
        params: Tensor of shape (2,) compatible with the current backend.

    Returns:
        Scalar real tensor representing the Z-expectation of qubit 0.
    """
    c = tc.Circuit(2)
    c.h(0)
    c.h(1)

    # Parameterized gates
    c.rx(0, theta=params[0])
    c.ry(1, theta=params[1])

    # Entanglement
    c.cx(0, 1)

    # Measurement: Expectation of Z on qubit 0
    # IMPORTANT: We return the real part to ensure compatibility with all interfaces
    # (some gradients interfaces expect real scalars).
    return tc.backend.real(c.expectation([tc.gates.x(), [0]]))


# -----------------------------------------------------------------------
# 2. PyTorch Interface Demo
# -----------------------------------------------------------------------


def demo_torch_interface():
    separation_line("Part 1: PyTorch Interface Tests")
    print(
        "Goal: Wrap generic TC legacy/other-backend functions as PyTorch Functions.\n"
    )

    # We will test wrapping the circuit running on different internal backends
    backends_to_test = ["jax", "tensorflow"]

    for bk in backends_to_test:
        print(f"--- Testing PyTorch Interface with internal backend: '{bk}' ---")

        with tc.runtime_backend(bk):
            # 1. Wrap the quantum function
            # jit=True is highly recommended for performance
            qcircuit_torch = tc.interfaces.torch_interface(
                universal_circuit, jit=True, enable_dlpack=True
            )

            # 2. Integrate into a PyTorch optimization loop
            # Initialize parameters
            params_torch = torch.tensor(
                [0.1, 0.1], requires_grad=True, dtype=torch.float32
            )

            initial_loss = qcircuit_torch(params_torch).item()
            print(f"    Initial Loss: {initial_loss:.4f}")

            loss = qcircuit_torch(params_torch)
            loss.backward()

            print(f"    Loss:   {loss:.4f}")
            print(f"    gradient: {params_torch.grad.numpy()}")


# -----------------------------------------------------------------------
# 3. JAX Interface Demo
# -----------------------------------------------------------------------


def demo_jax_interface():
    separation_line("Part 2: JAX Interface Tests")
    print(
        "Goal: Wrap generic TC functions as JAX compatible functions (jit/grad/vmap).\n"
    )

    backends_to_test = ["tensorflow", "pytorch"]

    for bk in backends_to_test:
        print(f"--- Testing JAX Interface with internal backend: '{bk}' ---")

        with tc.runtime_backend(bk):
            # 1. Wrap the quantum function
            qcircuit_jax_wrapped = tc.interfaces.jax_interface(
                universal_circuit,
                jit=True,
                output_shape=(),
                output_dtype=jnp.float32,
                enable_dlpack=True,
            )
            # better specify the output dtype and shape
            # to avoid precomputation especially when the function wrapped is heavy

            # 2. Use JAX transformations (value_and_grad)
            params_jax = jnp.array([0.1, 0.1])

            # Define a loss function
            def loss_fn(p):
                return qcircuit_jax_wrapped(p)

            print("  Checking JAX value_and_grad execution...")
            val, grads = jax.value_and_grad(loss_fn)(params_jax)

            print(f"    Value: {val:.4f}")
            print(f"    Grads: {grads}")


# -----------------------------------------------------------------------
# 4. TensorFlow Interface Demo
# -----------------------------------------------------------------------


def demo_tf_interface():
    separation_line("Part 3: TensorFlow Interface Tests")
    print("Goal: Wrap generic TC functions as TensorFlow Layers/Functions.\n")

    backends_to_test = ["jax", "pytorch"]

    for bk in backends_to_test:
        print(f"--- Testing TensorFlow Interface with internal backend: '{bk}' ---")

        with tc.runtime_backend(bk):
            # 1. Wrap the function
            # TF interface requires explicit output dtype usually
            qcircuit_tf_wrapped = tc.interfaces.tf_interface(
                universal_circuit, ydtype=tf.float32, jit=True
            )

            # 2. Use with TensorFlow GradientTape
            params_tf = tf.Variable(np.array([0.1, 0.1], dtype=np.float32))

            print("  Checking TensorFlow GradientTape execution...")
            with tf.GradientTape() as tape:
                tape.watch(params_tf)
                val = qcircuit_tf_wrapped(params_tf)

            grads = tape.gradient(val, params_tf)

            print(f"    Value: {val.numpy():.4f}")
            print(f"    Grads: {grads.numpy()}")


# -----------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------


def main():
    print("Starting TensorCircuit Interface Demonstration...")

    # Run all demos
    demo_torch_interface()
    demo_jax_interface()
    demo_tf_interface()


if __name__ == "__main__":
    main()
