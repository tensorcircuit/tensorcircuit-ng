"""
demonstration of vmap + grad like API
"""

import sys

sys.path.insert(0, "../")

import tensorcircuit as tc

# See issues in https://github.com/tencent-quantum-lab/tensorcircuit/issues/229#issuecomment-2600773780

for backend in ["tensorflow", "jax"]:
    with tc.runtime_backend(backend) as K:
        L = 2
        inputs = K.cast(K.ones([3, 2]), tc.rdtypestr)
        weights = K.cast(K.ones([3, 2]), tc.rdtypestr)

        def ansatz(thetas, alpha):
            c = tc.Circuit(L)
            for j in range(2):
                for i in range(L):
                    c.rx(i, theta=thetas[j])
                    c.ry(i, theta=alpha[j])
                for i in range(L - 1):
                    c.cnot(i, i + 1)
            return c

        def f(thetas, alpha):
            c = ansatz(thetas, alpha)
            observables = K.stack([K.real(c.expectation_ps(z=[i])) for i in range(L)])
            return K.mean(observables)

        print("grad_0", K.grad(f)(inputs[0], weights[0]))
        print("grad_1", K.grad(f, argnums=1)(inputs[0], weights[0]))
        print("vmap_0", K.vmap(f)(inputs, weights[0]))
        print("vmap_1", K.vmap(f, vectorized_argnums=1)(inputs[0], weights))
        print("vmap over grad_0", K.vmap(K.grad(f))(inputs, weights[0]))
        # wrong in tf due to https://github.com/google/TensorNetwork/issues/940
        # https://github.com/tensorflow/tensorflow/issues/52148
        print("vmap over grad_1", K.vmap(K.grad(f, argnums=1))(inputs, weights[0]))
        # wrong in tf
        print("vmap over jacfwd_0", K.vmap(K.jacfwd(f))(inputs, weights[0]))
        print("jacfwd_0 over vmap", K.jacfwd(K.vmap(f))(inputs, weights[0]))
        print("vmap over jacfwd_1", K.vmap(K.jacfwd(f, argnums=1))(inputs, weights[0]))
        print("jacfwd_1 over vmap", K.jacfwd(K.vmap(f), argnums=1)(inputs, weights[0]))
        r = K.vmap(K.jacrev(f))(inputs, weights[0])
        print("vmap over jacrev0", r)
        # wrong in tf
        r = K.jacrev(K.vmap(f))(inputs, weights[0])
        print("jacrev0 over vmap", r)
        r = K.vmap(K.jacrev(f, argnums=1))(inputs, weights[0])
        print("vmap over jacrev1", r)
        # wrong in tf
        r = K.jacrev(K.vmap(f), argnums=1)(inputs, weights[0])
        print("jacrev1 over vmap", r)
        r = K.vmap(K.jacrev(f, argnums=1), vectorized_argnums=1)(inputs[0], weights)
        print("vmap1 over jacrev1", r)
        r = K.jacrev(K.vmap(f, vectorized_argnums=1), argnums=1)(inputs[0], weights)
        print("jacrev1 over vmap1", r)
        r = K.vmap(K.hessian(f))(inputs, weights[0])
        print("vmap over hess0", r)
        # wrong in tf
        r = K.hessian(K.vmap(f))(inputs, weights[0])
        print("hess0 over vmap", r)
        r = K.vmap(K.hessian(f, argnums=1))(inputs, weights[0])
        print("vmap over hess1", r)
        # wrong in tf
        r = K.hessian(K.vmap(f), argnums=1)(inputs, weights[0])
        print("hess1 over vmap", r)

# lessons: never put vmap outside gradient function in tf
