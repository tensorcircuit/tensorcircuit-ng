"""
demonstration of vmap + grad like API
"""

import tensorcircuit as tc

# See issues in https://github.com/tencent-quantum-lab/tensorcircuit/issues/229#issuecomment-2600773780

for backend in ["tensorflow", "jax"]:
    with tc.runtime_backend(backend) as K:
        L = 2
        inputs = K.cast(K.ones([3, 2]), tc.rdtypestr)
        weights = K.cast(K.ones([2]), tc.rdtypestr)

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

        # f_vmap = K.vmap(f, vectorized_argnums=0)

        print("grad", K.grad(f)(inputs[0], weights))
        print("vmap", K.vmap(f)(inputs, weights))
        print("vmap over grad", K.vmap(K.grad(f))(inputs, weights))
        # wrong in tf due to https://github.com/google/TensorNetwork/issues/940
        # https://github.com/tensorflow/tensorflow/issues/52148
        print("vmap over jacfwd", K.vmap(K.jacfwd(f))(inputs, weights))
        print("jacfwd over vmap", K.jacfwd(K.vmap(f))(inputs, weights))
        r = K.vmap(K.jacrev(f))(inputs, weights)
        print("vmap over jacrev", r)
        # wrong in tf
        r = K.jacrev(K.vmap(f))(inputs, weights)
        print("jacrev over vmap", r)
        r = K.vmap(K.hessian(f))(inputs, weights)
        print("vmap over hess", r)
        # wrong in tf
        r = K.hessian(K.vmap(f))(inputs, weights)
        print("hess over vmap", r)

# lessons: never put vmap outside gradient function in tf
