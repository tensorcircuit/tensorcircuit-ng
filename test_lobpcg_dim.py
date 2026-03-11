import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

jax.config.update('jax_enable_x64', True)

def test_complex():
    N = 6
    A = jnp.diag(jnp.arange(N, dtype=jnp.complex128))

    def matvec(x):
        print("matvec got x shape:", x.shape)
        # x might be (N, k) where k>1 because LOBPCG operates on a subspace block?
        return -A @ x

    v0 = jax.random.normal(jax.random.PRNGKey(0), (N, 1), dtype=jnp.float64).astype(jnp.complex128)

    evals, evecs, i = lobpcg_standard(matvec, v0, m=20, tol=1e-5)
test_complex()
