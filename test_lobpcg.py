import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

jax.config.update('jax_enable_x64', True)

def test_complex():
    N = 6
    A = jnp.diag(jnp.arange(N, dtype=jnp.complex128))

    def matvec(x):
        return -A @ x

    v0 = jax.random.normal(jax.random.PRNGKey(0), (N, 1), dtype=jnp.float64).astype(jnp.complex128)
    # The error "The input carry component state[5] has type complex128[1,1] but the corresponding output carry component has type float64[1,1]"
    # This happens because the initial eigenvalue array (or some state) is complex128, but gets updated with float64.
    # We can try to explicitly cast the output of lobpcg_standard. But we can't patch lobpcg_standard internals easily.
    # Wait, the eigenvalues of a Hermitian matrix ARE real.
    # LOBPCG internally does rayleigh quotient `V.T @ A @ V`. For complex, it should be `V.conj().T @ A @ V`.
    # Let's check `lobpcg_standard` code: does it use `v.T.conj()`?

    try:
        evals, evecs, i = lobpcg_standard(matvec, v0, m=20, tol=1e-5)
        print("Lobpcg evals:", -evals)
    except Exception as e:
        print(e)
test_complex()
