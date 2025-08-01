import time
import optax
import tensorcircuit as tc

tc.set_dtype("complex64")
K = tc.set_backend("jax")
# jax is more GPU friendly for this task,
# while tf is more CPU efficient for this task
tc.set_contractor("cotengra-1024-8192")
# more trials for better contraction path

n, m, nlayers = 4, 4, 6
coord = tc.templates.graphs.Grid2DCoord(n, m)
h = tc.quantum.heisenberg_hamiltonian(
    coord.lattice_graph(), hzz=0.8, hxx=1.0, hyy=1.0, sparse=True
)


def singlet_init(circuit):  # assert n % 2 == 0
    nq = circuit._nqubits
    for i in range(0, nq - 1, 2):
        j = (i + 1) % nq
        circuit.X(i)
        circuit.H(i)
        circuit.cnot(i, j)
        circuit.X(j)
    return circuit


def vqe_forward(param):
    c = tc.Circuit(n * m)
    c = singlet_init(c)
    for i in range(nlayers):
        c = tc.templates.blocks.Grid2D_entangling(
            c, coord, tc.gates._zz_matrix, param[i, 0]
        )
        c = tc.templates.blocks.Grid2D_entangling(
            c, coord, tc.gates._xx_matrix, param[i, 1]
        )
        c = tc.templates.blocks.Grid2D_entangling(
            c, coord, tc.gates._yy_matrix, param[i, 2]
        )
    loss = tc.templates.measurements.operator_expectation(c, h)
    return loss


# vgf = tc.backend.jit(
#     tc.backend.value_and_grad(vqe_forward),
# )
# param = tc.backend.implicit_randn(stddev=0.1, shape=[nlayers, 3, 2 * n * m])


# for _ in range(5):
#     time0 = time.time()
#     loss, gr = vgf(param)
#     print(loss)
#     print(time.time()-time0)

vvgf = K.jit(
    K.vectorized_value_and_grad(vqe_forward),
)
param = tc.backend.implicit_randn(stddev=0.02, shape=[10, nlayers, 3, 2 * n * m])

optimizer = optax.adam(learning_rate=3e-3)
opt_state = optimizer.init(param)


@K.jit
def train_step(param, opt_state):
    # always using jitted optax paradigm when running on GPU!
    loss_val, grads = vvgf(param)
    updates, opt_state = optimizer.update(grads, opt_state, param)
    param = optax.apply_updates(param, updates)
    return param, opt_state, loss_val


for _ in range(1000):
    time0 = time.time()
    param, opt_state, losses = train_step(param, opt_state)
    print(K.mean(losses), time.time() - time0)
    # ~0.017s per iteration on A800
