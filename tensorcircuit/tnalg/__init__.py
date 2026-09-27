"""Fixed-shape JAX tensor-network algorithms.

The module provides dense and Abelian block-buffer MPS states together with
fixed-shape TEBD, one-site TDVP, and one-site DMRG. The ``make_*`` constructors
prepare the static plan for the common case; the ``make_*_from_plan`` variants
accept an explicitly prepared plan when it needs to be inspected or reused.
Wrap the returned step or sweep in ``jax.jit`` and reuse it with tensors that
match the static physical, bond, and sector layout. ``symmetry=None`` is the
dense case; Abelian layouts use fixed sector quotas and never densify in the
numerical hot path.

MPS, MPO, and environment data are tensor-only PyTrees. Checkpoints and
explicit legacy MPS/MPO conversions are available at the I/O boundary. The
benchmark examples compare one complete step or sweep with TeNPy and report
lowering, compilation, first execution, steady execution, and numerical
agreement separately.
"""

from .environment import EnvironmentState
from .layout import (
    BlockLayout,
    GateSpec,
    MPOSpec,
    MPSSpec,
)
from .interop import to_mpscircuit, to_tn_mpo, to_tn_mps
from .mps import (
    MPSState,
    as_mps,
    norm,
    overlap,
    product_state,
    random_mps,
)
from .mpo import MPOState, as_mpo, compile_mpo, expectation, variance
from .io import load_checkpoint, save_checkpoint
from .symmetry import AbelianSymmetry, SectorIndex
from .tensor import (
    SymTensor,
    FusionPlan,
    adjoint,
    conjugate,
    fuse_array,
    fuse_indices,
    from_dense,
    inner,
    site_view,
    to_dense,
    transpose,
    unfuse_array,
)
from .algorithms.tebd import (
    TEBDOptions,
    compile_tebd_gates,
    make_tebd_step,
    make_tebd_step_from_plan,
    prepare_tebd,
)
from .algorithms.tdvp import (
    TDVPOptions,
    make_tdvp_step,
    make_tdvp_step_from_plan,
    prepare_tdvp,
)
from .algorithms.dmrg import (
    DMRGOptions,
    make_dmrg_sweep,
    make_dmrg_sweep_from_plan,
    prepare_dmrg,
)

__all__ = [
    "DMRGOptions",
    "AbelianSymmetry",
    "BlockLayout",
    "EnvironmentState",
    "GateSpec",
    "FusionPlan",
    "MPOSpec",
    "MPOState",
    "MPSState",
    "MPSSpec",
    "TEBDOptions",
    "TDVPOptions",
    "SectorIndex",
    "SymTensor",
    "as_mps",
    "as_mpo",
    "adjoint",
    "compile_tebd_gates",
    "compile_mpo",
    "expectation",
    "variance",
    "conjugate",
    "from_dense",
    "fuse_array",
    "fuse_indices",
    "inner",
    "load_checkpoint",
    "make_tebd_step",
    "make_tebd_step_from_plan",
    "make_tdvp_step",
    "make_tdvp_step_from_plan",
    "make_dmrg_sweep",
    "make_dmrg_sweep_from_plan",
    "norm",
    "overlap",
    "prepare_tebd",
    "prepare_tdvp",
    "prepare_dmrg",
    "product_state",
    "random_mps",
    "to_mpscircuit",
    "to_tn_mpo",
    "site_view",
    "save_checkpoint",
    "to_dense",
    "to_tn_mps",
    "transpose",
    "unfuse_array",
]
