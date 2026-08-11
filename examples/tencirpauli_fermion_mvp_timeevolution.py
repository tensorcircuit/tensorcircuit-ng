"""
Two-dimensional Fermi-Hubbard evolution with a TenCirPauli MVP.

This short preview uses TensorCircuit-NG's Krylov time-evolution routine with
TenCirPauli's charge- and spin-restricted FermionOperator. For the complete
2D Fermi-Hubbard study and backend/native comparisons, see:
https://github.com/tensorcircuit/TenCirPauli/tree/main/examples/research/fermion_timeevolution_2d
"""

from __future__ import annotations

import numpy as np

import tencirpauli as tcp
import tensorcircuit as tc


def _hubbard_terms() -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    """Return an open-boundary 2x2 spinful Fermi-Hubbard Hamiltonian."""
    sites = 4
    bonds = ((0, 1), (0, 2), (1, 3), (2, 3))
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for site in range(sites):
        up, down = site, sites + site
        terms.append(
            (
                (
                    (up, "create"),
                    (up, "annihilate"),
                    (down, "create"),
                    (down, "annihilate"),
                ),
                4.0,
            )
        )
    for left, right in bonds:
        for spin in (0, 1):
            left_mode, right_mode = left + spin * sites, right + spin * sites
            terms.extend(
                (
                    (((left_mode, "create"), (right_mode, "annihilate")), -1.0),
                    (((right_mode, "create"), (left_mode, "annihilate")), -1.0),
                )
            )
    return terms


def main() -> None:
    """
    Evolve a charge- and spin-conserving state with TensorCircuit Krylov.
    """
    tc.set_backend("numpy")
    tc.set_dtype("complex128")

    space = tcp.OperatorSpace(fermions=8)
    number = tcp.AdditiveCharge(space, fermions={mode: 1 for mode in range(8)})
    spin = tcp.AdditiveCharge(
        space,
        name="2Sz",
        fermions={mode: (1 if mode < 4 else -1) for mode in range(8)},
    )
    sector = tcp.ChargeSector(((number, 4), (spin, 0)))
    operator = tcp.FermionOperator.from_terms(8, _hubbard_terms())
    restricted = operator.restrict_charge(sector)
    mvp = restricted.mvp_plan().apply

    occupations = (1, 0, 1, 0, 0, 1, 0, 1)
    state = np.zeros(sector.dimension, dtype=np.complex128)
    state[sector.rank(occupations)] = 1.0
    times = np.asarray([0.0, 0.1, 0.2], dtype=np.float64)
    trajectory = tc.timeevol.krylov_evol(
        mvp, state, times, subspace_dimension=min(12, sector.dimension)
    )

    print(f"restricted sector dimension: {sector.dimension}")
    print(f"trajectory shape: {np.asarray(trajectory).shape}")
    print(f"initial energy: {np.vdot(state, mvp(state)).real:.8f}")


if __name__ == "__main__":
    main()
