"""
QAOA portfolio selection with fund NAV data fetched by xalpha.

The example chooses one representative fund for each requested exposure:

- 110020: 易方达沪深300ETF联接A
- 090010: 大成中证红利指数A
- 110026: 易方达创业板ETF联接A
- 000307: 易方达黄金ETF联接A
- 270042: 广发纳斯达克100ETF联接人民币(QDII)A
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from typing import List, Sequence, Tuple

import numpy as np
import optax
import pandas as pd
import xalpha as xa

import tensorcircuit as tc
from tensorcircuit.applications.finance.portfolio import QUBO_from_portfolio
from tensorcircuit.templates.conversions import QUBO_to_Ising

K = tc.set_backend("jax")

pd.set_option("display.width", 160)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)


@dataclass(frozen=True)
class FundSpec:
    """Description of one fund exposure used in the portfolio QUBO."""

    code: str
    label: str
    exposure: str


@dataclass(frozen=True)
class SelectionSummary:
    selected_funds: List[FundSpec]
    selected_count: int
    objective_score: float
    annual_return_sum: float
    risk_term: float
    equal_weight_return: float
    equal_weight_volatility: float
    sharpe_like_ratio: float


DEFAULT_FUNDS = (
    FundSpec("110020", "易方达沪深300ETF联接A", "沪深300"),
    FundSpec("090010", "大成中证红利指数A", "红利指数"),
    FundSpec("110026", "易方达创业板ETF联接A", "创业板指"),
    FundSpec("000307", "易方达黄金ETF联接A", "黄金"),
    FundSpec("270042", "广发纳斯达克100ETF联接人民币(QDII)A", "纳指"),
)

X_MIXER = "X"
XY_MIXER = "XY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small TensorCircuit QAOA portfolio example using xalpha data."
    )
    parser.add_argument(
        "--start",
        default="2021-01-01",
        help="Start date for aligned NAV history, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional end date for aligned NAV history, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2,
        help="Number of funds to select.",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.5,
        help="Risk-aversion coefficient q in q * covariance - return.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=2.0,
        help="Quadratic penalty for violating the budget constraint.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="QAOA depth.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Adam optimization steps per random restart.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        help="Adam learning rate for QAOA parameters.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=8,
        help="Number of random QAOA parameter restarts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for QAOA parameter initialization.",
    )
    return parser.parse_args()


def load_nav_history(
    funds: Sequence[FundSpec], start: str, end: str | None
) -> np.ndarray:
    """Fetch adjusted NAV histories and return an aligned price matrix."""
    series = []

    for fund in funds:
        info = xa.fundinfo(fund.code, fetch=True, priceonly=True)
        price = info.price.copy()
        if "totvalue" not in price.columns:
            raise KeyError(f"xalpha price data for {fund.code} has no totvalue column")

        fund_series = price.set_index("date")["totvalue"].sort_index()
        fund_series = (
            fund_series.loc[start:end] if end is not None else fund_series.loc[start:]
        )
        fund_series.name = fund.code
        series.append(fund_series)

    navs = pd.concat(series, axis=1, join="inner").dropna().to_numpy(dtype=float)
    if navs.shape[0] < 60:
        raise ValueError(
            f"Only {navs.shape[0]} common NAV observations were found; "
            "choose a wider date range."
        )
    return navs


def annualized_return_and_covariance(
    navs: np.ndarray, trading_days: int = 252
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute annualized geometric return and covariance from aligned NAVs."""
    daily_returns = navs[1:] / navs[:-1] - 1.0
    n_days = daily_returns.shape[0]
    mean = np.prod(1.0 + daily_returns, axis=0) ** (trading_days / n_days) - 1.0
    covariance = np.cov(daily_returns, rowvar=False) * trading_days
    return mean, covariance


def ring_pairs(nqubits: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(nqubits - 1)] + [(nqubits - 1, 0)]


def build_qaoa_circuit(
    params: tc.Tensor,
    nlayers: int,
    pauli_terms: Sequence[Sequence[float]],
    weights: Sequence[float],
    mixer: str,
    initial_bits: np.ndarray | None = None,
) -> tc.Circuit:
    """Build a QAOA circuit for a diagonal Ising Hamiltonian."""
    nqubits = len(pauli_terms[0])
    circuit = tc.Circuit(nqubits)
    if initial_bits is None:
        for i in range(nqubits):
            circuit.h(i)
    else:
        for i, bit in enumerate(initial_bits):
            if bit:
                circuit.x(i)

    for layer in range(nlayers):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]

        for term, weight in zip(pauli_terms, weights):
            qubits = [i for i, value in enumerate(term) if value == 1]
            if len(qubits) == 1:
                circuit.rz(qubits[0], theta=2.0 * weight * gamma)
            elif len(qubits) == 2:
                circuit.rzz(qubits[0], qubits[1], theta=2.0 * weight * gamma)
            else:
                raise ValueError(f"Unsupported Ising term with {len(qubits)} Z factors")

        if mixer == X_MIXER:
            for i in range(nqubits):
                circuit.rx(i, theta=2.0 * beta)
        elif mixer == XY_MIXER:
            for i, j in ring_pairs(nqubits):
                circuit.rxx(i, j, theta=2.0 * beta)
                circuit.ryy(i, j, theta=2.0 * beta)
        else:
            raise ValueError(f"Unsupported mixer: {mixer}")

    return circuit


def ising_expectation(
    circuit: tc.Circuit,
    pauli_terms: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> tc.Tensor:
    loss = 0.0
    for term, weight in zip(pauli_terms, weights):
        qubits = [i for i, value in enumerate(term) if value == 1]
        loss += weight * circuit.expectation_ps(z=qubits)
    return K.real(loss)


def optimize_qaoa(
    pauli_terms: Sequence[Sequence[float]],
    weights: Sequence[float],
    offset: float,
    nlayers: int,
    steps: int,
    learning_rate: float,
    restarts: int,
    seed: int,
    mixer: str,
    initial_bits: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    if restarts < 1:
        raise ValueError("restarts must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")

    rng = np.random.default_rng(seed)

    def objective(params: tc.Tensor) -> tc.Tensor:
        circuit = build_qaoa_circuit(
            params, nlayers, pauli_terms, weights, mixer, initial_bits
        )
        return ising_expectation(circuit, pauli_terms, weights) + offset

    initial = rng.normal(size=(restarts, 2 * nlayers))
    params = K.convert_to_tensor(initial)
    periods = K.convert_to_tensor(np.tile([2.0 * np.pi, np.pi], nlayers))
    optimizer = K.optimizer(optax.adam(learning_rate))
    value_and_grad = K.jit(K.vvag(objective, argnums=0, vectorized_argnums=0))

    for _ in range(steps):
        _, gradients = value_and_grad(params)
        params = optimizer.update(gradients, params)
        params = K.mod(params, periods)

    value_tensor, _ = value_and_grad(params)
    values = K.numpy(value_tensor)
    best_index = int(np.argmin(values))
    return K.numpy(params)[best_index].astype(float), float(values[best_index])


def fixed_weight_initial_bits(nbits: int, budget: int) -> np.ndarray:
    bits = np.zeros(nbits, dtype=int)
    bits[:budget] = 1
    return bits


def bitstring_to_array(index: int, nbits: int) -> np.ndarray:
    return np.fromiter((int(bit) for bit in format(index, f"0{nbits}b")), dtype=int)


def qubo_cost(bits: np.ndarray, qmatrix: np.ndarray) -> float:
    return float(bits @ qmatrix @ bits)


def brute_force(qmatrix: np.ndarray, budget: int) -> Tuple[np.ndarray, float]:
    nbits = qmatrix.shape[0]
    best_bits = None
    best_value = np.inf
    for bits_tuple in product((0, 1), repeat=nbits):
        bits = np.asarray(bits_tuple, dtype=int)
        if int(bits.sum()) != budget:
            continue
        value = qubo_cost(bits, qmatrix)
        if value < best_value:
            best_bits = bits
            best_value = value
    if best_bits is None:
        raise ValueError("No feasible bitstring matches the requested budget")
    return best_bits, best_value


def summarize_selection(
    bits: np.ndarray,
    funds: Sequence[FundSpec],
    mean: np.ndarray,
    covariance: np.ndarray,
    qmatrix: np.ndarray,
    budget: int,
) -> SelectionSummary:
    selected_funds = [fund for bit, fund in zip(bits, funds) if bit == 1]
    selected_count = int(bits.sum())
    return_sum = float(bits @ mean)
    variance_sum = float(bits @ covariance @ bits)
    if selected_count == 0:
        equal_weight_return = 0.0
        equal_weight_volatility = 0.0
        sharpe_like_ratio = 0.0
    else:
        equal_weight_return = return_sum / selected_count
        equal_weight_volatility = float(np.sqrt(variance_sum)) / selected_count
        sharpe_like_ratio = (
            np.inf
            if equal_weight_volatility == 0.0
            else equal_weight_return / equal_weight_volatility
        )
    return SelectionSummary(
        selected_funds=selected_funds,
        selected_count=selected_count,
        objective_score=qubo_cost(bits, qmatrix),
        annual_return_sum=return_sum,
        risk_term=variance_sum,
        equal_weight_return=equal_weight_return,
        equal_weight_volatility=equal_weight_volatility,
        sharpe_like_ratio=sharpe_like_ratio,
    )


def fund_table(funds: Sequence[FundSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exposure": [fund.exposure for fund in funds],
            "code": [fund.code for fund in funds],
            "fund": [fund.label for fund in funds],
        }
    )


def bits_to_string(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits)


def percent(value: float) -> str:
    return f"{value:.2%}"


def print_asset_table(
    funds: Sequence[FundSpec], mean: np.ndarray, covariance: np.ndarray
) -> None:
    volatility = np.sqrt(np.diag(covariance))
    print("Asset universe from aligned xalpha NAV data")
    table = fund_table(funds)
    table["ann_return"] = mean
    table["ann_volatility"] = volatility
    print(
        table.to_string(
            index=False,
            formatters={
                "ann_return": percent,
                "ann_volatility": percent,
            },
        )
    )
    print()


def print_correlation_snapshot(
    exposures: Sequence[str], covariance: np.ndarray
) -> None:
    volatility = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(volatility, volatility)
    print("Correlation snapshot")
    table = pd.DataFrame(correlation, index=exposures, columns=exposures)
    print(table.to_string(float_format=lambda value: f"{value:.2f}"))
    print()


def print_model_settings(args: argparse.Namespace) -> None:
    print("Optimization setup")
    table = pd.DataFrame(
        [
            ("target holdings", f"{args.budget} funds"),
            ("risk-aversion q", f"{args.risk_aversion:.3f}"),
            ("QAOA depth", f"p={args.layers}"),
            ("random restarts", str(args.restarts)),
            ("optimizer", "optax Adam"),
            ("Adam steps", str(args.steps)),
            ("Adam learning rate", f"{args.learning_rate:.4f}"),
        ],
        columns=["setting", "value"],
    )
    print(table.to_string(index=False))
    print()


def print_selection(
    title: str, bits: np.ndarray, summary: SelectionSummary, budget: int
) -> None:
    status = "feasible" if summary.selected_count == budget else "budget mismatch"
    print(title)
    metrics = pd.DataFrame(
        [
            ("bitstring", bits_to_string(bits)),
            ("holdings", f"{summary.selected_count}/{budget}"),
            ("constraint_status", status),
            ("equal_weight_return", percent(summary.equal_weight_return)),
            ("equal_weight_volatility", percent(summary.equal_weight_volatility)),
            ("return_volatility_ratio", f"{summary.sharpe_like_ratio:.3f}"),
            ("selected_return_sum", f"{summary.annual_return_sum:.6f}"),
            ("selected_variance_term", f"{summary.risk_term:.6f}"),
            ("objective_score", f"{summary.objective_score:.6f}"),
        ],
        columns=["metric", "value"],
    )
    print(metrics.to_string(index=False))
    if summary.selected_funds:
        print("selected funds:")
        for fund in summary.selected_funds:
            print(f"  - {fund.exposure} | {fund.code} | {fund.label}")
    print()


def print_model_results(
    rows: Sequence[Tuple[str, float, float, float]], exact_cost: float
) -> None:
    print("Model results")
    table = pd.DataFrame(
        [
            {
                "method": method,
                "training_expectation": training_value,
                "selected_objective": selected_value,
                "selected_gap": selected_value - exact_cost,
                "mode_probability": mode_probability,
            }
            for method, training_value, selected_value, mode_probability in rows
        ]
    )
    print(
        table.to_string(
            index=False,
            formatters={
                "training_expectation": lambda value: f"{value:.6f}",
                "selected_objective": lambda value: f"{value:.6f}",
                "selected_gap": lambda value: f"{value:.6f}",
                "mode_probability": lambda value: f"{value:.3f}",
            },
        )
    )
    print(f"Best feasible brute-force objective: {exact_cost:.6f}")
    print()


def print_top_samples(
    title: str,
    probabilities: np.ndarray,
    qmatrix: np.ndarray,
    budget: int,
    topk: int = 5,
) -> None:
    rows = []
    for index in np.argsort(probabilities)[-topk:][::-1]:
        bits = bitstring_to_array(int(index), qmatrix.shape[0])
        rows.append(
            {
                "bits": bits_to_string(bits),
                "probability": probabilities[index],
                "holdings": int(bits.sum()),
                "feasible": int(bits.sum()) == budget,
                "objective_score": qubo_cost(bits, qmatrix),
            }
        )
    table = pd.DataFrame(rows)
    print(title)
    print(
        table.to_string(
            index=False,
            formatters={
                "probability": lambda value: f"{value:.3f}",
                "feasible": lambda value: "yes" if value else "no",
                "objective_score": lambda value: f"{value:.6f}",
            },
        )
    )
    print()


def main() -> None:
    args = parse_args()
    if args.budget < 1 or args.budget > len(DEFAULT_FUNDS):
        raise ValueError("budget must be between 1 and the number of funds")

    navs = load_nav_history(DEFAULT_FUNDS, args.start, args.end)
    mean, covariance = annualized_return_and_covariance(navs)
    exposures = [fund.exposure for fund in DEFAULT_FUNDS]
    print(f"Common NAV observations: {navs.shape[0]} from {args.start} onward")
    print()
    print_asset_table(DEFAULT_FUNDS, mean, covariance)
    print_correlation_snapshot(exposures, covariance)
    print_model_settings(args)

    objective_qmatrix = QUBO_from_portfolio(
        covariance,
        mean,
        q=args.risk_aversion,
        B=args.budget,
        t=0.0,
    )
    penalty_qmatrix = QUBO_from_portfolio(
        covariance,
        mean,
        q=args.risk_aversion,
        B=args.budget,
        t=args.penalty,
    )
    penalty_pauli_terms, penalty_weights, penalty_offset = QUBO_to_Ising(
        penalty_qmatrix
    )
    objective_pauli_terms, objective_weights, objective_offset = QUBO_to_Ising(
        objective_qmatrix
    )

    penalty_params, penalty_training_score = optimize_qaoa(
        penalty_pauli_terms,
        penalty_weights,
        float(penalty_offset),
        nlayers=args.layers,
        steps=args.steps,
        learning_rate=args.learning_rate,
        restarts=args.restarts,
        seed=args.seed,
        mixer=X_MIXER,
    )
    penalty_circuit = build_qaoa_circuit(
        penalty_params,
        args.layers,
        penalty_pauli_terms,
        penalty_weights,
        mixer=X_MIXER,
    )
    penalty_probabilities = K.numpy(penalty_circuit.probability()).astype(float)
    penalty_bits = bitstring_to_array(
        int(np.argmax(penalty_probabilities)), len(DEFAULT_FUNDS)
    )
    penalty_summary = summarize_selection(
        penalty_bits, DEFAULT_FUNDS, mean, covariance, objective_qmatrix, args.budget
    )

    initial_bits = fixed_weight_initial_bits(len(DEFAULT_FUNDS), args.budget)
    xy_params, xy_training_score = optimize_qaoa(
        objective_pauli_terms,
        objective_weights,
        float(objective_offset),
        nlayers=args.layers,
        steps=args.steps,
        learning_rate=args.learning_rate,
        restarts=args.restarts,
        seed=args.seed + 17,
        mixer=XY_MIXER,
        initial_bits=initial_bits,
    )
    xy_circuit = build_qaoa_circuit(
        xy_params,
        args.layers,
        objective_pauli_terms,
        objective_weights,
        mixer=XY_MIXER,
        initial_bits=initial_bits,
    )
    xy_probabilities = K.numpy(xy_circuit.probability()).astype(float)
    xy_bits = bitstring_to_array(int(np.argmax(xy_probabilities)), len(DEFAULT_FUNDS))
    xy_summary = summarize_selection(
        xy_bits, DEFAULT_FUNDS, mean, covariance, objective_qmatrix, args.budget
    )

    exact_bits, exact_cost = brute_force(objective_qmatrix, args.budget)
    exact_summary = summarize_selection(
        exact_bits, DEFAULT_FUNDS, mean, covariance, objective_qmatrix, args.budget
    )

    print_model_results(
        [
            (
                "Penalty X-mixer",
                penalty_training_score,
                penalty_summary.objective_score,
                float(np.max(penalty_probabilities)),
            ),
            (
                "Fixed-weight ring XY-mixer",
                xy_training_score,
                xy_summary.objective_score,
                float(np.max(xy_probabilities)),
            ),
        ],
        exact_cost,
    )
    print_selection(
        "Penalty X-mixer QAOA mode", penalty_bits, penalty_summary, args.budget
    )
    print_selection(
        "Fixed-weight ring XY-mixer QAOA mode", xy_bits, xy_summary, args.budget
    )
    print_selection(
        "Best feasible brute-force benchmark", exact_bits, exact_summary, args.budget
    )

    print_top_samples(
        "Penalty X-mixer top samples",
        penalty_probabilities,
        objective_qmatrix,
        args.budget,
    )
    print_top_samples(
        "Fixed-weight ring XY-mixer top samples",
        xy_probabilities,
        objective_qmatrix,
        args.budget,
    )


if __name__ == "__main__":
    main()
