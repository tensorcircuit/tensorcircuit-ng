"""
HS300 volatility-direction classification with TensorCircuit QNN.

This example predicts whether the next-horizon realized variance of HS300
will be larger than the immediately previous-horizon realized variance:

    label = 1[next_horizon_var > prior_horizon_var].

It compares a validation-tuned classical baseline grid with a return-encoded
TensorCircuit QNN.  The default QNN is the best-performing setting found in the
exploratory run: 10 qubits, 6 data-reuploading blocks, and scanned state
evolution for lower JAX staging overhead.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xalpha as xa
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


@dataclass(frozen=True)
class Split:
    """Chronological data split."""

    name: str
    x_qnn: np.ndarray
    x_classical: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    past_var: np.ndarray
    future_var: np.ndarray


@dataclass(frozen=True)
class ModelSelection:
    """Selected model metadata."""

    name: str
    threshold: float
    metrics: Dict[str, float]


def realized_variance(log_returns: np.ndarray) -> float:
    """Mean squared log return."""

    return float(np.mean(np.square(log_returns)))


def qnn_feature(path: np.ndarray, past_returns: np.ndarray, mode: str) -> np.ndarray:
    """Map one lookback window to the 60 angle-encoding inputs."""

    if mode == "returns":
        return np.concatenate([[0.0], past_returns])
    if mode == "path":
        return path - path[0]
    if mode == "demeaned_path":
        return path - np.mean(path)
    if mode == "path_and_returns":
        return 0.5 * (path - path[0]) + 0.5 * np.concatenate([[0.0], past_returns])
    raise ValueError(f"Unsupported feature mode: {mode}")


def make_classical_feature(path: np.ndarray, past_returns: np.ndarray) -> np.ndarray:
    """Classical baseline features from the same 60-day lookback window."""

    normalized_path = path - path[0]
    padded_returns = np.concatenate([[0.0], past_returns])
    recent_20 = past_returns[-min(20, len(past_returns)) :]
    recent_10 = past_returns[-min(10, len(past_returns)) :]
    stats = np.array(
        [
            np.mean(past_returns),
            np.std(past_returns),
            realized_variance(past_returns),
            np.mean(np.abs(past_returns)),
            realized_variance(recent_20),
            realized_variance(recent_10),
            normalized_path[-1],
            normalized_path[-1] - np.max(normalized_path),
            np.max(normalized_path) - np.min(normalized_path),
        ],
        dtype=np.float32,
    )
    return np.concatenate([normalized_path, padded_returns, stats])


def make_dataset(
    price: pd.DataFrame,
    lookback: int,
    horizon: int,
    feature_mode: str,
    classical_raw: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned windows without future leakage in the features."""

    df = price[["date", "netvalue"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["netvalue"] = pd.to_numeric(df["netvalue"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)

    values = df["netvalue"].to_numpy(dtype=float)
    log_values = np.log(values)
    log_returns = np.diff(log_values)

    x_qnn = []
    x_classical = []
    labels = []
    dates = []
    past_vars = []
    future_vars = []

    first_valid_end = max(lookback - 1, horizon)
    for end in range(first_valid_end, len(values) - horizon):
        start = end - lookback + 1
        path = log_values[start : end + 1]
        lookback_returns = log_returns[start:end]
        prior_returns = log_returns[end - horizon : end]
        future_returns = log_returns[end : end + horizon]
        past_var = realized_variance(prior_returns)
        future_var = realized_variance(future_returns)

        qnn_feat = qnn_feature(path, lookback_returns, feature_mode)
        x_qnn.append(qnn_feat)
        if classical_raw:
            x_classical.append(qnn_feat)
        else:
            x_classical.append(make_classical_feature(path, lookback_returns))
        labels.append(1 if future_var > past_var else 0)
        dates.append(df.loc[end, "date"].to_datetime64())
        past_vars.append(past_var)
        future_vars.append(future_var)

    return (
        np.asarray(x_qnn, dtype=np.float32),
        np.asarray(x_classical, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(dates),
        np.asarray(past_vars, dtype=np.float32),
        np.asarray(future_vars, dtype=np.float32),
    )


def chronological_splits(
    x_qnn: np.ndarray,
    x_classical: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    past_var: np.ndarray,
    future_var: np.ndarray,
) -> Tuple[Split, Split, Split]:
    """Use chronological train/validation/test splits."""

    n = len(y)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        Split(
            "train",
            x_qnn[:train_end],
            x_classical[:train_end],
            y[:train_end],
            dates[:train_end],
            past_var[:train_end],
            future_var[:train_end],
        ),
        Split(
            "validation",
            x_qnn[train_end:val_end],
            x_classical[train_end:val_end],
            y[train_end:val_end],
            dates[train_end:val_end],
            past_var[train_end:val_end],
            future_var[train_end:val_end],
        ),
        Split(
            "test",
            x_qnn[val_end:],
            x_classical[val_end:],
            y[val_end:],
            dates[val_end:],
            past_var[val_end:],
            future_var[val_end:],
        ),
    )


def standardize_angles(train_x: np.ndarray, *xs: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Standardize lookback features and scale them to stable rotation angles."""

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    out = []
    for x in xs:
        clipped = np.clip((x - mean) / std, -3.0, 3.0)
        out.append((np.pi / 3.0 * clipped).astype(np.float32))
    return tuple(out)


def classifier_metrics(
    y: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Accuracy-style metrics for a fixed probability threshold."""

    pred = (prob >= threshold).astype(np.int32)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    return {
        "acc": float(accuracy_score(y, pred)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc": float(auc),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pred_pos": float(np.mean(pred)),
        "threshold": float(threshold),
    }


def best_accuracy_threshold(
    y: np.ndarray, prob: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """Choose the threshold by validation accuracy only."""

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], prob)))
    best_threshold = 0.5
    best_metrics = classifier_metrics(y, prob, best_threshold)
    for threshold in thresholds:
        current = classifier_metrics(y, prob, float(threshold))
        current_score = current["acc"] + 1e-3 * current["auc"]
        best_score = best_metrics["acc"] + 1e-3 * best_metrics["auc"]
        if current_score > best_score:
            best_threshold = float(threshold)
            best_metrics = current
    return best_threshold, best_metrics


def print_metrics(
    label: str, y: np.ndarray, prob: np.ndarray, threshold: float = 0.5
) -> None:
    """Print compact classifier metrics."""

    metrics = classifier_metrics(y, prob, threshold)
    print(
        f"{label:34s} "
        f"acc={metrics['acc']:.4f} "
        f"bal_acc={metrics['bal_acc']:.4f} "
        f"auc={metrics['auc']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"pred_pos={metrics['pred_pos']:.3f} "
        f"thr={metrics['threshold']:.3f}"
    )


def positive_probability(model: object, x: np.ndarray) -> np.ndarray:
    """Return P(label=1) for sklearn classifiers."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def candidate_classical_models(random_state: int = 17) -> List[Tuple[str, object]]:
    """A compact but strong classical baseline grid."""

    models: List[Tuple[str, object]] = []
    for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        models.append(
            (
                f"logreg C={c_value:g}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            )
        )
    for max_depth in [3, 5, 7]:
        for min_leaf in [20, 50]:
            models.append(
                (
                    f"rf depth={max_depth} leaf={min_leaf}",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            )
    for max_leaf_nodes in [7, 15, 31]:
        for learning_rate in [0.03, 0.06]:
            models.append(
                (
                    f"hgb leaves={max_leaf_nodes} lr={learning_rate:g}",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=learning_rate,
                        max_leaf_nodes=max_leaf_nodes,
                        l2_regularization=0.1,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=random_state,
                    ),
                )
            )
    for hidden_layer_sizes in [(16,), (32,), (64,), (32, 16)]:
        for alpha in [0.01, 0.1, 1.0]:
            models.append(
                (
                    f"mlp layers={hidden_layer_sizes} alpha={alpha:g}",
                    make_pipeline(
                        StandardScaler(),
                        MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            alpha=alpha,
                            learning_rate_init=0.001,
                            batch_size=128,
                            max_iter=1500,
                            n_iter_no_change=50,
                            random_state=random_state,
                        ),
                    ),
                )
            )
    return models


def fit_classical(train: Split, val: Split) -> Tuple[str, object, float]:
    """Fit and select the classical baseline by validation accuracy."""

    best_name = ""
    best_model = None
    best_threshold = 0.5
    best_score = -float("inf")

    print("Validation classical model scan:")
    for name, model in candidate_classical_models():
        model.fit(train.x_classical, train.y)
        val_prob = positive_probability(model, val.x_classical)
        threshold, metrics = best_accuracy_threshold(val.y, val_prob)
        print(
            f"  {name:32s} "
            f"acc={metrics['acc']:.4f} "
            f"bal_acc={metrics['bal_acc']:.4f} "
            f"auc={metrics['auc']:.4f} "
            f"thr={threshold:.3f}"
        )
        score = metrics["acc"] + 1e-3 * metrics["auc"]
        if score > best_score:
            best_name = name
            best_model = model
            best_threshold = threshold
            best_score = score

    if best_model is None:
        raise RuntimeError("No classical model was selected.")
    return best_name, best_model, best_threshold


def init_qnn_params(seed: int, n_blocks: int, n_qubits: int) -> Dict[str, jnp.ndarray]:
    """Initialize QNN encoding, variational, and readout parameters."""

    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "enc_scale": 0.2 * jax.random.normal(k1, (n_blocks, n_qubits)),
        "enc_bias": 0.05 * jax.random.normal(k2, (n_blocks, n_qubits)),
        "var": 0.1 * jax.random.normal(k3, (n_blocks, n_qubits, 3)),
        "readout_w": 0.1 * jax.random.normal(k4, (n_qubits,)),
        "readout_b": jnp.array(0.0),
    }


def apply_reuploading_group(
    state: jnp.ndarray,
    group_inputs: Tuple[jnp.ndarray, ...],
    n_qubits: int,
    scan_each: int,
) -> jnp.ndarray:
    """Apply one scanned group of reuploading blocks to a state."""

    x_group, enc_scale_group, enc_bias_group, var_group = group_inputs
    circuit = tc.Circuit(n_qubits, inputs=state)
    for local_block in range(scan_each):
        for q in range(n_qubits):
            angle = (
                enc_scale_group[local_block, q] * x_group[local_block, q]
                + enc_bias_group[local_block, q]
            )
            circuit.ry(q, theta=angle)
            circuit.rz(q, theta=var_group[local_block, q, 0])
            circuit.rx(q, theta=var_group[local_block, q, 1])
            circuit.rz(q, theta=var_group[local_block, q, 2])
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)
        circuit.cnot(n_qubits - 1, 0)
    return circuit.state()


def qnn_observables_scan(
    params: Dict[str, jnp.ndarray],
    x: jnp.ndarray,
    n_qubits: int,
    n_blocks: int,
    scan_each: int,
) -> jnp.ndarray:
    """Run scanned data reuploading and measure all single-qubit Z values."""

    x_grouped = K.reshape(x, [n_blocks // scan_each, scan_each, n_qubits])
    scale_grouped = K.reshape(
        params["enc_scale"], [n_blocks // scan_each, scan_each, n_qubits]
    )
    bias_grouped = K.reshape(
        params["enc_bias"], [n_blocks // scan_each, scan_each, n_qubits]
    )
    var_grouped = K.reshape(
        params["var"], [n_blocks // scan_each, scan_each, n_qubits, 3]
    )

    state = tc.Circuit(n_qubits).state()
    state = K.scan(
        lambda s, block_inputs: apply_reuploading_group(
            s, block_inputs, n_qubits, scan_each
        ),
        (x_grouped, scale_grouped, bias_grouped, var_grouped),
        state,
    )
    final_circuit = tc.Circuit(n_qubits, inputs=state)
    return K.stack(
        [K.real(final_circuit.expectation_ps(z=[q])) for q in range(n_qubits)]
    )


def qnn_logit(
    params: Dict[str, jnp.ndarray],
    x: jnp.ndarray,
    n_qubits: int,
    n_blocks: int,
    scan_each: int,
) -> jnp.ndarray:
    """Linear readout from measured Z expectations."""

    obs = qnn_observables_scan(params, x, n_qubits, n_blocks, scan_each)
    return K.sum(obs * params["readout_w"]) + params["readout_b"]


_batch_logits = K.jit(K.vmap(qnn_logit, vectorized_argnums=1), static_argnums=(2, 3, 4))


def make_qnn_update(
    n_qubits: int,
    n_blocks: int,
    scan_each: int,
    optimizer: optax.GradientTransformation,
):
    """Create the jitted QNN training step."""

    batch_logits = K.vmap(qnn_logit, vectorized_argnums=1)

    def loss_fn(
        params: Dict[str, jnp.ndarray], xb: jnp.ndarray, yb: jnp.ndarray
    ) -> jnp.ndarray:
        logits = batch_logits(params, xb, n_qubits, n_blocks, scan_each)
        return K.mean(optax.sigmoid_binary_cross_entropy(logits, yb))

    loss_and_grad = K.jit(K.value_and_grad(loss_fn))

    @K.jit
    def update(
        params: Dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        xb: jnp.ndarray,
        yb: jnp.ndarray,
    ) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
        loss, grads = loss_and_grad(params, xb, yb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return update


def qnn_predict_prob(
    params: Dict[str, jnp.ndarray],
    x: np.ndarray,
    n_qubits: int,
    n_blocks: int,
    scan_each: int,
    batch_size: int,
) -> np.ndarray:
    """Predict QNN positive-class probabilities in batches."""

    probs = []
    xj = K.convert_to_tensor(x)
    for start in range(0, len(x), batch_size):
        logits = _batch_logits(
            params, xj[start : start + batch_size], n_qubits, n_blocks, scan_each
        )
        probs.append(K.numpy(K.sigmoid(logits)))
    return np.concatenate(probs)


def train_qnn_seed(
    seed: int,
    train: Split,
    val: Split,
    train_x_qnn: np.ndarray,
    val_x_qnn: np.ndarray,
    n_qubits: int,
    n_blocks: int,
    scan_each: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
) -> Tuple[Dict[str, jnp.ndarray], ModelSelection, float]:
    """Train one QNN seed and keep the best validation checkpoint."""

    params = init_qnn_params(seed, n_blocks, n_qubits)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    update = make_qnn_update(n_qubits, n_blocks, scan_each, optimizer)

    xj = jnp.asarray(train_x_qnn)
    yj = jnp.asarray(train.y.astype(np.float32))
    key = jax.random.PRNGKey(seed + 10_000)
    start_time = time.time()
    best_params = params
    best_selection = ModelSelection(
        "", 0.5, {"acc": -float("inf"), "auc": -float("inf")}
    )

    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (batch_size,), 0, len(train_x_qnn))
        params, opt_state, train_loss = update(params, opt_state, xj[idx], yj[idx])
        if step % eval_every == 0 or step == steps:
            val_prob = qnn_predict_prob(
                params, val_x_qnn, n_qubits, n_blocks, scan_each, batch_size
            )
            threshold, metrics = best_accuracy_threshold(val.y, val_prob)
            score = metrics["acc"] + 1e-3 * metrics["auc"]
            best_score = (
                best_selection.metrics["acc"] + 1e-3 * best_selection.metrics["auc"]
            )
            if score > best_score:
                best_params = jax.tree_util.tree_map(lambda z: z.copy(), params)
                best_selection = ModelSelection(
                    name=f"seed={seed}, step={step}",
                    threshold=threshold,
                    metrics=metrics,
                )
            print(
                f"  seed={seed} step={step:4d} "
                f"train_loss={float(train_loss):.5f} "
                f"val_acc={metrics['acc']:.4f} "
                f"val_auc={metrics['auc']:.4f} "
                f"thr={threshold:.3f} "
                f"best_acc={best_selection.metrics['acc']:.4f}"
            )

    return best_params, best_selection, time.time() - start_time


def fit_qnn(
    train: Split,
    val: Split,
    train_x_qnn: np.ndarray,
    val_x_qnn: np.ndarray,
    args: argparse.Namespace,
    n_blocks: int,
) -> Tuple[Dict[str, jnp.ndarray], ModelSelection]:
    """Train multiple seeds and select by validation accuracy."""

    best_params = None
    best_selection = ModelSelection(
        "", 0.5, {"acc": -float("inf"), "auc": -float("inf")}
    )
    for seed in args.seeds:
        params, selection, elapsed = train_qnn_seed(
            seed,
            train,
            val,
            train_x_qnn,
            val_x_qnn,
            args.n_qubits,
            n_blocks,
            args.scan_each,
            args.steps,
            args.batch_size,
            args.lr,
            args.eval_every,
        )
        print(
            f"{selection.name} finished elapsed={elapsed:.1f}s "
            f"best_val_acc={selection.metrics['acc']:.4f} "
            f"best_val_auc={selection.metrics['auc']:.4f}"
        )
        score = selection.metrics["acc"] + 1e-3 * selection.metrics["auc"]
        best_score = (
            best_selection.metrics["acc"] + 1e-3 * best_selection.metrics["auc"]
        )
        if score > best_score:
            best_params = params
            best_selection = selection

    if best_params is None:
        raise RuntimeError("No QNN checkpoint was selected.")
    return best_params, best_selection


def date_range(dates: np.ndarray) -> str:
    """Compact date range display."""

    return f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="SH000300", help="xalpha index code.")
    parser.add_argument("--start", default="20120101", help="xalpha start date.")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--feature-mode",
        choices=["returns", "path", "demeaned_path", "path_and_returns"],
        default="returns",
        help="QNN angle-encoding feature representation.",
    )
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--scan-each", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--classical-raw",
        action="store_true",
        help="Use raw features for classical models, identical to QNN inputs.",
    )
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-qnn", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the integrated classical-vs-QNN experiment."""

    args = parse_args()
    if args.lookback % args.n_qubits != 0:
        raise ValueError("--lookback must be divisible by --n-qubits.")
    n_blocks = args.lookback // args.n_qubits
    if n_blocks % args.scan_each != 0:
        raise ValueError("--lookback / --n-qubits must be divisible by --scan-each.")

    info = xa.indexinfo(args.code, start=args.start)
    x_qnn, x_classical, y, dates, past_var, future_var = make_dataset(
        info.price, args.lookback, args.horizon, args.feature_mode, args.classical_raw
    )
    train, val, test = chronological_splits(
        x_qnn, x_classical, y, dates, past_var, future_var
    )
    train_x_qnn, val_x_qnn, test_x_qnn = standardize_angles(
        train.x_qnn, train.x_qnn, val.x_qnn, test.x_qnn
    )

    print(
        f"Fetched {getattr(info, 'name', args.code)} ({args.code}) rows={len(info.price)}"
    )
    print(
        f"Task: classify future {args.horizon}d realized variance > "
        f"prior {args.horizon}d realized variance"
    )
    print(
        f"QNN: feature_mode={args.feature_mode}, {args.n_qubits} qubits, "
        f"{n_blocks} reuploading blocks, scan_each={args.scan_each}"
    )
    print(f"Classical raw features: {args.classical_raw}")
    for split in (train, val, test):
        print(
            f"{split.name:10s} n={len(split.y):4d} "
            f"date_end={date_range(split.dates)} "
            f"positive_rate={np.mean(split.y):.3f}"
        )
    print()

    train_majority = int(np.mean(train.y) >= 0.5)
    majority_val_prob = np.full_like(val.y, train_majority, dtype=float)
    majority_test_prob = np.full_like(test.y, train_majority, dtype=float)
    print_metrics("validation train-majority", val.y, majority_val_prob)
    print_metrics("test train-majority", test.y, majority_test_prob)

    classical_model = None
    classical_name = ""
    classical_threshold = 0.5
    if not args.skip_classical:
        classical_name, classical_model, classical_threshold = fit_classical(train, val)
        print()
        print(f"Selected classical model by validation accuracy: {classical_name}")
        print_metrics(
            f"validation {classical_name}",
            val.y,
            positive_probability(classical_model, val.x_classical),
            classical_threshold,
        )
        print_metrics(
            f"test {classical_name}",
            test.y,
            positive_probability(classical_model, test.x_classical),
            classical_threshold,
        )

    qnn_params = None
    qnn_selection = None
    if not args.skip_qnn:
        print()
        qnn_params, qnn_selection = fit_qnn(
            train, val, train_x_qnn, val_x_qnn, args, n_blocks
        )
        val_qnn_prob = qnn_predict_prob(
            qnn_params,
            val_x_qnn,
            args.n_qubits,
            n_blocks,
            args.scan_each,
            args.batch_size,
        )
        test_qnn_prob = qnn_predict_prob(
            qnn_params,
            test_x_qnn,
            args.n_qubits,
            n_blocks,
            args.scan_each,
            args.batch_size,
        )
        print()
        print(
            f"Selected QNN checkpoint by validation accuracy: {qnn_selection.name}, "
            f"threshold={qnn_selection.threshold:.3f}"
        )
        print_metrics(
            "validation TensorCircuit QNN",
            val.y,
            val_qnn_prob,
            qnn_selection.threshold,
        )
        print_metrics(
            "test TensorCircuit QNN",
            test.y,
            test_qnn_prob,
            qnn_selection.threshold,
        )

    if (
        classical_model is not None
        and qnn_params is not None
        and qnn_selection is not None
    ):
        print()
        print("Summary:")
        print_metrics(
            f"test {classical_name}",
            test.y,
            positive_probability(classical_model, test.x_classical),
            classical_threshold,
        )
        print_metrics(
            "test TensorCircuit QNN",
            test.y,
            test_qnn_prob,
            qnn_selection.threshold,
        )
        print_metrics("test train-majority", test.y, majority_test_prob)

        preview = pd.DataFrame(
            {
                "date_end": pd.to_datetime(test.dates),
                "actual_up": test.y,
                "qnn_prob_up": test_qnn_prob,
                "past_ann_vol": np.sqrt(252 * test.past_var),
                "future_ann_vol": np.sqrt(252 * test.future_var),
            }
        )
        print()
        print("Last 8 QNN test classifications:")
        print(preview.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
