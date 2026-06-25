"""
Evaluation for Challenge Suite Problem 8.
"""

import argparse
import importlib
import time

import numpy as np

DEFAULT_CONFIG = {
    "grid_side": 7,
    "n_qubits": 49,
    "n_samples": 8192,
    "ry_offset": 0.19,
    "ry_row_sin_scale": 0.07,
    "ry_row_sin_frequency": 0.83,
    "ry_col_cos_scale": 0.05,
    "ry_col_cos_frequency": 0.61,
    "ry_diag_sin_scale": 0.03,
    "ry_diag_sin_frequency": 0.29,
    "rzz_offset": 0.31,
    "rzz_edge_sin_scale": 0.09,
    "rzz_edge_sin_frequency": 0.47,
    "rzz_site_cos_scale": 0.06,
    "rzz_site_cos_frequency": 0.38,
    "rxx_offset": 0.27,
    "rxx_edge_cos_scale": 0.08,
    "rxx_edge_cos_frequency": 0.41,
    "rxx_site_sin_scale": 0.07,
    "rxx_site_sin_frequency": 0.33,
    "rx_offset": 0.17,
    "rx_row_cos_scale": 0.06,
    "rx_row_cos_frequency": 0.52,
    "rx_col_sin_scale": 0.04,
    "rx_col_sin_frequency": 0.44,
    "rx_diag_cos_scale": 0.02,
    "rx_diag_cos_frequency": 0.25,
    "single_z_tolerance": 0.03,
    "hidden_z_string_max_tolerance": 0.05,
    "hidden_z_string_mean_tolerance": 0.015,
}


HIDDEN_Z_STRINGS = [
    (0,),
    (6,),
    (7,),
    (12,),
    (18,),
    (24,),
    (30,),
    (42,),
    (0, 1),
    (2, 3),
    (8, 15),
    (10, 11),
    (14, 21),
    (16, 23),
    (24, 25),
    (31, 38),
    (40, 41),
    (47, 48),
    (0, 8),
    (5, 19),
    (7, 21),
    (9, 23),
    (16, 30),
    (18, 32),
    (27, 41),
    (28, 36),
    (0, 1, 7, 8),
    (3, 4, 10, 11),
    (8, 9, 15, 16),
    (18, 19, 25, 26),
    (22, 23, 29, 30),
    (32, 33, 39, 40),
    (40, 41, 47, 48),
    (2, 3, 4, 9, 10, 11),
    (14, 15, 16, 21, 22, 23),
    (25, 26, 27, 32, 33, 34),
    (7, 8, 9, 14, 15, 16),
    (1, 8, 15, 22, 29, 36),
    (3, 10, 17, 24, 31, 38, 45),
    (6, 13, 20, 27, 34, 41, 48),
    (0, 1, 2, 3, 4, 5, 6),
    (0, 7, 14, 21, 28, 35, 42),
    (8, 16, 24, 32, 40, 41, 42),
    (4, 10, 16, 22, 28, 34, 40),
]


EXACT_HIDDEN_Z_STRING_VALUES = np.asarray(
    [
        0.88235565787735,
        0.9513509979344562,
        0.8410546402488175,
        0.9160048805317923,
        0.9198632104127028,
        0.9161834911474817,
        0.9025978466956375,
        0.9340524695427191,
        0.8050509334897017,
        0.8610924642963674,
        0.7960402956880633,
        0.788609613912459,
        0.8621564877816011,
        0.9003471790590183,
        0.862280446064795,
        0.9630412230605648,
        0.9024843250356801,
        0.946974190713846,
        0.7576336555148323,
        0.9016214417133365,
        0.7126573639986975,
        0.7611035201350809,
        0.7826958156492164,
        0.8719021316091272,
        0.8849308416425875,
        0.7626021845174586,
        0.8059106240971841,
        0.8419225972079478,
        0.6539960370843456,
        0.8511405202044285,
        0.692162028840967,
        0.9283347781927921,
        0.9383070677201228,
        0.7694956531200131,
        0.6947047293378497,
        0.8305584162291031,
        0.5014507908937763,
        0.8624539021430481,
        0.9217668069334316,
        0.8738380122262001,
        0.6093344763746136,
        0.7899677899297161,
        0.5446731544504543,
        0.48428545341967616,
    ],
    dtype=np.float64,
)


def horizontal_edges(config):
    n_side = config["grid_side"]
    return [
        (row, col, row * n_side + col, row * n_side + col + 1)
        for row in range(n_side)
        for col in range(n_side - 1)
    ]


def vertical_edges(config):
    n_side = config["grid_side"]
    return [
        (row, col, row * n_side + col, (row + 1) * n_side + col)
        for row in range(n_side - 1)
        for col in range(n_side)
    ]


def ry_angle(row, col, config):
    return (
        config["ry_offset"]
        + config["ry_row_sin_scale"]
        * np.sin(config["ry_row_sin_frequency"] * (row + 1))
        + config["ry_col_cos_scale"]
        * np.cos(config["ry_col_cos_frequency"] * (col + 1))
        + config["ry_diag_sin_scale"]
        * np.sin(config["ry_diag_sin_frequency"] * (row + col + 2))
    )


def rzz_angle(row, col, edge_index, config):
    return (
        config["rzz_offset"]
        + config["rzz_edge_sin_scale"]
        * np.sin(config["rzz_edge_sin_frequency"] * (edge_index + 1))
        + config["rzz_site_cos_scale"]
        * np.cos(config["rzz_site_cos_frequency"] * (2 * row + col + 1))
    )


def rxx_angle(row, col, edge_index, config):
    return (
        config["rxx_offset"]
        + config["rxx_edge_cos_scale"]
        * np.cos(config["rxx_edge_cos_frequency"] * (edge_index + 1))
        + config["rxx_site_sin_scale"]
        * np.sin(config["rxx_site_sin_frequency"] * (row + 2 * col + 1))
    )


def rx_angle(row, col, config):
    return (
        config["rx_offset"]
        + config["rx_row_cos_scale"]
        * np.cos(config["rx_row_cos_frequency"] * (row + 1))
        - config["rx_col_sin_scale"]
        * np.sin(config["rx_col_sin_frequency"] * (col + 1))
        + config["rx_diag_cos_scale"]
        * np.cos(config["rx_diag_cos_frequency"] * (row + col + 2))
    )


def z_string_expectations_from_samples(samples, z_strings):
    z_samples = 1.0 - 2.0 * samples.astype(np.float64)
    values = []
    for support in z_strings:
        values.append(np.mean(np.prod(z_samples[:, support], axis=1)))
    return np.asarray(values, dtype=np.float64)


def bitstrings(samples):
    return ["".join(str(int(bit)) for bit in row) for row in samples]


def evaluate(solution_module, config):
    module = importlib.import_module(solution_module)

    start = time.perf_counter()
    results = module.run_solution(config)
    elapsed = time.perf_counter() - start

    samples = np.asarray(results["samples"])
    exact_values = EXACT_HIDDEN_Z_STRING_VALUES
    empirical_values = z_string_expectations_from_samples(samples, HIDDEN_Z_STRINGS)
    abs_errors = np.abs(empirical_values - exact_values)

    single_count = sum(len(support) == 1 for support in HIDDEN_Z_STRINGS)
    single_z_error = np.max(abs_errors[:single_count])
    hidden_max_error = np.max(abs_errors)
    hidden_mean_error = np.mean(abs_errors)

    criteria = {
        "sample shape": samples.shape == (config["n_samples"], config["n_qubits"]),
        "samples are binary": np.all((samples == 0) | (samples == 1)),
        "single-Z finite-sample error": single_z_error <= config["single_z_tolerance"],
        "hidden Z-string max error": hidden_max_error
        <= config["hidden_z_string_max_tolerance"],
        "hidden Z-string mean error": hidden_mean_error
        <= config["hidden_z_string_mean_tolerance"],
    }

    print("Challenge 8 evaluation")
    print(f"Solution module: {solution_module}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Grid side: {config['grid_side']}")
    print(f"Sample shape: {samples.shape}")
    print("First sampled bitstrings:")
    for value in bitstrings(samples[: min(10, len(samples))]):
        print(f"  {value}")
    print(f"Hidden observables checked: {len(HIDDEN_Z_STRINGS)}")
    print(f"Max single-site Z absolute error: {single_z_error:.10f}")
    print(f"Max hidden Z-string absolute error: {hidden_max_error:.10f}")
    print(f"Mean hidden Z-string absolute error: {hidden_mean_error:.10f}")
    print(f"Returned NumPy keys: {sorted(results)}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_8")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_CONFIG["n_samples"])
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config["n_samples"] = args.n_samples
    evaluate(args.solution, config)


if __name__ == "__main__":
    main()
