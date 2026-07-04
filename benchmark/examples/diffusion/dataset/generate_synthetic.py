import argparse
import json
import os

try:
    from dataset import wagenmakers
    from simulators import SIMULATORS
except ImportError:
    from . import wagenmakers
    from ..simulators import SIMULATORS

import pandas as pd

N_SIM = 17
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json")

N = wagenmakers.n_trials


def save_datasets(simulator, folder: str, n_sim: int = N_SIM, overwrite: bool = False):
    out_dir = os.path.join(BASE_DIR, folder)
    if os.path.exists(out_dir) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to regenerate it.")
    os.makedirs(out_dir, exist_ok=True)

    data = simulator.sample(n_sim)
    parameter_rows = []

    for i in range(n_sim):
        record = dict(
            rt=[float(x) for x in data["rt"][i]],
            condition=[int(x) for x in data["conditions"][i]],
            N=N,
        )
        path = os.path.join(out_dir, f"s{i}.json")
        with open(path, "w") as f:
            json.dump(record, f)

        row = {"id": f"s{i}"}
        for key in ("alpha", "nu", "tau"):
            values = data[key][i].reshape(-1)
            for j, value in enumerate(values):
                row[f"{key}_{j}"] = float(value)
        parameter_rows.append(row)

    pd.DataFrame(parameter_rows).to_csv(
        os.path.join(out_dir, "true_parameters.csv"),
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)

    selected = SIMULATORS.keys() if args.models == ["all"] else args.models
    for model in selected:
        save_datasets(
            SIMULATORS[model],
            folder=f"simulated_from_{model}",
            n_sim=args.n_sim,
            overwrite=args.overwrite,
        )
