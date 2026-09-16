# Gaussian case-study diagnostics

For Python training, see [the training guide](approximators/README.md). The
six presets cover D=20, N=10/100, and S=D/2D/4D: 24 networks across m1--m4.
All use `DeepSet(summary_dim=S)` and `CouplingFlow()` with BayesFlow network
defaults, including no summary MMD regularization. Training uses a shared
100 epochs, batch size 64, 128 batches per epoch, and Adam + CosineDecay
starting at 1e-4; these optimizer settings are project choices.

New files carry `_bf_default`, for example
`m1_s_20d_100n_bf_default.keras`. To compare both observation counts under
the new settings, retrain all 24 networks; adding only the eight missing
N=100, S=D/4D networks leaves the older models with different settings.
The training guide includes dry-run, complete-grid, eight-network, and smoke
commands. Original notebooks, saved weights, and cached results remain as
historical experiments. The older N=100 NPE names use `40d_100n` because their
actual summary size is S=40 (2D), with raw D=20. The separate direct classifier
retains `direct_s_20d_100n.keras`, which uses raw D/N in its name.

The analysis examples below refer to historical network tags. For the new
runs, explicitly select `20d_10n_bf_default 40d_10n_bf_default 80d_10n_bf_default`
with `--num-obs 10`, or
`20d_100n_bf_default 40d_100n_bf_default 80d_100n_bf_default` with
`--num-obs 100`, through `--network-tags`. Regenerate per-network calibration,
summary references, and OOD results; the training guide shows how to store
the new calibration separately and select its threshold file.

The Gaussian analysis follows the diffusion-case-study layout. It supports
`l2`, `linf`, `mmd`, and `density` diagnostics and calibrates approximation
errors separately on well-specified NPE versus analytical Gaussian reference
calculations.

For the current raw (D=20,n=10) networks, the saved models are discovered
from their Keras configuration rather than inferred from their filenames:

| Summary label | True summary dimension | Network tag |
| --- | ---: | --- |
| `S=1D` | 20 | `20d_10n` |
| `S=2D` | 40 | `40d_10n` |
| `S=4D` | 80 | `80d_10n` |

Thresholds are therefore specific to `summary_label × generating_model`; they
are never shared across summary-space sizes.

Run commands from the repository root with the `benchmark2` environment.

## 1. Generate calibration data and thresholds

The complete command discovers and runs all available `S=1D`, `S=2D`, and
`S=4D` networks for raw (D=20,n=10):

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline all \
  --num-dims 20 \
  --num-obs 10 \
  --num-datasets 30 \
  --num-posterior-samples 1000 \
  --quantile 0.95 \
  --signed-error-coverage 0.90 \
  --seed 2025
```

It generates one independent well-specified dataset bank, then evaluates every
summary configuration against it. Each configuration has its own NPE posterior,
NPE logML, and PMP thresholds: posterior MMD uses `[0, q95]`, while signed
logML/PMP errors use the central 90% interval `[q5, q95]`. The stages can also
be run separately:

```bash
# New, independent, well-specified m1--m4 datasets
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline generate --num-dims 20 --num-obs 10

# NPE and analytical posterior/logML/PMP calculations for every available S=kD
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline metrics --num-dims 20 --num-obs 10

# posterior MMD [0, q95] and signed logML/PMP central 90% thresholds
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline thresholds --num-dims 20 --num-obs 10
```

Existing calibration datasets are reused unless `--overwrite-datasets` is
specified. The summary-independent analytical posterior draws are saved once,
while matching-model NPE posterior draws are stored under their summary
configuration. Pass `--no-save-posterior-draws` to omit both kinds of draws.

To run only selected summary dimensions, use e.g.

```bash
# Only S=D and S=4D
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline all --num-dims 20 --num-obs 10 --summary-dims 20 80

# Equivalent selection by saved network names
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline all --num-dims 20 --num-obs 10 --network-tags 20d_10n 80d_10n
```

Outputs are written below:

```text
benchmark/examples/gaussian/calibration_outputs/20d_10n/
├── datasets/{m1,m2,m3,m4}/
├── analytical/posterior_draws/{m1,m2,m3,m4}/
├── metrics/
│   ├── S1D/{m1,m2,m3,m4}/per_dataset_metrics.csv
│   ├── S1D/npe_posterior_draws/{m1,m2,m3,m4}/
│   ├── S2D/{m1,m2,m3,m4}/per_dataset_metrics.csv
│   ├── S2D/npe_posterior_draws/{m1,m2,m3,m4}/
│   ├── S4D/{m1,m2,m3,m4}/per_dataset_metrics.csv
│   └── S4D/npe_posterior_draws/{m1,m2,m3,m4}/
├── metadata.json
├── per_dataset_metrics.csv
├── thresholds.csv
└── per_dataset_results.csv
```

The default 30 datasets per generating model matches the diffusion calibration. Because
the Gaussian analytical reference is inexpensive, use e.g. `--num-datasets 200`
when a more stable empirical 95th percentile is preferred. Dataset count,
posterior sample count, and logML method are part of the threshold grouping and
must match the intended analysis.

## 2. Compute OOD posterior, logML, PMP, and direct PMP

OOD calculations now run outside the notebooks. The full `S=D`, L2 workflow is:

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.analysis.pipeline all \
  --num-dims 20 \
  --num-obs 10 \
  --network-tags 20d_10n \
  --metrics l2 \
  --num-datasets 50 \
  --num-posterior-samples 1000 \
  --seed 2025
```

The stages can also be called independently:

```bash
# Generate or reuse the m1--m12 OOD dataset bank.
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.analysis.pipeline generate --network-tags 20d_10n

# Compute analytical/NPE posterior results, analytical/NPE logML, indirect PMP,
# and the direct classifier PMP.
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.analysis.pipeline inference --network-tags 20d_10n

# Compute L2 summary diagnostics and join the cached approximation errors.
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.analysis.pipeline diagnostics --network-tags 20d_10n --metrics l2
```

The computation-only tables are saved in
`results/ood_20d_10n/inference/{posterior,logml,pmp}.csv`. Full plotting tables
are saved in `results/ood_20d_10n/diagnostics/l2/`. The notebook
`notebooks/ood_analysis_l2_20d_10n.ipynb` only loads these cached tables and
plots posterior, logML, indirect-PMP, and direct-PMP results; it does not load
networks or rerun inference.

The plotting-only notebooks follow a regular `diagnostic × summary-size` grid:

| Diagnostic | S=D | S=2D | S=4D |
| --- | --- | --- | --- |
| L2 | `ood_analysis_l2_20d_10n.ipynb` | `ood_analysis_l2_40d_10n.ipynb` | `ood_analysis_l2_80d_10n.ipynb` |
| Linf | `ood_analysis_linf_20d_10n.ipynb` | `ood_analysis_linf_40d_10n.ipynb` | `ood_analysis_linf_80d_10n.ipynb` |
| Kernel | `ood_analysis_mmd_20d_10n.ipynb` | `ood_analysis_mmd_40d_10n.ipynb` | `ood_analysis_mmd_80d_10n.ipynb` |
| Density | `ood_analysis_density_20d_10n.ipynb` | `ood_analysis_density_40d_10n.ipynb` | `ood_analysis_density_80d_10n.ipynb` |

Every notebook loads posterior, logML, and PMP frames through the same helper.
It supports both the current `diagnostics/<metric>/` layout and the existing
legacy L2/Linf CSV locations, so the already-computed results can be plotted
without conversion or recomputation. Posterior MMD uses one panel per assumed
model and overlays all `m1`--`m12` source datasets in each panel. The notebook
`YSCALE` mapping selects `"linear"` or `"symlog"` independently for posterior
MMD, logML, and PMP, matching the diffusion plotting workflow. LogML errors
are displayed in base-10 units, with calibrated lower/upper threshold lines
and the corresponding shaded error interval.

Use `--overwrite-datasets` or `--overwrite-inference` when the corresponding
cache must be rebuilt. Python computation is plot-free by default; pass
`--plots` only when command-line figure generation is wanted.

## 3. Run Kernel and density diagnostics

After the threshold and OOD inference commands finish, this command evaluates
`mmd` and `density` for every available summary configuration:

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.analysis.pipeline diagnostics \
  --num-dims 20 \
  --num-obs 10 \
  --metrics mmd density \
  --n-fit 2000 \
  --n-calibration 2000 \
  --n-density-validation 2000 \
  --alpha 0.1 \
  --density-epochs 250 \
  --seed 2025
```

This command expects the OOD inference cache files for each summary network:
`results/ood_20d_10n`, `results/ood_40d_10n`, and `results/ood_80d_10n`.
It computes each NPE summary once per `(summary configuration, assumed model)`,
fits independent reference suites, and joins the matching threshold rows.

```text
benchmark/examples/gaussian/results/
├── ood_20d_10n/diagnostics/   # S=1D
├── ood_40d_10n/diagnostics/   # S=2D
└── ood_80d_10n/diagnostics/   # S=4D
```

The reference suite is reused per summary configuration. Use
`--overwrite-reference` to refit it. The command writes CSV tables only by
default; pass `--plots` to also generate three calibrated
error-versus-diagnostic figures per metric. To evaluate only a subset, use `--summary-dims 20 80` or
`--network-tags 20d_10n 80d_10n`. The same command can run all diagnostics with
`--metrics l2 linf mmd density`.

For another trained configuration, change `--num-dims` and `--num-obs`; for
example, `--num-dims 40 --num-obs 10`. The corresponding `m1_s_40d_10n.keras`
through `m4_s_40d_10n.keras` files and OOD cache directory must exist.
