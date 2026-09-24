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

## Shared raw datasets for diagnostic references

For each candidate model M1–M4, indirect networks and all direct losses use
the same saved raw datasets for diagnostic reference fitting, reference
calibration, and density validation. These are three separate splits. Fit
data estimate the reference mean/covariance, kernel reference, or density;
calibration data determine the reference median and 5th–95th percentiles;
validation data check the density fit. The validation split is also fixed and
shared across methods. No method redraws these splits independently.

The current D=20, N=10 bank is stored under
`results/diagnostic_reference_datasets/20d_10n/seed_2025_fit_2000_calibration_2000_validation_2000/`.
Each candidate has `{fit,calibration,validation}.npy`, with a shared
`manifest.json` recording the bank identity. Each split contains 2,000 datasets
per candidate. Raw dimensions, observation count, candidate model, split,
sample counts, and seed identify the bank; summary dimension and direct loss
do not. The shared entry point is
`analysis.reference_datasets.ensure_reference_datasets()`.

Every network embeds those shared observations using its own learned summary
network and fits its own reference distributions in that space. Sharing raw
inputs does not share embeddings, fitted densities, quantile bounds, or
surprise labels. Derived diagnostic caches must correspond to both the raw
bank and the network checkpoint. After changing the bank, recompute those
caches before redrawing diagnostic figures.

The separate approximation-error benchmark remains 120 raw datasets, 30 per
candidate, in `calibration_outputs/20d_10n/datasets/`. It is used to calibrate
approximation-error thresholds and is not an additional batch for the four
summary diagnostics. The plotted OOD observations remain the existing 600
datasets. This change does not retrain any inference network.

## Direct model comparison and its own summary diagnostics

`notebooks/direct_cent.ipynb`, `direct_exp.ipynb`, and `direct_logistic.ipynb`
train separate direct classifiers, each with a learned 12-dimensional summary.
Their outputs live under `results/direct/direct_{cross_entropy,exponential,logistic}/`.
`notebooks/pmp_direct_indirect_comparison.ipynb` reads these results and saves
combined figures under `results/direct/direct_loss_comparison/figures/`.

The comparison still reads observations and indirect S=4D logML from
`results/ood_80d_10n/datasets/*_logml_pmp.pkl`. These files contain distinct
indirect inference results and must be retained. Here `80d` denotes the indirect
summary dimension; the observations have shape `(10, 20)`.

Direct L2, Linf, Kernel, and density diagnostics use each loss's **own direct
checkpoint embeddings**. The shared raw fit, calibration, and validation
splits from M1–M4 are transformed by each classifier to fit four model-specific
reference laws in that classifier's learned space. No indirect reference
distributions or surprise labels are reused.
Reference fitting and calibration each use 2,000 datasets per candidate model;
density flows additionally use 2,000 validation datasets. The reference interval
is the central 90%, and `rho = (distance - median) / (high - median)`.
“All high surprise” means all four distances exceed their own upper reference
bound; lower-tail values retain the existing interpolation convention.

```bash
KERAS_BACKEND=jax /opt/anaconda3/envs/benchmark2/bin/python \
  -m benchmark.examples.gaussian.analysis.direct_diagnostics
```

The comparison notebook calls the same cache-aware entry point before reading
PMP tables. Checkpoint, input, implementation, and reference-setting fingerprints
prevent stale diagnostics after retraining; refreshed PMPs come from the same
checkpoint as the diagnostics. This only fits diagnostic density models and
does not retrain the direct classifier. Outputs include a long diagnostic CSV,
reference caches, observed embeddings, and provenance metadata per loss.

Direct PMP **error** thresholds are separate from these summary-space diagnostic
intervals. `analysis/direct_calibration.py` evaluates each saved direct classifier
on the exact existing `calibration_outputs/20d_10n/datasets/{m1,m2,m3,m4}/x.npy`
benchmark (30 datasets per generator). It recomputes analytical PMP from those
observations and calibrates `direct PMP - analytical PMP`. Following the indirect
protocol, each generating model pools all four PMP components (120 values) and
uses the linear-interpolated 5th and 95th percentiles. Thus plot row Mj uses the
error interval calibrated on generator Mj, pooling its four components; it is
not a component-j-only error interval.

```bash
KERAS_BACKEND=jax /opt/anaconda3/envs/benchmark2/bin/python \
  -m benchmark.examples.gaussian.analysis.direct_calibration
```

Each loss saves `calibration/{thresholds,per_dataset_metrics,per_dataset_results}.csv`
and `calibration/metadata.json` below its own `results/direct/direct_<loss>/`.
The comparison notebook checks these caches and matches the green background
style of `summary_dimension_comparison_M1_M4_4x4.ipynb` (`#DCEEDC`, alpha 0.70).
The diagnostic background spans from the smallest loss-specific `rho_low`
(using its median across datasets) to `rho = 1`: it is the envelope of the
three reference intervals, not a region jointly normal under all three losses.
The horizontal PMP-error background shows the intersection of the three loss
intervals, `[max(low), min(high)]`. If they do not overlap, it shows their
envelope `[min(low), max(high)]` with alpha 0.25 instead. Loss-specific horizontal
threshold lines retain their colors, and the vertical `rho = 1` line marks the
upper diagnostic reference boundary.
The three direct classifiers do not produce parameter posteriors or absolute
logML, so posterior-MMD and logML-error thresholds remain properties of the
corresponding indirect estimators. Summary-space Kernel MMD is a different quantity
from posterior-sample MMD.

The reused historical benchmark has a known overlap: all 30 M1 observations
exactly match OOD M1 IDs 0–29; the M2–M4 benchmark observations do not overlap
the plotted 600 OOD datasets. Reuse preserves the original indirect comparison
protocol, but the M1 error interval is not independent of those plotted samples.

The redundant 12 L2 `*_processed_80d_10n.pkl` caches were removed after checking
that their inference fields match retained `*_logml_pmp.pkl` and their diagnostic
values match retained CSVs. The `*_processed_80d_10n_linf.pkl` caches are retained:
they contain a different historical inference run, including posterior draws.

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
plots posterior, logML, and indirect-PMP results; it does not load networks or
rerun inference. Direct PMP diagnostics are shown only in
`notebooks/pmp_direct_indirect_comparison.ipynb`, using each direct network's
own summary space. The twelve OOD notebooks no longer plot old direct
predictions against indirect summary-space diagnostics.

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
