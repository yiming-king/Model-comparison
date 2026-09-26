# Diffusion Inference and Model Comparison Without Self-Consistency

This folder follows Simon's diffusion example structure. It compares ordinary
NPE-based indirect model probabilities with direct model-comparison classifiers,
using a shared Stan gold standard and diagnostics in each network's own learned
summary space. Neither method uses self-consistency training.

## Structure

- `dataset/`: Wagenmakers data loader and Stan JSON writer.
- `simulators/`: Racing diffusion simulators for the four assumed threshold structures.
- `distributions/`: Exact prior and likelihood objects matching the four Stan models.
- `networks/`: Simon's network factory functions.
- `approximators/`: NPE training and loading code.
- `analysis/`: direct PMP evaluation, calibration, and summary-space diagnostics.
- `results/`: log marginal likelihood estimation, Stan gold-standard merge, summary-space diagnostics, and plots.
- `stan/`: Stan files and bridge-sampling results copied from Simon's example.
- `notebooks/`: notebook entry point for the case study.

## Main notebook

Open one of the summary-dimension notebooks in `notebooks/`, for example `S_D.ipynb`.

Use the `benchmark2` environment:

```text
/opt/anaconda3/envs/benchmark2/bin/python
```

The twelve `density`, `l2`, `linf`, and `mmd` diagnostic notebooks (with-MMD,
no-MMD, and no-MMD rerun1) use two shared entry points in
`results/summary_dimension_comparison.py`:

```python
from benchmark.examples.diffusion.results.summary_dimension_comparison import (
    run_diagnostic_notebook,
    display_diagnostic_notebook,
)

result = run_diagnostic_notebook("density", variant="noMMD")
display_diagnostic_notebook(result)
```

Each notebook contains only setup, editable parameters, the workflow call, and
display. `variant` selects the matching checkpoints, 100-dataset thresholds,
and output folders. The notebooks expose axis scales, normalization, and the
no-MMD LOESS controls. The shared plotter uses `plot_style="standard"` for
with-MMD and `plot_style="combined"` for no-MMD; no notebook replaces global
plotting functions. Pass `output_root` and `overlay_output_root` to write figures
elsewhere, and `refresh_thresholds=False` to reuse the existing thresholds.
The twelve diagnostic notebooks now pass `refresh_thresholds=False` when
redrawing diagnostics; change it to `True` explicitly only when the separate
approximation-error calibration needs to be recalculated.

`calibration/thresholds.py` still calculates calibration intervals.
`results/multisource_pipeline.py`, `results/summary_diagnostic.py`, and
`results/posterior_diagnostic.py` supply cached data and diagnostics. Figure
layouts, markers, medians, LOESS, legends, and notebook display all live in
`results/summary_dimension_comparison.py`; this plotting workflow does not
retrain networks or run Stan.

## Main workflow

1. Train or load `M0_NPE` through `M3_NPE`.
2. Use simulations from all four models to build summary-space typical sets.
3. Project Wagenmakers empirical datasets into all four summary spaces.
4. Estimate NPE log marginal likelihoods and PMPs across the four models.
5. Merge the Stan bridge-sampling gold standard from `stan/results_4_models`.
6. Compare summary surprise with logML/PMP errors.

## Calibration with 100 datasets per model

The calibration pipeline generates 100 independent datasets for each of
`m0`–`m3` (400 datasets total), numbered `s000`–`s099`. The default base seed is
2025. Each dataset has its own seed, so generating all 100 in one run uses the
same seed schedule as the former 30 + 70 batches.

Only the complete 100-dataset reference and results are retained:

| Directory | Contents |
| --- | --- |
| `calibration/` | Generation, Stan fitting, metric, and threshold code. |
| `calibration_reference_100/` | Shared datasets, true parameters, Stan/bridge tables, and posterior draws. |
| `calibration_outputs_100_withMMD/` | Results for networks trained with MMD loss. |
| `calibration_outputs_100_noMMD/` | Results for the first no-MMD training run. |
| `calibration_outputs_100_noMMD_rerun1/` | Results for the second no-MMD training run. |

The three result directories contain `S=D`, `S=2D`, and `S=4D`, with 100
calibration datasets per generating model. Old `S=6D` calibration covered only
30 datasets and is not part of the retained results. It can be calculated
separately with `--summary-multipliers 6` and an explicit `--output-root`.

### Generate or reuse all 100 datasets

From the project root:

```bash
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py generate
```

The default is `--k 100`. The command validates and reuses matching existing
datasets; it does not overwrite them. No merge script or old 30/70 directory
is needed. For a new independent run, choose a new reference root:

```bash
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py generate \
  --reference-root /private/tmp/diffusion_calibration_100_new
```

Use `--base-seed` to choose another seed schedule. `--overwrite` explicitly
regenerates datasets; after changing datasets, also recompute their Stan fits
and metrics rather than combining them with old cached results.

### Fit Stan and calculate metrics

The default variant is `noMMD`. All three variants share
`calibration_reference_100`; each writes metrics and thresholds into its own
`calibration_outputs_100_<variant>` directory.

```bash
# Fit missing reference results once; reuse complete existing references.
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py mcmc

# Calculate or resume metrics for each training run.
for variant in withMMD noMMD noMMD_rerun1
do
  /opt/anaconda3/envs/benchmark2/bin/python \
    benchmark/examples/diffusion/calibration/pipeline.py metrics --variant "$variant"
  /opt/anaconda3/envs/benchmark2/bin/python \
    benchmark/examples/diffusion/calibration/thresholds.py --variant "$variant"
done
```

To run generation, MCMC, metrics, and thresholds for one variant together:

```bash
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py all --variant noMMD
```

`--reference-root` overrides the shared dataset/Stan location;
`--output-root` overrides the result location. When both refer to new locations,
`all` starts a fresh run. Existing compatible metric rows are reused directly
from the selected 100-dataset result directory.

### Rebuild thresholds without rerunning inference

```bash
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/thresholds.py --variant noMMD_rerun1
```

This reads `per_dataset_metrics.csv` and writes `thresholds.csv` and
`per_dataset_results.csv`. Posterior MMD uses `[0, q95]`; signed logML and PMP
errors use the central 90% interval `[q5, q95]`. Thresholds are calculated over
all 100 datasets together, never by averaging thresholds from the old batches.
With all fits converged, posterior-MMD and signed-logML rows have
`n_values=100`; signed-PMP rows have `n_values=400` because they pool all four
candidate-model PMP components. Stored logML errors use natural-log units;
analysis loaders convert errors and interval endpoints to log10 units.

Default analysis functions select the matching 100-dataset threshold file for
each training configuration. The retained `run_config.json` files document the
original 30+70 calculation, including historical `reuse_metrics` paths; these
are provenance records, not dependencies needed to load or rerun the results.

## Diagnostic references for indirect and direct methods

Indirect networks retain their model-specific summary spaces and reference
distributions for M0–M3. Their existing raw reference bank has 2,000 datasets
**per model** for fit, calibration, and density validation.

Each direct loss has one shared summary network. Its diagnostic reference is a
balanced pool across M0–M3, embedded by that direct network. Defaults
`n_fit=2000`, `n_calibration=2000`, and `n_density_validation=2000` are
**totals**: 500 datasets per model in each partition. One reference law and
threshold are fitted per diagnostic (L2, Linf, density, Kernel MMD), yielding
one score per observed dataset. The raw bank is separate from the independent
PMP-error calibration benchmark. The direct network is never retrained by
reference fitting.

## Direct versus indirect PMP comparison

`notebooks/pmp_direct_indirect_comparison.ipynb` reads the saved direct
checkpoints in `approximators/trained/direct/` (`cent`, `exponential`,
`logistic`) and compares them with the same S4D indirect baseline and Stan
gold-standard PMP. Results for the three losses are in
`results/direct/direct_{cross_entropy,exponential,logistic}/`; the cross-entropy
checkpoint is named `direct_cent.keras`.

`analysis/direct_diagnostics.py` embeds the balanced reference, calibration,
validation, and 136 observed datasets in each direct network's own shared
summary space. It saves one pooled reference and four scores per observed
dataset. It does not reuse indirect diagnostic scores or model-specific
surprise classes. `rho = distance / high`, and the green plot background shows
the central reference interval `rho_low <= rho <= 1`.

The independent direct PMP calibration reuses the 100 saved benchmark datasets
per generating model and their indirect-calibration gold PMP vectors. Each
direct network evaluates the same 400 datasets and calibrates a separate
central 90% signed PMP error interval for each candidate model. A dataset has
high PMP error when any candidate's signed error leaves its interval. The
comparison notebook reads those saved intervals.
`rho > 1` predicts high error; the
saved table reports TP/FP/TN/FN, FNR, FPR, and ROC AUC. Each direct network
has a 4×4 plot, with candidate models as rows and diagnostics as columns.
Horizontal lines mark each candidate's signed 90% PMP error bounds, with
simulated datasets as filled markers and empirical datasets as hollow markers. They are saved under
`results/direct/direct_loss_comparison/figures/`.

## Intentional differences from Simon's original scripts

- `approximators/indirect.py` uses BayesFlow 2's `BasicWorkflow`.
- `networks/SummaryNetwork()` uses `dropout=0.0` instead of `None`, because the installed BayesFlow version expects a numeric dropout value.
- `dataset.write_stan()` writes to `dataset/json/empirical`, matching the Stan scripts' read path.
