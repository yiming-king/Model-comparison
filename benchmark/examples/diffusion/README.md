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

## Shared raw datasets for diagnostic references

For each candidate model M0–M3, indirect networks and all direct losses reuse
the same persisted raw datasets in three separate splits: reference fit,
reference calibration, and density validation. Fit observations estimate
reference means/covariances, kernel references, and densities. Calibration
observations determine diagnostic medians and 5th–95th percentiles. The third,
held-out validation split is also fixed and shared across methods.

The default bank is
`reference_datasets/diagnostics/seed2025_fit2000_cal2000_validation2000/`.
Each candidate directory `{m0,m1,m2,m3}` contains
`{fit,calibration,validation}.npz`, each with 2,000 raw datasets under the
`observations` key. The bank's `manifest.json` records shapes, simulator and
condition provenance, and file/array hashes. The shared entry point is
`results.shared_reference_data.ensure_shared_reference_data()`.

Each network transforms those identical raw observations with its own learned
summary network. Its fitted references, normalized distances, and surprise
labels therefore remain specific to that network and candidate model.
Changing summary dimension, training variant, or direct loss does not redraw
the raw reference splits. Derived diagnostic caches must match both the shared
bank and the corresponding checkpoint before figures are regenerated.

The 400 datasets in `calibration_reference_100` remain a separate benchmark
for approximation-error thresholds. They are not added to the diagnostic
scoring batch: the four diagnostics still score only the 136 observed
datasets. Reference simulation and density fitting do not retrain the
indirect or direct inference networks.

## Direct versus indirect PMP comparison

Open `notebooks/pmp_direct_indirect_comparison.ipynb` in the `benchmark2`
environment. It discovers saved `networks/direct_<loss>.keras` checkpoints for
cross-entropy, exponential, and logistic losses. Currently only cross-entropy
is available. Missing losses are printed explicitly and omitted; the notebook
does not train classifiers or substitute another loss's checkpoint. Adding the
other saved checkpoints automatically adds their comparison rows and overlays.

The training notebooks `direct_cent.ipynb`, `direct_exp.ipynb`, and
`direct_logistic.ipynb` now call `ensure_direct_pmp_thresholds()` and
`ensure_direct_diagnostics()` in their evaluation cell after saving the
checkpoint. Each evaluates that saved checkpoint and writes to the same
`results/direct/direct_<loss>/` layout. Use the standalone comparison notebook
to load and evaluate saved checkpoints without rerunning the training cells.

The indirect baseline is `S4D` **with MMD**, matching `direct_cent.ipynb`, and
the gold standard is the existing Stan bridge-sampling PMP. Observed results
are joined by `(dataset, id)`, retaining empirical, simulated, and contaminated
source names. The current comparison includes 136 observed datasets. Both
methods use the same observations and gold probabilities. Recovery plots use
circles for indirect PMP and crosses for direct PMP, with one shared 0–1
`cividis` color bar showing the candidate model's gold-standard PMP.

Each direct loss's four diagnostics (L2, Linf, density, and Kernel) are computed
for the 136 observed datasets in that classifier's own learned summary
embeddings, with separate reference distributions for M0–M3 fitted from
the shared raw reference splits through each network's own embeddings. They do
not reuse indirect diagnostic values or surprise classes. Compatible caches
are validated against checkpoint and input
provenance. The first run may fit diagnostic reference densities; that is
separate from training a direct classifier.

Direct PMP error calibration uses the same `calibration_reference_100`
datasets and Stan reference as the indirect benchmark. These 400 datasets are
used only to calculate direct PMP error thresholds, not to score the four
summary diagnostics. Within each generating model, the four signed PMP errors
are pooled after requiring convergence for
all candidate fits. The 5th–95th percentiles form the interval; `n_values` is
400 when all 100 datasets are retained. The diagnostic row for M_j displays
the interval calibrated on datasets generated by M_j, pooling all four PMP
components. Direct models provide PMP only; no direct parameter-posterior or
logML error thresholds are produced.

Outputs are under `results/direct/direct_<loss>/`; combined PNG/PDF figures
are under `results/direct/direct_loss_comparison/figures`. The observed
`pmp_comparison.csv` and diagnostic rows drive the comparison figures.
`diagnostics/diagnostic_frame.csv` contains the four diagnostics for the 136
observed datasets. The 400 calibration datasets supply the PMP error
thresholds displayed in those plots; diagnostic reference distributions are
fitted separately for each network using the shared raw fit/calibration/
validation splits described above.

The diagnostic figure follows the Gaussian layout, without an overall title
or upper-left reference strips. Green vertical shading covers the union of
the available losses' normalized diagnostic reference intervals. Green
horizontal shading covers the intersection of their PMP calibration
intervals (or a lighter enclosing interval if disjoint); colored dashed lines
retain each loss's own bounds. The vertical shading is a display envelope,
not a joint classification across networks. Point shapes still use each
loss's own four-model high-surprise rule.

## Intentional differences from Simon's original scripts

- `approximators/indirect.py` uses BayesFlow 2's `BasicWorkflow`.
- `networks/SummaryNetwork()` uses `dropout=0.0` instead of `None`, because the installed BayesFlow version expects a numeric dropout value.
- `dataset.write_stan()` writes to `dataset/json/empirical`, matching the Stan scripts' read path.
