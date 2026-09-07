# Diffusion NPE Without Self-Consistency

This folder follows Simon's diffusion example structure and keeps the case study limited to ordinary NPE.

## Structure

- `dataset/`: Wagenmakers data loader and Stan JSON writer.
- `simulators/`: Racing diffusion simulators for the four assumed threshold structures.
- `distributions/`: Exact prior and likelihood objects matching the four Stan models.
- `networks/`: Simon's network factory functions.
- `approximators/`: NPE training and loading code.
- `results/`: log marginal likelihood estimation, Stan gold-standard merge, summary-space diagnostics, and plots.
- `stan/`: Stan files and bridge-sampling results copied from Simon's example.
- `notebooks/`: notebook entry point for the case study.

## Main notebook

Open one of the summary-dimension notebooks in `notebooks/`, for example `S_D.ipynb`.

Use the `benchmark2` environment:

```text
/opt/anaconda3/envs/benchmark2/bin/python
```

## Main workflow

1. Train or load `M0_NPE` through `M3_NPE`.
2. Use simulations from all four models to build summary-space typical sets.
3. Project Wagenmakers empirical datasets into all four summary spaces.
4. Estimate NPE log marginal likelihoods and PMPs across the four models.
5. Merge the Stan bridge-sampling gold standard from `stan/results_4_models`.
6. Compare summary surprise with logML/PMP errors.

Well-specified threshold calibration has a separate entry point and output tree;
the two Python files can be run directly from the project root:

```bash
# Compute or load the per-dataset NPE-MCMC posterior MMD, logML, and PMP errors.
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py metrics

# Rebuild posterior MMD [0, q95] and signed logML/PMP central 90% thresholds.
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/thresholds.py

# Rebuild the second without-MMD run from its own cached metrics.
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/thresholds.py \
  --calibration-root benchmark/examples/diffusion/calibration_outputs_rerun1
```

`pipeline.py metrics` writes `calibration_outputs/per_dataset_metrics.csv`.
`thresholds.py` reads that file and writes `thresholds.csv` plus
`per_dataset_results.csv`; it does not rerun NPE or MCMC. The defaults encode
the interval definitions, so neither run needs manual threshold edits after
regeneration.

### Extend threshold calibration from 30 to 100 datasets

First create one standard reference root from the original `s000--s029` data
and the incremental `s030--s099` bundle. The command hard-links the large JSON
and posterior files and concatenates the manifests and Stan CSV tables:

```bash
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/prepare_reference_100.py
```

Then compute the missing 70 per-dataset metrics for each training run. The
`--reuse-metrics` option imports the compatible first 30 datasets from the old
cache. It never combines old and incremental threshold values; each final
threshold is calculated once from the resulting 100-dataset metric sample.

```bash
# With MMD loss.
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py metrics \
  --reference-root benchmark/examples/diffusion/calibration_reference_100 \
  --output-root benchmark/examples/diffusion/calibration_outputs_100_withMMD \
  --reuse-metrics benchmark/examples/diffusion/calibration_outputs \
  --k 100 --training-settings with_mmd \
  --summary-multipliers 1 2 4 6

# Without MMD loss, run 1 (checkpoint suffix: noMMD).
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py metrics \
  --reference-root benchmark/examples/diffusion/calibration_reference_100 \
  --output-root benchmark/examples/diffusion/calibration_outputs_100_noMMD \
  --reuse-metrics benchmark/examples/diffusion/calibration_outputs \
  --k 100 --training-settings without_mmd \
  --without-mmd-run-suffix noMMD \
  --summary-multipliers 1 2 4 6

# Without MMD loss, run 2 (checkpoint suffix: noMMD_rerun1).
/opt/anaconda3/envs/benchmark2/bin/python \
  benchmark/examples/diffusion/calibration/pipeline.py metrics \
  --reference-root benchmark/examples/diffusion/calibration_reference_100 \
  --output-root benchmark/examples/diffusion/calibration_outputs_100_noMMD_rerun1 \
  --reuse-metrics benchmark/examples/diffusion/calibration_outputs_rerun1 \
  --k 100 --training-settings without_mmd \
  --without-mmd-run-suffix noMMD_rerun1 \
  --summary-multipliers 1 2 4 6
```

Finally calculate one set of 100-dataset thresholds in each output root:

```bash
for root in \
  calibration_outputs_100_withMMD \
  calibration_outputs_100_noMMD \
  calibration_outputs_100_noMMD_rerun1
do
  /opt/anaconda3/envs/benchmark2/bin/python \
    benchmark/examples/diffusion/calibration/thresholds.py \
    --calibration-root "benchmark/examples/diffusion/${root}"
done
```

With all Stan fits converged, every posterior-MMD and signed-logML threshold
row has `n_values=100`; every signed-PMP threshold row has `n_values=400`
because it pools the four candidate-model PMP components for each dataset.
Stored logML errors are in natural-log units. Plot/analysis loaders convert
both errors and interval endpoints to log10 units by dividing by `log(10)`.

Threshold and classification tables retain all four summary dimensions
(`S=D`, `S=2D`, `S=4D`, and `S=6D`). Summary-diagnostic figures intentionally
display only `S=D`, `S=2D`, and `S=4D`; `S=6D` is excluded from plotted points,
curves, threshold lines, legends, and axis-limit calculations.

## Intentional differences from Simon's original scripts

- `approximators/indirect.py` uses BayesFlow 2's `BasicWorkflow`.
- `networks/SummaryNetwork()` uses `dropout=0.0` instead of `None`, because the installed BayesFlow version expects a numeric dropout value.
- `dataset.write_stan()` writes to `dataset/json/empirical`, matching the Stan scripts' read path.
