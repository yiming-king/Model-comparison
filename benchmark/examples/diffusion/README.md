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

Threshold and classification tables retain all four summary dimensions
(`S=D`, `S=2D`, `S=4D`, and `S=6D`). Summary-diagnostic figures intentionally
display only `S=D`, `S=2D`, and `S=4D`; `S=6D` is excluded from plotted points,
curves, threshold lines, legends, and axis-limit calculations.

## Intentional differences from Simon's original scripts

- `approximators/indirect.py` uses BayesFlow 2's `BasicWorkflow`.
- `networks/SummaryNetwork()` uses `dropout=0.0` instead of `None`, because the installed BayesFlow version expects a numeric dropout value.
- `dataset.write_stan()` writes to `dataset/json/empirical`, matching the Stan scripts' read path.
