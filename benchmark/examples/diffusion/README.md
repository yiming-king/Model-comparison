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

## Intentional differences from Simon's original scripts

- `approximators/indirect.py` uses BayesFlow 2's `BasicWorkflow`.
- `networks/SummaryNetwork()` uses `dropout=0.0` instead of `None`, because the installed BayesFlow version expects a numeric dropout value.
- `dataset.write_stan()` writes to `dataset/json/empirical`, matching the Stan scripts' read path.
