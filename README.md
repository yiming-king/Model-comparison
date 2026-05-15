# benchmark

A research repository for posterior-approximation and model-comparison experiments on a 3-model Gaussian / Student-t toy family for different approaches: NPE and NPMP.

## Core terminology

This repository uses two different axes of "model", and they must not be conflated:

- **Source model**: the data-generating mechanism used to sample the observation `y`.
- **Assumed model**: the model assumption used during inference to compute `p(theta | y, M)` and/or `p(y | M)`.

This distinction is essential for interpreting the files and result fields in this repo.

## Model definitions

- **M1**: Normal prior mean `0`, Normal likelihood
- **M2**: Normal prior mean `0.5`, Normal likelihood
- **M3**: Normal prior mean `0`, Student-t likelihood with df=5

- **M4**: Normal prior mean `2`, Student-t likelihood with df=5 which will only be used as observation dataset to form a open world situation.

## Gold references used in this repo

- **Gold under M1**: analytical posterior samples and analytical log marginal likelihood
- **Gold under M2**: analytical posterior samples and analytical log marginal likelihood
- **Gold under M3**: Stan posterior samples and bridge-sampling log marginal likelihood

Important interpretation of the current naming:

- files such as `results/datasets/m1.pkl` and `stan/json_m1` refer to **source model = M1**
- fields such as `gold_post_samples_m3` and `gold_log_marginal_m3` refer to **assumed model = M3**
- therefore `stan/stan_output_m1` means: observations generated from **M1**, then evaluated under **M3**

## Repository layout

- `benchmark/examples/gaussian/notebooks/m1.ipynb`
  Train the NPE model under assumed model M1.
- `benchmark/examples/gaussian/notebooks/m2.ipynb`
  Train the NPE model under assumed model M2.
- `benchmark/examples/gaussian/notebooks/m3.ipynb`
  Train the NPE model under assumed model M3.
- `benchmark/examples/gaussian/notebooks/direct.ipynb`
  Train the NPMP model-comparison classifier.
- `benchmark/examples/gaussian/notebooks/calculation.ipynb`
  Generate observation datasets, merge gold references, NPE outputs, and NPMP outputs.
- `benchmark/examples/gaussian/notebooks/comparison.ipynb`
  Compare posterior diagnostics and log-marginal errors.
- `benchmark/examples/gaussian/notebooks/plot.ipynb`
  Produce distributions figures from saved datasets.
- `benchmark/examples/gaussian/stan/student_t.stan`
  Stan model used for the M3 gold reference.
- `benchmark/examples/gaussian/stan/student_t_gold_m1.r`
  Run Stan for observations generated from source model M1, under assumed model M3.
- `benchmark/examples/gaussian/stan/student_t_gold_m2.r`
  Run Stan for observations generated from source model M2, under assumed model M3.
- `benchmark/examples/gaussian/stan/student_t_gold_m3.r`
  Run Stan for observations generated from source model M3, under assumed model M3.
- `benchmark/examples/gaussian/stan/student_t_gold_m4.r`
  Run Stan for observations generated from source model M4, under assumed model M3.

## Python requirements

Install the Python environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebooks]"


## License

`benchmark` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
