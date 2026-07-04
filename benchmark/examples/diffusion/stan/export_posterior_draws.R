library(rstan)

# Export light posterior draw CSV files from existing .Rdata files.
# Usage:
#   Rscript stan/export_posterior_draws.R empirical all 2048 2025

args <- commandArgs(trailingOnly = TRUE)
dataset <- if (length(args) >= 1) args[1] else "empirical"
model_arg <- if (length(args) >= 2) tolower(args[2]) else "all"
n_posterior_draws <- if (length(args) >= 3) as.integer(args[3]) else 2048
posterior_seed <- if (length(args) >= 4) as.integer(args[4]) else 2025

model_specs <- data.frame(
  model = c("m0", "m1", "m2", "m3"),
  model_index = 0:3,
  stringsAsFactors = FALSE
)

num_alpha <- function(model_index) {
  c(1, 2, 2, 4)[model_index + 1]
}

export_posterior_draws <- function(stan_samples, model_index, id, output_dir, n_draws, seed) {
  draws <- rstan::extract(
    stan_samples,
    pars = c("log_alpha", "log_nu", "logit_tau"),
    permuted = TRUE
  )
  n_available <- length(draws$logit_tau)
  n_keep <- min(n_draws, n_available)
  set.seed(seed)
  keep <- sort(sample(seq_len(n_available), size = n_keep, replace = FALSE))

  output <- data.frame(
    draw = seq_len(n_keep),
    source_draw = keep,
    id = id,
    assumed_model = paste0("m", model_index),
    stringsAsFactors = FALSE
  )

  for (j in seq_len(num_alpha(model_index))) {
    output[[paste0("alpha_", j - 1)]] <- draws$log_alpha[keep, j]
  }
  for (j in seq_len(2)) {
    output[[paste0("nu_", j - 1)]] <- draws$log_nu[keep, j]
  }
  output[["tau"]] <- draws$logit_tau[keep]

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  write.csv(
    output,
    file = file.path(output_dir, paste0(id, ".csv")),
    row.names = FALSE
  )
}

if (model_arg != "all") {
  stopifnot(model_arg %in% model_specs$model)
  model_specs <- model_specs[model_specs$model == model_arg, , drop = FALSE]
}

command_args <- commandArgs(trailingOnly = FALSE)
file_arg <- command_args[grep("^--file=", command_args)][1]
script_path <- normalizePath(sub("^--file=", "", file_arg))
stan_dir <- dirname(script_path)
results_root <- file.path(stan_dir, "results_4_models", dataset)

for (row in seq_len(nrow(model_specs))) {
  model <- model_specs$model[row]
  model_index <- model_specs$model_index[row]
  results_dir <- file.path(results_root, model)
  posterior_dir <- file.path(results_dir, "posterior_draws")
  rdata_files <- sort(list.files(results_dir, pattern = "\\.Rdata$", full.names = TRUE))

  if (length(rdata_files) == 0) {
    stop("No .Rdata files found in: ", results_dir)
  }

  cat("Dataset:", dataset, "Assumed model:", model, "\n")
  for (i in seq_along(rdata_files)) {
    env <- new.env(parent = emptyenv())
    load(rdata_files[i], envir = env)
    id <- tools::file_path_sans_ext(basename(rdata_files[i]))
    export_posterior_draws(
      env$stan_samples,
      model_index = model_index,
      id = id,
      output_dir = posterior_dir,
      n_draws = n_posterior_draws,
      seed = posterior_seed + i
    )
  }
}

#cd /Users/yimingzang/Documents/Project/benchmark2

# /opt/anaconda3/envs/benchmark2/bin/Rscript \
#   benchmark/examples/diffusion/stan/export_posterior_draws.R empirical all 2048 2025