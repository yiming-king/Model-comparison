library(rstan)
library(bridgesampling)
library(jsonlite)

rstan::rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


# arguments:
# - dataset: select subfolder from from dataset/json/
# - model: select null/alternative
args    <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]  # "empirical", "null", or "alternative"
model   <- args[2]  # "null" or "alternative"

stopifnot(dataset %in% c("empirical", "null", "alternative"))
stopifnot(model   %in% c("null", "alternative"))

cat("Dataset:", dataset, "\n")
cat("Model:  ", model,   "\n\n")

data_path <- here::here("dataset", "json", dataset)
stan_file <- here::here("stan", sprintf("rdm_%s.stan", model))

results_dir <- here::here("stan", "results", dataset, model)
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

stan_model <- rstan::stan_model(stan_file)

data_files <- list.files(data_path, pattern = "\\.json$", full.names = FALSE)
df <- data.frame() # collect log_ml estimates

for (i in seq_along(data_files)) {
  data_file <- data_files[i]
  cat(sprintf("Fitting %s, %i/%i", data_file, i, length(data_files)), "\n")

  stan_data <- jsonlite::read_json(
    file.path(data_path, data_file),
    simplifyVector = TRUE
  )

  stan_samples <- rstan::sampling(
    stan_model,
    data    = stan_data,
    iter    = 10000,
    warmup  = 2000,
    chains  = 4,
    control = list(adapt_delta = 0.9),
    pars    = c("log_alpha", "log_nu", "logit_tau", "alpha", "nu", "t0")
  )

  bridge_samples <- bridgesampling::bridge_sampler(
    stan_samples,
    repetitions = 10,
    method      = "warp3",
    silent      = TRUE
  )

  log_ml <- bridgesampling::logml(bridge_samples)

  id    <- tools::file_path_sans_ext(data_file)
  df[i, "id"]       <- id
  df[i, "estimate"] <- log_ml
  df[i, "sd"]       <- sd(bridge_samples$logml)

  save(
    stan_samples, bridge_samples, log_ml,
    file = file.path(results_dir, paste0(id, ".Rdata"))
  )

  cat("Done!\n")
}

write.csv(
  df,
  file      = file.path(results_dir, "bridgesampling.csv"),
  row.names = FALSE
)