library(rstan)
library(bridgesampling)
library(jsonlite)
library(RcppCNPy)

rstan:: rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

path <- "/Users/yimingzang/Documents/thesis/benchmark2/benchmark/examples/gaussian/stan"
data_path <-"/Users/yimingzang/Documents/thesis/benchmark2/benchmark/examples/gaussian/stan/json_m4"
output_path <- "/Users/yimingzang/Documents/thesis/benchmark2/benchmark/examples/gaussian/stan/stan_output_m4"

dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
stan_model <- rstan:: stan_model("/Users/yimingzang/Documents/thesis/benchmark2/benchmark/examples/gaussian/stan/student_t.stan")

data_files <- list.files(data_path, pattern = "*.json", full.names = FALSE)
cat("Found", length(data_files), "data files\n\n")
results <- data.frame()
i <- 1
for (data_file in data_files) {
  cat(sprintf("[%3d/%3d] %s\n", i, length(data_files), data_file))
  file_base <- tools::file_path_sans_ext(data_file)

  obj <- jsonlite::read_json(
    file.path(data_path, data_file), 
    simplifyVector = FALSE
  )
  y_list <- lapply(obj$x, as.numeric)
  N<-length(y_list)
  D<-length(y_list[[1]])
  df<-as.numeric(obj$df)
  y_mat <- do.call(rbind, y_list)
  mu_prior_mean <- 0
  mu_prior_std <- 1.0
  likelihood_std <- 1
  
  
  stan_data<-list(
    N=N,
    D=D,
    y=y_mat,
    mu_prior_mean=mu_prior_mean,
    mu_prior_std=mu_prior_std,
    likelihood_std=likelihood_std,
    df=df
  )

    stan_fit <- rstan::sampling(
      stan_model,
      data = stan_data,
      chains = 4,
      iter = 8000,
      warmup = 2000,
      thin = 1,
      seed = 2025,
      control = list(
        adapt_delta = 0.99,
        max_treedepth = 12
      ),refresh = 0
    )
  
    # Extract posterior samples
    posterior <- extract(stan_fit)
    mu_posterior <- posterior$mu
    S<-nrow(mu_posterior)
    set.seed(2026)
    idx<-sample.int(S,size=2000,replace=(S<2000))
    mu_2000<-mu_posterior[idx, ,drop=FALSE]
    obj$gold_post_samples_m3<-unname(split(mu_2000, row(mu_2000)))
  
    bridge_result <- bridgesampling:: bridge_sampler(
      stan_fit,
      repetitions = 10,
      method = "warp3",
      silent = TRUE
    )

    log_ml <- bridgesampling::logml(bridge_result)
    obj$gold_log_marginal_m3 <- as.numeric(log_ml)
    
    out_file<-file.path(output_path, data_file)

    write_json(obj,out_file,pretty=TRUE,auto_unbox=TRUE)

  i <- i + 1
  cat("\n")
}
