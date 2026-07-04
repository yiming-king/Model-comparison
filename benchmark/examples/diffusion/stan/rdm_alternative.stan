functions {
  #include helpers/wald.stan
}
data {
  int<lower=0> N;
  array[N] real rt;
  array[N] int<lower=0, upper=1> condition; // used to select threshold (alpha)
}
transformed data {
  array[N] real<lower=0> abs_rt = abs(rt);
  real min_rt = min(abs_rt);
}
parameters {
   array[2] real log_alpha;
   array[2] real log_nu;
   real logit_tau;
}
transformed parameters {
    array[N] real log_lik;
    real total_log_lik = 0.0;
    array[2] real alpha = exp(log_alpha);
    array[2] real nu = exp(log_nu);
    real tau = inv_logit(logit_tau);
    real t0 = tau * min_rt;

    for (n in 1:N) {
        log_lik[n] = rdm_lpdf(rt[n] | alpha[condition[n] + 1], nu, t0);
        total_log_lik += log_lik[n];
    }
}
model {
    target += normal_lpdf(log_alpha | 0.0, 0.5);
    target += normal_lpdf(log_nu    | 0.0, 0.5);
    target += normal_lpdf(logit_tau | 0.0, 1.0);
    target += total_log_lik;
}
