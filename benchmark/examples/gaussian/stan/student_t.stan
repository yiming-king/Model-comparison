data {
  int<lower=1> N;              // Number of observations
  int<lower=1> D;              // Number of dimensions
  array[N] vector[D] y;        // Observations (N x D matrix)
  real mu_prior_mean;          // Prior mean for mu
  real<lower=0> mu_prior_std;  // Prior std for mu
  real<lower=0> likelihood_std; // Known likelihood std
  real<lower=2> df;            // Degrees of freedom for Student t
}

transformed data {
  real<lower=0> likelihood_scale;
  likelihood_scale = likelihood_std * sqrt((df - 2.0) / df);
}

parameters {
  vector[D] mu;                // Mean parameter (D-dimensional)
}

model {
  // Prior:  mu ~ N(mu_prior_mean, mu_prior_std^2 * I)
 target += normal_lpdf(mu | mu_prior_mean, mu_prior_std);
  
  // Likelihood:  y_i ~ Student_t(df, mu, likelihood_scale)
  for (n in 1:N) {
    target += student_t_lpdf(y[n] | df, mu, likelihood_scale);
  }
}

generated quantities {
  vector[N] log_lik;  
  for (n in 1:N) {
    log_lik[n] = student_t_lpdf(y[n] | df, mu, likelihood_scale);
  }
}

