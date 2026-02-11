import numpy as np
import bayesflow as bf

class Simulator:
    def __init__(self, mu_prior_mean: float, mu_prior_std: float, num_dims: int, num_obs: int, 
                 likelihood_std: float, df: float | None = None, use_student_t: bool = False, rng=None):
        """Initialize the simulator."""
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.likelihood_std = likelihood_std
        self.use_student_t = use_student_t
        self.df = df
        self.rng = rng or np.random.default_rng()

    def get_bayes_simulator(self) -> bf.make_simulator:
        """Create the BayesFlow simulator."""
        def prior():
            mu = self.rng.normal(loc=self.mu_prior_mean, scale=self.mu_prior_std, size=self.num_dims)
            return dict(mu=mu)
        def likelihood(mu):
            if self.use_student_t:
                obs_data = self.rng.standard_t(df=self.df, size=(self.num_obs,self.num_dims))*self.likelihood_std+mu
            else:
                obs_data = self.rng.normal(loc=mu, scale=self.likelihood_std, size=(self.num_obs, self.num_dims))
            return dict(obs_data=obs_data)
        simulator=bf.make_simulator([prior,likelihood])
        return simulator

    def get_observed_data(self) -> np.ndarray:
        """Generate observed data either from a Gaussian or Student's t likelihood."""
        prior_sample = self.rng.normal(loc=self.mu_prior_mean, scale=self.mu_prior_std, size=self.num_dims)
        if self.use_student_t:
            obs_data = self.rng.standard_t(df=self.df, size=(self.num_obs,self.num_dims))*self.likelihood_std+prior_sample
        else:
            obs_data = self.rng.normal(loc=prior_sample, scale=self.likelihood_std, size=(self.num_obs, self.num_dims))
        return obs_data