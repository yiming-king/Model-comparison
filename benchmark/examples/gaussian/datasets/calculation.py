import numpy as np
from benchmark.examples.gaussian.analytic.analytic import GaussianAnalytical as GA
from benchmark.examples.gaussian.approximators.estimation import MarginalLikelihoodEstimator as MLE

class Calculation:
    def __init__(self,workflow,mu_prior_mean: float, mu_prior_std: float,
                 num_dims:int,num_obs:int,
                 likelihood_std:float,num_samples:int,
                 df: float | None = None, use_student_t: bool = False, rng=None):
        
        self.workflow = workflow
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.likelihood_std = likelihood_std
        self.num_samples = num_samples
        self.df=df
        self.use_student_t=use_student_t
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def normal_analytical(self,obs_data):
        for i in range(len(obs_data)):
            dataset=obs_data[i]["obs_data"]
            analytical=GA(obs_data=dataset,
                                num_dims=self.num_dims,
                                mu_prior_mean=self.mu_prior_mean,
                                mu_prior_std=self.mu_prior_std,
                                num_obs=self.num_obs,
                                likelihood_std=self.likelihood_std,
                                num_samples=1000,
                                rng=self.rng)
            _,_,analytical_posterior_samples=analytical.analytical_posterior()
            log_marginal_analytical=analytical.log_marginal_analytical()
            obs_data[i]["gold_log_marginal"]=log_marginal_analytical
            obs_data[i]["gold_post_samples"]=analytical_posterior_samples
        return obs_data
    def npe_estimation(self,obs_data):
        for i in range(len(obs_data)):
            dataset=obs_data[i]["obs_data"]
            mu_samples=self.workflow.sample_posterior(obs_data=dataset,num_samples=self.num_samples)
            npe_post_samples=np.asarray(mu_samples['mu'][0],dtype=np.float32)
            estimator=MLE(workflow=self.workflow,mu=npe_post_samples,obs_data=dataset,
                                        mu_prior_mean=self.mu_prior_mean,mu_prior_std=self.mu_prior_std,
                                        num_dims=self.num_dims,num_obs=self.num_obs,
                                        likelihood_std=self.likelihood_std,df=self.df,
                                        use_student_t=self.use_student_t,
                                        rng=self.rng)
            npe_log_marginal=estimator.log_marginal_npe()
            obs_data[i]["npe_post_samples"]=npe_post_samples
            obs_data[i]["npe_log_marginal"]=npe_log_marginal
        return obs_data
            
