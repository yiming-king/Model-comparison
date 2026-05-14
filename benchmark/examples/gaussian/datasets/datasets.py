import numpy as np
class GetDatasets:
    def __init__(self,obs_mu_prior_mean: float, obs_mu_prior_std: float,num_dims:int,num_obs:int,
                 obs_likelihood_std:float,num_datasets:int,rng=None):
        self.obs_mu_prior_mean = obs_mu_prior_mean
        self.obs_mu_prior_std = obs_mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.obs_likelihood_std = obs_likelihood_std
        self.num_datasets = num_datasets
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_datasets_normal(self):
        datasets=[]
        for i in range(self.num_datasets):
            mu = self.rng.normal(loc=self.obs_mu_prior_mean, scale=self.obs_mu_prior_std, size=self.num_dims)
            x = self.rng.normal(loc=mu, scale=self.obs_likelihood_std, size=(self.num_obs, self.num_dims))
            datasets.append({
                "mu": mu,
                "x": x,
                "id": i
            })
        return datasets

    def get_datasets_student_t(self,df):
        datasets=[]
        for i in range(self.num_datasets):
            mu = self.rng.normal(loc=self.obs_mu_prior_mean, scale=self.obs_mu_prior_std, size=self.num_dims)
            scale = self.obs_likelihood_std
            x = self.rng.standard_t(df=df, size=(self.num_obs, self.num_dims)) * scale + mu
            datasets.append({
                "mu": mu,
                "x": x,
                "id": i,
                "df": df
            })
        return datasets
    

