import numpy as np
class GetDatasets:
    def __init__(self,mu_prior_mean: float, mu_prior_std: float,num_dims:int,num_obs:int,
                 likelihood_std:float,num_datasets:int,
                 use_student_t:bool=False, df:float=None,rng=None):
        """Initialize the MMD calculator."""
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_std = mu_prior_std
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.likelihood_std = likelihood_std
        self.num_datasets = num_datasets
        self.use_student_t = use_student_t
        self.df = df
        self.rng = rng if rng is not None else np.random.default_rng()

    def get_datasets_normal(self):
        datasets=[]
        for i in range(self.num_datasets):
            mu = self.rng.normal(loc=self.mu_prior_mean, scale=self.mu_prior_std, size=self.num_dims)
            obs_data = self.rng.normal(loc=mu, scale=self.likelihood_std, size=(self.num_obs, self.num_dims))
            datasets.append({
                "mu": mu,
                "obs_data": obs_data,
                "id": i
            })
        return datasets

    def get_datasets_student_t(self):
        datasets=[]
        for i in range(self.num_datasets):
            mu = self.rng.normal(loc=self.mu_prior_mean, scale=self.mu_prior_std, size=self.num_dims)
            obs_data = self.rng.standard_t(df=self.df, size=(self.num_obs,self.num_dims))*self.likelihood_std+mu
            datasets.append({
                "mu": mu,
                "obs_data": obs_data,
                "id": i,
                "df": self.df
            })
        return datasets
    

