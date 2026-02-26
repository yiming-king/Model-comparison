import numpy as np
import bayesflow as bf

class DirectSimulator:
    def __init__(self, num_dims: int, num_obs: int,prior_mean1:float,prior_mean2:float,
                 likelihood_std: float,df:float, rng=None):
        """Initialize the simulator."""
        self.num_dims = num_dims
        self.num_obs = num_obs
        self.prior_mean1=prior_mean1
        self.prior_mean2=prior_mean2
        self.likelihood_std = likelihood_std
        self.df=df
        self.rng = rng or np.random.default_rng()
    def get_direct_simulator(self):
        def prior_0():
            mu=self.rng.normal(loc=self.prior_mean1,scale=1,size=self.num_dims)
            return dict(mu=mu)
        def prior_10():
            mu=self.rng.normal(loc=self.prior_mean2,scale=1,size=self.num_dims)
            return dict(mu=mu)
        def likelihood_n(mu):
            x=self.rng.normal(loc=mu,scale=self.likelihood_std,size=(self.num_obs,self.num_dims))
            return dict(x=x)
        def likelihood_t(mu):
            x=self.rng.standard_t(df=self.df, size=(self.num_obs,self.num_dims))*self.likelihood_std+mu
            return dict(x=x)
        simulator_1=bf.make_simulator([prior_0,likelihood_n])
        simulator_2=bf.make_simulator([prior_10,likelihood_n])
        simulator_3=bf.make_simulator([prior_0,likelihood_t])
        simulator=bf.simulators.ModelComparisonSimulator(
            simulators=[simulator_1,simulator_2,simulator_3],
            use_mixed_batches=True,
        )
        return simulator
