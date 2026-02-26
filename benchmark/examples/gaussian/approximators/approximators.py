import numpy as np
import keras
import bayesflow as bf

class BayesFLowNPE:
    def __init__(self, simulator, summary_config=None, inference_config=None):
        self.simulator = simulator
        
        self.summary_config = summary_config or {
            'summary_dim': 30,
            'activation': 'relu',}

        self.inference_config = inference_config or {
            'depth': 5,
            'transform':  'spline', # can be 'affine'
            'subnet_kwargs': {
                'widths': (128, 128),
                'activation': 'relu',
                'dropout': 0.05
            }
        }
        self._build_workflow()
        
    def _build_workflow(self):
        # Adapter
        self.adapter = (bf.adapters.Adapter()
                        .convert_dtype("float64", "float32")
                        .rename('mu', 'inference_variables')
                        .rename('obs_data', 'summary_variables')
                )
        # networks
        self.summary_net = bf.networks.DeepSet(**self.summary_config)
        self.inference_net = bf.networks.CouplingFlow(**self.inference_config)
        self.workflow = bf.BasicWorkflow(simulator=self.simulator,
                                          adapter=self.adapter,
                                          summary_network=self.summary_net,
                                          inference_network=self.inference_net,
                                          standardize=None
                                          )
        self.workflow.approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4))
    def train(self,epochs:int,batch_size:int,num_batches_per_epoch:int):
        history = self.workflow.fit_online(
            epochs=epochs,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )
        return history
    def test_datasets(self,number_datasets:int,num_samples:int):
        test_sims=self.workflow.simulate(number_datasets)
        posterior_samples = self.workflow.sample(conditions=test_sims,
                                                                num_samples=num_samples)
        return test_sims,posterior_samples
    def sample_posterior(self, obs_data,num_samples:int):
        """Posterior samples given observed data.
        """
        obs_data = np.asarray(obs_data, dtype=np.float32)
        posterior_samples = self.workflow.sample(conditions={"obs_data": obs_data[np.newaxis,:,:]},
                                                                num_samples=num_samples)
        return posterior_samples

    
