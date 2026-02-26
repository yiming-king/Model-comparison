import numpy as np
import keras
import bayesflow as bf

class DirectBayesFLowNPE:
    def __init__(self, simulator,num_batches_per_epoch:int,batch_size:int,epochs:int, 
                 summary_config=None, classifier_config=None,eps: float=1e-12):
        self.simulator = simulator
        
        self.summary_config = summary_config or {
            'summary_dim':15,
            'activation': 'relu',
            'dropout':None,
            }
        self.classifier_config = classifier_config or {
            'widths': [64]*4,
            'activation': 'silu',
            'dropout': None,
            }
        self.num_batches_per_epoch=num_batches_per_epoch
        self.batch_size=batch_size
        self.epochs=epochs
        self.eps=eps
        self._build_workflow()
        
    def _build_workflow(self):
        # Adapter
        self.adapter = (bf.Adapter()
                        .as_set("x")
                        .convert_dtype("float64", "float32")
                        .rename('x', 'summary_variables')
                )
        # networks
        self.summary_network = bf.networks.DeepSet(**self.summary_config)
        self.classifier_network = bf.networks.MLP(**self.classifier_config)
        self.workflow = bf.approximators.ModelComparisonApproximator(
            num_models=3,
            classifier_network=self.classifier_network,
            summary_network=self.summary_network,
            adapter=self.adapter,
            standardize="summary_variables"
            
        )
        self.learning_rate=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-4, 
            decay_steps=self.epochs * self.num_batches_per_epoch
            )
        self.workflow.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        
    def train(self):
        history = self.workflow.fit(
            simulator=self.simulator,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_batches=self.num_batches_per_epoch,
            adapter=self.adapter,
        )
        return history
    
    def get_probs(self,obs_data,approximator):
        for i in range(len(obs_data)):
            df=obs_data[i]["obs_data"]
            pred_model=approximator.predict(conditions={'x':df[None,...]},probs=True)
            p = np.asarray(pred_model, dtype=float).squeeze(0)
            if not np.all(np.isfinite(p)):
                raise ValueError(f"Non-finite probs at i={i}: {p}")
            p = np.clip(p, self.eps, 1.0)
            p = p / p.sum()
            obs_data[i]["pred_model"]=p
            logp=np.log(p)
            
            obs_data[i]["logBF_12_direct"]=float(logp[0] - logp[1])
            obs_data[i]["logBF_13_direct"]=float(logp[0] - logp[1])
            obs_data[i]["logBF_23_direct"]=float(logp[0] - logp[1])
        return obs_data

    
