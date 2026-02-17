import numpy as np
import keras
import bayesflow as bf

class DirectBayesFLowNPE:
    def __init__(self, simulator,num_batches_per_epoch:int,batch_size:int,epochs:int, summary_config=None, inference_config=None):
        self.simulator = simulator
        
        self.summary_config = summary_config or {
            'summary_dim': 30,
            'activation': 'relu',
            }

        self.inference_config = inference_config or {
            'depth': 5,
            'transform':  'spline', # can be 'affine'
            'subnet_kwargs': {
                'widths': (128, 128),
                'activation': 'relu',
            }
        }
        self.num_batches_per_epoch=num_batches_per_epoch
        self.batch_size=batch_size
        self.epochs=epochs
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
        self.classifier_network = bf.networks.MLP(**self.inference_config)
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
            learning_rate=self.learning_rate,
            num_batches=self.num_batches_per_epoch,
            adapter=self.adapter,
        )
        return history
    
    def get_probs_M1(self,obs_data,approximator):
        for i in range(len(obs_data)):
            df=obs_data[i]["obs_data"]
            pred_model=approximator.predict(conditions={'x':df[None,...]},probs=True)
            obs_data[i]["pred_model"]=pred_model
            obs_data[i]["logBF_12_direct"]=np.log(pred_model[0][0]/pred_model[0][1])
            obs_data[i]["logBF_13_direct"]=np.log(pred_model[0][0]/pred_model[0][2])
            obs_data[i]["logBF_23_direct"]=np.log(pred_model[0][1]/pred_model[0][2])
        return obs_data
    
    def get_probs_M2(self,obs_data,approximator):
        for i in range(len(obs_data)):
            df=obs_data[i]["obs_data"]
            pred_model=approximator.predict(conditions={'x':df[None,...]},probs=True)
            obs_data[i]["pred_model"]=pred_model
            obs_data[i]["logBF_21_direct"]=np.log(pred_model[0][1]/pred_model[0][0])
            obs_data[i]["logBF_23_direct"]=np.log(pred_model[0][1]/pred_model[0][2])
            obs_data[i]["logBF_13_direct"]=np.log(pred_model[0][0]/pred_model[0][2])
        return obs_data
    
    def get_probs_M3(self,obs_data,approximator):
        for i in range(len(obs_data)):
            df=obs_data[i]["obs_data"]
            pred_model=approximator.predict(conditions={'x':df[None,...]},probs=True)
            obs_data[i]["pred_model"]=pred_model
            obs_data[i]["logBF_31_direct"]=np.log(pred_model[0][2]/pred_model[0][0])
            obs_data[i]["logBF_32_direct"]=np.log(pred_model[0][2]/pred_model[0][1])
            obs_data[i]["logBF_12_direct"]=np.log(pred_model[0][0]/pred_model[0][1])
        return obs_data

    
