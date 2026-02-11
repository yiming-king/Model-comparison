import numpy as np
import pandas as pd
import matplotlib as plt
from bayesflow.metrics import MaximumMeanDiscrepancy
import tensorflow as tf

class Analysis:
    
    def evaluate_mmd_err(
        dataset_list:list,name:str,
        key_npe_post="npe_post_samples",key_gold_post="gold_post_samples",
        key_npe_logml="npe_log_marginal",key_gold_logml="gold_log_marginal"
    ):
        rows=[]
        mmd=MaximumMeanDiscrepancy(kernel='gaussian')
        for ds in dataset_list:
            ds_id=ds['id']
            X=tf.convert_to_tensor(np.asarray(ds[key_npe_post],dtype=np.float32))
            Y=tf.convert_to_tensor(np.asarray(ds[key_gold_post],dtype=np.float32))
            MMD=float(mmd(X,Y))
            
            npe_lm = float(ds[key_npe_logml])
            gold_lm = float(ds[key_gold_logml])
            abs_err = abs(npe_lm - gold_lm)
            
            rows.append({
                "dataset_group": name,
                "id": ds_id,
                "mmd": MMD,
                "abs_logml_error": abs_err,
                "npe_logml": npe_lm,
                "gold_logml": gold_lm
            })
        return pd.DataFrame(rows)
    