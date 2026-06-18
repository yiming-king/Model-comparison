import numpy as np
import keras
import bayesflow as bf

def direct_get_probs(obs_data,approximator):
    eps=1e-12
    for i in range(len(obs_data)):
        x=obs_data[i]["x"]
        pred_model=approximator.predict(conditions={'x':x[None,...]},probs=True)
        p = np.asarray(pred_model, dtype=float).squeeze(0)
        if not np.all(np.isfinite(p)):
            raise ValueError(f"Non-finite probs at i={i}: {p}")
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()
        obs_data[i]["p_direct"]=p
        logp=np.log(p)
            
        # obs_data[i]["logBF_12_direct"]=float(logp[0] - logp[1])
        # obs_data[i]["logBF_13_direct"]=float(logp[0] - logp[2])
        # obs_data[i]["logBF_23_direct"]=float(logp[1] - logp[2])
    return obs_data

def softmax_stable(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def indirect_get_probs(obs_data, assumed_models):
    for item in obs_data:
        gold = np.array([item[f"gold_log_marginal_{m}"] for m in assumed_models], dtype=float)
        npe = np.array([item[f"npe_log_marginal_{m}"] for m in assumed_models], dtype=float)
        item["p_gold"] = softmax_stable(gold)
        item["p_npe"] = softmax_stable(npe)
        
        # item["logBF_12_gold"] = float(gold_logml[0] - gold_logml[1])
        # item["logBF_13_gold"] = float(gold_logml[0] - gold_logml[2])
        # item["logBF_23_gold"] = float(gold_logml[1] - gold_logml[2])

        # item["logBF_12_npe"] = float(npe_logml[0] - npe_logml[1])
        # item["logBF_13_npe"] = float(npe_logml[0] - npe_logml[2])
        # item["logBF_23_npe"] = float(npe_logml[1] - npe_logml[2])
        
    return obs_data
