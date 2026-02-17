import pandas as pd
import numpy as np


class Comparison:
    def evidence_extract(datasets):
        evidence_dic={}
        for d in datasets:
            dataset_id = d["id"]
            evidence_dic[dataset_id] = {
                "id": dataset_id,
                "gold_lm_M1": d["gold_log_marginal"],
                "npe_lm_M1": d["npe_log_marginal"]
                }
        return evidence_dic
    
    def add_evidence(dictionary,dataset,model_name):
        for d in dataset:
            dataset_id=d['id']
            dictionary[dataset_id][f"gold_lm_{model_name}"]=d["gold_log_marginal"]
            dictionary[dataset_id][f"npe_lm_{model_name}"]=d["npe_log_marginal"]
        return dictionary
    
    def BF_calculator_M1(dict):
        dict=pd.DataFrame(dict.values())
        dict["logBF_12_gold"]=dict["gold_lm_M1"] - dict["gold_lm_M2"]
        dict["logBF_13_gold"]=dict["gold_lm_M1"] - dict["gold_lm_M3"]
        dict["logBF_23_gold"]=dict["gold_lm_M2"] - dict["gold_lm_M3"]

        dict["logBF_12_npe"]=dict["npe_lm_M1"] - dict["npe_lm_M2"]
        dict["logBF_13_npe"]=dict["npe_lm_M1"] - dict["npe_lm_M3"]
        dict["logBF_23_npe"]=dict["npe_lm_M2"] - dict["npe_lm_M3"]
        return dict
    
    def BF_calculator_M2(dict):
        dict=pd.DataFrame(dict.values())
        dict["logBF_21_gold"]=dict["gold_lm_M2"] - dict["gold_lm_M1"]
        dict["logBF_23_gold"]=dict["gold_lm_M2"] - dict["gold_lm_M3"]
        dict["logBF_13_gold"]=dict["gold_lm_M1"] - dict["gold_lm_M3"]

        dict["logBF_21_npe"]=dict["npe_lm_M2"] - dict["npe_lm_M1"]
        dict["logBF_23_npe"]=dict["npe_lm_M2"] - dict["npe_lm_M3"]
        dict["logBF_13_npe"]=dict["npe_lm_M1"] - dict["npe_lm_M3"]
        return dict
    
    def BF_calculator_M3(dict):
        dict=pd.DataFrame(dict.values())
        dict["logBF_31_gold"]=dict["gold_lm_M3"] - dict["gold_lm_M1"]
        dict["logBF_32_gold"]=dict["gold_lm_M3"] - dict["gold_lm_M2"]
        dict["logBF_12_gold"]=dict["gold_lm_M1"] - dict["gold_lm_M2"]

        dict["logBF_31_npe"]=dict["npe_lm_M3"] - dict["npe_lm_M1"]
        dict["logBF_32_npe"]=dict["npe_lm_M3"] - dict["npe_lm_M2"]
        dict["logBF_12_npe"]=dict["npe_lm_M1"] - dict["npe_lm_M2"]
        return dict
    
    def prob_calculator(dict):
        cols_gold= ["gold_lm_M1","gold_lm_M2","gold_lm_M3"]
        gold_logits = dict[cols_gold].to_numpy()
        cols_npe= ["npe_lm_M1","npe_lm_M2","npe_lm_M3"]
        npe_logits = dict[cols_npe].to_numpy()
        def softmax_stable(x):
            x = x - np.max(x, axis=1, keepdims=True)
            e = np.exp(x)
            return e / np.sum(e, axis=1, keepdims=True)
        gold_probs = softmax_stable(gold_logits)
        dict["p_gold"] = list(gold_probs)

        npe_probs = softmax_stable(npe_logits)
        dict["p_npe"] = list(npe_probs)
        return dict
    
    