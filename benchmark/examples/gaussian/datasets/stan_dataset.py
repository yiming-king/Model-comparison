import numpy as np
from pathlib import Path
import json
def _to_jsonable(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
                return x.item()
        return x
    
class StanDataset:
    def save_for_stan(obs_data,output_dir:str):
        output_path=Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest=[]
        for i in range(len(obs_data)):
            mu = _to_jsonable(obs_data[i]['mu'])
            dataset = _to_jsonable(obs_data[i]['obs_data'])
            id = obs_data[i]['id']
            df = obs_data[i]['df']
            npe_post_samples = _to_jsonable(obs_data[i]['npe_post_samples'])
            npe_log_marginal = _to_jsonable(obs_data[i]['npe_log_marginal'])
            stan_data={
                'mu':mu,
                'obs_data':dataset,
                'id':id,
                'df':df,
                'npe_post_samples':npe_post_samples,
                'npe_log_marginal':npe_log_marginal
            }
            filename=f"Student_{id}.json"
            filepath = output_path / filename
            with open(filepath, 'w') as f:
                json.dump(stan_data, f, indent=2)
            manifest.append({"id": id, "file": filename})
        print(f"\n✓ Saved {len(manifest)} datasets with unique IDs")
        return manifest
    
    def load_datasets_from_json(dir_path: str):
        dir_path = Path(dir_path)
        datasets = []
        for fp in dir_path.glob("*.json"):
            with open(fp, "r") as f:
                obj = json.load(f)
            datasets.append(obj)
        datasets.sort(key=lambda d: d["id"])
        return datasets




        
        
        
        