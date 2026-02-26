import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def plot(df,title:str):
        d = df.copy()
        d["err12_npe"] = d["logBF_12_npe"]    - d["logBF_12_gold"]
        d["err12_dir"] = d["logBF_12_direct"] - d["logBF_12_gold"]
        
        d["err13_npe"] = d["logBF_13_npe"]    - d["logBF_13_gold"]
        d["err13_dir"] = d["logBF_13_direct"] - d["logBF_13_gold"]
        
        d["err23_npe"] = d["logBF_23_npe"]    - d["logBF_23_gold"]
        d["err23_dir"] = d["logBF_23_direct"] - d["logBF_23_gold"]
        
        d["abs_err12_npe"] = np.abs(d["err12_npe"])
        d["abs_err12_dir"] = np.abs(d["err12_dir"])

        d["abs_err13_npe"] = np.abs(d["err13_npe"])
        d["abs_err13_dir"] = np.abs(d["err13_dir"])
        
        d["abs_err23_npe"] = np.abs(d["err23_npe"])
        d["abs_err23_dir"] = np.abs(d["err23_dir"])
        summary = pd.DataFrame({
            "pair": ["12","12","13","13","23","23"],
            "method": ["NPE","Direct","NPE","Direct","NPE","Direct"],
            "MAE": [
                d["abs_err12_npe"].mean(),
                d["abs_err12_dir"].mean(),
                d["abs_err13_npe"].mean(),
                d["abs_err13_dir"].mean(),
                d["abs_err23_npe"].mean(),
                d["abs_err23_dir"].mean(),
            ],
            "Median_AE": [
                d["abs_err12_npe"].median(),
                d["abs_err12_dir"].median(),
                d["abs_err13_npe"].median(),
                d["abs_err13_dir"].median(),
                d["abs_err23_npe"].median(),
                d["abs_err23_dir"].median(),
            ]
        })

        print(summary.to_string(index=False))
        
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        for ax in axes:
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
            
        axes[0].boxplot([d["err12_npe"], d["err12_dir"]],
                          labels=["NPE", "Direct"])
        axes[0].axhline(0)
        axes[0].set_ylabel("Error in log BF12")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].boxplot([d["err13_npe"], d["err13_dir"]],
                      labels=["NPE", "Direct"])
        axes[1].axhline(0)
        axes[1].set_ylabel("Error in log BF13")
        axes[1].grid(True, axis="y", alpha=0.3)
        
        axes[2].boxplot([d["err23_npe"], d["err23_dir"]],
                      labels=["NPE", "Direct"])
        axes[2].axhline(0)
        axes[2].set_ylabel("Error in log BF23")
        axes[2].grid(True, axis="y", alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

        return d, summary
  