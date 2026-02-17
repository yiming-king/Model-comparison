import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def plot_M1(df):
        d = df.copy()
        d["err12_npe"] = d["logBF_12_npe"]    - d["logBF_12_gold"]
        d["err12_dir"] = d["logBF_12_direct"] - d["logBF_12_gold"]
        
        d["err13_npe"] = d["logBF_13_npe"]    - d["logBF_13_gold"]
        d["err13_dir"] = d["logBF_13_direct"] - d["logBF_13_gold"]
        
        d["abs_err12_npe"] = np.abs(d["err12_npe"])
        d["abs_err12_dir"] = np.abs(d["err12_dir"])

        d["abs_err13_npe"] = np.abs(d["err13_npe"])
        d["abs_err13_dir"] = np.abs(d["err13_dir"])
        summary = pd.DataFrame({
            "pair": ["12","12","13","13"],
            "method": ["NPE","Direct","NPE","Direct"],
            "MAE": [
                d["abs_err12_npe"].mean(),
                d["abs_err12_dir"].mean(),
                d["abs_err13_npe"].mean(),
                d["abs_err13_dir"].mean(),
            ],
            "Median_AE": [
                d["abs_err12_npe"].median(),
                d["abs_err12_dir"].median(),
                d["abs_err13_npe"].median(),
                d["abs_err13_dir"].median(),
            ]
        })

        print(summary.to_string(index=False))
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0,0].boxplot([d["err12_npe"], d["err12_dir"]],
                          labels=["NPE", "Direct"])
        axes[0,0].axhline(0)
        axes[0,0].set_title("Normal_0: Error in log BF12")
        axes[0,0].set_ylabel("Δ log BF12")
        axes[0,0].grid(True, axis="y", alpha=0.3)

        axes[0,1].boxplot([d["err13_npe"], d["err13_dir"]],
                      labels=["NPE", "Direct"])
        axes[0,1].axhline(0)
        axes[0,1].set_title("Normal_0: Error in log BF13")
        axes[0,1].set_ylabel("Δ log BF13")
        axes[0,1].grid(True, axis="y", alpha=0.3)
        axes[1,0].boxplot([d["abs_err12_npe"], d["abs_err12_dir"]],
                      labels=["NPE", "Direct"])
        axes[1,0].set_title("Normal_0: Absolute Error |Δ log BF12|")
        axes[1,0].set_ylabel("|Δ log BF12|")
        axes[1,0].grid(True, axis="y", alpha=0.3)

        axes[1,1].boxplot([d["abs_err13_npe"], d["abs_err13_dir"]],
                      labels=["NPE", "Direct"])
        axes[1,1].set_title("Normal_0: Absolute Error |Δ log BF13|")
        axes[1,1].set_ylabel("|Δ log BF13|")
        axes[1,1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        return d, summary
    
    def plot_M2(df):
        d = df.copy()
        d["err21_npe"] = d["logBF_21_npe"]    - d["logBF_21_gold"]
        d["err21_dir"] = d["logBF_21_direct"] - d["logBF_21_gold"]
        
        d["err23_npe"] = d["logBF_23_npe"]    - d["logBF_23_gold"]
        d["err23_dir"] = d["logBF_23_direct"] - d["logBF_23_gold"]
        
        d["abs_err21_npe"] = np.abs(d["err21_npe"])
        d["abs_err21_dir"] = np.abs(d["err21_dir"])
        
        d["abs_err23_npe"] = np.abs(d["err23_npe"])
        d["abs_err23_dir"] = np.abs(d["err23_dir"])

        summary = pd.DataFrame({
            "pair": ["21","21","23","23"],
            "method": ["NPE","Direct","NPE","Direct"],
            "MAE": [
                d["abs_err21_npe"].mean(),
                d["abs_err21_dir"].mean(),
                d["abs_err23_npe"].mean(),
                d["abs_err23_dir"].mean(),
            ],
            "Median_AE": [
                d["abs_err21_npe"].median(),
                d["abs_err21_dir"].median(),
                d["abs_err23_npe"].median(),
                d["abs_err23_dir"].median(),
            ]
        })

        print(summary.to_string(index=False))
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0,0].boxplot([d["err21_npe"], d["err21_dir"]],
                          labels=["NPE", "Direct"])
        axes[0,0].axhline(0)
        axes[0,0].set_title("Normal_10: Error in log BF21")
        axes[0,0].set_ylabel("Δ log BF21")
        axes[0,0].grid(True, axis="y", alpha=0.3)

        axes[0,1].boxplot([d["err23_npe"], d["err23_dir"]],
                      labels=["NPE", "Direct"])
        axes[0,1].axhline(0)
        axes[0,1].set_title("Normal_10: Error in log BF23")
        axes[0,1].set_ylabel("Δ log BF23")
        
        axes[0,1].grid(True, axis="y", alpha=0.3)
        axes[1,0].boxplot([d["abs_err21_npe"], d["abs_err21_dir"]],
                      labels=["NPE", "Direct"])
        axes[1,0].set_title("Normal_10: Absolute Error |Δ log BF21|")
        axes[1,0].set_ylabel("|Δ log BF21|")
        axes[1,0].grid(True, axis="y", alpha=0.3)

        axes[1,1].boxplot([d["abs_err23_npe"], d["abs_err23_dir"]],
                      labels=["NPE", "Direct"])
        axes[1,1].set_title("Normal_10: Absolute Error |Δ log BF23|")
        axes[1,1].set_ylabel("|Δ log BF23|")
        axes[1,1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        return d, summary
    
    def plot_M3(df):
        d = df.copy()
        d["err31_npe"] = d["logBF_31_npe"]    - d["logBF_31_gold"]
        d["err31_dir"] = d["logBF_31_direct"] - d["logBF_31_gold"]
        
        d["err32_npe"] = d["logBF_32_npe"]    - d["logBF_32_gold"]
        d["err32_dir"] = d["logBF_32_direct"] - d["logBF_32_gold"]
        
        d["abs_err31_npe"] = np.abs(d["err31_npe"])
        d["abs_err31_dir"] = np.abs(d["err31_dir"])
        
        d["abs_err32_npe"] = np.abs(d["err32_npe"])
        d["abs_err32_dir"] = np.abs(d["err32_dir"])

        summary = pd.DataFrame({
            "pair": ["31","31","32","32"],
            "method": ["NPE","Direct","NPE","Direct"],
            "MAE": [
                d["abs_err31_npe"].mean(),
                d["abs_err31_dir"].mean(),
                d["abs_err32_npe"].mean(),
                d["abs_err32_dir"].mean(),
            ],
            "Median_AE": [
                d["abs_err31_npe"].median(),
                d["abs_err31_dir"].median(),
                d["abs_err32_npe"].median(),
                d["abs_err32_dir"].median(),
            ]
        })

        print(summary.to_string(index=False))
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0,0].boxplot([d["err31_npe"], d["err31_dir"]],
                          labels=["NPE", "Direct"])
        axes[0,0].axhline(0)
        axes[0,0].set_title("Student: Error in log BF31")
        axes[0,0].set_ylabel("Δ log BF31")
        axes[0,0].grid(True, axis="y", alpha=0.3)

        axes[0,1].boxplot([d["err32_npe"], d["err32_dir"]],
                      labels=["NPE", "Direct"])
        axes[0,1].axhline(0)
        axes[0,1].set_title("Student: Error in log BF32")
        axes[0,1].set_ylabel("Δ log BF32")
        
        axes[0,1].grid(True, axis="y", alpha=0.3)
        axes[1,0].boxplot([d["abs_err31_npe"], d["abs_err31_dir"]],
                      labels=["NPE", "Direct"])
        axes[1,0].set_title("Student: Absolute Error |Δ log BF31|")
        axes[1,0].set_ylabel("|Δ log BF31|")
        axes[1,0].grid(True, axis="y", alpha=0.3)

        axes[1,1].boxplot([d["abs_err32_npe"], d["abs_err32_dir"]],
                      labels=["NPE", "Direct"])
        axes[1,1].set_title("Student: Absolute Error |Δ log BF32|")
        axes[1,1].set_ylabel("|Δ log BF32|")
        axes[1,1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

        return d, summary