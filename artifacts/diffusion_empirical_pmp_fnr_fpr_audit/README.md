# Empirical signed PMP error / Table 8 verification

Recomputed from the diagnostic CSVs and calibration thresholds used by `benchmark/examples/diffusion/notebooks/summary_noMMDloss_runs_M1_M3_2x4.ipynb`. The audit includes all four models M0–M3, rather than just M1/M3 displayed in the plot. No plot or workbook was modified.

All 96 combinations (2 runs × 4 models × 3 S × 4 diagnostics) agree with the two workbooks, including counts, thresholds, FNR, FPR and AUC. Rates in the attached CSV are percentages. There are exactly 17 empirical cases in each combination, with finite error values, diagnostic scores and thresholds.

44 FNR entries and 44 AUC entries are undefined. In every instance there are zero error-positive cases and 17 error-negative cases. FNR=FN/(FN+TP)=0/0 is undefined, not zero. AUC requires both classes. FPR remains defined because its denominator FP+TN is 17.

Zero-positive groups: M0, both runs, all three S; M2, Run 1, all three S; M2, Run 2, S=2D and S=4D. Each group yields four N/A entries because it is assessed by four diagnostics. No M1/M3 FNR entries are N/A.

Run 1 / M0 / S=D / L2: TP=0,FN=0,FP=14,TN=3; FNR=N/A, FPR=14/17=82.4%, AUC=N/A. Its largest absolute empirical PMP error is about 5.33e-8, within the reference interval [-0.005361,0.004340].

Run 2 / M2 / S=D has only one error-positive empirical case, p6: signed PMP error 0.0470428 exceeds q95=0.00348768. All four diagnostics detect this case and rank it above the 16 negatives, giving FNR=0% and AUC=100%. These values describe one positive case; they do not establish broadly perfect detection performance.

Workbook references: `FNR-FPR/FNR-FPR tables without MMD.xlsx` (Run 1), `FNR-FPR/FNR-FPR tables without MMD rerun1.xlsx` (Run 2). `Empirical data` ranges: M0 C168:F179; M1 C188:F199; M2 C208:F219; M3 C228:F239. Source rates rows: M0 370:381, M1 34:45, M2 386:397, M3 50:61.
