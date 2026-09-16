# Diffusion signed PMP plot / FNR–FPR reconciliation

The notebook's raw plotted data were reclassified using their original, network-specific PMP reference intervals. The 48 combinations (2 runs × 2 assumed models × 3 S settings × 4 diagnostics) match both source workbooks: all 96 FNR/FPR values, all 240 TP/FP/FN/TN/N values, and all 96 interval endpoints agree.

Source notebook: `benchmark/examples/diffusion/notebooks/summary_noMMDloss_runs_M1_M3_2x4.ipynb`.
Run 1 workbook: `FNR-FPR/FNR-FPR tables without MMD.xlsx`.
Run 2 workbook: `FNR-FPR/FNR-FPR tables without MMD rerun1.xlsx`.
Workbook locations: `Simulated data!C188:F199` (M1), `C228:F239` (M3); raw counts in `Source rates!H130:P141` and `H146:P157`; definitions in `Method!B8:B13`.

For each combination, simulated data comprise all 119 nonempirical rows: 17 each from `simulated_from_m0`, `simulated_from_m1`, `simulated_from_m2`, `simulated_from_m3`, `m3_fast_30`, `m3_slow_30`, and `m3_fast_slow_30`. The additional 17 empirical observations, dataset medians, and fitted curves drawn in the original figure are not cases in these rates.

Actual positive: raw signed PMP error < q05 or > q95. Diagnostic positive: rho > 1. FNR = FN/(FN+TP); FPR = FP/(FP+TN). The lower diagnostic cutoff does not define predicted positive. Symlog affects display only.

Run 1 / M1 / S=D / L2: TP=5, FN=8, FP=39, TN=67; FNR=8/13=61.5%, FPR=39/106=36.8%.
Run 1 / M3 / S=2D / L2: TP=14, FN=46, FP=20, TN=39; FNR=46/60=76.7%, FPR=20/59=33.9%. The reference interval is approximately [-3.24188e-5, 1.09227e-5]. 37 of these 46 false negatives have absolute PMP error <= 1e-3.

The original green horizontal band spans the minimum lower and maximum upper threshold across all three S settings. Thus it does not indicate the correct negative region for every colored point. Each point must be compared against its own S-specific threshold lines. In the attached audit plot, each panel contains only one S setting and only simulated raw observations; orange-red crosses identify FN.

`plot_vs_table_counts.csv` contains aggregate counts and rates (rates are percentages). `point_classifications.csv` contains the plotted simulated points and classifications. `reconciliation.json` records the workbook comparison.
