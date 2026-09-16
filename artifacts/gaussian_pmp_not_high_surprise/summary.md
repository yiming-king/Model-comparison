# Gaussian PMP error：at least one not high surprise

模型指 assumed model M1–M4，包含全部生成模型 M1–M12。S=D、2D、4D 分别对应 20d_10n、40d_10n、80d_10n 的现代 diagnostic cache，与 summary dimension comparison notebook 一致。

筛选 at_least_one_not_extrapolative == True，即至少一个 assumed model 的 regime 不是 extrapolation。误差为 abs(signed_pmp_error_npe_Mj)，原始 NPE − analytical PMP error 取绝对值，未按 calibration threshold 归一化。

已核对 12 个 cache 的分类标记、dataset 唯一性、误差有限性，以及误差与 p_npe - p_gold 的一致性（容许浮点舍入误差）。

## 最大绝对误差

| Diagnostic | Assumed model | S=D | S=2D | S=4D |
|---|---|---:|---:|---:|
| l2 | M1 | 0.02410331712 | 0.01758704794 | 0.02645123509 |
| l2 | M2 | 8.54025739e-11 | 3.07838677e-13 | 5.114619839e-11 |
| l2 | M3 | 1.037830092e-41 | 1.239553684e-41 | 3.894000886e-42 |
| l2 | M4 | 0.02410331712 | 0.01758704794 | 0.02645123509 |
| linf | M1 | 0.6688152299 | 0.03336810249 | 0.02455195518 |
| linf | M2 | 8.900086719e-06 | 0.01364861401 | 0.05894282754 |
| linf | M3 | 0.9999999698 | 2.161251053e-41 | 2.699047118e-42 |
| linf | M4 | 0.8579319305 | 0.03342751545 | 0.06722693146 |
| density | M1 | 0.7893423988 | 0.01029640206 | 0.01810517528 |
| density | M2 | 0.02233159408 | 2.586711734e-09 | 4.013610044e-07 |
| density | M3 | 1 | 1.239553684e-41 | 3.894000886e-42 |
| density | M4 | 0.8579319605 | 0.01029640206 | 0.01810517528 |
| mmd | M1 | 0.02410331712 | 0.01242662997 | 0.01826537141 |
| mmd | M2 | 8.54025739e-11 | 3.07838677e-13 | 1.02915456e-09 |
| mmd | M3 | 1.037830092e-41 | 1.239553684e-41 | 3.894000886e-42 |
| mmd | M4 | 0.02410331712 | 0.01242662997 | 0.01826537141 |

## 最小绝对误差

除以下两个组合外，最小值均为 CSV 中存储的 0。0 和极小值可能反映浮点精度或下溢，不代表理论误差必定为零。

| Diagnostic | Assumed model | Summary | Min abs error | Dataset |
|---|---|---|---:|---|
| density | M1 | S=4D | 2.112505434e-284 | m3, id=16 |
| density | M4 | S=4D | 1.075115242e-285 | m3, id=16 |

## 极值对应 dataset

dataset 使用 source_model + 原始 id。并列最小值仅列一个示例，括号标记并列数量。

| Diagnostic | Assumed model | Summary | N selected / total | Min abs error | Min dataset (ties) | Max abs error | Max dataset | Signed error at max |
|---|---|---|---:|---:|---|---:|---|---:|
| l2 | M1 | S=D | 450/600 | 0 | m7, id=0 (50) | 0.02410331712 | m9, id=13 | -0.02410331712 |
| l2 | M2 | S=D | 450/600 | 0 | m2, id=0 (98) | 8.54025739e-11 | m5, id=15 | -8.54025739e-11 |
| l2 | M3 | S=D | 450/600 | 0 | m3, id=0 (100) | 1.037830092e-41 | m1, id=27 | 1.037830092e-41 |
| l2 | M4 | S=D | 450/600 | 0 | m7, id=0 (50) | 0.02410331712 | m9, id=13 | 0.02410331712 |
| l2 | M1 | S=2D | 397/600 | 0 | m7, id=0 (40) | 0.01758704794 | m9, id=43 | -0.01758704794 |
| l2 | M2 | S=2D | 397/600 | 0 | m2, id=0 (87) | 3.07838677e-13 | m5, id=32 | -3.07838677e-13 |
| l2 | M3 | S=2D | 397/600 | 0 | m3, id=0 (88) | 1.239553684e-41 | m1, id=38 | 1.239553684e-41 |
| l2 | M4 | S=2D | 397/600 | 0 | m7, id=0 (40) | 0.01758704794 | m9, id=43 | 0.01758704794 |
| l2 | M1 | S=4D | 314/600 | 0 | m7, id=3 (10) | 0.02645123509 | m10, id=17 | 0.02645123509 |
| l2 | M2 | S=4D | 314/600 | 0 | m2, id=0 (58) | 5.114619839e-11 | m5, id=15 | -5.114619839e-11 |
| l2 | M3 | S=4D | 314/600 | 0 | m3, id=0 (57) | 3.894000886e-42 | m1, id=27 | -3.894000886e-42 |
| l2 | M4 | S=4D | 314/600 | 0 | m7, id=3 (10) | 0.02645123509 | m10, id=17 | -0.02645123509 |
| linf | M1 | S=D | 463/600 | 0 | m7, id=0 (50) | 0.6688152299 | m8, id=3 | 0.6688152299 |
| linf | M2 | S=D | 463/600 | 0 | m2, id=0 (97) | 8.900086719e-06 | m5, id=21 | 8.900086719e-06 |
| linf | M3 | S=D | 463/600 | 0 | m3, id=0 (100) | 0.9999999698 | m8, id=1 | 0.9999999698 |
| linf | M4 | S=D | 463/600 | 0 | m7, id=0 (50) | 0.8579319305 | m8, id=1 | -0.8579319305 |
| linf | M1 | S=2D | 485/600 | 0 | m7, id=0 (45) | 0.03336810249 | m5, id=41 | 0.03336810249 |
| linf | M2 | S=2D | 485/600 | 0 | m2, id=0 (102) | 0.01364861401 | m5, id=9 | 0.01364861401 |
| linf | M3 | S=2D | 485/600 | 0 | m3, id=0 (94) | 2.161251053e-41 | m1, id=38 | 2.161251053e-41 |
| linf | M4 | S=2D | 485/600 | 0 | m7, id=0 (45) | 0.03342751545 | m5, id=41 | -0.03342751545 |
| linf | M1 | S=4D | 449/600 | 0 | m7, id=3 (20) | 0.02455195518 | m10, id=22 | 0.02455195518 |
| linf | M2 | S=4D | 449/600 | 0 | m2, id=0 (68) | 0.05894282754 | m5, id=47 | 0.05894282754 |
| linf | M3 | S=4D | 449/600 | 0 | m3, id=0 (68) | 2.699047118e-42 | m1, id=38 | 2.699047118e-42 |
| linf | M4 | S=4D | 449/600 | 0 | m7, id=3 (20) | 0.06722693146 | m5, id=47 | -0.06722693146 |
| density | M1 | S=D | 511/600 | 0 | m7, id=0 (50) | 0.7893423988 | m8, id=30 | -0.7893423988 |
| density | M2 | S=D | 511/600 | 0 | m2, id=0 (110) | 0.02233159408 | m5, id=9 | -0.02233159408 |
| density | M3 | S=D | 511/600 | 0 | m3, id=0 (100) | 1 | m8, id=22 | 1 |
| density | M4 | S=D | 511/600 | 0 | m7, id=0 (50) | 0.8579319605 | m8, id=1 | -0.8579319605 |
| density | M1 | S=2D | 302/600 | 0 | m7, id=12 (4) | 0.01029640206 | m1, id=29 | 0.01029640206 |
| density | M2 | S=2D | 302/600 | 0 | m2, id=0 (51) | 2.586711734e-09 | m5, id=29 | -2.586711734e-09 |
| density | M3 | S=2D | 302/600 | 0 | m3, id=0 (52) | 1.239553684e-41 | m1, id=38 | 1.239553684e-41 |
| density | M4 | S=2D | 302/600 | 0 | m7, id=12 (4) | 0.01029640206 | m1, id=29 | -0.01029640206 |
| density | M1 | S=4D | 292/600 | 2.112505434e-284 | m3, id=16 (1) | 0.01810517528 | m1, id=32 | -0.01810517528 |
| density | M2 | S=4D | 292/600 | 0 | m2, id=0 (46) | 4.013610044e-07 | m5, id=20 | -4.013610044e-07 |
| density | M3 | S=4D | 292/600 | 0 | m3, id=0 (47) | 3.894000886e-42 | m1, id=27 | -3.894000886e-42 |
| density | M4 | S=4D | 292/600 | 1.075115242e-285 | m3, id=16 (1) | 0.01810517528 | m1, id=32 | 0.01810517528 |
| mmd | M1 | S=D | 448/600 | 0 | m7, id=0 (50) | 0.02410331712 | m9, id=13 | -0.02410331712 |
| mmd | M2 | S=D | 448/600 | 0 | m2, id=0 (98) | 8.54025739e-11 | m5, id=15 | -8.54025739e-11 |
| mmd | M3 | S=D | 448/600 | 0 | m3, id=0 (99) | 1.037830092e-41 | m1, id=27 | 1.037830092e-41 |
| mmd | M4 | S=D | 448/600 | 0 | m7, id=0 (50) | 0.02410331712 | m9, id=13 | 0.02410331712 |
| mmd | M1 | S=2D | 418/600 | 0 | m7, id=0 (50) | 0.01242662997 | m10, id=16 | -0.01242662997 |
| mmd | M2 | S=2D | 418/600 | 0 | m2, id=0 (96) | 3.07838677e-13 | m5, id=32 | -3.07838677e-13 |
| mmd | M3 | S=2D | 418/600 | 0 | m3, id=0 (100) | 1.239553684e-41 | m1, id=38 | 1.239553684e-41 |
| mmd | M4 | S=2D | 418/600 | 0 | m7, id=0 (50) | 0.01242662997 | m10, id=16 | 0.01242662997 |
| mmd | M1 | S=4D | 398/600 | 0 | m7, id=0 (50) | 0.01826537141 | m10, id=6 | -0.01826537141 |
| mmd | M2 | S=4D | 398/600 | 0 | m2, id=0 (99) | 1.02915456e-09 | m5, id=1 | -1.02915456e-09 |
| mmd | M3 | S=4D | 398/600 | 0 | m3, id=0 (100) | 3.894000886e-42 | m1, id=27 | -3.894000886e-42 |
| mmd | M4 | S=4D | 398/600 | 0 | m7, id=0 (50) | 0.01826537141 | m10, id=6 | 0.01826537141 |

## 数据源

- [benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/l2/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/l2/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/l2/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/l2/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/l2/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/l2/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/linf/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/linf/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/linf/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/linf/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/linf/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/linf/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/density/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/density/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/density/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/density/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/density/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/density/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/mmd/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_20d_10n/diagnostics/mmd/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/mmd/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_40d_10n/diagnostics/mmd/pmp_ambiguity_frame.csv)
- [benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/mmd/pmp_ambiguity_frame.csv](/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/ood_80d_10n/diagnostics/mmd/pmp_ambiguity_frame.csv)
