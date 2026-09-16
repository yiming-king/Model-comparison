# Gaussian：使用 Python 统一训练

Python 入口将 m1–m4 的 DeepSet 和 CouplingFlow 统一为 BayesFlow 默认网络配置，并覆盖 D=20、N=10/100、S=D/2D/4D。原训练 notebook 保留为历史实验记录；已有 notebook、模型权重及结果不会因为 Python 配置改变而自动更新。

如果保持旧训练配置，只补 N=100 下 S=D 和 S=4D，确实还缺 **2 个摘要维数 × 4 个模型 = 8 个网络**。现在改变了网络配置，为了按相同设定比较 N=10 和 N=100，应训练 **2 个 N × 3 个 S × 4 个模型 = 24 个网络**，然后重新计算对应的 calibration / OOD 结果。

## 配置与命名

所有预设的参数/数据维数 D 都是 20。预设中的 `20d` / `40d` / `80d` 表示 DeepSet 输出的摘要维数 S。

| `--preset` | D | N | S | 摘要大小 |
| --- | ---: | ---: | ---: | --- |
| `20d_10n` | 20 | 10 | 20 | S=D |
| `40d_10n` | 20 | 10 | 40 | S=2D |
| `80d_10n` | 20 | 10 | 80 | S=4D |
| `20d_100n` | 20 | 100 | 20 | S=D |
| `40d_100n` | 20 | 100 | 40 | S=2D |
| `80d_100n` | 20 | 100 | 80 | S=4D |

m1–m4 均使用：

```python
summary_network = bf.networks.DeepSet(summary_dim=S)
inference_network = bf.networks.CouplingFlow()
```

`summary_dim` 是实验变量，其余网络参数交给已安装 BayesFlow 的默认值。当前环境为 **BayesFlow 2.0.12**：DeepSet 默认 `base_distribution=None`，没有摘要分布 MMD 正则化；CouplingFlow 默认 depth=6，MLP widths=(256,256)、norm=None，使用 affine transform、random permutation、ActNorm 和 normal 推断基分布。后者是推断网络的基分布，与 DeepSet 的摘要正则化是两项设置。

训练设置也在全部 24 个组合间统一：**100 epochs、batch size 64、每 epoch 128 batches、Adam + CosineDecay 初始学习率 1e-4**，使用 `standardize="all"` 和 seed=2025。优化器与学习率调度沿用本项目设置；“BayesFlow 默认”指网络构造，不表示采用 BayesFlow 的默认训练预算或优化器。每个网络默认共 12,800 次梯度更新、819,200 个模拟数据集；每个数据集包含 N 条观测。

模型分布保持原设定：

| 模型 | prior mean | prior std | likelihood std |
| --- | ---: | ---: | ---: |
| `m1` | 0 | 1 | 1 |
| `m2` | 3 | 1 | 1 |
| `m3` | 0 | 1 | 3 |
| `m4` | 0.1 | 1 | 1 |

新网络名自动带 `_bf_default`，例如 `m1_s_20d_100n_bf_default.keras`，因此不会因旧的 `m1_s_20d_100n.keras` 存在而跳过新训练。每次训练保存 `.keras`、`.history.json` 和 `.config.json`；配置文件记录实际训练设置及 BayesFlow / Keras 版本。默认目录是 `benchmark/examples/gaussian/networks/`，批量训练日志在 `benchmark/examples/gaussian/training_logs/`。

`--summary-mmd` 显式启用 `base_distribution="normal"` 并增加 `_mmd`，例如 `m1_s_20d_100n_bf_default_mmd.keras`。`--no-summary-mmd` 保留为显式选择默认行为。`--run-suffix trial` 放在这些标识之后，例如 `m1_s_20d_100n_bf_default_trial.keras`。改变 epochs、学习率或 seed 等设置时，使用新的 run suffix 区分实验；升级 BayesFlow 后也应使用新的 run suffix，因为库的默认值可能变化。

## 从仓库根目录运行

以下命令在 `/Users/yimingzang/Documents/Project/benchmark2` 下执行，使用已有的 `benchmark2` 环境。

先查看完整的 24 个任务及输出路径，不开始训练：

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.approximators.train_grid --dry-run
```

统一训练全部 24 个网络：

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.approximators.train_grid --max-workers 1
```

只训练 N=100、S=D 和 S=4D 的 8 个网络：

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.approximators.train_grid \
  --presets 20d_100n 80d_100n --models m1 m2 m3 m4 --max-workers 1
```

这条 8 网络命令使用新的默认配置；要做统一配置的比较，还需重训其余 16 个组合。完整命令可再次运行：遇到本次配置的同名模型时默认复用/跳过，`--overwrite` 才会从头重训并替换它。

长时间运行前，可先检查 N=100、S=4D 的一个小规模训练，输出放在临时目录：

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.approximators.indirect \
  --preset 80d_100n --models m1 --epochs 1 --batch-size 4 --num-batches 1 \
  --validation-size 4 --output-dir /private/tmp/gaussian-training-smoke --run-suffix smoke
```

小规模训练仅用于检查环境、训练和保存流程，不能作为比较结果。重复检查同一路径时可加 `--overwrite`。

单独训练一组配置，或添加独立验证数据：

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.approximators.indirect \
  --preset 40d_100n --models m1 m2 m3 m4 --validation-size 200
```

验证数据使用独立随机数生成器，不消耗训练模拟器的 NumPy 随机数序列。批量运行默认一个 worker、每 worker 四个线程；增加 worker 数会增加内存及设备资源占用。

## 后续 calibration / OOD

新旧网络在同一目录时，分析必须用 `--network-tags` 选择一套实验。默认新标签为：

| N | `--network-tags` |
| ---: | --- |
| 10 | `20d_10n_bf_default 40d_10n_bf_default 80d_10n_bf_default` |
| 100 | `20d_100n_bf_default 40d_100n_bf_default 80d_100n_bf_default` |

如果使用 `--run-suffix` 或 `--summary-mmd`，这些标签也要加上相应后缀。分开运行两个 N 的校准，并将本轮输出保存到独立目录：

```bash
/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline all \
  --num-dims 20 --num-obs 10 \
  --network-tags 20d_10n_bf_default 40d_10n_bf_default 80d_10n_bf_default \
  --calibration-root benchmark/examples/gaussian/calibration_outputs/bf_default

/opt/anaconda3/envs/benchmark2/bin/python -m benchmark.examples.gaussian.calibration.pipeline all \
  --num-dims 20 --num-obs 100 \
  --network-tags 20d_100n_bf_default 40d_100n_bf_default 80d_100n_bf_default \
  --calibration-root benchmark/examples/gaussian/calibration_outputs/bf_default
```

随后按 [分析指南](../README.md) 运行 OOD pipeline，设置匹配的 `--num-obs`、上述 `--network-tags`，并通过 `--threshold-path` 指向本轮的 `calibration_outputs/bf_default/20d_10n/thresholds.csv` 或 `calibration_outputs/bf_default/20d_100n/thresholds.csv`。原分析指南的无 `_bf_default` 标签对应历史模型，使用新模型时须替换。

重新训练后，必须重新计算每个网络的后验、logML、PMP、summary reference 和阈值；旧结果不能直接用于新网络。新标签会区分 OOD 输出；如果原地重训了同一个新标签，用 `--overwrite-inference --overwrite-reference` 重新生成它的结果。现有 summary diagnostics 使用拟合得到的经验参考分布，不要求 DeepSet 输出为标准正态，因此默认关闭摘要 MMD 后仍可使用；训练中的摘要 MMD 正则化与分析中的 Kernel diagnostic 也不是同一项设置。

## 常用参数与 Python 接口

| 参数 | 用途 |
| --- | --- |
| `--preset` / `--presets` | 单组/批量入口选择预设；批量默认全部六个 |
| `--models` | 选择 m1–m4 中的模型；默认全部 |
| `--num-dims` / `--num-obs` / `--summary-dim` | 单组入口覆盖 D、N、S |
| `--epochs` / `--batch-size` / `--num-batches` | 覆盖共同训练预算 |
| `--learning-rate` / `--seed` | 覆盖初始学习率和训练种子 |
| `--summary-mmd` / `--no-summary-mmd` | 启用/关闭摘要 MMD；默认关闭 |
| `--validation-size` / `--validation-freq` / `--validation-seed` | 验证集大小、频率和种子 |
| `--output-dir` / `--run-suffix` | 输出目录与实验后缀 |
| `--overwrite` | 从头重训并替换已有同名结果 |
| `--dry-run` | 只列出任务及配置 |
| `--max-workers` / `--threads-per-worker` | 批量入口并发数与线程数 |

```python
from benchmark.examples.gaussian.approximators.config import TrainingConfig
from benchmark.examples.gaussian.approximators.indirect import (
    train_approximator,
    load_approximator,
    load_history,
)

config = TrainingConfig.from_preset("20d_100n")
approximator, history = train_approximator("m1", config=config)

# 后续进程使用同一配置加载本次模型。
approximator = load_approximator("m1", config=config)
history = load_history("m1", config=config)
```

`config.py` 管理维数、共同训练设置与命名，`simulators.py` 管理模拟器，`indirect.py` 构建并训练网络，`train_grid.py` 负责独立子进程中的批量训练。优化器通过 `BasicWorkflow(optimizer=...)` 传入，使 Adam + CosineDecay 在首次 `fit` 时仍然生效。Python 入口同时固定 NumPy 模拟器和 Keras 的种子；默认不执行 notebook 中的 posterior sampling 或绘图。
