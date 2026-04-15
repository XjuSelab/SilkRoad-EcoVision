# Hugging Face 同步说明

这份文档的目标不是教你写训练代码，而是帮你看懂当前仓库里的 `data/` 和 `artifacts/` 到底分别代表什么，以及它们在同步到 Hugging Face 时应该怎么拆。

当前建议采用“两个私有仓库”方案：

1. `datasets/XJU-SeLab/csiro-biomass-private`
2. `XJU-SeLab/csiro-biomass-server-models`

不要把数据和权重混在一个仓库里。原因很简单：

- `data/` 更像数据资产，适合放在 Hugging Face Dataset Repo
- `artifacts/server/` 更像训练产物，适合放在 Hugging Face Model Repo
- 分开后，下载、权限管理、后续公开范围控制都会更清楚

## 总体原则

这套仓库的代码默认依赖下面这几个本地路径：

```text
data/raw/csiro-biomass/
data/processed/csiro-biomass/
artifacts/server/<experiment>/
```

所以 Hugging Face 上的内容也应该围绕这三个目录来组织，而不是重新发明一套目录结构。  
最稳的做法是：

- Dataset Repo 里保存 `raw + processed`
- Model Repo 里保存 `artifacts/server`
- 下载回来后，按原路径还原到本地

这样现有 `configs/` 和 CLI 都不用改。

## `data/` 到底是什么

`data/` 不是模型输出，而是训练和推理依赖的数据输入。

当前代码里，`prepare-data` 会把 Kaggle 原始包整理成两层：

```text
data/
├── raw/
│   └── csiro-biomass/
└── processed/
    └── csiro-biomass/
```

### `data/raw/csiro-biomass/`

这一层是“原始比赛数据的解压结果”，本质上还是原始数据，只是已经落成目录。

典型结构：

```text
data/raw/csiro-biomass/
├── train/
├── test/
├── train.csv
├── test.csv
└── sample_submission.csv
```

这些文件分别表示：

- `train/`: 训练图片
- `test/`: 测试图片
- `train.csv`: 训练标签与元信息原表
- `test.csv`: 测试元信息原表
- `sample_submission.csv`: 官方提交模板

这部分的用途是：

- 给训练数据集类读取图片
- 给 `prepare-data` 生成宽表、长表和 fold manifest
- 在需要回溯原始字段时作为原始来源

### `data/raw/csiro-biomass.zip`

这是 Kaggle 原始压缩包本体。

它和 `data/raw/csiro-biomass/` 的关系是：

- `csiro-biomass.zip` 是归档包
- `data/raw/csiro-biomass/` 是解压后的可直接使用目录

默认不建议把两者都作为主要下载入口同时上传，因为内容高度重复。  
推荐策略是：

- 以 `data/raw/csiro-biomass/` 作为默认下载入口
- `csiro-biomass.zip` 只作为可选归档备份

如果 Hugging Face 空间或带宽敏感，可以不传 zip，只传解压目录。

### `data/processed/csiro-biomass/`

这一层不是新数据集，而是代码为训练和推理准备的“结构化清单”。

典型结构：

```text
data/processed/csiro-biomass/
├── folds/
│   └── folds_v1.parquet
└── metadata/
    ├── submission_template.csv
    ├── summary.json
    ├── test_long.parquet
    ├── test_wide.parquet
    ├── train_long.parquet
    └── train_wide.parquet
```

这些文件分别表示：

- `train_long.parquet`: 训练原表的长表镜像，基本保留原始逐 target 记录
- `test_long.parquet`: 测试原表的长表镜像
- `train_wide.parquet`: 一张图一行的训练 manifest，已经把 5 个 biomass target 摊平到同一行
- `test_wide.parquet`: 一张图一行的测试 manifest，保留推理所需字段
- `folds_v1.parquet`: 交叉验证划分结果，告诉训练脚本每张图属于哪个 fold
- `submission_template.csv`: 便于后续导出提交文件时对齐列结构
- `summary.json`: 数据准备阶段的汇总信息，例如 train/test 图像数、fold 分布、target 列表

其中最关键的三个文件是：

- `metadata/train_wide.parquet`
- `metadata/test_wide.parquet`
- `folds/folds_v1.parquet`

因为当前训练和推理配置直接依赖它们。

## `artifacts/server/` 到底是什么

`artifacts/server/` 是服务器训练产物，不是原始数据。

每个实验根目录通常对应一种 backbone 或输入尺寸，例如：

- `dinov2-vitg-518`
- `dinov2-vitl-518`
- `dinov3-vitl-896`
- `dinov3-vitl-1024`
- `dinov3-vitl-896-timm`
- `siglip-so400m-384`
- `siglip-so400m-448`

一个完整实验根目录通常长这样：

```text
artifacts/server/<experiment>/
├── fold0_seed42/
├── fold0_seed3407/
├── fold1_seed42/
├── fold1_seed3407/
├── fold2_seed42/
├── fold2_seed3407/
├── oof_metrics.csv
├── oof_predictions.parquet
├── oof_summary.json
└── run_summaries.json
```

含义可以拆成两层理解：

1. `foldX_seedY/` 是单次训练 run
2. 实验根目录里的 `oof_*` 和 `run_summaries.json` 是跨 run 聚合结果

### `foldX_seedY/` 子目录

例如：

```text
artifacts/server/dinov2-vitg-518/fold0_seed42/
├── best.pt
├── history.json
├── summary.json
├── valid_metrics.csv
└── valid_predictions.parquet
```

这些文件分别表示：

- `best.pt`: 当前 run 的最佳 checkpoint，真正可复用的模型权重
- `history.json`: 每个 epoch 的训练/验证过程记录
- `summary.json`: 当前 run 的摘要信息，例如 backbone、image_size、fold、seed、最佳验证分数
- `valid_metrics.csv`: 当前 run 在验证集上的指标明细
- `valid_predictions.parquet`: 当前 run 在验证集上的逐样本预测结果

如果只想“把模型传上去方便下载”，`best.pt` 是最核心文件。  
但如果希望后续还能复盘、筛选、做 ensemble 分析，建议把同目录下的 `history.json`、`summary.json`、`valid_metrics.csv`、`valid_predictions.parquet` 一并保留。

### 实验根目录聚合文件

这些文件不是单个模型成员，而是把多个 `fold_seed` run 聚合后的结果。

- `oof_predictions.parquet`: 整个实验根目录的 OOF 逐样本预测
- `oof_metrics.csv`: 聚合后的按 target 指标
- `oof_summary.json`: 聚合后的总体分数摘要，例如 `oof_weighted_r2`
- `run_summaries.json`: 收集每个子 run 的 `summary.json`

这几类文件的主要用途是：

- 公平比较不同实验谁更强
- 后续做 teacher selection
- 分析哪些 target 学得好或学得差
- 判断不同 backbone 之间是否有异构性

如果只保留 `best.pt`，以后还能推理；  
但如果把 `oof_*` 也保留下来，后续做模型筛选会轻松很多。

## 推荐的 Hugging Face 仓库拆分

### Dataset Repo

当前仓库：

```text
datasets/XJU-SeLab/csiro-biomass-private
```

建议内容：

```text
csiro-biomass/
├── raw/
│   └── csiro-biomass/
│       ├── train/
│       ├── test/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
└── processed/
    └── csiro-biomass/
        ├── folds/
        │   └── folds_v1.parquet
        └── metadata/
            ├── submission_template.csv
            ├── summary.json
            ├── test_long.parquet
            ├── test_wide.parquet
            ├── train_long.parquet
            └── train_wide.parquet
```

推荐上传：

- 解压后的 `raw/csiro-biomass/`
- 完整的 `processed/csiro-biomass/`

可选上传：

- `data/raw/csiro-biomass.zip`

不建议把 `zip` 当成唯一入口，因为当前代码和配置主要消费的是解压目录。

### Model Repo

当前仓库：

```text
XJU-SeLab/csiro-biomass-server-models
```

建议内容：

```text
server/
├── dinov2-vitg-518/
├── dinov2-vitl-518/
├── dinov3-vitl-896/
├── dinov3-vitl-1024/
├── dinov3-vitl-896-timm/
├── siglip-so400m-384/
└── siglip-so400m-448/
```

每个实验根目录建议保留：

- 完整的 `foldX_seedY/`
- `oof_predictions.parquet`
- `oof_metrics.csv`
- `oof_summary.json`
- `run_summaries.json`

其中每个 `foldX_seedY/` 内建议保留：

- `best.pt`
- `history.json`
- `summary.json`
- `valid_metrics.csv`
- `valid_predictions.parquet`

## 哪些内容应该跳过

不是所有服务器目录都应该直接上传。

默认跳过这些内容：

- 空目录
- 中途中断、没有 `best.pt` 的 run
- 没有 `valid_predictions.parquet` 且无法确认是否跑完的 run
- 临时日志、监控输出、shell 重定向日志

比如你给出的结构里：

- `artifacts/server/dinov3-vitl-1024/fold0_seed42`
- `artifacts/server/dinov3-vitl-896/fold0_seed42`

如果目录里没有完整产物，就不应当作为正式发布内容上传。

更稳的筛选标准是：

1. 子目录里至少存在 `best.pt`
2. 最好同时存在 `history.json`、`summary.json`、`valid_metrics.csv`、`valid_predictions.parquet`
3. 实验根目录最好存在 `oof_summary.json`

只有满足这些条件的实验，才算“可下载、可解释、可复盘”。

## 什么时候该传数据，什么时候该传权重

可以用下面的判断方式：

- 想让别人复现训练或继续训练，就必须有 `data/raw` 和 `data/processed`
- 想让别人直接做推理或继续微调，至少要有 `best.pt`
- 想让别人理解哪个实验最好、为什么选它，还需要 `oof_*` 和 `run_summaries.json`

所以实际最合理的上传组合不是单独某一类文件，而是：

- Dataset Repo: `raw + processed`
- Model Repo: `best.pt + valid_* + history/summary + oof_*`

## 为什么不建议把所有东西混成一个“大仓库”

虽然技术上能传，但不建议这么做。

主要问题有三个：

1. 权限边界不清楚  
   后续如果模型能公开、数据不能公开，拆分就会非常麻烦。

2. 下载成本高  
   只想拿权重的人，不应该被迫同步整套原始图片。

3. 目录语义混乱  
   数据资产和训练结果是两种不同生命周期的对象，放一起后很难维护。

## 下载后如何还原到本地

不管 Hugging Face 上叫什么名字，下载回来后都建议还原成仓库默认路径。

推荐还原后的本地结构：

```text
data/
├── raw/
│   └── csiro-biomass/
└── processed/
    └── csiro-biomass/

artifacts/
└── server/
```

也就是说：

- Dataset Repo 里的 `raw/csiro-biomass/` 还原到 `data/raw/csiro-biomass/`
- Dataset Repo 里的 `processed/csiro-biomass/` 还原到 `data/processed/csiro-biomass/`
- Model Repo 里的 `server/<experiment>/...` 还原到 `artifacts/server/<experiment>/...`

这样现有配置例如：

- `data.image_root: data/raw/csiro-biomass`
- `data.train_manifest: data/processed/csiro-biomass/metadata/train_wide.parquet`
- `data.fold_manifest: data/processed/csiro-biomass/folds/folds_v1.parquet`
- `train.output_dir: artifacts/server/<experiment>`

都可以直接继续工作。

## 私有发布阶段的建议口径

当前更适合先按“私有镜像”来理解，而不是按“公开发布资产”来理解。

建议口径是：

- 先完整保存内部可复现所需的 `raw + processed + server artifacts`
- 后续如果要公开，再单独重新审查哪些内容允许公开

尤其需要谨慎对待：

- 比赛原始图片
- 比赛原始 CSV
- 由比赛原始数据直接派生出的 metadata

如果未来要公开，应该先重新确认比赛条款和授权边界，再决定哪些内容能转成公开仓库。

## 一句话总结

如果你现在的目标只是“把服务器上的东西传到 Hugging Face 上方便下载”，最稳的做法就是：

- 用一个私有 Dataset Repo 放 `data/raw` 和 `data/processed`
- 用一个私有 Model Repo 放 `artifacts/server`
- 默认只上传完整实验，不上传空目录和半成品
- 下载后按仓库原路径还原，不改现有配置
