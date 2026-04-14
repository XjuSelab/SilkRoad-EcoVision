# 服务器训练说明

## 快速开始

先准备数据：

```bash
uv sync --dev
uv run csiro-biomass prepare-data --zip-path csiro-biomass.zip --extract-images
```

直接启动一个服务器实验：

```bash
uv run csiro-biomass train-supervised --config configs/server/supervised-dinov3-vitl-896.yaml
```

跑完一个实验根目录后，聚合 `OOF`：

```bash
uv run csiro-biomass oof aggregate \
  --experiment-root artifacts/server/dinov3-vitl-896 \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet
```

多个实验跑完后，做 teacher 选择：

```bash
uv run csiro-biomass oof select \
  --experiment-root artifacts/server/dinov3-vitl-896 \
  --experiment-root artifacts/server/dinov3-vitl-1024 \
  --experiment-root artifacts/server/dinov2-vitl-518 \
  --experiment-root artifacts/server/dinov2-vitg-518 \
  --experiment-root artifacts/server/siglip-so400m-384 \
  --experiment-root artifacts/server/siglip-so400m-448 \
  --output-dir artifacts/server/teacher-selection \
  --top-k 4 \
  --correlation-threshold 0.985
```

如果要做带 `TTA` 和 checkpoint averaging 的推理：

```bash
uv run csiro-biomass infer --config configs/infer-ensemble.yaml
```

## 当前主线目标

当前服务器主线不是为了提前做蒸馏，而是为了把 CSIRO 数据集上的分数做高。

所以顺序应该是：

1. 跑强 backbone 和不同输入尺寸的服务器实验。
2. 做完整 `OOF` 聚合。
3. 筛出最强 single model 和最优 ensemble。
4. 最后再考虑蒸馏或边缘部署。

这和当前仓库策略一致：

- 先做 score-first server workflow
- 蒸馏是后续部署路线，不是当前主线

## 当前支持的服务器配置

仓库内已经提供这些服务器配置：

- `configs/server/supervised-dinov3-vitl-896.yaml`
- `configs/server/supervised-dinov3-vitl-1024.yaml`
- `configs/server/supervised-dinov2-vitl-518.yaml`
- `configs/server/supervised-dinov2-vitg-518.yaml`
- `configs/server/supervised-siglip-so400m-384.yaml`
- `configs/server/supervised-siglip-so400m-448.yaml`

它们的意义不是一次只跑一个模型，而是把 `8xH200` 当作实验池，快速并行筛 backbone、分辨率、`fold` 和 `seed`。

## 一次训练到底在做什么

可以把一次训练理解成“把图像变成 biomass 数值，再用验证集筛模型”。

主要步骤是：

1. 读取训练 manifest 和 fold manifest。  
   程序先知道哪部分样本用来训练，哪部分样本用来验证。

2. 把原始宽图切成左右两半。  
   当前任务不是单图输入，而是双流输入。每张原图会拆成左图和右图。

3. 做数据增强。  
   包括翻转、旋转、亮度对比度变化、噪声、缩放等，目的是避免模型死记硬背。

4. 左右图分别进入 backbone。  
   比如 `DINOv3-L`、`DINOv2-L/G`、`SigLIP SO400M`。

5. 融合左右特征。  
   当前实现会把左右流特征投影后融合，再送入 trunk。

6. 输出 `10` 个头。  
   其中：
   - `5` 个回归头直接预测目标值
   - `5` 个区间分类头预测每个目标落在哪个区间

7. 计算 loss。  
   当前主损失是：
   - `SmoothL1` 管回归误差
   - `CrossEntropy` 管区间分类误差

8. 反向传播并更新参数。  
   用 `AdamW` 按 batch 逐步更新模型参数。

9. 在验证集上评估。  
   每个 epoch 结束后，模型会在验证集上输出 `valid_loss`、`valid_r2` 等指标。

10. 保存最优 checkpoint。  
    只要验证效果更好，就更新 `best.pt`。

## 为什么训练要分阶段

当前训练一般分成两个阶段：

### Stage 1：冻结 backbone

- 只训练后面的融合层和 heads
- 相当于先学会如何使用已有视觉特征

这样更稳，尤其当前数据量不大时，不容易一上来就把预训练表征训坏。

### Stage 2：解冻 backbone

- 让 backbone 也参与微调
- 让特征更适配 biomass 这个任务

可以理解成：

- 第一阶段先学会“怎么答题”
- 第二阶段再去微调“看题的方法”

## 为什么要跑 folds 和 seeds

单个实验结果不稳，可能只是运气。

### `fold`

不同 `fold` 对应不同验证集划分。  
如果模型在不同 `fold` 上都稳，说明它不是只适配某一批样本。

### `seed`

不同 `seed` 会改变初始化和数据顺序。  
如果不同 `seed` 都稳，说明模型不是靠随机性碰巧出成绩。

所以服务器上跑 `fold x seed` 的目的，是把运气因素压下去。

## 为什么要做 OOF

`OOF` 可以理解为：

- 每个样本都在“没见过它的模型”上被预测过一次

这样得到的预测更公平，也更适合做这些事情：

- 比较不同实验谁更强
- 做 teacher selection
- 看哪些 target 仍然学得差
- 做后续 ensemble

当前仓库里，`oof aggregate` 就是在做这件事。

## 为什么要做 TTA 和 checkpoint averaging

### TTA

推理时对同一张图做几个变换后再平均，例如：

- 原图
- 水平翻转
- 垂直翻转
- `rot90`

这样通常能降低单次推理的偶然误差。

### checkpoint averaging

同一个成员可以先把多个 checkpoint 平均，再作为一个 ensemble 成员使用。

这样做的好处是：

- 比单个 checkpoint 更稳
- 通常比盲目增大 ensemble 成员数量更划算

## 训练显存为什么会大

主要原因有五个。

### 1. 当前是双流输入

每个样本不是一张图，而是左右两张图。  
即使 backbone 权重共享，也要前向两次，成本接近翻倍。

### 2. ViT 的显存和分辨率增长很快

真正吃显存的不只是参数，而是中间激活，尤其 attention。  
图像边长增加时，token 数按面积增长，所以显存涨得很快。

### 3. 训练比推理贵得多

训练时除了前向，还要保存：

- 激活
- 梯度
- optimizer state
- 反向传播图

所以训练显存远高于推理显存。

### 4. 当前 backbone 本身就很大

`DINOv3-L`、`DINOv2-G`、`SigLIP SO400M` 都不是轻量模型。

### 5. 当前主线配置没有极致省显存优化

比如没有默认打开：

- gradient checkpointing
- 更激进的并行切分
- 更复杂的内存优化策略

所以目前的显存预估是“能稳跑，但不是极致压显存”的口径。

## 训练显存占用预估

下面是当前服务器配置的大致单卡训练峰值预估。

### `dinov3-vitl-896`

- `batch_size: 4`
- `grad_accum_steps: 4`
- 预计约 `35GB - 55GB`

### `dinov3-vitl-1024`

- `batch_size: 2`
- `grad_accum_steps: 8`
- 预计约 `40GB - 65GB`

### `dinov2-vitl-518`

- `batch_size: 4`
- `grad_accum_steps: 4`
- 预计约 `18GB - 30GB`

### `dinov2-vitg-518`

- `batch_size: 2`
- `grad_accum_steps: 8`
- 预计约 `35GB - 55GB`

### `siglip-so400m-384`

- `batch_size: 4`
- `grad_accum_steps: 4`
- 预计约 `18GB - 28GB`

### `siglip-so400m-448`

- `batch_size: 4`
- `grad_accum_steps: 4`
- 预计约 `22GB - 32GB`

这些值是经验估算，不是严格 benchmark。  
真正跑时，还是要用 `nvidia-smi` 实测。

建议监控命令：

```bash
watch -n 1 nvidia-smi
```

## 推理显存占用预估

推理显存会比训练小很多，因为不需要反向传播。

### `dinov3-vitl-896`

- 预计约 `12GB - 22GB`

### `dinov3-vitl-1024`

- 预计约 `16GB - 28GB`

### `dinov2-vitl-518`

- 预计约 `6GB - 12GB`

### `dinov2-vitg-518`

- 预计约 `12GB - 22GB`

### `siglip-so400m-384`

- 预计约 `5GB - 10GB`

### `siglip-so400m-448`

- 预计约 `6GB - 12GB`

注意：

- 当前是双流输入，所以推理也不是普通单图成本
- `TTA` 会明显增加总耗时，但当前实现是顺序执行，不会把峰值显存简单乘上 `4`
- checkpoint averaging 主要影响加载阶段，对峰值推理显存影响不大
- 真正明显影响推理显存的是 `infer.batch_size`

## H200 服务器该怎么用

`H200` 不应该只是拿来把一个小实验跑得更快，而应该当成实验工厂。

更合理的用法是：

- 并行跑多个 backbone 和输入尺寸
- 并行跑不同 `fold` 和 `seed`
- 尽快完成 `OOF` 聚合
- 用结果筛出最优 single model 和 ensemble

也就是说：

- `H200` 的价值不在“让一个短实验更快结束”
- 而在“扩大搜索空间，提高研究效率”

## 推荐的服务器使用顺序

建议按风险和收益这样排：

1. 先跑 `supervised-dinov3-vitl-896.yaml`
2. 再跑 `supervised-dinov3-vitl-1024.yaml`
3. 再跑 `supervised-dinov2-vitl-518.yaml`
4. 再补 `supervised-dinov2-vitg-518.yaml`
5. 最后用 `siglip` 两档分辨率补多样性

原因是：

- `dinov3-vitl-896` 通常是最稳的第一站
- `1024` 更接近冠军题解主力尺寸
- `dinov2` 和 `siglip` 更重要的价值是给 ensemble 带来异构性

## 结果目录长什么样

一个完整实验根目录通常会写成这样：

```text
artifacts/server/<experiment>/
  fold0_seed42/
  fold0_seed3407/
  fold1_seed42/
  fold1_seed3407/
  fold2_seed42/
  fold2_seed3407/
  oof_predictions.parquet
  oof_metrics.csv
  oof_summary.json
  run_summaries.json
```

每个 `fold_seed` 子目录内通常会包含：

- `best.pt`
- `history.json`
- `valid_predictions.parquet`
- 若干 summary 文件

## 当前最重要的验收标准

服务器实验不是只看单个数字。

至少要同时看：

- `mean_3fold` 层面的整体效果
- 各 target 的 `corr / MAE / RMSE`
- `pred_std` 是否正常，避免近常数预测
- 不同 `fold` 之间是否稳定

尤其要重点关注：

- `Dry_Green_g`
- `GDM_g`
- `Dry_Total_g`

这几个目标更容易看出模型是否真的学到了东西。

## 推理阶段的配置示意

当前仓库支持在推理配置里声明 `TTA` 和 checkpoint averaging，例如：

```yaml
infer:
  tta_policies: [identity, hflip, vflip, rot90]
  members:
    - checkpoints:
        - artifacts/server/dinov3-vitl-1024/fold0_seed42/best.pt
        - artifacts/server/dinov3-vitl-1024/fold0_seed3407/best.pt
      weight: 0.35
```

这里的意思是：

- 先把两个 checkpoint 平均成一个 member
- 再按 member 级别参与整体 ensemble 加权

## 总结

当前服务器文线的核心逻辑是：

1. 用大 backbone 和高分辨率做强监督训练。
2. 用 `fold x seed` 压低运气因素。
3. 用 `OOF` 做公平比较和筛选。
4. 用 `TTA + checkpoint averaging + ensemble` 做最终提升。

当前阶段最重要的不是蒸馏，而是把 teacher 练强。  
只有在最优 server-side teacher 稳定之后，边缘 student 或 Hailo-8 路线才有意义。
