# 服务器实验记录

## 当前目标

这份文档用于持续记录 `CSIRO biomass` 在服务器上的实验结果，并给出当前阶段建议要跑的训练清单。

当前目标仍然是：

- 先把 server-side single model 和 ensemble 做强
- 先筛强 teacher
- 蒸馏和边缘部署排在后面

## 当前环境

- 主数据集：`CSIRO biomass`
- 当前服务器：`4xL40`
- 当前主要运行方式：
  - `1 GPU = 1 个独立实验`
  - 不优先用 `4 卡 DDP` 跑单个实验

这样做的原因是：

- 当前数据量不大
- `fold x seed` sweep 更适合并行实验池
- 比起把 4 张卡绑给一个实验，更值得并行比较不同 backbone

## 已完成实验

### 汇总表

| 实验 | 配置 | 状态 | OOF Weighted R2 | 结论 |
| --- | --- | --- | ---: | --- |
| `dinov2-vitl-518` | `configs/server/supervised-dinov2-vitl-518.yaml` | 已完成 | `0.4347` | 当前强基线，保留 |
| `siglip-so400m-384` | `configs/server/supervised-siglip-so400m-384.yaml` | 已完成 | `0.0695` | 明显偏弱，暂时降级 |

### `dinov2-vitl-518`

实验目录：

- `artifacts/server/dinov2-vitl-518`

总分：

- `oof_weighted_r2 = 0.43469774247886117`

按 target 指标：

| target | corr | mae | rmse | pred_std | true_std | weighted_component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Dry_Green_g` | `0.6773` | `12.3282` | `19.0579` | `13.6147` | `25.4012` | `0.4355` |
| `Dry_Dead_g` | `0.1025` | `8.8960` | `13.1249` | `3.2610` | `12.4020` | `-0.1231` |
| `Dry_Clover_g` | `-0.2626` | `6.7940` | `13.4142` | `1.0946` | `12.1178` | `-0.2289` |
| `GDM_g` | `0.7913` | `10.6788` | `15.7002` | `16.3076` | `24.9358` | `0.6025` |
| `Dry_Total_g` | `0.7917` | `11.8211` | `17.4133` | `20.3011` | `27.9840` | `0.6117` |

当前判断：

- 明显有效，是当前最强已完成实验
- `Dry_Green_g / GDM_g / Dry_Total_g` 已经很强
- `Dry_Dead_g` 仍然偏弱
- `Dry_Clover_g` 仍然是最差 target
- 没有常数塌缩问题，`pred_std` 正常

### `siglip-so400m-384`

实验目录：

- `artifacts/server/siglip-so400m-384`

总分：

- `oof_weighted_r2 = 0.0694691849065068`

按 target 指标：

| target | corr | mae | rmse | pred_std | true_std | weighted_component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Dry_Green_g` | `0.3680` | `16.1807` | `24.6120` | `4.5718` | `25.4012` | `0.0585` |
| `Dry_Dead_g` | `0.0221` | `8.7582` | `13.1463` | `1.6593` | `12.4020` | `-0.1268` |
| `Dry_Clover_g` | `-0.2359` | `6.7316` | `13.3451` | `0.7602` | `12.1178` | `-0.2162` |
| `GDM_g` | `0.4695` | `16.0826` | `23.5580` | `5.4320` | `24.9358` | `0.1050` |
| `Dry_Total_g` | `0.4608` | `17.7998` | `25.7054` | `8.0679` | `27.9840` | `0.1539` |

当前判断：

- 明显弱于 `dinov2-vitl-518`
- 没有完全塌缩，但输出动态范围偏小
- 当前更适合作为异构补充候选，不适合作为主力

## 已知结论

截至目前，可以先定下这些判断：

1. `dinov2-vitl-518` 是当前最强已完成实验。
2. `siglip-so400m-384` 明显偏弱，不应追加过多预算。
3. 当前最值得继续看的待跑配置是：
   - `dinov3-vitl-896`
   - `siglip-so400m-448`
4. 如果 `dinov3-vitl-896` 超过 `0.4347`，它将成为新的主力候选。
5. 如果 `siglip-so400m-448` 仍然接近 `384` 结果，`siglip` 这一家更多只剩下异构 ensemble 价值。

## 待跑实验

### 高优先级

| 优先级 | 实验 | 配置 | 目的 |
| --- | --- | --- | --- |
| P0 | `dinov3-vitl-896` | `configs/server/supervised-dinov3-vitl-896.yaml` | 争夺新的主力 single model |
| P0 | `siglip-so400m-448` | `configs/server/supervised-siglip-so400m-448.yaml` | 验证 `siglip` 是否受限于输入尺寸 |

### 中优先级

| 优先级 | 实验 | 配置 | 目的 |
| --- | --- | --- | --- |
| P1 | `dinov2-vitg-518` | `configs/server/supervised-dinov2-vitg-518.yaml` | 测试更大 `DINOv2-G` 的上限 |

### 低优先级

| 优先级 | 实验 | 配置 | 目的 |
| --- | --- | --- | --- |
| P2 | `dinov3-vitl-1024` | `configs/server/supervised-dinov3-vitl-1024.yaml` | 更接近冠军题解主力尺寸，但在 `L40` 上风险较高 |

## 4xL40 建议启动方式

### 先决条件

如果数据未准备：

```bash
uv sync --dev
uv run csiro-biomass prepare-data --zip-path csiro-biomass.zip --extract-images
```

### 当前推荐并行启动

如果 `GPU0` 和 `GPU1` 已经完成，那么下一步建议：

```bash
CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com uv run csiro-biomass train-supervised --config configs/server/supervised-siglip-so400m-448.yaml > logs.siglip-448.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 HF_ENDPOINT=https://hf-mirror.com uv run csiro-biomass train-supervised --config configs/server/supervised-dinov3-vitl-896.yaml > logs.dinov3-vitl-896.txt 2>&1 &
```

如果前两张卡空出来，也可以继续开：

```bash
CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com uv run csiro-biomass train-supervised --config configs/server/supervised-dinov2-vitg-518.yaml > logs.dinov2-vitg-518.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 HF_ENDPOINT=https://hf-mirror.com uv run csiro-biomass train-supervised --config configs/server/supervised-dinov3-vitl-1024.yaml > logs.dinov3-vitl-1024.txt 2>&1 &
```

但要注意：

- `dinov2-vitg-518` 风险中等
- `dinov3-vitl-1024` 在 `L40 48GB` 上风险最高

### 更稳妥的启动顺序

如果不想四卡全压高风险配置，建议这样排：

1. `dinov3-vitl-896`
2. `siglip-so400m-448`
3. `dinov2-vitg-518`
4. 最后再试 `dinov3-vitl-1024`

## 训练监控命令

实时看显存：

```bash
watch -n 1 nvidia-smi
```

看某个实验日志尾部：

```bash
tail -f logs.dinov3-vitl-896.txt
tail -f logs.siglip-448.txt
```

## 跑完后要执行的命令

### 聚合 OOF

```bash
uv run csiro-biomass oof aggregate \
  --experiment-root artifacts/server/dinov3-vitl-896 \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet

uv run csiro-biomass oof aggregate \
  --experiment-root artifacts/server/siglip-so400m-448 \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet

uv run csiro-biomass oof aggregate \
  --experiment-root artifacts/server/dinov2-vitg-518 \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet

uv run csiro-biomass oof aggregate \
  --experiment-root artifacts/server/dinov3-vitl-1024 \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet
```

### 汇总筛选 teacher

```bash
uv run csiro-biomass oof select \
  --experiment-root artifacts/server/dinov2-vitl-518 \
  --experiment-root artifacts/server/siglip-so400m-384 \
  --experiment-root artifacts/server/siglip-so400m-448 \
  --experiment-root artifacts/server/dinov3-vitl-896 \
  --experiment-root artifacts/server/dinov2-vitg-518 \
  --experiment-root artifacts/server/dinov3-vitl-1024 \
  --output-dir artifacts/server/teacher-selection \
  --top-k 4 \
  --correlation-threshold 0.985
```

## 建议的下一步决策规则

跑完剩余实验后，按下面规则判断：

1. 谁的 `oof_weighted_r2` 最高。
2. 看 `Dry_Green_g / GDM_g / Dry_Total_g` 三个主目标是否明显更强。
3. 看是否有 target 出现近常数预测。
4. 看不同 backbone family 是否能提供异构性，而不只是重复。

如果后续结果满足：

- `dinov3-vitl-896 > dinov2-vitl-518`

那么新的主力单模型优先考虑 `dinov3-vitl-896`。  
否则当前主力仍是 `dinov2-vitl-518`。

## 追加记录模板

后续每完成一个实验，就按下面格式追加：

```md
### <experiment_name>

- config:
- status:
- oof_weighted_r2:
- conclusion:

| target | corr | mae | rmse | pred_std | true_std | weighted_component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dry_Green_g |  |  |  |  |  |  |
| Dry_Dead_g |  |  |  |  |  |  |
| Dry_Clover_g |  |  |  |  |  |  |
| GDM_g |  |  |  |  |  |  |
| Dry_Total_g |  |  |  |  |  |  |
```
