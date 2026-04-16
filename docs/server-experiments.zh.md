# 服务器实验记录

## 当前目标

这份文档用于持续记录 `CSIRO biomass` 在服务器上的实验结果，并给出当前阶段建议要跑的训练清单。

当前目标仍然是：

- 先把 server-side single model 和 ensemble 做强
- 先筛强 teacher
- 蒸馏和边缘部署排在后面

## 当前环境

- 主数据集：`CSIRO biomass`
- 当前服务器：
  - `4xL40`
  - `H200x8`
- 当前主要运行方式：
  - `1 GPU = 1 个独立实验`
  - 不优先用 `4 卡 DDP` 跑单个实验

这样做的原因是：

- 当前数据量不大
- `fold x seed` sweep 更适合并行实验池
- 比起把 4 张卡绑给一个实验，更值得并行比较不同 backbone

## 最新主线 OOF 汇总

汇总 CSV：

- [server-experiments-summary.csv](./server-experiments-summary.csv)

这份汇总基于服务器上已经生成的 `oof_summary.json` 与 `oof_metrics.csv`，统一覆盖当前主线的 `9` 个实验。

### 总览表

| 排名 | 实验 | 平台 | OOF Weighted R2 | num_runs | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| `1` | `dinov3-vitl-896-timm` | `H200` | `0.4933` | `6` | 当前最强单模主线 |
| `2` | `dinov3-vitl-1024-timm` | `H200` | `0.4914` | `6` | 更高分辨率未超过 `896-timm` |
| `3` | `dinov3-vithplus-896-timm` | `H200` | `0.4873` | `6` | 更大模型未超过 `ViT-L 896-timm` |
| `4` | `dinov2-vitl-reg4-518` | `H200` | `0.4687` | `6` | `reg4` 有改进，但未改主线格局 |
| `5` | `dinov2-vitg-reg4-518` | `H200` | `0.4605` | `6` | 当前 `DINOv2-G` 家族最稳版本 |
| `6` | `dinov2-vitg-518` | `H200` | `0.4546` | `6` | H200 重炼后仍低于 `reg4` 与 `DINOv3-L` 主线 |
| `7` | `dinov2-vitl-518` | `L40` | `0.4347` | `6` | 旧强基线，已被新主线明显超过 |
| `8` | `siglip-so400m-384` | `L40` | `0.0695` | `6` | 明显偏弱，不适合作为主线 |
| `9` | `siglip-so400m-448` | `L40` | `0.0356` | `6` | 比 `384` 更差，进一步确认失败线 |

### 当前缺失 OOF 聚合结果

以下实验目录当前没有本地可用的 `oof_summary.json` / `oof_metrics.csv`，因此不纳入主汇总表：

- `dinov3-vitl-896`
- `dinov3-vitl-1024`

### 关键结论

1. `dinov3-vitl-896-timm = 0.4933` 在 H200 上仍然是当前最强单模。
2. 更高分辨率并未自动涨分：`dinov3-vitl-1024-timm = 0.4914 < 0.4933`。
3. 更大模型并未自动涨分：`dinov3-vithplus-896-timm = 0.4873 < 0.4933`。
4. `reg4` 版本有提升，但 `DINOv2-L = 0.4687`、`DINOv2-G = 0.4605` 仍未接近当前 `DINOv3-L 896-timm` 主线。
5. `siglip` 线已经可以视为主线失败：`384 = 0.0695`，`448 = 0.0356`。
6. 当前总分上限主要不是被 backbone 卡住，而是被少数短板 target 持续拖住。

### Per-target 重点观察

先看当前最值得比较的几条 `DINO` 主线：

| 实验 | 平台 | `Dry_Green_g` | `Dry_Dead_g` | `Dry_Clover_g` | `GDM_g` | `Dry_Total_g` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dinov3-vitl-896-timm` | `H200` | `0.5387` | `-0.1042` | `-0.2186` | `0.6572` | `0.6806` |
| `dinov3-vitl-1024-timm` | `H200` | `0.5314` | `-0.1010` | `-0.2214` | `0.6453` | `0.6828` |
| `dinov3-vithplus-896-timm` | `H200` | `0.5687` | `-0.1289` | `-0.2122` | `0.6671` | `0.6622` |
| `dinov2-vitl-reg4-518` | `H200` | `0.4866` | `-0.0992` | `-0.2258` | `0.6224` | `0.6560` |
| `dinov2-vitg-518` | `H200` | `0.4960` | `-0.1303` | `-0.2128` | `0.6379` | `0.6235` |
| `dinov2-vitl-518` | `L40` | `0.4355` | `-0.1231` | `-0.2289` | `0.6025` | `0.6117` |

其中表内数值为各 target 的 `weighted_component`。

固定可以得出这些观察：

- `dinov3-vitl-896-timm` 的领先主要来自：
  - `Dry_Total_g = 0.6806`
  - `GDM_g = 0.6572`
  - `Dry_Green_g = 0.5387`
- `dinov3-vitl-1024-timm` 没有超过 `896-timm`，原因不是单一 target 崩掉，而是三个主贡献目标都略弱：
  - `Dry_Green_g` 更低
  - `GDM_g` 更低
  - `Dry_Total_g` 更低
- `dinov3-vithplus-896-timm` 的 `GDM_g` 和 `Dry_Green_g` 很强，但 `Dry_Dead_g` 退化明显，抵消了更大模型的收益。
- `dinov2-vitl-reg4-518` 仍是当前 `DINOv2` 家族里最强的一条。
- `dinov2-vitg-518` 在 H200 重炼后只到 `0.4546`，说明更大 `DINOv2-G` 也没有改写当前格局。
- `DINOv2` 家族的主要增益仍来自 `Dry_Total_g / GDM_g`，没有真正修好短板 target。

### 短板 target 结论

当前最值得盯的仍然是：

- `Dry_Dead_g`
- `Dry_Clover_g`

原因很直接：

- `Dry_Dead_g` 在所有当前有竞争力的 `DINO` 模型里都是负贡献，最好也只到约 `-0.099` 量级。
- `Dry_Clover_g` 在所有当前有竞争力的 `DINO` 模型里也都是负贡献，没有任何一条主线把它拉回正值。
- `dinov3-vitl-896-timm` 已经把 `Dry_Total_g / GDM_g / Dry_Green_g` 推得很高，但总分仍停在 `0.4933`，说明真正还没修掉的就是这两个 target 的系统性拖分。

### `siglip` 为什么可以降级

`siglip` 两条线都表现出主目标贡献明显不足：

| 实验 | `Dry_Green_g` | `Dry_Dead_g` | `Dry_Clover_g` | `GDM_g` | `Dry_Total_g` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `siglip-so400m-384` | `0.0585` | `-0.1268` | `-0.2162` | `0.1050` | `0.1539` |
| `siglip-so400m-448` | `0.0287` | `-0.0933` | `-0.2082` | `0.0396` | `0.1099` |

这说明：

- 问题不是“输入尺寸不够大”，因为 `448` 比 `384` 更差。
- 问题也不是“只差一点异构补充”，因为它在主目标上的贡献量级本身就太低。
- 因此当前 `siglip` 更适合作为失败线参考，而不是继续投入主线预算。

## 当前下一步执行清单

当前阶段不再优先扩 backbone 搜索，而是先基于 H200 本地已经齐备的 `6` 组结果做 `teacher selection -> ensemble -> postprocess` 闭环。

### 候选池

当前 H200 本地分析组固定为这 `6` 个实验：

- `dinov3-vitl-896-timm`
- `dinov3-vitl-1024-timm`
- `dinov3-vithplus-896-timm`
- `dinov2-vitl-reg4-518`
- `dinov2-vitg-reg4-518`
- `dinov2-vitg-518`

这组候选覆盖了：

- 当前 H200 本地最强 `DINOv3`
- 同家族更大模型
- 同家族更高分辨率
- `DINOv2-L / G` 两条主补充线
- `reg4` 与非 `reg4` 的差异

`siglip` 不再进入当前主 teacher pool。

### Step 1：先做 H200 本地 teacher selection

```bash
uv run csiro-biomass oof select \
  --experiment-root artifacts/server/dinov3-vitl-896-timm \
  --experiment-root artifacts/server/dinov3-vitl-1024-timm \
  --experiment-root artifacts/server/dinov3-vithplus-896-timm \
  --experiment-root artifacts/server/dinov2-vitl-reg4-518 \
  --experiment-root artifacts/server/dinov2-vitg-reg4-518 \
  --experiment-root artifacts/server/dinov2-vitg-518 \
  --output-dir artifacts/server/teacher-selection-h200 \
  --top-k 4 \
  --correlation-threshold 0.985
```

产物：

- `artifacts/server/teacher-selection-h200/teacher_selection.csv`
- `artifacts/server/teacher-selection-h200/teacher_correlations.csv`

### Step 2：做 H200 本地 OOF ensemble + postprocess 分析

新增脚本：

- `scripts/analyze_oof_ensemble.py`

推荐命令：

```bash
uv run python scripts/analyze_oof_ensemble.py \
  --train-manifest data/processed/csiro-biomass/metadata/train_wide.parquet \
  --experiment-root artifacts/server/dinov3-vitl-896-timm \
  --experiment-root artifacts/server/dinov3-vitl-1024-timm \
  --experiment-root artifacts/server/dinov3-vithplus-896-timm \
  --experiment-root artifacts/server/dinov2-vitl-reg4-518 \
  --experiment-root artifacts/server/dinov2-vitg-reg4-518 \
  --experiment-root artifacts/server/dinov2-vitg-518 \
  --output-dir artifacts/server/ensemble-analysis-h200 \
  --min-combination-size 1 \
  --max-combination-size 4 \
  --top-n 20
```

### Step 3：追踪分析结果文件

当前实际需要追踪和查看的是：

- `artifacts/server/ensemble-analysis-h200/combination_scores.csv`
- `artifacts/server/ensemble-analysis-h200/best_by_size.csv`
- `artifacts/server/teacher-selection-h200/teacher_selection.csv`

查看命令：

```bash
sed -n '1,30p' artifacts/server/ensemble-analysis-h200/combination_scores.csv
sed -n '1,40p' artifacts/server/ensemble-analysis-h200/best_by_size.csv
sed -n '1,20p' artifacts/server/teacher-selection-h200/teacher_selection.csv
```

### Step 4：按分析结果锁定 H200 本地 `4` 个 teacher

锁定 teacher 时，不只看单模总分，还要同时看：

1. `teacher_selection.csv` 里的多样性筛选结果
2. `combination_scores.csv` 里的组合总分
3. `Dry_Dead_g / Dry_Clover_g` 在 postprocess 前后的改善幅度

优先保留：

- `dinov3-vitl-1024-timm`
- `dinov3-vithplus-896-timm`
- 至少 `1` 个 `DINOv2` 异构成员

## OOF ensemble 分析产物

运行 `scripts/analyze_oof_ensemble.py` 后，固定检查这三个文件：

- `artifacts/server/ensemble-analysis-h200/combination_scores.csv`
- `artifacts/server/ensemble-analysis-h200/combination_metrics.csv`
- `artifacts/server/ensemble-analysis-h200/best_by_size.csv`

解释口径：

- `combination_scores.csv`：每个组合的总分、`post_delta`、以及各 target 的 `weighted_component`
- `combination_metrics.csv`：每个组合在 `raw` / `postprocess` 两个模式下的 per-target `corr / mae / rmse / weighted_component`
- `best_by_size.csv`：`1/2/3/4` 模型组合各自最优的 `raw` 和 `postprocess` 摘要

## 建议的下一步决策规则

进入 pseudo / online 之前，按下面规则判断：

1. H200 本地最优 ensemble 是否稳定高于当前 H200 本地最强单模 `dinov3-vitl-896-timm = 0.4933`
2. `postprocess` 是否带来稳定正增益，而不是只在个别组合上偶然加分
3. `Dry_Dead_g / Dry_Clover_g` 是否明显改善

如果满足以下任一条件，就进入 pseudo / online：

- 最优 ensemble 相比 `0.4933` 有清晰正增益
- 总分提升有限，但 `Dry_Dead_g / Dry_Clover_g` 的负贡献明显收敛

如果 ensemble 和 postprocess 都没有明显收益，就先停在分析阶段，不继续扩 pseudo 链路。

## 历史记录

以下旧计划保留为阶段性记录，不再代表当前最高优先级：

- 继续扩大 backbone 搜索
- 继续把 `siglip` 当主线候选
- 按单模 sweep 继续扩 L40/H200 训练矩阵

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
