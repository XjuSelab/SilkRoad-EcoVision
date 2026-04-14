# ViT 蒸馏到 Hailo-8 学生模型报告

## 范围

这份文档总结当前仓库上下文，并提出一条面向边缘部署的路线：

- 在服务器上训练强 `ViT` teacher
- 蒸馏一个适合 `Hailo-8` 的单帧学生模型
- 在 `Raspberry Pi 5` 上利用视频流多帧融合来逼近 teacher 的效果

目标不是把当前双流 `DINOv3-L` 直接搬到 `Hailo-8` 上，而是在满足边缘设备约束的前提下，尽量保留 teacher 的预测能力。

## 当前上下文

当前仓库已经具备围绕 CSIRO biomass 数据集的主线训练能力：

- 将原始宽图切成左右双流输入
- 使用 `5` 个回归头和 `5` 个区间分类头进行监督训练
- 支持多 `fold`、多 `seed` 的服务器 sweep
- 支持 `OOF` 聚合和模型筛选
- 支持推理阶段的 `TTA` 和 checkpoint averaging

相关文档：

- [reproduction.md](/home/winbeau/Selab-gh/SilkRoad-EcoVision/docs/reproduction.md)
- [server-training.md](/home/winbeau/Selab-gh/SilkRoad-EcoVision/docs/server-training.md)

当前训练目标仍然应该是：

- 先把 CSIRO 数据集上的效果做到尽量强
- 蒸馏作为后续部署路线，而不是当前冲分主线

这个判断没有变。学生模型应该建立在已经稳定的强 teacher 之上。

## 为什么不该直接部署当前模型

当前仓库里的强候选 backbone 主要是：

- `DINOv3 ViT-L`
- `DINOv2 ViT-L / ViT-G`
- `SigLIP SO400M`

它们并不适合直接部署到 `Raspberry Pi 5 + Hailo-8`，原因是：

- `Hailo-8` 不是通用 GPU
- 部署受 `Hailo Dataflow Compiler` 支持的图结构和算子限制
- 大型 transformer 风格模型更难稳定编译并高效运行
- 当前模型还是双流结构，边缘推理成本更高

实际含义就是：

- 不要试图把当前 teacher 架构原样塞进 `Hailo-8`
- 应该把 teacher 的行为蒸馏到一个更小、更友好的 student

## 推荐的边缘架构

建议采用分层架构：

1. `Hailo-8` 负责单帧轻量学生模型推理。
2. `Raspberry Pi 5` 维护视频流上的短时窗口。
3. `Pi` 端融合多帧预测，输出最终 biomass 结果。

这比把重型时序模型或 transformer 直接编译到 `Hailo-8` 上更符合硬件特点。

### 推荐的运行图

```text
video frame
  -> preprocess
  -> Hailo-8 student
       -> 5 regression outputs
       -> 5 interval-classification outputs
       -> optional low-dim embedding
  -> Pi 5 temporal fusion over last K frames
       -> final biomass prediction
       -> optional confidence / stability score
```

## 为什么多帧边缘推理通常会优于单帧

前提是相邻帧里确实有新信息，而不是纯重复帧。

满足这个前提时，多帧融合通常更强，主要有四个原因：

- 可以降低单帧中的随机误差，例如模糊、压缩噪声、曝光波动、局部遮挡
- 可以逐步积累轻微变化的视角信息
- 可以利用时间连续性约束结果，避免预测在相邻帧之间大幅跳动
- 可以用代价很低的时序融合头，补回一部分学生模型在压缩 teacher 时损失的信息

对 biomass 任务来说，这一点尤其合理，因为单帧里植物纹理、覆盖和密度信息经常不够稳定，而短时间窗口通常会更稳。

## ViT teacher 能否蒸馏成适配 Hailo-8 的 student

可以，但要问对问题。

不是“student 能否复制 teacher 的结构”，答案是否定的。  
真正的问题是“student 能否复制 teacher 足够多的行为”，这件事是可行的。

跨架构蒸馏是标准做法：

- `ViT` 做 teacher
- 轻量 `CNN` 或轻量 `hybrid` 做 student

student 不需要逐层复刻 teacher，而是学习：

- 最终输出
- logits 结构
- 投影后的特征关系
- 排序和置信度行为

合理预期是：

- 设计得当的 student 可以明显逼近一个强 single teacher
- 但通常仍然会落后于最强 ensemble
- 如果目标是边缘实时和低功耗，这个折中是合理的

## 学生模型候选

学生模型应该优先按“能部署”选，而不是先按“学术上最强”选。

最现实的候选有：

- `MobileNetV3`
- `EfficientNet-Lite`
- `EdgeNeXt`
- `EfficientFormer`
- 小型 `ConvNeXt`，前提是部署图可以接受

筛选标准：

- ONNX 导出稳定
- 与 Hailo 支持算子兼容性好
- 在 `224-320` 输入分辨率下延迟低
- 对回归任务有足够容量，而不仅仅适合粗分类

## 输入设计选项

学生模型有三种现实可行的输入方案。

### 方案 A：单帧拼接图

- 将原始左右信息保留在一张合成图里
- 部署路径最简单
- 最容易导出和编译

代价：

- 学生模型必须自己在内部学会左右信息交互

### 方案 B：轻量双分支

- 保留两个小分支，后期再做融合
- 对当前 teacher 更忠实

代价：

- 部署更复杂
- 对编译器可能没那么友好

### 方案 C：单帧 student 加视频窗口

- 每次只推理一个紧凑单帧表示
- 靠时序融合补偿去掉显式双流后丢失的信息

代价：

- 系统最简洁
- 更多负担落在 Pi 的时序融合层上

v1 建议：

- 先做 `方案 C`
- 用单帧 student，加短窗口视频融合来补精度

## 蒸馏目标

student 不应该只学习 hard label。

推荐从最优 teacher 或小 ensemble 中蒸馏这些目标：

- `5` 个回归输出
- `5` 个区间分类 logits 或概率
- 可选的中间特征投影向量
- 可选的不确定性或置信度

推荐的损失组成：

- hard supervised regression loss
- hard supervised interval-classification loss
- teacher regression distillation loss
- teacher logit distillation loss
- 可选的 feature distillation loss
- 面向视频窗口训练的 temporal consistency loss

一个示意性的形式：

```text
L_total =
  0.4 * hard_regression +
  0.2 * hard_interval_cls +
  0.2 * teacher_regression +
  0.1 * teacher_logits +
  0.1 * temporal_consistency
```

实际权重应通过实验调优。

## 训练策略

### Stage 1：先训练并锁定最强 teacher

在服务器上的 teacher 没稳定之前，不要开始 edge 线。

输入：

- 当前服务器 sweep 里最优的 single model
- 可选地再加最优小 ensemble 用于离线生成更稳的软目标

输出：

- teacher checkpoints
- 每个样本的 teacher 预测
- 可选的中间 embedding

### Stage 2：生成 student 的训练目标

对每张训练图或视频帧：

- 保存 teacher 的回归输出
- 保存 teacher 的区间分类 logits 或概率
- 可选地保存特征投影

如果已经有视频数据，则对短窗口采样并保存：

- 每帧 teacher 输出
- 窗口级融合后的 teacher target

### Stage 3：训练单帧 student

先不加时序模块，只训练一个可部署的单帧 student。

目标：

- 架构稳定、可导出
- 单帧效果可接受
- 为后续时序融合提供较好校准

### Stage 4：训练或校准 Pi 端时序融合模块

在 student 跑完短窗口后，对最近若干帧结果做融合，可选方式包括：

- EMA
- 基于置信度的加权平均
- 小型 `GRU`
- 小型 `1D TCN`
- 对最近结果堆叠后跑小 `MLP`

v1 建议：

- 先做基于置信度的 EMA
- 如果简单规则已经到瓶颈，再试很小的 `GRU`

## Raspberry Pi 5 上的时序融合设计

时序融合模块建议运行在 Pi 的 CPU 上，而不是强行放进 Hailo。

原因：

- 这部分本身很小，用 CPU 足够
- 它需要状态、窗口和控制逻辑
- 比重新编译设备图更容易迭代

每帧建议送入的内容：

- `5` 个回归值
- `5 x bins` 个区间概率
- 可选置信度
- 可选 `64` 或 `128` 维 student embedding

窗口大小建议：

- 从 `K = 8` 开始
- 再比较 `K = 4, 8, 16`

帧筛选策略建议：

- 不要无脑融合所有原始帧
- 只保留有足够视觉新信息的帧
- 对近重复帧做跳过，避免同一视角被过度加权

## 评估计划

需要看三层评估。

### 第一层：离线精度

对比：

- 最优 teacher single model
- 最优 teacher ensemble
- student 单帧
- student 加时序融合

指标：

- weighted `R2`
- 各 target 的相关系数
- `MAE`
- `RMSE`

### 第二层：边缘部署可行性

测：

- ONNX 导出是否成功
- Hailo 编译是否成功
- 设备上的显存和吞吐
- `Pi 5 + Hailo-8` 上的端到端延迟

### 第三层：视频流稳定性

测：

- 帧间结果波动
- 对模糊和曝光变化的鲁棒性
- 对重复帧的敏感性
- 视角变化很小时的退化程度

## 预期精度折中

合理预期应该是：

- 最优 teacher ensemble 仍然最强
- 最优 single teacher 是 edge imitation 的现实上限
- 单帧 student 会落后于 single teacher
- student 加短窗口时序融合可以回补一部分差距

推论：

- 如果视频流里确实有视角和信息变化，时序系统通常会明显优于同一个 student 的单帧推理
- 如果视频几乎静止，收益就会小得多

## 风险

- 对 ImageNet 好用的轻量模型，不一定对 biomass 回归也好
- `Hailo` 编译支持可能会淘汰一些看起来不错的学生架构
- 视频帧冗余太高时，时序融合价值会下降
- 激进量化可能会伤害 `Dry_Clover_g` 这类小值目标
- student 可能学到 teacher 的偏差，而不仅仅是 teacher 的优点

## 推荐的项目顺序

1. 先完成并锁定最优的 CSIRO teacher 流程。
2. 选择 `2-3` 个导出友好的 student 架构。
3. 用 hard labels 和 teacher distillation 训练单帧 student。
4. 尽早验证 ONNX 导出和 Hailo 编译。
5. 在最优 student 上增加 Pi 端时序融合。
6. 对比：
   - 单帧 student
   - student 加时序融合
   - single teacher
7. 再决定 edge 线是否值得进入产品化阶段。

## 实际建议

对这个仓库来说，最合适的第一版 edge 原型是：

- teacher：当前服务器上最优的 single model 或小 ensemble
- student：`MobileNetV3` 或 `EdgeNeXt-S`
- 输入：单帧紧凑表示，不再照搬当前双流 teacher 图结构
- 运行时：`Hailo-8` 做逐帧推理，`Pi 5` 做时序融合
- 时序融合：先做置信度加权 EMA，再考虑极小 `GRU`

这是当前最现实的一条路径：既能继承现有强 `ViT` teacher 栈的能力，又真正有机会落到可部署的边缘系统。

## 参考资料

- Hailo Model Zoo: https://github.com/hailo-ai/hailo_model_zoo
- Hailo Applications: https://github.com/hailo-ai/hailo-apps
- Hailo Community 关于设备和编译约束的说明: https://community.hailo.ai/t/vision-acceleration-on-rpi-hailo-8l/12402/2
- Hailo Community 关于 transformer 支持的讨论: https://community.hailo.ai/t/support-for-vision-transformers/14715
- Online Model Distillation for Efficient Video Inference: https://www.ri.cmu.edu/publications/online-model-distillation-for-efficient-video-inference/
- Cross-modality online distillation for multi-view action recognition: https://doi.org/10.1016/j.neucom.2021.05.077
- Cross-Architecture Knowledge Distillation: https://openaccess.thecvf.com/content/ACCV2022/papers/Liu_Cross-Architecture_Knowledge_Distillation_ACCV_2022_paper.pdf
- Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation: https://arxiv.org/abs/2107.01378
- MiniViT: Compressing Vision Transformers with Weight Multiplexing: https://doi.org/10.1109/CVPR52688.2022.01183
- EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications: https://doi.org/10.1007/978-3-031-25082-8_1
- Knowledge Distillation: A Survey: https://doi.org/10.1007/s11263-021-01453-z
- Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks: https://doi.org/10.1109/TPAMI.2021.3055564
