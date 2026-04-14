# 丝路绿野 (SilkRoad-EcoVision) 🌿

> **基于 DINOv2 与边缘 AI 的高精度智慧牧场监测平台**
>
> 针对中国西北牧区（新疆）复杂环境下产草量监测难题，构建“算力中心预训练 + 边缘端实时感知”的端协同架构，实现厘米级生物量（Biomass）高精度反演。

---

## 📍 项目愿景 (Vision)
本项目旨在将前沿的视觉大模型技术（DINOv2）与生成式 AI（DiT）应用于智慧农业，解决牧场“草畜平衡”动态监测的痛点。通过在高性能计算中心（H200）训练出的知识，蒸馏部署于低功耗边缘设备（Raspberry Pi 5 + Hailo-8），实现低成本、高效率的数字化牧场管理。

---

## 🛠️ 技术架构 (Architecture)

| 模块 | 核心硬件 | 关键技术栈 |
| :--- | :--- | :--- |
| **中央算力站 (Compute Hub)** | **NVIDIA H200** | DINOv2-Giant, DiT (Data Engine), PyTorch |
| **边缘感知端 (Edge Node)** | **RPi 5 + Hailo-8** | ViT-Adapter, Quantization, Hailo SDK, ROS |
| **数字化大屏 (Frontend)** | - | Vue 3, Tailwind CSS, ECharts |
| **中控引擎 (Backend)** | Server | FastAPI, PostgreSQL, Redis |

---

## 🚀 项目路线图 (Project Roadmap)

### 🟢 Phase 1: 算法基准与数据构建 (2026 Q2)
- [ ] **DINOv2 特征分析：** 基于 CSIRO Biomass 数据集验证 DINOv2 底座在不同牧草密度下的特征敏感度。
- [ ] **Baseline 训练：** 在 H200 上完成基于 Vision Transformer (ViT) 的区间分类（Interval Classification）回归模型构建。
- [ ] **数据引擎：** 利用 **Diffusion Transformer (DiT)** 生成新疆特定牧草（如荒漠绿洲草场）的合成图像，缓解小样本难题。

### 🟡 Phase 2: 模型优化与边算协同 (2026 Q3)
- [ ] **模型压缩与蒸馏：** 以 H200 上的 DINO-Giant 为教师模型，通过知识蒸馏训练适配边缘端的轻量化感知模型。
- [ ] **Hailo-8 部署适配：** 完成模型量化（INT8/FP16）并编译为 `.hef` 固件，利用 26 TOPS 算力实现实时推断。
- [ ] **通信闭环：** 开发边缘端与算力站之间的异步数据回传协议，支持离线存储与择机同步。

### 🟠 Phase 3: 系统集成与实地验证 (2026 Q4)
- [ ] **全栈平台开发：** 完成 FastAPI 后端与 Vue 3 前端开发，集成历史产草量趋势分析。
- [ ] **野外终端原型：** 组装树莓派 5 移动监测站，集成 GPS 与 4G 通信模块。
- [ ] **新疆实地测试：** 赴阿勒泰或伊犁牧区进行实地数据采集与模型精度比对，获取第三方试用证明。

### 🔴 Phase 4: 成果转化与竞赛冲刺 (2027 Q1)
- [ ] **科研产出：** 整理实验数据，撰写高质量学术论文（目标 CV/AI 相关 Conference）。
- [ ] **大创结项：** 准备国家级大学生创新创业训练计划结项材料，制作演示视频。
- [ ] **开源建设：** 开放部分脱敏数据集与预训练权重。

---

## 💎 项目亮点 (Core Highlights)

1. **顶配算力支持：** 利用 **NVIDIA H200 (141GB HBM3e)** 进行超大规模视觉特征学习，确保模型具备极强的泛化性。
2. **边缘智能落地：** 针对 **Hailo-8** 的数据流计算架构深度优化，让高深的大模型算法能跑在低功耗硬件上。
3. **生成式增强：** 创新性引入 **DiT** 技术解决传统遥感数据不足的问题，构建闭环的“合成-训练-验证”流程。
4. **地域定制化：** 专门针对新疆干旱/半干旱牧草特性进行模型微调（Fine-tuning），具备极高的实效性。

---

## 🧑‍💻 核心团队与贡献 (Team)
- **负责人：** 担任架构设计与核心算法（H200 训练侧）
- **算法组：** 负责 DINOv2 微调与模型蒸馏
- **硬件组：** 负责树莓派、Hailo-8 部署与传感器集成
- **全栈组：** 负责可视化平台与 API 接口

---

## 📈 状态 (Status)
![Progress](https://img.shields.io/badge/Status-Initializing-blue)
![GPU](https://img.shields.io/badge/Computing-H200-green)
![Edge](https://img.shields.io/badge/Hardware-Hailo--8-orange)

---
*© 2026 丝路绿野项目组 - 专注于 AI 驱动的生态可持续性发展*
