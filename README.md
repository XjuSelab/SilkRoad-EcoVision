# SilkRoad-EcoVision

CSIRO `Image2Biomass` 题解复现工程，按本地开发、`uv` 管理依赖、GitHub 作为唯一代码中枢的方式组织。

## Environment

本地开发：

```bash
uv sync --dev
```

如果需要 Hugging Face 镜像拉权重，推荐先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

服务器运行：

```bash
uv sync --no-dev
```

统一入口：

```bash
uv run csiro-biomass --help
```

## Workflow

1. 把比赛原始压缩包放到 `data/raw/csiro-biomass.zip`，或在仓库根目录保留 `csiro-biomass.zip`。
2. 生成宽表、fold 和 submission 模板：

```bash
uv run csiro-biomass prepare-data --zip-path csiro-biomass.zip --extract-images
```

3. 监督训练：

```bash
uv run csiro-biomass train-supervised --config configs/supervised-vitl.yaml
```

4. 伪标签在线训练：

```bash
uv run csiro-biomass train-pseudo --config configs/pseudo-online.yaml
```

5. 最终推理与提交文件导出：

```bash
uv run csiro-biomass infer --config configs/infer-ensemble.yaml
```

## Repository Rules

- 依赖由 `uv` 管理，不提交 `uv.lock`。
- 服务器通过 `git pull` 从 GitHub 同步代码。
- `data/`、`artifacts/`、`outputs/` 等生成物不纳入 Git。
- Hugging Face 数据同步预留为后续私有仓库流程，不是首轮关键路径。

## Layout

```text
src/csiro_biomass/    Python package
configs/              YAML experiment configs
docs/                 Reproduction notes
scripts/              Thin wrapper scripts
tests/                Lightweight smoke tests
```
