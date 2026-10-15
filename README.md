# PIE Pedestrian Intention & Action Prediction

基于 PIE 数据集的多行人时空图学习实验：使用共享的 GATv2 空间编码器，对比因果 Temporal GAT 与单向 GRU，并通过 2×2 消融实验分析 `look` 行为特征对过街意图和当前动作预测的影响。

## 项目流程

```mermaid
flowchart TB
    A1["空间与行为标注<br/>bbox · action · look"]
    A2["行人属性<br/>intention · intersection · signal"]
    A3["自车状态<br/>OBD speed"]

    A1 --> B
    A2 --> B
    A3 --> B
    B["① 数据预处理<br/>多源标注按帧对齐<br/>视频级 70 / 15 / 15 划分"]
    B --> C["② 序列样本生成<br/>15 帧观察窗口 · stride 15<br/>仅保留全程可见的目标行人"]

    C --> D["③ 单帧多行人图构建<br/><br/>Ped: 位置 + 尺寸 + 速度 + optional look<br/>Vehicle: 几何信息 + is_car<br/>Crosswalk: 几何信息 + 信号灯状态<br/><br/>边: Ped KNN k=5 · Vehicle→Ped<br/>Crosswalk→Ped · Self-loop"]

    D --> E["④ 空间编码器<br/>类型专属投影 → 128 维<br/>EgoFiLM 速度调制<br/>2 × GATv2 · 4 heads · residual · LayerNorm"]

    E --> F{"⑤ 时间编码器"}
    F -->|方案 A| F1["Causal Temporal GAT<br/>2 层 · 历史帧→当前帧"]
    F -->|方案 B| F2["Unidirectional GRU<br/>2 层 · 顺序时序编码"]
    F1 --> G["行人时序表示<br/>取第 15 帧输出"]
    F2 --> G

    G --> H1["Intention Head<br/>时序表示 + 静态场景上下文"]
    G --> H2["Action Head<br/>时序表示"]
    A2 -. intersection / signal / lanes .-> H1

    H1 --> I["⑥ 联合训练与评估<br/>BCEWithLogits · class weight<br/>intention label smoothing = 0.05"]
    H2 --> I
    I --> J["2 × 2 消融实验<br/>Temporal GAT / GRU × Baseline / +look<br/>Accuracy · Precision · Recall · F1"]

    classDef inputNode fill:#E8F1FF,stroke:#4C78A8,color:#17324D,stroke-width:1.5px;
    classDef processNode fill:#EAF7EF,stroke:#3A8D5D,color:#173D28,stroke-width:1.5px;
    classDef modelNode fill:#F2ECFF,stroke:#7557B7,color:#32215B,stroke-width:1.5px;
    classDef outputNode fill:#FFF3E3,stroke:#D9822B,color:#5A3210,stroke-width:1.5px;
    class A1,A2,A3 inputNode;
    class B,C,D processNode;
    class E,F,F1,F2,G modelNode;
    class H1,H2,I,J outputNode;
```

## 结果摘要

| 模型 | look | 参数量 | 意图 F1 | 动作 F1 |
|---|---:|---:|---:|---:|
| Temporal GAT | 否 | 239,106 | 0.7934 | **0.7545** |
| Temporal GAT | 是 | 239,234 | **0.8287** | 0.7504 |
| GRU | 否 | 369,410 | 0.8112 | 0.7347 |
| GRU | 是 | 369,538 | 0.8182 | 0.7346 |

加入 `look` 后，Temporal GAT 的意图 F1 提升 0.0353，GRU 提升 0.0070；动作 F1 没有改善。详细设计、训练设置与讨论见 [研究摘要](docs/RESEARCH_REPORT.md)，完整指标见 [CSV](results/final_report_metrics.csv)。

![PIE look 标签分布](docs/look-label-distribution.png)

## 项目结构

```text
configs/                 四组最终消融实验配置
src/preprocess.py        PIE XML 解析、视频级划分与窗口生成
src/graph_builder.py     多行人空间图构建
src/dataset.py           15 帧序列数据集
src/models/              GRU 与 Temporal GAT 模型
src/train.py             统一训练入口
analysis/                数据集统计脚本
docs/                    匿名化研究说明与图表
results/                 最终报告指标
```

## 环境

- Python 3.10+
- PyTorch 2.x
- PyTorch Geometric
- scikit-learn、PyYAML

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

从 [PIE 官方仓库](https://github.com/aras62/PIE) 获取标注。项目不会重新分发数据；本地目录需包含：

```text
PIE_ROOT/
├── annotations/
├── annotations_attributes/
└── annotations_vehicle/
```

预处理：

```bash
python src/preprocess.py \
  --pie-root /path/to/PIE_ROOT \
  --output data/processed
```

该命令使用最终报告设置：15 帧窗口、步长 15、随机种子 42、按视频 70/15/15 划分。
在本项目使用的 53 个视频标注上，会得到 12,393 个训练窗口和 3,292 个验证窗口；训练配置从验证集固定抽取 30%（987 个窗口）。

## 训练四组实验

```bash
python src/train.py --config configs/temporal_gat_baseline.yaml
python src/train.py --config configs/temporal_gat_look.yaml
python src/train.py --config configs/gru_baseline.yaml
python src/train.py --config configs/gru_look.yaml
```

权重输出到 `outputs/`，该目录默认不提交。由于训练依赖随机初始化、硬件环境及上游数据版本，复现实验数值可能出现小幅波动。

## 数据分析

```bash
python analysis/analyze_pie_dataset.py --help
```

## 说明

- PIE 数据与训练权重不随仓库分发，请按上述步骤自行准备。
- 项目代码采用 MIT License；数据使用请遵守 [PIE 官方许可](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)。

## 引用

```bibtex
@inproceedings{Rasouli2019PIE,
  author    = {Amir Rasouli and Iuliia Kotseruba and Toni Kunic and John K. Tsotsos},
  title     = {PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```
