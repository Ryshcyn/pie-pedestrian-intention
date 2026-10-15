# PIE Pedestrian Intention & Action Prediction

基于 PIE 数据集的多行人时空图学习实验：使用共享的 GATv2 空间编码器，对比因果 Temporal GAT 与单向 GRU，并通过 2×2 消融实验分析 `look` 行为特征对过街意图和当前动作预测的影响。

## 项目流程

```mermaid
flowchart LR
    subgraph DATA["① 数据输入与预处理"]
        direction TB
        A1["空间与行为 XML<br/>bbox · action · look"]
        A2["行人属性 XML<br/>intention · intersection · signal"]
        A3["自车 XML<br/>OBD speed"]
        B1["按帧对齐多源标注"]
        B2["按视频划分<br/>70% Train · 15% Val · 15% Test"]
        B3["15 帧观察窗口<br/>stride = 15 · full observation"]
        A1 --> B1
        A2 --> B1
        A3 --> B1
        B1 --> B2 --> B3
    end

    subgraph GRAPH["② 单帧多行人空间图"]
        direction TB
        P["Pedestrian<br/>[cx, cy, w, h, vx, vy]<br/>+ optional is_looking"]
        V["Vehicle<br/>[cx, cy, w, h, is_car]"]
        C["Crosswalk<br/>geometry + traffic-light state"]
        E["Ego vehicle<br/>normalized OBD speed"]
        G["Graph construction<br/>Ped KNN k=5 · Vehicle→Ped<br/>Crosswalk→Ped · Self-loop"]
        P --> G
        V --> G
        C --> G
        E -. global context .-> G
    end

    subgraph SPATIAL["③ 空间关系编码"]
        direction TB
        S1["类型专属线性投影<br/>统一至 128 维"]
        S2["EgoFiLM<br/>自车速度调制节点表示"]
        S3["2 × GATv2<br/>4 heads · residual · LayerNorm"]
        S4["每帧行人表示<br/>P × 128"]
        S1 --> S2 --> S3 --> S4
    end

    subgraph TEMPORAL["④ 时间建模：二选一"]
        direction TB
        T0{"Temporal encoder"}
        T1["2 × Causal Temporal GAT<br/>历史帧 → 当前帧"]
        T2["2-layer Unidirectional GRU<br/>按时间顺序编码"]
        T3["15 帧行人时序表示<br/>取最后时间步"]
        T0 --> T1 --> T3
        T0 --> T2 --> T3
    end

    subgraph OUTPUT["⑤ 双任务预测与消融评估"]
        direction TB
        O1["Intention head<br/>时序表示 + 静态场景上下文"]
        O2["Action head<br/>时序表示"]
        O3["BCEWithLogits<br/>class weight · label smoothing"]
        O4["Accuracy · Precision<br/>Recall · F1"]
        O5["2 × 2 Ablation<br/>Temporal GAT / GRU<br/>× Baseline / +look"]
        O1 --> O3
        O2 --> O3
        O3 --> O4 --> O5
    end

    B3 --> G
    G --> S1
    S4 --> T0
    T3 --> O1
    T3 --> O2
    A2 -. static context .-> O1

    classDef data fill:#E8F1FF,stroke:#4C78A8,color:#17324D,stroke-width:1.5px;
    classDef graph fill:#EAF7EF,stroke:#3A8D5D,color:#173D28,stroke-width:1.5px;
    classDef model fill:#F2ECFF,stroke:#7557B7,color:#32215B,stroke-width:1.5px;
    classDef output fill:#FFF3E3,stroke:#D9822B,color:#5A3210,stroke-width:1.5px;
    class A1,A2,A3,B1,B2,B3 data;
    class P,V,C,E,G graph;
    class S1,S2,S3,S4,T0,T1,T2,T3 model;
    class O1,O2,O3,O4,O5 output;
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
