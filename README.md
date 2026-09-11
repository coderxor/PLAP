# PLAP

两阶段日志解析与异常预测。

- 阶段 1：基于虚拟嵌入的日志解析（`log3p_parsing`）
- 阶段 2：滑动时间窗口 + 提示特征的异常预测（`log3p_prediction`）

代码在 `plap-main/` 下。改参数在 **`plap-main/config.py`**。

## 环境

见`requirements.txt`


## 数据与路径

仓库不包含日志和预训练模型。克隆后先改 `config.py` 里的数据集路径

**阶段 1 `Parsing`**

| 项 | 说明 |
|---|---|
| `CURRENT_DATASET` | `hdfs` / `bgl` / `android` / `mac` / `zookeeper` |
| `DATASETS[*]["file_path"]` | Drain 解析后的 `*_structured.csv` |
| `MODEL_OPTIONS` | 本地模型目录，或改成 Hugging Face 名称，如 `roberta-base` |
| `TOP_N` / `EPOCHS` / `BATCH_SIZE` | 解析超参 |

**阶段 2 `Prediction`**

| 项 | 说明 |
|---|---|
| `DATASET` | `HDFS` 或 `BGL` |
| `*_LOG_FILE_PATH` | Drain 结构化日志 |
| `HDFS_LABEL_FILE_PATH` | HDFS 异常标签 `anomaly_label.csv`（BGL 不需要） |
| `*_EMBEDDING_FILE_PATH` | 模板嵌入 CSV（可用 `generate_embedding_bert.py` 生成） |
| `WINDOW_DURATION` | 时间窗口，单位秒；论文默认 HDFS 8s、BGL 9s |
| `USE_SEMANTIC` / `USE_SEQUENTIAL` / `USE_TIME` / `USE_PARAM` | 特征开关 |

日志数据可从 [Loghub](https://github.com/logpai/loghub) 获取。HDFS 标签来自 Loghub 的 `anomaly_label.csv`。

## 阶段 1：日志解析

单数据集训练与评估：

```bash
python -m log_parsing.log3p_parsing.main
```

Top-N 消融实验（5 个数据集 × `TOP_N=2,4,…,52`）：

```bash
python log_parsing/log3p_parsing/other_file/run_experiment.py
```

脚本结束时会写出 CSV 并画图。结果默认写到当前目录下的 `experiments/`。

## 阶段 2：异常预测

先用 Drain 得到结构化日志，再用 BERT 生成模板嵌入（可选，若已有 CSV 可跳过）：

```bash
python log_anomaly_prediction/logprompt/log3p_prediction/generate_embedding_bert.py
```

单次训练与评估：

```bash
python -m log_anomaly_prediction.logprompt.log3p_prediction.main
python -m log_anomaly_prediction.logprompt.log3p_prediction.main --dataset BGL
```

窗口大小实验（默认 HDFS 与 BGL）：

```bash
python -m log_anomaly_prediction.logprompt.log3p_prediction.experiment_window_size
```

窗口实验结束时会写出 CSV 并画图。日志到达率图（需原始 `HDFS.log` / `BGL.log`）：

```bash
python log_anomaly_prediction/logprompt/log3p_prediction/plot_log_arrival_rate.py --hdfs <HDFS.log> --bgl <BGL.log>
```


