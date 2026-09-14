# PLAP

Two-stage log parsing and anomaly prediction.

- Stage 1: virtual-embedding-based log parsing (`log3p_parsing`)
- Stage 2: sliding time-window anomaly prediction with prompt features (`log3p_prediction`)

The code lives under `plap-main/`. Edit hyperparameters in **`plap-main/config.py`**.

## Environment

See `requirements.txt`.

## Data and paths

This repository does not include logs or pretrained models. After cloning, update dataset paths in `config.py`.

**Stage 1: Parsing**

| Item | Description |
|---|---|
| `CURRENT_DATASET` | `hdfs` / `bgl` / `android` / `mac` / `zookeeper` |
| `DATASETS[*]["file_path"]` | Drain-parsed `*_structured.csv` |
| `MODEL_OPTIONS` | Local model directory, or a Hugging Face name such as `roberta-base` |
| `TOP_N` / `EPOCHS` / `BATCH_SIZE` | Parsing hyperparameters |

**Stage 2: Prediction**

| Item | Description |
|---|---|
| `DATASET` | `HDFS` or `BGL` |
| `*_LOG_FILE_PATH` | Drain-structured logs |
| `HDFS_LABEL_FILE_PATH` | HDFS anomaly labels `anomaly_label.csv` (not required for BGL) |
| `*_EMBEDDING_FILE_PATH` | Template embedding CSV (can be generated with `generate_embedding_bert.py`) |
| `WINDOW_DURATION` | Time window in seconds; paper defaults are HDFS 8s and BGL 9s |
| `USE_SEMANTIC` / `USE_SEQUENTIAL` / `USE_TIME` / `USE_PARAM` | Feature switches |

Logs are available from [Loghub](https://github.com/logpai/loghub). HDFS labels come from Loghub's `anomaly_label.csv`.

## Stage 1: Log parsing

Train and evaluate on a single dataset:

```bash
python -m log_parsing.log3p_parsing.main
```

Top-N ablation (5 datasets × `TOP_N=2,4,…,52`):

```bash
python log_parsing/log3p_parsing/other_file/run_experiment.py
```

The script writes a CSV and plots at the end. Results go to `experiments/` in the current directory by default.

## Stage 2: Anomaly prediction

First obtain Drain-structured logs, then generate template embeddings with BERT (optional if the CSV already exists):

```bash
python log_anomaly_prediction/logprompt/log3p_prediction/generate_embedding_bert.py
```

Train and evaluate once:

```bash
python -m log_anomaly_prediction.logprompt.log3p_prediction.main
python -m log_anomaly_prediction.logprompt.log3p_prediction.main --dataset BGL
```

Window-size experiment (HDFS and BGL by default):

```bash
python -m log_anomaly_prediction.logprompt.log3p_prediction.experiment_window_size
```

The window experiment writes a CSV and plots. Log arrival-rate plot (requires raw `HDFS.log` / `BGL.log`):

```bash
python log_anomaly_prediction/logprompt/log3p_prediction/plot_log_arrival_rate.py --hdfs <HDFS.log> --bgl <BGL.log>
```
