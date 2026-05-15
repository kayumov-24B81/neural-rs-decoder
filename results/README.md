# Experiments

This file documents experiments conducted with the benchmark stand. Each entry references the underlying CSV and YAML files under `runs/`

## 01a — Baseline on Gilbert-Elliott moderate

Train on GE moderate, test on same channel. 1000 benchmark samples.

| | classic | oracle | neural |
|--|--|--|--|
| FER | 0.456 | 0.018 | **0.027** |

Model: MLP 4×512, 500 epochs, seed=42.  
Results: [`runs\20260515_134052_01a_no_earlystop_moderate_seed42.csv`](runs\20260515_134052_01a_no_earlystop_moderate_seed42.csv)
