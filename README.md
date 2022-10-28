# WaveBound

Official implementation of "WaveBound: Dynamic Error Bounds for Stable Time Series Forecasting" (NeurIPS 2022). [[arxiv]](https://arxiv.org/abs/2210.14303)

## Prerequisites

We tested our project in the following environment:

```bash
Anaconda
python 3.7.10
pytorch 1.11.0
numpy 1.20.1
torchvision 0.12.0
```

## Running WaveBound

You can download the datasets used in our experiments from the Autoformer repository (https://github.com/thuml/Autoformer).
The dataset files should be located in "./dataset/...".

Then, if you run the script below, checkpoints and validation/test results will be saved in the results directory.

```bash
bash ./scripts/ETTm2_Autoformer+EMA_M_96.sh
```

In the default setting, the dataset files and results directory are expected to be located as follows:

```bash
┌── dataset
│   ├── electricity
│   ├── ETT-small
│   ├── exchange_rate
│   ├── illness
│   ├── traffic
│   └── weather
└── save
    ├── checkpoints
    └── results
        ├── test_metrics
        └── valid_metrics
```