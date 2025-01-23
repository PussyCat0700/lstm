#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=gatherer
#SBATCH --partition=ADA6000,RTX4090,RTX3090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量
#SBATCH --output=./logs/gather.out
#SBATCH --error=./logs/gather.err

# 定义参数列表
months=("1m" "3m" "6m" "9m" "12m")
models=("XGBOOST_P" "GradientBoost" "Greek" "CNN_LSTM" "FFNN")
plantsets=("china" "nmg" "china_add")

# 遍历所有组合
for month in "${months[@]}"; do
  for model in "${models[@]}"; do
    for plantset in "${plantsets[@]}"; do
      echo "Executing: python gather_results.py $model $month $plantset --zip"
      python gather_results.py "$model" "$month" "$plantset" --zip
    done
  done
done