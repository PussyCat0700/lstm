#!/bin/bash

#SBATCH --job-name=gatherer
#SBATCH --partition=ai4earth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量
#SBATCH --output=./logs/gather.out
#SBATCH --error=./logs/gather.err

# 定义参数列表
months=("1m" "3m" "6m" "9m" "12m")
models=("CROSS_VIVIT")
plantsets=("china" "nmg")

# 遍历所有组合
for month in "${months[@]}"; do
  for model in "${models[@]}"; do
    for plantset in "${plantsets[@]}"; do
      echo "Executing: python gather_results.py $model $month $plantset --zip"
      python gather_results.py "$model" "$month" "$plantset" --zip
    done
  done
done

