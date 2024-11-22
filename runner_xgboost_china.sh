#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=xgboost_china
#SBATCH --partition=RTX3090,RTX4090,A100,ADA6000
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --array=0-398:10%1       # 任务ID范围
#SBATCH --mem=30GB
#SBATCH --qos=ne_ablation
#SBATCH --output=./logs/station_logs/china/xgboost_%A_%a.out
#SBATCH --error=./logs/station_logs/china/xgboost_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=10
# 计算任务的显卡ID
gpu_id=$((task_id % 1))
echo $1

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    logdir=/data1/yfliu/logs/solar/china/xgboost/${1}_${task_id}_${plant_number}
    mkdir -p $logdir
    CUDA_VISIBLE_DEVICES=$gpu_id python train_xgboost.py $plant_number $1 "china" > "$logdir/log.txt" 2>&1 &
done
timeout 30m wait