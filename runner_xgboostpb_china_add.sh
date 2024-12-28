#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=xgboostpb_china_add
#SBATCH --time=00:15:00
#SBATCH --partition=RTX3090,RTX4090,ADA6000
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --array=0-243:10%1       # 任务ID范围
#SBATCH --mem=40GB
#SBATCH --qos=ne_ablation
#SBATCH --output=./logs/china_add/xgboostpb_%A_%a.out
#SBATCH --error=./logs/china_add/xgboostpb_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=10
echo $1

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    logdir=/data1/yfliu/logs/solar/china_add/xgboostpb/${1}_${task_id}_${plant_number}
    echo "exporting to" "$logdir/log.txt"
    mkdir -p $logdir
    python sklearn_train.py 4 --plant_number $plant_number --months $1 --plant_set "china_add"> "$logdir/log.txt" 2>&1 &
done
wait