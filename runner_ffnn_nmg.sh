#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=ffnn_nmg
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --array=0-438:16%2       # 任务ID范围
#SBATCH --mem=80GB
#SBATCH --qos=ne_ablation
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/station_logs/nmg/ffnn_%A_%a.out
#SBATCH --error=./logs/station_logs/nmg/ffnn_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=16
# 计算任务的显卡ID
gpu_id=$((task_id % 1))
echo $1

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    logdir=/data1/yfliu/logs/solar/nmg/ffnn/${1}_${task_id}_${plant_number}
    echo "exporting to logdir $logdir/log.txt"
    mkdir -p $logdir
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py 1 --plant_set nmg --plant_number $plant_number --months $1 --batch_size 1024 --period 24 > "$logdir/log.txt" 2>&1 &
done
wait