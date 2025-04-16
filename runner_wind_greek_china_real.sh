#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=greek_china_real_eclimit_wnd
#SBATCH --time=00:15:00
#SBATCH --partition=RTX3090,RTX4090,L40S,ADA6000
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --array=0-286:10%1       # 任务ID范围
#SBATCH --mem=40GB
#SBATCH --qos=ne_ablation
#SBATCH --output=./logs/station_logs/china_real_eclimit/greek_%A_%a.out
#SBATCH --error=./logs/station_logs/china_real_eclimit/greek_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=10
echo $1
echo $2

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    logdir=/data0/yfliu/logs/solar/china_real_eclimit/greek/${1}_${task_id}_${plant_number}
    echo "exporting to" "$logdir/log.txt"
    mkdir -p $logdir
    python sklearn_train.py 3 --plant_number $plant_number --months $1 --plant_set "china_real_eclimit" --plant_type 0 --period $2 > "$logdir/log.txt" 2>&1 &
done
wait