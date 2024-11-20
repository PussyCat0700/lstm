#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=cnnlstmapx
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --cpus-per-task=24  # 每个进程的CPU数量
#SBATCH --array=0-481:10%1       # 任务ID范围
#SBATCH --mem=300GB
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/station_logs/cnnlstm_%A_%a.out
#SBATCH --error=./logs/station_logs/cnnlstm_%A_%a.err
#SBATCH --mail-type=all     # 设置邮件通知类型，可选all, end, fail, begin
#SBATCH --mail-user=1729372667@qq.com # 设置通知邮箱
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=10
# 计算任务的显卡ID
gpu_id=$((task_id % 1))
echo $task_id

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    ckpt_dir=/data1/yfliu/solar_baseline/runs_cnnlstm/lstm_$plant_number
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py 3 --plant_number $plant_number --checkpoint_dir $ckpt_dir --batch_size 1024 --num_epochs 1000 > "$ckpt_dir/log.txt" 2>&1 &
done
wait