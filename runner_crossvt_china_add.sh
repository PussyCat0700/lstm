#!/bin/bash

#SBATCH --job-name=crossvt_china_add
#SBATCH --partition=ai4earth
#SBATCH --array=75-243:6%1       # 任务ID范围
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/china_add/crossvt_%A_%a.out
#SBATCH --error=./logs/china_add/crossvt_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=6

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    python train.py 4 --plant_set china_add --plant_number $plant_number --months $1 --batch_size 20 --num_epochs 16 --plant_type 1 &
done
wait
