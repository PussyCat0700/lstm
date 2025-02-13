#!/bin/bash

#SBATCH --job-name=40crossvt_nanwang
#SBATCH --partition=ai4earth
#SBATCH --array=47-95:6%1       # 任务ID范围
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/nanwang/crossvt_%A_%a.out
#SBATCH --error=./logs/nanwang/crossvt_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=6

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    python train.py 4 --plant_set nanwang --plant_number $plant_number --months $1 --batch_size 20 --num_epochs 16 --plant_type 1 --period 40 &
done
wait
