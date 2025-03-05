#!/bin/bash

#SBATCH --job-name=24crossvt_newly_built
#SBATCH --partition=ai4earth
#SBATCH --array=0-36:6%2      # 任务ID范围
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/newly_built/crossvt_%A_%a.out
#SBATCH --error=./logs/newly_built/crossvt_%A_%a.err
#SBATCH --time=7-00:00:00

# 获取当前任务ID
task_id=$SLURM_ARRAY_TASK_ID
offset=6
hours=$1
echo "hours=" $hours

for i in $(seq 0 $((offset-1)))
do
    plant_number=$((task_id + i))
    echo "Running task for plant_number: $plant_number"
    python train.py 4 --plant_set newly_built --plant_number $plant_number --months "12m" --batch_size 20 --num_epochs 16 --period $hours &
done
wait
