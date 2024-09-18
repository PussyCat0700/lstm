#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=lstm
#SBATCH --partition=RTX4090,RTX3090,ADA6000 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/lstm.out
#SBATCH --error=./logs/lstm.err
#SBATCH --time=50:00:00

for i in {0..183}
do
    # 替换 %d 为当前的数字 i
    ckpt_dir="/data1/yfliu/windpower_baseline/lstm_hourly/lstm_$i"
    # 执行命令，传递参数并替换 %d
    python train.py 0 --plant_number $i --checkpoint_dir $ckpt_dir --num_epochs 1000
done
python gather_results.py 0