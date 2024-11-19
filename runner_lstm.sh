#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=9mcnnlstm
#SBATCH --partition=RTX4090,RTX3090,ADA6000 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./logs/9mcnnlstm.out
#SBATCH --error=./logs/9mcnnlstm.err
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=1729372667@qq.com

for i in {0..436}
do
    # 替换 %d 为当前的数字 i
    ckpt_dir="/data1/yfliu/solar_baseline/ablation_ffnn/9m/lstm_$i"
    # 执行命令，传递参数并替换 %d
    python train.py 3 --plant_number $i --checkpoint_dir $ckpt_dir --num_epochs 1000 --batch_size 1024
done