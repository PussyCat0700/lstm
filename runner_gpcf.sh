#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=gpcf
#SBATCH --partition=RTX4090,RTX3090,ADA6000,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --output=./logs/gpcf.out
#SBATCH --error=./logs/gpcf.err
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=1729372667@qq.com

for i in {0..328}
do
    # 执行命令，传递参数并替换 %d
    python train_gpcf.py $i
done
python gather_results.py 4