#!/bin/bash

#SBATCH --job-name=export
#SBATCH --partition=ai4earth
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH --time=7-00:00:00

echo $1
python preprocess.py /mnt/petrelfs/guxinyu/sftp-src/yfliu_dev/lstm/solardata/china_data_add/china_add_info_new.csv $1 --overwrite
