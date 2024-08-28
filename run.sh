#!/bin/bash
#SBATCH --account=c_earth
#SBATCH --job-name=xrd
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00
#SBATCH --output="%x_%j.out"
nice -n 19 python diff2struct_lstm.py