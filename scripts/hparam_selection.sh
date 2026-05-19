#!/bin/bash
#SBATCH -p spyder
#SBATCH -c 2
#SBATCH --mem-per-cpu=20GB
#SBATCH -o logs/%J.out
#SBATCH -t 2-0
cmd="python -u ../src/utils/hparam_selection.py --results_dir ../results/hyperparam_search --output ../best_hparams.json"
echo $(date)
echo $cmd
$cmd
