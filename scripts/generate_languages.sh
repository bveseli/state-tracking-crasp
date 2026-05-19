#!/bin/bash
#SBATCH -p spyder
#SBATCH -c 2
#SBATCH --mem-per-cpu=20GB
#SBATCH -o logs/%J.out
#SBATCH -t 2-0
cmd="python -u ../src/language_creation/generate_languages.py"
echo $(date)
echo $cmd
$cmd
