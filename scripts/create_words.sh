#!/bin/bash
#SBATCH -p spyder
#SBATCH -c 2
#SBATCH --mem-per-cpu=20GB
#SBATCH -o logs/%J.out
#SBATCH -t 2-0
cmd="python -u ../src/data/create_words.py --languages_csv ../languages.csv --output ../data --bins (min,50),(51,100),(101,150),(151,200),(201,250),(251,300),(301,350),(351,400),(401,450),(451,500)"
echo $(date)
echo $cmd
$cmd
