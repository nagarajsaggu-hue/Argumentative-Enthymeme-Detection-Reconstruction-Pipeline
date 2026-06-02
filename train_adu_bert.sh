#!/bin/bash
#SBATCH --job-name=check_packages
#SBATCH --output=logs/check_packages_%j.out
#SBATCH --error=logs/check_packages_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gammaweb
#SBATCH --gres=gpu:1

cd $HOME/argsme-project
mkdir -p logs
/usr/bin/python3 scripts/check_packages.py
