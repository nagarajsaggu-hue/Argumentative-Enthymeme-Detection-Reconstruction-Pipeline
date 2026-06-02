#!/bin/bash
#SBATCH --job-name=mtg_step1_load
#SBATCH --output=logs/mtg_step1_%j.out
#SBATCH --error=logs/mtg_step1_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gammaweb

echo "=========================================="
echo "MIND THE GAP - STEP 1: Load Dataset"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

cd $HOME/argsme-project

mkdir -p logs
mkdir -p data/processed

/usr/bin/python3 -m spacy download en_core_web_sm
/mnt/ceph/storage/data-tmp/2026/zuyi6708/enthymeme_detection/venv/bin/python3 scripts/mtg_step1_load.py

exit_code=$?
echo "=========================================="
echo "Completed: $(date)"
echo "Exit code: $exit_code"
echo "=========================================="

exit $exit_code
