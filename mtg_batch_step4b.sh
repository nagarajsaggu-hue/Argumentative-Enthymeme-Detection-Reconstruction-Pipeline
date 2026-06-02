#!/bin/bash
#SBATCH --job-name=mtg_step4a_adu
#SBATCH --output=logs/mtg_step4a_%j.out
#SBATCH --error=logs/mtg_step4a_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gammaweb
#SBATCH --gres=gpu:1

echo "=========================================="
echo "MIND THE GAP - STEP 4a: ADU Filtering"
echo "Started: $(date)"
echo "=========================================="

cd /mnt/ceph/storage/data-tmp/2026/zuyi6708/argsme-project

/mnt/ceph/storage/data-tmp/2026/zuyi6708/enthymeme_detection/venv_final_2026/bin/python3 scripts/mtg_step4a_adu_inference.py

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
