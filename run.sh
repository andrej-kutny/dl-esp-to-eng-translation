#!/bin/bash
#SBATCH --job-name=dl_a2
#SBATCH --output=logs/%j_dl_a2.out
#SBATCH --error=logs/%j_dl_a2.err
#SBATCH --time=0-02:59:59
#SBATCH --gres=gpu:1
#SBATCH --partition=priority

mkdir -p logs

export TF_GPU_ALLOCATOR=cuda_malloc_async
export KERAS_BACKEND=tensorflow

# usage:
#   sbatch run.sh --mha keras  --epochs 20
#   sbatch run.sh --mha custom --epochs 20
#   sbatch run.sh --mha custom --epochs 20 --visualize --sentence "tom esta en la cocina"
#
# follow logs:
#   tail logs/<jobid>_dl_a2.out --follow

python src/cli.py "$@"
