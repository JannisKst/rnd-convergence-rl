#!/bin/bash

#SBATCH --job-name=experiments
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-65
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source .venv/bin/activate

TASK=${SLURM_ARRAY_TASK_ID}

ENVS=(
    marsrover
    minigrid_empty
    minigrid_doorkey5
    minigrid_doorkey8
    cartpole
    lunarlander
)

if [ "$TASK" -lt 60 ]; then
    # 10 normal runs per environment
    ENV_INDEX=$((TASK / 10))
    SEED=$((TASK % 10))
    ENV=${ENVS[$ENV_INDEX]}

    echo "Running ${ENV}, seed ${SEED}"
    echo "Array task: ${TASK}"

    python scripts/train.py \
        env=${ENV} \
        seed=${SEED}

else
    # One frozen-policy run per environment
    ENV_INDEX=$((TASK - 60))
    ENV=${ENVS[$ENV_INDEX]}

    echo "Running ${ENV}, frozen policy"
    echo "Array task: ${TASK}"

    python scripts/train.py \
        env=${ENV} \
        frozen=true
fi