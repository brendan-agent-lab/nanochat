#!/bin/bash --login
#SBATCH -p gpuL    # gpuA, gpuL
#SBATCH -G 1       # 1 gpu
#SBATCH -t 0-2       # days-hours
#SBATCH -n 8       # (or --ntasks=) Optional number of cores. The amount of host RAM
                   # available to your job is affected by this setting.

echo "Job is using $SLURM_NTASKS CPU core(s)"

module purge
module load libs/cuda

source ~/scratch/nanochat/.venv/bin/activate

which python
python --version

python -m nanochat.report reset

python -m nanochat.dataset -n 32
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node="gpu" -m scripts.base_train -- \
    --depth=8 \
    --run="d8" \
    --model-tag="d8" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \