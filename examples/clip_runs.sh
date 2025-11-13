#!/bin/bash

set -e

export HF_DATASETS_CACHE=/opt/dlami/nvme/hf_cache
export TRANSFORMERS_CACHE=/opt/dlami/nvme/hf_cache
export HF_HOME=/opt/dlami/nvme/hf_cache


# Check if the user is already logged into WandB
if wandb status &>/dev/null; then
    echo "Already logged into WandB"
else
    echo "Not logged into WandB. Please log in:"
    wandb login --relogin
fi



for seed in 40; do
    for lora_rank in 0; do
        for proportion in 1; do
            for patch_size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do

                echo "Running with alpha=$alpha, lora_rank=$lora_rank, proportion=$proportion, seed=$seed"

                python clip_finetuning.py ++params.dataset=uoft-cs/cifar10 ++params.label_key=label ++params.zeroshot_dataset=uoft-cs/cifar10 ++params.zero_label=label ++params.use_lora=False ++params.lora_rank=$lora_rank ++params.spur_proportion=$proportion ++params.patch_size=$patch_size ++params.use_spurious=True

            done
        done
    done
done
