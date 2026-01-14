#!/bin/bash

#SBATCH --job-name=MixedPE_2D_run
#SBATCH --time=12:00:00
#SBATCH --account=deep_learning
#SBATCH --output=logs/MixedPE_2D_%j.out
#SBATCH --error=logs/MixedPE_2D_%j.err
#SBATCH --mem=32G

# Load modules
. /etc/profile.d/modules.sh
module add cuda/12.4

# Activate environment
source .venv_cluster/bin/activate

# WandB login
export WANDB_API_KEY=
wandb login $WANDB_API_KEY

# Set python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training
python train.py \
  --exp_name MixedPE_2D_run_L6_H4_nocarry \
  --n_layer 6 \
  --n_embd 256 \
  --n_head 4 \
  --learning_rate 5e-4 \
  --batch_size 256 \
  --max_iters 10000 \
  --steps_per_epoch 500 \
  --max_train_digits 10 \
  --val_step 4 \
  --max_val_digits 50 \
  --num_workers 2 \
  --pos_emb_type abc_mixed \
  --no-explicit-carry
