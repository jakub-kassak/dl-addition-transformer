#!/bin/bash

#SBATCH --job-name=MultiOperandAddition01
#SBATCH --time=1:00:00
#SBATCH --account=deep_learning
#SBATCH --output=logs/MultiOperandAddition01_%j.out
#SBATCH --error=logs/MultiOperandAddition01_%j.err
#SBATCH --mem=32G

# Load modules
. /etc/profile.d/modules.sh
module add cuda/12.4

# Activate environment
source .venv_cluster/bin/activate

# WandB login
export WANDB_API_KEY=bccc3e2d0a4ff78b206521ca8ebc99653d888316
wandb login $WANDB_API_KEY

# Set python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training
python train.py \
  --exp_name MultiOperandAddition02_no_curriculum \
  --n_layer 2 \
  --n_embd 384 \
  --n_head 2 \
  --n_ffwd_depth 2 \
  --learning_rate 3e-4 \
  --batch_size 256 \
  --max_iters 5000 \
  --steps_per_epoch 500 \
  --min_train_digits 1 \
  --max_train_digits 10 \
  --max_val_digits 15 \
  --min_operands 2 \
  --max_operands 10 \
  --max_val_operands 15 \
  --val_step 2 \
  --val_operand_step 2 \
  --num_workers 2 \
  --pos_emb_type mixed \
  --debug_data \
  --use_wandb
