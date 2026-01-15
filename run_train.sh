#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=deep_learning
#SBATCH --output=slurm_logs/MO%j.out
#SBATCH --error=slurm_logs/MO%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs logs experiments

# Enable modules in batch scripts
. /etc/profile.d/modules.sh
module add cuda/12.9

cd ~/addition-transformer
source .venv/bin/activate

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
nvidia-smi
python -V
which python
echo "============================"

python train.py \
  --exp_name "MO" \
  --n_embd 384 --n_head 4 --n_layer 6 \
  --min_operands 2 --max_operands 5 \
  --min_train_digits 1 --max_train_digits 5 \
  --data_mode padded \
  --data_type digit_combinations \
  --pos_emb_type mixed \
  --learning_rate 5e-4 \
  --batch_size 256 \
  --max_iters 10000 \
  --steps_per_epoch 1000 \
  --eval_interval 500 \
  --rope_theta 10000.0 \
  --no-random_offsets \
  --debug_data \
  --val_batch_size 20 \
  --max_val_operands 7 \
  --val_operand_step 1 \
  --max_val_digits 8

echo "=== Job finished: $(date) ==="