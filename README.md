# Walkthrough - 2D Positional Addition Transformer

This project implements a Transformer model for multi-digit addition with a focus on high performance and length generalization using 2D positional encodings.

## Key Components

- **data.py**: Optimized vectorized data generation with multiple validation sets and Epoch-based Curriculum Learning.
- **model.py**: Transformer with dual positional embeddings, structural accuracy metrics, and Masked Loss training.
- **train.py**: Main entry point with Rich terminal monitoring.

## Key Features

### 1. Training Diagnostics
To monitor stability and convergence, we now automatically log:
- **Gradient Norm**: Determine if gradients are exploding or vanishing.
- **Max Logits**: Track `logits.max()` to detect value scaling issues before they cause overflow.
- **Learning Rate**: `lr-AdamW` via `LearningRateMonitor`.

### 2. Targeted Training (Masked Loss)
We have optimized the training to focus only on the result generation. The model receives gradients solely for predicting:
- The first digit of the sum (triggered by `=`)
- All subsequent sum digits. 

It is not penalized for predicting the input equation itself, making learning more efficient.

### 3. Validation Grid & Live Table
Validation sets are now automatically generated to cover the full spectrum of lengths, plus critical curriculum points. Example if `val_step=4`: L1, L5, L9, L13... PLUS your current curriculum level.

After every validation epoch, a rich formatted table appears in your terminal:

| Dataset | Token Acc | Seq Accuracy | Loss |
| :--- | :--- | :--- | :--- |
| val_L1 | 1.0000 | 1.0000 | 0.01 |
| val_L5 | 0.9998 | 0.9950 | 0.05 |
| val_L7 | 0.9500 | 0.8000 | 0.40 |
| val_L8 | 0.8500 | 0.1200 | 1.20 |

### 4. Curriculum Learning
- **Start**: Epoch 0 begins with 1-3 digit sums.
- **Progression**: Every epoch adds +1 to the max digit length.
- **Config**: `--curriculum_start` (default 3), `--steps_per_epoch` (default 1000).

### 5. Configurable FeedForward Layer
The Transformer FeedForward block is now parameterizable via CLI:
- `--n_ffwd_width`: Expansion factor for the middle dimension (default 4).
- `--n_ffwd_depth`: Number of linear layers in the block (default 1).

### 6. Stability & Robustness
- **DataLoader Fix**: Added `persistent_workers` logic to handle systems where `num_workers=0` without crashing.
- **Manual Grad Logging**: Manual logging of gradient norms and max logits for better stability monitoring.
- **Improved Checkpointing**: Keeps the 3 best models based on `val_avg_seq_acc` (average across all validation sets), ensuring overall performance is prioritized.

## Running Experiments

### Train 10M Param Model
```bash
python train.py \
  --exp_name 10M_run_01 \
  --n_layer 6 \
  --n_embd 384 \
  --n_head 6 \
  --learning_rate 6e-4 \
  --batch_size 256 \
  --max_iters 50000 \
  --steps_per_epoch 2000 \
  --max_train_digits 15 \
  --curriculum_start 3 \
  --val_step 4 \
  --max_val_digits 30
```

### Inspect Data
To see exactly what the model sees (tokens + positional encodings) and verify the masking logic structure:

```bash
python inspect_data.py --min_digits 3 --max_digits 5
```