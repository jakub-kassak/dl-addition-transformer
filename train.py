"""
Absolute Positional Encoding Addition Transformer Training Script.
Features experiment isolation, validation-only mode, and length generalization tracking.
"""

from math import ceil
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from model import GPTLightningModule
from data_1dPE import AdditionDataModule


class ValidationTableCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        console = Console()
        table = Table(title=f"Validation Results (Epoch {trainer.current_epoch})")

        table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
        table.add_column("Token Acc", justify="right", style="green")
        table.add_column("Seq Accuracy", justify="right", style="magenta")
        table.add_column("Loss", justify="right", style="red")

        # Get validation names from datamodule
        val_names = getattr(trainer.datamodule, "val_names", [])
        if not val_names:
            return

        metrics = trainer.callback_metrics

        for i, name in enumerate(val_names):
            # Keys are typically val_acc_token/dataloader_idx_N
            # If there's only one dataloader, no suffix is added by PL sometimes?
            # But we force list of dataloaders, so suffixes should exist.

            suffix = f"/dataloader_idx_{i}"

            # Helper to safely get metric
            def get_m(key_base):
                # Try with suffix
                val = metrics.get(f"{key_base}{suffix}")
                if val is None and i == 0:
                    # Fallback for single dataloader case if it happens
                    val = metrics.get(key_base)
                return val

            token_acc = get_m("val_acc_token")
            seq_acc = get_m("val_acc_seq")
            loss = get_m("val_loss")

            # MV Metrics
            seq_mv = get_m("val_acc_seq_mv")
            digit_mv = get_m("val_acc_digit_mv")

            if token_acc is not None:
                row = [
                    name,
                    f"{token_acc:.4f}",
                    f"{seq_acc:.4f}" if seq_acc is not None else "N/A",
                    f"{loss:.4f}" if loss is not None else "N/A",
                ]

                # Dynamically add columns for MV if they appear (only once)
                if seq_mv is not None and len(table.columns) == 4:
                    table.add_column("Seq MV", justify="right", style="blue")
                    table.add_column("Digit MV", justify="right", style="yellow")

                if seq_mv is not None:
                    row.append(f"{seq_mv:.4f}")
                    row.append(f"{digit_mv:.4f}")

                table.add_row(*row)

        console.print(table)


class ProgressBarCallback(Callback):
    """Add tqdm progress bar for training and validation"""
    
    def on_train_start(self, trainer, pl_module):
        self.train_pbar = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        if self.train_pbar is not None:
            self.train_pbar.close()
        total_batches = trainer.num_training_batches
        self.train_pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {trainer.current_epoch}",
            unit="batch",
            leave=True
        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.train_pbar is not None:
            # Update with loss and accuracy if available
            metrics = {}
            if 'loss' in outputs:
                metrics['loss'] = f"{outputs['loss']:.4f}"
            self.train_pbar.set_postfix(metrics)
            self.train_pbar.update(1)
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_pbar is not None:
            self.train_pbar.close()
            self.train_pbar = None
    
    def on_validation_start(self, trainer, pl_module):
        total_batches = sum(trainer.num_val_batches)
        self.val_pbar = tqdm(
            total=total_batches,
            desc="Validation",
            unit="batch",
            leave=False
        )
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.val_pbar is not None:
            self.val_pbar.update(1)
    
    def on_validation_end(self, trainer, pl_module):
        if self.val_pbar is not None:
            self.val_pbar.close()


def main():
    parser = argparse.ArgumentParser()
    # Experiment Management
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default_exp",
        help="Name for isolating checkpoints and logs.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint to load.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Run validation only on the specified checkpoint.",
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument("--min_train_digits", type=int, default=1)
    parser.add_argument("--max_train_digits", type=int, default=7)
    parser.add_argument("--max_val_digits", type=int, default=15)
    parser.add_argument("--val_step", type=int, default=3)

    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="Define length of one curriculum epoch.",
    )
    parser.add_argument("--curriculum_start", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_ffwd_width", type=int, default=4)
    parser.add_argument("--n_ffwd_depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--dataset_type", type=str, default="position_coupling", 
                       choices=["absolute", "position_coupling"],
                       help="Type of dataset to use")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--val_k",
        type=int,
        default=1,
        help="Number of samples for majority voting validation.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    # Calculate max position: offset_range (100) + sequence length
    # Sequence length = 3L + 3 (for L-digit addition)
    max_pos = 100 + 3 * args.max_val_digits + 3

    # 1. Data Module
    dm = AdditionDataModule(
        min_train_digits=args.min_train_digits,
        max_train_digits=args.max_train_digits,
        max_val_digits=args.max_val_digits,
        val_step=args.val_step,
        batch_size=args.batch_size,
        curriculum_start=args.curriculum_start,
        num_workers=0 if args.smoke_test else args.num_workers,
        dataset_type=args.dataset_type,
        seed=args.seed,
    )
    dm.setup()

    # 2. Model Module
    if args.ckpt_path:
        # Load from checkpoint with hyperparams preserved (override val_k)
        model = GPTLightningModule.load_from_checkpoint(
            args.ckpt_path, val_k=args.val_k
        )
        print(f"✅ Loaded model from {args.ckpt_path} for validation.")
    else:
        model = GPTLightningModule(
            vocab_size=dm.vocab_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            n_ffwd_width=args.n_ffwd_width,
            n_ffwd_depth=args.n_ffwd_depth,
            total_steps=5 if args.smoke_test else args.max_iters,
            max_pos=max_pos,
            pad_token=dm.stoi["#"],
            eq_token=dm.stoi["="],
            val_k=args.val_k,
        )

    # 3. Trainer Setup
    exp_dir = os.path.join("experiments", args.exp_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="gpt-abs-add-{step:04d}-acc={val_avg_seq_acc:.4f}",
        every_n_train_steps=args.eval_interval,
        save_top_k=3,
        monitor="val_avg_seq_acc",
        mode="max",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = ProgressBarCallback()

    trainer_kwargs = {
        "max_steps": 5 if args.smoke_test else args.max_iters,
        "val_check_interval": 2 if args.smoke_test else args.eval_interval,
        "limit_val_batches": 2 if args.smoke_test else args.eval_batch_size,
        "limit_train_batches": 2 if args.smoke_test else args.steps_per_epoch,
        "reload_dataloaders_every_n_epochs": 1,
        "callbacks": [checkpoint_callback, lr_monitor, ValidationTableCallback(), progress_bar],
        "accelerator": "auto",
        "devices": 1,
        "gradient_clip_val": args.grad_clip,
        "logger": TensorBoardLogger("logs", name=args.exp_name),
        "enable_progress_bar": False,  # Disable default progress bar to use our custom one
    }

    trainer = pl.Trainer(**trainer_kwargs)

    # Handle Automatic Resumption
    if not args.ckpt_path and not args.validate_only:
        last_ckpt = os.path.join(exp_dir, "checkpoints", "last.ckpt")
        if os.path.exists(last_ckpt):
            args.ckpt_path = last_ckpt
            print(
                f"🔁 Resuming from last checkpoint for experiment '{args.exp_name}': {last_ckpt}"
            )

    # 4. Action
    if args.validate_only:
        if not args.ckpt_path:
            raise ValueError("Must provide --ckpt_path when using --validate_only.")
        trainer.validate(model, datamodule=dm, ckpt_path=args.ckpt_path)
    else:
        # Debug print
        batch = next(iter(dm.train_dataloader()))
        x, y, pos = batch
        print(f"\n--- Training Experiment: {args.exp_name} ---")
        print(f"Dataset Type: {args.dataset_type}")
        decode = lambda l: "".join([dm.itos[i.item()] for i in l])
        print(f"Sample Input: {decode(x[0])}")
        print(f"Sample Positions: {pos[0].tolist()}")
        print("---------------------------\n")

        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

    # 5. Final Inference Test
    if not args.smoke_test:
        print("\n--- Spot Check (123 + 456) ---")
        model.eval()
        model.to("cpu")

        n1_str, n2_str = "123", "456"
        L = len(n1_str)
        prompt_str = f"{n1_str}+{n2_str}="
        tokens = [dm.stoi[c] for c in prompt_str]
        
        # Position encoding based on dataset type
        offset = 50
        if args.dataset_type == "absolute":
            # Absolute positions: just sequential with an offset
            pos = list(range(len(tokens)))
            pos = [p + offset for p in pos]
        elif args.dataset_type == "position_coupling":
            # Position coupling: decreasing for operands, 0 for operators
            # n1: [L, L-1, ..., 1], +: [0], n2: [L, L-1, ..., 1], =: [0]
            pos_n1 = list(range(L, 0, -1))  # [3, 2, 1]
            pos_plus = [0]
            pos_n2 = list(range(L, 0, -1))  # [3, 2, 1]
            pos_eq = [0]
            pos = pos_n1 + pos_plus + pos_n2 + pos_eq
            pos = [p + offset for p in pos]
        
        x_in = torch.tensor([tokens], dtype=torch.long)
        pos_in = torch.tensor([pos], dtype=torch.long)

        generated = model.generate(x_in, pos_in, max_new_tokens=L + 1)
        decode = lambda l: "".join([dm.itos[i.item()] for i in l])
        print(f"Dataset Type: {args.dataset_type}")
        print(f"Prompt: {prompt_str}")
        print(f"Positions: {pos}")
        print(f"Output: {decode(generated[0][len(tokens) :])}")


if __name__ == "__main__":
    main()