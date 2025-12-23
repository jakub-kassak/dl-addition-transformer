"""
2D Positional Addition Transformer Training Script.
Features experiment isolation, validation-only mode, and length generalization tracking.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from rich.console import Console
from rich.table import Table

from model import GPTLightningModule
from data import AdditionDataModule


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

            if token_acc is not None:
                table.add_row(
                    name,
                    f"{token_acc:.4f}",
                    f"{seq_acc:.4f}" if seq_acc is not None else "N/A",
                    f"{loss:.4f}" if loss is not None else "N/A",
                )

        console.print(table)


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
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_ffwd_width", type=int, default=4)
    parser.add_argument("--n_ffwd_depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # 1. Data Module
    dm = AdditionDataModule(
        min_train_digits=args.min_train_digits,
        max_train_digits=args.max_train_digits,
        max_val_digits=args.max_val_digits,
        val_step=args.val_step,
        batch_size=args.batch_size,
        curriculum_start=args.curriculum_start,
        num_workers=0 if args.smoke_test else args.num_workers,
    )
    dm.setup()

    # 2. Model Module
    if args.ckpt_path:
        # Load from checkpoint with hyperparams preserved
        model = GPTLightningModule.load_from_checkpoint(args.ckpt_path)
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
            max_pos2=1500,
            pad_token=dm.stoi["#"],
            eq_token=dm.stoi["="],
        )

    # 3. Trainer Setup
    exp_dir = os.path.join("experiments", args.exp_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_dir, "checkpoints"),
        filename="gpt-2d-add-{step:04d}-acc={val_avg_seq_acc:.4f}",
        every_n_train_steps=args.eval_interval,
        save_top_k=3,
        monitor="val_avg_seq_acc",
        mode="max",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer_kwargs = {
        "max_steps": 5 if args.smoke_test else args.max_iters,
        "val_check_interval": 2 if args.smoke_test else args.eval_interval,
        "limit_val_batches": 2 if args.smoke_test else 50,
        "limit_train_batches": 2 if args.smoke_test else args.steps_per_epoch,
        "reload_dataloaders_every_n_epochs": 1,
        "callbacks": [checkpoint_callback, lr_monitor, ValidationTableCallback()],
        "accelerator": "auto",
        "devices": 1,
        "gradient_clip_val": args.grad_clip,
        "logger": TensorBoardLogger("logs", name=args.exp_name),
        # "compile": True,
    }

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.compile_model = True

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
        x, y, p1, p2 = batch
        print(f"\n--- Training Experiment: {args.exp_name} ---")
        decode = lambda l: "".join([dm.itos[i.item()] for i in l])
        print(f"Sample Input: {decode(x[0])}")
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
        p1 = ([1] * (L + 1)) + ([2] * (L + 1))
        p2 = list(range(1, L + 1)) + [1] + list(range(1, L + 1)) + [1]

        offset = 50
        p2 = [p + offset for p in p2]

        x_in = torch.tensor([tokens], dtype=torch.long)
        p1_in = torch.tensor([p1], dtype=torch.long)
        p2_in = torch.tensor([p2], dtype=torch.long)

        generated = model.generate(x_in, p1_in, p2_in, max_new_tokens=L + 1)
        decode = lambda l: "".join([dm.itos[i.item()] for i in l])
        print(f"Prompt: {prompt_str}")
        print(f"Output: {decode(generated[0][len(tokens) :])}")


if __name__ == "__main__":
    main()
