"""
2D Positional Addition Transformer Training Script.
Features experiment isolation, validation-only mode, and length generalization tracking.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb
import argparse
from rich.console import Console
from rich.table import Table

from model import GPTLightningModule
from data import AdditionDataModule


class ValidationTableCallback(Callback):
    def __init__(self, exp_name):
        super().__init__()
        self.exp_name = exp_name
        self.current_config_idx = -1
        self.config_seq_accs = []
        self.config_token_accs = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if outputs is None or "config_idx" not in outputs:
            return

        config_idx = outputs["config_idx"]
        if hasattr(config_idx, "item"):
            config_idx = config_idx.item()

        # If config changed, print results for the previous one
        if config_idx != self.current_config_idx and self.current_config_idx != -1:
            self._print_config_progress(trainer)

        if config_idx != self.current_config_idx:
            self.current_config_idx = config_idx
            self.config_seq_accs = []
            self.config_token_accs = []

        self.config_seq_accs.append(outputs["seq_acc"])
        self.config_token_accs.append(outputs["token_acc"])

    def _print_config_progress(self, trainer):
        if not self.config_seq_accs:
            return

        val_names = getattr(trainer.datamodule, "val_config_names", [])
        if self.current_config_idx < len(val_names):
            name = val_names[self.current_config_idx]
            avg_seq = (
                torch.stack(self.config_seq_accs).mean().item()
                if isinstance(self.config_seq_accs[0], torch.Tensor)
                else sum(self.config_seq_accs) / len(self.config_seq_accs)
            )
            avg_token = (
                torch.stack(self.config_token_accs).mean().item()
                if isinstance(self.config_token_accs[0], torch.Tensor)
                else sum(self.config_token_accs) / len(self.config_token_accs)
            )
            print(
                f"  [Progress] Config {self.current_config_idx:3}: {name:<12} | Seq Acc: {avg_seq:.4f} | Token Acc: {avg_token:.4f}"
            )

            # Log to WandB if available
            if trainer.loggers:
                for logger in trainer.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.experiment.log(
                            {
                                f"val_acc_seq/{name}": avg_seq,
                                f"val_acc_token/{name}": avg_token,
                            },
                        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Print the last config progress before ending
        if self.current_config_idx != -1:
            self._print_config_progress(trainer)
            self.current_config_idx = -1  # Reset for next epoch

        console = Console()

        # Get validation names from datamodule
        val_names = getattr(trainer.datamodule, "val_config_names", [])
        if not val_names:
            val_names = getattr(trainer.datamodule, "val_names", [])
            if not val_names:
                return

        metrics = trainer.callback_metrics

        # Parse names: val_L{L}_N{N}
        import re

        parsed_configs = []
        for name in val_names:
            match = re.match(r"val_L(\d+)_N(\d+)", name)
            if match:
                L, N = int(match.group(1)), int(match.group(2))
                parsed_configs.append({"name": name, "L": L, "N": N})

        if not parsed_configs:
            return

        unique_ls = sorted(list(set(c["L"] for c in parsed_configs)))
        unique_ns = sorted(list(set(c["N"] for c in parsed_configs)))

        # Define styles for formatting
        styles = {"val_acc_token": "green", "val_acc_seq": "magenta", "val_loss": "red"}

        def create_grid_table(title, metric_key):
            table = Table(title=title)
            table.add_column("Digits \\ Ops", justify="left", style="cyan")
            for N in unique_ns:
                table.add_column(
                    f"N={N}", justify="right", style=styles.get(metric_key, "")
                )

            # Markdown table construction
            md = f"### {title}\n\n"
            md += (
                "| Digits \\ Ops | "
                + " | ".join([f"N={N}" for N in unique_ns])
                + " |\n"
            )
            md += "|:---|" + "|".join(["---:"] * len(unique_ns)) + "|\n"

            for L in unique_ls:
                row_vals = [f"L={L}"]
                md_row = [f"L={L}"]
                for N in unique_ns:
                    # Find matching name
                    config_name = next(
                        (
                            c["name"]
                            for c in parsed_configs
                            if c["L"] == L and c["N"] == N
                        ),
                        None,
                    )
                    val = None
                    if config_name:
                        val = metrics.get(f"{metric_key}/{config_name}")

                    if val is not None:
                        val_str = f"{val:.4f}"
                    else:
                        val_str = "-"
                    row_vals.append(val_str)
                    md_row.append(val_str)
                table.add_row(*row_vals)
                md += "| " + " | ".join(md_row) + " |\n"

            return table, md

        # Create the three tables
        t_token, md_token = create_grid_table(
            f"Token Accuracy (Epoch {trainer.current_epoch})", "val_acc_token"
        )
        t_seq, md_seq = create_grid_table(
            f"Sequence Accuracy (Epoch {trainer.current_epoch})", "val_acc_seq"
        )
        t_loss, md_loss = create_grid_table(
            f"Loss (Epoch {trainer.current_epoch})", "val_loss"
        )

        full_md = md_token + "\n" + md_seq + "\n" + md_loss

        # Print to console
        console.print(t_token)
        console.print(t_seq)
        console.print(t_loss)

        # Log to loggers
        if trainer.loggers:
            for logger in trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_text(
                        "validation_tables", full_md, global_step=trainer.global_step
                    )
                elif isinstance(logger, WandbLogger):
                    # Log as markdown in WandB
                    logger.experiment.log(
                        {
                            "validation_tables": wandb.Html(
                                f"<pre style='font-family: \"Courier New\", Courier, monospace; line-height: 1.2;'>{full_md}</pre>"
                            )
                        },
                        step=trainer.global_step,
                    )

        # Save to CSV
        import csv

        exp_dir = os.path.join("experiments", self.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        csv_path = os.path.join(exp_dir, "val_results.csv")
        file_exists = os.path.exists(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "epoch",
                        "step",
                        "config_name",
                        "L",
                        "N",
                        "val_acc_token",
                        "val_acc_seq",
                        "val_loss",
                    ]
                )

            for c in parsed_configs:
                name = c["name"]
                L = c["L"]
                N = c["N"]
                acc_token = metrics.get(f"val_acc_token/{name}", 0.0)
                acc_seq = metrics.get(f"val_acc_seq/{name}", 0.0)
                loss = metrics.get(f"val_loss/{name}", 0.0)

                # Metrics might be tensors
                if hasattr(acc_token, "item"):
                    acc_token = acc_token.item()
                if hasattr(acc_seq, "item"):
                    acc_seq = acc_seq.item()
                if hasattr(loss, "item"):
                    loss = loss.item()

                writer.writerow(
                    [
                        trainer.current_epoch,
                        trainer.global_step,
                        name,
                        L,
                        N,
                        acc_token,
                        acc_seq,
                        loss,
                    ]
                )


def print_data_sample(dm, max_digits, debug_data=False, prefix=""):
    from data import MultiOperandAdditionDataset

    temp_ds = MultiOperandAdditionDataset(
        dm.hparams.min_train_digits,
        max_digits,
        batch_size=1,
        offset_range=dm.hparams.offset_range,
        random_offsets=dm.hparams.random_offsets,
        min_operands=2,
        max_operands=dm.hparams.max_operands,
        data_mode=dm.hparams.data_mode,
        explicit_carry=dm.hparams.explicit_carry,
    )
    batch = temp_ds.generate_batch()
    x, y, p1, p2, p3 = batch

    if prefix:
        print(f"\n--- {prefix} ---")

    def decode_and_mod(l):
        return "".join(
            [
                str(dm.itos[i.item()])
                if not isinstance(dm.itos[i.item()], str)
                or not dm.itos[i.item()].isdigit()
                else str(int(dm.itos[i.item()]) % 10)
                for i in l
            ]
        )

    # Simple decode for debugging raw tokens
    decode_raw = lambda l: "".join([f"[{dm.itos[i.item()]}]" for i in l])
    print(f"Sample Input (mod 10): {decode_and_mod(x[0])}")
    print(f"Sample Input (raw):    {decode_raw(x[0])}")

    if debug_data:
        print(f"Sample Pos1:           {p1[0].tolist()}")
        print(f"Sample Pos2:           {p2[0].tolist()}")
        print(f"Sample Pos3:           {p3[0].tolist()}")
        print("\nDetailed Token-Position alignment:")
        tokens_str = [f"{dm.itos[i.item()]}" for i in x[0]]
        p1_str = [str(p.item()) for p in p1[0]]
        p2_str = [str(p.item()) for p in p2[0]]
        p3_str = [str(p.item()) for p in p3[0]]

        # Simple alignment
        for t, p1_v, p2_v, p3_v in zip(tokens_str, p1_str, p2_str, p3_str):
            print(f"  Token: {t:4} | Pos1: {p1_v} | Pos2: {p2_v} | Pos3: {p3_v}")

    print("---------------------------\n")


class CurriculumLoggerCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.last_max = -1
        self.last_ops = -1

    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if hasattr(dm, "train_ds"):
            # Update Curriculum
            current_epoch = trainer.current_epoch
            new_max = min(
                dm.hparams.curriculum_start + current_epoch, dm.hparams.max_train_digits
            )
            dm.train_ds.max_digits = new_max

            pl_module.log(
                "curriculum/max_digits",
                float(new_max),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            # Update Operands Curriculum
            if dm.hparams.curriculum_operands_start is not None:
                new_ops = min(
                    dm.hparams.curriculum_operands_start + current_epoch,
                    dm.hparams.max_operands,
                )
                dm.train_ds.max_operands = new_ops
                pl_module.log(
                    "curriculum/max_operands",
                    float(new_ops),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            else:
                new_ops = dm.train_ds.max_operands

            # Print sample if curriculum advanced OR if debug_data is enabled
            if (
                new_max != self.last_max
                or new_ops != self.last_ops
                or self.args.debug_data
            ):
                print_data_sample(
                    dm,
                    new_max,
                    debug_data=self.args.debug_data,
                    prefix=f"Training Epoch {current_epoch} (max_digits={new_max}, max_ops={new_ops})",
                )
                self.last_max = new_max
                self.last_ops = new_ops


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
    parser.add_argument("--val_batch_size", type=int, default=20)
    parser.add_argument(
        "--val_samples",
        type=int,
        default=None,
        help="Number of examples per validation configuration. Defaults to val_batch_size.",
    )
    parser.add_argument(
        "--debug_data",
        action="store_true",
        help="Print detailed data samples (inputs, pos1, pos2) before training.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument("--min_train_digits", type=int, default=1)
    parser.add_argument("--max_train_digits", type=int, default=7)
    parser.add_argument("--max_val_digits", type=int, default=15)
    parser.add_argument("--min_val_digits", type=int, default=None)
    parser.add_argument("--val_step", type=int, default=3)

    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="Define length of one curriculum epoch.",
    )
    parser.add_argument("--curriculum_start", type=int, default=None)
    parser.add_argument("--curriculum_operands_start", type=int, default=None)
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
    parser.add_argument("--rope_theta", type=float, default=40000)
    parser.add_argument(
        "--pos_emb_type",
        type=str,
        default="rope",
        choices=["rope", "learned", "mixed"],
        help="Type of positional embedding to use.",
    )
    parser.add_argument(
        "--random-offsets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable random positional offsets for pos2.",
    )
    parser.add_argument("--offset_range", type=int, default=0)
    parser.add_argument("--min_operands", type=int, default=2)
    parser.add_argument("--max_operands", type=int, default=5)
    parser.add_argument("--max_val_operands", type=int, default=10)
    parser.add_argument("--min_val_operands", type=int, default=None)
    parser.add_argument("--val_operand_step", type=int, default=2)
    parser.add_argument(
        "--data_mode",
        type=str,
        default="variable",
        choices=["variable", "padded"],
        help="Defines wheter the numbers should be padded to the same length or not.",
    )
    parser.add_argument(
        "--explicit_carry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable explicit carry tokens (10-19) in scratchpad.",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable WandB logging."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="addition-transformer",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="WandB entity/username."
    )

    args = parser.parse_args()
    args.curriculum_start = (
        args.curriculum_start
        if args.curriculum_start is not None
        else args.max_train_digits
    )
    args.curriculum_operands_start = (
        args.curriculum_operands_start
        if args.curriculum_operands_start is not None
        else args.max_operands
    )
    args.val_samples = (
        args.val_samples if args.val_samples is not None else args.val_batch_size
    )

    pl.seed_everything(args.seed)

    # 1. Data Module
    dm = AdditionDataModule(
        min_train_digits=args.min_train_digits,
        max_train_digits=args.max_train_digits,
        max_val_digits=args.max_val_digits,
        val_step=args.val_step,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        curriculum_start=args.curriculum_start,
        num_workers=0 if args.smoke_test else args.num_workers,
        offset_range=args.offset_range,
        random_offsets=args.random_offsets,
        min_operands=args.min_operands,
        max_operands=args.max_operands,
        max_val_operands=args.max_val_operands,
        val_operand_step=args.val_operand_step,
        min_val_digits=args.min_val_digits,
        min_val_operands=args.min_val_operands,
        data_mode=args.data_mode,
        curriculum_operands_start=args.curriculum_operands_start,
        explicit_carry=args.explicit_carry,
        val_samples=args.val_samples,
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
            eq_token=dm.stoi["="],
            rope_theta=args.rope_theta,
            pos_emb_type=args.pos_emb_type,
            explicit_carry=args.explicit_carry,
        )

    # 2b. Safety Checks
    if args.pos_emb_type == "learned" and args.random_offsets:
        # Check if max possible p2 (max_len + offset_range) fits in embedding table
        # max_len = max_val_digits + ceil(log10(max_val_operands))
        import math

        max_len = args.max_val_digits + math.ceil(math.log10(args.max_val_operands))
        if max_len + args.offset_range >= model.hparams.max_pos2:
            print(
                f"⚠️  [WARNING] Learned PEs might overflow! max_len({max_len}) + offset_range({args.offset_range}) >= max_pos2({model.hparams.max_pos2})"
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
        "limit_val_batches": 2 if args.smoke_test else 1.0,
        "limit_train_batches": 2 if args.smoke_test else args.steps_per_epoch,
        "reload_dataloaders_every_n_epochs": 1,
        "callbacks": [
            checkpoint_callback,
            lr_monitor,
            ValidationTableCallback(args.exp_name),
            CurriculumLoggerCallback(args),
        ],
        "accelerator": "auto",
        "devices": 1,
        "gradient_clip_val": args.grad_clip,
        "logger": [],
        # "compile": True,
    }

    # Setup Loggers
    loggers = [TensorBoardLogger("logs", name=args.exp_name)]
    if args.use_wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=args.exp_name,
                entity=args.wandb_entity,
                config=args,
            )
        )
    trainer_kwargs["logger"] = loggers

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
        print("\n--- Validation ---")
        trainer.validate(model, datamodule=dm, ckpt_path=args.ckpt_path)
    else:
        # Initial debug print
        print_data_sample(
            dm,
            min(dm.hparams.curriculum_start, dm.hparams.max_train_digits),
            debug_data=args.debug_data,
            prefix=f"Initial State Experiment: {args.exp_name}",
        )

        print("\n--- Training ---")
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
