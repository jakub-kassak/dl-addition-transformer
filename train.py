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

        # Markdown table construction
        md_table = "| Dataset | Token Acc | Seq Accuracy | Loss |\n"
        md_table += "|:---|---:|---:|---:|\n"

        # Get validation names from datamodule
        val_names = getattr(trainer.datamodule, "val_config_names", [])
        if not val_names:
            # Fallback if names not found or using old logic
            val_names = getattr(trainer.datamodule, "val_names", [])
            if not val_names:
                return

        metrics = trainer.callback_metrics

        for name in val_names:
            # New format: val_acc_token/val_L1_N2

            def get_m(key_base):
                # Try specific name (Sequential logic)
                val = metrics.get(f"{key_base}/{name}")
                # Fallback to old dataloader_idx logic if name lookup fails (backward compact)
                if val is None:
                    # This fallback is tricky without index, but we assume new logic prevails
                    pass
                return val

            token_acc = get_m("val_acc_token")
            seq_acc = get_m("val_acc_seq")
            loss = get_m("val_loss")

            if token_acc is not None:
                token_acc_val = f"{token_acc:.4f}"
                seq_acc_val = f"{seq_acc:.4f}" if seq_acc is not None else "N/A"
                loss_val = f"{loss:.4f}" if loss is not None else "N/A"

                table.add_row(name, token_acc_val, seq_acc_val, loss_val)
                md_table += (
                    f"| {name} | {token_acc_val} | {seq_acc_val} | {loss_val} |\n"
                )

        console.print(table)

        # Log to TensorBoard
        if hasattr(trainer.logger, "experiment") and hasattr(
            trainer.logger.experiment, "add_text"
        ):
            trainer.logger.experiment.add_text(
                "validation_table", md_table, global_step=trainer.global_step
            )
            print(md_table)


def print_data_sample(dm, max_digits, debug_data=False, prefix=""):
    from data import MultiOperandAdditionDataset

    temp_ds = MultiOperandAdditionDataset(
        dm.hparams.min_train_digits,
        max_digits,
        batch_size=1,
        offset_range=100,
        random_offsets=dm.hparams.random_offsets,
        min_operands=2,
        max_operands=dm.hparams.max_operands,
        data_mode=dm.hparams.data_mode,
        data_type=dm.hparams.data_type,
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
            if dm.hparams.data_type == "default":
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
            elif dm.hparams.data_type == "digit_combinations":
                new_max = dm.train_ds.max_digits

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
            
            with open("logs/log.txt", "a") as f:
                f.write(f"UPDATE: max-ops {new_ops}")

            # Print sample if curriculum advanced OR if debug_data is enabled
            if (
                
                # new_max != self.last_max
                new_ops != self.last_ops
                or self.args.debug_data
            ):
                print_data_sample(
                    dm,
                    new_max,
                    debug_data=self.args.debug_data,
                    prefix=f"Training Epoch {current_epoch} max_ops={new_ops})",
                )
                # self.last_max = new_max
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
    parser.add_argument("--val_step", type=int, default=3)

    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="Define length of one curriculum epoch.",
    )
    parser.add_argument("--curriculum_start", type=int, default=3)
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
        "--random_offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable random positional offsets for pos2.",
    )
    parser.add_argument("--min_operands", type=int, default=2)
    parser.add_argument("--max_operands", type=int, default=5)
    parser.add_argument("--max_val_operands", type=int, default=10)
    parser.add_argument("--val_operand_step", type=int, default=2)
    parser.add_argument(
        "--data_mode",
        type=str,
        default="variable",
        choices=["variable", "padded"],
        help="Defines wheter the numbers should be padded to the same length or not.",
    )
    parser.add_argument("--data_type", type=str, default="default")

    args = parser.parse_args()

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
        random_offsets=args.random_offsets,
        min_operands=args.min_operands,
        max_operands=args.max_operands,
        max_val_operands=args.max_val_operands,
        val_operand_step=args.val_operand_step,
        data_mode=args.data_mode,
        curriculum_operands_start=args.curriculum_operands_start,
        data_type=args.data_type,
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
            ValidationTableCallback(),
            CurriculumLoggerCallback(args),
        ],
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
        # Initial debug print
        print_data_sample(
            dm,
            min(dm.hparams.curriculum_start, dm.hparams.max_train_digits),
            debug_data=args.debug_data,
            prefix=f"Initial State Experiment: {args.exp_name}",
        )

        print("\n--- Training ---")
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

    # 5. Final Inference Test
    if not args.smoke_test:
        print("\n--- Spot Check (Multi Operand) ---")
        model.eval()
        model.to("cpu")

        # Simple test: 2 operands, non-random
        # We need to construct a valid batch manually or just skip complex inference here.
        # Given complexity of new PEs and Dataset, manual construction is error prone.
        # We will skip manual construction and rely on print_data_sample verifying data.
        pass


if __name__ == "__main__":
    main()
