import torch
import torch.nn as nn
from model import GPTLightningModule
from data import AdditionDataModule
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math


class HeadAblator:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.ablated_heads = set()  # (layer_idx, head_idx)
        self.n_layer = model.hparams.n_layer
        self.n_head = model.hparams.n_head
        self.head_size = model.hparams.n_embd // model.hparams.n_head

    def _get_ablation_hook(self, head_idx):
        def hook(module, input):
            # input is a tuple, first element is (B, T, C)
            x = input[0]
            start = head_idx * self.head_size
            end = (head_idx + 1) * self.head_size
            x_new = x.clone()
            x_new[:, :, start:end] = 0
            return (x_new,)

        return hook

    def ablate_head(self, layer_idx, head_idx):
        if (layer_idx, head_idx) in self.ablated_heads:
            return

        module = self.model.blocks[layer_idx].sa.c_proj
        hook = module.register_forward_pre_hook(self._get_ablation_hook(head_idx))
        self.hooks.append(hook)
        self.ablated_heads.add((layer_idx, head_idx))

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.ablated_heads = set()

    def evaluate(self, dataloader, device="cpu", max_batches=None):
        self.model.eval()
        self.model.to(device)
        total_correct = 0
        total_count = 0
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
                if max_batches and i >= max_batches:
                    break

                if len(batch) == 6:
                    idx, targets, p1, p2, p3, _ = batch
                else:
                    idx, targets, p1, p2, p3 = batch

                idx, targets, p1, p2, p3 = [
                    b.to(device) for b in [idx, targets, p1, p2, p3]
                ]

                logits, loss = self.model(idx, p1, p2, p3, targets=targets)
                total_loss += loss.item()
                total_batches += 1

                # Same masking logic as model.py
                is_first_eq = (idx == self.model.hparams.eq_token) & (p3 == 1)
                is_scratchpad_or_res = p3 >= 2
                keep_mask = is_first_eq | is_scratchpad_or_res

                targets_masked = targets.clone()
                targets_masked[~keep_mask] = self.model.hparams.pad_token

                valid_acc_mask = targets_masked != self.model.hparams.pad_token
                preds = torch.argmax(logits, dim=-1)
                correct = (preds == targets_masked) & valid_acc_mask

                # Sequence accuracy: all tokens in a row must be correct
                # (correct | (~valid_acc_mask)) is True if either correct or we don't care
                row_correct = (correct | (~valid_acc_mask)).all(dim=1)
                total_correct += row_correct.sum().item()
                total_count += row_correct.size(0)

        avg_acc = total_correct / total_count if total_count > 0 else 0
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        return avg_acc, perplexity

    def sweep_heads(self, dataloader, device="cpu", max_batches=10):
        baseline_acc, baseline_ppl = self.evaluate(dataloader, device, max_batches)
        print(f"Baseline: Accuracy {baseline_acc:.4f}, Perplexity {baseline_ppl:.4f}")

        results = []
        for l in range(self.n_layer):
            for h in range(self.n_head):
                self.clear_hooks()
                self.ablate_head(l, h)
                acc, ppl = self.evaluate(dataloader, device, max_batches)
                acc_drop = baseline_acc - acc
                ppl_inc = ppl - baseline_ppl
                results.append(
                    {
                        "layer": l,
                        "head": h,
                        "accuracy": acc,
                        "drop": acc_drop,
                        "perplexity": ppl,
                        "ppl_inc": ppl_inc,
                    }
                )
                print(
                    f"L{l}H{h}: Acc {acc:.4f} (Drop {acc_drop:.4f}), PPL {ppl:.4f} (Inc {ppl_inc:.4f})"
                )

        self.clear_hooks()
        return results, (baseline_acc, baseline_ppl)


def plot_importance(results, save_path):
    df = pd.DataFrame(results)
    pivot = df.pivot(index="layer", columns="head", values="drop")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0)
    plt.title("Head Importance (Accuracy Drop when Ablated)")
    plt.savefig(save_path)
    print(f"Importance heatmap saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=str, default="head_importance.png")

    # Validation data config
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=10)
    parser.add_argument("--n_operands", type=int, default=2)

    args = parser.parse_args()

    print(f"Loading model from {args.ckpt}...")
    model = GPTLightningModule.load_from_checkpoint(args.ckpt)

    dm = AdditionDataModule(
        min_train_digits=args.min_digits,
        max_train_digits=args.max_digits,
        batch_size=args.batch_size,
        explicit_carry=getattr(model.hparams, "explicit_carry", True),
    )
    dm.setup()
    val_loader = (
        dm.val_dataloader()[0]
        if isinstance(dm.val_dataloader(), list)
        else dm.val_dataloader()
    )

    ablator = HeadAblator(model)
    results, baseline = ablator.sweep_heads(
        val_loader, device=args.device, max_batches=args.max_batches
    )

    plot_importance(results, args.output)


if __name__ == "__main__":
    main()
