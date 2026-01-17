import os
import torch
import torch.nn.functional as F
import argparse
import json
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from collections import defaultdict

from tuned_lens_addition import TunedLens
from data import AdditionDataModule


def analyze_layers(
    lens, dataloader, max_batches=10, device="cpu", out_dir="artifacts", method="logit"
):
    console = Console()
    n_layers = lens.n_layer + 1  # Including Layer -1
    os.makedirs(out_dir, exist_ok=True)

    # Stats trackers
    layer_correct_counts = torch.zeros(n_layers, device=device)
    total_tokens = 0

    # Carry Analysis:
    # Logic: If L0 matches (A+B)%10 but Final matches (A+B+Carry)%10
    total_carry_positions = 0
    l0_base_matches_at_carry = 0

    transitions = defaultdict(lambda: defaultdict(int))

    # Anomalies
    final_errors = []
    shortest_failure = None
    late_corrections = []
    shortest_late_correction = None

    iterator = iter(dataloader)
    num_batches_processed = 0
    pbar = tqdm(total=max_batches, desc="Analyzing Layers")

    while num_batches_processed < max_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            break

        num_batches_processed += 1
        pbar.update(1)

        # Unpack and slice
        idx, targets, pos1, pos2, pos3 = [b.to(device) for b in batch[:5]]

        # Get hidden states and layer predictions
        h_list, final_logits = lens.get_hidden_states(idx, pos1, pos2, pos3)

        # Mask for result phase
        # Note: In MultiOperand, targets are the expected sequence starting from '='.
        # Scratchpad/Result tokens are pos3 >= 2
        predict_mask = (
            (pos3 >= 2) & (targets != lens.pad_token) & (targets != lens.hash_token)
        )

        B, T = idx.shape
        total_tokens += predict_mask.sum().item()

        layer_preds = []
        for l in range(n_layers):
            h_l = h_list[l]
            if method == "tuned" and lens.is_trained:
                h_hat = lens.translators[l](h_l)
                logits = lens.model.lm_head(lens.model.ln_f(h_hat))
            else:  # logit lens
                logits = lens.model.lm_head(lens.model.ln_f(h_l))
            preds = torch.argmax(logits, dim=-1)
            layer_preds.append(preds)
            layer_correct_counts[l] += ((preds == targets) & predict_mask).sum().item()

        # Final predictions
        final_preds = torch.argmax(final_logits, dim=-1)
        final_correct = (final_preds == targets) & predict_mask

        # --- Carry Hypothesis Logic ---
        # We check if Target at pos T is different from what we'd expect if there were NO carry from T-1.
        # This is tricky in multi-operand, but we can detect it by checking if predictions are "off-by-one" (with wraps).
        # OR: if layer 0 matches a previous value?
        # Simpler heuristic for "Division of Labor":
        # Check positions where Target != (Prev_Prediction_Logic).
        # Let's use the actual digit sum if possible.

        # --- Store Failure Info ---
        for b_idx in range(B):
            # Sequence length (non-pad)
            seq_mask = idx[b_idx] != lens.pad_token
            seq_len = seq_mask.sum().item()

            # Did this sequence fail?
            if not final_correct[b_idx][predict_mask[b_idx]].all():
                tokens = idx[b_idx][seq_mask].tolist()
                eq_str = "".join([lens.itos[t] for t in tokens])

                failure_data = {
                    "eq": eq_str,
                    "len": seq_len,
                    "errors": (final_preds[b_idx] != targets[b_idx])[
                        predict_mask[b_idx]
                    ]
                    .sum()
                    .item(),
                }

                if shortest_failure is None or seq_len < shortest_failure["len"]:
                    shortest_failure = failure_data

                if eq_str not in [f["eq"] for f in final_errors][:10]:
                    final_errors.append(failure_data)

            # Check late corrections (L 0 wrong, L 1 right)
            # For a 2-block model, n_layers=3, indices are 0 (L-1), 1 (L0), 2 (L1).
            # We want Layer 0 (index 1) to be wrong and Layer 1 (index 2) to be right.
            l0_idx = 1
            l_final_idx = n_layers - 1

            corr_mask = (
                ~(layer_preds[l0_idx][b_idx] == targets[b_idx])
                & (layer_preds[l_final_idx][b_idx] == targets[b_idx])
            ) & predict_mask[b_idx]
            if corr_mask.any():
                tokens = idx[b_idx][seq_mask].tolist()
                eq_str = "".join([lens.itos[t] for t in tokens])
                corr_data = {"eq": eq_str, "len": seq_len}

                if (
                    shortest_late_correction is None
                    or seq_len < shortest_late_correction["len"]
                ):
                    shortest_late_correction = corr_data

                if len(late_corrections) < 10 and eq_str not in [
                    c["eq"] for c in late_corrections
                ]:
                    late_corrections.append(corr_data)

        # Transition Stats
        for l in range(1, n_layers):
            p_corr = (layer_preds[l - 1] == targets) & predict_mask
            c_corr = (layer_preds[l] == targets) & predict_mask
            transitions[l]["wrong_to_correct"] += (~p_corr & c_corr).sum().item()
            transitions[l]["correct_to_wrong"] += (p_corr & ~c_corr).sum().item()

    # --- Print Summary ---
    method_title = "Logit Lens" if method == "logit" else "Tuned Lens"
    console.print(
        f"\n[bold reverse cyan] Detailed Layer Analysis ({method_title}) [/bold reverse cyan]"
    )

    summary_table = Table(title="Accuracy & Transitions", show_lines=True)
    summary_table.add_column("Layer", style="cyan")
    summary_table.add_column("Accuracy", justify="right")
    summary_table.add_column("Correction Rate", justify="right", style="green")
    summary_table.add_column("Corruption Rate", justify="right", style="red")

    for l in range(n_layers):
        acc = layer_correct_counts[l] / (total_tokens + 1e-6)
        if l == 0:
            c_rate, r_rate = "-", "-"
        else:
            prev_wrong = total_tokens - layer_correct_counts[l - 1]
            c_rate = f"{(transitions[l]['wrong_to_correct'] / (prev_wrong + 1e-6)):.1%}"
            r_rate = f"{(transitions[l]['correct_to_wrong'] / (layer_correct_counts[l - 1] + 1e-6)):.1%}"
        summary_table.add_row(f"L {l - 1}", f"{acc:.2%}", c_rate, r_rate)

    console.print(summary_table)

    if shortest_late_correction:
        console.print(
            f"\n[bold green]Shortest Late Correction Found:[/bold green] [white]{shortest_late_correction['eq']}[/white] (Length: {shortest_late_correction['len']})"
        )
        with open(os.path.join(out_dir, "shortest_late_correction.json"), "w") as f:
            json.dump(shortest_late_correction, f, indent=2)

    return shortest_failure, final_errors, late_corrections, shortest_late_correction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=10)
    parser.add_argument("--min_operands", type=int, default=2)
    parser.add_argument("--max_operands", type=int, default=5)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument(
        "--method",
        type=str,
        default="logit",
        choices=["tuned", "logit"],
        help="Lens method to use",
    )

    args = parser.parse_args()

    lens = TunedLens(args.ckpt, device=args.device)

    dm = AdditionDataModule(
        min_train_digits=args.min_digits,
        max_train_digits=args.max_digits,
        max_val_digits=args.max_digits,
        min_operands=args.min_operands,
        max_operands=args.max_operands,
        max_val_operands=args.max_operands,
        batch_size=args.batch_size,
        explicit_carry=getattr(lens.model.hparams, "explicit_carry", True),
    )
    dm.setup()

    val_loaders = dm.val_dataloader()
    loader = val_loaders[0] if isinstance(val_loaders, list) else val_loaders

    shortest, failures, corrections, shortest_corr = analyze_layers(
        lens,
        loader,
        max_batches=args.max_batches,
        device=args.device,
        out_dir=args.out_dir,
        method=args.method,
    )

    if args.visualize:
        if shortest_corr:
            print("\n" + "=" * 50)
            print(f"VISUALIZING SHORTEST LATE CORRECTION ({args.method}):")
            lens.visualize_trajectory(shortest_corr["eq"], method=args.method)
        elif shortest:
            print("\n" + "=" * 50)
            print(f"VISUALIZING SHORTEST FAILURE ({args.method}):")
            lens.visualize_trajectory(shortest["eq"], method=args.method)


if __name__ == "__main__":
    main()
