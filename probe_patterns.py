import torch
import torch.nn.functional as F
import argparse
import json
from rich.console import Console
from rich.table import Table

from tuned_lens_addition import TunedLens
from data import construct_addition_batch


def run_probe(lens, queries, device="cpu", method="logit"):
    """
    Runs a list of (equation_string, target_val) pairs and analyzes transitions.
    """
    results = []

    for eq in queries:
        # Step 1: Parse equation string into list of strings
        operand_strs = [op.strip() for op in eq.split("+") if op.strip()]

        # Step 2: Convert to list of tensors [ (B, L), ... ]
        max_op_len = max(len(s) for s in operand_strs)

        operands_digits = []
        for s in operand_strs:
            s_padded = s.zfill(max_op_len)
            digits = [int(d) for d in s_padded]
            operands_digits.append(torch.tensor([digits], dtype=torch.long))

        # Step 3: Construct batch
        full_seq, pos1, pos2, pos3 = construct_addition_batch(
            operands_digits, lens.stoi, random_offsets=False
        )

        idx = full_seq[:, :-1].to(device)
        targets = full_seq[:, 1:].to(device)
        p1 = pos1[:, :-1].to(device)
        p2 = pos2[:, :-1].to(device)
        p3 = pos3[:, :-1].to(device)

        # Step 4: Get hidden states
        h_list, _ = lens.get_hidden_states(idx, p1, p2, p3)

        predict_mask = (
            (p3[0] >= 2)
            & (targets[0] != lens.pad_token)
            & (targets[0] != lens.hash_token)
        )

        indices = torch.where(predict_mask)[0]

        eq_results = []
        for step_idx in indices:
            target_token = targets[0, step_idx].item()
            l_preds = []
            for l in range(lens.n_layer + 1):
                h_l = h_list[l][:, step_idx]
                if method == "tuned" and lens.is_trained:
                    h_hat = lens.translators[l](h_l)
                    logits = lens.model.lm_head(lens.model.ln_f(h_hat))
                else:
                    logits = lens.model.lm_head(lens.model.ln_f(h_l))
                pred = torch.argmax(logits, dim=-1).item()
                l_preds.append(pred)

            eq_results.append(
                {
                    "pos": step_idx.item(),
                    "target": lens.itos[target_token],
                    "layers": [lens.itos[p] for p in l_preds],
                    "target_num": target_token,  # integer representation
                }
            )

        results.append({"eq": eq, "steps": eq_results})

    return results


def analyze_mistakes(results, title, console, method):
    mistake_map = {}
    total_steps = 0
    correct_final = 0

    for res in results:
        for step in res["steps"]:
            total_steps += 1
            l0 = step["layers"][1]
            l_final = step["layers"][-1]
            target = step["target"]
            if l_final == target:
                correct_final += 1

            if l0 != target:
                key = (l0, target, res["eq"])
                mistake_map[key] = mistake_map.get(key, 0) + 1

    table = Table(title=f"{title} - {method}", show_lines=True)
    table.add_column("Top Examples", style="cyan")
    table.add_column("L0 Pred", style="red")
    table.add_column("Target", style="green")
    table.add_column("Correction", style="bold yellow")

    pattern_stats = {}
    for (l0, target, eq), count in mistake_map.items():
        pattern_stats[(l0, target)] = pattern_stats.get((l0, target), []) + [eq]

    for (l0, target), eqs in sorted(
        pattern_stats.items(), key=lambda x: len(x[1]), reverse=True
    ):
        # Determine if it's a late correction (L0 wrong, L_final right)
        # We'll just assume most are corrections if the mistake map is non-empty but overall acc is high
        table.add_row(
            ", ".join(eqs[:4]) + ("..." if len(eqs) > 4 else ""),
            str(l0),
            str(target),
            "YES",
        )

    console.print(table)
    acc = correct_final / total_steps if total_steps > 0 else 0
    console.print(f"Final Model Accuracy on this set: {acc:.1%}")

    # Copying Bias Check
    copy_hits = 0
    for (l0, target), eqs in pattern_stats.items():
        for eq in eqs:
            operands = [op.strip() for op in eq.split("+") if op.strip()]
            ops_clean = [str(int(op)) for op in operands]
            if l0 in operands or l0 in ops_clean:
                copy_hits += 1

    if total_steps > 0 and len(mistake_map) > 0:
        console.print(
            f"Copying Bias: {copy_hits / len(mistake_map):.1%} of L0 unique mistakes match an input digit."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--method", type=str, default="logit", choices=["tuned", "logit"]
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    lens = TunedLens(args.ckpt, device=args.device)
    console = Console()

    # 1. Homogenous Grid (X+X, XX+XX, XXX+XXX)
    homo_queries = []
    for x in range(1, 10):
        homo_queries.append(f"{x}+{x}")
        homo_queries.append(f"{x}{x}+{x}{x}")
        homo_queries.append(f"{x}{x}{x}+{x}{x}{x}")

    console.print(
        f"\n[bold magenta]Test A: Homogeneity Scaling (X+X to XXX+XXX)[/bold magenta]"
    )
    results_a = run_probe(lens, homo_queries, device=args.device, method=args.method)
    analyze_mistakes(results_a, "Homogeneity Mistake Patterns", console, args.method)

    # 2. Long Carry Chain (99...9 + 1)
    carry_queries = []
    for length in range(2, 11):
        nines = "9" * length
        carry_queries.append(f"{nines}+1")

    console.print(
        f"\n[bold magenta]Test B: Long Carry Chains (99...9 + 1)[/bold magenta]"
    )
    results_b = run_probe(lens, carry_queries, device=args.device, method=args.method)
    analyze_mistakes(results_b, "Carry Chain Mistake Patterns", console, args.method)


if __name__ == "__main__":
    main()
