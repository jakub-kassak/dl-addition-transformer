import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description="Plot validation accuracy heatmap.")
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to val_results.csv"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="heatmap.png",
        help="Path to save the heatmap",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_acc_seq",
        choices=["val_acc_seq", "val_acc_token", "val_loss"],
        help="Metric to plot",
    )

    # Training range parameters
    parser.add_argument("--min_train_digits", type=int, default=1)
    parser.add_argument("--max_train_digits", type=int, default=7)
    parser.add_argument("--min_train_operands", type=int, default=2)
    parser.add_argument("--max_train_operands", type=int, default=5)

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File {args.csv_path} not found.")
        return

    # Load data
    data = []
    with open(args.csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "L": int(row["L"]),
                    "N": int(row["N"]),
                    "val_acc_seq": float(row["val_acc_seq"]),
                    "val_acc_token": float(row["val_acc_token"]),
                    "val_loss": float(row["val_loss"]),
                }
            )

    if not data:
        print("No data found in CSV.")
        return

    # Filter for the latest step
    latest_step = max(d["step"] for d in data)
    latest_data = [d for d in data if d["step"] == latest_step]

    # Get unique L and N values
    all_ls = sorted(list(set(d["L"] for d in data)))
    all_ns = sorted(list(set(d["N"] for d in data)))

    # Create grid
    grid = np.zeros((len(all_ls), len(all_ns)))
    grid[:] = np.nan

    l_to_idx = {l: i for i, l in enumerate(all_ls)}
    n_to_idx = {n: i for i, n in enumerate(all_ns)}

    for d in latest_data:
        l_idx = l_to_idx[d["L"]]
        n_idx = n_to_idx[d["N"]]
        grid[l_idx, n_idx] = d[args.metric]

    # Plot
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        grid,
        annot=True,
        fmt=".2f" if args.metric != "val_loss" else ".4f",
        xticklabels=all_ns,
        yticklabels=all_ls,
        cmap="viridis",
        cbar_kws={"label": args.metric},
        vmin=0,
        vmax=1,
    )
    plt.xlabel("Number of Operands (N)")
    plt.ylabel("Number of Digits (L)")
    plt.title(f"{args.metric} Heatmap (Step {latest_step})")

    # Invert y-axis to have L increasing upwards if desired,
    # but standard heatmap has (0,0) at top left.
    # Let's keep default but ensure labels are correct.

    # Mark training area
    # Find indices for the training range
    # Note: if training range is outside validation range, highlight might be partial or empty

    train_l_start = None
    train_l_end = None
    train_n_start = None
    train_n_end = None

    for i, l in enumerate(all_ls):
        if l >= args.min_train_digits and train_l_start is None:
            train_l_start = i
        if l <= args.max_train_digits:
            train_l_end = i

    for i, n in enumerate(all_ns):
        if n >= args.min_train_operands and train_n_start is None:
            train_n_start = i
        if n <= args.max_train_operands:
            train_n_end = i

    if all(
        v is not None for v in [train_l_start, train_l_end, train_n_start, train_n_end]
    ):
        # Rectangle(xy, width, height)
        # xy is top-left in heatmap coordinate? No, (n_idx, l_idx) for data.
        # But sns.heatmap plots labels. train_n_start is the index on x-axis.
        # train_l_start is the index on y-axis.

        rect = Rectangle(
            (train_n_start, train_l_start),
            train_n_end - train_n_start + 1,
            train_l_end - train_l_start + 1,
            fill=False,
            edgecolor="red",
            lw=4,
            label="Training Range",
        )
        ax.add_patch(rect)
        plt.legend(handles=[rect], labels=["Training Range"], loc="upper right")

    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Heatmap saved to {args.output_path}")


if __name__ == "__main__":
    main()
