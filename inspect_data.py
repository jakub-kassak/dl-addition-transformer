import torch
from data import AdditionDataModule
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=200)
    args = parser.parse_args()

    print(f"--- Data Inspection (L={args.min_digits}-{args.max_digits}) ---")

    # Initialize DataModule
    dm = AdditionDataModule(
        min_train_digits=args.min_digits,
        max_train_digits=args.max_digits,
        batch_size=args.batch_size,
        curriculum_start=args.max_digits,  # Force max digits immediately for inspection
        seed=args.seed,
    )
    dm.setup()

    # We need to manually set the max_digits if we want to test a specific range
    # effectively bypassing the curriculum start if needed, but here we set curriculum_start = max_digits

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    x, y, p1, p2 = batch

    # Vocabulary for decoding
    itos = dm.itos

    def decode(indices):
        return "".join([itos[i.item()] for i in indices])

    print(f"\nBatch Shapes:")
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
    print(f"p1: {p1.shape}")
    print(f"p2: {p2.shape}")

    for i in range(min(args.n_samples, x.shape[0])):
        print(f"\nSample {i+1}:")

        # Decode Equation
        seq_x = decode(x[i])
        seq_y = decode(y[i])

        print(f"x (Input):  {seq_x}")
        print(f"y (Target): {seq_y}")

        # Positional Encodings
        # We'll align them with x for visualization
        tokens_x = [itos[idx.item()] for idx in x[i]]
        pos1_vals = p1[i].tolist()
        pos2_vals = p2[i].tolist()

        print("-" * 60)
        print(f"{'Token':<6} | {'Pos1 (Role)':<12} | {'Pos2 (Digit+Offset)':<20}")
        print("-" * 60)

        for t, p1_v, p2_v in zip(tokens_x, pos1_vals, pos2_vals):
            role = "N1" if p1_v == 1 else "N2" if p1_v == 2 else "Sum"
            print(f"{t:<6} | {p1_v:<4} ({role:<5}) | {p2_v:<20}")

    print("\n✅ Inspection Complete")


if __name__ == "__main__":
    main()
