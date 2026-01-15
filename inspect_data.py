import torch
from data import AdditionDataModule
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--min_operands", type=int, default=2)
    parser.add_argument("--max_operands", type=int, default=3)
    parser.add_argument("--max_val_operands", type=int, default=5)
    parser.add_argument("--val_operand_step", type=int, default=2)
    parser.add_argument("--val_step", type=int, default=3)
    parser.add_argument("--data_mode", type=str, default="variable")
    parser.add_argument("--random_offsets", type=bool, default=False)
    parser.add_argument("--data_type", type=str, default="digit_combinations")
    args = parser.parse_args()

    print(
        f"--- Data Inspection (L={args.min_digits}-{args.max_digits}, N={args.min_operands}-{args.max_operands}, mode={args.data_mode}) ---"
    )

    # Initialize DataModule
    dm = AdditionDataModule(
        min_train_digits=args.min_digits,
        max_train_digits=args.max_digits,
        batch_size=args.batch_size,
        min_operands=args.min_operands,
        max_operands=args.max_operands,
        data_mode=args.data_mode,
        random_offsets=args.random_offsets,  # Explicitly enable for inspection
        data_type=args.data_type,
    )
    dm.setup()

    # Need to call val_dataloader to populate val_names
    _ = dm.val_dataloader()

    loader = dm.train_dataloader()
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("No data in loader!")
        return

    x, y, p1, p2, p3 = batch

    # Vocabulary for decoding
    itos = dm.itos

    def decode(indices):
        # print(indices)
        return   " ".join([f'{itos[i.item()]:>2}' for i in indices])

    print(f"\nBatch Shapes:")
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
    print(f"p1: {p1.shape}")
    print(f"p2: {p2.shape}")
    print(f"p3: {p3.shape}")

    for i in range(min(args.n_samples, x.shape[0])):
        print(f"\nSample {i}:")

        # Decode Equation
        seq_x = decode(x[i])

        # Helper to show targets (mask padding)
        # We don't have mask here easily unless we replicate logic.
        # Just show raw y
        seq_y = decode(y[i])

        print(f"y (Target):  {seq_y}")
        print(f"x  (Input):  {seq_x}")
        print(f"p1 (Block):  {' '.join(f'{p:>2}' for p in p1[i].tolist())}")
        print(f"p2 (Digit):  {' '.join(f'{p:>2}' for p in p2[i].tolist())}")
        print(f"p3  (Type):  {' '.join(f'{p:>2}' for p in p3[i].tolist())}")

        # Positional Encodings
        tokens_x = [itos[idx.item()] for idx in x[i]]
        pos1_vals = p1[i].tolist()
        pos2_vals = p2[i].tolist()
        pos3_vals = p3[i].tolist()

        if True:
            print("-" * 80)
            print(
                f"{'Token':<6} | {'Pos1 (Block)':<12} | {'Pos2 (Digit+Offset)':<20} | {'Pos3 (Type)':<12}"
            )
            print("-" * 80)

            for t, p1_v, p2_v, p3_v in zip(tokens_x, pos1_vals, pos2_vals, pos3_vals):
                # map p3 to description
                type_desc = "Input" if p3_v == 1 else "Scratch" if p3_v == 2 else "Result"
                print(f"{t:<6} | {p1_v:<12} | {p2_v:<20} | {p3_v:<4} ({type_desc})")

    print("\n✅ Inspection Complete")


if __name__ == "__main__":
    main()
