import torch
from data import AdditionDataModule
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--dataset_type", type=str, default="1dCoT", 
                       choices=["2dPE", "1dCoT"], help="Dataset type: standard or cot")
    parser.add_argument("--separator", type=str, default="|")
    parser.add_argument("--seed", type=int, default=200)
    args = parser.parse_args()

    print(f"--- Data Inspection (Dataset: {args.dataset_type.upper()}, L={args.min_digits}-{args.max_digits}) ---")

    # Initialize DataModule
    cot_params = {   # additional params for 1dCoT
        'separator': args.separator,
        'use_padding': True
    } if args.dataset_type == "1dCoT" else None

    dm = AdditionDataModule(
        min_train_digits=args.min_digits,
        max_train_digits=args.max_digits,
        batch_size=args.batch_size,
        curriculum_start=args.max_digits,  # Force max digits immediately for inspection
        dataset_type=args.dataset_type,
        cot_params=cot_params,
        seed=args.seed,
    )
    dm.setup()

    # Vocabulary for decoding
    vocab_info = dm.get_vocab_info()
    itos = vocab_info['itos']
    dataset_type = vocab_info['dataset_type']
    def decode(indices):
        return "".join([itos[i.item()] for i in indices])

    # We need to manually set the max_digits if we want to test a specific range
    # effectively bypassing the curriculum start if needed, but here we set curriculum_start = max_digits

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    if dataset_type == "1dCoT":
        # CoT format: (x, y_input, y_target, x_roles, y_roles)
        x, y_input, y_target, x_roles, y_roles = batch
        
        print(f"\nBatch Shapes (CoT format):")
        print(f"x (input): {x.shape}")
        print(f"y_input (CoT input): {y_input.shape}")
        print(f"y_target (CoT target): {y_target.shape}")
        print(f"x_roles: {x_roles.shape}")
        print(f"y_roles: {y_roles.shape}")
        
        
        for i in range(min(args.n_samples, x.shape[0])):
            print("-" * 60)
            print(f"Sample {i+1}:")
            # Decode sequences
            seq_x = decode(x[i])
            seq_y_input = decode(y_input[i])
            seq_y_target = decode(y_target[i])
            
            # Parse the problem
            if '+' in seq_x and '=' in seq_x:
                n1_part, rest = seq_x.split('+', 1)
                n2_part = rest.split('=', 1)[0]
                try:
                    n1 = int(n1_part) if n1_part else 0
                    n2 = int(n2_part) if n2_part else 0
                    actual_sum = n1 + n2
                    print(f"Problem: {n1} + {n2} = {actual_sum}")
                except:
                    print(f"Problem: {seq_x}")
            
            print(f"\nx (Input problem):  {seq_x}")
            print(f"y_input (CoT start): {seq_y_input}")
            print(f"y_target (CoT cont): {seq_y_target}")

            # Positional Encodings
            role_names = {
                0: "N1_digit",
                1: "Operator",
                2: "N2_digit", 
                3: "Step_separator",
                4: "CoT_digit",
                5: "Final_digit"
            }

            tokens_x = [itos[idx.item()] for idx in x[i]]
            pos_x = x_roles[i].tolist()
            tokens_y = [itos[idx.item()] for idx in y_input[i]]
            pos_y = y_roles[i].tolist()
            print(f"{'Token':<6} | {'Pos: Role':<12}")
            for t, px in zip(tokens_x, pos_x):
                role_name = role_names.get(px, f"Unknown_{px}")
                print(f"{t:<6} | {px:<8} | {role_name:<12}")
            for t, py in zip(tokens_y, pos_y):
                role_name = role_names.get(py, f"Unknown_{py}")
                print(f"{t:<6} | {py:<8} | {role_name:<12}")

            # Extract final result from CoT
            cot_parts = seq_y_target.split(args.separator)
            if len(cot_parts) >= 2:
                final_result = cot_parts[-2]  # Last part except the ending separator
                try:
                    cot_final = int(final_result) if final_result else 0
                    print(f"\nCoT Final Result: {cot_final}")
                except:
                    print(f"\nCoT Final Result (raw): '{final_result}'")

    if dataset_type == "2dPE":
        x, y, p1, p2 = batch

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
