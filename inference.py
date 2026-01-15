import torch
import torch.nn.functional as F
import argparse
import random
import math
from model import GPTLightningModule


def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path} to {device}...")
    model = GPTLightningModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    model.freeze()
    return model


def encode(arr: list):
    return " ".join(f"{i:<2}" for i in arr)


def decode_tokens(tokens: list, itos: dict):
    return list(itos[i] for i in tokens)

def generate_correct_scratchpad(operands, max_len):
    # Replicate logic from data.py lines 92-110 using torch
    B = 1
    device = "cpu"

    # 2. Generate Op Digits with strict padding
    operands_digits = []
    for op in operands:
        # Pad to max_len (zeros at front/MSB)
        op_padded = op.zfill(max_len)
        d = [int(c) for c in op_padded]
        t = torch.tensor(d, dtype=torch.long, device=device).unsqueeze(
            0
        )  # (1, max_len)
        operands_digits.append(t)

    # 3. Compute Partial Sums
    scratchpad_segments = []
    current_sum = torch.zeros((B, max_len), dtype=torch.long, device=device)
    scratchpad_segments.append(current_sum.clone())  # S0 (MSB first, but all zeros)

    for operand in operands_digits:
        carry = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(max_len - 1, -1, -1):
            d_acc = current_sum[:, i]
            d_op = operand[:, i]
            total = d_acc + d_op + carry
            carry = total // 10
            current_sum[:, i] = total

        scratchpad_segments.append(
            current_sum.flip(1)
        )  # Append current sum (LSB first) for S1..Sn
        current_sum %= 10

    # Construct Token List for String Generation
    full_tokens = []

    # S0
    full_tokens.extend(scratchpad_segments[0].squeeze(0).tolist())

    for k in range(1, len(scratchpad_segments)):
        full_tokens.append(">")
        full_tokens.extend(scratchpad_segments[k].squeeze(0).tolist())

    full_tokens.append("#")

    return full_tokens


def inference(model, operands):
    device = model.device

    # Prepare vocab
    chars = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    plus_token = stoi["+"]
    eq_token = stoi["="]
    greater_token = stoi[">"]
    hash_token = stoi["#"]

    N = len(operands)
    max_digits = max(len(op) for op in operands)
    carry_allowance = math.ceil(math.log10(N))
    max_len = max_digits + carry_allowance
    print(f"Max digits: {max_digits}")
    print(f"Carry allowance: {carry_allowance}")
    print(f"Max length: {max_len}")

    # 1. Construct Prompt Tokens
    # Format: N1 + N2 + ... + Nk =
    tokens_list = []
    p1_list = []
    p2_list = []
    p3_list = []

    # offset = random.randint(0, offset_range)
    offset = 0

    for k, op_str in enumerate(operands):
        # Pad op_str to max_len with zeros
        padded_op = op_str.zfill(max_len)
        op_tokens = [stoi[c] for c in padded_op]

        L = len(op_tokens)
        tokens_list.extend(op_tokens)
        p1_list.extend([k + 1] * (L + 1))
        # PosID2: (L+1) down to 2
        p2_list.extend(range(L, -1, -1))
        p3_list.extend([1] * (L + 1))

        if k < N - 1:
            # Plus
            tokens_list.append(plus_token)
        else:
            # Equals
            tokens_list.append(eq_token)

    print("=== Prompt ===")
    print("Tokens: ", encode(decode_tokens(tokens_list, itos)))
    print("P1:     ", encode(p1_list))
    print("P2:     ", encode(p2_list))
    print("P3:     ", encode(p3_list))

    x = torch.tensor(tokens_list).unsqueeze(0).to(device)
    pos1 = torch.tensor(p1_list).unsqueeze(0).to(device)
    pos2 = torch.tensor(p2_list).unsqueeze(0).to(device)
    pos3 = torch.tensor(p3_list).unsqueeze(0).to(device)

    # 2. Generation Loop using model.generate
    # We generate until # or max_gen

    special_tokens = {
        ">": greater_token,
        "+": plus_token,
        "=": eq_token,
        "#": hash_token,
    }

    print(f"\nTask: {' + '.join(operands)}")
    print(f"Padded Max Length: {max_len}")



    # Run generation
    # model.generate returns (idx, pos1, pos2, pos3)
    # x input shape is (1, T)
    prompt_len = x.shape[1]

    correct_scratchpad = generate_correct_scratchpad(operands, max_len)
    print(f"Correct Scratchpad:   {encode(correct_scratchpad)}")

    # Safety limit for generation
    max_gen = (max_len + 1) * (N + 1) + 10

    x, pos1, pos2, pos3 = model.generate(
        x,
        pos1,
        pos2,
        pos3,
        max_new_tokens=max_gen,
        special_tokens=special_tokens,
        offset=offset,
    )

    # Extract generated portion
    # x is (1, T_total)
    generated_ids = x[0, prompt_len:].tolist()
    generated_tokens = decode_tokens(generated_ids, itos)
    generated_tokens_str = encode(generated_tokens)

    print(f"Generated Scratchpad: {generated_tokens_str}")

    # 3. Final Visualization
    print("\nToken-Position Alignment:")
    header = f"{'Token':<8} | {'P1':<4} | {'P2':<4} | {'P3':<4} | Segment"
    print(header)
    print("-" * len(header))

    # Combine everything for display
    # x contains full sequence IDs
    all_tokens = x[0].tolist()

    # pos tensors contain full sequence positions
    p_len = pos1.shape[1]
    # Ensure lengths match (generate ensures this)

    p1_all = pos1[0].tolist()
    p2_all = pos2[0].tolist()
    p3_all = pos3[0].tolist()

    all_tokens_correct = decode_tokens(all_tokens[:prompt_len], itos) + correct_scratchpad
    all_tokens_correct_str = encode(all_tokens_correct)
    all_tokens_str = encode(decode_tokens(all_tokens, itos))
    p1_all_str = encode(p1_all)
    p2_all_str = encode(p2_all)
    p3_all_str = encode(p3_all)
    print(f"Tokens:  {all_tokens_str}")
    print(f"Correct: {all_tokens_correct_str}")
    print(f"P1:      {p1_all_str}")
    print(f"P2:      {p2_all_str}")
    print(f"P3:      {p3_all_str}")

    # return
    # for i in range(len(all_tokens)):
    #     t_id = all_tokens[i]
    #     char = itos[t_id]
    #     p1 = p1_all[i]
    #     p2 = p2_all[i]
    #     p3 = p3_all[i]

    #     seg = "Input" if p3 == 1 else "Scratch" if p3 == 2 else "Result"
    #     print(f"{char:<8} | {p1:<4} | {p2:<4} | {p3:<4} | {seg}")

    # 4. Decode Result
    # Result is the LAST partial sum Sn (before #)
    sum_segments = []
    current_seg = []
    separators = ["+", "=", ">", "#"]

    for t in generated_tokens:
        if t in separators:
            if current_seg:
                sum_segments.append(current_seg)
            current_seg = []
        else:
            # Digit token (0-19)
            try:
                val = int(t) % 10
                current_seg.append(val)
            except ValueError:
                # Handle cases where t might be something else unexpectedly
                continue

    if current_seg:
        sum_segments.append(current_seg)

    if not sum_segments:
        print("❌ No sum segments found in scratchpad.")
        return

    final_digits_lsb = sum_segments[-1]
    # Convert LSB to number
    final_val = 0
    for i, d in enumerate(final_digits_lsb):
        final_val += d * (10**i)

    correct_val = sum(int(op) for op in operands)

    print(f"Final Decoded Sum: {final_val}")
    print(f"Correct Sum:       {correct_val}")

    if final_val == correct_val:
        print("✅ Correct!")
    else:
        print("❌ Incorrect.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--operands",
        type=str,
        default="123,456",
        help="Comma-separated list of operands",
    )
    args = parser.parse_args()

    # Device handling
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.ckpt_path, device)

    operands = args.operands.split(",")
    inference(model, operands)


if __name__ == "__main__":
    main()
