from PIL.ImageChops import offset
import torch
import torch.nn.functional as F
import argparse
import random
import math
from model import GPTLightningModule
from data import construct_addition_batch


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


def inference(model, operands, offset=0):
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

    # 1. Use construct_addition_batch to get ground truth
    # We need to construct operands_digits as list of tensors (1, max_len)
    operands_digits = []
    for op in operands:
        op_padded = op.zfill(max_len)
        d = [int(c) for c in op_padded]
        t = torch.tensor(d, dtype=torch.long).unsqueeze(0)  # (1, max_len)
        operands_digits.append(t)

    gt_full_seq, gt_pos1, gt_pos2, gt_pos3 = construct_addition_batch(
        operands_digits,
        stoi,
        random_offsets=False,
        offset_range=0,
        explicit_carry=getattr(model.hparams, "explicit_carry", True),
    )

    if offset > 0:
        gt_pos1 = gt_pos1 + offset
        gt_pos2 = gt_pos2 + offset

    # Extract Prompt parts
    # The prompt is the sequence up to the equals sign.
    # But wait, construct_addition_batch returns the FULL sequence (Input + Scratchpad).
    # We need to find where the input ends.
    # The input ends at the equals sign token.
    # Let's find the index of the last equals sign?
    # Actually, construct_addition_batch constructs: N1 + N2 ... + Nk = (S0 ... #)
    # So we can just find the index of '='.

    # We can just iterate to find '='
    # full_seq is (1, L)
    full_seq_list = gt_full_seq[0].tolist()
    try:
        eq_idx = full_seq_list.index(eq_token)
    except ValueError:
        print("Error: Could not find '=' in constructed sequence")
        return

    # Prompt is everything up to and including '='
    prompt_tokens = gt_full_seq[:, : eq_idx + 1]
    prompt_pos1 = gt_pos1[:, : eq_idx + 1]
    prompt_pos2 = gt_pos2[:, : eq_idx + 1]
    prompt_pos3 = gt_pos3[:, : eq_idx + 1]

    # Correct scratchpad is everything AFTER '='
    correct_scratchpad_tokens = gt_full_seq[:, eq_idx + 1 :]

    x = prompt_tokens.to(device)
    pos1 = prompt_pos1.to(device)
    pos2 = prompt_pos2.to(device)
    pos3 = prompt_pos3.to(device)

    print("=== Prompt ===")
    print("Tokens: ", encode(decode_tokens(x[0].tolist(), itos)))
    print("P1:     ", encode(pos1[0].tolist()))
    print("P2:     ", encode(pos2[0].tolist()))
    print("P3:     ", encode(pos3[0].tolist()))

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

    correct_scratchpad = correct_scratchpad_tokens[0].tolist()
    print(f"Correct Scratchpad:   {encode(decode_tokens(correct_scratchpad, itos))}")

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

    # Combine everything for display
    all_tokens = x[0].tolist()

    p1_all = pos1[0].tolist()
    p2_all = pos2[0].tolist()
    p3_all = pos3[0].tolist()

    all_tokens_correct = decode_tokens(
        all_tokens[:prompt_len] + correct_scratchpad, itos
    )
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
    parser.add_argument(
        "--offset", type=int, default=0, help="Positional offset for testing"
    )
    args = parser.parse_args()

    # Device handling
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.ckpt_path, device)

    operands = args.operands.split(",")
    inference(model, operands, offset=args.offset)


if __name__ == "__main__":
    main()
