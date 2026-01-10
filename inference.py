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


def inference(model, operands, offset_range=10):
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
    carry_allowance = math.ceil(math.log10(model.hparams.get("max_operands", N)))
    max_len = max_digits + carry_allowance

    # 1. Construct Prompt Tokens
    # Format: N1 + N2 + ... + Nk =
    tokens_list = []
    p1_list = []
    p2_list = []
    p3_list = []

    offset = random.randint(0, offset_range)

    for k, op_str in enumerate(operands):
        # Pad op_str to max_len with zeros
        padded_op = op_str.zfill(max_len)
        op_tokens = [stoi[c] for c in padded_op]

        L = len(op_tokens)
        tokens_list.extend(op_tokens)
        p1_list.extend([k + 1] * L)
        # PosID2: (L+1) down to 2
        ids = list(range(L + 1, 1, -1))
        p2_list.extend([i + offset for i in ids])
        p3_list.extend([1] * L)

        if k < N - 1:
            # Plus
            tokens_list.append(plus_token)
            p1_list.append(k + 1)
            p2_list.append(1 + offset)
            p3_list.append(1)
        else:
            # Equals
            tokens_list.append(eq_token)
            p1_list.append(k + 1)
            p2_list.append(1 + offset)
            p3_list.append(1)

    x = torch.tensor(tokens_list).unsqueeze(0).to(device)
    pos1 = torch.tensor(p1_list).unsqueeze(0).to(device)
    pos2 = torch.tensor(p2_list).unsqueeze(0).to(device)
    pos3 = torch.tensor(p3_list).unsqueeze(0).to(device)

    # 2. Generation Loop
    # We generate until #
    generated_tokens = []

    # State tracking for positional IDs during generation
    curr_block = 0  # S0 starts block 1
    curr_p2 = 1

    max_gen = (max_len + 1) * (N + 1) + 10  # Safety limit

    print(f"\nTask: {' + '.join(operands)}")
    print(f"Padded Max Length: {max_len}")

    for _ in range(max_gen):
        with torch.no_grad():
            # Support both old (2-pos) and new (3-pos) forward calls if needed,
            # but we assume the new version.
            logits, _ = model(x, pos1, pos2, pos3_ids=pos3)

        # Predict next token
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=0)
        _, next_token_id = torch.max(probs, dim=0)

        next_id = next_token_id.item()
        next_char = itos[next_id]
        generated_tokens.append(next_char)

        if next_id == hash_token:
            break

        # Update inputs for next step
        x = torch.cat([x, next_token_id.view(1, 1)], dim=1)

        # update positional ids for next step
        # Logic:
        # If we just generated next_id, what are ITS pos IDs?
        # Actually x we just updated ALREADY includes next_id.
        # But for the NEXT iteration, we need the PosIDs of the token at x[0, -1].

        # Wait, the `model` call uses current x, pos1, pos2, pos3.
        # So we need to append the PosIDs for the token we just added to x.

        if next_id == greater_token or next_id == plus_token:
            # Separator > or +
            pos1 = torch.cat(
                [pos1, torch.tensor([[curr_block + 1]], device=device)], dim=1
            )
            pos2 = torch.cat(
                [pos2, torch.tensor([[max_len + 1 + offset]], device=device)], dim=1
            )
            pos3 = torch.cat([pos3, torch.tensor([[2]], device=device)], dim=1)

            # Prepare for next partial sum block
            curr_block += 1
            curr_p2 = 1
        else:
            # Digit in Scratchpad or Result
            pos1 = torch.cat(
                [pos1, torch.tensor([[curr_block + 1]], device=device)], dim=1
            )
            pos2 = torch.cat(
                [pos2, torch.tensor([[curr_p2 + offset]], device=device)], dim=1
            )
            pos3 = torch.cat([pos3, torch.tensor([[2]], device=device)], dim=1)
            curr_p2 += 1

    full_gen_str = "".join(generated_tokens)
    print(f"Generated Scratchpad: {full_gen_str}")

    # 3. Final Visualization
    print("\nToken-Position Alignment:")
    header = f"{'Token':<8} | {'P1':<4} | {'P2':<4} | {'P3':<4} | Segment"
    print(header)
    print("-" * len(header))

    # Combine everything for display
    all_tokens = tokens_list + [stoi[t] for t in generated_tokens]
    p1_all = pos1.squeeze().tolist()
    p2_all = pos2.squeeze().tolist()
    p3_all = pos3.squeeze().tolist()

    for i in range(len(all_tokens)):
        t_id = all_tokens[i]
        char = itos[t_id]
        p1 = p1_all[i]
        p2 = p2_all[i]
        p3 = p3_all[i]

        seg = "Input" if p3 == 1 else "Scratch" if p3 == 2 else "Result"
        print(f"{char:<8} | {p1:<4} | {p2:<4} | {p3:<4} | {seg}")

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
    print(f"Correct Sum:      {correct_val}")

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
