import torch
import torch.nn.functional as F
import argparse
import random
from model import GPTLightningModule


def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path} to {device}...")
    model = GPTLightningModule.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    model.freeze()
    return model


def inference(model, n1_str, n2_str, offset_range=10):
    device = model.device

    # Prepare vocab
    chars = [str(i) for i in range(20)] + ["+", "="]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Training assumes max(len(n1), len(n2)) roughly.
    L = max(len(n1_str), len(n2_str))

    # Construct Tokens
    # Format: n1... + n2... =
    n1_str = f"{n1_str:0>{L}}"
    n2_str = f"{n2_str:0>{L}}"
    prompt_str = f"{n1_str}+{n2_str}="
    tokens = [stoi[c] for c in prompt_str]
    x = torch.tensor(tokens).unsqueeze(0).to(device)  # (1, T)

    # Construct Pos1 (Roles)
    # n1 digits -> 1
    # + -> 1
    # n2 digits -> 2
    # = -> 2
    p1_n1 = [1] * len(n1_str)
    p1_plus = [1]
    p1_n2 = [2] * len(n2_str)
    p1_eq = [2]
    p1_seq = p1_n1 + p1_plus + p1_n2 + p1_eq
    pos1 = torch.tensor(p1_seq).unsqueeze(0).to(device)

    # Construct Pos2 (Significance/Column) with Random Offset
    # n1: 1..len(n1)
    # +: 1
    # n2: 1..len(n2)
    # =: 1
    offset = random.randint(0, offset_range)

    p2_n1 = [i + 1 + offset for i in range(len(n1_str))][
        ::-1
    ]  # 1 to L -> Reversed to L to 1
    p2_plus = [0 + offset]
    p2_n2 = [i + 1 + offset for i in range(len(n2_str))][
        ::-1
    ]  # 1 to L -> Reversed to L to 1
    p2_eq = [0 + offset]
    p2_seq = p2_n1 + p2_plus + p2_n2 + p2_eq
    pos2 = torch.tensor(p2_seq).unsqueeze(0).to(device)

    # Generation Loop
    # We generate L+1 digits (including carry)
    # pos2 for result goes: L + offset, L-1 + offset, ..., 0 + offset

    result_digits = []

    print(f"\nTask: {n1_str} + {n2_str}")
    print(f"Length: {L}")
    print(f"Input Tokens: {prompt_str}")
    print(f"Position 1:   {pos1}")
    print(f"Position 2:   {pos2}")

    # Next Pos2 values for generation
    # From L down to 0
    # Next Pos2 values for generation
    # From 1 up to L+1 (LSB .. MSB)
    next_pos2_vals = [i + offset for i in range(1, L + 2)]

    for step, next_p2 in enumerate(next_pos2_vals):
        # Get logits
        with torch.no_grad():
            logits, _ = model(x, pos1, pos2)

        # Predict next token from last position
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=0)
        _, next_token_id = torch.max(probs, dim=0)

        # Decode
        next_token = itos[next_token_id.item()]

        # Store
        result_digits.append(next_token)

        # Update inputs for next step
        # x -> append next_token_id
        x = torch.cat([x, next_token_id.view(1, 1)], dim=1)

        # pos1 -> append 3 (Sum role)
        pos1 = torch.cat([pos1, torch.tensor([[3]], device=device)], dim=1)

        # pos2 -> append next_p2 (which corresponds to the digit we just generated)
        # Wait, the p2 we append is for the token we just added.
        # That token is `sum[current]`.
        # Its p2 is `next_p2`.
        pos2 = torch.cat([pos2, torch.tensor([[next_p2]], device=device)], dim=1)

    print(f"Result (Reversed): {''.join(result_digits)}")

    # Reverse back to normal number
    # result_digits contains raw tokens as strings (from itos)
    # Convert them to int % 10 then back to string for joining
    final_res_str = "".join([str(int(d) % 10) for d in result_digits])
    final_res = int(final_res_str[::-1])
    correct_res = int(n1_str) + int(n2_str)
    print(f"Final Answer:   {final_res}")
    print(f"Correct Answer: {correct_res}")
    if final_res == correct_res:
        print("✅ Correct!")
    else:
        print("❌ Incorrect.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n1", type=str, default="123")
    parser.add_argument("--n2", type=str, default="456")
    args = parser.parse_args()

    # Device handling
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.ckpt_path, device)
    inference(model, args.n1, args.n2)


if __name__ == "__main__":
    main()
