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


def inference(model, n1_str, n2_str, offset_range=10, val_k=1):
    device = model.device

    # Prepare vocab
    chars = "0123456789+=#"
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Training assumes max(len(n1), len(n2)) roughly.
    L = max(len(n1_str), len(n2_str))

    # Construct Tokens
    # Format: n1... + n2... =
    n1_str_pad = f"{n1_str:0>{L}}"
    n2_str_pad = f"{n2_str:0>{L}}"
    prompt_str = f"{n1_str_pad}+{n2_str_pad}="
    tokens = [stoi[c] for c in prompt_str]
    x = torch.tensor(tokens).unsqueeze(0).to(device)  # (1, T)

    # Construct Pos1 (Roles)
    p1_seq = ([1] * (L + 1)) + ([2] * (L + 1))
    pos1 = torch.tensor(p1_seq).unsqueeze(0).to(device)

    # Construct Pos2 (Significance/Column) with Random Offset
    offset = random.randint(0, offset_range)
    offset = 0
    idx_1_L = torch.arange(L, -1, -1)
    p2_seq = torch.cat([idx_1_L, idx_1_L]) + offset
    pos2 = p2_seq.unsqueeze(0).to(device)

    print(f"\nTask: {n1_str} + {n2_str}")
    print(f"Length: {L}")
    print(f"Input Tokens: {prompt_str}")

    # Repeat for K samples
    x_k = x.repeat(val_k, 1)
    p1_k = pos1.repeat(val_k, 1)
    p2_k = pos2.repeat(val_k, 1)

    # Generate
    model.eval()
    with torch.no_grad():
        # generated_k = model.generate(x_k, p1_k, p2_k, max_new_tokens=L + 1)
        generated_k = model.generate(x_k, p1_k, p2_k, max_new_tokens=L + 1)

    # Extract results
    results_sum_k = generated_k[:, len(tokens) :]  # (K, sum_len)

    # 1. Standard Prediction (First Sample)
    first_pred_tokens = results_sum_k[0].tolist()
    first_pred_str = "".join([itos[t] for t in first_pred_tokens]).split("#")[0]

    # 2. Sequence Majority Vote
    seqs = [tuple(s.tolist()) for s in results_sum_k]
    most_common_seq = max(set(seqs), key=seqs.count)
    seq_mv_str = "".join([itos[t] for t in most_common_seq]).split("#")[0]

    # 3. Digit Majority Vote
    # Move to CPU because torch.mode is not implemented on MPS
    digit_mv_tokens = torch.mode(results_sum_k.cpu(), dim=0).values.tolist()
    digit_mv_str = "".join([itos[t] for t in digit_mv_tokens]).split("#")[0]

    print(f"Result (First Sample): {first_pred_str}")
    if val_k > 1:
        print(f"Result (Seq MV):      {seq_mv_str}")
        print(f"Result (Digit MV):    {digit_mv_str}")

    # Final logic (using Seq MV as primary)
    try:
        final_res = int(seq_mv_str[::-1]) if seq_mv_str else 0
    except ValueError:
        final_res = -1

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
    parser.add_argument(
        "--val_k", type=int, default=1, help="Number of samples for majority voting."
    )
    parser.add_argument(
        "--offset", type=int, default=10, help="Random offset range for pos2."
    )
    args = parser.parse_args()

    # Device handling
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.ckpt_path, device)
    inference(model, args.n1, args.n2, offset_range=args.offset, val_k=args.val_k)


if __name__ == "__main__":
    main()
