import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class AttentionExplorer:
    def __init__(self, model):
        """
        Initialize the explorer with a trained GPTLightningModule.
        """
        self.model = model
        self.model.eval()

        # Token mapping based on data.py
        # Token mapping based on data.py (22 tokens: 0-19, +, =)
        # Token mapping based on data.py (22 tokens: 0-19, +, =, >, #)
        self.chars = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token = -1
        self.plus_token = self.stoi["+"]
        self.eq_token = self.stoi["="]
        self.greater_token = self.stoi[">"]
        self.hash_token = self.stoi["#"]

    def encode_equation(self, equation_string):
        """
        Encodes an equation string into tokens and positional IDs (P1, P2, P3).
        Mimics data.py generate_batch logic for specific operands.
        """
        # Parse operands from equation string
        if "=" in equation_string:
            lhs, _ = equation_string.split("=")  # Ignore RHS, we calculate it
        else:
            lhs = equation_string

        operands = lhs.split("+")
        N_Ops = len(operands)

        # Determine max_digits and length
        max_digits = max(len(op) for op in operands)
        carry_allowance = 1  # math.ceil(math.log10(N_Ops)) if N_Ops > 1 else 0 # Simplified for visualizer
        if N_Ops > 1:
            import math

            carry_allowance = math.ceil(math.log10(N_Ops))

        max_len = max_digits + carry_allowance

        B = 1

        # 2. Generate Op Digits with strict padding
        operands_digits = []
        for op in operands:
            # Pad to max_len (zeros at front/MSB)
            op_padded = op.zfill(max_len)
            d = [int(c) for c in op_padded]
            t = torch.tensor(d, dtype=torch.long).unsqueeze(0)  # (1, max_len)
            operands_digits.append(t)

        # 3. Compute Partial Sums
        scratchpad_segments = []
        current_sum = torch.zeros((B, max_len), dtype=torch.long)
        scratchpad_segments.append(current_sum.clone())  # S0

        for operand in operands_digits:
            carry = torch.zeros(B, dtype=torch.long)
            for i in range(max_len - 1, -1, -1):
                d_acc = current_sum[:, i]
                d_op = operand[:, i]
                total = d_acc + d_op + carry
                carry = total // 10
                current_sum[:, i] = total

            scratchpad_segments.append(current_sum.flip(1))  # LSB first
            current_sum %= 10

        # 4. Construct Full Sequence
        p1_list, p2_list, p3_list, tokens_list = [], [], [], []

        plus = torch.full((B, 1), self.plus_token, dtype=torch.long)
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long)
        # greater = torch.full((B, 1), self.greater_token, dtype=torch.long) # Need to add > to init if missing
        # hash_t = torch.full((B, 1), self.hash_token, dtype=torch.long) # Need to add # to init if missing

        # Fallbacks for tokens that might be missing in __init__
        greater_token = self.stoi.get(
            ">", self.stoi.get("=")
        )  # Fallback? No, should exist
        hash_token = self.stoi.get("#", self.stoi.get("="))

        greater = torch.full((B, 1), greater_token, dtype=torch.long)
        hash_t = torch.full((B, 1), hash_token, dtype=torch.long)

        # -- Input Phase --
        for k in range(N_Ops):
            L_op = operands_digits[k].shape[1]
            tokens_list.append(operands_digits[k])
            p1_list.append(torch.full((B, L_op), k + 1, dtype=torch.long))
            ids = torch.arange(L_op, 0, -1).unsqueeze(0).expand(B, -1)
            p2_list.append(ids)
            p3_list.append(torch.full((B, L_op), 1, dtype=torch.long))

            if k < N_Ops - 1:
                tokens_list.append(plus)
                p1_list.append(torch.full((B, 1), k + 1, dtype=torch.long))
                p2_list.append(torch.zeros((B, 1), dtype=torch.long))
                p3_list.append(torch.full((B, 1), 1, dtype=torch.long))
            else:
                tokens_list.append(eq)
                p1_list.append(torch.full((B, 1), k + 1, dtype=torch.long))
                p2_list.append(torch.zeros((B, 1), dtype=torch.long))
                p3_list.append(torch.full((B, 1), 1, dtype=torch.long))

        # -- Scratchpad Phase --
        for k, seg in enumerate(scratchpad_segments):
            L_seg = seg.shape[1]
            tokens_list.append(seg)
            p1_list.append(torch.full((B, L_seg), k + 1, dtype=torch.long))
            ids = torch.arange(1, L_seg + 1).unsqueeze(0).expand(B, -1)
            p2_list.append(ids)
            p3_list.append(torch.full((B, L_seg), 2, dtype=torch.long))

            if k < len(scratchpad_segments) - 1:
                tokens_list.append(greater)
                p1_list.append(torch.full((B, 1), k + 1, dtype=torch.long))
                p2_list.append(torch.full((B, 1), L_seg + 1, dtype=torch.long))
                p3_list.append(torch.full((B, 1), 2, dtype=torch.long))

        # End Token [#]
        tokens_list.append(hash_t)
        p1_list.append(torch.full((B, 1), N_Ops + 1, dtype=torch.long))
        p2_list.append(torch.zeros((B, 1), dtype=torch.long))
        p3_list.append(torch.full((B, 1), 2, dtype=torch.long))

        full_seq = torch.cat(tokens_list, dim=1)
        pos1 = torch.cat(p1_list, dim=1)
        pos2 = torch.cat(p2_list, dim=1)
        pos3 = torch.cat(p3_list, dim=1)

        # Prepare for return
        # idx, pos1_ids, pos2_ids, tokens (list)
        return full_seq, pos1, pos2, pos3, full_seq[0].tolist()

    def visualize_addition(self, equation_string, save_path=None, all_tokens=False):
        """
        Runs model, extracts attention, and plots the staircase pattern.
        """
        idx, pos1_ids, pos2_ids, pos3_ids, tokens = self.encode_equation(
            equation_string
        )

        # Ensure model is on CPU or same device
        device = next(self.model.parameters()).device
        idx = idx.to(device)
        pos1_ids = pos1_ids.to(device)
        pos2_ids = pos2_ids.to(device)
        pos3_ids = pos3_ids.to(device)

        with torch.no_grad():
            logits, _, all_attn = self.model(
                idx, pos1_ids, pos2_ids, pos3_ids, return_weights=True
            )

        probs = torch.softmax(logits[0], dim=-1).cpu()

        def decode_mod10(t_id):
            char = self.itos[t_id]
            if char in ["+", "=", ">", "#"]:
                return char
            return str(int(char) % 10)

        input_tk_names = [decode_mod10(t) for t in tokens]
        output_tk_names = input_tk_names[1:] + [""]

        table = [
            ["position"] + [str(i) for i in range(len(tokens))],
            ["input_tk"] + input_tk_names,
            ["output_tk"] + output_tk_names,
        ]
        for i in range(len(self.itos)):
            char = self.itos[i]
            label = (
                char if char in ["+", "=", ">", "#"] else f"{char}({int(char) % 10})"
            )
            table.append(
                [f"pr_{label}"] + [f"{probs[j, i]:.2f}" for j in range(len(tokens))]
            )

        # Determine column widths for nice printing
        col_widths = [max(len(str(item)) for item in col) for col in zip(*table)]

        # Print with nice separators
        header_sep = "-+-".join("-" * w for w in col_widths)
        print("\nSequence Prediction Analysis (Probabilities per position):")
        for r_idx, row in enumerate(table):
            print(
                " | ".join(
                    f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row)
                )
            )
            if r_idx in [0, 2]:  # After position, after output_tk
                print(header_sep)

        # Result indices (where we want to see what tokens we attended to)
        res_indices = [i for i, p in enumerate(pos1_ids[0]) if p == 3]
        input_token_labels = [decode_mod10(t) for t in tokens]

        # Determine model predictions and probabilities for result tokens
        # The prediction for result token at index i is generated by the token at index i-1
        pred_indices = [i - 1 for i in res_indices]
        res_logits = logits[0, pred_indices, :]
        res_probs = F.softmax(res_logits, dim=-1)

        preds = torch.argmax(res_logits, dim=-1)
        ground_truth = idx[0, res_indices]

        # Mapping from prediction index to its label
        pred_to_label = {}
        all_correct = True
        print(f"\nPrediction Analysis for calculated tokens:")
        for i, pred_idx in enumerate(pred_indices):
            gt_idx = ground_truth[i].item()
            gt_char = self.itos[gt_idx]
            pred_idx_val = preds[i].item()
            pred_char = self.itos[pred_idx_val]
            gt_prob = res_probs[i, gt_idx].item()

            gt_display = decode_mod10(gt_idx)
            pred_display = decode_mod10(pred_idx_val)

            if gt_idx == pred_idx_val:
                label = f"{gt_display} ({gt_prob:.1%})"
                print(
                    f"  Token {i}: Correct! Target '{gt_char}' ({gt_display}) has prob {gt_prob:.4f}"
                )
            else:
                label = f"{gt_display} (✗ {pred_display} {res_probs[i, pred_idx_val]:.1%}, gt: {gt_prob:.1%})"
                all_correct = False
                print(
                    f"  Token {i}: WRONG! Expected '{gt_char}' ({gt_display}) ({gt_prob:.4f}), got '{pred_char}' ({pred_display}) ({res_probs[i, pred_idx_val]:.4f})"
                )
            pred_to_label[pred_idx] = label

        status_text = "PASSED" if all_correct else "FAILED"
        print(f"Overall Result: {status_text}\n")

        # Determine which rows to show on Y-axis
        if all_tokens:
            # Show all tokens up to the last prediction made
            y_indices = list(range(len(tokens)))
            y_labels = []
            for i in y_indices:
                if i in pred_to_label:
                    y_labels.append(pred_to_label[i])
                else:
                    # For input tokens, we show itos[tokens[i]] -> itos[tokens[i+1]]
                    y_labels.append(
                        f"{decode_mod10(tokens[i])} \u2192 {decode_mod10(tokens[i + 1])}"
                    )
        else:
            # Just the result producing steps
            y_indices = pred_indices
            y_labels = [pred_to_label[i] for i in y_indices]

        # Reverse Y-axis (Time flows upwards: Start at bottom, End at top)
        y_indices = y_indices[::-1]
        y_labels = y_labels[::-1]

        n_layers = len(all_attn)
        n_heads = all_attn[0].shape[1]

        fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 6, n_layers * 5))
        if n_layers == 1:
            axes = np.array([axes])
        if n_heads == 1:
            axes = np.array([[ax] for ax in axes])

        # Identifying indices for staircase
        n1_indices = [
            i
            for i, p in enumerate(pos1_ids[0])
            if p == 1 and tokens[i] != self.plus_token
        ]
        n2_indices = [
            i
            for i, p in enumerate(pos1_ids[0])
            if p == 2 and tokens[i] != self.eq_token
        ]
        L = len(n1_indices)

        for l in range(n_layers):
            attn = all_attn[l][0]  # (heads, T, T)
            for h in range(n_heads):
                head_attn = attn[h]  # (T, T)
                # Select requested rows
                plot_attn = head_attn[y_indices, :].cpu().numpy()

                ax = axes[l, h]
                sns.heatmap(
                    plot_attn,
                    ax=ax,
                    xticklabels=input_token_labels,
                    yticklabels=y_labels,
                    cmap="viridis",
                    annot=False,
                    cbar=True,
                    vmin=0,
                    vmax=1,
                )

                # Highlight "Staircase" Pattern (only for result positions)
                for i_y, y_idx in enumerate(y_indices):
                    if y_idx in pred_indices:
                        j = pred_indices.index(y_idx)  # Which result digit is this?

                        target_cols = []
                        if L - 1 - j >= 0:
                            target_cols.append(n1_indices[L - 1 - j])
                            target_cols.append(n2_indices[L - 1 - j])
                        if j > 0:
                            if L - j >= 0:
                                target_cols.append(n1_indices[L - j])
                                target_cols.append(n2_indices[L - j])

                        for col in target_cols:
                            ax.add_patch(
                                plt.Rectangle(
                                    (col, i_y), 1, 1, fill=False, edgecolor="red", lw=2
                                )
                            )

                ax.set_title(f"Layer {l} | Head {h}")
                ax.set_xlabel("Input Sequence")
                ax.set_ylabel("Generated Tokens")

        plt.suptitle(f"Attention Analysis: {equation_string}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()


def load_model_from_checkpoint(checkpoint_path, model_class):
    """
    Helper to load the model.
    """
    return model_class.load_from_checkpoint(checkpoint_path)


if __name__ == "__main__":
    import argparse
    from model import GPTLightningModule

    parser = argparse.ArgumentParser(
        description="Visualize Transformer Attention for Addition"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--equation", type=str, default="123+456=975", help="Equation to visualize"
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all tokens on Y-axis (input + result)"
    )
    parser.add_argument("--save", type=str, help="Path to save the plot")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = GPTLightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu")

    explorer = AttentionExplorer(model)
    print(f"Generating visualization for: {args.equation}")
    explorer.visualize_addition(args.equation, save_path=args.save, all_tokens=args.all)
