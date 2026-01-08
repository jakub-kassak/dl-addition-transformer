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
        self.chars = [str(i) for i in range(20)] + ["+", "="]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token = -1
        self.plus_token = self.stoi["+"]
        self.eq_token = self.stoi["="]

    def encode_equation(self, equation_string):
        """
        Encodes an equation string into tokens and positional IDs.
        Matches the logic in data.py
        """
        if "=" not in equation_string:
            raise ValueError("Equation must contain '='")

        prefix, result_str = equation_string.split("=")
        if "+" not in prefix:
            raise ValueError("Prefix must contain '+'")

        n1_str, n2_str = prefix.split("+")
        L = len(n1_str)
        if len(n2_str) != L:
            # We handle same-length additions primarily as per data.py
            # If they differ, we'd need to know how the user pads them.
            # Assuming they are same length for now.
            pass

        n1_tokens = [self.stoi[c] for c in n1_str]
        n2_tokens = [self.stoi[c] for c in n2_str]

        # Calculate Carries for result tokens (matches data.py logic)
        res_tokens = []
        carry = 0
        # Result digits are LSB first in the sequence as per data.py
        # We need to calculate carries from LSB to MSB
        n1_digits = [int(c) for c in n1_str]
        n2_digits = [int(c) for c in n2_str]

        for d1, d2 in zip(reversed(n1_digits), reversed(n2_digits)):
            total = d1 + d2 + carry
            res_tokens.append(self.stoi[str(total)])
            carry = total // 10

        # Append final carry
        res_tokens.append(self.stoi[str(carry)])

        # The equation provided by the user in --equation might be incomplete or have the wrong result.
        # We use the CALCULATED result tokens for the forward pass to match training data distribution.

        # tokens = [n1] + [+] + [n2] + [=] + [s] + [0]
        tokens = (
            n1_tokens + [self.plus_token] + n2_tokens + [self.eq_token] + res_tokens + [0]
        )
        idx = torch.tensor(tokens).unsqueeze(0)

        # Positional IDs (pos1)
        # n1+ is segment 1, n2= is segment 2, result is segment 3
        pos1 = (
            ([1] * (len(n1_tokens) + 1))
            + ([2] * (len(n2_tokens) + 1))
            + ([3] * (len(res_tokens) + 1))
        )
        pos1_ids = torch.tensor(pos1).unsqueeze(0)

        # Positional IDs (pos2)
        # matches data.py: [L...1, 0] [L...1, 0] [1...L+1]
        idx_1_L_n1 = list(range(L, 0, -1)) + [0]  # n1 digits + '+'
        idx_1_L_n2 = list(range(L, 0, -1)) + [0]  # n2 digits + '='
        idx_L_0 = list(range(1, len(res_tokens) + 2))

        pos2 = idx_1_L_n1 + idx_1_L_n2 + idx_L_0
        pos2_ids = torch.tensor(pos2).unsqueeze(0)

        return idx, pos1_ids, pos2_ids, tokens

    def visualize_addition(self, equation_string, save_path=None, all_tokens=False):
        """
        Runs model, extracts attention, and plots the staircase pattern.
        """
        idx, pos1_ids, pos2_ids, tokens = self.encode_equation(equation_string)

        # Ensure model is on CPU or same device
        device = next(self.model.parameters()).device
        idx = idx.to(device)
        pos1_ids = pos1_ids.to(device)
        pos2_ids = pos2_ids.to(device)

        with torch.no_grad():
            logits, _, all_attn = self.model(
                idx, pos1_ids, pos2_ids, return_weights=True
            )

        probs = torch.softmax(logits[0], dim=-1).cpu()

        def decode_mod10(t_id):
            char = self.itos[t_id]
            if char in ["+", "="]:
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
            label = char if char in ["+", "="] else f"{char}({int(char) % 10})"
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
