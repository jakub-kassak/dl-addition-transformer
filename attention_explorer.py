import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
from data import construct_addition_batch


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
        Uses data.construct_addition_batch logic.
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
        carry_allowance = math.ceil(math.log10(N_Ops))
        max_len = max_digits + carry_allowance

        # Prepare operands_digits list of tensors
        operands_digits = []
        for op in operands:
            # Pad to max_len (zeros at front/MSB)
            op_padded = op.zfill(max_len)
            d = [int(c) for c in op_padded]
            t = torch.tensor(d, dtype=torch.long).unsqueeze(0)  # (1, max_len)
            operands_digits.append(t)

        # Call construct_addition_batch
        full_seq, pos1, pos2, pos3 = construct_addition_batch(
            operands_digits,
            self.stoi,
            random_offsets=False,
            explicit_carry=getattr(self.model.hparams, "explicit_carry", True),
        )

        # Prepare for return
        # idx, pos1_ids, pos2_ids, pos3_ids, tokens (list)
        return full_seq, pos1, pos2, pos3, full_seq[0].tolist()

    def prediction_correctness(
        self, pred_indices, ground_truth, preds, res_probs, decode, pred_to_label
    ):
        print("\nPrediction Analysis for calculated tokens:")
        all_correct = True
        for i, pred_idx in enumerate(pred_indices):
            gt_idx = ground_truth[i].item()
            gt_char = self.itos[gt_idx]
            pred_idx_val = preds[i].item()
            pred_char = self.itos[pred_idx_val]
            gt_prob = res_probs[i, gt_idx].item()

            gt_display = decode(gt_idx)
            pred_display = decode(pred_idx_val)

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

    def pred_table(self, tokens, decode, probs):
        input_tk_names = [decode(t) for t in tokens]
        output_tk_names = input_tk_names[1:] + [""]

        table = [
            ["position"] + [str(i) for i in range(len(tokens))],
            ["input_tk"] + input_tk_names,
            ["output_tk"] + output_tk_names,
        ]
        for i in range(len(self.itos)):
            label = self.itos[i]
            table.append([label] + [f"{probs[j, i]:.2f}" for j in range(len(tokens))])

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

    def visualize_addition(
        self,
        equation_string,
        save_path=None,
        all_tokens=False,
        show_transitions=False,
        print_pred_table=False,
        model_name="",
        start=None,
        end=None,
    ):
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

        def decode(t_id):
            return self.itos[t_id]

        if print_pred_table:
            self.pred_table(tokens, decode, probs)

        # Result indices (where we want to see what tokens we attended to)
        res_indices = [i for i, p in enumerate(pos3_ids[0]) if p == 2]

        # Slice result indices if start/end are provided
        if start is not None or end is not None:
            res_indices = res_indices[start:end]

        input_token_labels = [decode(t) for t in tokens]

        # Determine model predictions and probabilities for result tokens
        # The prediction for result token at index i is generated by the token at index i-1
        pred_indices = [i - 1 for i in res_indices]
        res_logits = logits[0, pred_indices, :]
        res_probs = F.softmax(res_logits, dim=-1)
        preds = torch.argmax(res_logits, dim=-1)
        ground_truth = idx[0, res_indices]

        # Mapping from prediction index to its label
        pred_to_label = {}
        self.prediction_correctness(
            pred_indices, ground_truth, preds, res_probs, decode, pred_to_label
        )

        # Determine which rows to show on Y-axis
        if all_tokens:
            # Show all tokens up to the last prediction made
            y_indices = list(range(len(tokens) - 1))
            y_labels = []
            for i in y_indices:
                # target_display is what we predict/expect at position i+1
                if i in pred_to_label:
                    # Use the descriptive label (e.g. "5 (99.7%)")
                    target_display = pred_to_label[i]
                else:
                    # Use the actual token at i+1
                    target_display = decode(tokens[i + 1])

                if show_transitions:
                    y_labels.append(f"{decode(tokens[i])} \u2192 {target_display}")
                else:
                    y_labels.append(target_display)
        else:
            # Just the result producing steps
            y_indices = pred_indices
            if show_transitions:
                y_labels = []
                for i in y_indices:
                    # For result indices i, tokens[i] is the token that makes the prediction
                    # However pred_indices are res_indices - 1.
                    # So for res_idx, the prediction is made by token at res_idx-1.
                    # If i is in pred_indices, then i+1 is the res_idx.
                    y_labels.append(f"{decode(tokens[i])} \u2192 {pred_to_label[i]}")
            else:
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
        l = len(tokens)
        n1_indices = []
        n2_indices = []
        for i in range(l):
            for j in range(l):
                if (  # input operand Xi
                    pos1_ids[0][i] == pos1_ids[0][j] - 1
                    and pos2_ids[0][i] == pos2_ids[0][j] + 1
                    and pos3_ids[0][i] == 1
                    and pos3_ids[0][j] == 2
                ):
                    n1_indices.append((i, j))
                    n2_indices.append((i + 1, j))
                if (  # intermediate sum S{i-1}
                    pos1_ids[0][i] == pos1_ids[0][j] - 1
                    and pos2_ids[0][i] == pos2_ids[0][j] + 1
                    and pos3_ids[0][i] == 2
                    and pos3_ids[0][j] == 2
                ):
                    n1_indices.append((i, j))
                    n2_indices.append((i - 1, j))

        print("\nAverage Attention Probability for n2_indices (Staircase adjacent):")
        for l_idx in range(n_layers):
            attn = all_attn[l_idx][0]  # (heads, T, T)
            for h_idx in range(n_heads):
                vals = []
                for src_idx, dest_idx in n2_indices:
                    if 0 <= src_idx < attn.shape[2] and 0 <= dest_idx < attn.shape[1]:
                        vals.append(attn[h_idx, dest_idx, src_idx].item())

                if vals:
                    avg = sum(vals) / len(vals)
                    vals.sort()
                    median = vals[len(vals) // 2]
                    print(
                        f"  Layer {l_idx}, Head {h_idx}: avg={avg:.4f}, median={median:.4f}"
                    )
                else:
                    print(f"  Layer {l_idx}, Head {h_idx}: N/A (no valid indices)")

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
                    # vmax=1,
                )

                # Highlight "Staircase" Pattern
                for src_idx, dest_idx in n1_indices:
                    # src_idx is Input (column), dest_idx is Scratchpad (row)
                    if dest_idx in y_indices:
                        plot_y = y_indices.index(dest_idx)
                        ax.add_patch(
                            plt.Rectangle(
                                (src_idx, plot_y),
                                1,
                                1,
                                fill=False,
                                edgecolor="red",
                                lw=2,
                            )
                        )

                # Highlight "Staircase" Pattern
                for src_idx, dest_idx in n2_indices:
                    # src_idx is Input (column), dest_idx is Scratchpad (row)
                    if dest_idx in y_indices:
                        plot_y = y_indices.index(dest_idx)
                        ax.add_patch(
                            plt.Rectangle(
                                (src_idx, plot_y),
                                1,
                                1,
                                fill=False,
                                edgecolor="green",
                                lw=2,
                            )
                        )

                ax.set_title(f"Layer {l} | Head {h}")
                ax.set_xlabel("Input Sequence")
                ax.set_ylabel("Generated Tokens")

        plt.suptitle(
            f"Attention Analysis: {equation_string}\nModel: {model_name}", fontsize=16
        )
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
    parser.add_argument(
        "--show_transitions",
        action="store_true",
        help="Show transitions (x -> y) on Y-axis",
    )
    parser.add_argument("--save", type=str, help="Path to save the plot")
    parser.add_argument(
        "--start", type=int, default=None, help="Start index of result region to show"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index of result region to show"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = GPTLightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu")

    explorer = AttentionExplorer(model)
    print(f"Generating visualization for: {args.equation}")
    model_name = str(args.checkpoint).split("/")[-3].split(".")[0]
    explorer.visualize_addition(
        args.equation,
        save_path=args.save,
        all_tokens=args.all,
        show_transitions=args.show_transitions,
        model_name=model_name,
        start=args.start,
        end=args.end,
    )
