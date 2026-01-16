import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from model import GPTLightningModule
from data import AdditionDataModule, construct_addition_batch
import math


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step.
    This optimizer is intended for 2D parameters (matrices) only.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Newton-Schulz iteration
                update = self._newton_schulz(g, ns_steps)
                p.data.add_(update, alpha=-lr)

    @staticmethod
    def _newton_schulz(G, steps=5):
        """
        Hyper-optimized Newton-Schulz iteration to replace G with an approximate orthogonal matrix.
        """
        assert len(G.shape) == 2
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16() if G.is_cuda and G.dtype != torch.float16 else G
        X /= X.norm() + 1e-7  # ensure top singular value <= 1
        if G.shape[0] > G.shape[1]:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.shape[0] > G.shape[1]:
            X = X.T
        return X.to(G.dtype)


class TunedLensTranslator(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.affine = nn.Linear(n_embd, n_embd)

    def forward(self, h):
        return self.affine(h)


class TunedLens:
    def __init__(
        self, model_checkpoint_path, save_path=None, device="cpu", force_train=False
    ):
        self.device = device
        self.save_path = save_path
        self.model = GPTLightningModule.load_from_checkpoint(model_checkpoint_path)
        self.model.to(device)
        self.model.eval()
        self.n_layer = self.model.hparams.n_layer
        self.n_embd = self.model.hparams.n_embd

        # Translators for each layer (including embedding layer at index 0)
        self.translators = nn.ModuleList(
            [TunedLensTranslator(self.n_embd) for _ in range(self.n_layer + 1)]
        ).to(device)

        # Token mapping based on data.py (22 tokens: 0-19, +, =, >, #)
        self.chars = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token = -1
        self.plus_token = self.stoi["+"]
        self.eq_token = self.stoi["="]
        self.greater_token = self.stoi[">"]
        self.hash_token = self.stoi["#"]

        self.is_trained = False

        # Load if exists and not forced to retrain
        if save_path and os.path.exists(save_path) and not force_train:
            print(f"Loading Tuned Lens from {save_path}...")
            try:
                self.translators.load_state_dict(
                    torch.load(save_path, map_location=device)
                )
                self.is_trained = True
                print("✅ Successfully loaded Tuned Lens.")
            except Exception as e:
                print(f"⚠️ Failed to load Tuned Lens: {e}. Will retrain.")

    def get_hidden_states(self, idx, pos1_ids, pos2_ids, pos3_ids):
        """
        Captured hidden states after each block.
        """
        # We need to ensure we don't compute gradients for the main model
        with torch.no_grad():
            x = self.model.token_embedding_table(idx)

            head_size = self.model.hparams.n_embd // self.model.hparams.n_head
            rope = None

            if self.model.hparams.pos_emb_type == "learned":
                # Absolute for all
                p1 = self.model.pos1_embedding_table(pos1_ids)
                p2 = self.model.pos2_embedding_table(pos2_ids)
                p3 = self.model.pos3_embedding_table(pos3_ids)
                x = x + p1 + p2 + p3
            elif self.model.hparams.pos_emb_type == "rope":
                d = head_size
                d1 = (d // 3) // 2 * 2
                d2 = (d // 3) // 2 * 2
                d3 = d - d1 - d2
                rope = self.model.prepare_multidim_rope(
                    [pos1_ids, pos2_ids, pos3_ids], [d1, d2, d3]
                )
            elif self.model.hparams.pos_emb_type == "mixed":
                p3 = self.model.pos3_embedding_table(pos3_ids)
                x = x + p3
                d = head_size
                d1 = (d // 2) // 2 * 2
                d2 = d - d1
                rope = self.model.prepare_multidim_rope([pos1_ids, pos2_ids], [d1, d2])

            hidden_states = [x]  # Layer -1
            for block in self.model.blocks:
                x = block(x, rope=rope)
                hidden_states.append(x)

            # Final logits
            final_x = self.model.ln_f(x)
            final_logits = self.model.lm_head(final_x)

        return hidden_states, final_logits

    def train(self, dataloader, max_steps=2000, lr=0.02, patience=50, min_delta=1e-4):
        if self.is_trained:
            print("Tuned Lens is already trained. Skipping training.")
            return

        # Separate 2D params (matrices) for Muon and 1D params (biases) for AdamW
        muon_params = []
        adam_params = []

        for translator in self.translators:
            for name, p in translator.named_parameters():
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        optimizers = []
        if muon_params:
            optimizers.append(Muon(muon_params, lr=lr))
        if adam_params:
            optimizers.append(torch.optim.AdamW(adam_params, lr=0.01))

        print(f"Training Tuned Lens with Fresh Data (Muon + AdamW)...")
        print(f"Muon params: {len(muon_params)}, AdamW params: {len(adam_params)}")

        iterator = iter(dataloader)
        pbar = tqdm(range(max_steps), desc="Training")

        # Early Stopping state
        best_loss = float("inf")
        steps_without_improvement = 0
        running_loss = 0.0
        smooth_factor = 0.05

        for step in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            idx, targets, pos1_ids, pos2_ids, pos3_ids = [
                b.to(self.device) for b in batch
            ]

            h_list, final_logits = self.get_hidden_states(
                idx, pos1_ids, pos2_ids, pos3_ids
            )

            # Target distribution (Soft Target)
            with torch.no_grad():
                target_probs = F.softmax(final_logits, dim=-1)
                B, T, V = target_probs.shape
                target_probs = target_probs.view(B * T, V)

            total_loss = 0

            # Zero grads
            for opt in optimizers:
                opt.zero_grad()

            for l in range(self.n_layer + 1):
                h_l = h_list[l].detach()  # Important: detach from main model graph

                # Forward pass through translator
                h_hat = self.translators[l](h_l)  # (B, T, D)
                h_hat_flat = h_hat.view(B * T, -1)

                final_x_hat = self.model.ln_f(h_hat_flat)
                early_logits = self.model.lm_head(final_x_hat)
                early_log_probs = F.log_softmax(early_logits, dim=-1)

                loss_l = F.kl_div(early_log_probs, target_probs, reduction="batchmean")
                total_loss += loss_l

            # Backward
            avg_loss = total_loss / (self.n_layer + 1)
            avg_loss.backward()

            # Step
            for opt in optimizers:
                opt.step()

            # Logging and Early Stopping
            loss_val = avg_loss.item()
            if step == 0:
                running_loss = loss_val
            else:
                running_loss = (
                    1 - smooth_factor
                ) * running_loss + smooth_factor * loss_val

            pbar.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "patience": f"{patience - steps_without_improvement}",
                }
            )

            # Check stopping every 50 steps to be stable
            if step % 50 == 0:
                if running_loss < best_loss - min_delta:
                    best_loss = running_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if steps_without_improvement >= patience:
                    print(
                        f"\nEarly stopping triggered at step {step}. Best Loss: {best_loss:.4f}"
                    )
                    break

        self.is_trained = True

        # Save
        if self.save_path:
            print(f"Saving Tuned Lens to {self.save_path}...")
            torch.save(self.translators.state_dict(), self.save_path)

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

    @torch.no_grad()
    def visualize_trajectory(
        self,
        equation,
        method="tuned",
        theoretical_no_carry=False,
        show_probs=False,
        top_k=1,
    ):
        """
        visualize the trajectory of predictions across layers using Teacher Forcing.
        """
        idx, pos1_ids, pos2_ids, pos3_ids, tokens = self.encode_equation(equation)

        # Move to device
        idx = idx.to(self.device)
        pos1_ids = pos1_ids.to(self.device)
        pos2_ids = pos2_ids.to(self.device)
        pos3_ids = pos3_ids.to(self.device)

        h_list, final_logits = self.get_hidden_states(idx, pos1_ids, pos2_ids, pos3_ids)

        results = []
        seq_len = idx.shape[1]

        # We iterate over the sequence to capture predictions at each step
        for t in range(seq_len - 1):  # Can't predict after the last token
            curr_token_idx = idx[0, t].item()
            curr_token_str = self.itos[curr_token_idx]

            # Prediction target is the next token
            next_token_idx = idx[0, t + 1].item()
            next_token_str = self.itos[next_token_idx]

            step_results = {"Input": curr_token_str, "Target": next_token_str}

            # Layers (including Layer -1)
            for l in range(self.n_layer + 1):
                h_l_t = h_list[l][:, t, :]
                if method == "tuned":
                    h_hat = self.translators[l](h_l_t)
                    logits = self.model.lm_head(self.model.ln_f(h_hat))
                else:  # logit lens
                    logits = self.model.lm_head(self.model.ln_f(h_l_t))

                probs = F.softmax(logits, dim=-1)

                # Get Top K
                top_v, top_i = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                top_v = top_v[0].tolist()
                top_i = top_i[0].tolist()

                layer_label = f"Layer {l - 1}" if l > 0 else "Layer -1"

                if top_k > 1:
                    lines = [
                        f"{self.itos[idx]} ({p:.1%})" for p, idx in zip(top_v, top_i)
                    ]
                    step_results[layer_label] = "\n".join(lines)
                elif show_probs:
                    step_results[layer_label] = (
                        f"{self.itos[top_i[0]]} ({top_v[0]:.1%})"
                    )
                else:
                    step_results[layer_label] = self.itos[top_i[0]]

                # Store top 1 index for cell coloring
                step_results[f"{layer_label}_idx"] = top_i[0]

            # Model final output (after all blocks)
            f_probs = F.softmax(final_logits[:, t, :], dim=-1)
            f_v, f_i = torch.topk(f_probs, k=min(top_k, f_probs.size(-1)), dim=-1)
            f_v = f_v[0].tolist()
            f_i = f_i[0].tolist()

            if top_k > 1:
                lines = [f"{self.itos[idx]} ({p:.1%})" for p, idx in zip(f_v, f_i)]
                step_results["Model Final"] = "\n".join(lines)
            elif show_probs:
                step_results["Model Final"] = f"{self.itos[f_i[0]]} ({f_v[0]:.1%})"
            else:
                step_results["Model Final"] = self.itos[f_i[0]]

            step_results["Model Final_idx"] = f_i[0]
            step_results["Target_idx"] = next_token_idx
            results.append(step_results)

        # Display table
        console = Console()
        method_name = "Tuned Lens" if method == "tuned" else "Logit Lens"
        table = Table(title=f"{method_name} Prediction Trajectory (Teacher Forcing)")
        table.caption = f"Equation: {equation}"

        table.add_column("Pos", justify="right", style="dim")
        table.add_column("Input", justify="center", style="bold")
        for l in range(self.n_layer + 1):
            layer_label = f"L {l - 1}" if l > 0 else "L -1"
            table.add_column(layer_label, justify="center")
        table.add_column("Final", justify="center", style="bold green")
        table.add_column("Target", justify="center", style="bold yellow")

        is_input_phase = True
        for i, res in enumerate(results):
            row_style = None
            if is_input_phase:
                row_style = "bright_black"  # Grey for input
                if res["Target"] == "=":
                    is_input_phase = False

                row = [str(i), res["Input"]]
                for l in range(self.n_layer + 1):
                    layer_label = f"Layer {l - 1}" if l > 0 else "Layer -1"
                    row.append(res[layer_label])
                row.append(res["Model Final"])
                row.append(res["Target"])
            else:
                # Prediction phase: color individual cells
                target_idx = res["Target_idx"]
                row = [str(i), res["Input"]]

                for l in range(self.n_layer + 1):
                    layer_label = f"Layer {l - 1}" if l > 0 else "Layer -1"
                    cell_val = res[layer_label]
                    cell_idx = res[f"{layer_label}_idx"]
                    color = "green" if cell_idx == target_idx else "red"
                    row.append(f"[{color}]{cell_val}[/{color}]")

                # Model Final cell
                final_val = res["Model Final"]
                final_idx = res["Model Final_idx"]
                color = "green" if final_idx == target_idx else "red"
                row.append(f"[{color}]{final_val}[/{color}]")

                row.append(res["Target"])

            table.add_row(*row, style=row_style)

        console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_steps", type=int, default=2000, help="Max training steps"
    )
    parser.add_argument("--lr", type=float, default=0.02, help="Muon learning rate")
    parser.add_argument("--equation", type=str, default="123+456=")
    parser.add_argument(
        "--no_carry", action="store_true", help="Show no-carry analysis"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save/load tuned lens"
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Force retraining even if save exists",
    )
    parser.add_argument(
        "--show_probs", action="store_true", help="Show probability of predictions"
    )
    parser.add_argument("--top_k", type=int, default=1, help="Show top K predictions")
    parser.add_argument(
        "--method",
        type=str,
        default="tuned",
        choices=["tuned", "logit"],
        help="Lens method to use",
    )

    args = parser.parse_args()

    # Define default save path if not provided
    if args.save_path is None:
        exp_dir = os.path.dirname(os.path.dirname(args.ckpt))  # experiments/exp_name
        args.save_path = os.path.join(exp_dir, "tuned_lens.pt")
        print(f"Defaulting save path to: {args.save_path}")

    lens = TunedLens(
        args.ckpt,
        save_path=args.save_path,
        device=args.device,
        force_train=args.force_train,
    )

    if args.method == "tuned":
        # Load data for training translators
        dm = AdditionDataModule(
            min_train_digits=1,
            max_train_digits=7,
            batch_size=args.batch_size,
            explicit_carry=getattr(lens.model.hparams, "explicit_carry", True),
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        lens.train(train_loader, max_steps=args.max_steps, lr=args.lr)

    # Visualize
    lens.visualize_trajectory(
        args.equation,
        method=args.method,
        theoretical_no_carry=args.no_carry,
        show_probs=args.show_probs,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
