import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from model import GPTLightningModule
from data import AdditionDataModule, VectorizedAdditionDataset


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

    def get_hidden_states(self, idx, pos1_ids, pos2_ids):
        """
        Captured hidden states after each block.
        """
        # We need to ensure we don't compute gradients for the main model
        with torch.no_grad():
            x = self.model.token_embedding_table(idx)  # (B, T, n_embd)

            rope = None
            if self.model.hparams.pos_emb_type == "rope":
                rope = self.model.prepare_rope(pos1_ids, pos2_ids)
            elif self.model.hparams.pos_emb_type == "learned":
                pos1_emb = self.model.pos1_embedding_table(pos1_ids)  # (B, T, n_embd)
                # Bound Check for pos2 to avoid IndexError
                max_p2 = self.model.pos2_embedding_table.num_embeddings
                if (pos2_ids >= max_p2).any() or (pos2_ids < 0).any():
                    pos2_ids = torch.clamp(pos2_ids, 0, max_p2 - 1)
                pos2_emb = self.model.pos2_embedding_table(pos2_ids)  # (B, T, n_embd)
                x = x + pos1_emb + pos2_emb  # (B, T, n_embd)
            elif self.model.hparams.pos_emb_type == "abc_mixed":
                # pos1 is added as learned absolute PE
                pos1_emb = self.model.pos1_embedding_table(pos1_ids)  # (B, T, n_embd)
                x = x + pos1_emb
                # pos2 is RoPE spanning the entire head dimension
                head_size = self.model.hparams.n_embd // self.model.hparams.n_head
                rope = self.model.rope(pos2_ids, head_size).unsqueeze(
                    1
                )  # (B, 1, T, head_size//2, 2, 2)

            hidden_states = [x]  # Add embedding layer at index 0
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
            optimizers.append(
                torch.optim.AdamW(adam_params, lr=0.01)
            )  # Standard LR for AdamW

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

            idx, targets, pos1_ids, pos2_ids = [b.to(self.device) for b in batch]

            h_list, final_logits = self.get_hidden_states(idx, pos1_ids, pos2_ids)

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

    @torch.no_grad()
    def visualize_trajectory(
        self,
        equation,
        stoi,
        itos,
        method="tuned",
        theoretical_no_carry=False,
        show_probs=False,
        top_k=1,
    ):
        """
        visualize the trajectory of predictions across layers.
        """

        # Split equation to get n1 and n2
        if "=" in equation:
            prompt = equation.split("=")[0] + "="
        else:
            prompt = equation + "="

        n1_str, n2_str = prompt.split("+")
        n2_str = n2_str.replace("=", "")

        L1 = len(n1_str)
        L2 = len(n2_str)
        tokens = [stoi[c] for c in prompt]

        # Reconstruction of pos2 for prompt:
        # n1 digits + plus
        p2_n1_plus = list(range(L1, -1, -1))
        # n2 digits + eq
        p2_n2_eq = list(range(L2, -1, -1))
        p2 = p2_n1_plus + p2_n2_eq

        # p1 logic
        p1 = ([1] * (L1 + 1)) + ([2] * (L2 + 1))

        # Use offset=0 for visualization to avoid running out of embedding table range
        offset = 0
        p2 = [p + offset for p in p2]

        idx_in = torch.tensor([tokens], device=self.device)
        p1_in = torch.tensor([p1], device=self.device)
        p2_in = torch.tensor([p2], device=self.device)

        # We need to generate the result tokens to see the trajectory over the result
        max_gen = max(L1, L2) + 1
        curr_idx = idx_in
        curr_p1 = p1_in
        curr_p2 = p2_in

        results = []  # List of dicts for each generation step

        for step in range(max_gen):
            h_list, final_logits = self.get_hidden_states(curr_idx, curr_p1, curr_p2)

            # We are interested in the last position's predictions
            step_results = {}

            # Layers (including Layer -1)
            for l in range(self.n_layer + 1):
                h_l_last = h_list[l][:, -1, :]
                if method == "tuned":
                    h_hat = self.translators[l](h_l_last)
                    logits = self.model.lm_head(self.model.ln_f(h_hat))
                else:  # logit lens
                    logits = self.model.lm_head(self.model.ln_f(h_l_last))

                probs = F.softmax(logits, dim=-1)

                # Get Top K
                top_v, top_i = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                top_v = top_v[0].tolist()
                top_i = top_i[0].tolist()

                layer_label = f"Layer {l - 1}" if l > 0 else "Layer -1"

                if top_k > 1:
                    lines = [f"{itos[idx]} ({p:.1%})" for p, idx in zip(top_v, top_i)]
                    step_results[layer_label] = "\n".join(lines)
                elif show_probs:
                    step_results[layer_label] = f"{itos[top_i[0]]} ({top_v[0]:.1%})"
                else:
                    step_results[layer_label] = itos[top_i[0]]

            # Final output
            final_probs = F.softmax(final_logits[:, -1, :], dim=-1)
            f_v, f_i = torch.topk(
                final_probs, k=min(top_k, final_probs.size(-1)), dim=-1
            )
            f_v = f_v[0].tolist()
            f_i = f_i[0].tolist()

            if top_k > 1:
                lines = [f"{itos[idx]} ({p:.1%})" for p, idx in zip(f_v, f_i)]
                step_results["Final"] = "\n".join(lines)
            elif show_probs:
                step_results["Final"] = f"{itos[f_i[0]]} ({f_v[0]:.1%})"
            else:
                step_results["Final"] = itos[f_i[0]]

            results.append(step_results)

            # Update for next token
            next_token = f_i[0]
            curr_idx = torch.cat(
                [curr_idx, torch.tensor([[next_token]], device=self.device)], dim=1
            )

            # Update positions
            last_p1 = curr_p1[:, -1:]
            # For results, data.py uses pos1=3
            next_p1 = torch.tensor([[3]], device=self.device)
            curr_p1 = torch.cat([curr_p1, next_p1], dim=1)

            # For results, data.py uses INCREASING pos2: 1, 2, 3...
            next_p2 = torch.tensor([[step + 1 + offset]], device=self.device)
            curr_p2 = torch.cat([curr_p2, next_p2], dim=1)

        # Display table
        console = Console()
        method_name = "Tuned Lens" if method == "tuned" else "Logit Lens"
        table = Table(title=f"{method_name} Prediction Trajectory")
        table.caption = f"Equation: {equation}"

        table.add_column("Layer / Step", justify="left", style="bold cyan")
        for i in range(max_gen):
            table.add_column(f"Digit {i}", justify="center")

        # Row for each layer (including -1)
        for l in range(self.n_layer + 1):
            layer_label = f"Layer {l - 1}" if l > 0 else "Layer -1"
            row = [layer_label]
            for i in range(max_gen):
                row.append(results[i][layer_label])
            table.add_row(*row)

        # # Row for Final Output
        # row_final = ["Final Output"]
        # for i in range(max_gen):
        #     row_final.append(results[i]["Final"])
        # table.add_row(*row_final, style="bold green")

        # Row for Correct Result
        try:
            n1_digits = [int(c) for c in n1_str]
            n2_digits = [int(c) for c in n2_str]
            L1 = len(n1_digits)
            L2 = len(n2_digits)
            max_L = max(L1, L2)
            s_digits_rev = []
            carry = 0
            for i in range(max_L):
                d1 = n1_digits[L1 - 1 - i] if i < L1 else 0
                d2 = n2_digits[L2 - 1 - i] if i < L2 else 0
                total = d1 + d2 + carry
                carry = total // 10
                s_digits_rev.append(total)
            s_digits_rev.append(carry)

            row_correct = ["Correct Result"]
            for i in range(max_gen):
                token = s_digits_rev[i] if i < len(s_digits_rev) else 0
                row_correct.append(itos[token])
            table.add_row(*row_correct, style="bold yellow")
        except Exception as e:
            print(f"Error calculating correct result: {e}")
            pass

        if theoretical_no_carry:
            # Theoretical no-carry digits
            n1_rev = n1_str[::-1]
            n2_rev = n2_str[::-1]
            row_theory = ["No-Carry Theory"]
            for i in range(max_gen):
                d1 = int(n1_rev[i]) if i < len(n1_rev) else 0
                d2 = int(n2_rev[i]) if i < len(n2_rev) else 0
                row_theory.append(str((d1 + d2) % 10))
            table.add_row(*row_theory, style="dim")

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
            min_train_digits=1, max_train_digits=4, batch_size=args.batch_size
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        lens.train(train_loader, max_steps=args.max_steps, lr=args.lr)
        stoi, itos = dm.stoi, dm.itos
    else:
        # For logit lens we still need stoi/itos
        # We can get them from the model if they are stored there, or just create a dummy DM
        dm = AdditionDataModule()
        dm.setup()
        stoi, itos = dm.stoi, dm.itos

    # Visualize
    lens.visualize_trajectory(
        args.equation,
        stoi,
        itos,
        method=args.method,
        theoretical_no_carry=args.no_carry,
        show_probs=args.show_probs,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
