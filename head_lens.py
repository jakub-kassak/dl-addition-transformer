import torch
import torch.nn as nn
import torch.nn.functional as F
from model import GPTLightningModule
from data import AdditionDataModule, construct_addition_batch
import math
import argparse
import os
from tqdm import tqdm
from rich.console import Console
from rich.table import Table


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
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
                update = self._newton_schulz(g, ns_steps)
                p.data.add_(update, alpha=-lr)

    @staticmethod
    def _newton_schulz(G, steps=5):
        assert len(G.shape) == 2
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16() if G.is_cuda and G.dtype != torch.float16 else G
        X /= X.norm() + 1e-7
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


class HeadResultExtractor:
    def __init__(self, model):
        self.model = model
        self.n_layer = model.hparams.n_layer
        self.n_head = model.hparams.n_head
        self.head_size = model.hparams.n_embd // model.hparams.n_head
        self.head_results = {}  # (layer, head) -> result tensor

    def get_head_results(self, idx, pos1, pos2, pos3):
        self.head_results = {}
        hooks = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                # input[0] is the concatenated head outputs before projection: (B, T, C)
                # module is sa.c_proj
                x = input[0]
                B, T, C = x.shape

                # Split x into heads
                heads = x.view(B, T, self.n_head, self.head_size)

                # weight has shape (C, C). We want to multiply each head by its part of weight.
                # output = x @ weight.T + bias
                # weight.T has shape (C, C).
                # Slice weight.T for each head.
                weight_T = module.weight.T  # (C, C)

                for h in range(self.n_head):
                    h_start = h * self.head_size
                    h_end = (h + 1) * self.head_size
                    w_h = weight_T[h_start:h_end, :]  # (head_size, C)

                    # head_output = heads[:, :, h, :] @ w_h
                    res = heads[:, :, h, :] @ w_h
                    self.head_results[(layer_idx, h)] = res.detach()

            return hook

        for l in range(self.n_layer):
            h = self.model.blocks[l].sa.c_proj.register_forward_hook(get_hook(l))
            hooks.append(h)

        with torch.no_grad():
            self.model(idx, pos1, pos2, pos3)

        for h in hooks:
            h.remove()

        return self.head_results


class HeadLens:
    def __init__(
        self, model_checkpoint_path, save_path=None, device="cpu", force_train=False
    ):
        self.device = device
        self.save_path = save_path
        self.model = GPTLightningModule.load_from_checkpoint(model_checkpoint_path)
        self.model.to(device)
        self.model.eval()
        self.extractor = HeadResultExtractor(self.model)
        self.n_layer = self.model.hparams.n_layer
        self.n_head = self.model.hparams.n_head
        self.n_embd = self.model.hparams.n_embd

        # Translators for each head in each layer
        self.translators = nn.ModuleDict()
        for l in range(self.n_layer):
            for h in range(self.n_head):
                self.translators[f"L{l}H{h}"] = TunedLensTranslator(self.n_embd)
        self.translators.to(device)

        # Vocab mapping
        self.chars = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.is_trained = False
        if save_path and os.path.exists(save_path) and not force_train:
            print(f"Loading Head Tuned Lens from {save_path}...")
            try:
                self.translators.load_state_dict(
                    torch.load(save_path, map_location=device)
                )
                self.is_trained = True
            except Exception as e:
                print(f"Failed to load: {e}. Will retrain.")

    def analyze_equation(self, equation_string):
        if "=" in equation_string:
            lhs = equation_string.split("=")[0]
        else:
            lhs = equation_string

        operands = lhs.split("+")
        max_digits = max(len(op) for op in operands)
        carry_allowance = math.ceil(math.log10(len(operands)))
        max_len = max_digits + carry_allowance

        operands_digits = []
        for op in operands:
            op_padded = op.zfill(max_len)
            d = [int(c) for c in op_padded]
            t = torch.tensor(d, dtype=torch.long).unsqueeze(0)
            operands_digits.append(t)

        idx, p1, p2, p3 = construct_addition_batch(
            operands_digits,
            self.stoi,
            random_offsets=False,
            explicit_carry=getattr(self.model.hparams, "explicit_carry", True),
        )

        idx, p1, p2, p3 = [b.to(self.device) for b in [idx, p1, p2, p3]]
        head_results = self.extractor.get_head_results(idx, p1, p2, p3)

        return idx, head_results

    def train(self, dataloader, max_steps=1000, lr=0.02):
        if self.is_trained:
            print("Tuned Lens is already trained.")
            return

        muon_params = []
        adam_params = []
        for translator in self.translators.values():
            for p in translator.parameters():
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

        optimizers = []
        if muon_params:
            optimizers.append(Muon(muon_params, lr=lr))
        if adam_params:
            optimizers.append(torch.optim.AdamW(adam_params, lr=0.01))

        print(f"Training Head Tuned Lens...")
        iterator = iter(dataloader)
        pbar = tqdm(range(max_steps), desc="Training")

        for step in pbar:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

            idx, targets, p1, p2, p3 = [b.to(self.device) for b in batch[:5]]

            with torch.no_grad():
                logits_final, _ = self.model(idx, p1, p2, p3)
                target_probs = F.softmax(logits_final, dim=-1).view(
                    -1, self.model.hparams.vocab_size
                )
                head_results = self.extractor.get_head_results(idx, p1, p2, p3)

            total_loss = 0
            for opt in optimizers:
                opt.zero_grad()

            for (l, h), res in head_results.items():
                res_in = res.detach()
                h_hat = self.translators[f"L{l}H{h}"](res_in)
                early_logits = self.model.lm_head(self.model.ln_f(h_hat)).view(
                    -1, self.model.hparams.vocab_size
                )
                loss = F.kl_div(
                    F.log_softmax(early_logits, dim=-1),
                    target_probs,
                    reduction="batchmean",
                )
                total_loss += loss

            avg_loss = total_loss / (self.n_layer * self.n_head)
            avg_loss.backward()
            for opt in optimizers:
                opt.step()
            pbar.set_postfix({"loss": f"{avg_loss.item():.4f}"})

        self.is_trained = True
        if self.save_path:
            torch.save(self.translators.state_dict(), self.save_path)

    def decode_results(self, idx, head_results, method="logit", top_k=1):
        B, T = idx.shape
        decoded = {}  # (layer, head) -> list of top predictions per token

        for (l, h), res in head_results.items():
            if method == "tuned" and self.is_trained:
                h_hat = self.translators[f"L{l}H{h}"](res)
                logits = self.model.lm_head(self.model.ln_f(h_hat))
            else:
                # Apply Logit Lens: LM_Head(LN(head_output))
                ln_res = self.model.ln_f(res)
                logits = self.model.lm_head(ln_res)

            probs = F.softmax(logits, dim=-1)
            top_v, top_i = torch.topk(probs, k=top_k, dim=-1)
            decoded[(l, h)] = (top_i[0], top_v[0])

        return decoded

    def visualize(self, equation, method="logit", top_k=1, only_prediction=True):
        idx, head_results = self.analyze_equation(equation)
        decoded = self.decode_results(idx, head_results, method=method, top_k=top_k)

        T = idx.shape[1]
        tokens = [self.itos[i.item()] for i in idx[0]]

        console = Console()

        is_predicting = False
        for t in range(T - 1):
            curr_token = tokens[t]
            next_token = tokens[t + 1]

            # Check for carry: if curr_token is a digit >= 10
            has_carry = False
            try:
                val = int(curr_token)
                if val >= 10:
                    has_carry = True
            except ValueError:
                pass

            if curr_token == "=":
                is_predicting = True

            if only_prediction and not is_predicting:
                continue

            title = f"Step {t}: Input '{curr_token}' -> Target '{next_token}'"
            if has_carry:
                title += " [bold yellow](Carry from prev)[/bold yellow]"

            table = Table(title=title)
            table.add_column("Layer", justify="right")
            for h in range(self.model.hparams.n_head):
                table.add_column(f"Head {h}", justify="center")

            for l in range(self.model.hparams.n_layer):
                row = [f"L{l}"]
                for h in range(self.model.hparams.n_head):
                    top_idx, top_prob = decoded[(l, h)]
                    pred_token = self.itos[top_idx[t].item()]
                    prob = top_prob[t].item()

                    color = "green" if pred_token == next_token else "red"
                    style = f"[{color}]" if prob > 0.05 else "[dim]"
                    row.append(
                        f"{style}{pred_token} ({prob:.1%})[/{color if prob > 0.05 else 'dim'}]"
                    )
                table.add_row(*row)

            console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--equation", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--all_tokens",
        action="store_true",
        help="Show all tokens, not just prediction phase",
    )
    parser.add_argument(
        "--method", type=str, default="logit", choices=["logit", "tuned"]
    )
    parser.add_argument("--max_train_digits", type=int, default=10)
    parser.add_argument("--max_train_operands", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--force_train", action="store_true")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(os.path.dirname(args.ckpt), "head_tuned_lens.pt")

    lens = HeadLens(
        args.ckpt,
        save_path=args.save_path,
        device=args.device,
        force_train=args.force_train,
    )

    if args.method == "tuned":
        dm = AdditionDataModule(
            min_train_digits=1,
            max_train_digits=args.max_train_digits,
            min_operands=2,
            max_operands=args.max_train_operands,
            batch_size=64,
            explicit_carry=getattr(lens.model.hparams, "explicit_carry", True),
        )
        dm.setup()
        lens.train(dm.train_dataloader(), max_steps=args.max_steps)

    lens.visualize(
        args.equation,
        method=args.method,
        top_k=args.top_k,
        only_prediction=not args.all_tokens,
    )


if __name__ == "__main__":
    main()
