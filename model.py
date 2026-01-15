import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl

# --- Compatibility Fallback ---
if not hasattr(F, "scaled_dot_product_attention"):

    def scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        B, nh, T, hs = q.shape
        att = (q @ k.transpose(-2, -1)) * (1.0 / (hs**0.5))
        if is_causal:
            mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if dropout_p > 0:
            att = F.dropout(att, p=dropout_p)
        return att @ v
else:
    scaled_dot_product_attention = F.scaled_dot_product_attention

# --- Model Components ---


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, n_embd * 3)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False, rope=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if rope is not None:
            # rope shape: (B, 1, T, head_size//2, 2, 2)
            # q, k shape: (B, n_head, T, head_size)
            shape = q.shape  # (B, n_head, T, head_size)

            # Reshape q, k for RoPE: [B, n_head, T, head_size//2, 2]
            q_rot = q.view(*shape[:-1], -1, 2)
            k_rot = k.view(*shape[:-1], -1, 2)

            # rope is (B, 1, T, head_size//2, 2, 2)
            # We want [x*cos - y*sin, x*sin + y*cos]
            # cos/sin shape: (B, 1, T, head_size//2)
            # To multiply (B, n_head, T, head_size//2, 2) by (B, 1, T, head_size//2),
            # we need to unsqueeze cos/sin to (B, 1, T, head_size//2, 1)
            # q_rot:      (B, n_head, T, head_size//2, 2)
            # cos/sin:    (B, 1, T, head_size//2, 1) -> broadcasting works
            cos = rope[..., 0, 0].unsqueeze(-1)
            sin = rope[..., 1, 0].unsqueeze(-1)

            x_raw = q_rot[..., 0:1]
            y_raw = q_rot[..., 1:2]

            q_new = torch.cat(
                [x_raw * cos - y_raw * sin, x_raw * sin + y_raw * cos], dim=-1
            )

            x_k = k_rot[..., 0:1]
            y_k = k_rot[..., 1:2]
            k_new = torch.cat([x_k * cos - y_k * sin, x_k * sin + y_k * cos], dim=-1)

            # Reshape back to (B, n_head, T, head_size)
            q = q_new.view(shape)
            k = k_new.view(shape)

        if return_weights:
            att = (q @ k.transpose(-2, -1)) * (1.0 / ((C // self.n_head) ** 0.5))
            mask = torch.tril(torch.ones(T, T, device=q.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        else:
            y = scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0,
                is_causal=True,
            )
            att = None

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_weights:
            return y, att
        return y


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout, n_ffwd_width=4, n_ffwd_depth=1):
        super().__init__()
        layers = []
        hidden_dim = n_ffwd_width * n_embd

        # Input layer
        layers.append(nn.Linear(n_embd, hidden_dim))
        layers.append(nn.ReLU())

        # Additional hidden layers (if depth > 1)
        for _ in range(n_ffwd_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, n_embd))
        layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, n_ffwd_width=4, n_ffwd_depth=1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffwd = FeedFoward(n_embd, dropout, n_ffwd_width, n_ffwd_depth)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_weights=False, rope=None):
        if return_weights:
            out, att = self.sa(self.ln1(x), return_weights=True, rope=rope)
            x = x + out
            x = x + self.ffwd(self.ln2(x))
            return x, att
        else:
            x = x + self.sa(self.ln1(x), rope=rope)
            x = x + self.ffwd(self.ln2(x))
            return x


# --- Lightning Module ---


class GPTLightningModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
        learning_rate,
        n_ffwd_width=4,
        n_ffwd_depth=1,
        total_steps=10000,
        max_pos2=1500,
        pad_token=-1,
        eq_token=-1,
        rope_theta=20000,
        pos_emb_type="rope",  # "learned", "rope", "mixed"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.theta = rope_theta

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Learned Absolute PE tables
        # pos1: Block index (max operands ~20 for safety)
        self.pos1_embedding_table = nn.Embedding(50, n_embd)
        # pos2: Significance index (max 1500 digits)
        self.pos2_embedding_table = nn.Embedding(max_pos2, n_embd)
        # pos3: Segment type (1, 2, 3)
        self.pos3_embedding_table = nn.Embedding(10, n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, dropout, n_ffwd_width, n_ffwd_depth)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie weights
        self.lm_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def rope(self, pos, head_size):
        assert head_size % 2 == 0
        scale = (
            torch.arange(0, head_size, 2, dtype=pos.dtype, device=pos.device)
            / head_size
        )
        radians = 1.0 / (self.theta**scale)
        out = torch.einsum("...n,d->...nd", pos, radians)
        cos, sin = torch.cos(out), torch.sin(out)
        out = torch.stack([cos, -sin, sin, cos], dim=-1)
        out = out.reshape(*out.shape[:-1], 2, 2)
        return out

    def prepare_multidim_rope(self, pos_list, dims_list):
        """
        Splits head_dim into chunks determined by dims_list.
        Applies RoPE to each chunk using corresponding pos from pos_list.
        Concatenates results.
        pos_list: [pos1, pos2, ...]
        dims_list: [dim1, dim2, ...] sums to head_size
        """
        assert len(pos_list) == len(dims_list)
        rope_all = []

        for pos, dim in zip(pos_list, dims_list):
            if dim == 0:
                continue
            # Each dim chunk is further split for cos/sin (needs mixed logic?)
            # No, rope() takes head_size (dim) and returns (..., dim//2, 2, 2)
            if dim % 2 != 0:
                raise ValueError(f"RoPE chunk dimension {dim} must be even")

            r = self.rope(pos, dim)
            rope_all.append(r)

        # rope has dimensions (B, T, dims//2, 2, 2)
        # We simply concatenate them along the feature dimension (dim 2)
        if all(x.shape == rope_all[0].shape for x in rope_all[1:]):
            stacked = torch.stack(rope_all, dim=3)
            full_rope = stacked.flatten(2, 3)
        else:
            full_rope = torch.cat(rope_all, dim=2)  # (B, T, head_size//2, 2, 2)
        return full_rope.unsqueeze(1)  # (B, 1, T, head_size//2, 2, 2)

    def forward(
        self, idx, pos1_ids, pos2_ids, pos3_ids, targets=None, return_weights=False
    ):
        B, T = idx.shape
        x = self.token_embedding_table(idx)  # (B, T, n_embd)

        head_size = self.hparams.n_embd // self.hparams.n_head
        rope = None

        if self.hparams.pos_emb_type == "learned":
            # Absolute for all
            p1 = self.pos1_embedding_table(pos1_ids)
            p2 = self.pos2_embedding_table(pos2_ids)
            p3 = self.pos3_embedding_table(pos3_ids)
            x = x + p1 + p2 + p3

        elif self.hparams.pos_emb_type == "rope":
            # Multidimensional RoPE for pos1, pos2, pos3
            # Split head logic: e.g. 1/3, 1/3, 1/3? or 1/4, 2/4, 1/4?
            # Let's do even split if possible, or remainder to last.

            # For simplicity: Split into 3 chunks for 3 positions.
            # Dims must be even.

            d = head_size
            d1 = (d // 3) // 2 * 2  # make even
            d2 = (d // 3) // 2 * 2
            d3 = d - d1 - d2

            rope = self.prepare_multidim_rope(
                [pos1_ids, pos2_ids, pos3_ids], [d1, d2, d3]
            )

        elif self.hparams.pos_emb_type == "mixed":
            # "RoPE for pos1,2 and absolute learned PE for pos3"

            # Add absolute pos3
            p3 = self.pos3_embedding_table(pos3_ids)
            x = x + p3

            # RoPE for pos1, pos2
            d = head_size
            d1 = (d // 2) // 2 * 2
            d2 = d - d1

            rope = self.prepare_multidim_rope([pos1_ids, pos2_ids], [d1, d2])

        all_attn = []
        for block in self.blocks:
            if return_weights:
                x, att = block(x, return_weights=True, rope=rope)
                all_attn.append(att)
            else:
                x = block(x, rope=rope)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Mask out loss for tokens strictly before '='
            # We want to predict starting from the token AFTER '='
            # Actually, with new MultiOperand format:
            # Input=... #=...
            # We want to predict Scratchpad and Result.
            # So mask out Input part.
            # Input part has pos3_ids == 1.
            # Scratchpad (pos3==2) and Result (pos3==3) should be trained on.
            # Also the first '=' (end of input) predicts first token of scratchpad.

            # Valid targets are where we are NOT in Input (or the transition from input).
            # The transition token is '=' (at end of inputs).
            # pos3=1 usually.

            # Simple rule: Train on everything EXCEPT the Input tokens.
            # Input tokens correspond to pos3==1 (and pos1 < N? no pos1 increments).

            # Wait, let's just use the `eq_token` logic but extended.
            # We want to predict everything after the first `=`.
            # First `=` is part of Input (pos3=1).
            # So if input token is `=` and pos3=1, we predict next (start of scratchpad).
            # If input token is in Scratchpad (pos3=2), we predict next.
            # If input token is in Result (pos3=3), we predict next.

            # So Keep Mask:
            # (idx == eq_token AND pos3 == 1) OR (pos3 >= 2)

            # Also, we likely have padding in batch. Mask padding too. (targets != pad_token) covers it?
            # Just relying on ignore_index=pad_token in cross_entropy is enough for padding?
            # But we must mask out the Input Questions themselves (N1+N2...).

            if self.hparams.eq_token == -1:
                raise Exception("eq_token must be provided")

            is_first_eq = (idx == self.hparams.eq_token) & (pos3_ids == 1)
            is_scratchpad_or_res = pos3_ids >= 2
            keep_mask = is_first_eq | is_scratchpad_or_res

            # Apply mask to targets
            targets = targets.clone()
            targets[~keep_mask] = self.hparams.pad_token

            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
                ignore_index=self.hparams.pad_token,
            )

        if return_weights:
            return logits, loss, all_attn
        return logits, loss

    def training_step(self, batch, batch_idx):
        idx, targets, pos1_ids, pos2_ids, pos3_ids = batch
        logits, loss = self(idx, pos1_ids, pos2_ids, pos3_ids, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 6:
            idx, targets, pos1_ids, pos2_ids, pos3_ids, config_idx = batch
        else:
            # Fallback for sanity checks or old datasets
            idx, targets, pos1_ids, pos2_ids, pos3_ids = batch
            config_idx = None

        logits, loss = self(idx, pos1_ids, pos2_ids, pos3_ids, targets)

        # Determine metric suffix
        suffix = ""
        if config_idx is not None and hasattr(
            self.trainer.datamodule, "val_config_names"
        ):
            # Assuming purely sequential batches from one config, config_idx should be identical in batch
            # We take the first one.
            c_id = config_idx
            name = self.trainer.datamodule.val_config_names[c_id]
            suffix = f"/{name}"

        # Log Loss
        self.log(
            f"val_loss{suffix}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # Accuracy Calculation
        is_first_eq = (idx == self.hparams.eq_token) & (pos3_ids == 1)
        is_scratchpad_or_res = pos3_ids >= 2
        keep_mask = is_first_eq | is_scratchpad_or_res

        targets_masked = targets.clone()
        targets_masked[~keep_mask] = self.hparams.pad_token

        # Also ensure we don't count padding in accuracy
        # The mask setup above sets non-predict tokens to pad_token.
        # But if the TARGET itself was pad_token (from data loader), it remains pad_token.
        # So check targets_masked != pad_token
        valid_acc_mask = targets_masked != self.hparams.pad_token

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets_masked

        # 1. Token-wise Accuracy
        token_acc = (
            correct & valid_acc_mask
        ).sum().float() / valid_acc_mask.sum().float()
        self.log(
            f"val_acc_token{suffix}",
            token_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # 2. Sequence-wise Accuracy
        row_correct = (correct | (~valid_acc_mask)).all(dim=1)
        seq_acc = row_correct.float().mean()
        self.log(
            f"val_acc_seq{suffix}",
            seq_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return loss

    def on_validation_epoch_end(self):
        # We need to manually aggregate because we are logging dynamic keys
        # Actually, PL handles aggregation of "val_loss/val_L1_N2" automatically across the epoch if identifiers match.
        # We want to compute an overall "val_avg_seq_acc" for checkpointing.

        metrics = self.trainer.callback_metrics
        # Look for any key starting with val_acc_seq/val_
        seq_accs = [v for k, v in metrics.items() if "val_acc_seq/val_" in k]
        losses = [v for k, v in metrics.items() if "val_loss/val_" in k]

        if seq_accs:
            # Helper for Weighted Log Mean
            def calculate_weighted_log_mean(values, weights, epsilon=1e-6):
                # values: list or tensor of accuracies (0 to 1)
                # weights: list or tensor of weights
                v = torch.as_tensor(values, device=self.device, dtype=torch.float32)
                w = torch.as_tensor(weights, device=self.device, dtype=torch.float32)

                # Metric = exp( sum(w * log(v + eps)) / sum(w) )
                weighted_log_sum = torch.sum(w * torch.log(v + epsilon))
                sum_weights = torch.sum(w)
                return torch.exp(weighted_log_sum / sum_weights)

            # Parse configs to get weights
            import re

            # Prepare data
            configs = []
            valid_seq_vals = []
            valid_token_vals = []

            # We iterate through the metrics keys to find matching pairs
            # This relies on the fact that for every 'val_acc_seq/X', there should be 'val_acc_token/X' ideally,
            # but we can process them independently.

            # Let's gather all configs present in metrics
            config_map = {}  # name -> {L, N, seq_val, token_val}

            for k, v in metrics.items():
                if k.startswith("val_acc_seq/val_"):
                    name = k.replace("val_acc_seq/", "")
                    match = re.match(r"val_L(\d+)_N(\d+)", name)
                    if match:
                        if name not in config_map:
                            config_map[name] = {}
                        config_map[name]["L"] = int(match.group(1))
                        config_map[name]["N"] = int(match.group(2))
                        config_map[name]["seq"] = v

                if k.startswith("val_acc_token/val_"):
                    name = k.replace("val_acc_token/", "")
                    match = re.match(r"val_L(\d+)_N(\d+)", name)
                    if match:
                        if name not in config_map:
                            config_map[name] = {}
                        config_map[name]["L"] = int(match.group(1))
                        config_map[name]["N"] = int(match.group(2))
                        config_map[name]["token"] = v

            # Arrays for computation
            seq_values = []
            token_values = []
            weights_combined = []  # N * L
            weights_unit = []  # 1

            for name, data in config_map.items():
                if "seq" in data and "token" in data:
                    seq_values.append(data["seq"])
                    token_values.append(data["token"])
                    L, N = data["L"], data["N"]
                    weights_combined.append(L * N)
                    weights_unit.append(1.0)

            if seq_values:
                # 1. Combined (w = L * N)
                self.log(
                    "val_wlm_combined_seq",
                    calculate_weighted_log_mean(seq_values, weights_combined),
                    prog_bar=True,
                )
                self.log(
                    "val_wlm_combined_token",
                    calculate_weighted_log_mean(token_values, weights_combined),
                    prog_bar=True,
                )

                # 2. Unweighted (w = 1)
                self.log(
                    "val_wlm_unweighted_seq",
                    calculate_weighted_log_mean(seq_values, weights_unit),
                    prog_bar=True,
                )
                self.log(
                    "val_wlm_unweighted_token",
                    calculate_weighted_log_mean(token_values, weights_unit),
                    prog_bar=True,
                )

            avg_seq_acc = torch.stack(seq_accs).mean()
            self.log("val_avg_seq_acc", avg_seq_acc, prog_bar=True)
            self.log(
                "val_avg_loss",
                torch.stack(losses).mean() if losses else 0.0,
                prog_bar=False,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.hparams.total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @torch.no_grad()
    def generate(
        self,
        idx,
        pos1_ids,
        pos2_ids,
        pos3_ids,
        max_new_tokens,
        special_tokens=None,
        offset=0,
    ):
        """
        Autoregressively generate tokens with correct 2D-Positional Update logic.
        special_tokens: dict with keys '>', '+', '=', '#' mapping to IDs.
        """
        assert special_tokens is not None, "Must provide special_tokens dict"
        greater_token = special_tokens[">"]
        hash_token = special_tokens["#"]
        eq_token = self.hparams.eq_token

        for _ in range(max_new_tokens):
            # 1. Forward pass
            logits, _ = self(idx, pos1_ids, pos2_ids, pos3_ids)
            last_logits = logits[:, -1, :]
            # probs = F.softmax(last_logits, dim=-1)
            # idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.argmax(last_logits, dim=-1, keepdim=True)

            # 2. Determine Next Positions based on WHAT WE JUST GENERATED (idx_next)
            # and the context of the LAST token we were at.

            last_token = idx[:, -1]
            last_p1 = pos1_ids[:, -1]
            last_p2 = pos2_ids[:, -1]
            last_p3 = pos3_ids[:, -1]

            # Base logic: Continue segment (increment p2, keep p1, p3=2)
            new_p1 = last_p1.clone()
            new_p2 = last_p2 + 1
            new_p3 = torch.full_like(last_p3, 2)

            # Masks for Previous Token Context
            is_prev_eq = last_token == eq_token
            is_prev_greater = last_token == greater_token

            # Context Case 1: Previous was '=' (End of Input -> Start S0)
            # p1 becomes 1, p2 becomes 1 + offset
            new_p1 = torch.where(is_prev_eq, torch.tensor(1, device=idx.device), new_p1)
            new_p2 = torch.where(
                is_prev_eq, torch.tensor(1 + offset, device=idx.device), new_p2
            )

            # Context Case 2: Previous was '>' (Separator -> Start Next Segment)
            # p1 stays as is (it was already incremented when generating '>'), p2 resets to 1 + offset
            new_p2 = torch.where(
                is_prev_greater, torch.tensor(1 + offset, device=idx.device), new_p2
            )

            # Masks for Current Generated Token
            is_curr_greater = idx_next.squeeze(-1) == greater_token
            is_curr_hash = idx_next.squeeze(-1) == hash_token

            # Current Case 1: Generated '>' (Separator)
            # p1 increments, p2 becomes 0
            new_p1 = torch.where(is_curr_greater, last_p1 + 1, new_p1)
            new_p2 = torch.where(
                is_curr_greater, torch.tensor(0, device=idx.device), new_p2
            )

            # Current Case 2: Generated '#' (End Token)
            # p1 increments, p2 becomes 0
            new_p1 = torch.where(is_curr_hash, last_p1 + 1, new_p1)
            new_p2 = torch.where(
                is_curr_hash, torch.tensor(0, device=idx.device), new_p2
            )

            # Reshape for concatenation
            new_p1 = new_p1.unsqueeze(1)
            new_p2 = new_p2.unsqueeze(1)
            new_p3 = new_p3.unsqueeze(1)

            idx = torch.cat((idx, idx_next), dim=1)
            pos1_ids = torch.cat((pos1_ids, new_p1), dim=1)
            pos2_ids = torch.cat((pos2_ids, new_p2), dim=1)
            pos3_ids = torch.cat((pos3_ids, new_p3), dim=1)

            if is_curr_hash.all():
                break

        return idx, pos1_ids, pos2_ids, pos3_ids
