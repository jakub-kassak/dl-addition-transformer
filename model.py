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
            # dtype = q.dtype
            shape = q.shape
            q = q.reshape(*shape[:-1], -1, 1, 2)
            k = k.reshape(*shape[:-1], -1, 1, 2)
            q = rope[..., 0] * q[..., 0] + rope[..., 1] * q[..., 1]
            k = rope[..., 0] * k[..., 0] + rope[..., 1] * k[..., 1]
            q = q.reshape(shape)
            k = k.reshape(shape)

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
    ):
        super().__init__()
        self.save_hyperparameters()
        self.theta = rope_theta

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

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
        scale = torch.arange(0, head_size, 2, dtype=pos.dtype, device=pos. device) / head_size
        radians = 1. / (self.theta ** scale)
        out = torch.einsum("...n,d->...nd", pos, radians)
        cos, sin = torch.cos(out), torch.sin(out)
        out = torch.stack([cos, -sin, sin, cos], dim=-1)
        out = out.reshape(*out.shape[:-1], 2, 2)
        return out
    
    def prepare_rope(self, pos1_ids, pos2_ids):
        head_size = self.hparams.n_embd // self.hparams.n_head // 4
        rope1 = self.rope(pos1_ids, head_size)
        rope2 = self.rope(pos2_ids, head_size)
        rope = torch.cat([rope1, rope2, rope1, rope2], dim=-3)
        return rope.unsqueeze(1)

    def forward(self, idx, pos1_ids, pos2_ids, targets=None, return_weights=False):
        B, T = idx.shape

        x = self.token_embedding_table(idx)  # (B, T, n_embd)
        rope = self.prepare_rope(pos1_ids, pos2_ids)

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
            # The input that predicts the first result digit is '=' itself.
            # So valid inputs are: (input == eq_token) OR (pos1 == 3)
            # pos1 == 3 corresponds to result digits.

            # Mask: 1 for valid positions, 0 for ignored
            # Make sure eq_token is valid
            if self.hparams.eq_token == -1:
                # If not provided, assume we train on everything (backward compat)
                raise Exception("eq_token must be provided")
            else:
                is_eq = idx == self.hparams.eq_token
                is_result = pos1_ids == 3
                keep_mask = is_eq | is_result

                # Apply mask to targets: set ignored to pad_token
                # We clone targets to avoid modifying the original tensor if it's used elsewhere
                targets = targets.clone()
                targets[~keep_mask] = self.hparams.pad_token

            # Flatten for loss calculation
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
        idx, targets, pos1_ids, pos2_ids = batch
        logits, loss = self(idx, pos1_ids, pos2_ids, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train/max_logits",
            logits.max(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, targets, pos1_ids, pos2_ids = batch
        logits, loss = self(idx, pos1_ids, pos2_ids, targets)

        # Determine dataset name via dataloader_idx
        # But here we can just log with dataloader_idx or rely on automatic suffixes?
        # Lightning adds /dataloader_idx_N

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

        # Accuracy Calculation
        # Mask out padding
        if self.hparams.eq_token == -1:
            raise Exception("eq_token must be provided")
        else:
            is_eq = idx == self.hparams.eq_token
            is_result = pos1_ids == 3
            keep_mask = is_eq | is_result

            # Apply mask to targets: set ignored to pad_token
            # We clone targets to avoid modifying the original tensor if it's used elsewhere
            targets = targets.clone()
            targets[~keep_mask] = self.hparams.pad_token

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets

        # 1. Token-wise Accuracy
        # Only count non-padding tokens
        token_acc = (correct & keep_mask).sum().float() / keep_mask.sum().float()
        self.log(
            "val_acc_token",
            token_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

        # 2. Sequence-wise Accuracy (Whole Result)
        # A sequence is correct if ALL its non-masked tokens are correct
        # dim=1 is sequence dimension
        # correct shape: (B, T)
        # mask shape: (B, T)

        # We need to ensure that for a given row, all positions where mask=True are correct.
        # If mask=False, it doesn't matter (treat as correct).
        # So check: correct | (~mask)
        row_correct = (correct | (~keep_mask)).all(dim=1)
        seq_acc = row_correct.float().mean()
        self.log(
            "val_acc_seq",
            seq_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

        return loss

    def on_validation_epoch_end(self):
        # Calculate average metrics across all validation dataloaders
        metrics = self.trainer.callback_metrics

        # Collect all per-dataloader metrics
        seq_accs = [v for k, v in metrics.items() if "val_acc_seq/dataloader_idx_" in k]
        losses = [v for k, v in metrics.items() if "val_loss/dataloader_idx_" in k]

        if seq_accs:
            avg_seq_acc = torch.stack(seq_accs).mean()
            self.log("val_avg_seq_acc", avg_seq_acc, prog_bar=True)

        if losses:
            avg_loss = torch.stack(losses).mean()
            self.log("val_avg_loss", avg_loss, prog_bar=False)

    def on_before_optimizer_step(self, optimizer):
        # Manually log gradient norms since track_grad_norm is deprecated
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.log("grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        # OneCycleLR Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.hparams.total_steps,
            pct_start=0.1,  # Warmup for 10% of steps
            anneal_strategy="cos",
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update per step
            },
        }

    @torch.no_grad()
    def generate(self, idx, pos1_ids, pos2_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx, pos1_ids, pos2_ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            next_pos1 = pos1_ids[:, -1:]
            next_pos2 = pos2_ids[:, -1:] - 1

            idx = torch.cat((idx, idx_next), dim=1)
            pos1_ids = torch.cat((pos1_ids, next_pos1), dim=1)
            pos2_ids = torch.cat((pos2_ids, next_pos2), dim=1)

        return idx
