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

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0, is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# --- Lightning Module ---


class GPTLightningModule(pl.LightningModule):
    """
    This model is amended to be compatible to 1-dim PE
    """

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
        max_pos=1500,
        pad_token=-1,
        eq_token=-1,
        val_k=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Absolute Positional Embedding (1D)
        self.pos_embedding_table = nn.Embedding(max_pos, n_embd)

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

    def forward(self, idx, pos_ids, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.pos_embedding_table(pos_ids)  # (B, T, n_embd)

        x = tok_emb + pos_emb  # (B, T, n_embd)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Find '=' position to determine where to start computing loss
            if self.hparams.eq_token == -1:
                raise Exception("eq_token must be provided")
            
            # Mask: only compute loss on tokens at and after '='
            # This includes the '=' token itself (which predicts first digit)
            # and all subsequent tokens
            eq_mask = idx == self.hparams.eq_token
            
            # Find the position of '=' in each sequence
            # Create cumsum to mark everything from '=' onwards
            keep_mask = eq_mask.cumsum(dim=1) > 0
            
            # Apply mask to targets
            targets = targets.clone()
            targets[~keep_mask] = self.hparams.pad_token

            # Flatten for loss calculation
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
                ignore_index=self.hparams.pad_token,
            )

        return logits, loss

    def training_step(self, batch, batch_idx):
        idx, targets, pos_ids = batch
        logits, loss = self(idx, pos_ids, targets)
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
        idx, targets, pos_ids = batch
        
        # Calculate L (number of digits in operands)
        # Find '=' position
        eq_positions = (idx == self.hparams.eq_token).nonzero(as_tuple=True)[1]
        eq_pos = eq_positions[0].item() if len(eq_positions) > 0 else -1
        
        # Find '+' position
        plus_token = 10  # Hardcoded based on vocab
        plus_positions = (idx == plus_token).nonzero(as_tuple=True)[1]
        plus_pos = plus_positions[0].item() if len(plus_positions) > 0 else -1
        
        L = plus_pos  # Number of digits in first operand
        
        if batch_idx == 0:
            print(f"DEBUG: validation_step L={L}")
        
        logits, loss = self(idx, pos_ids, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

        # Accuracy Calculation
        eq_mask = idx == self.hparams.eq_token
        keep_mask = eq_mask.cumsum(dim=1) > 0
        
        # Apply mask to targets
        targets_masked = targets.clone()
        targets_masked[~keep_mask] = self.hparams.pad_token

        preds = torch.argmax(logits, dim=-1)
        correct = preds == targets_masked

        # 1. Token-wise Accuracy
        token_acc = (correct & keep_mask).sum().float() / keep_mask.sum().float()
        self.log(
            "val_acc_token",
            token_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=True,
        )

        # 2. Sequence-wise Accuracy
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

        # 3. Majority Voting Validation
        if self.hparams.val_k > 1:
            # Find '=' position
            eq_indices = (idx == self.hparams.eq_token).nonzero(as_tuple=True)
            eq_idx = eq_indices[1][0].item()

            prefix_idx = idx[:, : eq_idx + 1]
            prefix_pos = pos_ids[:, : eq_idx + 1]

            target_sum = targets[:, eq_idx:]  # (B, sum_len)
            target_mask = target_sum != self.hparams.pad_token
            sum_len = target_sum.shape[1]

            # Ground truth positions for the sum part
            rest_pos = pos_ids[:, eq_idx + 1 :]

            k = self.hparams.val_k
            B = idx.shape[0]

            # Repeat for k samples
            x_k = prefix_idx.repeat_interleave(k, dim=0)
            pos_k = prefix_pos.repeat_interleave(k, dim=0)
            rest_pos_k = rest_pos.repeat_interleave(k, dim=0)

            # Generate (B*k, eq_idx + 1 + sum_len)
            out_k = self.generate(
                x_k, pos_k, max_new_tokens=sum_len, external_pos=rest_pos_k
            )
            pred_sum_k = out_k[:, eq_idx + 1 :]  # (B*k, sum_len)

            # Reshape to (B, k, sum_len)
            pred_sum_k = pred_sum_k.view(B, k, sum_len)

            # 3a. Sequence-wise Majority Vote
            final_pred_seq = []
            for i in range(B):
                seqs = [tuple(s.tolist()) for s in pred_sum_k[i]]
                most_common = max(set(seqs), key=seqs.count)
                final_pred_seq.append(torch.tensor(most_common, device=self.device))
            final_pred_seq = torch.stack(final_pred_seq)  # (B, sum_len)

            # Calculate MV Seq Accuracy
            seq_mv_correct = (final_pred_seq == target_sum) | (~target_mask)
            seq_mv_acc = seq_mv_correct.all(dim=1).float().mean()
            self.log(
                "val_acc_seq_mv", seq_mv_acc, on_epoch=True, add_dataloader_idx=True
            )

            # 3b. Digit-wise Majority Vote
            final_pred_digit = torch.mode(pred_sum_k.cpu(), dim=1).values.to(
                self.device
            )  # (B, sum_len)

            digit_mv_correct = final_pred_digit == target_sum
            digit_mv_acc = (
                digit_mv_correct & target_mask
            ).sum().float() / target_mask.sum().float()
            self.log(
                "val_acc_digit_mv", digit_mv_acc, on_epoch=True, add_dataloader_idx=True
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
    def generate(self, idx, pos_ids, max_new_tokens, external_pos=None):
        """
        Generate tokens with absolute positional encoding.
        If external_pos is provided, use those positions.
        Otherwise, increment position by 1 for each new token.
        """
        for i in range(max_new_tokens):
            logits, _ = self(idx, pos_ids)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            if external_pos is not None:
                next_pos = external_pos[:, i : i + 1]
            else:
                # Simply increment position by 1
                last_pos = pos_ids[:, -1:]
                next_pos = last_pos + 1

            idx = torch.cat((idx, idx_next), dim=1)
            pos_ids = torch.cat((pos_ids, next_pos), dim=1)

        return idx
