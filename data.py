import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
import random
import math


class MultiOperandAdditionDataset(IterableDataset):
    """
    Generates multi-operand addition examples.
    Format: N1 + N2 + ... + Nn = S0 + S1 + ... + Sn = R #
    """

    def __init__(
        self,
        min_digits,
        max_digits,
        batch_size,
        min_operands=2,
        max_operands=5,
        data_mode="variable",  # "padded" or "variable"
        offset_range=100,
        random_offsets=True,
    ):
        super().__init__()
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.batch_size = batch_size
        self.min_operands = min_operands
        self.max_operands = max_operands
        self.data_mode = data_mode
        self.offset_range = offset_range
        self.random_offsets = random_offsets

        # Vocab: 0-19 (digits+carry), +, =, >, #
        self.chars = [str(i) for i in range(20)] + ["+", "=", ">", "#"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        self.plus_token = self.stoi["+"]
        self.eq_token = self.stoi["="]
        self.greater_token = self.stoi[">"]
        self.hash_token = self.stoi["#"]

        # Pad token for batching variable lengths
        self.pad_token = -1

    def __iter__(self):
        while True:
            yield self.generate_batch()

    def generate_batch(self):
        B = self.batch_size
        N = random.randint(self.min_operands, self.max_operands)

        # 1. Determine max_len (padding everything to this length)
        # Allowance for carry: log10(max_operands)
        carry_allowance = math.ceil(math.log10(self.max_operands))
        max_len = self.max_digits + carry_allowance

        # 2. Generate Op Digits with strict padding
        # In Figure 4, operands are zero-padded to match the length of the result.
        operands_digits = []
        for _ in range(N):
            # Generate random number with 1 to max_digits
            L_actual = random.randint(self.min_digits, self.max_digits)
            d = torch.randint(0, 10, (B, L_actual))

            # Pad to max_len with zeros (at the front/MSB for input)
            padding = torch.zeros((B, max_len - L_actual), dtype=torch.long)
            full_d = torch.cat([padding, d], dim=1)
            operands_digits.append(full_d)

        # 3. Compute Partial Sums (S0=0, S1=N1, ..., SN=Sum(N1..NN))
        # Each Si is stored LSB-first in scratchpad_segments (length max_len)
        scratchpad_segments = []
        current_sum = torch.zeros((B, max_len), dtype=torch.long)
        scratchpad_segments.append(current_sum.clone())

        for operand in operands_digits:
            carry = torch.zeros(B, dtype=torch.long)
            for i in range(max_len-1, -1, -1):
                d_acc = current_sum[:, i]
                d_op = operand[:, i]

                total = d_acc + d_op + carry
                carry = total // 10
                current_sum[:, i] = total

            scratchpad_segments.append(current_sum.flip(1))
            current_sum %= 10

        # 4. Construct Full Sequence
        # Input: N1 + N2 + ... + Nk =
        # SP: S0 > S1 > ... > Sn #
        p1_list, p2_list, p3_list, tokens_list = [], [], [], []

        plus = torch.full((B, 1), self.plus_token, dtype=torch.long)
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long)
        greater = torch.full((B, 1), self.greater_token, dtype=torch.long)
        hash_t = torch.full((B, 1), self.hash_token, dtype=torch.long)

        # -- Input Phase --
        for k in range(N):
            # Operand Nk gets PosID2 = k+1
            L = operands_digits[k].shape[1]
            tokens_list.append(operands_digits[k])
            p1_list.append(torch.full((B, L), k + 1, dtype=torch.long))
            # PosID1: MSB first uses decreasing IDs. Separator gets 1.
            # Figure 4 shows 4 3 2 1 for 3-digit number + sep.
            # So IDs are (L+1) down to 2, then 1 for separator.
            ids = torch.arange(L, 0, -1).unsqueeze(0).expand(B, -1)
            p2_list.append(ids)
            p3_list.append(torch.full((B, L), 1, dtype=torch.long))

            if k < N - 1:
                # Separator [+]
                tokens_list.append(plus)
                p1_list.append(torch.full((B, 1), k + 1, dtype=torch.long))
                p2_list.append(torch.zeros((B, 1), dtype=torch.long))
                p3_list.append(torch.full((B, 1), 1, dtype=torch.long))
            else:
                # Separator [=]
                tokens_list.append(eq)
                p1_list.append(torch.full((B, 1), k + 1, dtype=torch.long))
                p2_list.append(torch.zeros((B, 1), dtype=torch.long))
                p3_list.append(torch.full((B, 1), 1, dtype=torch.long))

        # -- Scratchpad Phase --
        # S0 gets PosID2 = 1 (couples with N1)
        # S_k gets PosID2 = k+1 (couples with N_{k+1})
        for k, seg in enumerate(scratchpad_segments):
            L = seg.shape[1]
            tokens_list.append(seg)
            p1_list.append(torch.full((B, L), k + 1, dtype=torch.long))
            # PosID1: LSB first uses increasing IDs. 1, 2, ..., L
            ids = torch.arange(1, L + 1).unsqueeze(0).expand(B, -1)
            p2_list.append(ids)
            p3_list.append(torch.full((B, L), 2, dtype=torch.long))

            if k < len(scratchpad_segments) - 1:
                # Separator [>]
                tokens_list.append(greater)
                p1_list.append(torch.full((B, 1), k + 1, dtype=torch.long))
                # Figure 4 shows > gets ID L+1 (4 for 3 digits)
                p2_list.append(torch.full((B, 1), L + 1, dtype=torch.long))
                p3_list.append(torch.full((B, 1), 2, dtype=torch.long))

        # End Token [#]
        tokens_list.append(hash_t)
        p1_list.append(torch.full((B, 1), N + 1, dtype=torch.long))  # Final block
        p2_list.append(torch.zeros((B, 1), dtype=torch.long))
        p3_list.append(torch.full((B, 1), 2, dtype=torch.long))

        full_seq = torch.cat(tokens_list, dim=1)
        pos1 = torch.cat(p1_list, dim=1)
        pos2 = torch.cat(p2_list, dim=1)
        pos3 = torch.cat(p3_list, dim=1)

        if self.random_offsets:
            offsets = torch.randint(0, self.offset_range, (B, 1))
            pos2 = pos2 + offsets

        x = full_seq[:, :-1]
        y = full_seq[:, 1:]
        p1 = pos1[:, :-1]
        p2 = pos2[:, :-1]
        p3 = pos3[:, :-1]

        return x, y, p1, p2, p3


class AdditionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        min_train_digits=1,
        max_train_digits=7,
        max_val_digits=15,
        val_step=3,
        batch_size=64,
        num_workers=0,
        curriculum_start=3,
        val_batch_size=None,
        random_offsets=True,
        min_operands=2,
        max_operands=5,
        max_val_operands=10,
        val_operand_step=2,
        data_mode="variable",
        curriculum_operands_start=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = 24  # 0-19, +, =, >, #

    def setup(self, stage=None):
        # Training dataset: range [min_train, max_train]
        self.train_ds = MultiOperandAdditionDataset(
            self.hparams.min_train_digits,
            self.hparams.max_train_digits,
            self.hparams.batch_size,
            min_operands=self.hparams.min_operands,
            max_operands=self.hparams.max_operands,
            data_mode=self.hparams.data_mode,
            random_offsets=self.hparams.random_offsets,
        )

        # Normalize curriculum start immediately
        # This ensures setup() results are consistent regardless of max_train_digits
        initial_max = min(self.hparams.curriculum_start, self.hparams.max_train_digits)
        self.train_ds.max_digits = initial_max

        if self.hparams.curriculum_operands_start is not None:
            initial_ops = min(
                self.hparams.curriculum_operands_start, self.hparams.max_operands
            )
            self.train_ds.max_operands = initial_ops
        self.stoi = self.train_ds.stoi
        self.itos = self.train_ds.itos
        self.vocab_size = self.train_ds.vocab_size

        # Validation batch size
        self.val_bs = self.hparams.val_batch_size or self.hparams.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=None, num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        # Dynamic validation sets based on current curriculum progress
        current_max_digits = self.train_ds.max_digits
        max_val_digits = self.hparams.max_val_digits
        min_train_digits = self.hparams.min_train_digits
        val_step_digits = self.hparams.val_step

        max_train_ops = self.hparams.max_operands
        max_val_ops = self.hparams.max_val_operands
        val_step_ops = self.hparams.val_operand_step

        # 1. Lengths Grid
        lengths = sorted(
            list(
                set(
                    list(
                        range(
                            min_train_digits,
                            max_val_digits + 1,
                            max(1, val_step_digits),
                        )
                    )
                    + [current_max_digits, min(current_max_digits + 1, max_val_digits)]
                )
            )
        )

        # 2. Operands Grid
        # We validate on a fixed set of operand counts to see generalization
        # Including: In-distribution (max_train_ops) and OOD (up to max_val_ops)
        operands = sorted(
            list(
                set(
                    [max_train_ops]
                    + list(
                        range(
                            max_train_ops + val_step_ops,
                            max_val_ops + 1,
                            max(1, val_step_ops),
                        )
                    )
                )
            )
        )

        dataloaders = []
        self.val_names = []  # Reset and repopulate

        # We create a specific selection of (L, N) pairs to avoid exponential explosion
        # e.g., for each L, validate on max_train_ops and maybe max_val_ops

        for L in lengths:
            for N in operands:
                ds = MultiOperandAdditionDataset(
                    L,
                    L,
                    self.val_bs,
                    min_operands=N,
                    max_operands=N,
                    data_mode=self.hparams.data_mode,
                    random_offsets=self.hparams.random_offsets,
                )
                dataloaders.append(
                    DataLoader(
                        ds,
                        batch_size=None,
                        num_workers=self.hparams.num_workers,
                        persistent_workers=self.hparams.num_workers > 0,
                    )
                )
                self.val_names.append(f"val_L{L}_N{N}")

        return dataloaders
