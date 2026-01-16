import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
import random
import math


def decode_batch(x, itos):
    x = x.tolist()
    for i in range(len(x)):
        x[i] = " ".join([f"{itos[i]:<2}" for i in x[i]])
    return x


def construct_addition_batch(
    operands_digits, stoi, random_offsets=False, offset_range=100, explicit_carry=True
):
    """
    Constructs the full sequence and positional encodings given the operands digits.
    operands_digits: List of tensors [ (B, max_len), ... ]
    stoi: dict mapping token to index
    """
    B = operands_digits[0].shape[0]
    max_len = operands_digits[0].shape[1]
    N_Ops = len(operands_digits)

    plus_token = stoi["+"]
    eq_token = stoi["="]
    greater_token = stoi[">"]
    hash_token = stoi["#"]

    # 3. Compute Partial Sums (S0=0, S1=N1, ..., SN=Sum(N1..NN))
    # Each Si is stored LSB-first in scratchpad_segments (length max_len)
    scratchpad_segments = []
    current_sum = torch.zeros((B, max_len), dtype=torch.long)
    scratchpad_segments.append(current_sum.clone())

    for operand in operands_digits:
        carry = torch.zeros(B, dtype=torch.long)
        for i in range(max_len - 1, -1, -1):
            d_acc = current_sum[:, i]
            d_op = operand[:, i]

            total = d_acc + d_op + carry
            carry = total // 10
            if not explicit_carry:
                total %= 10
            current_sum[:, i] = total

        scratchpad_segments.append(current_sum.flip(1))
        current_sum %= 10

    # 4. Construct Full Sequence
    # Input: N1 + N2 + ... + Nk =
    # SP: S0 > S1 > ... > Sn #
    p1_list, p2_list, p3_list, tokens_list = [], [], [], []

    plus = torch.full((B, 1), plus_token, dtype=torch.long)
    eq = torch.full((B, 1), eq_token, dtype=torch.long)
    greater = torch.full((B, 1), greater_token, dtype=torch.long)
    hash_t = torch.full((B, 1), hash_token, dtype=torch.long)

    # -- Input Phase --
    for k in range(N_Ops):
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

        if k < N_Ops - 1:
            # Separator [+]
            tokens_list.append(plus)
        else:
            # Separator [=]
            tokens_list.append(eq)
        p1_list.append(torch.full((B, 1), k + 2, dtype=torch.long))
        p2_list.append(torch.full((B, 1), L + 1, dtype=torch.long))
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
            p1_list.append(torch.full((B, 1), k + 2, dtype=torch.long))
            # Figure 4 shows > gets ID L+1 (4 for 3 digits)
            p2_list.append(torch.full((B, 1), 0, dtype=torch.long))
            p3_list.append(torch.full((B, 1), 2, dtype=torch.long))

    # End Token [#]
    tokens_list.append(hash_t)
    p1_list.append(torch.full((B, 1), N_Ops + 1, dtype=torch.long))  # Final block
    p2_list.append(torch.zeros((B, 1), dtype=torch.long))
    p3_list.append(torch.full((B, 1), 2, dtype=torch.long))

    full_seq = torch.cat(tokens_list, dim=1)
    pos1 = torch.cat(p1_list, dim=1)
    pos2 = torch.cat(p2_list, dim=1)
    pos3 = torch.cat(p3_list, dim=1)

    if random_offsets:
        offsets = torch.randint(0, offset_range, (B, 1))
        pos2 = pos2 + offsets

    return full_seq, pos1, pos2, pos3


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
        explicit_carry=True,
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
        self.explicit_carry = explicit_carry

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
        B_size = self.batch_size
        N_Ops_max = 0
        N_Dig_max = 0
        batch = []
        sub_batch_size = 8
        for i in range(0, B_size, sub_batch_size):
            B = min(sub_batch_size, B_size - i)
            N_Ops = random.randint(self.min_operands, self.max_operands)
            N_Dig = random.randint(self.min_digits, self.max_digits)
            N_Ops_max = max(N_Ops, N_Ops_max)
            N_Dig_max = max(N_Dig, N_Dig_max)
            # with open("logs/log.txt", "a") as f:
            #     f.write(f"OPS: {self.min_operands}-{self.max_operands} => {N_Ops}, Digits: {self.min_digits}-{self.max_digits} => {N_Dig}\n")

            # 1. Determine max_len (padding everything to this length)
            # Allowance for carry: log10(max_operands)
            carry_allowance = math.ceil(math.log10(N_Ops))
            max_len = N_Dig + carry_allowance

            # 2. Generate Op Digits with strict padding
            # In Figure 4, operands are zero-padded to match the length of the result.
            operands_digits = []
            for _ in range(N_Ops):
                # Generate random number with 1 to max_digits
                d = torch.randint(0, 10, (B, N_Dig))

                # Pad to max_len with zeros (at the front/MSB for input)
                padding = torch.zeros((B, max_len - N_Dig), dtype=torch.long)
                full_d = torch.cat([padding, d], dim=1)
                operands_digits.append(full_d)

            full_seq, pos1, pos2, pos3 = construct_addition_batch(
                operands_digits,
                self.stoi,
                random_offsets=self.random_offsets,
                offset_range=self.offset_range,
                explicit_carry=self.explicit_carry,
            )

            x = full_seq[:, :-1]
            y = full_seq[:, 1:]
            p1 = pos1[:, :-1]
            p2 = pos2[:, :-1]
            p3 = pos3[:, :-1]
            batch.append((x, y, p1, p2, p3))

            # with open("logs/log.txt", "a") as f:
            #     f.write(f"x: {x[0]},\ny: {y[0]},\np1: {p1[0]},\np2: {p2[0]},\np3: {p3[0]}\n")
        batch2 = []
        max_len = max(x.shape[1] for x, y, p1, p2, p3 in batch)
        for x, y, p1, p2, p3 in batch:
            l = x.shape[1]
            if l < max_len:
                x = torch.cat(
                    (
                        x,
                        torch.full(
                            (x.shape[0], max_len - l), self.stoi["#"], dtype=torch.long
                        ),
                    ),
                    dim=1,
                )
                y = torch.cat(
                    (
                        y,
                        torch.full(
                            (y.shape[0], max_len - l), self.stoi["#"], dtype=torch.long
                        ),
                    ),
                    dim=1,
                )
                p1_x = p1[0, -1]
                p1 = torch.cat(
                    (
                        p1,
                        torch.full((p1.shape[0], max_len - l), p1_x, dtype=torch.long),
                    ),
                    dim=1,
                )
                p2_x = p2[0, -1]
                p2 = torch.cat(
                    (
                        p2,
                        torch.arange(p2_x + 1, p2_x + max_len - l + 1, dtype=torch.long)
                        .unsqueeze(0)
                        .expand(p2.shape[0], -1),
                    ),
                    dim=1,
                )
                p3 = torch.cat(
                    (p3, torch.full((p3.shape[0], max_len - l), 2, dtype=torch.long)),
                    dim=1,
                )
            batch2.append((x, y, p1, p2, p3))
        x, y, p1, p2, p3 = zip(*batch2)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        p1 = torch.cat(p1, dim=0)
        p2 = torch.cat(p2, dim=0)
        p3 = torch.cat(p3, dim=0)
        return x, y, p1, p2, p3


class SequentialMultiOperandAdditionDataset(IterableDataset):
    """
    Sequentially yields batches from multiple dataset configurations.
    Used for validation to allow a single persistent worker pool.
    """

    def __init__(
        self,
        configurations,  # List of (digits, operands)
        samples_per_config,
        batch_size,
        data_mode="variable",
        offset_range=100,
        random_offsets=True,
        explicit_carry=True,
    ):
        super().__init__()
        self.configurations = configurations
        self.samples_per_config = samples_per_config
        self.batch_size = batch_size
        self.data_mode = data_mode
        self.offset_range = offset_range
        self.random_offsets = random_offsets
        self.explicit_carry = explicit_carry

        # Reuse the logic/vocab from the main dataset
        # We can just instantiate a helper object or copy logic.
        # Ideally, we refactor the generation logic out, but for now,
        # we will instantiate a temporary helper to access properties/methods if needed,
        # or just re-implement the loop calling a generator.

        # Helper dataset to delegate generation to
        # We will create one "template" dataset and update its params on the fly
        self.template_ds = MultiOperandAdditionDataset(
            min_digits=1,
            max_digits=1,  # placeholders
            batch_size=batch_size,
            min_operands=2,
            max_operands=2,  # placeholders
            data_mode=data_mode,
            offset_range=offset_range,
            random_offsets=random_offsets,
            explicit_carry=explicit_carry,
        )

        self.vocab_size = self.template_ds.vocab_size
        self.stoi = self.template_ds.stoi
        self.itos = self.template_ds.itos

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # If multiple workers, we need to split the work?
        # Actually validation usually doesn't need strict splitting effectively if we just want "some" samples.
        # But for determinism and exact sample counts, we should be careful.
        # Simpler: Each worker does the FULL sequence? No, that duplicates work.
        # Each worker should do a subset.

        total_samples = self.samples_per_config

        if worker_info is not None:
            # Simple split: divide samples_per_config by num_workers
            per_worker = int(math.ceil(total_samples / worker_info.num_workers))
            worker_id = worker_info.id
            # TODO: Handle remainders correctly, but ceil is safe for validation (more is valid)
        else:
            per_worker = total_samples

        for config_idx, (digits, operands) in enumerate(self.configurations):
            # Update template dataset
            self.template_ds.min_digits = digits
            self.template_ds.max_digits = digits
            self.template_ds.min_operands = operands
            self.template_ds.max_operands = operands

            # Yield batches
            for _ in range(per_worker):
                batch = self.template_ds.generate_batch()
                # batch is (x, y, p1, p2, p3)
                # Append config_idx
                yield batch + (config_idx,)


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
        explicit_carry=True,
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
            explicit_carry=self.hparams.explicit_carry,
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
        max_val_digits = self.hparams.max_val_digits
        min_train_digits = self.hparams.min_train_digits
        val_step_digits = self.hparams.val_step

        min_val_ops = self.hparams.min_operands
        max_val_ops = self.hparams.max_val_operands
        val_step_ops = self.hparams.val_operand_step

        # 1. Lengths Grid
        lengths = sorted(
            list(range(min_train_digits, max_val_digits + 1, max(1, val_step_digits)))
        )

        # 2. Operands Grid
        operands = sorted(
            list(range(min_val_ops, max_val_ops + 1, max(1, val_step_ops)))
        )

        # Generate Configurations
        self.val_config_names = []
        configurations = []

        for L in lengths:
            for N in operands:
                configurations.append((L, N))
                self.val_config_names.append(f"val_L{L}_N{N}")

        # Samples per config: We had limit_val_batches=50 in train.py?
        # Or val_bs * limit_val_batches?
        # Standard validation often runs for a fixed number of steps per loader.
        # train.py uses limit_val_batches=50.
        # Let's target 20 batches per config (arbitrary, user can tune).
        batches_per_config = 20

        # Single Sequential Dataset
        val_ds = SequentialMultiOperandAdditionDataset(
            configurations=configurations,
            samples_per_config=batches_per_config,
            batch_size=self.val_bs,
            data_mode=self.hparams.data_mode,
            random_offsets=self.hparams.random_offsets,
            explicit_carry=self.hparams.explicit_carry,
        )

        return DataLoader(
            val_ds,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
