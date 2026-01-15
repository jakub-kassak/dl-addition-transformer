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
        data_type="default" 
        #digit_combinations- puts all combinations of operands and digits in single batch- no cirriculum training 
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
        self.data_type = data_type

        # Pad token for batching variable lengths
        self.pad_token = -1

        # Vocab: 0-19 (digits+carry), +, =, >, #
        self.chars = [str(i) for i in range(20)] + ["+", "=", ">"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.stoi['#'] = self.pad_token
        self.itos[self.pad_token] = "#"
        self.vocab_size = len(self.stoi.keys())

        self.plus_token = self.stoi["+"]
        self.eq_token = self.stoi["="]
        self.greater_token = self.stoi[">"]
        self.hash_token = self.stoi["#"]

       

    def __iter__(self):
        while True:
            with open("logs/log.txt", "a") as f:
                f.write(f"TRAINING \n")
            yield self.generate_batch()

    def genereate_default_operands(self, N_Ops):
        B = self.batch_size
        N_Dig = random.randint(self.min_digits, self.max_digits)
        with open("logs/log.txt", "a") as f:
            f.write(f"OPS: {self.min_operands}-{self.max_operands} => {N_Ops}, Digits: {self.min_digits}-{self.max_digits} => {N_Dig}\n")

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
        
        return operands_digits, max_len
    
    def generate_varied_digits_per_operand(self, N_Ops):
        operand_digits = []
        B = self.batch_size
        with open("logs/log.txt", "a") as f:
            f.write(f"OPS: {self.min_operands}-{self.max_operands} => {N_Ops} \n")
        num_digits = self.max_digits - self.min_digits + 1

        # 1. Determine max_len (padding everything to this length)
        # Allowance for carry: log10(max_operands)
        carry_allowance = math.ceil(math.log10(N_Ops))
        max_len = self.max_digits + carry_allowance

        ## split batch into chunks where digits will vary
        base, rem = divmod(B, num_digits)
        chunk_sizes = [base] * num_digits
        for i in torch.randperm(num_digits)[:rem].tolist():  # randomize who gets the +1
            chunk_sizes[i] += 1
        
        ## generate a batch
        for _ in range(N_Ops):
            batch = torch.empty(B, max_len, dtype=torch.long)
            idx = 0
            for i, size in enumerate(chunk_sizes):
                n_dig = self.min_digits + i
        
                d = torch.randint(0, 10, (size, n_dig))

                # Pad to max_len with zeros (at the front/MSB for input)
                carry_allowance = math.ceil(math.log10(N_Ops))
                max_len_ndig = n_dig + carry_allowance
                padding_zeros = torch.zeros((size, max_len_ndig - n_dig), dtype=torch.long)
                full_d = torch.cat([padding_zeros, d], dim=1)
                padding_pound = torch.full((size,max_len-max_len_ndig), -1)
                full_final = torch.cat([full_d, padding_pound], dim=1)

                batch[idx : idx + size] = full_final
                idx += size
            operand_digits.append(batch)

        return operand_digits, max_len

    def generate_batch(self):
        B = self.batch_size
        operands_digits = []
        max_len = 0
        N_Ops = random.randint(self.min_operands, self.max_operands)

        if self.data_type == "default":
            operands_digits, max_len = self.genereate_default_operands(N_Ops)
        elif self.data_type == "digit_combinations":
            operands_digits, max_len = self.generate_varied_digits_per_operand(N_Ops)
        


        # 3. Compute Partial Sums (S0=0, S1=N1, ..., SN=Sum(N1..NN))
        # Each Si is stored LSB-first in scratchpad_segments (length max_len)
        scratchpad_segments = []
        current_sum = torch.zeros((B, max_len), dtype=torch.long)
        ex = operands_digits[0] 
        cond = ex !=-1
        scratchpad_segments.append(torch.where(cond,torch.zeros_like(ex), ex ))
       
        for operand in operands_digits:
            carry = torch.zeros(B, dtype=torch.long)
            for i in range(max_len - 1, -1, -1):
                d_acc = current_sum[:, i]
                d_op = operand[:, i]
                # d_op = torch.where(d_op == -1, torch.zeros_like(d_op), d_op)
                # print(d_acc)
                # Only sum values where d_op is not -1; else, keep as -1
                # If either d_op or d_acc is -1, keep as -1, else add
                cond = (d_op == -1) | (d_acc == -1) 
                # print (cond)
                total = torch.where(cond, torch.full_like(d_op, -1), d_acc + d_op + carry)
                
                current_sum[:, i] = total
                # everywhere there is -1, make it 0 in total
                total = torch.where(total == -1, torch.zeros_like(total), total)
                carry = total // 10
                # 
            flipped = current_sum.clone()
            for row_idx, row in enumerate(current_sum):
                mask = row != -1
                values = row[mask]
                flipped_row = row.clone()
                flipped_row[mask] = values.flip(0)
                flipped[row_idx] = flipped_row
            scratchpad_segments.append(flipped)
            # print(current_sum)
            current_sum %= 10
        # print(scratchpad_segments)

        # 4. Construct Full Sequence
        # Input: N1 + N2 + ... + Nk =
        # SP: S0 > S1 > ... > Sn #
        p1_list, p2_list, p3_list, tokens_list = [], [], [], []

        plus = torch.full((B, 1), self.plus_token, dtype=torch.long)
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long)
        greater = torch.full((B, 1), self.greater_token, dtype=torch.long)
        hash_t = torch.full((B, 1), self.hash_token, dtype=torch.long)

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
        p1_list.append(torch.full((B, 1), N_Ops + 1, dtype=torch.long))  # Final block
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

        # with open("logs/log.txt", "a") as f:
        #     f.write(f"x: {x[0]},\ny: {y[0]},\np1: {p1[0]},\np2: {p2[0]},\np3: {p3[0]}\n")
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
    ):
        super().__init__()
        self.configurations = configurations
        self.samples_per_config = samples_per_config
        self.batch_size = batch_size
        self.data_mode = data_mode
        self.offset_range = offset_range
        self.random_offsets = random_offsets

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
        )

        self.vocab_size = self.template_ds.vocab_size
        self.stoi = self.template_ds.stoi
        self.itos = self.template_ds.itos

    def __iter__(self):
        with open("logs/log.txt", "a") as f:
            f.write(f"VALIDATING \n")
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
        data_type="default",
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
            data_type=self.hparams.data_type,
        )

        # Normalize curriculum start immediately
        # This ensures setup() results are consistent regardless of max_train_digits
        if self.hparams.data_type == "default":
            initial_max = min(self.hparams.curriculum_start, self.hparams.max_train_digits)
        elif self.hparams.data_type == "digit_combinations":
            initial_max = self.hparams.max_train_digits
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
        )

        return DataLoader(
            val_ds,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
