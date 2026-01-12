
import torch
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

class AbsolutePositionAdditionDataset(IterableDataset):
    """
    Generates addition examples batches vectorized.
    [n1] + [n2] = [reversed_sum]
    n1, n2 are L-digit numbers (or smaller padded).
    Uses absolute position encoding with same offset in a batch.
    """
    def __init__(self, min_digits, max_digits, batch_size, offset_range=20, seed=None):
        super().__init__()
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.batch_size = batch_size
        self.offset_range = offset_range
        self.seed = seed
        self.chars = [str(i) for i in range(10)] + ["+", "=", "#"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        # Pre-compute token IDs for special chars
        self.plus_token = self.stoi["+"]
        self.pad_token = self.stoi["#"]
        self.eq_token = self.stoi["="]
        
        # Initialize RNG for seed support
        if self.seed is not None:
            self.rng = random.Random(self.seed)
        else:
            self.rng = random.Random()
    
    def __iter__(self):
        while True:
            yield self.generate_batch()
    
    def generate_batch(self):
        B = self.batch_size
        # Sample one L for the vector batch to allow pure tensor ops without padding mess
        L = self.rng.randint(self.min_digits, self.max_digits)
        
        # 1. Generate digits directly (B, L)
        n1_digits = torch.randint(0, 10, (B, L))
        n2_digits = torch.randint(0, 10, (B, L))
        
        # 2. Compute Sum with Carry Propagation
        # We process from LSB (index L-1) to MSB (index 0). Results will have length L+1 to accommodate final carry
        s_digits_rev = []  # Stores digits from LSB to MSB
        carry = torch.zeros(B, dtype=torch.short)
        for i in range(L - 1, -1, -1):
            d1 = n1_digits[:, i]
            d2 = n2_digits[:, i]
            total = d1 + d2 + carry
            carry = total // 10
            s_digits_rev.append(total % 10)
        # Final carry becomes the MSB of the sum
        s_digits_rev.append(carry)
        # s_digits_rev is ALREADY [LSB, ..., MSB], which matches the target reversed sum order.
        s_digits = torch.stack(s_digits_rev, dim=1)  # (B, L+1)
        
        # 3. Construct Token Batch
        plus = torch.full((B, 1), self.plus_token, dtype=torch.long)
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long)
        tokens = torch.cat([n1_digits, plus, n2_digits, eq, s_digits], dim=1)
        
        # 4. Construct Absolute Positional Encoding
        # Sequence length is: L (n1) + 1 (plus) + L (n2) + 1 (eq) + L+1 (sum) = 3L + 3
        seq_len = tokens.size(1)
        # Generate absolute positions [0, 1, 2, ..., seq_len-1]
        abs_pos = torch.arange(seq_len)       
        # Sample a single offset for the entire batch
        offset = self.rng.randint(0, self.offset_range - 1)
        abs_pos = abs_pos + offset
        pos = abs_pos.unsqueeze(0).expand(B, -1)
        
        # 5. Form (x, y, pos)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        pos = pos[:, :-1]
        
        return x, y, pos


class PositionCouplingAdditionDataset(AbsolutePositionAdditionDataset):
    """
    Generates addition examples batches vectorized.
    [n1] + [n2] = [reversed_sum]
    n1, n2 are both L-digit numbers (with possible leading zeros).
    Uses position coupling encoding.
    """
    def generate_batch(self):
        B = self.batch_size
        L = random.randint(self.min_digits, self.max_digits)

        # 1. Generate L-digit numbers (leading zeros allowed)
        n1_digits = torch.randint(0, 10, (B, L))
        n2_digits = torch.randint(0, 10, (B, L))

        # 2. Compute Sum with Carry Propagation
        s_digits_rev = []  # Stores digits from LSB to MSB
        carry = torch.zeros(B, dtype=torch.long)

        for i in range(L - 1, -1, -1):
            d1 = n1_digits[:, i]
            d2 = n2_digits[:, i]
            total = d1 + d2 + carry
            rem = total % 10
            carry = total // 10
            s_digits_rev.append(rem)

        # Final carry becomes the MSB of the sum
        s_digits_rev.append(carry)

        # s_digits_rev is [LSB, ..., MSB], which matches the target reversed sum order
        s_digits = torch.stack(s_digits_rev, dim=1)  # (B, L+1)

        # 3. Construct Token Batch
        plus = torch.full((B, 1), self.plus_token, dtype=torch.long)
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long)

        tokens = torch.cat([n1_digits, plus, n2_digits, eq, s_digits], dim=1)

        # 4. Construct Positional Batch
        # Position coupling: both operands have length L
        # For n1, the index decreases from L to 1, with + at position 0
        idx_n1 = torch.arange(L, 0, -1)  # [L, L-1, ..., 1]
        idx_n1_with_plus = torch.cat([idx_n1, torch.tensor([0])])
        
        # For n2, the index decreases from L to 1, with = at position 0
        idx_n2 = torch.arange(L, 0, -1)  # [L, L-1, ..., 1]
        idx_n2_with_eq = torch.cat([idx_n2, torch.tensor([0])])
        
        # For the sum, the index increases from 1 to sum_len
        sum_len = s_digits.size(1)
        idx_sum = torch.arange(1, sum_len + 1)  # [1, 2, ..., sum_len]
        
        pos_couple = torch.cat([idx_n1_with_plus, idx_n2_with_eq, idx_sum])
        
        offset = self.rng.randint(0, self.offset_range - 1)
        pos = pos_couple + offset
        pos = pos.unsqueeze(0).expand(B, -1)  # (B, SeqLen)
        
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        p = pos[:, :-1]

        return x, y, p


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
        seed=None,
        dataset_type="position_coupling",  # "absolute" or "position_coupling"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = 12
        self.dataset_type = dataset_type  # Store which dataset is being used
    
    def setup(self, stage=None):
        # Training dataset: range [min_train, max_train]
        if self.dataset_type == "absolute":
            self.train_ds = AbsolutePositionAdditionDataset(
                self.hparams.min_train_digits,
                self.hparams.max_train_digits,
                self.hparams.batch_size,
                seed=self.hparams.seed,
            )
        elif self.dataset_type == "position_coupling":
            self.train_ds = PositionCouplingAdditionDataset(
                self.hparams.min_train_digits,
                self.hparams.max_train_digits,
                self.hparams.batch_size,
                seed=self.hparams.seed,
            )
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}. Must be 'absolute' or 'position_coupling'")
        
        # Validation datasets: Multiple datasets for generalization
        # Start from max_train + 1, go up to max_val
        val_lengths = list(
            range(
                self.hparams.max_train_digits + 1,
                self.hparams.max_val_digits + 1,
                self.hparams.val_step,
            )
        )
        # Also always include a validation set within training distribution
        train_dist_val = (
            self.hparams.max_train_digits + self.hparams.min_train_digits
        ) // 2
        self.val_datasets = []
        self.val_names = []
        
        # 1. In-distribution set
        if self.dataset_type == "absolute":
            val_ds = AbsolutePositionAdditionDataset(
                train_dist_val, train_dist_val, self.hparams.batch_size, seed=self.hparams.seed
            )
        else:  # position_coupling
            val_ds = PositionCouplingAdditionDataset(
                train_dist_val, train_dist_val, self.hparams.batch_size, seed=self.hparams.seed
            )
        self.val_datasets.append(val_ds)
        self.val_names.append(f"val_L{train_dist_val}")
        
        # 2. Generalization sets
        for L in val_lengths:
            if self.dataset_type == "absolute":
                val_ds = AbsolutePositionAdditionDataset(
                    L, L, self.hparams.batch_size, seed=self.hparams.seed
                )
            else:  # position_coupling
                val_ds = PositionCouplingAdditionDataset(
                    L, L, self.hparams.batch_size, seed=self.hparams.seed
                )
            self.val_datasets.append(val_ds)
            self.val_names.append(f"val_L{L}")
        
        self.stoi = self.train_ds.stoi
        self.itos = self.train_ds.itos
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=None, num_workers=self.hparams.num_workers
        )
    
    def on_train_epoch_start(self):
        # Curriculum: At epoch 0, max_digits = curriculum_start
        # Each epoch increases max_digits by 1 until max_train_digits
        current_epoch = self.trainer.current_epoch
        new_max = min(
            self.hparams.curriculum_start + current_epoch, self.hparams.max_train_digits
        )
        self.train_ds.max_digits = new_max
        print("\n" + "=" * 50)
        print(f"   CURRICULUM UPDATE: Epoch {current_epoch}")
        print(f"   Training range: 1-{new_max} digits")
        print(f"   Dataset type: {self.dataset_type}")
        print("=" * 50 + "\n")
    
    def val_dataloader(self):
        # Dynamic validation sets based on current curriculum progress
        current_max = self.train_ds.max_digits
        max_val = self.hparams.max_val_digits
        min_train = self.hparams.min_train_digits
        val_step = self.hparams.val_step
        # 1. Regular Grid (Coarse view across entire range)
        # e.g., 1, 5, 9, 13, 17... if step is 4
        lengths_grid = list(range(min_train, max_val + 1, max(1, val_step)))
        # 2. Critical Curriculum Points
        # Always check exactly where we are training, and the immediate next step
        curriculum_points = [current_max, min(current_max + 1, max_val)]
        # Combine and Deduplicate
        lengths = sorted(list(set(lengths_grid + curriculum_points)))
        dataloaders = []
        self.val_names = []  # Reset and repopulate
        for L in lengths:
            if self.dataset_type == "absolute":
                ds = AbsolutePositionAdditionDataset(
                    L, L, self.hparams.batch_size, seed=self.hparams.seed
                )
            else:  # position_coupling
                ds = PositionCouplingAdditionDataset(
                    L, L, self.hparams.batch_size, seed=self.hparams.seed
                )
            dataloaders.append(
                DataLoader(
                    ds,
                    batch_size=None,
                    num_workers=self.hparams.num_workers,
                    persistent_workers=self.hparams.num_workers > 0,
                )
            )
            self.val_names.append(f"val_L{L}")
        return dataloaders