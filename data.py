import torch
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
import random


class VectorizedAdditionDataset(IterableDataset):
    """
    Generates addition examples batches vectorized.
    [n1] + [n2] = [reversed_sum]
    Different digit lengths for n1 and n2 are supported as L1 and L2
    """

    def __init__(
        self,
        min_digits,
        max_digits,
        batch_size,
        offset_range_bottom,
        offset_range_top,
        val=False,
        seed = None
    ):
        super().__init__()
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.batch_size = batch_size
        self.offset_range_bottom = offset_range_bottom
        self.offset_range_top = offset_range_top
        self.seed = seed

        self.chars = "0123456789+=#"
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_token = self.stoi["#"]

        # Pre-compute token IDs for special chars
        self.plus_token = self.stoi["+"]
        self.eq_token = self.stoi["="]

        if self.seed is not None:
            self.set_seed(self.seed)

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._seed = seed

    def __iter__(self):
        while True:
            yield self.generate_batch()

    def generate_batch(self):
        B = self.batch_size
        L1 = random.randint(self.min_digits, self.max_digits)
        L2 = random.randint(self.min_digits, self.max_digits)
        # L for padding
        L = max(L1, L2)

        # 1. Generate digits and pad to maximum length L
        n1_digits_full = torch.randint(0, 10, (B, L1))
        n2_digits_full = torch.randint(0, 10, (B, L2))
        # make sure that the first digit is non-zero, unless L = 1
        if L1 > 1:
            n1_digits_full[:, 0] = torch.randint(1, 10, (B,))
        if L2 > 1:
            n2_digits_full[:, 0] = torch.randint(1, 10, (B,))
        n1_digits = torch.zeros(B, L, dtype=torch.long)
        n2_digits = torch.zeros(B, L, dtype=torch.long)

        # non-zero digits on the right
        n1_digits[:, L-L1:] = n1_digits_full  
        n2_digits[:, L-L2:] = n2_digits_full

        # 2. Compute Sum with Carry Propagation
        # We process from LSB (index L-1) to MSB (index 0)
        # Result s will have length L+1 to accommodate final carry. It could be 0 in mamy cases.
        s_digits_rev = []  # Stores digits from LSB to MSB
        carry = torch.zeros(B, dtype=torch.long)

        for i in range(L - 1, -1, -1):
            d1 = n1_digits[:, i]
            d2 = n2_digits[:, i]
            total = d1 + d2 + carry  # total < 20 always for 2 addends!
            rem = total % 10
            carry = total // 10
            s_digits_rev.append(rem)

        # Final carry becomes the MSB of the sum
        s_digits_rev.append(carry)  # carry = 0 or 1

        # s_digits_rev is ALREADY [LSB, ..., MSB], which matches the target reversed sum order.
        s_digits = torch.stack(s_digits_rev, dim=1)  # (B, L+1)

        # 3. Construct Token Batch
        plus = torch.full((B, 1), self.plus_token, dtype=torch.long)
        eq = torch.full((B, 1), self.eq_token, dtype=torch.long)

        n1_tokens = n1_digits_full  # note this must be the original number
        n2_tokens = n2_digits_full

        tokens = torch.cat([n1_tokens, plus, n2_tokens, eq, s_digits], dim=1)

        # 4. Construct Positional Batch
        # 1st: role encoding: n1=1, n2=2, sum=3
        p1_seg1 = torch.full((B, L1 + 1), 1, dtype=torch.long) # n1 & +
        p1_seg2 = torch.full((B, L2 + 1), 2, dtype=torch.long) # n2 & =
        p1_seg3 = torch.full((B, s_digits.size(1)), 3, dtype=torch.long) # sum
        pos1 = torch.cat([p1_seg1, p1_seg2, p1_seg3], dim=1)

        # 2nd: position coupling
        # for n1, the index decreases from L1 to 1, with + at position 0
        idx_n1 = torch.arange(L1, 0, -1)  # [L1, L1-1, ..., 1]
        idx_n1_with_plus = torch.cat([idx_n1, torch.tensor([0])])
        idx_n2 = torch.arange(L2, 0, -1)  # [L2, L2-1, ..., 1]
        idx_n2_with_eq = torch.cat([idx_n2, torch.tensor([0])])
        
        # for the sum, the index increases from 1 to sum_len
        sum_len = s_digits.size(1)
        idx_sum = torch.arange(1, sum_len + 1)  # [1, 2, ..., sum_len]
        
        pos2_seq = torch.cat([idx_n1_with_plus, idx_n2_with_eq, idx_sum])
        pos2 = pos2_seq.unsqueeze(0).expand(B, -1)  # (B, SeqLen)
        
        offsets = torch.randint(self.offset_range_bottom, self.offset_range_top, (B, 1))
        pos2 = pos2 + offsets  # random shift per sample

        # 5. Form (x, y) fro causal language modelling
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        p1 = pos1[:, :-1]
        p2 = pos2[:, :-1]

        return x, y, p1, p2


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
        max_pos2=3 * 15,
        seed=200,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = 13
        self.seed = seed

    def setup(self, stage=None):

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Training dataset: range [min_train, max_train]
        self.train_ds = VectorizedAdditionDataset(
            self.hparams.min_train_digits,
            self.hparams.max_train_digits,
            self.hparams.batch_size,
            0,
            self.hparams.max_pos2 - self.hparams.max_train_digits,
            seed=self.seed,
        )

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
        self.val_datasets.append(
            VectorizedAdditionDataset(
                train_dist_val,
                train_dist_val,
                self.hparams.batch_size,
                self.hparams.max_val_digits / 4,
                3 * self.hparams.max_val_digits / 4,
            )
        )
        self.val_names.append(f"val_L{train_dist_val}")

        # 2. Generalization sets
        for L in val_lengths:
            self.val_datasets.append(
                VectorizedAdditionDataset(
                    L,
                    L,
                    self.hparams.batch_size,
                    self.hparams.max_val_digits / 4,
                    3 * self.hparams.max_val_digits / 4,
                )
            )
            self.val_names.append(f"val_L{L}")

        self.stoi = self.train_ds.stoi
        self.itos = self.train_ds.itos

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
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
        print(f"    CURRICULUM UPDATE: Epoch {current_epoch}")
        print(f"    Training range: 1-{new_max} digits")
        print("=" * 50 + "\n")
        # different seed per epoch
        epoch_seed = self.seed + 1000 * current_epoch
        self.train_ds.set_seed(epoch_seed)

    def val_dataloader(self):
        # Dynamic validation sets based on current curriculum progress
        current_max = self.train_ds.max_digits
        max_val = self.hparams.max_val_digits
        min_train = self.hparams.min_train_digits
        val_step = self.hparams.val_step

        # 1. Regular Grid (Coarse view across entire range)
        # e.g., 1, 5, 9, 13, 17... if step is 4
        lengths_grid = list(range(min_train, max_val, max(1, val_step)))

        # 2. Critical Curriculum Points
        # Always check exactly where we are training, and the immediate next step
        # curriculum_points = [current_max, min(current_max + 1, max_val)]

        # Combine and Deduplicate
        lengths = sorted(list(set(lengths_grid)))

        dataloaders = []
        self.val_names = []  # Reset and repopulate
        offset_range_bottom = self.hparams.max_val_digits // 3
        offset_range_top = 4 * self.hparams.max_val_digits // 3
        for i, L in enumerate(lengths):
            ds = VectorizedAdditionDataset(
                max(1, L),
                max(1, L),
                self.hparams.batch_size,
                offset_range_bottom=offset_range_bottom,
                offset_range_top=offset_range_top,
                seed=self.seed + 100 + i   # different seed per validation set
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
