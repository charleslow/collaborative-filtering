import math
from collections.abc import Iterable

import numpy as np
import pytorch_lightning as pl
import scipy
import scipy.sparse
import torch
from typing_extensions import Self

PADDING_IDX = 0


class MFExhaustiveDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a batch of users and items. Ratings are stored in a sparse matrix and
    queried to return ratings for each (user, item) pair in the batch.
    """

    def __init__(
        self,
        R: scipy.sparse.csc_matrix,
        user_batch_size: int,
        item_batch_size: int,
        shuffle: bool = False,
    ):
        self.n_users = R.shape[0]
        self.user_inner_ids = np.arange(self.n_users)
        self.user_batch_size = user_batch_size
        self.n_user_batches = math.ceil(self.n_users / self.user_batch_size)

        self.n_items = R.shape[1]
        self.item_inner_ids = np.arange(self.n_items)
        self.item_batch_size = item_batch_size
        self.n_item_batches = math.ceil(self.n_items / self.item_batch_size)

        self.R = R
        if shuffle:
            self.user_inner_ids = np.random.shuffle(self.user_inner_ids)
            self.item_inner_ids = np.random.shuffle(self.item_inner_ids)

    def __len__(self):
        """Returns the number of batches."""
        return self.n_user_batches * self.n_item_batches

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a batch of data.

        Imagine that we are traversing a matrix where rows are users and columns are items.
        We traverse blocks of the matrix left to right, then top to bottom.
        Suppose both user/item_batch_size = 10, and n_users/items = 100.
            At idx=0: load users[0:10], items[0:10]
            At idx=1: load users[0:10], items[10:20]
            ...
            At idx=9: load users[0:10], items[90:100]
            At idx=10: load users[10:20], items[0:10]
        And so on.
        """
        user_batch_idx = idx // self.n_item_batches
        item_batch_idx = idx % self.n_item_batches
        user_start_idx = user_batch_idx * self.user_batch_size
        user_end_idx = min(user_start_idx + self.user_batch_size, self.n_users)
        item_start_idx = item_batch_idx * self.item_batch_size
        item_end_idx = min(item_start_idx + self.item_batch_size, self.n_items)
        item_batch = self.item_inner_ids[item_start_idx:item_end_idx]
        user_batch = self.user_inner_ids[user_start_idx:user_end_idx]
        batch_R = self.R[user_batch, item_batch].toarray()
        batch_R = torch.tensor(batch_R, dtype=torch.float32)
        return batch_R


class MFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ratings: list[tuple],
        user_batch_size: int = 128,
        item_batch_size: int = 128,
        num_workers: int = 1,
        shuffle: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ratings"])
        self.raw2inner_user, self.raw2inner_item, self.R = self.parse_ratings(ratings)
        self.dataset = MFExhaustiveDataset(
            R=self.R,
            user_batch_size=self.hparams.user_batch_size,
            item_batch_size=self.hparams.item_batch_size,
            shuffle=self.hparams.shuffle,
        )
        self.n_users = self.R.shape[0]
        self.n_items = self.R.shape[1]

    def parse_ratings(
        self: Self, ratings_raw: Iterable[tuple]
    ) -> tuple[dict, dict, scipy.sparse.csc_matrix]:
        """Ingest raw ratings tuples.

        Given ratings data in the form of list[tuple] where each tuple is:
            (user_id, item_id, rating)
        Return the raw2inner_user/item dictionaries and the sparse ratings matrix.
        """
        raw2inner_user = {}
        raw2inner_item = {}
        current_user_id = 0
        current_item_id = 0
        rows = []
        cols = []
        ratings = []
        for user_id, item_id, rating in ratings_raw:
            if user_id not in raw2inner_user:
                raw2inner_user[user_id] = current_user_id
                current_user_id += 1
            if item_id not in raw2inner_item:
                raw2inner_item[item_id] = current_item_id
                current_item_id += 1
            rows.append(raw2inner_user[user_id])
            cols.append(raw2inner_item[item_id])
            ratings.append(rating)
        sparse_matrix = scipy.sparse.coo_matrix(
            (ratings, (rows, cols)),
            shape=(len(raw2inner_user), len(raw2inner_item)),
        )
        return raw2inner_user, raw2inner_item, sparse_matrix.tocsc()

    def train_dataloader(self):
        """Create a DataLoader for training."""
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
