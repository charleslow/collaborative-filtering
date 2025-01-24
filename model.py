from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from typing_extensions import Self


class Model(ABC):
    model_param_cls: type[BaseModel]

    def __init__(self, **kwargs):
        self.hparams = self.model_params_cls.model_validate(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.loss(**kwargs)

    @abstractmethod
    def loss(
        self,
        *,
        scores: torch.Tensor,
        ratings: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ):
        raise NotImplementedError()


class WMFParams(BaseModel):
    lambda_u: float = 0.01
    lambda_v: float = 0.01
    b: float = 0.01


class WMFModel(Model):
    model_param_cls = WMFParams

    def loss(
        self,
        *,
        scores: torch.Tensor,
        ratings: torch.Tensor,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ):
        pos_mask = (ratings > 0).float()
        neg_mask = (ratings <= 0).float()
        C = self.hparams.b * neg_mask + pos_mask
        squared_error = torch.square(ratings - scores)
        loss_1 = torch.sum(C * squared_error)
        loss_2 = self.hparams.lambda_u * torch.norm(
            user_embeddings, p=2
        ) + self.hparams.lambda_v * torch.norm(item_embeddings, p=2)
        total_loss = loss_1 + loss_2
        return total_loss, loss_1, loss_2


class MFLightningModule(pl.LightningModule):
    def __init__(
        self: Self,
        *,
        model_type: str = "wmf",
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.U = torch.nn.Parameter(torch.empty(n_users, embedding_dim))
        self.V = torch.nn.Parameter(torch.empty(n_items, embedding_dim))
        self.save_hyperparameters()
        self.reset_parameters()
        if model_type == "wmf":
            self.model = WMFModel(**kwargs)
        else:
            raise ValueError()

    def reset_parameters(self):
        """Initialize weights with Xavier uniform initialization."""
        torch.nn.init.xavier_uniform_(self.U).to(torch.float32)
        torch.nn.init.xavier_uniform_(self.V).to(torch.float32)

    def training_step(self, batch, batch_idx: int):
        """Training step for PyTorch Lightning."""
        batch_R, user_ids, item_ids = batch
        user_embeddings = self.U[user_ids]
        item_embeddings = self.V[item_ids]
        scores = torch.matmul(user_embeddings, item_embeddings.t())
        loss, loss_1, loss_2 = self.model(
            scores=scores,
            ratings=batch_R,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
        )
        self.log_dict(
            {"num_batch_users": batch_R.size(0), "num_batch_items": batch_R.size(1)}
        )
        model_name = type(self.model).__name__
        self.log(f"train/{model_name}Loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def score(
        self,
        user_idx: int,
        item_idx: list[int] | None = None,
    ):
        """Predict scores for a user and item(s)."""
        if item_idx is None:
            return torch.matmul(self.V, self.U[user_idx, :])
        return torch.dot(self.V[item_idx, :], self.U[user_idx, :])
