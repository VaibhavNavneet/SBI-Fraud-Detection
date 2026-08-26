"""PyTorch autoencoder trained on non-defaulters only."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AutoencoderConfig:
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (128, 64)
    epochs: int = 80
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    seed: int = 42


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, config: AutoencoderConfig):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous = input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        encoder_layers.append(nn.Linear(previous, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        previous = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


class NormalAutoencoder:
    def __init__(self, config: AutoencoderConfig | None = None):
        self.config = config or AutoencoderConfig()
        self.model: MLPAutoencoder | None = None
        self.input_dim: int | None = None

    def fit(self, x: np.ndarray) -> "NormalAutoencoder":
        _seed_everything(self.config.seed)
        x = np.asarray(x, dtype=np.float32)
        if len(x) < 2:
            raise ValueError("At least two non-defaulter rows are required.")
        self.input_dim = x.shape[1]
        self.model = MLPAutoencoder(self.input_dim, self.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        split = max(1, int(len(x) * 0.1))
        train_x, valid_x = x[:-split], x[-split:]
        train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x)), batch_size=self.config.batch_size, shuffle=True)
        valid_tensor = torch.from_numpy(valid_x).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        loss_fn = nn.MSELoss()
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(self.config.epochs):
            self.model.train()
            for (batch,) in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                _, reconstruction = self.model(batch)
                loss_fn(reconstruction, batch).backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                _, reconstruction = self.model(valid_tensor)
                valid_loss = float(loss_fn(reconstruction, valid_tensor).cpu())
            if valid_loss < best_loss:
                best_loss, best_state, stale = valid_loss, {k: v.cpu().clone() for k, v in self.model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        return self

    def transform(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Autoencoder must be fitted before transform().")
        tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            z, reconstruction = self.model(tensor)
        errors = torch.mean((tensor - reconstruction) ** 2, dim=1).numpy().reshape(-1, 1)
        return z.numpy(), errors

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_dim is None:
            raise RuntimeError("Cannot save an unfitted autoencoder.")
        torch.save({"config": asdict(self.config), "input_dim": self.input_dim, "state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "NormalAutoencoder":
        checkpoint = torch.load(path, map_location="cpu")
        config = AutoencoderConfig(**checkpoint["config"])
        instance = cls(config)
        instance.input_dim = checkpoint["input_dim"]
        instance.model = MLPAutoencoder(instance.input_dim, config)
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.model.eval()
        return instance


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
