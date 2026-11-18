"""Deep autoencoder for unsupervised anomaly detection in DCC correlations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - surfaced clearly outside a torch environment
    raise ImportError(
        "PyTorch is required for src.ml. Install the project requirements "
        "(for example, `pip install torch`)."
    ) from error


def upper_triangle_features(correlation_matrices: np.ndarray) -> np.ndarray:
    """Extract the 10 unique off-diagonal elements from 5-bank matrices."""
    matrices = np.asarray(correlation_matrices, dtype=np.float32)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Correlation matrices must have shape (time, assets, assets).")
    if matrices.shape[1] < 2:
        raise ValueError("Correlation matrices must contain at least two assets.")
    if not np.isfinite(matrices).all():
        raise ValueError("Correlation matrices contain non-finite values.")
    row, col = np.triu_indices(matrices.shape[1], k=1)
    features = matrices[:, row, col]
    return features


def extract_upper_triangle_features(correlation_matrices: np.ndarray) -> np.ndarray:
    """Backward-friendly explicit alias for upper-triangle feature extraction."""
    return upper_triangle_features(correlation_matrices)


def _validate_features(features: np.ndarray, input_dim: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Autoencoder input must be a 2-D array of observations.")
    if values.shape[1] != input_dim:
        raise ValueError(
            f"Autoencoder input has {values.shape[1]} features; expected {input_dim}."
        )
    if len(values) == 0:
        raise ValueError("Autoencoder input cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError("Autoencoder input contains non-finite values.")
    return values


class CorrelationAutoencoder(nn.Module):
    """Dense ReLU autoencoder for the upper-triangle DCC feature vector."""

    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 3,
        hidden_dims: tuple[int, ...] = (8, 5),
    ) -> None:
        super().__init__()
        if input_dim < 2:
            raise ValueError("input_dim must be at least 2.")
        if latent_dim < 1 or latent_dim >= input_dim:
            raise ValueError("latent_dim must be between 1 and input_dim - 1.")

        encoder_sizes = [input_dim, *hidden_dims, latent_dim]
        decoder_sizes = [latent_dim, *reversed(hidden_dims), input_dim]
        encoder_layers: list[nn.Module] = []
        for in_features, out_features in zip(encoder_sizes[:-1], encoder_sizes[1:]):
            encoder_layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
        decoder_layers: list[nn.Module] = []
        for in_features, out_features in zip(decoder_sizes[:-1], decoder_sizes[1:-1]):
            decoder_layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
        decoder_layers.append(nn.Linear(decoder_sizes[-2], decoder_sizes[-1]))

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, features: Tensor) -> Tensor:
        return self.encoder(features)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, features: Tensor) -> Tensor:
        return self.decode(self.encode(features))

    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        values = _validate_features(features, self.input_dim)
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=next(self.parameters()).device)
            return self(tensor).detach().cpu().numpy()

    def score(self, features: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error for each observation."""
        values = _validate_features(features, self.input_dim)
        reconstructed = self.reconstruct(values)
        return np.mean((values - reconstructed) ** 2, axis=1)

    def fit(
        self,
        features: np.ndarray,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        seed: int = 20251115,
        device: str | torch.device | None = None,
    ) -> "TrainingResult":
        return train_autoencoder(
            features,
            model=self,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )


@dataclass
class TrainingResult:
    model: CorrelationAutoencoder
    losses: list[float]
    device: torch.device

    def __iter__(self):
        yield self.model
        yield self.losses


def build_autoencoder(
    input_dim: int = 10,
    latent_dim: int = 3,
    hidden_dims: tuple[int, ...] = (8, 5),
) -> CorrelationAutoencoder:
    return CorrelationAutoencoder(input_dim, latent_dim, hidden_dims)


def train_autoencoder(
    features: np.ndarray,
    *,
    model: CorrelationAutoencoder | None = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 20251115,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Train on normal observations only; callers control the chronological split."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            capability = torch.cuda.get_device_capability()
            use_cuda = capability[0] >= 7
        target_device = torch.device("cuda" if use_cuda else "cpu")
    else:
        requested = torch.device(device)
        if requested.type == "cuda" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 7
        ):
            requested = torch.device("cpu")
        target_device = requested
    autoencoder = model or CorrelationAutoencoder()
    values = _validate_features(features, autoencoder.input_dim)
    autoencoder.to(target_device)
    autoencoder.train()

    tensor = torch.as_tensor(values, dtype=torch.float32, device=target_device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch)
        losses.append(epoch_loss / len(tensor))

    autoencoder.eval()
    return TrainingResult(autoencoder, losses, target_device)


def reconstruction_errors(
    model: CorrelationAutoencoder,
    features: np.ndarray,
) -> np.ndarray:
    """Vector of per-observation mean-squared reconstruction errors."""
    return model.score(features)


DeepAutoencoder = CorrelationAutoencoder
