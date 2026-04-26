# %load reference/kg_mtp_rebuild.py
"""
KG-MTP baseline rebuild mirroring the original feature extraction pipeline.

The implementation reuses the combinatorial kernel bank, dilation schedule,
and pooling logic from the KG-MTP publication while keeping the code local to
this project. No modules are imported from the original KG-MTP repository.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from scipy import fftpack
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from libs.misc import is_ipython_notebook

PROJECT_ROOT = Path('.').resolve() if is_ipython_notebook() else Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.hydra_basic import Hydra  # noqa: E402
from reference import kgmtp_core  # noqa: E402

SUPPORTED_EXTENSIONS = (".tsv", ".txt", ".csv")
TRANSFORM_NAMES = ("base", "hilbert", "diff")
POOLING_FEATURES_PER_KERNEL = 5
HYDRA_FEATURES_PER_KERNEL = 2


@dataclass(frozen=True)
class HydraEnsembleConfig:
    k: int = 16
    g: int = 64
    batch_size: int = 256
    device: str = "auto"

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

@dataclass(frozen=True)
class FeatureLayout:
    pooling_per_transform: int
    hydra_per_transform: int

    @property
    def features_per_transform(self) -> int:
        return self.pooling_per_transform + self.hydra_per_transform


def create_kgmtp_weights() -> np.ndarray:
    """Generate the 62-kernel weight bank used by KG-MTP."""
    kernel_length = 6
    all_weights: List[np.ndarray] = []

    for value_length, scale in [
        (1, 1.0),
        (2, 1.0),
        (3, 1.0),
        (4, 2.0),
        (5, 5.0),
    ]:
        indices = np.array(
            [combo for combo in combinations(np.arange(kernel_length), value_length)],
            dtype=np.int32,
        )
        weights = np.ones((len(indices), kernel_length), dtype=np.float32) * -1
        for i, combo in enumerate(indices):
            weights[i, combo] = (kernel_length - value_length) / value_length
        weights *= scale
        all_weights.append(weights)

    return np.vstack(all_weights)


def _hilbert_transform(X: np.ndarray) -> np.ndarray:
    """
    Hilbert transform matching upstream exactly: row-by-row fftpack.hilbert
    stored into float32 (matching upstream's incidental dtype truncation).

    Note: scipy.fftpack.hilbert(x) == -np.imag(scipy.signal.hilbert(x))
    but the two use different FFT backends, so numerical results differ slightly.
    We use fftpack to match upstream bit-for-bit.
    """
    out = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = fftpack.hilbert(X[i])
    return out.astype(np.float64)


def find_dataset_files(dataset_dir: Path) -> Tuple[Path, Path]:
    """Locate the TRAIN and TEST files for a UCR-format dataset."""
    name = dataset_dir.name
    for ext in SUPPORTED_EXTENSIONS:
        train = dataset_dir / f"{name}_TRAIN{ext}"
        test = dataset_dir / f"{name}_TEST{ext}"
        if train.exists() and test.exists():
            return train, test

    train = test = None
    for file in dataset_dir.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            upper = file.name.upper()
            suffix = file.suffix.upper()
            if upper.endswith("_TRAIN" + suffix):
                train = file
            elif upper.endswith("_TEST" + suffix):
                test = file
    if train is None or test is None:
        raise FileNotFoundError(f"Missing TRAIN/TEST files in {dataset_dir}")
    return train, test


def load_ucr_split(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a UCR dataset split, falling back to whitespace separation."""
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    if df.shape[1] <= 2:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    y = df.iloc[:, 0].to_numpy()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    return X, y


def prepare_resample_indices(
    y_train: np.ndarray,
    y_test: np.ndarray,
    num_resamples: int,
    random_state: int,
    *,
    dataset_name: Optional[str] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield train/test index splits beginning with the original UCR partition."""

    n_train = len(y_train)
    n_test = len(y_test)
    total = n_train + n_test

    y_all = np.hstack([y_train, y_test])
    dummy = np.zeros((total, 1), dtype=np.float32)

    yield np.arange(n_train, dtype=np.int64), np.arange(n_train, total, dtype=np.int64)

    if num_resamples <= 1:
        return
    splitter = StratifiedShuffleSplit(
        n_splits=num_resamples - 1,
        train_size=n_train,
        test_size=n_test,
        random_state=random_state,
    )

    for train_idx, test_idx in splitter.split(dummy, y_all):
        yield train_idx.astype(np.int64), test_idx.astype(np.int64)


def split_feature_blocks(features: np.ndarray, layout: FeatureLayout) -> Tuple[np.ndarray, np.ndarray]:
    """Split combined features into pooling and hydra blocks."""
    pooling_dim = layout.pooling_per_transform * len(TRANSFORM_NAMES)
    return features[:, :pooling_dim], features[:, pooling_dim:]


def extract_hydra_features(
    X_train: np.ndarray,  # [N, T]
    X_test: np.ndarray,   # [N, T]
    cfg: HydraEnsembleConfig,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Hydra features for training and test sets."""
    device = cfg.resolved_device()
    N, T = X_train.shape
    hydra_model = Hydra(
        input_length=T,
        k=cfg.k,
        g=cfg.g,
        seed=seed,
        device=device,
    )
    hydra_model.eval()

    def _transform(X: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(X).float().unsqueeze(1) # [N, C=1, T]
        with torch.no_grad():
            feats = hydra_model.batch(tensor.to(device), batch_size=cfg.batch_size)
        return feats.cpu().numpy()

    return _transform(X_train), _transform(X_test)


class KGMTPRebuild:
    """Wrapper combining base, Hilbert, and first-order-difference KG-MTP transforms."""

    def __init__(
        self,
        num_features: int = 50_000,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
    ) -> None:
        self.num_features = max(1, int(num_features))
        self.random_state = random_state
        self.n_jobs = max(1, n_jobs) if n_jobs != 0 else 1
        self.weights = create_kgmtp_weights()

        # Layout is set after fit_transform when actual output dims are known.
        self.layout: Optional[FeatureLayout] = None

        self._models: Dict[str, kgmtp_core.KGMTP] = {}

    def _prepare_transforms(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float64, order="C")
        hilbert = _hilbert_transform(X)
        diff = np.diff(X, axis=1)
        diff = np.asarray(diff, dtype=np.float64, order="C")
        return {"base": X, "hilbert": hilbert, "diff": diff}

    def fit_transform(self, X: np.ndarray, chunk_size: Optional[int] = None) -> np.ndarray:
        transformed = self._prepare_transforms(X)

        features_per_transform = self.num_features // len(TRANSFORM_NAMES)
        return_full = (
            chunk_size is None
            or chunk_size <= 0
            or chunk_size >= transformed["base"].shape[0]
        )

        pooling_blocks: List[np.ndarray] = []
        hydra_blocks: List[np.ndarray] = []
        for name in TRANSFORM_NAMES:
            model = kgmtp_core.KGMTP(
                num_features=features_per_transform,
                weights=self.weights,
                seed=self.random_state
            )
            pooled, hydra = model.fit(transformed[name], return_transform=return_full)
            if return_full:
                pooling_blocks.append(np.nan_to_num(pooled))
                hydra_blocks.append(hydra)
            self._models[name] = model

        if return_full:
            self.layout = FeatureLayout(
                pooling_per_transform=pooling_blocks[0].shape[1],
                hydra_per_transform=hydra_blocks[0].shape[1],
            )
            return self._combine_blocks(pooling_blocks, hydra_blocks)

        # Determine actual layout from fitted model
        first_model = self._models[TRANSFORM_NAMES[0]]
        sample_pool, sample_hydra = first_model.predict(transformed[TRANSFORM_NAMES[0]][:1])
        self.layout = FeatureLayout(
            pooling_per_transform=sample_pool.shape[1],
            hydra_per_transform=sample_hydra.shape[1],
        )
        return self._transform_with_prepared(transformed, chunk_size)

    def transform(self, X: np.ndarray, chunk_size: Optional[int] = None) -> np.ndarray:
        if not self._models:
            raise RuntimeError("KGMTPRebuild must be fitted before calling transform.")

        transformed = self._prepare_transforms(X)
        return self._transform_with_prepared(transformed, chunk_size)

    def _combine_blocks(
        self,
        pooling_blocks: List[np.ndarray],
        hydra_blocks: List[np.ndarray],
    ) -> np.ndarray:
        ordered = pooling_blocks + hydra_blocks
        if len(ordered) == 1:
            return ordered[0]
        return np.concatenate(ordered, axis=1)

    def _transform_with_prepared(
        self,
        transformed: Dict[str, np.ndarray],
        chunk_size: Optional[int],
    ) -> np.ndarray:
        total = transformed["base"].shape[0]
        if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
            pooling_blocks = []
            hydra_blocks = []
            for name in TRANSFORM_NAMES:
                pooled, hydra = self._models[name].predict(transformed[name])
                pooling_blocks.append(np.nan_to_num(pooled))
                hydra_blocks.append(hydra)
            return self._combine_blocks(pooling_blocks, hydra_blocks)

        chunk_size = int(chunk_size)
        combined_chunks: List[np.ndarray] = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            pooling_blocks = []
            hydra_blocks = []
            for name in TRANSFORM_NAMES:
                pooled, hydra = self._models[name].predict(transformed[name][start:end])
                pooling_blocks.append(np.nan_to_num(pooled))
                hydra_blocks.append(hydra)
            combined_chunks.append(self._combine_blocks(pooling_blocks, hydra_blocks))
        return np.vstack(combined_chunks)

