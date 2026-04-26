"""Run KG-MTP (3 transforms) with a per-split loss-based adaptive Savitzky-Golay gate.

This script is an alternative to `run_3trans.py`:
  - It proposes adaptive Savitzky-Golay parameters per dataset via
    `get_adaptive_transform_for_dataset` (noise/length heuristics).
  - Instead of validating with accuracy on full train-only resamples, it runs
    small loss-driven simulations on *subsampled* per-class training slices.
  - Uses a custom margin hinge-like loss on the classifier decision function.
  - Chooses the adaptive filter if the average validation loss does NOT exceed
    the baseline (no filter) loss by more than a tolerance.

Why loss? The classifier (RidgeClassifierCV) optimizes a regression objective;
direct access to its internal loss is not exposed, so we approximate with a
margin-based hinge surrogate:
  Binary: margin = y * decision; loss = max(0, 1 - margin)
  Multiclass: margin = f_correct - max(f_other); loss = max(0, 1 - margin)

Subset sampling rationale: Using only a small number of samples per class for
the validation simulations reduces overfitting in the decision gate and speeds
up filtering choice. The main evaluation still uses the full data.

We use the Welford algorithm to keep the number of simulations low.

Parameters you can tune below:
  SUBSET_PER_CLASS         -> Max samples per class for gate simulations
  MAX_VALIDATION_RESAMPLES -> Maximum number of simulations
  LOSS_TOLERANCE           -> Allowed *increase* in loss to still keep adaptive

Decision rule:
  Keep adaptive filter if adaptive_loss <= baseline_loss + LOSS_TOLERANCE
  Else fall back to baseline (no filtering).



The adaptive Savitzky-Golay filter determination is performed separately for
each resample split.

The results are logged in two formats:
1. Wide format (Main): Datasets as rows, Splits as columns + aggregates.
2. Long format (Details): Detailed per-split metadata (time, window, poly).

The CSVs are written incrementally to: results/kgtmp_3_trans_loss_gate*.csv

For each dataset and for each split, the script writes out arithmetic mean,
minimum, maximum, and standard deviation.
"""

import sys
from libs.misc import is_ipython_notebook

# Only show this when actually invoked from CLI with no args (not in notebooks)
if not is_ipython_notebook() and len(sys.argv) == 1:
    print(f"Running with default parameters; execute `{sys.argv[0]} --help` to see all available options.\n")

import config

# Cheeky hack for better user experience
print(f"Compiling NUMBA optimized code -- this will take approximately 5 minutes on the first run...")

import os
import time
import csv
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
import hashlib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import roc_auc_score, f1_score

from aeon.transformations.collection.convolution_based import MultiRocket

from reference.kg_mtp_rebuild import (
    HydraEnsembleConfig,
    find_dataset_files,
    load_ucr_split,
    KGMTPRebuild,
    split_feature_blocks,
    prepare_resample_indices,
    extract_hydra_features,
)
from libs.hydra_basic import SparseScaler
from libs.adaptive_transform import get_adaptive_transform, apply_pretransform
from libs.misc import pseudo_nll_from_dec

# Suppress CUDA capability warning on GB10 (sm_121) until a build with sm_121 exists
warnings.filterwarnings(
    "ignore",
    message=r"Found GPU0 NVIDIA GB10 which is of cuda capability 12\.1\.",
    category=UserWarning,
)


HYDRA_CFG = HydraEnsembleConfig(
    # Change if you want non-default values
    # These are the KG-MTP defaults
    # k=16,
    # g=64,
    # batch_size=256,

    # Set to "cpu" if you prefer to wait
    device="auto",
)

def extract_mr_features(
        X_train: np.ndarray,
        X_test: np.ndarray,
        seed: int,
        n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract MultiRocket features."""
    mr_model = MultiRocket(
        random_state=seed,
        n_jobs=n_jobs,
    )
    mr_model.fit(X_train)
    X_train_mr: np.ndarray = mr_model.transform(X_train) # type: ignore
    X_test_mr: np.ndarray = mr_model.transform(X_test) # type: ignore

    return X_train_mr, X_test_mr

def compute_margin_loss(decisions: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> float:
    """Compute hinge-like margin loss from decision function outputs.

    Args:
        decisions: (n_samples,) for binary or (n_samples, n_classes) for multiclass.
        y_true: Encoded integer labels matching indices in classes.
        classes: Array of class labels in the same order as classifier output.

    Returns:
        Mean hinge-like loss (float).
    """
    if decisions.ndim == 1:  # binary case
        # Map y_true integers to +/-1 according to classes ordering
        positive_class = classes[1]
        signs = np.where(y_true == np.where(classes == positive_class)[0][0], 1.0, -1.0)
        margin = signs * decisions
        loss = np.maximum(0.0, 1.0 - margin)
        return float(np.mean(loss))
    else:  # multiclass hinge: f_correct - max(f_other)
        correct_scores = decisions[np.arange(len(y_true)), y_true]
        mask = np.ones_like(decisions, dtype=bool)
        mask[np.arange(len(y_true)), y_true] = False
        other_max = np.max(np.where(mask, decisions, -np.inf), axis=1)
        margin = correct_scores - other_max
        loss = np.maximum(0.0, 1.0 - margin)
        return float(np.mean(loss))


def _compute_pipeline_loss(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    window: int,
    poly: int,
    model_seed: int,
) -> float:
    """Apply filtering -> KG-MTP extraction -> Ridge classifier -> Return margin loss."""
    if window > 0 and poly > 0:
        X_train_f = apply_pretransform(X_train)
        X_val_f = apply_pretransform(X_val)
    else:
        X_train_f = X_train
        X_val_f = X_val

    # IMPORTANT: keep random features identical between baseline/adaptive comparisons.
    # Otherwise, even identical X can yield different features and a non-zero delta.
    kgmtp = KGMTPRebuild(num_features=config.NUM_FEATURES, random_state=model_seed, n_jobs=config.N_JOBS)
    X_train_feats = kgmtp.fit_transform(X_train_f)
    X_val_feats = kgmtp.transform(X_val_f)

    layout = kgmtp.layout
    X_train_pool, X_train_hydra = split_feature_blocks(X_train_feats, layout)
    X_val_pool, X_val_hydra = split_feature_blocks(X_val_feats, layout)

    pool_scaler = StandardScaler()
    X_train_pool_scaled = pool_scaler.fit_transform(X_train_pool)
    X_val_pool_scaled = pool_scaler.transform(X_val_pool)

    hydra_block_scaler = SparseScaler()
    X_train_hydra_scaled = hydra_block_scaler.fit_transform(
        torch.from_numpy(X_train_hydra.astype(np.float32))
    ).numpy()
    X_val_hydra_scaled = hydra_block_scaler.transform(
        torch.from_numpy(X_val_hydra.astype(np.float32))
    ).numpy()

    X_train_all = np.c_[X_train_pool_scaled, X_train_hydra_scaled]
    X_val_all = np.c_[X_val_pool_scaled, X_val_hydra_scaled]

    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(X_train_all, y_train)
    decisions = clf.decision_function(X_val_all)
    return compute_margin_loss(decisions, y_val, clf.classes_)

def _hash(
    s: Optional[str] = None,
    seed0: Optional[int] = None,
    seed1: Optional[int] = None,
) -> int:
    """Return a 32-bit signed hash of the inputs all hashed together."""
    assert s is not None or seed0 is not None or seed1 is not None

    h = hashlib.blake2s(digest_size=4)

    if s is not None:
        h.update(s.encode("utf-8"))

    for seed in (seed0, seed1):
        if seed is not None:
            # 8 bytes = handle 64-bit inputs gracefully
            h.update(seed.to_bytes(8, "little", signed=False))

    return int.from_bytes(h.digest(), "little") & 0x7fff_ffff # clamp to 0..INT32_MAX


def run_loss_gate_on_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prop_window: int,
    prop_poly: int,
    heuristic_expl: str,
    seed: int,
    dataset_name: str,
    split_index: int,  # 1-based
) -> Tuple[int, int, str]:
    """Determine (window, poly) using loss-based validation gate on a specific split.

    - Use Welford's algorithm to track the mean/variance of (adaptive_loss - baseline_loss).
    - Apply a simple sequential test with early stopping based on a confidence interval around the mean delta.
    """
    if config.GATE_MAX_SIMS <= 0:
        msg = f"Gate skipped ({config.GATE_MAX_SIMS=} is less than or equal to 0); no simulations configured"
        print(f"    [Loss Gate] {msg}", flush=True)
        return 0, 0, msg

    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_train)
    n_classes = len(unique_classes)

    baseline_losses: List[float] = []
    adaptive_losses: List[float] = []
    decisions_prefix: List[int] = []  # 1 = keep adaptive, 0 = reject (based on mean delta vs tolerance)

    # Welford state for delta_i = adaptive_loss_i - baseline_loss_i
    n = 0
    mean_delta = 0.0
    M2_delta = 0.0

    early_decision: Optional[bool] = None  # True=KEEP, False=REJECT, None=not decided

    print("  [Loss Gate] Running subset simulations...", flush=True)

    # We append rows to disk incrementally
    def _append_gate_sim_row(
        sim_idx: int,
        base_loss: float,
        adapt_loss: float,
        base_avg: float,
        adapt_avg: float,
        delta_avg: float,
        decision_keep: bool,
        subset_size: int,
        sim_seed: int,
    ) -> None:
        decision_int = 1 if decision_keep else 0
        with open(config.GATE_SIMS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dataset_name,
                split_index,
                sim_idx,
                f"{base_loss:.8f}",
                f"{adapt_loss:.8f}",
                f"{base_avg:.8f}",
                f"{adapt_avg:.8f}",
                f"{delta_avg:.8f}",
                decision_int,
                prop_window,
                prop_poly,
                heuristic_expl,
                f"{config.GATE_LOSS_TOLERANCE:.8f}",
                subset_size,
                n_classes,
                seed,
                sim_seed,
            ])

    for sim_idx in range(config.GATE_MAX_SIMS):
        # Build subset indices per class
        subset_indices = []
        min_class_size = float('inf')
        for c in unique_classes:
            cls_indices = np.where(y_train == c)[0]
            if len(cls_indices) > config.GATE_MAX_CLASS_SAMPLES:
                chosen = rng.choice(cls_indices, config.GATE_MAX_CLASS_SAMPLES, replace=False)
            else:
                chosen = cls_indices
            subset_indices.append(chosen)
            min_class_size = min(min_class_size, len(chosen))

        if min_class_size < 2:
            msg = f"Gate skipped (min class size {min_class_size} < 2); {heuristic_expl}"
            print(f"    {msg}", flush=True)
            return 0, 0, msg

        subset_indices = np.concatenate(subset_indices)
        rng.shuffle(subset_indices)

        subset_X = X_train[subset_indices]
        subset_y = y_train[subset_indices]
        subset_n = len(subset_y)

        val_count = max(int(0.3 * subset_n), n_classes)
        max_val_count = subset_n - n_classes
        if max_val_count < n_classes:
            msg = f"Gate skipped (subset too small for stratified split); {heuristic_expl}"
            print(f"    {msg}", flush=True)
            return 0, 0, msg
        val_count = min(val_count, max_val_count)

        sim_seed = _hash("loss gate inner split", seed, sim_idx)
        inner_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_count,
            random_state=sim_seed,
        )
        try:
            (train_idx_sub, val_idx_sub) = next(inner_split.split(np.zeros((subset_n, 1)), subset_y))
        except ValueError:
            msg = f"Gate skipped (stratified split failed); {heuristic_expl}"
            print(f"    {msg}", flush=True)
            return 0, 0, msg

        X_sub_train = subset_X[train_idx_sub]
        y_sub_train = subset_y[train_idx_sub]
        X_sub_val = subset_X[val_idx_sub]
        y_sub_val = subset_y[val_idx_sub]

        # Use the same model_seed for baseline/adaptive inside this simulation.
        model_seed = _hash("loss gate kgmtp", seed, sim_idx)
        base_loss = _compute_pipeline_loss(
            X_sub_train, y_sub_train, X_sub_val, y_sub_val, 0, 0, model_seed
        )
        baseline_losses.append(base_loss)

        adapt_loss = _compute_pipeline_loss(
            X_sub_train, y_sub_train, X_sub_val, y_sub_val, prop_window, prop_poly, model_seed
        )
        adaptive_losses.append(adapt_loss)

        if (base_loss > 1_000 or base_loss < -1_000) or (adapt_loss > 1_000 or adapt_loss < -1_000):
            # Rarely, the loss blows up (we have observed values in the 10e11+ range) -- this is fine
            # Because we take the raw Ridge values, sometimes these are way out of proportion. However, we do not have
            # a good way to normalize these values in a way that would actually help us.
            # When this happens, that event will dominate the loss calculation. The gate in general, and the Welford algorithm
            # in particular, handles this well, so we do not handle it as a special case.
            pass

        # Arithmetic means for logging / interpretability
        base_avg = float(np.mean(baseline_losses))
        adapt_avg = float(np.mean(adaptive_losses))
        delta_avg = adapt_avg - base_avg  # should coincide with mean_delta from Welford

        # Prefix decision based on arithmetic mean (for CSV + binary sequence)
        decision_keep_prefix = delta_avg <= config.GATE_LOSS_TOLERANCE

        _append_gate_sim_row(
            sim_idx=sim_idx + 1,              # 1-based
            base_loss=base_loss,
            adapt_loss=adapt_loss,
            base_avg=base_avg,
            adapt_avg=adapt_avg,
            delta_avg=delta_avg,
            decision_keep=decision_keep_prefix,
            subset_size=subset_n,
            sim_seed=sim_seed,
        )

        # --- Welford update for delta_i = adapt_loss - base_loss ---
        d_i = adapt_loss - base_loss
        n += 1
        if n == 1:
            mean_delta = d_i
            M2_delta = 0.0
        else:
            delta = d_i - mean_delta
            mean_delta += delta / n
            M2_delta += delta * (d_i - mean_delta)

        if n > 1:
            var_delta = M2_delta / (n - 1)
            se_delta = float(np.sqrt(var_delta / n)) if var_delta > 0 else 0.0
            ci_low = mean_delta - config.Z_CONF * se_delta
            ci_high = mean_delta + config.Z_CONF * se_delta
        else:
            se_delta = float("inf")
            ci_low = float("nan")
            ci_high = float("nan")

        # Status wrt sequential confidence interval test
        status = "continue"
        if n >= config.GATE_MIN_SIMS and n > 1 and se_delta != float("inf"):
            if ci_high < config.GATE_LOSS_TOLERANCE:
                status = "early KEEP"
                early_decision = True
            elif ci_low > config.GATE_LOSS_TOLERANCE:
                status = "early REJECT"
                early_decision = False

        if config.DEBUG_PRINT_GATE_PROGRESS:
            decisions_prefix.append(1 if decision_keep_prefix else 0)
            seq_str = "".join("1" if d == 1 else "0" for d in decisions_prefix)
            flips = sum(
                1 for i in range(1, len(decisions_prefix))
                if decisions_prefix[i] != decisions_prefix[i - 1]
            )

            print(
                f"    Gate split {sim_idx+1}/{config.GATE_MAX_SIMS}: "
                f"baseline_loss={base_loss:.4f} adaptive_loss={adapt_loss:.4f} | "
                f"avg_base={base_avg:.4f} avg_adapt={adapt_avg:.4f} "
                f"(meanΔ={delta_avg:+.4f}, Welford_meanΔ={mean_delta:+.4f}) | "
                f"CI=[{ci_low:+.4f}, {ci_high:+.4f}] LT=+{config.GATE_LOSS_TOLERANCE:.4f} "
                f"status={status} | "
                f"prefix_dec={'KEEP' if decision_keep_prefix else 'REJECT'} "
                f"seq={seq_str} flips={flips}",
                flush=True,
            )

        if early_decision is not None:
            print(
                f"    Early stopping after {n} simulations: "
                f"{'KEEP' if early_decision else 'REJECT'} "
                f"(meanΔ={mean_delta:+.4f}, CI=[{ci_low:+.4f}, {ci_high:+.4f}], "
                f"LT=+{config.GATE_LOSS_TOLERANCE:.4f})",
                flush=True,
            )
            break

    # Final decision based on simulations run (may be early-stopped)
    base_avg = float(np.mean(baseline_losses))
    adapt_avg = float(np.mean(adaptive_losses))
    delta = adapt_avg - base_avg

    if early_decision is None:
        # Fall back to the original mean-based rule
        keep = adapt_avg <= base_avg + config.GATE_LOSS_TOLERANCE
    else:
        keep = early_decision

    if keep:
        msg = (
            f"Adaptive kept (loss {adapt_avg:.4f} vs baseline {base_avg:.4f}, "
            f"delta={delta:+.4f} ; early={early_decision is not None}); "
            f"{heuristic_expl}"
        )
        print(f"  Decision: {msg}", flush=True)
        return prop_window, prop_poly, msg
    else:
        msg = (
            f"Adaptive rejected (loss {adapt_avg:.4f} vs baseline {base_avg:.4f}, "
            f"delta={delta:+.4f} ; early={early_decision is not None}); "
            f"{heuristic_expl}"
        )
        print(f"  Decision: {msg}", flush=True)
        return 0, 0, msg


def evaluate_dataset_per_split(
    dataset_dir: Path,
    num_features: int,
    num_resamples: int,
    random_state: int,
    n_jobs: int,
    hydra_cfg: HydraEnsembleConfig,
    dataset_name: str,
) -> List[Dict]:
    """Evaluate dataset with per-split gating."""
    train_path, test_path = find_dataset_files(dataset_dir)
    X_train_raw, y_train_raw = load_ucr_split(train_path)
    X_test_raw, y_test_raw = load_ucr_split(test_path)

    # Calculate total splits for display
    total_splits = num_resamples if num_resamples > 0 else 1

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    X_all = np.vstack([X_train_raw, X_test_raw])
    y_all = np.hstack([y_train, y_test])


    N, T = X_train_raw.shape
    n_classes = len(np.unique(y_train))
    assert n_classes >= 2, f"Need at least 2 classes to classify, got {n_classes}"
    is_binary = n_classes == 2

    results = []
    print(f"StratifiedShuffleSplit {num_resamples}x with random seed {random_state}...")
    for split_idx, (train_idx, test_idx) in enumerate(
        prepare_resample_indices(y_train, y_test, num_resamples, random_state,
                                 dataset_name=dataset_name,)
    ):
        # Adjust split display index to be 1-based
        current_split_num = split_idx + 1

        print(f"\n--- Split {current_split_num}/{total_splits} ---", flush=True)
        start = time.perf_counter()

        X_train_split = X_all[train_idx]
        y_train_split = y_all[train_idx]
        X_test_split = X_all[test_idx]
        y_test_split = y_all[test_idx]

        apply_f, prop_w, prop_p, heuristic_expl = get_adaptive_transform(
            X_train=X_train_split,
            y_train=y_train_split,
        )

        chosen_w, chosen_p = 0, 0
        decision_expl = "Disabled (heuristic)"
        gate_seed, kgmtp_seed, hydra_seed = None, None, None

        if apply_f:
            gate_seed = _hash("gate", random_state, split_idx)
            print(f"{gate_seed=}")
            chosen_w, chosen_p, decision_expl = run_loss_gate_on_split(
                X_train_split, y_train_split, prop_w, prop_p, heuristic_expl, gate_seed, dataset_name, current_split_num
            )
        else:
            print(f"  Decision: Heuristic says no filter ({heuristic_expl})", flush=True)

        if chosen_w > 0 and chosen_p > 0:
            X_train_proc = apply_pretransform(X_train_split)
            X_test_proc = apply_pretransform(X_test_split)
        else:
            X_train_proc = X_train_split
            X_test_proc = X_test_split

        kgmtp_seed = int(os.environ.get("DEBUG_KGMTP_SEED", 0)) or _hash("KGMTP", random_state, split_idx)
        print(f"{kgmtp_seed=}")
        kgmtp = KGMTPRebuild(
            num_features=num_features,
            random_state=kgmtp_seed,
            n_jobs=n_jobs,
        )
        X_train_features = kgmtp.fit_transform(X_train_proc)
        X_test_features = kgmtp.transform(X_test_proc)

        layout = kgmtp.layout
        X_train_pool, X_train_hydra = split_feature_blocks(X_train_features, layout)
        X_test_pool, X_test_hydra = split_feature_blocks(X_test_features, layout)

        pool_scaler = StandardScaler()
        X_train_pool_scaled = pool_scaler.fit_transform(X_train_pool)
        X_test_pool_scaled = pool_scaler.transform(X_test_pool)

        hydra_block_scaler = SparseScaler()
        X_train_hydra_scaled = hydra_block_scaler.fit_transform(
            torch.from_numpy(X_train_hydra.astype(np.float32))
        ).numpy()
        X_test_hydra_scaled = hydra_block_scaler.transform(
            torch.from_numpy(X_test_hydra.astype(np.float32))
        ).numpy()

        X_train_final = np.c_[X_train_pool_scaled, X_train_hydra_scaled]
        X_test_final = np.c_[X_test_pool_scaled, X_test_hydra_scaled]


        # Extract Hydra features - we always need those

        hydra_seed = _hash("hydra", random_state, split_idx)
        print(f"{hydra_seed=}")
        X_train_h_raw, X_test_h_raw = extract_hydra_features(
            X_train_split, X_test_split, hydra_cfg, hydra_seed
        )

        h_scaler = SparseScaler()
        X_train_h_only = h_scaler.fit_transform(
            torch.from_numpy(X_train_h_raw.astype(np.float32))
        ).numpy()
        X_test_h_only = h_scaler.transform(
            torch.from_numpy(X_test_h_raw.astype(np.float32))
        ).numpy()

        if config.KG_ONLY:
            # Match upstream: StandardScaler on combined [raw_pooling | SparseScaled_hydra]
            combined_scaler = StandardScaler()
            X_train_kg = combined_scaler.fit_transform(np.c_[X_train_pool, X_train_hydra_scaled])
            X_test_kg = combined_scaler.transform(np.c_[X_test_pool, X_test_hydra_scaled])

            kg_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            kg_clf.fit(X_train_kg, y_train_split)
            kg_dec = kg_clf.decision_function(X_test_kg)
            kg_accuracy = float(kg_clf.score(X_test_kg, y_test_split))
            classes = kg_clf.classes_
            if kg_dec.ndim == 1:
                kg_pred = np.where(kg_dec >= 0, classes[1], classes[0])
            else:
                kg_pred = classes[np.argmax(kg_dec, axis=1)]
        else:
            # Create the monstrous concatenation of Hydra + MultiRocket + KG

            # MultiRocket

            mr_seed = hydra_seed
            X_train_mr_raw, X_test_mr_raw = extract_mr_features(
                X_train_split,
                X_test_split,
                seed=mr_seed,  # type: ignore
                n_jobs=config.N_JOBS,
            )

            mr_scaler = StandardScaler()
            X_train_mr = mr_scaler.fit_transform(X_train_mr_raw)
            X_test_mr = mr_scaler.transform(X_test_mr_raw)

            if config.WEIGHT_MR > 0:
                mr_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                mr_clf.fit(X_train_mr, y_train_split)
                mr_dec = mr_clf.decision_function(X_test_mr)
                mr_accuracy = float(mr_clf.score(X_test_mr, y_test_split))
            else:
                mr_dec = None
                mr_accuracy = 0.0

            # MR + Hydra

            if config.WEIGHT_MRH > 0:
                X_train_mrh = np.concatenate((X_train_mr, X_train_h_only), axis=1)
                X_test_mrh = np.concatenate((X_test_mr, X_test_h_only), axis=1)

                mrh_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                mrh_clf.fit(X_train_mrh, y_train_split)
                mrh_dec = mrh_clf.decision_function(X_test_mrh)
                mrh_accuracy = float(mrh_clf.score(X_test_mrh, y_test_split))
            else:
                mrh_dec = None
                mrh_accuracy = 0.0


            # KG + Hydra + Multirocket

            X_train_all = np.concatenate((X_train_mr, X_train_h_only, X_train_pool_scaled), axis=1)
            X_test_all = np.concatenate((X_test_mr, X_test_h_only, X_test_pool_scaled), axis=1)

            all_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            all_clf.fit(X_train_all, y_train_split)
            all_dec = all_clf.decision_function(X_test_all)
            all_accuracy = float(all_clf.score(X_test_all, y_test_split))

            classes = all_clf.classes_
            if all_dec.ndim == 1:
                all_pred = np.where(all_dec >= 0, classes[1], classes[0])
            else:
                all_pred = classes[np.argmax(all_dec, axis=1)]

            # TODO refactor
            kg_clf, kg_dec, kg_pred, kg_accuracy = all_clf, all_dec, all_pred, all_accuracy

        ensemble_dec, ensemble_accuracy, ensemble_pred = kg_dec, kg_accuracy, kg_pred
        hydra_accuracy = 0.0
        weight = 0.0
        hydra_seed = None


        if config.WEIGHT_HYDRA > 0:

            hydra_clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            hydra_clf.fit(X_train_h_only, y_train_split)
            hydra_dec = hydra_clf.decision_function(X_test_h_only)
            hydra_accuracy = float(hydra_clf.score(X_test_h_only, y_test_split))

            weight = float(np.clip(config.WEIGHT_HYDRA, 0.0, 1.0))

            ensemble_dec = (1.0 - weight) * kg_dec + weight * hydra_dec
            if kg_dec.ndim == 1:
                ensemble_pred = np.where(ensemble_dec >= 0, classes[1], classes[0])
            else:
                ensemble_pred = classes[np.argmax(ensemble_dec, axis=1)]

            ensemble_accuracy = float(np.mean(ensemble_pred == y_test_split))
            print(
                f"    hydra_acc={hydra_accuracy:.4f} kg_acc={kg_accuracy:.4f} ensemble={ensemble_accuracy:.4f} (w={weight:.2f})",
                flush=True,
            )
        else:
            print(
                f"    hydra skipped (weight={config.WEIGHT_HYDRA:.2f}); kg_acc={kg_accuracy:.4f}",
                flush=True,
            )
        
        durations = time.perf_counter() - start

        # Compute auxiliary aggregates, and fall back to NaNs in case of error
        NAN = float("nan")
        assert np.isnan(NAN)
        try:
            if is_binary:
                # For binary, apply sigmoid to normalize decision values to [0, 1]
                ensemble_proba = 1.0 / (1.0 + np.exp(-ensemble_dec))
            else:
                # For multiclass, convert decision scores to probabilities using softmax
                ensemble_proba = softmax(ensemble_dec, axis=1)
            # XXX FIXME ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'
            ensemble_auroc = roc_auc_score(
                y_test_split,
                ensemble_proba,
                **({} if is_binary else {"multi_class": "ovr", "average": "macro"}),
            )
        except Exception as e:
            decision_expl += (
                f"{decision_expl}; roc_auc_score could not be computed: "
                f"{e.__class__.__name__}: {e}"
            )
            ensemble_auroc = NAN

        try:
            ensemble_f1 = f1_score(
                y_test_split,
                ensemble_pred,
                **({} if is_binary else {"average": "macro"}),
            )
        except Exception as e:
            decision_expl += (
                f"{decision_expl}; f1_score could not be computed: "
                f"{e.__class__.__name__}: {e}"
            )
            ensemble_f1 = NAN

        try:
            ensemble_nll = pseudo_nll_from_dec(
                y_true=y_test_split,
                decisions=ensemble_dec,
                classes=classes,
            )
        except Exception as e:
            decision_expl += (
                f"{decision_expl}; pseudo_nll_from_dec could not be computed: "
                f"{e.__class__.__name__}: {e}"
            )
            ensemble_nll = NAN

        
        results.append({
            "split_idx": current_split_num,
            "accuracy": ensemble_accuracy,
            "auroc": ensemble_auroc,
            "f1": ensemble_f1,
            "nll": ensemble_nll,
            "time": durations,
            "window": chosen_w,
            "poly": chosen_p,
            "random_state": random_state,
            "gate_seed": gate_seed,
            "kgmtp_seed": kgmtp_seed,
            "hydra_seed": hydra_seed,
            "notes": decision_expl,
            "train_idx": train_idx.tolist() if train_idx is not None else None,
            "test_idx": test_idx.tolist() if test_idx is not None else None,
        })

        print(
            f"  Split {current_split_num}/{total_splits} "
            f"- accuracy={ensemble_accuracy:.4f} time={durations:.1f}s",
            flush=True,
        )

    return results


def main():
    print(f"NUMBA compilation completed successfully.")
    if not config.DATASETS_DIR.exists():
        print(f"Datasets directory not found: {config.DATASETS_DIR}")
        return

    if config.DATASETS_FILE.exists():
        with open(config.DATASETS_FILE, "r") as f:
            datasets_to_run = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: list directories in DATASETS_DIR
        datasets_to_run = [d.name for d in config.DATASETS_DIR.iterdir() if d.is_dir()]
        datasets_to_run.sort()

    config.OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine columns based on NUM_RESAMPLES (0 -> 1 original split, N -> N resamples)
    n_cols = config.NUM_SPLITS if config.NUM_SPLITS > 0 else 1

    # Initialize CSV files
    split_cols = [f"split_{i+1}" for i in range(n_cols)]
    stat_cols = [
        "min",
        "max",
        "mean",
        "median",
        "std",
        "auroc",              # area under the ROC curve
        "f1",                 # F1 score
        "nll",                # negative log-likelihood (log loss)
        "time",              # average time to run the dataset (seconds)
    ]
    csv_headers = ["dataset"] + split_cols + stat_cols
    
    if not config.OUTPUT_CSV.exists():
        with open(config.OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            
    details_headers = ["dataset", "split", "accuracy", "time", "window", "poly",
            "random_state", "gate_seed", "kgmtp_seed", "hydra_seed",
            "train_idx", "test_idx",
            "notes",
    ]
    if not config.DETAILS_CSV.exists():
        with open(config.DETAILS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(details_headers)

    # Per-simulation gate logging - only write headers here
    gate_sims_headers = [
        "dataset",        # dataset name
        "split",          # 1-based split index
        "sim_idx",        # 1-based gate simulation index
        "baseline_loss",
        "adaptive_loss",
        "baseline_avg",   # avg baseline loss up to this sim
        "adaptive_avg",   # avg adaptive loss up to this sim
        "delta",          # adaptive_avg - baseline_avg
        "decision",       # 1 = keep adaptive, 0 = reject, at this prefix
        "window",         # proposed window (prop_window)
        "poly",           # proposed poly (prop_poly)
        "heuristic",      # heuristic explanation string
        "loss_tolerance", # LOSS_TOLERANCE at run time
        "subset_size",    # number of samples used in this subset
        "num_classes",    # num classes in dataset
        "gate_seed",      # gate RNG seed
        "sim_seed",       # per-simulation StratifiedShuffleSplit seed
    ]
    if not config.GATE_SIMS_CSV.exists():
        with open(config.GATE_SIMS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(gate_sims_headers)

    print(f"Starting evaluation on {len(datasets_to_run)} datasets...")
    print(f"Results will be saved to {config.OUTPUT_CSV}")

    for ds_idx, ds_name in enumerate(datasets_to_run, 1):
        dataset_path = config.DATASETS_DIR / ds_name
        if not dataset_path.exists():
            print(f"Dataset path not found: {dataset_path}, skipping.")
            continue

        print(f"\n=== {ds_idx}/{len(datasets_to_run)} {ds_name} ===", flush=True)
        try:
            seed = _hash(ds_name, config.RANDOM_STATE)
            results = evaluate_dataset_per_split(
                dataset_dir=dataset_path,
                num_features=config.NUM_FEATURES,
                num_resamples=config.NUM_SPLITS,
                random_state=seed,
                n_jobs=config.N_JOBS,
                hydra_cfg=HYDRA_CFG,
                dataset_name=ds_name,
            )

            # 1. Write Details
            with open(config.DETAILS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                for res in results:
                    writer.writerow([
                        ds_name,
                        res['split_idx'],
                        f"{res['accuracy']:.8f}",
                        f"{res['time']:.2f}",
                        res['window'],
                        res['poly'],
                        res['random_state'],
                        res['gate_seed'],
                        res['kgmtp_seed'],
                        res['hydra_seed'],
                        res['train_idx'],
                        res['test_idx'],
                        res['notes'],
                    ])

            # 2. Write Wide Row (Main)
            accuracies = [r['accuracy'] for r in results]
            aurocs = [r["auroc"] for r in results]
            f1_scores = [r["f1"] for r in results]
            nlls = [r["nll"] for r in results]
            train_predict_times = [r["time"] for r in results]
            row_stats = [
                np.min(accuracies) if accuracies else "",
                np.max(accuracies) if accuracies else "",
                np.mean(accuracies) if accuracies else "",
                np.median(accuracies) if accuracies else "",
                np.std(accuracies) if accuracies else "",
                np.mean(aurocs) if aurocs else "",
                np.mean(f1_scores) if f1_scores else "",
                np.mean(nlls) if nlls else "",
                np.mean(train_predict_times) if train_predict_times else "",
            ]
            
            # Pad if fewer results than expected (e.g. error)
            acc_cells = [""] * n_cols
            for i, acc in enumerate(accuracies):
                if i < n_cols:
                    acc_cells[i] = f"{acc:.8f}"
            
            final_row = [ds_name] + acc_cells + [f"{v:.8f}" if isinstance(v, float) else v for v in row_stats]
            
            with open(config.OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(final_row)
            
            print(f"Completed {ds_name}: Mean Acc={np.mean(accuracies):.4f}")
            
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # --- Final Aggregation (Column-wise) ---
    print("\nRunning final aggregation...")
    try:
        df = pd.read_csv(config.OUTPUT_CSV)
        # Filter out existing aggregate rows if re-running
        df = df[~df['dataset'].isin(['MEAN', 'MIN', 'MAX', 'STD'])]
        
        # Columns to aggregate: split_1..split_N + stats
        numeric_cols = split_cols + stat_cols
        
        aggs = {}
        aggs['MEAN'] = df[numeric_cols].mean()
        aggs['MIN'] = df[numeric_cols].min()
        aggs['MAX'] = df[numeric_cols].max()
        aggs['STD'] = df[numeric_cols].std()
        
        with open(config.OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            for label, series in aggs.items():
                # Construct row: [label, val_split1, ..., val_stat]
                row = [label]
                for col in numeric_cols:
                    val = series.get(col, np.nan)
                    row.append(f"{val:.8f}")
                writer.writerow(row)
                
        print("Aggregation complete.")
        
    except Exception as e:
        print(f"Error during aggregation: {e}")

if __name__ == "__main__":
    if is_ipython_notebook():
        pass  # Avoid running main automatically in notebooks
    else:
        main()

