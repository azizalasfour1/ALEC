import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Literal

from libs.misc import is_ipython_notebook


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

def _env_bool(name: str, default: bool) -> bool:
    try:
        _value = str(os.getenv(name, default)).lower().strip()
        if _value in ["1", "true", "yes", "on"]:
            return True
        elif _value in ["0", "false", "no", "off"]:
            return False
    except (TypeError, ValueError):
        pass
    return default

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, str(default))


def _get_weight(algo: Literal["hydra","kg","mr","mrh"], default: float|int) -> float|int:
    env_var_name = f"{algo.upper()}_WEIGHT"
    _weight = os.getenv(env_var_name, None)
    # keep the boundaries a nice and crisp int, use a float for everything in between
    # this should help us avoid float comparison and rounding errors
    if _weight is None:
        return default
    _weight = int(_weight) if _weight in ["0", "1"] else float(_weight)
    assert 0 <= _weight <= 1, f"{env_var_name} must be between 0 and 1 - got: {_weight}"
    return _weight

VALID_TRANSFORMS=["BASELINE", "MA", "EXP", "GF", "SG", "DFT", "SIV_PRIME"]

# Standard normal critical values Z_p (upper tail)
# Source (verbatim table):
#   NIST/SEMATECH e-Handbook of Statistical Methods
#   Section 1.3.6.7 — "Critical Values of the Normal Distribution"
#   https://www.itl.nist.gov/div898/handbook/eda/section3/eda3671.htm
#
# Definition:
#   Z_p satisfies P(Z <= Z_p) = p for Z ~ N(0, 1)
#
# NIST table excerpt (p >= 0.90):
#   p     : 0.900  0.950  0.975  0.990  0.995  0.999
#   Z_p   : 1.282  1.645  1.960  2.326  2.576  3.090
#
# Keys are fixed-width decimal strings to avoid float equality issues.
Z_P_LOOKUP = {
    "0.900": 1.282,
    "0.950": 1.645,
    "0.975": 1.960,
    "0.990": 2.326,
    "0.995": 2.576,
    "0.999": 3.090,
}

def print_effective_args(args: argparse.Namespace) -> None:
    t = args.transform

    transform_order = ["window", "polyorder", "radius", "numerator"]

    transform_visible = set()
    if t in {"MA", "EXP", "GF", "SG"}:
        transform_visible.add("window")
    if t == "SG":
        transform_visible.add("polyorder")
    if t == "DFT":
        transform_visible.add("radius")
    if t == "SIV_PRIME":
        transform_visible.add("numerator")

    print("Effective arguments:")

    # 1. print non-transform args in argparse order
    for k in vars(args):
        if k not in transform_order:
            print(f"  {k} = {getattr(args, k)}")

    # 2. print transform args in fixed order
    for k in transform_order:
        if k in transform_visible:
            print(f"  {k} = {getattr(args, k)}")


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=(
            "Runner for Adaptive Loss probing Ensemble Classifier (ALEC)"
        )
    )

    classifier_group = parser.add_argument_group("classifier selection")
    classifier_group.add_argument(
        "--classifier",
        type=str.lower,
        default="alec",
        action="store",
        choices=["alec", "kg_base", "kg_only"],
        help=(
            "Allows emulation of a base classifier by setting the corresponding weight to 1 "
            "and others to 0, and optionally disabling the gate by setting GATE_MAX_SIMS=0.\n"
            "  alec (default): full ALEC ensemble with configured weights and gate\n"
            "  kg_base: only KG-MTP (no gate or ensembling)\n"
            "  kg_only: only KG-MTP with the gate enabled (no ensembling)"
        ),
    )

    # Splits
    splits = parser.add_argument_group("splits")
    splits.add_argument(
        "--num-splits",
        type=int,
        default=_env_int("NUM_SPLITS", 1),
        help=(
            "Number of random splits to evaluate.\n"
            "First split is always the original UCR split."
        ),
    )

    # Reproducibility
    repro_args = parser.add_argument_group("reproducibility")
    repro_args.add_argument(
        "--seed",
        type=int,
        default=1337,
        help=(
            "Master seed for all stochastic components."
        )
    )

    # Compute
    compute_args = parser.add_argument_group("compute")
    compute_args.add_argument(
        "--jobs",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Number of parallel workers.\n"
            "Use -1 to use all available CPU cores."
        )
    )

    gate_args = parser.add_argument_group("gate")
    gate_args.add_argument(
        "--gate-max-class-samples",
        type=int,
        default=_env_int("GATE_MAX_CLASS_SAMPLES", 40),
        metavar="N",
        help=(
            "Maximum samples per class used in each gate simulation.\n"
            "Controls per-simulation class subsampling."
        )
    )
    gate_args.add_argument(
        "--gate-min-sims",
        type=int,
        default=_env_int("GATE_MIN_SIMS", 3),
        metavar="N",
        help=(
            "Minimum number of gate simulations before early stopping is allowed. "
            "Used by the Welford sequential test. "
        )
    )
    gate_args.add_argument(
        "--gate-max-sims",
        type=int,
        default=_env_int("GATE_MAX_SIMS", 60),
        metavar="N",
        help=(
            "Maximum number of simulations evaluated by the gate.\n"
            "Acts as a hard cap even if early stopping does not trigger."
        )
    )
    gate_args.add_argument(
        "--gate-confidence",
        type=float,
        default=_env_float("GATE_CONFIDENCE", 0.95),
        metavar="P",
        help=(
            "One-sided confidence level for the Welford sequential test.\n"
            "Expressed as a probability"
        )
    )
    gate_args.add_argument(
        "--gate-loss-tolerance",
        type=float,
        default=_env_float("GATE_LOSS_TOLERANCE", 0.0),
        metavar="D",
        help=(
            "Allowed loss increase versus baseline to keep a transform adaptive.\n"
            "Usually set to 0.0; retained for bias experiments."
        )
    )

    transform_group = parser.add_argument_group("transform")
    transform_group.add_argument(
        "--transform",
        choices=VALID_TRANSFORMS,
        type=str.upper,
        default=_env_str("TRANSFORM", "SG"),
        help=(
            "Transform applied to input signals.\n"
            "baseline  : identity transform (returns X unchanged)\n"
            "MA        : moving average\n"
            "EXP       : exponential smoothing\n"
            "GF        : Gaussian filtering\n"
            "SG        : Savitzky–Golay smoothing\n"
            "DFT       : Fourier approximation\n"
            "SIV_prime : sieve median approximation\n"
        ),
    )

    params_group = parser.add_argument_group("transform parameters")
    params_group.add_argument(
        "--window",
        type=int,
        default=_env_int("TRANSFORM_WINDOW", 21),
        help=(
            "Window size parameter. Must be >= 2.\n"
            "Used by: MA, EXP, GF, SG"
        ),
    )
    params_group.add_argument(
        "--polyorder",
        type=int,
        default=_env_int("POLYORDER", 5),
        help=(
            "Polynomial order.\n"
            "Used by: SG (Savitzky–Golay)"
        )
        
    )

    params_group.add_argument(
        "--radius",
        type=float,
        default=_env_float("RADIUS", 0.1),
        help=(
            "Frequency radius.\n"
            "Used by: DFT (Fourier approximation)"
        )
    )
    params_group.add_argument(
        "--numerator",
        type=int,
        default=_env_int("NUMERATOR", 5),
        metavar="N",
        help=(
            "Numerator parameter.\n"
            "Used by: SIV_prime (Sieve median approximation)"
        )
    )

    is_pytest = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules
    if is_pytest:
        args, _ = parser.parse_known_args([])
    else:
        args = parser.parse_args()
        print_effective_args(args)
    return args

class HelpFormatter(
    argparse.RawTextHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


_args = _parse_args()


_program_name = "alec"

# File locations
PROJECT_ROOT = Path('.').resolve() if is_ipython_notebook() else Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_FILE = PROJECT_ROOT / "datasets_to_run.txt"
OUTPUT_CSV = PROJECT_ROOT / "results" / f"{_program_name}_results.csv"
DETAILS_CSV = PROJECT_ROOT / "results" / f"{_program_name}_details.csv"
GATE_SIMS_CSV = PROJECT_ROOT / "results" / f"{_program_name}_loss_gate_sims.csv"

# Savitzky-Golay parameters for the adaptive filter proposal
WINDOW = _args.window
POLYORDER = _args.polyorder 

# TODO: implement as arg
KG_ONLY = _env_bool("KG_ONLY", True)

# Transform selection and parameters
TRANSFORM = _args.transform
assert TRANSFORM in VALID_TRANSFORMS

# transform params
# most transforms have a window
WINDOW = _args.window
assert WINDOW >= 2

# SavGol
_is_odd = lambda x: x % 2 != 0
if TRANSFORM == "SG":
    assert WINDOW >= 3
    assert _is_odd(WINDOW)
    assert POLYORDER < WINDOW

# DFT
RADIUS = _args.radius

# SIV_prime
NUMERATOR = _args.numerator

# Enjoy voluminous loss gate logs in the console
DEBUG_PRINT_GATE_PROGRESS = True

# Simulation configuration
GATE_MAX_CLASS_SAMPLES = _args.gate_max_class_samples # Max samples per class used in each validation simulation
GATE_MAX_SIMS = _args.gate_max_sims                   # Max number of subset splits to evaluate for gate
GATE_LOSS_TOLERANCE = _args.gate_loss_tolerance      # Allowed loss increase vs baseline to keep adaptive - we don't really need this with Welford; adjust if you want bias
GATE_CONFIDENCE = _args.gate_confidence

# Sequential test hyperparameters
conf_str = f"{GATE_CONFIDENCE:.3f}"
if conf_str not in Z_P_LOOKUP:
    print(f"Error: unsuported value for --gate-confidence. Supported values are: {", ".join(Z_P_LOOKUP.keys())}")
    sys.exit(1)
Z_CONF = Z_P_LOOKUP[conf_str]

RANDOM_STATE = _args.seed

# Main evaluation configuration
NUM_FEATURES = 50_000
NUM_SPLITS = _args.num_splits # number of stratified shuffle resplits, or "0" to use the UCR test/train split
N_JOBS = _args.jobs           # Number of processes to run in parallel, or -1 to use all CPU cores

GATE_MIN_SIMS = _args.gate_min_sims # Don't early-stop before this many sims


# Ensembling weights
WEIGHT_KG   : float|int = _get_weight("kg"   , 0.5)
WEIGHT_HYDRA: float|int = _get_weight("hydra", 0.5)
WEIGHT_MR   : float|int = _get_weight("mr"   , 0  )
WEIGHT_MRH  : float|int = _get_weight("mrh"  , 0  )

# Explicit classifier selection overrides gate settings, so do this after
# defining those
CLASSIFIER = _args.classifier
if CLASSIFIER in ["kg_only", "kg_base"]:
    KG_ONLY = True

    # Override weights
    WEIGHT_KG = 1.0
    WEIGHT_HYDRA = 0.0
    WEIGHT_MR = 0.0
    WEIGHT_MRH = 0.0

    if CLASSIFIER == "kg_base":
        # Also disable gate by setting sims to 0. This will automatically skip the filtering.
        GATE_MAX_SIMS = 0
        GATE_MIN_SIMS = 0 # not strictly necessary, since the gate is effectively disabled, but we set it to 0 for clarity
        # Rest of the GATE_* settings will be effectively ignored

else:
    # Also set KG_ONLY=True by default, because computing the MR and MRH features is costly
    KG_ONLY = _env_bool("KG_ONLY", True)

# Sanity check -- we should probably fix this more principally, but for now,
# just make sure the config is consistent and won't lead to silent errors
if KG_ONLY:
    assert WEIGHT_MR == 0 and WEIGHT_MRH == 0, f"Non-KG_ONLY weights are set, but KG_ONLY is enabled. Most certainly this is an error; aborting. {KG_ONLY=}, {WEIGHT_MR=}, {WEIGHT_MRH=}"


# Determinism for our HYDRA implementation is a bit tricky, so we provide
# separate flags to control it. KG-MTP determinism is always enabled, on the
# code level, and does not require any special flags.
#
# Determinism for Hydra is enabled by setting HYDRA_DETERMINISTIC=1 (enabled by
# default).  The strict flag is provided for completeness, in case some
# hardware/software setups may need it; in our testing, the strict mode produces
# self-consistent, but *different* results vs non-strict mode, and most often is not needed.

# Produce detterministic results across runs on the same hardware and software setup
HYDRA_DETERMINISTIC = _env_bool("HYDRA_DETERMINISTIC", True)
# Stricter mode -- enable more aggressive determinism
HYDRA_DETERMINISTIC_STRICT = _env_bool("HYDRA_DETERMINISTIC_STRICT", True)
