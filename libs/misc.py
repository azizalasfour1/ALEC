import numpy as np


def is_ipython_notebook():
    try:
        from IPython import get_ipython
        # Check if get_ipython() returns a non-None value (i.e., an active IPython shell)
        return get_ipython() is not None
    except NameError:
        return False  # IPython is not defined
    except ImportError:
        return False  # IPython module not found

if is_ipython_notebook():
    print("Running in an IPython notebook environment.")
else:
    # print("Not running in an IPython notebook environment.")
    pass


def pseudo_nll_from_dec(y_true, decisions, classes, eps: float = 1e-15):
    """
    Compute a pseudo negative log-likelihood (NLL) from unnormalized
    decision_function outputs.

    Motivation
    ----------
    Some classifiers (e.g. RidgeClassifier / RidgeClassifierCV) expose
    `decision_function` scores but not calibrated class probabilities.
    This helper converts those scores into *pseudo-probabilities* using:

      - a sigmoid transform for binary classification
      - a softmax transform for multiclass classification

    The resulting values are not guaranteed to be well-calibrated
    probabilities, but they allow:
      - a smooth, proper loss for model comparison
      - approximate log-loss / NLL reporting
      - consistency across binary and multiclass settings

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.

    decisions : array-like
        Output of `decision_function`:
          - shape (n_samples,) for binary classification
          - shape (n_samples, n_classes) for multiclass classification

    classes : array-like of shape (n_classes,)
        Class labels in the order corresponding to the decision function.
        For binary classification, classes[1] is treated as the positive class.

    eps : float, default=1e-15
        Numerical stability constant used for clipping probabilities.

    Returns
    -------
    float
        Mean pseudo negative log-likelihood.

    """
    y = np.asarray(y_true)
    d = np.asarray(decisions)
    classes = np.asarray(classes)

    if classes.size < 2:
        raise ValueError(f"need at least 2 classes; got classes={classes}")

    # Binary case
    if d.ndim == 1:
        if classes.size != 2:
            raise ValueError(
                f"binary decisions but classes has size {classes.size}"
            )

        y01 = (y == classes[1]).astype(np.int64)
        p1 = 1.0 / (1.0 + np.exp(-d))          # sigmoid
        p1 = np.clip(p1, eps, 1.0 - eps)

        return float(
            -np.mean(y01 * np.log(p1) + (1 - y01) * np.log(1.0 - p1))
        )

    # Multiclass case
    if d.ndim != 2:
        raise ValueError(f"multiclass decisions must be 2D, got shape {d.shape}")

    n, k = d.shape
    if k != classes.size:
        raise ValueError(
            f"decisions has K={k} columns but classes has {classes.size}"
        )

    # stable softmax
    d = d - np.max(d, axis=1, keepdims=True)
    p = np.exp(d)
    p /= np.sum(p, axis=1, keepdims=True)
    p = np.clip(p, eps, 1.0 - eps)

    idx = np.searchsorted(classes, y)
    return float(-np.mean(np.log(p[np.arange(n), idx])))
