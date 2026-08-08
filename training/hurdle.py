"""Utilities for decoding the auxiliary activity/magnitude hurdle output."""

import numpy as np
import torch


def decode_hurdle_tensor(raw_output, activity_threshold=0.5,
                         activity_logit_bias=0.0):
    """Decode ``[..., conditional_magnitude, activity_logit]`` tensors."""
    if raw_output.shape[-1] != 2:
        raise ValueError("Hurdle decoding requires exactly two model outputs.")
    conditional = raw_output[..., 0:1]
    activity_logit = raw_output[..., 1:2] + float(activity_logit_bias)
    activity_probability = torch.sigmoid(activity_logit)
    expected = activity_probability * conditional
    thresholded = torch.where(
        activity_probability >= float(activity_threshold),
        conditional,
        torch.zeros_like(conditional),
    )
    return {
        'conditional': conditional,
        'activity_logit': activity_logit,
        'activity_probability': activity_probability,
        'expected': expected,
        'thresholded': thresholded,
    }


def decode_hurdle_numpy(raw_output, activity_threshold=0.5,
                        activity_logit_bias=0.0):
    """NumPy equivalent used by evaluation and artifact generation."""
    raw_output = np.asarray(raw_output)
    if raw_output.shape[-1] != 2:
        raise ValueError("Hurdle decoding requires exactly two model outputs.")
    conditional = raw_output[..., 0:1]
    activity_logit = raw_output[..., 1:2] + float(activity_logit_bias)
    clipped_logit = np.clip(activity_logit, -60.0, 60.0)
    activity_probability = 1.0 / (1.0 + np.exp(-clipped_logit))
    expected = activity_probability * conditional
    thresholded = np.where(
        activity_probability >= float(activity_threshold),
        conditional,
        0.0,
    )
    return {
        'conditional': conditional.astype(np.float32, copy=False),
        'activity_logit': activity_logit.astype(np.float32, copy=False),
        'activity_probability': activity_probability.astype(np.float32, copy=False),
        'expected': expected.astype(np.float32, copy=False),
        'thresholded': thresholded.astype(np.float32, copy=False),
    }


def split_hurdle_targets(targets):
    """Return normalized magnitude and explicit activity target arrays."""
    targets = np.asarray(targets)
    if targets.shape[-1] < 2:
        raise ValueError(
            "Hurdle targets require magnitude and activity channels."
        )
    return targets[..., 0:1], targets[..., 1:2]


def fit_activity_logit_bias(activity_logits, activity_targets,
                            lower=-20.0, upper=20.0, iterations=80):
    """Fit an intercept-only probability calibration on validation data.

    For fixed logits, the Bernoulli maximum-likelihood intercept is the value
    whose mean predicted probability equals validation prevalence. Bisection
    is deterministic and does not require SciPy.
    """
    logits = np.asarray(activity_logits, dtype=np.float64).reshape(-1)
    targets = (np.asarray(activity_targets).reshape(-1) >= 0.5).astype(np.float64)
    if logits.size != targets.size or logits.size == 0:
        raise ValueError("Activity logits and targets must be non-empty and aligned.")
    prevalence = float(targets.mean())
    if prevalence <= 0.0:
        return float(lower)
    if prevalence >= 1.0:
        return float(upper)

    for _ in range(int(iterations)):
        midpoint = (lower + upper) / 2.0
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(logits + midpoint, -60.0, 60.0))
        )
        if float(probability.mean()) < prevalence:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0)
