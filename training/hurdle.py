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


def fit_activity_threshold(activity_probability, activity_targets,
                           objective='f1', minimum=0.05, maximum=0.95):
    """Select a deterministic activity threshold on validation predictions.

    The threshold is evaluated only at distinct predicted probabilities.  The
    highest threshold wins ties, which avoids silently increasing the number
    of positive forecasts when two operating points have the same score.

    Returns:
        ``(threshold, details)`` where ``details`` is JSON serializable.
    """
    probability = np.asarray(activity_probability, dtype=np.float64).reshape(-1)
    targets = (np.asarray(activity_targets).reshape(-1) >= 0.5).astype(np.int64)
    if probability.size != targets.size or probability.size == 0:
        raise ValueError(
            "Activity probabilities and targets must be non-empty and aligned."
        )
    if not np.all(np.isfinite(probability)):
        raise ValueError("Activity probabilities must be finite.")
    if objective not in {'f1', 'balanced_accuracy'}:
        raise ValueError("objective must be 'f1' or 'balanced_accuracy'.")
    if not 0.0 <= float(minimum) < float(maximum) <= 1.0:
        raise ValueError("Threshold bounds must satisfy 0 <= minimum < maximum <= 1.")

    probability = np.clip(probability, 0.0, 1.0)
    order = np.argsort(-probability, kind='mergesort')
    sorted_probability = probability[order]
    sorted_targets = targets[order]
    cumulative_tp = np.cumsum(sorted_targets)
    cumulative_fp = np.cumsum(1 - sorted_targets)

    # Predictions change only after the final item for a distinct score.
    distinct_end = np.r_[
        sorted_probability[:-1] != sorted_probability[1:], True
    ]
    candidate_indices = np.flatnonzero(distinct_end)
    candidate_thresholds = sorted_probability[candidate_indices]
    bounded = (
        (candidate_thresholds >= float(minimum))
        & (candidate_thresholds <= float(maximum))
    )
    candidate_indices = candidate_indices[bounded]
    candidate_thresholds = candidate_thresholds[bounded]

    if candidate_thresholds.size == 0:
        fallback = float(np.clip(0.5, minimum, maximum))
        prediction = probability >= fallback
        tp_value = float(np.sum(prediction & (targets == 1)))
        fp_value = float(np.sum(prediction & (targets == 0)))
        fn_value = float(np.sum((~prediction) & (targets == 1)))
        tn_value = float(np.sum((~prediction) & (targets == 0)))
        precision_value = (
            tp_value / (tp_value + fp_value) if tp_value + fp_value else 0.0
        )
        recall_value = (
            tp_value / (tp_value + fn_value) if tp_value + fn_value else 0.0
        )
        f1_value = (
            2.0 * precision_value * recall_value
            / (precision_value + recall_value)
            if precision_value + recall_value else 0.0
        )
        specificity_value = (
            tn_value / (tn_value + fp_value) if tn_value + fp_value else 0.0
        )
        score_value = (
            f1_value if objective == 'f1'
            else 0.5 * (recall_value + specificity_value)
        )
        return fallback, {
            'source': 'validation_only_fallback',
            'objective': objective,
            'objective_score': float(score_value),
            'threshold': fallback,
            'precision': float(precision_value),
            'recall': float(recall_value),
            'f1_score': float(f1_value),
            'specificity': float(specificity_value),
            'predicted_positive_rate': float(prediction.mean()),
            'prevalence': float(targets.mean()),
            'n_validation_entries': int(targets.size),
            'minimum': float(minimum),
            'maximum': float(maximum),
        }

    tp = cumulative_tp[candidate_indices].astype(np.float64)
    fp = cumulative_fp[candidate_indices].astype(np.float64)
    total_positive = float(targets.sum())
    total_negative = float(targets.size - targets.sum())
    fn = total_positive - tp
    tn = total_negative - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    specificity = np.divide(
        tn, tn + fp, out=np.zeros_like(tn), where=(tn + fp) > 0
    )
    scores = f1 if objective == 'f1' else 0.5 * (recall + specificity)
    best = int(np.argmax(scores))
    threshold = float(candidate_thresholds[best])

    details = {
        'source': 'validation_only',
        'objective': objective,
        'objective_score': float(scores[best]),
        'threshold': threshold,
        'precision': float(precision[best]),
        'recall': float(recall[best]),
        'f1_score': float(f1[best]),
        'specificity': float(specificity[best]),
        'predicted_positive_rate': float((tp[best] + fp[best]) / targets.size),
        'prevalence': float(targets.mean()),
        'n_validation_entries': int(targets.size),
        'minimum': float(minimum),
        'maximum': float(maximum),
    }
    return threshold, details
