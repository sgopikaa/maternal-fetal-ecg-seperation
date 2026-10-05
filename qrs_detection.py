"""Fetal QRS peak detection (Pan-Tompkins style) and HR feature extraction."""

import numpy as np
from scipy.signal import find_peaks


def detect_fetal_peaks(fetal_signal, fs, max_fhr=220, min_fhr=60):
    """Simple Pan-Tompkins-style pipeline: derivative -> square -> peak pick,
    with a refractory period matched to plausible fetal heart rates."""
    sig = fetal_signal - np.mean(fetal_signal)
    sig = sig / (np.std(sig) + 1e-8)

    derivative = np.diff(sig, prepend=sig[0])
    squared = derivative ** 2

    window = max(1, int(0.05 * fs))
    kernel = np.ones(window) / window
    envelope = np.convolve(squared, kernel, mode="same")

    min_distance = int(fs * 60.0 / max_fhr)
    height = np.mean(envelope) + 0.6 * np.std(envelope)

    peaks, _ = find_peaks(envelope, distance=min_distance, height=height)
    return peaks


def compute_fhr_series(peaks, fs, min_fhr=60, max_fhr=220):
    """Instantaneous fetal heart rate (bpm) from consecutive peak intervals."""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    rr_intervals = np.diff(peaks) / fs  # seconds
    fhr = 60.0 / rr_intervals
    valid = (fhr >= min_fhr) & (fhr <= max_fhr)
    times = peaks[1:][valid] / fs
    return times, fhr[valid]


def extract_features(fhr_values, rr_intervals=None):
    """Handcrafted features for the ML classifier."""
    if len(fhr_values) == 0:
        return {
            "mean_fhr": 0, "std_fhr": 0, "min_fhr": 0, "max_fhr": 0,
            "range_fhr": 0, "rmssd": 0,
        }
    mean_fhr = float(np.mean(fhr_values))
    std_fhr = float(np.std(fhr_values))
    min_fhr = float(np.min(fhr_values))
    max_fhr = float(np.max(fhr_values))
    diffs = np.diff(fhr_values)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
    return {
        "mean_fhr": mean_fhr,
        "std_fhr": std_fhr,
        "min_fhr": min_fhr,
        "max_fhr": max_fhr,
        "range_fhr": max_fhr - min_fhr,
        "rmssd": rmssd,
    }


def label_from_features(feats):
    """Rule for generating training labels on synthetic data (ground truth
    substitute): mirrors standard cardiotocography (CTG) categories."""
    mean_fhr = feats["mean_fhr"]
    std_fhr = feats["std_fhr"]
    if mean_fhr == 0:
        return "undetermined"
    if mean_fhr < 110:
        return "bradycardia"
    if mean_fhr > 160:
        return "tachycardia"
    if std_fhr < 2.0:
        return "low_variability_distress"
    return "normal"
