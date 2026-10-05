"""Preprocessing: bandpass filtering, baseline-wander removal, normalization."""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, low=0.5, high=45.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def remove_baseline_wander(signal, fs, cutoff=0.6, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)


def normalize(signal):
    signal = signal - np.mean(signal)
    std = np.std(signal)
    return signal / std if std > 1e-8 else signal


def preprocess_channels(channels, fs):
    """Apply bandpass + baseline removal + normalization to every channel."""
    out = np.zeros_like(channels)
    for i in range(channels.shape[0]):
        sig = bandpass_filter(channels[i], fs)
        sig = remove_baseline_wander(sig, fs)
        out[i] = normalize(sig)
    return out
