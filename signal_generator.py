"""
Synthetic Maternal-Fetal Abdominal ECG Generator
--------------------------------------------------
Real datasets (e.g. PhysioNet ADFECGDB, NInFEA) require manual download and
are not bundled here. This generator produces physiologically plausible
multi-channel abdominal ECG mixtures (maternal ECG + fetal ECG + noise) so
the full pipeline can be demonstrated and trained end-to-end offline.

Swap `generate_dataset()` for a real-data loader when PhysioNet files are
available locally (see README for the drop-in path).
"""

import numpy as np


def _gaussian_pulse(t, center, width, amplitude):
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def _synth_ecg_beat(t, beat_center, hr_bpm, amp_scale=1.0):
    """Builds one PQRST complex as a sum of Gaussians, scaled to heart rate."""
    rr = 60.0 / hr_bpm
    # (offset from R peak as fraction of RR, width in seconds, amplitude)
    components = [
        (-0.20 * rr, 0.025 * rr, 0.10 * amp_scale),   # P wave
        (-0.05 * rr, 0.008 * rr, -0.12 * amp_scale),  # Q dip
        (0.0,        0.006 * rr, 1.00 * amp_scale),   # R peak
        (0.04 * rr,  0.010 * rr, -0.20 * amp_scale),  # S dip
        (0.20 * rr,  0.040 * rr, 0.25 * amp_scale),   # T wave
    ]
    sig = np.zeros_like(t)
    for offset, width, amp in components:
        sig += _gaussian_pulse(t, beat_center + offset, width, amp)
    return sig


def _generate_ecg_track(duration_s, fs, hr_bpm, amp_scale, hr_variability=0.03, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)
    sig = np.zeros_like(t)

    beat_time = 0.0
    while beat_time < duration_s:
        instantaneous_hr = hr_bpm * (1 + rng.normal(0, hr_variability))
        instantaneous_hr = np.clip(instantaneous_hr, hr_bpm * 0.7, hr_bpm * 1.3)
        sig += _synth_ecg_beat(t, beat_time, instantaneous_hr, amp_scale)
        beat_time += 60.0 / instantaneous_hr

    return t, sig


def generate_mixture(
    duration_s=10.0,
    fs=250,
    maternal_hr=78,
    fetal_hr=145,
    n_channels=4,
    fetal_distress=False,
    snr_db=8,
    seed=None,
):
    """
    Returns a dict with:
      t                : time vector
      channels         : (n_channels, N) mixed abdominal ECG observations
      maternal_ref      : ground-truth maternal ECG (channel 0 scale)
      fetal_ref         : ground-truth fetal ECG (channel 0 scale)
      fetal_hr_trend    : per-second instantaneous fetal HR (ground truth)
    """
    rng = np.random.default_rng(seed)

    fetal_hr_used = fetal_hr
    hr_var = 0.03
    if fetal_distress:
        # simulate late deceleration / reduced variability pattern
        fetal_hr_used = fetal_hr * 0.72
        hr_var = 0.015

    t, m_sig = _generate_ecg_track(duration_s, fs, maternal_hr, amp_scale=1.0,
                                    hr_variability=0.02, seed=(seed or 0) + 1)
    _, f_sig = _generate_ecg_track(duration_s, fs, fetal_hr_used, amp_scale=0.22,
                                    hr_variability=hr_var, seed=(seed or 0) + 2)

    # respiration-driven baseline wander (maternal breathing ~0.25 Hz)
    baseline = 0.15 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))

    channels = np.zeros((n_channels, len(t)))
    for ch in range(n_channels):
        # different electrode placements pick up maternal/fetal signal with
        # different (mixing) gains -> what makes BSS / ICA meaningful
        m_gain = 0.7 + 0.3 * rng.random()
        f_gain = 0.5 + 0.7 * rng.random()
        noise_power = 10 ** (-snr_db / 20)
        noise = rng.normal(0, noise_power, size=len(t))
        channels[ch] = m_gain * m_sig + f_gain * f_sig + baseline + noise

    # ground-truth instantaneous fetal HR trend, 1 Hz resolution
    n_points = int(duration_s)
    fetal_hr_trend = fetal_hr_used * (1 + rng.normal(0, hr_var, size=n_points))
    fetal_hr_trend = np.clip(fetal_hr_trend, 60, 220)

    return {
        "t": t,
        "channels": channels,
        "maternal_ref": m_sig,
        "fetal_ref": f_sig,
        "fetal_hr_trend": fetal_hr_trend,
        "fs": fs,
        "fetal_distress": fetal_distress,
    }


def generate_dataset(n_samples=40, duration_s=6.0, fs=250, seed=42):
    """Batch of mixtures for training the deep-learning models, with a mix
    of normal and distress-pattern fetal traces for the classifiers."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        distress = rng.random() < 0.35
        maternal_hr = rng.uniform(65, 95)
        fetal_hr = rng.uniform(120, 160)
        sample = generate_mixture(
            duration_s=duration_s,
            fs=fs,
            maternal_hr=maternal_hr,
            fetal_hr=fetal_hr,
            fetal_distress=distress,
            snr_db=rng.uniform(4, 12),
            seed=seed + i,
        )
        samples.append(sample)
    return samples
