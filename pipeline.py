"""Orchestrates the full 4-model pipeline for one abdominal ECG recording."""

import numpy as np

import signal_generator
import preprocessing
import qrs_detection
from models import ica_model, cnn_autoencoder, rf_classifier, cnn_lstm


def snr_db(reference, estimate):
    reference = reference / (np.std(reference) + 1e-8)
    estimate = estimate / (np.std(estimate) + 1e-8)
    # align sign (BSS outputs have arbitrary sign/scale)
    if np.corrcoef(reference, estimate)[0, 1] < 0:
        estimate = -estimate
    noise = reference - estimate
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(signal_power / noise_power))


class ModelStore:
    """Trains and caches all 4 models once; reused across API requests."""

    def __init__(self, n_channels=4, train_samples=40, duration_s=6.0, fs=250,
                 ae_epochs=15, lstm_epochs=25, verbose=0):
        self.n_channels = n_channels
        self.fs = fs
        self.ready = False
        self.train_samples = train_samples
        self.duration_s = duration_s
        self.ae_epochs = ae_epochs
        self.lstm_epochs = lstm_epochs
        self.verbose = verbose

    def train_all(self):
        dataset = signal_generator.generate_dataset(
            n_samples=self.train_samples, duration_s=self.duration_s, fs=self.fs
        )
        # DL1: CNN Autoencoder for separation
        self.ae_model, self.ae_history = cnn_autoencoder.train_autoencoder(
            dataset, epochs=self.ae_epochs, verbose=self.verbose
        )
        # ML2: Random Forest classifier
        self.rf_model = rf_classifier.train_rf(dataset)
        # DL2: CNN-LSTM temporal distress model
        self.lstm_model, self.lstm_history = cnn_lstm.train_cnn_lstm(
            dataset, epochs=self.lstm_epochs, verbose=self.verbose
        )
        self.ready = True
        return {
            "ae_final_loss": float(self.ae_history.history["loss"][-1]),
            "ae_final_val_loss": float(self.ae_history.history["val_loss"][-1]),
            "lstm_final_acc": float(self.lstm_history.history["accuracy"][-1]),
            "lstm_final_val_acc": float(self.lstm_history.history["val_accuracy"][-1]),
            "train_samples": self.train_samples,
        }


def run_pipeline(store: ModelStore, maternal_hr=78, fetal_hr=145,
                  fetal_distress=False, snr_db_input=8, seed=None):
    if not store.ready:
        raise RuntimeError("Models not trained yet")

    sample = signal_generator.generate_mixture(
        duration_s=store.duration_s,
        fs=store.fs,
        maternal_hr=maternal_hr,
        fetal_hr=fetal_hr,
        n_channels=store.n_channels,
        fetal_distress=fetal_distress,
        snr_db=snr_db_input,
        seed=seed,
    )

    clean_channels = preprocessing.preprocess_channels(sample["channels"], sample["fs"])

    # ---- Stage 1: Separation ----
    ica_result = ica_model.separate_ica(clean_channels)
    fetal_ica = ica_result["fetal_component"]

    fetal_cnn = cnn_autoencoder.extract_fetal_ecg(store.ae_model, clean_channels)

    ica_snr = snr_db(sample["fetal_ref"], fetal_ica)
    cnn_snr = snr_db(sample["fetal_ref"], fetal_cnn)

    # ---- Fetal QRS detection + HR trend (use the better-SNR estimate) ----
    best_estimate = fetal_cnn if cnn_snr >= ica_snr else fetal_ica
    best_estimate_name = "cnn_autoencoder" if cnn_snr >= ica_snr else "ica"

    peaks = qrs_detection.detect_fetal_peaks(best_estimate, sample["fs"])
    times, fhr_series = qrs_detection.compute_fhr_series(peaks, sample["fs"])
    feats = qrs_detection.extract_features(fhr_series)

    # ---- Stage 2: Monitoring ----
    rf_label, rf_proba = "undetermined", {}
    if feats["mean_fhr"] > 0:
        rf_label, rf_proba = rf_classifier.predict_condition(store.rf_model, feats)

    trend_for_lstm = fhr_series if len(fhr_series) > 0 else sample["fetal_hr_trend"]
    lstm_label, lstm_prob = cnn_lstm.predict_distress(store.lstm_model, trend_for_lstm)

    def downsample(arr, max_points=1500):
        if len(arr) <= max_points:
            return arr.tolist()
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        return arr[idx].tolist()

    return {
        "fs": sample["fs"],
        "duration_s": store.duration_s,
        "time": downsample(sample["t"]),
        "raw_channel": downsample(sample["channels"][0]),
        "ground_truth": {
            "maternal_ecg": downsample(sample["maternal_ref"]),
            "fetal_ecg": downsample(sample["fetal_ref"]),
            "fetal_distress_simulated": bool(sample["fetal_distress"]),
        },
        "separation": {
            "ica": {
                "fetal_component": downsample(fetal_ica),
                "snr_db": round(ica_snr, 2),
            },
            "cnn_autoencoder": {
                "fetal_component": downsample(fetal_cnn),
                "snr_db": round(cnn_snr, 2),
            },
            "best_model": best_estimate_name,
        },
        "monitoring": {
            "fetal_peaks_detected": len(peaks),
            "fhr_series_time": times.tolist() if len(times) else [],
            "fhr_series_bpm": fhr_series.tolist() if len(fhr_series) else [],
            "features": feats,
            "random_forest": {
                "label": rf_label,
                "probabilities": rf_proba,
            },
            "cnn_lstm": {
                "label": lstm_label,
                "probability": round(lstm_prob, 3),
            },
        },
    }
