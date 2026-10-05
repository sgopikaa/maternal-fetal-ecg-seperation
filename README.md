# Fetal Cardiac Monitor — Maternal–Fetal ECG Separation

An end-to-end demo that separates fetal ECG from a mixed maternal abdominal
recording and monitors fetal cardiac status, using **2 classical ML models**
and **2 deep learning models**:

| Stage | Model | Type | File |
|---|---|---|---|
| Separation | FastICA (blind source separation) | ML | `backend/models/ica_model.py` |
| Separation | 1D CNN Autoencoder | DL | `backend/models/cnn_autoencoder.py` |
| Monitoring | Random Forest (CTG-style classifier) | ML | `backend/models/rf_classifier.py` |
| Monitoring | CNN–LSTM (temporal distress risk) | DL | `backend/models/cnn_lstm.py` |

The frontend is a single-page "instrument console" styled like real ECG
chart paper, served by the same Flask app.

## ⚠️ About the data

No real clinical dataset is bundled (PhysioNet requires manual download and
this environment couldn't fetch it for you). Instead, `signal_generator.py`
synthesizes physiologically plausible multi-channel maternal+fetal ECG
mixtures on the fly, which is what all four models are trained/demoed on.
**Do not use this as-is for any real clinical decision.**

To use real data instead:
1. Download **ADFECGDB** or **NInFEA** from PhysioNet.
2. Replace the body of `signal_generator.generate_dataset()` /
   `generate_mixture()` with a loader that reads your `.dat`/`.hea` (WFDB)
   files into the same `{t, channels, maternal_ref, fetal_ref, fetal_hr_trend}`
   dict shape used everywhere downstream — nothing else needs to change.

## Project structure

```
maternal_fetal_ecg_project/
├── backend/
│   ├── app.py                 # Flask API + serves the frontend
│   ├── pipeline.py            # orchestrates all 4 models end-to-end
│   ├── signal_generator.py    # synthetic maternal+fetal ECG mixtures
│   ├── preprocessing.py       # bandpass filter, baseline removal, normalize
│   ├── qrs_detection.py       # fetal peak detection + HR feature extraction
│   ├── models/
│   │   ├── ica_model.py         # ML  — FastICA separation
│   │   ├── cnn_autoencoder.py   # DL  — CNN autoencoder separation
│   │   ├── rf_classifier.py     # ML  — Random Forest CTG classifier
│   │   └── cnn_lstm.py          # DL  — CNN-LSTM distress-risk model
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## Setup (VS Code / local machine)

1. **Extract the zip** and open the folder in VS Code.

2. **Create a virtual environment** (recommended) and install dependencies:
   ```bash
   cd maternal_fetal_ecg_project/backend
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   python app.py
   ```
   You should see `Starting server. Training initial models in the background...`

4. **Open the app**: go to `http://localhost:5000` in your browser.
   - The status pill in the top-right will show "Training models…" for
     roughly 15–30 seconds on first load (training happens once, in a
     background thread, on freshly generated synthetic recordings).
   - Once it turns green ("Models ready"), set your parameters on the left
     and click **Run analysis**.

## How the pipeline works

1. `signal_generator.py` synthesizes a multi-channel abdominal ECG: a
   maternal ECG waveform + a much smaller fetal ECG waveform + respiration
   baseline wander + noise, mixed with random per-channel gains (this
   per-channel mixing is exactly what makes source separation meaningful).
2. `preprocessing.py` bandpass-filters (0.5–45 Hz) and removes baseline
   wander from every channel.
3. **Separation stage** — both models attempt to recover the fetal ECG from
   the mixed channels:
   - **FastICA** treats the channels as a linear mixture of independent
     sources and un-mixes them; the fetal component is picked out via a
     kurtosis heuristic (QRS-like peaky sources score higher).
   - **CNN Autoencoder** learns a direct nonlinear multi-channel → fetal-ECG
     mapping from training examples, which can help with overlapping
     maternal/fetal QRS complexes that trip up linear ICA.
   - Both are scored by SNR against the known synthetic fetal reference;
     the higher-SNR estimate feeds the monitoring stage.
4. `qrs_detection.py` finds fetal R-peaks (derivative → square → threshold,
   Pan-Tompkins style) on the winning estimate and derives an instantaneous
   fetal heart-rate (FHR) trend plus summary features (mean, std, RMSSD,
   range).
5. **Monitoring stage**:
   - **Random Forest** classifies the FHR window into a cardiotocography
     (CTG) style category: normal / bradycardia / tachycardia / low
     variability (possible distress).
   - **CNN–LSTM** looks at the *shape* of the FHR trend over time (not just
     summary stats) to flag temporal distress patterns such as sustained
     decelerations.
6. The frontend renders the raw mixture, ground truth, both separation
   estimates (with SNR), the FHR trend chart, and both models' verdicts.

## Retraining

Models train once when the server starts. Click **Retrain models** in the
UI (or `POST /api/train`) to regenerate a fresh synthetic training set and
retrain all 4 models from scratch — useful after you change
`signal_generator.py` or model hyperparameters.

## Extending this project

- Swap in real PhysioNet recordings (see "About the data" above).
- Add an adaptive filter (RLS/LMS) as an alternative ML separation baseline.
- Replace the kurtosis-based ICA component picker with a supervised
  classifier once you have labeled sources.
- Add persistence (save/load trained models with `model.save()` /
  `joblib.dump()`) so retraining isn't required on every server restart.
- Tighten fetal QRS detection with a dedicated wavelet-based detector for
  noisier recordings.
