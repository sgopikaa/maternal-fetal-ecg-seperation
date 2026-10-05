"""
Flask API for the Maternal-Fetal ECG Separation & Monitoring project.

Endpoints:
  GET  /api/health        -> server + model training status
  POST /api/train         -> (re)train all 4 models on fresh synthetic data
  POST /api/run           -> run the full pipeline on a new synthetic case
                              body: { maternal_hr, fetal_hr, fetal_distress,
                                      snr_db, seed }
"""

import os
import threading

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from pipeline import ModelStore, run_pipeline

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

store = ModelStore(train_samples=40, duration_s=6.0, fs=250, ae_epochs=15, lstm_epochs=25)
_train_lock = threading.Lock()
_training_in_progress = {"value": False}


def _train_in_background():
    with _train_lock:
        _training_in_progress["value"] = True
        try:
            store.train_all()
        finally:
            _training_in_progress["value"] = False


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": store.ready,
        "training_in_progress": _training_in_progress["value"],
    })


@app.route("/api/train", methods=["POST"])
def train():
    if _training_in_progress["value"]:
        return jsonify({"status": "already_training"}), 202
    thread = threading.Thread(target=_train_in_background, daemon=True)
    thread.start()
    return jsonify({"status": "training_started"}), 202


@app.route("/api/run", methods=["POST"])
def run():
    if not store.ready:
        return jsonify({"error": "models_not_ready",
                         "message": "Call /api/train first and wait for training to finish."}), 409

    body = request.get_json(force=True, silent=True) or {}
    try:
        result = run_pipeline(
            store,
            maternal_hr=float(body.get("maternal_hr", 78)),
            fetal_hr=float(body.get("fetal_hr", 145)),
            fetal_distress=bool(body.get("fetal_distress", False)),
            snr_db_input=float(body.get("snr_db", 8)),
            seed=body.get("seed"),
        )
        return jsonify(result)
    except Exception as exc:  # surface a clean error to the frontend
        return jsonify({"error": "pipeline_failed", "message": str(exc)}), 500


if __name__ == "__main__":
    print("Starting server. Training initial models in the background...")
    threading.Thread(target=_train_in_background, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
