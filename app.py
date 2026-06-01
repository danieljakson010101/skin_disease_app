"""
app.py — Flask server for Skin Disease Pattern Recognition System
"""

import os, sys, json, threading
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from model.classifier import (
    run_pipeline, plot_class_distribution, train_and_save,
    load_saved_model, _find_dataset_split, CLASSES, DISPLAY_NAMES
)

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXT    = {"png","jpg","jpeg","bmp","webp"}
MAX_CONTENT_MB = 16

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Training state (per-job) ──────────────────────────────────────────────────
_train_status = {}   # job_id → {"status": "running"/"done"/"error", "log": [...]}
_train_lock   = threading.Lock()

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXT


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Return whether dataset & trained models exist."""
    dataset_ok = _find_dataset_split() is not None
    trained = {}
    for m in ["cnn","svm","rf","knn"]:
        _, meta = load_saved_model(m, "clahe", "minmax")
        trained[m] = meta["metrics"] if meta else None
    return jsonify({"dataset": dataset_ok, "trained": trained})


@app.route("/api/distribution")
def api_distribution():
    return jsonify({"chart": plot_class_distribution()})


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    POST /api/train
    Body: { model, enhancement, normalization }
    Starts background training; returns job_id.
    """
    data      = request.get_json(silent=True) or {}
    model_type = data.get("model",        "cnn")
    enh_method = data.get("enhancement",  "clahe")
    norm_method= data.get("normalization","minmax")

    job_id = f"{model_type}_{enh_method}_{norm_method}"
    with _train_lock:
        if _train_status.get(job_id, {}).get("status") == "running":
            return jsonify({"job_id": job_id, "message": "Already training"}), 200
        _train_status[job_id] = {"status":"running","log":[]}

    def _run():
        log_lines = []
        def logger(status, msg):
            log_lines.append([status, msg])
            with _train_lock:
                _train_status[job_id]["log"] = list(log_lines)

        try:
            metrics = train_and_save(model_type, enh_method, norm_method, log_fn=logger)
            if metrics:
                with _train_lock:
                    _train_status[job_id]["status"]  = "done"
                    _train_status[job_id]["metrics"] = metrics
            else:
                with _train_lock:
                    _train_status[job_id]["status"] = "error"
                    _train_status[job_id]["message"]= "No dataset found. Place images in dataset/train/"
        except Exception as e:
            with _train_lock:
                _train_status[job_id]["status"]  = "error"
                _train_status[job_id]["message"] = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/train/status/<job_id>")
def api_train_status(job_id):
    with _train_lock:
        return jsonify(_train_status.get(job_id, {"status":"unknown"}))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error":"No file in request"}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error":"Unsupported file type"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error":"Empty file"}), 400

    model_type  = request.form.get("model",         "cnn")
    enh_method  = request.form.get("enhancement",   "clahe")
    norm_method = request.form.get("normalization", "minmax")
    aug_flip    = request.form.get("aug_flip",   "0") == "1"
    aug_rotate  = request.form.get("aug_rotate", "1") == "1"
    aug_zoom    = request.form.get("aug_zoom",   "0") == "1"

    valid = {
        "model":         {"cnn","svm","rf","knn"},
        "enhancement":   {"clahe","hist","gaussian","none"},
        "normalization": {"minmax","zscore","none"},
    }
    for key, val in [("model",model_type),("enhancement",enh_method),
                     ("normalization",norm_method)]:
        if val not in valid[key]:
            return jsonify({"error":f"Invalid {key}: {val}"}), 400

    try:
        result = run_pipeline(
            image_bytes=image_bytes, model_type=model_type,
            norm_method=norm_method, enh_method=enh_method,
            aug_flip=aug_flip, aug_rotate=aug_rotate, aug_zoom=aug_zoom,
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Pipeline error")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    print("="*60)
    print("  Skin Disease Pattern Recognition System")
    print("  http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, host="0.0.0.0", port=5000)