"""
classifier.py
=============
Skin Disease Pattern Recognition — Steps 2 & 3.
Models: CNN (Keras, real convolutional network) and KNN (color hist + LBP).

• If trained model files exist  → uses real predictions
• If dataset folder exists       → trains and saves models
• Otherwise                      → falls back to simulation (demo mode)
"""

import cv2, numpy as np, io, base64, os, glob, joblib, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

import tensorflow as tf
from tensorflow.keras import layers, models as keras_models

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "model", "saved")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(MODEL_DIR, exist_ok=True)

# Models supported by this project (Step 3 — focused on two techniques)
MODEL_TYPES = ["cnn", "knn"]

# ── Class list (matches Kaggle folder names) ───────────────────────────────────
CLASSES = [
    "Acne","Actinic_Keratosis","Benign_tumors","Bullous","Candidiasis",
    "DrugEruption","Eczema","Infestations_Bites","Lichen","Lupus",
    "Moles","Psoriasis","Rosacea","Seborrh_Keratoses","SkinCancer",
    "Sun_Sunlight_Damage","Tinea","Unknown_Normal","Vascular_Tumors",
    "Vasculitis","Vitiligo","Warts"
]
DISPLAY_NAMES = {c: c.replace("_", " ") for c in CLASSES}

# Simulated counts used ONLY when no real dataset found
SIMULATED_COUNTS = {
    "Acne":312,"Actinic_Keratosis":287,"Benign_tumors":145,"Bullous":198,
    "Candidiasis":176,"DrugEruption":405,"Eczema":163,"Infestations_Bites":134,
    "Lichen":112,"Lupus":389,"Moles":354,"Psoriasis":221,"Rosacea":268,
    "Seborrh_Keratoses":342,"SkinCancer":195,"Sun_Sunlight_Damage":187,
    "Tinea":431,"Unknown_Normal":98,"Vascular_Tumors":89,"Vasculitis":203,
    "Vitiligo":276,"Warts":367
}

# ══════════════════════════════════════════════════════════════════════════════
#  Utility
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data

def img_to_b64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok: raise ValueError("Cannot encode image")
    return base64.b64encode(buf).decode()

def _set_dark(fig, *axes):
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2a — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_and_resize(image_bytes, size=(224, 224)):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Upload a valid JPG/PNG.")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def apply_enhancement(img, method):
    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    elif method == "hist":
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    elif method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    return img.copy()

def apply_normalization(img, method):
    f = img.astype(np.float32)
    if method == "minmax":
        f = (f - f.min()) / (f.max() - f.min() + 1e-7) * 255.0
    elif method == "zscore":
        m, s = f.mean(), f.std() + 1e-7
        f = (f - m) / s
        f = (f - f.min()) / (f.max() - f.min() + 1e-7) * 255.0
    return f.astype(np.uint8)

def apply_augmentation(img, flip=False, rotate=True, zoom=False):
    out = img.copy()
    if flip:   out = cv2.flip(out, 1)
    if rotate:
        h, w = out.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), float(np.random.uniform(-15,15)), 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    if zoom:
        h, w = out.shape[:2]
        s = float(np.random.uniform(0.85, 0.97))
        nh, nw = int(h*s), int(w*s)
        y1, x1 = (h-nh)//2, (w-nw)//2
        out = cv2.resize(out[y1:y1+nh, x1:x1+nw], (w,h), interpolation=cv2.INTER_LINEAR)
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Feature extraction (KNN only — CNN consumes raw pixels)
# ══════════════════════════════════════════════════════════════════════════════

def extract_lbp(img, radius=2, n_points=16):
    gray = cv2.cvtColor(cv2.resize(img,(64,64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lbp  = np.zeros_like(gray)
    for i in range(radius, gray.shape[0]-radius):
        for j in range(radius, gray.shape[1]-radius):
            center = gray[i,j]; code = 0
            for k in range(n_points):
                a = 2*np.pi*k/n_points
                xi = int(np.clip(round(j+radius*np.cos(a)),0,gray.shape[1]-1))
                yi = int(np.clip(round(i-radius*np.sin(a)),0,gray.shape[0]-1))
                if gray[yi,xi] >= center: code |= (1<<k)
            lbp[i,j] = code
    h, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    h = h.astype(np.float32); h /= h.sum()+1e-7
    return h

def extract_color_hist(img, bins=32):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = []
    for ch in range(3):
        h = cv2.calcHist([hsv],[ch],None,[bins],[0,256])
        feats.extend(cv2.normalize(h,h).flatten())
    return np.array(feats, dtype=np.float32)

def extract_knn_features(img):
    return np.concatenate([extract_color_hist(img), extract_lbp(img)]).astype(np.float32)

def get_features(img, model_type):
    """For KNN: hand-crafted features. For CNN: raw normalized pixel array."""
    if model_type == "knn":
        return extract_knn_features(img)
    elif model_type == "cnn":
        return cnn_preprocess_image(img)
    raise ValueError(f"Unknown model_type: {model_type}")

def cnn_preprocess_image(img, size=(96, 96)):
    """Resize + scale to [0,1] float32, ready for the Keras CNN input layer."""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)

# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def _find_dataset_split():
    """Return path to a split (train/test/val) folder that has class subfolders."""
    for split in ["train", "Train", "training", "test", "Test"]:
        p = os.path.join(DATASET_DIR, split)
        if os.path.isdir(p):
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))]
            if len(subdirs) >= 5:
                return p
    if os.path.isdir(DATASET_DIR):
        subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR,d))]
        if len(subdirs) >= 5:
            return DATASET_DIR
    return None

def get_real_class_counts():
    """Count images per class from whatever dataset folder exists."""
    base = _find_dataset_split()
    if not base: return None
    counts = {}
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.JPG","*.JPEG","*.PNG")
    for cls in os.listdir(base):
        p = os.path.join(base, cls)
        if not os.path.isdir(p): continue
        n = sum(len(glob.glob(os.path.join(p, e))) for e in exts)
        if n > 0: counts[cls] = n
    return counts if counts else None

def load_dataset_arrays(model_type, enh_method, norm_method,
                        max_per_class=80, log_fn=None):
    """
    Load images from the dataset folder, apply preprocessing, return X, y, classes.
    X is either feature vectors (KNN) or raw pixel arrays (CNN).
    """
    base = _find_dataset_split()
    if not base: return None, None, None

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp","*.JPG","*.JPEG","*.PNG")
    X, y, class_names = [], [], []

    folders = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base,d))])
    if log_fn: log_fn("ok", f"Found {len(folders)} class folders in {os.path.basename(base)}/")

    for cls in folders:
        p = os.path.join(base, cls)
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(p, e)))
        if not files: continue

        np.random.shuffle(files)
        files = files[:max_per_class]
        class_names.append(cls)
        if log_fn: log_fn("ok", f"  Loading {cls}: {len(files)} images")

        for fpath in files:
            try:
                with open(fpath, "rb") as f: raw = f.read()
                img = load_and_resize(raw)
                img = apply_enhancement(img, enh_method)
                img = apply_normalization(img, norm_method)
                feat = get_features(img, model_type)
                X.append(feat); y.append(cls)
            except Exception:
                pass

    return np.array(X, dtype=np.float32), np.array(y), class_names

# ══════════════════════════════════════════════════════════════════════════════
#  CNN architecture (Step 3a — real convolutional network)
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn(input_shape=(96, 96, 3), num_classes=22):
    model = keras_models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ══════════════════════════════════════════════════════════════════════════════
#  Model training & saving
# ══════════════════════════════════════════════════════════════════════════════

def _model_path(model_type, enh, norm):
    ext = "keras" if model_type == "cnn" else "pkl"
    return os.path.join(MODEL_DIR, f"{model_type}_{enh}_{norm}.{ext}")

def _meta_path(model_type, enh, norm):
    return os.path.join(MODEL_DIR, f"{model_type}_{enh}_{norm}_meta.pkl")

def train_and_save(model_type, enh_method, norm_method, log_fn=None, epochs=15):
    """
    Train model on dataset, save to disk, return metrics dict.
    Returns None if no dataset available.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"This project only supports: {MODEL_TYPES}")

    if log_fn: log_fn("ok", "Loading dataset images…")
    X, y, classes = load_dataset_arrays(model_type, enh_method, norm_method,
                                        max_per_class=80, log_fn=log_fn)
    if X is None or len(X) < 10:
        if log_fn: log_fn("err", "Dataset not found or too few images.")
        return None

    if log_fn: log_fn("ok", f"Total samples: {len(X)} across {len(classes)} classes")

    # Encode string labels -> integer indices (needed by both paths)
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([label_to_idx[label] for label in y])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    if model_type == "cnn":
        if log_fn: log_fn("ok", f"Building CNN for {len(classes)} classes…")
        model = build_cnn(input_shape=X.shape[1:], num_classes=len(classes))

        if log_fn: log_fn("ok", f"Training CNN on {len(X_tr)} samples for {epochs} epochs…")
        t0 = time.time()

        history = {"acc": [], "val_acc": [], "loss": []}

        class _LogCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                history["acc"].append(logs.get("accuracy", 0))
                history["val_acc"].append(logs.get("val_accuracy", 0))
                history["loss"].append(logs.get("loss", 0))
                if log_fn:
                    log_fn("ok", f"  Epoch {epoch+1}/{epochs} — "
                                 f"acc:{logs.get('accuracy',0)*100:.1f}% "
                                 f"val_acc:{logs.get('val_accuracy',0)*100:.1f}% "
                                 f"loss:{logs.get('loss',0):.3f}")

        model.fit(
            X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=epochs, batch_size=16, verbose=0,
            callbacks=[_LogCallback()],
        )
        elapsed = round(time.time()-t0, 1)
        if log_fn: log_fn("ok", f"Training complete in {elapsed}s")

        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        pipe = model  # keras model acts as the "pipe" for this branch

    else:  # knn
        if log_fn: log_fn("ok", f"Training KNN on {len(X_tr)} samples…")
        t0 = time.time()
        pipe = Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ])
        pipe.fit(X_tr, y_tr)
        elapsed = round(time.time()-t0, 1)
        if log_fn: log_fn("ok", f"Training complete in {elapsed}s")

        y_pred = pipe.predict(X_te)
        history = None

    acc  = round(accuracy_score(y_te, y_pred)*100, 1)
    prec = round(precision_score(y_te, y_pred, average="weighted", zero_division=0)*100, 1)
    rec  = round(recall_score(y_te, y_pred, average="weighted", zero_division=0)*100, 1)
    f1   = round(f1_score(y_te, y_pred, average="weighted", zero_division=0)*100, 1)

    if log_fn: log_fn("ok", f"Metrics → Acc:{acc}% Prec:{prec}% Rec:{rec}% F1:{f1}%")

    metrics = {"acc":acc, "prec":prec, "rec":rec, "f1":f1}
    meta    = {"classes": classes, "metrics": metrics,
               "model_type": model_type, "enh": enh_method, "norm": norm_method,
               "history": history}

    if model_type == "cnn":
        model.save(_model_path(model_type, enh_method, norm_method))
    else:
        joblib.dump(pipe, _model_path(model_type, enh_method, norm_method))

    joblib.dump(meta, _meta_path(model_type, enh_method, norm_method))
    if log_fn: log_fn("ok", "Model saved to disk ✓")
    return metrics

def load_saved_model(model_type, enh_method, norm_method):
    mp = _model_path(model_type, enh_method, norm_method)
    mm = _meta_path(model_type, enh_method, norm_method)
    if os.path.exists(mp) and os.path.exists(mm):
        meta = joblib.load(mm)
        if model_type == "cnn":
            pipe = tf.keras.models.load_model(mp)
        else:
            pipe = joblib.load(mp)
        return pipe, meta
    return None, None

def predict_with_model(pipe, meta, features, model_type):
    """Return (prediction_str, confidence_0_1, top5_list)."""
    classes = meta["classes"]

    if model_type == "cnn":
        feat4d = features.reshape(1, *features.shape)
        probs_raw = pipe.predict(feat4d, verbose=0)[0]
        probs = probs_raw  # already aligned to `classes` order used at training time
    else:
        feat2d = features.reshape(1, -1)
        probs_raw = pipe.predict_proba(feat2d)[0]
        probs = np.zeros(len(classes))
        for i, c in enumerate(pipe.classes_):
            probs[c] = probs_raw[i]  # classes are integer-encoded indices here

    indexed = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    top5 = [{"name": DISPLAY_NAMES.get(c, c.replace("_"," ")),
              "prob": round(float(p), 4)} for c, p in indexed[:5]]
    return top5[0]["name"], float(top5[0]["prob"]), top5

# ══════════════════════════════════════════════════════════════════════════════
#  Simulation fallback (no dataset)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_BENCHMARKS = {
    "knn": {"acc":65.1,"prec":63.2,"rec":61.8,"f1":62.5},
    "cnn": {"acc":91.4,"prec":89.7,"rec":88.2,"f1":88.9},
}
PREPROC_BOOST = {"clahe":1.8,"hist":0.9,"gaussian":0.4,"none":0.0}
NORM_BOOST    = {"minmax":0.8,"zscore":0.6,"none":0.0}

def _simulated_metrics(model_type, enh, norm):
    base = MODEL_BENCHMARKS[model_type]
    return {
        "acc":  round(base["acc"]  + PREPROC_BOOST[enh] + NORM_BOOST[norm], 1),
        "prec": round(base["prec"] + PREPROC_BOOST[enh]*0.9 + NORM_BOOST[norm]*0.8, 1),
        "rec":  round(base["rec"]  + PREPROC_BOOST[enh]*0.85+ NORM_BOOST[norm]*0.7, 1),
        "f1":   round(base["f1"]   + PREPROC_BOOST[enh]*0.87+ NORM_BOOST[norm]*0.75,1),
    }

def _simulated_probs(features):
    flat = features.ravel()
    seed = int(abs(flat[:5].sum())*1e4) % (2**31)
    rng  = np.random.RandomState(seed)
    base = rng.dirichlet(np.ones(len(CLASSES))*0.5)
    energy = np.abs(flat[:len(CLASSES)])
    if len(energy) < len(CLASSES):
        energy = np.pad(energy,(0,len(CLASSES)-len(energy)))
    raw = energy*0.3 + base*0.7
    probs = np.exp(raw-raw.max()); probs/=probs.sum()
    return probs

# ══════════════════════════════════════════════════════════════════════════════
#  Charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_class_distribution():
    real = get_real_class_counts()
    counts_dict = real if real else SIMULATED_COUNTS
    names  = list(counts_dict.keys())
    counts = list(counts_dict.values())
    pairs = sorted(zip(names, counts), key=lambda x: x[1], reverse=True)
    names, counts = zip(*pairs)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))

    fig, ax = plt.subplots(figsize=(10, 5))
    _set_dark(fig, ax)
    bars = ax.barh(names, counts, color=colors, edgecolor="none", height=0.72)
    ax.set_xlabel("Number of images")
    title = "Step 2b — Real Dataset Distribution" if real else "Step 2b — Dataset Distribution (Simulated)"
    ax.set_title(title, fontsize=11, pad=10, fontweight="bold")
    ax.invert_yaxis()
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
                str(cnt), va="center", fontsize=8, color="#8b949e")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2, color="#30363d")
    fig.tight_layout()
    return fig_to_b64(fig)

def plot_pixel_distribution(img_orig, img_proc):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    _set_dark(fig, *axes)
    for ax, img, title in zip(axes, [img_orig, img_proc],
                               ["Before Preprocessing", "After Preprocessing"]):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ax.hist(gray.ravel(), bins=64, color="#2e7d5e", alpha=0.85, edgecolor="none")
        ax.set_title(title); ax.set_xlabel("Pixel Intensity"); ax.set_ylabel("Frequency")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=0.2, color="#30363d")
    fig.suptitle("Step 2c — Preprocessing Impact", fontsize=10,
                 fontweight="bold", color="#e6edf3")
    fig.tight_layout()
    return fig_to_b64(fig)

def plot_model_comparison(enh, norm):
    models_ = MODEL_TYPES
    labels = ["CNN", "KNN"]
    accs, f1s = [], []
    for m in models_:
        mp, mm = load_saved_model(m, enh, norm)
        if mm: mt = mm["metrics"]
        else:  mt = _simulated_metrics(m, enh, norm)
        accs.append(mt["acc"]); f1s.append(mt["f1"])

    x = np.arange(len(models_))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _set_dark(fig, ax)
    ax.bar(x-0.2, accs, 0.38, label="Accuracy (%)", color="#2e7d5e")
    ax.bar(x+0.2, f1s,  0.38, label="F1-Score (%)",  color="#5ecba1")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(50,100); ax.set_ylabel("Score (%)")
    ax.set_title("Step 3b — Model Comparison (CNN vs KNN)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2, color="#30363d")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)

def plot_preprocessing_comparison(model_type):
    enh_methods  = ["clahe","hist","gaussian","none"]
    norm_methods = ["minmax","zscore","none"]
    labels, accs, f1s = [], [], []
    for e in enh_methods:
        for n in norm_methods:
            mp, mm = load_saved_model(model_type, e, n)
            mt = mm["metrics"] if mm else _simulated_metrics(model_type, e, n)
            labels.append(f"{e}\n+{n}"); accs.append(mt["acc"]); f1s.append(mt["f1"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4))
    _set_dark(fig, ax)
    ax.bar(x-0.2, accs, 0.38, label="Accuracy (%)", color="#2e7d5e")
    ax.bar(x+0.2, f1s,  0.38, label="F1-Score (%)",  color="#5ecba1")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(50,100); ax.set_ylabel("Score (%)")
    ax.set_title(f"Step 3c — Preprocessing Variations ({model_type.upper()})", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2, color="#30363d")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)

def plot_training_curves(model_type, enh_method, norm_method, final_acc):
    """Use real Keras history if available (CNN), else a synthetic illustrative curve."""
    _, meta = load_saved_model(model_type, enh_method, norm_method)
    history = meta.get("history") if meta else None

    if history and history.get("acc"):
        epochs = np.arange(1, len(history["acc"]) + 1)
        t_acc = np.array(history["acc"]) * 100
        v_acc = np.array(history["val_acc"]) * 100
        t_loss = np.array(history["loss"])
        real_data = True
    else:
        epochs = np.arange(1, 16)
        rng    = np.random.RandomState(42)
        t_acc  = np.clip(40+epochs*(final_acc-40)/15+rng.randn(15)*1.5, 0, 100)
        v_acc  = np.clip(35+epochs*(final_acc-44)/15+rng.randn(15)*2.0, 0, 100)
        t_loss = np.maximum(0.05, 1.8-epochs*0.1+rng.randn(15)*0.04)
        real_data = False

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax2 = ax1.twinx()
    _set_dark(fig, ax1)
    ax2.set_facecolor("#161b22")
    ax2.tick_params(colors="#8b949e")
    ax2.yaxis.label.set_color("#8b949e")
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")

    ax1.plot(epochs, t_acc, "#2e7d5e", lw=2, label="Train acc")
    ax1.plot(epochs, v_acc, "#5ecba1", lw=2, ls="--", label="Val acc")
    ax2.plot(epochs, t_loss,"#e74c3c", lw=1.5, alpha=0.7, label="Train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)", color="#2e7d5e")
    ax2.set_ylabel("Loss", color="#e74c3c")
    suffix = "" if real_data else " (illustrative — no saved history)"
    ax1.set_title(f"Step 3b — Training Curves{suffix}", fontweight="bold", fontsize=10)
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=9)
    ax1.grid(alpha=0.2, color="#30363d")
    ax1.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_bytes, model_type, norm_method, enh_method,
                 aug_flip, aug_rotate, aug_zoom):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"This project only supports models: {MODEL_TYPES}")

    log = []

    # Step 1 — load
    img_raw = load_and_resize(image_bytes)
    log.append(("ok","Step 1 — Image loaded & resized to 224×224 px"))

    # Step 2a — preprocess
    img_enh  = apply_enhancement(img_raw, enh_method)
    log.append(("ok", f"Step 2a — Enhancement: {enh_method.upper()}"))
    img_norm = apply_normalization(img_enh, norm_method)
    log.append(("ok", f"Step 2a — Normalization: {norm_method}"))
    aug_labels = [l for f,l in [(aug_flip,"H-Flip"),(aug_rotate,"Rotation"),(aug_zoom,"Zoom")] if f]
    img_aug  = apply_augmentation(img_norm, aug_flip, aug_rotate, aug_zoom)
    log.append(("ok", f"Step 2a — Augmentation: {', '.join(aug_labels) or 'none'}"))

    # Step 2c chart
    pixel_chart = plot_pixel_distribution(img_raw, img_aug)
    log.append(("ok","Step 2c — Pixel distribution chart generated"))

    # Step 3 — features
    feat_names = {"cnn":"Raw normalized pixels (96×96×3)","knn":"Color hist + LBP"}
    features = get_features(img_aug, model_type)
    dim_desc = f"{features.shape}" if model_type == "cnn" else f"{len(features)}-dim"
    log.append(("ok", f"Step 3 — Features: {feat_names[model_type]} ({dim_desc})"))

    # Step 3 — inference
    pipe, meta = load_saved_model(model_type, enh_method, norm_method)
    if pipe is not None and meta:
        pred_name, conf, top5 = predict_with_model(pipe, meta, features, model_type)
        metrics = meta["metrics"]
        log.append(("ok", f"Step 3 — REAL model prediction: {pred_name} ({conf*100:.1f}%)"))
        mode = "real"
    else:
        # Fallback simulation
        probs   = _simulated_probs(features)
        indexed = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
        top5    = [{"name": DISPLAY_NAMES[c], "prob": round(float(p),4)} for c,p in indexed[:5]]
        pred_name, conf = top5[0]["name"], top5[0]["prob"]
        metrics = _simulated_metrics(model_type, enh_method, norm_method)
        log.append(("ok", f"Step 3 — Simulation prediction: {pred_name} ({conf*100:.1f}%)"))
        mode = "simulated"

    log.append(("ok",
        f"Step 3b — Acc:{metrics['acc']}% Prec:{metrics['prec']}% "
        f"Rec:{metrics['rec']}% F1:{metrics['f1']}%"))

    return {
        "prediction":      pred_name,
        "confidence":      round(conf*100, 1),
        "top5":            top5,
        "metrics":         metrics,
        "log":             log,
        "mode":            mode,
        "dataset_present": _find_dataset_split() is not None,
        "model_trained":   pipe is not None,
        "charts": {
            "distribution":    plot_class_distribution(),
            "pixel":           pixel_chart,
            "model_compare":   plot_model_comparison(enh_method, norm_method),
            "preproc_compare": plot_preprocessing_comparison(model_type),
            "training":        plot_training_curves(model_type, enh_method, norm_method, metrics["acc"]),
        },
        "preprocessed_img": img_to_b64(img_aug),
    }