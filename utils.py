
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

try:  
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

IMG_SIZE = 224
CLASS_NAMES = ["peace", "stop", "thumbs_up"]



def load_rgb(src):
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_array(pil_img, size=IMG_SIZE):
    return np.asarray(pil_img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def center_square(pil_img):
    w, h = pil_img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return pil_img.crop((l, t, l + s, t + s))



def _low_light(img, rng):
    x = (img / 255.0) ** 2.5 * 0.8
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def _overexposed(img, rng):
    return np.clip(img.astype(np.float32) * 1.6 + 30, 0, 255).astype(np.uint8)


def _motion_blur(img, rng):
    k = 15
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _gaussian_noise(img, rng):
    noise = rng.normal(0, 25, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(img, rng):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 10])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _defocus(img, rng):
    return cv2.GaussianBlur(img, (0, 0), 3.0)


def _color_cast(img, rng):
    gains = [(1.25, 1.0, 0.75), (0.75, 1.0, 1.25)][int(rng.integers(0, 2))]
    return np.clip(img.astype(np.float32) * np.array(gains), 0, 255).astype(np.uint8)


CONDITIONS = {
    "normal": lambda img, rng: img,
    "low_light": _low_light,
    "overexposed": _overexposed,
    "motion_blur": _motion_blur,
    "gaussian_noise": _gaussian_noise,
    "jpeg_compression": _jpeg,
    "defocus_blur": _defocus,
    "color_cast": _color_cast,
}
TRAIN_CONDITIONS = ["low_light", "overexposed", "motion_blur",
                    "gaussian_noise", "jpeg_compression"]
HELDOUT_CONDITIONS = ["defocus_blur", "color_cast"]


def apply_condition(img, name, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    return CONDITIONS[name](img, rng)


def enhance_exposure(img):
    """Lightweight exposure polishing: auto-gamma + CLAHE on the L channel.

    NOTE: this is a simple classical stand-in. It is NOT the ArtAug model.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    mean = max(float(l.mean()) / 255.0, 1e-3)
    gamma = float(np.clip(np.log(0.5) / np.log(mean), 0.4, 1.0))
    l = (255.0 * np.power(l / 255.0, gamma)).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def predict_probs(model, arr_uint8):
    x = arr_uint8[None].astype("float32")
    return model(x, training=False).numpy()[0]


def avg_latency_ms(model, arr_uint8, runs=20, warmup=3):
    x = tf.convert_to_tensor(arr_uint8[None].astype("float32"))
    for _ in range(warmup):
        model(x, training=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x, training=False).numpy()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))



def gradcam(model, arr_uint8):
    x = tf.convert_to_tensor(arr_uint8[None].astype("float32"))
    nested = [l for l in model.layers if isinstance(l, tf.keras.Model)]

    if nested:
        base = nested[0]
        resc = next(l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling))
        gap = next(l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
        drop = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dropout))
        dense = next(l for l in model.layers if isinstance(l, tf.keras.layers.Dense))
        with tf.GradientTape() as tape:
            conv = base(resc(x), training=False)
            tape.watch(conv)
            probs = dense(drop(gap(conv), training=False))
            score = probs[:, tf.argmax(probs[0])]
    else:
        last = model.get_layer("last_conv")
        grad_model = tf.keras.Model(model.inputs, [last.output, model.output])
        with tf.GradientTape() as tape:
            conv, probs = grad_model(x, training=False)
            score = probs[:, tf.argmax(probs[0])]

    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.reduce_sum(conv[0] * weights, axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy(), probs.numpy()[0]


def overlay_heatmap(arr_uint8, heat, alpha=0.45):
    h, w = arr_uint8.shape[:2]
    heat = cv2.resize(heat.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(arr_uint8, 1 - alpha, colored, alpha, 0)



_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
_DETECTOR = None


def detect_hands(img_uint8, task_path="hand_landmarker.task"):
    global _DETECTOR
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if _DETECTOR is None:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
        _DETECTOR = vision.HandLandmarker.create_from_options(options)

    img = np.ascontiguousarray(img_uint8)
    res = _DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    out = img.copy()
    h, w = out.shape[:2]
    n_landmarks = 0
    for hand in res.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        n_landmarks += len(pts)
        for a, b in _HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(out, p, 4, (255, 0, 0), -1)
    return out, len(res.hand_landmarks), n_landmarks
