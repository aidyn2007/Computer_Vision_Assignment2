# HandGesture AI (Computer Vision, Assignment 2)

Hand gesture recognition (peace / stop / thumbs up) with MobileNetV2, synthetic data from a
text-to-video model (CogVideoX-2B), synthetic stress-test generator, Grad-CAM and MediaPipe landmarks.

## Run the app locally
```
pip install -r requirements.txt
streamlit run app.py
```
Needs the folders `models/` (trained .keras weights) and `results/`, plus `hand_landmarker.task` (MediaPipe, optional).
The app is also launched from the notebook in Google Colab through a Cloudflare tunnel.

## Reproduce the experiments
Open the notebook in Colab (GPU runtime), upload `hand_gesture_dataset.zip` and run all cells.
Seed=42, epochs=12, image size=224. Environment: {"python": "3.13.18", "tensorflow": "2.21.0", "platform": "Linux-6.6.159+-x86_64-with-glibc2.39", "device": "CPU", "gpu_name": "none"}

## Ethics
Synthetic media (text-to-video frames and code-generated degraded images) are labelled SYNTHETIC and kept in separate folders.
Only photos of the authors' own hands are used; the app does not store uploads. The dataset is small and made by few people,
so performance for other skin tones, hand sizes, lighting and backgrounds is not guaranteed.
