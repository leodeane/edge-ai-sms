import os
import time
from PIL import Image
import psutil
import onnxruntime as ort
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "../datasets/objects")  # adjust relative path as needed
IMAGE_FOLDER = os.path.normpath(IMAGE_FOLDER)
ONNX_MODEL_PATH = "vit.onnx"
RESULTS_FILE = "results.txt"

# Initialize processor, model config, and ONNX session with profiling enabled
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

session_options = ort.SessionOptions()
session_options.enable_profiling = True  # ONNX Runtime internal profiling
session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=session_options)

# Helper to get system usage snapshot
def get_system_usage():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    return cpu, ram

# Gather image file paths
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"No images found in folder {IMAGE_FOLDER}")
    exit(1)

# Lists to store measurements
preprocess_times = []
inference_times = []
cpu_usages = []
ram_usages = []

print(f"Profiling {len(image_files)} images from {IMAGE_FOLDER}...\n")

with open(RESULTS_FILE, 'w') as results:
    results.write(f"Profiling {len(image_files)} images from {IMAGE_FOLDER}...\n")

for img_path in image_files:
    try:
        start_pre = time.time()
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="np")
        preprocess_time = time.time() - start_pre

        # Measure system usage before inference
        cpu_before, ram_before = get_system_usage()

        start_inf = time.time()
        outputs = session.run(None, {"pixel_values": inputs["pixel_values"]})
        inference_time = time.time() - start_inf

        cpu_after, ram_after = get_system_usage()

        # Store times and usage
        preprocess_times.append(preprocess_time)
        inference_times.append(inference_time)
        cpu_usages.append((cpu_before + cpu_after) / 2)
        ram_usages.append((ram_before + ram_after) / 2)

        logits = outputs[0]
        predicted_idx = logits.argmax(-1).item()
        label = model.config.id2label[predicted_idx]

        time_usage = (f"{os.path.basename(img_path):20} → {label:25} "
              f"Preproc: {preprocess_time:.3f}s, Inf: {inference_time:.3f}s, "
              f"CPU: {cpu_usages[-1]:5.1f}%, RAM: {ram_usages[-1]:5.1f}%")
        
        print(time_usage)

        with open(RESULTS_FILE, 'a') as results:
            results.write(time_usage)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Summarize results
def summarize(name, data):
    return f"{name}: mean={np.mean(data):.3f}, min={np.min(data):.3f}, max={np.max(data):.3f}"

print("\n--- Profiling Summary ---")
print(summarize("Preprocessing time (s)", preprocess_times))
print(summarize("Inference time (s)", inference_times))
print(summarize("CPU usage (%)", cpu_usages))
print(summarize("RAM usage (%)", ram_usages))

with open(RESULTS_FILE, 'a') as results:
    results.write("\n--- Profiling Summary ---")
    results.write(summarize("Preprocessing time (s)", preprocess_times))
    results.write(summarize("Inference time (s)", inference_times))
    results.write(summarize("CPU usage (%)", cpu_usages))
    results.write(summarize("RAM usage (%)", ram_usages))

# Save ONNX Runtime profiling file
profile_file = session.end_profiling()
print(f"\nONNX Runtime internal profiling saved to: {profile_file}")
print("You can open it in Chrome tracing viewer: chrome://tracing")
