import os
import random
import numpy as np
from PIL import Image
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import onnxruntime as ort
from transformers import ViTImageProcessor, ViTForImageClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "../datasets/objects")
SOUND_FOLDER = os.path.join(SCRIPT_DIR, "../datasets/sounds")

# === IMAGE CLASSIFIER SETUP ===
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
image_model_config = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
image_session = ort.InferenceSession("vit.onnx")

def classify_random_image():
    image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise RuntimeError("No image files found in the objects folder.")
    
    chosen_img_path = random.choice(image_files)
    image = Image.open(chosen_img_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="np")
    outputs = image_session.run(None, {"pixel_values": inputs["pixel_values"]})
    logits = outputs[0]
    predicted_idx = logits.argmax(-1).item()
    label = image_model_config.config.id2label[predicted_idx]
    print(f"\nSelected Image: {os.path.basename(chosen_img_path)} → {label}")
    return label

# === AUDIO CLASSIFIER SETUP ===
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_map_path).readlines()[1:]]

def classify_random_audio():
    audio_files = [os.path.join(SOUND_FOLDER, f) for f in os.listdir(SOUND_FOLDER)
                   if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a'))]
    if not audio_files:
        raise RuntimeError("No audio files found in the sounds folder.")
    
    chosen_audio_path = random.choice(audio_files)
    waveform, sr = librosa.load(chosen_audio_path, sr=16000, mono=True)
    waveform = waveform.astype(np.float32)
    scores, _, _ = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(mean_scores)
    label = class_names[top_class.numpy()]
    print(f"Selected Audio: {os.path.basename(chosen_audio_path)} → {label}")
    return label

# === LANGUAGE MODEL SETUP ===
flan_model = ORTModelForSeq2SeqLM.from_pretrained("./models/flan_onnx", use_cache=False)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def generate_incident_report(obj_label, sound_label, max_length=64):
    prompt = (
        f"You are monitoring a security system. Generate an incident report that "
        f"describes what just happened based on the following object and sound. "
        f"Object: {obj_label}. Sound: {sound_label}."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = flan_model.generate(**inputs, max_length=max_length)
    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return report

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        for i in range(10):
            object_label = classify_random_image()
            sound_label = classify_random_audio()
            report = generate_incident_report(object_label, sound_label)
            print(f"\nIncident Report:\n", report)

    except Exception as e:
        print(f"\n[ERROR] {e}")
