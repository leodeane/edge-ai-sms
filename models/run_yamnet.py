import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
import librosa
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_FOLDER = os.path.join(SCRIPT_DIR, "../datasets/sounds")

# Load the YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# load class names
class_names_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.strip().split(',')[2] for line in open(class_names_path).readlines()[1:]]

# Supported audio file extensions
valid_extensions = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')

# Loop through each file in the directory
for filename in os.listdir(SOUNDS_FOLDER):
    if not filename.lower().endswith(valid_extensions):
        continue  # skip non-audio files

    file_path = os.path.join(SOUNDS_FOLDER, filename)

    try:
        # Load and resample the audio to 16 kHz mono
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)

        # Run inference
        scores, embeddings, spectrogram = yamnet_model(waveform)

        # Get the top-scoring class
        mean_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(mean_scores)
        top_class_name = class_names[top_class.numpy()]

        print(f"{filename}: {top_class_name}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
