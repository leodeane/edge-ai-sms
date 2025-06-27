from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# Paths
model_path = "./models/flan_onnx"
tokenizer_name = "google/flan-t5-small"

# Load ONNX-wrapped model and tokenizer
model = ORTModelForSeq2SeqLM.from_pretrained(model_path, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def generate_incident_report(obj, sound, max_length=64):
    prompt = (
        f"You are monitoring a security system. Generate an incident report that "
        f"describes what just happened based on the following object and sound. "
        f"Object: {obj}. Sound: {sound}."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Change these as needed
object_name = "hammer"
sound_name = "glass breaking"

report = generate_incident_report(object_name, sound_name)
print("\nIncident Report: ", report.split())
