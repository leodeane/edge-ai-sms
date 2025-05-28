from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import requests

# Load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Load processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.eval()  # Set to eval mode

# Process the image
inputs = processor(images=image, return_tensors="pt")

# Define dummy input for export
dummy_input = inputs["pixel_values"]

# Export to ONNX
torch.onnx.export(
    model,                                 # model being run
    (dummy_input,),                        # model input (a tuple)
    "vit.onnx",                            # where to save the model
    input_names=["pixel_values"],          # input tensor names
    output_names=["logits"],               # output tensor names
    dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},  # dynamic batch size
    opset_version=14,                      # ONNX opset version
    do_constant_folding=True               # optimization
)

print("Model successfully exported to vit.onnx")