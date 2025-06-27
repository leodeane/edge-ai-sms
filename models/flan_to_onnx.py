from optimum.exporters.onnx import main_export

model_id = "google/flan-t5-small"
output_dir = "./models/flan_onnx"

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="text2text-generation",
    opset=14
)

print(f"ONNX model exported to: {output_dir}")
