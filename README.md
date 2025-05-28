# edge-ai-sms
repo for my edge-ai project, smart monitoring system

---

# models

This directory includes the models that are used for the security system.

vit_to_onnx.py converts the google/vit-base-patch16-224 model from PyTorch to ONNX.

The ONNX version of the model will be saved as vit.onnx.

run_vit_as_onnx.py is the script that is used to pass images into and run the Vit model as ONNX.

To use the model, first,  place a folder of images you wish to categorize into the datasets directory in a directory named "objects".

Then, cd to the models directory and run `python vit_to_onnx.py`, when this completes you should see vit.onnx in the models directory.

Next, run `python run_vit_as_onnx.py`. 

Depending on how many images are in your datasets/objects folder this will take some time. 

Once the model is finished categorizing your images you will have a couple of performance reports, a profile and a general results text file.

Results example:

![image](https://github.com/user-attachments/assets/d3717529-e4d7-4b70-a285-5e3e1097e81a)

