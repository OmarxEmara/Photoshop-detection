from ultralytics import YOLO

import os

base_path = "/home/workstation/ashry/ekyc/services/yolo_models"
for model in os.listdir(base_path):
    model = YOLO(f"{base_path}/{model}")

    # Export the model to ONNX format
    model.export(format="onnx")

# import os
# # Get the list of all files and directories
# path = ""
# dir_list = os.listdir(path)
# print("Files and directories in '", path, "' :")
# # prints all files
# print(dir_list)
