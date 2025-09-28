from ultralytics import YOLO

model = YOLO("/home/user/Desktop/ekyc-id-front/models/detect_id_card.pt")

img_pth = "/home/user/Desktop/ekyc-id-front/data/omarID.png"

model.predict(img_pth, save= True,device="cpu")

# print all class names
print(model.names)          # dict: {0: 'photo', 1: 'invalid_photo', 2: 'firstName', ...}

# print them one per line
for idx, name in model.names.items():
    print(idx, name)