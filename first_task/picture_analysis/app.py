# pip install -U ultralytics, pandas
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

results = model("https://img.championat.com/news/big/r/c/chempionat-mira-2022-po-futbolu-rezultaty-matchej-24-noyabrya-turnirnaya-tablica_16693249971950732690.jpg")

results.print()
results.save()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

# Fusing layers...
# YOLOv5s summary: 124 layers, 7,225,885 parameters, 0 gradients, 16.4 GFLOPs
# Adding AutoShape...
# image 1/1: 1024x1024 1 cat
# Speed: 491.5ms pre-process, 116.2ms inference, 2.1ms NMS per image at shape (1, 3, 640, 640)
# Saved 1 image to runs\detect\exp