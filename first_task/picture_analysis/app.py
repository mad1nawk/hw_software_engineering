# pip install -U ultralytics, pandas
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

results = model("https://img.championat.com/news/big/r/c/chempionat-mira-2022-po-futbolu-rezultaty-matchej-24-noyabrya-turnirnaya-tablica_16693249971950732690.jpg")

results.print()
results.save()

results.xyxy[0]
results.pandas().xyxy[0]
