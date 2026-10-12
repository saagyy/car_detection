import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Zpracování videa pomocí YOLOv8 na GPU.")
parser.add_argument("video", type=str, help="Cesta ke vstupnímu videu")
args = parser.parse_args()

video_path = Path(args.video)
if not video_path.exists():
    print(f"Soubor {video_path} neexistuje.")
    exit(1)

video_out_path = video_path.with_name(f"{video_path.stem}_out{video_path.suffix}")

#nacteni modelu
model = YOLO("yolov8n.pt")
#yolo pobezi na gpu
#model.to("cuda")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Nelze otevřít video.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.6, verbose=False)
    result = results[0]
    
    save_frame = False

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        
        if class_name == "car":
            save_frame = True
            #print("naslo se auto")
            break  

    if save_frame:
        out.write(frame)
    else:
        # pokud neni auto preskoc 5 snimku
        for _ in range(5):
            if not cap.grab():
                break

cap.release()
out.release()
print(f"Video bylo uloženo do: {video_out_path}")