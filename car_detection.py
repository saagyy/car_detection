# ukonceni pomoci CTRL C
from pathlib import Path
import time
import cv2
from ultralytics import YOLO

#url streamu, a nazev vysledneho videa
STREAM_URL = ""
OUTPUT = "output.mp4"

# nacteni modelu
model = YOLO("yolov8n.pt")
# moznost aby to jelo na gpu
# model.to("cuda")

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Nelze se připojit k: {STREAM_URL}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#soubor pro ulozeni snimků
video_out_path = Path(OUTPUT)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))


try:
    print("Ukonči pomocí CTRL C")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ztráta spojení se streamem")
            time.sleep(1)
            continue
        # 60 % pravdepodobnost ze je to auto
        results = model(frame, conf=0.6, verbose=False)
        result = results[0]
        
        save_frame = False

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            
            #detekce auta
            if class_name == "car":
                save_frame = True
                break  

        #ulozeni snimku s autem
        if save_frame:
            out.write(frame)

# control c
except KeyboardInterrupt:
    print("\nUkončení")

finally:
    #ulozeni videa
    cap.release()
    out.release()
    print(f"Video bylo uloženo do: {video_out_path}")