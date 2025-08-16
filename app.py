from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np

yolo_model = YOLO("license_plate.pt")

# Load PaddleOCR once
ocr_reader = PaddleOCR(lang='en')

# Open webcam
cap = cv2.VideoCapture(0) 

all_texts = []  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, verbose=False)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        plate_img = frame[y1:y2, x1:x2]

        ocr_result = ocr_reader.predict(plate_img)

        if ocr_result and len(ocr_result) > 0:
            res = ocr_result[0]

            if isinstance(res, dict) and "rec_texts" in res:
                texts = res["rec_texts"]
                scores = res["rec_scores"]

                if texts and scores:
                    best_idx = int(np.argmax(scores))
                    text = texts[best_idx]
                    score = scores[best_idx]

                    if score > 0.25:
                        all_texts.append(text)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
    cv2.imshow("License Plate Recognition (Press Q to exit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
