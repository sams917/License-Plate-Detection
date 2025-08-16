from ultralytics import YOLO
import easyocr
import cv2

yolo_model = YOLO("license_plate.pt") 
ocr_reader = easyocr.Reader(['en']) 

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        plate_img = frame[y1:y2, x1:x2]

        
        ocr_result = ocr_reader.readtext(plate_img)
        for (_, text, prob) in ocr_result:
            if prob > 0.25:
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    cv2.imshow("Live License Plate Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
