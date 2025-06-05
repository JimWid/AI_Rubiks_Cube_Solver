import cv2
import sys
import torch
import numpy as np

# Path to the trained model
CUSTOM_MODEL_PATH = "./best.pt"
CONFIDENCE_THRESHOLD = 0.50

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit()

# Loading Custom Trained YOLOv5 Model using torch.hub
try:
    print(f"Loading YOLOv5 custom model from torch.hub: {CUSTOM_MODEL_PATH}")
    model = torch.hub.load(
        'ultralytics/yolov5', # The repository
        'custom',
        path=CUSTOM_MODEL_PATH, # Path to custom model
        force_reload=False,
    )
    model.conf = CONFIDENCE_THRESHOLD
    print("YOLOv5 model loaded successfully via torch.hub!")

except Exception as e:
    print(f"An error occurred during model loading with torch.hub: {e}")
    print("Please ensure:")
    print(f"1. The CUSTOM_MODEL_PATH '{CUSTOM_MODEL_PATH}' is correct.")
    print(f"2. You have a working internet connection for the first time loading (to fetch repo info).")
    print(f"3. The yolov5 repository (ultralytics/yolov5) is accessible by torch.hub.")
    print(f"4. Your PyTorch and yolov5 compatible dependencies are correctly installed.")
    print(f"5. Try with force_reload=True if you suspect caching issues.")
    sys.exit()

print(f"Using confidence threshold: {model.conf}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame_flipped = cv2.flip(frame, 1)

    # --- INPUT PREPROCESSING ---    
    img_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)

    # --- RESULTS ---
    results = model(img_rgb) # Pass the RGB frame to the model

    # --- DETECTIONS ---
    detections_df = results.pandas().xyxy[0]  # Detections for the first (and only) image

    for index, row in detections_df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = float(row['confidence'])
        class_id = int(row['class']) # Class ID
        label = row['name'] # Class name string

        print(f"Detected: {label} with confidence {confidence:.2f} at [{x1},{y1},{x2},{y2}]")

       
        if label.lower() == 'rubiks_cube':
            cv2.rectangle(frame_flipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame_flipped, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("No Rubik's Cube detected")


    cv2.imshow("Rubiks Cube Solver", frame_flipped)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()