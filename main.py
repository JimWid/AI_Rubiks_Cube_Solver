import cv2
import sys
import torch
import numpy as np

# Path to the trained model
CUSTOM_MODEL_PATH = "./yolov5/runs/train/rubiks_cube_detector10/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5

# Color Detection Setup - HSV = HUE, SATURATION, VALUE
HSV_RANGES = {
    'red':    [[160, 100, 100], [179, 255, 255]], # Also covers a bit of the 0-10 range in the get_color_name function
    'orange': [[10, 100, 100], [25, 255, 255]],
    'yellow': [[26, 100, 100], [40, 255, 255]],
    'green':  [[41, 80, 80], [85, 255, 255]],
    'blue':   [[86, 100, 100], [130, 255, 255]],
    'white':  [[0, 0, 180], [179, 80, 255]] # White has low saturation and high value
    }
        
# Cube State Representation Setup: up, right, front, down, left and back
KOCIEMBA_FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

# Each letter reprents a color, we are using the standar color scheme
COLOR_TO_FACE_MAP = {
    'white':  'U',
    'red':    'R',
    'green':  'F',
    'yellow': 'D',
    'orange': 'L',
    'blue':   'B'
    }
        
# Dictionary to hold and the state as we scan the cube faces
# Initializing cube state as None, meaning the faces haven't been scanned
cube_state = {face : [None] * 9 for face in KOCIEMBA_FACE_ORDER}

# Utility Functions:
# Getting the name for each color
def get_color_name(h, s, v): # HUE, SATURATION, VALUE

    # Checking for white first, since its HUE can be anything but sarutation is low.
    if HSV_RANGES["white"][0][1] <= s <= HSV_RANGES["white"][1][1] and \
       HSV_RANGES["white"][0][2] <= v <= HSV_RANGES["white"][1][2]:
       return 'white'
        
    # Cheking for Red, which HUE is close to 180 / highest
    if (0 <= h <= 10 or 160 <= h <= 179) and s > 100 and v > 100:
        return "red"
        
    # For Loop to iterate HSV_Ranges and return the color
    for color, (lower, upper) in HSV_RANGES.items():
        if color in ["red", "white"]: # Skiping these two since we handled them already
            continue

        if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
            return color

    return None

# Generating the kociemba string
def generate_kociemba_string(state):
        
    if any(None in face_colors for face_colors in state.values()):
        return "Error: not all faces have been scanned"
        
    face_to_color_map = {v : k for k, v in COLOR_TO_FACE_MAP.items()}
            
    kociemba_string = ""

    for face_char in KOCIEMBA_FACE_ORDER:
        center_color_name = face_to_color_map[face_char]

        scanned_colors = state[face_char]

        for color_name in scanned_colors:
            if color_name in COLOR_TO_FACE_MAP:
                    kociemba_string += COLOR_TO_FACE_MAP[color_name]

            else:
                kociemba_string += "?" # This should not happen but good for debuggin purposes.

    return kociemba_string

# Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera D:")
    sys.exit()

# Loading Custom Trained YOLOv5 Model using torch.hub
try:
    print(f"Loading YOLOv5 custom model from torch.hub: {CUSTOM_MODEL_PATH}")
    # force_reload=True can help if you have caching issues
    model = torch.hub.load(
        'ultralytics/yolov5', # The repository
        'custom',             # Specify 'custom' for your own weights
        path=CUSTOM_MODEL_PATH, # Path to your .pt file
        force_reload=False,   # Set to True if you suspect cache issues or updated the model
    )
    model.conf = CONFIDENCE_THRESHOLD 
    print("YOLOv5 model loaded successfully via torch.hub!")

except Exception as e:
    print(f"An error occurred during model loading: {e}")
    sys.exit()

print(f"Using confidence threshold: {model.conf}")

current_face_state = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)

    display_frame = frame.copy()

    # Converting frame to RGB for the model YOLOv5   
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb) # Passing the RGB frame to the model
    detections_df = results.pandas().xyxy[0] 

    # Now we are only detecting the cube with the highest confidence %

    if not detections_df.empty:
        row = detections_df.iloc[0]
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])

        # Drawing the main bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Rubiks Cube (Confidence:{row['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Sticker Region Caculation (Aprox.)
        box_width = x2 - x1
        box_height = y2 - y1

        # Calculation ROI: Region Of Interest
        # For every sticker in a 3x3 grid + adding a padding to avoid black borders between stickers
        padding_x = int(box_width * 0.05) // 3
        padding_y = int(box_height * 0.05) // 3
        sticker_width = (box_height - 2 * padding_x * 3) // 3
        sticker_height = (box_height - 2 * padding_y  *3) // 3

        detected_face_colors = []

        for row in range(3): # Row of 3
            for col in range(3): # Column of 3
                sticker_x1 = x1 + col * (sticker_width + 2 * padding_x) + padding_x
                sticker_y1 = y1 + row * (sticker_height + 2 * padding_y) + padding_y
                sticker_x2 = sticker_x1 + sticker_width
                sticker_y2 = sticker_y1 + sticker_height   

                # ROI = Region Of Interest
                sticker_roi = frame[sticker_y1:sticker_y2, sticker_x1:sticker_x2]

                if sticker_roi.size > 0:
                    # Converting ROI to HSV
                    hsv_roi = cv2.cvtColor(sticker_roi, cv2.COLOR_BGR2HSV)

                    # Calculating the median HSV value of the ROI
                    h, s, v = np.median(hsv_roi.reshape(-1, 3), axis=0).astype(int)

                    # Getting color names
                    color_name = get_color_name(h, s, v)
                    detected_face_colors.append(color_name)

                    # Drawing feedback on display frame
                    cv2.rectangle(display_frame, (sticker_x1, sticker_y1), (sticker_x2, sticker_y2), (255, 0, 0), 1)
                    if color_name:
                        text_color = (0, 0, 0) if color_name in ["white", "yellow"] else (255, 255, 255)
                        cv2.putText(display_frame, color_name[0].upper(), (sticker_x1 + 5, sticker_y1 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                else:
                    detected_face_colors.append(None) # Append None if ROI is empty (should not happen)

        current_face_state = detected_face_colors

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Map: Key to Face:
    key_to_face = {'w': 'U', 'r': 'R', 'g': 'F', 'y': 'D', 'o': 'L', 'b': 'B'}

    if chr(key) in key_to_face:
        face_char = key_to_face[chr(key)]
        if current_face_state and len(current_face_state) == 9:
            
            cube_state[face_char] = current_face_state
            print(f"Scanned Face: '{face_char}': {current_face_state}")

        else:
            print(f"Could not scan face '{face_char}'. Cube not clearly visible.")

    if key == ord('c'):
        cube_state = {face: [None] * 9 for face in KOCIEMBA_FACE_ORDER}
        print("\n--- All scanned data cleared. ---\n")

    # Displaying Instructions and Status
    y_offset = 30
    scanned_faces = [face for face, colors in cube_state.items() if colors[0] is not None]
    cv2.putText(display_frame, "Press key for the face's CENTER color:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    cv2.putText(display_frame, "'w:white r:red g:green y:yellow o:orange b:blue", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(display_frame, f"Scanned Faces: {scanned_faces}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Checking if all faces are scanned and generate the final string
    if len(scanned_faces) == 6:
        y_offset += 30
        k_string = generate_kociemba_string(cube_state)
        cv2.putText(display_frame, "All faces scanned! Press 'c' to clear.", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(display_frame, f"Kociemba: {k_string}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Rubiks Cube Scanner!!!", display_frame)

cap.release()
cv2.destroyAllWindows()