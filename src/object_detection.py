from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import pyttsx3
import time

# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully.")

# ==========================================
# TEXT TO SPEECH SETUP
# ==========================================

engine = pyttsx3.init()

# Better speech settings
engine.setProperty('rate', 135)
engine.setProperty('volume', 1.0)

# Try Indian English voice
voices = engine.getProperty('voices')

for voice in voices:
    if "india" in voice.id.lower():
        engine.setProperty('voice', voice.id)
        break

# ==========================================
# CAMERA SETUP
# ==========================================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)

picam2.start()

time.sleep(2)

print("Camera started successfully.")

# ==========================================
# SPEECH CONTROL
# ==========================================

last_spoken = ""
last_time = time.time()

# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        frame = picam2.capture_array()

        # Convert BGRA to BGR
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        height, width, _ = frame.shape

        # YOLO inference
        results = model(frame)

        annotated_frame = results[0].plot()

        detected_sentence = ""

        # ==========================================
        # PROCESS DETECTIONS
        # ==========================================

        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box center
            obj_center_x = (x1 + x2) // 2

            # Bounding box width
            box_width = x2 - x1

            # ==========================================
            # DIRECTION DETECTION
            # ==========================================

            if obj_center_x < width // 3:
                direction = "left"

            elif obj_center_x > 2 * width // 3:
                direction = "right"

            else:
                direction = "front"

            # ==========================================
            # DISTANCE ESTIMATION
            # ==========================================

            # Bigger box = closer object
            if box_width > 300:
                distance = "very close"

            elif box_width > 180:
                distance = "close"

            elif box_width > 100:
                distance = "near"

            else:
                distance = "far"

            # ==========================================
            # ALERT MESSAGE
            # ==========================================

            if distance == "very close":
                speech_text = f"Warning! {label} very close on the {direction}"

            else:
                speech_text = f"{label} on the {direction}, {distance}"

            detected_sentence = speech_text

            # Draw extra text
            cv2.putText(
                annotated_frame,
                speech_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Speak only first object
            break

        # ==========================================
        # SPEECH OUTPUT
        # ==========================================

        if detected_sentence != "":

            if (
                detected_sentence != last_spoken
                or time.time() - last_time > 4
            ):

                print(detected_sentence)

                engine.say(detected_sentence)
                engine.runAndWait()

                last_spoken = detected_sentence
                last_time = time.time()

        # ==========================================
        # DISPLAY WINDOW
        # ==========================================

        cv2.imshow("Smart Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ==========================================
# EXIT
# ==========================================

except KeyboardInterrupt:

    print("\nStopped by user.")

finally:

    picam2.stop()

    cv2.destroyAllWindows()

    print("Program exited safely.")
