from picamera2 import Picamera2
import cv2
import pyttsx3
import pytesseract
import time

# ==========================================
# TEXT TO SPEECH SETUP
# ==========================================

engine = pyttsx3.init()

engine.setProperty('rate', 135)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')

for voice in voices:
    if "india" in voice.id.lower():
        engine.setProperty('voice', voice.id)
        break

# ==========================================
# CAMERA SETUP
# ==========================================

print("Starting Camera...")

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

print("Camera started successfully.")

# ==========================================
# OCR CONTROL
# ==========================================

last_ocr_text = ""
last_ocr_time = time.time()

# ==========================================
# MAIN LOOP
# ==========================================

try:

    while True:

        frame = picam2.capture_array()

        # Convert BGRA -> BGR if needed
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR
            )

        display = frame.copy()

        # ==========================================
        # IMAGE PREPROCESSING
        # ==========================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ==========================================
        # OCR
        # ==========================================

        ocr_text = pytesseract.image_to_string(
            thresh
        )

        ocr_text = ocr_text.strip()

        # ==========================================
        # DISPLAY & SPEECH
        # ==========================================

        if len(ocr_text) > 5:

            cv2.putText(
                display,
                "TEXT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            print("\nDetected Text:")
            print(ocr_text)

            if (
                ocr_text != last_ocr_text
                or time.time() - last_ocr_time > 8
            ):

                engine.say(ocr_text)
                engine.runAndWait()

                last_ocr_text = ocr_text
                last_ocr_time = time.time()

        # ==========================================
        # DISPLAY WINDOW
        # ==========================================

        cv2.imshow(
            "OCR Reader",
            display
        )

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
