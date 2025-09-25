import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")  # path to your weights

# Find the class ID for elephant
# You can check the model's class names with model.names
elephant_class_id = None
for idx, name in model.names.items():
    if name.lower() == "elephant":
        elephant_class_id = idx
        break

if elephant_class_id is None:
    print("Elephant class not found in model!")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Filter detections to only elephants
    for result in results:
        mask = result.boxes.cls == elephant_class_id
        if mask.any():
            result.boxes = result.boxes[mask]  # Keep only elephant boxes
        else:
            result.boxes = []  # No elephants in this frame

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Elephant Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
