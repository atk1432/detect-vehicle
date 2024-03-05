import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 50)
fontScale              = 1
fontColor              = (255, 0, 0) 
thickness              = 2
lineType               = 2

target_fps = 15
start_time = time.time()


while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        count = sum((results[0].boxes.cls >= 1) & (results[0].boxes.cls <= 7))
        cv2.putText(annotated_frame, f'Number trans: {count}',
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()