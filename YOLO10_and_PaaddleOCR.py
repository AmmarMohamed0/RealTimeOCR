import cv2
import pandas as pd
import os
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone

# Initialize the YOLO Model
yolo_model = YOLO("best.pt")

# Initialize PaddleOCR
ocr_model = PaddleOCR()

# Load class labels from the file
with open("coco.txt", "r") as class_file:
    class_labels_data = class_file.read()
    class_list = class_labels_data.split("\n")

# Capture the video
video_capture = cv2.VideoCapture('nr.mp4')

# Define the Region of Interest (ROI) polygon
roi_polygon = [(124, 339), (127, 451), (485, 440), (460, 328)]

def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("The image is None")
    result = ocr_model.ocr(image_array, rec=True)

    detected_text = []
    if result[0] is not None:
        for line in result[0]:
            text = line[1][0]
            detected_text.append(text)
        return ''.join(detected_text)

while True:            

    is_frame_read, video_frame = video_capture.read()
    if not is_frame_read:
        break
    
    # Resize the video frame
    video_frame = cv2.resize(video_frame, (1000, 400))

    # Get predictions from the YOLO model
    results = yolo_model.predict(video_frame)
    bounding_box_data = results[0].boxes.data
    bounding_box_data = bounding_box_data.cpu()
    
    bounding_box_df = pd.DataFrame(bounding_box_data).astype(float)
    # if need do one step
    #bounding_box_df = pd.DataFrame(results[0].boxes.data.cpu()).astype(float)

    detected_objects_list = []

    for index, row in bounding_box_df.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        label = int(row[5])
        class_name = class_list[label]

        # Calculate the center coordinates
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Check if the object is inside the defined polygon
        result = cv2.pointPolygonTest(np.array(roi_polygon, np.int32), (center_x, center_y), False)
        if result >= 0:
            # Draw a rectangle around the detected object
            cv2.rectangle(video_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Crop the object from the frame for OCR
            cropped_image = video_frame[y1:y2, x1:x2]
            cropped_image = cv2.resize(cropped_image, (110, 30))
            
            # Perform OCR on the cropped image
            detected_text = perform_ocr(cropped_image)
            
            # Display the detected text on the frame
            cvzone.putTextRect(video_frame, f"{detected_text}", (center_x - 50, center_y - 30), 1, 2)

    # Display the processed frame
    cv2.imshow("frame", video_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()