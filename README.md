# RealTimeOCR

RealTimeOCR is a computer vision project that combines YOLO (You Only Look Once) for object detection and PaddleOCR for optical character recognition (OCR) to identify and read text from objects in real-time video feeds. This project can be used for various applications, such as automated text extraction from documents, license plates, or any text-containing objects in videos.

## Features

- Real-time object detection using YOLO
- Text extraction from detected objects using PaddleOCR
- Customizable Region of Interest (ROI) for focused detection
- Easy integration with video streams

## Requirements

Make sure you have the following dependencies installed:

- Python 3.7+
- OpenCV
- Pandas
- NumPy
- PaddleOCR
- Ultralytics YOLO
- CVZone

You can install the necessary packages using pip:

```bash
pip install opencv-python pandas numpy paddleocr ultralytics cvzone
```
## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AmmarMohamed0/RealTimeOCR.git
   cd RealTimeOCR
2. **Download the YOLO weights:**
- Make sure to place your `best.pt` weights file in the project directory.

3. **Prepare the class labels:**
- Create a `coco.txt` file in the project directory with the class labels (one per line).

4.  **Capture a video:**
- Place a sample video file named `nr.mp4` in the project directory or modify the code to use your video file.

5.  **Run the project:**
```bash
python YOLO10_and_PaaddleOCR.py
```

## How It Works
1. The video feed is captured using OpenCV.
2. YOLO model predicts the bounding boxes for objects in the video.
3. The detected objects' bounding boxes are checked against a defined polygonal Region of Interest (ROI).
4. If an object is detected within the ROI, it is cropped, resized, and processed using PaddleOCR to extract any text.
5. The detected text is displayed on the video frame.

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [OpenCV](https://opencv.org/)
