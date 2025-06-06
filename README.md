# Multi-Object Tracking with YOLOv8 + ByteTrack for Traffic Analysis

## Giới thiệu
Project này sử dụng YOLOv8x và ByteTrack cho bài toán multi-object Tracking. 

---

## Tính năng

-  Vehicle detection với YOLOv8
-  Object tracking qua các frame sử dụng ByteTrack
-  Có thể tùy ý sử dụng với input video khác

---

## Cấu trúc project
<pre> <code> MOT_YOLO_ByteTrack_traffic/
│
├── yolo/ # YOLOv8 detection code and model loading
│ └── yolov8x.pt # [Not included] Large model file – download separately
│
├── tracker/ # ByteTrack implementation
│
├── video/ # Input videos for testing
│
├── main.py # Main entry for detection + tracking
├── utils.py # Utility functions
├── requirements.txt # Required Python packages
└── README.md # Project documentation
</code> </pre>

## Demo

https://github.com/user-attachments/assets/866ba262-d650-4132-a947-143f0a3721da




