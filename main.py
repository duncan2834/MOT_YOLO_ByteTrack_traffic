from object_tracking import ObjectTracking
from ultralytics import YOLO
import sys
print(">>> Python executable:", sys.executable)
INPUT_PATH = "video_traffic.mp4"
OUTPUT_PATH = "video_traffic_result.mp4"

if __name__ == "__main__":
    obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH)
    obj.process()