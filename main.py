import cv2
import time
import csv
import os
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


class Tracker:
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        matched = set()
        for track_id, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det = None
            best_det_idx = None
            for idx, det in enumerate(detections):
                if idx in matched:
                    continue
                iou_val = iou(track["box"], det["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det
                    best_det_idx = idx
            if best_det is not None:
                matched.add(best_det_idx)
                self.tracks[track_id] = {
                    "box": best_det["box"],
                    "cls": best_det["cls"],
                    "lost": 0
                }
                detections[best_det_idx]["id"] = track_id
            else:
                track["lost"] += 1
                if track["lost"] > self.max_lost:
                    del self.tracks[track_id]

        for idx, det in enumerate(detections):
            if idx not in matched:
                det["id"] = self.next_id
                self.tracks[self.next_id] = {
                    "box": det["box"],
                    "cls": det["cls"],
                    "lost": 0
                }
                self.next_id += 1

        return detections


def get_browser_compatible_writer(output_path, fourcc_str, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, fourcc_str
    writer.release()
    return None, None


def create_video_writer(output_path, fps, width, height):
    codecs = ["avc1", "H264", "mp4v"]
    for codec in codecs:
        writer, used = get_browser_compatible_writer(
            output_path, codec, fps, width, height
        )
        if writer is not None:
            return writer, used
    fallback = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fallback, fps, (width, height)), "mp4v"


def validate_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"valid": False, "error": "Cannot open video file"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "valid": True,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }


def process_video(input_video, target_width=640):
    info = validate_video(input_video)
    if not info["valid"]:
        raise ValueError(f"Invalid input video: {info.get('error')}")

    output_video = os.path.join(OUTPUT_FOLDER, "output_video.mp4")
    csv_file = os.path.join(OUTPUT_FOLDER, "speed_log.csv")

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = target_width
    height = int(orig_height * (target_width / orig_width))
    height = height if height % 2 == 0 else height + 1

    writer, used_codec = create_video_writer(output_video, fps, width, height)

    tracker = Tracker()
    vehicle_positions = {}

    csv_out = open(csv_file, "w", newline="")
    csv_writer = csv.writer(csv_out)
    csv_writer.writerow(["Vehicle_ID", "Type", "Speed_km_h"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        results = get_model()(frame, verbose=False)
        boxes_data = results[0].boxes

        detections = []
        if boxes_data is not None and boxes_data.cls is not None:
            classes = boxes_data.cls.cpu().numpy().astype(int)
            coordinates = boxes_data.xyxy.cpu().numpy()

            for cls, box in zip(classes, coordinates):
                if cls in VEHICLE_CLASSES:
                    detections.append({
                        "box": box,
                        "cls": cls
                    })

        detections = tracker.update(detections)

        for det in detections:
            vehicle_id = det["id"]
            cls = det["cls"]
            x1, y1, x2, y2 = map(int, det["box"])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_time = time.time()

            speed = 0.0
            if vehicle_id in vehicle_positions:
                old_x, old_y, old_time = vehicle_positions[vehicle_id]
                pixel_dist = np.hypot(center_x - old_x, center_y - old_y)
                time_diff = current_time - old_time
                if time_diff > 0:
                    meters = pixel_dist / 10.0
                    speed = (meters / time_diff) * 3.6

            vehicle_positions[vehicle_id] = (center_x, center_y, current_time)

            label = f"{VEHICLE_CLASSES[cls]} ID:{vehicle_id} {speed:.1f} km/h"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            csv_writer.writerow([vehicle_id, VEHICLE_CLASSES[cls], round(speed, 2)])

        writer.write(frame)

    cap.release()
    writer.release()
    csv_out.close()

    return output_video, csv_file
