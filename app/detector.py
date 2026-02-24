"""YOLO-based vehicle detector/tracker wrapper."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import settings

# YOLO class names that map to vehicle categories
_YOLO_VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "motorbike",
    "van", "lorry", "bicycle",
}

BBoxResult = Tuple[int, int, int, int, float, str]  # x1,y1,x2,y2,conf,class_name


class Detector:
    """Lazy-loads a YOLO model and runs track() on frames."""

    def __init__(self) -> None:
        self._model = None
        self._allowed: List[str] = settings.allowed_classes

    def _load(self) -> None:
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(settings.model_path)

    def detect_and_track(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, int, float, str]]:
        """Run YOLOv8 tracking on *frame*.

        Returns list of (x1, y1, x2, y2, track_id, confidence, class_name).
        """
        if self._model is None:
            self._load()

        results = self._model.track(  # type: ignore[union-attr]
            frame,
            persist=True,
            conf=settings.conf_threshold,
            iou=settings.iou_threshold,
            verbose=False,
        )
        detections: List[Tuple[int, int, int, int, int, float, str]] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                class_name = result.names[cls_id]
                if class_name not in self._allowed:
                    continue
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                track_id = (
                    int(boxes.id[i].item()) if boxes.id is not None else -1
                )
                detections.append((x1, y1, x2, y2, track_id, conf, class_name))

        return detections
