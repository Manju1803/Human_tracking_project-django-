# website_app/camera_worker.py
import threading
import time
import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import face_recognition
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None
from django.conf import settings

CAPTURE_INTERVAL = 3.0
FACE_TOLERANCE = 0.6

CAPTURE_DIR = Path(settings.MEDIA_ROOT) / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

FACES_DIR = Path(settings.MEDIA_ROOT) / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "yolov8n.pt"

class PerCameraThread(threading.Thread):
    def __init__(self, cam_name: str, source: str, known_encs, known_names):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.source = source
        self.known_encs = known_encs
        self.known_names = known_names
        self.last_captures = {}
        self.latest_frame = None
        self.running = threading.Event()
        self.running.set()
        self.cap = None
        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception as e:
                print(f"⚠️ YOLO load fail for {cam_name}: {e}")

    def run(self):
        print(f"🔁 [Thread] Starting camera thread: {self.cam_name} -> {self.source}")
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not (self.cap and self.cap.isOpened()):
                print(f"❌ [Thread] Could not open source for {self.cam_name}")
                return
        except Exception as e:
            print(f"❌ [Thread] Exception opening source {self.cam_name}: {e}")
            return

        frame_count = 0
        last_fps_time = time.time()

        while self.running.is_set():
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print(f"⚠️ [Thread:{self.cam_name}] read error: {e}")
                ret = False
                frame = None

            if not ret or frame is None:
                # small backoff to avoid busy loop
                time.sleep(0.5)
                # try to reopen if closed
                try:
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                except Exception:
                    pass
                continue

            frame_count += 1
            if frame_count % 2 != 0:
                self.latest_frame = frame.copy()
                continue

            # face detection/recognition block (kept same logic)
            if self.model:
                try:
                    results = self.model(frame, conf=0.35, imgsz=640)
                except Exception as e:
                    results = []
                    print(f"⚠️ [Thread:{self.cam_name}] YOLO error: {e}")

                for r in results:
                    for box in r.boxes:
                        try:
                            cls_id = int(box.cls[0])
                        except Exception:
                            continue
                        if cls_id >= len(self.model.names):
                            continue
                        if self.model.names[cls_id] != "person":
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                        person_roi = frame[y1:y2, x1:x2].copy()
                        name = "Unknown"
                        try:
                            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_roi, model="hog")
                            for (top, right, bottom, left) in face_locations:
                                face_encs = face_recognition.face_encodings(rgb_roi, [(top, right, bottom, left)])
                                if face_encs:
                                    enc = face_encs[0]
                                    if len(self.known_encs) > 0:
                                        distances = face_recognition.face_distance(self.known_encs, enc)
                                        best_match = np.argmin(distances)
                                        min_dist = distances[best_match]
                                        if min_dist < FACE_TOLERANCE:
                                            name = self.known_names[best_match]
                        except Exception as e:
                            print(f"⚠️ [Thread:{self.cam_name}] face_rec error: {e}")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, name, (x1 + 5, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        now = time.time()
                        last_time = self.last_captures.get(name, 0)
                        if now - last_time > CAPTURE_INTERVAL:
                            self.last_captures[name] = now
                            self.save_cropped_person(person_roi, name)

            # display FPS on frame
            if time.time() - last_fps_time >= 1.0:
                fps = frame_count / (time.time() - last_fps_time)
                frame_count = 0
                last_fps_time = time.time()
                try:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                except Exception:
                    pass

            self.latest_frame = frame

        # cleanup
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        print(f"🛑 [Thread] Camera thread stopped: {self.cam_name}")

    def stop(self):
        """Stop the thread loop; release capture in run cleanup."""
        print(f"⏹️ [Thread] stop() called for {self.cam_name}")
        self.running.clear()

    def save_cropped_person(self, person_crop, name):
        if person_crop is None or getattr(person_crop, "size", 0) == 0:
            return
        h, w = person_crop.shape[:2]
        try:
            cv2.rectangle(person_crop, (0, 0), (w - 1, 25), (0, 255, 0), cv2.FILLED)
            cv2.putText(person_crop, name, (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        except Exception:
            pass

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name == "Unknown":
            save_dir = CAPTURE_DIR / "Unknown"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"Unknown_{ts}.jpg"
        else:
            save_path = CAPTURE_DIR / f"{name}_{ts}.jpg"

        try:
            cv2.imwrite(str(save_path), person_crop)
            print(f"📸 Saved {name} at {save_path}")
        except Exception as e:
            print(f"⚠️ [Thread] Failed to save crop: {e}")


class CameraManager:
    def __init__(self):
        self.threads = {}
        self.known_encs, self.known_names = self.load_known_faces()
        self._started = False
        self._lock = threading.Lock()

    def load_known_faces(self):
        encodings = []
        names = []
        if not FACES_DIR.exists():
            print("⚠️ Faces dir missing:", FACES_DIR)
            return encodings, names

        for person_dir in FACES_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            person_name = person_dir.name
            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                try:
                    img = face_recognition.load_image_file(str(img_path))
                    boxes = face_recognition.face_locations(img, model="hog")
                    encs = face_recognition.face_encodings(img, boxes)
                    for enc in encs:
                        encodings.append(enc)
                        names.append(person_name)
                except Exception as e:
                    print(f"⚠️ Failed to load known face {img_path}: {e}")
        print(f"✅ Loaded {len(encodings)} encodings for {len(set(names))} persons")
        return encodings, names

    def start_all(self):
        with self._lock:
            if self._started:
                print("ℹ️ CameraManager.start_all() called but already started.")
                return
            sources = getattr(settings, "CAM_SOURCES", {})
            print(f"🔁 CameraManager starting {len(sources)} sources.")
            for cam_name, source in sources.items():
                if cam_name in self.threads:
                    continue
                t = PerCameraThread(cam_name, source, self.known_encs, self.known_names)
                t.start()
                self.threads[cam_name] = t
            self._started = True

    def stop_all(self, timeout=5.0):
        with self._lock:
            if not self._started:
                print("ℹ️ CameraManager.stop_all() called but manager not started.")
                return
            print("🛑 CameraManager stopping all threads...")
            for name, t in list(self.threads.items()):
                try:
                    t.stop()
                except Exception:
                    pass

            for name, t in list(self.threads.items()):
                try:
                    t.join(timeout)
                    if t.is_alive():
                        print(f"⚠️ Thread {name} did not stop in {timeout}s")
                except Exception as e:
                    print(f"⚠️ Error joining thread {name}: {e}")

            self.threads = {}
            self._started = False
            print("✅ CameraManager stopped all threads.")

    def get_latest_frame(self, cam_name):
        t = self.threads.get(cam_name)
        if not t:
            return None
        return t.latest_frame

    def is_running(self):
        return self._started

manager = CameraManager()
