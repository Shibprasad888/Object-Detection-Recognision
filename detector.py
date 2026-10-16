from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from alert_manager import AlertManager
from alert_rules import AlertRules
from dashboard_manager import DashboardManager
from scene_analytics import SceneAnalytics


def build_detection_label(class_name: str, recognized_name: str | None, track_id: int | None, confidence: float) -> str:
    """Create the compact overlay label shown above each detected object."""
    candidate = (recognized_name or "").strip()
    if candidate and candidate.lower() not in {"unknown", "none", ""}:
        return candidate

    fallback = (class_name or "object").strip()
    if not fallback:
        return "object"

    friendly_names = {
        "person": "human",
        "people": "human",
        "man": "human",
        "woman": "human",
    }
    return friendly_names.get(fallback.lower(), fallback)


class ObjectDetector:
    """Load a YOLOv8 model once and run detection, tracking, recognition, and management."""

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.35,
        imgsz: int = 480,
        recognition_threshold: float = 0.30,
        device: str = "cpu",
    ) -> None:
        self.confidence = confidence
        self.imgsz = imgsz
        self.recognition_threshold = recognition_threshold
        self.device = str(device or "cpu")
        self.model_path = Path(model_path or Path(__file__).resolve().parent / "models" / "yolov8n.pt")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.tracker_config = "bytetrack.yaml"
        self.db_path = self.model_path.parent / "object_registry.db"
        self.max_detections = 80
        self.model = self._load_model()
        self.class_names = getattr(self.model, "names", {})
        self.clip_model: Optional[Any] = None
        self.clip_preprocess: Optional[Any] = None
        self.embedding_cache: Dict[int, Dict[str, Any]] = {}
        self.recognition_state: Dict[int, Dict[str, Any]] = {}
        self._registered_objects: List[Dict[str, Any]] = []
        self.scene_analytics = SceneAnalytics()
        self.alert_manager = AlertManager(base_dir=self.model_path.parent)
        self.alert_rules = AlertRules(self.scene_analytics, self.alert_manager)
        self.dashboard_manager = DashboardManager(base_dir=self.model_path.parent)
        self._initialize_database()
        self._load_registered_objects()

    def _load_model(self) -> YOLO:
        """Load the YOLOv8 model from the local project path if available."""
        if self.model_path.exists():
            return YOLO(str(self.model_path), task="detect")

        try:
            model = YOLO("yolov8n.pt", task="detect")
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the YOLOv8 Nano model. Make sure the model file exists or the download succeeds."
            ) from exc

        if hasattr(model, "ckpt_path") and model.ckpt_path:
            try:
                shutil.copy2(str(model.ckpt_path), str(self.model_path))
            except Exception:
                pass

        return model

    def _load_clip_model(self, device: str = "cpu") -> Tuple[Any, Any]:
        """Load the OpenAI CLIP model for image embeddings on the selected device."""
        try:
            model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            model.eval()
            return model, preprocess
        except Exception as exc:
            raise RuntimeError("Unable to load the CLIP model. Please install the required dependency.") from exc

    def _connect_db(self) -> sqlite3.Connection:
        """Create a SQLite database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_database(self) -> None:
        """Create the SQLite schema if it does not already exist."""
        try:
            with self._connect_db() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS OBJECTS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        object_name TEXT NOT NULL UNIQUE,
                        object_class TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        date_registered TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        recognition_count INTEGER NOT NULL DEFAULT 0,
                        notes TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS RECOGNITION_HISTORY (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        object_id INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        track_id INTEGER,
                        FOREIGN KEY(object_id) REFERENCES OBJECTS(id) ON DELETE CASCADE
                    )
                    """
                )
        except sqlite3.DatabaseError as exc:
            print(f"Warning: database is corrupted or unreadable: {exc}")
            if self.db_path.exists():
                self.db_path.unlink(missing_ok=True)
            self._initialize_database()

    def _load_registered_objects(self) -> None:
        """Load registered objects from SQLite into memory for fast recognition."""
        self._registered_objects = []
        try:
            with self._connect_db() as conn:
                rows = conn.execute(
                    "SELECT id, object_name, object_class, embedding, date_registered, last_seen, recognition_count, notes FROM OBJECTS ORDER BY id"
                ).fetchall()
                for row in rows:
                    self._registered_objects.append(
                        {
                            "id": row["id"],
                            "object_name": row["object_name"],
                            "object_class": row["object_class"],
                            "embedding": self._deserialize_embedding(row["embedding"]),
                            "date_registered": row["date_registered"],
                            "last_seen": row["last_seen"],
                            "recognition_count": row["recognition_count"],
                            "notes": row["notes"],
                        }
                    )
        except sqlite3.Error as exc:
            print(f"Warning: failed to read registered objects: {exc}")
            self._registered_objects = []

    def _serialize_embedding(self, embedding: np.ndarray) -> str:
        """Convert an embedding array into a JSON string for SQLite storage."""
        return json.dumps(embedding.astype(np.float32).tolist())

    def _deserialize_embedding(self, embedding_text: str) -> np.ndarray:
        """Convert a stored JSON embedding back into a numpy array."""
        try:
            values = json.loads(embedding_text)
            return np.asarray(values, dtype=np.float32)
        except (TypeError, json.JSONDecodeError):
            return np.zeros(512, dtype=np.float32)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a CLIP embedding to unit length."""
        embedding_array = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            return embedding_array
        return embedding_array / norm

    def _embed_image(self, frame: Any, bounding_box: List[int]) -> Optional[np.ndarray]:
        """Create a CLIP embedding for an image crop from the given bounding box."""
        if self.clip_model is None or self.clip_preprocess is None:
            self.clip_model, self.clip_preprocess = self._load_clip_model(self.device)

        x1, y1, x2, y2 = [max(0, int(value)) for value in bounding_box]
        height, width = frame.shape[:2]
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image_input = self.clip_preprocess(image).unsqueeze(0).to("cpu")
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input).cpu().numpy().reshape(-1)
            return self._normalize_embedding(embedding)
        except Exception as exc:
            print(f"Warning: failed to generate CLIP embedding: {exc}")
            return None

    def _cosine_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        return float(np.dot(self._normalize_embedding(embedding_a), self._normalize_embedding(embedding_b)))

    def _recognize_embedding(self, embedding: Optional[np.ndarray]) -> Dict[str, Any]:
        """Compare an embedding against the registered objects and return recognition info."""
        if embedding is None or not self._registered_objects:
            return {
                "recognition_status": "Unknown",
                "object_name": "Unknown",
                "similarity_score": 0.0,
                "object_id": None,
            }

        best_match: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for entry in self._registered_objects:
            try:
                stored_embedding = entry.get("embedding", np.zeros(512, dtype=np.float32))
                similarity = self._cosine_similarity(embedding, stored_embedding)
            except Exception:
                continue

            if similarity > best_score:
                best_score = similarity
                best_match = entry

        if best_match is not None and best_score >= self.recognition_threshold:
            return {
                "recognition_status": "Known",
                "object_name": best_match.get("object_name", "Unknown"),
                "similarity_score": round(float(best_score), 3),
                "object_id": best_match.get("id"),
            }

        return {
            "recognition_status": "Unknown",
            "object_name": "Unknown",
            "similarity_score": round(float(best_score), 3),
            "object_id": None,
        }

    def _update_recognition_stats(self, object_id: int, confidence: float, track_id: Optional[int]) -> None:
        """Update recognition statistics and append to history only when necessary."""
        if object_id is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._connect_db() as conn:
                conn.execute(
                    "UPDATE OBJECTS SET last_seen = ?, recognition_count = recognition_count + 1 WHERE id = ?",
                    (timestamp, object_id),
                )
                conn.execute(
                    "INSERT INTO RECOGNITION_HISTORY (object_id, timestamp, confidence, track_id) VALUES (?, ?, ?, ?)",
                    (object_id, timestamp, confidence, track_id),
                )
        except sqlite3.Error as exc:
            print(f"Warning: failed to update recognition history: {exc}")
            return

        for entry in self._registered_objects:
            if entry.get("id") == object_id:
                entry["last_seen"] = timestamp
                entry["recognition_count"] = int(entry.get("recognition_count", 0)) + 1
                break

    def _get_cached_embedding(self, track_id: Optional[int], frame: Any, bounding_box: List[int]) -> Optional[np.ndarray]:
        """Reuse cached embeddings for stable track IDs and recompute only when needed."""
        if track_id is None:
            return self._embed_image(frame, bounding_box)

        cached = self.embedding_cache.get(track_id)
        if cached is None:
            embedding = self._embed_image(frame, bounding_box)
            if embedding is not None:
                self.embedding_cache[track_id] = {"embedding": embedding, "bounding_box": bounding_box}
            return embedding

        previous_box = cached.get("bounding_box", [])
        previous_area = max(1, abs(previous_box[2] - previous_box[0]) * abs(previous_box[3] - previous_box[1]))
        current_area = max(1, abs(bounding_box[2] - bounding_box[0]) * abs(bounding_box[3] - bounding_box[1]))
        area_change = abs(current_area - previous_area) / previous_area
        if area_change > 0.15:
            embedding = self._embed_image(frame, bounding_box)
            if embedding is not None:
                self.embedding_cache[track_id] = {"embedding": embedding, "bounding_box": bounding_box}
            return embedding

        return cached.get("embedding")

    def _build_predict_kwargs(self) -> Dict[str, Any]:
        """Build YOLO predict kwargs compatible with the installed Ultralytics version."""
        return {
            "stream": False,
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "device": self.device,
            "tracker": self.tracker_config,
            "max_det": self.max_detections,
            "agnostic_nms": True,
            "iou": 0.35,
            "half": self.device != "cpu",
        }

    def detect(self, frame: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        """Run inference on a frame and return annotated output plus tracking and recognition detections."""
        if frame is None:
            raise ValueError("Frame cannot be None")

        results = self.model(frame, **self._build_predict_kwargs())
        result = results[0]
        annotated_frame = frame.copy()
        detections: List[Dict[str, Any]] = []

        for box in getattr(result, "boxes", []):
            if box is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else -1
            confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
            class_name = self.class_names.get(class_id, "object")

            track_id = None
            if getattr(box, "id", None) is not None:
                track_ids = box.id.cpu().tolist()
                if track_ids:
                    track_id = int(track_ids[0])

            bounding_box = [int(x1), int(y1), int(x2), int(y2)]
            center_x = int(round((x1 + x2) / 2))
            center_y = int(round((y1 + y2) / 2))
            recognition = {
                "recognition_status": "Unknown",
                "object_name": "Unknown",
                "similarity_score": 0.0,
                "object_id": None,
            }
            should_run_recognition = bool(self._registered_objects) and confidence >= max(self.confidence, 0.35)
            if should_run_recognition:
                embedding = self._get_cached_embedding(track_id, frame, bounding_box)
                recognition = self._recognize_embedding(embedding)

            object_id = recognition.get("object_id")
            if recognition["recognition_status"] == "Known" and track_id is not None and object_id is not None:
                object_id_int = int(object_id)
                previous_state = self.recognition_state.get(track_id)
                if previous_state is None or previous_state.get("object_id") != object_id_int:
                    self._update_recognition_stats(object_id_int, confidence, track_id)
                    self.recognition_state[track_id] = {
                        "object_id": object_id_int,
                        "object_name": recognition.get("object_name"),
                    }

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bounding_box": bounding_box,
                    "center_point": [center_x, center_y],
                    "recognition_status": recognition["recognition_status"],
                    "object_name": recognition["object_name"],
                    "similarity_score": recognition["similarity_score"],
                }
            )

            color = (0, 255, 0)
            if recognition["recognition_status"] == "Unknown":
                color = (0, 165, 255)

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            recognized_name = recognition.get("object_name")
            display_name = build_detection_label(
                class_name,
                recognized_name if isinstance(recognized_name, str) else None,
                track_id,
                confidence,
            )
            cv2.putText(
                annotated_frame,
                display_name,
                (int(x1), max(16, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            if track_id is not None:
                cv2.putText(
                    annotated_frame,
                    f"Track #{track_id}",
                    (int(x1), max(40, int(y1) + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    annotated_frame,
                    f"{confidence * 100:.0f}%",
                    (int(x1), max(40, int(y1) + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)

        self.scene_analytics.update_tracks(detections, annotated_frame.shape)
        self.scene_analytics.draw(annotated_frame)
        self.alert_rules.evaluate(frame, annotated_frame)
        summary = self.scene_analytics.get_summary()
        summary["fps"] = 0
        self.dashboard_manager.update_dashboard(
            summary=summary,
            detections=detections,
            registered_objects=self.get_registered_objects(),
            alerts=self.alert_manager.get_alerts(),
            frame_shape=annotated_frame.shape,
        )
        self.dashboard_manager.draw_heatmap(annotated_frame)
        return annotated_frame, detections

    def register_object_from_detection(self, frame: Any, detection: Dict[str, Any], object_name: str, notes: str = "") -> Tuple[bool, str]:
        """Create a new registry entry from a tracked detection using the selected crop embedding."""
        if not object_name or not object_name.strip():
            return False, "Object name cannot be empty."

        bounding_box = detection.get("bounding_box")
        if not bounding_box:
            return False, "No valid bounding box selected."

        embedding = self._embed_image(frame, bounding_box)
        if embedding is None:
            return False, "Unable to create embedding for the selected object."

        normalized_name = object_name.strip()
        for entry in self._registered_objects:
            if entry.get("object_name", "").lower() == normalized_name.lower():
                return False, "A registered object with that name already exists."

        object_class = detection.get("class_name", "object")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._connect_db() as conn:
                cursor = conn.execute(
                    "INSERT INTO OBJECTS (object_name, object_class, embedding, date_registered, last_seen, recognition_count, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        normalized_name,
                        object_class,
                        self._serialize_embedding(embedding),
                        timestamp,
                        timestamp,
                        0,
                        notes.strip(),
                    ),
                )
                object_id = cursor.lastrowid
        except sqlite3.Error as exc:
            return False, f"Failed to save object: {exc}"

        self._registered_objects.append(
            {
                "id": object_id,
                "object_name": normalized_name,
                "object_class": object_class,
                "embedding": embedding,
                "date_registered": timestamp,
                "last_seen": timestamp,
                "recognition_count": 0,
                "notes": notes.strip(),
            }
        )
        return True, f"Registered '{normalized_name}' successfully."

    def add_object(self, object_name: str, object_class: str, embedding: np.ndarray, notes: str = "") -> Tuple[bool, str]:
        """Insert a new object into the SQLite database."""
        if not object_name or not object_name.strip():
            return False, "Object name cannot be empty."

        normalized_name = object_name.strip()
        for entry in self._registered_objects:
            if entry.get("object_name", "").lower() == normalized_name.lower():
                return False, "A registered object with that name already exists."

        if embedding is None:
            return False, "Unable to create embedding for the selected object."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._connect_db() as conn:
                cursor = conn.execute(
                    "INSERT INTO OBJECTS (object_name, object_class, embedding, date_registered, last_seen, recognition_count, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        normalized_name,
                        object_class,
                        self._serialize_embedding(embedding),
                        timestamp,
                        timestamp,
                        0,
                        notes.strip(),
                    ),
                )
                object_id = cursor.lastrowid
        except sqlite3.Error as exc:
            return False, f"Failed to save object: {exc}"

        self._registered_objects.append(
            {
                "id": object_id,
                "object_name": normalized_name,
                "object_class": object_class,
                "embedding": embedding,
                "date_registered": timestamp,
                "last_seen": timestamp,
                "recognition_count": 0,
                "notes": notes.strip(),
            }
        )
        return True, f"Registered '{normalized_name}' successfully."

    def edit_object(self, object_id: int, object_name: str, notes: str = "") -> Tuple[bool, str]:
        """Edit an existing object in the database."""
        if not object_name or not object_name.strip():
            return False, "Object name cannot be empty."

        normalized_name = object_name.strip()
        for entry in self._registered_objects:
            if entry.get("id") != object_id and entry.get("object_name", "").lower() == normalized_name.lower():
                return False, "A registered object with that name already exists."

        try:
            with self._connect_db() as conn:
                conn.execute(
                    "UPDATE OBJECTS SET object_name = ?, notes = ? WHERE id = ?",
                    (normalized_name, notes.strip(), object_id),
                )
        except sqlite3.Error as exc:
            return False, f"Failed to update object: {exc}"

        for entry in self._registered_objects:
            if entry.get("id") == object_id:
                entry["object_name"] = normalized_name
                entry["notes"] = notes.strip()
                break
        return True, f"Updated '{normalized_name}' successfully."

    def delete_object(self, object_id: int) -> Tuple[bool, str]:
        """Delete an object from the database and memory cache."""
        try:
            with self._connect_db() as conn:
                conn.execute("DELETE FROM OBJECTS WHERE id = ?", (object_id,))
        except sqlite3.Error as exc:
            return False, f"Failed to delete object: {exc}"

        self._registered_objects = [entry for entry in self._registered_objects if entry.get("id") != object_id]
        return True, "Object deleted successfully."

    def get_registered_objects(self) -> List[Dict[str, Any]]:
        """Return the current in-memory list of registered objects."""
        return list(self._registered_objects)

    def search_objects(self, query: str) -> List[Dict[str, Any]]:
        """Search registered objects by name or class."""
        if not query:
            return list(self._registered_objects)

        needle = query.strip().lower()
        matches: List[Dict[str, Any]] = []
        for entry in self._registered_objects:
            if needle in entry.get("object_name", "").lower() or needle in entry.get("object_class", "").lower():
                matches.append(entry)
        return matches

    def get_object_history(self, object_id: int) -> List[Dict[str, Any]]:
        """Return recognition history for a registered object."""
        try:
            with self._connect_db() as conn:
                rows = conn.execute(
                    "SELECT timestamp, confidence, track_id FROM RECOGNITION_HISTORY WHERE object_id = ? ORDER BY id DESC",
                    (object_id,),
                ).fetchall()
                return [
                    {
                        "timestamp": row["timestamp"],
                        "confidence": row["confidence"],
                        "track_id": row["track_id"],
                    }
                    for row in rows
                ]
        except sqlite3.Error as exc:
            print(f"Warning: failed to read object history: {exc}")
            return []
