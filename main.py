from __future__ import annotations

import time

import cv2

from camera_manager import CameraManager
from config_manager import ConfigManager
from detector import ObjectDetector


window_name = "Object Detection & Recognition"


def main() -> None:
    config = ConfigManager()
    try:
        detector = ObjectDetector(device=config.get("device", "cpu"))
    except RuntimeError as exc:
        print(f"Model loading failed: {exc}")
        return

    camera_manager = CameraManager(lambda device: ObjectDetector(device=device), config)
    default_camera = {"name": "Camera 1", "source": 0, "device": config.get("device", "cpu")}
    camera_manager.add_camera(default_camera)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Webcam not available.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    last_detections = []
    registration_pending = False
    edit_pending = False
    delete_pending = False
    search_pending = False
    history_pending = False
    drawing_line = False
    drawing_zone = False
    draw_start = None
    zone_points = []
    selected_track_id = None
    selected_object_id = None

    def mouse_callback(event: int, x: int, y: int, flags: int, _params: object) -> None:
        nonlocal registration_pending, edit_pending, delete_pending, search_pending, history_pending, drawing_line, drawing_zone, draw_start, zone_points, selected_track_id, selected_object_id
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if drawing_line:
            if draw_start is None:
                draw_start = (x, y)
            else:
                detector.scene_analytics.create_line(draw_start, (x, y), f"Line {len(detector.scene_analytics.lines) + 1}")
                draw_start = None
                drawing_line = False
            return

        if drawing_zone:
            zone_points.append((x, y))
            if len(zone_points) >= 3:
                detector.scene_analytics.create_zone(zone_points.copy(), f"Zone {len(detector.scene_analytics.zones) + 1}")
                zone_points = []
                drawing_zone = False
            return

        if registration_pending:
            for detection in last_detections:
                x1, y1, x2, y2 = detection.get("bounding_box", [])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_track_id = detection.get("track_id")
                    registration_pending = False
                    break
        elif edit_pending or delete_pending or search_pending:
            selected_object_id = None
            for entry in detector.get_registered_objects():
                if entry.get("object_name") and len(entry.get("object_name", "")) > 0:
                    pass

    cv2.setMouseCallback(window_name, mouse_callback)

    prev_time = cv2.getTickCount()
    target_fps = 60.0
    frame_period = 1.0 / target_fps
    last_detect_time = 0.0
    last_annotated_frame = None

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("Camera read failed.")
                break

            loop_start = time.perf_counter()
            should_detect = (loop_start - last_detect_time) >= frame_period
            if should_detect:
                annotated_frame, detections = detector.detect(frame)
                last_detections = detections
                last_annotated_frame = annotated_frame
                last_detect_time = loop_start
            else:
                annotated_frame = last_annotated_frame.copy() if last_annotated_frame is not None else frame.copy()
                detections = last_detections

            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(
                annotated_frame,
                f"FPS: {min(target_fps, fps):.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if registration_pending:
                cv2.putText(
                    annotated_frame,
                    "Register mode: click a tracked object",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if drawing_line:
                cv2.putText(
                    annotated_frame,
                    "Draw line: click twice to confirm",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            if drawing_zone:
                cv2.putText(
                    annotated_frame,
                    "Draw zone: click at least 3 points",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if edit_pending:
                cv2.putText(
                    annotated_frame,
                    "Edit mode: use terminal commands",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if delete_pending:
                cv2.putText(
                    annotated_frame,
                    "Delete mode: use terminal commands",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            if search_pending:
                cv2.putText(
                    annotated_frame,
                    "Search mode: use terminal commands",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            summary = detector.scene_analytics.get_summary()
            cv2.putText(annotated_frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Detected: {len(detections)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Tracked: {summary['tracked_objects']}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Registered: {len(detector.get_registered_objects())}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Avg Speed: {summary['average_scene_speed']:.2f} m/s", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Fastest: Track #{summary['fastest_object']}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Safety Violations: {summary['safe_distance_violations']}", (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Predicted Collisions: {summary['predicted_collisions']}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Line IN/OUT/TOTAL: {summary['line_counts']['IN']}/{summary['line_counts']['OUT']}/{summary['line_counts']['TOTAL']}", (10, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Zones: {summary['zone_count']}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            alert_summary = detector.alert_manager.get_summary()
            latest_alert = alert_summary["latest"]
            if latest_alert is not None:
                cv2.putText(annotated_frame, f"Latest Alert: {latest_alert['alert_type']} [{latest_alert['severity']}]", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, latest_alert.get("description", ""), (10, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Alerts: {alert_summary['total']}", (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            dashboard_data = detector.dashboard_manager.analytics_cache
            if dashboard_data:
                cv2.putText(annotated_frame, f"Dashboard: {dashboard_data.get('timestamp', '')}", (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"Heatmap Points: {len(dashboard_data.get('heatmap_points', []))}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if not detections:
                    print("No tracked objects detected right now.")
                else:
                    registration_pending = True
                    edit_pending = False
                    delete_pending = False
                    search_pending = False
                    history_pending = False
                    selected_track_id = None
                    print("Registration mode enabled. Click a tracked object.")
            elif key == ord("e"):
                registration_pending = False
                edit_pending = True
                delete_pending = False
                search_pending = False
                history_pending = False
                print("Edit mode enabled. Enter the object ID to edit.")
            elif key == ord("d"):
                registration_pending = False
                edit_pending = False
                delete_pending = True
                search_pending = False
                history_pending = False
                print("Delete mode enabled. Enter the object ID to delete.")
            elif key == ord("s"):
                registration_pending = False
                edit_pending = False
                delete_pending = False
                search_pending = True
                history_pending = False
                print("Search mode enabled. Enter a search query.")
            elif key == ord("h"):
                registration_pending = False
                edit_pending = False
                delete_pending = False
                search_pending = False
                history_pending = True
                print("History mode enabled. Enter the object ID to inspect.")
            elif key == ord("l"):
                registration_pending = False
                edit_pending = False
                delete_pending = False
                search_pending = False
                history_pending = False
                drawing_line = True
                drawing_zone = False
                draw_start = None
                zone_points = []
                print("Line drawing enabled. Click twice to create a line.")
            elif key == ord("z"):
                registration_pending = False
                edit_pending = False
                delete_pending = False
                search_pending = False
                history_pending = False
                drawing_line = False
                drawing_zone = True
                draw_start = None
                zone_points = []
                print("Zone drawing enabled. Click at least 3 points to create a zone.")
            elif key == ord("t"):
                detector.scene_analytics.toggle_trails()
                print("Trails toggled.")
            elif key == ord("c"):
                detector.scene_analytics.reset_counters()
                print("Counters reset.")
            elif key == ord("p"):
                detector.scene_analytics.toggle_pause()
                print("Playback paused." if detector.scene_analytics.paused else "Playback resumed.")
            elif key == ord("m"):
                detector.scene_analytics.set_calibration(pixel_to_meter_ratio=0.01)
                print("Calibration updated to default pixel-to-meter ratio.")
            elif key == ord("a"):
                detector.alert_manager.set_sounds_enabled(not detector.alert_manager._sounds_enabled)
                print("Alert sounds toggled.")
            elif key == ord("i"):
                detector.dashboard_manager.toggle_heatmap()
                print("Heatmap toggled.")
            elif key == ord("o"):
                detector.dashboard_manager.reset_heatmap()
                print("Heatmap reset.")
            elif key == ord("u"):
                report_path = detector.dashboard_manager.export_report("Daily")
                print(f"Report exported to {report_path}")
            elif key == 27:
                drawing_line = False
                drawing_zone = False
                draw_start = None
                zone_points = []
                print("Drawing cancelled.")

            if selected_track_id is not None:
                matching_detection = next(
                    (item for item in detections if item.get("track_id") == selected_track_id),
                    None,
                )
                if matching_detection is not None:
                    object_name = input("Enter object name: ").strip()
                    notes = input("Optional notes: ").strip()
                    if object_name:
                        success, message = detector.register_object_from_detection(
                            frame=frame,
                            detection=matching_detection,
                            object_name=object_name,
                            notes=notes,
                        )
                        print(message)
                    else:
                        print("Empty object name. Registration cancelled.")
                else:
                    print("Selected object is no longer visible. Registration cancelled.")
                selected_track_id = None

            if edit_pending:
                try:
                    object_id = int(input("Enter object ID to edit: ").strip())
                except ValueError:
                    print("Invalid object ID.")
                    edit_pending = False
                    continue
                object_name = input("Enter new object name: ").strip()
                notes = input("Enter new notes: ").strip()
                success, message = detector.edit_object(object_id, object_name, notes)
                print(message)
                edit_pending = False

            if delete_pending:
                try:
                    object_id = int(input("Enter object ID to delete: ").strip())
                except ValueError:
                    print("Invalid object ID.")
                    delete_pending = False
                    continue
                confirm = input("Delete this object? (y/n): ").strip().lower()
                if confirm in {"y", "yes"}:
                    success, message = detector.delete_object(object_id)
                    print(message)
                else:
                    print("Deletion cancelled.")
                delete_pending = False

            if search_pending:
                query = input("Enter search query: ").strip()
                results = detector.search_objects(query)
                if results:
                    for item in results:
                        print(
                            f"ID={item['id']} Name={item['object_name']} Class={item['object_class']} Registered={item['date_registered']} LastSeen={item['last_seen']} Recognitions={item['recognition_count']}"
                        )
                else:
                    print("No matching objects found.")
                search_pending = False

            if history_pending:
                try:
                    object_id = int(input("Enter object ID for history: ").strip())
                except ValueError:
                    print("Invalid object ID.")
                    history_pending = False
                    continue
                history = detector.get_object_history(object_id)
                if history:
                    for item in history:
                        print(f"{item['timestamp']} | confidence={item['confidence']:.3f} | track_id={item['track_id']}")
                else:
                    print("No history found for that object.")
                history_pending = False
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
