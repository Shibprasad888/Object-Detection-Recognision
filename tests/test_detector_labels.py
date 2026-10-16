from detector import build_detection_label


def test_build_detection_label_prefers_recognized_name() -> None:
    assert build_detection_label("person", "Unknown", None, 0.0) == "human"
    assert build_detection_label("person", "Backpack", 7, 0.92) == "Backpack"


def test_build_detection_label_includes_track_id_when_available() -> None:
    label = build_detection_label("car", "Unknown", 12, 0.81)
    assert label == "car"
