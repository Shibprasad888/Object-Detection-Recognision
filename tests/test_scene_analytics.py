from scene_analytics import direction_from_delta, point_in_polygon, point_to_line_side


def test_point_in_polygon_detects_inside_and_outside_points() -> None:
    polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]

    assert point_in_polygon((2, 2), polygon) is True
    assert point_in_polygon((5, 5), polygon) is False


def test_point_to_line_side_detects_opposite_sides() -> None:
    line_start = (0, 0)
    line_end = (4, 0)

    assert point_to_line_side((1, 1), line_start, line_end) > 0
    assert point_to_line_side((1, -1), line_start, line_end) < 0


def test_direction_from_delta_handles_cardinal_motion() -> None:
    assert direction_from_delta(5, 0) == "E"
    assert direction_from_delta(0, -5) == "N"
    assert direction_from_delta(-3, 4) == "NW"
