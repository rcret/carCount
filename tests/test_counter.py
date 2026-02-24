"""Unit tests for lane-counting logic."""

from __future__ import annotations

import pytest

from app.counter import LaneCounter, has_crossed_line, point_in_polygon, signed_distance_to_line
from app.config import LaneConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config() -> LaneConfig:
    """Simple 960×720 frame split horizontally at y=360."""
    return LaneConfig(
        lane1_polygon=[(0, 0), (960, 0), (960, 360), (0, 360)],
        lane2_polygon=[(0, 360), (960, 360), (960, 720), (0, 720)],
        counting_line=((0, 360), (960, 360)),
        direction="any",
    )


# ---------------------------------------------------------------------------
# point_in_polygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    def test_point_inside_lane1(self):
        poly = [(0, 0), (960, 0), (960, 360), (0, 360)]
        assert point_in_polygon((480, 180), poly) is True

    def test_point_inside_lane2(self):
        poly = [(0, 360), (960, 360), (960, 720), (0, 720)]
        assert point_in_polygon((480, 540), poly) is True

    def test_point_outside(self):
        poly = [(0, 0), (960, 0), (960, 360), (0, 360)]
        assert point_in_polygon((480, 600), poly) is False

    def test_point_on_boundary_corner(self):
        # A point exactly at a vertex – behaviour may be True or False,
        # but must not raise.
        poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
        result = point_in_polygon((0, 0), poly)
        assert isinstance(result, bool)

    def test_triangle_inside(self):
        poly = [(0, 0), (100, 0), (50, 100)]
        assert point_in_polygon((50, 40), poly) is True

    def test_triangle_outside(self):
        poly = [(0, 0), (100, 0), (50, 100)]
        assert point_in_polygon((90, 90), poly) is False


# ---------------------------------------------------------------------------
# Line crossing
# ---------------------------------------------------------------------------


class TestLineCrossing:
    def test_crosses_any_direction(self):
        assert has_crossed_line(-10.0, 10.0, "any") is True
        assert has_crossed_line(10.0, -10.0, "any") is True

    def test_no_cross_same_side(self):
        assert has_crossed_line(5.0, 10.0, "any") is False
        assert has_crossed_line(-5.0, -10.0, "any") is False

    def test_positive_direction_only(self):
        assert has_crossed_line(-5.0, 5.0, "positive") is True
        assert has_crossed_line(5.0, -5.0, "positive") is False

    def test_negative_direction_only(self):
        assert has_crossed_line(5.0, -5.0, "negative") is True
        assert has_crossed_line(-5.0, 5.0, "negative") is False

    def test_zero_prev_no_count(self):
        assert has_crossed_line(0.0, 10.0, "any") is False

    def test_zero_curr_no_count(self):
        assert has_crossed_line(10.0, 0.0, "any") is False


# ---------------------------------------------------------------------------
# Deduplication – each track_id counted once per crossing
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_each_track_id_counted_once(self):
        cfg = make_config()
        counter = LaneCounter(cfg)

        # Simulate track_id=1 moving from y=200 (lane1, above line)
        # to y=400 (lane1, below line)
        # counting_line is at y=360 across x=0..960
        # signed_distance_to_line at y=200: positive (above line), y=400: negative
        # Let's verify with actual distance calls first.
        d_above = signed_distance_to_line((480, 200), cfg.counting_line)
        d_below = signed_distance_to_line((480, 400), cfg.counting_line)
        # direction="any" so sign change → count
        assert (d_above > 0) != (d_below > 0), "Test setup: sign should change"

        # First update – establishes baseline, no count yet
        result = counter.update(1, (480, 200), "car")
        assert result is None

        # Second update – crosses the line → should count once
        result = counter.update(1, (480, 400), "car")
        assert result == 1
        assert counter.lane1_count == 1

        # Third update – same track_id crosses again → deduplicated, no new count
        result = counter.update(1, (480, 200), "car")
        assert result is None
        result = counter.update(1, (480, 400), "car")
        assert result is None
        assert counter.lane1_count == 1  # still 1

    def test_different_track_ids_counted_separately(self):
        cfg = make_config()
        counter = LaneCounter(cfg)

        for tid in (1, 2, 3):
            counter.update(tid, (480, 200), "car")  # above line
            counter.update(tid, (480, 400), "car")  # crosses → lane1

        assert counter.lane1_count == 3

    def test_lane2_counted_separately(self):
        cfg = make_config()
        counter = LaneCounter(cfg)

        # lane2 is below the line (y=360..720). Track starts at y=500 (below),
        # crosses upward to y=300 (above).
        counter.update(10, (480, 500), "truck")  # below
        result = counter.update(10, (480, 300), "truck")  # cross upward
        assert result == 2
        assert counter.lane2_count == 1
        assert counter.lane1_count == 0

    def test_reset_clears_state(self):
        cfg = make_config()
        counter = LaneCounter(cfg)
        counter.update(1, (480, 200), "car")
        counter.update(1, (480, 400), "car")
        assert counter.lane1_count == 1

        counter.reset()
        assert counter.lane1_count == 0
        assert counter.lane2_count == 0
        assert counter.total == 0
