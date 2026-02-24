"""Lane counting logic: point-in-polygon, line crossing, deduplication."""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from .config import LaneConfig, Point, Polygon, Line


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Ray-casting algorithm to test if *point* is inside *polygon*."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def signed_distance_to_line(point: Point, line: Line) -> float:
    """Signed perpendicular distance of *point* from *line*.

    Positive on one side, negative on the other.  We use the cross-product
    sign so the result is orientation-dependent (consistent across calls).
    """
    (x1, y1), (x2, y2) = line
    px, py = point
    # Cross product of (line vector) × (point - line start)
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def has_crossed_line(prev_dist: float, curr_dist: float, direction: str) -> bool:
    """Return True if the sign change represents a valid crossing.

    *direction* controls which direction counts:
    - "positive": from negative side to positive side
    - "negative": from positive side to negative side
    - "any": both
    (In the default geometry, "positive" means downward / increasing y.)
    """
    if prev_dist == 0.0 or curr_dist == 0.0:
        return False  # on the line – don't double-count
    crossed = (prev_dist > 0) != (curr_dist > 0)  # sign change
    if not crossed:
        return False
    if direction == "any":
        return True
    if direction == "positive":
        return prev_dist < 0 < curr_dist
    if direction == "negative":
        return prev_dist > 0 > curr_dist
    return True


# ---------------------------------------------------------------------------
# Per-lane counter
# ---------------------------------------------------------------------------


class LaneCounter:
    """Stateful counter for two lanes.

    Call :meth:`update` for every tracked detection in every frame.
    """

    def __init__(self, lane_config: LaneConfig) -> None:
        self.cfg = lane_config
        # track_id → last signed distance to counting line
        self._prev_dist: Dict[int, float] = {}
        # track_id → lane number at previous position (used when crossing is detected)
        self._prev_lane: Dict[int, Optional[int]] = {}
        # track_ids that have already been counted (global deduplication)
        self._counted: Set[int] = set()
        self.lane1_count: int = 0
        self.lane2_count: int = 0

    @property
    def total(self) -> int:
        return self.lane1_count + self.lane2_count

    def _lane_of(self, pt: Point) -> Optional[int]:
        if point_in_polygon(pt, self.cfg.lane1_polygon):
            return 1
        if point_in_polygon(pt, self.cfg.lane2_polygon):
            return 2
        return None

    def update(
        self,
        track_id: int,
        bbox_bottom_center: Point,
        class_name: str,
    ) -> Optional[int]:
        """Process one tracked object.

        Returns the lane number (1 or 2) if this update caused a new count,
        otherwise returns None.

        The lane assigned to a crossing is determined by where the vehicle
        *was* before crossing (its previous lane), so that e.g. a vehicle
        coming from lane 1 is counted as a lane-1 crossing even though it
        momentarily enters lane 2 as it crosses the line.
        """
        pt = bbox_bottom_center
        curr_dist = signed_distance_to_line(pt, self.cfg.counting_line)
        curr_lane = self._lane_of(pt)

        new_count: Optional[int] = None

        if track_id in self._prev_dist:
            prev_dist = self._prev_dist[track_id]
            prev_lane = self._prev_lane.get(track_id)
            if (
                prev_lane is not None
                and has_crossed_line(prev_dist, curr_dist, self.cfg.direction)
            ):
                if track_id not in self._counted:
                    self._counted.add(track_id)
                    if prev_lane == 1:
                        self.lane1_count += 1
                    else:
                        self.lane2_count += 1
                    new_count = prev_lane

        self._prev_dist[track_id] = curr_dist
        self._prev_lane[track_id] = curr_lane
        return new_count

    def reset(self) -> None:
        self._prev_dist.clear()
        self._prev_lane.clear()
        self._counted.clear()
        self.lane1_count = 0
        self.lane2_count = 0
