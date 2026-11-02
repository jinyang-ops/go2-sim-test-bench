"""Reference competition-map geometry from the inspected upstream implementation."""

import math


SEGMENTS = (
    (0.0, 12.0, "flat"),
    (12.0, 24.0, "pyramid_slope"),
    (24.0, 36.0, "rough"),
    (36.0, 48.0, "discrete_obstacles"),
    (48.0, 60.0, "waves"),
    (60.0, 72.0, "stairs_up"),
    (72.0, 84.0, "stairs_down"),
    (84.0, 96.0, "stepping_stones"),
    (96.0, 108.0, "tail_flat"),
)


def segment_at(x):
    if x is None or not math.isfinite(x):
        return "unknown"
    for start, end, name in SEGMENTS:
        if start <= x < end:
            return name
    return "outside_map"
