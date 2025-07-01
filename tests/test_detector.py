import numpy as np
from src.utils.detector import detect_lines


def test_detect_lines_basic():
    dem = np.zeros((50, 50))
    dem[25, 10:40] = 10  # horizontal ridge
    lines = detect_lines(dem, pixel_size=1, min_length=5, max_length=100)
    assert len(lines) == 1
    line = lines[0]
    assert line.length >= 20
