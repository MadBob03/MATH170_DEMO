import pytest
import numpy as np
from fractalpy.julia import generate_julia

def test_julia_structure():
    """Test that the function returns a valid numpy array with correct shape and type."""
    resolution = 100
    c = 0+0j
    max_iter = 50
    result = generate_julia(resolution, c, max_iter)

    assert isinstance(result, np.ndarray)
    assert result.shape == (resolution, resolution)
    assert np.issubdtype(result.dtype, np.integer)

def test_julia_unit_circle():
    """
    Test for the unit circle case (c = 0+0j).
    Points inside the unit circle (|z| < 1) should stay bounded.
    Points outside the unit circle (|z| > 1) should escape.

    Assumes coordinate system is centered at 0+0j with range [-2, 2] in both axes,
    consistent with conventions in existing tests.
    """
    resolution = 1000
    c = 0+0j
    max_iter = 100

    # Calculate indices based on assumed coordinate system [-2, 2]
    # Width = 4.0
    # Pixel size = 4.0 / resolution
    pixel_scale = 4.0 / resolution
    center_idx = resolution // 2

    # Index for point at radius 0.5 (should be bounded)
    # 0.5 distance from center
    offset_0_5 = int(0.5 / pixel_scale)
    idx_0_5 = center_idx + offset_0_5

    # Index for point at radius 1.5 (should escape)
    # 1.5 distance from center
    offset_1_5 = int(1.5 / pixel_scale)
    idx_1_5 = center_idx + offset_1_5

    result = generate_julia(resolution, c, max_iter)

    # Check bounded point (radius 0.5)
    # We check along the x-axis (imaginary part 0)
    assert result[center_idx, idx_0_5] == max_iter, \
        f"Point at radius 0.5 (index {idx_0_5}) should be bounded (value {max_iter})"

    # Check escaping point (radius 1.5)
    # We check along the x-axis (imaginary part 0)
    assert result[center_idx, idx_1_5] < max_iter, \
        f"Point at radius 1.5 (index {idx_1_5}) should escape (value < {max_iter})"
