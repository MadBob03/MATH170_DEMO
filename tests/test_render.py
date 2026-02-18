import pytest
import numpy as np
from fractalpy.render import colorize_fractal

def test_colorize_fractal_returns_rgba_array():
    """Test that colorize_fractal returns a 3D numpy array of shape (2, 2, 4)."""
    # Given a mock 2x2 numpy integer array
    mock_fractal = np.array([[0, 10], [20, 50]], dtype=int)
    colormap = "viridis"

    # When calling colorize_fractal
    result = colorize_fractal(mock_fractal, colormap)

    # Then it returns a numpy array with shape (2, 2, 4)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2, 4)

def test_colorize_fractal_values_are_valid_floats():
    """Test that output values are valid image data (floats between 0.0 and 1.0)."""
    # Given a mock 2x2 numpy integer array
    mock_fractal = np.array([[0, 10], [20, 50]], dtype=int)
    colormap = "viridis"

    # When calling colorize_fractal
    result = colorize_fractal(mock_fractal, colormap)

    # Then values are valid floats between 0.0 and 1.0
    assert np.issubdtype(result.dtype, np.floating)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
