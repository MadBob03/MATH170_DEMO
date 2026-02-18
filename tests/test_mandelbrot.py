import pytest
import numpy as np
from fractalpy.mandelbrot import generate_mandelbrot

def test_mandelbrot_structure():
    resolution = 100
    max_iter = 50
    result = generate_mandelbrot(resolution, max_iter)
    assert isinstance(result, np.ndarray)
    assert result.shape == (resolution, resolution)
    assert np.issubdtype(result.dtype, np.integer)

def test_mandelbrot_center():
    resolution = 101 # Odd resolution ensures center is exactly one pixel
    max_iter = 100
    result = generate_mandelbrot(resolution, max_iter)
    center_idx = resolution // 2
    # The center pixel should correspond to 0+0j which is inside the set
    assert result[center_idx, center_idx] == max_iter
