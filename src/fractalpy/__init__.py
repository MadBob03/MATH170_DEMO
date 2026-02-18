from .julia import generate_julia
from .mandelbrot import generate_mandelbrot
from .render import colorize_fractal, save_image

__all__ = ["generate_julia", "generate_mandelbrot", "colorize_fractal", "save_image"]
