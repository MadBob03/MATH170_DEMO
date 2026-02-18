import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def colorize_fractal(fractal_array: np.ndarray, colormap_name: str) -> np.ndarray:
    """
    Colorizes a 2D escape-time integer array using a matplotlib colormap.

    Args:
        fractal_array (np.ndarray): 2D array of integer escape times.
        colormap_name (str): Name of the matplotlib colormap to use.

    Returns:
        np.ndarray: 3D RGBA array of shape (height, width, 4) with float values between 0.0 and 1.0.
    """
    # Check for empty array
    if fractal_array.size == 0:
        return np.zeros(fractal_array.shape + (4,), dtype=float)

    max_val = np.max(fractal_array)

    if max_val == 0:
        # If max value is 0, normalization would divide by zero.
        # We can just return the colormap value for 0 for all pixels.
        normalized = np.zeros_like(fractal_array, dtype=float)
    else:
        # Normalize the array
        normalized = fractal_array / max_val

    # Get the colormap
    try:
        cmap = matplotlib.colormaps[colormap_name]
    except KeyError:
        raise ValueError(f"Colormap '{colormap_name}' is not a valid matplotlib colormap.")

    # Apply colormap
    rgba_image = cmap(normalized)

    return rgba_image

def save_image(image_array: np.ndarray, filename: str) -> None:
    """
    Saves the image array to a file.

    Args:
        image_array (np.ndarray): 3D RGBA array of shape (height, width, 4).
        filename (str): The path to save the image to.
    """
    plt.imsave(filename, image_array)
