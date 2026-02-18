import numpy as np

def generate_mandelbrot(resolution, max_iter):
    """
    Generates a Mandelbrot set for the region [-2.0, 2.0] for both real and imaginary parts.

    Args:
        resolution (int): The number of points along each axis (width and height).
        max_iter (int): The maximum number of iterations.

    Returns:
        numpy.ndarray: A 2D array of shape (resolution, resolution) containing the
                       iteration count at which the point escaped. Points in the set
                       will have the value max_iter.
    """
    # Create a grid of complex numbers
    # We use linspace to create evenly spaced points
    x = np.linspace(-2.0, 2.0, resolution)
    y = np.linspace(-2.0, 2.0, resolution)

    # Create a meshgrid
    # X corresponds to columns (real part), Y corresponds to rows (imaginary part)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Initialize Z and the iteration count array (fractal)
    Z = np.zeros_like(C)
    fractal = np.zeros(C.shape, dtype=int)

    # Iterate
    # We use a mask to track points that have not escaped yet.
    # Initially all points are considered 'in' (mask = True).
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        # Update Z only for points that haven't escaped
        Z[mask] = Z[mask]**2 + C[mask]

        # Check condition for points currently in mask
        # We only need to check the points that were processed
        # This updates the mask in-place for the True values
        mask[mask] = np.abs(Z[mask]) <= 2.0

        # Increment the count for points that haven't escaped yet
        fractal += mask

    return fractal
