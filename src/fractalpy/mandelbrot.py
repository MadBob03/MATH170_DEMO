import numpy as np


def generate_mandelbrot(resolution: int, max_iter: int) -> np.ndarray:
    """Generates the Mandelbrot set visualization data.

    The Mandelbrot set is a set of complex numbers $C$ for which the function
    $Z_{n+1} = Z_n^2 + C$ does not diverge to infinity when iterated starting from $Z_0 = 0$.
    Instead, it remains bounded.

    To create the image, we check a grid of points on the complex plane. For each point $C$,
    we repeatedly apply the formula. If the magnitude of $Z$ (its distance from the origin)
    exceeds 2, we know it will escape to infinity, so it's not part of the set. The number
    of iterations it takes to escape determines the color of that point in the image.

    Args:
        resolution: The width and height of the square image in pixels.
        max_iter: The maximum number of times to run the formula for each point.
            Higher values provide more detail at the edges of the set.

    Returns:
        A 2D NumPy array of integers with shape (resolution, resolution).
        Each value represents the number of iterations a point remained bounded
        (magnitude <= 2). Points within the Mandelbrot set will have the value `max_iter`.
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

    # Pre-allocate a buffer for magnitude calculations to avoid creating new arrays in loop
    abs_Z = np.empty(C.shape, dtype=float)

    for _ in range(max_iter):
        # Update Z only for points that haven't escaped: Z = Z^2 + C
        # We use in-place operations to save memory
        np.square(Z, out=Z, where=mask)
        np.add(Z, C, out=Z, where=mask)

        # Check condition for points currently in mask: |Z| <= 2.0
        # Calculate magnitude into pre-allocated buffer
        np.abs(Z, out=abs_Z, where=mask)
        # Update mask in-place based on condition
        np.less_equal(abs_Z, 2.0, out=mask, where=mask)

        # Increment the count for points that remain bounded
        np.add(fractal, 1, out=fractal, where=mask)

    return fractal
