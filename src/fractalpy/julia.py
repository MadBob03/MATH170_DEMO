import numpy as np

def generate_julia(resolution, c, max_iter):
    """
    Generates a Julia set using NumPy vectorization.

    Args:
        resolution (int): The width and height of the square grid.
        c (complex): The constant parameter for the Julia set formula Z = Z^2 + c.
        max_iter (int): The maximum number of iterations to perform.

    Returns:
        numpy.ndarray: A 2D integer array of shape (resolution, resolution).
                       Values represent the iteration index at which the point escaped (|Z| > 2).
                       Points that did not escape within max_iter have the value max_iter.
    """
    # Create a complex grid from -1.5 to 1.5
    real = np.linspace(-1.5, 1.5, resolution)
    imag = np.linspace(-1.5, 1.5, resolution)

    # meshgrid returns coordinate matrices from coordinate vectors.
    # real (x) varies across columns, imag (y) varies across rows.
    X, Y = np.meshgrid(real, imag)
    Z = X + 1j * Y

    # Initialize escape_times with max_iter (for bounded points)
    escape_times = np.full(Z.shape, max_iter, dtype=int)

    # Mask to track points that have not yet escaped
    mask = np.full(Z.shape, True, dtype=bool)

    for i in range(max_iter):
        # Optimization: stop if all points have escaped
        if not np.any(mask):
            break

        # Update Z for points that haven't escaped
        # We only compute for the mask to avoid overflows on already escaped large numbers,
        # and to save computation time.
        Z[mask] = Z[mask]**2 + c

        # Check for escape condition |Z| > 2
        # Use abs(Z) which computes the magnitude
        escaped_now = (np.abs(Z) > 2) & mask

        # Record the iteration number for points that just escaped
        escape_times[escaped_now] = i

        # Update mask to exclude points that have just escaped
        mask[escaped_now] = False

    return escape_times
