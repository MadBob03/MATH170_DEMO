import numpy as np

def generate_julia(resolution: int, c: complex, max_iter: int) -> np.ndarray:
    """
    Generates a Julia set for a given complex parameter c.

    The Julia set is the set of points in the complex plane that do not escape
    to infinity under the iterative mapping Z = Z^2 + c. This function uses
    the escape-time algorithm, returning the number of iterations before
    the magnitude of Z exceeds 2.

    Args:
        resolution (int): The number of points along each axis (width and height) of the square grid.
                          The grid spans from -1.5 to 1.5 in both real and imaginary axes.
        c (complex): The constant parameter for the Julia set formula.
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
        # Calculate magnitude only for points that haven't escaped yet to avoid unnecessary
        # computation and potential warnings/overflows on already escaped points.
        escaped_now = np.zeros(Z.shape, dtype=bool)

        # We use a temporary boolean array for the subset of points
        subset_escaped = np.abs(Z[mask]) > 2

        # Update the main escaped_now mask using the subset
        escaped_now[mask] = subset_escaped

        # Record the iteration number for points that just escaped
        escape_times[escaped_now] = i

        # Update mask to exclude points that have just escaped
        mask[escaped_now] = False

    return escape_times
