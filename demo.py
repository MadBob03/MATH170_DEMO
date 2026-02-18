import fractalpy
import matplotlib.pyplot as plt

def main():
    resolution = 2000
    max_iter = 100 # Default is usually enough for a demo

    print("Generating Mandelbrot set...")
    # generate_mandelbrot(resolution: int, max_iter: int)
    mandelbrot = fractalpy.generate_mandelbrot(resolution, max_iter)

    print("Generating Julia set...")
    julia_c = -0.8 + 0.156j
    # generate_julia(resolution: int, c: complex, max_iter: int)
    julia = fractalpy.generate_julia(resolution, julia_c, max_iter)

    print("Colorizing Mandelbrot set...")
    mandelbrot_img = fractalpy.colorize_fractal(mandelbrot, 'magma')

    print("Colorizing Julia set...")
    julia_img = fractalpy.colorize_fractal(julia, 'twilight_shifted')

    print("Saving Mandelbrot image...")
    fractalpy.save_image(mandelbrot_img, 'mandelbrot.png')

    print("Saving Julia image...")
    fractalpy.save_image(julia_img, 'julia.png')

    print("Displaying Mandelbrot set...")
    plt.imshow(mandelbrot_img)
    plt.title("Mandelbrot Set")
    plt.axis('off')
    plt.show()

    print("Displaying Julia set...")
    plt.imshow(julia_img)
    plt.title(f"Julia Set (c = {julia_c})")
    plt.axis('off')
    plt.show()

    print("Demo complete!")

if __name__ == "__main__":
    main()
