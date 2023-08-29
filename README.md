# Basic CUDA samples (Main README file)

Set of programs to teach the basics of CUDA programming for the scientific
community. Each directory is independent, self contained and could run by
itself.

## Program structure

Each directory, labeled `XX_name`, contains an independent code to check
different CUDA implementations, library usage, concepts, etc.

Additionally each directory has its own:

- A `READMEXX.md` file for each code
- A `Doxyfile` used by Doxygen to generate documentation
- A `Makefile` to compile using make
- Additional $\LaTeX$ file (if more info about the algorithm is needed)

## List of programs and its CUDA features to highlight

1. Factorial: Recursive device functions, basic kernel usage and Streams.
4. Pi Monte-Carlo: CURAND library, Macros for checking error and reduction.
5. Mandelbrot set: 2D kernels, cudaMalloc.
6. Square sum: Streams.
