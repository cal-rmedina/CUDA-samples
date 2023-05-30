# Factorial on GPU

Program to compute the [factorial](https://en.wikipedia.org/wiki/Factorial)
$n!$ of an integer number.

## Features

Parallel CUDA/C proof of concept code made with the purpose of explaining
different [CUDA](https://developer.nvidia.com/cuda-toolkit) concepts like:

- Kernel calls and launch
- Recursive device functions
- CUDA Streams; concurrency during kernel execution

For more details about each concept, refer to the [CUDA
Documentation](https://docs.nvidia.com/cuda/)

## Documentation

Explicit documentation for the whole code has been made using
[Doxygen](https://doxygen.nl/), each file includes comments with the  Doxygen
syntax, taken by Doxygen once the documentation file is generated. For more
details, go to the source code and check comments with the following structure,
as headers of each file and also after each function.

```
/** @file factorialGPU.h
 *
 * @brief       File with the kernel & kernel call computing \f$n!\f$.
 */

```

The documentation includes files structure, tree libraries and dependencies, as
well as detailed functions and kernel explanation about the algorithms and
implementation used, furthermore, a file `Doxyfile` is included as a
configuration file to determine all of its settings.

It is important to mention that the documentation file **is not included** in
the repository, it should be created running Doxygen as follows:

```
$doxygen Doxyfile
```
Doxygen should be installed on your computer before running the previous
command, refer to the [Doxygen
Manual](https://www.doxygen.nl/manual/starting.html) for more details.

Once Doxygen has run and the documentation is created, different directories
are generated, the documentation can be seen opening the file
`html/index.html`.

## Dependencies

To run the code a NVIDIA GPU on your computer is needed, to compile the code,
[nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) must
be installed on your computer, it is included in the [NVIDIA
CUDA-Toolkit](https://developer.nvidia.com/cuda-toolkit).

## Code structure

The code is divided in several files

- <b>Main file</b> (`./main.cu`): Libraries, included files & main program.

```
main(){
  ...
  function calls
  ...
  return 0;
}
```

- **Routines** (`./src/*.h`): Files with the algorithms and functions.

## Compiling

Make file (`Makefile`) can be used to compile the code directly on the terminal; 

### Compiling using Make
```
$make clean; make
```

Makefile should be modified depending on the GPU architecture (`-gencode
arch=compute_XX,code=sm_XX`), for more details and the value of `XX` of your
device, check the details of your target GPU architecture.

### Compiling directly on the terminal
**If you do not know what make does**, compile typing directly on the terminal.

```
$nvcc --default-stream per-thread -Wno-deprecated-declarations -gencode
arch=compute_61,code=compute_61 main.cu -o main.exe
```

The case where `XX=61` targets Pascal architecture GTX10xx series, and the `--default-stream per-thread` should be included to allows kernel execution overlap (CUDA-Streams).

## Testing the code (Macros)

Inside each function there are sections of code dedicated for testing 
each function, those sections have the following format:

```
#ifdef testing_flag_name
  testing code
#endif
```

Lines are omitted if the code is compiled without testing flags, to activate
them, modify `TFLAG` inside `Makefile` including `TFLAG=testing_flag_name`.

### Testing: Including factorial results

For this code there is only one Macro to print the values of $n!$,
to include/omit it uncomment/comment the `TFLAG` directly on the `Makefile`.

If not, add the compiling option with the following format `-D` + `PRINT_FACTORIAL`;
`-DPRINT_FACTORIAL` directly on the terminal

```
$nvcc -DPRINT_FACTORIAL --default-stream per-thread
-Wno-deprecated-declarations -gencode arch=compute_61,code=compute_61 main.cu
-o main.exe
```

## Run the code

Once the code is compiled, a executable file is created `main.exe`, run it
directly on the terminal: `$./main.exe` or with make `$make run`.

### README.md file

This file was created using [Markdown](https://www.markdownguide.org/), a
lightweight markup language for creating formatted text using a plain-text
editor. For syntax go to the
[Guide](https://www.markdownguide.org/basic-syntax/).
