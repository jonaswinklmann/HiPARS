# HiPARS: Highly-Parallel Atom Rearrangement Sequencer
This project is aimed at providing a solution for rearranging neutral atoms into arbitrary target configurations. At the moment, sequential and parallel movement towards a fully compactified target area is supported.

## Installation
### Linux
- Go to `resorting_cpp` directory
- `make all`
- Go to `resorting_pip` directory
- Check that `resorting_cpp.<python_suffix>.so` exists
- `pip install ./`

or

- run `compileCppAndInstallPip.sh` in `HiPARS` directory

### Windows
- Build HiPARS_VS project
- Go to `resorting_pip` directory
- Check that `resorting_cpp.<python_suffix>.pyd` exists
- `pip install ./`

### Usage 
Look at `scripts` directory for usage examples.
Many configuration options are, at the moment, hidden in the `.hpp` files. This will be changed in the future.

The [documentation](https://resorting.readthedocs.io/en/latest/) contains both the functions of the Python package ([hipars package](https://resorting.readthedocs.io/en/latest/hipars.html)) as well the exposed functions of the underlying C++ library ([resorting_cpp package](https://resorting.readthedocs.io/en/latest/resorting_cpp.html)).
