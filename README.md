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

## Functionality
### Tweezer
For larger tweezer spacings, there are two sorting functions. Accepting the same set of arguments, both functions sort towards a fully-occupied computational zone.
#### sort_sequentially
This function moves atoms one-by-one towards the computational zone. If enough atoms are present in a plus-shape spanned by the rows and columns of the computational zone, then there are at most as many moves as there are sites in the computational zone. If atoms in the corners have to be used, then some segmented moves may be required.
#### sort_parallel
This function parallelizes atom rearrangement through the use of multiple AOD tones. It works in a greedy fashion by executing the best move it can find at any given moment. Execution time does, at the moment, not scale well above around 1000 target sites. This function can only unfold its true potential if both movement between rows and columns is allowed, which can be configured through the `.hpp` files.
### Lattice
No sequential method exists for lattices as this combination should hardly ever be valuable. If it is required nonetheless, simply reduce the configurable limit to the AOD tones to one.
#### sort_parallel_lattice (under development)
Based on the sort_parallel function and extended for use on lattices, this greedy algorithm uses pathways were atoms may be moved to sort atoms towards an given target geometry. Not all target geometries are possible as all target sites need to be reachable. In its current form, this algorithm's runtime is prohibitive and its use in its current form is not recommended.
#### sort_parallel_lattice_by_row
Imposes a more strict movement scheme than sort_parallel_lattice to alleviate calculation-time concerns. It uses what we call a sorting channel that sweeps over the array and is wide enough to allow for arbitrary movement of atoms along its length. Idea developed together with Francisco Romão.
