# OCTANE
Optical flow Code for Tracking, Atmospheric motion vector, and Nowcasting Experiments (OCTANE)

## Author
Jason Apke

## Acknowledgements
OCTANE was developed under projects funded by the NASA New Investigators Program Award #80NSSC21K0919 (PI: J. Apke) and Office of Naval Research Award #N00014-21-1-2112 (PI: S. Miller).

## Uses
OCTANE is designed to ingest GOES-R imagery and output both pixel displacements and navigated speeds (in m/s).  The algorithm is useful for retrieving winds from cloud- and water-vapor-drift motions and tracking features at a near-pixel level.  The algorithm does not contain height assignment internally, though cloud-top heights can be injested for use or output within the optical flow methods.

## Dependencies
### Required
- NVIDIA CUDA capable machine with [CUDA](https://developer.nvidia.com/cuda-toolkit) version 9 or greater
- [netcdf](https://www.unidata.ucar.edu/software/netcdf/) and [netcdfcxx4](https://github.com/Unidata/netcdf-cxx4)

## Install

Edit the Makefile in the ./src directory, adding locations of CUDA, Netcdf, and Netcdfc++ libraries.  Also select the GENCODE_FLAGS variable with respect to your specific GPU architecture. Then to build the executable, in the .src/ directory, type:
```
$ make
```
This creates an executable in the ./build directory. Add the build directory to your $PATH environmental variable using setenv or export for easy access to octane commands.  To remove build files, in the ./src directory, type:
```
$ make clean
```

## Running OCTANE
When OCTANE is run without any commands, i.e.:
```
$ octane
```
the system will print a list of command line arguements.  A simple example of running the baseline variational optical flow approach would be
```
$ octane -i1 /path/to/data/GOES/file1.nc -i2 /path/to/data/GOES/file2.nc -alpha 5 -lambda 1 
```

## Development Stage and Plans
OCTANE is in its early stages of development, working in the crowded open-source market for optical flow algorithms.  The first version released deals with deriving motions specifically within GOES-R imagery, with plans to expand to other imagers (or combinations of imagers) and instruments (e.g. GLM, ground-based radars).  Unlike other optical flow collections, algorithms within OCTANE will be specifically designed for Atmospheric and Earth Science applications.  The first version released focuses on deriving dense (every image pixel) Atmospheric Motion Vectors from cloud- and water-vapor-drift motions from satellite imagery in the troposphere, with plans to add new applications related to motion tracking such as temporal interpolation, semi-Lagrangian property tracking, and image stereoscopy.
