# OCTANE
Optical Flow toolkit for Atmospheric and Earth Sciences (OCTANE)

## Author
Jason Apke

## Uses
OCTANE is designed to ingest GOES-R imagery and output both pixel displacements and navigated speeds (in m/s).  The algorithm is useful for retrieving winds from cloud- and water-vapor-drift motions and tracking features at a near-pixel level.

## Dependencies
Required
- NVIDIA CUDA capable machine with [CUDA](https://developer.nvidia.com/cuda-toolkit) version 9 or greater
- [netcdf](https://www.unidata.ucar.edu/software/netcdf/) and [netcdfcxx4](https://github.com/Unidata/netcdf-cxx4)

## Install

Edit the Makefile in the ./src directory, adding locations of CUDA, Netcdf, and Netcdfc++ libraries.  Also select the GENCODE_FLAGS variable with respect to your specific GPU architecture. Then to build the executable, in the .src/ directory, type:
```
make
```
This creates an executable in the ./build directory. To remove build files, in the ./src directory, type:
```
make clean
```

## Development Stage and Plans
OCTANE is in its early stages of development, working in the crowded open-source market for optical flow algorithms.  The first version released deals with deriving motions specifically within GOES-R imagery, with plans to expand to other imagers (or combinations of imagers) and instruments (e.g. GLM, ground-based radars).  Unlike other optical flow collections, algorithms within OCTANE will be specifically designed for Atmospheric and Earth Science applications.  The first version released focuses on deriving dense (every image pixel) Atmospheric Motion Vectors from cloud- and water-vapor-drift motions from satellite imagery in the troposphere, with plans to add new applications related to motion tracking such as temporal interpolation, semi-Lagrangian property tracking, and image stereoscopy.
