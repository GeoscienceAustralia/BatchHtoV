# BatchHtoV
Batch Horizontal to Vertical Spectrum Ratio and Time Frequency Analysis utility.

Based on the HtoV Toolbox by Krischer: https://github.com/krischer/HtoV-Toolbox

This utility has been extended to implement HVTFA using CWT and Stockwell transforms. Original HVSR processing (multitaper, single taper, etc) has been maintained.

## Requirements

* [mlpy v3.5] is based on NumPy, Scipy and GNU scientific libraries 
* [GSL] version 1.11 or greater
* [stockwell] version 0.0.5 or greater
* [mtspec] version 0.3.2 or greater

[mlpy v3.5]:http://mlpy.sourceforge.net/
[GSL]:https://www.gnu.org/software/gsl/
[Stockwell v0.0.5]:https://github.com/synergetics/stockwell_transform
[mtspec]:http://krischer.github.io/mtspec/

## Installation

### Gadi

#### Load Requisite Modules
* `module purge`
* `module load gsl`
* `module load fftw3/3.3.8`
* `module load python3-as-python`
* `module load openmpi/2.1.6-mt`
* `module load hdf5/1.10.5p`

#### mlpy
* Download [mlpy v3.5]
* `pip3.6 install --global-option=build_ext --global-option="-I/apps/gsl/1.6/include/" --global-option="-L/apps/gsl/2.6/lib/" mlpy-3.5.0.tar.gz --user`

#### stockwell-transform
* `pip3.6 install git+https://github.com/synergetics/stockwell_transform --user`

#### mtspec
* `pip3.6 install mtspec --user`

#### h5py
​
  1. `git clone --single-branch --branch 2.10.0.gadi_tweaks https://github.com/rh-downunder/h5py.git` Pull a branch (based on version 2.10.0) from h5py repository from github fork of h5py, adapted for Gadi, for purpose of custom build
  2. `cd h5py`
  3. `CC=mpicc python setup.py configure --mpi --hdf5=/apps/hdf5/1.10.5p/` Configure with mpi enabled  
  4. `python setup.py build` Build h5py
  5. `python setup.py install --user` Install in user space

#### pyasdf
* `pip3.6 install pyasdf --user`

