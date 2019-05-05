# BatchHtoV
Batch Horizontal to Vertical Spectrum Ratio and Time Frequency Analysis utility.

Based on the HtoV Toolbox by Krischer: https://github.com/krischer/HtoV-Toolbox

This utility has been extended to implement HVTFA using CWT and Stockwell transforms. Original HVSR processing (multitaper, single taper, etc) has been maintained.

## Requirements

* [mlpy v3.5] is based on NumPy, Scipy and GNU scientific libraries 
* [GSL] version 1.11 or greater

[mlpy v3.5]:http://mlpy.sourceforge.net/
[GSL]:https://www.gnu.org/software/gsl/

## Installation

### Raijin

* Download `mlpy`
* `module load gsl/1.15`
* `pip install --global-option=build_ext --global-option="-I/apps/gsl/1.15/include/" --global-option="-L/apps/gsl/1.15/lib/" mlpy-3.5.0.tar.gz --user`
