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

### Raijin

* Download `mlpy`
* `module load gsl/1.15`
* `pip install --global-option=build_ext --global-option="-I/apps/gsl/1.15/include/" --global-option="-L/apps/gsl/1.15/lib/" mlpy-3.5.0.tar.gz --user`
* module load fftw3/3.3.4
* `pip install --install-option="--prefix=$PREFIX_PATH" git+https://github.com/synergetics/stockwell_transform`
* `pip install mtspec`

