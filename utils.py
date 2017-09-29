from PyQt4 import QtCore

import numpy as np
from obspy.core import UTCDateTime
from scipy.signal.signaltools import detrend
from obspy.signal.tf_misfit import cwt
import mlpy.wavelet as wave
import stockwell.smt as smt
from scipy import interpolate
import functools


def toQDateTime(dt):
    """
    Converts a UTCDateTime object to a QDateTime object.
    """
    # Microseconds will get lost because QDateTime does not support them.
    return QtCore.QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                            dt.second, int(dt.microsecond / 1000.0),
                            QtCore.Qt.TimeSpec(1))


def fromQDateTime(dt):
    """
    Converts a QDateTime to a UTCDateTime object.
    """
    # XXX: Microseconds might be lost.
    return UTCDateTime(dt.toPyDateTime())


def getIntervalsInAreas(areas, min_interval=None):
    """
    Searches areas and gets all intervals in it.

    Areas is a list with tuples. The first item is the beginning of the area
    and the second the end. Both in samples.
    """
    # The smallest area will be the interval if none is given.
    if min_interval is None:
        min_interval = min([i[1] - i[0] for i in areas])
        # Choose 99 Percent.
        min_interval *= 0.99
    intervals = []
    # Loop over each area.
    for area in areas:
        # The span of one area.
        span = area[1] - area[0]
        # How many intervals fit in the area?
        count = int(span / min_interval)
        # Determine the rest to be able to distribute the intervals evenly in
        # each area. Remove one just for savety reasons.
        rest = span - count * min_interval - 1
        # Distribute evenly.
        if count > 1:
            space_inbetween = rest // (count - 1)
            for _i in xrange(count):
                start = area[0] + _i * (min_interval + space_inbetween)
                end = start + min_interval
                intervals.append((start, end))
        # Center.
        elif count == 1:
            start = area[0] + rest / 2
            end = start + min_interval
            intervals.append((start, end))
    return intervals


def getAreasWithinThreshold(c_funct, threshold, min_width, feather=0):
    """
    Parses any characteristic function and returns all areas in samples since
    start and length where the values of the function are below (or above) a
    certain threshold.

    :type c_funct: Tuple with two numpy.ndarrays.
    :param c_funct: The first array are the x-values and the second array are
                    the y-values. If it just is one array, then it will
                    be assumed to be the y-values and the x-values will be
                    created using numpy.arange(len(y-values)).
    :type threshold: Integer, Float
    :param threshold: Threshold value.
    :type min_width: Integer
    :param min_width: Minimum width of the returned areas in samples. Any
                      smaller areas will be discarded.
    """
    if type(c_funct) == np.ndarray:
        y_values = c_funct
        x_values = np.arange(len(y_values))
    else:
        x_values = c_funct[0]
        y_values = c_funct[1]
    if len(x_values) != len(y_values):
        raise
    areas = []
    # Init values for loop.
    start = 0
    last = False
    for _i, _j in zip(x_values, y_values):
        if _j < threshold:
            last = True
            continue
        # Already larger than threshold.
        if last:
            if _i - start < min_width:
                start = _i
                last = False
                continue
            areas.append((start + feather, _i - feather))
        start = _i
        last = False
    if last and x_values[-1] - start >= min_width:
        areas.append((start + feather, x_values[-1] - feather))
    return np.array(areas)


def cwt_TFA(data, delta, nf, f_min=0, f_max=0, w0=8, useMlpy=True):
    """
    :param data: time dependent signal.
    :param delta: time step between two samples in data (in seconds)
    :param nf: number of logarithmically spaced frequencies between fmin and
               fmax
    :param f_min: minimum frequency (in Hz)
    :param f_max: maximum frequency (in Hz)
    :param wf: wavelet to use
    :param w0: parameter w0 for morlet wavelets
    :param useMlpy: use the continuous wavelet transform from MLPY by default, otherwise
           use alternative implementation from ObsPy
    :return: 1. time frequency representation of data, type numpy.ndarray of complex
                values, shape = (nf, len(data)).
             2. frequency bins
             3. wavelet scales
    """

    wl = data.shape[0]
    if f_min == 0:
        f_min = 1.0 / wl
    if f_max == 0:
        f_max = 0.4 / delta

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), nf)

    if (useMlpy == False):
        # using cwt from obspy. Note that ObsPy only supports morlet
        # wavelets at present.
        cfs = cwt(data, delta, w0, f_min, f_max, nf)

        fperiods = 1. / freqs
        # using scales_from_fourier from mlpy to compute wavelet scales
        scales = wave.scales_from_fourier(fperiods, 'morlet', w0)

        return cfs, freqs, scales
    else:
        # using morlet cwt from mlpy
        npts = data.shape[0]

        fperiods = 1. / freqs
        scales = wave.scales_from_fourier(fperiods, 'morlet', w0)
        cfs = wave.cwt(data, delta, scales, 'morlet', w0)

        return cfs, freqs, scales
    # end if
# end func

def st_TFA(data, delta, nf, f_min=0, f_max=0):
    cfs = smt.st(data)
    npts = int(data.shape[0])

    # function that maps frequencies to indices
    def st_freq(f):
        return int(np.floor(f * npts / (1. / delta) + .5))
    # end func

    freqsOut = np.logspace(np.log10(f_min), np.log10(f_max), nf)
    indices = map(st_freq, freqsOut)

    return cfs[indices, :], freqsOut
#end func

def single_taper_spectrum(data, delta, taper_name=None):
    """
    Returns the spectrum and the corresponding frequencies for data with the
    given taper.
    """
    length = len(data)
    good_length = length // 2 + 1
    # Create the frequencies.
    # XXX: This might be some kind of hack
    freq = abs(np.fft.fftfreq(length, delta)[:good_length])
    # Create the tapers.
    if taper_name == 'bartlett':
        taper = np.bartlett(length)
    elif taper_name == 'blackman':
        taper = np.blackman(length)
    elif taper_name == 'boxcar':
        taper = np.ones(length)
    elif taper_name == 'hamming':
        taper = np.hamming(length)
    elif taper_name == 'hanning':
        taper = np.hanning(length)
    elif 'kaiser' in taper_name:
        taper = np.kaiser(length, beta=14)
    # Should never happen.
    else:
        msg = 'Something went wrong.'
        raise Exception(msg)
    # Detrend the data.
    data = detrend(data)
    # Apply the taper.
    data *= taper
    spec = abs(np.fft.rfft(data)) ** 2
    return spec, freq
