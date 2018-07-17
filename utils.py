#from PyQt4 import QtCore
import math
import numpy as np
from obspy.core import UTCDateTime
from scipy.signal.signaltools import detrend
from obspy.signal.tf_misfit import cwt
import mlpy.wavelet as wave
import stockwell.smt as smt
from scipy import interpolate
import functools
import scipy


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

def icwt_band(tfa, delta, freq, w0):
    fperiods = np.array([1/freq])
    scales = wave.scales_from_fourier(fperiods, 'morlet', w0)
    data = wave.cwt(tfa, delta, scales, 'morlet', w0)
    return data

from scipy.signal import cheby1, lfilter

def cheby_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, 5, [low,high], 'band', analog=True)
    return b, a
 
def cheby_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = cheby_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def zero_phase_cheby_bpf(data, fc, width, fs, order=4):
    forward = cheby_bandpass_filter(data, fc - 0.5 * width, fc + 0.5 * width, fs, order/2)
    reverse = np.flipud(forward)
    return cheby_bandpass_filter(reverse, fc - 0.5 * width, fc + 0.5 * width, fs, order/2)

def get_bands(data, delta, nf, f_min=0, f_max=0, freq_spacing='log', freqs=None):
    wl = data.shape[0]
    if f_min == 0:
        f_min = 1.0 / wl
    if f_max == 0:
        f_max = 0.4 / delta
    
    no_freqs = False
    fs = 1.0/delta
    #print "sample rate = " + str(fs)

    if type(freqs) is not np.ndarray:
        no_freqs = True
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), nf)

    if no_freqs and freq_spacing=='linear':
        oldnf = nf
        sr = 1.0/delta
        spacing = sr/wl
        f_min_idx = math.floor(f_min / spacing)
        f_max_idx = math.ceil(f_max / spacing)
        nf = f_max_idx - f_min_idx + 1
        f_min = f_min_idx + spacing
        f_max = f_max_idx + spacing
        freqs = np.linspace(f_min_idx, f_max_idx, nf) * spacing

    # matrix of time series filtered by each band in freqs
    cfs = np.tile(data,(freqs.shape[0],1))
    widths = np.zeros(freqs.shape[0])
    widths[1:-1] = 0.5 * ((freqs[2:] - freqs[1:-1]) + (freqs[1:-1] - freqs[:-2]))
    widths[0] = widths[1]
    widths[-1] = widths[-2]
    for i in xrange(freqs.shape[0]):
        cfs[i,:] = zero_phase_cheby_bpf(cfs[i,:], freqs[i], widths[i], fs)

    return cfs, freqs

def windowedSincFilterKernelLength(dt,b):
	bb = b * dt
	N = int(np.ceil((4 / bb)))
	if not N % 2: N += 1  # Make sure that N is odd.
	return N


def windowedSincFilterKernel(f,dt, b=0.5):
	#print "dt = " + str(dt)
	fL = f * dt * 2
	fH = fL
	N = windowedSincFilterKernelLength(dt,b)
	n = np.arange(N)
	 
	# low-pass filter
	hlpf = np.sinc(fH * (n - (N - 1) / 2.))
	hlpf *= np.blackman(N)
	hlpf = hlpf / np.sum(hlpf)
	 
	# high-pass filter 
	hhpf = np.sinc(fL * (n - (N - 1) / 2.))
	hhpf *= np.blackman(N)
	hhpf = hhpf / np.sum(hhpf)
	hhpf = -hhpf
	hhpf[(N - 1) / 2] += 1
	 
	h = np.convolve(hlpf, hhpf)
	return h

def get_ws_bands(data, delta, nf, f_min=0, f_max=0, freq_spacing='log', freqs=None):
    wl = data.shape[0]
    if f_min == 0:
        f_min = 1.0 / wl
    if f_max == 0:
        f_max = 0.4 / delta
    
    no_freqs = False
    fs = 1.0/delta
    #print "sample rate = " + str(fs)

    if type(freqs) is not np.ndarray:
        no_freqs = True
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), nf)

    if no_freqs and freq_spacing=='linear':
        oldnf = nf
        sr = 1.0/delta
        spacing = sr/wl
        f_min_idx = math.floor(f_min / spacing)
        f_max_idx = math.ceil(f_max / spacing)
        nf = f_max_idx - f_min_idx + 1
        f_min = f_min_idx + spacing
        f_max = f_max_idx + spacing
        freqs = np.linspace(f_min_idx, f_max_idx, nf) * spacing

    # matrix of time series filtered by each band in freqs
    cfs = np.tile(data,(freqs.shape[0],1))

    # zero-pad to nearest power of 2
    # FIXME we are not accounting for the full convolution in this padded length.
    # TODO pre-generate convolution filters based on wl and freq array. This will determine zpadlen.
    #      pre-calc fft of these kernels padded to zpadlen
    #      pre-calc fft of data. Since we have one b only, this is one FFT thus we are saving time.
    #      zpadlen should be computed for each filter to make convolved ifft size a power of 2
    b = 0.5
    N = windowedSincFilterKernelLength(delta,b)
    # kernel is bpf so it will be 2N-1
    zpadlen = int(2 ** math.ceil(math.log(wl+(2*N-1)-1,2)))
    # Detrend the data.
    dataz = np.zeros(zpadlen)
    dataz[:wl] = detrend(data)
    #good_wl = wl // 2 + 1
    good_wl = zpadlen // 2 + 1
    # Create the frequencies. Not used.
    ffreq = abs(np.fft.fftfreq(zpadlen, delta)[:good_wl])
    spec = np.fft.rfft(dataz)

    for i in xrange(freqs.shape[0]):
	h = np.zeros(zpadlen)
	hh = windowedSincFilterKernel(freqs[i],delta,b)
	hl = hh.shape[0]
	h[:hl] = hh
	off = int((hl - 1) / 2)
	#print "wl = " + str(wl)
	#print "off = " + str(off)
	#print "zpadlen = " + str(zpadlen)
        #r = np.fft.irfft(spec*np.fft.rfft(h))
	#print "r len = " + str(r.shape)
        #cfs[i,:] = r[off:wl+off]
        cfs[i,:] = np.fft.irfft(spec*np.fft.rfft(h))[off:wl+off]
        #print "Computed bpf signal for frequency " + str(freqs[i])

    return cfs, freqs

def cwt_TFA(data, delta, nf, f_min=0, f_max=0, w0=8, useMlpy=True, freq_spacing='log', freqs=None):
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

    no_freqs = False

    if type(freqs) is not np.ndarray:
        no_freqs = True
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
        # spacing option only applies to mlpy
        if no_freqs and freq_spacing=='linear':
            oldnf = nf
            sr = 1.0/delta
            spacing = sr/wl
            f_min_idx = math.floor(f_min / spacing)
            f_max_idx = math.ceil(f_max / spacing)
            nf = f_max_idx - f_min_idx + 1
            f_min = f_min_idx + spacing
            f_max = f_max_idx + spacing
            freqs = np.linspace(f_min_idx, f_max_idx, nf) * spacing

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

def single_taper_spectrum(data_in, delta_in, taper_name=None):
    """
    Returns the spectrum and the corresponding frequencies for data with the
    given taper.
    """
    length = len(data_in)
    # zero-pad to nearest power of 2
    zpadlen = int(2 ** math.ceil(math.log(length,2)))
    # Detrend the data.
    #data = detrend(data)
    data = np.zeros(zpadlen)
    data[:length] = detrend(data_in)
    #good_length = length // 2 + 1
    good_length = zpadlen // 2 + 1
    # compute delta from delta_in and length
    #delta = delta_in * length / zpadlen
    delta = delta_in
    # Create the frequencies.
    # XXX: This might be some kind of hack
    #freq = abs(np.fft.fftfreq(length, delta)[:good_length])
    freq = abs(np.fft.fftfreq(zpadlen, delta)[:good_length])
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
    elif taper_name == 'nuttall':
        taper = scipy.signal.nuttall(length)
    elif taper_name == 'tukey':
        taper = scipy.signal.tukey(length)
    # Should never happen.
    else:
        msg = 'Something went wrong.'
        raise Exception(msg)
    # Apply the taper.
    data[:length] *= taper
    spec = abs(np.fft.rfft(data)) ** 2
    return spec, freq
