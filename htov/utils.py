import math, os
import numpy as np
import glob
import pyasdf
from obspy import read
from obspy.core import UTCDateTime, Stream
from scipy.signal.signaltools import detrend
from obspy.signal.tf_misfit import cwt
import mlpy.wavelet as wave
import st
from scipy import interpolate
import functools
from collections import defaultdict

class StreamAdapter(object):
    def __init__(self, data_path, buffer_size_in_mb=2048):
        assert os.path.exists(data_path), 'Invalid path'

        self._data_path = data_path
        self._stations = []
        self._stations_metadata = {}
        self._files_dict = defaultdict(list)
        self._buffer_size_in_mb = buffer_size_in_mb
        self._ds = None

        if(os.path.isdir(data_path)):
            # harvest miniseed files
            self._input_type = 'mseed'

            files = glob.glob(data_path + '/*.*N')
            files += glob.glob(data_path + '/*.*E')
            files += glob.glob(data_path + '/*.*Z')

            if(len(files) % 3): raise NameError('Mismatch in component file-count detected..')

            # extract station names
            for f in files:
                try:
                    s = read(f)
                    self._files_dict[s.traces[0].stats.station].append(f)
                    self._stations.append(s.traces[0].stats.station)
                except:
                    raise NameError('Error reading file :%s'%(f))
                # end try
            # end for
            self._stations = list(set(self._stations))
        else:
            self._input_type = 'asdf'

            try:
                self._ds = pyasdf.ASDFDataSet(self._data_path, mode='r')
            except:
                raise NameError('Error reading file : %s'%(self._data_path))
            # end try

            for s in self._ds.ifilter(self._ds.q.station == "*"):
                sn = s._station_name.split('.')[1]
                self._stations.append(sn)
                self._stations_metadata[sn] = s
        # end if
    # end func

    def getStationNames(self):
        return self._stations
    # end func

    def getStream(self, station_name, start_time, end_time):
        if(station_name not in self._stations): raise NameError('Error: station-name not found.')

        if (type(start_time)!=UTCDateTime): raise NameError('start_time must be of type UTCDateTime')
        if (type(end_time) != UTCDateTime): raise NameError('end_time must be of type UTCDateTime')

        st = None
        if(self._input_type == 'mseed'):
            st = Stream()

            for f in self._files_dict[station_name]:
                cst = read(f)
                st += cst.slice(start_time, end_time)
            # end for
        elif(self._input_type=='asdf'):
            self._ds.single_item_read_limit_in_mb = self._buffer_size_in_mb

            st = self._ds.get_waveforms("*", station_name, "*", '*', start_time, end_time, '*')
        # end if

        # merge filling gaps
        st.merge(method=1, fill_value=0)
        return st
    # end func

    def getLonLat(self, station_name):
        if(self._input_type=='asdf'):
            try:
                return [self._stations_metadata[station_name].coordinates['longitude'],
                        self._stations_metadata[station_name].coordinates['latitude']]
            except:
                raise NameError('Station not found..')
            # end try
        # end if
        return []
    # end func
# end class

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


def cwt_TFA(data, delta, nf, f_min=0, f_max=0, w0=8, useMlpy=True, freq_spacing='log'):
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
        # spacing option only applies to mlpy
        if freq_spacing=='linear':
            oldnf = nf
            sr = 1.0/delta
            spacing = sr/wl
            f_min_idx = math.ceil(f_min / spacing)
            f_max_idx = math.ceil(f_max / spacing)
            nf = f_max_idx - f_min_idx + 1

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
    cfs = st.st(data)
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
