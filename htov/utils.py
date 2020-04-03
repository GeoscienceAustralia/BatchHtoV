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
import json
from scipy import interpolate
import functools
from collections import defaultdict

class SeisDB(object):
    '''
    Class for loading jason database that helps speed up random access to raw waveforms.
    This class was initially adapted from Ashby's implementation and has been copied over
    from the passive-seismic repository.
    '''
    def __init__(self, json_file=False, generate_numpy_index=True):
        self._json_loaded = False
        self._valid_index = False
        self._use_numpy_index = False
        self._generate_numpy_index = generate_numpy_index
        if json_file:
            try:
                f = open(json_file, 'r')
                self._json_dict = json.load(f)
                self._json_file = json_file
                self._json_loaded = True
                self.generateIndex()
            except IOError as e:
                print(("I/O error({0}): {1}".format(e.errorno, e.strerror)))
            except ValueError as e:
                print(("JSON Decoding has failed with a value error({0}): {1}".format(e.errorno, e.strerror)))

    def generateIndex(self):
        assert self._json_loaded, "Invalid SeisDB object. Try loading a valid JSON file first."

        if self._generate_numpy_index:

            try:
                # dictionary with keys as integer index (corresponding to numpy array indexes) and value which is ASDF tag
                self._indexed_dict = {}
                # List for numpy array building
                self._index_dict_list = []
                # container for dtypes
                type_list = []
                # check if the dtype has been populated
                dtype_pop = False
                for _i, (key, value) in enumerate(self._json_dict.items()):
                    self._indexed_dict[_i] = key
                    temp_list = []
                    for _j, (sub_key, sub_value) in enumerate(value.items()):
                        # only add some of the attributes to the numpy array to speed up lookup

                        if sub_key == "tr_starttime":
                            temp_list.append(float(sub_value))
                            if not dtype_pop:
                                type_list.append(('st', float))
                        elif sub_key == "tr_endtime":
                            temp_list.append(float(sub_value))
                            if not dtype_pop:
                                type_list.append(('et', float))
                        elif sub_key == "new_network":
                            temp_list.append(str(sub_value))
                            if not dtype_pop:
                                type_list.append(('net', 'S2'))
                        elif sub_key == "new_station":
                            temp_list.append(str(sub_value))
                            if not dtype_pop:
                                type_list.append(('sta', 'S5'))
                        elif sub_key == "new_channel":
                            temp_list.append(str(sub_value))
                            if not dtype_pop:
                                type_list.append(('cha', 'S3'))
                        elif sub_key == "new_location":
                            temp_list.append(str(sub_value))
                            if not dtype_pop:
                                type_list.append(('loc', 'S2'))

                    dtype_pop = True

                    self._index_dict_list.append(tuple(temp_list))
                dt = np.dtype(type_list)
                self._indexed_np_array = np.array(self._index_dict_list, dtype=dt)
                self._use_numpy_index = True
                self._valid_index = True

            except KeyError as e:
                print(("Indexing JSON dictionary has failed with a key error({0}): {1}".format(e.errorno, e.strerror)))

        else:
            try:
                # dictionary with keys as integer index (corresponding to numpy array indexes) and value which is ASDF tag
                self._indexed_dict = {}
                # new dictionary to be sure that starttime and endtime fields are float
                self._formatted_dict = {}
                for _i, (key, value) in enumerate(self._json_dict.items()):
                    self._indexed_dict[_i] = key
                    temp_dict = {}
                    for _j, (sub_key, sub_value) in enumerate(value.items()):
                        if sub_key == "tr_starttime":
                            temp_dict[sub_key] = float(sub_value)
                        elif sub_key == "tr_endtime":
                            temp_dict[sub_key] = float(sub_value)
                        else:
                            temp_dict[sub_key] = sub_value

                    self._formatted_dict[_i] = temp_dict
                self._valid_index = True
            except KeyError as e:
                print(("Indexing JSON dictionary has failed with a key error({0}): {1}".format(e.errorno, e.strerror)))

    def queryByTime(self, sta, chan, query_starttime, query_endtime):
        qs = query_starttime
        qe = query_endtime
        assert self._json_loaded, "Invalid SeisDB object. Try loading a valid JSON file first."
        assert self._valid_index, "Invalid SeisDB object. Index has not been generated."
        if not self._use_numpy_index:
            indices = []
            for _i, key in enumerate(self._formatted_dict.keys()):
                matched_entry = self._formatted_dict[key]
                if ((matched_entry['tr_starttime'] <= qs < matched_entry['tr_endtime'])
                    or (qs <= matched_entry['tr_starttime'] and matched_entry['tr_starttime'] < qe)) \
                        and ((matched_entry['new_station'] in sta) and (matched_entry['new_channel'] in chan)):
                    indices.append(_i)
            # indices_array = np.array(indices)
            # Print output
            # print(indices_array)
            # for index in indices_array:
            #    print(self._indexed_dict[index]['ASDF_tag'])
            # print(len(indices_array))
            # return {k:self._indexed_dict[self._index_dict_list[k]] for k in indices if k in self._index_dict_list}
            return {k: {"ASDF_tag": self._indexed_dict[k],
                        "new_station": self._formatted_dict[k]["new_station"],
                        "new_network": self._formatted_dict[k]["new_network"]}
                    for k in indices}
        else:
            _indexed_np_array_masked = None
            if(chan=='*'):
                _indexed_np_array_masked = np.where((np.in1d(self._indexed_np_array['sta'], sta))
                                                    & np.logical_or(
                    np.logical_and(self._indexed_np_array['st'] <= qs, qs < self._indexed_np_array['et']),
                    (np.logical_and(qs <= self._indexed_np_array['st'],
                                    self._indexed_np_array['st'] < qe))))
            else:
                _indexed_np_array_masked = np.where((np.in1d(self._indexed_np_array['sta'], sta))
                               & (np.in1d(self._indexed_np_array['cha'], chan))
                               & np.logical_or(np.logical_and(self._indexed_np_array['st'] <= qs,  qs < self._indexed_np_array['et']),
                                               (np.logical_and(qs <= self._indexed_np_array['st'],
                                                               self._indexed_np_array['st'] < qe))))
            # end if
            # print(_indexed_np_array_masked[0])
            # for index in _indexed_np_array_masked[0]:
            #    print(self._indexed_np_array[index, 6])
            # print(len(_indexed_np_array_masked[0]))
            # print(self._index_dict_list[0])
            # return {k:self._indexed_dict[self._indexed_np_array[k]] for k in _indexed_np_array_masked[0] if k in self._indexed_np_array}
            return {k: {"ASDF_tag": self._indexed_dict[k],
                        "new_station": self._indexed_np_array['sta'][k],
                        "new_network": self._indexed_np_array['net'][k]}
                    for k in _indexed_np_array_masked[0]}

    def queryToStream(self, ds, query, query_starttime, query_endtime,
                      decimation_factor=None):
        """
        Method to use output from seisds query to return waveform streams

        :type ds: ASDFDataSet
        :param ds: ASDFDataSet to fetch waveforms from
        :type query: dict
        :param query: Dictionary returned by function 'queryByTime'
        :type query_starttime: UTCDateTime or str
        :param query_starttime: Start time of waveform
        :type query_endtime: UTCDateTime or str
        :param query_endtime: End time of waveform
        :type decimation_factor: int
        :param decimation_factor: Decimation factor applied to returned waveforms

        :return : Stream object with merged waveform data, in which gaps are masked
        """

        # Open a new st object
        st = Stream()

        for matched_info in list(query.values()):

            # read the data from the ASDF into stream
            temp_tr = ds.waveforms[matched_info["new_network"] + '.' + matched_info["new_station"]][
                matched_info["ASDF_tag"]][0]

            # trim trace to start and endtime
            temp_tr.trim(starttime=UTCDateTime(query_starttime),
                         endtime=UTCDateTime(query_endtime))

            # append the asdf id tag into the trace stats so that the original data is accesbale
            temp_tr.stats.asdf.orig_id = matched_info["ASDF_tag"]

            # Decimate trace
            if(decimation_factor is not None): temp_tr.decimate(decimation_factor)

            # append trace to stream
            st += temp_tr

            # free memory
            temp_tr = None
        #end for

        if st.__nonzero__():
            # Attempt to merge all traces with matching ID'S in place
            #print('')
            #print('Merging %d Traces ....' % len(st))
            # filling no data with 0
            #st.print_gaps()
            # merge filling gaps
            try:
                st.merge(method=1, fill_value=0)
            except:
                print(('Merge failed for station %s ..' % (matched_info["new_station"])))
                st = Stream()
            # end try
        # end if

        return st
    #end func

    def fetchDataByTime(self, ds, sta, chan, query_starttime, query_endtime,
                        decimation_factor=None):
        """
        Wrapper to use output from seisds query to return waveform streams

        :type ds: ASDFDataSet
        :param ds: ASDFDataSet to fetch waveforms from
        :type sta: list
        :param sta: List of station names
        :type chan: str
        :param chan: Wildcard to be used to filter channels
        :type query_starttime: UTCDateTime or str
        :param query_starttime: Start time of waveform
        :type query_endtime: UTCDateTime or str
        :param query_endtime: End time of waveform
        :type decimation_factor: int
        :param decimation_factor: Decimation factor applied to returned waveforms

        :return : Stream object with merged waveform data, in which gaps are masked
        """

        qr = self.queryByTime(sta, chan, query_starttime, query_endtime)

        st = self.queryToStream(ds, qr, query_starttime, query_endtime,
                                decimation_factor=decimation_factor)

        return st
    #end func
#end class

class StreamAdapter(object):
    def __init__(self, data_path, buffer_size_in_mb=2048):
        assert os.path.exists(data_path), 'Invalid path'

        self._data_path = data_path
        self._stations = []
        self._stations_metadata = {}
        self._files_dict = defaultdict(list)
        self._buffer_size_in_mb = buffer_size_in_mb
        self._ds = None
        self._ds_jason_db = None
        self._has_jason_db = False

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
                self._ds = pyasdf.ASDFDataSet(self._data_path, mode='r', 
                                              single_item_read_limit_in_mb=self._buffer_size_in_mb)
            except:
                raise NameError('Error reading file : %s'%(self._data_path))
            # end try

            # look for json db
            files = glob.glob(os.path.dirname(self._data_path) + '/*.json')
            for f in files:
                if(os.path.splitext(os.path.basename(self._data_path))[0] in f):
                    try:
                        self._ds_jason_db = SeisDB(f)
                        self._has_jason_db = True
                    except:
                        raise RuntimeError('Failed to load json file:%s' % (f))
                    #end try
                    break
                # end if
            # end for

            for station in self._ds.waveforms:
                sn = station._station_name.split('.')[1]
                self._stations.append(sn)
                self._stations_metadata[sn] = station
            # end for
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
            if(self._has_jason_db):
                st = self._ds_jason_db.fetchDataByTime(self._ds, station_name, '*',
                                                       start_time.timestamp,
                                                       end_time.timestamp)
            else:
                st = self._ds.get_waveforms("*", station_name, "*", '*', start_time, end_time, '*')
            # end if
        # end if

        # merge filling gaps
        try:
            st.merge(method=1, fill_value=0)
        except:
            print(('Merge failed for station %s ..'%(station_name)))
            st = Stream()
        # end try
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
            for _i in range(count):
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
    indices = list(map(st_freq, freqsOut))

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
