# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: htov.py
#  Purpose: Routines for calculating HVSR.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

from copy import deepcopy
from math import ceil
from mtspec import mtspec, sine_psd
import numpy as np
np.set_printoptions(threshold='nan')
#from obspy.core.util import scoreatpercentile as quantile
from scipy.stats import scoreatpercentile as quantile
from obspy.signal.filter import highpass, lowpass, bandpass
#from obspy.signal.trigger import zDetect as zdetect
from obspy.signal.trigger import z_detect as zdetect
from scipy.signal import resample
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema

#import pywt
import mlpy.wavelet as wave

from utils import *
from konno_ohmachi_smoothing import calculate_smoothing_matrix

import sys


def resampleFilterAndCutTraces(stream, resampling_rate, lowpass_value,
                               highpass_value, zerophase, corners, starttime,
                               endtime, message_function=None):
    """
    Resamples, filters and cuts all Traces in a Stream object.
    
    It will always apply each operation to every trace in the order described
    above.

    :param stream: obspy.core.stream object
        Will be altered and has to contain at least one Trace.
    :param resampling_rate: float
        Desired new sample rate.
    :param lowpass_value: float
        High filter frequency.
    :param highpass_value: float
        Low filter frequency.
    :param zerophase: bool
        Whether or not to use a zerophase filter.
    :param corners: int
        Number of corners for the used Butterworth-Filter.
    :param starttime: obspy.core.UTCDateTime
        New starttime of each Trace.
    :param endtime: obspy.core.UTCDateTime
        New endtime of each Trace.
    :param message_function: Python function
        If given, a string will be passed to this function to document the
        current progress.
    """
    # Convert to floats for more exact handling. Also level the data.
    for trace in stream:
        trace.data = np.require(trace.data, 'float32')
        trace.data -= np.linspace(trace.data[0], trace.data[-1], len(trace.data))
    # The first step is to resample the data. This is done before trimming
    # so that any boundary effects that might occur can be cut away later
    # on.
    if resampling_rate != stream[0].stats.sampling_rate:
        time_range = stream[0].stats.endtime - \
                     stream[0].stats.starttime
        new_npts = time_range / \
                   (1 / resampling_rate) + 1
        new_freq = 1.0 / (time_range / float(new_npts - 1))
        for _i, trace in enumerate(stream):
            if message_function:
                msg = 'Resampling traces to %.2f Hz [%i/%i]...' % \
                        (resampling_rate, _i + 1, len(stream))
                message_function(msg)
            # Use scipy to resample the traces.
            trace.data = resample(trace.data, new_npts, window='hamming')
            trace.stats.sampling_rate = new_freq
    # Filter the trace. Differentiate between low-, high-, and bandpass
    if lowpass_value and highpass_value:
        if message_function:
            msg = 'Bandpass filtering traces from %.2f Hz to %.2f Hz...' % \
                    (highpass_value, highpass_value)
            message_function(msg)
        for trace in stream:
            trace.data = bandpass(trace.data, highpass_value,
                                  lowpass_value, trace.stats.sampling_rate,
                                  corners=corners, zerophase=zerophase)
    elif lowpass_value:
        if message_function:
            msg = 'Lowpass filtering traces with %.2f Hz...' % lowpass_value
            message_function(msg)
        for trace in stream:
            trace.data = lowpass(trace.data, lowpass_value,
                                  trace.stats.sampling_rate,
                                  corners=corners, zerophase=zerophase)
    elif highpass_value:
        if message_function:
            msg = 'Highpass filtering traces with %.2f Hz...' % highpass_value
            message_function(msg)
        for trace in stream:
            trace.data = highpass(trace.data, highpass_value,
                                  trace.stats.sampling_rate,
                                  corners=corners, zerophase=zerophase)
    # Trim the trace if it is necessary.
    if message_function:
        message_function('Trimming traces...')
    stream.trim(starttime, endtime)

def calculateCharacteristicNoiseFunction(stream, threshold, window_length,
                                         message_function=None):
    """
    Calculates a characteristic function for the noise and threshold.

    Uses the z-detector to do this. Returns a list with the characteristic
    functions and a second list with the thresholds.

    :param stream: obspy.core.Stream
    :param threshold: Percentile value, e.g 0.95 mean the 95% percentile.
    :param window_length: Window length passed to the z-detector.
    :param message_function: Python function
        If given, a string will be passed to this function to document the
        current progress.
    """
    charNoiseFunctions = []
    if message_function:
        message_function('Calculating characteristic noise function...')
    # Get the characteristic function for each Trace.
    for trace in stream:
        charNoiseFunctions.append(zdetect(trace.data, window_length))
    lengths = [len(tr.data) for tr in stream]
    if message_function:
        message_function('Applying threshold...')
    # Calculate the thresholds from the percentage values.
    thresholds = []
    for data in charNoiseFunctions:
        length = len(data)
        s_data = np.sort(data)
        thresholds.append(s_data[int(threshold * length)])
    return charNoiseFunctions, thresholds


def getQuietIntervals(charNoiseFunctions, thresholds, window_length, npts):
    """
    Parses the characteristic Noise Function and find the areas that are under
    the threshold and at least have a length of window length and they also all
    have to be in all Traces.

    :param charNoiseFunction: list
        Contains numpy.ndarrays with characteristic functions.
    :param thresholds: list
        Contains the threshold value for each characteristic function.
    :param window_length: int
        Resulting window length.
    :param npts: int
        Number of samples for one original Trace. Needed because the
        characteristic function might contain less or more samples then the
        Trace.

    Returns three things:
        1. A numpy.ndarray containing the interval start- and endsamples in
        samples.
        2. A list containing the quiet areas for each Trace.
        3. A list containing the common quiet areas for each Trace.
    """
    # Find the areas within the treshold.
    quiet_areas = []
    for _i, data in enumerate(charNoiseFunctions):
        quiet_areas.append(getAreasWithinThreshold(data,
                thresholds[_i], window_length, 0))

    # Find the common quiet areas.
    common_quiet_areas = findCommonQuietAreas(quiet_areas,
                              npts,
                              window_length)

    # Find the intervals in the areas.
    intervals = np.array(getIntervalsInAreas(common_quiet_areas,
                                         min_interval=window_length))
    return intervals, quiet_areas, common_quiet_areas

def findCommonQuietAreas(areas, length, min_length):
    """
    areas is a list with arrays. Each array contains quiet areas as a tuple
    of two samples which represent the start and the end of the quiet
    areas.

    This function returns one array with the areas that are common to each
    array in the areas list.

    Length is an integer to avoid having to calulate the maximum number in
    each sample.
    """
    common_quiet = np.zeros(length)
    # Loop over each area and over each zone. Write every quiet zone to the
    # same array. At the end all quiet zone will have a value of zero and
    # every not quiet zone a value of one.
    for area in areas:
        position = 0
        for start, end in area:
            common_quiet[position:start] = 1
            position = end + 1
        # Do the end seperately
        common_quiet[position:length + 1] = 1
    # Now again create tuples.
    # XXX: Search for faster way of doing this.
    common_quiet_times = getAreasWithinThreshold(common_quiet, 0.5,
                                                 min_length)
    return common_quiet_times

def calculateHVSR(stream, intervals, window_length, method, options,
                  master_method, cutoff_value, smoothing=None,
                  smoothing_count=1, smoothing_constant=40,
                  message_function=None, bin_samples=100,
                  bin_sampling='log',f_min=0.1,f_max=50.0,
                  frequencies=None,w0=None):
    """
    Calculates the HVSR curve.
    """
    # Some arithmetics.
    length = len(intervals)
    good_length = window_length // 2 + 1
    # The stream that will be used.
    # XXX: Add option to use the raw data stream.
    # Create the matrix that will be used to store the single spectra.
    hvsr_matrix = np.empty((length, good_length))
    rms_matrix = np.empty((length))
    if method == 'multitaper':
        if options['nfft']:
            good_length = options['nfft']// 2 + 1
            # Create the matrix that will be used to store the single
            # spectra.
            hvsr_matrix = np.empty((length, good_length))
        # Loop over each interval
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            # Figure out which traces are vertical and which are horizontal.
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_spec, v_freq = mtspec(v, stream[0].stats.delta,
                                    options['time_bandwidth'],
                                    nfft=options['nfft'],
                                    number_of_tapers=options['number_of_tapers'],
                                    quadratic=options['quadratic'],
                                    adaptive=options['adaptive'])
            h1_spec, h1_freq = mtspec(h1, stream[0].stats.delta,
                                      options['time_bandwidth'], nfft=options['nfft'],
                                      number_of_tapers=options['number_of_tapers'],
                                      quadratic=options['quadratic'],
                                      adaptive=options['adaptive'])
            h2_spec, h2_freq = mtspec(h2, stream[0].stats.delta,
                                      options['time_bandwidth'],
                                      nfft=options['nfft'],
                                      number_of_tapers=options['number_of_tapers'],
                                      quadratic=options['quadratic'],
                                      adaptive=options['adaptive'])
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h1_spec = np.dot(h1_spec, sm_matrix)
                        h2_spec = np.dot(h2_spec, sm_matrix)
            hv_spec = np.sqrt(h1_spec * h2_spec) / v_spec
            if _i == 0:
                good_freq = v_freq
            hvsr_matrix[_i, :] = hv_spec
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:length, :]
    elif method == 'sine multitaper':
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            # Figure out which traces are vertical and which are horizontal.
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_spec, v_freq = sine_psd(v, stream[0].stats.delta,
                              number_of_tapers=options['number_of_tapers'],
                              number_of_iterations=options['number_of_iterations'],
                              degree_of_smoothing=options['degree_of_smoothing'])
            h1_spec, h1_freq = sine_psd(h1, stream[0].stats.delta,
                              number_of_tapers=options['number_of_tapers'],
                              number_of_iterations=options['number_of_iterations'],
                              degree_of_smoothing=options['degree_of_smoothing'])
            h2_spec, h2_freq = sine_psd(h2, stream[0].stats.delta,
                              number_of_tapers=options['number_of_tapers'],
                              number_of_iterations=options['number_of_iterations'],
                              degree_of_smoothing=options['degree_of_smoothing'])
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h1_spec = np.dot(h1_spec, sm_matrix)
                        h2_spec = np.dot(h2_spec, sm_matrix)
            hv_spec = np.sqrt(h1_spec * h2_spec) / v_spec
            if _i == 0:
                good_freq = v_freq
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[_i,:] = hv_spec
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:length, :]
    # Use a single taper spectrum with different available tapers.
    elif method == 'single taper':
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_spec, v_freq = single_taper_spectrum(v,
                                    stream[0].stats.delta, options['taper'])
            h1_spec, h1_freq = single_taper_spectrum(h1,
                                    stream[0].stats.delta, options['taper'])
            h2_spec, h2_freq = single_taper_spectrum(h2,
                                    stream[0].stats.delta, options['taper'])
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h1_spec = np.dot(h1_spec, sm_matrix)
                        h2_spec = np.dot(h2_spec, sm_matrix)
            hv_spec = np.sqrt(0.5 * (h1_spec  +  h2_spec) / v_spec)
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.empty((length, good_length))
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[_i, :] = hv_spec
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:length, :]
    # Use a Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'cwt':
        good_length = bin_samples
        #f_min = 0.20
        #f_max = 20.0
        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_cwt, v_freq, v_scales = cwt_TFA(v,
                                    stream[0].stats.delta, good_length, f_min, f_max, useMlpy=False)
            h1_cwt, h1_freq, v_scales = cwt_TFA(h1,
                                    stream[0].stats.delta, good_length, f_min, f_max, useMlpy=False)
            h2_cwt, h2_freq, v_scales = cwt_TFA(h2,
                                    stream[0].stats.delta, good_length, f_min, f_max, useMlpy=False)
            # Convert to spectrum via vertical maxima search
            h_cwt = np.sqrt((np.abs(h1_cwt) ** 2 + np.abs(h2_cwt) ** 2)) # FIXME test 0.5 * inner
            v_cwt = np.abs(v_cwt)
            v_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            h_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            #h_spec = np.zeros(v_freq.shape[0])

            # Compute cone of influence for morlet wavelets (Torrence and Compo 1998)
            # COI = sqrt(2.) * s
            # , where s is wavelet scale.
            cois = np.int_(np.floor(np.sqrt(2.)*v_scales*stream[0].stats.sampling_rate + 0.5))
            for findex in xrange(v_freq.shape[0]):
                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                startIdx = rayleighDelay if (rayleighDelay>cois[findex]) else cois[findex]
                endIdx = v_cwt.shape[1] - cois[findex] - 1
                extrema = argrelextrema(v_cwt[findex,
                                        startIdx:endIdx], np.greater)
                e = extrema[0]
                if e.shape[0] == 0:
                    print "No peaks found for frequency " + str(f)
                    v_spec[findex] = np.ma.masked
                    h_spec[findex] = np.ma.masked
                else:
                    vmax = v_cwt[findex,e]
                    hmax_neg = h_cwt[findex,e-rayleighDelay]
                    hmax_pos = h_cwt[findex,e+rayleighDelay]
                    #for now, average it.
                    #h_numer = np.sqrt(hmax_neg**2 + hmax_pos**2) #RMS creates bias
                    h_numer = 0.5 * (hmax_neg + hmax_pos)
                    v_numer = vmax
                    h_spec[findex] = np.sum(h_numer) / h_numer.shape[0]
                    v_spec[findex] = np.sum(v_numer) / v_numer.shape[0]
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h_spec = np.dot(h_spec, sm_matrix)
            hv_spec = h_spec / v_spec
            if _i == 0:
                good_freq = v_freq
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[num_good_intervals, :] = hv_spec
            num_good_intervals += 1
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'cwtlog':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            if w0==None:
		w0 = 8 # 16
            v_cwt, v_freq, v_scales    = cwt_TFA(v,  stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            h1_cwt, h1_freq, h1_scales = cwt_TFA(h1, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            h2_cwt, h2_freq, h2_sclaes = cwt_TFA(h2, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            # Convert to spectrum via vertical maxima search
            #h_cwt = np.abs(np.sqrt((h1_cwt ** 2 + h2_cwt ** 2))) # FIXME test 0.5 * inner
            h_cwt = np.sqrt(0.5 * (np.abs(h1_cwt) ** 2 + np.abs(h2_cwt) ** 2)) # FIXME test 0.5 * inner
            v_cwt = np.abs(v_cwt)
            v_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            h_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            #h_spec = np.zeros(v_freq.shape[0])

            # Compute cone of influence for morlet wavelets (Torrence and Compo 1998)
            # COI = sqrt(2.) * s
            # , where s is wavelet scale.
            cois = np.int_(np.floor(np.sqrt(2.)*v_scales*stream[0].stats.sampling_rate + 0.5))
            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                startIdx = rayleighDelay if (rayleighDelay>cois[findex]) else cois[findex]
                endIdx = v_cwt.shape[1] - cois[findex] - 1
                extrema = argrelextrema(v_cwt[findex,
                                        startIdx:endIdx], np.greater)
                e = extrema[0]
                if e.shape[0] == 0:
                    print "No peaks found for frequency " + str(f)
                    v_spec[findex] = np.ma.masked
                    h_spec[findex] = np.ma.masked
                else:
                    e += startIdx
                    vmax = v_cwt[findex,e]
                    hmax_neg = h_cwt[findex,e-rayleighDelay]
                    hmax_pos = h_cwt[findex,e+rayleighDelay]
                    #for now, average it.
                    #h_numer = np.sqrt(hmax_neg**2 + hmax_pos**2) #RMS creates bias
                    #h_numer = 0.5 * (hmax_neg + hmax_pos)
                    h_numer = np.maximum(hmax_neg,hmax_pos)
                    v_numer = vmax
                    # likely incorrect. We should not sum because this biases to higher amplitudes.
                    h_spec[findex] = np.sum(h_numer) / h_numer.shape[0]
                    v_spec[findex] = np.sum(v_numer) / v_numer.shape[0]
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h_spec = np.dot(h_spec, sm_matrix)
            hv_spec = h_spec / v_spec
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.empty((length, good_length))
            # Store it into the matrix if it has the correct length.
            if np.all(np.isfinite(hv_spec)):
                hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
                num_good_intervals += 1
            else:
                print "Excluding window " + str(_i) + " due to non-finite values" 
                print hv_spec
            #hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
            #num_good_intervals += 1
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'raydecwslog':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the bands
            v_cheb, v_freq   = get_ws_bands(v,  stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='log',freqs=frequencies)
            h1_cheb, h1_freq = get_ws_bands(h1, stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='log',freqs=frequencies)
            h2_cheb, h2_freq = get_ws_bands(h2, stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='log',freqs=frequencies)

            #print v_freq

            # Vertical maxima search
            hv_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))

            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                deltalen = int(stream[0].stats.sampling_rate * 10.0/f)

                vbpf = v_cheb[findex,:]
                h1bpf = h1_cheb[findex,:]
                h2bpf = h2_cheb[findex,:] 
                if np.isnan(vbpf).any() or np.isnan(h1bpf).any() or np.isnan(h2bpf).any():
                    print "NAN detected in icwt"
                    sys.exit()

                # find all zero crossings in vertical
                e = np.where(np.diff(np.sign(vbpf)) > 0)[0]
                #print "VBPF " + str(vbpf)
                #print "Zero crossings for frequency " + str(f)+ ": " + str(e)
                #print "Computing zero crossings for frequency " + str(f)

                if e.shape[0] == 0:
                    print "No zero-crossings found for frequency " + str(f)
                    hv_spec[findex] = np.ma.masked
                else:
		    vfs = np.zeros(deltalen)
		    hfs = np.zeros(deltalen)
                    for e_ in xrange(e.shape[0]):
                        ei = e[e_]
                        if ei < rayleighDelay or ei + deltalen >= vbpf.shape[0]:
                            continue
                        hei = ei - rayleighDelay
                        #print "vmax at " + str(ei) + " with delayed h index " + str(hei) + " search idx = " + str((startIdx,endIdx))
                        # buffer
                        vfbi = vbpf[ei:ei+deltalen]
                        h1fbi = h1bpf[hei:hei+deltalen] 
                        h2fbi = h2bpf[hei:hei+deltalen] 
                        # solve for azimuth
                        azimuth = math.atan2(np.sum(vfbi*h1fbi),np.sum(vfbi*h2fbi))
                        hfbi = math.sin(azimuth)*h1fbi + math.cos(azimuth)*h2fbi
                        #print "Azimuth = " + str(azimuth) + " sin(a) = " + str(math.sin(azimuth)) + " cos(a) = " +str(math.cos(azimuth))
                        corweight = np.sum(vfbi*hfbi)/math.sqrt(np.sum(vfbi**2)*np.sum(hfbi**2)) 
                        #print "Correlation weight = " + str(corweight)
                        if corweight > 0.4:
                            vfs+=vfbi * (corweight ** 2)
                            hfs+=hfbi * (corweight ** 2)
                        #print "f = " + str(f) + " h = " + str(np.sum((hfbi * (corweight ** 2)) ** 2)) + " v = " + str(np.sum((vfbi * (corweight ** 2)) ** 2)) + " hv = " + str(np.sqrt(np.sum((hfbi * (corweight ** 2)) ** 2)/np.sum((vfbi * (corweight ** 2)) ** 2)))

                    if np.sum(vfs) > 0:
                        hv_spec[findex] = math.sqrt(np.sum(hfs**2) / np.sum(vfs**2))
                    else:
                        hv_spec[findex] = np.ma.masked
                    #print "hv("+str(f)+") = " + str(hv_spec[findex])
                    del vfs
                    del hfs
                    
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.zeros((length, good_length))
            # Store it into the matrix if it has the correct length and contains no NaNs
            if np.all(np.isfinite(hv_spec)):
                hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
                num_good_intervals += 1
            else:
                print "Excluding window " + str(_i) + " due to non-finite values" 
                print hv_spec
            print "Num good intervals = " + str(num_good_intervals)
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'raydecchebylin':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the bands
            v_cheb, v_freq   = get_ws_bands(v,  stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='linear',freqs=frequencies)
            h1_cheb, h1_freq = get_ws_bands(h1, stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='linear',freqs=frequencies)
            h2_cheb, h2_freq = get_ws_bands(h2, stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='linear',freqs=frequencies)

            #print v_freq

            # Vertical maxima search
            hv_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))

            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                deltalen = int(stream[0].stats.sampling_rate * 10.0/f)

                vbpf = v_cheb[findex,:]
                h1bpf = h1_cheb[findex,:]
                h2bpf = h2_cheb[findex,:] 
                if np.isnan(vbpf).any() or np.isnan(h1bpf).any() or np.isnan(h2bpf).any():
                    print "NAN detected in icwt"
                    sys.exit()

                # find all zero crossings in vertical
                e = np.where(np.diff(np.sign(vbpf)) > 0)[0]
                #print "VBPF " + str(vbpf)
                #print "Zero crossings for frequency " + str(f)+ ": " + str(e)
                #print "Computing zero crossings for frequency " + str(f)

                if e.shape[0] == 0:
                    print "No zero-crossings found for frequency " + str(f)
                    hv_spec[findex] = np.ma.masked
                else:
		    vfs = np.zeros(deltalen)
		    hfs = np.zeros(deltalen)
                    for e_ in xrange(e.shape[0]):
                        ei = e[e_]
                        if ei < rayleighDelay or ei + deltalen >= vbpf.shape[0]:
                            continue
                        hei = ei - rayleighDelay
                        #print "vmax at " + str(ei) + " with delayed h index " + str(hei) + " search idx = " + str((startIdx,endIdx))
                        # buffer
                        vfbi = vbpf[ei:ei+deltalen]
                        h1fbi = h1bpf[hei:hei+deltalen] 
                        h2fbi = h2bpf[hei:hei+deltalen] 
                        # solve for azimuth
                        azimuth = math.atan2(np.sum(vfbi*h1fbi),np.sum(vfbi*h2fbi))
                        hfbi = math.sin(azimuth)*h1fbi + math.cos(azimuth)*h2fbi
                        #print "Azimuth = " + str(azimuth) + " sin(a) = " + str(math.sin(azimuth)) + " cos(a) = " +str(math.cos(azimuth))
                        corweight = np.sum(vfbi*hfbi)/math.sqrt(np.sum(vfbi**2)*np.sum(hfbi**2)) 
                        #print "Correlation weight = " + str(corweight)
                        if corweight > 0.4:
                            vfs+=vfbi * (corweight ** 2)
                            hfs+=hfbi * (corweight ** 2)
                        #print "f = " + str(f) + " h = " + str(np.sum((hfbi * (corweight ** 2)) ** 2)) + " v = " + str(np.sum((vfbi * (corweight ** 2)) ** 2)) + " hv = " + str(np.sqrt(np.sum((hfbi * (corweight ** 2)) ** 2)/np.sum((vfbi * (corweight ** 2)) ** 2)))

                    hv_spec[findex] = math.sqrt(np.sum(hfs**2) / np.sum(vfs**2))
                    #print "hv("+str(f)+") = " + str(hv_spec[findex])
                    del vfs
                    del hfs
                    
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.zeros((length, good_length))
            # Store it into the matrix if it has the correct length and contains no NaNs
            if np.all(np.isfinite(hv_spec)):
                hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
                num_good_intervals += 1
            else:
                print "Excluding window " + str(_i) + " due to non-finite values" 
            print "Num good intervals = " + str(num_good_intervals)
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'raydeccwtlin':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            if w0==None:
		w0 = 8 # 16
            v_cwt, v_freq, v_scales    = cwt_TFA(v,  stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='linear',freqs=frequencies)
            h1_cwt, h1_freq, h1_scales = cwt_TFA(h1, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='linear',freqs=frequencies)
            h2_cwt, h2_freq, h2_scales = cwt_TFA(h2, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='linear',freqs=frequencies)
            # Convert to spectrum via vertical maxima search
            #v_cwt = np.abs(v_cwt)
            hv_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))

            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                deltalen = int(stream[0].stats.sampling_rate * 10.0/f)
                #startIdx = rayleighDelay if (rayleighDelay>cois[findex]) else cois[findex]
                #startIdx = cois[findex] + rayleighDelay
                #endIdx = v_cwt.shape[1] - cois[findex] - deltalen - 1
                #extrema = argrelextrema(v_cwt[findex,
                #                        startIdx:endIdx], np.greater)
                #e = extrema[0]

                vbpf = np.real(icwt_band(v_cwt[findex,:], stream[0].stats.delta, f, w0)).flatten()
                h1bpf = np.real(icwt_band(h1_cwt[findex,:], stream[0].stats.delta, f, w0)).flatten()
                h2bpf = np.real(icwt_band(h2_cwt[findex,:], stream[0].stats.delta, f, w0)).flatten()
                if np.isnan(vbpf).any() or np.isnan(h1bpf).any() or np.isnan(h2bpf).any():
                    print "NAN detected in icwt"
                    sys.exit()

                # find all zero crossings in vertical
                e = np.where(np.diff(np.sign(vbpf)) > 0)[0]
                #print "VBPF " + str(vbpf)
                #print "Zero crossings: " + str(e)

                if e.shape[0] == 0:
                    print "No zero-crossings found for frequency " + str(f)
                    hv_spec[findex] = np.ma.masked
                else:
		    vfs = np.zeros(deltalen)
		    hfs = np.zeros(deltalen)
                    for e_ in xrange(e.shape[0]):
                        ei = e[e_]
                        if ei < rayleighDelay or ei + deltalen >= vbpf.shape[0]:
                            continue
                        hei = ei - rayleighDelay
                        #print "vmax at " + str(ei) + " with delayed h index " + str(hei) + " search idx = " + str((startIdx,endIdx))
                        # buffer
                        vfbi = vbpf[ei:ei+deltalen]
                        h1fbi = h1bpf[hei:hei+deltalen] 
                        h2fbi = h2bpf[hei:hei+deltalen] 
                        # solve for azimuth
                        azimuth = math.atan2(np.sum(vfbi*h1fbi),np.sum(vfbi*h2fbi))
                        hfbi = math.sin(azimuth)*h1fbi + math.cos(azimuth)*h2fbi
                        #print "Azimuth = " + str(azimuth) + " sin(a) = " + str(math.sin(azimuth)) + " cos(a) = " +str(math.cos(azimuth))
                        corweight = np.sum(vfbi*hfbi)/math.sqrt(np.sum(vfbi**2)*np.sum(hfbi**2)) 
                        #print "Correlation weight = " + str(corweight)
                        if corweight > 0.4:
                            vfs+=vfbi * (corweight ** 2)
                            hfs+=hfbi * (corweight ** 2)
                        #print "f = " + str(f) + " h = " + str(np.sum((hfbi * (corweight ** 2)) ** 2)) + " v = " + str(np.sum((vfbi * (corweight ** 2)) ** 2)) + " hv = " + str(np.sqrt(np.sum((hfbi * (corweight ** 2)) ** 2)/np.sum((vfbi * (corweight ** 2)) ** 2)))

                    hv_spec[findex] = math.sqrt(np.sum(hfs**2) / np.sum(vfs**2))
                    #print "hv("+str(f)+") = " + str(hv_spec[findex])
                    del vfs
                    del hfs
                    
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.zeros((length, good_length))
            # Store it into the matrix if it has the correct length and contains no NaNs
            if np.all(np.isfinite(hv_spec)):
                hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
                num_good_intervals += 1
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'raydeccwtlog2':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            if w0==None:
		w0 = 8 # 16
            v_cwt, v_freq, v_scales    = cwt_TFA(v,  stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            h1_cwt, h1_freq, h1_scales = cwt_TFA(h1, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            h2_cwt, h2_freq, h2_scales = cwt_TFA(h2, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)

            hv_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))

            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                deltalen = int(stream[0].stats.sampling_rate * 10.0/f)
                #startIdx = rayleighDelay if (rayleighDelay>cois[findex]) else cois[findex]
                #startIdx = cois[findex] + rayleighDelay
                #endIdx = v_cwt.shape[1] - cois[findex] - deltalen - 1
                #extrema = argrelextrema(v_cwt[findex,
                #                        startIdx:endIdx], np.greater)
                #e = extrema[0]

                vbpf = np.real(icwt_band(v_cwt[findex,:], stream[0].stats.delta, f, w0)).flatten()
                h1bpf = np.real(icwt_band(h1_cwt[findex,:], stream[0].stats.delta, f, w0)).flatten()
                h2bpf = np.real(icwt_band(h2_cwt[findex,:], stream[0].stats.delta, f, w0)).flatten()
                if np.isnan(vbpf).any() or np.isnan(h1bpf).any() or np.isnan(h2bpf).any():
                    print "NAN detected in icwt"
                    sys.exit()

                # find all zero crossings in vertical
                e = np.where(np.diff(np.sign(vbpf)) > 0)[0]
                #print "VBPF " + str(vbpf)
                #print "Zero crossings: " + str(e)

                if e.shape[0] == 0:
                    print "No zero-crossings found for frequency " + str(f)
                    hv_spec[findex] = np.ma.masked
                else:
		    vfs = np.zeros(deltalen)
		    hfs = np.zeros(deltalen)
                    for e_ in xrange(e.shape[0]):
                        ei = e[e_]
                        if ei < rayleighDelay or ei + deltalen >= vbpf.shape[0]:
                            continue
                        hei = ei - rayleighDelay
                        #print "vmax at " + str(ei) + " with delayed h index " + str(hei) + " search idx = " + str((startIdx,endIdx))
                        # buffer
                        vfbi = vbpf[ei:ei+deltalen]
                        h1fbi = h1bpf[hei:hei+deltalen] 
                        h2fbi = h2bpf[hei:hei+deltalen] 
                        # solve for azimuth
                        azimuth = math.atan2(np.sum(vfbi*h1fbi),np.sum(vfbi*h2fbi))
                        hfbi = math.sin(azimuth)*h1fbi + math.cos(azimuth)*h2fbi
                        #print "Azimuth = " + str(azimuth) + " sin(a) = " + str(math.sin(azimuth)) + " cos(a) = " +str(math.cos(azimuth))
                        corweight = np.sum(vfbi*hfbi)/math.sqrt(np.sum(vfbi**2)*np.sum(hfbi**2)) 
                        #print "Correlation weight = " + str(corweight)
                        if corweight > 0.5:
                            vfs+=vfbi * (corweight ** 2)
                            hfs+=hfbi * (corweight ** 2)
                        #print "f = " + str(f) + " h = " + str(np.sum((hfbi * (corweight ** 2)) ** 2)) + " v = " + str(np.sum((vfbi * (corweight ** 2)) ** 2)) + " hv = " + str(np.sqrt(np.sum((hfbi * (corweight ** 2)) ** 2)/np.sum((vfbi * (corweight ** 2)) ** 2)))

                    if np.sum(vfs) > 0:
                        hv_spec[findex] = math.sqrt(np.sum(hfs**2) / np.sum(vfs**2))
                    else:
                        hv_spec[findex] = np.ma.masked
                    #print "hv("+str(f)+") = " + str(hv_spec[findex])
                    del vfs
                    del hfs
                    
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.zeros((length, good_length))
            # Store it into the matrix if it has the correct length and contains no NaNs
            if np.all(np.isfinite(hv_spec)):
                hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
                num_good_intervals += 1
            else:
                print "Excluding window " + str(_i) + " due to non-finite values" 
                print hv_spec
            print "Num good intervals = " + str(num_good_intervals)
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'raydeccwtlog':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            if w0==None:
		w0 = 8 # 16
            v_cwt, v_freq, v_scales    = cwt_TFA(v,  stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            h1_cwt, h1_freq, h1_scales = cwt_TFA(h1, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            h2_cwt, h2_freq, h2_scales = cwt_TFA(h2, stream[0].stats.delta, good_length, f_min, f_max, w0=w0, freq_spacing='log',freqs=frequencies)
            # Convert to spectrum via vertical maxima search
            v_cwt = np.abs(v_cwt)
            hv_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            if 'v_amps' not in locals():
		v_amps = [[] for i in xrange(v_freq.shape[0])]
		h_amps = [[] for i in xrange(v_freq.shape[0])]
		corweights = [[] for i in xrange(v_freq.shape[0])]
            # Compute cone of influence for morlet wavelets (Torrence and Compo 1998)
            # COI = sqrt(2.) * s
            # , where s is wavelet scale.
            cois = np.int_(np.floor(np.sqrt(2.)*v_scales*stream[0].stats.sampling_rate + 0.5))
            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                #deltalen = int(max(10.0,f)/f)
                deltalen = int(stream[0].stats.sampling_rate * 10.0/f)
                #startIdx = rayleighDelay if (rayleighDelay>cois[findex]) else cois[findex]
                startIdx = cois[findex] + rayleighDelay
                endIdx = v_cwt.shape[1] - cois[findex] - deltalen - 1
                extrema = argrelextrema(v_cwt[findex,
                                        startIdx:endIdx], np.greater)
                e = extrema[0]
                if e.shape[0] == 0:
                    print "No peaks found for frequency " + str(f)
                    hv_spec[findex] = np.ma.masked
                else:
                    #hv_f = np.zeros(e.shape[0])
                    for e_ in xrange(e.shape[0]):
                        ei = e[e_] + startIdx
                        hei = ei - rayleighDelay
                        #print "vmax at " + str(ei) + " with delayed h index " + str(hei) + " search idx = " + str((startIdx,endIdx))
                        v_amp = np.abs(v_cwt[findex,ei]) #** 2
                        h1_amp = np.abs(h1_cwt[findex,hei]) #** 2
                        h2_amp = np.abs(h2_cwt[findex,hei]) #** 2
                        # buffer
                        vfbi = np.abs(v_cwt[findex,ei:ei+deltalen]) #** 2
                        h1fbi = np.abs(h1_cwt[findex,hei:hei+deltalen]) #** 2
                        h2fbi = np.abs(h2_cwt[findex,hei:hei+deltalen]) #** 2
                        # solve for azimuth
                        azimuth = math.atan2(np.sum(vfbi*h1fbi),np.sum(vfbi*h2fbi))
                        hfbi = math.sin(azimuth)*h1fbi + math.cos(azimuth)*h2fbi
                        h_amp =  math.sin(azimuth)*h1_amp + math.cos(azimuth)*h2_amp
                        corweight = np.sum(vfbi*hfbi)/math.sqrt(np.sum(vfbi**2)*np.sum(hfbi**2)) 
                        v_amps[findex].append(v_amp)
                        h_amps[findex].append(h_amp)
                        corweights[findex].append(corweight)
                        #hv_f[e_] = corweight **2 * h_amp/v_amp
                    
                    # probably wrong
                    #hv_spec[findex] = hv_f.sum() / hv_f.shape[0]
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.zeros((length, good_length))
            # Store it into the matrix if it has the correct length.
            #hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
            num_good_intervals += 1
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
        master_curve = np.zeros(v_freq.shape[0])
        error = np.zeros(v_freq.shape[0])
        for findex in xrange(v_freq.shape[0]):
            #print v_freq[findex]
            cws = np.array(corweights[findex])
            #print cws
            # for a weighted mean, normalise weights to sum to 1
            cws = cws ** 2
            cws /= cws.sum()
            V1 = 1
            V2 = np.sum(cws ** 2)
            hv_arr  = np.log(np.array(h_amps[findex]) / np.array(v_amps[findex]))
            #print hv_arr
            master_curve[findex] = np.sum(cws * hv_arr)
            error[findex] = math.sqrt(np.sum(cws*((hv_arr - master_curve[findex])**2))/(V1 - (V2/V1))) # std of weighted mean
        return hvsr_matrix, good_freq, len(hvsr_matrix), master_curve, error
    # Use a mlpy Morlet CWT and isolate via the vertical spectrum maxima
    elif method == 'cwtmlpy':
        good_length = bin_samples

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            v_cwt, v_freq, v_scales    = cwt_TFA(v,  stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='linear')
            h1_cwt, h1_freq, h1_scales = cwt_TFA(h1, stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='linear')
            h2_cwt, h2_freq, h2_sclaes = cwt_TFA(h2, stream[0].stats.delta, good_length, f_min, f_max, freq_spacing='linear')
            # Convert to spectrum via vertical maxima search
            #h_cwt = np.abs(np.sqrt((h1_cwt ** 2 + h2_cwt ** 2))) # FIXME test 0.5 * inner
            h_cwt = np.sqrt(0.5 * (np.abs(h1_cwt) ** 2 + np.abs(h2_cwt) ** 2)) # FIXME test 0.5 * inner
            v_cwt = np.abs(v_cwt)
            v_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            h_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            #h_spec = np.zeros(v_freq.shape[0])

            # Compute cone of influence for morlet wavelets (Torrence and Compo 1998)
            # COI = sqrt(2.) * s
            # , where s is wavelet scale.
            cois = np.int_(np.floor(np.sqrt(2.)*v_scales*stream[0].stats.sampling_rate + 0.5))
            for findex in xrange(v_freq.shape[0]):

                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference

                # exclude cone of influence from regions searched for peaks.
                startIdx = rayleighDelay if (rayleighDelay>cois[findex]) else cois[findex]
                endIdx = v_cwt.shape[1] - cois[findex] - 1
                extrema = argrelextrema(v_cwt[findex,
                                        startIdx:endIdx], np.greater)
                e = extrema[0]
                if e.shape[0] == 0:
                    print "No peaks found for frequency " + str(f)
                    v_spec[findex] = np.ma.masked
                    h_spec[findex] = np.ma.masked
                else:
                    vmax = v_cwt[findex,e]
                    hmax_neg = h_cwt[findex,e-rayleighDelay]
                    hmax_pos = h_cwt[findex,e+rayleighDelay]
                    #for now, average it.
                    #h_numer = np.sqrt(hmax_neg**2 + hmax_pos**2) #RMS creates bias
                    #h_numer = 0.5 * (hmax_neg + hmax_pos)
                    h_numer = np.maximum(hmax_neg,hmax_pos)
                    v_numer = vmax
                    h_spec[findex] = np.sum(h_numer) / h_numer.shape[0]
                    v_spec[findex] = np.sum(v_numer) / v_numer.shape[0]
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h_spec = np.dot(h_spec, sm_matrix)
            hv_spec = h_spec / v_spec
            if _i == 0:
                good_freq = v_freq
                good_length = v_freq.shape[0]
                # Create the matrix that will be used to store the single
                # spectra.
                hvsr_matrix = np.empty((length, good_length))
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
            num_good_intervals += 1
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Use a Stockwell transform and isolate via the vertical spectrum maxima
    elif method == 'st':
        good_length = bin_samples
        #f_min = 0.25
        #f_max = 20.0

        hvsr_matrix = np.ma.empty((length, good_length))
        num_good_intervals = 0
        for _i, interval in enumerate(intervals):
            if message_function:
                message_function('Calculating HVSR %i of %i...' % \
                                 (_i+1, length))
            v = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'vertical']
            h = [_j for _j, trace in enumerate(stream) if \
                 trace.stats.orientation == 'horizontal']
            v = stream[v[0]].data[interval[0]: interval[0] + \
                               window_length]
            h1 = stream[h[0]].data[interval[0]: interval[0] + \
                                window_length]
            h2 = stream[h[1]].data[interval[0]: interval[0] + \
                                window_length]
            # Calculate the spectra.
            io = None
            v_cwt, v_freq   = st_TFA(v,  stream[0].stats.delta, good_length, f_min, f_max)
            h1_cwt, h1_freq = st_TFA(h1, stream[0].stats.delta, good_length, f_min, f_max)
            h2_cwt, h2_freq = st_TFA(h2, stream[0].stats.delta, good_length, f_min, f_max)
            # Convert to spectrum via vertical maxima search
            h_cwt = np.abs(np.sqrt((h1_cwt ** 2 + h2_cwt ** 2))) # FIXME test 0.5 * inner
            v_cwt = np.abs(v_cwt)
            v_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            h_spec = np.ma.array(np.zeros(v_freq.shape[0]),mask=np.ones(v_freq.shape[0]))
            #h_spec = np.zeros(v_freq.shape[0])
            for findex in xrange(v_freq.shape[0]):
                f = v_freq[findex]
                rayleighDelay = int(0.25 * (1.0/f) * stream[0].stats.sampling_rate + 0.5) # the + 0.5 at the end is to round to nearest for integer reference
                extrema = argrelextrema(v_cwt[findex,rayleighDelay:-rayleighDelay], np.greater)
                e = extrema[0]
                if e.shape[0] == 0:
                    print "No peaks found for frequency " + str(f)
                    v_spec[findex] = np.ma.masked
                    h_spec[findex] = np.ma.masked
                else:
                    vmax = v_cwt[findex,e]
                    hmax_neg = h_cwt[findex,e-rayleighDelay]
                    hmax_pos = h_cwt[findex,e+rayleighDelay]
                    #for now, average it.
                    #h_numer = np.sqrt(hmax_neg**2 + hmax_pos**2) #RMS creates bias
                    h_numer = 0.5 * (hmax_neg + hmax_pos)
                    v_numer = vmax
                    h_spec[findex] = np.sum(h_numer) / h_numer.shape[0]
                    v_spec[findex] = np.sum(v_numer) / v_numer.shape[0]
            # Apply smoothing.
            if smoothing:
                if 'konno-ohmachi' in smoothing.lower():
                    if _i == 0:
                        sm_matrix = calculate_smoothing_matrix(v_freq,
                                                           smoothing_constant)
                    for _j in xrange(smoothing_count):
                        v_spec = np.dot(v_spec, sm_matrix)
                        h_spec = np.dot(h_spec, sm_matrix)
            hv_spec = h_spec / v_spec
            if _i == 0:
                good_freq = v_freq
            # Store it into the matrix if it has the correct length.
            hvsr_matrix[num_good_intervals, 0:hv_spec.shape[0]] = hv_spec
            num_good_intervals += 1
        # Cut the hvsr matrix.
        hvsr_matrix = hvsr_matrix[0:num_good_intervals, :]
    # Should never happen.
    else:
        msg = 'Something went wrong.'
        raise Exception(msg)
    # Copy once to be able to calculate standard deviations.
    original_matrix = deepcopy(hvsr_matrix)
    # Sort it for quantile operations.
    hvsr_matrix.sort(axis=0)
    # Only senseful for mean calculations. Omitted for the median.
    if cutoff_value != 0.0 and master_method != 'median':
        hvsr_matrix = hvsr_matrix[int(length * cutoff_value):
                              int(ceil(length * (1 - cutoff_value))), :]
    length = len(hvsr_matrix)
    # Mean.
    if master_method == 'mean':
        master_curve = hvsr_matrix.mean(axis=0)
    # Geometric average.
    elif master_method == 'geometric average':
        master_curve = hvsr_matrix.prod(axis=0) ** (1.0 / length)
    # Median.
    elif master_method == 'median':
        # Use another method because interpolation might be necessary.
        master_curve = np.empty(len(hvsr_matrix[0, :]))
        error = np.empty((len(master_curve), 2))
        for _i in xrange(len(master_curve)):
            cur_row = hvsr_matrix[:, _i]
            master_curve[_i] = quantile(cur_row, 50)
            error[_i, 0] = quantile(cur_row, 25)
            error[_i, 1] = quantile(cur_row, 75)
    # Calculate the standard deviation for the two mean methods.
    if master_method != 'median':
        error = np.empty((len(master_curve), 2))
        std = (hvsr_matrix[:][:] - master_curve) ** 2
        std = std.sum(axis=0)
        std /= float(length)
        std **= 0.5
        error[:, 0] = master_curve - std
        error[:, 1] = master_curve + std
    return original_matrix, good_freq, length, master_curve, error

def detectTraceOrientation(stream):
    """
    Detects the orientation of each Trace in a Stream object simply based
    on it ending with z.

    If that does not work any 'z' in the channel attribute will do or else
    the last trace will be the vertical trace.
    """
    for trace in stream:
        if trace.stats.channel.lower().endswith('z'):
            trace.stats.orientation = 'vertical'
        else:
            trace.stats.orientation = 'horizontal'
    # Check if it worked. Only one vertical component should be available.
    check = [True for trace in stream if trace.stats.orientation=='vertical']
    if len(check) == 1:
        return
    # Try matchmaking based on any z in it.
    for trace in stream:
        if 'z' in trace.stats.channel.lower():
            trace.stats.orientation = 'vertical'
        else:
            trace.stats.orientation = 'horizontal'
    # Check if it worked. Only one vertical component should be available.
    check = [True for trace in stream if trace.stats.orientation=='vertical']
    if len(check) == 1:
        return
    # The last one will be the vertical component.
    for trace in stream:
        trace.stats.orientation == 'horizontal'
    stream[-1].stats.oriantatio = 'vertical'

