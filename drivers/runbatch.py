from htov import batch
from htov.utils import StreamAdapter
from mpi4py import MPI
import glob, os, sys
import obspy
from obspy.core import UTCDateTime
import numpy as np
import matplotlib
from collections import defaultdict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy
import calendar
import math
import sys
import click
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from htov.utils import SpooledMatrix, waveform_iterator3c
from seismic.ASDFdatabase.FederatedASDFDataSet import FederatedASDFDataSet
import re

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

def compute_master_curve(spooled_storage:SpooledMatrix,
                         master_curve_method:str, cutoff_value:float):
    """
    :param master_curve_method: string
        How to determine the master curve. Available are 'mean', 'geometric
        average', 'full gaussian' , and 'median'
    :param cutoff_value: float
        If given, than this value determines which part of the bottom and top
        frequencies are discarded for the calculation of the master HVSR curve.
        e.g. a value of 0.1 will throw away the bottom and top 10% for each
        frequency.
    """
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
        for _i in range(len(master_curve)):
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


    nwindows = len(hvsr_matrix)


    def find_nearest_idx(array, value):
        return (np.abs(array - value)).argmin()


    std = (np.log1p(hvsr_matrix[:][:]) - np.log1p(master_curve))
    errormag = np.zeros(nwindows)
    for i in range(nwindows):
        errormag[i] = np.dot(std[i, :], std[i, :].T)
    error = np.dot(std.T, std)
    error /= float(nwindows - 1)

    if (compute_sparse_covariance):
        print("Computing sparse model covariance")
        sp_model = GraphicalLassoCV()
        sp_model.fit(std)
        sp_cov = sp_model.covariance_
        sp_prec = sp_model.precision_
    # end if

    if CLIP_TO_FREQ:
        lclip = find_nearest_idx(hvsr_freq, lowest_freq)
        uclip = find_nearest_idx(hvsr_freq, highest_freq)
        master_curve = master_curve[lclip:uclip]
        hvsr_freq = hvsr_freq[lclip:uclip]
        error = error[lclip:uclip, lclip:uclip]
    # end if

    print("Master curve shape: " + str(master_curve.shape))
    print(master_curve)
    print("Frequencies shape: " + str(hvsr_freq.shape))
    print(hvsr_freq)
    print("Error shape: " + str(error.shape))
    print(error)

    diagerr = np.sqrt(np.diag(error))
    lerr = np.exp(np.log(master_curve) - diagerr)
    uerr = np.exp(np.log(master_curve) + diagerr)

    # set output prefix
    tag = ''
    if (output_prefix == '$station_name.$spec_method'):
        tag = '%s.%s' % (station, spectra_method.replace(' ', '_'))
    else:
        tag = output_prefix
    # end if
    saveprefix = os.path.join(output_path, tag)

    np.savetxt(saveprefix + '.hv.txt', np.column_stack((hvsr_freq, master_curve, lerr, uerr)))
    np.savetxt(saveprefix + '.std.txt', np.std(hvsr_matrix, axis=0))
    if (len(lonlat)): np.savetxt(saveprefix + '.lonlat.txt', np.array(lonlat).T)
    np.savetxt(saveprefix + '.error.txt', error)
    np.savetxt(saveprefix + '.inverror.txt', np.linalg.inv(error))
    logdeterr = np.linalg.slogdet(error)
    print("Log determinant of error matrix: " + str(logdeterr))
    np.savetxt(saveprefix + '.logdeterror.txt', np.array(logdeterr))

    if (compute_sparse_covariance):
        # sparse equivalent
        np.savetxt(saveprefix + '.sperror.txt', sp_cov)
        np.savetxt(saveprefix + '.invsperror.txt', sp_prec)
        logdetsperr = np.linalg.slogdet(sp_cov)
        print("Log determinant of sparse error matrix: " + str(logdetsperr))
        np.savetxt(saveprefix + '.logdetsperror.txt', np.array(logdetsperr))
    # end if

    f = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(4, 4, height_ratios=[40, 2, 40, 2])

    # Plot master curves and +/- 1 sigma
    a1 = plt.subplot(gs[:, 0])
    a1.plot(hvsr_freq, master_curve, 'r', lw=2)
    a1.plot(hvsr_freq, lerr, 'g', lw=0.5)
    a1.plot(hvsr_freq, uerr, 'b', lw=0.5)
    a1.set_yscale('log')
    a1.set_xscale('log')
    a1.grid(which='major', linestyle='-', linewidth=0.5, color='k')
    a1.grid(which='minor', linestyle=':', linewidth=0.5, color='grey')

    # Plot covariance
    a2 = plt.subplot(gs[0, 1])
    ca2 = a2.imshow(error, interpolation='nearest')
    cba2 = plt.subplot(gs[1, 1])
    cbar2 = f.colorbar(ca2, cax=cba2, orientation='horizontal')
    cbar2.ax.tick_params(labelsize=7)
    a2.title.set_text('Covariance Matrix')

    # Plot inverse of covariance
    a3 = plt.subplot(gs[0, 2])
    ca3 = a3.imshow(np.linalg.inv(error), interpolation='nearest')
    cba3 = plt.subplot(gs[1, 2])
    cbar3 = f.colorbar(ca3, cax=cba3, orientation='horizontal')
    cbar3.ax.tick_params(labelsize=7)
    a3.title.set_text('Inverse of covariance Matrix')

    if (compute_sparse_covariance):
        # Plot sparse covariance
        a22 = plt.subplot(gs[2, 1])
        ca22 = a22.imshow(sp_cov, interpolation='nearest')
        cba22 = plt.subplot(gs[3, 1])
        cbar22 = f.colorbar(ca22, cax=cba22, orientation='horizontal')
        cbar22.ax.tick_params(labelsize=7)
        a22.title.set_text('Sparse covariance Matrix')

        # Plot inverse of sparse covariance (precision)
        a23 = plt.subplot(gs[2, 2])
        ca23 = a23.imshow(sp_prec, interpolation='nearest')
        cba23 = plt.subplot(gs[3, 2])
        cbar23 = f.colorbar(ca23, cax=cba23, orientation='horizontal')
        cbar23.ax.tick_params(labelsize=7)
        a23.title.set_text('Precision Matrix')
    # end if

    # Plot histogram
    a4 = plt.subplot(gs[:, 3])
    a4.hist(errormag, 50)

    plt.tight_layout()
    plt.savefig(saveprefix + '.figure.png')
# end func

def get_stations_to_process(fds:FederatedASDFDataSet, station_list):
    netsta_list= fds.unique_coordinates.keys()
    result = []
    net_dict = defaultdict(list)

    fds_stations = set([netsta.split('.')[1] for netsta in netsta_list])
    for netsta in netsta_list: net_dict[netsta.split('.')[1]].append(netsta.split('.')[0])

    for station in station_list:
        if(station not in fds_stations):
            raise ValueError('Station {} not found. Aborting..'.format(station))
        else:
            for net in net_dict[station]: result.append('{}.{}'.format(net, station))
        # end if
    # end for

    return result
# end func

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('asdf-source',
                type=click.Path(exists=True))
@click.argument('spec-method',
                required=True,
                type=click.Choice(['single-taper', 'multitaper', 'st', 'cwt2']))
@click.argument('output-path', required=True,
                type=click.Path(exists=True))
@click.option('--win-length', default=200, help="Window length in seconds")
@click.option('--trigger-method', default='zdetect', type=click.Choice(['zdetect', 'stalta']),
              help="Triggering method to use")
@click.option('--trigger-wlen', default=0.5, type=float, help="Triggering window length in seconds if method='zdetect', otherwise"
                                                              " this is window size for short time average in 'stalta'")
@click.option('--trigger-wlen-long', default=30, type=float, help="Window length in seconds for long time average if method='stalta'; "
              "this parameter has no effect if method='zdetect'")
@click.option('--trigger-threshold', default=0.95, type=float, help="Threshold, as a percentile, for the characteristic function to find quiet areas")
@click.option('--trigger-lowpass-value', default=None, type=float, help="Lowpass filter value (Hz) to use for triggering only and not for data "
                                                            "data processing")
@click.option('--trigger-highpass-value', default=None, type=float, help="Highpass filter value (Hz) to use for triggering only and not for data "
                                                             "data processing")
@click.option('--nfreq', default=50, help="Number of frequency bins")
@click.option('--fmin', default=0.1, help="Lowest frequency")
@click.option('--fmax', default=40., help="Highest frequency, which is clipped to the Nyquist value if larger")
@click.option('--lowpass-value', default=None, type=float, help="Lowpass filter value (Hz)")
@click.option('--highpass-value', default=None, type=float, help="Highpass filter value (Hz)")
@click.option('--freq-sampling', default='log',
              type=click.Choice(['linear', 'log']),
              help="Sampling method for frequency bins")
@click.option('--resample-log-freq', is_flag=True,
              help="Resample frequency bins in log-space. Only applicable if --freq-sampling is 'linear' and --spec-method is 'cwt2")
@click.option('--smooth-spectra-method', default='konno-ohmachi',
              type=click.Choice(['konno-ohmachi', 'none']),
              help="Smooth spectra using the Konno & Ohmachi method; 'none' to skip smoothing")
@click.option('--clip-fmin', default=0.3,
              help="Minimum clip frequency for master HVSR-curve. Only applicable if --clip-freq is given")
@click.option('--clip-fmax', default=50.,
              help="Maximum clip frequency for master HVSR-curve. Only applicable if --clip-freq is given")
@click.option('--clip-freq', is_flag=True,
              help="Clip master HVSR-curve to range specified by --clip-fmin and --clip-fmax")
@click.option('--master-curve-method', default='mean',
              type=click.Choice(['mean', 'median', 'geometric-average']),
              help="Method for computing master HVSR curve")
@click.option('--output-prefix', default='$station_name.$spec_method',
              type=str,
              help="Prefix for output file names; default is composed of station name and spectra method")
@click.option('--compute-sparse-covariance', is_flag=True,
              help="Compute sparse covariance")
@click.option('--station-list', default='*', type=str,
              help="'A space-separated list of stations (within quotes) to process; default is '*', which "
                   "processes all available stations.",
              show_default=True)
@click.option('--start-time', default='1970-01-01T00:00:00',
              type=str,
              help="Date and time (in UTC format) to start from; default is year 1900.")
@click.option('--end-time', default='2100-01-01T00:00:00',
              type=str,
              help="Date and time (in UTC format) to stop at; default is year 2100.")
@click.option('--read-buffer-mb', default=2048,
              type=int,
              help="Data read buffer in MB (only applicable for asdf files)")
def process(asdf_source, spec_method, output_path, win_length,
            trigger_method, trigger_wlen, trigger_wlen_long, 
            trigger_threshold, trigger_lowpass_value, trigger_highpass_value,
            nfreq, fmin, fmax, lowpass_value, highpass_value, freq_sampling,
            resample_log_freq, smooth_spectra_method, clip_fmin, clip_fmax,
            clip_freq, master_curve_method, output_prefix,
            compute_sparse_covariance, station_list, start_time, end_time,
            read_buffer_mb):
    """
    ASDF_SOURCE: Path to text file containing paths to ASDF files\n
    SPEC_METHOD: Method for computing spectra; ['single-taper', 'st', 'cwt2']. \n
    OUTPUT_PATH: Output folder \n
    """

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    proc_stations = defaultdict(list)

    if(rank == 0):
        print('\n=== RunBatch Parameters ===\n')
        print(('ASDF-suource:            %s' % asdf_source))
        print(('Spec. Method:            %s' % spec_method))
        print(('Output-path:             %s' % output_path))
        print(('Win. Length:             %d (seconds)' % win_length))
        
        print(('Triggering method:       %s' % trigger_method))
        if(trigger_method=='zdetect'):
            print(('Trigger Win. Length:     %f (seconds)' % trigger_wlen))
        else:
            print(('Trigger Win. Length sta: %f (seconds)' % trigger_wlen))
            print(('Trigger Win. Length lta: %f (seconds)' % trigger_wlen_long))

        print(('Trigger threshold:       %3.2f' % trigger_threshold))
        if(trigger_lowpass_value):
            print(('Trigger lowpass value:   %3.2f' % trigger_lowpass_value))
        if(trigger_highpass_value):
            print(('Trigger highpass value:  %3.2f' % trigger_highpass_value))
        
        print(('nfreq:                   %d' % nfreq))
        print(('fmin:                    %f' % fmin))
        print(('fmax:                    %f' % fmax))
        if(lowpass_value):
            print(('lowpass_value:           %f (Hz)' % lowpass_value))
        if(highpass_value):
            print(('highpass_value:          %f (Hz)' % highpass_value))
        print(('freq_sampling:           %s' % freq_sampling))
        print(('resample_log_freq:       %d' % resample_log_freq))
        print(('smooth_spectra_method:   %s' % smooth_spectra_method))
        print(('clip_freq:               %d' % clip_freq))
        if(clip_freq):
            print(('\tclip_fmin:             %d' % clip_fmin))
            print(('\tclip_fmax:             %d' % clip_fmax))
        print(('Output-prefix:           %s' % output_prefix))
        print(('Start date and time:     %s' % start_time))
        print(('End date and time:       %s' % end_time))
        print('\n===========================\n')
    # end if

    # Removing '-' in options which are meant to avoid having to pass
    # a quoted string at the commandline
    if (spec_method == 'single-taper'): spec_method = 'single taper'
    if (master_curve_method == 'geometric-average'):
        master_curve_method = 'geometric average'

    if(smooth_spectra_method=='none'): smooth_spectra_method = None

    # Check input consistency
    if((resample_log_freq and spec_method != 'cwt2') or
       (resample_log_freq and freq_sampling != 'linear')):
        msg = "Error: Log-space frequency bin resampling can only be applied if --freq-sampling is 'linear' and --spec-method is 'cwt2'. Aborting..\n"
        sys.exit(msg)
    #end if

    # Triggering parameters
    triggering_options = {'method':trigger_method,
                          'trigger_wlen': trigger_wlen,
                          'trigger_wlen_long': trigger_wlen_long,
                          'trigger_threshold': trigger_threshold,
                          'trigger_lowpass_value': trigger_lowpass_value,
                          'trigger_highpass_value': trigger_highpass_value}

    # Get start and end times
    try:
        start_time = UTCDateTime(start_time)
        end_time = UTCDateTime(end_time)
    except:
        raise NameError('Failed to convert start or end time to UTCDateTime')
    # end try

    # check station names
    try:
        if(station_list != '*'):
            station_list = re.findall('\S+', station_list)
        # end if
    except:
        raise NameError('Invalid station-list..')
    # end try


    nfrequencies    = nfreq
    initialfreq     = fmin
    finalfreq       = fmax

    spectra_method  = spec_method
    CLIP_TO_FREQ    = clip_freq
    lowest_freq     = clip_fmin
    highest_freq    = clip_fmax

    fds = FederatedASDFDataSet(asdf_source)
    if(rank == 0):
        stations = get_stations_to_process(fds, station_list)
        print("")
        print('Stations to process:')
        print(stations)
        print("")

        def split_list(lst, npartitions):
            k, m = divmod(len(lst), npartitions)
            return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(npartitions)]
        # end func

        proc_stations = split_list(stations, nproc)
    # end if

    # broadcast workload to all procs
    proc_stations = comm.bcast(proc_stations, root=0)

    for station in proc_stations[rank]:
        print('\nProcessing station %s..\n'%(station))

        spooled_storage = SpooledMatrix(-1, # placeholder
                                        dtype=np.float32, max_size_mb=2048,
                                        prefix=station, dir='/tmp')


        lonlat = fds.unique_coordinates[station]
        net, sta = station.split('.')
        for st in waveform_iterator3c(fds, net, sta, start_time, end_time):
            if(not len(st)): continue # no data found
            else: print(st)

            batch.create_HVSR( st, spooled_storage,
                               spectra_method=spectra_method,
                               spectra_options={'time_bandwidth': 3.5,
                                                'number_of_tapers': None,
                                                'quadratic': False,
                                                'adaptive': True, 'nfft': None,
                                                'taper': 'blackman'},
                               window_length=win_length,
                               bin_samples=nfrequencies,
                               bin_sampling=freq_sampling,
                               f_min=initialfreq,
                               f_max=np.min([finalfreq, (1./st[0].stats.delta)*0.5]),
                               triggering_options=triggering_options,
                               lowpass_value = lowpass_value,
                               highpass_value = highpass_value,
                               resample_log_freq=resample_log_freq,
                               smoothing=smooth_spectra_method )
        # end for
    # end for
# end func

if (__name__ == '__main__'):
    process()
# end if
