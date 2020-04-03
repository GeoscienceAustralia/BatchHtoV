from mpi4py import MPI
from htov import batch
from htov.utils import StreamAdapter
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

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('spec-method',
                required=True,
                type=click.Choice(['single-taper', 'multitaper', 'st', 'cwt2']))
@click.argument('data-path',
                type= click.Path(exists=True) or click.File('r'))
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
@click.option('--station-names', default='*', type=str,
              help="Stations name(s) (space-delimited) to process; default is '*', which processes all available stations.")
@click.option('--start-time', default='1970-01-01T00:00:00',
              type=str,
              help="Date and time (in UTC format) to start from; default is year 1900.")
@click.option('--end-time', default='2100-01-01T00:00:00',
              type=str,
              help="Date and time (in UTC format) to stop at; default is year 2100.")
@click.option('--read-buffer-mb', default=2048,
              type=int,
              help="Data read buffer in MB (only applicable for asdf files)")
def process(spec_method, data_path, output_path, win_length,
            trigger_method, trigger_wlen, trigger_wlen_long, 
            trigger_threshold, trigger_lowpass_value, trigger_highpass_value,
            nfreq, fmin, fmax, lowpass_value, highpass_value, freq_sampling,
            resample_log_freq, smooth_spectra_method, clip_fmin, clip_fmax,
            clip_freq, master_curve_method, output_prefix,
            compute_sparse_covariance, station_names, start_time, end_time,
            read_buffer_mb):
    """
    SPEC_METHOD: Method for computing spectra; ['single-taper', 'st', 'cwt2']. \n
    DATA_PATH: Path to miniseed files or an ASDF file\n
    OUTPUT_PATH: Output folder \n
    """

    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    proc_stations = defaultdict(list)

    if(rank == 0):
        print('\n=== RunBatch Parameters ===\n')
        print(('Spec. Method:            %s' % spec_method))
        print(('Data-path:               %s' % data_path))
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
        if(station_names != '*'):
            station_names = set([s.upper().replace("'", "") for s in station_names.split(' ')])
    except:
        raise NameError('Invalid station-names list..')
    # end try


    nfrequencies    = nfreq
    initialfreq     = fmin
    finalfreq       = fmax
    dr              = data_path

    spectra_method  = spec_method
    CLIP_TO_FREQ    = clip_freq
    lowest_freq     = clip_fmin
    highest_freq    = clip_fmax

    sa = StreamAdapter(data_path, buffer_size_in_mb=read_buffer_mb)
    stations = sa.getStationNames()

    if(rank == 0):
        print("")
        print('Stations Found:')
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

        # skip stations based on user input
        if(station_names=='*'): pass
        else:
            if station not in station_names: continue
        # end if

        print('\nProcessing station %s..\n'%(station))

        st = sa.getStream(station, start_time=start_time, end_time=end_time)

        lonlat = sa.getLonLat(station)

        if(not len(st)): continue # no data found
        else: print(st)

        (master_curve, hvsr_freq,
         error, hvsr_matrix) = batch.create_HVSR( st, spectra_method=spectra_method,
                                                  spectra_options={'time_bandwidth': 3.5,
                                                                   'number_of_tapers': None,
                                                                   'quadratic': False,
                                                                   'adaptive': True, 'nfft': None,
                                                                   'taper': 'blackman'},
                                                  master_curve_method=master_curve_method,
                                                  cutoff_value=0.0,
                                                  window_length=win_length,
                                                  bin_samples=nfrequencies,
                                                  bin_sampling=freq_sampling,
                                                  f_min=initialfreq,
                                                  f_max=np.min([finalfreq, (1./st[0].stats.delta)*0.5]),
                                                  triggering_options=triggering_options,
                                                  lowpass_value = lowpass_value,
                                                  highpass_value = highpass_value,
                                                  resample_log_freq=resample_log_freq,
                                                  smoothing=smooth_spectra_method)
        if(master_curve is None): sys.exit('Failed to process data.')

        nwindows = len(hvsr_matrix)

        def find_nearest_idx(array, value):
            return (np.abs(array - value)).argmin()


        std = (np.log1p(hvsr_matrix[:][:]) - np.log1p(master_curve))
        errormag = np.zeros(nwindows)
        for i in range(nwindows):
            errormag[i] = np.dot(std[i, :], std[i, :].T)
        error = np.dot(std.T, std)
        error /= float(nwindows - 1)

        if(compute_sparse_covariance):
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
        #end if

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
        if(output_prefix=='$station_name.$spec_method'):
            tag = '%s.%s'%(station, spectra_method.replace(' ', '_'))
        else:
            tag = output_prefix
        # end if
        saveprefix = os.path.join(output_path, tag)

        np.savetxt(saveprefix + '.hv.txt', np.column_stack((hvsr_freq, master_curve, lerr, uerr)))
        np.savetxt(saveprefix + '.std.txt', np.std(hvsr_matrix, axis=0))
        if(len(lonlat)): np.savetxt(saveprefix + '.lonlat.txt', np.array(lonlat).T)
        np.savetxt(saveprefix + '.error.txt', error)
        np.savetxt(saveprefix + '.inverror.txt', np.linalg.inv(error))
        logdeterr = np.linalg.slogdet(error)
        print("Log determinant of error matrix: " + str(logdeterr))
        np.savetxt(saveprefix + '.logdeterror.txt', np.array(logdeterr))

        if(compute_sparse_covariance):
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

        if(compute_sparse_covariance):
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
    # end for
# end func

if (__name__ == '__main__'):
    process()
# end if
