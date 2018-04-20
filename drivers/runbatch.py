from htov import batch
import glob, os
import obspy
from obspy.core import Stream, UTCDateTime
from obspy import read
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy
import calendar
import math
import sys
import click
from sklearn.covariance import GraphLassoCV, ledoit_wolf

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('spec-method',
                required=True,
                type=click.Choice(['single-taper', 'st', 'cwt2']))
@click.argument('data-path',
                type=click.Path(exists=True))
@click.argument('output-path', required=True,
                type=click.Path(exists=True))
@click.option('--win-length', default=100, help="Window length in seconds")
@click.option('--zdetect-win-length', default=40, help="Zdetect window length in samples")
@click.option('--zdetect-threshold', default=0.95, help="Threshold, as a percentile, for the characteristic function to find quiet areas")
@click.option('--nfreq', default=50, help="Number of frequency bins")
@click.option('--fmin', default=0.4, help="Minimum frequency")
@click.option('--fmax', default=50., help="Minimum frequency")
@click.option('--freq-sampling', default='linear',
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
@click.option('--output-prefix', default='', help="Prefix for output file names")
def process(spec_method, data_path, output_path, win_length,
            zdetect_win_length, zdetect_threshold, nfreq, fmin,
            fmax, freq_sampling, resample_log_freq, smooth_spectra_method,
            clip_fmin, clip_fmax, clip_freq, master_curve_method,
            output_prefix):
    """
    SPEC_METHOD: Method for computing spectra; ['single-taper', 'st', 'cwt2']. \n
    DATA_PATH: Path to miniseed files \n
    OUTPUT_PATH: Output folder \n
    """

    print('\n=== RunBatch Parameters ===\n')
    print('Spec. Method:            %s' % spec_method)
    print('Data-path:               %s' % data_path)
    print('Output-path:             %s' % output_path)
    print('Win. Length:             %d (seconds)' % win_length)
    print('Zdetect Win. Length:     %d (samples)' % zdetect_win_length)
    print('Zdetect threshold:       %d' % zdetect_threshold)
    print('nfreq:                   %d' % nfreq)
    print('fmin:                    %f' % fmin)
    print('fmax:                    %f' % fmax)
    print('freq_sampling:           %s' % freq_sampling)
    print('resample_log_freq:       %d' % resample_log_freq)
    print('smooth_spectra_method:   %s' % smooth_spectra_method)
    print('clip_freq:               %d' % clip_freq)
    if(clip_freq):
        print('\tclip_fmin:         %d' % clip_fmin)
        print('\tclip_fmax:         %d' % clip_fmax)
    print('Output-prefix:           %s' % output_prefix)
    print('\n===========================\n')

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

    nfrequencies    = nfreq
    initialfreq     = fmin
    finalfreq       = fmax
    dr              = data_path
    runprefix       = output_prefix

    spectra_method  = spec_method
    CLIP_TO_FREQ    = clip_freq
    lowest_freq     = clip_fmin
    highest_freq    = clip_fmax

    st = Stream()
    for f in sorted(glob.glob(dr + '/*.EH*')):
        print "Loading " + f
        st += read(f)
    #end for

    st.merge(method=1, fill_value=0)
    st = st.slice(st[0].stats.starttime, st[0].stats.starttime+3600)
    print "stream length = " + str(len(st))

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
                                              f_min=initialfreq, f_max=finalfreq,
                                              zdetector_window_length=zdetect_win_length,
                                              resample_log_freq=resample_log_freq,
                                              smoothing=smooth_spectra_method,
                                              threshold=zdetect_threshold )
    if(master_curve is None): sys.exit('Failed to process data.')

    nwindows = len(hvsr_matrix)

    def find_nearest_idx(array, value):
        return (np.abs(array - value)).argmin()


    std = (np.log1p(hvsr_matrix[:][:]) - np.log1p(master_curve))
    errormag = np.zeros(nwindows)
    for i in xrange(nwindows):
        errormag[i] = np.dot(std[i, :], std[i, :].T)
    error = np.dot(std.T, std)
    error /= float(nwindows - 1)

    print "Computing model covariance"
    sp_model = GraphLassoCV()
    sp_model.fit(std)
    sp_cov = sp_model.covariance_
    sp_prec = sp_model.precision_

    if CLIP_TO_FREQ:
        lclip = find_nearest_idx(hvsr_freq, lowest_freq)
        uclip = find_nearest_idx(hvsr_freq, highest_freq)
        master_curve = master_curve[lclip:uclip]
        hvsr_freq = hvsr_freq[lclip:uclip]
        error = error[lclip:uclip, lclip:uclip]
    #end if

    print "Master curve shape: " + str(master_curve.shape)
    print master_curve
    print "Frequencies shape: " + str(hvsr_freq.shape)
    print hvsr_freq
    print "Error shape: " + str(error.shape)
    print error

    diagerr = np.sqrt(np.diag(error))
    lerr = np.exp(np.log(master_curve) - diagerr)
    uerr = np.exp(np.log(master_curve) + diagerr)

    tag = ''
    if(runprefix!=''): tag = runprefix+'.'
    saveprefix = os.path.join(output_path, tag + (spectra_method.replace(' ', '_')))

    np.savetxt(saveprefix + '.hv.txt', np.column_stack((hvsr_freq, master_curve, lerr, uerr)))
    np.savetxt(saveprefix + '.error.txt', error)
    np.savetxt(saveprefix + '.inverror.txt', np.linalg.inv(error))
    logdeterr = np.linalg.slogdet(error)
    print "Log determinant of error matrix: " + str(logdeterr)
    np.savetxt(saveprefix + '.logdeterror.txt', np.array(logdeterr))

    # sparse equivalent
    np.savetxt(saveprefix + '.sperror.txt', sp_cov)
    np.savetxt(saveprefix + '.invsperror.txt', sp_prec)
    logdetsperr = np.linalg.slogdet(sp_cov)
    print "Log determinant of sparse error matrix: " + str(logdetsperr)
    np.savetxt(saveprefix + '.logdetsperror.txt', np.array(logdetsperr))

    f = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(4, 4, height_ratios=[40, 1, 40, 1])
    a1 = plt.subplot(gs[:, 0])
    a1.plot(hvsr_freq, master_curve, 'r')
    a1.plot(hvsr_freq, lerr, ':g')
    a1.plot(hvsr_freq, uerr, ':b')
    a1.set_yscale('log')
    a1.set_xscale('log')
    a2 = plt.subplot(gs[0, 1])
    ca2 = a2.imshow(error, interpolation='nearest')
    cba2 = plt.subplot(gs[1, 1])
    cbar2 = f.colorbar(ca2, cax=cba2, orientation='horizontal')
    a3 = plt.subplot(gs[0, 2])
    ca3 = a3.imshow(np.linalg.inv(error), interpolation='nearest')
    cba3 = plt.subplot(gs[1, 2])
    cbar3 = f.colorbar(ca3, cax=cba3, orientation='horizontal')
    # sparse
    a22 = plt.subplot(gs[2, 1])
    ca22 = a22.imshow(sp_cov, interpolation='nearest')
    cba22 = plt.subplot(gs[3, 1])
    cbar22 = f.colorbar(ca22, cax=cba22, orientation='horizontal')
    a23 = plt.subplot(gs[2, 2])
    ca23 = a23.imshow(sp_prec, interpolation='nearest')
    cba23 = plt.subplot(gs[3, 2])
    cbar23 = f.colorbar(ca23, cax=cba23, orientation='horizontal')

    a4 = plt.subplot(gs[:, 3])
    a4.hist(errormag, 50)

    plt.savefig(saveprefix + '.figure.png')
# end func

if (__name__ == '__main__'):
    process()
# end if
