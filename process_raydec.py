import batch
from htov import *
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
from sklearn.covariance import GraphLassoCV, ledoit_wolf
import numpy.ma as ma
from konno_ohmachi_smoothing import calculate_smoothing_matrix_linlog 


if (len(sys.argv) < 8):
	print "Usage: python htov.py method /path/to/miniSEED/ nfrequencies f_min f_max window_length log|lin prefix w0"
	print "Method is 'raydeccwtlog'"
	exit(0)

spectra_method='raydeccwtlog2' #sys.argv[1]
dr = sys.argv[1]
nfrequencies = int(sys.argv[2])
initialfreq = float(sys.argv[3])
finalfreq = float(sys.argv[4])
windowlength = float(sys.argv[5])
sampling_type = sys.argv[6]
runprefix = sys.argv[7]
if len(sys.argv) > 8:
	w0 = int(sys.argv[8])
else:
	w0 = 16

dr_out = os.path.dirname(os.path.realpath(__file__))+'/'+runprefix+'/';
if not os.path.exists(dr_out):
	os.makedirs(dr_out)

saveprefix = dr_out+runprefix+(spectra_method.replace(' ','_'))

st = Stream()
if dr[-1]=='/':
	#for f in sorted(glob.glob(dr+'*.EH*')):
	for f in sorted(glob.glob(dr+'*.*')):
		print "Loading " + f
		st += read(f)
else:
	st += read(dr)

st.merge(method=1,fill_value=0)
#st = st.slice(st[0].stats.starttime, st[0].stats.starttime+28800)
print st
print "stream length = " + str(len(st))

freq_bins = None
logfreq_extended = None
logfreq = None

(master_curve, hvsr_freq, error, hvsr_matrix) = batch.create_HVSR(st,spectra_method=spectra_method,
						  spectra_options={'time_bandwidth':3.5,
								   'number_of_tapers':None,
								   'quadratic':False,
								   'adaptive':True,'nfft':None,
								   #'taper':'blackman'},
								   'taper':'nuttall'},
                                                                  #smoothing='konno-ohmachi',smoothing_constant=ko_bandwidth,
                                                                  smoothing=None,
                                                                  master_curve_method='mean',cutoff_value=0.0,
                                                                  window_length=windowlength,bin_samples=nfrequencies,
                                                                  f_min=initialfreq,f_max=finalfreq,frequencies=freq_bins,w0=w0)

nwindows = len(hvsr_matrix)

master_curve_full = np.exp(np.log(hvsr_matrix).mean(axis=0))
error = np.sqrt(np.sum((np.log(hvsr_matrix[:][:]) - np.log(master_curve)) ** 2,axis=0) / float(nwindows-1))

print "Master curve shape: " + str(master_curve.shape)
print master_curve
print "Frequencies shape: " + str(hvsr_freq.shape)
print hvsr_freq
print "Error shape: " + str(error.shape)
print error

#diagerr = np.sqrt(np.diag(error))
#diagerr = np.sqrt(std.var(axis=0) + master_curve_binlogvar)

lerr = np.exp(np.log(master_curve) - error)
uerr = np.exp(np.log(master_curve) + error)

np.savetxt(saveprefix+'hv.txt',np.column_stack((hvsr_freq,master_curve, error)))
np.savetxt(saveprefix+'error.txt',error)

f = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(1,1)
a1 = plt.subplot(gs[:,0])
for i in xrange(hvsr_matrix.shape[0]):
	a1.plot(hvsr_freq,hvsr_matrix[i,:],'k', alpha=0.1)
a1.plot(hvsr_freq,master_curve,'r')
a1.plot(hvsr_freq,lerr,':g')
a1.plot(hvsr_freq,uerr,':b')
a1.grid(True)
a1.set_yscale('log')
a1.set_xscale('log')

#plt.show()
plt.savefig(saveprefix+'_figure.png')

