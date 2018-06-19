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

CLIP_TO_FREQ = False
RESAMPLE_FREQ = True
APPLY_RESAMPLE_BIAS = False
APPLY_SMOOTHING_VARIANCE = True

if (len(sys.argv) < 10):
	print "Usage: python htov.py method /path/to/miniSEED/ nfrequencies f_min f_max window_length ko_bandwidth log|lin prefix w0"
	print "Method is 'single taper', 'st, 'cwtmlpy', 'cwtlog'"
	exit(0)

nfrequencies = int(sys.argv[3])
initialfreq = float(sys.argv[4])
finalfreq = float(sys.argv[5])
windowlength = float(sys.argv[6])
ko_bandwidth = float(sys.argv[7])
sampling_type = sys.argv[8]
runprefix = sys.argv[9]
if len(sys.argv) > 10:
	w0 = int(sys.argv[10])
else:
	w0 = 8
spectra_method=sys.argv[1]
dr = sys.argv[2]
demean_hvsr = False
shift_down = False
diff_hvsr = True

if sampling_type == 'linear' or spectra_method=='cwtlog':
	RESAMPLE_FREQ = False
else:
	RESAMPLE_FREQ = True


dr_out = os.path.dirname(os.path.realpath(__file__))+'/'+runprefix+'/';
if not os.path.exists(dr_out):
	os.makedirs(dr_out)

saveprefix = dr_out+runprefix+(spectra_method.replace(' ','_'))+"_diff"

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
                                                                  smoothing='konno-ohmachi',smoothing_constant=ko_bandwidth,
                                                                  #smoothing=None,
                                                                  master_curve_method='mean',cutoff_value=0.0,
                                                                  window_length=windowlength,bin_samples=nfrequencies,
                                                                  f_min=initialfreq,f_max=finalfreq,frequencies=freq_bins,w0=w0)

nwindows = len(hvsr_matrix)

lowest_freq = initialfreq #0.3
highest_freq = finalfreq #50.0
def find_nearest_idx(array,value):
	return (np.abs(array-value)).argmin()

# compute statistics on the difference between successive data points
hvsr_matrix = (hvsr_matrix[:,1:] - hvsr_matrix[:,:-1]) / (hvsr_freq[1:]-hvsr_freq[:-1])
hvsr_freq=0.5*(hvsr_freq[1:]+hvsr_freq[:-1])
	
print "HVSR matrix"
print hvsr_matrix

master_curve_full =hvsr_matrix.mean(axis=0)
std_full = hvsr_matrix[:][:] - master_curve_full

interp_hvsr_matrix = hvsr_matrix
nfrequencies = hvsr_freq.shape[0]
initialfreq = hvsr_freq[0]
finalfreq = hvsr_freq[nfrequencies-1]
# for non-resample case
master_curve_binlogvar = np.zeros(hvsr_freq.shape[0])
resample_bias = np.ones(nfrequencies)
master_curve = master_curve_full
std = std_full
diagerr = np.sqrt(std.var(axis=0))
errormag = np.zeros(nwindows)
for i in xrange(nwindows):	
	errormag[i] = np.dot(std[i,:],std[i,:].T)
error = np.dot(std.T,std) / float(nwindows-1) # + np.diag(master_curve_binlogvar)

if CLIP_TO_FREQ:
	lclip = find_nearest_idx(hvsr_freq,lowest_freq)
	uclip = find_nearest_idx(hvsr_freq,highest_freq)
	master_curve=master_curve[lclip:uclip]
	hvsr_freq=hvsr_freq[lclip:uclip]
	error=error[lclip:uclip,lclip:uclip]

print "Master curve shape: " + str(master_curve.shape)
print master_curve
print "Frequencies shape: " + str(hvsr_freq.shape)
print hvsr_freq
print "Error shape: " + str(error.shape)
print error

#diagerr = np.sqrt(np.diag(error))
#diagerr = np.sqrt(std.var(axis=0) + master_curve_binlogvar)
lerr = master_curve - diagerr
uerr = master_curve + diagerr

np.savetxt(saveprefix+'resample_bias.txt',resample_bias)
#np.savetxt(saveprefix+'hv.txt',np.column_stack((hvsr_freq,master_curve, lerr,uerr)))
np.savetxt(saveprefix+'hv.txt',np.column_stack((hvsr_freq,master_curve, diagerr)))
np.savetxt(saveprefix+'error.txt',error)
np.savetxt(saveprefix+'inverror.txt',np.linalg.inv(error))


#f,((a1,a2,a3),(cba1,cba2,cba3)) = plt.subplots(2,3,figsize=(18,6))
f = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(4, 4, height_ratios=[40, 1,40,1])
a1 = plt.subplot(gs[:,0])
for i in xrange(hvsr_matrix.shape[0]):
	a1.plot(hvsr_freq,hvsr_matrix[i,:],'k', alpha=0.1)
a1.plot(hvsr_freq,master_curve,'r')
a1.plot(hvsr_freq,lerr,':g')
a1.plot(hvsr_freq,uerr,':b')
a1.set_yscale('linear')
a1.set_xscale('log')


a4 = plt.subplot(gs[:,3])
a4.hist(errormag,50)

#plt.show()
#plt.show()
plt.savefig(saveprefix+'_figure.png')
