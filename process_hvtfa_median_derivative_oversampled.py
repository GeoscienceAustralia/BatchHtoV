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


if (len(sys.argv) < 9):
	print "Usage: python htov.py method /path/to/miniSEED/ nfrequencies f_min f_max window_length ko_bandwidths log|lin prefix w0"
	print "Method is 'raydeccwtlog'"
	exit(0)

spectra_method='cwtlog' #sys.argv[1]
dr = sys.argv[1]
nfrequencies = int(sys.argv[2])
initialfreq = float(sys.argv[3])
finalfreq = float(sys.argv[4])
windowlength = float(sys.argv[5])
ko_bandwidths = [float(kob) for kob in sys.argv[6].split(',')]
ko_bandwidthstr = ' '.join([kob for kob in sys.argv[6].split(',')])
nbws = len(ko_bandwidths)
sampling_type = sys.argv[7]
runprefix = sys.argv[8]
if len(sys.argv) > 9:
	w0 = int(sys.argv[9])
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

# chop stream to debug
#st.trim(starttime=st[0].stats.starttime,endtime=st[0].stats.starttime+800)
#print "Post trim"
#print st

freq_bins = None

# set linearly spaced frequency bins
#sr = 1.0/st[0].stats.delta
#spacing = sr/wl
#f_min_idx = math.floor(f_min / spacing)
#f_max_idx = math.ceil(f_max / spacing)
#nf = f_max_idx - f_min_idx + 1
#f_min = f_min_idx + spacing
#f_max = f_max_idx + spacing
#freqs = np.linspace(f_min_idx, f_max_idx, nf) * spacing

oversample = 4

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
                                                                  master_curve_method='median',cutoff_value=0.0,
                                                                  window_length=windowlength,bin_samples=nfrequencies * oversample,
                                                                  f_min=initialfreq,f_max=finalfreq,w0=w0,
)

nwindows = len(hvsr_matrix)
print "nwindows = " + str(nwindows)

master_curve_full = np.exp(np.mean(np.log(hvsr_matrix),axis=0))
error = np.sqrt(np.sum((np.log(hvsr_matrix[:][:]) - np.log(master_curve_full)) ** 2,axis=0) / float(nwindows-1))

# detect and remove outlier windows
# Simple 3 sigma test. Sum residuals over window, if above 3 sigma, reject.
outlier_mask = np.ones(hvsr_matrix.shape[0],dtype=bool)
log_hvsr_matrix = np.log(hvsr_matrix)
hvsr_matrix_masked = np.ma.array(log_hvsr_matrix, mask=np.logical_not(np.broadcast_to(outlier_mask[:, None],log_hvsr_matrix.shape)))
while True:
	error = np.sqrt(np.sum((hvsr_matrix_masked[:][:] - np.log(master_curve_full)) ** 2,axis=0) / float(outlier_mask.sum()-1))
	residuals = ((log_hvsr_matrix[:][:] - np.log(master_curve_full))/error) ** 2 * outlier_mask[:,None]
	#res_window_sum = np.sqrt(residuals.sum(axis=1) / master_curve.shape[0])
	#res_window_sum = np.sqrt(np.quantile(residuals,0.75,axis=1))
	res_window_sum = np.sqrt(np.median(residuals,axis=1))
	if np.any(res_window_sum > 1):
		# mask the max, then recompute master curve
		outlier_mask[np.argmax(res_window_sum)] = False
		#hvsr_matrix_masked = np.ma.array(log_hvsr_matrix, mask=np.logical_not(outlier_mask[:, None]))
		hvsr_matrix_masked = np.ma.array(log_hvsr_matrix, mask=np.logical_not(np.broadcast_to(outlier_mask[:, None],log_hvsr_matrix.shape)))
		master_curve_full = np.exp(np.ma.mean(hvsr_matrix_masked,axis=0))
	else:
		break
# using the mask we delete rows from hvsr_matrix
hvsr_matrix = hvsr_matrix[outlier_mask,:]
nwindows = len(hvsr_matrix)
print "Non-outlier nwindows = " + str(nwindows)
		
# generate log frequencies
logfreq_extended = np.zeros(nfrequencies+2)
c = (1.0/(nfrequencies-1))*np.log10(finalfreq/initialfreq)
for i in xrange(nfrequencies+2):
	logfreq_extended[i] = initialfreq*(10.0 ** (c*(i-1)))
logfreq = logfreq_extended[1:-1]
print "Logarithmically interpolated frequency bins"
print logfreq
	

dhvsr_freq=0.5*(logfreq[1:]+logfreq[:-1])
x1=logfreq[:-2]
x2=logfreq[1:-1]
x3=logfreq[2:]
ddhvsr_freq=0.25*((x3+x2)+(x2+x1))
all_hvsr = np.empty((nfrequencies,2 * nbws + 1))
all_dhvsr = np.empty((nfrequencies-1,2 * nbws + 1))
all_ddhvsr = np.empty((nfrequencies-2,2 * nbws + 1))
all_hvsr[:,0]=logfreq
all_dhvsr[:,0]=dhvsr_freq
all_ddhvsr[:,0]=ddhvsr_freq
# interpolate to log spacing
print "Number of windows computed = " + str(nwindows)
bwi=0


for ko_bandwidth in ko_bandwidths:
	smooth_hvsr_matrix = np.empty((nwindows, nfrequencies))
	#sm_matrix = calculate_smoothing_matrix(hvsr_freq,ko_bandwidth)
	sm_matrix = calculate_smoothing_matrix_linlog(logfreq,hvsr_freq,ko_bandwidth)
	print "Max freq = " + str(logfreq[-1])
	for i in xrange(nwindows):
		# use k-o smoothing to generate "mean" h/v curve
		smooth_hvsr_matrix[i,:] = np.dot(sm_matrix,hvsr_matrix[i,:])
	master_curve = np.exp(np.median(np.log(smooth_hvsr_matrix),axis=0))
	# for each interpolated frequency bin, compute sample error of weighted mean (i.e. interpolated value)

	dhvsr_matrix = (smooth_hvsr_matrix[:,1:] - smooth_hvsr_matrix[:,:-1]) / (logfreq[1:]-logfreq[:-1])
	dhvsr_curve = np.median(dhvsr_matrix,axis=0)
	dhvsr_deviation = np.absolute(dhvsr_matrix - dhvsr_curve)
	dhvsr_variance = np.dot(dhvsr_deviation.T,dhvsr_deviation) / float(nwindows - 1)
	dhvsr_totalvariance = dhvsr_variance

	# compute statistics on the difference between successive data points
	y1=smooth_hvsr_matrix[:,:-2]
	y2=smooth_hvsr_matrix[:,1:-1]
	y3=smooth_hvsr_matrix[:,2:]
	ddhvsr_matrix = 2 * (y1/((x2-x1)*(x3-x1)) - y2/((x3-x2)*(x2-x1)) + y3/((x3-x2)*(x3-x1)))
	ddhvsr_curve = np.median(ddhvsr_matrix,axis=0)
	ddhvsr_deviation = np.absolute(ddhvsr_matrix - ddhvsr_curve)
	ddhvsr_variance = np.dot(ddhvsr_deviation.T,ddhvsr_deviation) / float(nwindows - 1)
	ddhvsr_totalvariance = ddhvsr_variance
	
	std = (np.log(smooth_hvsr_matrix[:][:]) - np.log(master_curve))
	stackerror = np.dot(std.T,std) / float(nwindows-1)
	error = stackerror 
	diagerr = np.sqrt(np.diag(error))
	errormag = np.zeros(nwindows)
	for i in xrange(nwindows):	
		errormag[i] = np.dot(std[i,:],std[i,:].T)
	
	print "Master curve shape: " + str(master_curve.shape)
	print master_curve
	print "Frequencies shape: " + str(logfreq.shape)
	print logfreq
	print "Error shape: " + str(error.shape)
	print error
	
	lerr = np.exp(np.log(master_curve) - diagerr)
	uerr = np.exp(np.log(master_curve) + diagerr)

	saveprefix = dr_out+runprefix+(spectra_method.replace(' ','_'))+'_'+str(ko_bandwidth)+'_'
	
	np.savetxt(saveprefix+'hv.txt',np.column_stack((logfreq,master_curve, diagerr)))
	np.savetxt(saveprefix+'error.txt',error)
	np.savetxt(saveprefix+'inverror.txt',np.linalg.inv(error))
	logdeterr = np.linalg.slogdet(error)
	print "Log determinant of error matrix: " + str(logdeterr)
	np.savetxt(saveprefix+'logdeterror.txt',np.array(logdeterr))

	ddiagerr = np.sqrt(np.diag(dhvsr_totalvariance))
	dlerr = dhvsr_curve - ddiagerr
	duerr = dhvsr_curve + ddiagerr
	
	dddiagerr = np.sqrt(np.diag(ddhvsr_totalvariance))
	ddlerr = ddhvsr_curve - dddiagerr
	dduerr = ddhvsr_curve + dddiagerr

	np.savetxt(saveprefix+'dhv.txt',np.column_stack((dhvsr_freq,dhvsr_curve, ddiagerr)))
	np.savetxt(saveprefix+'ddhv.txt',np.column_stack((ddhvsr_freq,ddhvsr_curve, dddiagerr)))

	all_hvsr[:,1+bwi]=master_curve
	all_hvsr[:,1+bwi+nbws]=diagerr
	all_dhvsr[:,1+bwi]=dhvsr_curve
	all_dhvsr[:,1+bwi+nbws]=ddiagerr
	all_ddhvsr[:,1+bwi]=ddhvsr_curve
	all_ddhvsr[:,1+bwi+nbws]=dddiagerr
	bwi+=1
	
	f = plt.figure(figsize=(18,6))
	gs = gridspec.GridSpec(1, 4)
	a1 = plt.subplot(gs[0,0])
	for i in xrange(smooth_hvsr_matrix.shape[0]):
		a1.plot(logfreq,smooth_hvsr_matrix[i,:],'k', alpha=0.1)
	a1.plot(logfreq,master_curve,'r')
	a1.plot(logfreq,lerr,':g')
	a1.plot(logfreq,uerr,':b')
	a1.set_yscale('log')
	a1.set_xscale('log')

	a2 = plt.subplot(gs[0,1])
	for i in xrange(dhvsr_matrix.shape[0]):
		a2.plot(dhvsr_freq,dhvsr_matrix[i,:],'k', alpha=0.1)
	a2.plot(dhvsr_freq,dhvsr_curve,'r')
	a2.plot(dhvsr_freq,dlerr,':g')
	a2.plot(dhvsr_freq,duerr,':b')
	a2.set_xscale('log')
	
	a3 = plt.subplot(gs[0,2])
	for i in xrange(ddhvsr_matrix.shape[0]):
		a3.plot(ddhvsr_freq,ddhvsr_matrix[i,:],'k', alpha=0.1)
	a3.plot(ddhvsr_freq,ddhvsr_curve,'r')
	a3.plot(ddhvsr_freq,ddlerr,':g')
	a3.plot(ddhvsr_freq,dduerr,':b')
	a3.set_xscale('log')
	
	a4 = plt.subplot(gs[0,3])
	a4.hist(errormag,50)
	
	plt.savefig(saveprefix+'_figure.png')

np.savetxt(saveprefix+'allhv.txt',all_hvsr, header=ko_bandwidthstr, comments='')
np.savetxt(saveprefix+'alldhv.txt',all_dhvsr, header=ko_bandwidthstr, comments='')
np.savetxt(saveprefix+'allddhv.txt',all_ddhvsr, header=ko_bandwidthstr, comments='')

