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

if (len(sys.argv) < 7):
	print "Usage: python htov.py method /path/to/miniSEED/ nfrequencies f_min f_max prefix"
	print "Method is 'single taper', 'st, 'cwt2'"
	exit(0)

nfrequencies = int(sys.argv[3])
initialfreq = float(sys.argv[4])
finalfreq = float(sys.argv[5])
windowlength = float(sys.argv[6])
ko_bandwidth = float(sys.argv[7])

runprefix = sys.argv[8]

dr = sys.argv[2]
dr_out = os.path.dirname(os.path.realpath(__file__))+'/'+runprefix+'/';
if not os.path.exists(dr_out):
	os.makedirs(dr_out)

spectra_method=sys.argv[1]

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

(master_curve, hvsr_freq, error, hvsr_matrix) = batch.create_HVSR(st,spectra_method=spectra_method,
						  spectra_options={'time_bandwidth':3.5,
								   'number_of_tapers':None,
								   'quadratic':False,
								   'adaptive':True,'nfft':None,
								   #'taper':'blackman'},
								   'taper':'nuttall'},
                                                                  #smoothing='konno-ohmachi',
                                                                  smoothing=None,
                                                                  master_curve_method='mean',cutoff_value=0.0,
                                                                  window_length=windowlength,bin_samples=nfrequencies,
                                                                  f_min=initialfreq,f_max=finalfreq)

nwindows = len(hvsr_matrix)

lowest_freq = initialfreq #0.3
highest_freq = finalfreq #50.0
def find_nearest_idx(array,value):
	return (np.abs(array-value)).argmin()


master_curve_full = np.exp(np.log(hvsr_matrix).mean(axis=0))
std_full = (np.log(hvsr_matrix[:][:]) - np.log(master_curve))

if RESAMPLE_FREQ:
	# generate frequencies vector
	logfreq_extended = np.zeros(nfrequencies+2)
	c = (1.0/(nfrequencies-1))*np.log10(finalfreq/initialfreq)
	for i in xrange(nfrequencies+2):
		logfreq_extended[i] = initialfreq*(10.0 ** (c*(i-1)))
	logfreq = logfreq_extended[1:-1]
	print "Logarithmically interpolated frequency bins"
	print logfreq
	# interpolate to log spacing
	print "Number of windows computed = " + str(nwindows)
	interp_hvsr_matrix = np.empty((nwindows, nfrequencies))
	interpolation_variance = np.empty((nwindows,nfrequencies))
	sm_matrix = calculate_smoothing_matrix_linlog(logfreq,hvsr_freq,ko_bandwidth)
	deltalin = hvsr_freq[1]-hvsr_freq[0]
	deltalogb = 0.5 * (logfreq_extended[1:] + logfreq_extended[:-1])
	deltalog = deltalogb[1:] - deltalogb[:-1] # same dimensions as logfreq
	print "Max freq = " + str(hvsr_freq[-1])
	for i in xrange(nwindows):
		# use k-o smoothing to generate "mean" h/v curve
		interp_hvsr_matrix[i,:] = np.exp(np.dot(sm_matrix,np.log(hvsr_matrix[i,:])))
	master_curve = np.exp(np.log(interp_hvsr_matrix).mean(axis=0))
	hvsr_freq = logfreq
	v1 = sm_matrix.sum(axis=1) # nfrequencies x 1
	v2 = (sm_matrix**2).sum(axis=1) # nfrequencies x 1
	# for each interpolated frequency bin, compute sample error of weighted mean (i.e. interpolated value)
	for j in xrange(logfreq.shape[0]):
		# compute variance as squared error to interpolated spectrum value
		#interp_hvsr_matrix_var[i,:] = np.power(np.log(interp_hvsr_matrix[i,:]/hv_int),2)
		# compute sample error of weighted mean
		demeaned = np.log(hvsr_matrix) - np.log(master_curve[j])
		print demeaned.shape
		interpolation_variance[:,j] = np.dot(sm_matrix[j,:],demeaned.T ** 2) / (v1[j] - (v2[j]/v1[j]))
		#interpolation_variance[:,j] = (sm_matrix[j,:].T,np.dot(demeaned.T,demeaned)).sum(axis=0) / (v1[j] - (v2[j]/v1[j]))
		#interpolation_variance[:,j] = np.dot(sm_matrix[j,:],np.dot(np.log(hvsr_matrix) - np.log(master_curve[j]),(np.log(hvsr_matrix) - np.log(master_curve[j])).T)) / (v1[j] - (v2[j]/v1[j]))
		print ("Computed interpolation variance for frequency index " + str(j))
	#master_curve_binlogvar = (interpolation_variance).sum(axis=0) / float(nwindows - 1)
	master_curve_binlogvar = np.dot(interpolation_variance.T,interpolation_variance) / float(nwindows - 1)
	resample_bias = np.diag(master_curve_binlogvar)
	smoothing_variance_operator = sm_matrix ** 2 # currently unused
	std = (np.log(interp_hvsr_matrix[:][:]) - np.log(master_curve))
	stackerror = np.dot(std.T,std) / float(nwindows-1)
	error = stackerror + master_curve_binlogvar
	#diagerr = np.sqrt(std.var(axis=0))
	diagerr = np.sqrt(np.diag(error))
	# errormag doesn't include interpolation variance
	errormag = np.zeros(nwindows)
	for i in xrange(nwindows):	
		errormag[i] = np.dot(std[i,:],std[i,:].T)
else:
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
	#sp_model = GraphLassoCV()
	#sp_model.fit(std)
	#sp_cov = sp_model.covariance_# + np.diag(master_curve_binlogvar)
	#sp_prec = sp_model.precision_# + 1.0/np.diag(master_curve_binlogvar)



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
lerr = np.exp(np.log(master_curve) - diagerr)
uerr = np.exp(np.log(master_curve) + diagerr)
saveprefix = dr_out+runprefix+(spectra_method.replace(' ','_'))

np.savetxt(saveprefix+'resample_bias.txt',resample_bias)
np.savetxt(saveprefix+'hv.txt',np.column_stack((hvsr_freq,master_curve, lerr,uerr)))
np.savetxt(saveprefix+'error.txt',error)
np.savetxt(saveprefix+'inverror.txt',np.linalg.inv(error))
logdeterr = np.linalg.slogdet(error)
print "Log determinant of error matrix: " + str(logdeterr)
np.savetxt(saveprefix+'logdeterror.txt',np.array(logdeterr))

# sparse equivalent
#np.savetxt(saveprefix+'sperror.txt',sp_cov)
#np.savetxt(saveprefix+'invsperror.txt',sp_prec)
#logdetsperr = np.linalg.slogdet(sp_cov)
#print "Log determinant of sparse error matrix: " + str(logdetsperr)
#np.savetxt(saveprefix+'logdetsperror.txt',np.array(logdetsperr))

#f,((a1,a2,a3),(cba1,cba2,cba3)) = plt.subplots(2,3,figsize=(18,6))
f = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(4, 4, height_ratios=[40, 1,40,1])
a1 = plt.subplot(gs[:,0])
a1.plot(hvsr_freq,master_curve,'r')
a1.plot(hvsr_freq,lerr,':g')
a1.plot(hvsr_freq,uerr,':b')
a1.set_yscale('log')
a1.set_xscale('log')
#stacking error
a2 = plt.subplot(gs[0,1])
ca2 = a2.imshow(stackerror,interpolation='nearest')
cba2 = plt.subplot(gs[1,1])
cbar2 = f.colorbar(ca2,cax=cba2,orientation='horizontal')
a3 = plt.subplot(gs[0,2])
ca3 = a3.imshow(np.linalg.inv(stackerror),interpolation='nearest')
cba3 = plt.subplot(gs[1,2])
cbar3 = f.colorbar(ca3,cax=cba3,orientation='horizontal')
#sparse
# interpolation error
int_err = master_curve_binlogvar
a22 = plt.subplot(gs[2,1])
ca22 = a22.imshow(int_err,interpolation='nearest')
cba22 = plt.subplot(gs[3,1])
cbar22 = f.colorbar(ca22,cax=cba22,orientation='horizontal')
a23 = plt.subplot(gs[2,2])
ca23 = a23.imshow(np.linalg.inv(int_err),interpolation='nearest')
cba23 = plt.subplot(gs[3,2])
cbar23 = f.colorbar(ca23,cax=cba23,orientation='horizontal')


a4 = plt.subplot(gs[:,3])
a4.hist(errormag,50)

#plt.show()
#plt.show()
plt.savefig(saveprefix+'_figure.png')
