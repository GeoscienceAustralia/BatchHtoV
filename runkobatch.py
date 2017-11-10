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

if (len(sys.argv) < 7):
	print "Usage: python htov.py method /path/to/miniSEED/ nfrequencies f_min f_max prefix"
	print "Method is 'single taper', 'st, 'cwt2'"
	exit(0)

nfrequencies = int(sys.argv[3])
initialfreq = float(sys.argv[4])
finalfreq = float(sys.argv[5])
windowlength = 120.0

runprefix = sys.argv[6]

#dr = '/g/data/ha3/Passive/Stavely/'
dr = '/g/data/ha3/Passive/OvernightData/STAVELY/S06PS/Seismometer_data/S0600/S0600miniSEED/'
#dr = 'data/S0600miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/AdventureWay1/aw05/AW05_miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/Overshot1/OV04/OV04_miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/Eulo1/EU13/EU13_miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/EUCLA_PASSIVE/GUINEWARRA/GB12/GB12_miniSEED/'

dr = sys.argv[2]
dr_out = os.path.dirname(os.path.realpath(__file__))+'/'+runprefix+'/';
if not os.path.exists(dr_out):
	os.makedirs(dr_out)

spectra_method=sys.argv[1]
#spectra_method='cwt2'
#spectra_method='st'
#spectra_method='single taper'

st = Stream()
for f in sorted(glob.glob(dr+'*.EH*')):
	print "Loading " + f
	st += read(f)

st.merge(method=1,fill_value=0)
#st = st.slice(st[0].stats.starttime, st[0].stats.starttime+28800)
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


# assumes raw_f is linearly spaced
def mean_maxmin_interp(raw_f,int_f,y):
	# bin bounds for int_f
	spacing = raw_f[1]-raw_f[0] # assumes linear spacing
	ni = int_f.shape[0]
	bint_f = np.zeros(ni+1)
	bint_f[1:ni] = 0.5 * (int_f[1:ni]+int_f[0:ni-1])
	bint_f[0] = bint_f[1] - (bint_f[2] - bint_f[1])
	bint_f[ni] = bint_f[ni-1] + (bint_f[ni-1] - bint_f[ni-2]) # the end bins are not correctly sized, but this will do.
	#ind_f = np.digitize(raw_f,bint_f)
	#ind_count = np.bincount(ind_f)
	# using indices in ind_f
	ycs = np.cumsum(y)
	# y is a function of raw_f. Using the bin boundaries, iterpolate
	# the cumsum(y) at these bin boundaries and use this to integrate y over the bin.
	ycsint = interp1d(raw_f, ycs)
	rsycs = ycsint(bint_f)
	rsy = rsycs[1:] - rsycs[:-1] # integrate via cumsum @ upper bin boundary - cumsum @ lower bin boundary
	rsy /= (bint_f[1:] - bint_f[:-1])/spacing
	#
	# compute the variance
	y2 = ma.empty((len(bint_f)-1,len(y)))
	y2.data[...] = np.log(y)
	#y2.mask = bint_f[:-1,np.newaxis] <= np.digitize(raw_f,bint_f)-1 < bint_f[1:,np.newaxis]
	y2.mask = np.digitize(raw_f,bint_f)-1 != np.arange(ni)[:,np.newaxis]
	#y2.mask = bint_f[:-1,np.newaxis] <= raw_f < bint_f[1:,np.newaxis]
	yint = interp1d(raw_f, np.log(y))
	bin_yint = yint(bint_f)
	bcf = np.bincount(np.digitize(raw_f,bint_f))[1:-1] # clip off first element because digitize will always bin less than the first boundary into bin 0.
	min_bin_yint = np.minimum(bin_yint[1:],bin_yint[:-1])
	max_bin_yint = np.maximum(bin_yint[1:],bin_yint[:-1])
	rsymax = np.exp(np.maximum(y2.max(axis=1).filled(0),max_bin_yint))
	rsymin = np.exp(np.minimum(y2.min(axis=1).filled(99999),min_bin_yint))
	rsyvar = (y2.var(axis=1).filled(0) * bcf + bin_yint[1:] + bin_yint[:-1]) / (bcf + 1.0) # unbiased estimate: subtract 1 from denom when ddof=0 for np.var()
	# resampled y
	return (rsy,rsyvar,rsymin,rsymax)


# assumes raw_f is linearly spaced
def mean_interp(raw_f,int_f,y):
	# bin bounds for int_f
	spacing = raw_f[1]-raw_f[0] # assumes linear spacing
	ni = int_f.shape[0]
	bint_f = np.zeros(ni+1)
	bint_f[1:ni] = 0.5 * (int_f[1:ni]+int_f[0:ni-1])
	bint_f[0] = bint_f[1] - (bint_f[2] - bint_f[1])
	bint_f[ni] = bint_f[ni-1] + (bint_f[ni-1] - bint_f[ni-2]) # the end bins are not correctly sized, but this will do.
	#ind_f = np.digitize(raw_f,bint_f)
	#ind_count = np.bincount(ind_f)
	# using indices in ind_f
	ycs = np.cumsum(y)
	# y is a function of raw_f. Using the bin boundaries, iterpolate
	# the cumsum(y) at these bin boundaries and use this to integrate y over the bin.
	ycsint = interp1d(raw_f, ycs)
	rsycs = ycsint(bint_f)
	rsy = rsycs[1:] - rsycs[:-1] # integrate via cumsum @ upper bin boundary - cumsum @ lower bin boundary
	rsy /= (bint_f[1:] - bint_f[:-1])/spacing
	#
	# compute the variance
	y2 = ma.empty((len(bint_f)-1,len(y)))
	y2.data[...] = y
	#y2.mask = bint_f[:-1,np.newaxis] <= np.digitize(raw_f,bint_f)-1 < bint_f[1:,np.newaxis]
	y2.mask = np.digitize(raw_f,bint_f)-1 != np.arange(ni)[:,np.newaxis]
	#y2.mask = bint_f[:-1,np.newaxis] <= raw_f < bint_f[1:,np.newaxis]
	rsyvar = y2.var(axis=1).filled(0.001)
	# resampled y
	return (rsy,rsyvar)


if RESAMPLE_FREQ:
	# generate frequencies vector
	logfreq_extended = np.zeros(nfrequencies+2)
	c = (1.0/(nfrequencies-1))*np.log10(finalfreq/initialfreq)
	for i in xrange(nfrequencies+2):
		logfreq_extended[i] = initialfreq*(10.0 ** (c*(i-1)))
	logfreq = logfreq_extended[1:-1]
	# interpolate to log spacing
	print "Number of windows computed = " + str(nwindows)
	interp_hvsr_matrix = np.empty((nwindows, nfrequencies))
	interp_hvsr_matrix_var = np.empty((nwindows, nfrequencies))
	sm_matrix = calculate_smoothing_matrix_linlog(logfreq,hvsr_freq,40)
	sm_matrix_log = calculate_smoothing_matrix(logfreq,40)
	deltalin = hvsr_freq[1]-hvsr_freq[0]
	deltalogb = 0.5 * (logfreq_extended[1:] + logfreq_extended[:-1])
	deltalog = deltalogb[1:] - deltalogb[:-1] # same dimensions as logfreq
	resample_bias = np.dot(sm_matrix_log,deltalog) / deltalin
	for i in xrange(nwindows):
		# interp spectrum without rebinning and averaging
		#nint = interp1d(hvsr_freq, hvsr_matrix[i,:])
		#hv_spec2 = nint(logfreq)
		# rebin and average to reduce error in wide bins
		#(hv_spec2, hv_spec2_var, hvs_min, hvs_max) = mean_maxmin_interp(hvsr_freq,logfreq,hvsr_matrix[i,:])
		#interp_hvsr_matrix[i,:] = hv_spec2
		#interp_hvsr_matrix_var[i,:] = hv_spec2_var
		# use k-o smoothing to generate "mean" h/v curve
		interp_hvsr_matrix[i,:] = np.exp(np.dot(sm_matrix,np.log(hvsr_matrix[i,:])))
		# compute variance as squared error to interpolated spectrum value
		nint = interp1d(hvsr_freq, hvsr_matrix[i,:])
		hv_int = nint(logfreq)
		interp_hvsr_matrix_var[i,:] = np.power(np.log(interp_hvsr_matrix[i,:]/hv_int),2)
	hvsr_freq = logfreq
	#master_curve_binlogvar = interp_hvsr_matrix_var.mean(axis=0) / float(nwindows**2)
	master_curve_binlogvar = (interp_hvsr_matrix_var).sum(axis=0) / float(nwindows**2)
else:
	interp_hvsr_matrix = hvsr_matrix
	nfrequencies = hvsr_freq.shape[0]
	initialfreq = hvsr_freq[0]
	finalfreq = hvsr_freq[nfrequencies-1]
	# for non-resample case
	master_curve_binlogvar = np.zeros(hvsr_freq.shape[0])
	resample_bias = np.ones(nfrequencies)

#master_curve = interp_hvsr_matrix.mean(axis=0)
master_curve = np.exp(np.log(interp_hvsr_matrix).mean(axis=0))
#master_curve = np.median(interp_hvsr_matrix,axis=0)
std = (np.log(interp_hvsr_matrix[:][:]) - np.log(master_curve))
errormag = np.zeros(nwindows)
for i in xrange(nwindows):	
	errormag[i] = np.dot(std[i,:],std[i,:].T)
error = np.dot(std.T,std) / float(nwindows-1) # + np.diag(master_curve_binlogvar)
#error /= float(nwindows-1)

sp_model = GraphLassoCV()
sp_model.fit(std)
sp_cov = sp_model.covariance_# + np.diag(master_curve_binlogvar)
sp_prec = sp_model.precision_# + 1.0/np.diag(master_curve_binlogvar)



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
diagerr = np.sqrt(std.var(axis=0)*resample_bias)
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
np.savetxt(saveprefix+'sperror.txt',sp_cov)
np.savetxt(saveprefix+'invsperror.txt',sp_prec)
logdetsperr = np.linalg.slogdet(sp_cov)
print "Log determinant of sparse error matrix: " + str(logdetsperr)
np.savetxt(saveprefix+'logdetsperror.txt',np.array(logdetsperr))

#f,((a1,a2,a3),(cba1,cba2,cba3)) = plt.subplots(2,3,figsize=(18,6))
f = plt.figure(figsize=(18,6))
gs = gridspec.GridSpec(4, 4, height_ratios=[40, 1,40,1])
a1 = plt.subplot(gs[:,0])
a1.plot(hvsr_freq,master_curve,'r')
a1.plot(hvsr_freq,lerr,':g')
a1.plot(hvsr_freq,uerr,':b')
a1.set_yscale('log')
a1.set_xscale('log')
a2 = plt.subplot(gs[0,1])
ca2 = a2.imshow(error,interpolation='nearest')
cba2 = plt.subplot(gs[1,1])
cbar2 = f.colorbar(ca2,cax=cba2,orientation='horizontal')
a3 = plt.subplot(gs[0,2])
ca3 = a3.imshow(np.linalg.inv(error),interpolation='nearest')
cba3 = plt.subplot(gs[1,2])
cbar3 = f.colorbar(ca3,cax=cba3,orientation='horizontal')
#sparse
a22 = plt.subplot(gs[2,1])
ca22 = a22.imshow(sp_cov,interpolation='nearest')
cba22 = plt.subplot(gs[3,1])
cbar22 = f.colorbar(ca22,cax=cba22,orientation='horizontal')
a23 = plt.subplot(gs[2,2])
ca23 = a23.imshow(sp_prec,interpolation='nearest')
cba23 = plt.subplot(gs[3,2])
cbar23 = f.colorbar(ca23,cax=cba23,orientation='horizontal')


a4 = plt.subplot(gs[:,3])
a4.hist(errormag,50)

#plt.show()
#plt.show()
plt.savefig(saveprefix+'_figure.png')
