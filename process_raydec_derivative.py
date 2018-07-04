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

spectra_method='raydeccwtlog2' #sys.argv[1]
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

std_full = (np.log(hvsr_matrix[:][:]) - np.log(master_curve))

dhvsr_freq=0.5*(hvsr_freq[1:]+hvsr_freq[:-1])
x1=hvsr_freq[:-2]
x2=hvsr_freq[1:-1]
x3=hvsr_freq[2:]
ddhvsr_freq=0.25*((x3+x2)+(x2+x1))
all_hvsr = np.empty((nfrequencies,2 * nbws + 1))
all_dhvsr = np.empty((nfrequencies-1,2 * nbws + 1))
all_ddhvsr = np.empty((nfrequencies-2,2 * nbws + 1))
all_hvsr[:,0]=hvsr_freq
all_dhvsr[:,0]=dhvsr_freq
all_ddhvsr[:,0]=ddhvsr_freq
# interpolate to log spacing
print "Number of windows computed = " + str(nwindows)
bwi=0
for ko_bandwidth in ko_bandwidths:
	smooth_hvsr_matrix = np.empty((nwindows, nfrequencies))
	sm_matrix = calculate_smoothing_matrix(hvsr_freq,ko_bandwidth)
	print "Max freq = " + str(hvsr_freq[-1])
	for i in xrange(nwindows):
		# use k-o smoothing to generate "mean" h/v curve
		smooth_hvsr_matrix[i,:] = np.dot(sm_matrix,hvsr_matrix[i,:])
		#smooth_hvsr_matrix[i,:] = np.exp(np.dot(sm_matrix,np.log(hvsr_matrix[i,:])))
	master_curve = np.exp(np.log(smooth_hvsr_matrix).mean(axis=0))
	# for each interpolated frequency bin, compute sample error of weighted mean (i.e. interpolated value)
	smoothing_deviation_log = np.absolute(np.log(smooth_hvsr_matrix) -np.log( hvsr_matrix))
	smoothing_deviation = np.absolute(smooth_hvsr_matrix - hvsr_matrix)
	#smoothing_variance_log = (smoothing_deviation_log).sum(axis=0) / float(nwindows - 1)
	smoothing_variance_log = np.dot(smoothing_deviation_log.T,smoothing_deviation_log) / float(nwindows - 1)
	smoothing_variance = np.dot(smoothing_deviation.T,smoothing_deviation) / float(nwindows - 1)

	dhvsr_matrix = (smooth_hvsr_matrix[:,1:] - smooth_hvsr_matrix[:,:-1]) / (hvsr_freq[1:]-hvsr_freq[:-1])
	e1=smoothing_variance[:-1,:-1]
	e2=smoothing_variance[1:,1:]
	dserr = (e2 + e1) / ((hvsr_freq[1:]-hvsr_freq[:-1])**2)
	dhvsr_curve = dhvsr_matrix.mean(axis=0)
	dhvsr_deviation = np.absolute(dhvsr_matrix - dhvsr_curve)
	dhvsr_variance = np.dot(dhvsr_deviation.T,dhvsr_deviation) / float(nwindows - 1)
	dhvsr_totalvariance = dhvsr_variance + dserr

	# compute statistics on the difference between successive data points
	y1=smooth_hvsr_matrix[:,:-2]
	y2=smooth_hvsr_matrix[:,1:-1]
	y3=smooth_hvsr_matrix[:,2:]
	e1=smoothing_variance[:-2,:-2]
	e2=smoothing_variance[1:-1,1:-1]
	e3=smoothing_variance[2:,2:]
	ddhvsr_matrix = 2 * (y1/((x2-x1)*(x3-x1)) - y2/((x3-x2)*(x2-x1)) + y3/((x3-x2)*(x3-x1)))
	ddserr = 4 * (e1/(((x2-x1)*(x3-x1))**2) + e2/(((x3-x2)*(x2-x1))**2) + e3/(((x3-x2)*(x3-x1))**2))
	ddhvsr_curve = ddhvsr_matrix.mean(axis=0)
	ddhvsr_deviation = np.absolute(ddhvsr_matrix - ddhvsr_curve)
	ddhvsr_variance = np.dot(ddhvsr_deviation.T,ddhvsr_deviation) / float(nwindows - 1)
	ddhvsr_totalvariance = ddhvsr_variance + ddserr
	
	resample_bias = np.diag(smoothing_variance_log)
	smoothing_variance_operator = sm_matrix ** 2 # currently unused
	std = (np.log(smooth_hvsr_matrix[:][:]) - np.log(master_curve))
	stackerror = np.dot(std.T,std) / float(nwindows-1)
	error = stackerror + smoothing_variance_log
	#diagerr = np.sqrt(std.var(axis=0))
	diagerr = np.sqrt(np.diag(error))
	# errormag doesn't include interpolation variance
	errormag = np.zeros(nwindows)
	for i in xrange(nwindows):	
		errormag[i] = np.dot(std[i,:],std[i,:].T)
	
	print "Master curve shape: " + str(master_curve.shape)
	print master_curve
	print "Frequencies shape: " + str(hvsr_freq.shape)
	print hvsr_freq
	print "Error shape: " + str(error.shape)
	print error
	
	#diagerr = np.sqrt(np.diag(error))
	#diagerr = np.sqrt(std.var(axis=0) + smoothing_variance_log)
	
	lerr = np.exp(np.log(master_curve) - diagerr)
	uerr = np.exp(np.log(master_curve) + diagerr)

	saveprefix = dr_out+runprefix+(spectra_method.replace(' ','_'))+'_'+str(ko_bandwidth)+'_'
	
	np.savetxt(saveprefix+'resample_bias.txt',resample_bias)
	#np.savetxt(saveprefix+'hv.txt',np.column_stack((hvsr_freq,master_curve, lerr,uerr)))
	np.savetxt(saveprefix+'hv.txt',np.column_stack((hvsr_freq,master_curve, diagerr)))
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
	
	# sparse equivalent
	#np.savetxt(saveprefix+'sperror.txt',sp_cov)
	#np.savetxt(saveprefix+'invsperror.txt',sp_prec)
	#logdetsperr = np.linalg.slogdet(sp_cov)
	#print "Log determinant of sparse error matrix: " + str(logdetsperr)
	#np.savetxt(saveprefix+'logdetsperror.txt',np.array(logdetsperr))
	
	#f,((a1,a2,a3),(cba1,cba2,cba3)) = plt.subplots(2,3,figsize=(18,6))
	f = plt.figure(figsize=(18,6))
	gs = gridspec.GridSpec(1, 4)
	a1 = plt.subplot(gs[0,0])
	for i in xrange(hvsr_matrix.shape[0]):
		a1.plot(hvsr_freq,hvsr_matrix[i,:],'k', alpha=0.1)
	a1.plot(hvsr_freq,master_curve,'r')
	a1.plot(hvsr_freq,lerr,':g')
	a1.plot(hvsr_freq,uerr,':b')
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

