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

CLIP_TO_FREQ = False 
RESAMPLE_FREQ = True

if (len(sys.argv) < 7):
	print "Usage: python htov.py method /path/to/miniSEED/ nfrequencies f_min f_max prefix"
	print "Method is 'single taper', 'st, 'cwt2'"
	exit(0)

nfrequencies = int(sys.argv[3])
initialfreq = float(sys.argv[4])
finalfreq = float(sys.argv[5])

runprefix = sys.argv[6]

#dr = '/g/data/ha3/Passive/Stavely/'
dr = '/g/data/ha3/Passive/OvernightData/STAVELY/S06PS/Seismometer_data/S0600/S0600miniSEED/'
#dr = 'data/S0600miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/AdventureWay1/aw05/AW05_miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/Overshot1/OV04/OV04_miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/Eulo1/EU13/EU13_miniSEED/'
#dr = '/g/data/ha3/Passive/OvernightData/EUCLA_PASSIVE/GUINEWARRA/GB12/GB12_miniSEED/'

dr = sys.argv[2]

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
								   'taper':'blackman'},
                                                                  master_curve_method='mean',cutoff_value=0.0,
                                                                  window_length=100.0,bin_samples=nfrequencies,
                                                                  f_min=initialfreq,f_max=finalfreq)

nwindows = len(hvsr_matrix)

lowest_freq = initialfreq #0.3
highest_freq = finalfreq #50.0
def find_nearest_idx(array,value):
	return (np.abs(array-value)).argmin()

if RESAMPLE_FREQ:
	# generate frequencies vector
	logfreq = np.zeros(nfrequencies)
	c = (1.0/(nfrequencies-1))*np.log10(finalfreq/initialfreq)
	for i in xrange(nfrequencies):
		logfreq[i] = initialfreq*(10.0 ** (c*i))
	# interpolate to log spacing
	print "Number of windows computed = " + str(nwindows)
	interp_hvsr_matrix = np.empty((nwindows, nfrequencies))
	for i in xrange(nwindows):
		nint = interp1d(hvsr_freq, hvsr_matrix[i,:])
		hv_spec2 = nint(logfreq)
		interp_hvsr_matrix[i,:] = hv_spec2
	hvsr_freq = logfreq
else:
	interp_hvsr_matrix = hvsr_matrix
	nfrequencies = hvsr_freq.shape[0]
	initialfreq = hvsr_freq[0]
	finalfreq = hvsr_freq[nfrequencies-1]
#master_curve = interp_hvsr_matrix.mean(axis=0)
master_curve = np.median(interp_hvsr_matrix,axis=0)
std = (np.log1p(interp_hvsr_matrix[:][:]) - np.log1p(master_curve))
errormag = np.zeros(nwindows)
for i in xrange(nwindows):	
	errormag[i] = np.dot(std[i,:],std[i,:].T)
error = np.dot(std.T,std)
error /= float(nwindows-1)

sp_model = GraphLassoCV()
sp_model.fit(std)
sp_cov = sp_model.covariance_
sp_prec = sp_model.precision_


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

diagerr = np.sqrt(np.diag(error))
lerr = np.exp(np.log(master_curve) - diagerr)
uerr = np.exp(np.log(master_curve) + diagerr)
saveprefix = dr+runprefix+(spectra_method.replace(' ','_'))

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
