import batch
from htov import *
import glob, os
import obspy
from obspy.core import Stream, UTCDateTime
from obspy import read
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy
import calendar
import math
from scipy import signal
import pywt
from wavelets import WaveletAnalysis
import mlpy.wavelet as wave

#dr = '/g/data/ha3/Passive/Stavely/'
#dr = '/g/data/ha3/Passive/OvernightData/Southern_Thompson_2016/AdventureWay1/aw05/AW05_miniSEED/'
dr = 'data/AW05_miniSEED/'

st = Stream()
for f in sorted(glob.glob(dr+'*.EH*')):
	print "Loading " + f
	st += read(f)

st.merge(method=1,fill_value=0)
st.filter("highpass",freq=0.2)
print "stream length = " + str(len(st))

#widths = np.arange(1.0,100.0)
#
strt = 13000
wdw = 50000 + strt
#x = 0.5*(st[0][strt:wdw] + st[1][strt:wdw])
#y = st[2][strt:wdw]
#cwtmatr = signal.cwt(x + 1j * y, signal.morlet, widths)




#st = obspy.read()
fig = plt.figure()
f_min = 0.1
f_max = 40
nf = 100
scalograms = []
frequencies = []
t = []

omega0 = 8
dj     = 0.25
wf     = 'morlet'
for i in xrange(3):
	tr = st[i]
	npts = wdw - strt #tr.stats.npts
	dt = tr.stats.delta
	t.append(np.linspace(0, dt * npts, npts))

        freqs    = np.logspace(np.log10(f_min), np.log10(f_max), nf)
        fperiods = 1./freqs
        scales   = wave.scales_from_fourier(fperiods, wf, omega0)
        cfs      = wave.cwt(tr[strt:wdw], dt, scales, wf, omega0)

	scalograms.append(cfs)
	frequencies.append(freqs)

#scalograms[1] = np.sqrt(0.5*(scalograms[0]**2 + scalograms[1]**2))
scalograms[1] = np.sqrt(0.5*(np.abs(scalograms[0])**2 + np.abs(scalograms[1])**2))
scalograms[2] = np.abs(scalograms[2])

# search for maxima in the vertical spectrum for each frequency band and get horizontal amplitude at 90 degree offsets for Rayleigh Wave ellipticity.

fval = frequencies[0]
sfreq = st[0].stats.sampling_rate
hv = np.zeros(scalograms[2].shape[0])

for findex in xrange(scalograms[2].shape[0]):
	f = fval[findex]
	rayleighDelay = int(0.25 * (1.0/f) * sfreq + 0.5) # the + 0.5 at the end is to round to nearest for integer reference
	extrema = scipy.signal.argrelextrema(scalograms[2][findex,rayleighDelay:-rayleighDelay], np.greater)
	print extrema
	hv_numer = 0.0
	e = extrema[0]
	vmax = scalograms[2][findex,e]
	hmax_neg = scalograms[1][findex,e-rayleighDelay]
	hmax_pos = scalograms[1][findex,e+rayleighDelay]
	#for now, average it.
	hv_numer = (hmax_neg + hmax_pos) / (2.0 * vmax)
	hv[findex] = np.sum(hv_numer) / hv_numer.shape[0] # get mean

#Plot HVTFA
ax = fig.add_subplot(3,1,1)
ax.plot(fval,hv)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("HVTFA")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(f_min, f_max)
	
for i in xrange(1,3):
	ax = fig.add_subplot(3,1,i+1)
	
	periods = 1./frequencies[i]
	ax.contourf(t[i], frequencies[i], (scalograms[i]), cmap=obspy_sequential)

	ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
	ax.set_ylabel("Frequency [Hz]")
	ax.set_yscale('log')
	ax.invert_yaxis()
plt.show()

#f = plt.figure(figsize=(10,6))
#gs = gridspec.GridSpec(3,1)
#a1 = plt.subplot(gs[0,0])
#a1.plot(x)
#a2 = plt.subplot(gs[1,0])
#a2.plot(y)
#a3 = plt.subplot(gs[2,0])
#a3.plot(st[2][strt:wdw])
#a3.imshow(cwtmatr,interpolation='nearest',aspect=10)
#plt.show()
