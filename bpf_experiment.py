#import plotly.plotly as py
#import plotly.graph_objs as go
#from plotly.tools import FigureFactory as FF

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import math
#import pandas as pd
import scipy

from scipy import signal

#data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')
#data = np.loadtxt('wind_speed_laurel_nebraska.csv',delimiter=',',skiprows=1)
sr = 100
dt = 1.0 /sr
sinfreq = 0.5
datax = np.arange(1000)
data = np.sin(datax * dt * 2 * math.pi * sinfreq)

def windowedSincFilterKernel(f,dt, b=0.5):
	fL = f * dt * 2 * math.pi
	b *= dt
	fH = fL
	N = int(np.ceil((4 / b)))
	if not N % 2: N += 1  # Make sure that N is odd.
	n = np.arange(N)
	 
	# low-pass filter
	hlpf = np.sinc(fH * (n - (N - 1) / 2.))
	hlpf *= np.blackman(N)
	hlpf = hlpf / np.sum(hlpf)
	 
	# high-pass filter 
	hhpf = np.sinc(fL * (n - (N - 1) / 2.))
	hhpf *= np.blackman(N)
	hhpf = hhpf / np.sum(hhpf)
	hhpf = -hhpf
	hhpf[(N - 1) / 2] += 1
	 
	h = np.convolve(hlpf, hhpf)
	return h

h = windowedSincFilterKernel(sinfreq,dt)
#s = list(data['10 Min Std Dev'])
s = list(data[:])
new_signal = np.convolve(s, h)

#fig = plt.figure()
#plt.plot(range(len(new_signal)),new_signal)
#plt.show()

#fig = plt.figure()
#plt.plot(datax,data)
#plt.show()


# now try with fft
M = len(s)
H = h.shape[0]
print "Data size = " +str(M)
print "Filter size = " +str(H)
padlen = int(2 ** np.ceil(np.log2(M + H - 1))) 
print "Padded window size = " + str(padlen)
pad_s = np.zeros(padlen)
pad_s[:M] = np.array(s)

# set up the filter response
pad_h = np.zeros(padlen)
pad_h[:h.shape[0]] = h
Fs = np.fft.fft(pad_s)
Fh = np.fft.fft(pad_h)
new_signal_ifft = np.fft.ifft(Fs*Fh)
# trim
#off = (N - 1) / 2
off = (H - 1) / 2
new_signal_ifft_trim = new_signal_ifft[off:M+off]
xs = range(len(s))
s = np.array(s)
s /= s.max()
s *= float(new_signal_ifft_trim.max())

fig2 = plt.figure()
#plt.plot(range(len(new_signal)),new_signal,'b',range(len(new_signal_ifft_trim)),new_signal_ifft_trim,'rx',range(len(new_signal_ifft)),new_signal_ifft,'g:')
#plt.plot(range(len(new_signal)),new_signal,'b',range(len(s)),s,'rx',range(len(new_signal_ifft)),new_signal_ifft,'g:')
plt.plot(range(len(new_signal_ifft_trim)),new_signal_ifft_trim,'b',xs,s,'r:',)
#plt.plot(range(len(new_signal_ifft_trim)),new_signal_ifft_trim,'b')
plt.show()

if False:
	from scipy import signal
	import matplotlib.pyplot as plt
	b, a = signal.cheby1(4, 5, [99,101], 'band', analog=True)
	w, h = signal.freqs(b, a)
	plt.plot(w, 20 * np.log10(abs(h)))
	plt.xscale('log')
	plt.title('Chebyshev Type I frequency response (rp=5)')
	plt.xlabel('Frequency [radians / second]')
	plt.ylabel('Amplitude [dB]')
	plt.margins(0, 0.1)
	plt.grid(which='both', axis='both')
	plt.axvline(100, color='green') # cutoff frequency
	plt.axhline(-5, color='green') # rp
	plt.show()
