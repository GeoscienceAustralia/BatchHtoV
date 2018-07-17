import numpy as np
import matplotlib.pyplot as plt

#a = np.loadtxt('raydechvtfa.txt')
rd = np.loadtxt('BAR1_raydec_md_cwt_os_01_50/BAR1_raydec_md_cwt_os_01_50raydeccwtlog2_60.0_allhv.txt',skiprows=1)
cwt = np.loadtxt('BAR1_cwtlog_md_cwt_os_01_50/BAR1_cwtlog_md_cwt_os_01_50cwtlog_60.0_allhv.txt',skiprows=1)
st = np.loadtxt('BAR1_singletaper_md_cwt_os_01_50/BAR1_singletaper_md_cwt_os_01_50single_taper_60.0_allhv.txt',skiprows=1)
plt.figure()
def applylogerr(a):
	af = a[:,0]
	ahv = a[:,1]
	aerr = a[:,2]
	alerr = np.exp(np.log(ahv) - aerr)
	auerr = np.exp(np.log(ahv) + aerr)
	return (af,ahv,alerr,auerr)
(rdf,rdhv,rdlerr,rduerr) = applylogerr(rd)
(cwtf,cwthv,cwtlerr,cwtuerr) = applylogerr(cwt)
(stf,sthv,stlerr,stuerr) = applylogerr(st)
#plt.plot(a[:,0],a[:,1],'b',b[:,0],b[:,1],'r')
plt.plot(rdf,rdhv,'b',label='Raydec')
plt.hold(True)
plt.plot(rdf,rdlerr,':b',rdf,rduerr,':b')
plt.plot(cwtf,cwthv,'r',label='HVTFA')
plt.plot(cwtf,cwtlerr,':r',cwtf,cwtuerr,':r')
#plt.hold(True)
plt.plot(stf,sthv,'g',label='Single taper FFT')
plt.plot(stf,stlerr,':g',stf,stuerr,':g')
#plt.hold(True)
plt.legend(loc='upper right')
#plt.yscale('log')
plt.xscale('log')
plt.show()
#plt.figure()
#plt.plot(a[:,0],a[:,1],'b',b[:,0],np.exp(np.log(b[:,1])- (np.max(np.log(b[:,1]))-np.max(np.log(a[:,1])))),'r')
#plt.yscale('log')
#plt.xscale('log')
#plt.show()
