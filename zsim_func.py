import numpy.fft
import numpy as N
import math as M
from scipy.fft import fft, fftfreq, fftshift
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
import configparser
from scipy.interpolate import CubicSpline

class Parameters():
    def __init__(self, filename):
        config = configparser.ConfigParser()
        config.read(filename)
        self.Path_Fara_width = config['PATH']['path_fara_width']
        self.Path_Spec_ind = config['PATH']['path_spec_ind']
        self.outpath = config['PATH']['outpath']
        Para = config['PARAMETER']
        freq_min = float(Para['freq_min'])
        freq_max = float(Para['freq_max'])
        dfreq = float(Para['dfreq'])
        # in MHz
        freq_table = N.arange(freq_min,freq_max,dfreq)
        self.freq_table = freq_table
        #self.xs_table = 2*(c_light/freq_table)**2
        self.xi = float(Para['xi'])
        self.beta = float(Para['beta'])
        Nside = int(Para['nside'])
        self.Nside = Nside
        self.npix = hp.pixelfunc.nside2npix(Nside)
        self.lmax = 3 * Nside - 1
        self.nlms = int((self.lmax + 2) * (self.lmax + 1) / 2)
        sigma_in = hp.fitsfunc.read_map(self.Path_Fara_width)
        spec_in = hp.fitsfunc.read_map(self.Path_Spec_ind)
        self.sig_max = N.max(sigma_in)
        self.sig_min = N.min(sigma_in)
    def input_maps(self):
        sigma_in = hp.fitsfunc.read_map(self.Path_Fara_width)
        spec_in = hp.fitsfunc.read_map(self.Path_Spec_ind)
        if self.Nside == 128:
            sigma_map = sigma_in
            spec_map = nside_downgrade(spec_in, 512, 128)
        elif self.Nside == 512:
            sigma_map = nside_upgrade(sigma_in, 128, 512)
            spec_map = spec_in
        return sigma_map, spec_map
    def savefreq(self):
        return N.savetxt('frequency_table.txt', self.freq_table, delimiter=',')
        
class RandomLOS():
    def __init__(self,sigma_max,xi):
        psi_max   =4.* sigma_max
        delta_psi1=0.2*xi
        #n1=2.*psi_max/delta_psi1
        #n=int(2**M.ceil(M.log2(n1)))
        n=int(2.*psi_max/delta_psi1)
        delta_psi=2.*psi_max/n
        self.n=n
        self.delta_psi=delta_psi
        self.psi_max=psi_max
        self.psi_vec=N.linspace(-psi_max,+psi_max,num=n)
        x=N.arange(n)/n
        x=x*(1.-x)
        factor=M.pi*xi*n/psi_max
        self.mask=N.exp(-0.25*(factor*x)**2)
        self.norm=1/N.sqrt((1./n)*N.sum(self.mask**2))
        self.xn=2*M.pi*fftfreq(n,delta_psi)
        #   fft.fftfreq(n, d=1.0)[source]
        #   f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
        #   f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
    def randJ(self):
        jVec=N.random.standard_normal(self.n)+1.j*N.random.standard_normal(self.n)
        jVec*=self.mask
        jVec=self.norm*fft(jVec)
        #jVec=jVec.real
        return(jVec)

def nside_downgrade(map_in, nside_in, nside_out): # The function used to change the Nside of input maps.
    npix_in = hp.nside2npix(nside_in)
    npix_out = hp.nside2npix(nside_out)
    ratio = int(npix_in/npix_out)
    i_ratio = float(1./ratio)
    map_out = N.zeros(shape=(npix_out,))
    for ii in range(npix_out):
        inest_out = hp.ring2nest(nside_out,ii)
        tot = 0.
        for jj in range(ratio):
            iring_in = hp.nest2ring(nside_in, jj+ratio*inest_out)
            tot += map_in[iring_in]
        map_out[ii]=tot*i_ratio
    return map_out

def nside_upgrade(map_in, nside_in, nside_out): #1. input map is in "ring"; nside_in < nside_out. eg: nside_out=512, nside_in=128.
    npix_in = hp.nside2npix(nside_in)
    npix_out = hp.nside2npix(nside_out)
    ratio = int(npix_out/npix_in)
    map_out = N.zeros(shape=(npix_out,))
    for ii in range(npix_in):
        value = map_in[ii]
        inest_in = hp.ring2nest(nside_in,ii)
        for jj in range(ratio):
            iring_out = hp.nest2ring(nside_out, jj+inest_in*ratio)
            map_out[iring_out] = value
    return map_out

def psFact(l,beta):
   if l<=10:
     ps=10**(-beta/2.)
   else:
     ps=l**(-beta/2.)
   return(ps)

def maps_mu_psi_spin(para, rlos, c_bb_over_c_ee):
    nlms=para.nlms
    n=rlos.n
    npix=para.npix
    Nside = para.Nside
    lmax = para.lmax
    beta = para.beta
    mu_lm_e = N.zeros(shape=(n,nlms),dtype=complex)
    mu_lm_b = N.zeros(shape=(n,nlms),dtype=complex)
    mu_psi_maps_Q = N.zeros(shape=(n,npix),dtype=float)
    mu_psi_maps_U = N.zeros(shape=(n,npix),dtype=float)
    for l in range(lmax+1):
        psFactor=psFact(l,beta)
        for m in range(0,l+1):
            index = hp.sphtfunc.Alm.getidx(lmax,l,m)
            mu_lm_e[:,index] =                        psFactor*rlos.randJ()
            mu_lm_b[:,index] = M.sqrt(c_bb_over_c_ee)*psFactor*rlos.randJ()
            if m == 0:
              mu_lm_e[:,index]=M.sqrt(2)*mu_lm_e[:,index].real
              mu_lm_b[:,index]=M.sqrt(2)*mu_lm_b[:,index].real
    alm_t=N.zeros((nlms,),dtype=N.complex)
    for k in range(n):
        map_i, mu_psi_maps_Q[k,:], mu_psi_maps_U[k,:]= hp.alm2map( (alm_t,
                      mu_lm_e[k,:],
                      mu_lm_b[k,:]),
                      Nside,pol=True)
    cube=mu_psi_maps_Q+mu_psi_maps_U*1j
    return(cube)
    
def apply_mask(rlos, cube, sigma_map):
    psi_vec = fftshift(rlos.psi_vec)
    npsi, npix=cube.shape
    #aux_sigma = 2*sigma_map**2
    #aux_psi = -psi_vec**2
    #cube *= N.exp(aux_psi[:, N.newaxis]/aux_sigma)
    for ipix in range(npix):
       mask=N.exp(-psi_vec**2/(2.*sigma_map[ipix]**2))
       cube[:,ipix]*=mask
    return

def generate_freq_maps(para, rlos, spec_map, cube):
    npsi=rlos.n
    npix=para.npix
    psiConjVec=rlos.xn
    freqVecMHz=para.freq_table
    xNeededVec=1.8e5/freqVecMHz**2
    #xs=para.xs_table
    #cube = N.fft.fft(cube, axis=0)
    #cs = CubicSpline(xn[1:int(n/2)], cube[1:int(n/2),:], axis=0)
    #del cube
    #aux=fs[:, N.newaxis]**spec_map
    #freq_maps = aux*cs(xs)
    nfreq=len(freqVecMHz)
    freq_maps=N.zeros((nfreq,npix),dtype=complex) 
    for ipix in range(npix): 
       jpsiVec=cube[:,ipix]
       jxVec  =N.fft.fft(jpsiVec)
       mySpline=CubicSpline(psiConjVec[0:int(npsi/2)], jxVec[0:int(npsi/2)],extrapolate=False)
       fVals=mySpline(xNeededVec)
       freq_maps[:,ipix]=fVals*((freqVecMHz/300.)**spec_map[ipix])
    return(freq_maps)
    
def get_x_and_frequency_slices(para,rlos,spec_map,cube):
    npsi=rlos.n
    npix=para.npix
    psiConjVec=rlos.xn
    freqVecMHz=para.freq_table
    xNeededVec=1.8e5/freqVecMHz**2
    #XinRange = []
    xleft=-1
    xright=-1
    for i in N.arange(npsi):
        if psiConjVec[i]>xNeededVec[0]:
           xright=i
           break
    for i in N.arange(npsi):
        if psiConjVec[i]>xNeededVec[-1]:
           xleft=i
           break
    nfreq=len(freqVecMHz)
    freq_maps=N.zeros((nfreq,npix),dtype=complex)
    x_maps=N.zeros((int(xright-xleft),npix),dtype=complex)
    freq_x_maps = N.sqrt(1.8e5/psiConjVec[xleft:xright])
    N.savetxt('x_frequency_table.txt', freq_x_maps, delimiter=',')
    for ipix in range(npix):
       jpsiVec=cube[:,ipix]
       jxVec  =N.fft.fft(jpsiVec)
       mySpline=CubicSpline(psiConjVec[0:int(npsi/2)], jxVec[0:int(npsi/2)],extrapolate=False)
       fVals=mySpline(xNeededVec)
       freq_maps[:,ipix]=fVals*((freqVecMHz/300.)**spec_map[ipix])
       x_maps[:,ipix]=jxVec[xleft:xright]*((freq_x_maps/300.)**spec_map[ipix])
    return(freq_maps,x_maps)


