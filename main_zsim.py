import numpy as N
import math as M
import scipy.stats
from scipy.fft import fft, fftfreq, fftshift
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt

from zsim_func import *

        
para = Parameters('example.ini')
para.savefreq()

rlos = RandomLOS(para.sig_max,para.xi)

############## Unify the pixelization of input maps ############################
sigma_map, spec_map = para.input_maps()

############## Simulating ############################
print("Generating mu cubes ...")

cube = maps_mu_psi_spin(para, rlos, 1)


'''
hdu = fits.PrimaryHDU(cube[:50].real)
hdul = fits.HDUList([hdu])
hdul.writeto(para.outpath+'psi_Q.fits', overwrite=True)

hdu = fits.PrimaryHDU(cube[:50].imag)
hdul = fits.HDUList([hdu])
hdul.writeto(para.outpath+'psi_U.fits', overwrite=True)

print("Mu si cubes saved!")
'''


apply_mask(rlos, cube, sigma_map)

ref_map, frequency_maps = generate_freq_maps(para, rlos, spec_map, cube)
do_normalization(frequency_maps,para,ref_map)


hdu = fits.PrimaryHDU(frequency_maps.real)
hdul = fits.HDUList([hdu])
hdul.writeto(para.outpath+'f_maps_q.fits', overwrite=True)

hdu = fits.PrimaryHDU(frequency_maps.imag)
hdul = fits.HDUList([hdu])
hdul.writeto(para.outpath+'f_maps_u.fits', overwrite=True)

hdu = fits.PrimaryHDU(N.absolute(frequency_maps))
hdul = fits.HDUList([hdu])
hdul.writeto(para.outpath+'f_maps_p.fits', overwrite=True)

