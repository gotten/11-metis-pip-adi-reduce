from hciplot import plot_frames, plot_cubes
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import numpy as np
from packaging import version

import vip_hci as vip
from vip_hci.metrics import completeness_curve, completeness_map
from vip_hci.config import VLT_NACO
from vip_hci.fm import cube_planet_free, normalize_psf
from vip_hci.metrics import inverse_stim_map, snr, snrmap, stim_map
from vip_hci.psfsub import median_sub, pca, pca_annular
from vip_hci.fits import open_fits, write_fits, info_fits
from vip_hci.metrics import contrast_curve, detection, significance, snr, snrmap, throughput
from vip_hci.preproc import frame_crop
from vip_hci.var import fit_2dgaussian, frame_center

"""
This plotting routine is based on the tutorial of VIP_HCI (Original authors: Carlos Alberto Gomez Gonzalez and Valentin Christiaens) which uses an MIT-license which is replicated here.
Edited by Gilles Otten
Requires VIP_HCI 1.3.1 (https://github.com/vortex-exoplanet/VIP)
psf_on and psf_off were generated with a combination of HEEPS+Scopesim
"""

psfref = 'psf_off_scopesim_RAVC_HCI_L_long.fits' # off-axis psf generated with HEEPS+Scopesim
#cubename = '/home/gotten/psf_sum_scopesim_RAVC_HCI_L_long.fits'  # planetary system scene generated with HEEPS+Scopesim (coronagraphic PSF on-axis + two off-axis PSFs for planets, deltaL=7.7, at 100 mas)

#cube=fits.getdata(cubename)
newhourangles=np.zeros([120])
hourangles=np.linspace(-0.5/24.*360.,0.5/24.*360.,12000) # replicating the hour angles of 1 hour of observations
for i in np.arange(120):
#    newcube[i,:,:]=np.mean(cube[i*100:(i+1)*100,101:-101,101:-101],0) # for rebinning the images for speed for this example program
    newhourangles[i]=np.mean(hourangles[i*100:(i+1)*100]) # for rebinning the hour angles by factor 100
hourangles=newhourangles*1.
#cube=np.reshape(cube,[120,100,403,403]) # 
#newcube=np.mean(cube,0)
#cube=newcube-np.median(newcube)

from astropy.io import fits
#fits.writeto("testcube.fits",cube,overwrite=1)
cube=fits.getdata("psf_on_scopesim_RAVC_HCI_L_long_binned.fits") # reading in the previously binned datacube (12000 frames original, 120 frames after binning)
#cube.reshape([100,120,403,403])
print(cube.shape)
#cube=np.mean(cube,0)
psf = open_fits(psfref)
starphot= 36077395.284 #np.sum(psf)

#hourangles=np.linspace(-0.5/24.*360.,0.5/24.*360.,12000)
#hourangles.reshape([100,120])
#hourangles=np.mean(hourangles,0)
#hourangles.reshape([600,20]).mean(0)
#hourangles=np.reshape(hourangles,[100,120])
#hourangles=np.mean(hourangles,0)
#print(hourangles.shape)

lat = -24.59            # deg
dec = -51.0665168055                # deg -51 == beta pic
hr = np.deg2rad(hourangles)
dr = np.deg2rad(dec)
lr = np.deg2rad(lat)
# parallactic angle in deg
pa = -np.rad2deg(np.arctan2(-np.sin(hr), np.cos(dr)*np.tan(lr)
     - np.sin(dr)*np.cos(hr)))
pa = (pa + 360)%360




derot_off = 0. # derotator offset for this observation 
TN = 0.5         # Position angle of true north for instrument at the epoch of observation

angs = pa-derot_off-TN

DF_fit = fit_2dgaussian(psf, crop=True, cropsize=9, debug=True, full_output=True)

fwhm_ref = np.mean([DF_fit['fwhm_x'],DF_fit['fwhm_y']])

psfn = normalize_psf(psf, fwhm_ref, size=191, imlib='vip-fft')

pxscale_ref = 5.47e-3
print(5.47e-3, "arcsec/px")

#pca_img = median_sub(cube, angs, verbose=False, imlib='vip-fft', interpolation=None)
#plot_frames(pca_img, backend='bokeh')

#snrmap_1 = snrmap(pca_img, fwhm=fwhm_ref, plot=True)
r_b =  0.100/pxscale_ref # planet was injected at 100 mas
theta_b = 0.+90 # note the VIP convention is to measure angles from positive x axis
theta_b2= 0.+270. # two planets were injected at 180 degrees from eachother
f_b = 10**(-0.4*7.7)*starphot # the injection contrast was 7.7 magnitudes wrt the star

cube_emp1 = cube_planet_free([(r_b, theta_b, f_b)], cube, angs, psfn=psfn) # manually subtracting one of the two injected planets
cube_emp2=cube_planet_free([(r_b, theta_b2, f_b)], cube_emp1, angs, psfn=psfn) # manually subtracting second injected planet
pca_img = median_sub(cube_emp1, angs, verbose=False, imlib='vip-fft', interpolation=None) # median subtraction ADI on image with only one planet
#plot_frames(pca_img, backend='bokeh')

snrmap_1 = snrmap(pca_img, fwhm=fwhm_ref, plot=True)


#pca_emp = median_sub(cube_emp, angs,verbose=False)
pca2=median_sub(cube_emp2,angs,verbose=False)
#plot_frames((pca2), axis=False)

#fr_pca = pca_annular(cube, angs, ncomp=7)
#snrmap_1=snrmap(cube_emph,fwhm=fwhm_ref,plot=True)

detection(pca2, fwhm=fwhm_ref, psf=psfn, bkg_sigma=5, debug=False, mode='log',
          snr_thresh=5, plot=True, verbose=True)



cc_1 = contrast_curve(cube_emp2, angs, psfn, fwhm=fwhm_ref, pxscale=pxscale_ref, starphot=starphot,
                      sigma=5, nbranch=1, algo=median_sub, debug=True)
#plot_images(cc_1)
cc_1.plot(x='distance_arcsec',y='sensitivity_gaussian')
plt.show()
# plt.show will now output a list of diagnostic plots that are used in the DRLD document
