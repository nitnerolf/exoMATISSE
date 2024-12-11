"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Example script to post-process MATISSE data of beta Pic b
Author: fmillour
Date: 06/12/2024
Project: OPTRA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from op_corrflux   import *
from op_rawdata    import *
from op_flux       import *
from op_vis        import *
from op_oifits     import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
import os
from scipy.ndimage import median_filter
from scipy         import *
from scipy         import stats

bbasedir = os.path.expanduser('~/Documents/ExoMATISSE/beta_Pic_b/')
#basedir  = bbasedir+'2022-11-09_OIFITS/'
basedir  = bbasedir+'2024-11-18_OIFITS/'
basename = os.path.basename(os.path.normpath(basedir))
print(basename)
starfiles = os.listdir(basedir)
timebin = 0.5  #in hours
bcd = "OUTOUT"
#bcd = "ININ"
#bcd = "INOUT"
#bcd = "OUTIN"
#bcd=''
fitsfiles = [f for f in starfiles if "oi.fits" in f]
BCDfiles  = [f for f in fitsfiles if bcd in f]
ININfiles = [f for f in BCDfiles if "SCI" in f]
#ININfiles = fitsfiles
ININfiles.sort()

starfiles = [f for f in BCDfiles if "CAL" in f]

def op_read_oifits_sequence(basedir, filelist):
    
    hdus = []
    vis  = []
    for ifile, file in enumerate(filelist):
        print('reading file: ', file)
        
        ihdu = op_read_oifits(basedir + file)
        
        if ifile == 0:
            wlen = ihdu['OI_WAVELENGTH'].data['EFF_WAVE']
            band = ihdu['OI_WAVELENGTH'].data['EFF_BAND']
            dit = ihdu[0].header['ESO DET SEQ1 DIT']
        
        ivis    = ihdu['OI_VIS'].data['VISAMP'] * np.exp(1j * ihdu['OI_VIS'].data['VISPHI'])
        nbase   = 6;
        nframes = np.shape(ivis)[0]//nbase
        nwlen   = np.shape(ivis)[1]
        ivis2   = ivis.reshape((nframes,nbase,nwlen))
        print('ivis shape: ', np.shape(ivis2))
        
        imedvis = np.median(np.abs(ivis), axis=-1)
        print('imedvis : ', imedvis)
        
        vis.append(ivis2)
        hdus.append(ihdu)    
        ihdu.close()
    return hdus, vis, wlen, band, dit
#print(wlen)
#print(ihdu)

hdus, vis, wlen, band, dit = op_read_oifits_sequence(basedir, ININfiles)
nfiles  = np.shape(vis)[0]
nframes = np.shape(vis)[1]
nbase   = np.shape(vis)[2]
nwlen   = np.shape(vis)[3]

print('vis shape: ', np.shape(vis))

totvis = np.sum(vis,axis=-1)

cvis = vis * np.exp(-1j * np.angle(totvis[...,None]))
#print(np.angle(totvis))
#print(np.angle(np.sum(cvis,axis=-1)))
rvis = cvis.reshape((nfiles*nframes,nbase,nwlen))
print('rvis shape: ', np.shape(rvis))

medvis = np.median(np.abs(rvis), axis=-1)
print('medvis shape: ', np.shape(medvis))
print('medvis',medvis)
threshold = 1100
threshup  = 3000
mask = (np.abs(medvis) < threshold) ^ (np.abs(medvis) > threshup)





shdus, svis, swlen, sband, sdit = op_read_oifits_sequence(basedir, starfiles)
snfiles  = np.shape(svis)[0]
snframes = np.shape(svis)[1]
snbase   = np.shape(svis)[2]
snwlen   = np.shape(svis)[3]

print('star vis shape: ', np.shape(svis))

stotvis = np.sum(svis,axis=-1)

scvis = svis * np.exp(-1j * np.angle(stotvis[...,None]))
#print(np.angle(totvis))
#print(np.angle(np.sum(cvis,axis=-1)))
srvis = scvis.reshape((snfiles*snframes,snbase,snwlen))
print('star rvis shape: ', np.shape(srvis))

smedvis = np.median(np.abs(srvis), axis=-1)
print('star medvis shape: ', np.shape(smedvis))
print('star medvis',smedvis)
sthreshold = 1100
starmask = (np.abs(smedvis) < sthreshold)


#print('mask',mask)
print('Keeping these frames per baseline for the planet')
print(np.sum(~mask,axis=0))
print('over',nfiles*nframes,'frames')
print('Keeping these frames per baseline for the star')
print(np.sum(~starmask,axis=0))
mask2 = np.repeat(mask[...,None],nwlen,axis=-1)
starmask2 = np.repeat(starmask[...,None],nwlen,axis=-1)

rvis2 = np.ma.masked_array(rvis, mask=mask2)
starvis2 = np.ma.masked_array(srvis, mask=starmask2)


fig1, ax1 = plt.subplots(nbase, 2, figsize=(8, 8), sharex=1, sharey=0)
#print('Shape of ax1:', ax1.shape)
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#33AAA5', '#A133FF', '#F5AA33']
mean_vis_planet = np.mean(rvis2,axis=0)
for i in np.arange(nbase):
    #print('i:', i)
    for j in np.arange(nfiles*nframes):
        ax1[i,0].plot(wlen, np.abs(rvis2[j,i,:]), color=colors[i], alpha=0.1)
        
    meanvis = np.abs(np.mean(rvis2[:,i,:],axis=0))
    ax1[i,0].plot(wlen, meanvis, color='black', alpha=1)
    selvis = meanvis[(wlen > 3.5e-6) & (wlen < 4.0e-6)]
    print('mean corr. flux',i+1,':',np.mean(selvis))
    
    ax1[i,0].set_ylim(0,3000)
    if i == 0 and nbase == 7:
        ax1[i,0].set_ylabel(f'flux {i+1}')
    else:
        ax1[i,0].set_ylabel(f'corr. flux {i+1}')
    for j in np.arange(nfiles*nframes):
        ax1[i,1].plot(wlen, np.angle(rvis2[j,i,:]), color=colors[i], alpha=0.1)
    ax1[i,1].plot(wlen, np.angle(np.mean(rvis2[:,i,:],axis=0)), color='black', alpha=1)
    ax1[i,1].set_ylabel(f'phase {i+1}')
    ax1[i,1].set_ylim(-1,1)
plt.suptitle('Sum CF data (1 exposure)')
plt.tight_layout()
plt.savefig(os.path.expanduser(bbasedir+f'{basename}_{bcd}_corrflux_planet.png'))


fig2, ax2 = plt.subplots(nbase, 2, figsize=(8, 8), sharex=1, sharey=0)
#print('Shape of ax1:', ax1.shape)
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#33AAA5', '#A133FF', '#F5AA33']
mean_vis_star = np.mean(starvis2,axis=0)
for i in np.arange(nbase):
    #print('i:', i)
    for j in np.arange(snfiles*snframes):
        ax2[i,0].plot(wlen, np.abs(starvis2[j,i,:]) * dit / sdit, color=colors[i], alpha=0.1)
    
    meanvis = np.abs(np.mean(starvis2[:,i,:] * dit / sdit,axis=0))
    ax2[i,0].plot(wlen, meanvis, color='black', alpha=1)
    selvis = meanvis[(wlen > 3.5e-6) & (wlen < 4.0e-6)]
    print('mean corr. flux star',i+1,':',np.mean(selvis))
    
    ax2[i,0].set_ylim(0)
    if i == 0 and nbase == 7:
        ax2[i,0].set_ylabel(f'flux {i+1}')
    else:
        ax2[i,0].set_ylabel(f'corr. flux {i+1}')
    for j in np.arange(snfiles*snframes):
        ax2[i,1].plot(wlen, np.angle(starvis2[j,i,:]), color=colors[i], alpha=0.1)
    ax2[i,1].plot(wlen, np.angle(np.mean(starvis2[:,i,:],axis=0)), color='black', alpha=1)
    ax2[i,1].set_ylabel(f'phase {i+1}')
    ax2[i,1].set_ylim(-1,1)
plt.suptitle('Sum CF data (1 exposure)')
plt.tight_layout()
plt.savefig(os.path.expanduser(bbasedir+f'{basename}_{bcd}_corrflux_star.png'))


fig3, ax3 = plt.subplots(nbase, 2, figsize=(8, 8), sharex=1, sharey=0)
#print('Shape of ax1:', ax1.shape)
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#33AAA5', '#A133FF', '#F5AA33']
mean_vis_star_planet = mean_vis_planet / mean_vis_star * sdit / dit
for i in np.arange(nbase):
    #print('i:', i)
    ax3[i,0].plot(wlen, np.abs(mean_vis_star_planet[i,:]), color='black', alpha=1)
    ax3[i,0].set_ylim(0,2e-3)
    ax3[i,0].set_ylabel(f'corr. flux {i+1}')

    ax3[i,1].plot(wlen, np.angle(mean_vis_star_planet[i,:]), color='black', alpha=1)
    ax3[i,1].set_ylabel(f'phase {i+1}')
    ax3[i,1].set_ylim(-1.5,1.5)
plt.suptitle('Sum CF data (1 exposure)')
plt.tight_layout()
plt.savefig(os.path.expanduser(bbasedir+f'{basename}_{bcd}_corrflux_planet_star.png'))


plt.show()