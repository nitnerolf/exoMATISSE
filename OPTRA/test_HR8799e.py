from op_corrflux   import *
from op_rawdata    import *
from op_flux       import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io    import fits
import os
from scipy.ndimage import median_filter
from scipy         import *
from scipy         import stats

#plt.ion()

plot    = True
plotdsp = True
verb    = True

bbasedir = '~/Documents/G+/'
#bbasedir = '~/SynologyDrive/driveFlorentin/GRAVITY+/HR8799e/'
bbasedir = '~/Documents/G+/'
basedir  = bbasedir+'GPAO_HR8799e/'
starfile = basedir + 'MATISSE_OBS_SIPHOT_LM_OBJECT_272_0001.fits'
skyfile  = basedir + 'MATISSE_OBS_SIPHOT_LM_SKY_272_0001.fits'

caldir    = bbasedir+'CALIB2024/'
kappafile = caldir+'KAPPA_MATRIX_L_MED.fits'
shiftfile = caldir+'SHIFT_L_MED.fits'
flatfile  = caldir+'FLATFIELD_L_SLOW.fits'
badfile   = caldir+'BADPIX_L_SLOW.fits'

##########################################################

bdata = op_loadAndCal_rawdata(starfile, skyfile, badfile, flatfile, verbose=True)

##########################################################

print('Shape of bdata:', bdata['INTERF']['data'].shape)

if plot:
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
    # Plot the first frame of intf
    ax1.imshow(np.mean(bdata['INTERF']['data'], axis=0), cmap='viridis')
    ax1.set_title('average intf')

    plt.show()
    
cfdata, wlen = op_get_corrflux(bdata, shiftfile)

print(wlen)

#########################################################


op_demodulate(cfdata, wlen, verbose=True, plot=False)

#scfdata = op_sortout_peaks(cfdata, verbose=True)
#scfdata = cfdata



colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']
import matplotlib.animation as animation
fig, ax = plt.subplots()
#lines = [ax.plot([], [], color=colors[i])[0] for i in range(1)]
lines  = [ax.plot([], [], color=colors[i])[0] for i in np.arange(7)]
#lines2 = [ax.plot([], [], '--', color=colors[i])[0] for i in np.arange(7)]
#ax.set_xlim(0, cfdata['CF']['CF'].shape[2])
ax.set_xlim(np.min(wlen), np.max(wlen))
ax.set_ylim(-np.pi, np.pi)
ax.set_title('Phase as a function of the wavelength for CF Data')
def init():
    for line in lines:
        line.set_data([], [])
    return lines
def update(frame):
    for i, line in enumerate(lines):
        if i == 5:
            #line.set_data(wlen, np.angle(cfdata['CF']['CF'][i, frame, :]  * np.conjugate(cfdata['CF']['mod_phasor'][2, frame, :])))
            
            # CF 1 phi 5 -> 6
            # CF 2 phi 0 -> 1
            # CF 3 phi 3 -> 4
            # CF 4 phi 4 -> 5
            # CF 5 phi 1 -> 2
            # CF 6 phi 2 -> 3
            line.set_data(wlen, np.angle(cfdata['CF']['CF_demod'][i, frame, :]))
            
            #lines2[i].set_data(wlen, np.angle(cfdata['CF']['mod_phasor'][i-1, frame, :]))
            
    return lines
ani = animation.FuncAnimation(fig, update, frames=cfdata['CF']['CF'].shape[1], init_func=init, blit=True)
plt.show()

    
if plotdsp:
    plt.figure(5)
    for i in np.arange(6)+1:
        if i == 5:
            plt.plot(np.abs(cfdata['CF']['CF'][i,iframe,:]) / np.abs(cfdata['CF']['CF'][0,0,:])*3,color=colors[i])
            plt.plot(np.max(np.abs(cfdata['CF']['data'][i,iframe,:,:]),axis=1) / np.abs(cfdata['CF']['CF'][0,0,:])*3*7,':',color=colors[i])
    plt.ylim(-0.2,1.2)
    plt.title('This one should resemble a visibility curve')



plt.show()
