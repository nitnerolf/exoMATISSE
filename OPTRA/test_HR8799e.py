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

plotFringes   = False
plotPhi       = True
plotDsp       = False

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

if plotFringes:
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
    # Plot the first frame of intf
    ax1.imshow(np.mean(bdata['INTERF']['data'], axis=0), cmap='viridis')
    ax1.set_title('average intf')

    plt.show()
    
cfdata, wlen = op_get_corrflux(bdata, shiftfile)

print(wlen)

#########################################################

cfdem = op_demodulate(cfdata, wlen, verbose=True, plot=False)

print('Shape of cfdata:', cfdem['CF']['CF_demod'].shape)
cf = cfdem['CF']['CF_demod']
sumcf = np.sum(cf, axis=1)
print('Shape of sumcf:', sumcf.shape)

fig1, ax1 = plt.subplots(6, 2, figsize=(8, 8))
print('Shape of ax1:', ax1.shape)
colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF5', '#F5FF33']
for i in np.arange(6):
    print('i:', i)
    ax1[i  ].plot(wlen,   np.abs(sumcf[i+1,:]), color=colors[i])
    #ax1[2*i+1].plot(wlen, np.angle(sumcf[i+1,:]), color=colors[i])
plt.title('Sum of the CF data')
#plt.show()

iframe = 0
fig2, ax2 = plt.subplots(6, 2, figsize=(8, 4))
for i in np.arange(6)+1:
    ax2[2*i-1].plot(wlen, np.abs(cf[i,iframe,:]), color=colors[i])
    ax2[2*i].plot(wlen, np.angle(cf[i,iframe,:]), color=colors[i])
plt.title(f'frame {iframe} of CF data')
plt.show()


            #line.set_data(wlen, np.angle(cfdata['CF']['CF'][i, frame, :]  * np.conjugate(cfdata['CF']['mod_phasor'][2, frame, :])))
            

'''
#scfdata = op_sortout_peaks(cfdata, verbose=True)
#scfdata = cfdata

iframe = 0
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
    
if plotDsp:
    plt.figure(5)
    for i in np.arange(6)+1:
        if i == 5:
            plt.plot(np.abs(cfdata['CF']['CF'][i,iframe,:]) / np.abs(cfdata['CF']['CF'][0,0,:])*3,color=colors[i])
            plt.plot(np.max(np.abs(cfdata['CF']['data'][i,iframe,:,:]),axis=1) / np.abs(cfdata['CF']['CF'][0,0,:])*3*7,':',color=colors[i])
    plt.ylim(-0.2,1.2)
    plt.title('This one should resemble a visibility curve')



plt.show()
'''