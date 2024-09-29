from op_corrflux import *
from op_cosmetics import *
from op_flux import *
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from scipy.ndimage import median_filter
from scipy import *
from scipy import stats

basedir  = '~/Documents/G+/GPAO_HR8799e/'
starfile = 'MATISSE_OBS_SIPHOT_LM_OBJECT_272_0005.fits'
skyfile  = 'MATISSE_OBS_SIPHOT_LM_SKY_272_0001.fits'

caldir    = '~/Documents/G+/CALIB2024/'
kappafile = 'KAPPA_MATRIX_L_MED.fits'
shiftfile = 'SHIFT_L_MED.fits'
flatfile  = 'FLATFIELD_L_SLOW.fits'
badfile   = 'BADPIX_L_SLOW.fits'

# Load the star and sky data
tardata  = op_load_rawdata(basedir+starfile)
starname = tardata['hdr']['OBJECT']
print(starname)

# Load the sky data
skydata  = op_load_rawdata(basedir+skyfile)

# Subtract the sky from the star data
stardata = op_subtract_sky(tardata, skydata)

# Load the calibration data
bpm = op_load_bpm(caldir+badfile)
ffm = op_load_ffm(caldir+flatfile)

fdata = op_apply_ffm(stardata, ffm, verbose=True)
bdata = op_apply_bpm(fdata, bpm, verbose=True)

print('Shape of bdata:', bdata['INTERF']['data'].shape)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)
# Plot the first frame of intf
ax1.imshow(np.mean(bdata['INTERF']['data'],axis=0), cmap='gray')
ax1.set_title('average intf')

plt.show()

adata = op_apodize(bdata, verbose=True)

# Compute the average of intf after apodization
avg_intf = np.mean(adata['INTERF']['data'], axis=0)
vmn = 1e-9
vmx = np.max(avg_intf)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.tight_layout()
split_images = np.array_split(avg_intf, 3, axis=0)
for ax, img in zip(axes, split_images):
    ax.imshow(img, cmap='gray')
plt.title('Average of intf after Apodization')
plt.show()

#compute fft
fdata = op_calc_fft(adata)

# Compute the average of intf after apodization
sum_dsp = np.log(fdata['FFT']['sum_dsp'])
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.tight_layout()
split_images = np.array_split(sum_dsp, 3, axis=0)
for ax, img in zip(axes, split_images):
    ax.imshow(img, cmap='gray')
plt.title('Sum of dsp after Apodization')
plt.show()


print(np.shape(intf))
import matplotlib.animation as animation

# Function to update the frames
def update_frame(i):
    # Split the image into three parts
    return axes

# ani = animation.FuncAnimation(fig, update_frame, frames=nframe, interval=50, blit=True, repeat=True)
# fig.suptitle(f'Animation of Processed Frames (vmin={vmn}, vmax={vmx})', fontsize=16)

plt.show()

# Compute FFT 1D of intf along the time axis
fft_intf = np.fft.fft(intf, axis=2)
fft_intf_magnitude = np.abs(fft_intf)
dsp_intf = fft_intf_magnitude**2
sum_dsp_intf = np.sum(dsp_intf, axis=0)
sdi_resh = np.fft.fftshift(sum_dsp_intf, axes=1)
print('Shape of sum_dsp_intf:', sum_dsp_intf.shape)

print('Shape of fft_intf_magnitude:', fft_intf_magnitude.shape)
# Normalize the FFT magnitude for better visualization
dsp_intfn = dsp_intf / np.max(dsp_intf, axis=(1, 2))[:, None, None]

#print(sum_dsp_intf[100,:])

plt.figure()
#plt.imshow(sum_dsp_intf+10)
plt.semilogy(sum_dsp_intf[800,:])
plt.show()

plt.figure()
#plt.imshow(sum_dsp_intf+10)
plt.imshow(np.log(sum_dsp_intf))
plt.show()

print('Shape of dsp_intfn:', dsp_intfn.shape)
fig_fft, axes_fft = plt.subplots(1, 3, figsize=(18, 6))

split_fft_images = np.array_split(sum_dsp_intf, 3, axis=0)

print('Shape of split_fft_images:', np.shape(split_fft_images))
for ax, img in zip(axes_fft, split_fft_images):
    print('Shape of img:', np.shape(img))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

plt.show()