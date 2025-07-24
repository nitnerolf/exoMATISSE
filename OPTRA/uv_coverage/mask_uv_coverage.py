#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:46:37 2025

@author: nsaucourt
"""
################################## IMPORT ######################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator




#################################################################################
# Convert UV coords to spatial frequency (Mλ)
def uv_to_sf(u, v, wl, wlen_ref= 3.5e-6):
    return (u / wl *  wlen_ref, v / wl * wlen_ref)

#################################################################################
# Compute the mask of a uv-coverage
def mask_uv_coverage(uCoord,vCoord,wlen,wlen_ref = 3.5e-6,dx = 1e-3,dxy = 1000, plot = True):
    # Compute max SF radius
    R = 1.50 * np.max(np.hypot(uCoord, vCoord))
    xs = np.linspace(-R, R, dxy)
    ys = np.linspace(-R, R, dxy)
    X, Y = np.meshgrid(xs, ys)
    final_mask = np.zeros((dxy, dxy), dtype=bool)

    # Min and max of wlen
    wl0 = min(wlen)
    wlN = max(wlen)

    # Precompute grid in SF space for mask evaluation
    X_sf_w0, Y_sf_w0 = uv_to_sf(X, Y, wlen_ref, wlen_ref )
    X_sf_wN, Y_sf_wN = uv_to_sf(X, Y, wlen_ref, wlen_ref )
    pts_w0 = np.vstack([X_sf_w0.ravel(), Y_sf_w0.ravel()]).T
    pts_wN = np.vstack([X_sf_wN.ravel(), Y_sf_wN.ravel()]).T


    # Build masks for all tracks
    envelope_paths = []
    for i in range(6):
        u, v = uCoord[i], vCoord[i]
        
        # Build the envelope around the uv coverage
        def build_mask(u, v):
            u0, v0 = uv_to_sf(u, v, wl0, wlen_ref)
            uN, vN = uv_to_sf(u, v, wlN, wlen_ref)
            
            # Compute the polynome around the wl = 2.8 and wl = 4.9
            poly = np.vstack([np.column_stack((u0, v0)),        # first curve 
                              np.column_stack((uN, vN))[::-1]]) #second curve: closed loop 

            path = Path(poly)
            mask = path.contains_points(pts_w0) | path.contains_points(pts_wN)
            envelope_paths.append(path)
            return mask.reshape(final_mask.shape)
        
        
        
        final_mask |= build_mask(u, v)
        final_mask |= build_mask(-u, -v)
        
        
        
        
    ########################## PLOTTING ##########################
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
    
        # Plot Mask 
        cmap = ListedColormap(['lightgrey', 'white'])
        ax.imshow(final_mask.astype(int), extent=[-R, R, -R, R], origin='lower',
                  cmap=cmap, interpolation='nearest', zorder=0)
    
        # Plot tracks and UV curves
        colors = ['red', 'blue', 'lightgreen', 'orange', 'purple', 'cyan']
        for i in range(6):
            u, v = uCoord[i], vCoord[i]
            for sign in [+1, -1]:
                u_s, v_s = sign * u, sign * v
                ax.plot(u_s, v_s, color=colors[i], lw=2, zorder=3)
                ax.scatter(u_s, v_s, color=colors[i], zorder=4)
    
                # Plot SF lines at start/mid/end points
                for j in [0, len(u) // 2,len(u)-1]:
                    sf_u, sf_v = uv_to_sf(u_s[j], v_s[j], wlen, wlen_ref)
                    ax.plot(sf_u, sf_v, color=colors[i], lw=2)
    
        # Labels and aesthetics
        ax.set_title('uv-coverage mask', fontsize=16)
        ax.set_xlabel('U (Mλ)', fontsize=12)
        ax.set_ylabel('V (Mλ)', fontsize=12)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.grid(True, linestyle='--', linewidth=0.7)
        plt.tight_layout()
        plt.show()
    
    return final_mask



#################################################################################
# Compute the fft of the mask
def fft_mask(mask, plot = True):
    
    mask = mask.astype(int)
    ny, nx = mask.shape

    # Zero Padding factor 
    pad_factor = 10
    ny_pad = ny * pad_factor
    nx_pad = nx * pad_factor
    pady = (ny_pad - ny) // 2
    padx = (nx_pad - nx) // 2
    mask_padded = np.pad(mask, ((pady, pady), (padx, padx)), mode='constant')

    # Shifting the mask 
    shifted_mask = np.fft.fftshift(mask_padded)

    # Put the 0,0 at the sum
    shifted_mask[0,0]= np.sum(shifted_mask)

    # FFT
    fft = np.fft.fft2(shifted_mask,s=(ny_pad, nx_pad))
    shifted_fft = np.abs(np.fft.fftshift(fft))
    mean = np.median(shifted_fft)



    ########################## PLOT OF FFT ##########################
    if plot :
        plt.figure()
        plt.imshow(shifted_fft - mean,cmap = 'seismic',origin='lower', interpolation='nearest', aspect='equal' )
        plt.colorbar()
        plt.xlim(4500,5500)
        plt.ylim(4500,5500)
        plt.title("Magnitude of FFT")
    
    return shifted_fft