
# ###############################################
# #
# Jules Scigliuto
# 
# 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tkinter as tk

# ###############################################
# #
# Notations : #
# #
# sf = Spatial Filtering #
# c = Centered # 
# off/on = off/on-axis #
# plan = Planète #
# flx = Flux #
# psd = power spectral density #
# ##############################################

##### Paramètres 
lam = 3.5e-6 #Longueur d'onde
n_pix = 1024 #Nombre de pixel
D = 8 #Diamètre du télescope
D1 = 50
sep = 534 #Séparation étoile-planète en mas
sep_rad = (sep*1e-3/3600) * (np.pi/180) #Séparation étoile-planète en rad
sep_lam = sep_rad/(lam/D) #Séparation étoile-planète en lam/D
plan_flx = 8e-3 #Flux de la planète en Jy
star_flx = 10 #Flux de l'étoile en Jy
holediam = 1.5 #En lam/D à 3.5µm
a = 10 #Borne pour le plot final en lam/D

plotquisertarien = False

r0_500nm = 0.1 #Paramètre de Fried à 500 nm
r0 = r0_500nm * (lam/500e-9)**(6/5) #Rayon de Fried en mètre à lam souhaitée
k0 = 2*np.pi/D1 #Nombre d'onde (grande échelle)

print(sep_lam)

##### Masque de phase - Kolmogorov
fx, fy = np.meshgrid((np.arange(n_pix) - n_pix//2)*2*np.pi/D, (np.arange(n_pix) - n_pix//2)*2*np.pi/D)
freq = np.sqrt(fx**2 + fy**2)
k_psd = 0.023*r0**(-5/3)*freq**(-11/3)
k_psd[n_pix//2,n_pix//2] = 0.023*r0**(-5/3)*k0**(-11/3)

random_table = np.random.random((n_pix,n_pix))
random_table_spectrum = np.fft.fft2(random_table)
k_phase_screen_spectrum = random_table_spectrum * np.sqrt(k_psd)

k_phase_screen = np.real(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(k_phase_screen_spectrum))))

if plotquisertarien:
    plt.figure(figsize=(14,4))
    plt.subplot(131)
    plt.imshow(np.abs(random_table_spectrum)**2,origin='lower',norm=LogNorm())
    plt.title('Random spectrum (white noise)')
    plt.subplot(132)
    plt.imshow(np.abs(k_phase_screen_spectrum)**2,origin='lower',norm=LogNorm())
    plt.title('PSD of the turbulence')
    plt.subplot(133)
    plt.imshow(k_phase_screen,origin='lower')
    plt.title('Phase screen')
    plt.show()


##### Modélisation du plan pupille + masque de phase
npad = 12
x = np.linspace(-npad*D, npad*D, n_pix)
xx,yy = np.meshgrid(x,x)
r = np.sqrt(xx**2+yy**2)
pupil = (r<D/2)
pupil = 1.0*pupil
pupil /= pupil.sum()

if plotquisertarien:
    plt.figure()
    plt.imshow(pupil, origin='lower', extent=(np.min(x), np.max(x), np.min(x), np.max(x)))
    plt.xlim(-5,5)
    plt.ylim(-5,5)

    plt.figure()
    plt.imshow(k_phase_screen*pupil)
    

##### Calcul de la PSF d'un point source à travers un télescope de 8
E_img = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pupil)))
psf = np.abs(E_img*E_img)
x_f = np.fft.fftshift(np.fft.fftfreq(n_pix,(2*npad*D)/n_pix))*D

if plotquisertarien:
    ##### Cut de PSF de d'un point source à travers un télescope de 8m
    plt.figure()
    plt.plot(x_f, psf[n_pix//2]/np.max(psf))
    plt.xlim(0,20)
    plt.xlabel('λ/D');
    plt.yscale('log')
    plt.ylabel('$I_{normalized}$')
    plt.ylim(10e-7)

    ##### PSF de d'un point source à travers un télescope de 8m
    plt.figure()
    plt.imshow(np.log(psf), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f)))
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    #print(x_f)

##### Modélisation du pinhole
x2, y2 = np.meshgrid(x_f, x_f)
r2 = np.sqrt(x2**2+y2**2)
pinhole = (r2<((holediam/3.5e-6)*(lam/2)))

##### Shifted pinhole
x3 = x2-sep_lam
y3 = y2
r3 = np.sqrt(x3**2+y3**2)
pinhole_off = (r3<((holediam/3.5e-6)*(lam/2)))

###### Planète
coordxy = np.where(r3==np.min(r3)) #Coordonnées de la planète dans r3
Eplan_img = np.roll(E_img, coordxy[1][0]-n_pix//2, axis=1) #Champ électrique de la planète à sa position

##### Filtrage spatial par le pinhole pour étoile on-axis
sf = pinhole * E_img

##### Filtrage spatial par le pinhole pour étoile off-axis
sf2 = pinhole_off * E_img
sf2_c = np.roll(sf2, n_pix//2-coordxy[1][0], axis=1)

##### Filtrage spatial par le pinhole pour planète off-axis
sf3 = pinhole * Eplan_img

##### Filtrage spatial par le pinhole pour planète on-axis
sf4 = pinhole_off * Eplan_img
sf4_c = np.roll(sf4, n_pix//2-coordxy[1][0], axis=1)

if plotquisertarien:
    plt.figure()
    plt.imshow(np.abs(Eplan_img), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f)))
    plt.title('Eplan')
    plt.xlim(-15,15)
    plt.ylim(-5,5)

    plt.figure()
    plt.imshow(np.abs(sf4), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f)))
    plt.title('sf')
    plt.xlim(-10,10)
    plt.ylim(-5,5)

##### Intensité du signal du point source à la sortie du pinhole de MATISSE 
# I_pin = np.abs(pinhole*pinhole)

# plt.figure();
# plt.plot(x_f, I_pin[n_pix//2]/np.max(I_pin), 'r');
# plt.yscale('log');
# plt.xlabel('λ/D');
# plt.xlim(0,2)

##### Modélisation de l'OTF de la pupille
E_pup_on = np.fft.fftshift(np.fft.fft2(sf))/n_pix
otf_pup = np.abs(E_pup_on*E_pup_on)

E_pup_off = np.fft.fftshift(np.fft.fft2(sf2_c))/n_pix 
# otf_pup_filt = np.abs(E_pup_off*E_pup_off)

Eplan_pup_off = np.fft.fftshift(np.fft.fft2(sf3))/n_pix 
# otf_pup_plan = np.abs(Eplan_pup_off*Eplan_pup_off)

Eplan_pup_on = np.fft.fftshift(np.fft.fft2(sf4_c))/n_pix 
# otf_pup_plan = np.abs(Eplan_pup_on*Eplan_pup_on)

x_otf = np.linspace(0,n_pix,n_pix )
otf_abs = np.abs(Eplan_pup_on)

if plotquisertarien:
    plt.figure()
    plt.plot(x_f, otf_abs[n_pix//2])
    plt.title('OTF')

    plt.figure()
    plt.imshow(otf_abs)


##### Cut de l'OTF 
# plt.figure()
# plt.plot(x, otf_pup[n_pix//2])
# plt.xlim(0,20)
# plt.yscale('log')
# plt.ylim(10e-7)

##### OTF 
# plt.figure()
# plt.imshow(otf_pup/np.max(otf_pup))

##### 'Pupil Stop' 
pup_stop = r<D/2

E_pup_stop = E_pup_on*pup_stop

E_pup_off_stop = E_pup_off*pup_stop

Eplan_pup_off_stop = Eplan_pup_off*pup_stop

Eplan_pup_on_stop = Eplan_pup_on*pup_stop

if plotquisertarien:
    plt.figure()
    plt.imshow(np.abs(Eplan_pup_off_stop))

##### PSF du point source après le 'Pupil Stop'
E_onstar = np.fft.fft2(E_pup_stop)/n_pix
psf_on = np.abs(E_onstar*E_onstar)

E_offstar = np.fft.fft2(E_pup_off_stop)/n_pix
psf_off= np.abs(E_offstar*E_offstar)

Eplan_off = np.fft.fft2(Eplan_pup_off_stop)/n_pix
psf_offplan = np.abs(Eplan_off*Eplan_off)

Eplan_on = np.fft.fft2(Eplan_pup_on_stop)/n_pix
psf_onplan = np.abs(Eplan_on*Eplan_on)

plt.rcParams['legend.title_fontsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15


##### Plots finals 
fig, ax1 = plt.subplots(figsize=(8,5))

ax1.plot(x_f+sep_lam, psf_off[n_pix//2]/np.max(psf_on), 'mediumblue', label='Star')
ax1.plot(x_f+sep_lam, (psf_onplan[n_pix//2]/np.max(psf_on))*(plan_flx/star_flx), 'darkred', linestyle = '-', label='Planet')

ax1.axvline(x=0, color='mediumblue', linestyle='--')
ax1.axvline(x=sep_lam, color='darkred', linestyle='--')

#ax1.axvline(x=sep_lam+holediam/2, color='grey', linestyle=':')
#ax1.axvline(x=sep_lam-holediam/2, color='grey', linestyle=':')
ax1.axvspan(sep_lam-holediam/2, sep_lam+holediam/2, facecolor='mintcream', edgecolor='grey', alpha=1, label='Pinhole', linestyle=':')

ax1.set_xlim(-2.5,a)
ax1.set_xlabel('Separation (λ/D)', fontsize=15)
ax1.set_yscale('log')
ax1.set_ylabel('Normalized intensity', fontsize=15)
ax1.set_ylim(1e-12, 2)
#ax1.get_yaxis().set_visible(False)
ax1.set_facecolor('lightgrey')

ax1.legend(title="MATISSE flux", loc='lower right', title_fontproperties={'weight':'bold'})

#plt.rcParams["figure.figsize"] = (8, 6)

ax2 = ax1.twiny()

ax2.get_xaxis().set_visible(False)
ax2.set_xlim(-2.5*(sep/sep_lam), a*(sep/sep_lam))
#ax2.set_xlabel('Separation (mas)', fontsize=15)

plt.tight_layout()

plt.savefig('figbetapic1.pdf', format='pdf')
plt.savefig('figbetapic1.png', format='png')
#plt.show()



##############################################################



fig, ax3 = plt.subplots(figsize=(8,5))

ax3.plot(x_f, psf_on[n_pix//2]/np.max(psf_on), 'mediumblue', label='Star')
ax3.plot(x_f, (psf_offplan[n_pix//2]/np.max(psf_on))*(plan_flx/star_flx), 'darkred', linestyle='-', label='Planet')

ax3.axvline(x=0, color='mediumblue', linestyle='--')
ax3.axvline(x=sep_lam, color='darkred', linestyle='--')

ax3.axvspan(-holediam/2, holediam/2, facecolor='mintcream', alpha=1, edgecolor='grey', label='Pinhole', linestyle=':')

ax3.set_xlim(-2.5,a)
#ax3.set_xlabel('Separation (λ/D)', fontsize=15)
ax3.set_yscale('log')
ax3.set_ylabel('Normalized intensity', fontsize=15)
ax3.set_ylim(1e-12,2)
ax3.legend(title="MATISSE flux",loc='upper right', title_fontproperties={'weight':'bold'})


ax3.set_facecolor('lightgrey')
ax3.get_xaxis().set_visible(False)

ax4 = ax3.twiny()


ax4.set_xlim(-2.5*(sep/sep_lam), a*(sep/sep_lam))
ax4.set_xlabel('Separation (mas)', fontsize=15)
plt.tight_layout()
plt.savefig('figbetapic2.pdf', format='pdf')
plt.savefig('figbetapic2.png', format='png')
plt.show()