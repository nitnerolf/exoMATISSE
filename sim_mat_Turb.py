# ###############################################
# 
# Jules Scigliuto
# 
# 

import numpy as np
import matplotlib.pyplot as plt

################################
#Functions

def dist(NAXIS):
    '''
    Create a 2D array with the distance from the center of the array

    Parameters:
    NAXIS (int): The size of the array

    Returns:
    tuple: A tuple containing the 2D array with distances, and the x and y coordinates of the array
    '''
    axis = np.linspace(-NAXIS/2, NAXIS/2-1, NAXIS)
    
    print(axis)
    
    
    result = np.sqrt(axis**2 + axis[:,np.newaxis]**2)
    x, y = np.meshgrid(axis, axis)
    xnorm = 2 * x / NAXIS
    ynorm = 2 * y / NAXIS
    return np.roll(result, int(NAXIS/2+1), axis=(0,1)), x, y, xnorm, ynorm

def wfgeneration(dim, dimm, length, L0, r0, wl, plot, seed="seed"):
    """
    Generates a wavefront screen using the specified parameters.

    The model of the simulated phase screen has been taken from the equation (3) of Carbillet et al. 2010.

    Parameters:
    - dim (int): Dimension of the wavefront screen.
    - length (float): Length of the wavefront screen.
    - L0 (float): Outer scale of the turbulence.
    - r0 (float): Fried parameter of the turbulence.
    - wl (float): Wavelength of the light.
    - seed (str or int, optional): Seed for the random number generator. 
      If "seed" or "Seed" or "SEED", a fixed seed of 1 is used. 
      If "random" or "Random" or "RANDOM", a random seed is used. 
      If an integer is provided, that integer is used as the seed. 
      Defaults to "seed".

    Returns:
    - screen (ndarray): Generated wavefront screen.
    - x_ps (ndarray): x-coordinates of the points on the screen.
    - y_ps (ndarray): y-coordinates of the points on the screen.
    """

    if seed in ["seed", "Seed", "SEED"]:
        np.random.seed(1)
    elif isinstance(seed, int):
        np.random.seed(seed)
    elif seed in ["random", "Random", "RANDOM"]: 
        np.random.seed()
    
    phase = np.random.uniform(-np.pi, np.pi, (dim, dim))
    rr, x_ps, y_ps, x_p, y_p = dist(dim)
    
    print(x_ps, x_ps.shape)
    #modul = (rr**2+(length/L0)**2)**(-11/12)
    modul = (x_ps**2 + y_ps**2 + (length / L0)**2)**(-11./12.)
    
    if plot:
        plt.plot(np.sqrt(x_ps**2 + y_ps**2), modul)
        plt.xlabel('radius element number')
        plt.ylabel('energy')
        plt.title('amplitude of FFT of phase screen')
        plt.loglog()
        plt.show()
    
        plt.imshow(modul)
        plt.imshow(np.fft.fftshift(np.log(modul)))
        plt.colorbar().set_label('energy')
        plt.title('amplitude of FFT of phase screen')
        plt.show()
    
    screen = np.fft.ifft2(np.fft.fftshift(modul)*np.exp(1j*phase))
    screen2 = np.fft.fftshift(screen)
    fact   = np.sqrt(2)*np.sqrt(.0228)*(length/r0)**(5./6.)# * (2*np.pi) / wl 
    print('fact',fact)
    screen3 = fact * screen2
    # screen -= np.mean(screen)

    scaling = dimm/dim
    print(scaling)

    if plot:
        xm = np.linspace(0, dim, dim) * scaling #1pix = 9.7mm
        #plt.imshow(screen.real * 1e6, origin='lower', extent=(np.min(xm), np.max(xm), np.min(xm), np.max(xm))) #phase screen on a 10-meter square
        plt.imshow(screen3.real) #phase screen on a 10-meter square
        plt.colorbar().set_label('Astmospheric piston [µm]')
        plt.title('Real part of phase screen')
        plt.show()

    return screen, scaling, x_ps

def generate_electric_field(D, wl, phase_screen, plot):
    """
    Generates the electric field and point spread function (PSF) for a given phase screen.

    Parameters:
    - D (float): Diameter of the pupil.
    - wl (float): Wavelength of the light.
    - phase_screen (ndarray): Phase screen representing the wavefront distortion.
    - x_ps (ndarray): x-coordinates of the points on the phase screen.

    Returns:
    - E_img (ndarray): Electric field in the image plane.
    - psf (ndarray): Point spread function.
    - x_f (ndarray): x-coordinates of the points in the Fourier plane.
    """
    # phase_screen *= 2*np.pi/wl 
    # ones_array = np.ones_like(phase_screen)
    # phase_screen = ones_array

    ps_dim = phase_screen.shape[0] * dimm / dim #phase screen dimension in meters
    n_pad = 8

    x = np.linspace(-n_pad * ps_dim / 2, n_pad * ps_dim / 2, x_ps.shape[0])
    xp, yp = np.meshgrid(x, x)
    rp = np.sqrt(xp**2 + yp**2)
    pupil = (rp < D / 2)
    pupil = pupil.astype(np.complex128)
    
    pscreen = pupil * phase_screen
    # pupil *= np.exp(1j * phase_screen) 
    
    E_img = np.fft.fftshift(np.fft.fft2(pscreen))
    psf = np.abs(E_img)**2

    print(E_img.shape, ps_dim)

    if plot:
        fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=(15, 15))

        ax2.imshow(np.log(psf))
        ax2.imshow(psf)
        ax2.set_title('PSF')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        ax1.imshow(pscreen.real, origin='lower', extent=(np.min(x), np.max(x), np.min(x), np.max(x)))
        ax1.set_title('Turbulence on the Pupil')
        ax1.set_xlim(-8, 8)
        ax1.set_ylim(-8, 8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        plt.tight_layout()
        plt.show()
    
    return E_img, psf, ps_dim

def MATISSE(E_img, dim, ps_dim, holediam, wl, sep, x_ps, plot):
    '''
    Generates the electric field for the on-axis star, off-axis star, on-axis planet, and off-axis planet.

    Parameters:
    - E_img (ndarray): Electric field in the image plane.
    - dim (int): Dimension of the wavefront screen.
    - holediam (float): Diameter of the pinhole.
    - wl (float): Wavelength of the light.
    - sep (float): Separation between the star and the planet.
    - x_ps (ndarray): x-coordinates of the points on the phase screen.

    Returns:
    - E_onstar (ndarray): Electric field of the on-axis star.
    - E_offstar (ndarray): Electric field of the off-axis star.
    - Eplan_off (ndarray): Electric field of the off-axis planet.
    - Eplan_on (ndarray): Electric field of the on-axis planet.
    '''

    # x_f = np.fft.fftshift(np.fft.fftfreq(x_ps[0], (2*n_pad*E_img[0]*scaling)/x_ps[0]))*D #spatial frequency in rad
    x_f = np.fft.fftshift(np.fft.fftfreq(x_ps.shape[0], np.max(x_ps) / x_ps.shape[0])) * D
    print(x_f)

    # Pinhole
    x_ph, y_ph = np.meshgrid(x_f, x_f)
    r_ph = np.sqrt(x_ph**2 + y_ph**2)
    pinhole = (r_ph < ((holediam / 3.5) * (wl * 1e6 / 2)))

    # Shifted pinhole
    sep_rad = (sep * 1e-3 / 3600) * (np.pi / 180) #Séparation étoile-planète en rad
    sep_lam = sep_rad / (wl / D) #Séparation étoile-planète en lam/D

    x_sph = x_ph - sep_lam
    y_sph = y_ph
    r_sph = np.sqrt(x_sph**2 + y_sph**2)
    pinhole_off = (r_sph < ((holediam / 3.5) * (wl * 1e6 / 2)))

    # Planète coordonnées + champ électrique shiftée
    coordxy = np.where(r_sph == np.min(r_sph)) #Coordonnées de la planète dans r_sph
    Eplan_img = np.roll(E_img, coordxy[1][0] - dim // 2, axis=1) #Chp électrique de planète à sa position

    # Pupil stop
    x = np.linspace(-np.max(x_ps) / 2, np.max(x_ps) / 2, x_ps.shape[0])
    xp, yp = np.meshgrid(x, x)
    rp = np.sqrt(xp**2 + yp**2)
    pup_stop = rp < D / 2

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

        ax1.imshow(pinhole, extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f)))
        ax1.set_title('Pinhole')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)

        ax2.imshow(pup_stop, extent=(np.min(x), np.max(x), np.min(x), np.max(x)))
        ax2.set_title('Pupil Stop')
        ax2.set_xlim(-8, 8)
        ax2.set_ylim(-8, 8)
    
        plt.tight_layout()
        plt.show()

    # Filtrage spatial sur l'étoile on- et off-axis
    sf = pinhole * E_img
    sf2 = pinhole_off * E_img
    sf2_c = np.roll(sf2, dim // 2 - coordxy[1][0], axis=1)

    # Filtrage spatial sur la planète off- et on-axis
    sf3 = pinhole * Eplan_img
    sf4 = pinhole_off * Eplan_img
    sf4_c = np.roll(sf4, dim // 2 - coordxy[1][0], axis=1)

    if plot:
        pl_flx = 8e-3 #Jy
        st_flx = 10 #Jy
        rt_flx = pl_flx / st_flx #unitless
    
        fig, ax = plt.subplots(2, 2)
        fig.colorbar(ax[0, 0].imshow(np.log(np.abs(sf)), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f))))
        ax[0, 0].set_title('Star through pinhole')
        ax[0, 0].set_xlim(-1, 1)
        ax[0, 0].set_ylim(-1, 1)

        fig.colorbar(ax[0, 1].imshow(np.log(np.abs(sf2_c)), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f))))
        ax[0, 1].set_title('Star through offset pinhole')
        ax[0, 1].set_xlim(-1, 1)
        ax[0, 1].set_ylim(-1, 1)

        fig.colorbar(ax[1, 0].imshow(np.log(np.abs(sf3 * rt_flx)), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f))))
        ax[1, 0].set_title('Planet through pinhole')
        ax[1, 0].set_xlim(-1, 1)
        ax[1, 0].set_ylim(-1, 1)

        fig.colorbar(ax[1, 1].imshow(np.log(np.abs(sf4_c * rt_flx)), origin='lower', extent=(np.min(x_f), np.max(x_f), np.min(x_f), np.max(x_f))))
        ax[1, 1].set_title('Planet through offset pinhole')
        ax[1, 1].set_xlim(-1, 1)
        ax[1, 1].set_ylim(-1, 1)
        plt.tight_layout()

    ### Calcul de la PSF de l'étoile et de la planète
    pup_stop_flag = True
    
    # Etoile on-axis
    # E_onstar = np.fft.fftshift(np.fft.fft2(sf)) / dim #after the spatial filtering
    E_onstar = np.fft.fft2(sf) / dim
    if pup_stop_flag:
        E_pup_stop = E_onstar * pup_stop #pupil stop
        E_onstar = np.fft.fft2(E_pup_stop) / dim #after the pupil stop

    # Etoile off-axis
    # E_offstar = np.fft.fftshift(np.fft.fft2(sf2_c)) / dim 
    E_offstar = np.fft.fft2(sf2_c) / dim
    if pup_stop_flag:
        E_pup_off_stop = E_offstar * pup_stop
        E_offstar = np.fft.fft2(E_pup_off_stop) / dim

    # Planète off-axis
    # Eplan_off = np.fft.fftshift(np.fft.fft2(sf3)) / dim 
    Eplan_off = np.fft.fft2(sf3) / dim
    if pup_stop_flag:
        Eplan_pup_off_stop = Eplan_off * pup_stop
        Eplan_off = np.fft.fft2(Eplan_pup_off_stop) / dim

    # Planète on-axis
    # Eplan_on = np.fft.fftshift(np.fft.fft2(sf4_c)) / dim 
    Eplan_on = np.fft.fft2(sf4_c) / dim 
    if pup_stop_flag:
        Eplan_pup_on_stop = Eplan_on * pup_stop
        Eplan_on = np.fft.fft2(Eplan_pup_on_stop) / dim

    ##############################################
    #Module Optic Anamorphose
    
    return E_onstar, E_offstar, Eplan_off, Eplan_on, x_f

dim  = 1024 #pix -> 1024pix = 10m -> 1pix = 9.7mm
dimm = 16. #m physical size of the phase screen
L0   = 30.0 #m
D    = 8.0 #m
r0   = 0.1 #m
lamb = 0.5e-6 #m
lam  = 3.5e-6 #m 
sep  = 534 #mas
sep_rad = (sep * 1e-3 / 3600) * (np.pi / 180) #Séparation étoile-planète en rad
sep_lam = sep_rad / (lam / D) #Séparation étoile-planète en lam/D
holediam = 1.5 #lam/D at 3.5µm
pl_flx = 8e-3 #Jy
st_flx = 10 #Jy
rt_flx = pl_flx / st_flx #unitless

plot = True

# Generate phase screen
screen, scaling, x_ps = wfgeneration(dim, dimm, D, L0, r0, lamb, plot, seed="seed")

#E_img, psf, ps_dim = generate_electric_field(D, lamb, screen, plot)

#E_onstar, E_offstar, Eplan_off, Eplan_on, x_f = MATISSE(E_img, dim, ps_dim, holediam, lam, sep, x_ps, plot)

stop()

psf_on      = np.abs(E_onstar)**2
psf_off     = np.abs(E_offstar)**2
psf_offplan = np.abs(Eplan_off)**2
psf_onplan  = np.abs(Eplan_on)**2

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(x_f, psf_on[dim // 2] / np.max(psf_on), label='On-axis Star')
ax.plot(x_f + sep_lam, psf_off[dim // 2] / np.max(psf_on), label='Off-axis Star')
ax.plot(x_f + sep_lam, psf_onplan[dim // 2] / np.max(psf_on) * rt_flx, label='On-axis Planet')
ax.plot(x_f, psf_offplan[dim // 2] / np.max(psf_on) * rt_flx, label='Off-axis Planet')

ax.set_yscale('log')
ax.set_ylim(10**-14, 2) 
ax.set_xlim(-10, 10)
ax.set_title('PSF Comparison')
ax.set_xlabel(r'Spatial Frequency $\frac{\lambda}{D}$')
ax.set_ylabel('Normalized Intensity')
ax.legend()
ax.legend()
ax.axvline(1.22, color='red', linestyle='--', label=r'1.22 $\frac{\lambda}{D}$')
ax.axvline(sep_lam, color='green', linestyle='--', label='Planet Position')
ax.legend()

ax2 = ax.twiny()
ax2.set_xlim(-10 * (sep / sep_lam), 10 * (sep / sep_lam))
ax2.set_xlabel('mas')
plt.show()

plt.tight_layout()
plt.show()
