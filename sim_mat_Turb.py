################################################
# 
# Jules Scigliuto, Florentin Millour
# 
# 

import numpy as np
import matplotlib.pyplot as plt

################################################
# Functions

def twoDtukey(NAXIS, alpha):
    '''
    Create a 2D Tukey window

    Parameters:
    NAXIS (int): The size of the array
    alpha (float): The alpha parameter of the Tukey window

    Returns:
    ndarray: A 2D Tukey window
    '''
    axis = np.linspace(-NAXIS/2, NAXIS/2-1, NAXIS)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x**2 + y**2)
    tukey = np.zeros((NAXIS, NAXIS))
    tukey = np.where((r >= (1 - alpha) * NAXIS/2) & (r < NAXIS/2), 0.5 * (1 + np.cos(np.pi * (r - (1 - alpha) * NAXIS/2) / (alpha * NAXIS/2))), tukey)
    tukey = np.where(r < (1 - alpha) * NAXIS/2, 1, tukey)
    return np.double(tukey)

################################################

def dist(NAXIS):
    '''
    Create a 2D array with the distance from the center of the array

    Parameters:
    NAXIS (int): The size of the array

    Returns:
    tuple: A tuple containing the 2D array with distances, and the x and y coordinates of the array
    '''
    axis = np.linspace(-NAXIS/2, NAXIS/2-1, NAXIS)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x**2 + y**2)
    xnorm = 2 * x / NAXIS
    ynorm = 2 * y / NAXIS
    return r, x, y, xnorm, ynorm

################################################

def generate_phase_screen(dim, length, L0, r0, plot=False, seed="seed", filter_nmodes=0, rejection=100.):
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
    
    phase = np.random.uniform(low=-np.pi, high=np.pi, size=(dim, dim))
    rr, x_ps, y_ps, x_p, y_p = dist(dim)
    
    #print(x_ps, x_ps.shape)
    modul = (rr**2 + (length / L0)**2)**(-11./12.)
    
    if filter_nmodes > 0:
        nmradius = np.sqrt(np.pi * filter_nmodes)
        nmr_int = int(np.ceil(nmradius))
        # Modes antialiasing
        camembert_smooth = twoDtukey(nmr_int, 5./nmradius)
        if plot:
            plt.imshow(camembert_smooth)
            plt.show()
        
        if nmr_int % 2 == 0:
            pcs = np.pad(camembert_smooth, ((dim//2 - camembert_smooth.shape[0]//2, dim//2 - camembert_smooth.shape[0]//2),
                                            (dim//2 - camembert_smooth.shape[1]//2, dim//2 - camembert_smooth.shape[1]//2)), mode='constant')
        else:
            pcs = np.pad(camembert_smooth, ((dim//2 - camembert_smooth.shape[0]//2, dim//2 - camembert_smooth.shape[0]//2-1),
                                            (dim//2 - camembert_smooth.shape[1]//2, dim//2 - camembert_smooth.shape[1]//2-1)), mode='constant')
        #print('shape of pcs', np.shape(pcs))
        
        imod = np.where(rr > nmradius)
        modval = np.max(modul[imod])
        #print('modval',modval)
        #modul = np.where(rr > nmradius, modul, 0.1 * modval)
        modul = modul * (1-pcs) + modval * pcs * modul / rejection
    
    #modul = (x_ps**2 + y_ps**2 + (length / L0)**2)**(-11./12.)
    
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
    
        plt.imshow(phase)
        plt.imshow(np.fft.fftshift(phase))
        plt.colorbar().set_label('energy')
        plt.title('phase of FFT of phase screen')
        plt.show()
    
    screen0 = np.fft.fftshift(modul) * np.exp(1j * phase)
    #print(screen0)
    screen  = np.fft.fft2(screen0).real
    screen2 = np.fft.fftshift(screen)
    fact    = np.sqrt(2)*np.sqrt(.0228)*(length/r0)**(5./6.)# * (2*np.pi) / wl 
    #print('fact',fact)
    screen3 = fact * screen2
    screen3 -= np.mean(screen3)
    screen3 *= 1# 2.2 / 3.5
    screen3 *= 1 / 3.5

    scaling = length/dim
    #print('scaling',scaling)

    if plot:
        xm = np.linspace(0, dim, dim) * scaling #1pix = 9.7mm
        #plt.imshow(screen.real * 1e6, origin='lower', extent=(np.min(xm), np.max(xm), np.min(xm), np.max(xm))) #phase screen on a 10-meter square
        plt.imshow(screen3) #phase screen on a 10-meter square
        plt.colorbar().set_label('Astmospheric phase (radians)')
        plt.title('Real part of phase screen')
        plt.show()

    return screen3, scaling, x_ps

################################################

def generate_pupil_obstr(dim, D, D_obstr, plot=False):
    """
    Generates the pupil of the VLT.

    Parameters:
    - dim (int): Dimension of the wavefront screen.
    - D (float): Diameter of the pupil (in meters).
    - D_obstr (float): Diameter of the central obscuration (in meters).

    Returns:
    - pupil (ndarray): Pupil of the VLT.
    """
    x = np.linspace(-D / 2, D / 2, dim)
    xp, yp = np.meshgrid(x, x)
    rp = np.sqrt(xp**2 + yp**2)
    
    pupil = (rp < D / 2) * (rp > D_obstr / 2)
    #pupil = twoDtukey(dim, 2./dim) * (1-twoDtukey(dim*D_obstr/D, 2./(dim/2)))
    
    if plot:
        plt.imshow(pupil, origin='lower', extent=(-D / 2, D / 2, -D / 2, D / 2))
        plt.title('Pupil of the VLT')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    return pupil

################################################

def generate_turbulent_psf(D, sz, phase_screen, plot=False, n_pad=2):
    """
    Generates the electric field and point spread function (PSF) for a given phase screen.

    Parameters:
    - D (float): Diameter of the pupil (in meters).
    - Physical size of the phase screen (in meters).
    - phase_screen (ndarray): Phase screen representing the wavefront distortion.

    Returns:
    - E_img (ndarray): Electric field in the image plane.
    - psf (ndarray): Point spread function.
    - x_f (ndarray): x-coordinates of the points in the Fourier plane.
    """
    dim = phase_screen.shape[0]
    x = np.linspace(-sz / 2, sz / 2, dim)
    xmin = np.min(x)
    xmax = np.max(x)
    step = x[1] - x[0]
    
    pupil = generate_pupil_obstr(dim, D, 1.2, plot=plot)
    
    pscreen = pupil * np.exp(1j * phase_screen)
    wp = np.where(pscreen == 0)
    pscreen[wp] = 0;
    
    paddim = 2**int(np.ceil(np.log2(dim * n_pad)))
    #print('paddim',paddim)
    padfact2 = int(paddim // 2 - dim/2)
    #print('padfact2',padfact2)
    ppscreen = np.pad(pscreen, ((padfact2, padfact2), (padfact2, padfact2)), mode='constant')
    x_pad = np.arange(-paddim/2*step, paddim/2*step, step)
    min_xp = np.min(x_pad)
    max_xp = np.max(x_pad)
    
    E_img = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ppscreen)))
    psf = np.abs(E_img)**2

    #print(E_img.shape)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
        
        ax1.imshow(np.angle(pscreen), origin='lower', extent=(xmin, xmax, xmin, xmax))
        ax1.set_title('Turbulence on the Pupil')
        #ax1.set_xlim(-8, 8)
        #ax1.set_ylim(-8, 8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        ax2.imshow(np.angle(ppscreen), origin='lower', extent=(min_xp, max_xp, min_xp, max_xp))
        ax2.set_title('Turbulence on the Pupil (zero padded)')
        #ax1.set_xlim(-8, 8)
        #ax1.set_ylim(-8, 8)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        ax3.imshow(np.log(psf))
        #ax3.imshow(psf)
        ax3.set_title('PSF')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')

        plt.tight_layout()
        plt.show()
    
    return E_img, psf, x_pad


################################################

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

dim  = 100 #pix -> 1024pix = 10m -> 1pix = 9.7mm
phsz = 8.0 #m physical size of the phase screen (in meters)
L0   = 30.0 # outer scale m
D    = 8.0 # telescope diameter m
D_obs = 1.2 # central obscuration m
r0   = 0.2 # Fried diameter m
lamb = 0.5e-6 #m
lam  = 3.5e-6 #m 
sep  = 534 #mas
sep_rad = (sep * 1e-3 / 3600) * (np.pi / 180) #Séparation étoile-planète en rad
sep_lam = sep_rad / (lam / D) #Séparation étoile-planète en lam/D
holediam = 1.5 #lam/D at 3.5µm
pl_flx = 8e-3 #Jy
st_flx = 10 #Jy
rt_flx = pl_flx / st_flx #unitless

plot = False

# Generate phase screen
#screen, scaling, x_ps = generate_phase_screen(dim, phsz, L0, r0, plot, seed="seed", filter_nmodes=800)

#screenM = screen / 2 / np.pi * lam

PSF50 = []
for i in range(200):
    print(i)
    screen, scaling, x_ps = generate_phase_screen(dim, phsz, L0, r0, plot=plot, seed="random", filter_nmodes=50)
    E_img, psf, ps_dim = generate_turbulent_psf(D, phsz, screen, plot=plot, n_pad=6)
    PSF50.append(psf)
average_PSF50 = np.mean(PSF50, axis=0)

PSF500 = []
for i in range(200):
    print(i)
    screen, scaling, x_ps = generate_phase_screen(dim, phsz, L0, r0, plot=plot, seed="random", filter_nmodes=500)
    E_img, psf, ps_dim = generate_turbulent_psf(D, phsz, screen, plot=plot, n_pad=6)
    PSF500.append(psf)
average_PSF500 = np.mean(PSF500, axis=0)


dim = average_PSF500.shape[0]
#print('dim',dim)

fig0, ax0 = plt.subplots(2, 2, figsize=(15, 8))

ax0[0,0].imshow(np.log(average_PSF50))
ax0[1,0].imshow(np.log(average_PSF500))

slice50   = average_PSF50[dim//2,dim//2:]
#slice50  /= np.max(slice50)
slice500  = average_PSF500[dim//2,dim//2:]
#slice500 /= np.max(slice500)
ax0[0,1].plot(slice50)
ax0[0,1].plot(slice500)
ax0[0,1].set_yscale('log')
ax0[1,1].plot(slice500/slice50)
ax0[1,1].set_yscale('log')

plt.tight_layout()
plt.show()

stop()

psf_on      = np.abs(E_onstar )**2
psf_off     = np.abs(E_offstar)**2
psf_offplan = np.abs(Eplan_off)**2
psf_onplan  = np.abs(Eplan_on )**2

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
