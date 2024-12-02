################################################
# 
# Jules Scigliuto, Florentin Millour
# with the help of Marcel Carbillet
# 

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import jv


################################################
# Functions

def twoDtukey(NAXIS, alpha, tukDiam='same'):
    '''
    Create a 2D Tukey window

    Parameters:
    NAXIS (int): The size of the array
    alpha (float): The alpha parameter of the Tukey window

    Returns:
    ndarray: A 2D Tukey window
    '''
    if tukDiam == 'same':
        tukDiam = NAXIS
    axis = np.linspace(-NAXIS/2, NAXIS/2-1, NAXIS)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x**2 + y**2)
    
    tukey = np.zeros((NAXIS, NAXIS))
    tukey = np.where((r >= tukDiam/2 - alpha * NAXIS/2) & (r <= tukDiam/2), 0.5 * (1 + np.cos(np.pi * (r - tukDiam/2 - alpha * NAXIS/2) / (alpha * NAXIS/2))), tukey)
    tukey = np.where(r < tukDiam/2 - alpha * NAXIS/2, 1, tukey)
    
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

def generate_pupil_obstr(dim, D=8, D_obstr=1.2, plot=False):
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
    pupil  = twoDtukey(dim, 4./dim)
    pupil2 = twoDtukey(dim, 4./dim, tukDiam=dim*D_obstr/D)
    pupil *= (1 - pupil2)
    
    #plot=1
    
    if plot:
        #plt.imshow(pupil, origin='lower', extent=(-D / 2, D / 2, -D / 2, D / 2))
        plt.imshow(pupil, origin='lower')
        plt.title('Pupil of the VLT')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    return pupil

################################################

def generate_phase_screen_subharmonics(dim, length, L0, sha):
    '''
    directly inspired from the IDL routines of Marcel Carbillet
    '''
    # Additional parameters
    nsub = 3
    null = nsub // 2

    # Initialize the low-frequencies screen
    low_freq_screen = np.zeros((dim, dim), dtype=np.complex128)

    # Frequency (modulus), fx & fy (coordinates) "nsub*nsub" frequency arrays init.
    freq_x = np.float64(np.tile(np.arange(nsub) - 1, (nsub, 1)))
    #print(freq_x)
    freq_y = np.rot90(freq_x)
    #print(freq_y)
    freq_mod = np.sqrt(freq_x**2 + freq_y**2)
    #print(freq_mod)

    # xx and yy "dim*dim" screens
    xx = np.tile(np.arange(dim) / dim, (dim, 1))
    yy = xx.T

    # Cycle over order of sub-division (depth) of the null freq. px
    depth = 0
    while depth != sha:
        depth += 1

        phase = np.random.uniform(-0.5, 0.5, (nsub, nsub))
        freq_mod /= nsub
        freq_mod[null, null] = 1.0

        freq_x /= nsub
        freq_y /= nsub

        modul = (freq_mod**2 + (length / L0)**2)**(-11/12.)
        modul[null, null] = 0.0

        for i in range(nsub):
            for j in range(nsub):
                sh = np.exp(2j * np.pi * (xx * freq_x[i, j] + yy * freq_y[i, j] + phase[i, j]))
                sh0 = np.sum(sh) / dim**2

                low_freq_screen += (1.0 / nsub**depth) * modul[i, j] * (sh - sh0)

    low_freq_screen *= np.sqrt(0.0228) * dim**(5/6.)

    return low_freq_screen.real

################################################

def generate_phase_screen(dim, length, L0, r0, r0_lambda, plot=False, seed="seed", filter_nmodes=0, rejection=8.):
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

    Directly inspired from the IDL routines of Marcel Carbillet
    [marcel.carbillet@unice.fr],
    lab. Lagrange (UCA, OCA, CNRS), Feb. 2013.
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
    
    # Set amplitude of mode 0 to 0
    #modul[np.where(rr==0)] = 0.0
    #compute subharmonics
    subh_term = generate_phase_screen_subharmonics(dim, length, L0, 9)
    
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
        modul = modul * (1-pcs) + pcs * modul / rejection
    
    #modul = (x_ps**2 + y_ps**2 + (length / L0)**2)**(-11./12.)
    
    if plot:
        plt.figure()
        plt.plot(np.sqrt(x_ps**2 + y_ps**2), modul)
        plt.xlabel('radius element number')
        plt.ylabel('energy')
        plt.title('amplitude of FFT of phase screen')
        plt.loglog()
    
        plt.figure()
        plt.imshow(modul)
        plt.imshow(np.fft.fftshift(np.log(modul)))
        plt.colorbar().set_label('energy')
        plt.title('amplitude of FFT of phase screen')
    
        plt.figure()
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
    screen4 = screen3 + subh_term / rejection    
    screen5 = screen4 - np.mean(screen4)
    
    # Take wavelength of definition of r0
    #screen3 *= 2.2 / 3.5
    screen6 = screen5 * r0_lambda / (2*np.pi)
    
    scaling = length/dim
    #print('scaling',scaling)

    if plot:
        xm = np.linspace(0, dim, dim) * scaling #1pix = 9.7mm
        #plt.imshow(screen.real * 1e6, origin='lower', extent=(np.min(xm), np.max(xm), np.min(xm), np.max(xm))) #phase screen on a 10-meter square
        plt.imshow(screen6) #phase screen on a 10-meter square
        plt.colorbar().set_label('Astmospheric phase (radians)')
        plt.title('Real part of phase screen')
        plt.show()

    return screen6, scaling, x_ps

################################################

def generate_turbulent_psf(D, sz, phase_screen, lambda_obs, plot=False, n_pad=2, gen_perfect_psf=False):
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
    phase_screen = phase_screen * 2 * np.pi / lambda_obs
    
    dim = phase_screen.shape[0]
    x = np.linspace(-sz / 2, sz / 2, dim)
    xmin = np.min(x)
    xmax = np.max(x)
    step = x[1] - x[0]
    
    pupil = generate_pupil_obstr(dim, D, 1.2, plot=plot)
    
    pscreen = np.copy(pupil) * np.exp(1j * phase_screen)
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

    if gen_perfect_psf:
        # Generate a perfect PSF for comparison
        pup_pad = np.pad(pupil, ((padfact2, padfact2), (padfact2, padfact2)), mode='constant')
        E_perfect = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(pup_pad)))
        psf_perfect = np.abs(E_perfect)**2 # perfect PSF
    
    
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
    
    if gen_perfect_psf:
        return E_img, psf, x_pad, E_perfect, psf_perfect
    else:
        return E_img, psf, x_pad




################################################
################################################

dim  = 100 #pix -> 1024pix = 10m -> 1pix = 9.7mm
n_pad= 4
phsz = 8.0 #m physical size of the phase screen (in meters)
L0   = 30.0 # outer scale m
D    = 8.0 # telescope diameter m
D_obs = 1.2 # central obscuration m
r0   = 0.1 # Fried diameter m
r0_lamb = 0.5e-6 #m
rejection = 8.
#lam  = 0.655e-6 #m 
#lam  = 1.2e-6 #m 
#lam  = 2.2e-6 #m 
lam  = 3.5e-6 #m 
sep  = 534 #mas
sep_rad = (sep * 1e-3 / 3600) * (np.pi / 180) #Séparation étoile-planète en rad
sep_lam = sep_rad / (lam / D) #Séparation étoile-planète en lam/D
holediam = 1.5 #lam/D at 3.5µm
pl_flx = 8e-3 #Jy
st_flx = 10 #Jy
rt_flx = pl_flx / st_flx #unitless
nrepeat = 250

#plot = True
plot = False

# Generate phase screen
#screen, scaling, x_ps = generate_phase_screen(dim, phsz, L0, r0, plot, seed="seed", filter_nmodes=800)

#screenM = screen / 2 / np.pi * lam

PSF50 = []
for i in range(nrepeat):
    print(i)
    screen, scaling, x_ps = generate_phase_screen(dim, phsz, L0, r0, r0_lamb, plot=plot, seed="random", filter_nmodes=50, rejection=rejection)
    if i == 0:
        E_img, psf, ps_dim, E_perfect, psf_perfect = generate_turbulent_psf(D, phsz, screen, lam, plot=plot, n_pad=n_pad, gen_perfect_psf=True)
    else:
        E_img, psf, ps_dim = generate_turbulent_psf(D, phsz, screen, lam, plot=plot, n_pad=n_pad)
    PSF50.append(psf)
average_PSF50 = np.mean(PSF50, axis=0)
average_PSF_PERFECT = psf_perfect

PSF500 = []
for i in range(nrepeat):
    print(i)
    screen, scaling, x_ps = generate_phase_screen(dim, phsz, L0, r0, r0_lamb, plot=plot, seed="random", filter_nmodes=500, rejection=rejection)
    E_img, psf, ps_dim = generate_turbulent_psf(D, phsz, screen, lam, plot=plot, n_pad=n_pad)
    PSF500.append(psf)
average_PSF500 = np.mean(PSF500, axis=0)


dim = average_PSF500.shape[0]
#print('dim',dim)

fig0, ax0 = plt.subplots(2, 3, figsize=(15, 8))

plt.suptitle(f'Comparison of PSF with 50 and 500 modes ({lam*1e6} µm)')

ax0[0,0].set_title('Average PSF with 50 modes')
ax0[0,0].imshow(np.log(average_PSF50))

ax0[0,1].set_title('Average PSF with 500 modes')
ax0[0,1].imshow(np.log(average_PSF500))

ax0[0,2].set_title('Perfect PSF')
ax0[0,2].imshow(np.log(average_PSF_PERFECT))

slice50   = average_PSF50[dim//2,dim//2:]
#slice50  /= np.max(slice50)
slice500  = average_PSF500[dim//2,dim//2:]
slicePERFECT  = average_PSF_PERFECT[dim//2,dim//2:]
#slice500 /= np.max(slice500)

ax0[1,0].set_title('Slice of PSF')
ax0[1,0].plot(slicePERFECT, label='Perfect')
ax0[1,0].plot(slice50, label='50 modes')
ax0[1,0].plot(slice500, label='500 modes')
ax0[1,0].set_yscale('log')
mx = np.max(slicePERFECT)
ax0[1,0].set_ylim(mx/1e6, mx)
ax0[1,0].set_xlim(0, 200)
ax0[1,0].legend()

maxp = np.max(slicePERFECT)
ax0[1,1].set_title('Strehl ratio')
ax0[1,1].plot(slicePERFECT / maxp, label='Perfect')
ax0[1,1].plot(slice50 / maxp, label='50 modes')
ax0[1,1].plot(slice500 / maxp, label='500 modes')
ax0[1,1].set_xlim(0, 2)
ax0[1,1].set_ylim(0, 1)
ax0[1,1].legend()

ax0[1,2].set_title('Speckle background improvement')
ax0[1,2].plot(slicePERFECT/slice50, label='Perfect/50 modes')
ax0[1,2].plot(slice500/slice50, label='500 modes/50 modes')
#ax0[1,2].set_yscale('log')
ax0[1,2].set_ylim(0, 1.5)
ax0[1,2].legend()

plt.tight_layout()

print('gain in speckle background',np.mean((slice500/slice50)[30:90]))

plt.show()
