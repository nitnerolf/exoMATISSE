# ###############################################
# 
# Jules Scigliuto
# 
# 
########################################################
# Figure 12 of Houllé et al. 2025
# Plot of flux vs separation for exoplanets
# The fluxes are taken from the literature
# or extrapolated from existing photometry using exoplanets models
# The separations are taken from the literature
# The fluxes are in mJy
# The separations are in mas


import matplotlib.pyplot as plt
import numpy as np

flux = (8.3,    # beta Pic b:    Lagrange+2010 Magnitude converted into Jy
        4,      # HD984 b:       Meshkat+2015 Magnitude converted into Jy
        4,      # beta Pic c:    Nowak+2020 planet parameters, spectrum simulated with ATMO models (Phillips+2020)
        1,      # 
        1.2,    # HD206893 b:    Hinkley+2023 planet parameters, spectrum simulated with ATMO models (Phillips+2020)
        0.4,    # PDS70 b/c:     Wang+2021 planet parameters, spectrum simulated with ATMO models (Phillips+2020)
        0.2,    # PDS70 b/c:     Wang+2021 planet parameters, spectrum simulated with ATMO models (Phillips+2020)
        0.4,    # HIP21152 B:    Bonavita+2022 spectrum simulated with ATMO models (Phillips+2020)
        0.3,    # HR8799 b/c/d:  Currie+2011 Magnitude converted into Jy
        0.3,    # HR8799 b/c/d:  Currie+2011 Magnitude converted into Jy
        0.3,    # HR8799e (d/c): Doelman+2022 Direct Flux
        0.2,    # AF Lep b:      Mesa+2023/De Rosa+2023/Franson+2023/Zhang+2023/Gratton+2024 spectrum simulated with ATMO models (Phillips+2020)
        0.2,    # HIP65426 b:    Janson+2012 spectrum simulated with ATMO models (Phillips+2020)
        0.1,    # 51 Eri b:      Brown-Sevilla+2022 spectrum simulated with ATMO models (Phillips+2020)
        0.3,    # HR8799 b/c/d:  Currie+2011 Magnitude converted into Jy
        0.13,   # HD 19467 B:    Greenbaum+2023 NACO spectrum
        1.7)    # PZ Tel b:      Musso Barucci+2019 spectrum simulated with ATMO models (Phillips+2020)

















names = [r'$\beta$ Pic b',
         'HD984 b',
         r'$\beta$ Pic c',
         'ABAur b',
         'HD206893 b',
         'PDS70 b',
         'PDS70 c',
         'HIP21152 B', 
        'HR8799 c',
        'HR8799 d',
        'HR8799 e',
        'AF Lep b',
        'HIP65426 b',
        '51 Eri b',
        'HR8799 b',
        'HD 19467 B',
        'PZ Tel B']

sep = [559,
       241,
       83,
       569,
       190,
       149,
       208,
       403,
       956,
       699,
       402,
       312,
       813,
       283,
       1720,
       1600,
       500] #mas

print(len(names), len(flux), len(sep))

for i, name in enumerate(names):
 plt.text(sep[i], flux[i], name, ha='center', va='bottom', rotation=25)

plt.scatter(sep, flux, marker='.')
plt.xticks(rotation=45)
plt.xlabel('Separation [mas]')
plt.ylabel('Flux [mJy]')
plt.yscale('log')
plt.ylim(0, 3*10**1)
plt.xscale('log')
#plt.title('Flux vs Separation')
plt.show()