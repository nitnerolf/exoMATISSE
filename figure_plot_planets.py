# ###############################################
# 
# Jules Scigliuto, Florentin Millour
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
# Data for exoplanets
exoplanets = [
        {"name": r'$\beta$ Pic b', "flux": 8.34, "fluxMes": 1, "sep": 552, "sep5yrs": 129, "sep10yrs": 440, "sepMin": 8,    "sepMax": 567, "comment": "beta Pic b: Lagrange+2010 Magnitude converted into Jy"},
        {"name": r'$\beta$ Pic c', "flux": 3.97,    "fluxMes": 0, "sep": 15, "sep5yrs": 552, "sep10yrs": 552, "sepMin": 1.65, "sepMax": 183,  "comment": "beta Pic c: Nowak+2020 planet parameters, spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'PDS70 b',        "flux": 0.3,  "fluxMes": 1, "sep": 145, "sep5yrs": 121, "sep10yrs": 110, "sepMin": 200 * np.cos(45*np.pi/180),  "sepMax": 200, "comment": "PDS70 b: Wang+2021 planet parameters, spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'PDS70 c',        "flux": 0.15, "fluxMes": 1, "sep": 207, "sep5yrs": 203, "sep10yrs": 204, "sepMin": 266 * np.cos(45*np.pi/180),  "sepMax": 266, "comment": "PDS70 c: Wang+2021 planet parameters, spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'HD984 b',        "flux": 4.1,    "fluxMes": 1, "sep": 243, "sep5yrs": 171, "sep10yrs": 205, "sepMin": 71,   "sepMax": 1046, "comment": "HD984 b: Meshkat+2015 Magnitude converted into Jy"},
        {"name": 'ABAur b',        "flux": 0.6,  "fluxMes": 0, "sep": 600, "sep5yrs": 1, "sep10yrs": 1, "sepMin": 288,  "sepMax": 913, "comment": "Flux estimé par Jules a partir de Biddle+2024 et Zhou+2022"},
        {"name": 'HD206893 b',     "flux": 0.5,  "fluxMes": 0, "sep": 193, "sep5yrs": 201, "sep10yrs": 206, "sepMin": 186, "sepMax": 285, "comment": "HD206893 b: Hinkley+2023 planet parameters, spectrum simulated with ATMO models (Phillips+2020) + ExoREM"},
        {"name": 'HIP21152 B',     "flux": 0.25, "fluxMes": 0, "sep": 370, "sep5yrs": 1, "sep10yrs": 1, "sepMin": 22,  "sepMax": 534, "comment": "HIP21152 B: Bonavita+2022 spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'HR8799 b',       "flux": 0.3,  "fluxMes": 1, "sep": 1719, "sep5yrs": 1716, "sep10yrs": 1713, "sepMin": 1523,"sepMax": 1726,"comment": "HR8799 b: Currie+2011 Magnitude converted into Jy"},
        {"name": 'HR8799 c',       "flux": 0.3,  "fluxMes": 1, "sep": 956, "sep5yrs": 966, "sep10yrs": 979, "sepMin": 961, "sepMax": 1089, "comment": "HR8799 c: Currie+2011 Magnitude converted into Jy"},
        {"name": 'HR8799 d',       "flux": 0.4,  "fluxMes": 1, "sep": 700, "sep5yrs": 701, "sep10yrs": 690, "sepMin": 545, "sepMax": 753, "comment": "HR8799 d: Currie+2011 Magnitude converted into Jy"},
        {"name": 'HR8799 e',       "flux": 0.3,  "fluxMes": 1, "sep": 402, "sep5yrs": 421, "sep10yrs": 434, "sepMin": 321, "sepMax": 479, "comment": "HR8799e (d/c): Doelman+2022 Direct Flux"},
        {"name": 'AF Lep b',       "flux": 0.29, "fluxMes": 1, "sep": 297, "sep5yrs": 202, "sep10yrs": 334, "sepMin": 175, "sepMax": 339, "comment": "AF Lep b: Mesa+2023/De Rosa+2023/Franson+2023/Zhang+2023/Gratton+2024 spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'HIP65426 b',     "flux": 0.18, "fluxMes": 0, "sep": 813, "sep5yrs": 800, "sep10yrs": 784, "sepMin": 116, "sepMax": 1234, "comment": "HIP65426 b: Janson+2012 spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": '51 Eri b',       "flux": 0.07, "fluxMes": 0, "sep": 266, "sep5yrs": 201, "sep10yrs": 418, "sepMin": 138, "sepMax": 562, "comment": "51 Eri b: Brown-Sevilla+2022 spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'HD19467 B',     "flux": 0.17,  "fluxMes": 1, "sep": 1581, "sep5yrs": 1545, "sep10yrs": 1496, "sepMin": 617,"sepMax": 2431,"comment": "HD 19467 B: Greenbaum+2023 NACO spectrum"},
        {"name": 'PZ Tel B',       "flux": 1.3,  "fluxMes": 0, "sep": 696, "sep5yrs": 779, "sep10yrs": 848, "sepMin": 7, "sepMax": 797, "comment": "PZ Tel b: Musso Barucci+2019 spectrum simulated with ATMO models (Phillips+2020)"},
        {"name": 'HD206893 c',     "flux": 0.5,  "fluxMes": 0, "sep": 56, "sep5yrs": 54, "sep10yrs": 74, "sepMin": 46,"sepMax": 128,"comment": ""}
        
]

print(len(exoplanets))

for exoplanet in exoplanets:
        #plt.text((exoplanet["sepMin"] + exoplanet["sepMax"]) / 2, exoplanet["flux"], exoplanet["name"], ha='center', va='bottom', rotation=25)
        color = 'blue' if exoplanet["fluxMes"] == 1 else 'red'
        if exoplanet["name"] == r'$\beta$ Pic c':
                plt.text(55, exoplanet["flux"], " "+exoplanet["name"], ha='left', va='bottom', rotation=22, color=color)
        elif exoplanet["name"] == 'HD206893 c':
                plt.text(55, exoplanet["flux"], " "+exoplanet["name"], ha='left', va='bottom', rotation=22, color=color)
        elif exoplanet["name"] == 'HIP21152 B':
                plt.text(exoplanet["sep"], exoplanet["flux"], " "+exoplanet["name"], ha='left', va='center', rotation=22, color=color)
        else:   
                plt.text(exoplanet["sep"], exoplanet["flux"], " "+exoplanet["name"], ha='left', va='baseline', rotation=22, color=color)
        plt.scatter(exoplanet["sepMin"], exoplanet["flux"], marker='.', color=color, alpha=0.2)
        plt.scatter(exoplanet["sepMax"], exoplanet["flux"], marker='.', color=color, alpha=0.2)
        plt.scatter(exoplanet["sep"], exoplanet["flux"], marker='*', color=color)
        plt.plot([exoplanet["sepMin"], exoplanet["sepMax"]], [exoplanet["flux"], exoplanet["flux"]], color=color, alpha=0.2)

plt.xticks(rotation=45)
plt.xlabel('Separation [mas]')
plt.ylabel('Flux [mJy]')
plt.yscale('log')
plt.ylim(0, 15)
plt.xscale('log')
#plt.title('Flux vs Separation')
plt.xlim(70)
plt.tight_layout()

plt.savefig('figure_planets.png', dpi=300)
plt.show()

