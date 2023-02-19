import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lambwaves import Lamb


def teflon_plate():
    E = 575e6          # E = Young's modulus, in Pa.
    p = 2175            # p = Density (rho), in kg/m3.
    v = 0.42         # v = Poisson's ratio (nu). #Polytetrafluoroethylene

    c_L = np.sqrt(E*(1-v) / (p*(1+v)*(1-2*v)))
    c_S = np.sqrt(E / (2*p*(1+v)))
    c_R = c_S * ((0.862+1.14*v) / (1+v))

    # Example: A 10 mm aluminum plate.

    teflon = Lamb(thickness=10, 
                nmodes_sym=5, 
                nmodes_antisym=5, 
                fd_max=10000, 
                vp_max=15000, 
                c_L=c_L, 
                c_S=c_S, 
                c_R=c_R, 
                material='teflon')

    # Plot phase velocity, group velocity and wavenumber.

    teflon.plot_phase_velocity()
    teflon.plot_group_velocity()
    teflon.plot_wave_number()

    # Plot wave structure (displacement profiles across thickness) for A0 
    # and S0 modes at different fd values.

    teflon.plot_wave_structure(mode='A0', nrows=3, ncols=2, 
                            fd=[500,1000,1500,2000,2500,3000])

    teflon.plot_wave_structure(mode='S0', nrows=4, ncols=2, 
                            fd=[500,1000,1500,2000,2500,3000,3500,4000])

    # Generate animations for A0 and S0 modes at 1000 kHz mm.

    teflon.animate_displacement(mode='S0', fd=1000)
    teflon.animate_displacement(mode='A0', fd=1000)

    # Save all results to a txt file.

    teflon.save_results()

    plt.show()