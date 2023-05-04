"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
from matplotlib import pyplot as plt

from main_scripts.test_on_real_simulations import test_on_real_simulations
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()

    # * Call one of the functions found in /main_scripts

    # * In theory these parameters should provide the best results:
    # - Many sensors
    # - Long array length (high aperture)
    # - Small wavelength, so either high frequency or low phase velocity
    # - Long signal length, so that many periods are present.
    # - High SNR
    # - Low attenuation
    SORTED_ESTIMATED_ANGLES = [
        -30.328182409403528,
        -25.332412794369677,
        -0.909610510877382,
        4.218204890523926,
    ]
    test_on_real_simulations()

    # Plot all figures at the same time
    plt.show()


if __name__ == "__main__":
    main()
