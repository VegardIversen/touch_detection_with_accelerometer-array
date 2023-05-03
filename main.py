"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
from matplotlib import pyplot as plt

from main_scripts.estimate_touch_location import estimate_touch_location
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
        -32.057085649677958,
        -27.269618204913666,
        10.623327537454102,
        14.118904906478052,
    ]
    estimate_touch_location(
        sorted_estimated_angles=SORTED_ESTIMATED_ANGLES,
        center_frequency=200000,
        t_var=1e-7,
        propagation_speed_mps=992,
        snr_dB=40,
        attenuation_dBpm=0,
        crop_end=None,
        number_of_sensors=20,
        sensor_spacing_m=0.00248,
        actuator_coordinates=np.array([0.50, 0.20]),
    )

    # Plot all figures at the same time
    plt.show()


if __name__ == "__main__":
    main()
