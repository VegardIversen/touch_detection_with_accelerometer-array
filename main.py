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
        -30.328182409403528,
        -25.332412794369677,
        -0.909610510877382,
        4.218204890523926,
    ]
    estimate_touch_location(
        sorted_estimated_angles_deg=SORTED_ESTIMATED_ANGLES,
        center_frequency_Hz=25000,
        t_var=1e-8,
        propagation_speed_mps=250,
        snr_dB=40,
        attenuation_dBpm=0,
        crop_end_s=None,
        number_of_sensors=30,
        sensor_spacing_m=0.005,
        actuator_coordinates=np.array([0.50, 0.15]),
    )

    # Plot all figures at the same time
    plt.show()


if __name__ == "__main__":
    main()
