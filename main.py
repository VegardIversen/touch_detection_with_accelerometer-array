"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
from matplotlib import pyplot as plt
from main_scripts.estimate_touch_location import estimate_touch_location

from main_scripts.test_on_real_simulations import test_on_real_simulations
from utils.data_processing.preprocessing import crop_data, crop_to_signal
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
    NUMBER_OF_SENSORS = 10
    SORTED_ESTIMATED_ANGLES = [
        -42.182571606842359,
        -5.182571606842359,
        25.477947317871088,
        25.477947317872083,
    ]

    simulated_data = test_on_real_simulations(
        noise=False,
        filter_signals=True,
        number_of_sensors=NUMBER_OF_SENSORS,
    )
    simulated_data = crop_data(
        simulated_data,
        time_start=0.0005,
        time_end=0.001,
        apply_window_function=True,
    )

    # Pass sorted_estimated_angles_deg=None to export analytic signal,
    # or pass the SORTED_ESTIMATED_ANGLES to estimate touch location
    estimate_touch_location(
        measurements=simulated_data,
        sorted_estimated_angles_deg=SORTED_ESTIMATED_ANGLES,
        center_frequency_Hz=25000,
        propagation_speed_mps=954,
        crop_end_s=0.001,
        number_of_sensors=NUMBER_OF_SENSORS,
        sensor_spacing_m=0.01,
        actuator_coordinates=np.array([0.50, 0.35]),
    )

    # Plot all figures at the same time
    plt.show()


if __name__ == "__main__":
    main()
