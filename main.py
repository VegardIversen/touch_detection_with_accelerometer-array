"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from main_scripts.estimate_touch_location import estimate_touch_location

from main_scripts.test_on_real_simulations import test_on_real_simulations
from utils.data_processing.preprocessing import crop_data
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
    NUMBER_OF_SENSORS = 15
    estimated_angles_filename = (
        "results_angles_estimation"
    )
    sorted_estimated_angles = import_estimated_angles(estimated_angles_filename)

    simulated_data = test_on_real_simulations(
        noise=False,
        filter_signals=True,
        number_of_sensors=NUMBER_OF_SENSORS,
    )
    simulated_data = crop_data(
        simulated_data,
        time_start=0.0006,
        time_end=0.002,
        apply_window_function=True,
    )

    # Pass sorted_estimated_angles_deg=None to export analytic signal,
    # or pass the sorted_estimated_angles to estimate touch location
    estimate_touch_location(
        measurements=simulated_data,
        sorted_estimated_angles_deg=sorted_estimated_angles,
        center_frequency_Hz=23000,
        propagation_speed_mps=434,
        crop_end_s=0.001,
        number_of_sensors=NUMBER_OF_SENSORS,
        sensor_spacing_m=0.01,
        actuator_coordinates=np.array([0.50, 0.35]),
    )

    # Plot all figures at the same time
    plt.show()


def import_estimated_angles(
    file_name: str,
):
    # Put the angles from results_simulations_10_mm_Teflon_COMSOL_25kHz_10sensors.csv into a dataframe
    estimated_angles = pd.read_csv(
        f"{file_name}.csv",
    )
    return estimated_angles


if __name__ == "__main__":
    main()
