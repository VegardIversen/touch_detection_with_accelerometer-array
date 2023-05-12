"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from main_scripts.estimate_touch_location import estimate_touch_location
from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab
from main_scripts.test_on_real_simulations import test_on_real_simulations
from utils.data_visualization.visualize_data import set_fontsizes
from utils.global_constants import FIGURES_SAVE_PATH
from utils.plate_setups import Setup5, Setup6


def main():
    set_fontsizes()

    # * Call one of the functions found in /main_scripts

    # * In theory these parameters should provide the best results:
    # - Many sensors
    # - Long array length (high aperture), but avoid
    # - Small wavelength, so either high frequency or low phase velocity
    # - Long signal length, so that many periods are present.
    # - High SNR
    # - Low attenuation

    SETUP_UCA = Setup6(
        actuator_coordinates=np.array([0.50, 0.35]),
        number_of_sensors=8,
        array_spacing_m=0.01,
    )
    SETUP_UCA.draw()
    plt.show()

    NUMBER_OF_SENSORS = 7
    estimated_angles_filename = "results_angles_estimation"
    sorted_estimated_angles = import_estimated_angles(
        estimated_angles_filename,
        s0=False,
    )

    CENTER_FREQUENCY_HZ = 22000

    simulated_data = test_on_real_simulations(
        noise=False,
        crop=False,
        number_of_sensors=NUMBER_OF_SENSORS,
        critical_frequency_Hz=0,
        filter_order=4,
    )

    SENSOR_SPACING_M = 0.01
    SETUP = Setup5(
        actuator_coordinates=np.array([0.50, 0.35]),
        number_of_sensors=NUMBER_OF_SENSORS,
        array_spacing_m=SENSOR_SPACING_M,
    )

    analytic_signals = generate_signals_for_matlab(
        simulated_data,
        center_frequency_Hz=CENTER_FREQUENCY_HZ,
        propagation_speed_mps=434,
        crop_end_s=0.001,
        number_of_sensors=NUMBER_OF_SENSORS,
    )

    # Pass sorted_estimated_angles_deg=None to export analytic signal,
    # or pass the sorted_estimated_angles to estimate touch location
    estimate_touch_location(
        setup=SETUP,
        sorted_estimated_angles_deg=sorted_estimated_angles,
        center_frequency_Hz=CENTER_FREQUENCY_HZ,
        number_of_sensors=NUMBER_OF_SENSORS,
        sensor_spacing_m=SENSOR_SPACING_M,
        actuator_coordinates=SETUP.actuator_coordinates,
    )

    # Plot all figures at the same time
    plt.show()


def import_estimated_angles(
    file_name: str,
    s0: bool = False,
):
    # Put the angles from results_simulations_10_mm_Teflon_COMSOL_25kHz_10sensors.csv into a dataframe
    estimated_angles = pd.read_csv(
        f"{file_name}.csv",
    )
    if s0:
        # Switch place between rows at index 1 and 3 in sorted_estimated_angles_deg
        estimated_angles.iloc[[1, 3]] = estimated_angles.iloc[[3, 1]].values
        estimated_angles.iloc[[2, 3]] = estimated_angles.iloc[[3, 2]].values

    return estimated_angles


def plot_far_field():
    # Plot the far field limit for multiple number of sensors, as a function of wavelength
    WAVELENGTHS = np.linspace(0.008, 0.03, 1000)
    NUMBER_OF_SENSORS = np.arange(3, 12, 2)
    SENSOR_SPACING_M = WAVELENGTHS / 2
    fig, ax = plt.subplots()
    for number_of_sensors in NUMBER_OF_SENSORS:
        ax.plot(
            1000 * WAVELENGTHS,
            1000
            * (2 * ((number_of_sensors - 1) * SENSOR_SPACING_M) ** 2)
            / WAVELENGTHS,
            label=f"{number_of_sensors} sensors",
            # Pick colors that can be reused later
            color=f"C{list(NUMBER_OF_SENSORS).index(number_of_sensors)}",
        )
    WAVELENGTHS_DASHED = np.linspace(0, 0.008, 1000)
    SENSOR_SPACING_M_DASHED = WAVELENGTHS_DASHED / 2
    for number_of_sensors in NUMBER_OF_SENSORS:
        ax.plot(
            1000 * WAVELENGTHS_DASHED,
            1000
            * (2 * ((number_of_sensors - 1) * SENSOR_SPACING_M_DASHED) ** 2)
            / WAVELENGTHS_DASHED,
            linestyle="--",
            # color=f"C{list(NUMBER_OF_SENSORS).index(number_of_sensors)}",
            color="lightgray",
        )
    # Show x-axis in mm
    ax.set_xlabel("Wavelength [mm]")
    ax.set_ylabel("Far Field Limit [mm]")
    ax.set_ylim(0, 500)
    ax.set_xlim(0, 30)
    ax.legend(loc="upper right")
    ax.grid()
    # Save figure as pdf
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/far_field_limits.pdf",
        bbox_inches="tight",
    )

    plt.show()


def test_uca_points():
    # Define the points as tuples
    points = [
        (6.307, 5),
        (5.924, 5.924),
        (5, 6.307),
        (4.076, 5.924),
        (3.693, 5),
        (4.076, 4.076),
        (5, 3.693),
        (5.924, 4.076),
    ]

    # Extract the x and y coordinates into separate lists
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the points using a scatter plot
    ax.scatter(x_coords, y_coords)

    # Set the axis labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_ylim(0, 8)
    ax.set_xlim(0, 8)
    # Set figure size
    fig.set_size_inches(5, 5)
    ax.set_title("Scatter plot of points")

    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()
