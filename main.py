"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from main_scripts.estimate_touch_location import (
    estimate_touch_location_UCA,
    estimate_touch_location_ULA,
)
from main_scripts.generate_ideal_signal import generate_ideal_signal
from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab
from main_scripts.physical_measurements import (
    combine_measurements_into_dataframe,
    measure_phase_velocity,
)
from utils.data_processing.preprocessing import crop_data, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals, set_fontsizes
from utils.global_constants import FIGURES_SAVE_PATH, SAMPLE_RATE, x, y
from utils.plate_setups import Setup5, Setup6


def main():
    set_fontsizes()

    matplotlib.use("TkAgg")
    # * Call one of the functions found in /main_scripts

    # * In theory these parameters should provide the best results:
    # - Many sensors
    # - Long array length (high aperture), but avoid near-field effects
    # - Small wavelength, so either high frequency or low phase velocity
    # - Long signal length, so that many periods are present.
    # - High SNR
    # - Low attenuation

    """Select array type ULA or UCA"""
    ARRAY_TYPE = "ULA"
    # ARRAY_TYPE = "UCA"
    """Set parameters for the array"""
    CENTER_FREQUENCY_HZ = 40000
    PHASE_VELOCITY_MPS = 850
    GROUP_VELOCITY_MPS = 1000
    NUMBER_OF_SENSORS = 7
    SENSOR_SPACING_M = 0.01
    ACTUATOR_COORDINATES = np.array([0.45, 0.40])
    UCA_CENTER_COORDINATES = np.array([0.05, 0.05])
    FILE_FOLDER = (
        f"Plate_10mm/Setup5/25kHz/"
        f"x{100 * ACTUATOR_COORDINATES[x]:.0f}"
        f"y{100 * ACTUATOR_COORDINATES[y]:.0f}"
    )
    FILTER_ORDER = 1
    FILTER_Q_VALUE = 0.01
    CROP_TIME_START = 0.00045
    CROP_TIME_END = 0.0009

    parameters = {
        "ARRAY_TYPE": ARRAY_TYPE,
        "CENTER_FREQUENCY_HZ": float(CENTER_FREQUENCY_HZ),
        "PHASE_VELOCITY_MPS": float(PHASE_VELOCITY_MPS),
        "GROUP_VELOCITY_MPS": float(GROUP_VELOCITY_MPS),
        "NUMBER_OF_SENSORS": float(NUMBER_OF_SENSORS),
        "SENSOR_SPACING_M": float(SENSOR_SPACING_M),
        "ACTUATOR_COORDINATES": ACTUATOR_COORDINATES,
        "UCA_CENTER_COORDINATES": UCA_CENTER_COORDINATES,
        "FILTER_ORDER": float(FILTER_ORDER),
        "FILTER_Q_VALUE": float(FILTER_Q_VALUE),
        "FILE_FOLDER": FILE_FOLDER,
        "CROP_TIME_START": float(CROP_TIME_START),
        "CROP_TIME_END": float(CROP_TIME_END),
    }
    print()
    print("Parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")
    print()

    if ARRAY_TYPE == "ULA":
        SETUP = Setup5(
            actuator_coordinates=ACTUATOR_COORDINATES,
            number_of_sensors=NUMBER_OF_SENSORS,
            array_spacing_m=SENSOR_SPACING_M,
        )
    elif ARRAY_TYPE == "UCA":
        SETUP = Setup6(
            actuator_coordinates=ACTUATOR_COORDINATES,
            array_center_coordinates=UCA_CENTER_COORDINATES,
            number_of_sensors=NUMBER_OF_SENSORS,
            array_spacing_m=SENSOR_SPACING_M,
        )
    else:
        raise ValueError("ARRAY_TYPE must be either ULA or UCA")

    measurements = combine_measurements_into_dataframe(
        file_folder=FILE_FOLDER,
        file_names=[
            "sensors123_v1",
            "sensors456_v1",
            "sensors78_v1",
        ],
        setup=SETUP,
        group_velocity_mps=GROUP_VELOCITY_MPS,
        center_frequency=CENTER_FREQUENCY_HZ,
        filter_order=FILTER_ORDER,
        filter_q_value=FILTER_Q_VALUE,
    )

    measurements = crop_data(
        signals=measurements,
        time_start=CROP_TIME_START,
        time_end=CROP_TIME_END,
    )

    measurements = filter_signal(
        signals=measurements,
        critical_frequency=CENTER_FREQUENCY_HZ,
        filtertype="bandpass",
        order=FILTER_ORDER,
        q=FILTER_Q_VALUE,
        plot_response=True,
        sample_rate=SAMPLE_RATE,  # type: ignore
    )

    # Pad the measurements with zeros after the end of the signal
    # measurements = measurements.append(
    #     pd.DataFrame(
    #         np.zeros((2 * measurements.shape[0], measurements.shape[1])),
    #         columns=measurements.columns,
    #     ),
    #     ignore_index=True,
    # )

    # Export the ideal signals
    generate_signals_for_matlab(
        parameters=parameters,
        measurements=measurements,
        center_frequency_Hz=CENTER_FREQUENCY_HZ,
        number_of_sensors=NUMBER_OF_SENSORS,
        array_type=ARRAY_TYPE,
    )

    estimated_angles_ULA = import_estimated_angles("results_angles_estimation_ULA")
    estimated_angles_UCA = import_estimated_angles("results_angles_estimation_UCA")

    if ARRAY_TYPE == "ULA":
        estimate_touch_location_ULA(
            setup=SETUP,
            sorted_estimated_angles_deg=estimated_angles_ULA,  # type: ignore
            center_frequency_Hz=CENTER_FREQUENCY_HZ,
            number_of_sensors=NUMBER_OF_SENSORS,
            sensor_spacing_m=SENSOR_SPACING_M,
            actuator_coordinates=ACTUATOR_COORDINATES,
        )
    elif ARRAY_TYPE == "UCA":
        estimate_touch_location_UCA(
            setup=SETUP,
            sorted_estimated_angles_deg=estimated_angles_UCA,  # type: ignore
            center_frequency_Hz=CENTER_FREQUENCY_HZ,
            number_of_sensors=NUMBER_OF_SENSORS,
            actuator_coordinates=ACTUATOR_COORDINATES,
            uca_center_coordinates=UCA_CENTER_COORDINATES,
        )
    else:
        raise ValueError("ARRAY_TYPE must be either ULA or UCA")

    plt.show()


def import_estimated_angles(
    file_name: str,
):
    # Put the angles from results_simULAtions_10_mm_Teflon_COMSOL_25kHz_10sensors.csv into a dataframe
    try:
        estimated_angles = pd.read_csv(
            f"{file_name}.csv",
        )
    # If file does not exist, get the file from Windows
    except FileNotFoundError:
        estimated_angles = pd.read_csv(
            f"/mnt/c/Users/nikla/Documents/GitHub/touch_detection_with_accelerometer-array/matlab/{file_name}.csv",
        )

    return estimated_angles


def plot_far_field(
    start_number_of_sensors: int = 3,
    end_number_of_sensors: int = 12,
    step_number_of_sensors: int = 1,
):
    # Plot the far field limit for multiple number of sensors, as a function of wavelength
    WAVELENGTHS = np.linspace(0.008, 0.03, 1000)
    NUMBER_OF_SENSORS = np.arange(
        start_number_of_sensors,
        end_number_of_sensors,
        step_number_of_sensors,
    )
    SENSOR_SPACING_M = WAVELENGTHS / 2
    fig, ax = plt.subplots(figsize=(10, 6))
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
    ax.legend(loc="upper left")
    ax.grid()
    # Save figure as pdf
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/far_field_limits.pdf",
        bbox_inches="tight",
    )

    plt.show()


def test_UCA_points():
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
