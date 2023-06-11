import os
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

from main_scripts.estimate_touch_location import (
    estimate_touch_location_UCA,
    estimate_touch_location_ULA,
)
from main_scripts.test_on_real_simulations import prepare_simulation_data
from main_scripts.generate_ideal_signal import (
    compare_to_ideal_signal,
    generate_ideal_signal,
)
from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab
from main_scripts.physical_measurements import (
    combine_measurements_into_dataframe,
    measure_phase_velocity,
)
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals, set_fontsizes
from utils.global_constants import ACTUATOR_1, FIGURES_SAVE_PATH, SAMPLE_RATE, x, y
from utils.plate_setups import Setup5, Setup6


def full_touch_localization():
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
    """Select whether to use COMSOL simulation data or real measurements"""
    DATA_SOURCE = "Measurements"
    # DATA_SOURCE = "COMSOL"
    """Set parameters for the array"""
    CENTER_FREQUENCY_HZ = 22000
    PHASE_VELOCITY_MPS = 442.7 * 1
    GROUP_VELOCITY_MPS = 564.4 * 1
    # PHASE_VELOCITY_MPS = 442.7 * 1.79
    # GROUP_VELOCITY_MPS = 564.4 * 1.79
    NUMBER_OF_SENSORS = 7
    NUMBER_OF_SIGNALS = 3
    SENSOR_SPACING_M = 0.01
    ACTUATOR_COORDINATES = np.array([0.50, 0.20])
    UCA_CENTER_COORDINATES = np.array([0.05, 0.05])
    MEASUREMENTS_FILE_FOLDER = (
        f"Plate_10mm/Setup5/25kHz/"
        f"x{100 * ACTUATOR_COORDINATES[x]:.0f}"
        f"y{100 * ACTUATOR_COORDINATES[y]:.0f}"
    )
    FILTER_ORDER = 3
    FILTER_Q_VALUE = 0.05
    CROP_TIME_START = 0.000
    CROP_TIME_END = 0.001
    SPATIAL_SMOOTHING = 1
    FORWARD_BACKWARD = 1
    ATTENUATION_DBPM = 35
    FILE_VERSION = "v1"

    parameters = {
        "ARRAY_TYPE": ARRAY_TYPE,
        "CENTER_FREQUENCY_HZ": float(CENTER_FREQUENCY_HZ),
        "PHASE_VELOCITY_MPS": float(PHASE_VELOCITY_MPS),
        "GROUP_VELOCITY_MPS": float(GROUP_VELOCITY_MPS),
        "NUMBER_OF_SENSORS": float(NUMBER_OF_SENSORS),
        "NUMBER_OF_SIGNALS": float(NUMBER_OF_SIGNALS),
        "SENSOR_SPACING_M": float(SENSOR_SPACING_M),
        "ACTUATOR_COORDINATES": ACTUATOR_COORDINATES,
        "UCA_CENTER_COORDINATES": UCA_CENTER_COORDINATES,
        "FILTER_ORDER": float(FILTER_ORDER),
        "FILTER_Q_VALUE": float(FILTER_Q_VALUE),
        "FILE_FOLDER": MEASUREMENTS_FILE_FOLDER,
        "CROP_TIME_START": float(CROP_TIME_START),
        "CROP_TIME_END": float(CROP_TIME_END),
        "SPATIAL_SMOOTHING": float(SPATIAL_SMOOTHING),
        "FORWARD_BACKWARD": float(FORWARD_BACKWARD),
        "ATTENUATION": float(ATTENUATION_DBPM),
        "FILE_VERSION": FILE_VERSION,
    }
    print()
    print("Parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")
    print()

    # SETUP_FOR_DRAWING = Setup5(
    #     actuator_coordinates=[
    #         ACTUATOR_COORDINATES,
    #         np.array([0.45, 0.30]),
    #         np.array([0.55, 0.40]),
    #         np.array([0.45, 0.40]),
    #         np.array([0.55, 0.30]),
    #         np.array([0.50, 0.20]),
    #         np.array([0.50, 0.15]),
    #         np.array([0.50, 0.08]),
    #         np.array([0.50, 0.05]),
    #     ],
    #     number_of_sensors=NUMBER_OF_SENSORS,
    #     array_spacing_m=SENSOR_SPACING_M,
    # )
    # SETUP_FOR_DRAWING.draw()
    # plt.savefig(
    #     "results/physical_tests_setup.pdf",
    #     bbox_inches="tight",
    # )

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

    if DATA_SOURCE == "Measurements":
        measurements = combine_measurements_into_dataframe(
            file_folder=MEASUREMENTS_FILE_FOLDER,
            file_names=[
                f"sensors123_{FILE_VERSION}",
                f"sensors456_{FILE_VERSION}",
                f"sensors78_{FILE_VERSION}",
            ],
            setup=SETUP,
            sensitivites_should_be_corrected=True,
        )
    elif DATA_SOURCE == "COMSOL":
        measurements = prepare_simulation_data(
            array_type=ARRAY_TYPE,
            number_of_sensors=NUMBER_OF_SENSORS,
            crop=True,
            crop_start=CROP_TIME_START,
            crop_end=CROP_TIME_END,
        )
    else:
        raise ValueError("DATA_SOURCE must be either COMSOL or MEASUREMENTS")

    # compare_to_ideal_signal(
    #     setup=SETUP,
    #     measurements=measurements,
    #     attenuation_dBpm=ATTENUATION_DBPM,
    #     group_velocity_mps=GROUP_VELOCITY_MPS,
    #     signal_model="signal_generator_touch_pulse",
    #     critical_frequency=CENTER_FREQUENCY_HZ,
    #     filter_order=FILTER_ORDER,
    #     filter_q_value=FILTER_Q_VALUE,
    # )

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

    return


def import_estimated_angles(
    file_name: str,
):
    # Put the angles from results_simulations_10_mm_Teflon_COMSOL_25kHz_10sensors.csv into a dataframe
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


def extract_touch_pulse_from_signal_generator(
    measurements: pd.DataFrame,
    plot: bool = False,
    save: bool = False,
):
    measurements = crop_data(
        signals=measurements,
        time_start=0.0000,
        time_end=0.00015,
    )
    if plot:
        fig, ax = plt.subplots(squeeze=False)
        compare_signals(
            fig,
            ax,
            [measurements["Actuator"]],
            plots_to_plot=["time"],
        )
    if save:
        measurements["Actuator"].to_csv(
            os.path.join(
                "Measurements",
                "Plate_10mm",
                "Setup5",
                "25kHz",
                "signal_generator_touch_pulse.csv",
            ),
            index=False,
        )
