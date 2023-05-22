"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
import pandas as pd
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
from main_scripts.plot_speed_information import plot_speed_information
from main_scripts.test_on_real_simulations import test_on_real_simulations_UCA
from utils.data_processing.preprocessing import crop_data, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals, set_fontsizes
from utils.global_constants import FIGURES_SAVE_PATH, SAMPLE_RATE
from utils.plate_setups import Setup5, Setup6


def main():
    set_fontsizes()

    plot_speed_information()

    # * Call one of the functions found in /main_scripts

    # * In theory these parameters should provide the best results:
    # - Many sensors
    # - Long array length (high aperture), but avoid
    # - Small wavelength, so either high frequency or low phase velocity
    # - Long signal length, so that many periods are present.
    # - High SNR
    # - Low attenuation

    """Select array type ULA or UCA"""
    # ARRAY_TYPE = "ULA"
    ARRAY_TYPE = "UCA"
    """Set parameters for the array"""
    CENTER_FREQUENCY_HZ = 20000
    PHASE_VELOCITY_MPS = 442.7
    GROUP_VELOCITY_MPS = 564.4
    NUMBER_OF_SENSORS = 8
    SENSOR_SPACING_M = 0.01
    ACTUATOR_COORDINATES = np.array([0.50, 0.35])
    UCA_CENTER_COORDINATES = np.array([0.05, 0.05])

    # measurements = combine_measurements_into_dataframe(
    #     f"Plate_10mm/Setup5/{str(CENTER_FREQUENCY_HZ - 10000)[:2]}kHz",
    #     "1period_pulse_from_middle_to_sensors_123",
    #     "1period_pulse_from_middle_to_sensors_456",
    #     "1period_pulse_from_middle_to_sensors_78",
    # )

    measurements = test_on_real_simulations_UCA(
        noise=False,
        crop=True,
        number_of_sensors=NUMBER_OF_SENSORS,
        critical_frequency_Hz=CENTER_FREQUENCY_HZ,
        filter_order=1,
    )

    # measurements = crop_data(
    #     signals=measurements,
    #     time_start=0.0,
    #     time_end=0.01,
    # )

    # Pad the measurements with zeros after the end of the signal
    # measurements = measurements.append(
    #     pd.DataFrame(
    #         np.zeros((measurements.shape[0], measurements.shape[1])),
    #         columns=measurements.columns,
    #     ),
    #     ignore_index=True,
    # )

    # measure_phase _velocity(measurements=measurements)

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

    # ideal_signals, _ = generate_ideal_signal(
    #     setup=SETUP,
    #     signal_model="gaussian",
    #     group_velocity_mps=GROUP_VELOCITY_MPS,
    #     phase_velocity_mps=PHASE_VELOCITY_MPS,
    #     signal_length_s=0.2,
    #     center_frequency_Hz=CENTER_FREQUENCY_HZ,
    #     t_var=20e-7,
    #     snr_dB=50,
    #     attenuation_dBpm=0,
    # )

    # ideal_signals = crop_to_signal(ideal_signals)

    # Plot each sensor in the ideal signal on a separate row
    fig, ax = plt.subplots(
        nrows=SETUP.number_of_sensors,
        ncols=1,
        sharex=True,
        sharey=True,
    )
    for i, sensor in enumerate(SETUP.sensors):
        ax[i].plot(measurements[sensor.name])
        ax[i].set_ylabel(f"Sensor {sensor.name}")
    ax[-1].set_xlabel("Time [s]")

    # Export the ideal signals
    generate_signals_for_matlab(
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
            sorted_estimated_angles_deg=estimated_angles_ULA,
            center_frequency_Hz=CENTER_FREQUENCY_HZ,
            number_of_sensors=NUMBER_OF_SENSORS,
            sensor_spacing_m=SENSOR_SPACING_M,
            actuator_coordinates=ACTUATOR_COORDINATES,
        )
    elif ARRAY_TYPE == "UCA":
        estimate_touch_location_UCA(
            setup=SETUP,
            sorted_estimated_angles_deg=estimated_angles_UCA,
            center_frequency_Hz=CENTER_FREQUENCY_HZ,
            number_of_sensors=NUMBER_OF_SENSORS,
            actuator_coordinates=ACTUATOR_COORDINATES,
            uca_center_coordinates=UCA_CENTER_COORDINATES,
        )

    plt.show()


def import_estimated_angles(
    file_name: str,
    s0: bool = False,
):
    # Put the angles from results_simULAtions_10_mm_Teflon_COMSOL_25kHz_10sensors.csv into a dataframe
    estimated_angles = pd.read_csv(
        f"{file_name}.csv",
    )
    if s0:
        # Switch place between rows at index 1 and 3 in sorted_estimated_angles_deg, not sure why
        estimated_angles.iloc[[1, 3]] = estimated_angles.iloc[[3, 1]].values
        estimated_angles.iloc[[2, 3]] = estimated_angles.iloc[[3, 2]].values

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
