"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
import numpy as np

from main_scripts.generate_ideal_signal import generate_ideal_signal
from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab
from utils.data_processing.preprocessing import crop_to_signal
from utils.data_visualization.visualize_data import set_fontsizes
from utils.plate_setups import Setup5
from utils.global_constants import x, y


def estimate_touch_location(
    sorted_estimated_angles_deg: list,
    center_frequency_Hz: int = 25000,
    t_var: float = 4e-9,
    propagation_speed_mps: float = 992,
    snr_dB: float = 10,
    attenuation_dBpm: float = 15,
    crop_end_s: float = 0.001,
    number_of_sensors: int = 8,
    sensor_spacing_m: float = 0.01,
    actuator_coordinates: np.ndarray = np.array([0.50, 0.35]),
):
    SENSORS_START_COORDINATES = np.array([0.05, 0.05])
    SENSORS_CENTER_COORDINATES = np.array(
        [
            SENSORS_START_COORDINATES[x],
            SENSORS_START_COORDINATES[y]
            + sensor_spacing_m * (number_of_sensors - 1) / 2,
        ]
    )
    y_s_a = actuator_coordinates[y] - SENSORS_CENTER_COORDINATES[y]
    x_s_a = actuator_coordinates[x] - SENSORS_CENTER_COORDINATES[x]

    SETUP = Setup5(
        actuator_coordinates=actuator_coordinates,
        number_of_sensors=number_of_sensors,
        array_spacing_m=sensor_spacing_m,
    )

    ideal_signals, _ = generate_ideal_signal(
        setup=SETUP,
        critical_frequency_Hz=center_frequency_Hz,
        attenuation_dBpm=attenuation_dBpm,
        signal_length_s=1,
        propagation_speed_mps=propagation_speed_mps,
        signal_model="gaussian",
        t_var=t_var,
        snr_dB=snr_dB,
    )
    ideal_signals = crop_to_signal(
        ideal_signals,
        # threshold=0.2,
        padding_percent=0.5,
    )
    generate_signals_for_matlab(
        ideal_signals,
        center_frequency_Hz=center_frequency_Hz,
        t_var=t_var,
        propagation_speed_mps=propagation_speed_mps,
        crop_end_s=crop_end_s,
        number_of_sensors=number_of_sensors,
    )

    # Real angles:
    real_phi_1_deg = calculate_phi_1(
        y_s_a=y_s_a,
        x_s_a=x_s_a,
    )
    real_phi_2_deg = calculate_phi_2(
        y_s_a=y_s_a,
        y_s_c=SENSORS_CENTER_COORDINATES[y],
        x_s_a=x_s_a,
    )
    real_phi_3_deg = calculate_phi_3(
        y_s_a=y_s_a,
        x_s_a=x_s_a,
        x_s_c=SENSORS_CENTER_COORDINATES[x],
    )
    real_phi_4_deg = calculate_phi_4(
        y_s_a=y_s_a,
        y_s_c=SENSORS_CENTER_COORDINATES[y],
        x_s_a=x_s_a,
        x_s_c=SENSORS_CENTER_COORDINATES[x],
    )
    if all([angle < 0 for angle in sorted_estimated_angles_deg]):
        # All angles are negative if y_s_a < y_s_c,
        # and a different order should be used
        phi_1_deg = sorted_estimated_angles_deg[1]
        phi_2_Deg = sorted_estimated_angles_deg[3]
        phi_3 = sorted_estimated_angles_deg[0]
        phi_4 = sorted_estimated_angles_deg[2]
    else:
        phi_1_deg = sorted_estimated_angles_deg[3]
        phi_2_Deg = sorted_estimated_angles_deg[0]
        phi_3 = sorted_estimated_angles_deg[2]
        phi_4 = sorted_estimated_angles_deg[1]
    print_angles_info(
        real_phi_1_deg,
        real_phi_2_deg,
        real_phi_3_deg,
        real_phi_4_deg,
        phi_1_deg,
        phi_2_Deg,
        phi_3,
        phi_4,
    )
    # Plot estimation result
    plt.figure()
    SETUP.draw()
    r_sa = calculate_r_sa(
        x_s_c=SENSORS_CENTER_COORDINATES[x],
        y_s_c=SENSORS_CENTER_COORDINATES[y],
        phi_1=phi_1_deg,
        phi_2=phi_2_Deg,
        phi_3=phi_3,
    )
    r_sa = np.array(r_sa)
    print(f"r_sa: [{r_sa[x]:.3f}, {r_sa[y]:.3f}]")
    estimated_location_error = np.linalg.norm(
        r_sa - (actuator_coordinates - SENSORS_CENTER_COORDINATES)
    )
    print(f"Estimated location error: {estimated_location_error:.3f} m")
    touch_location = SENSORS_CENTER_COORDINATES + r_sa
    plot_estimated_location(touch_location)
    plt.show()
    return 0


def print_angles_info(
    real_phi_1,
    real_phi_2,
    real_phi_3,
    real_phi_4,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
):
    print()
    print(f"Real phi_1: {real_phi_1:.3f}, Estimated phi_1: {phi_1:.3f}")
    print(f"Real phi_2: {real_phi_2:.3f}, Estimated phi_2: {phi_2:.3f}")
    print(f"Real phi_3: {real_phi_3:.3f}, Estimated phi_3: {phi_3:.3f}")
    print(f"Real phi_4: {real_phi_4:.3f}, Estimated phi_4: {phi_4:.3f}")

    # Errors in angles:
    print(
        f"Error in phi_1: {np.abs(real_phi_1 - phi_1):.3f} degrees."
    )
    print(
        f"Error in phi_2: {np.abs(real_phi_2 - phi_2):.3f} degrees."
    )
    print(
        f"Error in phi_3: {np.abs(real_phi_3 - phi_3):.3f} degrees."
    )
    print(
        f"Error in phi_4: {np.abs(real_phi_4 - phi_4):.3f} degrees."
    )


def plot_estimated_location(touch_location):
    plt.scatter(
        touch_location[x],
        touch_location[y],
        marker="x",
        color="red",
        s=25,
        zorder=10,
    )


def calculate_phi_1(
    y_s_a,
    x_s_a,
):
    # Will return +/- 90 degrees if x_s_a == 0
    radians = np.arctan2(
        y_s_a,
        x_s_a,
    )
    return radians_to_degrees(radians)


def calculate_phi_2(
    y_s_a,
    y_s_c,
    x_s_a,
):
    # Will return +/- 90 degrees if x_s_a == 0
    radians = np.arctan2(-y_s_a - 2 * y_s_c, x_s_a)
    return radians_to_degrees(radians)


def calculate_phi_3(
    y_s_a,
    x_s_a,
    x_s_c,
):
    radians = np.arctan2(y_s_a, x_s_a + 2 * x_s_c)
    return radians_to_degrees(radians)


def calculate_phi_4(
    y_s_a,
    y_s_c,
    x_s_a,
    x_s_c,
):
    radians = np.arctan2(-y_s_a - 2 * y_s_c, x_s_a + 2 * x_s_c)
    return radians_to_degrees(radians)


def radians_to_degrees(radians):
    return radians * 180 / np.pi


def calculate_r_sa(x_s_c, y_s_c, phi_1, phi_2, phi_3):
    # Calculates the vector r_sa
    if abs(phi_1) == 90 or abs(phi_2) == 90:
        r_sa = [
            0,
            np.tan(np.radians(phi_3)) * (2 * x_s_c),
        ]
        return r_sa

    r_sa = [
        -2 * y_s_c / (np.tan(np.radians(phi_2)) + np.tan(np.radians(phi_1))),
        np.tan(np.radians(phi_3))
        * (
            -2 * y_s_c / (np.tan(np.radians(phi_2)) + np.tan(np.radians(phi_1)))
            + 2 * x_s_c
        ),
    ]
    return r_sa


if __name__ == "__main__":
    raise NotImplementedError
