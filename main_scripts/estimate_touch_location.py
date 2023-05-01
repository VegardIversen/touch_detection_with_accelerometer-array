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


def estimate_touch_location():
    set_fontsizes()
    # Call one of the functions found in /main_scripts

    SENSOR_CENTER_COORDINATES = np.array([0.05, 0.085])
    ACTUATOR_COORDINATES = np.array([0.50, 0.35])
    y_s_a = ACTUATOR_COORDINATES[y] - SENSOR_CENTER_COORDINATES[y]
    x_s_a = ACTUATOR_COORDINATES[x] - SENSOR_CENTER_COORDINATES[x]

    SETUP = Setup5(
        actuator_coordinates=ACTUATOR_COORDINATES,
    )

    # Parameters for the Gaussian-modulated pulse
    CENTER_FREQUENCY = 35000
    T_VAR = 1e-9

    ideal_signals, _ = generate_ideal_signal(
        setup=SETUP,
        critical_frequency=CENTER_FREQUENCY,
        attenuation_dBpm=15,
        signal_length_s=5,
        propagation_speed_mps=1000,
        signal_model="gaussian",
        t_var=T_VAR,
        snr_dB=10,
    )
    ideal_signals = crop_to_signal(
        ideal_signals,
        threshold=0.2,
    )
    generate_signals_for_matlab(
        ideal_signals,
        center_frequency=CENTER_FREQUENCY,
        t_var=T_VAR,
    )

    # Real angles:
    real_phi_1 = calculate_phi_1(
        y_s_a=y_s_a,
        x_s_a=x_s_a,
    )
    real_phi_2 = calculate_phi_2(
        y_s_a=y_s_a,
        y_s_c=SENSOR_CENTER_COORDINATES[y],
        x_s_a=x_s_a,
    )
    real_phi_3 = calculate_phi_3(
        y_s_a=y_s_a,
        x_s_a=x_s_a,
        x_s_c=SENSOR_CENTER_COORDINATES[x],
    )
    real_phi_4 = calculate_phi_4(
        y_s_a=y_s_a,
        y_s_c=SENSOR_CENTER_COORDINATES[y],
        x_s_a=x_s_a,
        x_s_c=SENSOR_CENTER_COORDINATES[x],
    )
    # Angles from Root-WSF, sorted from lowest to highest:
    SORTED_ANGLES = np.array(
        [
            -42.539838862910912,
            -31.990446448370097,
            24.412457820142198,
            29.912970224888895,
        ]
    )
    phi_1 = np.abs(SORTED_ANGLES[3])
    phi_2 = np.abs(SORTED_ANGLES[0])
    phi_3 = np.abs(SORTED_ANGLES[2])
    phi_4 = np.abs(SORTED_ANGLES[1])
    print_angles_info(
        real_phi_1,
        real_phi_2,
        real_phi_3,
        real_phi_4,
        phi_1,
        phi_2,
        phi_3,
        phi_4,
    )
    # Plot estimation result
    plt.figure()
    SETUP.draw()
    r_sa = calculate_r_sa(
        x_s_c=SENSOR_CENTER_COORDINATES[x],
        y_s_c=SENSOR_CENTER_COORDINATES[y],
        phi_1=np.abs(phi_1),
        phi_2=np.abs(phi_2),
        phi_3=np.abs(phi_3),
    )
    print(f"r_sa: [{r_sa[x]:.3f}, {r_sa[y]:.3f}]")
    estimated_location_error = np.linalg.norm(
        r_sa - (ACTUATOR_COORDINATES - SENSOR_CENTER_COORDINATES)
    )
    print(f"Estimated location error: {estimated_location_error:.3f} m")
    touch_location = np.array([0.05, 0.085]) + r_sa
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
        f"Error in phi_1: {np.abs(real_phi_1 - phi_1):.3f} degrees, {np.abs(real_phi_1 - phi_1) / real_phi_1 * 100:.3f}%"
    )
    print(
        f"Error in phi_2: {np.abs(real_phi_2 - phi_2):.3f} degrees, {np.abs(real_phi_2 - phi_2) / real_phi_2 * 100:.3f}%"
    )
    print(
        f"Error in phi_3: {np.abs(real_phi_3 - phi_3):.3f} degrees, {np.abs(real_phi_3 - phi_3) / real_phi_3 * 100:.3f}%"
    )
    print(
        f"Error in phi_4: {np.abs(real_phi_4 - phi_4):.3f} degrees, {np.abs(real_phi_4 - phi_4) / real_phi_4 * 100:.3f}%"
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
    radians = np.arctan2(y_s_a + 2 * y_s_c, x_s_a)
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
    radians = np.arctan2(y_s_a + 2 * y_s_c, x_s_a + 2 * x_s_c)
    return radians_to_degrees(radians)


def radians_to_degrees(radians):
    return radians * 180 / np.pi


def calculate_r_sa(x_s_c, y_s_c, phi_1, phi_2, phi_3):
    # Calculates the vector r_sa
    if abs(phi_1) == 90 or abs(phi_2) == 90:
        raise ValueError("Phi1 and Phi2 should not be 90 degrees.")

    r_sa = [
        2 * y_s_c / (np.tan(np.radians(phi_2)) - np.tan(np.radians(phi_1))),
        np.tan(np.radians(phi_3))
        * (
            2 * y_s_c / (np.tan(np.radians(phi_2)) - np.tan(np.radians(phi_1)))
            + 2 * x_s_c
        ),
    ]

    return r_sa


if __name__ == "__main__":
    main()
