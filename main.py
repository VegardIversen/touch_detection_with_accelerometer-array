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


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts

    x = 0
    y = 1
    SENSOR_CENTER_COORDINATES = np.array([0.05, 0.085])
    ACTUATOR_COORDINATES = np.array([0.50, 0.35])
    y_s_a = ACTUATOR_COORDINATES[y] - SENSOR_CENTER_COORDINATES[y]
    x_s_a = ACTUATOR_COORDINATES[x] - SENSOR_CENTER_COORDINATES[x]

    SETUP = Setup5(actuator_coordinates=ACTUATOR_COORDINATES)
    ideal_signals, _ = generate_ideal_signal(
        setup=SETUP,
        critical_frequency=25000,
        attenuation_dBpm=15,
        signal_length_s=5,
        propagation_speed_mps=1000,
        signal_model="gaussian",
        snr_dB=10,
    )
    ideal_signals = crop_to_signal(ideal_signals, threshold=0.2)
    generate_signals_for_matlab(ideal_signals)

    # Real angles:
    phi_1 = calculate_phi_1(
        y_s_a=y_s_a,
        x_s_a=x_s_a,
    )
    phi_2 = calculate_phi_2(
        y_s_a=y_s_a,
        y_s_c=SENSOR_CENTER_COORDINATES[y],
        x_s_a=x_s_a,
    )
    phi_3 = calculate_phi_3(
        y_s_a=y_s_a,
        x_s_a=x_s_a,
        x_s_c=SENSOR_CENTER_COORDINATES[x],
    )
    phi_4 = calculate_phi_4(
        y_s_a=y_s_a,
        y_s_c=SENSOR_CENTER_COORDINATES[y],
        x_s_a=x_s_a,
        x_s_c=SENSOR_CENTER_COORDINATES[x],
    )
    print(f"phi_1: {phi_1}")
    print(f"phi_2: {phi_2}")
    print(f"phi_3: {phi_3}")
    print(f"phi_4: {phi_4}")

    # Plot estimation result
    plt.figure()
    SETUP.draw()
    r_sa = calculate_r_sa(
        x_s_c=SENSOR_CENTER_COORDINATES[x],
        y_s_c=SENSOR_CENTER_COORDINATES[y],
        phi_1=phi_1,
        phi_2=phi_2,
        phi_3=phi_3,
    )
    r_sa = np.array([0.004102398341860, 0.098374496796131])
    touch_location = np.array([0.05, 0.085]) + r_sa
    plt.scatter(
        touch_location[0],
        touch_location[1],
        marker="x",
        color="red",
        s=25,
        zorder=10,
    )
    plt.show()
    return 0


def calculate_phi_1(y_s_a, x_s_a):
    radians = np.arctan2(y_s_a, x_s_a)
    return radians_to_degrees(radians)


def calculate_phi_2(y_s_a, y_s_c, x_s_a):
    radians = np.arctan2(y_s_a + 2 * y_s_c, x_s_a)
    return radians_to_degrees(radians)


def calculate_phi_3(y_s_a, x_s_a, x_s_c):
    radians = np.arctan2(y_s_a, x_s_a + 2 * x_s_c)
    return radians_to_degrees(radians)


def calculate_phi_4(y_s_a, y_s_c, x_s_a, x_s_c):
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
