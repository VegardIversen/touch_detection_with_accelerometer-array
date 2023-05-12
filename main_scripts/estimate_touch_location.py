"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.data_visualization.drawing import plot_legend_without_duplicates
from utils.global_constants import FIGURES_SAVE_PATH, x, y
from utils.plate_setups import Setup


def estimate_touch_location(
    setup: Setup,
    sorted_estimated_angles_deg: list = None,
    center_frequency_Hz: int = 25000,
    number_of_sensors: int = 8,
    sensor_spacing_m: float = 0.01,
    actuator_coordinates: np.ndarray = np.array([0.50, 0.35]),
):
    SENSORS_START_COORDINATES = setup.sensors[0].coordinates
    SENSORS_CENTER_COORDINATES = np.array(
        [
            SENSORS_START_COORDINATES[x],
            SENSORS_START_COORDINATES[y]
            + sensor_spacing_m * (number_of_sensors - 1) / 2,
        ]
    )
    y_s_a = actuator_coordinates[y] - SENSORS_CENTER_COORDINATES[y]
    x_s_a = actuator_coordinates[x] - SENSORS_CENTER_COORDINATES[x]

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

    setup.draw()
    for method in sorted_estimated_angles_deg.keys():
        if all([angle < 0 for angle in sorted_estimated_angles_deg[method]]):
            # All angles are negative if y_s_a < y_s_c,
            # and a different order should be used
            phi_1_deg = sorted_estimated_angles_deg[method][1]
            phi_2_deg = sorted_estimated_angles_deg[method][3]
            phi_3_deg = sorted_estimated_angles_deg[method][0]
            phi_4_deg = sorted_estimated_angles_deg[method][2]
        # Else if the number of angles is three
        elif len(sorted_estimated_angles_deg[method]) == 3:
            phi_1_deg = sorted_estimated_angles_deg[method][2]
            phi_2_deg = sorted_estimated_angles_deg[method][0]
            phi_3_deg = sorted_estimated_angles_deg[method][1]
            phi_4_deg = 0
        else:
            phi_1_deg = sorted_estimated_angles_deg[method][3]
            phi_2_deg = sorted_estimated_angles_deg[method][0]
            phi_3_deg = sorted_estimated_angles_deg[method][2]
            phi_4_deg = sorted_estimated_angles_deg[method][1]
        print_angles_info(
            method,
            real_phi_1_deg,
            real_phi_2_deg,
            real_phi_3_deg,
            real_phi_4_deg,
            phi_1_deg,
            phi_2_deg,
            phi_3_deg,
            phi_4_deg,
        )
        r_sa = calculate_r_sa(
            x_s_c=SENSORS_CENTER_COORDINATES[x],
            y_s_c=SENSORS_CENTER_COORDINATES[y],
            phi_1=phi_1_deg,
            phi_2=phi_2_deg,
            phi_3=phi_3_deg,
        )
        r_sa = np.array(r_sa)
        print(f"r_sa: [{r_sa[x]:.3f}, {r_sa[y]:.3f}]")
        estimated_location_error = np.linalg.norm(
            r_sa - (actuator_coordinates - SENSORS_CENTER_COORDINATES)
        )
        print(f"Estimated location error: {estimated_location_error:.3f} m")
        # Plot estimation result
        touch_location = SENSORS_CENTER_COORDINATES + r_sa
        plot_estimated_location(touch_location)
        # Add legend that shows which color marker corresponds to which method
        plot_legend_without_duplicates()
    # Save plot as pdf
    plt.savefig(
        f"{FIGURES_SAVE_PATH}/{center_frequency_Hz // 1000}kHz_{number_of_sensors}sensors.pdf",
        bbox_inches="tight",
    )
    return 0


def print_angles_info(
    method,
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
    print(f"Method: {method}")
    print(f"Real phi_1: {real_phi_1:.3f}, Estimated phi_1: {phi_1:.3f}")
    print(f"Real phi_2: {real_phi_2:.3f}, Estimated phi_2: {phi_2:.3f}")
    print(f"Real phi_3: {real_phi_3:.3f}, Estimated phi_3: {phi_3:.3f}")
    print(f"Real phi_4: {real_phi_4:.3f}, Estimated phi_4: {phi_4:.3f}")

    # Errors in angles:
    print(f"Error in phi_1: {np.abs(real_phi_1 - phi_1):.3f} degrees.")
    print(f"Error in phi_2: {np.abs(real_phi_2 - phi_2):.3f} degrees.")
    print(f"Error in phi_3: {np.abs(real_phi_3 - phi_3):.3f} degrees.")
    print(f"Error in phi_4: {np.abs(real_phi_4 - phi_4):.3f} degrees.")


def plot_estimated_location(touch_location):
    plt.scatter(
        touch_location[x],
        touch_location[y],
        marker="x",
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