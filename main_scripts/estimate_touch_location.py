"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.data_visualization.drawing import plot_legend_without_duplicates
from utils.global_constants import FIGURES_SAVE_PATH, x, y
from utils.plate_setups import Setup


def estimate_touch_location_ULA(
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
    y_sa = actuator_coordinates[y] - SENSORS_CENTER_COORDINATES[y]
    x_sa = actuator_coordinates[x] - SENSORS_CENTER_COORDINATES[x]

    # Real angles:
    real_phi_1_deg = calculate_phi_1(
        y_sa=y_sa,
        x_sa=x_sa,
    )
    real_phi_2_deg = calculate_phi_2(
        y_sa=y_sa,
        y_cs=SENSORS_CENTER_COORDINATES[y],
        x_sa=x_sa,
    )
    real_phi_3_deg = calculate_phi_3(
        y_sa=y_sa,
        x_sa=x_sa,
        x_cs=SENSORS_CENTER_COORDINATES[x],
        array_type="ULA",
    )
    real_phi_4_deg = calculate_phi_4(
        y_sa=y_sa,
        y_cs=SENSORS_CENTER_COORDINATES[y],
        x_sa=x_sa,
        x_cs=SENSORS_CENTER_COORDINATES[x],
        array_type="ULA",
    )

    setup.draw()
    for method in sorted_estimated_angles_deg.keys():
        if all([angle < 0 for angle in sorted_estimated_angles_deg[method]]):
            # All angles are negative if y_sa < y_cs,
            # and a different order should be used
            phi_1_deg = sorted_estimated_angles_deg[method][1]
            phi_2_deg = sorted_estimated_angles_deg[method][3]
            phi_3_deg = sorted_estimated_angles_deg[method][0]
            phi_4_deg = sorted_estimated_angles_deg[method][2]
        # Assume that there are three angles if there is a NaN value
        elif (
            sum(np.isnan(sorted_estimated_angles_deg[method])) == 1
            or sorted_estimated_angles_deg[method].size == 3
        ):
            # If there are only three angles, then phi_4_deg = 0
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
            x_cs=SENSORS_CENTER_COORDINATES[x],
            y_cs=SENSORS_CENTER_COORDINATES[y],
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

    plt.tight_layout(pad=0.5, h_pad=0)
    try:
        plt.savefig(
            (
                f"{FIGURES_SAVE_PATH}/"
                f"{center_frequency_Hz // 1000}kHz_"
                f"{number_of_sensors}sensors_"
                f"x{100 * actuator_coordinates[x]:.0f}y{100 * actuator_coordinates[y]:.0f}"
                "_ULA.pdf"
            ),
            bbox_inches="tight",
        )
    except FileNotFoundError:
        print(f"\nCould not save figure at {FIGURES_SAVE_PATH}\n")
    return 0


def estimate_touch_location_UCA(
    setup: Setup,
    sorted_estimated_angles_deg: list = None,
    center_frequency_Hz: int = 25000,
    number_of_sensors: int = 8,
    actuator_coordinates: np.ndarray = np.array([0.50, 0.35]),
    uca_center_coordinates: np.ndarray = np.array([0.05, 0.05]),
):
    y_sa = actuator_coordinates[y] - uca_center_coordinates[y]
    x_sa = actuator_coordinates[x] - uca_center_coordinates[x]

    # Real angles:
    real_phi_1_deg = calculate_phi_1(
        y_sa=y_sa,
        x_sa=x_sa,
    )
    real_phi_2_deg = calculate_phi_2(
        y_sa=y_sa,
        y_cs=uca_center_coordinates[y],
        x_sa=x_sa,
    )
    real_phi_3_deg = calculate_phi_3(
        y_sa=y_sa,
        x_sa=x_sa,
        x_cs=uca_center_coordinates[x],
        array_type="UCA",
    )
    real_phi_4_deg = calculate_phi_4(
        y_sa=y_sa,
        y_cs=uca_center_coordinates[y],
        x_sa=x_sa,
        x_cs=uca_center_coordinates[x],
        array_type="UCA",
    )

    setup.draw()
    for method in sorted_estimated_angles_deg.keys():
        if all([angle < 0 for angle in sorted_estimated_angles_deg[method]]):
            # All angles are between {180, 360} if y_sa < y_cs,
            # and a different order should be used
            # Assuming phi = 0 along the x-axis for now
            phi_1_deg = sorted_estimated_angles_deg[method][3]
            phi_2_deg = sorted_estimated_angles_deg[method][2]
            phi_3_deg = sorted_estimated_angles_deg[method][0]
            phi_4_deg = sorted_estimated_angles_deg[method][1]
        elif (
            sum(np.isnan(sorted_estimated_angles_deg[method])) == 1
            or sorted_estimated_angles_deg[method].size == 3
        ):
            # If there are only three angles, then phi_4_deg = 0
            phi_1_deg = sorted_estimated_angles_deg[method][1]
            phi_2_deg = sorted_estimated_angles_deg[method][0]
            phi_3_deg = sorted_estimated_angles_deg[method][2]
            phi_4_deg = 0
        else:
            phi_1_deg = sorted_estimated_angles_deg[method][2]
            phi_2_deg = sorted_estimated_angles_deg[method][1]
            phi_3_deg = sorted_estimated_angles_deg[method][3]
            phi_4_deg = sorted_estimated_angles_deg[method][0]
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
            x_cs=uca_center_coordinates[x],
            y_cs=uca_center_coordinates[y],
            phi_1=phi_1_deg,
            phi_2=phi_2_deg,
            phi_3=phi_3_deg,
        )
        r_sa = np.array(r_sa)
        print(f"r_sa: [{r_sa[x]:.3f}, {r_sa[y]:.3f}]")
        estimated_location_error = np.linalg.norm(
            r_sa - (actuator_coordinates - uca_center_coordinates)
        )
        print(f"Estimated location error: {estimated_location_error:.3f} m")
        # Plot estimation result
        touch_location = uca_center_coordinates + r_sa
        plot_estimated_location(touch_location)
        # Add legend that shows which color marker corresponds to which method
        plot_legend_without_duplicates()

    plt.tight_layout(pad=0.5, h_pad=0)
    try:
        plt.savefig(
            f"{FIGURES_SAVE_PATH}/{center_frequency_Hz // 1000}kHz"
            f"_{number_of_sensors}sensors_UCA.pdf",
            bbox_inches="tight",
        )
    except FileNotFoundError:
        print(f"\nCould not save figure at {FIGURES_SAVE_PATH}\n")
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
    y_sa,
    x_sa,
):
    # Will return +/- 90 degrees if x_sa == 0
    radians = np.arctan2(
        y_sa,
        x_sa,
    )
    return radians_to_degrees(radians)


def calculate_phi_2(
    y_sa,
    y_cs,
    x_sa,
):
    # Will return +/- 90 degrees if x_sa == 0
    radians = np.arctan2(-y_sa - 2 * y_cs, x_sa)
    return radians_to_degrees(radians)


def calculate_phi_3(
    y_sa,
    x_sa,
    x_cs,
    array_type,
):
    if array_type == "ULA":
        radians = np.arctan2(y_sa, x_sa + 2 * x_cs)
    elif array_type == "UCA":
        radians = np.arctan2(y_sa, -x_sa - 2 * x_cs)
    return radians_to_degrees(radians)


def calculate_phi_4(
    y_sa,
    y_cs,
    x_sa,
    x_cs,
    array_type,
):
    if array_type == "ULA":
        radians = np.arctan2(-y_sa - 2 * y_cs, x_sa + 2 * x_cs)
    elif array_type == "UCA":
        radians = np.arctan2(-y_sa - 2 * y_cs, -x_sa - 2 * x_cs)
    return radians_to_degrees(radians)


def radians_to_degrees(radians):
    return radians * 180 / np.pi


def calculate_r_sa(x_cs, y_cs, phi_1, phi_2, phi_3):
    # Calculates the vector r_sa
    if abs(phi_1) == 90 or abs(phi_2) == 90:
        r_sa = [
            0,
            np.tan(np.radians(phi_3)) * (2 * x_cs),
        ]
        return r_sa

    r_sa = [
        -2 * y_cs / (np.tan(np.radians(phi_2)) + np.tan(np.radians(phi_1))),
        np.tan(np.radians(phi_3))
        * (
            -2 * y_cs / (np.tan(np.radians(phi_2)) + np.tan(np.radians(phi_1)))
            + 2 * x_cs
        ),
    ]
    return r_sa


if __name__ == "__main__":
    raise NotImplementedError
