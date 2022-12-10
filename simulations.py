import numpy as np


def phase_velocities_chipboard(frequencies: np.ndarray):
    """Return the phase velocity of a chipboard plate."""
    plate_thickness = 0.02  # m
    youngs_modulus = 3.8 * 10 ** 9  # Pa
    density = (650 + 800) / 2  # kg/m^3
    poisson_ratio = 0.2  # -
    phase_velocity_longitudinal = np.sqrt(youngs_modulus /
                                          (density * (1 - poisson_ratio ** 2)))
    phase_velocities = np.sqrt(np.pi / np.sqrt(3) * phase_velocity_longitudinal *
                               plate_thickness * frequencies)
    return phase_velocities
