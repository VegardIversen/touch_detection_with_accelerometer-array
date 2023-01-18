"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np


def simulated_phase_velocities(frequencies: np.ndarray):
    """Return the phase velocity of a chipboard plate."""
    plate_thickness = 0.02  # m
    youngs_modulus = 3.8 * 10 ** 9  # Pa
    density = (650 + 800) / 2  # kg/m^3
    poisson_ratio = 0.2  # -
    phase_velocity_longitudinal = np.sqrt(youngs_modulus /
                                          (density * (1 - poisson_ratio ** 2)))
    phase_velocity_shear = (phase_velocity_longitudinal *
                            np.sqrt((1 - poisson_ratio) / 2))
    phase_velocities_flexural = np.sqrt(1.8 * phase_velocity_longitudinal *
                                        plate_thickness * frequencies)
    # group_velocity_flexural = (2 * np.sqrt(2 * np.pi * frequencies) *
    #                            (youngs_modulus) ** (1 / 4))
    """As the plate is not considered 'thin', we need to correct the
    calculated velocities with correction factor depending on the
    Poisson ratio:
    poisson_ratio = 0.2: correction_factor = 0.689
    poisson_ratio = 0.3: correction_factor = 0.841
    (according to Vigran's 'Building acoustics').
    """
    correction_factor = 0.689
    c_G = phase_velocity_shear  # mysterious factor that the source doesnt explain
    corrected_phase_velocities = (1 /
                                  ((1 / (phase_velocities_flexural ** 3)) +
                                   (1 / ((correction_factor ** 3) *
                                    (c_G ** 3))))) ** (1 / 3)
    return (phase_velocities_flexural,
            corrected_phase_velocities,
            phase_velocity_shear)
