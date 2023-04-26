"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
import numpy as np
from main_scripts.find_propagation_velocities import find_propagation_velocities
from utils.data_visualization.visualize_data import set_fontsizes
from utils.plate_setups import Setup5


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts
    find_propagation_velocities()
    plt.show()
    return 0


if __name__ == "__main__":
    main()
