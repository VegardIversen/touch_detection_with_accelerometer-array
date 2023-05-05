"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import numpy as np
from matplotlib import pyplot as plt

from main_scripts.plot_speed_information import plot_speed_information
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()

    # * Call one of the functions found in /main_scripts
    plot_speed_information()

    # Plot all figures at the same time
    plt.show()


if __name__ == "__main__":
    main()
