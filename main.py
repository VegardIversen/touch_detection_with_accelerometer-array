"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


from matplotlib import pyplot as plt
from main_scripts.find_propagation_velocities import find_propagation_velocities
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts
    find_propagation_velocities()
    plt.show()


if __name__ == "__main__":
    main()
