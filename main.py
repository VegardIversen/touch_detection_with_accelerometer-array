"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
from main_scripts.linear_array_by_edge import linear_array_by_edge
# from main_scripts.project_thesis import (setup1_results,
#                                          setup2_results,
#                                          setup3_results,)
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()

    linear_array_by_edge()
    plt.show()

    return 0


if __name__ == '__main__':
    main()
