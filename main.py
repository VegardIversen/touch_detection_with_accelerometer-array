import matplotlib.pyplot as plt
from generate_results import (results_setup3_2,
                              results_setup7,
                              results_setup9,
                              custom_plots)
from data_visualization.visualize_data import set_fontsizes


def main():
    """Run results_setup*n() functions from generate_results.py
    NOTE:   Individual figures can be plotted by wrapping it with
            plt.close('all') and plt.show()
    """
    set_fontsizes()

    # results_setup3_2()

    results_setup9()

    # custom_plots()

    plt.show()
    return 0


if __name__ == '__main__':
    main()
