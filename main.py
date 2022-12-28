import matplotlib.pyplot as plt
from generate_results import (setup3_2_results,
                              setup7_results,
                              setup9_results,
                              custom_plots)
from data_visualization.visualize_data import set_fontsizes


def main():
    """Run results_setup*n() functions from generate_results.py
    NOTE:   Individual figures can be plotted by wrapping it with
            plt.close('all') and plt.show()
    """
    set_fontsizes()

    print('Generating results...')
    # results_setup3_2()

    # results_setup7()

    setup9_results()

    # custom_plots()

    plt.show()
    return 0


if __name__ == '__main__':
    main()
