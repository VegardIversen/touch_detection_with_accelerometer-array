"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

from collections import OrderedDict

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def plot_legend_without_duplicates(placement: str = None):
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    # Make the handles circles
    for i, handle in enumerate(handles):
        if isinstance(handle, patches.Circle):
            handles[i] = plt.Line2D(
                [],
                [],
                markerfacecolor=handle.get_facecolor(),
                markeredgecolor=handle.get_edgecolor(),
                marker="o",
                linestyle="None",
            )
        if isinstance(handle, patches.Rectangle):
            # Use a square with lengths 0.003 as the handle
            handles[i] = plt.Line2D(
                [],
                [],
                markerfacecolor=handle.get_facecolor(),
                markeredgecolor=handle.get_edgecolor(),
                marker="s",
                linestyle="None",
                markersize=3,
            )
    by_label = OrderedDict(zip(labels, handles))
    if placement:
        plt.legend(by_label.values(), by_label.keys(), loc=placement)
    else:
        plt.legend(by_label.values(), by_label.keys())


def plot_legend_without_duplicates_ax(ax):
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    return
