"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict

from utils.data_processing.detect_echoes import (find_mirrored_source,
                                           flip_sensors,
                                           flip_sources)
from utils.objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor


def plot_legend_without_duplicates(placement: str = None):
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    if placement:
        plt.legend(by_label.values(), by_label.keys(), loc=placement)
    else:
        plt.legend(by_label.values(), by_label.keys())


def plot_legend_without_duplicates_ax(ax):
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')


def draw_setup_example():
    table = Table()

    sensor_2 = Sensor(coordinates=np.array([table.LENGTH / 2,
                                            table.WIDTH - 0.05]))
    sensor_1 = Sensor(coordinates=(sensor_2.x - 0.02, sensor_2.y))
    sensor_3 = Sensor(coordinates=(sensor_2.x + 0.02, sensor_2.y))
    sensor_4 = Sensor(coordinates=(sensor_2.x - 0.02, sensor_2.y - 0.02))
    sensor_5 = Sensor(coordinates=(sensor_2.x, sensor_2.y - 0.02))
    sensor_6 = Sensor(coordinates=(sensor_2.x + 0.02, sensor_2.y - 0.02))
    sensor_7 = Sensor(coordinates=(sensor_2.x - 0.02, sensor_2.y - 0.04))
    sensor_8 = Sensor(coordinates=(sensor_2.x, sensor_2.y - 0.04))
    sensor_9 = Sensor(coordinates=(sensor_2.x + 0.02, sensor_2.y - 0.04))

    actuator = Actuator(coordinates=np.array([sensor_5.x - 0.10,
                                              sensor_5.y]))

    plt.axes()
    table.draw()
    actuator.draw()
    sensor_2.draw()
    sensor_1.draw()
    sensor_3.draw()
    sensor_4.draw()
    sensor_5.draw()
    sensor_6.draw()
    sensor_7.draw()
    sensor_8.draw()
    sensor_9.draw()

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()

    return actuator, sensor_1, sensor_2, sensor_3

