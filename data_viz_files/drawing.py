import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict

from data_processing.detect_echoes import find_mirrored_source
from Table import Table


def draw_table(plot_lines=True):
    "Draw the table with the real dimensions, including the lines."
    TABLE_SURFACE_COLOUR = '#fbe5b6'
    TABLE_LINE_COLOUR = '#f0c18b'

    table = patches.Rectangle((0, 0),
                              Table.LENGTH,
                              Table.WIDTH,
                              fc=TABLE_SURFACE_COLOUR,
                              ec=TABLE_LINE_COLOUR,
                              lw=2,
                              zorder=0)

    plt.gca().add_patch(table)

    if plot_lines:
        for i in range(1, 3):
            line_x = plt.Line2D((i * Table.LENGTH / 3, i * Table.LENGTH / 3),
                                (0, Table.WIDTH),
                                color=TABLE_LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            line_y = plt.Line2D((0, Table.LENGTH),
                                (i * Table.WIDTH / 3, i * Table.WIDTH / 3),
                                color=TABLE_LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            plt.gca().add_patch(line_x)
            plt.gca().add_patch(line_y)


def draw_actuator(actuator_coord: np.array):
    ACTUATOR_COLOUR = '#D4434A'
    actuator = plt.Circle(actuator_coord,
                          radius=0.01,
                          fc=ACTUATOR_COLOUR,
                          ec='dimgray',
                          label='Actuator')
    plt.gca().add_patch(actuator)


def draw_sensor(actuator_coord):
    ACTUATOR_COLOUR = '#AEAFA7'
    actuator = plt.Circle(actuator_coord,
                          radius=0.01,
                          fc=ACTUATOR_COLOUR,
                          ec='dimgray',
                          label='Sensor')
    plt.gca().add_patch(actuator)


def draw_waves(actuator_coord: np.array, n_waves=3):
    """Draw arcs representing the waves.
    TODO:   Automate to have the correct wavelength (and amplitude?)
            for a given frequency. Might also be better with circles.
    """
    ARC_COLOUR = '#f0c18b'
    for i in range(1, n_waves + 1):
        arc = patches.Arc(actuator_coord,
                          0.07 * i,     # could be automated with wavelengths
                          0.07 * i,     # could be automated with wavelengths
                          theta1=0,
                          theta2=180,
                          color=ARC_COLOUR,
                          lw=0.75,
                          linestyle='--')
        plt.gca().add_patch(arc)


def flip_and_draw_sources(source_coords: np.array,
                          edges_to_flip_around: np.array):
    """Draw the sources in the flipped positions.
    The table edges are numbered as:

         _______1_______
        |               |
      4 |               | 2
        |               |
        |_______________|
                3

    """

    for edge in edges_to_flip_around:
        for source_coord in source_coords:
            if edge == Table.TOP_EDGE:
                new_coord = source_coord + np.array([0, 2 * (Table.WIDTH - source_coord[1])])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)
            elif edge == Table.RIGHT_EDGE:
                new_coord = source_coord + np.array([2 * (Table.LENGTH - source_coord[0]), 0])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)
            elif edge == Table.BOTTOM_EDGE:
                new_coord = source_coord + np.array([0, -2 * source_coord[1]])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)
            elif edge == Table.LEFT_EDGE:
                new_coord = source_coord + np.array([-2 * source_coord[0], 0])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)


def flip_and_draw_sensors(sensor_coords: np.array,
                          edges_to_flip_around: np.array):
    """Draw the sensors in the flipped positions.
    The table edges are numbered as:

         _______1_______
        |               |
      4 |               | 2
        |               |
        |_______________|
                3

    """
    for edge in edges_to_flip_around:
        for sensor_coord in sensor_coords:
            if edge == Table.TOP_EDGE:
                new_coord = sensor_coord + np.array([0, 2 * (Table.WIDTH - sensor_coord[1])])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)
            elif edge == Table.RIGHT_EDGE:
                new_coord = sensor_coord + np.array([2 * (Table.LENGTH - sensor_coord[0]), 0])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)
            elif edge == Table.BOTTOM_EDGE:
                new_coord = sensor_coord + np.array([0, -2 * sensor_coord[1]])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)
            elif edge == Table.LEFT_EDGE:
                new_coord = sensor_coord + np.array([-2 * sensor_coord[0], 0])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)


def plot_legend_without_duplicates():
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def draw_setup_1():
    plt.axes()
    draw_table()

    ACTUATOR_COORD = np.array([Table.LENGTH / 2, 1 * Table.WIDTH / 6])
    draw_actuator(ACTUATOR_COORD)

    SENSOR_2_COORD = np.array([Table.LENGTH / 2, Table.WIDTH - 0.05])
    SENSOR_2_OFFSET = np.array([-0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    SENSOR_2_COORD = SENSOR_2_COORD + SENSOR_2_OFFSET
    SENSOR_3_OFFSET = np.array([0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    SENSOR_3_COORD = SENSOR_2_COORD + SENSOR_3_OFFSET
    draw_sensor(SENSOR_2_COORD)
    draw_sensor(SENSOR_2_COORD)
    draw_sensor(SENSOR_3_COORD)

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()


def draw_a_setup():
    ACTUATOR_COORD = np.array([1 / 6 * Table.LENGTH,
                               1 / 2 * Table.WIDTH])
    SENSOR_1_COORD = np.array([2 / 3 * Table.LENGTH,
                               5 / 6 * Table.WIDTH])
    SENSOR_2_COORD = np.array([2 / 3 * Table.LENGTH,
                               1 / 3 * Table.WIDTH])

    source_coords = np.array([ACTUATOR_COORD])
    sensor_coords = np.array([SENSOR_1_COORD])

    plt.axes()

    draw_table()
    draw_actuator(ACTUATOR_COORD)
    # Draw each sensor_coord in sensor_coords
    [draw_sensor(sensor_coord) for sensor_coord in sensor_coords]

    EDGES_TO_FLIP_AROUND = np.array([Table.LEFT_EDGE,
                                     Table.BOTTOM_EDGE])

    mirrored_source_coord = find_mirrored_source(ACTUATOR_COORD, EDGES_TO_FLIP_AROUND)
    draw_actuator(mirrored_source_coord)

    # flip_and_draw_sources(source_coords, EDGES_TO_FLIP_AROUND)
    # flip_and_draw_sensors(sensor_coords, EDGES_TO_FLIP_AROUND)
    # [flip_and_draw_sources(source_coords, np.array([edge])) for edge in EDGES_TO_FLIP_AROUND]

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()
