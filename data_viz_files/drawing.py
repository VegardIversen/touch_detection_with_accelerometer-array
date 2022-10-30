import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict


def draw_table(plot_lines=True):
    "Draw the table with the real dimensions, including the lines."
    TABLE_LENGTH = 0.716    # m
    TABLE_WIDTH = 0.597     # m
    TABLE_SURFACE_COLOUR = '#fbe5b6'
    TABLE_LINE_COLOUR = '#f0c18b'

    table = patches.Rectangle((0, 0),
                              TABLE_LENGTH,
                              TABLE_WIDTH,
                              fc=TABLE_SURFACE_COLOUR,
                              ec=TABLE_LINE_COLOUR,
                              lw=2,
                              zorder=0)

    plt.gca().add_patch(table)

    if plot_lines:
        for i in range(1, 3):
            line_x = plt.Line2D((i * TABLE_LENGTH / 3, i * TABLE_LENGTH / 3),
                                (0, TABLE_WIDTH),
                                color=TABLE_LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            line_y = plt.Line2D((0, TABLE_LENGTH),
                                (i * TABLE_WIDTH / 3, i * TABLE_WIDTH / 3),
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
    TABLE_LENGTH = 0.716    # m
    TABLE_WIDTH = 0.597     # m

    for edge in edges_to_flip_around:
        for source_coord in source_coords:
            if edge == 1:
                # Top edge of the table
                new_coord = source_coord + np.array([0, 2 * (TABLE_WIDTH - source_coord[1])])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)
            elif edge == 2:
                # Right edge of the table
                new_coord = source_coord + np.array([2 * (TABLE_LENGTH - source_coord[0]), 0])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)
            elif edge == 3:
                # Bottom edge of the table
                new_coord = source_coord + np.array([0, -2 * source_coord[1]])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)
            elif edge == 4:
                # Left edge of the table
                new_coord = source_coord + np.array([-2 * source_coord[0], 0])
                source_coords = np.vstack((source_coords, new_coord))
                draw_actuator(new_coord)


def flip_and_draw_sensors(sensor_coords,
                          edges_to_flip_around):
    """Draw the sensors in the flipped positions.
    The table edges are numbered as:

         _______1_______
        |               |
      4 |               | 2
        |               |
        |_______________|
                3

    """
    TABLE_LENGTH = 0.716    # m
    TABLE_WIDTH = 0.597     # m

    for edge in edges_to_flip_around:
        for sensor_coord in sensor_coords:
            if edge == 1:
                # Top edge of the table
                new_coord = sensor_coord + np.array([0, 2 * (TABLE_WIDTH - sensor_coord[1])])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)
            elif edge == 2:
                # Right edge of the table
                new_coord = sensor_coord + np.array([2 * (TABLE_LENGTH - sensor_coord[0]), 0])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)
            elif edge == 3:
                # Bottom edge of the table
                new_coord = sensor_coord + np.array([0, -2 * sensor_coord[1]])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)
            elif edge == 4:
                # Left edge of the table
                new_coord = sensor_coord + np.array([-2 * sensor_coord[0], 0])
                sensor_coords = np.vstack((sensor_coords, new_coord))
                draw_sensor(new_coord)


def plot_legend_without_duplicates():
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def draw_setup_1():
    TABLE_LENGTH = 0.716    # m
    TABLE_WIDTH = 0.597     # m

    plt.axes()
    draw_table()

    actuator_coord = np.array([TABLE_LENGTH / 2, 1 * TABLE_WIDTH / 6])
    draw_actuator(actuator_coord)

    sensor_2_coord = np.array([TABLE_LENGTH / 2, TABLE_WIDTH - 0.05])
    sensor_1_coord = sensor_2_coord + np.array([-0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    sensor_3_coord = sensor_2_coord + np.array([0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    draw_sensor(sensor_2_coord)
    draw_sensor(sensor_1_coord)
    draw_sensor(sensor_3_coord)

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()


def draw_a_setup():
    TABLE_LENGTH = 0.716    # m
    TABLE_WIDTH = 0.597     # m
    ACTUATOR_COORD = np.array([TABLE_LENGTH / 6,
                               TABLE_WIDTH / 2])
    SENSOR_1_COORD = np.array([2 * TABLE_LENGTH / 3,
                               5 * TABLE_WIDTH / 6])
    SENSOR_2_COORD = np.array([2 * TABLE_LENGTH / 3,
                               1 * TABLE_WIDTH / 3])

    source_coords = np.array([ACTUATOR_COORD])
    sensor_coords = np.array([SENSOR_1_COORD])

    plt.axes()

    draw_table()
    draw_actuator(ACTUATOR_COORD)
    # Draw each sensor_coord in sensor_coords
    [draw_sensor(sensor_coord) for sensor_coord in sensor_coords]

    EDGES_TO_FLIP_AROUND = np.array([1, 3])

    flip_and_draw_sensors(sensor_coords, EDGES_TO_FLIP_AROUND)
    flip_and_draw_sources(source_coords, EDGES_TO_FLIP_AROUND)
    # [flip_and_draw_sources(source_coords, np.array([edge])) for edge in EDGES_TO_FLIP_AROUND]

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()
