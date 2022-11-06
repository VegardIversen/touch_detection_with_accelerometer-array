"""Objects such as the table, acutators and sensors have their own class.
TODO:   - Give actuators and sensors a name, e.g. 'channel 1' or 'sensor 1'
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Table:
    """Represents the table and its edges."""
    """Table dimensions"""
    LENGTH = 0.80
    WIDTH = 0.60     # m
    """Enum for representing edges"""
    TOP_EDGE = 1
    RIGHT_EDGE = 2
    BOTTOM_EDGE = 3
    LEFT_EDGE = 4
    """Colour settings for drawing"""
    SURFACE_COLOUR = '#fbe5b6'
    LINE_COLOUR = '#f0c18b'

    def print_edges(self, edges):
        """Print the name of an edge."""
        if edges == self.TOP_EDGE:
            print('Top edge')
        elif edges == self.RIGHT_EDGE:
            print('Right edge')
        elif edges == self.BOTTOM_EDGE:
            print('Bottom edge')
        elif edges == self.LEFT_EDGE:
            print('Left edge')

    def draw(self):
        """Draw the table."""
        "Draw the table with the real dimensions, including the lines."
        table = patches.Rectangle((0, 0),
                                  self.LENGTH,
                                  self.WIDTH,
                                  fc=self.SURFACE_COLOUR,
                                  ec=self.LINE_COLOUR,
                                  lw=2,
                                  zorder=0)

        plt.gca().add_patch(table)

        for i in range(1, 3):
            line_x = plt.Line2D((i / 3 * self.LENGTH, i / 3 * self.LENGTH),
                                (0, self.WIDTH),
                                color=self.LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            line_y = plt.Line2D((0, self.LENGTH),
                                (i / 3 * self.WIDTH, i / 3 * self.WIDTH),
                                color=self.LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            plt.gca().add_patch(line_x)
            plt.gca().add_patch(line_y)


class Sensor:
    """Represents a sensor.
    NOTE:   Not sure just how to represent coordinates yet,
            or if get_/set_coordinates() are necessary.
    """
    def __init__(self, coordinates: np.ndarray, radius: float = 0.007):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.radius = radius

    def set_coordinates(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def draw(self):
        """Draw the sensor."""
        sensor = plt.Circle(self.coordinates,
                            radius=self.radius,
                            fc=self.FILL_COLOUR,
                            ec=self.EDGE_COLOUR,
                            label='Sensor',
                            zorder=10)
        plt.gca().add_patch(sensor)

    """Colour settings for drawing"""
    FILL_COLOUR = '#AEAFA7'
    EDGE_COLOUR = 'dimgray'


class Actuator:
    """Represents an actuator.
    NOTE:   Not sure just how to represent coordinates yet,
            or if get_/set_coordinates() are necessary.
    """
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def copy(self):
        """Define a function to copy the actuator."""
        return Actuator(self.coordinates)

    def set_coordinates(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def draw(self):
        """Draw the actuator."""
        actuator = plt.Circle(self.coordinates,
                              radius=self.RADIUS,
                              fc=self.FILL_COLOUR,
                              ec=self.EDGE_COLOUR,
                              label='Actuator',
                              zorder=10)
        plt.gca().add_patch(actuator)

    RADIUS = 0.01  # m
    """Colour settings for drawing"""
    FILL_COLOUR = '#D4434A'
    EDGE_COLOUR = 'dimgray'


class MirroredSource(Actuator):
    """Represents a mirrored source."""
    def __init__(self, coordinates: np.ndarray):
        super().__init__(coordinates)

    def draw(self):
        """Draw the mirrored source."""
        mirrored_source = plt.Circle(self.coordinates,
                                     radius=self.RADIUS,
                                     fc=self.FILL_COLOUR,
                                     ec=self.EDGE_COLOUR,
                                     label='Mirrored source',
                                     zorder=10)
        plt.gca().add_patch(mirrored_source)

    # Colour settings for drawing
    FILL_COLOUR = 'pink'
    EDGE_COLOUR = 'dimgray'


class MirroredSensor(Sensor):
    """Represents a mirrored sensor."""
    def __init__(self, coordinates: np.ndarray):
        super().__init__(coordinates)

    def draw(self):
        """Draw the mirrored sensor."""
        mirrored_sensor = plt.Circle(self.coordinates,
                                     radius=self.radius,
                                     fc=self.FILL_COLOUR,
                                     ec=self.EDGE_COLOUR,
                                     label='Mirrored sensor',
                                     zorder=10)
        plt.gca().add_patch(mirrored_sensor)

    # Colour settings for drawing
    FILL_COLOUR = 'white'
    EDGE_COLOUR = 'dimgray'
