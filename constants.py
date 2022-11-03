"""Global consts"""
SAMPLE_RATE = 150000
CHIRP_CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3', 'chirp']

"""An enum for representing edges 1 to 4"""
class Edge(Enum):
    EDGE_1 = 1
    EDGE_2 = 2
    EDGE_3 = 3
    EDGE_4 = 4

    """Print the name of the edge"""
    def __str__(self):
        """Print the name of an edge."""
        if edges == self.TOP_EDGE:
            print('Top edge')
        elif edges == self.RIGHT_EDGE:
            print('Right edge')
        elif edges == self.BOTTOM_EDGE:
            print('Bottom edge')
        elif edges == self.LEFT_EDGE:
            print('Left edge')