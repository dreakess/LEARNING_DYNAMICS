import numpy as np

class Config:

    """
    This class is responsible for creating the environment configuration,
    including the grid layout and the initial positions of agents and doorways.
    """
    
    def __init__(self, num_agents, env_num):
        self.num_agents = num_agents
        self.env_num = env_num

        # ENV 1: Cross Shape (21 States) - 5x5 Grid (Padded to 7x7 for safety if you want)
        # 0 = Wall, 1-21 = Unique State IDs
        self.grid_1 = np.array([
            [1,  2,  0,  3,  4],
            [5,  6,  0,  7,  8],
            [9, 10, 11, 12, 13],  # 11 is the Doorway (Center)
            [14,15,  0, 16, 17],
            [18,19,  0, 20, 21]
        ])

        # ENV 2: Four Rooms (36 States) - 7x7 Grid
        self.grid_2 = np.array([
            [0,  1,  2,  0,  3,  4,  0],
            [5,  6,  7,  0,  8,  9, 10],
            [11,12, 13, 14, 15, 16, 17], # Horizontal Hallway
            [0,  0, 18,  0, 19,  0,  0], # Vertical connectors
            [20,21, 22, 23, 24, 25, 26], # Horizontal Hallway
            [27,28, 29,  0, 30, 31, 32],
            [0, 33, 34,  0, 35, 36,  0]
        ])

    def get_grid(self):
        '''
            method to get the grid
        '''

        if self.env_num == 1:
            return self.grid_1
        else:
            return self.grid_2

    def set_initial_position(self):
        """
            Defines Start/Goal coordinates (y, x) for each scenario.
            These match the figures in the PDF.
        """
        positioning = {}

        if self.env_num == 1:
            positioning["player_0"] = [(0, 0), (4, 4)]
            positioning["player_1"] = [(0, 4), (4, 0)]
            positioning["collision"] = [(2, 2)]

        else:

            positioning["player_0"] = [(0, 1), (6, 5)]
            positioning["player_1"] = [(0, 5), (6, 1)]
            positioning["player_2"] = [(6, 5), (0, 1)]
            positioning["player_3"] = [(6, 1), (0, 5)]
            positioning["collision"] = [
                (2, 3), 
                (3, 2), 
                (4, 3), 
                (3, 4)  
            ]

        return positioning

    def get_grid_config_init(self):
        '''
            this is wrapper method to get grid, initial position, doorways position and goal position
        '''
        return self.get_grid(), self.set_initial_position()