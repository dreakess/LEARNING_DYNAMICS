import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from collections import defaultdict
from Config import Config
from Player import Player

class Environment(ParallelEnv):

    '''
    This class is responsible to orchestrate Player class (Physics about the position),
    and for the Training class (reward for each agent, calculation of the goal...)
    '''

    metadata = {"name": "Environment"}

    def __init__(self, num_agents, env_num):


        # Primary settings
        self.env_num = env_num
        self.steps = 0

        # Load Config
        self.conf = Config(num_agents, env_num)
        self.grid, self.initial_position = self.conf.get_grid_config_init()
        self.doorway_coords = self.initial_position.get("collision", []) 


        # Setup Agent Names
        if num_agents == 4:
            self.agents = ["player_0", "player_1", "player_2", "player_3"]
        else:
            self.agents = ["player_0", "player_1"]

        # Initialize the variable that we are going to truck
        self.terminations = {a: False for a in self.agents} 
        self.infos = {a: {} for a in self.agents}
        self.observations = {a : {} for a in self.agents} 
        self.collision = 0


        # Creating player obj and setting the initial position
        
        self.agents_obj = {}

        for name in self.agents:
            if name in self.initial_position:
                data = self.initial_position[name]
                start_pos = data[0]
                goal_pos  = data[1]
                # Create Player
                self.agents_obj[name] = Player(name, start_pos, goal_pos)



    def reset(self, seed=None, options=None):

        '''
        Resets all agents before the new episode starting.
        '''

        # Initialization of the variable
        self.steps = 0
        self.collision = 0
        self.finish_times = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        observations = {}
        infos = {a: {} for a in self.agents}

        # Rinitialization of the position of the agents inside the map
        current_positions = {}
        for name, player in self.agents_obj.items():
            player.reset() 
            player.set_position(player.start_pos[0], player.start_pos[1])
            current_positions[name] = player.get_position()

        # Initialization for the active perception
        for name, player in self.agents_obj.items():
            my_pos = current_positions[name]
            visible_neighbors = []
            for other_name, other_pos in current_positions.items():
                if name == other_name: continue
                # Calculating the manhattan distance
                dist = abs(my_pos[0] - other_pos[0]) + abs(my_pos[1] - other_pos[1])
                if dist <= 2:
                    visible_neighbors.append(other_pos)
            
            # Initialization of the observation dictionary
            observations[name] = {
                "id": player.get_state(self.grid),
                "coords": my_pos,
                "visible_neighbors": tuple(sorted(visible_neighbors)) 
            }

            infos[name] = {
                "step": 0 
            }

        return observations, self.terminations, infos
    

    
    
    def step(self, actions):
        
        '''
        Handles physics, rewards, collisions, and termination.
        '''

        # Initialization of the variable for this step
        self.steps += 1
        self.collision = 0
        rewards = {a: 0.0 for a in self.agents}
        infos = {a: {} for a in self.agents}
        observations = {}

        # Handling physics 
        for name, action in actions.items():
            player = self.agents_obj[name]
            if player.check_goal:
                pass
            else:
                if action == 4: rewards[name] -= 0.8
                player.step(action, self.grid)

        # Update the position
        current_positions = {}
        for name in self.agents:
            current_positions[name] = self.agents_obj[name].get_position()


        # check if there are someone near the agents (Active perception)
        for name in self.agents:
            my_pos = current_positions[name]
            neighbors_in_view = []
            
            for other_name, other_pos in current_positions.items():
                if name == other_name: continue 
                
                dist = abs(my_pos[0] - other_pos[0]) + abs(my_pos[1] - other_pos[1])
                if dist <= 2:
                    # Append of the agents because is within the FOV
                    neighbors_in_view.append(other_pos)
            
            neighbors_in_view.sort()
            

        # Update of the observation for every agents
        current_positions = {}
        for name in self.agents:
            player = self.agents_obj[name]
            pos = player.get_position()
            current_positions[name] = pos
            observations[name] = {
                "id": player.get_state(self.grid),
                "coords": pos,
                "visible_neighbors": tuple(neighbors_in_view) 
            }


        # this section of the code is trying to check if there is misscordination at this step
        agents_at_pos = defaultdict(list)
        for name in self.agents:
            # We use current_positions which you calculated earlier in step()
            pos = current_positions[name]
            agents_at_pos[pos].append(name)

        # Check Doorways Only
        for door_pos in self.doorway_coords:
            agents_here = agents_at_pos[door_pos]
            
            # If 2 or more robots are in this door
            if len(agents_here) > 1:
                self.collision += 1  # Count 1 Accident
                
                # Punish EVERYONE involved
                for name in agents_here:
                    rewards[name] -= 35

        # --- TERMINATION / GOAL LOGIC ---
        for name in self.agents:
            player = self.agents_obj[name]
            
            # Skip if already done
            if player.check_goal:
                self.terminations[name] = True 
                continue

            # Check Goal
            if player.get_position() == player.goal_pos:
                #print(f"{name} reached goal!")
                player.check_goal = True
                player.update_step(self.steps)
                
                self.terminations[name] = True
                rewards[name] += 10 # One-time Reward
            
            else:
                self.terminations[name] = False

        # Update the infos about the step for every agents
        for name in self.agents:
            player = self.agents_obj[name]
            infos[name] = {
                "step": player.get_step() 
            }

        return observations, rewards, self.terminations, self.collision, infos


    # API Requirement
    def action_space(self, agent): return self.action_spaces[agent]
    def observation_space(self, agent): return self.observation_spaces[agent]
