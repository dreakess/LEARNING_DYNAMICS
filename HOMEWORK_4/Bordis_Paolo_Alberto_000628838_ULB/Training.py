import numpy as np
import time
import os

from Environment import Environment
from IndependentQLearner import IndependentQLearner
from SparseQLearner import SparseQLearner
from IPython.display import clear_output
from PIL import Image, ImageDraw 

class Training:
    def __init__(self, num_agents, env_num, alpha):
        self.env = Environment(num_agents, env_num)
        self.env_num = env_num
        self.num_agents = num_agents
        self.alpha = alpha
        self.NUM_EPISODES = 10000
        self.MAX_STEPS = 100

        if self.env_num == 1:
          self.num_states = 21 
        else:
          self.num_states = 36 



    def evaluate_indipendent(self, agents, num_eval_episodes=32):
        """
        Greedy evaluations for indipendent learners.
        Return: (Avg Reward, Avg Collisions, Avg Steps)
        """

        all_eval_paths = []
        saved_eps = {}
        for name, agent in agents.items():
            saved_eps[name] = agent.epsilon
            agent.epsilon = 0.0 

        total_eval_reward = 0
        total_eval_collisions = 0
        total_eval_steps = 0

        for _ in range(num_eval_episodes):
            # Reset Environment
            obs, terms, infos = self.env.reset()

            episode_paths = {name: [obs[name]['coords']] for name in agents}
            
            terms = {name: False for name in agents}
            episode_collisions = 0

            for step in range(self.MAX_STEPS):
                actions = {}
                for name in agents:
                    if not terms[name]:
                        state_id = obs[name]["id"]
                        actions[name] = agents[name].select_action(state_id)

                # Step Environment
                next_obs, rewards, terms, collision, infos = self.env.step(actions)

                episode_collisions += collision
                
                # Accumulated Rewards (Discounted)
                for name in agents:
                    discounted_reward = rewards[name] * (0.98 ** step)
                    total_eval_reward += discounted_reward
                    if name in next_obs:
                        pos = next_obs[name]['coords']
                        episode_paths[name].append(pos)
                    elif terms[name]: 
                        episode_paths[name].append(episode_paths[name][-1])
                
                obs = next_obs
                
                # Global Stop
                if all(terms.values()):
                    break

            # END EPISODE
            all_eval_paths.append(episode_paths)
            total_eval_collisions += episode_collisions
            
            finish_times = [infos[n].get('step', 0) for n in agents]
            adj_times = [t if t > 0 else self.MAX_STEPS for t in finish_times]
            total_eval_steps += max(adj_times)

        # 2. Restore Epsilon
        for name, agent in agents.items():
            agent.epsilon = saved_eps[name]

        # 3. Calculate Averages
        all_eval_paths.append(episode_paths)
        avg_reward = total_eval_reward / (num_eval_episodes * self.num_agents)
        avg_col = total_eval_collisions /num_eval_episodes
        avg_steps = total_eval_steps / num_eval_episodes
        
        return avg_reward, avg_col, avg_steps, all_eval_paths
    

    def evaluate_sparse(self, agents, num_eval_episodes=32):
        """
        Greedy evaluations for sparse learners.
        Return: (Avg Reward, Avg Collisions, Avg Steps)
        """

        saved_eps = {}
        for name, agent in agents.items():
            saved_eps[name] = agent.epsilon
            agent.epsilon = 0.0 


        all_eval_paths = []
        total_eval_reward = 0
        total_eval_collisions = 0
        total_eval_steps = 0

        for _ in range(num_eval_episodes):
            obs, terms, infos = self.env.reset()
            episode_collisions = 0
            episode_paths = {name: [obs[name]['coords']] for name in agents}
            
            if not terms: terms = {name: False for name in agents}

            for step in range(self.MAX_STEPS):
                actions = {}
                for name in agents:
                    if not terms[name]:
                        state_id = obs[name]["id"]
                        my_coords = obs[name]["coords"]
                        neighbors = obs[name]["visible_neighbors"]
                        
                        decision = agents[name].select_action(state_id, my_coords, neighbors)
                        actions[name] = decision[1]

                # Step
                next_obs, rewards, terms, collision, infos = self.env.step(actions)
                
                episode_collisions += collision

                # Accumulated Rewards (Discounted)
                for name in agents:
                    discounted_reward = rewards[name] * (0.98 ** step)
                    total_eval_reward += discounted_reward
                    if name in next_obs:
                        pos = next_obs[name]['coords']
                        episode_paths[name].append(pos)
                    elif terms[name]: 
                        episode_paths[name].append(episode_paths[name][-1])
                
                obs = next_obs
                
                if all(terms.values()):
                    break
            
            # END EPISODE
            total_eval_collisions += episode_collisions

            finish_times = [infos[n].get('step', 0) for n in agents]
            adj_times = [t if t > 0 else self.MAX_STEPS for t in finish_times]
            total_eval_steps += max(adj_times)

        # 2. Restore Epsilon
        for name, agent in agents.items():
            agent.epsilon = saved_eps[name]

        # 3. Calculate Averages
        all_eval_paths.append(episode_paths)
        avg_reward = total_eval_reward / (num_eval_episodes * self.num_agents)
        avg_col = total_eval_collisions / num_eval_episodes
        avg_steps = total_eval_steps / num_eval_episodes
        
        return avg_reward, avg_col, avg_steps, all_eval_paths


    def save_replay_as_text(self, filename="replay/replay_log.txt", custom_paths=None, grids_per_row=5):
        """
        Saves the last episode as a sequence of side-by-side ASCII grids in a text file.
        
        Args:
            filename (str): The output path for the text file.
            custom_paths (dict): Optional dictionary of agent paths to replay. 
                               If None, uses the current environment state.
            grids_per_row (int): Number of grid snapshots to display per horizontal row.
        """
        
        # 1. Retrieve Agent Paths
        if custom_paths:
            paths = custom_paths
        else:
            paths = {}
            # Validate environment state
            if not hasattr(self.env, 'agents_obj') or not self.env.agents_obj:
                print("Error: No agents found in the environment!")
                return
            # Extract sequences from current agents
            for name, player in self.env.agents_obj.items():
                paths[name] = player.sequence

        if not paths:
            print("No paths available to save.")
            return

        # Calculate the maximum duration of the episode
        max_steps = 0
        for seq in paths.values():
            if len(seq) > max_steps: max_steps = len(seq)

        rows, cols = self.env.grid.shape
        
        # --- HELPER FUNCTION: GENERATE A SINGLE GRID (LIST OF STRINGS) ---
        def generate_single_grid(t):
            grid_lines = []
            
            # 1. Calculate exact grid width for alignment
            # Each cell is " . " (3 chars)
            # Side borders "|" (2 chars total: 1 left, 1 right)
            grid_width = (cols * 3) + 2 
            
            # 2. Header (Centered)
            header_text = f"Step {t:02d}"
            grid_lines.append(header_text.center(grid_width))
            
            # 3. Top Border
            grid_lines.append("+" + "---" * cols + "+")
            
            # Determine agent positions at time t
            current_positions = {}
            for name in self.env.agents:
                path = paths.get(name, [])
                if not path: continue
                
                # Use position at t, or last known position if finished
                if t < len(path): 
                    current_positions[name] = path[t]
                else: 
                    current_positions[name] = path[-1]

            # 4. Grid Rows
            for r in range(rows):
                row_str = "|"
                for c in range(cols):
                    content = " . " # Default: Floor
                    
                    # Check for Walls
                    if self.env.grid[r, c] == 0: 
                        content = "###"
                    # Check for Doorways
                    elif (r, c) in self.env.doorway_coords: 
                        content = " D "
                    
                    # Check for Goals (Draw G only if empty)
                    is_goal = any(self.env.agents_obj[n].goal_pos == (r,c) for n in self.env.agents)
                    if is_goal and content == " . ": 
                        content = " G "

                    # Check for Agents (Overwrites everything else)
                    agents_here = []
                    for name, pos in current_positions.items():
                        if pos == (r, c):
                            # Extract short ID (e.g. "player_0" -> "0")
                            short_id = name.split('_')[-1]
                            agents_here.append(short_id)

                    if len(agents_here) > 1: 
                        content = " X " # Collision indicator
                    elif len(agents_here) == 1: 
                        content = f" {agents_here[0]} " # Single agent

                    row_str += content
                grid_lines.append(row_str + "|")
            
            # 5. Bottom Border
            grid_lines.append("+" + "---" * cols + "+")
            
            return grid_lines

        # --- WRITE TO FILE IN BLOCKS ---
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== REPLAY LOG (Total Steps: {max_steps}) ===\n\n")

            # Iterate through time steps in batches (defined by grids_per_row)
            for t_start in range(0, max_steps, grids_per_row):
                
                # Collect grids for the current row
                batch_grids = []
                for offset in range(grids_per_row):
                    t = t_start + offset
                    if t < max_steps:
                        batch_grids.append(generate_single_grid(t))
                
                # Stop if no grids collected
                if not batch_grids: break
                
                # Print row by row (side-by-side concatenation)
                grid_height = len(batch_grids[0])
                
                for line_idx in range(grid_height):
                    line_str = ""
                    for grid in batch_grids:
                        # Add current grid line + fixed spacing
                        line_str += grid[line_idx] + "    " 
                    f.write(line_str + "\n")
                
                f.write("\n") # Vertical spacing between rows of grids

    
    
    def save_replay_as_gif(self, filename="replay/replay.gif", custom_paths=None, duration=400):
        """
        Saves the last episode (or custom paths) as an animated GIF.
        
        Args:
            filename (str): Output path for the GIF.
            custom_paths (dict): Optional dictionary of agent paths to replay.
            duration (int): Duration of each frame in milliseconds.
        """
        
        # 1. Retrieve paths
        if custom_paths:
            paths = custom_paths
        else:
            # Retrieve full history of the last played episode from the environment
            paths = {}
            # Ensure agents exist
            if hasattr(self.env, 'agents_obj') and self.env.agents_obj:
                for name, player in self.env.agents_obj.items():
                    paths[name] = player.sequence

        if not paths:
            print("No paths found to save.")
            return

        # Calculate maximum duration
        max_steps = 0
        for seq in paths.values():
            if len(seq) > max_steps: max_steps = len(seq)

        rows, cols = self.env.grid.shape
        
        # Graphic settings
        cell_size = 65
        padding = 30
        img_w = cols * cell_size + 2 * padding
        img_h = rows * cell_size + 2 * padding + 30 # Extra height for text
        
        frames = []

        # 2. Time Loop (Frame by Frame)
        for t in range(max_steps):
            # Create a new black canvas
            img = Image.new('RGB', (img_w, img_h), 'black')
            draw = ImageDraw.Draw(img)
            
            # Draw Step Info
            draw.text((10, 5), f"Step: {t}/{max_steps}", fill="white")

            # Calculate current positions for this timestamp
            current_positions = {}
            for name in self.env.agents:
                if name not in paths: continue
                path = paths[name]
                
                # If agent is moving, get position at t
                if t < len(path):
                    current_positions[name] = path[t]
                # If agent finished early, keep at last position (Visual Parking Brake)
                else:
                    current_positions[name] = path[-1]

            # 3. Draw Grid
            for r in range(rows):
                for c in range(cols):
                    x = padding + c * cell_size
                    y = padding + 20 + r * cell_size
                    
                    # Default Background
                    fill_color = "gray"
                    text = "."
                    
                    # Walls
                    if self.env.grid[r, c] == 0:
                        fill_color = "white"
                        text = "#"
                    # Doorways
                    elif (r, c) in self.env.doorway_coords:
                        fill_color = "yellow"
                        text = "D"
                    
                    # Goal Check (Draw 'G' only if empty)
                    for ag_name in self.env.agents:
                         if self.env.agents_obj[ag_name].goal_pos == (r,c):
                             fill_color = "lime"
                             text = "G"

                    # Check for Agents at this cell
                    agents_here = []
                    for name, pos in current_positions.items():
                        if pos == (r, c):
                            # Extract Short ID (e.g., "player_0" -> "0")
                            short_id = name.split('_')[-1]
                            agents_here.append(short_id)

                    # Collision Coloring
                    if len(agents_here) > 1:
                        fill_color = "red" # Crash
                        text = "X"
                    elif len(agents_here) == 1:
                        fill_color = "cyan"
                        text = agents_here[0]

                    # Draw Cell Rectangle
                    draw.rectangle([x, y, x+cell_size-2, y+cell_size-2], fill=fill_color)
                    # Draw Text Centered (Manual approximation)
                    draw.text((x+15, y+10), text, fill="black")

            frames.append(img)

        # 4. Save File
        if frames:
            frames[0].save(
                filename, 
                save_all=True, 
                append_images=frames[1:], 
                duration=duration, 
                loop=0 # Infinite loop
            )

    def sparse(self):
        target_episode = int(self.NUM_EPISODES * (2/3)) 
        epsilon_min = 0.15
        
        decay_rate = (epsilon_min / 1.0) ** (1 / target_episode)
        
        self.agents = {}
        for name in self.env.agents:
            self.agents[name] = SparseQLearner(
                agent_id=name,
                num_states=self.num_states,
                action_size=5, 
                alpha=self.alpha,
                gamma=0.98,
                epsilon=1.0,
                epsilon_min=epsilon_min,
                epsilon_decay=decay_rate
            )

        history = {
            "train_rewards": [], # Noisy training data
            "eval_rewards": [],  # Clean evaluation data
            "eval_points": [],   # X-axis points (0, 1000...)
            "eval_collisions": [],
            "eval_steps": [],
            "collisions": [],
            "steps": []
        }


        eval, collision, steps, _ = self.evaluate_sparse(self.agents)
        history["eval_rewards"].append(eval)
        history["eval_points"].append(0)
        history["eval_collisions"].append(collision)
        history["eval_steps"].append(steps)

        for episode in range(1, self.NUM_EPISODES + 1):
            
            obs, terms, infos = self.env.reset()
            active_agents = list(self.agents.keys())
            
            episode_agent_rewards = 0
            collision_count = 0

            for step in range(self.MAX_STEPS):
                
                # SELECT ACTIONS
                actions = {}
                full_decisions = {} 
                
                for name in active_agents:
                    agent = self.agents[name]
                    
                    state_id = obs[name]["id"]
                    my_coords = obs[name]["coords"]
                    neighbors = obs[name]["visible_neighbors"] 
                    
                    decision = agent.select_action(state_id, my_coords, neighbors)
                    
                    full_decisions[name] = decision
                    actions[name] = decision[1] 

                # STEP
                next_obs, rewards, terms, collision, infos = self.env.step(actions)
                
                collision_count += collision    

                # UPDATE
                next_active_agents = []
                
                for name in active_agents:
                    agent = self.agents[name]
                    
                    meta, phys = full_decisions[name]
                    decision_tuple = (meta, phys)
                    s_id = obs[name]["id"]
                    my_coords = obs[name]["coords"]
                    neighbors = obs[name]["visible_neighbors"]
                    
                    ns_id = next_obs[name]["id"]
                    next_coords = next_obs[name]["coords"]
                    next_neighbors = next_obs[name]["visible_neighbors"]
                    
                    rew = rewards[name]
                    discounted_reward = rew * (0.98 ** step)
                    episode_agent_rewards += discounted_reward
                    
                    agent.update(
                        s_id, my_coords, neighbors, 
                        decision_tuple, rew, 
                        ns_id, next_coords, next_neighbors
                    )
                    
                    if not terms[name]:
                        next_active_agents.append(name)

                obs = next_obs
                active_agents = next_active_agents

                if len(active_agents) == 0:
                    break

            for agent in self.agents.values(): agent.update_epsilon()

            history["train_rewards"].append(episode_agent_rewards / self.num_agents)
            history["collisions"].append(collision_count)

            finish_times = [infos[n].get('step', 0) for n in self.agents]
            adj_times = [t if t > 0 else self.MAX_STEPS for t in finish_times]
            history["steps"].append(max(adj_times))

            if episode % 1000 == 0:
                avg_score, collision, steps, path = self.evaluate_sparse(self.agents)
                history["eval_rewards"].append(avg_score)
                history["eval_points"].append(episode)
                history["eval_collisions"].append(collision)
                history["eval_steps"].append(steps)
        
            if max(adj_times) < self.MAX_STEPS:
                self.save_replay_as_gif(filename="replay/replay_sparse_agents_"+str(self.num_agents)+"_env_"+str(self.env_num)+".gif")
                self.save_replay_as_text(filename="print/replay_sparse_agents_"+str(self.num_agents)+"_env_"+str(self.env_num)+".text")
        return history

    def indipendent(self): 
        # 1. Initialize Agents
        target_episode = int(self.NUM_EPISODES * (2/3)) 
        epsilon_min = 0.15
        
        decay_rate = (epsilon_min / 1.0) ** (1 / target_episode)

        self.agents = {}
        for name in self.env.agents:
            self.agents[name] = IndependentQLearner(
                action_size=4,
                num_states=self.num_states,
                alpha=self.alpha,        
                gamma=0.98,       
                epsilon=1.0,       
                epsilon_min=epsilon_min,   
                epsilon_decay=decay_rate 
            )
        

        history = {
            "train_rewards": [], # Noisy training data
            "eval_rewards": [],  # Clean evaluation data
            "eval_points": [],   # X-axis points (0, 1000...)
            "eval_collisions": [],
            "eval_steps": [],
            "collisions": [],
            "steps": []
        }

        eval, collision, steps, _ = self.evaluate_indipendent(self.agents)
        history["eval_rewards"].append(eval)
        history["eval_points"].append(0)
        history["eval_collisions"].append(collision)
        history["eval_steps"].append(steps)



        for episode in range(1, self.NUM_EPISODES+1):
            obs, terms, _ = self.env.reset()
            self.terms = terms
            collision_count = 0
            episode_agent_rewards = 0
            active_agents = list(self.agents.keys())
            for step in range(self.MAX_STEPS):
                # SELECT ACTIONS 
                actions = {}

                for name in self.agents:
                    if name in active_agents:
                        state_id = obs[name]["id"]
                        actions[name] = self.agents[name].select_action(state_id)

                # STEP 
                next_obs, rewards, terms, collision, infos = self.env.step(actions)
                collision_count += collision

                total_reward = 0
                next_active_agents = []
            
                for name in active_agents:
                    agent = self.agents[name]
                
                    state_id = obs[name]["id"]
                    action = actions[name]
                    reward = rewards[name]
                    next_state_id = next_obs[name]["id"]
                    discounted_reward = reward * (0.98 ** step) 
                    episode_agent_rewards += discounted_reward

                    agent.update(state_id, action, reward, next_state_id)
                
                    if not terms[name]:
                        next_active_agents.append(name)
                    else:
                        pass

                obs = next_obs
                active_agents = next_active_agents

                if len(active_agents) == 0:
                    break
            # END EPISODE 
            for agent in self.agents.values():
                agent.update_epsilon()

            finish_times = []
            for name in self.agents:
                t = infos[name].get('step', 0) 
                finish_times.append(t)

            adjusted_times = [t if t > 0 else self.MAX_STEPS for t in finish_times]
            episode_duration = max(adjusted_times)
            history["steps"].append(episode_duration)            
            history["train_rewards"].append(episode_agent_rewards / self.num_agents)
            history["collisions"].append(collision_count)


            if episode % 1000 == 0:
                avg_score, collision, steps, path = self.evaluate_indipendent(self.agents)
                history["eval_rewards"].append(avg_score)
                history["eval_points"].append(episode)
                history["eval_collisions"].append(collision)
                history["eval_steps"].append(steps)

        return history