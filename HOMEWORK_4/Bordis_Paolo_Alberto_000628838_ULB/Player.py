import numpy as np

class Player:
  def __init__(self, name, start_pos, goal_pos):
      self.name = name
      self.start_pos = start_pos
      self.goal_pos = goal_pos
      self.check_goal = False

      self.step_to_reach_goal = 0

      # Unpack tuples: start_pos is usually (Row, Col)
      # We map Row -> y (Vertical), Col -> x (Horizontal)
      self.y, self.x = start_pos


      # Save goal for termination checks in the Env
      self.goal_pos = goal_pos


      self.sequence = []
      self.sequence.append((self.y, self.x))

  def reset(self):
     self.check_goal = False
     self.step_to_reach_goal = 0

  def update_step(self, step):
    self.step_to_reach_goal = step

  def get_step(self):
    return self.step_to_reach_goal


  def get_sequence_steps(self):
    return self.sequence

  def get_state(self, grid_map):
      """
      Returns the State ID (Integer) from the grid at current position.
      Used for Q-Table lookups.
      """
      # CRITICAL FIX: Access as [Row, Col] -> [y, x]
      return grid_map[self.y, self.x]

  def get_position(self):
      """Returns (Row, Col) for the Environment to use."""
      return (self.y, self.x)

  def set_position(self, r, c):
        # Niente scambio! r va in y, c va in x
        self.y = r 
        self.x = c
        self.sequence = [(r, c)]


  def step(self, action, grid_map):
        """
        Executes the move with 80% success / 20% slip probability.
        action: 0=N, 1=S, 2=E, 3=W, 4=Coordinate
        grid_map: 0=Wall, 1=Floor
        """

        # STOCHASTIC PHYSICS (The 80/20 Rule) ---
        slips = {
            0: [3, 2], # North (0) -> Slips West (3) or East (2)
            1: [2, 3], # South (1) -> Slips East (2) or West (3)
            2: [0, 1], # East  (2) -> Slips North (0) or South (1)
            3: [1, 0]  # West  (3) -> Slips South (1) or North (0)
        }

        outcomes = [action, slips[action][0], slips[action][1]]
        probs = [0.8, 0.1, 0.1]
        actual_move = np.random.choice(outcomes, p=probs)


        # --- CALCULATE COORDINATES ---
        next_y, next_x = self.y, self.x

        if actual_move == 0:   # North
            next_y -= 1        # Move UP (lower index)
        elif actual_move == 1: # South
            next_y += 1        # Move DOWN (higher index)
        elif actual_move == 2: # East
            next_x += 1
        elif actual_move == 3: # West
            next_x -= 1

        # WALL & BOUNDARY CHECK ---
        rows, cols = grid_map.shape

        if not (0 <= next_y < rows and 0 <= next_x < cols):
            next_y, next_x = self.y, self.x

        elif grid_map[next_y, next_x] == 0:
            next_y, next_x = self.y, self.x

        self.y = next_y
        self.x = next_x
        self.sequence.append((self.y, self.x))
