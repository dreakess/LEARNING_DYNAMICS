import numpy as np
import random

class IndependentQLearner:

    """
        A Q-Learning agent that acts independently, ignoring the presence of other agents.
        It maintains a single Q-table mapping local states to action values.
    """

    def __init__(self, action_size, num_states, alpha=0.1, gamma=0.98,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        

        # 1. Dimensions
        self.action_size = action_size
        self.num_states = num_states

        # 2. Hyperparameters
        self.gamma = gamma      
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Handling the learning rate, we can choose between fixed and dynamically learning rate
        if alpha == 0.0:
            self.dynamic_lr = True
            self.alpha = None 
            self.count_table = np.zeros((num_states + 1, action_size))
        else:
            self.dynamic_lr = False
            self.alpha = alpha

        # 3. Q-Table (State-Aware)
        # Dimensions: [States, Actions]
        self.q_table = np.zeros((num_states + 1, action_size))

    def select_greedy(self, state):
        """
        Helper to pick the best action for a specific state, handling ties randomly.
        """
        q_values = self.q_table[state]
        max_val = np.max(q_values)
        optimal_actions = np.flatnonzero(np.isclose(q_values, max_val))
        return np.random.choice(optimal_actions)

    def select_action(self, state):
        """
        Epsilon-Greedy Selection.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return self.select_greedy(state)

    def update(self, state, action, reward, next_state):
        """
        Standard Q-Learning Update with Dynamic Alpha support.
        """
        
        if self.dynamic_lr:
            self.count_table[state, action] += 1
            current_alpha = 1.0 / self.count_table[state, action]
        else:
            current_alpha = self.alpha

        old_val = self.q_table[state, action]
        max_future = np.max(self.q_table[next_state])

        new_val = old_val + current_alpha * (reward + self.gamma * max_future - old_val)

        self.q_table[state, action] = new_val

    def update_epsilon(self):
        """
        Decays the exploration rate.
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)