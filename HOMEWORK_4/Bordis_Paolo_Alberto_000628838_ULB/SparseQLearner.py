import numpy as np
import random
from collections import defaultdict

class SparseQLearner:
    """
    Implements the 'Learning with Sparse Interactions' algorithm (Melo & Veloso, 2009).
    It maintains two separate Q-functions:
    1. Q_local: A standard Q-table for independent navigation (Dense Matrix).
    2. Q_common: A sparse Q-table for coordination in specific joint states (Dictionary).
    """

    def __init__(self, agent_id, num_states, action_size=5, 
                 alpha=0.1, gamma=0.98, epsilon=1.0, 
                 epsilon_min=0.05, epsilon_decay=0.999):
        
        self.agent_id = agent_id
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if alpha == 0.0:
            self.dynamic_lr = True
            self.alpha = None 
            self.count_local = np.zeros((num_states + 1, action_size))
            self.count_common = defaultdict(int)
        else:
            self.dynamic_lr = False
            self.alpha = alpha

        # --- Q-TABLES ---
        self.q_local = np.zeros((num_states + 1, action_size))
        self.q_common = defaultdict(float)

    
    def select_action(self, state_id, my_coords, visible_neighbors):
        """
        Hierarchical Action Selection:
        1. Choose a Meta-Action (Navigation vs. Coordinate) using Q_local.
        2. Resolve the Physical Action (North/South/East/West).
        """

        if np.random.random() < self.epsilon:
            meta_action = np.random.randint(self.action_size)
        else:
            meta_action = self._get_greedy(self.q_local[state_id])

        if meta_action == 4: 
            if len(visible_neighbors) > 0:
                joint_context = (my_coords, visible_neighbors)
                q_vals = [self.q_common[(joint_context, a)] for a in range(4)]
                physical_action = self._get_greedy(q_vals)
            else:
                q_vals = self.q_local[state_id][:4]
                physical_action = self._get_greedy(q_vals)
            return (4, physical_action)
        else:
            return (meta_action, meta_action)


    def update(self, state_id, my_coords, visible_neighbors, 
               decision_tuple, reward, 
               next_state_id, next_my_coords, next_visible_neighbors):
        
        """
        Performs the Q-Learning update.
        - Q_local is ALWAYS updated (learning the value of the 'intent').
        - Q_common is conditionally updated (only if coordination was active).
        """
        
        meta_action, physical_action = decision_tuple

        # ==================================================
        # UPDATE 1: LOCAL TABLE (Q*)
        # ==================================================
        
        # Determine Alpha (Fixed vs Dynamic)

        if self.dynamic_lr:
            self.count_local[state_id, meta_action] += 1
            alpha_local = 1.0 / self.count_local[state_id, meta_action]
        else:
            alpha_local = self.alpha

        old_val = self.q_local[state_id, meta_action]
        max_next = np.max(self.q_local[next_state_id])
        
        self.q_local[state_id, meta_action] += alpha_local * (reward + self.gamma * max_next - old_val)


        # ==================================================
        # UPDATE 2: COMMON TABLE (Q^C)
        # ==================================================
        # Update ONLY if we chose to coordinate AND neighbors were present.

        if meta_action == 4 and len(visible_neighbors) > 0:
            
            current_key = ((my_coords, visible_neighbors), physical_action)
            
            if self.dynamic_lr:
                self.count_common[current_key] += 1
                alpha_common = 1.0 / self.count_common[current_key]
            else:
                alpha_common = self.alpha

            old_val_c = self.q_common[current_key]
            
            next_context = (next_my_coords, next_visible_neighbors)
            next_vals = [self.q_common[(next_context, a)] for a in range(4)]
            max_next_c = max(next_vals) if next_vals else 0.0
            
            self.q_common[current_key] += alpha_common * (reward + self.gamma * max_next_c - old_val_c)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def _get_greedy(self, q_vals):
        """
        Selects the index with the highest value.
        Breaks ties randomly to prevent deterministic loops.
        """
        vals = np.array(q_vals)
        max_val = np.max(vals)
        best_indices = np.flatnonzero(np.isclose(vals, max_val))
        return np.random.choice(best_indices)