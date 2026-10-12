import numpy as np

class Storage:

    """
    This class serves as a centralized data repository for experimental results.
    It aggregates metrics (rewards, collisions, steps) from multiple independent training runs
    to facilitate statistical analysis (mean and standard deviation) for plotting.
    """

    def __init__(self):
        self.runs_train_rewards = []
        self.runs_eval_rewards = []
        
        self.runs_collisions = []      # Training Collisions
        self.runs_eval_collisions = [] # Evaluation Collisions 
        
        self.runs_steps = []           # Training Steps
        self.runs_eval_steps = []      # Evaluation Steps 
        
        # Stores the X-axis points for evaluation (e.g., 0, 1000, 2000...)
        # Saved once since it is consistent across all runs.
        self.eval_x_points = None      

    def add_run(self, history):
        """
        Appends the complete history of a single training run to the storage lists.
        
        Args:
            history (dict): A dictionary containing lists of metrics for one run 
                            (e.g., history['train_rewards'] = [r0, r1, ...]).
        """
        
        self.runs_train_rewards.append(history["train_rewards"])
        self.runs_eval_rewards.append(history["eval_rewards"])
        
        self.runs_collisions.append(history["collisions"])
        self.runs_steps.append(history["steps"])
        if "eval_collisions" in history:
            self.runs_eval_collisions.append(history["eval_collisions"])
        
        if "eval_steps" in history:
            self.runs_eval_steps.append(history["eval_steps"])
        
        if self.eval_x_points is None and "eval_points" in history:
            self.eval_x_points = history["eval_points"]



    def get_statistics(self, metric_name):
        """
        Calculates the Mean and Standard Deviation across all stored runs for a given metric.
        
        Args:
            metric_name (str): The key of the metric to analyze (e.g., 'train_rewards').
            
        Returns:
            tuple: (mean_array, std_array) representing the statistics at each episode step.
        """
        if metric_name == 'train_rewards':
            data = np.array(self.runs_train_rewards)
        elif metric_name == 'eval_rewards':
            data = np.array(self.runs_eval_rewards)
            
        elif metric_name == 'collisions':
            data = np.array(self.runs_collisions)
        elif metric_name == 'eval_collisions':
            data = np.array(self.runs_eval_collisions)
            
        elif metric_name == 'steps':
            data = np.array(self.runs_steps)
        elif metric_name == 'eval_steps':
            data = np.array(self.runs_eval_steps)
            
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        if data.size == 0:
            print(f"Warning: No data for {metric_name}")
            return None, None

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        return mean, std