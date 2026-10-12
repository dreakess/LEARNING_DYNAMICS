import concurrent.futures
import time
import numpy as np
from tqdm import tqdm
from Storage import Storage
from Utils import plot_training_rewards_evaluations, plot_not_rewards
from Training import Training

def _worker_run(config):
    """
    Function called to compute for one training set.
    Executes in a separate process.
    """

    algo_type, num_agents, env_num = config
    
    # Istance of Training class
    if num_agents == 4:
        t = Training(num_agents=num_agents, env_num=env_num, alpha=0.)
    else:
        t = Training(num_agents=num_agents, env_num=env_num, alpha=0)
    
    # Choosing of the algorithm to compute
    if algo_type == "independent":
        obj = t.indipendent()
        return obj
    elif algo_type == "sparse":
        obj = t.sparse()
        return obj

# --- CLASSE RUNNER ---
class ExperimentRunner:
    '''
    This class is responsible to compute the result using multiprocessing across 7 training runs.
    '''


    def __init__(self, num_runs=7, num_agents=2, env_num=1):
        self.num_runs = num_runs
        self.num_agents = num_agents
        self.env_num = env_num
        self.storage = Storage()
        self.results = None

    def run_experiment(self, algo_type="independent"):
        """
        Run parallell 7 Training set
        algo_type: "independent" o "sparse"
        """
        print(f"\n--- Loading of the experiments: {algo_type.upper()} ({self.num_runs} Runs) ---")
        start_time = time.time()
        
        # Preparing this tuple to identify the Training running
        task_config = (algo_type, self.num_agents, self.env_num)
        
        # Reset of the Storange
        self.storage = Storage()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Creating of the n task
            futures = [executor.submit(_worker_run, task_config) for _ in range(self.num_runs)]
            
            # Progress bar definition
            for f in tqdm(concurrent.futures.as_completed(futures), total=self.num_runs, desc="Progress"):
                try:
                    hist= f.result()
                    self.storage.add_run(hist)
                except Exception as e:
                    print(f"Error inside the running: {e}")

        print(f"Completed with {time.time() - start_time:.2f} seconds.\n")

    def plot_results(self, algo_name="Algorithm"):
        """
        Generation of the plot for (Reward, Collision, Steps).
        """
        if not self.storage.runs_train_rewards:
            print("there are no data to plot!")
            return

        # Recover of the statistics data
        rew_mean, rew_std = self.storage.get_statistics('train_rewards')
        eval_mean, eval_std = self.storage.get_statistics('eval_rewards')
        coll_mean, coll_std = self.storage.get_statistics('collisions')
        step_mean, step_std = self.storage.get_statistics('steps')
        eval_coll_mean, eval_coll_std = self.storage.get_statistics('eval_collisions')
        eval_step_mean, eval_step_std = self.storage.get_statistics('eval_steps')

        # set different color for different algorithm
        base_color = "blue" if "independent" == algo_name else "darkorange"

        # Plot Rewards
        plot_training_rewards_evaluations(
            mean=rew_mean, 
            std=rew_std, 
            title=f"{algo_name}: Average Discounted Rewards", 
            ylabel="Reward", 
            color=base_color,
            eval_mean=eval_mean, 
            eval_std=eval_std, 
            eval_x=self.storage.eval_x_points,
            save_folder="results_pdf"
        )

        # Plot Collisions
        plot_not_rewards(
            mean=coll_mean, 
            std=coll_std, 
            title=f"{algo_name}: Miscoordinations per Episode", 
            ylabel="Collisions", 
            color="purple",
            eval_mean=eval_coll_mean, 
            eval_std=eval_coll_std, 
            eval_x=self.storage.eval_x_points,
            save_folder="results_pdf"
        )   

        # Plot Steps
        plot_not_rewards(
            mean=step_mean, 
            std=step_std, 
            title=f"{algo_name}: Episode Duration (Steps)", 
            ylabel="Steps", 
            color="green",
            eval_mean=eval_step_mean, 
            eval_std=eval_step_std, 
            eval_x=self.storage.eval_x_points,
            save_folder="results_pdf"
        )