import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

def plot_training_rewards_evaluations(mean, std, title, ylabel, color='blue', eval_mean=None, eval_std=None, eval_x=None,save_folder="plots"):
    """
    Generates a dual-plot for Rewards:
    1. Continuous line with shaded error region (Std Dev) for Training data.
    2. Discrete points with error bars for Evaluation data (Greedy policy).
    """
    plt.figure(figsize=(12, 7))
    
    x_axis = np.arange(len(mean))
    plt.plot(x_axis, mean, color=color, alpha=0.9, label='Training (Avg)', linewidth=1)    
    upper_bound = mean + std
    lower_bound = mean - std
    
    plt.fill_between(
        x_axis, 
        lower_bound, 
        upper_bound, 
        color=color, 
        alpha=0.2 
    )

    # Evaluation Data 
    if eval_mean is not None:
        if eval_x is None:
            eval_x = np.linspace(0, len(mean), len(eval_mean))
            
        plt.errorbar(
            eval_x,    
            eval_mean,     
            yerr=eval_std,   
            fmt='o-',     
            color='red', 
            capsize=5,       
            linewidth=2, 
            label='Evaluation (Greedy ± Std)'
        )

    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-50, 20)
    
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True) 
        safe_title = title.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        filename = f"{save_folder}/{safe_title}.pdf" 
        plt.savefig(filename, format='pdf', dpi=300) 
        print(f"Plot saved into: {filename}")

    plt.show()

def plot_not_rewards(mean, std, title, ylabel, color='blue', eval_mean=None, eval_std=None, eval_x=None,save_folder="plots"):
    """
    Generates plots for non-reward metrics (Collisions and Steps).
    Handles integer Y-axis ticks and specific zoom limits based on the metric type.
    """
    x_axis = np.arange(len(mean))

    plt.figure(figsize=(12, 7))

    plt.plot(
        x_axis, 
        mean, 
        color=color, 
        alpha=0.9, 
        label=f'{ylabel} Training', 
        linewidth=1
    )

    plt.fill_between(
        x_axis, 
        mean - std, 
        mean + std, 
        color=color, 
        alpha=0.2 
    )

    # Evaluation Data 
    if eval_mean is not None:
        if eval_x is None:
            eval_x = np.linspace(0, len(mean), len(eval_mean))
            
        plt.errorbar(
            eval_x,        
            eval_mean,     
            yerr=eval_std, 
            fmt='o-',      
            color='red', 
            capsize=5,      
            linewidth=2, 
            label='Evaluation (Greedy ± Std)'
        )

    plt.xlabel('Episodes')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    if "Collisions" in ylabel or "Miscoordination" in ylabel:
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(-0.5, 6) 
        
    elif "Steps" in ylabel:
        plt.ylim(-140, 140)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_folder:
        os.makedirs(save_folder, exist_ok=True) 
        safe_title = title.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        filename = f"{save_folder}/{safe_title}.pdf"
        
        plt.savefig(filename, format='pdf', dpi=300) 
        print(f"Plot saved into: {filename}")

    plt.show()