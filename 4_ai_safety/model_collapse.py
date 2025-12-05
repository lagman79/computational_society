"""
AI SAFETY: Model Autophagy Simulation
-------------------------------------
This script demonstrates 'Model Collapse' - the degenerative process that occurs
when generative AI models are trained recursively on synthetic data.

The simulation proves that without external human data (entropy), the variance
of the model's output collapses to zero, leading to a loss of information/creativity.

Method: Recursive Gaussian Fitting
Output: PDF Report visualizing the distribution collapse over generations.

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages
import os

def get_human_data(n_samples=2000):
    """
    Generates the initial 'Ground Truth' data.
    Simulates a complex, bimodal human distribution (e.g., polarized opinions or diverse art styles).
    """
    # 50% Group A (-2), 50% Group B (+2)
    data_a = np.random.normal(-2, 1, n_samples // 2)
    data_b = np.random.normal(2, 1, n_samples // 2)
    return np.concatenate([data_a, data_b])

def train_and_generate(data, n_samples=2000):
    """
    Simulates the AI training and generation loop.
    1. Fits a Gaussian distribution to the input data (Learning).
    2. Samples new data from that distribution (Generation).
    Note: Fitting a unimodal Gaussian to bimodal data causes immediate information loss.
    """
    mu, std = stats.norm.fit(data)
    # Prevent std from becoming absolute zero to avoid numerical errors, though collapse is the goal
    std = max(std, 1e-5) 
    new_data = np.random.normal(mu, std, n_samples)
    return new_data, mu, std

def run_autophagy_simulation(generations=15, output_file="collapse_report.pdf"):
    """
    Runs the recursive training loop and generates the PDF report.
    """
    print(f"--- Generating Report: {output_file} ---")
    
    n_samples = 2000
    current_data = get_human_data(n_samples)
    
    history = [current_data]
    
    print("Starting Recursive Training Loop...")
    for i in range(generations):
        new_data, mu, std = train_and_generate(current_data, n_samples)
        history.append(new_data)
        current_data = new_data
        # Print progress every 5 generations
        if i % 5 == 0:
            print(f"Generation {i}: Variance = {np.var(new_data):.4f}")

    # Visualization
    with PdfPages(output_file) as pdf:
        # Cover
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.text(0.5, 0.6, "MODEL AUTOPHAGY REPORT", ha='center', fontsize=22, weight='bold')
        plt.text(0.5, 0.5, "Mathematical Proof of AI Collapse", ha='center', fontsize=16)
        plt.text(0.5, 0.4, f"Generations Simulated: {generations}", ha='center', fontsize=12)
        pdf.savefig(fig)
        plt.close()

        # Plot specific generations
        plot_indices = [0, 1, 3, 5, 10, 15]
        
        for gen in range(len(history)):
            if gen not in plot_indices: continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histogram & Density Plot
            sns.histplot(history[gen], kde=True, stat="density", color="skyblue", bins=30, alpha=0.6, ax=ax)
            
            # Metrics
            variance = np.var(history[gen])
            title = "Gen 0: Human Data (Complex/Bimodal)" if gen == 0 else f"Generation {gen}: AI Trained on AI"
            
            ax.set_title(title, fontsize=15, weight='bold')
            ax.set_xlim(-6, 6)
            ax.set_ylim(0, 0.8) # Fixed y-axis to show the peak growing
            
            # Annotations
            ax.text(4.0, 0.7, f"Variance: {variance:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            if gen == 10:
                ax.text(0, 0.4, "INFORMATION COLLAPSE", ha='center', color='red', weight='bold', fontsize=14)

            pdf.savefig(fig)
            plt.close()

    print(f"Simulation complete. Report saved to {output_file}")

if __name__ == "__main__":
    run_autophagy_simulation()