"""
SOCIAL DYNAMICS: Thermodynamics of Polarization
-----------------------------------------------
This script simulates opinion dynamics using the Deffuant (Bounded Confidence) Model.
It investigates the 'Phase Transition' phenomenon where a society abruptly switches
from consensus to sectarianism based on the tolerance threshold.

Model: Agent-Based Modeling (ABM)
Output: A multi-page PDF analyzing convergence over time.

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os

def run_deffuant_simulation(n_agents=200, steps=8000, threshold=0.2, convergence=0.5):
    """
    Executes the Deffuant model logic.
    Agents interact only if their opinion difference is below the threshold.
    """
    opinions = np.random.rand(n_agents)
    history = [opinions.copy()]
    
    for _ in range(steps):
        # Select two random agents
        i, j = np.random.randint(0, n_agents, 2)
        diff = abs(opinions[i] - opinions[j])
        
        # Interaction Rule: Bounded Confidence
        if diff < threshold:
            change = diff * convergence
            if opinions[i] < opinions[j]:
                opinions[i] += change
                opinions[j] -= change
            else:
                opinions[i] -= change
                opinions[j] += change
        
        history.append(opinions.copy())
        
    return np.array(history)

def generate_polarization_report(output_file="polarization_report.pdf"):
    """
    Runs batch experiments across a range of thresholds and compiles results.
    """
    print(f"--- Generating Report: {output_file} ---")
    
    # Define experiment parameters
    t_values = [0.05, 0.15, 0.25, 0.28, 0.30, 0.32, 0.35, 0.50]
    
    # Create DataFrame for structured processing
    df_experiments = pd.DataFrame({'threshold': t_values})
    df_experiments['description'] = df_experiments['threshold'].apply(
        lambda x: "Phase Transition (Tipping Point)" if 0.28 <= x <= 0.32 else f"Simulation T={x}"
    )

    with PdfPages(output_file) as pdf:
        # Cover Page
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.6, "OPINION DYNAMICS REPORT", ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.5, "Sensitivity Analysis of the Deffuant Model", ha='center', fontsize=16)
        plt.text(0.5, 0.4, f"Experiments: {len(df_experiments)}", ha='center', fontsize=12)
        pdf.savefig(fig)
        plt.close()

        # Batch Processing
        for index, row in df_experiments.iterrows():
            t_val = row['threshold']
            desc = row['description']
            
            print(f"Running Experiment {index+1}/{len(df_experiments)}: Threshold {t_val}...")
            
            sim_data = run_deffuant_simulation(n_agents=200, steps=10000, threshold=t_val)
            
            # Plotting
            fig = plt.figure(figsize=(10, 6))
            plt.title(f"Threshold = {t_val} | {desc}", fontsize=16)
            plt.xlabel("Time Steps (Interactions)")
            plt.ylabel("Opinion Spectrum (0=Left, 1=Right)")
            plt.ylim(-0.05, 1.05)
            
            for agent_id in range(sim_data.shape[1]):
                plt.plot(sim_data[:, agent_id], lw=0.3, alpha=0.5)
                
            plt.figtext(0.5, 0.01, f"Parameters: N=200, Convergence=0.5", ha="center", fontsize=10)
            
            pdf.savefig(fig)
            plt.close()

    print(f"Report saved successfully to {output_file}")

if __name__ == "__main__":
    generate_polarization_report()