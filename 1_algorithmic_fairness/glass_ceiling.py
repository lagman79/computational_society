"""
ALGORITHMIC FAIRNESS: The Glass Ceiling Simulation
--------------------------------------------------
This script simulates a job market using Graph Theory to demonstrate how 
algorithmic strictness creates systemic barriers to social mobility.

It performs a sensitivity analysis by varying the 'strictness' parameter of a
matching algorithm and visualizing the connectivity between low-skill workers
and high-skill jobs.

Output: A multi-page PDF report visualizing the topological separation.

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
import os

def generate_population(n_agents=80, seed=42):
    """
    Generates a fixed population of workers and jobs in a 2D skill space.
    Using a fixed seed ensures reproducibility across experiments.
    """
    np.random.seed(seed)
    
    # Group A: Junior Workers (Low Skill Cluster - Bottom Left)
    workers_x = np.random.normal(0.3, 0.08, n_agents)
    workers_y = np.random.normal(0.3, 0.08, n_agents)
    
    # Group B: Senior Jobs (High Skill Cluster - Top Right)
    jobs_x = np.random.normal(0.7, 0.08, n_agents)
    jobs_y = np.random.normal(0.7, 0.08, n_agents)
    
    return workers_x, workers_y, jobs_x, jobs_y

def run_matching_algorithm(strictness, wx, wy, jx, jy):
    """
    Constructs a graph where edges represent valid job matches based on
    Euclidean distance and the algorithmic strictness threshold.
    """
    G = nx.Graph()
    n = len(wx)
    
    # Add Nodes
    for i in range(n):
        G.add_node(f"W{i}", pos=(wx[i], wy[i]), type='worker')
        G.add_node(f"J{i}", pos=(jx[i], jy[i]), type='job')
    
    # Add Edges (The Matching Logic)
    matches = 0
    
    for i in range(n):
        w_pos = np.array([wx[i], wy[i]])
        for j in range(n):
            j_pos = np.array([jx[j], jy[j]])
            dist = np.linalg.norm(w_pos - j_pos)
            
            # If distance is within the allowed radius, create a link
            if dist < strictness:
                G.add_edge(f"W{i}", f"J{j}")
                matches += 1
                
    return G, matches

def generate_report(output_file="glass_ceiling_report.pdf"):
    """
    Runs the simulation for multiple strictness levels and saves a PDF report.
    """
    print(f"--- Generating Report: {output_file} ---")
    
    # Sensitivity Analysis Parameters
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    wx, wy, jx, jy = generate_population() # Generate once, use many times

    with PdfPages(output_file) as pdf:
        # Cover Page
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.6, "ALGORITHMIC SOCIAL MOBILITY AUDIT", ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.5, "Simulation: 'The Opportunity Gap'", ha='center', fontsize=16)
        plt.text(0.5, 0.4, f"Scenarios Tested: {len(thresholds)}", ha='center', fontsize=12)
        pdf.savefig(fig)
        plt.close()

        # Iterate through scenarios
        for val in thresholds:
            print(f"Processing Strictness Radius: {val}...")
            
            G, matches = run_matching_algorithm(val, wx, wy, jx, jy)
            
            # Visualization
            plt.figure(figsize=(10, 8))
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw Workers (Blue)
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n,d in G.nodes(data=True) if d['type']=='worker'], 
                                   node_color='#3498db', node_size=50, label='Workers')
            # Draw Jobs (Red)
            nx.draw_networkx_nodes(G, pos, nodelist=[n for n,d in G.nodes(data=True) if d['type']=='job'], 
                                   node_color='#e74c3c', node_size=50, label='Jobs')
            # Draw Edges
            nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')

            # Metrics & Annotations
            status = "SEGREGATED" if matches < 10 else "CONNECTED"
            if matches == 0: status = "TOTAL EXCLUSION"
            
            plt.title(f"Strictness: {val} | Status: {status} | Matches: {matches}", fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
            # Visual Aid: The Search Radius
            circle = plt.Circle((0.3, 0.3), val, color='green', fill=False, linestyle='--', alpha=0.5, label='Search Radius')
            plt.gca().add_patch(circle)
            
            plt.legend(loc='upper left')
            plt.grid(True, linestyle=':', alpha=0.3)
            
            pdf.savefig()
            plt.close()

    print(f"Report saved successfully to {output_file}")

if __name__ == "__main__":
    generate_report()