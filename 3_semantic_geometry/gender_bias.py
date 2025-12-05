"""
BIAS ANALYSIS: The Geometry of Gender Stereotypes
------------------------------------------------
This script demonstrates how social biases are encoded in high-dimensional
vector spaces (Word Embeddings). using the GloVe model.

It performs two tasks:
1. Vector Arithmetic: Solves analogies like "Man is to Programmer as Woman is to X".
2. PCA Visualization: Maps gender-neutral professions onto a 2D gender axis.

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api
import os

def load_model():
    """Loads the GloVe model (100 dimensions) from gensim-data."""
    print("--- Loading GloVe Model (this may take a moment)... ---")
    try:
        model = api.load("glove-wiki-gigaword-100")
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_analogy_test(model):
    """Performs vector arithmetic to reveal bias in analogies."""
    print("\n--- 1. Vector Arithmetic: The Bias Test ---")
    
    tests = [
        ('man', 'doctor', 'woman'),
        ('man', 'programmer', 'woman'),
        ('man', 'boss', 'woman'),
        ('woman', 'homemaker', 'man') # The reverse test
    ]

    for (a, b, c) in tests:
        # Equation: A is to B as C is to ?  =>  B - A + C = ?
        try:
            result = model.most_similar(positive=[b, c], negative=[a], topn=1)
            print(f"{a.capitalize()} : {b.capitalize()}  ->  {c.capitalize()} : {result[0][0].upper()}")
        except KeyError as e:
            print(f"Word not found in vocabulary: {e}")

def plot_gender_profession_bias(model, output_file="visuals/gender_bias_map.png"):
    """
    Creates a PCA scatter plot showing how professions cluster 
    around the 'He' vs 'She' vector space.
    """
    print("\n--- 2. Generating Visualization: The 'Gengression' Chart ---")
    
    # Define anchor words (Gender) and target words (Professions)
    gender_anchors = ['he', 'she', 'man', 'woman']
    professions = [
        'mechanic', 'nurse', 'engineer', 'teacher', 
        'architect', 'receptionist', 'boss', 'hairdresser',
        'captain', 'nanny', 'physicist', 'maid'
    ]
    
    all_words = gender_anchors + professions
    
    # Extract vectors
    valid_words = [w for w in all_words if w in model]
    vectors = [model[w] for w in valid_words]
    
    if len(vectors) < 5:
        print("Not enough words found in model to plot.")
        return

    # Dimensionality Reduction (100D -> 2D)
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    
    # Plotting
    plt.figure(figsize=(12, 9))
    
    # Plot Gender Anchors (Red)
    # Assuming first 4 words are the gender anchors
    plt.scatter(result[:4, 0], result[:4, 1], c='crimson', s=150, edgecolors='k', label='Gender Pole')
    
    # Plot Professions (Blue)
    plt.scatter(result[4:, 0], result[4:, 1], c='royalblue', s=100, edgecolors='k', label='Profession')
    
    # Annotate words
    for i, word in enumerate(valid_words):
        # Add some offset to text so it doesn't overlap the dot
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), 
                     xytext=(6, 4), textcoords='offset points', 
                     fontsize=11, weight='bold', alpha=0.9)
    
    plt.title("Semantic Bias: Mapping Professions on the Gender Axis", fontsize=16)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Check if directory exists, if not create it
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    plt.savefig(output_file)
    print(f"Chart saved to: {output_file}")
    plt.close() # Close plot to free memory

if __name__ == "__main__":
    # Main Execution Flow
    model = load_model()
    
    if model:
        run_analogy_test(model)
        plot_gender_profession_bias(model)
        # --- SANITY CHECK: MATHEMATICAL PROOF ---
        print("\n--- 3. Mathematical Proof (Cosine Similarity) ---")
        word = 'nurse'
        sim_she = model.similarity(word, 'she')
        sim_he = model.similarity(word, 'he')
        
        print(f"Ομοιότητα '{word}' με 'she': {sim_she:.4f}")
        print(f"Ομοιότητα '{word}' με 'he':  {sim_he:.4f}")
        
        if sim_she > sim_he:
            print("✅ ΣΥΜΠΕΡΑΣΜΑ: Το μοντέλο συνδέει το 'nurse' περισσότερο με γυναίκες.")
        else:
            print("❓ ΣΥΜΠΕΡΑΣΜΑ: Κάτι περίεργο συμβαίνει.")

        word2 = 'captain'
        sim_she2 = model.similarity(word2, 'she')
        sim_he2 = model.similarity(word2, 'he')
        print(f"\nΟμοιότητα '{word2}' με 'she': {sim_she2:.4f}")
        print(f"Ομοιότητα '{word2}' με 'he':  {sim_he2:.4f}")
        print("\nAnalysis Complete.")