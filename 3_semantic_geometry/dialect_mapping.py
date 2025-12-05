"""
SEMANTIC GEOMETRY: Dialect Mapping (Standard Greek vs Cretan)
-------------------------------------------------------------
This script investigates whether modern NLP models (SpaCy) contain latent 
knowledge of Greek dialects.

Output: 
1. PCA Map (Visualizing distance)
2. Similarity Heatmap (Quantifying alignment for business reporting)

Author: [Your Name]
"""

import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # <--- Η νέα βιβλιοθήκη για το Heatmap
from sklearn.decomposition import PCA
import os

# --- 1. ΦΟΡΤΩΣΗ ΜΟΝΤΕΛΟΥ ---
def load_greek_model():
    print("--- Loading Greek Language Model (el_core_news_md)... ---")
    try:
        nlp = spacy.load("el_core_news_md")
        print("Model loaded successfully!")
        return nlp
    except OSError:
        print("Error: Model 'el_core_news_md' not found.")
        print("Please run in terminal: python -m spacy download el_core_news_md")
        return None

# --- 2. ΤΟ PCA ΓΡΑΦΗΜΑ (ΟΠΤΙΚΟΠΟΙΗΣΗ) ---
def analyze_dialect_pca(nlp, pairs, output_file="visuals/cretan_dialect_map.png"):
    print(f"\n--- Generating PCA Map: {output_file} ---")
    
    found_vectors = []
    found_labels = []

    for greek, cretan in pairs:
        token_greek = nlp(greek)
        token_cretan = nlp(cretan)
        
        if token_greek.has_vector and token_cretan.has_vector:
            found_vectors.append(token_greek.vector)
            found_labels.append(greek + " (GR)")
            found_vectors.append(token_cretan.vector)
            found_labels.append(cretan + " (CR)")

    if len(found_vectors) < 3:
        print("Not enough vectors to plot.")
        return

    pca = PCA(n_components=2)
    result = pca.fit_transform(np.array(found_vectors))
    
    plt.figure(figsize=(12, 10))
    
    for i in range(len(result)):
        x, y = result[i]
        label = found_labels[i]
        color = 'royalblue' if '(GR)' in label else 'forestgreen'
        
        plt.scatter(x, y, c=color, s=120, edgecolors='k', alpha=0.8)
        plt.annotate(label, (x, y), xytext=(6, 6), textcoords='offset points', 
                     weight='bold', color=color, fontsize=11)

    plt.title("Dialectometry: Standard Greek vs Cretan Semantic Space", fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Chart saved to: {output_file}")
    plt.close()

# --- 3. TO HEATMAP (ETAIΡΙΚΗ ΑΝΑΦΟΡΑ) ---
def plot_similarity_heatmap(nlp, pairs, output_file="visuals/dialect_heatmap.png"):
    print(f"\n--- Generating Corporate Heatmap: {output_file} ---")
    
    greek_words = [p[0] for p in pairs]
    cretan_words = [p[1] for p in pairs]
    
    # Φτιάχνουμε έναν πίνακα (Matrix) με τις ομοιότητες
    similarity_matrix = np.zeros((len(pairs), len(pairs)))
    
    for i, g_word in enumerate(greek_words):
        for j, c_word in enumerate(cretan_words):
            token_g = nlp(g_word)
            token_c = nlp(c_word)
            
            if token_g.has_vector and token_c.has_vector:
                similarity_matrix[i, j] = token_g.similarity(token_c)
            else:
                similarity_matrix[i, j] = 0.0

    plt.figure(figsize=(10, 8))
    
    # Σχεδίαση Heatmap με το Seaborn
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=cretan_words, yticklabels=greek_words)
    
    plt.title("Semantic Alignment Matrix: Standard Greek vs Cretan", fontsize=14)
    plt.xlabel("Cretan Dialect")
    plt.ylabel("Standard Greek")
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Heatmap saved to: {output_file}")
    plt.close()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    nlp_model = load_greek_model()
    
    if nlp_model:
        # Ορίζουμε τα ζευγάρια ΕΔΩ, μία φορά, για να τα χρησιμοποιούν όλες οι συναρτήσεις
        dialect_pairs = [
            ('εδώ', 'επαέ'),
            ('κρυώνω', 'εργώ'),
            ('όλοι', 'ούλοι'),
            ('τρελός', 'κουζουλός'),
            ('παιδί', 'κοπέλι'),
            ('φίλος', 'κουμπάρος'), 
            ('κοιτάζω', 'θωρώ'),
            ('μιλώ', 'συντυχάνω'),
            ('καρέκλα', 'κάθικλα'),
            ('αυτί', 'ατζί')
        ]
        
        # 1. Κάλεσμα της συνάρτησης για το PCA (Οι τελείες)
        analyze_dialect_pca(nlp_model, dialect_pairs)
        
        # 2. Κάλεσμα της συνάρτησης για το Heatmap (Ο πίνακας)
        plot_similarity_heatmap(nlp_model, dialect_pairs)
        
        print("\nAll tasks completed successfully.")