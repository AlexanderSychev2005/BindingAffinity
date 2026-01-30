---
title: Binding Affinity Prediction
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: mit
---

# 💊 Drug Binding Affinity Prediction with GNNs & Cross-Attention
This project is the implementation of the Deep Learning model to predict the **Binding Affinity ($pK_d$)** between drug candidates (ligand) and target proteins. The feature of that system is that it solves the "Black Box" in drug discovery field by presenting an **Explainable AI (XAI)** module powered by **Cross-Attention weights** and **LLM interpretation**.

## Architecture: The "Hybrid" Approach
The model uses a dual-encoder architecture with a Cross-Attention mechanism:
1.  **Ligand Encoder (Graph):**
     * **GAT (Graph Attention Network):** Treats atoms as nodes and bonds as edges. Uses 4 attention heads to capture complex chemical substructures.
2.  **Protein Encoder (Sequence):** 
     * **CNN** 1D Convolution to capture local protein structures. Considering the small size of the PDbind refined set, it outperformes complex architecture (Transformer).
3. **Cross-Attention Layer** Makes the model to understand the chemical bond between the ligands and the proteins, allows atoms of the ligand to look at the protein sequence, and 'bind' to specific regions of the protein sequence, mimicking the physical binding process. It gives a chance to understand the relationship between different atoms of the ligand with different acids of the protein, specifically which atom interacts most with which amino acid.

## Explainability (XAI)
1.  Extracts attention weights from the Cross-Attention layer.
2.  Identifies the **Top-15 atoms** responsible for binding process. (Got the atom number in the SMILES ligand sequence, name of atom, and the imprortance of that atom in the binding process (0 - min, 1 - max))
3. Calculate the values for the ligand using the Lipinski's Rule of 5.
4.  Uses **Google Gemini API** to generate a chemical explanation of *why* these atoms are critical (e.g., hydrogen bonds, hydrophobic pockets).
