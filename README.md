---
title: Binding Affinity Prediction
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: mit
---

# 💊 Drug Binding Affinity Prediction with GNNs + CNN + Cross-Attention & LLM Interpretation
<img width="1284" height="582" alt="image" src="https://github.com/user-attachments/assets/cb814000-5ae2-4967-8b9f-f25869dd5d53" />

This project is the implementation of the Deep Learning model to predict the **Binding Affinity ($pK_d$)** between drug candidates (ligand) and target proteins. The feature of that system is that it solves the "Black Box" problem in drug discovery field by presenting an **Explainable AI (XAI)** module powered by **Cross-Attention weights** and **LLM interpretation**, which allows researchers to visualize the active site of the ligand and which atoms play a vital role in the binding process.


## Architecture: The "Hybrid" Approach
The model uses a dual-encoder architecture with a Cross-Attention mechanism, mimicking the physical binding process:

<img width="3756" height="1797" alt="binding_affinity drawio" src="https://github.com/user-attachments/assets/1e510205-c9c2-468d-8372-2a8a0b45aae7" />

1.  **Ligand Encoder (Graph):**
     * **GAT (Graph Attention Network):** Treats atoms as nodes and bonds as edges. Uses 4 attention heads to capture complex chemical substructures.
2.  **Protein Encoder (Sequence):** 
     * **CNN** 1D Convolution to capture local protein structures. Considering the small size of the PDbind refined set, it outperformes complex architecture (Transformer).
3. **Cross-Attention Layer** Core feature of the project gives an understanding about the chemical bond between the ligands and the proteins, allows atoms of the ligand to look at the protein sequence, and 'bind' to specific regions of the protein sequence. It gives a chance to understand the relationship between different atoms of the ligand with different acids of the protein, specifically which atom interacts most with which amino acid.

## Results
We compared multiple architectures on the **PDBbind Refined** dataset. The Hybrid GAT+CNN with the Cross-Attention mechanism model achieved State-of-the-Art (SOTA) level performance for this scope. In conclusion, the CNN & Cross-Attention based model outperforms the Transformer based one.
| Model Architecture | RMSE |  MAE |  Pearson Correlation |
|--------------|---------------|---------------|---------------|
| GCN + Transformer for proteins | 1.5190 | 1.1957 | 0.6285  |
| GAT + Transformer for proteins | 1.5117  | 1.2074  | 0.6310  |
| GAT + 1 CNN for proteins + Cross-Attention | 1.3867  | 1.0947  | 0.7013  |


## Explainability (XAI)
The key moment is that the model does not give only a number, but an asnwer why it predicted that specific number.
1.  Extracts attention weights from the Cross-Attention layer.
2.  Identifies the **Top-15 atoms** responsible for binding process. (Got the atom number in the SMILES ligand sequence, name of atom, and the imprortance of that atom in the binding process (0 - min, 1 - max))
3.  Check the drug likeliness of the ligand according to the Lipinski's Rule of 5.
4.  Uses **Google Gemini API** to generate a chemical explanation of *why* these atoms are critical (e.g., hydrogen bonds, hydrophobic interactions).

---
*Created by Alex Sychov*
