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

<img width="1673" height="734" alt="image" src="https://github.com/user-attachments/assets/62cd937a-e5df-4acd-ab1c-1158588d49ea" />

This project is the implementation of the Deep Learning model to predict the **Binding Affinity ($pK_d$)** between drug candidates (ligand) and target proteins. The feature of that system is that it solves the "Black Box" problem in drug discovery field by presenting an **Explainable AI (XAI)** module powered by **Cross-Attention weights** and **LLM interpretation**, which allows researchers to visualize the active site of the ligand and which atoms play a vital role in the binding process.


## Architecture: The "Hybrid" Approach
The model uses a dual-encoder architecture with a Cross-Attention mechanism, mimicking the physical binding process:

<img width="100%" alt="binding_affinity drawio" src="https://github.com/user-attachments/assets/1e510205-c9c2-468d-8372-2a8a0b45aae7" />

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
| **GAT + Deep CNN + Cross-Attention** | **1.3867** | **1.0947** | **0.7013** |


## Explainability (XAI)
The key moment is that the model does not give only a number, but an asnwer why it predicted that specific number.
1.  Extracts attention weights from the Cross-Attention layer.
2.  Identifies the **Top-15 atoms** responsible for binding process. (Got the atom number in the SMILES ligand sequence, name of atom, and the imprortance of that atom in the binding process (0 - min, 1 - max))
3.  Check the drug likeliness of the ligand according to the Lipinski's Rule of 5.
4.  Uses **Google Gemini API** to generate a chemical explanation of *why* these atoms are critical (e.g., hydrogen bonds, hydrophobic interactions).

## 🧪 Case Study: HIV-1 Protease Inhibitor (PDB: 6e9a)
To validate the model on high-complexity ligands, we tested it on a potent HIV-1 protease inhibitor (Darunavir analog, PDB: 6e9a).
* **Ligand: Sulfonamide-based inhibitor ($C_{29}H_{37}N_3O_7S$, MW 575.7 Da). SMILES:** `COc1ccc(S(=O)(=O)N(CC(C)C)C[C@@H](O)[C@H](Cc2ccccc2)NC(=O)O[C@@H]2C[C@@H]3NC(=O)O[C@@H]3C2)cc1`
* **Target:** HIV-1 Protease Chain A.

* **Molecule:** Sulfonamide-based inhibitor ($C_{29}H_{37}N_3O_7S$, MW 575.7 Da).
* **Predicted Affinity ($pK_d$):** `7.22` (Classified as **Strong Binder**)
* **Real Affinity:** High potency confirmed by PDB data.

The Cross-Attention mechanism identified the key pharmacophore features without prior knowledge:
1. **Polar Anchors (Oxygen #16, #34):** The model assigned high attention scores to the oxygen atoms. Chemically, these act as hydrogen bond acceptors, critical for anchoring the drug to the protein's backbone (Asp29/Asp30 residues).
2.  **Hydrophobic Core (Carbon #0):** The model highlighted the aromatic carbon in the terminal ring, which is essential for hydrophobic packing in the S2' pocket of the protease.
<img width="800" alt="Molecule Visualization" src="https://github.com/user-attachments/assets/245be900-41ff-44e9-b31a-69be6d42be8e" />

### 3. Top Critical Atoms
Below are the atoms with the highest attention weights contributed to the decision:

| Rank | Atom Index | Type | Attention Score | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| 1 | #57 | H | 1.000 | Hydrogen Bond Donor |
| 2 | #16 | O | 0.729 | **Sulfonamide Oxygen (Key Anchor)** |
| 3 | #0 | C | 0.676 | **Aromatic Ring (Hydrophobic)** |
| 4 | #34 | O | 0.581 | Ether Oxygen (H-bond Acceptor) |
| 5 | #22, #23 | C | ~0.600 | Hydrophobic Scaffold |

### 4. Drug-Likeness & Gemini Report
The system automatically generates a report to assist chemists:

#### 💊 Lipinski's Rule of 5 Analysis
* **Status:** Poor (2 violations) 🔴
* **Mass:** 575.68 Da (Violation: > 500)
* **H-Acceptors:** 11 (Violation: > 10)
* *Note: HIV protease inhibitors are often large molecules that break these rules but remain effective.*

#### 🤖 Google Gemini Analysis
>Affinity Analysis: The predicted binding affinity (pKd = 7.22) suggests moderate to strong binding for this ligand to the target protein. A pKd > 7 generally indicates a promising starting point for drug >discovery, implying significant interaction.
>
>Structural Basis: The highlighted atoms, particularly Oxygen (idx 16) and Nitrogen (implicit in the sulfonamide and carbamate groups), likely participate in hydrogen bonding as donors or acceptors. Aromatic >carbons (e.g., idx 0 for the methoxy-substituted phenyl ring) are key for pi-pi stacking interactions within the protein's binding pocket. The specific arrangement of these functional groups and their positions >relative to the protein are critical for recognition.
>
>Drug-Likeness: With 2 Lipinski violations, this molecule exhibits poor drug-likeness, particularly concerning oral bioavailability. It may face challenges with absorption and membrane permeability.
>
>Conclusion: While the binding affinity is encouraging, the poor drug-likeness warrants caution. Further structural optimization to improve Lipinski compliance would be essential before proceeding with this >molecule as a drug candidate.


---
*Created by Alex Sychov*
