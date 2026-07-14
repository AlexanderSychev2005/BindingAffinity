# Binding Affinity Prediction

This project is a state-of-the-art Deep Learning system for predicting the binding affinity (pKd) between protein targets and drug-like ligand molecules. 

It leverages a hybrid architecture combining a 3D Equivariant Graph Neural Network (EGNN) for the ligand's precise geometry and a Deep 1D Convolutional Neural Network (CNN) acting over ESM-2 embeddings for the protein sequence.

## Architecture

Our model fuses two distinct representations:

1. **Ligand Tower (3D EGNN):** 
   - Uses RDKit (ETKDG) to generate 3D coordinates for all ligand atoms, including hydrogens.
   - A multi-layer Equivariant Graph Convolutional Network (EGNN) processes the ligand graph, updating both node features and 3D coordinates to capture rigid spatial structures.
   - Coordinate Noise augmentation is used to prevent overfitting.

2. **Protein Tower (Deep 1D CNN + ESM-2):**
   - The protein is embedded using a 35-million parameter ESM-2 language model (`facebook/esm2_t12_35M_UR50D`).
   - A deep 1D CNN processes the 480-dimensional ESM-2 sequence embeddings. This CNN is exceptionally capable of extracting side-chain chemistry and sequence-level motifs.

3. **Fusion (Cross-Attention):**
   - The structural ligand representation and the chemical protein representation are fused using a Multi-Head Cross-Attention Layer. This allows every atom of the ligand to attend to every amino acid residue of the protein.

> **TODO FOR USER: [Insert Architecture Diagram Here]**
> *Please draw a diagram showing: 1) Ligand SMILES -> RDKit 3D -> EGNN Tower, 2) Protein Sequence -> ESM-2 (35M) -> Deep CNN Tower, 3) Both towers feeding into a Cross-Attention block -> MLP -> Binding Affinity Score. Use tools like draw.io or Excalidraw.*

## Directory Structure

```
BindingAffinity/
├── training/            # All model training scripts and data prep
│   ├── scripts/         # Scripts for Colab execution and caching (e.g. train_3d_colab.py)
│   └── src/             # Core ML source code (dataset, models, utils)
├── frontend/            # FastAPI Web Interface
│   ├── app.py           # Main backend entrypoint
│   ├── templates/       # HTML UI
│   └── static/          # CSS and assets
├── notebooks/           # Jupyter Notebooks for exploration
└── README.md            # Documentation
```

## Quick Start (Training)

1. Pre-compute ESM-2 embeddings locally:
   ```bash
   python training/scripts/prepare_esm_cache.py
   ```
2. Archive the `refined-set` folder and upload to Colab.
3. Run the training script on Colab (make sure paths match):
   ```bash
   !python /content/drive/MyDrive/BindingAffinity/training/scripts/train_3d_colab.py \
       --data-dir /content/refined-set \
       --csv-file /content/pdbbind_refined_dataset.csv \
       --log-dir /content/runs \
       --fp16 \
       --batch-size 64
   ```

## Web Interface

Run the FastAPI frontend server locally:
```bash
uvicorn frontend.app:app --reload
```
Access the application at `http://localhost:8000`.

> **TODO FOR USER: [Insert Pipeline Diagram Here]**
> *Please draw a flowchart of the user journey: User inputs SMILES & Sequence -> Backend runs Inference -> AI generates explanation -> Results displayed in UI with py3Dmol visualization.*
