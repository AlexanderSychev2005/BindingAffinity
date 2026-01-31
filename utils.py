import torch
import numpy as np
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import py3Dmol
from jinja2 import Environment, FileSystemLoader
from google import genai
from decouple import config
import time

GEMINI_API_KEY = config("GEMINI_API_KEY")


from dataset import get_atom_features, get_protein_features
from model_attention import BindingAffinityModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_PATH = "runs/experiment_attention20260124_104439_optuna/models/model_ep041_mse1.9153.pth"
#
GAT_HEADS = 2
HIDDEN_CHANNELS = 256

MODEL_PATH = "models/model_ep041_attention_mse1.9153.pth"
# MODEL_PATH = "models/model_ep028_weighted_loss6.7715.pth"
# GAT_HEADS = 4
# HIDDEN_CHANNELS = 128


def get_inference_data(ligand_smiles, protein_sequence, model_path=MODEL_PATH):
    # Prepare ligand
    mol = Chem.MolFromSmiles(ligand_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    # Graph data
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_features), dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Protein data
    tokens = [get_protein_features(c) for c in protein_sequence]
    if len(tokens) > 1200:
        tokens = tokens[:1200]
    else:
        tokens.extend([0] * (1200 - len(tokens)))
    protein_sequence_tensor = (
        torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    )

    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data]).to(DEVICE)
    num_features = x.shape[1]

    # Model
    model = BindingAffinityModel(
        num_features, hidden_channels=HIDDEN_CHANNELS, gat_heads=GAT_HEADS
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=False)
    )
    model.eval()

    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.batch, protein_sequence_tensor)
        attention_weights = model.cross_attention.last_attention_weights[0]

    # Attention processing
    real_prot_len = len([t for t in tokens if t != 0])
    importance = attention_weights[:, :real_prot_len].max(dim=1).values.cpu().numpy()

    if importance.max() > 0:
        importance = (importance - importance.min()) / (
            importance.max() - importance.min()
        )

    importance[importance < 0.01] = 0
    return mol, importance, pred.item()


def get_lipinski_properties(mol):
    mw = Descriptors.MolWt(mol)
    hba = Descriptors.NOCount(mol)
    hbd = Descriptors.NHOHCount(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    violations = 0
    bad_params = []
    if mw > 500:
        violations += 1
        bad_params.append("Mass > 500")
    if logp > 5:
        violations += 1
        bad_params.append("LogP > 5")
    if hbd > 5:
        violations += 1
        bad_params.append("H-Donors > 5")
    if hba > 10:
        violations += 1
        bad_params.append("H-Acceptors > 10")

    if violations == 0:
        status = "Excellent (Drug-like) 🟢"
        css_class = "success"
    elif violations == 1:
        status = "Acceptable (1 violation) 🟡"
        css_class = "warning"
    else:
        status = f"Poor ({violations} violations) 🔴"
        css_class = "danger"

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "TPSA": round(tpsa, 2),
        "violations": violations,
        "status_text": status,
        "css_class": css_class,
        "bad_params": ", ".join(bad_params) if bad_params else "None",
    }


def get_py3dmol_view(mol, importance):
    view = py3Dmol.view(width="100%", height="600px")
    view.addModel(Chem.MolToMolBlock(mol), "sdf")
    view.setBackgroundColor("white")

    view.setStyle({}, {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})

    indices_sorted = np.argsort(importance)[::-1]
    top_indices = set(indices_sorted[:15])

    conf = mol.GetConformer()

    for i, val in enumerate(importance):
        if i in top_indices:
            pos = conf.GetAtomPosition(i)
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            label_text = f"{i}:{symbol}:{val:.2f}"

            view.addLabel(
                label_text,
                {
                    "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "fontSize": 14,
                    "fontColor": "white",
                    "backgroundColor": "black",
                    "backgroundOpacity": 0.7,
                    "borderThickness": 0,
                    "inFront": True,
                    "showBackground": True,
                },
            )
    view.zoomTo()
    return view


def save_standalone_ngl_html(mol, importance, filepath):
    pdb_block = Chem.MolToPDBBlock(mol)
    mol_pdb = Chem.MolFromPDBBlock(pdb_block, removeHs=False)

    for i, atom in enumerate(mol_pdb.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        if info:
            info.SetTempFactor(float(importance[i]))

    final_pdb_block = Chem.MolToPDBBlock(mol_pdb)
    final_pdb_block = final_pdb_block.replace("`", "\\`")

    indices_sorted = np.argsort(importance)[::-1]
    top_indices = indices_sorted[:15]

    selection_list = [str(i) for i in top_indices]
    selection_str = "@" + ",".join(selection_list)

    if not selection_list:
        selection_str = "@-1"

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("ngl_view.html")

    rendered_html = template.render(
        pdb_block=final_pdb_block, selection_str=selection_str
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(rendered_html)


def get_gemini_explanation(
    ligand_smiles, protein_sequence, affinity, top_atoms, lipinski
):
    if not GEMINI_API_KEY:
        return "<p class='text-warning'>API Key for Gemini not found. Please set GOOGLE_API_KEY environment variable.</p>"

    # Forming a list of top important atoms for a prompt
    atoms_desc = ", ".join(
        [f"{a['symbol']}(idx {a['id']}, score {a['score']})" for a in top_atoms[:10]]
    )

    # Cut a protein to not spend too many tokens
    prot_short = (
        protein_sequence[:100] + "..."
        if len(protein_sequence) > 100
        else protein_sequence
    )

    prompt = f"""
    You are an expert Computational Chemist and Drug Discovery Scientist.
    Analyze the following interaction results between a Ligand and a Protein.
    
    **Data:**
    1. **Ligand (SMILES):** `{ligand_smiles}`
    2. **Target Protein (Start):** `{prot_short}`
    3. **Predicted Binding Affinity (pKd):** {affinity} (Note: >7 is usually good, <5 is weak).
    4. **Top Active Atoms (Attention Weights):** {atoms_desc}. These atoms had the highest attention scores in the Graph Neural Network with attention.
    5. **Lipinski Properties:** {lipinski['status_text']} (Violations: {lipinski['violations']}).
    
    **Task:**
    Write a concise, professional scientific summary (in HTML format, use <p>, <ul>, <li>, <b>).
    Cover these points:
    1. **Affinity Analysis:** Is the binding strong? What does a pKd of {affinity} imply for a drug candidate?
    2. **Structural Basis:** Why might the model have focused on the atoms listed above (e.g., Nitrogen/Oxygen often act as H-bond donors/acceptors, Rings for stacking)?
    3. **Drug-Likeness:** Comment on the Lipinski status. Is it suitable for oral administration?
    4. **Conclusion:** Verdict on whether to proceed with this molecule.
    Keep it relatively short (max 150 words). Do not include markdown code blocks (```html), just return the raw HTML tags.
    """
    max_retries = 3
    client = genai.Client(api_key=GEMINI_API_KEY)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            return response.text

        except Exception as e:
            error_msg = str(e).lower()

            if "503" in error_msg or "overloaded" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    print(f"Gemini overloaded, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            return f"<p class='text-danger'>Error generating explanation: {str(e)}</p>"

    return "<p class='text-danger'>Error: Gemini unavailable after retries.</p>"
