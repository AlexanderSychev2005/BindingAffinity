import torch
import numpy as np
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import py3Dmol
from jinja2 import Environment, FileSystemLoader

# Убираем лишние импорты (nglview тут больше не нужен для standalone HTML)
from dataset import get_atom_features, get_protein_features
from model_attention import BindingAffinityModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Обновите путь, если нужно
MODEL_PATH = "runs/experiment_attention20260124_104439_optuna/models/model_ep041_mse1.9153.pth"

GAT_HEADS = 2
HIDDEN_CHANNELS = 256


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
    protein_sequence_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data]).to(DEVICE)
    num_features = x.shape[1]

    # Model
    model = BindingAffinityModel(num_features, hidden_channels=HIDDEN_CHANNELS, gat_heads=GAT_HEADS).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.batch, protein_sequence_tensor)
        attention_weights = model.cross_attention.last_attention_weights[0]

    # Attention processing
    real_prot_len = len([t for t in tokens if t != 0])
    importance = attention_weights[:, :real_prot_len].max(dim=1).values.cpu().numpy()

    if importance.max() > 0:
        importance = (importance - importance.min()) / (importance.max() - importance.min())

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
        "bad_params": ", ".join(bad_params) if bad_params else "None"
    }


def get_py3dmol_view(mol, importance):
    view = py3Dmol.view(width="100%", height="600px")
    view.addModel(Chem.MolToMolBlock(mol), "sdf")
    view.setBackgroundColor('white')

    view.setStyle({}, {
        'stick': {'radius': 0.15},
        'sphere': {'scale': 0.25}
    })

    indices_sorted = np.argsort(importance)[::-1]
    top_indices = set(indices_sorted[:15])

    conf = mol.GetConformer()

    for i, val in enumerate(importance):
        if i in top_indices:
            pos = conf.GetAtomPosition(i)
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            label_text = f"{i}:{symbol}:{val:.2f}"

            view.addLabel(label_text, {
                'position': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                'fontSize': 14,
                'fontColor': 'white',
                'backgroundColor': 'black',
                'backgroundOpacity': 0.7,
                'borderThickness': 0,
                'inFront': True,
                'showBackground': True
            })
    view.zoomTo()
    return view


def save_standalone_ngl_html(mol, importance, filepath):
    pdb_block = Chem.MolToPDBBlock(mol)
    mol_pdb = Chem.MolFromPDBBlock(pdb_block, removeHs=False)

    for i, atom in enumerate(mol_pdb.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        if info:
            info.SetTempFactor(float(importance[i]) * 100)

    final_pdb_block = Chem.MolToPDBBlock(mol_pdb)
    final_pdb_block = final_pdb_block.replace("`", "\\`")


    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('ngl_view.html')

    rendered_html = template.render(pdb_block=final_pdb_block)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(rendered_html)