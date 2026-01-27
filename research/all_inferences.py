import datetime
import os.path
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import nglview as nv
import py3Dmol
from nglview import write_html


import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from dataset import get_atom_features, get_protein_features
from model_attention import BindingAffinityModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = (
    "runs/experiment_attention20260124_104439_optuna/models/model_ep041_mse1.9153.pth"
)

GAT_HEADS = 2
HIDDEN_CHANNELS = 256


def get_inference_data(ligand_smiles, protein_sequence, model_path):
    """
    Returns:
        - mol: RDKit molecule object with 3D coordinates
        - importance: list of importance scores for each atom
        - predicted_affinity: predicted binding affinity value
    """
    # Prepare ligand molecule with geometry RDKit
    mol = Chem.MolFromSmiles(ligand_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    # Graph data PyTorch
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_features), dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    tokens = [get_protein_features(c) for c in protein_sequence]
    if len(tokens) > 1200:
        tokens = tokens[:1200]
    else:
        tokens.extend([0] * (1200 - len(tokens)))
    protein_sequence = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data]).to(DEVICE)
    num_features = x.shape[1]

    # Model loading
    model = BindingAffinityModel(
        num_features, hidden_channels=HIDDEN_CHANNELS, gat_heads=GAT_HEADS
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Prediction
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.batch, protein_sequence)
        attention_weights = model.cross_attention.last_attention_weights[0]

    # Attention importance, Max + Normalize
    real_prot_len = len([t for t in tokens if t != 0])
    importance = attention_weights[:, :real_prot_len].max(dim=1).values.cpu().numpy()

    # Normalize to [0, 1]
    if importance.max() > 0:
        importance = (importance - importance.min()) / (
            importance.max() - importance.min()
        )

    # Noise reduction
    importance[importance < 0.01] = 0
    return mol, importance, pred.item()


def print_atom_scores(mol, importance):
    print("Atom importance scores:")

    atom_data = []
    for i, score in enumerate(importance):
        if score > 0.1:
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            atom_data.append((i, symbol, score))

    atom_data.sort(key=lambda x: x[2], reverse=True)

    for idx, symbol, score in atom_data:
        fire = "🔥" if score > 0.8 else ("✨" if score > 0.5 else "")
        print(f"Atom {idx} ({symbol}): Importance = {score:.3f} {fire}")


def get_py3dmol(mol, importance, score):

    view = py3Dmol.view(width=1000, height=800)
    view.addModel(Chem.MolToMolBlock(mol), "sdf")
    view.setBackgroundColor("white")

    # 1. БАЗОВЫЙ СТИЛЬ (ГРУНТОВКА)
    # Задаем единый размер для всей молекулы сразу
    # scale: 0.25 — оптимальный средний размер
    view.setStyle(
        {},
        {
            "stick": {"color": "#cccccc", "radius": 0.1},
            "sphere": {"color": "#cccccc", "scale": 0.25},
        },
    )

    red_atoms = []
    orange_atoms = []
    blue_atoms = []

    indices_sorted = np.argsort(importance)[::-1]
    top_indices = set(indices_sorted[:15])
    labels_to_add = []

    conf = mol.GetConformer()

    # 2. СОРТИРОВКА (ТОЛЬКО ЦВЕТА)
    for i, val in enumerate(importance):
        if val >= 0.70:
            red_atoms.append(i)
        elif val >= 0.55:
            orange_atoms.append(i)
        elif val >= 0.40:
            blue_atoms.append(i)

        if i in top_indices and val > 0.1:
            pos = conf.GetAtomPosition(i)
            symbol = mol.GetAtomWithIdx(i).GetSymbol()
            labels_to_add.append(
                {
                    "text": f"{i}:{symbol}:{val:.2f}",
                    "pos": {"x": pos.x, "y": pos.y, "z": pos.z},
                }
            )

    # 3. ПРИМЕНЕНИЕ СТИЛЕЙ
    # Обрати внимание: scale везде 0.25 (или 0.28, чтобы чуть выделить цветные)
    # Мы меняем ТОЛЬКО ЦВЕТ.

    if red_atoms:
        view.addStyle(
            {"serial": red_atoms},
            {
                "sphere": {"color": "#FF0000", "scale": 0.28},
                "stick": {"color": "#FF0000", "radius": 0.12},
            },
        )

    if orange_atoms:
        view.addStyle(
            {"serial": orange_atoms},
            {
                "sphere": {"color": "#FF8C00", "scale": 0.28},
                "stick": {"color": "#FF8C00", "radius": 0.12},
            },
        )

    if blue_atoms:
        view.addStyle(
            {"serial": blue_atoms}, {"sphere": {"color": "#7777FF", "scale": 0.28}}
        )

    # 4. МЕТКИ
    for label in labels_to_add:
        view.addLabel(
            label["text"],
            {
                "position": label["pos"],
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
    view.addLabel(
        f"Predicted pKd: {float(score):.2f}",
        {
            "position": {"x": -5, "y": 10, "z": 0},
            "backgroundColor": "black",
            "fontColor": "white",
        },
    )

    return view


def get_ngl(mol, importance):
    pdb_temp = Chem.MolToPDBBlock(mol)
    mol_pdb = Chem.MolFromPDBBlock(pdb_temp, removeHs=False)

    for i, atom in enumerate(mol_pdb.GetAtoms()):
        info = atom.GetPDBResidueInfo()
        if info:
            val = float(importance[i] * 100.0)
            info.SetTempFactor(val)
    final_pdb_block = Chem.MolToPDBBlock(mol_pdb)
    structure = nv.TextStructure(final_pdb_block, ext="pdb")
    view = nv.NGLWidget(structure)
    view.clear_representations()

    view.add_representation(
        "ball+stick",
        colorScheme="bfactor",
        colorScale=["blue", "white", "red"],
        colorDomain=[10, 80],
        radiusScale=1.0,
    )

    indices_sorted = np.argsort(importance)[::-1]
    top_indices = indices_sorted[:15]

    selection_str = "@" + ",".join(map(str, top_indices))
    view.add_representation(
        "label",
        selection=selection_str,  # Подписываем только избранных
        labelType="atomindex",  # Показывать Индекс (0, 1, 2...)
        color="black",  # Черный текст
        radius=2.0,  # Размер шрифта (попробуйте 1.5 - 3.0)
        zOffset=1.0,
    )  # Чуть сдвинуть к камере

    view.center()
    return view


if __name__ == "__main__":
    smiles = "COc1ccc(S(=O)(=O)N(CC(C)C)C[C@@H](O)[C@H](Cc2ccccc2)NC(=O)O[C@@H]2C[C@@H]3NC(=O)O[C@@H]3C2)cc1"
    protein = "PQITLWKRPLVTIKIGGQLKEALLDTGADDTVIEEMSLPGRWKPKMIGGIGGFIKVRQYDQIIIEIAGHKAIGTVLVGPTPVNIIGRNLLTQIGATLNF"
    affinity = 11.92

    file_name_py3dmol = "html_results/py3dmol_result.html"
    file_name_ngl = "html_results/ngl_result.html"

    mol, importance, score = get_inference_data(smiles, protein, MODEL_PATH)
    print_atom_scores(mol, importance)
    py3dmol_view = get_py3dmol(mol, importance, score)
    py3dmol_view.write_html(file_name_py3dmol)

    ngl_widget = get_ngl(mol, importance)
    nv.write_html(file_name_ngl, ngl_widget)
