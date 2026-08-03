"""Joint ligand+pocket graph dataset (GIGN-style featurization).

Each complex becomes a single PyG Data object: ligand atoms and pocket
atoms share one node set, with two edge types -
`edge_index_intra` (chemical bonds, ligand and pocket separately) and
`edge_index_inter` (cross ligand<->pocket atoms within `dist_threshold`
Angstrom). Graphs are cached to disk as .pt files since the RDKit parsing
is the slow part and doesn't change across epochs.
"""

import os

import pandas as pd
import torch
from rdkit import Chem, rdBase
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

rdBase.DisableLog("rdApp.*")

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Unknown"]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
VALENCES = [0, 1, 2, 3, 4, 5, 6]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS = [0, 1, 2, 3, 4]

NODE_DIM = len(ATOM_SYMBOLS) + len(DEGREES) + len(VALENCES) + len(HYBRIDIZATIONS) + 1 + len(NUM_HS)


def _one_hot(value, allowed):
    return [value == a for a in allowed]


def _one_hot_unk(value, allowed):
    if value not in allowed:
        value = allowed[-1]
    return [value == a for a in allowed]


def atom_features(atom):
    feats = (
        _one_hot_unk(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_hot_unk(atom.GetDegree(), DEGREES)
        + _one_hot_unk(atom.GetImplicitValence(), VALENCES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [atom.GetIsAromatic()]
        + _one_hot_unk(atom.GetTotalNumHs(), NUM_HS)
    )
    return torch.tensor(feats, dtype=torch.float)


def mol_to_graph(mol):
    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, pos, edge_index


def load_ligand(complex_dir, pdb_id):
    sdf_path = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
    if os.path.exists(sdf_path):
        mol = next((m for m in Chem.SDMolSupplier(sdf_path, sanitize=True) if m is not None), None)
        if mol is not None:
            return mol
    mol2_path = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
    if os.path.exists(mol2_path):
        return Chem.MolFromMol2File(mol2_path, sanitize=True)
    return None


def load_pocket(complex_dir, pdb_id):
    pocket_path = os.path.join(complex_dir, f"{pdb_id}_pocket.pdb")
    if not os.path.exists(pocket_path):
        return None
    return Chem.MolFromPDBFile(pocket_path, removeHs=True, sanitize=True)


def restrict_pocket_to_radius(pocket, ligand_pos, radius):
    """Keep only whole residues with >=1 atom within `radius` of the ligand.

    The pre-extracted *_pocket.pdb files on disk were cut at a wider radius
    (~8A) than the 5A GIGN uses. This re-selects by residue (not by atom, to
    avoid chopping amino acids in half) on top of the already-loaded pocket,
    so no PyMOL re-extraction is needed.
    """
    pocket_pos = torch.tensor(pocket.GetConformer().GetPositions(), dtype=torch.float)
    min_dist = torch.cdist(pocket_pos, ligand_pos).min(dim=1).values

    res_key = [
        (info.GetChainId(), info.GetResidueNumber()) if (info := atom.GetPDBResidueInfo()) else (None, atom.GetIdx())
        for atom in pocket.GetAtoms()
    ]
    keep_residues = {res_key[i] for i in range(len(res_key)) if min_dist[i] < radius}
    drop_atoms = sorted((i for i, k in enumerate(res_key) if k not in keep_residues), reverse=True)

    rw = Chem.RWMol(pocket)
    for idx in drop_atoms:
        rw.RemoveAtom(idx)
    return rw.GetMol()


def build_complex_graph(ligand, pocket, affinity, dist_threshold=5.0, pocket_radius=5.0):
    x_l, pos_l, edge_l = mol_to_graph(ligand)
    if pocket_radius is not None:
        pocket = restrict_pocket_to_radius(pocket, pos_l, pocket_radius)
    x_p, pos_p, edge_p = mol_to_graph(pocket)
    n_l = x_l.size(0)

    x = torch.cat([x_l, x_p], dim=0)
    pos = torch.cat([pos_l, pos_p], dim=0)
    edge_index_intra = torch.cat([edge_l, edge_p + n_l], dim=1)

    cross_dist = torch.cdist(pos_l, pos_p)
    lig_idx, pocket_idx = (cross_dist < dist_threshold).nonzero(as_tuple=True)
    pocket_idx = pocket_idx + n_l
    edge_index_inter = torch.stack(
        [torch.cat([lig_idx, pocket_idx]), torch.cat([pocket_idx, lig_idx])]
    )

    return Data(
        x=x, pos=pos,
        edge_index_intra=edge_index_intra,
        edge_index_inter=edge_index_inter,
        y=torch.tensor([affinity], dtype=torch.float),
    )


class PocketLigandDataset(Dataset):
    def __init__(self, csv_path, data_dir, cache_dir, dist_threshold=5.0, pocket_radius=5.0):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.dist_threshold = dist_threshold
        self.pocket_radius = pocket_radius
        os.makedirs(cache_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        self.entries = list(df[["pdb_id", "affinity"]].itertuples(index=False, name=None))

    def __len__(self):
        return len(self.entries)

    def _cache_path(self, pdb_id):
        return os.path.join(self.cache_dir, f"{pdb_id}_{self.dist_threshold}A_pocket{self.pocket_radius}A.pt")

    def __getitem__(self, idx):
        pdb_id, affinity = self.entries[idx]
        cache_path = self._cache_path(pdb_id)

        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=False)

        complex_dir = os.path.join(self.data_dir, pdb_id)
        ligand = load_ligand(complex_dir, pdb_id)
        pocket = load_pocket(complex_dir, pdb_id)
        if ligand is None or pocket is None:
            return None

        data = build_complex_graph(ligand, pocket, affinity, self.dist_threshold, self.pocket_radius)
        torch.save(data, cache_path)
        return data

    @staticmethod
    def collate(batch):
        batch = [d for d in batch if d is not None]
        return Batch.from_data_list(batch)
