import os
import argparse
from tqdm import tqdm
import warnings
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.spatial import cKDTree
from rdkit import Chem

warnings.simplefilter('ignore', PDBConstructionWarning)

class ResidueSelect(Select):
    def __init__(self, residue_ids):
        self.residue_ids = residue_ids
    
    def accept_residue(self, residue):
        return residue.get_full_id() in self.residue_ids

def get_ligand_coords(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
    coords = []
    for mol in supplier:
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            break # just take the first molecule
    return coords

def extract_pocket(pdb_id, data_dir, radius=10.0):
    folder = os.path.join(data_dir, pdb_id)
    prot_path = os.path.join(folder, f"{pdb_id}_protein.pdb")
    sdf_path = os.path.join(folder, f"{pdb_id}_ligand.sdf")
    mol2_path = os.path.join(folder, f"{pdb_id}_ligand.mol2")
    out_path = os.path.join(folder, f"{pdb_id}_pocket.pdb")

    if os.path.exists(out_path):
        return True

    if not os.path.exists(prot_path):
        return False

    lig_coords = []
    if os.path.exists(sdf_path):
        lig_coords = get_ligand_coords(sdf_path)
    
    # fallback to mol2 if sdf failed or missing (though RDKit mol2 parser is brittle, SDF is better)
    if len(lig_coords) == 0 and os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=False)
        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                lig_coords.append([pos.x, pos.y, pos.z])

    if len(lig_coords) == 0:
        return False

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, prot_path)
    except Exception:
        return False

    prot_atoms = []
    prot_coords = []
    for atom in structure.get_atoms():
        prot_atoms.append(atom)
        prot_coords.append(atom.get_coord())

    if len(prot_coords) == 0:
        return False

    # Find protein atoms near ligand atoms
    prot_tree = cKDTree(prot_coords)
    close_indices = set()
    for l_coord in lig_coords:
        indices = prot_tree.query_ball_point(l_coord, r=radius)
        close_indices.update(indices)

    if not close_indices:
        return False

    # Find which residues these atoms belong to
    close_residues = set()
    for idx in close_indices:
        atom = prot_atoms[idx]
        close_residues.add(atom.get_parent().get_full_id())

    # Write out the selected residues
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, ResidueSelect(close_residues))
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract protein binding pockets")
    parser.add_argument("--data-dir", type=str, default="general-set", help="Path to general-set")
    parser.add_argument("--radius", type=float, default=10.0, help="Radius in Angstroms")
    args = parser.parse_args()

    pdb_ids = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    print(f"Found {len(pdb_ids)} complexes. Extracting pockets...")

    success = 0
    with tqdm(total=len(pdb_ids)) as pbar:
        for pdb_id in pdb_ids:
            if extract_pocket(pdb_id, args.data_dir, args.radius):
                success += 1
            pbar.update(1)

    print(f"Successfully extracted {success}/{len(pdb_ids)} pockets.")

if __name__ == "__main__":
    main()
