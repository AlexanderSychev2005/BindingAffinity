from rdkit import Chem
from rdkit.Chem import Descriptors

smiles_1a1e = "Cc1cc(C)cc(c1)CN2C(=O)N(Cc3cc(C)cc(C)c3)[C@H](Cc4cc(C)cc(C)c4)[C@H](O)[C@@H]2O"
mol = Chem.MolFromSmiles(smiles_1a1e)

# 2. Проверка массы (должно быть ~568)
mw = Descriptors.MolWt(mol)
print(f"REAL Molecular Weight for 1a1e: {mw:.2f}")