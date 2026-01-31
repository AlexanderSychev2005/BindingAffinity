import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np


def generate_final_viz():
    # 1. Твой лиганд
    smiles = "Cc1cc(C)cc(c1)CN2C(=O)N(Cc3cc(C)cc(C)c3)[C@H](Cc4cc(C)cc(C)c4)[C@H](O)[C@@H]2O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Важно! Твоя модель работала с H, значит индексы с H

    # Генерация координат для красивой картинки
    AllChem.Compute2DCoords(mol)

    # 2. Твои результаты Attention (из логов)
    # Словарь {Atom_Index: Score}
    # Я взял твои топ значения
    raw_attention = {
        73: 1.000, 71: 0.968,  # H
        33: 0.699, 35: 0.670,  # O
        39: 0.484, 43: 0.484, 44: 0.484, 49: 0.484, 53: 0.484, 57: 0.484,  # H group
        0: 0.479, 4: 0.479  # C
    }

    # 3. Агрегация: переносим вес с H на тяжелые атомы
    num_atoms = mol.GetNumAtoms()
    aggregated_weights = np.zeros(num_atoms)

    for idx, score in raw_attention.items():
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'H':
            # Отдаем вес соседу (Heavy Atom)
            neighbors = atom.GetNeighbors()
            if neighbors:
                parent_idx = neighbors[0].GetIdx()
                aggregated_weights[parent_idx] += score
        else:
            aggregated_weights[idx] += score

    # Нормализация для цвета (0..1)
    if aggregated_weights.max() > 0:
        aggregated_weights /= aggregated_weights.max()

    # 4. Выделяем топ атомы для подсветки (уже без H)
    highlight_atoms = []
    highlight_colors = {}

    for i in range(num_atoms):
        score = aggregated_weights[i]
        if score > 0.3:  # Порог, чтобы не светить всё подряд
            highlight_atoms.append(i)
            # Цвет: Красный, прозрачность зависит от скора
            highlight_colors[i] = (1.0, 0.0, 0.0, score)

    # 5. Рисуем (Убираем H для красоты картинки)
    mol_no_h = Chem.RemoveHs(mol)
    # Нам нужно смапить индексы обратно, но для простоты нарисуем с H,
    # или (лучше) просто используем индексы тяжелых атомов, если они совпадают до AddHs.
    # В данном случае проще нарисовать структуру с H, но скрыть их лейблы,
    # либо оставить как есть, так как агрегация уже подсветит нужные C/N/O.

    d = rdMolDraw2D.MolDraw2DCairo(1000, 600)
    d.drawOptions().annotationFontScale = 0.7
    d.drawOptions().addAtomIndices = True  # Можно убрать для чистоты

    rdMolDraw2D.PrepareAndDrawMolecule(d, mol,
                                       highlightAtoms=highlight_atoms,
                                       highlightAtomColors=highlight_colors)
    d.FinishDrawing()
    d.WriteDrawingText("final_hiv_viz.png")
    print("✅ Картинка готова: assets/final_hiv_viz.png")


if __name__ == "__main__":
    generate_final_viz()