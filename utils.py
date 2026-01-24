import torch
import numpy as np
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
import nglview as nv
import py3Dmol


from dataset import get_atom_features, get_protein_features
from model_attention import BindingAffinityModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "runs/experiment_attention20260124_104439_optuna/models/model_ep041_mse1.9153.pth"

GAT_HEADS = 2
HIDDEN_CHANNELS = 256

def get_inference_data(ligand_smiles, protein_sequence, model_path=MODEL_PATH):
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
    if len(tokens) > 1200: tokens = tokens[:1200]
    else: tokens.extend([0] * (1200 - len(tokens)))
    protein_sequence = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    data = Data(x=x, edge_index=edge_index)
    batch = Batch.from_data_list([data]).to(DEVICE)
    num_features = x.shape[1]

    # Model loading
    model = BindingAffinityModel(num_features, hidden_channels=HIDDEN_CHANNELS, gat_heads=GAT_HEADS).to(DEVICE)
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
        importance = (importance - importance.min()) / (importance.max() - importance.min())

    # Noise reduction
    importance[importance < 0.01] = 0
    return mol, importance, pred.item()



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


    indices_sorted = np.argsort(importance)[::-1]
    top_indices = indices_sorted[:15]


    selection_list = [str(i) for i in top_indices]
    selection_str = "@" + ",".join(selection_list)

    # Защита от пустой выборки
    if not selection_list:
        selection_str = "@-1"

    html_content = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NGL Visualization</title>
        <script src="https://unpkg.com/ngl@2.0.0-dev.37/dist/ngl.js"></script>
        <style>
            html, body {{ width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; font-family: sans-serif; }}
            #viewport {{ width: 100%; height: 100%; }}

            /* Стиль подсказки */
            #tooltip {{
                display: none;
                position: absolute;
                z-index: 100;
                pointer-events: none; /* Чтобы мышь не 'застревала' на подсказке */
                background-color: rgba(20, 20, 20, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                white-space: nowrap;
                border: 1px solid rgba(255,255,255,0.2);
                transition: opacity 0.1s ease;
            }}

            /* Панель управления */
            #controls {{
                position: absolute;
                top: 20px;
                right: 20px;
                z-index: 50;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
            }}

            /* Стили переключателя */
            .switch-container {{
                display: flex;
                align-items: center;
                gap: 10px;
                cursor: pointer;
                font-weight: bold;
                color: #333;
            }}

            input[type=checkbox] {{
                transform: scale(1.5);
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div id="controls">
            <label class="switch-container">
                <input type="checkbox" id="heatmapToggle" checked>
                <span>Show Heatmap</span>
            </label>
        </div>

        <div id="tooltip"></div>

        <div id="viewport"></div>

        <script>
            var pdbData = `{final_pdb_block}`;
            var selectionString = "{selection_str}";
            var component; // Глобальная переменная для доступа к модели

            document.addEventListener("DOMContentLoaded", function () {{
                var stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});
                var tooltip = document.getElementById("tooltip");
                var toggle = document.getElementById("heatmapToggle");

                // Загружаем данные
                var stringBlob = new Blob([pdbData], {{type: 'text/plain'}});

                stage.loadFile(stringBlob, {{ ext: 'pdb' }}).then(function (o) {{
                    component = o; // Сохраняем ссылку

                    // Рисуем начальное состояние
                    updateVisualization();
                    o.autoView();
                }});

                // --- ФУНКЦИЯ ОБНОВЛЕНИЯ ВИДА ---
                function updateVisualization() {{
                    if (!component) return;

                    // Очищаем старые представления (чтобы не накладывались)
                    component.removeAllRepresentations();

                    var useHeatmap = toggle.checked;

                    if (useHeatmap) {{
                        // 1. РЕЖИМ HEATMAP
                        component.addRepresentation("ball+stick", {{
                            colorScheme: "bfactor",
                            colorDomain: [20, 80],
                            colorScale: ["blue", "white", "red"],
                            radiusScale: 1.0
                        }});
                    }} else {{
                        // 2. ОБЫЧНЫЙ РЕЖИМ (По элементам)
                        component.addRepresentation("ball+stick", {{
                            colorScheme: "element",
                            radiusScale: 1.0
                        }});
                    }}

                    // Добавляем метки (они нужны всегда)
                    if (selectionString.length > 1 && selectionString !== "@-1") {{
                        component.addRepresentation("label", {{
                            sele: selectionString,
                            labelType: "atomindex",
                            color: "black",     
                            radius: 1.1,        
                            yOffset: 0.0,       
                            zOffset: 2.0,       
                            attachment: "middle_center",
                            pickable: true // ВАЖНО: Делаем текст интерактивным
                        }});
                    }}
                }}

                // Слушаем переключатель
                toggle.addEventListener("change", updateVisualization);

                // --- УМНЫЙ TOOLTIP ---
                stage.mouseControls.remove("hoverPick"); // Убираем стандартное поведение

                stage.signals.hovered.add(function (pickingProxy) {{
                    // Проверяем, навели ли мы на атом ИЛИ на метку (текст)
                    // NGL возвращает pickingProxy.atom даже если мы навели на label этого атома
                    if (pickingProxy && (pickingProxy.atom || pickingProxy.closestBondAtom)) {{
                        var atom = pickingProxy.atom || pickingProxy.closestBondAtom;
                        var score = atom.bfactor.toFixed(2);

                        tooltip.innerHTML = `
                            <div style="margin-bottom:2px;"><b>Atom ID:</b> ${{atom.index}} (${{atom.element}}:  ${{atom.atomname}})</div>
                            <div style="color: #ffcccc;"><b>Importance:</b> ${{(score/100).toFixed(3)}}</div>
                        `;
                        tooltip.style.display = "block";
                        tooltip.style.opacity = "1";

                        // Позиционирование: сдвиг вправо и вниз, чтобы не мешать
                        var cp = pickingProxy.canvasPosition;
                        tooltip.style.left = (cp.x + 20) + "px";
                        tooltip.style.top = (cp.y + 20) + "px";

                    }} else {{
                        // Скрываем, если увели мышь
                        tooltip.style.display = "none";
                        tooltip.style.opacity = "0";
                    }}
                }});

                // Ресайз окна
                window.addEventListener("resize", function(event){{
                    stage.handleResize();
                }}, false);
            }});
        </script>
    </body>
    </html>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)