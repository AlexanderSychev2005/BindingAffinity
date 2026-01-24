import os
import uuid

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from utils import get_inference_data, get_py3dmol_view,save_standalone_ngl_html
import nglview as nv


app = FastAPI()

os.makedirs("html_results", exist_ok=True)
app.mount("/results", StaticFiles(directory="html_results"), name="results")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    smiles_ligand: str = Form(...),
    sequence_protein: str = Form(...)
):
    mol, importance, affinity = get_inference_data(smiles_ligand, sequence_protein)

    atom_list = []
    sorted_indices = sorted(range(len(importance)), key=lambda k: importance[k], reverse=True)

    for idx in sorted_indices[:15]:
        val = importance[idx]
        symbol = mol.GetAtomWithIdx(idx).GetSymbol()

        icon = ""
        if val >= 0.9: icon = "🔥"
        elif val >= 0.7: icon = "✨"
        elif val >= 0.5: icon = "⭐"
        atom_list.append({
            "id": idx,
            "symbol": symbol,
            "score": f"{val:.3f}",
            "icon": icon
        })

    unique_id = str(uuid.uuid4())


    filename_ngl = f"ngl_{unique_id}.html"
    filepath_ngl = os.path.join("html_results", filename_ngl)

    py3dmol_view = get_py3dmol_view(mol, importance)
    py3dmol_content = py3dmol_view._make_html()

    # ngl_view = get_ngl_view(mol, importance)
    # nv.write_html(filepath_ngl, ngl_view)

    save_standalone_ngl_html(mol, importance, filepath_ngl)

    ngl_url_link = f"/results/{filename_ngl}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result_ready": True,
        "smiles": smiles_ligand,
        "protein": sequence_protein,
        "affinity": f"{affinity:.2f}",
        "atom_list": atom_list,
        "html_py3dmol": py3dmol_content,
        "url_ngl": ngl_url_link
    })





