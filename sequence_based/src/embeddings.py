"""Pretrained encoders for sequence-only binding affinity: ESM-2 for protein
sequences, ChemBERTa for ligand SMILES.

`compute_*_embeddings` run the encoders frozen (no_grad) for one-off cache
building - that's the local-GPU path. `load_*_encoder` just construct the
model/tokenizer and are reused as-is by the cloud fine-tuning model, which
keeps requires_grad on and runs them inside the training loop instead.
"""

import time

import torch

PROTEIN_MODEL_NAME = "esm2_t12_35M_UR50D"
PROTEIN_EMBED_DIMS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t48_15B_UR50D": 5120,
}
PROTEIN_EMBED_DIM = PROTEIN_EMBED_DIMS[PROTEIN_MODEL_NAME]
LIGAND_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
LIGAND_EMBED_DIM = 768


def load_protein_encoder(device, model_name=PROTEIN_MODEL_NAME, train_mode=False):
    import esm

    model, alphabet = getattr(esm.pretrained, model_name)()
    model.to(device)
    model.train() if train_mode else model.eval()
    return model, alphabet


@torch.no_grad()
def compute_protein_embeddings(sequences, device, batch_size=1, model_name=PROTEIN_MODEL_NAME,
                                embeddings=None, save_callback=None, save_interval_sec=60):
    """batch_size=1: ESM-2 attention is O(L^2) per head/layer, and BindingDB
    sequences run up to ~1500 residues - padding a long sequence into a
    multi-item batch blows well past a 4GB card. One-at-a-time avoids that
    at the cost of some throughput, which is fine for a one-off cache build.
    Bump batch_size when running this on a big-VRAM box.

    `embeddings`/`save_callback`: write into a caller-owned dict and flush it
    to disk every `save_interval_sec`, so an interrupted run only loses the
    last partial interval, not the whole thing - useful for the local 4GB
    card where the ligand set alone can be hundreds of thousands of items.
    """
    model, alphabet = load_protein_encoder(device, model_name)
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    embeddings = {} if embeddings is None else embeddings
    last_save = time.monotonic()
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        data = [(str(j), seq) for j, seq in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[num_layers], return_contacts=False)
        reprs = out["representations"][num_layers]
        for seq, rep, tok_row in zip(batch, reprs, tokens):
            true_len = (tok_row != alphabet.padding_idx).sum().item()
            embeddings[seq] = rep[1:true_len - 1].mean(0).cpu()  # drop BOS/EOS
        del tokens, out, reprs
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if save_callback is not None and time.monotonic() - last_save > save_interval_sec:
            save_callback()
            last_save = time.monotonic()
    return embeddings


def load_ligand_encoder(train_mode=False):
    """seyonec/ChemBERTa-zinc-base-v1 only ships a legacy pytorch_model.bin
    (no safetensors), and transformers now refuses to torch.load those
    unless torch>=2.6 - which would mean bumping torch past what the legacy
    torch-geometric pipeline was pinned to. Load the state dict ourselves
    (weights_only=True is exactly the safe mode transformers is asking for)
    instead of going through AutoModel.from_pretrained's guarded path.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LIGAND_MODEL_NAME)
    config = AutoConfig.from_pretrained(LIGAND_MODEL_NAME)
    model = AutoModel.from_config(config)
    weights_path = hf_hub_download(LIGAND_MODEL_NAME, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.train() if train_mode else model.eval()
    return tokenizer, model


@torch.no_grad()
def compute_ligand_embeddings(smiles_list, device, batch_size=64,
                               embeddings=None, save_callback=None, save_interval_sec=60):
    tokenizer, model = load_ligand_encoder()
    model = model.eval().to(device)

    embeddings = {} if embeddings is None else embeddings
    last_save = time.monotonic()
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        out = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        for smiles, vec in zip(batch, pooled):
            embeddings[smiles] = vec.cpu()

        if save_callback is not None and time.monotonic() - last_save > save_interval_sec:
            save_callback()
            last_save = time.monotonic()
    return embeddings
