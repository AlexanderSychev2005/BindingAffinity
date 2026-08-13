#!/usr/bin/env bash
# Run ON the cloud instance, from the synced repo root (e.g. ~/BindingAffinity),
# after sync_to_cloud.sh has copied the code over (lands under training/ on
# the remote - see the note in sync_to_cloud.sh about local/remote naming).
set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root, if invoked as training/scripts/cloud_env_setup.sh

uv venv --python 3.12
source .venv/bin/activate

# ROCm PyTorch build - same command the user already uses successfully.
uv pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1

# Everything the sequence-based pipeline needs. No torch-geometric here -
# that's only for the legacy structure-based GIGN model, not shipped to the cloud.
uv pip install fair-esm transformers huggingface_hub rdkit pandas numpy tqdm tensorboard

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
