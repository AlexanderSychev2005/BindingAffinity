#!/usr/bin/env bash
# Run from the repo root on your local machine:
#   HOST=user@1.2.3.4 ./sequence_based/scripts/sync_to_cloud.sh
#
# Copies only what the sequence-based pipeline needs: scripts/src and the
# small split CSVs. Deliberately skips: structure_based/ (GIGN pipeline,
# irrelevant here), data/raw/ (the 9GB parsed-away BindingDB TSV),
# data/embeddings*/ (the local encoder caches - not reusable at a different
# embedding dim, cheap to recompute on the box with real GPU headroom
# instead of shipping it over the wire).
#
# Note: the remote layout still uses a bare "training/" directory name
# (set up before the local repo split into sequence_based/structure_based) -
# REMOTE_DIR below points at the repo root, and the training/ subpath is
# the existing remote convention, independent of the local folder name.
set -euo pipefail

HOST="${HOST:?set HOST=user@ip, e.g. HOST=user@1.2.3.4 ./sequence_based/scripts/sync_to_cloud.sh}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-~/BindingAffinity}"

ssh -p "$PORT" "$HOST" "mkdir -p $REMOTE_DIR/training/data"

rsync -avz --progress -e "ssh -p $PORT" \
  sequence_based/scripts/ "$HOST:$REMOTE_DIR/training/scripts/"

rsync -avz --progress -e "ssh -p $PORT" \
  sequence_based/src/ "$HOST:$REMOTE_DIR/training/src/"

rsync -avz --progress -e "ssh -p $PORT" \
  sequence_based/data/bindingdb_train.csv \
  sequence_based/data/bindingdb_val.csv \
  sequence_based/data/bindingdb_test.csv \
  sequence_based/data/bindingdb_colddrug_train.csv \
  sequence_based/data/bindingdb_colddrug_val.csv \
  sequence_based/data/bindingdb_colddrug_test.csv \
  "$HOST:$REMOTE_DIR/training/data/"

echo "synced. Remote has training/{scripts,src,data/*.csv} only."
echo "Bring results back with:"
echo "  rsync -avz -e 'ssh -p $PORT' $HOST:$REMOTE_DIR/runs/ ./sequence_based/runs_cloud/"
