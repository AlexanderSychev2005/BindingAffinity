"""Shared training-loop bookkeeping: run config dump + top-K checkpoint
retention. Factored out so the (planned) cloud fine-tuning script reuses the
same run-dir layout instead of duplicating this.
"""

import json
import os


def save_config(run_dir, args):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


class TopKCheckpoints:
    """Keeps the K checkpoints with the lowest metric value on disk under
    run_dir/checkpoints/, deleting the rest as new ones arrive.
    """

    def __init__(self, run_dir, k=3):
        self.dir = os.path.join(run_dir, "checkpoints")
        self.k = k
        os.makedirs(self.dir, exist_ok=True)
        self.kept = []  # list of (metric, path), sorted ascending

    def offer(self, epoch, metric, state_dict, save_fn):
        path = os.path.join(self.dir, f"epoch{epoch:04d}_rmse{metric:.4f}.pt")
        save_fn(state_dict, path)
        self.kept.append((metric, path))
        self.kept.sort(key=lambda x: x[0])
        while len(self.kept) > self.k:
            _, worst_path = self.kept.pop()
            os.remove(worst_path)

    @property
    def best_path(self):
        return self.kept[0][1]
