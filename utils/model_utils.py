
from glob import glob


def get_active_learning_checkpoint(path, default_checkpoint="model.pt"):
    files = list(glob(f"{path}/active_learning-*.pt"))
    if not files:
        return default_checkpoint
    else:
        return max(files).split("/")[-1]

