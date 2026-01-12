import os
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow (optional if installed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

    # PyTorch (optional if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
