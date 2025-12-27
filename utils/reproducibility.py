"""Reproducibility utilities"""

import torch
import numpy as np
import random
import os

GLOBAL_SEED = 42


def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    set_all_seeds()
    print(f"Seeds set to {GLOBAL_SEED}")
