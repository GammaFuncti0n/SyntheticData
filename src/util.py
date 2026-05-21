import random
import numpy as np
import torch
import os
import logging
import yaml
from datetime import datetime

def setup_logging(path):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=f"{path}/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8"
    )
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True