import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / "src"), str(PACKAGE / "scripts")]
torch.set_num_threads(2)
