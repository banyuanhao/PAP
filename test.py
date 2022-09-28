import os
from pathlib import Path

base = Path('perturbations','1','2')

if not os.path.isdir(base):
    os.makedirs(base)