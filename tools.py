from typing import Dict, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torchrs.datasets import BrazilianCoffeeScenes

