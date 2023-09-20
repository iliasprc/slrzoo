from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.googlenet_tconvs.googlenet_tconvs import GoogLeNet_TConvs
# from main_weaklys import SEED
SEED = 1234
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)
MAX_LENGTH = 30
# torch.cuda.manual_seed(SEED)
