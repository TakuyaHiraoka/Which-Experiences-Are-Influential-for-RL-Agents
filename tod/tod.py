from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union
import torch
import torch.nn as nn
import numpy as np


class TurnOverDropout(nn.Module):
    def __init__(self, size: Union[int, Tuple], p: float = 0.5, seed: int = 777):
        super(TurnOverDropout, self).__init__()
        self.size = size
        self.p = p
        self.seed = seed
        if self.seed != 777:
            print("tod seed is changed from 777 to", self.seed)

        # make a random projection for hashing; this does not have to be updated (and even saved)
        proj = (torch.rand((size, ), generator=torch.Generator("cpu").manual_seed(seed)) + 0.1) * 1000.0
        self.proj = nn.parameter.Parameter(proj, requires_grad=False)

    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None, flips: Optional[Union[torch.Tensor, bool]] = None):
        if indices is not None:
            # randomize indices as binary d-dim vectors. arbitrary function can be used.
            indices = indices + 199
            # modified for TH (2D input) setting 20221015
            proj = self.proj.view(1, -1) # 1 x size mat
            hash = torch.matmul(indices, proj) #
            masks = ( (hash % 10.0 ) <= (self.p * 10.0) ).float()

            if (flips is not None) and (flips == True): # bugfix: add flups == True .TH20221017
                if isinstance(flips, torch.Tensor):
                    assert flips.shape == indices.shape
                    masks = torch.where(flips[..., None], 1.0 - masks, masks)
                else:
                    if flips:
                        masks = 1 - masks
                x = x * masks / (1.0 - self.p)
            else:
                x = x * masks / self.p
        else:
            assert flips is None
        return x
