from typing import NamedTuple

import torch


hmm_fields = [
    ('states', torch.Tensor),
    ('observations', torch.Tensor),
]

HMMOutput = NamedTuple('HMMOutput', hmm_fields)


