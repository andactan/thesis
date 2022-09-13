import torch

from algorithms.vmpo import VMPO
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_method
from rlpyt.utils.collections import namedarraytuple