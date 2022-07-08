import torch
import torch.nn as nn

import numpy as np
import pandas as pd

# Prepare the training data


def input_data(seq, ws):
    out = []
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out
