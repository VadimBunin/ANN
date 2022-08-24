from asyncio import windows_events
import time
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kit

# Create & plot data points

x = torch.linspace(0, 799, steps=800)
y = torch.sin(x*2*3.1416/40)

#plt.figure(figsize=(12, 4))
#plt.xlim(-10, 810)
# plt.grid(True)
#plt.plot(y.numpy(), c='b', lw=2)
# plt.show()

"""Create a train and test sets"""
test_size = int(len(y)*.15)
train_set = y[:-test_size]
test_set = y[-test_size:]

"""Prepare the training data"""
"""
When working with LSTM models,
we start by dividing the training sequence into a series of overlapping "windows".
Each window consists of a connected string of samples.
The label used for comparison is equal to the next value in the sequence.
In this way our network learns what value should follow a given pattern of preceding values.
Note: although the LSTM layer produces a prediction for each sample in the window, we only care about the last one.
"""
"""we'll define a function called input_data that builds a list of (seq, label) tuples.
Windows overlap, so the first tuple might contain  ([𝑥1,..,𝑥5],[𝑥6]) ,
the second would have  ([𝑥2,..,𝑥6],[𝑥7]) , etc.
Here  𝑘  is the width of the window. Due to the overlap,
we'll have a total number of (seq, label) tuples equal to  len(𝑠𝑒𝑟𝑖𝑒𝑠)−𝑘"""


def input_data(seq, ws):
    out = []
    L = len(seq)

    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out


k = 40
print(input_data(train_set, k)[:2])
