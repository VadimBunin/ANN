from asyncio import windows_events
import time
from turtle import forward
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kit

# Create & plot data points

x = torch.linspace(0, 799, steps=800)
y = torch.sin(x*2*3.1416/40)

# plt.figure(figsize=(12, 4))
# plt.xlim(-10, 810)
# plt.grid(True)
# plt.plot(y.numpy(), c='b', lw=2)
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
train_data = input_data(train_set, k)
"""
X = []
y = []
for seq, lebel in train_data:
    X.append(seq)
    y.append(lebel)


print(X[0].view(len(X[0]), 1, 1))
"""


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, out_size=1, batch_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

        self.hidden = (torch.zeros(num_layers, batch_size, hidden_size),
                       torch.zeros(num_layers, batch_size, hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), self.batch_size, self.input_size), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), self.hidden_size))
        return pred[-1]


model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 150
losses = []
future = 40  # forcast 40 points in to future

for i in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(model.num_layers, model.batch_size, model.hidden_size),
                        torch.zeros(model.num_layers, model.batch_size, model.hidden_size))
    y_pred = model(seq)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    # print training result
    print(f'Epoch: {i+1:2} Loss: {loss.item():10.8f}')

plt.plot(range(epochs), losses)
plt.ylabel('RMSE Loss')
plt.xlabel('epoch')
plt.show()
