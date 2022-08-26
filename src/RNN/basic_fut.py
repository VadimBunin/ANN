from asyncio import futures
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kit as k

# Create & plot data points

x = torch.linspace(0, 799, steps=800)
y = torch.sin(x*2*3.1416/40)

"""Prepare seqs and labels"""

ws = 40
all_data = k.input_data(y, ws)

""""Create the LSTM Model"""


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1, batch_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = (torch.zeros(num_layers, batch_size, hidden_size),
                       torch.zeros(num_layers, batch_size, hidden_size))

    def forward(self, seq):
        lstm_out, (self.hidden) = self.lstm(
            seq.view(len(seq), self.batch_size, self.input_size), self.hidden)
        pred = self.linear(lstm_out.view(-1, self.hidden_size))
        return pred[-1]


model = LSTM()
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 7
losses = []
for i in range(epochs):

    # tuple-unpack the entire set of data
    for seq, y_train in all_data:

        # reset the parameters and hidden states
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))

        y_pred = model(seq)

        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

    # print training result
    print(f'Epoch: {i+1:2} Loss: {loss.item():10.8f}')


"""Predict future values, plot the result"""
future = 40
preds = y[-ws:].tolist()

for i in range(future):
    seq = torch.FloatTensor(preds)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        preds.append(model(seq).item())


plt.figure(figsize=(12, 4))
plt.xlim(-10, 841)
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(800, 800+future), preds[ws:])
plt.show()
