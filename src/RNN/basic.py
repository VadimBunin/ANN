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

# plt.figure(figsize=(12, 4))
# plt.xlim(-10, 801)
# plt.grid(True)
# plt.plot(y.numpy(), c='r', ls=':', lw=3)
# plt.show()

# Create train and test sets
test_size = 40
train_set = y[:-test_size]
test_set = y[-test_size:]
# Load the data
window_size = 40
train_data = kit.input_data(train_set, window_size)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size, hidden_size)

        # Add a fully-connected layer:
        self.linear = nn.Linear(hidden_size, out_size)

        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1, 1, hidden_size),
                       torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, 1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]   # we only care about the last predic


torch.manual_seed(42)
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print(model)


epochs = 7
window_size = 40
future = 40

# Create the full set of sequence/label tuples:
all_data = kit.input_data(y, window_size)
len(all_data)  # this should equal 800-40

start_time = time.time()

for i in range(epochs):
    for seq, label in all_data:
        # reset the parameters and hidden states

        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
    # print training result
    print(f'Epoch: {i+1:2} Loss: {loss.item():10.8f}')

print(f'\nDuration: {time.time() - start_time:.0f} seconds')

preds = y[-window_size:].tolist()

for i in range(future):
    seq = torch.FloatTensor(preds)
    with torch.no_grad():
        # Reset the hidden parameters
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        preds.append(model(seq).item())

plt.figure(figsize=(12, 4))
plt.xlim(-10, 841)
plt.grid(True)
plt.plot(y.numpy())
plt.plot(range(800, 800+future), preds[window_size:])
plt.show()
