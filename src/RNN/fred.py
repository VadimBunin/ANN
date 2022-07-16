import time
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kit

# This relates to plotting datetime values with matplotlib:
register_matplotlib_converters()

df = pd.read_csv('data/Building Materials and Supplies Dealers.csv',
                 index_col=0, parse_dates=True)
print(df.head())

# plt.figure(figsize=(12, 4))
# plt.title('Biulding Materials')
# plt.ylabel('Sales in Mio')
# plt.grid(True)
# plt.autoscale(axis='x', tight=True)
# plt.plot(df['MRTSSM4441USN'])
# plt.show()

y = df['MRTSSM4441USN'].values.astype(float)
test_size = 12

train_set = y[:-test_size]
test_set = y[-test_size:]

# Normalize the data

scaler = MinMaxScaler(feature_range=(-1, 1))

train_norm = scaler.fit_transform(train_set.reshape(-1, 1))

# Convert to tensor

train_norm = torch.FloatTensor(train_norm).view(-1)


print(train_norm.shape)
print(train_norm.ndim)
window_size = 12

# Prepare data for LSTM

window_size = 12


train_data = kit.input_data(train_norm, window_size)

# Define the model


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=500, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        # Add an LSTM layer:
        self.lstm = nn.LSTM(input_size, hidden_size)

        # Add a fully-connected layer:
        self.linear = nn.Linear(hidden_size, output_size)

        # Initialize h0 and c0:
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]  # we only want the last value


model = LSTM()


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 100

start_time = time.time()

for epoch in range(epochs):

    # extract the sequence & label from the training data
    for seq, y_train in train_data:

        # reset the parameters and hidden states
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))

        y_pred = model(seq)

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # print training result
    if epoch % 50 == 0:
        print(f'Epoch: {epoch+1:2} Loss: {loss.item():10.8f}')

print(f'\nDuration: {time.time() - start_time:.0f} seconds')


# Run predictions and compare to known test set

future = 12

# Add the last window of training values to the list of predictions

preds = train_norm[-window_size:].tolist()

# Set the model to evaluation mode

PATH = "state_dict_model.pt"
torch.save(model.state_dict(), PATH)
