# Perform standard imports¶
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the MNIST dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root='../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../data', train=False,
                           download=True, transform=transform)

print(test_data)

# Create loaders
# When working with images, we want relatively small batches a batch size of 4 is not uncommon
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Define layers

conv1 = nn.Conv2d(1, 6, 3, 1)
conv2 = nn.Conv2d(6, 16, 3, 1)

# Grab the first MNIST record
for i, (X_train, y_train) in enumerate(train_data):
    break

# Create a rank-4 tensor to be passed into the model
# (train_loader will have done this already)

x = X_train.view(1, 1, 28, 28)
print('The shape of the x: ', x.shape)

image, label = train_data[0]
print('Shape:', image.shape, '\nLabel:', label)

#plt.imshow(train_data[0][0].reshape((28, 28)), cmap="gray")
# plt.show()

# create the model


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


torch.manual_seed(42)
model = CNN()
print(model)

# Define loss function & optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
start_time = time.time()

epochs = 3
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        # Apply the model
        y_pred = model(X_train)  # we don't flatten X-train here
        loss = criterion(y_pred, y_train)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b % 600 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/60000]  loss: {loss.item():10.8f}  \
accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

    train_losses.append(loss.item())
    train_correct.append(trn_corr.item())

    # Run the testing batches
    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

# print the time elapsed
print(f'\nDuration: {time.time() - start_time:.0f} seconds')
# print the time elapsed
print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# Plot the loss and accuracy comparisons

plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validatiion loss')
plt.title('Loss')
plt.legend()
plt.show()

# Evaluate Test Data

test_load_all = DataLoader(test_data, batch_size=10000)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

x = 2019
plt.figure(figsize=(1, 1))
plt.imshow(test_data[x][0].reshape((28, 28)), cmap="gist_yarg")
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(test_data[x][0].view(1, 1, 28, 28)).argmax()
print("Predicted value:", new_pred.item())
