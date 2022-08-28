from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This relates to plotting datetime values with matplotlib:
register_matplotlib_converters()


df = pd.read_csv("data\Ice Cream and Frozen Dessert.csv",
                 index_col=0, parse_dates=True)
""""
plt.figure(figsize=(12, 4))
plt.title(
    'Industrial Production Index for Ice Cream and Frozen Dessert')
plt.ylabel('Index 2017=100, Not Seasonally Adjusted')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df['IPN31152N'])
plt.show()
"""
"""
Divide the data into train and test sets
Working with a window_size of 12, divide the dataset into a sequence of 313 training records (including the window),
and a test set of 12 records."""

y = df['IPN31152N'].values.astype(float)

test_size = 24

window_size = 12

train_set = y[:-test_size]
test_set = y[-test_size:]

# Run the code below to check your results:
print(f'Train: {len(train_set)}')
print(f'Test:  {len(test_set)}')

"""Normalize the training set
Feature scale the training set to fit within the range [-1,1]."""


scaler = MinMaxScaler(feature_range=(-1, 1))
train_norm = scaler.fit_transform(train_set.reshape(-1, 1))

print(f'First item, original: {train_set[0]}')
print(f'First item, scaled: {train_norm[0]}')
