import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kit

df = pd.read_csv('data/NYCTaxiFares.csv')
df['dist_km'] = kit.haversine_distance(
    df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['EDTdate'] = pd.to_datetime(df['pickup_datetime']) - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
df['Weekday'] = df['EDTdate'].dt.strftime('%a')

# Categorical and continual features
cat_cols = ['Hour', 'AMorPM', 'Weekday']
y_col = ['fare_amount']  # this column contains the labels

cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
             'dropoff_longitude', 'passenger_count', 'dist_km']
# Categorify

for col in cat_cols:
    df[col] = df[col].astype('category')

# print(df['Weekday'].cat.categories)
# print(df['Weekday'].cat.codes.unique())
# print(df['Weekday'].cat.codes.values)

hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], axis=1)

# Convert continuous variables to a tensor
conts = np.stack([df[col].values for col in cont_cols], 1)

conts = torch.tensor(conts, dtype=torch.float)

# Set an embedding size
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
print(cat_szs)
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
