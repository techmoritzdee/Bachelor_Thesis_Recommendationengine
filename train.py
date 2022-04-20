import pandas as pd
import numpy as np
import torch

from data import load_train_data, load_track_features
from CSV_categorize import sns_plot, sns_plot_count, filter_df, convert_df_to_array, convert_df_to_numeric,\
    convert_array_to_torch
from logistic_regression import LogisticRegression

MODEL_SAVE_PATH = r'/Users/moritzdeecke/Desktop/Spotify_Dataset/model.pth'

if __name__ == "__main__":

  pd.options.display.max_columns = None
  pd.options.display.max_rows = 20

  track_features = load_track_features()
  train_data = load_train_data()
  ds_train = pd.merge(train_data, track_features, on="track_id", how="inner")

  print(ds_train['track_id'].nunique())
  print(ds_train['not_skipped'].value_counts())

  labels_tr = convert_df_to_array(ds_train["not_skipped"])
  labels_torch_tr = convert_array_to_torch(labels_tr)

  # Immediately after merge, drop label (and track_id) from dataset
  ds_train = ds_train.drop(['not_skipped', 'track_id'], axis=1)
  musicdata_tr = convert_df_to_array(convert_df_to_numeric(ds_train))
  musicdata_tr = (musicdata_tr - np.mean(musicdata_tr, axis=0)) / (np.std(musicdata_tr, axis=0) + 1.0e-6)
  musicdata_torch_train = convert_array_to_torch(musicdata_tr)

  data_torch_train = musicdata_torch_train
  model = LogisticRegression(data_torch_train.shape[1])  # Instantiate model w/ number of features in data (e.g. 14 for musicdata)

  dataset_tr = torch.utils.data.TensorDataset(data_torch_train, labels_torch_tr)
  loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=32, shuffle=True)

  optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
  loss = torch.nn.BCELoss()
  obs = {"l": []}

  for x, y in loader_tr:
    y_hat = model(x)

    l = loss(y_hat, y)
    obs["l"].append(l.item())

    l.backward()

    optimizer.step()
    optimizer.zero_grad()

  # Store model to disk
  torch.save(model, MODEL_SAVE_PATH)
  print("Model parameters after learning on training set.")
  print('Betas:', np.vstack((ds_train.columns.values, model.linear.weight.data)).T)
  print('Bias term:', model.linear.bias.data)

