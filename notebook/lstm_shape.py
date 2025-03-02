#%%
import os

import matplotlib.pyplot as plt
import torch.utils.data as th_data
import torch as th

from state_forecast.data import SensorData

#%%
PROJECT_ROOT = "../"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")

#%%
dataset = SensorData(n_hist=1600, root=DATA_ROOT, mode="train")
loader = th_data.DataLoader(dataset, batch_size=32)

#%%
rs, ss, t_rs, t_ss = next(iter(loader))

#%%
lstm = th.nn.LSTM(input_size=1600, hidden_size=512, batch_first=True)

#%%
outs, _ = lstm(ss)

#%%
