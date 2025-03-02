#%%
import os

import matplotlib.pyplot as plt
import torch as th

from state_forecast.data import SensorData

#%%
PROJECT_ROOT = "../"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")

#%%
dataset = SensorData(n_hist=1600, root=DATA_ROOT, mode="train")

# %%
plt.plot(dataset.times[:, 0], dataset.refs[:, 0])

# %%
plt.plot(dataset.times[:, 0], dataset.sensors[:, 0])

#%%
rs, ss, t_rs, t_ss  = dataset[0]

#%%
