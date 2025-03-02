#%%
import os

import torch as th
import torch.utils.data as th_data

from state_forecast.data import SensorData
from state_forecast.forecasters import ForecasterLSTM

#%%
PROJECT_ROOT = "../"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
N_HIST: int = 1600
BATCH_SIZE: int = 32
HIDDEN_SIZE: int = 512

#%%
dataset = SensorData(n_hist=N_HIST, root=DATA_ROOT, mode="train")
loader = th_data.DataLoader(dataset, batch_size=BATCH_SIZE)

#%%
forecaster = ForecasterLSTM(n_hist=N_HIST,
                            hidden_size=HIDDEN_SIZE,
                            num_layers=1,
                            post_lstm=th.nn.Linear(in_features=HIDDEN_SIZE,
                                                   out_features=1))
forecaster.fit(train_loader=loader, n_iter=1000)

#%%
