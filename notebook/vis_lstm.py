#%%
import os
from typing import List

import matplotlib.pyplot as plt
import torch as th
import torch.utils.data as th_data

from state_forecast.data import SensorData
from state_forecast.forecasters import ForecasterLSTM

th.set_grad_enabled(False)

%matplotlib ipympl

#%%
PROJECT_ROOT = "../"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
CKPT_PATH = os.path.join(
    PROJECT_ROOT,
    "output",
    "lstm_time",
    "ckpt",
    "version_2",
    "epoch=499-step=0.ckpt",
)
N_HIST = 1600
HIDDEN_SIZE = 512
USE_TIME = True

#%%
val_set = SensorData(n_hist=N_HIST, root=DATA_ROOT, mode="val")
val_loader = th_data.DataLoader(val_set,
                                batch_size=1,
                                num_workers=0,
                                shuffle=False)
forecaster = ForecasterLSTM.load_from_checkpoint(
    checkpoint_path=CKPT_PATH,
    map_location=th.device("cpu"),
    n_hist=N_HIST,
    hidden_size=HIDDEN_SIZE,
    num_layers=1,
    post_lstm=th.nn.Linear(in_features=HIDDEN_SIZE, out_features=1),
    use_time=USE_TIME,
)
forecaster.eval()

#%%
preds: List[float] = list()
reals: List[float] = list()
for refs, sensors, _, times_sens in val_loader:
    refs: th.Tensor
    sensors: th.Tensor
    refs_ps: th.Tensor = forecaster(sensors, times_sens)
    preds.append(refs_ps.item())
    reals.append(refs.item())

#%%
snr = 10 * th.log10((th.tensor(reals)**2).mean() /
                    ((th.tensor(preds) - th.tensor(reals))**2).mean())

#%%
ts = val_set.times[val_set.n_hist:]
plt.plot(ts, preds, label="pred")
plt.plot(ts, reals, label="real")
plt.legend()

#%%
