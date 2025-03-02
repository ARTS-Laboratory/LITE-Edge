import csv
import os
from typing import List, Literal, Tuple

import torch as th
import torch.utils.data as th_data


class SensorData(th_data.Dataset[Tuple[th.Tensor, th.Tensor]]):

    n_hist: int

    times: th.Tensor
    refs: th.Tensor
    sensors: th.Tensor

    def __init__(self, n_hist: int, root: str, mode: Literal["train",
                                                             "val"]) -> None:
        super().__init__()
        self.n_hist = n_hist
        fn: str = "train.csv" if mode == "train" else "test.csv"
        # read from csv
        times_l: List[float] = list()
        refs_l: List[float] = list()
        sensors_l: List[float] = list()
        with open(os.path.join(root, fn), "r") as ds_file:
            ds_reader = csv.DictReader(ds_file)
            for row in ds_reader:
                times_l.append(float(row["time"]))
                refs_l.append(float(row["ref"]))
                sensors_l.append(float(row["sensor"]))
        # to pytorch tensor
        self.times = th.tensor(times_l)[:, None]
        self.refs = th.tensor(refs_l)[:, None]
        self.sensors = th.tensor(sensors_l)[:, None]

    def __getitem__(
            self,
            index: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        idx: int = index
        ref: th.Tensor = self.refs[idx + self.n_hist]
        sens: th.Tensor = self.sensors[idx:idx + self.n_hist].permute((1, 0))
        time_ref: th.Tensor = self.times[idx + self.n_hist]
        times_sens: th.Tensor = self.times[idx:idx + self.n_hist].permute((1, 0))
        return ref, sens, time_ref, times_sens

    def __len__(self) -> int:
        return self.times.shape[0] - self.n_hist
