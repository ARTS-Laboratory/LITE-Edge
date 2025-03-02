import sys
from argparse import ArgumentParser
from typing import List
import os

import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch as th
import torch.utils.data as th_data

from state_forecast.data import SensorData
from state_forecast.forecasters import ForecasterLSTM
from state_forecast.callbacks import ForceModelCheckpoint


def parse_args(args: List[str]):
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--n_hist", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--use_time", type=bool, default=False)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--name", type=str, default="train_lstm")
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="auto")
    argv = parser.parse_args(args)
    return argv


def main(args: List[str]):
    argv = parse_args(args)
    data_root: str = argv.data_root
    n_hist: int = argv.n_hist
    batch_size: int = argv.batch_size
    hidden_size: int = argv.hidden_size
    # load dataset
    train_set = SensorData(n_hist=n_hist, root=data_root, mode="train")
    train_loader = th_data.DataLoader(train_set,
                                      batch_size=batch_size,
                                      num_workers=argv.n_workers)
    val_set = SensorData(n_hist=n_hist, root=data_root, mode="val")
    val_loader = th_data.DataLoader(val_set,
                                    batch_size=batch_size,
                                    num_workers=argv.n_workers,
                                    shuffle=False)
    # train model
    logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join(argv.output_root, argv.name, "tfb"),
        name="",
    )
    callbacks: List[pl_callbacks.Callback] = [
        ForceModelCheckpoint(
            dirpath=os.path.join(argv.output_root, argv.name, "ckpt",
                                 f"version_{logger.version}"),
            every_n_epochs=10,
            save_top_k=-1,
        )
    ]
    forecaster = ForecasterLSTM(
        n_hist=n_hist,
        hidden_size=hidden_size,
        num_layers=1,
        post_lstm=th.nn.Linear(in_features=hidden_size, out_features=1),
        use_time=argv.use_time,
    )
    forecaster.fit(train_loader=train_loader,
                   n_iter=argv.n_iter,
                   logger=logger,
                   callbacks=callbacks,
                   val_loader=val_loader,
                   accelerator=argv.accelerator,
                   ckpt_path=argv.ckpt_path)


if __name__ == "__main__":
    main(sys.argv[1:])