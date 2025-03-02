from typing import Iterable, List, Optional, Tuple, Union

import pytorch_lightning as pl
import pytorch_lightning.accelerators as pl_accelerators
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.plugins as pl_plugins
import pytorch_lightning.strategies as pl_strategies
import torch as th
import torch.utils.data as th_data


class ForecasterLSTM(pl.LightningModule):

    pre_lstm: th.nn.Module
    lstm: th.nn.LSTM
    post_lstm: th.nn.Module

    use_time: bool

    _lr: float

    def __init__(
        self,
        n_hist: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        pre_lstm: th.nn.Module = th.nn.Identity(),
        post_lstm: th.nn.Module = th.nn.Identity(),
        use_time: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = th.nn.LSTM(
            input_size=n_hist * 2 - 1 if use_time else n_hist,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.pre_lstm = pre_lstm
        self.post_lstm = post_lstm
        self.use_time = use_time
        # set on fit
        self._lr = 1e-3
        # pl flags
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=[
            "pre_lstm",
            "post_lstm",
        ])

    def forward(
        self,
        sensors: th.Tensor,
        times_sens: th.Tensor,
    ) -> th.Tensor:
        inputs: th.Tensor = self._prep_inputs(sensors, times_sens)
        inputs = self.pre_lstm(inputs)
        lstm_outs, _ = self.lstm(inputs)
        lstm_outs = lstm_outs[:, 0, :]
        outs = self.post_lstm(lstm_outs)
        return outs

    def _prep_inputs(
        self,
        sensors: th.Tensor,
        times_sens: th.Tensor,
    ) -> th.Tensor:
        if self.use_time == False:
            return sensors
        times_sens = times_sens.diff(dim=2)
        inputs: th.Tensor = th.cat((sensors, times_sens), dim=2)
        return inputs

    def fit(self,
            train_loader: th_data.DataLoader,
            n_iter: int = 1000,
            lr: float = 1e-3,
            logger: Union[pl_loggers.Logger, Iterable[pl_loggers.Logger],
                          bool] = True,
            callbacks: Optional[Union[List[pl_callbacks.Callback],
                                      pl_callbacks.Callback]] = None,
            val_loader: Optional[th_data.DataLoader] = None,
            ckpt_path: Optional[str] = None,
            to_show_progress: bool = True,
            accelerator: Optional[Union[str,
                                        pl_accelerators.Accelerator]] = None,
            devices: Optional[Union[List[int], str, int]] = None,
            strategy: Optional[Union[str, pl_strategies.Strategy]] = None,
            plugins: Optional[Union[pl_plugins.PLUGIN_INPUT,
                                    List[pl_plugins.PLUGIN_INPUT]]] = None):
        self._lr = lr
        trainer = pl.Trainer(logger=logger,
                             callbacks=callbacks,
                             max_epochs=n_iter,
                             accelerator=accelerator,
                             devices=devices,
                             strategy=strategy,
                             plugins=plugins,
                             enable_progress_bar=to_show_progress,
                             log_every_n_steps=1)
        trainer.fit(self, train_loader, val_loader, ckpt_path=ckpt_path)

    def configure_optimizers(self) -> th.optim.Optimizer:
        return th.optim.Adam(self.parameters(), lr=self._lr)

    def training_step(
        self,
        batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
        batch_idx: int,
    ):
        refs, sensors, _, times_sens = batch
        refs_preds: th.Tensor = self.forward(sensors, times_sens)
        loss: th.Tensor = th.nn.functional.mse_loss(refs_preds, refs)
        optim: th.optim.Optimizer = self.optimizers(
            use_pl_optimizer=False)  # type:ignore
        optim.zero_grad()
        loss.backward()
        optim.step()
        self.log_dict({"train/mse": loss.item()}, prog_bar=True)

    def validation_step(
        self,
        batch: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor],
        batch_idx: int,
    ):
        refs, sensors, _, times_sens = batch
        refs_preds: th.Tensor = self.forward(sensors, times_sens)
        loss: th.Tensor = th.nn.functional.mse_loss(refs_preds, refs)
        self.log_dict({"val/mse": loss.item()}, prog_bar=True)
