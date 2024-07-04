
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data

from functools import partial
from pathlib import Path
from typing import Callable, List
from pydantic import BaseModel



from detect_malignant.src.callbacks.callbacks import (CustomSchedulerCallback,  EarlyStoppingCallback,
                                                               SaveBestModel,
                                                              SaveTrainingResult)
from detect_malignant.src.configs.config import ExperimentationConfig
from detect_malignant.src.configs.paths_config import PathsConfig
from detect_malignant.src.loader.dataset  import MalignantDataset
from detect_malignant.src.metrics.metric import MetricParams



class TrainConfig(BaseModel):

    """The training configuration."""

    expconfig: ExperimentationConfig
    paths_config: PathsConfig

    def get_model(self) -> torch.nn.Module:
        return self.expconfig.models[self.expconfig.model_name]

    def get_num_classes(self, path=None, df=None):
        return self.expconfig.model_kwargs["num_classes"]



    def call_exp_paths(self):
        paths = self.paths_config
        return paths.get_experiment_datasheet_path(exp_name=self.expconfig.exp_name)




    def get_optimizer(self) -> optim.Optimizer:
        mom = self.expconfig.optimizer_kwargs.pop("mom", 0.9)
        eps = self.expconfig.optimizer_kwargs.pop("eps", 1e-8)
        lr = self.expconfig.optimizer_kwargs.pop("lr", 0.001)  # Get the learning rate from optimizer_kwargs
        if self.expconfig.optimizer is not None:
            return partial(
                self.expconfig.optimizers[self.expconfig.optimizer],
                lr=lr,
                mom=mom,
                eps=eps,
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.expconfig.optimizer}")


    def get_callbacks(self, learner, save_dir: Path):
        verbose = self.expconfig.verbose   
        processed_kwargs = self.expconfig.kwagrs_learner.copy()
        scheduler_kwargs = self.expconfig.scheduler_kwargs
        scheduler_func = self.expconfig.schedulers[self.expconfig.scheduler]
        # Fix the typo here; it should be cb_mapping
        cb_mapping = {
            "best_save_cb": SaveBestModel(self),
            "save_train_result_cb": SaveTrainingResult(self, save_dir),           
            "lr_sched_cb": CustomSchedulerCallback(scheduler_func, scheduler_kwargs, learner, verbose),
            "early_stoppping_cb":EarlyStoppingCallback(self.expconfig),     
        }

        if "cbs" in processed_kwargs:
            selected_cbs = processed_kwargs["cbs"]
            out = [cb_mapping[cb_name] for cb_name in selected_cbs if cb_name in cb_mapping]
            print("Using callbacks: {}".format(out))
            return out
        else:
            return []
    def get_loss(self, num_classes) -> Callable:
        device = self.expconfig.device
        if self.expconfig.loss == "MalignantLoss":
            return self.expconfig.losses[self.expconfig.loss](self.expconfig.loss_kwargs, num_classes, self.expconfig.loss_kwargs.loss_dict).to(device)
        else:
            raise ValueError(f"Invalid loss: {self.expconfig.loss}")



    def get_metrics(self) -> List[Callable]:
        metric_params = MetricParams(one_hot_labels=self.expconfig.one_hot_labels)

        available_metrics = {
            "accuracy": metric_params.accuracy,
            "macro_f1": metric_params.macro_f1,
        }

        metric_functions = []
        for metric_name in self.expconfig.metrics:
            if metric_name in available_metrics:
                metric_functions.append(available_metrics[metric_name])
            else:
                raise ValueError(f"Invalid metric: {metric_name}")

        return metric_functions

    def get_dataset(
        self,
        mode: str,   
    ) :
     
        data_sheet = self.paths_config.get_experiment_datasheet_path(exp_name=self.expconfig.exp_name)       
        
        num_classes = self.get_num_classes()
        if self.expconfig.dataset == "MalignantDataset":
            dataset = MalignantDataset(
                config=self,             
                data_df=data_sheet,
                num_classes=num_classes,
                mode=mode
            )

        return dataset

    