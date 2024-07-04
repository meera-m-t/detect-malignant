import datetime
import os
import numpy as np
import torch.nn as nn

from fastai.callback.schedule import ParamScheduler
from fastai.callback.tracker import EarlyStoppingCallback, TrackerCallback
from fastcore.basics import store_attr
from fastcore.nb_imports import *
from fastcore.foundation import store_attr
from detect_malignant.src.utils.config_utils import get_subm_folder



class SaveBestModel(TrackerCallback):
    _only_train_loop = True

    def __init__(
        self,
        config,
        monitor="valid_loss",
        comp=None,
        min_delta=0.0,
        with_opt=False,
        reset_on_fit=True,
        outfile="",
    ):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        store_attr("with_opt")
        self.config = config
        self.best_metrics = {"loss": float("inf"), "acc": 0, "f1": 0}
        self.metrics_pos_dict = None
        self.outfile = outfile

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def before_fit(self):
        super().before_fit()
        self.metrics_pos_dict = {metric: idx - 1 for idx, metric in enumerate(self.recorder.metric_names)}

    def _get_current_metrics(self):
        metrics = {
            "loss": self.recorder.values[-1][self.metrics_pos_dict["valid_loss"]],
            "acc": self.recorder.values[-1][self.metrics_pos_dict["accuracy"]],
            "f1": self.recorder.values[-1][self.metrics_pos_dict["macro_f1"]],
        }
        return metrics


    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        current_metrics = self._get_current_metrics()

        for metric, value in current_metrics.items():
            if metric == "loss":
                condition = value < self.best_metrics[metric]
            else:  # For acc and f1
                condition = value > self.best_metrics[metric]

            if condition:
                self.best_metrics[metric] = value
                self._save(f"best_{metric}" + self.outfile)

        for metric in self.best_metrics:
            self._save(f"last_{metric}" + self.outfile)


class SaveTrainingResult(TrackerCallback):
    _only_train_loop = True

    def __init__(self, config, save_dir):
        super().__init__()
        self.config = config
        self.save_path = self._prepare_save_path(save_dir)
        self.start_epoch_time = None

    def _prepare_save_path(self, save_dir):
        save_path = os.path.join(get_subm_folder(save_dir), "train_loss.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        return save_path

    def _write_to_file(self, data):
        with open(self.save_path, "a") as f:
            f.write(",".join(map(str, data)) + "\n")

    def before_fit(self):
        self._write_to_file(self.recorder.metric_names)
        self._write_to_file(["-1"] + [""] * (len(self.recorder.metric_names) - 2) + [str(datetime.datetime.now())])

    def before_epoch(self):
        self.start_epoch_time = datetime.datetime.now()

    def after_epoch(self):
        elapsed_time = datetime.datetime.now() - self.start_epoch_time
        minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
        formatted_time = f"{int(minutes)}:{int(seconds)}"
        self._write_to_file([self.learn.epoch] + self.recorder.values[-1] + [formatted_time])






class CustomSchedulerCallback(TrackerCallback):
    def __init__(self, scheduler_func, scheduler_kwargs, learner, print_schedular_lr):
        super().__init__()          
        scheduler_kwargs = {str(k): float(v) for k, v in (pair.split('=') for pair in str(scheduler_kwargs).split())}        
        self.scheduler_func = scheduler_func(*scheduler_kwargs.values()) if scheduler_func else None
        self.verbose = print_schedular_lr
        self.scheduler_kwargs = scheduler_kwargs
        if learner is not None:
            self.learn = learner
            self.learn.hps = {'lr': []}  # Initialize a dictionary to store learning rates in the learner
        else:
            print("Warning: Learner object is None.")

    def before_fit(self):
        if not hasattr(self, 'learn') or self.learn is None:
            print("Warning: Learner object is None. Skipping before_fit.")
            return
  
        self.learn.hps = {'lr': []}  # Reset here
        self.learn.add_cb(ParamScheduler({'lr': self.scheduler_func}))

    def after_epoch(self):
        # Adjust the monitor to match exactly a name from learn.recorder.metric_names
        self.monitor = "macro_f1"  # Adjust this based on the actual metric you want to monitor
        self.learn =self.learn.module if isinstance(self.learn, nn.DataParallel) else self.learn        
        if self.monitor in self.learn.recorder.metric_names:           
            metric_value = self.learn.recorder.values[-1][3]  # Accesses the metric value for the last epoch
            if self.best is None or self.comp(metric_value, self.best + self.min_delta):
                self.best = metric_value
                # Assuming scheduler_func returns the new learning rate
                new_lr = self.scheduler_func(pos=self.learn.epoch)
                self.learn.opt.set_hyper('lr', new_lr)
                if self.verbose:
                    print(f"{self.monitor} : {metric_value}. Adjusting Learning Rate to {new_lr}.")
        else:
            print(f"Warning: Monitored metric '{self.monitor}' not found.")



class EarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, config):   
        monitor_metric = config.early_stopping_kwargs.watched_metric
        
        # Determine if we're looking for improvement ('min' for decrease, 'max' for increase)
        if config.early_stopping_kwargs.watched_metric_polarity == "Negative":
            monitor_comp = np.less
        else:  # Assuming 'Positive' polarity means we want the metric to increase
            monitor_comp = np.greater
        
        super().__init__(
            monitor=monitor_metric,
            min_delta=config.early_stopping_kwargs.min_delta,
            patience=config.early_stopping_kwargs.patience,
            comp=monitor_comp
        )