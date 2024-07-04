import json
import shutil
import torch

from fastai.data.core import DataLoaders
from fastai.vision.all import *
from torchsummary import summary
from functools import *
from pathlib import Path
from fastai.learner import Learner
from torch.nn.parallel import DistributedDataParallel as DDP



from detect_malignant.src.configs.config import ExperimentationConfig
from detect_malignant.src.configs.paths_config import PathsConfig
from detect_malignant.src.configs.train_config import TrainConfig
from detect_malignant.src.loader.dataset import get_fastai_dataloaders
from detect_malignant.src.utils.utils import SimpleLogger, make_dirs

def train_model(model, config, config_paths, config_train, num_classes, logger):
    """Train the model."""


    exp_dir = Path(config.save_dir) / config.exp_name
    save_dir = config_paths.get_save_dir()



    if not save_dir.exists():
        make_dirs([save_dir])

    datasheet_dest_path = save_dir / "data.csv"

    logger.log(f"Copying datasheet from {config.datasheet_path} to {datasheet_dest_path}")

    shutil.copy(config.datasheet_path, datasheet_dest_path)

    with (save_dir / "TrainConfig.json").open("w") as frozen_settings_file:
        json.dump(config.dict(exclude_none=True), frozen_settings_file, indent=2)
        logger.log(f"Saved training configuration in {save_dir / 'TrainConfig.json'}")


    logger.log(f"Saving experiment to {exp_dir}")

    train_set = config_train.get_dataset(mode="Train")
    logger.log(f"Training set size {len(train_set)}")

    valid_set = config_train.get_dataset(mode="Valid")

    logger.log(f"Validation set size {len(valid_set)}")



    data = DataLoaders(*get_fastai_dataloaders(config, train_set, valid_set), device=config.device  )  
   
    # logger.log(summary(model=model, input_size=train_set[0][0].shape))
  
    # defining the optimizer, loss function, metrics and callbacks
    opt_func = config_train.get_optimizer()
    loss = config_train.get_loss(num_classes)
    metrics = config_train.get_metrics()

    logger.log(f"STARTING LEARNER WITH exp_dir: {save_dir}") 

    learn =  Learner(
        dls=data,
        model=model,
        opt_func=opt_func,
        loss_func=loss,
        lr=config.kwagrs_learner["lr"],
        metrics=metrics,
        wd=config.kwagrs_learner["weight_decay"],
        model_dir=save_dir,
    )
    learn.to(config.device)
    if len(config.device_ids) > 1:      
        learn = torch.nn.DataParallel(learn, device_ids=config.device_ids) 
        learn = learn.module 

    cbs = config_train.get_callbacks(learn, save_dir)
   
    logger.log(f"STARTING LEARNER WITH exp_dir: {save_dir}")   
    learn.fit_one_cycle(config.epochs, cbs=cbs)
     


def train(config: ExperimentationConfig):
    """Train the model."""
    config_paths = PathsConfig(expconfig=config)
    config_train = TrainConfig(expconfig=config, paths_config=config_paths)
    assert config.mode == "train", "Incorrect settings"
    logger = SimpleLogger(config.model_name + "-Trainer")    

    logger.log(json.dumps(config.dict(exclude_none=True), indent=2))
    
    Model = config_train.get_model()

    logger.log(f"Using model {config.model_name}")
    kwargs = config.model_kwargs

    num_classes = config_train.get_num_classes(config.datasheet_path)
    model = Model(**kwargs)
    device = config.device
    model = model.to(device)

    if len(config.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    model.train()
    train_model(model, config, config_paths, config_train, num_classes, logger)
































