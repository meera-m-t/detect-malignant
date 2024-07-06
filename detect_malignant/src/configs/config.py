import os
import shutil
from typing import ClassVar, Dict, List, Optional, Type

import torch.optim as optim
import torch.utils.data
from fastai.callback.schedule import SchedNo, SchedCos, SchedLin, SchedExp
from fastai.optimizer import ranger
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

from detect_malignant.src.configs.misc_config import ( EarlyStoppingConfig, 
                                                      SchedulerConfig,
                                                               )
from detect_malignant.src.loader.dataset import MalignantDataset
from detect_malignant.src.losses.loss import MalignantLoss, SmartCrossEntropyLoss
from detect_malignant.src.models.senet import SENet
from detect_malignant.src.models.densenet import DenseNet_




class ExperimentationConfig(BaseModel):

    """The experimentation configuration."""

    exp_name: str = Field(..., description="The name of the experiment")  

    models: ClassVar[Dict[str, torch.nn.Module]] = {
        "SENet": SENet,
        "DenseNet": DenseNet_, 
             
    }

    model_name: str = Field(..., description="The model to train/test with")


    save_dir: str = Field(..., description="The directory for the saved models while training")

    mode: str = Field(
        default="train",
        description="The network mode i.e. `train` or `test` or `finetune`",
    )

    epochs: Optional[int] = Field(default=30, description="The number of epochs when training")

    batch_size: int = Field(default=64, description="The batch size when training")

    imsize: int = Field(default=224, description="The image size")
 

    early_stopping_kwargs: Optional[EarlyStoppingConfig] = Field(default=EarlyStoppingConfig())    


    save_dir: Optional[str] = Field(
        default=None,
        description="The directory for the saved models while training",
    )

    num_workers: int = Field(default=14, description="The number of workers to use in dataloaders")

    optimizers: ClassVar[Dict[str, Type]] = {
        "Ranger": ranger,
        "Adam": optim.Adam,
    }

    optimizer: str = Field(
        default="Ranger",  # Set a default optimizer here
        description="The optimizer name",
    )
    optimizer_kwargs: Optional[Dict] = Field(default={}, description="The keyword arguments for the optimizer")

    schedulers: ClassVar[Dict[str, Type]] = {
        "SchedLin":SchedLin,
        "SchedExp": SchedExp,  
        "SchedNo": SchedNo, 
        "SchedCos": SchedCos, }

    scheduler: str = Field(default="SchedNo", description="The scheduler to use")

    scheduler_kwargs: Optional[SchedulerConfig] = Field(
        default=SchedulerConfig().dict(),
        description="The keyword arguments for the scheduler",
    )
    

    model_kwargs: Optional[Dict] = Field(default={}, description="The keyword arguments for the model")

    losses: ClassVar[Dict[str, Type]] = {
        "MalignantLoss": MalignantLoss,
        "SmartCrossEntropyLoss": SmartCrossEntropyLoss,
        
    }

    loss: str = Field(default="MalignantLoss", description="The loss to use")

    loss_kwargs: Optional[Dict] = Field(default={}, description="The keyword arguments for the loss")

    metrics: List[str] = Field(default=["accuracy", "macro_f1"], description="The metrics to use")

   
    datasheet_path: Optional[str] = Field(..., description="The path to the datasheet")

    
    device: Optional[str] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu", description="The device to use"
    )

    device_ids: Optional[List[int]] = Field(
        default_factory=lambda: list(range(torch.cuda.device_count())), description="The device ids to use"
    )

    datasets: ClassVar[Dict[str, Type]] = {
        "MalignantDataset": MalignantDataset,
        
    }

    dataset: str = Field(default="MalignantDataset", description="The dataset to use")

    kwargs_augmentation: Optional[Dict] = Field(default={}, description="The keyword arguments for the augmentation")

    one_hot_labels: bool = Field(default=False, description="Whether the labels are one-hot encoded")

    kwagrs_learner: Optional[Dict] = Field(default={}, description="The keyword arguments for the learner")

    verbose: bool = Field(default=False, description="Whether to print verbose information")
    

    @validator("mode", always=True)
    def mode_validator(cls, value):
        if value is None:
            return "train"
        return value

    @validator("model_name", always=True)
    def model_name_validator(cls, value):
        if value not in cls.models:
            raise ValidationError(f"Model name {value} is not valid")
        return value

    @validator("optimizer", always=True, pre=True)
    def optimizer_name_validator(cls, value, values):
        mode = values.get("mode")
        if mode == "train":
            if value not in cls.optimizers:
                available_optimizers = ", ".join(cls.optimizers.keys())
                raise ValidationError(
                    f"Optimizer name '{value}' is not valid. Available optimizers: {available_optimizers}"
                )
        return value

    @validator("mode", always=True)
    def check_and_remove_tmp(cls, mode):
        if mode == "train":
            tmp_path = "tmp"
            if os.path.exists(tmp_path):
                shutil.rmtree(tmp_path)
        return mode


    @root_validator(skip_on_failure=True)
    def exp_name_validator(cls, values):
        mode = values.get("mode")
        exp_name = values.get("exp_name")
        if os.path.exists(f"experiments/{exp_name}") and mode == "train":
            print(f"**** exp_name {exp_name}  already exist *****")
            raise ValidationError(f"exp_name {exp_name}  already exist")
        return values  # return the entire values dict

    @root_validator(skip_on_failure=True)
    def datasheet_path_validator(cls, values):
        datasheet_path = values.get("datasheet_path")
        mode = values.get("mode")
        billy_datasheet_path = values.get("billy_datasheet_path")

        if (not billy_datasheet_path and not datasheet_path) and mode == "train":
            raise ValueError("datasheet_path  and  billy_datasheet_path is None")

        if (
            (billy_datasheet_path is not None and not os.path.exists(billy_datasheet_path))
            and (datasheet_path is not None and not os.path.exists(datasheet_path))
        ) and mode == "train":
            print(f"Datasheet path {billy_datasheet_path}  and {datasheet_path} do not exist")
            raise ValidationError(f"Datasheet path {billy_datasheet_path}  and {datasheet_path} do not exist")
        return values

    class Config:
        arbitrary_types_allowed = True
        allow_extra = False
        allow_mutation = False