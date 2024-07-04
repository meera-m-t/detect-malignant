from pathlib import Path
from pydantic import BaseModel

from detect_malignant.src.configs.config import ExperimentationConfig


class PathsConfig(BaseModel):
    expconfig: ExperimentationConfig

    def get_exp_dir(self, config=None, exp_name=None, path_format="experiments/{}"):
        if config is None and exp_name is None:
            raise ValueError("Either config or exp_name must be provided")
        if exp_name is None:
            exp_name = config.exp_name
        return path_format.format(exp_name)

    def get_experiment_datasheet_path(self, config=None, exp_name=None, path_format="experiments/{}/data.csv"):
        if config is None and exp_name is None:
            raise ValueError("Either config or exp_name must be provided")
        if exp_name is None:
            exp_name = config.exp_name
        return path_format.format(exp_name)

    def get_save_dir(self):        
        return Path(f"{self.expconfig.save_dir}/{self.expconfig.exp_name}/")