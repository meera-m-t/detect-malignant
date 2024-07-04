import importlib
import os

import pandas as pd


def get_classes(datasheet, only_training=True):
    df = pd.read_csv(datasheet)
    classes = []
    if only_training:
        df = df[df.Split == "Train"]
    ys = df["y"].unique()
    class_map = {}
    for y in range(max(ys) + 1):
        class_map[y] = df["Species_Name"].loc[df["y"] == y].values[0]
        classes.append(class_map[y])
    # return class_map,len(classes),classes
    return classes


def load_module_from_file(file_path):
    modname = os.path.splitext(os.path.split(file_path)[1])[0]
    spec = importlib.util.spec_from_file_location(f"train.{modname}", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_config(cfg_path: str):
    if os.path.exists(cfg_path):
        config = load_module_from_file(cfg_path).config
        config.cfg_path = cfg_path
    else:
        raise ValueError(f"{cfg_path} does not exist")
    return config


def get_subm_folder(exp_dir: str):
    return os.path.join(exp_dir, "subm")