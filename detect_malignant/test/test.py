import json
import torch
import pandas as pd
import warnings


from detect_malignant.src.configs.config import ExperimentationConfig
from detect_malignant.src.configs.paths_config import PathsConfig
from detect_malignant.src.configs.train_config import TrainConfig
from detect_malignant.src.configs.train_config import ExperimentationConfig
from detect_malignant.test.metrics import (get_all_metrics,  quick_report)
from detect_malignant.test.test_utils import  (create_results_dataframe, setup_experiment_environment)

import warnings
# To ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")


def test_model(
    config: ExperimentationConfig,   
    config_train: TrainConfig,
    model: torch.nn.Module, 
    num_classes: int,
    data_loader: torch.utils.data.DataLoader,
    data_df: pd.DataFrame,
    experiment_dir: str,
    prefix="test"):


    names = ['Benign', 'Malignant']
    exp_name = config.exp_name
    model.eval()
    num_samples = len(data_loader.dataset)
    num_samples, y_true, y_pred, test_loss = create_results_dataframe(config, config_train, num_classes,
                                                                            data_loader, model
                                                                                            ) 

   
    _, preds_fn = setup_experiment_environment(experiment_dir, prefix, exp_name) 
  
    test_loss /= num_samples
    print(f"Test loss: {test_loss:.4f}")
    quick_report(y_true, y_pred)
    # get all metrics
    get_all_metrics(y_true, y_pred, names, experiment_dir, prefix, exp_name, save=True)    
    # Add predictions to the test data and save as CSV
    data_df["predictions"] = y_pred
    data_df.to_csv(preds_fn, index=False)





def test(experiment_dir: str, config=None):
    with open(experiment_dir + "/TrainConfig.json", "r") as json_file:
        if config is None:
            settings_json = json.load(json_file)
            settings_json["mode"] = "test"  # Change the mode to "test"
            print(settings_json["mode"])
            config = ExperimentationConfig.parse_obj(settings_json)
        config_paths = PathsConfig(expconfig=config)
        config_train = TrainConfig(expconfig=config, paths_config=config_paths)
        # Todo: put the below in a function in config_train (maybe called pred model?)
        Model = config_train.get_model()

        kwargs = config.model_kwargs
        num_classes = config_train.get_num_classes(f"{experiment_dir}/data.csv")     
        model = Model(**kwargs)
        device = config.device
        model = model.to(device)
        

        model.load_state_dict(torch.load(experiment_dir + "/best_acc.pth"))      

        if len(config.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=config.device_ids)

     
        test_set = config_train.get_dataset(mode="Test")
        data_df = pd.read_csv("tmp/current_run_Test_dataset.csv")
        custom_prefix = "test"

        test_loader = torch.utils.data.DataLoader(
            test_set,
            config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        print("testing {} on the general class map.".format(custom_prefix))
        test_model(
            config, 
            config_train,        
            model,
            num_classes,
            data_loader=test_loader,   
            data_df= data_df,       
            experiment_dir=experiment_dir,
            prefix=custom_prefix
        )
       
