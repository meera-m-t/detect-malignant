import torch
from tqdm.auto import tqdm
from detect_malignant.src.utils.utils import make_dirs
from detect_malignant.test.test_paths import  get_test_paths



def create_results_dataframe(config, config_train, num_classes, data_loader, model):
        y_true = []        
        y_outputs = []        
        loss_function = config_train.get_loss(num_classes=num_classes)
        test_loss = 0.0
        num_samples = 0   
        model = model.module if len(config.device_ids) > 1 else model
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                images, labels = data               
                images = images.to(config.device)
                labels = labels.to(config.device)         
                outputs = model.forward(images)                    
            
                loss = loss_function(outputs, labels)
                test_loss += loss.item() * images.size(0)
                outputs = torch.max(outputs, 1)[1]  

                num_samples += images.size(0)             
                y_true.extend(labels.cpu().numpy())               
                y_outputs.extend(outputs.cpu().numpy())
   
        return num_samples, y_true, y_outputs, test_loss          


def setup_experiment_environment(experiment_dir, prefix, exp_name):
    test_dir, preds_fn = get_test_paths(experiment_dir, prefix, exp_name)
    make_dirs([test_dir])
    return test_dir, preds_fn


