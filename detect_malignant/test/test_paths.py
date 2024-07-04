from pathlib import Path
def get_test_paths(experiment_dir, prefix, exp_name):
    test_dir = f"{experiment_dir}/{prefix}_predictions"
    test_dir = Path(test_dir)
    
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
    data_with_predictions_fn = f"{experiment_dir}/{prefix}_predictions/{exp_name}_{prefix}_data_with_predictions.csv"
    return test_dir, data_with_predictions_fn



def get_confusion_matrix_path(experiment_dir, prefix, exp_name):
    test_dir = Path(f"{experiment_dir}/{prefix}_predictions")
    
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
    
    return f"{experiment_dir}/{prefix}_predictions/{exp_name}_{prefix}_confusion_matrix.csv"


def get_metrics_path(test_dir , prefix, exp_name):
    return f"{test_dir}/{prefix}_predictions/{exp_name}_{prefix}_metrics.csv"



def get_precision_recall_curve_path(experiment_dir, prefix, exp_name):
    return Path(experiment_dir) / f"{prefix}_predictions" / f"{exp_name}_{prefix}_precision_recall_curve.png"