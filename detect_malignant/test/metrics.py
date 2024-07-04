import math
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from detect_malignant.test.test_paths import (get_confusion_matrix_path, get_metrics_path,
                                                      get_precision_recall_curve_path)



def get_all_metrics(y_true, y_pred, names, experiment_dir, prefix, exp_name, save=True, region=False):
    names = names

    metrics_df, cm_norm = balanced_classification_report(
        y_true, y_pred, names, experiment_dir, prefix, exp_name, save=save
    )   
    return metrics_df, cm_norm


def quick_report(y_true, y_pred):
    correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    print(
        "\nQuick Accuracy for the CNN:",
        correct_predictions / (len(y_true)),
        "\n",
        "\tcorrect predictions:",
        correct_predictions,
        " --\tincorrect predictions:",
        len(y_true) - correct_predictions,
        " --\ttotal predictions:",
        len(y_true),
    )
    return


def balanced_classification_report(y_true, y_pred, names, experiment_dir, prefix, exp_name, save=True):
    _, cm_raw, cm_norm = generate_confusion_matrix(
        y_true,
        y_pred,
        names,
        get_confusion_matrix_path(experiment_dir, prefix, exp_name),
        save=save,
    ) 
    recall = np.diag(cm_norm) / (np.sum(cm_norm, axis=1) + 1e-10)
    precision = np.diag(cm_norm) / (np.sum(cm_norm, axis=0) + 1e-10)

    pred_counts = np.sum(cm_raw, axis=0)
    prec_unbalanced = np.zeros(pred_counts.shape, dtype=float)
    for i in range(len(pred_counts)):
        if pred_counts[i] != 0:
            prec_unbalanced[i] = cm_raw[i, i].astype("float") / pred_counts[i]
        else:
            prec_unbalanced[i] = math.nan
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    f1_unbalanced = 2 * prec_unbalanced * recall / (prec_unbalanced + recall + 1e-10)

    metrics_df = pd.DataFrame(
        {
            "Classes": names,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Precision Unbalanced": prec_unbalanced,
            "F1 Unbalanced": f1_unbalanced,
            "Samples": np.sum(cm_raw, axis=1),
        }
    )

    # metrics_df.to_csv(get_metrics_path(experiment_dir, prefix).replace(".csv", "_pre.csv"))
    avgs = get_macro_and_micro_avgs(metrics_df)
    metrics_df = pd.concat([metrics_df, avgs], ignore_index=True)

    if save:
        metrics_df.to_csv(get_metrics_path(experiment_dir, prefix, exp_name), index=False)
    return metrics_df, cm_norm



def get_macro_and_micro_avgs(metrics):
    skip_cols = ["Samples", "Classes"]
    avgs = pd.DataFrame(np.zeros((3, len(metrics.columns))), columns=metrics.columns)
    avgs.iloc[0] = ""
    avgs["Classes"] = ["", "macro_avg", "micro_avg"]
    for col in metrics.columns:
        if col not in skip_cols:
            avgs[col].iloc[1] = metrics[col].mean()
            for i in range(len(metrics[col])):
                if metrics["Samples"].iloc[i] != 0:
                    avgs[col].iloc[2] += metrics[col].iloc[i] * metrics["Samples"].iloc[i]
            avgs[col].iloc[2] /= metrics["Samples"].sum()
    avgs["Samples"].iloc[1] = len(metrics[col])
    avgs["Samples"].iloc[2] = metrics["Samples"].sum()
    return avgs


def generate_confusion_matrix(y_true, y_pred, names, confusion_matrix_path, save=True):
   
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))

    # Calculate normalized confusion matrix to avoid divide by 0 errors
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = np.true_divide(cm, cm.sum(axis=1)[:, np.newaxis])
        cm_normalized[~np.isfinite(cm_normalized)] = 0  # Replace NaNs with 0

    cm_df = pd.DataFrame(cm_normalized, index=names, columns=names)

    if save:
        cm_df.to_csv(confusion_matrix_path)

        num_classes = len(cm_df)

        # Scale based on the number of classes
        scale_factor = 1      

        fontsize_base = max(5, 8 * scale_factor)
        title_fontsize = fontsize_base + 4
        tick_fontsize = fontsize_base + 4

        # Adjust figure size based on the number of classes
        plt.figure(figsize=(max(5, num_classes * scale_factor), max(5, num_classes * scale_factor)))

        # Display heatmap with values to one decimal place and bold numbers
        hm = sns.heatmap(
            cm_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            square=True,
            annot_kws={"size": fontsize_base, "weight": "bold"},
        )

        # Adjust tick labels font size
        hm.set_xticklabels(hm.get_xticklabels(), size=tick_fontsize, rotation=90, weight="bold")
        hm.set_yticklabels(
            hm.get_yticklabels(), size=tick_fontsize, rotation=0, weight="bold"
        )  # Made y-axis labels horizontal

        plt.xlabel("Predicted", fontsize=fontsize_base + 2, fontweight="bold")
        plt.ylabel("True", fontsize=fontsize_base + 2, fontweight="bold")
        plt.title("Confusion Matrix (Percentages)", fontsize=title_fontsize, fontweight="bold")

        plt.tight_layout()

        # Save the figure as a PDF
        plt.savefig(str(confusion_matrix_path).replace(".csv", ".pdf"), dpi=300)

        # Save the figure as a PNG
        plt.savefig(str(confusion_matrix_path).replace(".csv", ".png"), dpi=300)
        plt.close()

    return cm_df, cm, cm_normalized




