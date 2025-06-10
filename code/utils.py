import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch


def value_dist(dict_data):
    dict_res = {}
    for name in dict_data:
        data = dict_data[name].value_counts()
        n = data.sum()
        data_counts = {
            key: f"{value} ({round(value/n*100, 1)}%)"
            for key, value in data.to_dict().items()
        }
        dict_res[name] = data_counts
    return pd.DataFrame(dict_res)


def calculate_performance(arr_gs, arr_preds, arr_labels, col_label, df_data, df_train_data):
    dict_perf = {
        'precision': precision_score(
            y_true=arr_gs,
            y_pred=arr_preds,
            average=None,
            labels=arr_labels,
            zero_division=0.0
        ),
        'recall': recall_score(
            y_true=arr_gs,
            y_pred=arr_preds,
            average=None,
            labels=arr_labels,
            zero_division=0.0
        ),
        'f1': f1_score(
            y_true=arr_gs,
            y_pred=arr_preds,
            average=None,
            labels=arr_labels,
            zero_division=0.0
        )
    }
    arr_res = []
    for i in range(len(arr_labels)):
        label = arr_labels[i]
        p, r, f1 = dict_perf['precision'][i], dict_perf['recall'][i], dict_perf['f1'][i]
        n_train = len(df_train_data[df_train_data[col_label] == label])
        n_eval = len(df_data[df_data[col_label] == label])
        arr_res.append({
            "label": label,
            "precision": p,
            "recall": r,
            "f1": f1,
            "n_train": n_train,
            "n_val": n_eval
        })
    df_perf = pd.DataFrame(arr_res)

    df_perf.sort_values(
        by=["f1", "recall", "precision"],
        ascending=True
    )

    return df_perf


# BERT

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics_text_class(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(
        labels,
        preds,
        average='weighted',
        zero_division=.0
    )
    precision = precision_score(
        labels,
        preds,
        average='weighted',
        zero_division=.0
    )
    recall = recall_score(
        labels,
        preds,
        average='weighted',
        zero_division=.0
    )
    accuracy = accuracy_score(
        labels,
        preds
    )
    return {
        'accuracy': round(accuracy * 100, 1),
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1': round(f1 * 100, 1)
    }