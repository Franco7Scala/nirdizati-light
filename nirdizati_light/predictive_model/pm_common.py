from enum import Enum
import numpy as np
import torch
from funcy import flatten
from pandas import DataFrame


class ClassificationMethods(str, Enum):
    """Available classification methods"""
    RANDOM_FOREST = "randomForestClassifier"
    KNN = "knn"
    XGBOOST = "xgboost"
    SGDCLASSIFIER = "SGDClassifier"
    PERCEPTRON = "perceptron"
    LSTM = "lstm"
    CUSTOM_PYTORCH = "customPytorch"
    MLP = "mlp"
    SVM = "svc"
    DT = "DecisionTree"
    TORCH_MOE = "torch_moe"          


class RegressionMethods(str, Enum):
    """Available regression methods"""
    RANDOM_FOREST = "randomForestRegressor"


def get_tensor(df: DataFrame, prefix_length: int) -> np.ndarray:
    """Costruisce il tensore 3D [n_tracce, prefix_length, n_feat] per modelli sequenziali."""
    trace_attributes = [att for att in df.columns if "prefix_" not in att]
    event_attributes = [att[:-2] for att in df.columns if att.endswith("_1")]

    reshaped_data = {
        trace_index: {
            prefix_index: list(flatten(
                feat_values if isinstance(feat_values, tuple) else [feat_values]
                for feat_name, feat_values in trace.items()
                if feat_name in trace_attributes
                or feat_name in [f"{ea}_{prefix_index}" for ea in event_attributes]
            ))
            for prefix_index in range(1, prefix_length + 1)
        }
        for trace_index, trace in df.iterrows()
    }

    flattened_features = max(
        len(reshaped_data[trace][prefix])
        for trace in reshaped_data
        for prefix in reshaped_data[trace]
    )

    tensor = np.zeros((len(df), prefix_length, flattened_features))
    for i, trace_index in enumerate(reshaped_data):
        for j, prefix_index in enumerate(reshaped_data[trace_index]):
            vals = reshaped_data[trace_index][prefix_index]
            tensor[i, j, :len(vals)] = vals

    return tensor


def shape_label_df(df: DataFrame) -> np.ndarray:
    """One-hot delle etichette (assume label numeriche 0..K-1)."""
    labels_arr = np.asarray(df["label"]).astype(int)
    n_classes = int(labels_arr.max()) + 1
    labels = np.zeros((len(labels_arr), n_classes), dtype=float)
    labels[np.arange(len(labels_arr)), labels_arr] = 1.0
    return labels


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
