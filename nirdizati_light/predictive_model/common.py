from enum import Enum
import torch
import numpy as np
from funcy import flatten
from pandas import DataFrame


class ClassificationMethods(Enum):
    """
    Available classification methods
    """
    RANDOM_FOREST = 'randomForestClassifier'
    KNN = 'knn'
    XGBOOST = 'xgboost'
    SGDCLASSIFIER = 'SGDClassifier'
    PERCEPTRON = 'perceptron'
    LSTM = 'lstm'
    CUSTOM_PYTORCH = 'customPytorch'
    MLP = 'mlp'
    SVM = 'svc'
    DT = 'DecisionTree'
    MOE = 'MixtureOfExperts'


class RegressionMethods(Enum):
    """
    Available regression methods
    """
    RANDOM_FOREST = 'randomForestRegressor'


def get_tensor(df: DataFrame, prefix_length):
    trace_attributes = [att for att in df.columns if 'prefix_' not in att]
    event_attributes = [att[:-2] for att in df.columns if att[-2:] == '_1']

    reshaped_data = {
            trace_index: {
                prefix_index:
                    list(flatten(
                        feat_values if isinstance(feat_values, tuple) else [feat_values]
                        for feat_name, feat_values in trace.items()
                        if feat_name in trace_attributes + [event_attribute + '_' + str(prefix_index) for event_attribute in event_attributes]
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

    tensor = np.zeros((
        len(df),                # sample
        prefix_length,          # time steps
        flattened_features      # features x single time step (trace and event attributes)
    ))

    for i, trace_index in enumerate(reshaped_data):  # prefix
        for j, prefix_index in enumerate(reshaped_data[trace_index]):  # steps of the prefix
            for single_flattened_value in range(len(reshaped_data[trace_index][prefix_index])):
                tensor[i, j, single_flattened_value] = reshaped_data[trace_index][prefix_index][single_flattened_value]

    return tensor


def get_tensor_alt(df, prefix_length, aggregate=True, trace_att_filter=['trace_id', 'label'],
                   event_att_filter=['label', 'prefix', 'Prefix']):
    def is_trace_attribute(att):
        try:
            return not int(att[len(att) - att[::-1].index('_'):]) > 0
        except ValueError:
            return True

    def create_numerical_feature_list(trace, selected_attributes):
        feature_count = 0
        numerical_feature_list = []
        # print('trace: ', trace)
        for feat_name, feat_values in trace.items():
            if feat_name in selected_attributes:
                # print('\nisinstance(feat_values, tuple): ', isinstance(feat_values, tuple))
                if isinstance(feat_values, tuple):
                    # print('not inserted feature: ', feat_name, '\n \t with feat_values', feat_values)
                    feature_count += len(feat_values)
                else:
                    numerical_feature_list.append(feature_count)
                    # print('inserting feature: ', feat_name, '\n \t with count ', feature_count)
                    feature_count += 1
        return numerical_feature_list

    trace_attributes = [att for att in df.columns if is_trace_attribute(att)]
    event_attributes = [att[:-2] for att in df.columns if att[-2:] == '_1' and not is_trace_attribute(att)]
    # print('trace_attributes: ', trace_attributes)
    # print('event_attributes: ', event_attributes)

    reshaped_data = {
        trace_index: {
            prefix_index:
                list(flatten(
                    feat_values if isinstance(feat_values, tuple) else [feat_values]
                    for feat_name, feat_values in trace.items()
                    if feat_name in [trace_attribute for trace_attribute in trace_attributes if
                                     trace_attribute not in trace_att_filter] + [
                        event_attribute + '_' + str(prefix_index) for event_attribute in event_attributes if
                        event_attribute not in event_att_filter]
                ))
            for prefix_index in range(1, prefix_length + 1)
        }
        for trace_index, trace in df.iterrows()
    }

    flattened_feature_length = max(
        len(reshaped_data[trace][prefix])
        for trace in reshaped_data
        for prefix in reshaped_data[trace]
    )

    trace_number = len(df)
    tensor = torch.zeros((
        trace_number,  # samples
        prefix_length,  # time steps
        flattened_feature_length  # features per single time step (trace and event attributes)
    ))

    selected_attributes = [trace_attribute for trace_attribute in trace_attributes if
                           trace_attribute not in trace_att_filter] + [event_attribute + '_1' for event_attribute in
                                                                       event_attributes if
                                                                       event_attribute not in event_att_filter]

    numerical_feature_list = create_numerical_feature_list(df.loc[0], selected_attributes) if aggregate else []
    # print('numerical_feature_list: ', numerical_feature_list)

    for i, trace_index in enumerate(reshaped_data):  # prefix
        for j, prefix_index in enumerate(reshaped_data[trace_index]):  # steps of the prefix
            for single_flattened_value in range(len(reshaped_data[trace_index][prefix_index])):
                tensor[i, j, single_flattened_value] = reshaped_data[trace_index][prefix_index][single_flattened_value]
                if aggregate and j in numerical_feature_list:
                    tensor[i, j, single_flattened_value] /= trace_number

    # print('tensor.shape: ', tensor.shape)
    if aggregate:
        return torch.sum(tensor, dim=1)
    else:
        return tensor

def shape_label_df(df: DataFrame):
    labels_list = df['label'].tolist()
    labels = np.zeros((len(labels_list), int(max(df['label'].nunique(), int(max(df['label'].values))) + 1)))
    for label_idx, label_val in enumerate(labels_list):
        labels[int(label_idx), int(label_val)] = 1

    return labels

# General purpose class to wrap a lambda function as a torch module
class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
# Class for early stopping
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False