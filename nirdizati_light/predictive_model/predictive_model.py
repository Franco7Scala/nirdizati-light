import os
import logging
import numpy as np
import torch
from typing import Union, Optional, Type
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from nirdizati_light.evaluation.common import evaluate_classifier, evaluate_regressor
from nirdizati_light.predictive_model.pm_common import (
    ClassificationMethods, RegressionMethods,
    get_tensor, shape_label_df, LambdaModule, EarlyStopper
)
from .torch_moe_model import MOEHead

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    return df.drop(['trace_id', 'label'], axis=1)


class PredictiveModel:
    """
    A class representing a predictive model.
    """

    def __init__(
        self,
        model_type: Union[ClassificationMethods, RegressionMethods, str],
        train_df: DataFrame,
        validate_df: DataFrame,
        test_df: DataFrame,
        prefix_length: int,
        hyperopt_space: Optional[dict] = None,
        custom_model_class: Optional[Type[Module]] = None
    ):
        self.model_type = model_type
        self.config = None
        self.model = None

        self.full_train_df = train_df
        self.train_df = drop_columns(train_df)
        self.train_df_shaped = None

        self.full_validate_df = validate_df
        self.validate_df = drop_columns(validate_df)
        self.validate_df_shaped = None

        self.full_test_df = test_df
        self.test_df = drop_columns(test_df)
        self.test_df_shaped = None

        self.hyperopt_space = hyperopt_space
        self.custom_model_class = custom_model_class

        if model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
            self.train_tensor = get_tensor(self.train_df, prefix_length)
            self.validate_tensor = get_tensor(self.validate_df, prefix_length)
            self.test_tensor = get_tensor(self.test_df, prefix_length)

            self.train_label = shape_label_df(self.full_train_df)
            self.validate_label = shape_label_df(self.full_validate_df)
            self.test_label = shape_label_df(self.full_test_df)

        elif model_type == ClassificationMethods.MLP.value:
            self.train_label = self.full_train_df['label'].nunique()
            self.validate_label = self.full_validate_df['label'].nunique()
            self.test_label = self.full_test_df['label'].unique()

    def train_and_evaluate_configuration(self, config, target):
        try:
            self.model = self._instantiate_model(config)
            self._fit_model(self.model, config)
            actual = self.full_validate_df['label']

            if self.model_type in [
                ClassificationMethods.LSTM.value,
                ClassificationMethods.CUSTOM_PYTORCH.value,
                ClassificationMethods.TORCH_MOE.value
            ]:
                if isinstance(actual.iloc[0], (list, np.ndarray)):
                    actual = np.array(actual.to_list())

            if self.model_type in [item.value for item in ClassificationMethods]:
                predicted, scores = self.predict(test=False)
                result = evaluate_classifier(actual, predicted, scores, loss=target)
            elif self.model_type in [item.value for item in RegressionMethods]:
                predicted = self.model.predict(self.validate_df)
                result = evaluate_regressor(actual, predicted, loss=target)
            else:
                raise Exception('Unsupported model_type')

            return {
                'status': STATUS_OK,
                'loss': - result['loss'],
                'exception': None,
                'config': config,
                'model': self.model,
                'result': result,
            }
        except Exception as e:
            return {
                'status': STATUS_FAIL,
                'loss': 0,
                'exception': str(e)
            }

    def _instantiate_model(self, config):
        if self.model_type == ClassificationMethods.RANDOM_FOREST.value:
            model = RandomForestClassifier(**config)

        elif self.model_type == ClassificationMethods.DT.value:
            model = DecisionTreeClassifier(**config)

        elif self.model_type == ClassificationMethods.KNN.value:
            model = KNeighborsClassifier(**config)

        elif self.model_type == ClassificationMethods.XGBOOST.value:
            model = XGBClassifier(**config)

        elif self.model_type == ClassificationMethods.SGDCLASSIFIER.value:
            model = SGDClassifier(**config)

        elif self.model_type == ClassificationMethods.PERCEPTRON.value:
            base = Perceptron(**config)
            model = CalibratedClassifierCV(base, cv=10, method='isotonic')

        elif self.model_type == ClassificationMethods.MLP.value:
            model = MLPClassifier(**config)

        elif self.model_type == RegressionMethods.RANDOM_FOREST.value:
            model = RandomForestRegressor(**config)

        elif self.model_type == ClassificationMethods.SVM.value:
            model = SVC(**config, probability=True)

        elif self.model_type == ClassificationMethods.LSTM.value:
            model = torch.nn.Sequential(
                torch.nn.LSTM(
                    input_size=self.train_tensor.shape[2],
                    hidden_size=int(config['lstm_hidden_size']),
                    num_layers=int(config['lstm_num_layers']),
                    batch_first=True
                ),
                LambdaModule(lambda x: x[0][:, -1, :]),
                torch.nn.Linear(int(config['lstm_hidden_size']), self.train_label.shape[1]),
                torch.nn.Softmax(dim=1),
            ).to(torch.float32)

        elif self.model_type == ClassificationMethods.CUSTOM_PYTORCH.value:
            model = self.custom_model_class(
                input_dim=self.train_tensor.shape[2],
                output_dim=self.train_label.shape[1],
                config=config,
            ).to(torch.float32)

        elif self.model_type == ClassificationMethods.TORCH_MOE.value:
            input_dim = self.train_df.shape[1]
            num_classes = int(self.full_train_df['label'].nunique())
            num_experts = int(config.get('num_experts', 4))
            hidden = int(config.get('hidden', 128))
            dropout = float(config.get('dropout', 0.1))
            model = MOEHead(
                input_dim=input_dim,
                num_classes=num_classes,
                num_experts=num_experts,
                hidden=hidden,
                dropout=dropout
            )

        else:
            raise Exception('unsupported model_type')

        return model

    def _fit_model(self, model, config=None):
        if self.model_type in [
            ClassificationMethods.LSTM.value,
            ClassificationMethods.CUSTOM_PYTORCH.value,
            ClassificationMethods.TORCH_MOE.value
        ]:
            max_num_epochs = int(config.get('max_num_epochs', 20))
            batch_size = int(config.get('batch_size', 64))
            lr = float(config.get('lr', 1e-3))

            # Dataset: LSTM usa tensori 3D; MoE/CUSTOM PyTorch usano 2D tabellare
            if self.model_type == ClassificationMethods.LSTM.value:
                train_dataset = TensorDataset(
                    torch.tensor(self.train_tensor, dtype=torch.float32),
                    torch.tensor(self.train_label, dtype=torch.float32)
                )
                validate_dataset = TensorDataset(
                    torch.tensor(self.validate_tensor, dtype=torch.float32),
                    torch.tensor(self.validate_label, dtype=torch.float32)
                )
            else:
                Xtr = torch.tensor(self.train_df.values, dtype=torch.float32)
                ytr = torch.tensor(shape_label_df(self.full_train_df), dtype=torch.float32)
                Xva = torch.tensor(self.validate_df.values, dtype=torch.float32)
                yva = torch.tensor(shape_label_df(self.full_validate_df), dtype=torch.float32)
                train_dataset = TensorDataset(Xtr, ytr)
                validate_dataset = TensorDataset(Xva, yva)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()

            early_stopper = EarlyStopper(
                patience=int(config.get('patience', 3)),       # allineato con hyperopt
                min_delta=float(config.get('min_delta', 0.0))  # allineato con hyperopt
            )

            for _ in range(max_num_epochs):
                # training
                model.train()
                for inputs, labels in train_loader:
                    y = torch.argmax(labels, dim=1) if labels.ndim == 2 else labels.long()
                    logits = model(inputs)
                    loss = criterion(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # validation
                model.eval()
                validate_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in validate_loader:
                        y = torch.argmax(labels, dim=1) if labels.ndim == 2 else labels.long()
                        logits = model(inputs)
                        validate_loss += criterion(logits, y).item()
                validate_loss /= max(1, len(validate_loader))

                if early_stopper.early_stop(validate_loss):
                    break

        else:
            model.fit(self.train_df, self.full_train_df['label'])

    def predict(self, test: bool = True):
        """
        Returns predictions and (if available) scores.
        """
        data = self.test_df if test else self.validate_df

        if self.model_type in [
            ClassificationMethods.LSTM.value,
            ClassificationMethods.CUSTOM_PYTORCH.value,
            ClassificationMethods.TORCH_MOE.value
        ]:
            if self.model_type == ClassificationMethods.LSTM.value:
                data_tensor = torch.tensor(self.test_tensor if test else self.validate_tensor, dtype=torch.float32)
            else:
                data_tensor = torch.tensor(data.values, dtype=torch.float32)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(data_tensor)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predicted = np.argmax(probabilities, axis=1)
            scores = probabilities[np.arange(len(probabilities)), predicted]
        else:
            predicted = self.model.predict(data)
            scores = self.model.predict_proba(data)[:, 1] if hasattr(self.model, 'predict_proba') else None

        return predicted, scores

    def save(self, path: str, name: str):
        """
        Save the model to the given path.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        path_with_name = os.path.join(path, name)

        if self.model_type in [
            ClassificationMethods.LSTM.value,
            ClassificationMethods.CUSTOM_PYTORCH.value,
            ClassificationMethods.TORCH_MOE.value
        ]:
            path_with_name += '.pt'
            torch.save(self.model.state_dict(), path_with_name)
        else:
            path_with_name += '.joblib'
            import joblib
            joblib.dump(self.model, path_with_name)

        return path_with_name
