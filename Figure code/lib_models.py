# Copy of the file in the Figure code folder, unfortunately
import json
import os
import pickle
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple

import numpy
import pandas
import scipy.special
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ModelInputOutput(NamedTuple):
    cell_type_mapping: List[str]
    input_mapping: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "cell_type_mapping": self.cell_type_mapping,
            "input_mapping": self.input_mapping
        }

    @staticmethod
    def from_dict(dictionary: Dict[str, List[str]]) -> "ModelInputOutput":
        return ModelInputOutput(cell_type_mapping=dictionary["cell_type_mapping"],
                                input_mapping=dictionary["input_mapping"])


class OurModel(ABC):

    @abstractmethod
    def get_input_output(self) -> ModelInputOutput:
        """Describes the input and output of the model: what input parameters does it need, and what cell types does it
        output?"""
        ...

    @abstractmethod
    def fit(self, x_train: ndarray, y_train: ndarray, class_weights: Dict[int, float], epochs: int = 1):
        """Fits the model to the training data."""
        ...

    @abstractmethod
    def evaluate(self, x_test: ndarray, y_test: ndarray) -> Dict[str, float]:
        """Returns evaluation metrics, like "accuracy"."""
        ...

    @abstractmethod
    def save(self, folder: str):
        """Saves the model to a folder."""
        ...

    @abstractmethod
    def predict(self, x_values: ndarray) -> ndarray:
        """Gives you the probability of all cell types."""
        ...


class _KerasModel(OurModel):
    _input_output: ModelInputOutput
    _keras_model: "keras.Model"

    def __init__(self, input_output: ModelInputOutput, keras_model: "keras.Model"):
        self._input_output = input_output
        self._keras_model = keras_model

    def fit(self, x_train: ndarray, y_train: ndarray, class_weights: Dict[int, float], epochs: int = 1):
        self._keras_model.fit(x_train, y_train,
                              epochs=epochs, class_weight=class_weights, verbose=False)

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> Dict[str, float]:
        return self._keras_model.evaluate(x_test, y_test, return_dict=True, verbose=0)

    def save(self, folder: str):
        self._keras_model.save(os.path.join(folder, "model.keras"))
        with open(os.path.join(folder, "settings.json"), "w") as handle:
            json.dump({
                "type": "cell_type_from_manual_features",
                "cell_types": self._input_output.cell_type_mapping,
                "input_parameters": self._input_output.input_mapping
            }, handle)

    def predict(self, x_values: ndarray) -> ndarray:
        return self._keras_model.predict(x_values)

    def get_input_output(self) -> ModelInputOutput:
        return self._input_output


class _LinearModel(OurModel):
    _input_output: ModelInputOutput
    scaler: StandardScaler
    logistic_regression: LogisticRegression

    def __init__(self, input_output: ModelInputOutput, scaler: StandardScaler, regressor: LogisticRegression):
        self._input_output = input_output
        self.scaler = scaler
        self.logistic_regression = regressor

    def get_input_output(self) -> ModelInputOutput:
        return self._input_output

    def fit(self, x_train: ndarray, y_train: ndarray, class_weights: Dict[int, float], epochs: int = 1):
        class_weights_array = numpy.zeros(y_train.shape[0], dtype=numpy.float64)
        for index, output_class in enumerate(y_train):
            class_weights_array[index] = class_weights[output_class]

        self.logistic_regression.fit(self.scaler.transform(x_train), y_train, sample_weight=class_weights_array)

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> Dict[str, float]:
        return {
            "accuracy": self.logistic_regression.score(self.scaler.transform(x_test), y_test)
        }

    def save(self, folder: str):
        pickle_file = os.path.join(folder, "linear_model_pickled.sav")
        with open(pickle_file, "wb") as handle:
            pickle.dump(self._input_output.to_dict(), handle)
            pickle.dump(self.scaler, handle)
            pickle.dump(self.logistic_regression, handle)

    def predict(self, x_values: ndarray) -> ndarray:
        # Scale and clip the input array
        x_values = numpy.clip(self.scaler.transform(x_values), -3, 3)

        # Do the prediction, obtaining logits
        probabilities = self.logistic_regression.decision_function(x_values)

        # Convert from logit to probabilities
        scipy.special.expit(probabilities, out=probabilities)

        # Normalize
        probabilities /= probabilities.sum(axis=1).reshape((probabilities.shape[0], -1))

        return probabilities


class _RandomForestModel(OurModel):
    """Uses a random forest to classify cell types."""

    _input_output: ModelInputOutput
    _random_forest: RandomForestClassifier

    def __init__(self, input_output: ModelInputOutput, random_forest: RandomForestClassifier):
        self._input_output = input_output
        self._random_forest = random_forest

    def get_input_output(self) -> ModelInputOutput:
        return self._input_output

    def fit(self, x_train: ndarray, y_train: ndarray, class_weights: Dict[int, float], epochs: int = 1):
        class_weights_array = numpy.zeros(y_train.shape[0], dtype=numpy.float64)
        for index, output_class in enumerate(y_train):
            class_weights_array[index] = class_weights[output_class]
        self._random_forest.fit(x_train, y_train, sample_weight=class_weights_array)

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> Dict[str, float]:
        return {
            "accuracy": self._random_forest.score(x_test, y_test)
        }

    def save(self, folder: str):
        pickle_file = os.path.join(folder, "random_forest_pickled.sav")
        with open(pickle_file, "wb") as handle:
            pickle.dump(self._input_output.to_dict(), handle)
            pickle.dump(self._random_forest, handle)

    def predict(self, x_values: ndarray) -> ndarray:
        return self._random_forest.predict_proba(x_values)


class _GradientBoostingModel(OurModel):
    """Uses a gradient boosting model to classify cell types."""

    _input_output: ModelInputOutput
    _gradient_boosting: GradientBoostingClassifier

    def __init__(self, input_output: ModelInputOutput, gradient_boosting: GradientBoostingClassifier):
        self._input_output = input_output
        self._gradient_boosting = gradient_boosting

    def get_input_output(self) -> ModelInputOutput:
        return self._input_output

    def fit(self, x_train: ndarray, y_train: ndarray, class_weights: Dict[int, float], epochs: int = 1):
        class_weights_array = numpy.zeros(y_train.shape[0], dtype=numpy.float64)
        for index, output_class in enumerate(y_train):
            class_weights_array[index] = class_weights[output_class]
        self._gradient_boosting.fit(x_train, y_train, sample_weight=class_weights_array)

    def evaluate(self, x_test: ndarray, y_test: ndarray) -> Dict[str, float]:
        return {
            "accuracy": self._gradient_boosting.score(x_test, y_test)
        }

    def save(self, folder: str):
        pickle_file = os.path.join(folder, "gradient_boosting_pickled.sav")
        with open(pickle_file, "wb") as handle:
            pickle.dump(self._input_output.to_dict(), handle)
            pickle.dump(self._gradient_boosting, handle)

    def predict(self, x_values: ndarray) -> ndarray:
        return self._gradient_boosting.predict_proba(x_values)


def load_model(folder: str) -> OurModel:
    # Needed to load pickle files
    dir_name = os.path.dirname(__file__)
    if dir_name not in sys.path:
        sys.path.append(dir_name)

    pickle_file_for_linear_model = os.path.join(folder, "linear_model_pickled.sav")
    if os.path.exists(pickle_file_for_linear_model):
        with open(pickle_file_for_linear_model, "rb") as handle:
            input_output = ModelInputOutput.from_dict(pickle.load(handle))
            scaler = pickle.load(handle)
            regressor = pickle.load(handle)
        return _LinearModel(input_output, scaler, regressor)

    pickle_file_for_random_forest = os.path.join(folder, "random_forest_pickled.sav")
    if os.path.exists(pickle_file_for_random_forest):
        with open(pickle_file_for_random_forest, "rb") as handle:
            input_output = ModelInputOutput.from_dict(pickle.load(handle))
            random_forest = pickle.load(handle)
        return _RandomForestModel(input_output, random_forest)

    pickle_file_for_gradient_boosting = os.path.join(folder, "gradient_boosting_pickled.sav")
    if os.path.exists(pickle_file_for_gradient_boosting):
        with open(pickle_file_for_gradient_boosting, "rb") as handle:
            input_output = ModelInputOutput.from_dict(pickle.load(handle))
            gradient_boosting = pickle.load(handle)
        return _GradientBoostingModel(input_output, gradient_boosting)

    if os.path.exists(os.path.join(folder, "settings.json")):
        import tensorflow
        keras_model = tensorflow.keras.models.load_model(folder)
        with open(os.path.join(folder, "settings.json")) as handle:
            settings = json.load(handle)
            cell_types = settings["cell_types"]
            input_parameters = settings["input_parameters"]
        return _KerasModel(ModelInputOutput(cell_type_mapping=cell_types, input_mapping=input_parameters),
                           keras_model)
    raise ValueError(f"No model found in {folder}.")



def build_gradient_boosting(input_output: ModelInputOutput, n_estimators: int = 100) -> OurModel:
    """Builds a gradient boosting model. It will initially be untrained, call OurModel.fit to train it."""
    return _GradientBoostingModel(input_output, GradientBoostingClassifier(n_estimators=n_estimators, random_state=1))


def build_random_forest(input_output: ModelInputOutput, tree_count: int = 100) -> OurModel:
    """Builds a random forest model. It will initially be untrained, call OurModel.fit to train it."""
    return _RandomForestModel(input_output, RandomForestClassifier(max_features="sqrt", n_estimators=tree_count, random_state=1))


def calculate_class_weights(cell_types: pandas.Series) -> Dict[int, float]:
    counts_by_type = dict()

    array = cell_types.cat.codes.array
    for i in numpy.unique(array):
        count = numpy.count_nonzero(array == i)
        counts_by_type[i] = count

    max_count = max(counts_by_type.values())
    weights_by_type = dict()
    for i, count in counts_by_type.items():
        weights_by_type[i] = min(max_count / count, 2)

    return weights_by_type


def build_shallow_model(input_output: ModelInputOutput, x_train: ndarray, hidden_neurons: int) -> OurModel:
    """Builds a shallow model, so with one hidden layer. If hidden_neurons==0, then we have no hidden layer at all, and
    instead we'll have a linear model."""
    if hidden_neurons == 0:
        # Just use a linear classifier, solved analytically
        scaler = StandardScaler()
        scaler.fit(x_train)
        logistic_regression = LogisticRegression(max_iter=1000)
        return _LinearModel(input_output, scaler, logistic_regression)

    # Build a shallow Keras model
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)

    model = keras.models.Sequential([
        normalizer,
        keras.layers.Dense(hidden_neurons, activation="relu"),
        keras.layers.Dense(len(input_output.cell_type_mapping))
    ])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return _KerasModel(input_output, model)


def build_deep_model(input_output: ModelInputOutput, x_train: ndarray, hidden_neurons: int, hidden_layers: int) -> OurModel:
    """Builds a deep model, so with multiple hidden layers. If hidden_neurons==0, then we have no hidden layer at all, and
    instead we'll have a linear model."""
    if hidden_neurons == 0:
        # Just use a linear classifier, solved analytically
        scaler = StandardScaler()
        scaler.fit(x_train)
        logistic_regression = LogisticRegression(max_iter=1000)
        return _LinearModel(input_output, scaler, logistic_regression)

    # Build a deep Keras model
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train)

    layers = [normalizer]
    for _ in range(hidden_layers):
        layers.append(keras.layers.Dense(hidden_neurons, activation="relu"))
    layers.append(keras.layers.Dense(len(input_output.cell_type_mapping)))

    model = keras.models.Sequential(layers)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    return _KerasModel(input_output, model)