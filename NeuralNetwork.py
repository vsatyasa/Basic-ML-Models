"""Feed-forward neural network utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray | Sequence[float]


def _as_numpy_2d(array: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, received shape {arr.shape}.")
    return arr


class MultiLayerNN:
    """Simple two layer feed-forward neural network."""

    class Layer:
        def __init__(self, input_size: int, output_size: int) -> None:
            limit = 1 / math.sqrt(input_size)
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
            self.bias = np.random.rand(1, output_size)
            self.last_input: np.ndarray | None = None
            self.last_output: np.ndarray | None = None

        def forward(self, X: np.ndarray) -> np.ndarray:
            return np.dot(X, self.weights) + self.bias

        def activation(self, X: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-X))

        def forward_with_activation(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            linear = self.forward(X)
            activated = self.activation(linear)
            self.last_input = X
            self.last_output = activated
            return activated, linear

    def __init__(
        self,
        X_train: ArrayLike | Sequence[ArrayLike],
        y_train: ArrayLike | Sequence[ArrayLike],
        epochs: int,
        learning_rate: float,
        hidden_layer_size: int,
        one_hot_classes_map: Dict[str, np.ndarray],
    ) -> None:
        self.X_train = _as_numpy_2d(X_train)
        self.y_train = _as_numpy_2d(y_train)
        self.one_hot_classes_map = one_hot_classes_map
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)

        self.layers = [
            self.Layer(self.X_train.shape[1], hidden_layer_size),
            self.Layer(hidden_layer_size, self.y_train.shape[1]),
        ]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_2d = _as_numpy_2d(x)
        max_vals = np.max(x_2d, axis=1, keepdims=True)
        exp = np.exp(x_2d - max_vals)
        sums = np.sum(exp, axis=1, keepdims=True)
        return exp / sums

    @staticmethod
    def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def forward_propagation(self, X: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
        inputs = _as_numpy_2d(X)
        first_out, _ = self.layers[0].forward_with_activation(inputs)
        _, second_linear = self.layers[1].forward_with_activation(first_out)
        return self._softmax(second_linear)

    def backward_propagation(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        last_error = y_pred - y_true
        last_delta = last_error
        last_input = self.layers[1].last_input
        assert last_input is not None
        self.layers[1].weights -= self.learning_rate * np.dot(last_input.T, last_delta)
        self.layers[1].bias -= self.learning_rate * np.sum(last_delta, axis=0, keepdims=True)

        first_error = np.dot(last_delta, self.layers[1].weights.T)
        first_output = self.layers[0].last_output
        assert first_output is not None
        first_delta = first_error * self._sigmoid_derivative(first_output)
        first_input = self.layers[0].last_input
        assert first_input is not None
        self.layers[0].weights -= self.learning_rate * np.dot(first_input.T, first_delta)
        self.layers[0].bias -= self.learning_rate * np.sum(first_delta, axis=0, keepdims=True)

    def train(self) -> None:
        for _ in range(self.epochs):
            outputs = self.forward_propagation(self.X_train)
            self.backward_propagation(self.y_train, outputs)

    def predict(self, X: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
        probabilities = self.forward_propagation(X)
        indices = np.argmax(probabilities, axis=1)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(one_hot.shape[0]), indices] = 1
        return one_hot

    def test_and_get_accuracy(
        self,
        X_test: ArrayLike | Sequence[ArrayLike],
        y_test: ArrayLike | Sequence[ArrayLike],
    ) -> float:
        X_arr = _as_numpy_2d(X_test)
        y_arr = _as_numpy_2d(y_test)
        predictions = self.predict(X_arr)
        matches = np.all(predictions == y_arr, axis=1)
        return float(np.mean(matches) * 100) if len(matches) else 0.0

    def get_emotion_class(self, one_hot_class: np.ndarray) -> str | None:
        target = np.asarray(one_hot_class).flatten()
        for label, encoding in self.one_hot_classes_map.items():
            if np.array_equal(np.asarray(encoding).flatten(), target):
                return label
        return None

    def predict_and_print_file(
        self,
        id_tweet_vec: Sequence[Tuple[str, str, ArrayLike]],
        file_name: str,
    ) -> None:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write("id,tweet,emotions\n")
            for tweet_id, tweet_text, vec in id_tweet_vec:
                predicted = self.predict(vec)[0]
                emotion = self.get_emotion_class(predicted) or "Unknown"
                file.write(f"{tweet_id},{str(tweet_text).replace(',', ';')},{emotion}\n")


class CrossValidationTrain:
    def __init__(
        self,
        k: int,
        X_train: ArrayLike | Sequence[ArrayLike],
        Y_train: ArrayLike | Sequence[ArrayLike],
        vocab_map: Dict[str, np.ndarray],
        epochs: int,
        learning_rate: float,
        hidden_layer_size: int = 10,
    ) -> None:
        self.k = int(k)
        self.vocab_map = vocab_map
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.hidden_layer_size = int(hidden_layer_size)
        self.x_folds, self.y_folds = self._divide_into_folds(X_train, Y_train)

    def _divide_into_folds(
        self,
        X_train: ArrayLike | Sequence[ArrayLike],
        Y_train: ArrayLike | Sequence[ArrayLike],
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
        X_arr = _as_numpy_2d(X_train)
        Y_arr = _as_numpy_2d(Y_train)
        indices = np.arange(len(X_arr))
        np.random.shuffle(indices)
        x_folds: List[List[np.ndarray]] = [[] for _ in range(self.k)]
        y_folds: List[List[np.ndarray]] = [[] for _ in range(self.k)]
        for fold_index, fold in enumerate(np.array_split(indices, self.k)):
            for idx in fold:
                x_folds[fold_index].append(X_arr[idx])
                y_folds[fold_index].append(Y_arr[idx])
        return x_folds, y_folds

    def _train_test_split(self, fold_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_x, train_y, test_x, test_y = [], [], [], []
        for idx in range(self.k):
            if idx == fold_index:
                test_x.extend(self.x_folds[idx])
                test_y.extend(self.y_folds[idx])
            else:
                train_x.extend(self.x_folds[idx])
                train_y.extend(self.y_folds[idx])
        return (
            _as_numpy_2d(train_x),
            _as_numpy_2d(train_y),
            _as_numpy_2d(test_x),
            _as_numpy_2d(test_y),
        )

    def run(self) -> Tuple[float, float]:
        training_scores: List[float] = []
        testing_scores: List[float] = []
        for fold_idx in range(self.k):
            X_train_fold, Y_train_fold, X_test_fold, Y_test_fold = self._train_test_split(fold_idx)
            model = MultiLayerNN(
                X_train_fold,
                Y_train_fold,
                self.epochs,
                self.learning_rate,
                self.hidden_layer_size,
                self.vocab_map,
            )
            model.train()
            training_scores.append(model.test_and_get_accuracy(X_train_fold, Y_train_fold))
            testing_scores.append(model.test_and_get_accuracy(X_test_fold, Y_test_fold))
        return float(np.mean(training_scores)), float(np.mean(testing_scores))


def _run_default_training() -> None:
    from DataRepresentationBuilder import DataRepresentationBuilder

    data_builder = DataRepresentationBuilder("train.csv")
    X_train, Y_train, vocab_map = data_builder.get_training_set_and_vocab()
    id_tweet_vec = data_builder.get_test_set_representation()

    model = MultiLayerNN(X_train, Y_train, epochs=5000, learning_rate=0.001, hidden_layer_size=10, one_hot_classes_map=vocab_map)
    model.train()
    model.predict_and_print_file(id_tweet_vec, "output.csv")


if __name__ == "__main__":
    _run_default_training()
