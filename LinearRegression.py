"""Utilities for multinomial logistic regression models."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _as_numpy_2d(array: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
    """Return *array* as a two dimensional numpy array."""

    arr = np.asarray(array)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array, received shape {arr.shape}.")
    return arr


class LiogisticRegression:
    """Simple multinomial logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        X_train: ArrayLike | Sequence[ArrayLike],
        y_train: ArrayLike | Sequence[ArrayLike],
        one_hot_classes_map: Dict[str, np.ndarray],
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.X_train = _as_numpy_2d(X_train)
        self.y_train = _as_numpy_2d(y_train)
        self.one_hot_classes_map = one_hot_classes_map

        self._seed_weights(self.X_train.shape[1], self.y_train.shape[1])

    def _seed_weights(self, n_features: int, n_classes: int) -> None:
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, n_classes))
        self.b = np.zeros((1, n_classes))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_2d = _as_numpy_2d(z)
        max_vals = np.max(z_2d, axis=1, keepdims=True)
        exp = np.exp(z_2d - max_vals)
        sums = np.sum(exp, axis=1, keepdims=True)
        return exp / sums

    def predict_proba(self, X: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
        X_2d = _as_numpy_2d(X)
        logits = np.dot(X_2d, self.W) + self.b
        return self._softmax(logits)

    def predict_one_hot_class(self, X: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(one_hot.shape[0]), predictions] = 1
        return one_hot

    def predict_and_get_accuracy(
        self,
        X_test: ArrayLike | Sequence[ArrayLike],
        y_test: ArrayLike | Sequence[ArrayLike],
    ) -> float:
        X_arr = _as_numpy_2d(X_test)
        y_arr = _as_numpy_2d(y_test)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        predictions = self.predict_one_hot_class(X_arr)
        matches = np.all(predictions == y_arr, axis=1)
        return float(np.mean(matches) * 100) if len(matches) else 0.0

    def update_weights(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        self.W = self.W - self.learning_rate * np.dot(X.T, (y_pred - y))
        self.b = self.b - self.learning_rate * np.sum(y_pred - y, axis=0, keepdims=True)

    def train(self) -> None:
        for _ in range(self.epochs):
            predicted = self.predict_proba(self.X_train)
            self.update_weights(self.X_train, self.y_train, predicted)

    def get_emotion_class(self, one_hot_class: np.ndarray) -> str | None:
        one_hot_flat = np.asarray(one_hot_class).flatten()
        for label, encoding in self.one_hot_classes_map.items():
            if np.array_equal(np.asarray(encoding).flatten(), one_hot_flat):
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
                predicted = self.predict_one_hot_class(vec)[0]
                emotion = self.get_emotion_class(predicted) or "Unknown"
                file.write(f"{tweet_id},{str(tweet_text).replace(',', ';')},{emotion}\n")


class CrossValidationTrain:
    """Performs simple K-Fold cross validation for logistic regression."""

    def __init__(
        self,
        k: int,
        X_train: ArrayLike | Sequence[ArrayLike],
        Y_train: ArrayLike | Sequence[ArrayLike],
        vocab_map: Dict[str, np.ndarray],
        learning_rate: float,
        epochs: int,
    ) -> None:
        self.k = int(k)
        self.vocab_map = vocab_map
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)

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
            model = LiogisticRegression(
                self.learning_rate,
                self.epochs,
                X_train_fold,
                Y_train_fold,
                self.vocab_map,
            )
            model.train()
            training_scores.append(model.predict_and_get_accuracy(X_train_fold, Y_train_fold))
            testing_scores.append(model.predict_and_get_accuracy(X_test_fold, Y_test_fold))

        return float(np.mean(training_scores)), float(np.mean(testing_scores))


def _run_default_training() -> None:
    from DataRepresentationBuilder import DataRepresentationBuilder

    data_builder = DataRepresentationBuilder("train.csv")
    X_train, Y_train, vocab_map = data_builder.get_training_set_and_vocab()
    id_tweet_vec = data_builder.get_test_set_representation()

    model = LiogisticRegression(0.1, 5000, X_train, Y_train, vocab_map)
    model.train()
    model.predict_and_print_file(id_tweet_vec, "output.csv")


if __name__ == "__main__":
    _run_default_training()
