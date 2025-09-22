"""Entry point for training the available text classification models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from AttentionModel import AttentionModel
from DataRepresentationBuilder import DataRepresentationBuilder
from LinearRegression import LiogisticRegression
from NeuralNetwork import MultiLayerNN
from RNN import RNN


ArrayLike = np.ndarray | Sequence[float]


@dataclass
class Dataset:
    """Container for the features, labels and metadata needed to train models."""

    features: List[np.ndarray]
    labels: List[np.ndarray]
    vocab_map: Dict[str, np.ndarray]
    test_vectors: List[Tuple[str, str, np.ndarray]]

    def as_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.features), np.asarray(self.labels)

    @property
    def vocab_size(self) -> int:
        return len(self.features[0]) if self.features else 0

    @property
    def num_classes(self) -> int:
        return len(self.labels[0]) if self.labels else 0


def load_dataset(train_path: str = "train.csv") -> Dataset:
    builder = DataRepresentationBuilder(train_path)
    features, labels, vocab_map = builder.get_training_set_and_vocab()
    test_vectors = builder.get_test_set_representation()

    feature_arrays = [np.asarray(vector) for vector in features]
    label_arrays = [np.asarray(label) for label in labels]
    return Dataset(feature_arrays, label_arrays, vocab_map, test_vectors)


def train_logistic_regression(
    dataset: Dataset,
    learning_rate: float = 0.0008,
    epochs: int = 2000,
    output_file: str = "test_lr.csv",
) -> LiogisticRegression:
    X_train, Y_train = dataset.as_matrices()
    model = LiogisticRegression(learning_rate, epochs, X_train, Y_train, dataset.vocab_map)
    model.train()
    model.predict_and_print_file(dataset.test_vectors, output_file)
    return model


def train_multilayer_nn(
    dataset: Dataset,
    learning_rate: float = 0.001,
    epochs: int = 2000,
    hidden_layer_size: int = 10,
    output_file: str = "test_nn.csv",
) -> MultiLayerNN:
    X_train, Y_train = dataset.as_matrices()
    model = MultiLayerNN(X_train, Y_train, epochs, learning_rate, hidden_layer_size, dataset.vocab_map)
    model.train()
    model.predict_and_print_file(dataset.test_vectors, output_file)
    return model


def _prepare_rnn_training_data(dataset: Dataset) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    sequences = [np.asarray(vec).reshape(1, -1) for vec in dataset.features]
    targets = [np.asarray(label).reshape(1, -1) for label in dataset.labels]
    return sequences, targets


def _prepare_rnn_test_data(
    test_vectors: Sequence[Tuple[str, str, ArrayLike]]
) -> List[Tuple[str, str, np.ndarray]]:
    prepared = []
    for tweet_id, tweet_text, vec in test_vectors:
        array = np.asarray(vec).reshape(1, -1)
        prepared.append((tweet_id, tweet_text, array))
    return prepared


def train_rnn_model(
    dataset: Dataset,
    learning_rate: float = 0.001,
    epochs: int = 100,
    hidden_layer_size: int = 64,
    output_file: str = "test_rnn.csv",
) -> RNN | None:
    sequences, targets = _prepare_rnn_training_data(dataset)
    if not sequences or not targets:
        print("RNN: Training data is empty. Skipping model training.")
        return None

    model = RNN(
        X_train=sequences,
        y_train=targets,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_layer_size=hidden_layer_size,
        one_hot_classes_map=dataset.vocab_map,
        vocab_size=dataset.vocab_size,
    )
    model.train()
    model.predict_and_print_file(_prepare_rnn_test_data(dataset.test_vectors), output_file)
    return model


def train_attention_model(
    dataset: Dataset,
    attention_dim: int = 64,
    learning_rate: float = 0.001,
    epochs: int = 50,
    output_file: str = "test_attention.csv",
) -> AttentionModel | None:
    if not dataset.features or not dataset.labels:
        print("Attention Model: Training data is empty. Skipping model training.")
        return None

    queries = [np.asarray(vec) for vec in dataset.features]
    key_values = [vec.reshape(1, -1) for vec in queries]
    targets = [np.asarray(label).reshape(1, -1) for label in dataset.labels]

    model = AttentionModel(
        input_dim=queries[0].shape[0],
        attention_dim=attention_dim,
        output_dim=targets[0].shape[1],
        one_hot_classes_map=dataset.vocab_map,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    model.train(queries, key_values, targets)

    test_sequences = [
        (
            tweet_id,
            tweet_text,
            np.asarray(vec),
            np.asarray(vec).reshape(1, -1),
        )
        for tweet_id, tweet_text, vec in dataset.test_vectors
    ]
    model.predict_and_print_file(test_sequences, output_file)
    return model


def main() -> None:
    dataset = load_dataset()

    print("..................Beginning of Logistic Regression................")
    train_logistic_regression(dataset)
    print("..................End of Logistic Regression................\n")

    print("..................Beginning of Neural Network................")
    train_multilayer_nn(dataset)
    print("..................End of Neural Network................\n")

    print("..................Beginning of RNN Model................")
    train_rnn_model(dataset)
    print("..................End of RNN Model................\n")

    print("..................Beginning of Attention Model................")
    train_attention_model(dataset)
    print("..................End of Attention Model................")


if __name__ == "__main__":
    main()
