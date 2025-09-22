"""Recurrent neural network classifier."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class RNN:
    @staticmethod
    def _ensure_sequence(sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence)
        if seq.ndim == 1:
            return seq.reshape(1, -1)
        if seq.ndim != 2:
            raise ValueError(f"Expected sequence to be 1D or 2D, received shape {seq.shape}.")
        return seq

    @staticmethod
    def _ensure_target(target: np.ndarray) -> np.ndarray:
        arr = np.asarray(target)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(
                "Target labels must be convertible to shape (1, num_classes)."
            )
        return arr

    def __init__(
        self,
        X_train,
        y_train,
        epochs,
        learning_rate,
        hidden_layer_size,
        one_hot_classes_map,
        sequence_length=None,
        vocab_size=None,
    ):
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.hidden_layer_size = int(hidden_layer_size)
        self.one_hot_classes_map = one_hot_classes_map

        self.X_train = [self._ensure_sequence(np.asarray(seq)) for seq in X_train]
        self.y_train = [self._ensure_target(np.asarray(label)) for label in y_train]

        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train must have the same number of samples.")
        if not self.X_train:
            raise ValueError("Training data cannot be empty.")

        first_sequence = self.X_train[0]
        self.sequence_length = sequence_length or first_sequence.shape[0]
        self.vocab_size = int(vocab_size) if vocab_size is not None else first_sequence.shape[1]

        num_output_classes = self.y_train[0].shape[1]

        # Wxh (input to hidden)
        limit_Wxh = math.sqrt(6 / (self.vocab_size + self.hidden_layer_size))
        self.Wxh = np.random.uniform(-limit_Wxh, limit_Wxh, (self.vocab_size, self.hidden_layer_size))

        # Whh (hidden to hidden)
        limit_Whh = math.sqrt(6 / (self.hidden_layer_size + self.hidden_layer_size))
        self.Whh = np.random.uniform(-limit_Whh, limit_Whh, (self.hidden_layer_size, self.hidden_layer_size))

        # Why (hidden to output)
        limit_Why = math.sqrt(6 / (self.hidden_layer_size + num_output_classes))
        self.Why = np.random.uniform(-limit_Why, limit_Why, (self.hidden_layer_size, num_output_classes))

        # bh (hidden bias)
        self.bh = np.zeros((1, self.hidden_layer_size))

        # by (output bias)
        self.by = np.zeros((1, num_output_classes))

    def softmax(self, x):
        max_val = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max_val)
        sum_val = np.sum(e_x, axis=1, keepdims=True)
        f_x = e_x / sum_val
        return f_x

    def forward(self, X_sequence):
        """
        Performs forward propagation for a single input sequence.
        X_sequence: A 2D numpy array of shape (sequence_length, vocab_size)
                    where each row is a time step.
        """
        X_sequence = self._ensure_sequence(X_sequence)
        h_prev = np.zeros((1, self.hidden_layer_size))

        self.last_inputs = []
        self.last_hs = {} # Store hidden states, h_prev is h_0
        self.last_hs[-1] = np.copy(h_prev) # Store initial h_prev as h_-1 or h_0
        self.last_outputs = []

        for t in range(X_sequence.shape[0]):
            xt = X_sequence[t : t + 1, :]
            self.last_inputs.append(xt)

            # Current hidden state
            ht = np.tanh(np.dot(xt, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh)

            # Output for current time step (raw, without softmax)
            yt = np.dot(ht, self.Why) + self.by

            self.last_hs[t] = ht
            self.last_outputs.append(yt)
            
            h_prev = ht # Update h_prev for the next time step

        return self.last_outputs

    def backward(self, Y_sequence, Y_pred_sequence):
        """
        Performs backpropagation through time (BPTT) for a single sequence.
        Y_sequence: List or array of true target labels for each time step.
                    Each element is expected to be a 1D array or list of class indices
                    or a 2D one-hot encoded array.
        Y_pred_sequence: List of raw output vectors from the forward pass (self.last_outputs).
        """
        num_classes = self.Why.shape[1]

        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((1, self.hidden_layer_size)) # Gradient from next hidden state

        # Iterate backward through time steps
        for t in reversed(range(len(Y_pred_sequence))):
            yt_pred_raw = Y_pred_sequence[t]
            yt_pred_softmax = self.softmax(yt_pred_raw) # Apply softmax to raw output

            # Prepare true label yt_true
            # Assuming Y_sequence[t] is a one-hot encoded vector (1, num_classes)
            # If Y_sequence[t] is a class index, it needs conversion to one-hot.
            # For this implementation, we'll assume Y_sequence[t] is already one-hot.
            yt_true = Y_sequence[t]
            if yt_true.ndim == 1:
                yt_true = yt_true.reshape(1, -1) # Ensure (1, num_classes)
            
            if yt_true.shape[1] != num_classes:
                raise ValueError(f"Shape mismatch for Y_sequence[{t}]. Expected ({num_classes},) or (1, {num_classes}), got {Y_sequence[t].shape}")


            # Error in output (gradient of cross-entropy loss w.r.t. input to softmax)
            dy = yt_pred_softmax - yt_true # Shape (1, num_classes)

            # Retrieve stored values from forward pass
            ht = self.last_hs[t]              # Shape (1, hidden_layer_size)
            h_prev_t = self.last_hs[t-1]      # Shape (1, hidden_layer_size)
            xt = self.last_inputs[t]          # Shape (1, vocab_size)

            # Gradients for output layer
            dWhy += np.dot(ht.T, dy) # (hidden_size, 1) x (1, num_classes) -> (hidden_size, num_classes)
            dby += dy                # (1, num_classes)

            # Gradient of loss w.r.t. current hidden state ht
            dh = np.dot(dy, self.Why.T) + dh_next # (1, num_classes) x (num_classes, hidden_size) + (1, hidden_size) -> (1, hidden_size)

            # Gradient for tanh activation
            dtanh = (1 - ht * ht) * dh # Element-wise: (1 - ht^2) * dh

            # Gradients for hidden layer
            dbh += dtanh                                # (1, hidden_size)
            dWhh += np.dot(h_prev_t.T, dtanh)           # (hidden_size, 1) x (1, hidden_size) -> (hidden_size, hidden_size)
            dWxh += np.dot(xt.T, dtanh)                 # (vocab_size, 1) x (1, hidden_size) -> (vocab_size, hidden_size)
            
            # Update dh_next for the previous time step
            dh_next = np.dot(dtanh, self.Whh.T) # (1, hidden_size) x (hidden_size, hidden_size) -> (1, hidden_size)

        # Clip gradients to prevent explosion
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights and biases
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh  -= self.learning_rate * dbh
        self.by  -= self.learning_rate * dby

    def get_emotion_class(self, one_hot_class):
        """
        Translates a one-hot encoded class vector back to its original string label.
        one_hot_class: A 1D numpy array or list representing the one-hot vector.
        """
        # Ensure one_hot_class is a numpy array for comparison
        one_hot_class_arr = np.asarray(one_hot_class).flatten() 
        for key, value in self.one_hot_classes_map.items():
            # Ensure value from map is also a flat numpy array for comparison
            map_value_arr = np.asarray(value).flatten()
            if np.array_equal(map_value_arr, one_hot_class_arr):
                return key
        return None # Or raise an error if not found

    def predict(self, X_sequence):
        """
        Predicts the class for a single input sequence.
        X_sequence: A 2D numpy array of shape (sequence_length, vocab_size).
        Returns a one-hot encoded prediction vector.
        """
        raw_outputs = self.forward(X_sequence)
        # Assuming many-to-one: prediction is based on the output of the last time step
        last_raw_output = raw_outputs[-1] # Shape (1, num_classes)
        
        probabilities = self.softmax(last_raw_output) # Shape (1, num_classes)
        
        predicted_class_idx = np.argmax(probabilities, axis=1)[0] # Get the index
        
        num_classes = self.Why.shape[1]
        one_hot_prediction = np.zeros((1, num_classes))
        one_hot_prediction[0, predicted_class_idx] = 1
        
        return one_hot_prediction

    def train(self):
        """
        Trains the RNN model.
        Assumes self.X_train is a list of sequences (2D numpy arrays)
        and self.y_train is a list of corresponding one-hot encoded target labels (2D numpy arrays, 1 x num_classes).
        This implements Stochastic Gradient Descent (SGD) as it processes one sample at a time.
        """
        num_classes = self.Why.shape[1]

        for epoch in range(self.epochs):
            for index, (X_sample, Y_target) in enumerate(zip(self.X_train, self.y_train)):
                raw_outputs = self.forward(X_sample)

                sequence_len = X_sample.shape[0]
                Y_sequence_for_bptt = [np.zeros((1, num_classes)) for _ in range(sequence_len - 1)]

                if Y_target.shape[1] != num_classes:
                    raise ValueError(
                        f"Y_target at index {index} has incorrect number of classes. Expected {num_classes}, got {Y_target.shape[1]}"
                    )
                Y_sequence_for_bptt.append(Y_target.reshape(1, num_classes))

                self.backward(Y_sequence_for_bptt, raw_outputs)

            if (epoch + 1) % 100 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}/{self.epochs} completed.")

    def predict_and_print_file(self, id_tweet_vec_sequences, file_name):
        """
        Predicts emotions for a list of tweet sequences and writes the results to a CSV file.
        id_tweet_vec_sequences: A list of tuples, where each tuple is 
                                 (id_val, tweet_text, sequence_vector).
                                 sequence_vector is a 2D numpy array (e.g., (1, vocab_size)).
        file_name: The name of the CSV file to write the results to.
        """
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("id,tweet,emotions\n")
                for id_val, tweet_text, sequence_vec in id_tweet_vec_sequences:
                    sequence_vec_np = self._ensure_sequence(sequence_vec)
                    y_pred_one_hot = self.predict(sequence_vec_np)

                    # self.get_emotion_class expects a 1D array/list
                    pred_emotion = self.get_emotion_class(y_pred_one_hot.flatten()) or "Unknown"

                    # Ensure tweet_text doesn't contain characters that break CSV (like commas within the tweet itself)
                    # A simple replacement, more robust CSV handling might be needed for complex text.
                    processed_tweet_text = str(tweet_text).replace(',', ';')


                    f.write(str(id_val) + "," + processed_tweet_text + "," + str(pred_emotion) + "\n")
            print(f"Successfully wrote predictions to {file_name}")
        except IOError:
            print(f"Error: Could not write to file {file_name}")
        except Exception as e:
            print(f"An unexpected error occurred during predict_and_print_file: {e}")


if __name__ == '__main__':
    # Example Usage (Illustrative)
    # This part is just for basic testing and might be removed or modified later.
    # It requires X_train and y_train to be defined.
    
    # Sample X_train (e.g., 10 sequences, each of 5 one-hot encoded words from a vocab of 20)
    # This is a placeholder. Actual X_train structure will depend on the specific problem.
    # If X_train represents multiple sequences, it might be a list of 2D arrays,
    # or a 3D array. For now, let's assume X_train for __init__ is primarily for shape inference.
    # If X_train is (num_samples, vocab_size) for a non-sequential model or (num_sequences, sequence_len, vocab_size)
    # the current vocab_size inference X_train.shape[1] might need adjustment.
    # Let's assume X_train for shape inference is (some_dimension, vocab_size)
    
    # Example: vocab_size = 20, hidden_layer_size = 50, num_classes = 3
    # X_train_sample = np.random.rand(10, 20) # 10 samples, vocab_size 20
    # y_train_sample = np.random.randint(0, 2, size=(10, 3)) # 10 samples, 3 classes
    # one_hot_map_sample = {'a':0, 'b':1, 'c':2}

    # rnn = RNN(X_train=X_train_sample,
    #           y_train=y_train_sample,
    #           epochs=100,
    #           learning_rate=0.01,
    #           hidden_layer_size=50,
    #           one_hot_classes_map=one_hot_map_sample)

    # print("Wxh shape:", rnn.Wxh.shape)
    # print("Whh shape:", rnn.Whh.shape)
    # print("Why shape:", rnn.Why.shape)
    # print("bh shape:", rnn.bh.shape)
    # print("by shape:", rnn.by.shape)
    # print("Vocab size:", rnn.vocab_size)
    pass
