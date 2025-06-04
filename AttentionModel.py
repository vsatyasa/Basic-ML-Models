import numpy as np
import math

class AttentionLayer():
    def __init__(self, input_dim, attention_dim):
        self.W_query = np.random.randn(input_dim, attention_dim) * 0.1
        self.W_key = np.random.randn(input_dim, attention_dim) * 0.1
        self.W_value = np.random.randn(input_dim, attention_dim) * 0.1
        self.attention_dim = attention_dim # Store for consistent scaling

        self.last_query_input_raw = None
        self.last_key_value_sequence_raw = None
        self.last_proj_query = None
        self.last_proj_keys = None
        self.last_proj_values = None
        self.last_scores = None

    def forward(self, query, key_value_sequence):
        self.last_query_input_raw = query
        self.last_key_value_sequence_raw = key_value_sequence

        proj_query = np.dot(query, self.W_query)
        proj_keys = np.dot(key_value_sequence, self.W_key)
        proj_values = np.dot(key_value_sequence, self.W_value)

        self.last_proj_query = proj_query
        self.last_proj_keys = proj_keys
        self.last_proj_values = proj_values

        scores = np.dot(proj_keys, proj_query) / np.sqrt(self.attention_dim)
        self.last_scores = scores

        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

        context_vector = np.sum(np.expand_dims(attention_weights, axis=1) * proj_values, axis=0)

        return context_vector, attention_weights

class AttentionModel():
    def __init__(self, input_dim, attention_dim, output_dim, one_hot_classes_map, learning_rate=0.001, epochs=100):
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.output_dim = output_dim
        self.one_hot_classes_map = one_hot_classes_map
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.attention_layer = AttentionLayer(input_dim, attention_dim)

        self.W_output = np.random.randn(attention_dim, output_dim) * 0.1
        self.b_output = np.zeros((1, output_dim))

        self.dW_query, self.dW_key, self.dW_value = None, None, None
        self.dW_output, self.db_output = None, None

        self.last_query_input, self.last_key_value_input = None, None
        self.last_attention_weights, self.last_context_vector = None, None
        self.last_output_logits, self.last_output_softmax = None, None

    def softmax(self, x):
        max_val = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max_val)
        sum_val = np.sum(e_x, axis=1, keepdims=True)
        return e_x / sum_val

    def forward(self, query_input, key_value_input):
        self.last_query_input = query_input
        self.last_key_value_input = key_value_input

        context_vector, attention_weights = self.attention_layer.forward(query_input, key_value_input)
        self.last_context_vector = context_vector
        self.last_attention_weights = attention_weights

        logits = np.dot(context_vector.reshape(1, -1), self.W_output) + self.b_output
        self.last_output_logits = logits
        self.last_output_softmax = self.softmax(logits)

        return self.last_output_softmax

    def backward(self, y_true):
        if y_true.ndim == 1: y_true = y_true.reshape(1, -1)

        d_logits = self.last_output_softmax - y_true
        self.dW_output = np.dot(self.last_context_vector.reshape(-1, 1), d_logits)
        self.db_output = np.sum(d_logits, axis=0, keepdims=True)
        d_context = np.dot(d_logits, self.W_output.T).flatten()

        d_proj_values = np.outer(self.last_attention_weights, d_context)
        d_attention_weights = np.dot(self.attention_layer.last_proj_values, d_context)

        s = self.last_attention_weights
        dL_ds_dot_s = np.dot(d_attention_weights, s)
        d_scores = s * (d_attention_weights - dL_ds_dot_s)

        scaling_factor = np.sqrt(self.attention_dim)
        d_proj_keys = np.outer(d_scores, self.attention_layer.last_proj_query) / scaling_factor
        d_proj_query = np.dot(self.attention_layer.last_proj_keys.T, d_scores) / scaling_factor

        self.dW_value = np.dot(self.attention_layer.last_key_value_sequence_raw.T, d_proj_values)
        self.dW_key = np.dot(self.attention_layer.last_key_value_sequence_raw.T, d_proj_keys)
        self.dW_query = np.outer(self.attention_layer.last_query_input_raw, d_proj_query)

        self.attention_layer.W_query -= self.learning_rate * self.dW_query
        self.attention_layer.W_key -= self.learning_rate * self.dW_key
        self.attention_layer.W_value -= self.learning_rate * self.dW_value
        self.W_output -= self.learning_rate * self.dW_output
        self.b_output -= self.learning_rate * self.db_output

    def predict(self, query_input, key_value_input):
        probabilities = self.forward(query_input, key_value_input)
        predicted_class_idx = np.argmax(probabilities, axis=1)[0]
        one_hot_prediction = np.zeros((1, self.output_dim))
        one_hot_prediction[0, predicted_class_idx] = 1
        return one_hot_prediction

    def get_emotion_class(self, one_hot_class):
        one_hot_class_arr = np.asarray(one_hot_class).flatten()
        for key, value in self.one_hot_classes_map.items():
            if np.array_equal(np.asarray(value).flatten(), one_hot_class_arr):
                return key
        return None

    def train(self, X_query_train, X_kv_train, Y_train):
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(len(X_query_train)):
                query_i, kv_i, y_i = X_query_train[i], X_kv_train[i], Y_train[i]

                query_i = np.asarray(query_i)
                kv_i = np.asarray(kv_i)
                y_i = np.asarray(y_i)

                if y_i.ndim == 1: y_i = y_i.reshape(1, -1)

                predictions = self.forward(query_i, kv_i)
                loss = -np.sum(y_i * np.log(predictions + 1e-9))
                epoch_loss += loss
                self.backward(y_i)

            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                avg_loss = epoch_loss / len(X_query_train) if len(X_query_train) > 0 else 0
                print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")

    def predict_and_print_file(self, id_tweet_query_kv_list, file_name):
        try:
            with open(file_name, "w") as f:
                f.write("id,tweet,emotions\n")
                for id_val, tweet_text, query_input, kv_input in id_tweet_query_kv_list:
                    query_np, kv_np = np.asarray(query_input), np.asarray(kv_input)

                    if query_np.ndim == 0: query_np = query_np.reshape(1)
                    elif query_np.ndim > 1: query_np = query_np.flatten()

                    if query_np.shape[0] != self.input_dim:
                         raise ValueError(f"Query for id {id_val} has dim {query_np.shape[0]}, expected {self.input_dim}")
                    if kv_np.ndim != 2 or kv_np.shape[1] != self.input_dim:
                        raise ValueError(f"Key-value for id {id_val} has shape {kv_np.shape}, expected (seq_len, {self.input_dim})")

                    pred_one_hot = self.predict(query_np, kv_np)
                    pred_emotion = self.get_emotion_class(pred_one_hot.flatten())
                    processed_tweet_text = str(tweet_text).replace(',', ';')
                    f.write(f"{id_val},{processed_tweet_text},{pred_emotion}\n")
            print(f"Successfully wrote predictions to {file_name}")
        except IOError as e:
            print(f"IOError: Could not write to file {file_name}: {e}")
        except Exception as e:
            print(f"Unexpected error in predict_and_print_file: {e}")

if __name__ == '__main__':
    input_dim, attention_dim, output_dim, seq_len = 10, 8, 3, 5
    one_hot_map = {
        "class_A": np.array([1,0,0]), "class_B": np.array([0,1,0]), "class_C": np.array([0,0,1])
    }
    model = AttentionModel(input_dim, attention_dim, output_dim, one_hot_map, learning_rate=0.01, epochs=50)

    num_samples = 10
    X_q_train = [np.random.rand(input_dim) for _ in range(num_samples)]
    X_kv_train = [np.random.rand(seq_len, input_dim) for _ in range(num_samples)]
    Y_labels_train = [one_hot_map[np.random.choice(list(one_hot_map.keys()))] for _ in range(num_samples)]

    print("Starting dummy training...")
    model.train(X_q_train, X_kv_train, Y_labels_train)
    print("Dummy training complete.")

    print("\nTesting prediction on the first training sample:")
    pred_one_hot = model.predict(X_q_train[0], X_kv_train[0])
    pred_label = model.get_emotion_class(pred_one_hot.flatten())
    true_label_text = model.get_emotion_class(Y_labels_train[0].flatten())
    print(f"Predicted One-Hot: {pred_one_hot}, Predicted Label: {pred_label}, True Label: {true_label_text}")

    print("\nTesting predict_and_print_file:")
    test_samples_for_file = []
    for i in range(min(3, num_samples)):
         test_samples_for_file.append((
             i+1, f"Test tweet {i+1}", X_q_train[i], X_kv_train[i]
         ))
    if not test_samples_for_file: # Ensure list is not empty if num_samples was 0
        test_samples_for_file.append((1, "Default test tweet", np.random.rand(input_dim), np.random.rand(seq_len, input_dim)))

    model.predict_and_print_file(test_samples_for_file, "test_attention_output.csv")
