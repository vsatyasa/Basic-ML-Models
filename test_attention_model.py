import unittest
import numpy as np
from AttentionModel import AttentionLayer, AttentionModel # Assuming AttentionModel.py is in PYTHONPATH

class TestAttentionLayer(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.attention_dim = 8
        self.seq_len = 5
        self.layer = AttentionLayer(self.input_dim, self.attention_dim)

        self.query = np.random.rand(self.input_dim)
        self.key_value_sequence = np.random.rand(self.seq_len, self.input_dim)

    def test_forward_output_shapes(self):
        context_vector, attention_weights = self.layer.forward(self.query, self.key_value_sequence)
        self.assertEqual(context_vector.shape, (self.attention_dim,), "Context vector shape mismatch")
        self.assertEqual(attention_weights.shape, (self.seq_len,), "Attention weights shape mismatch")

    def test_attention_weights_sum_to_one(self):
        _, attention_weights = self.layer.forward(self.query, self.key_value_sequence)
        self.assertAlmostEqual(np.sum(attention_weights), 1.0, places=6, msg="Attention weights do not sum to 1")

    def test_score_calculation_basic_properties(self):
        # Check that weights are non-negative and exist for each item in sequence.
        _, attention_weights = self.layer.forward(self.query, self.key_value_sequence)
        self.assertTrue(np.all(attention_weights >= 0), "Attention weights should be non-negative")
        self.assertEqual(len(attention_weights), self.seq_len, "Attention weights length mismatch with sequence length")


class TestAttentionModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 12
        self.attention_dim = 10
        self.output_dim = 4 # Number of classes
        self.seq_len = 6

        self.one_hot_map = {
            "class0": np.array([1,0,0,0]),
            "class1": np.array([0,1,0,0]),
            "class2": np.array([0,0,1,0]),
            "class3": np.array([0,0,0,1]),
        }
        # Use a fixed seed for reproducibility of weight initialization during tests if needed
        # np.random.seed(42) # Optional: for reproducible random weight initialization
        self.model = AttentionModel(self.input_dim, self.attention_dim, self.output_dim, self.one_hot_map, learning_rate=0.01, epochs=1)

        self.query_input = np.random.rand(self.input_dim)
        self.key_value_input = np.random.rand(self.seq_len, self.input_dim)
        # y_true_one_hot is a 1D array, e.g. (4,). AttentionModel.backward handles reshaping.
        self.y_true_one_hot = self.one_hot_map["class1"]


    def test_forward_output_shape_and_sum(self):
        probabilities = self.model.forward(self.query_input, self.key_value_input)
        self.assertEqual(probabilities.shape, (1, self.output_dim), "Forward pass output shape mismatch")
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=6, msg="Forward pass probabilities do not sum to 1")

    def test_predict_output_shape_and_properties(self):
        one_hot_prediction = self.model.predict(self.query_input, self.key_value_input)
        self.assertEqual(one_hot_prediction.shape, (1, self.output_dim), "Predict output shape mismatch")
        self.assertAlmostEqual(np.sum(one_hot_prediction), 1.0, places=6, msg="Prediction is not a valid one-hot vector (sum should be 1)")
        self.assertTrue(np.all((one_hot_prediction == 0) | (one_hot_prediction == 1)), "Prediction is not one-hot (values should be 0 or 1)")
        self.assertEqual(np.count_nonzero(one_hot_prediction == 1), 1, "Prediction should have exactly one element as 1")

    def test_get_emotion_class(self):
        self.assertEqual(self.model.get_emotion_class(np.array([0,1,0,0])), "class1")
        # Test with 2D input, as model.get_emotion_class flattens it.
        self.assertEqual(self.model.get_emotion_class(np.array([0,0,0,1]).reshape(1,4)), "class3")
        self.assertIsNone(self.model.get_emotion_class(np.array([0.5,0.5,0,0])), "Should be None for non-binary/non-exact one-hot")
        self.assertIsNone(self.model.get_emotion_class(np.array([1,1,0,0])), "Should be None for multi-hot")

    def test_training_step_runs_and_weights_change(self):
        initial_W_query = np.copy(self.model.attention_layer.W_query)
        initial_W_output = np.copy(self.model.W_output)
        initial_b_output = np.copy(self.model.b_output)

        self.model.forward(self.query_input, self.key_value_input)
        self.model.backward(self.y_true_one_hot)

        self.assertFalse(np.array_equal(self.model.attention_layer.W_query, initial_W_query), "AttentionLayer W_query did not change")
        self.assertFalse(np.array_equal(self.model.W_output, initial_W_output), "Model W_output did not change")
        self.assertFalse(np.array_equal(self.model.b_output, initial_b_output), "Model b_output did not change")

    def test_train_method_runs_without_error(self):
        num_dummy_samples = 3
        X_q_dummy = [np.random.rand(self.input_dim) for _ in range(num_dummy_samples)]
        X_kv_dummy = [np.random.rand(self.seq_len, self.input_dim) for _ in range(num_dummy_samples)]
        Y_labels_dummy = [self.one_hot_map[np.random.choice(list(self.one_hot_map.keys()))] for _ in range(num_dummy_samples)]

        try:
            original_epochs = self.model.epochs
            self.model.epochs = 2 # Small number of epochs for smoke test
            self.model.train(X_q_dummy, X_kv_dummy, Y_labels_dummy)
            self.model.epochs = original_epochs # Restore original epochs
        except Exception as e:
            self.fail(f"model.train() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
