Project1 Files

## Models Implemented

This project includes implementations of several machine learning models for text classification.

### 3. Attention Model

*   **Purpose:** The Attention Model is designed for sequence classification tasks. It incorporates an attention mechanism (scaled dot-product attention) that allows the model to dynamically weigh the importance of different parts of an input sequence (or features, in the case of TF-IDF vectors treated as sequences of length 1) when making a prediction.
*   **File:** The implementation resides in `AttentionModel.py`.
*   **Usage:**
    *   The model is integrated into `main.py`.
    *   **Data Format:**
        *   `query_input`: For the current integration with TF-IDF vectors, this is the TF-IDF vector itself (shape: `(input_dim,)`).
        *   `key_value_input`: The TF-IDF vector reshaped as a sequence of length 1 (shape: `(1, input_dim)`).
    *   **Configuration in `main.py`:**
        *   `attention_internal_dim`: Sets the internal dimension of the attention mechanism's projections.
        *   `learning_rate`: The learning rate for training.
        *   `epochs`: The number of training epochs.
    *   **Output:** Predictions on the test set are saved to `test_attention.csv`.
*   **Testing:** Unit tests for the attention model are in `test_attention_model.py`.
