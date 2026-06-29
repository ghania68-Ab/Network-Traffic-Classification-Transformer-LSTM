"""TensorFlow/Keras model definitions for network traffic classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam


def compile_classifier(model: Model, learning_rate: float) -> Model:
    """Compile all models with the same loss/metric family for fair comparison."""

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_simple_lstm(
    input_shape: tuple[int, int],
    num_classes: int,
    lstm_units: int = 96,
    dropout_rate: float = 0.25,
    dense_units: int = 96,
    learning_rate: float = 8e-4,
) -> Model:
    """Build the required Simple LSTM baseline.

    The baseline intentionally stays standard: one LSTM layer, one dropout
    layer, one dense layer, and a softmax output layer.
    """

    inputs = Input(shape=input_shape, name="network_features")
    x = LSTM(lstm_units, name="lstm_encoder")(inputs)
    x = Dropout(dropout_rate, name="dropout")(x)
    x = Dense(dense_units, activation="relu", name="dense_classifier")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    return compile_classifier(Model(inputs, outputs, name="Simple_LSTM"), learning_rate)


def build_transformer_lstm(
    input_shape: tuple[int, int],
    num_classes: int,
    lstm_units: int = 128,
    attention_heads: int = 4,
    dropout_rate: float = 0.25,
    dense_units: int = 128,
    ffn_units: int | None = None,
    learning_rate: float = 5e-4,
) -> Model:
    """Build the proposed Transformer-Enhanced LSTM model.

    Important implementation note:
    The preprocessed tabular data is reshaped to (features, 1). Applying
    LayerNormalization directly to that input would normalize over a single
    channel and remove most of the signal. The model therefore first projects
    the feature sequence with LSTM, then applies Transformer-style normalization
    and attention on the richer LSTM representation.
    """

    if lstm_units % attention_heads != 0:
        raise ValueError("lstm_units must be divisible by attention_heads for a clean attention projection.")

    key_dim = lstm_units // attention_heads
    ffn_units = ffn_units or lstm_units * 2
    inputs = Input(shape=input_shape, name="network_features")

    # LSTM sequence encoder: keeps one hidden vector per feature timestep so
    # attention can learn relationships among the encoded feature positions.
    x = LSTM(lstm_units, return_sequences=True, name="lstm_sequence_encoder")(inputs)
    x = Dropout(dropout_rate, name="lstm_dropout")(x)

    # Pre-norm self-attention block. MultiHeadAttention returns lstm_units
    # channels, so the residual Add has valid matching dimensions.
    attention_input = LayerNormalization(epsilon=1e-6, name="attention_input_norm")(x)
    attention_output = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=key_dim,
        value_dim=key_dim,
        dropout=dropout_rate,
        output_shape=lstm_units,
        name="multi_head_self_attention",
    )(attention_input, attention_input)
    attention_output = Dropout(dropout_rate, name="attention_dropout")(attention_output)
    x = Add(name="attention_residual")([x, attention_output])
    x = LayerNormalization(epsilon=1e-6, name="attention_output_norm")(x)

    # Transformer feed-forward block with a second residual connection. The
    # expanded hidden size gives attention features more capacity, then the
    # final projection returns to lstm_units so Add remains dimensionally valid.
    ffn_input = x
    ffn = Dense(ffn_units, activation="relu", name="transformer_ffn_dense")(ffn_input)
    ffn = Dropout(dropout_rate, name="transformer_ffn_dropout")(ffn)
    ffn = Dense(lstm_units, name="transformer_ffn_projection")(ffn)
    x = Add(name="ffn_residual")([ffn_input, ffn])
    x = LayerNormalization(epsilon=1e-6, name="ffn_output_norm")(x)

    # GlobalAveragePooling1D converts the attended sequence representation into
    # one vector for classification without favoring a single timestep.
    x = GlobalAveragePooling1D(name="global_average_pooling")(x)
    x = Dropout(dropout_rate, name="classifier_dropout")(x)
    x = Dense(dense_units, activation="relu", name="classifier_dense")(x)
    x = Dropout(dropout_rate, name="classifier_dense_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    return compile_classifier(Model(inputs, outputs, name="Transformer_Enhanced_LSTM"), learning_rate)


def set_global_determinism(seed: int = 42) -> None:
    """Set random seeds so experiments are easier to reproduce."""

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Some TensorFlow builds do not expose deterministic kernels on Windows.
        pass
