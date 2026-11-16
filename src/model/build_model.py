#!/usr/bin/env python3

"""
Definizione e compilazione del modello EloChessV1.

Input:
- board: tensore (8, 8, 12) che rappresenta la disposizione dei pezzi.
- state: vettore (71,) contenente informazioni aggiuntive sulla posizione.
- elo_side: Elo normalizzato del giocatore al tratto, forma (1,).
- elo_opponent: Elo normalizzato dell'avversario, forma (1,).
- legal_mask: vettore (num_moves,) che identifica le mosse legali.

Architettura:
La scacchiera viene elaborata tramite una rete convoluzionale (CNN).

Le caratteristiche estratte vengono combinate con lo stato della
posizione e con gli Elo dei due giocatori.

Il parametro board_head determina come le caratteristiche spaziali
della CNN vengono trasformate prima della fusione con gli altri input.

La rappresentazione condivisa alimenta due teste di output:
- Policy: distribuzione di probabilità sulle mosse legali.
- Value: stima del risultato atteso per il giocatore al tratto.

Output:
- "policy": distribuzione sulle num_moves mosse del vocabolario.
  Le mosse illegali vengono mascherate prima della softmax.
- "value": valore scalare nell'intervallo [0, 1], interpretato come
  expected score del giocatore al tratto:
      1.0 = vittoria
      0.5 = patta
      0.0 = sconfitta

Il Value rappresenta un risultato atteso, non direttamente
la probabilità di vittoria.

Compilazione:
La costruzione e la compilazione sono operazioni separate:
- build_model() costruisce l'architettura e restituisce il modello.
- compile_model() configura ottimizzatore, loss e metriche.

Loss:
- Policy: (1-alpha)*CrossEntropy - alpha*(2*y_value-1)*p_mossa.
  Il target Policy contiene [indice mossa, y_value].
- Value: BinaryCrossentropy.

La loss complessiva combina le due componenti:
    total_loss = policy_loss + lambda_value * value_loss

La pipeline usa sample weights unitari: alpha e il risultato sono
gestiti direttamente dalla loss Policy.

Metriche Policy:
- accuracy: accuratezza della prima mossa suggerita (Top-1).
- top3: accuratezza considerando le prime tre mosse.
- top5: accuratezza considerando le prime cinque mosse.
- unweighted_ce: SparseCategoricalCrossentropy senza sample weights,
  utile per confrontare esperimenti con alpha differenti.

Metriche Value:
- brier: MeanSquaredError tra expected score previsto e risultato
  osservato. Il nome storico "brier" identifica qui un MSE sul
  risultato codificato come 0, 0.5 o 1.

Le metriche sono diagnostiche: l'interpretazione della qualità
scacchistica delle mosse richiede una valutazione separata.

Questo modulo definisce il modello e la sua compilazione.
La preparazione dei dati, il bilanciamento dei sample weights,
la suddivisione training/validation, i callback e il salvataggio
dei checkpoint sono gestiti dalla pipeline di training.
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import DEFAULTS, validate_settings  # noqa: E402
from src.config.utils import resolve_path  # noqa: E402

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

DEFAULT_VOCAB = DEFAULTS["vocab"]
DEFAULT_DROPOUT = DEFAULTS["dropout"]
DEFAULT_LEARNING_RATE = DEFAULTS["learning_rate"]
DEFAULT_LAMBDA_VALUE = DEFAULTS["lambda_value"]
DEFAULT_ALPHA = DEFAULTS["alpha"]

@keras.utils.register_keras_serializable(package="EloChess")
class LegalMaskSoftmax(layers.Layer):
    """Applica la legal mask ai logits e restituisce una distribuzione softmax."""
    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        logits, legal_mask = inputs
        masked_logits = tf.where(
            legal_mask > 0,
            logits,
            tf.cast(-1e9, logits.dtype),
        )
        return tf.nn.softmax(masked_logits, axis=-1)



def _unpack_move(y_true):
    """Prima colonna = indice UCI; seconda = y_value."""
    return tf.cast(y_true[..., 0], tf.int32)


@keras.utils.register_keras_serializable(package="EloChess")
class OutcomeBlendLoss(keras.losses.Loss):
    """(1-alpha)*CE - alpha*(2*y-1)*p_m, per singola posizione."""
    def __init__(self, alpha=DEFAULT_ALPHA, name="policy_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        if not 0 <= alpha <= 1:
            raise ValueError("alpha deve essere compreso tra 0 e 1")
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        move = _unpack_move(y_true)
        outcome = tf.cast(y_true[..., 1], y_pred.dtype)
        ce = keras.losses.sparse_categorical_crossentropy(move, y_pred)
        p_move = tf.gather(y_pred, move, axis=1, batch_dims=1)
        reward = (2.0 * outcome - 1.0) * p_move
        return (1.0 - self.alpha) * ce - self.alpha * reward

    def get_config(self):
        return {**super().get_config(), "alpha": self.alpha}


@keras.utils.register_keras_serializable(package="EloChess")
class PackedAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_unpack_move(y_true), y_pred, sample_weight)


@keras.utils.register_keras_serializable(package="EloChess")
class PackedTopK(keras.metrics.SparseTopKCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_unpack_move(y_true), y_pred, sample_weight)


@keras.utils.register_keras_serializable(package="EloChess")
class PackedUnweightedCE(keras.metrics.SparseCategoricalCrossentropy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(_unpack_move(y_true), y_pred, sample_weight)

def build_model(num_moves: int, dropout_rate: float = DEFAULT_DROPOUT, board_head: str = DEFAULTS["board_head"]) -> keras.Model:
    """Costruisce EloChess V1 senza compilare il modello."""
    if num_moves <= 0:
        raise ValueError("num_moves deve essere positivo")
    validate_settings({"dropout": dropout_rate, "board_head": board_head})
    # Input
    board_input = keras.Input(shape=(8, 8, 12), dtype=tf.float32, name="board")
    state_input = keras.Input(shape=(71,), dtype=tf.float32, name="state")
    elo_side_input = keras.Input(shape=(1,), dtype=tf.float32, name="elo_side")
    elo_opponent_input = keras.Input(shape=(1,), dtype=tf.float32, name="elo_opponent")
    legal_mask_input = keras.Input(shape=(num_moves,), dtype=tf.float32, name="legal_mask")
    # Board
    x_board = layers.Conv2D(64, (3, 3), use_bias=False, padding="same", name="board_conv1")(board_input)
    x_board = layers.BatchNormalization(name="board_bn1")(x_board)
    x_board = layers.ReLU(name="board_relu1")(x_board)
    x_board = layers.Conv2D(64, (3, 3), use_bias=False, padding="same", name="board_conv2")(x_board)
    x_board = layers.BatchNormalization(name="board_bn2")(x_board)
    x_board = layers.ReLU(name="board_relu2")(x_board)
    x_board = layers.Conv2D(128, (3, 3), use_bias=False, padding="same", name="board_conv3")(x_board)
    x_board = layers.BatchNormalization(name="board_bn3")(x_board)
    x_board = layers.ReLU(name="board_relu3")(x_board)
    x_board = layers.Conv2D(128, (3, 3), use_bias=False, padding="same", name="board_conv4")(x_board)
    x_board = layers.BatchNormalization(name="board_bn4")(x_board)
    x_board = layers.ReLU(name="board_relu4")(x_board)
    x_board = layers.Conv2D(256, (3, 3), use_bias=False, padding="same", name="board_conv5")(x_board)
    x_board = layers.BatchNormalization(name="board_bn5")(x_board)
    x_board = layers.ReLU(name="board_relu5")(x_board)
    if board_head == "gap":
        x_board = layers.GlobalAveragePooling2D(name="board_gap")(x_board)
    elif board_head == "flatten":
        x_board = layers.Conv2D(32, (1, 1), use_bias=False, name="board_reduce")(x_board)
        x_board = layers.BatchNormalization(name="board_reduce_bn")(x_board)
        x_board = layers.ReLU(name="board_reduce_relu")(x_board)
        x_board = layers.Flatten(name="board_flatten")(x_board)
    else:
        raise ValueError("board_head deve essere 'gap' o 'flatten'")
    # State
    x_state = layers.Dense(32, name="state_dense1")(state_input)
    x_state = layers.ReLU(name="state_relu1")(x_state)
    x_state = layers.Dense(32, name="state_dense2")(x_state)
    x_state = layers.ReLU(name="state_relu2")(x_state)
    # Elo branch
    elo_input = layers.Concatenate(name="elo_concat")([elo_side_input, elo_opponent_input])
    x_elo = layers.Dense(32, name="elo_dense1")(elo_input)
    x_elo = layers.ReLU(name="elo_relu1")(x_elo)
    x_elo = layers.Dense(32, name="elo_dense2")(x_elo)
    x_elo = layers.ReLU(name="elo_relu2")(x_elo)
    # Fusion
    x = layers.Concatenate(name="fusion")([x_board, x_state, x_elo])
    x = layers.Dense(256, name="fusion_dense1")(x)
    x = layers.ReLU(name="fusion_relu1")(x)
    x = layers.Dropout(dropout_rate, name="fusion_dropout")(x)
    x = layers.Dense(128, name="fusion_dense2")(x)
    x = layers.ReLU(name="fusion_relu2")(x)
    # Policy
    policy_logits = layers.Dense(num_moves, name="policy_logits")(x)
    policy_output = LegalMaskSoftmax(name="policy")([policy_logits, legal_mask_input])
    # Value
    x_value = layers.Dense(64, name="value_dense1")(x)
    x_value = layers.ReLU(name="value_relu1")(x_value)
    value_output = layers.Dense(1, activation="sigmoid", name="value")(x_value)
    # Model
    model = keras.Model(inputs=[board_input, state_input, elo_side_input, elo_opponent_input, legal_mask_input], outputs={"policy": policy_output, "value": value_output}, name="EloChessV1")
    return model

def compile_model(
    model: keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lambda_value: float = DEFAULT_LAMBDA_VALUE,
    policy_loss_name: str = DEFAULTS["policy_loss"],
    value_loss_name: str = DEFAULTS["value_loss"],
    optimizer_name: str = DEFAULTS["optimizer"],
    alpha: float = DEFAULT_ALPHA,
) -> None:
    validate_settings({
        "learning_rate": learning_rate,
        "lambda_value": lambda_value,
        "policy_loss": policy_loss_name,
        "value_loss": value_loss_name,
        "optimizer": optimizer_name,
        "alpha": alpha,
    })
    policy_losses = {
        "outcome_blend": lambda name: OutcomeBlendLoss(alpha=alpha, name=name),
    }
    value_losses = {
        "binary_crossentropy": keras.losses.BinaryCrossentropy,
        "mean_squared_error": keras.losses.MeanSquaredError,
    }
    optimizers = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD}
    policy_loss = policy_losses[policy_loss_name](name="policy_loss")
    value_loss = value_losses[value_loss_name](name="value_loss")
    model.compile(
        optimizer=optimizers[optimizer_name](learning_rate=learning_rate),
        loss={"policy": policy_loss, "value": value_loss},
        loss_weights={"policy": 1.0, "value": lambda_value},
        metrics={"policy": [
                PackedAccuracy(name="accuracy"),
                PackedTopK(k=3, name="top3"),
                PackedTopK(k=5, name="top5"),
                PackedUnweightedCE(name="unweighted_ce"),
                ], "value": [keras.metrics.MeanSquaredError(name="brier"),
                ]}
    )

def build_parser() -> argparse.ArgumentParser:
    """Costruisce il parser CLI."""
    parser = argparse.ArgumentParser(description="Costruisce e mostra EloChess V1.")
    parser.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
        help="Percorso di move2idx_all.json.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout della fusione.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate iniziale.",
    )
    parser.add_argument(
        "--lambda_value",
        type=float,
        default=DEFAULT_LAMBDA_VALUE,
        help="Peso della loss Value.",
    )
    parser.add_argument(
        "--board_head",
        type=str,
        choices=["gap", "flatten"],
        default=DEFAULTS["board_head"],
        help="Uscita del ramo board: media globale o mappa spaziale appiattita."
    )
    parser.add_argument("--policy_loss", choices=["outcome_blend"], default=DEFAULTS["policy_loss"])
    parser.add_argument("--value_loss", choices=["binary_crossentropy", "mean_squared_error"], default=DEFAULTS["value_loss"])
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=DEFAULTS["optimizer"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser

def main() -> None:
    """Costruisce il modello e ne stampa la struttura."""
    args = build_parser().parse_args()
    with open(resolve_path(args.vocab), encoding="utf-8") as f:
        num_moves = len(json.load(f))
    model = build_model(
        num_moves=num_moves,
        dropout_rate=args.dropout,
        board_head = args.board_head
    )
    compile_model(model, args.learning_rate, args.lambda_value, args.policy_loss, args.value_loss, args.optimizer, args.alpha)
    model.summary()


if __name__ == "__main__":
    main()
