"""Valori di riferimento per gli esperimenti EloChess.

Importare DEFAULTS nei parser CLI; gli argomenti espliciti da terminale
prevalgono sui default. I parser di train_model.py e build_model.py leggono queste impostazioni.

I nomi delle chiavi corrispondono agli argomenti CLI degli script.
Le opzioni specifiche di un esperimento, come quelle della V1 nel launcher,
vanno passate esplicitamente; il training conserva la configurazione effettiva
in config.json.
"""
from collections.abc import Mapping
from typing import Any

# Un unico punto dove modificare i default degli esperimenti.
DEFAULTS = {
    # Apprendimento
    "alpha": 0.5,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 1,
    "dropout": 0.2,
    "lambda_value": 0.5,
    # Dataset e riproducibilita
    "data_dir": "data/processed/2013-07",
    "validation_fraction": 0.1,
    "split_seed": 42,
    "seed": 42,
    "max_files": None,
    "vocab": "src/config/move2idx_all.json",
    "split_file": None,  # percorso generato da data_dir e split_seed
    # Architettura e training
    "board_head": "flatten",
    "monitor": "val_policy_loss",
    "patience": 6,
    "save_every_steps": 5000,
    "out_dir": "models/v1",
    "resume": False,
    "init_model": None,
    # Funzioni obiettivo e ottimizzazione (selezionabili in compile_model)
    "policy_loss": "outcome_blend",
    "value_loss": "binary_crossentropy",
    "optimizer": "adam",
    "reduce_lr_patience": 2,
    "reduce_lr_factor": 0.5,
    "min_learning_rate": 1e-6,
}


def validate_settings(values: Mapping[str, Any]) -> None:
    """Valida i parametri di configurazione presenti, inclusi gli override CLI."""
    constraints = {
        "alpha": (lambda value: 0 <= value <= 1, "deve essere compreso tra 0 e 1"),
        "learning_rate": (lambda value: value > 0, "deve essere positivo"),
        "batch_size": (lambda value: value > 0, "deve essere positivo"),
        "epochs": (lambda value: value > 0, "deve essere positivo"),
        "dropout": (lambda value: 0 <= value < 1, "deve essere in [0, 1)"),
        "lambda_value": (lambda value: value >= 0, "deve essere non negativo"),
        "validation_fraction": (
            lambda value: 0 < value < 1,
            "deve essere compreso tra 0 e 1, esclusi",
        ),
        "patience": (lambda value: value > 0, "deve essere positivo"),
        "save_every_steps": (lambda value: value >= 0, "deve essere non negativo"),
        "max_files": (
            lambda value: value is None or value > 0,
            "deve essere positivo oppure None",
        ),
        "reduce_lr_patience": (
            lambda value: value >= 0,
            "deve essere non negativo",
        ),
        "reduce_lr_factor": (
            lambda value: 0 < value < 1,
            "deve essere compreso tra 0 e 1, esclusi",
        ),
        "min_learning_rate": (lambda value: value > 0, "deve essere positivo"),
    }
    for key, (is_valid, message) in constraints.items():
        if key in values and not is_valid(values[key]):
            raise ValueError(f"{key} {message}.")

    choices = {
        "board_head": {"gap", "flatten"},
        "policy_loss": {"outcome_blend"},
        "value_loss": {"binary_crossentropy", "mean_squared_error"},
        "optimizer": {"adam", "sgd"},
    }
    for key, allowed in choices.items():
        if key in values and values[key] not in allowed:
            raise ValueError(f"{key} deve essere uno tra: {', '.join(sorted(allowed))}.")

    if values.get("resume") and values.get("init_model"):
        raise ValueError("--resume e --init_model sono alternativi.")


def validate_defaults() -> None:
    """Valida i default del modulo usando le stesse regole degli override CLI."""
    validate_settings(DEFAULTS)
