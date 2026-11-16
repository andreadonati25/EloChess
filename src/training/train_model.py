#!/usr/bin/env python3
"""
train_model.py

Allena EloChess V1 sugli shard NPZ prodotti da parquet_to_npz.py.
Costruisce (o riprende) il modello, gestisce lo split train/validation e salva
checkpoint e cronologia. La definizione del modello e' in build_model.py.

Dati:
    Gli shard NPZ vengono letti uno alla volta (streaming): in memoria c'e' uno
    shard per volta e non l'intero dataset. Un'epoca e' una passata completa su
    tutti gli shard selezionati.

Split:
    Lo split train/validation e' fatto a livello di partita (game_id), non di
    posizione, cosi' le posizioni di una stessa partita non finiscono in entrambi
    gli insiemi. La scelta e' deterministica (--split_seed) e viene salvata in un
    manifest in data/validation/ con gli ID delle partite di validation: gli ID
    di training sono tutte le altre partite. Ai training successivi il manifest
    viene riutilizzato, cosi' la validation non entra mai nel training.

Shuffle:
    A ogni epoca l'ordine degli shard e' casuale e, dentro ogni shard, le
    posizioni di training sono mescolate. Non si fa nessun bilanciamento
    artificiale dei dati: la distribuzione naturale dei giocatori resta visibile.

Obiettivo Policy:
    L = (1-alpha) * [-log(p_m)] + alpha * [-r*p_m], r=2*y_value-1.
    La prima componente imita la mossa; la seconda premia una probabilita'
    alta dopo una vittoria e bassa dopo una sconfitta. alpha in [0,1].
    Nessuna statistica preaggregata. La Policy riceve target [indice UCI,
    y_value] e sample weight 1. La validation usa lo stesso obiettivo.
    val_policy_unweighted_ce resta la metrica di imitazione separata.

Checkpoint e resume:
    - best.keras: miglior modello secondo --monitor (default val_policy_loss);
    - last.keras: ultimo modello, salvato a fine epoca e ogni --save_every_steps
      batch, perche' un'epoca puo' durare ore;
    - --resume riprende da last.keras con pesi, ottimizzatore e learning rate.
    - --init_model avvia un nuovo esperimento dai pesi di un modello .keras,
      ricompilando loss e ottimizzatore e salvando in una nuova --out_dir.
      Dopo un'interruzione a meta' epoca l'epoca riparte dall'inizio, ma i pesi
      sono quelli dell'ultimo salvataggio. Per cambiare l'ordine dei dati in un
      resume basta cambiare --seed (lo split non cambia, dipende da --split_seed).

Output (cartella --out_dir, default models/v1):
    best.keras, last.keras, history.json, config.json, best_metric.json

Esempio:
    python src/training/train_model.py --data_dir data/processed/2013-07 --epochs 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.build_model import build_model, compile_model # noqa: E402
from src.config.utils import utc_now, resolve_path, write_json_atomic, load_num_moves, find_npz_files # noqa: E402
from src.config.settings import DEFAULTS, validate_settings  # noqa: E402

DEFAULT_VOCAB = DEFAULTS["vocab"]
DEFAULT_VALIDATION_DIR = "data/validation"
TRAIN_KEYS = ("board", "state", "elo_side", "elo_opponent", "move", "legal_mask", "y_value", "game_id")

# Split train/validation per partita
def collect_game_ids(files: list[Path]) -> np.ndarray:
    """
    Raccoglie gli ID delle partite leggendo soltanto game_id.

    Una partita non deve essere distribuita tra shard diversi: in quel caso lo
    split per partita potrebbe produrre leakage. parquet_to_npz.py non spezza mai
    una partita, quindi qui si verifica soltanto che sia cosi'.
    """
    seen: set[int] = set()
    for path in tqdm(files, desc="Lettura game_id", unit="file"):
        with np.load(path, allow_pickle=False) as archive:
            if "game_id" not in archive.files:
                raise ValueError(f"Chiave game_id assente in {path}")
            local_ids = np.unique(archive["game_id"])
        local_set = {int(game_id) for game_id in local_ids}
        overlap = seen.intersection(local_set)
        if overlap:
            example = sorted(overlap)[:5]
            raise ValueError(f"Partite presenti in piu' shard; esempi game_id: {example}")
        seen.update(local_set)
    if len(seen) < 2:
        raise ValueError("Servono almeno due partite per creare lo split.")
    return np.asarray(sorted(seen), dtype=np.int64)

def fingerprint_game_ids(game_ids: np.ndarray) -> str:
    """Impronta stabile di un insieme di partite."""
    canonical = np.asarray(game_ids, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()

def load_or_create_split(files: list[Path], split_path: Path, validation_fraction: float, seed: int) -> tuple[set[int], dict]:
    """
    Crea o ricarica lo split globale per partita.

    Lo split viene scelto una sola volta e salvato su disco, cosi' tutti i
    training successivi usano gli stessi game_id di validation. Se il manifest
    esiste ma non corrisponde al dataset (o a seed e percentuale richiesti) il
    programma si ferma, invece di creare in silenzio uno split diverso.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve essere compreso tra 0 e 1.")
    game_ids = collect_game_ids(files)
    fingerprint = fingerprint_game_ids(game_ids)
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("game_ids_sha256") != fingerprint:
            raise ValueError(f"Lo split {split_path} non corrisponde al dataset attuale. Verifica gli NPZ e scegli uno split nuovo se il dataset e' cambiato.")
        if manifest.get("seed") != seed:
            raise ValueError("Lo split_seed non coincide con quello del manifest esistente.")
        if not math.isclose(float(manifest.get("validation_fraction", -1)), validation_fraction):
            raise ValueError("La percentuale validation non coincide con il manifest esistente.")
        validation_ids = {int(game_id) for game_id in manifest["validation_game_ids"]}
        if not validation_ids or not validation_ids.issubset(set(game_ids.tolist())):
            raise ValueError("Il manifest contiene game_id non validi.")
        print(f"Split esistente riutilizzato: {split_path}")
    else:
        n_validation = int(round(len(game_ids) * validation_fraction))
        n_validation = max(1, min(len(game_ids) - 1, n_validation))
        rng = np.random.default_rng(seed)
        validation_ids = {
            int(game_id)
            for game_id in rng.choice(
                game_ids,
                size=n_validation,
                replace=False,
            )
        }
        manifest = {
            "created_at_utc": utc_now(),
            "seed": seed,
            "validation_fraction": validation_fraction,
            "num_games": int(len(game_ids)),
            "num_validation_games": len(validation_ids),
            "game_ids_sha256": fingerprint,
            "validation_game_ids": sorted(validation_ids),
        }
        write_json_atomic(split_path, manifest)
        print(f"Nuovo split salvato: {split_path}")
    print(f"Partite totali: {len(game_ids)}")
    print(f"Partite validation: {len(validation_ids)}")
    print(f"Partite training: {len(game_ids) - len(validation_ids)}")
    return validation_ids, manifest

def count_positions_and_steps(files: list[Path], validation_ids: set[int], batch_size: int) -> dict[str, int]:
    """
    Conta posizioni e batch attesi leggendo soltanto game_id.

    Il numero di batch deve coincidere con quello che produce il generatore:
    ogni shard finisce con un batch parziale, quindi si somma ceil(n / batch)
    shard per shard e non ceil(totale / batch).
    """
    val_ids = np.asarray(sorted(validation_ids), dtype=np.int64)
    train_positions = 0
    val_positions = 0
    train_steps = 0
    val_steps = 0
    for path in tqdm(files, desc="Conteggio posizioni", unit="file"):
        with np.load(path, allow_pickle=False) as archive:
            game_ids = archive["game_id"]
        in_validation = np.isin(game_ids, val_ids)
        n_val = int(in_validation.sum())
        n_train = len(game_ids) - n_val
        train_positions += n_train
        val_positions += n_val
        train_steps += math.ceil(n_train / batch_size) if n_train else 0
        val_steps += math.ceil(n_val / batch_size) if n_val else 0
    if not train_steps or not val_steps:
        raise ValueError("Train o validation sono vuoti. Controlla gli shard e il manifest.")
    return {
        "train_positions": train_positions,
        "validation_positions": val_positions,
        "train_steps": train_steps,
        "validation_steps": val_steps,
    }

# Dataset in streaming

def iter_shard_batches(path: Path, split: str, validation_ids: np.ndarray, batch_size: int, num_moves: int, rng: np.random.Generator):
    """
    Legge uno shard e restituisce i batch dello split richiesto.

    Ogni batch e' una tripla (inputs, targets, sample_weights) nel formato che
    Keras si aspetta da model.fit.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = set(TRAIN_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Chiavi mancanti in {path}: {sorted(missing)}")
        data = {key: archive[key] for key in TRAIN_KEYS}
    n = len(data["game_id"])
    # Controlli di coerenza dello shard: meglio fermarsi subito che allenare
    # su dati con la forma sbagliata.
    if data["board"].shape != (n, 8, 8, 12):
        raise ValueError(f"Shape board non valida in {path}")
    if data["state"].shape != (n, 71):
        raise ValueError(f"Shape state non valida in {path}")
    if data["legal_mask"].shape != (n, num_moves):
        raise ValueError(f"Shape legal_mask non valida in {path}")
    for key in TRAIN_KEYS:
        if len(data[key]) != n:
            raise ValueError(f"Numero di righe incoerente per {key} in {path}")
    # Selezione delle posizioni dello split: la validation e' l'insieme delle
    # partite del manifest, il training e' tutto il resto.
    is_validation = np.isin(data["game_id"], validation_ids)
    keep = is_validation if split == "validation" else ~is_validation
    indices = np.flatnonzero(keep)
    # Mescola le posizioni dello shard solo in training.
    if split == "train":
        rng.shuffle(indices)
    # La loss della Policy segue lo stesso criterio nel training e nella
    # validation. I pesi di validation non aggiornano i parametri del modello,
    # ma rendono coerenti checkpoint, scheduler ed early stopping.
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        y_value = data["y_value"][batch_indices].astype(np.float32, copy=False)
        y_move = data["move"][batch_indices].astype(np.int32, copy=False)
        if not np.all(np.isin(y_value, [0.0, 0.5, 1.0])):
            raise ValueError(f"y_value non valido in {path}")
        if np.any(y_move < 0) or np.any(y_move >= num_moves):
            raise ValueError(f"Indice move fuori dal vocabolario in {path}")
        # La Policy riceve indice mossa e risultato nella stessa etichetta;
        # la loss costruisce autonomamente il compromesso mediante alpha.
        y_policy = np.column_stack((y_move, y_value)).astype(np.float32)
        # Gli array negli NPZ sono uint8/(N,): il modello vuole float32 e gli
        # Elo con forma (N, 1).
        inputs = {
            "board": data["board"][batch_indices].astype(np.float32),
            "state": data["state"][batch_indices].astype(np.float32, copy=False),
            "elo_side": data["elo_side"][batch_indices, None].astype(np.float32, copy=False),
            "elo_opponent": data["elo_opponent"][batch_indices, None].astype(np.float32, copy=False),
            "legal_mask": data["legal_mask"][batch_indices].astype(np.float32),
        }
        targets = {
            "policy": y_policy,
            "value": y_value[:, None],
        }
        # Il Value non viene pesato.
        sample_weights = {
            "policy": np.ones(len(batch_indices), dtype=np.float32),
            "value": np.ones(len(batch_indices), dtype=np.float32),
        }
        yield inputs, targets, sample_weights

def make_tf_dataset(files: list[Path], split: str, validation_ids: set[int], batch_size: int, num_moves: int, seed: int, num_batches: int) -> tf.data.Dataset:
    """Crea un dataset in streaming che attraversa tutti gli shard a ogni epoca."""
    rng = np.random.default_rng(seed)
    validation_array = np.asarray(sorted(validation_ids), dtype=np.int64)

    def generator():
        order = np.arange(len(files))
        if split == "train":
            rng.shuffle(order)
        for index in order:
            yield from iter_shard_batches(
                path=files[int(index)],
                split=split,
                validation_ids=validation_array,
                batch_size=batch_size,
                num_moves=num_moves,
                rng=rng,
            )

    output_signature = (
        {
            "board": tf.TensorSpec((None, 8, 8, 12), tf.float32),
            "state": tf.TensorSpec((None, 71), tf.float32),
            "elo_side": tf.TensorSpec((None, 1), tf.float32),
            "elo_opponent": tf.TensorSpec((None, 1), tf.float32),
            "legal_mask": tf.TensorSpec((None, num_moves), tf.float32),
        },
        {
            "policy": tf.TensorSpec((None, 2), tf.float32),
            "value": tf.TensorSpec((None, 1), tf.float32),
        },
        {
            "policy": tf.TensorSpec((None,), tf.float32),
            "value": tf.TensorSpec((None,), tf.float32),
        },
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    # Un dataset da generatore ha lunghezza sconosciuta: Keras non saprebbe
    # quando finisce un'epoca ne' quanti batch aspettarsi (niente barra di
    # avanzamento con totale) e, passando steps_per_epoch, rischia di non
    # ricreare l'iteratore alla seconda epoca. Dichiarare la lunghezza risolve
    # entrambe le cose e, se il generatore produce un numero di batch diverso
    # da quello contato, TensorFlow si ferma con un errore.
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_batches))
    return dataset.prefetch(1)

# Callback e cronologia
def monitor_mode(monitor: str) -> str:
    """Le metriche di errore si minimizzano, le accuracy si massimizzano."""
    name = monitor.lower()
    return (
        "min"
        if any(term in name for term in ("loss", "mae", "mse", "brier", "_ce"))
        else "max"
    )

class BestModelCheckpoint(tf.keras.callbacks.Callback):
    """Salva il miglior modello e conserva il valore best anche dopo un resume."""

    def __init__(self, path: Path, metric_path: Path, monitor: str, mode: str):
        super().__init__()
        self.path = path
        self.metric_path = metric_path
        self.monitor = monitor
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        # Dopo un resume il best precedente va ricordato, altrimenti la prima
        # epoca nuova sovrascriverebbe un best.keras migliore.
        if metric_path.exists():
            with open(metric_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            if (previous.get("monitor") != monitor or previous.get("mode") != mode):
                raise ValueError("Il monitor del checkpoint precedente e' diverso. Usa una nuova cartella di output per cambiare monitor.")
            self.best = float(previous["best_value"])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.monitor not in logs:
            raise ValueError(f"Metric {self.monitor!r} non trovata. Metriche disponibili: {sorted(logs.keys())}")
        current = float(logs[self.monitor])
        improved = (
            current < self.best
            if self.mode == "min"
            else current > self.best
        )
        if not improved:
            return
        temporary = self.path.with_name("best.tmp.keras")
        self.model.save(temporary)
        os.replace(temporary, self.path)
        self.best = current
        write_json_atomic(
            self.metric_path,
            {
                "monitor": self.monitor,
                "mode": self.mode,
                "best_value": current,
                "epoch_in_run": epoch + 1,
                "updated_at_utc": utc_now(),
            },
        )
        print(f"\nNuovo best: {self.monitor}={current:.6f}")

class LastModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Salva l'ultimo modello a fine epoca e, se richiesto, ogni N batch.

    Con epoche lunghe il salvataggio periodico limita la perdita di lavoro in
    caso di interruzione (spegnimento, crash, Ctrl+C).
    """
    def __init__(self, path: Path, every_steps: int = 0):
        super().__init__()
        self.path = path
        self.every_steps = every_steps

    def _save(self) -> None:
        temporary = self.path.with_name("last.tmp.keras")
        self.model.save(temporary)
        os.replace(temporary, self.path)

    def on_train_batch_end(self, batch, logs=None):
        if self.every_steps and (batch + 1) % self.every_steps == 0:
            self._save()

    def on_epoch_end(self, epoch, logs=None):
        self._save()

def compute_bests(history: dict[str, list[float]]) -> dict:
    """Riassume il valore migliore osservato per ciascuna metrica."""
    bests = {}
    for name, values in history.items():
        if not values:
            continue
        low_is_better = any(
            term in name.lower()
            for term in ("loss", "mae", "mse", "brier", "_ce")
        )
        index = int(np.argmin(values) if low_is_better else np.argmax(values))
        bests[name] = {
            "best_value": float(values[index]),
            "epoch_in_run": index + 1,
            "mode": "min" if low_is_better else "max",
        }
    return bests

class RunHistoryLogger(tf.keras.callbacks.Callback):
    """
    Registra in history.json la cronologia di questa esecuzione.

    Il file viene aggiornato a ogni epoca (non solo alla fine), cosi' la
    cronologia non si perde se il training viene interrotto. Ogni esecuzione
    (nuovo training o resume) aggiunge una voce a "runs".
    """
    def __init__(self, path: Path, run_args: dict):
        super().__init__()
        self.path = path
        self.entry = {
            "started_at_utc": utc_now(),
            "ended_at_utc": None,
            "args": run_args,
            "history": {},
            "best_metrics": {},
        }
        self.document: dict = {"runs": []}

    def on_train_begin(self, logs=None):
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.document = json.load(f)
        if not isinstance(self.document, dict) or not isinstance(self.document.get("runs"), list):
            raise ValueError(f"Formato history.json non valido: {self.path}")
        self.document["runs"].append(self.entry)
        write_json_atomic(self.path, self.document)

    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            self.entry["history"].setdefault(name, []).append(float(value))

        self.entry["best_metrics"] = compute_bests(self.entry["history"])
        self.entry["ended_at_utc"] = utc_now()
        write_json_atomic(self.path, self.document)

def check_resume_split(config_path: Path, game_ids_sha256: str, validation_ids_sha256: str) -> None:
    """
    Verifica che un resume usi lo stesso split del training precedente.

    Se la validation cambiasse, partite gia' viste in training potrebbero
    finire in validation e la loss non sarebbe piu' significativa.
    """
    if not config_path.exists():
        print("ATTENZIONE: config.json assente, non posso verificare che lo split coincida con quello del training precedente.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        previous = json.load(f)
    for key, current in (
        ("game_ids_sha256", game_ids_sha256),
        ("validation_ids_sha256", validation_ids_sha256),
    ):
        if previous.get(key) != current:
            raise ValueError(f"Il resume usa uno split diverso da quello del training precedente ({key} non coincide).")

# Programma principale
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training in streaming di EloChess V1 su shard NPZ.")
    parser.add_argument("--data_dir", default=DEFAULTS["data_dir"], help="Cartella degli shard NPZ.")
    parser.add_argument("--out_dir", default=DEFAULTS["out_dir"], help="Cartella dei checkpoint.")
    parser.add_argument("--vocab", default=DEFAULTS["vocab"], help="Percorso di move2idx_all.json.")
    parser.add_argument("--split_file", default=DEFAULTS["split_file"], help="Manifest dello split; None = percorso automatico.")
    parser.add_argument("--validation_fraction", type=float, default=DEFAULTS["validation_fraction"], help="Frazione di partite in validation.")
    parser.add_argument("--split_seed", type=int, default=DEFAULTS["split_seed"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"], help="Compromesso tra imitazione (0) ed efficacia (1).")
    parser.add_argument("--learning_rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--lambda_value", type=float, default=DEFAULTS["lambda_value"])
    parser.add_argument("--board_head", choices=["gap", "flatten"], default=DEFAULTS["board_head"])
    parser.add_argument("--policy_loss", choices=["outcome_blend"], default=DEFAULTS["policy_loss"])
    parser.add_argument("--value_loss", choices=["binary_crossentropy", "mean_squared_error"], default=DEFAULTS["value_loss"])
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=DEFAULTS["optimizer"])
    parser.add_argument("--monitor", default=DEFAULTS["monitor"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--reduce_lr_patience", type=int, default=DEFAULTS["reduce_lr_patience"])
    parser.add_argument("--reduce_lr_factor", type=float, default=DEFAULTS["reduce_lr_factor"])
    parser.add_argument("--min_learning_rate", type=float, default=DEFAULTS["min_learning_rate"])
    parser.add_argument("--save_every_steps", type=int, default=DEFAULTS["save_every_steps"])
    parser.add_argument("--max_files", type=int, default=DEFAULTS["max_files"])
    parser.add_argument("--resume", action="store_true", default=DEFAULTS["resume"], help="Riprende da out_dir/last.keras con stato optimizer.")
    parser.add_argument("--init_model", default=DEFAULTS["init_model"], help="Checkpoint .keras da cui inizializzare una nuova versione (senza stato optimizer).")
    return parser

def main() -> None:
    args = build_parser().parse_args()
    validate_settings(vars(args))
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = resolve_path(args.data_dir)
    out_dir = resolve_path(args.out_dir)
    vocab_path = resolve_path(args.vocab)
    all_files = find_npz_files(data_dir)
    num_moves = load_num_moves(vocab_path)
    # Split
    # Calcolato sempre su tutti gli shard, anche con --max_files, cosi' non
    # cambia tra un test breve e il training completo.
    split_path = (
        resolve_path(args.split_file)
        if args.split_file
        else ROOT / DEFAULT_VALIDATION_DIR
        / f"{data_dir.name}_game_split_seed{args.split_seed}.json"
    )
    validation_ids, manifest = load_or_create_split(
        files=all_files,
        split_path=split_path,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    validation_ids_sha256 = fingerprint_game_ids(
        np.asarray(sorted(validation_ids), dtype=np.int64)
    )
    files = all_files
    if args.max_files is not None:
        files = files[:args.max_files]
        print(f"ATTENZIONE: test limitato a {len(files)} shard su {len(all_files)}.")
    counts = count_positions_and_steps(
        files=files,
        validation_ids=validation_ids,
        batch_size=args.batch_size,
    )
    print(f"\nShard utilizzati: {len(files)}")
    print(f"Mosse nel vocabolario: {num_moves}")
    print(f"Posizioni training: {counts['train_positions']}")
    print(f"Posizioni validation: {counts['validation_positions']}")
    print(f"Batch training per epoca: {counts['train_steps']}")
    print(f"Batch validation per epoca: {counts['validation_steps']}")
    print(f"Policy: (1-alpha)*CrossEntropy - alpha*r*p; alpha={args.alpha}")

    # Modello (nuovo o ripreso)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.keras"
    best_path = out_dir / "best.keras"
    metric_path = out_dir / "best_metric.json"
    config_path = out_dir / "config.json"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Checkpoint per resume non trovato: {last_path}")
        check_resume_split(
            config_path,
            game_ids_sha256=manifest["game_ids_sha256"],
            validation_ids_sha256=validation_ids_sha256,
        )
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                previous_config = json.load(f)
            if previous_config.get("policy_objective") != "(1-alpha)*(-log(p_move)) - alpha*(2*y_value-1)*p_move":
                raise ValueError("Checkpoint con obiettivo Policy precedente: usa --init_model, non --resume.")
            old_args = previous_config.get("args", {})
            for key in ("alpha", "policy_loss"):
                if key in old_args and old_args[key] != getattr(args, key):
                    raise ValueError(f"--resume non puo' cambiare {key}; usa --init_model in una nuova cartella.")
        print(f"Riprendo il modello da {last_path}")
        model = tf.keras.models.load_model(last_path, compile=True)
        print("Pesi, optimizer e learning rate ripristinati dal checkpoint.")
        print("Con --resume si mantiene la compilazione del checkpoint: learning_rate, dropout, lambda_value, optimizer, loss e board_head CLI non cambiano il modello.")
    else:
        if last_path.exists() or best_path.exists():
            raise FileExistsError(f"Checkpoint gia' presenti in {out_dir}. Usa --resume oppure scegli una nuova cartella di output.")
        if args.init_model:
            initial_path = resolve_path(args.init_model)
            if not initial_path.is_file():
                raise FileNotFoundError(f"Checkpoint iniziale non trovato: {initial_path}")
            # Evita trasferimenti silenziosi con indici UCI diversi, anche
            # se i due vocabolari hanno lo stesso numero di mosse.
            source_config = initial_path.parent / "config.json"
            if source_config.is_file():
                with open(source_config, encoding="utf-8") as f:
                    source_metadata = json.load(f)
                source_vocab_arg = source_metadata.get("args", {}).get("vocab")
                if source_vocab_arg:
                    source_vocab = resolve_path(source_vocab_arg)
                    if source_vocab.is_file():
                        with open(source_vocab, encoding="utf-8") as f:
                            source_mapping = json.load(f)
                        with open(vocab_path, encoding="utf-8") as f:
                            target_mapping = json.load(f)
                        if source_mapping != target_mapping:
                            raise ValueError("Vocabolario mosse incompatibile con il checkpoint iniziale.")
                    else:
                        print("ATTENZIONE: vocabolario originario non trovato; verifica manualmente la corrispondenza degli indici UCI.")
            else:
                print("ATTENZIONE: config.json originario assente; verifica manualmente il vocabolario.")
            # Il modello viene caricato senza la compilazione precedente:
            # pesi conservati, optimizer nuovo e loss definite per questo esperimento.
            model = tf.keras.models.load_model(initial_path, compile=False)
            shapes = {tensor.name.split(":")[0]: tuple(tensor.shape[1:]) for tensor in model.inputs}
            expected_shapes = {"board": (8, 8, 12), "state": (71,), "elo_side": (1,),
                               "elo_opponent": (1,), "legal_mask": (num_moves,)}
            if shapes != expected_shapes:
                raise ValueError(f"Modello iniziale incompatibile. Input: {shapes}; attesi: {expected_shapes}")
            if set(model.output_names) != {"policy", "value"}:
                raise ValueError(f"Output modello iniziale inattesi: {model.output_names}")
            print(f"Nuovo esperimento inizializzato dai pesi di {initial_path}")
            print("ATTENZIONE: architettura/dropout provengono dal checkpoint, non dai nuovi argomenti CLI.")
        else:
            model = build_model(
                num_moves=num_moves,
                dropout_rate=args.dropout,
                board_head=args.board_head,
            )
        compile_model(
            model,
            learning_rate=args.learning_rate,
            lambda_value=args.lambda_value,
            policy_loss_name=args.policy_loss,
            value_loss_name=args.value_loss,
            optimizer_name=args.optimizer,
            alpha=args.alpha,
        )
        # config.json descrive il modello e lo split del training iniziale e
        # non viene riscritto dai resume (le loro opzioni sono in history.json).
        write_json_atomic(
            config_path,
            {
                "model": "EloChessV1",
                "created_at_utc": utc_now(),
                "args": vars(args),
                "num_moves": num_moves,
                "data_dir": str(data_dir),
                "npz_files_found": len(all_files),
                "npz_files_used": [path.name for path in files],
                "split_file": str(split_path),
                "game_ids_sha256": manifest["game_ids_sha256"],
                "validation_ids_sha256": validation_ids_sha256,
                "num_games": manifest["num_games"],
                "num_validation_games": manifest["num_validation_games"],
                "positions": counts,
                "policy_objective": "(1-alpha)*(-log(p_move)) - alpha*(2*y_value-1)*p_move",
                "init_model": str(resolve_path(args.init_model)) if args.init_model else None,
            },
        )
    # Dataset
    train_ds = make_tf_dataset(
        files=files,
        split="train",
        validation_ids=validation_ids,
        batch_size=args.batch_size,
        num_moves=num_moves,
        seed=args.seed + 1,
        num_batches=counts["train_steps"],
    )
    val_ds = make_tf_dataset(
        files=files,
        split="validation",
        validation_ids=validation_ids,
        batch_size=args.batch_size,
        num_moves=num_moves,
        seed=args.seed + 2,
        num_batches=counts["validation_steps"],
    )
    # Callback
    # L'ordine conta: i callback girano a fine epoca nell'ordine della lista.
    # ReduceLROnPlateau viene prima del salvataggio di last.keras, cosi' il
    # modello salvato porta il learning rate gia' aggiornato.
    mode = monitor_mode(args.monitor)
    callbacks = [
        BestModelCheckpoint(best_path, metric_path, args.monitor, mode),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=args.monitor,
            mode=mode,
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=args.monitor,
            mode=mode,
            patience=args.patience,
            restore_best_weights=False,  # il best e' gia' salvato in best.keras
            verbose=1,
        ),
        LastModelCheckpoint(last_path, every_steps=args.save_every_steps),
        RunHistoryLogger(out_dir / "history.json", run_args=vars(args)),
    ]
    # Training
    print(f"\nInizio training: {args.epochs} epoche, batch_size={args.batch_size}")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    print("\nTraining completato.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"History:    {out_dir / 'history.json'}")
    print(f"Config:     {config_path}")

if __name__ == "__main__":
    main()