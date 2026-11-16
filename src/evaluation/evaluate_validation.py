#!/usr/bin/env python3
"""Valutazione incrociata, senza training, sulle validation di EloChess.

Esempio:
    python src/evaluation/evaluate_validation.py --models models/v1_1/best.keras models/v1_2/best.keras --months 2013-07 2013-12

Le stesse posizioni e i medesimi manifest di split sono usati per tutti i
checkpoint. Non vengono generati o modificati split, modelli o checkpoint.
Default: tutte le posizioni di validation. --sample_positions limita ogni
mese a un campione deterministico, proporzionale al numero di righe per shard.
La validazione usa le stesse metriche e loss di train_model.py, ricompilando il
checkpoint in memoria con alpha/lambda_value specificati.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.utils import resolve_path, load_num_moves  # noqa: E402
from src.config.settings import DEFAULTS  # noqa: E402

REQUIRED = ("board", "state", "elo_side", "elo_opponent", "legal_mask", "move", "y_value", "game_id")
MODEL_INPUTS = ("board", "state", "elo_side", "elo_opponent", "legal_mask")
SCORES = (
    "loss", "policy_loss", "policy_accuracy", "policy_top3", "policy_top5",
    "policy_unweighted_ce", "value_loss", "value_brier",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ValidationPlan:
    month: str
    files: list[Path]
    validation_ids: np.ndarray
    quotas: list[int]
    total: int
    selected: int
    split_sha256: str
    game_ids_sha256: str
    validation_games: int
    seed: int

    @property
    def batches(self) -> int:
        return sum(math.ceil(q / self.batch_size) for q in self.quotas if q)

    batch_size: int = DEFAULTS["batch_size"]


def distribute(counts: list[int], cap: int) -> list[int]:
    """Quote proporzionali per shard; zero (cap) significa campione completo."""
    total = sum(counts)
    if total <= 0:
        raise ValueError("Validation vuota.")
    if cap == 0 or cap >= total:
        return list(counts)
    if cap < 0:
        raise ValueError("--sample_positions deve essere >= 0.")
    expected = [cap * count / total for count in counts]
    result = [int(math.floor(value)) for value in expected]
    remaining = cap - sum(result)
    order = sorted(range(len(counts)), key=lambda i: (-(expected[i] - result[i]), i))
    for i in order[:remaining]:
        result[i] += 1
    assert sum(result) == cap and all(0 <= q <= n for q, n in zip(result, counts))
    return result


def create_plan(month: str, split_seed: int, sample_positions: int, seed: int, batch_size: int) -> ValidationPlan:
    folder = ROOT / "data" / "processed" / month
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"Nessuno shard NPZ in {folder}.")
    split_path = ROOT / "data" / "validation" / f"{month}_game_split_seed{split_seed}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Manifest esistente obbligatorio: {split_path}. Non sara' generato uno split nuovo.")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if int(manifest.get("seed", -1)) != split_seed:
        raise ValueError(f"Seed di split incompatibile: {split_path}")
    validation_set = {int(x) for x in manifest["validation_game_ids"]}
    if len(validation_set) != len(manifest["validation_game_ids"]):
        raise ValueError(f"Game ID duplicati nel manifest {split_path}")
    seen: set[int] = set()
    counts = []
    ids = np.asarray(sorted(validation_set), dtype=np.int64)
    print(f"\n[{month}] Controllo manifest e conteggio validation: {len(files)} shard", flush=True)
    for ix, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as npz:
            if "game_id" not in npz.files:
                raise ValueError(f"game_id assente: {path}")
            game_ids = npz["game_id"]
        local_ids = {int(x) for x in np.unique(game_ids)}
        duplicate = seen.intersection(local_ids)
        if duplicate:
            raise ValueError(f"Una partita compare in piu' shard: {path}; esempio: {min(duplicate)}")
        seen.update(local_ids)
        counts.append(int(np.isin(game_ids, ids).sum()))
        if ix % 50 == 0 or ix == len(files):
            print(f"  Controllati {ix}/{len(files)} shard", flush=True)
    all_ids = np.asarray(sorted(seen), dtype="<i8")
    computed_fingerprint = hashlib.sha256(all_ids.tobytes()).hexdigest()
    if computed_fingerprint != manifest.get("game_ids_sha256"):
        raise ValueError(f"Impronta game_id differente dal manifest per {month}: il dataset e' cambiato.")
    if not validation_set or not validation_set.issubset(seen):
        raise ValueError(f"Game ID della validation inesistenti nel dataset {month}.")
    if manifest.get("num_games") != len(seen) or manifest.get("num_validation_games") != len(validation_set):
        raise ValueError(f"Conteggio partite incoerente nel manifest di {month}.")
    quotas = distribute(counts, sample_positions)
    plan = ValidationPlan(
        month=month, files=files, validation_ids=ids, quotas=quotas,
        total=sum(counts), selected=sum(quotas), split_sha256=sha256_file(split_path),
        game_ids_sha256=computed_fingerprint, validation_games=len(validation_set),
        seed=seed, batch_size=batch_size,
    )
    print(f"[{month}] {plan.total:,} posizioni di validation; {plan.selected:,} selezionate; "
          f"{plan.batches:,} batch (batch_size={batch_size}).", flush=True)
    return plan


def shard_indices(plan: ValidationPlan, ix: int, game_ids: np.ndarray) -> np.ndarray:
    candidates = np.flatnonzero(np.isin(game_ids, plan.validation_ids))
    quota = plan.quotas[ix]
    if quota > len(candidates):
        raise ValueError(f"Validation mutata dopo il conteggio: {plan.files[ix]}")
    if not quota:
        return np.empty(0, dtype=np.int64)
    if quota == len(candidates):
        return candidates
    # Seed per file indipendente dall'ordine di caricamento e dal checkpoint.
    material = f"{plan.seed}|{plan.month}|{plan.files[ix].name}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.sort(np.random.default_rng(local_seed).choice(candidates, size=quota, replace=False))


def iter_batches(plan: ValidationPlan, num_moves: int):
    """Stessa codifica degli input/target di train_model.iter_shard_batches()."""
    for ix, path in enumerate(plan.files):
        if not plan.quotas[ix]:
            continue
        with np.load(path, allow_pickle=False) as npz:
            missing = set(REQUIRED) - set(npz.files)
            if missing:
                raise ValueError(f"Chiavi mancanti in {path}: {sorted(missing)}")
            arrays = {key: npz[key] for key in REQUIRED}
        n = len(arrays["game_id"])
        if (arrays["board"].shape != (n, 8, 8, 12)
                or arrays["state"].shape != (n, 71)
                or arrays["legal_mask"].shape != (n, num_moves)):
            raise ValueError(f"Shape incompatibili in {path}")
        if any(len(a) != n for a in arrays.values()):
            raise ValueError(f"Dimensioni incoerenti in {path}")
        indices = shard_indices(plan, ix, arrays["game_id"])
        for start in range(0, len(indices), plan.batch_size):
            idx = indices[start:start + plan.batch_size]
            y_move = arrays["move"][idx].astype(np.int32, copy=False)
            y_value = arrays["y_value"][idx].astype(np.float32, copy=False)
            if np.any(y_move < 0) or np.any(y_move >= num_moves):
                raise ValueError(f"Indice UCI non valido in {path}")
            if not np.all(np.isin(y_value, [0.0, 0.5, 1.0])):
                raise ValueError(f"y_value non valido in {path}")
            inputs = {
                "board": arrays["board"][idx].astype(np.float32),
                "state": arrays["state"][idx].astype(np.float32),
                "elo_side": arrays["elo_side"][idx, None].astype(np.float32),
                "elo_opponent": arrays["elo_opponent"][idx, None].astype(np.float32),
                "legal_mask": arrays["legal_mask"][idx].astype(np.float32),
            }
            targets = {
                "policy": np.column_stack((y_move, y_value)).astype(np.float32),
                "value": y_value[:, None],
            }
            weights = {
                "policy": np.ones(len(idx), dtype=np.float32),
                "value": np.ones(len(idx), dtype=np.float32),
            }
            yield inputs, targets, weights


def make_dataset(plan: ValidationPlan, num_moves: int, tf):
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
    ds = tf.data.Dataset.from_generator(lambda: iter_batches(plan, num_moves), output_signature=output_signature)
    ds = ds.apply(tf.data.experimental.assert_cardinality(plan.batches))
    return ds.prefetch(1)


def save_json(path: Path, payload: dict):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def emit_summary(output: Path) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "results").glob("*.json"))]
    if not rows:
        return
    fields = ["model", "month", "positions_evaluated", "validation_positions_total",
              "policy_accuracy", "policy_top3", "policy_top5", "policy_unweighted_ce",
              "policy_loss", "value_brier", "value_loss", "loss", "alpha", "lambda_value",
              "sample_positions", "seed", "batch_size", "checkpoint_sha256",
              "split_sha256", "created_at_utc", "checkpoint"]
    tmp = output / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["month"], r["model"], r["sample_positions"], r["checkpoint_sha256"])))
    tmp.replace(output / "summary.csv")

    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    report = [
        "# EloChess — Confronto incrociato delle validation",
        "",
        "Il file `summary.csv` contiene le misure originali. Le percentuali nel report sono",
        "arrotondate per leggibilita'. Ogni riga usa esclusivamente le partite di validation",
        "del manifest del rispettivo mese, senza training e senza analisi Stockfish.",
        "",
        "| Mese | Checkpoint | Posizioni | Top-1 umana | Top-5 umana | CE umana | Value MSE | Policy loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda r: (r["month"], r["model"])):
        report.append(f"| {r['month']} | {r['model']} | {r['positions_evaluated']:,} | "
                      f"{pct(r['policy_accuracy'])} | {pct(r['policy_top5'])} | "
                      f"{r['policy_unweighted_ce']:.4f} | {r['value_brier']:.5f} | "
                      f"{r['policy_loss']:.4f} |")
    report += [
        "", "## Significato delle metriche", "",
        "- **Top-1/Top-5 umana:** frazione delle posizioni in cui la mossa realmente",
        "  giocata compare al primo posto / fra le prime cinque mosse della Policy.",
        "- **CE umana:** media di `-log(p_mossa_umano)`; piu' piccola e' meglio.",
        "- **Value MSE:** media di `(V - risultato)^2`, dove il risultato e' 0/0,5/1",
        "  per il giocatore al tratto; piu' piccola e' meglio.",
        "- **Policy loss:** obiettivo di training con alpha fissato (default 0.5);",
        "  non equivale alla CE umana ne' alla forza scacchistica.",
        "- Le due validation hanno distribuzioni diverse: confrontare checkpoint",
        "  **all'interno della stessa riga-mese**, non i valori assoluti luglio vs dicembre.",
        "- Le validation sono state usate anche per selezionare `best.keras` durante",
        "  l'allenamento del rispettivo mese: sono separate dal training ma non",
        "  completamente indipendenti dalla selezione dei checkpoint.",
        "- Un campione `--sample_positions` e' deterministico e proporzionale per",
        "  shard, non e' una nuova suddivisione delle partite.",
        "- Non misura la qualita' causale delle mosse, Elo effettivo o prestazioni",
        "  in partite giocate da EloChess.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport: {output / 'REPORT.md'}\nTabella: {output / 'summary.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confronta checkpoint EloChess sulle validation fisse di piu' mesi, senza addestramento.")
    parser.add_argument("--models", nargs="+", required=True, help="Checkpoint best.keras da confrontare.")
    parser.add_argument("--months", nargs="+", required=True, help="Mesi dei dataset NPZ, es. 2013-07 2013-12.")
    parser.add_argument("--split_seed", type=int, default=DEFAULTS["split_seed"])
    parser.add_argument("--sample_positions", type=int, default=0, help="0 = validation intera per ogni mese; >0 = campione fisso per mese.")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="Seed per il campione di validation; indipendente dal training.")
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    parser.add_argument("--lambda_value", type=float, default=DEFAULTS["lambda_value"])
    parser.add_argument("--vocab", default=DEFAULTS["vocab"])
    parser.add_argument("--out_dir", default="archive/manual_inspection/validation_crosscheck")
    parser.add_argument("--overwrite", action="store_true", help="Ricalcola anche le combinazioni gia' valutate con gli stessi parametri.")
    parser.add_argument("--dry_run", action="store_true", help="Verifica manifest, dataset e campioni senza importare TensorFlow.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.sample_positions < 0 or not 0 <= args.alpha <= 1 or args.lambda_value < 0:
        parser.error("batch_size>0, sample_positions>=0, alpha fra 0 e 1, lambda_value>=0 richiesti.")
    if len(set(args.months)) != len(args.months) or len(set(args.models)) != len(args.models):
        parser.error("Nessun mese o modello duplicato.")
    models = [resolve_path(value).resolve() for value in args.models]
    for model in models:
        if not model.is_file():
            raise FileNotFoundError(f"Checkpoint mancante: {model}")
    output = resolve_path(args.out_dir)
    num_moves = load_num_moves(resolve_path(args.vocab))
    plans = [create_plan(month, args.split_seed, args.sample_positions, args.seed, args.batch_size)
             for month in args.months]
    print("\nMatrice prevista:")
    for model in models:
        for plan in plans:
            print(f"  {model.parent.name}/{model.name} × {plan.month}: {plan.selected:,} posizioni")
    if args.dry_run:
        print("\n--dry_run concluso: nessun modello caricato e nessun file modificato.")
        return

    import tensorflow as tf
    from src.model.build_model import compile_model

    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(parents=True, exist_ok=True)
    for model_path in models:
        label = model_path.parent.name if model_path.name == "best.keras" else f"{model_path.parent.name}_{model_path.stem}"
        fingerprint = sha256_file(model_path)
        config_path = model_path.parent / "config.json"
        if config_path.is_file():
            source_config = json.loads(config_path.read_text(encoding="utf-8"))
            if int(source_config.get("num_moves", -1)) != num_moves:
                raise ValueError(f"Vocabolario incompatibile col checkpoint {model_path}")
            original_alpha = source_config.get("args", {}).get("alpha")
            if original_alpha is not None and not math.isclose(float(original_alpha), args.alpha, abs_tol=1e-9):
                raise ValueError(f"Alpha {args.alpha} diverso da alpha del checkpoint {label}: {original_alpha}")
        for plan in plans:
            tag = (f"{label}_{plan.month}_{fingerprint[:12]}_" 
                   f"n{plan.selected}_s{plan.seed}_b{plan.batch_size}_a{args.alpha}_l{args.lambda_value}")
            result_path = output / "results" / f"{tag}.json"
            if result_path.is_file() and not args.overwrite:
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if (previous.get("split_sha256") == plan.split_sha256
                        and previous.get("checkpoint_sha256") == fingerprint):
                    print(f"Gia' valutato, riutilizzo: {label} / {plan.month}")
                    continue
            print(f"\nVALUTAZIONE: {label} / {plan.month} ({plan.selected:,} posizioni)", flush=True)
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(model_path, compile=False)
            if set(model.output_names) != {"policy", "value"}:
                raise ValueError(f"Output non compatibile per {model_path}: {model.output_names}")
            compile_model(model, alpha=args.alpha, lambda_value=args.lambda_value)
            dataset = make_dataset(plan, num_moves, tf)
            measures = model.evaluate(dataset, return_dict=True, verbose=1)
            del dataset, model
            gc.collect()
            if not all(key in measures for key in SCORES):
                raise ValueError(f"Metriche mancanti in model.evaluate: {set(SCORES) - set(measures)}; disponibili {sorted(measures)}")
            row = {
                "model": label, "checkpoint": str(model_path), "checkpoint_sha256": fingerprint,
                "month": plan.month, "validation_positions_total": plan.total,
                "positions_evaluated": plan.selected, "validation_games": plan.validation_games,
                "sample_positions": args.sample_positions, "seed": args.seed,
                "batch_size": args.batch_size, "split_sha256": plan.split_sha256,
                "game_ids_sha256": plan.game_ids_sha256, "alpha": args.alpha,
                "lambda_value": args.lambda_value,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **{key: float(measures[key]) for key in SCORES},
            }
            save_json(result_path, row)
            print(f"  Top-1: {row['policy_accuracy']:.2%} | CE: {row['policy_unweighted_ce']:.4f} | "
                  f"Value MSE: {row['value_brier']:.5f}", flush=True)
    emit_summary(output)


if __name__ == "__main__":
    main()
