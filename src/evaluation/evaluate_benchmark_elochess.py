#!/usr/bin/env python3
"""Valuta UN checkpoint EloChess su un benchmark con analisi Stockfish salvata.

Modalita' hybrid (default): Elo originali nelle posizioni Lichess; griglia Elo
nelle posizioni delle suite. In modalita' grid la griglia vale per ogni FEN.

Nessun nuovo calcolo Stockfish, nessun addestramento, nessun salvataggio
incrementale. I risultati di modelli distinti sono in cartelle distinte.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import DEFAULTS  # noqa: E402
from src.config.utils import resolve_path  # noqa: E402

def encode_position(board: chess.Board, elo_side: int, elo_opponent: int, move2idx: dict[str, int]) -> dict[str, np.ndarray]:
    """Codifica FEN -> input del modello usando le funzioni condivise."""
    from src.config.utils import fill_board, fill_state
    pieces = np.zeros((8, 8, 12), dtype=np.float32)
    state = np.zeros(71, dtype=np.float32)
    fill_board(board, pieces)
    fill_state(board, state)

    mask = np.zeros(len(move2idx), dtype=np.float32)
    for move in board.legal_moves:
        index = move2idx.get(move.uci())
        if index is None:
            raise ValueError(f"Mossa legale assente nel vocabolario: {move.uci()}")
        mask[index] = 1.0
    if mask.sum() == 0:
        raise ValueError("La posizione non contiene mosse legali: impossibile interrogare la Policy.")

    return {
        "board": pieces[None],
        "state": state[None],
        "elo_side": np.array([[(elo_side - 1000) / 1000]], dtype=np.float32),
        "elo_opponent": np.array([[(elo_opponent - 1000) / 1000]], dtype=np.float32),
        "legal_mask": mask[None],
    }

DEFAULT_ELOS = [500, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400]
BASIC_COLUMNS = ("position_id", "fen", "source", "source_file", "theme")
OPTIONAL_COLUMNS = (
    "source_id", "stage", "elo_band", "game_id", "ply", "side_to_move",
    "white_elo", "black_elo", "elo_side_original", "elo_opponent_original",
    "played_move_uci", "result", "y_value_to_move",
)
INPUT_NAMES = ("board", "state", "elo_side", "elo_opponent", "legal_mask")


def benchmark_digest(rows: list[dict]) -> str:
    payload = [(r["position_id"], r["fen"]) for r in rows]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_positions(path: Path) -> list[dict]:
    schema = pq.read_schema(path)
    missing = set(BASIC_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"Benchmark privo di colonne obbligatorie: {sorted(missing)}")
    columns = [c for c in (*BASIC_COLUMNS, *OPTIONAL_COLUMNS) if c in schema.names]
    rows = pq.read_table(path, columns=columns).to_pylist()
    if not rows:
        raise ValueError("Benchmark vuoto")
    ids = [r["position_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("position_id duplicati nel benchmark")
    for row in rows:
        board = chess.Board(row["fen"])
        if not board.is_valid() or board.is_game_over(claim_draw=False):
            raise ValueError(f"Posizione non valida o terminale: {row['position_id']}")
    return rows


def grid_pairs(elos: list[int]) -> list[tuple[int, int]]:
    """Stessa regola usata nel benchmark preliminare (anche coppie invertite)."""
    elos = sorted(elos)

    def allowed(e1: int, e2: int) -> bool:
        weaker, stronger = min(e1, e2), max(e1, e2)
        limit = next((elo for elo in elos if elo >= 2 * weaker), elos[-1])
        return stronger <= limit

    return [(a, b) for a in elos for b in elos if allowed(a, b)]


def elo_settings(row: dict, mode: str, pairs: list[tuple[int, int]]):
    """Ritorna (elo_al_tratto, elo_avversario, origine_Elo)."""
    original_a = row.get("elo_side_original")
    original_b = row.get("elo_opponent_original")
    if (original_a is None) != (original_b is None):
        raise ValueError(f"Elo originali incompleti: {row['position_id']}")
    if mode == "hybrid" and original_a is not None:
        if int(original_a) <= 0 or int(original_b) <= 0:
            raise ValueError(f"Elo originali non validi: {row['position_id']}")
        yield int(original_a), int(original_b), "original"
    else:
        for e1, e2 in pairs:
            yield e1, e2, "grid"


def validate_stockfish(benchmark: Path, all_rows: list[dict], sf_path: Path, top_n: int):
    meta_path = sf_path.with_suffix(".metadata.json")
    if not sf_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"Analisi Stockfish incompleta: {sf_path} / {meta_path}. "
            "Esegui prima evaluate_benchmark_stockfish.py."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("benchmark_sha256") != benchmark_digest(all_rows):
        raise ValueError("I dati Stockfish non corrispondono al benchmark (SHA256)")
    if metadata.get("top_n", 0) < top_n:
        raise ValueError(f"Stockfish ha MultiPV {metadata.get('top_n')}, richiesto {top_n}")
    sf_rows = pq.read_table(sf_path).to_pylist()
    if len(sf_rows) != len(all_rows) or any(
        sf["position_id"] != row["position_id"] or sf["fen"] != row["fen"]
        for sf, row in zip(sf_rows, all_rows)
    ):
        raise ValueError("L'ordine o le FEN Stockfish non corrispondono al benchmark")
    if any(not row.get("moves") for row in sf_rows):
        raise ValueError("Analisi Stockfish con mosse mancanti")
    return sf_rows


def analyze(args, rows: list[dict], sf_rows: list[dict], model_path: Path):
    import tensorflow as tf
    from src.model.build_model import LegalMaskSoftmax  # noqa: F401 (registrazione Keras)

    with open(resolve_path(args.vocab), encoding="utf-8") as handle:
        move2idx = json.load(handle)
    if sorted(move2idx.values()) != list(range(len(move2idx))):
        raise ValueError("Vocabolario di mosse con indici non contigui")
    idx2move = {idx: uci for uci, idx in move2idx.items()}
    pairs = grid_pairs(args.elos)
    model = tf.keras.models.load_model(model_path, compile=False)
    results: list[dict] = []
    counts: Counter = Counter()

    # I risultati sono salvati solo alla fine; i blocchi servono unicamente
    # a limitare la RAM per codifica degli input e inferenza del modello.
    for start in tqdm(range(0, len(rows), args.positions_per_chunk), desc="EloChess", unit="chunk"):
        block_rows = rows[start:start + args.positions_per_chunk]
        block_sf = sf_rows[start:start + args.positions_per_chunk]
        encoded_arrays = {name: [] for name in INPUT_NAMES}
        keys = []
        for row, sf in zip(block_rows, block_sf):
            board = chess.Board(row["fen"])
            human_move = row.get("played_move_uci")
            if human_move is not None and human_move not in {m.uci() for m in board.legal_moves}:
                raise ValueError(f"Mossa umana non legale nella FEN: {row['position_id']} / {human_move}")
            for elo_side, elo_opponent, setting in elo_settings(row, args.elo_mode, pairs):
                encoded = encode_position(board, elo_side, elo_opponent, move2idx)
                for name in INPUT_NAMES:
                    encoded_arrays[name].append(encoded[name][0])
                keys.append((row, sf, board.turn, elo_side, elo_opponent, setting))
                counts[setting] += 1

        inputs = {name: np.asarray(values, dtype=np.float32)
                  for name, values in encoded_arrays.items()}
        for offset in range(0, len(keys), args.batch_size):
            stop = min(offset + args.batch_size, len(keys))
            batch = {name: arr[offset:stop] for name, arr in inputs.items()}
            prediction = model(batch, training=False)
            if not isinstance(prediction, dict) or set(prediction) != {"policy", "value"}:
                raise ValueError("Il modello deve restituire il dizionario {'policy', 'value'}")
            policy = np.asarray(prediction["policy"])
            values = np.asarray(prediction["value"]).reshape(-1)
            if policy.shape != (stop - offset, len(move2idx)) or values.shape != (stop - offset,):
                raise ValueError(f"Shape delle predizioni non valida: {policy.shape}, {values.shape}")
            if not np.isfinite(policy).all() or not np.isfinite(values).all():
                raise ValueError("Le predizioni contengono NaN o infinito")

            for i in range(stop - offset):
                row, sf, white_turn, elo_side, elo_opponent, setting = keys[offset + i]
                legal = np.flatnonzero(batch["legal_mask"][i])
                if not len(legal):
                    raise ValueError(f"Nessuna mossa legale: {row['position_id']}")
                p_legal = policy[i, legal]
                if np.any(p_legal < 0):
                    raise ValueError(f"Probabilita' negative: {row['position_id']}")
                ranked = legal[np.argsort(-p_legal, kind="stable")]
                moves = [
                    {"rank": rank, "uci": idx2move[int(idx)],
                     "probability": float(policy[i, idx])}
                    for rank, idx in enumerate(ranked[:args.top_n], 1)
                ]
                sf_moves = sf["moves"]
                sf_ranks = {m["uci"]: m["rank"] for m in sf_moves}
                observed = row.get("played_move_uci")
                human_prob = None
                human_rank = None
                human_agreement = None
                if observed is not None and setting == "original":
                    if observed not in move2idx:
                        raise ValueError(f"Mossa umana fuori vocabolario: {observed}")
                    human_prob = float(policy[i, move2idx[observed]])
                    human_rank = int(np.flatnonzero(ranked == move2idx[observed])[0]) + 1
                    human_agreement = moves[0]["uci"] == observed
                y = row.get("y_value_to_move")
                value = float(values[i])
                results.append({
                    "position_id": row["position_id"], "fen": row["fen"],
                    "source": row["source"], "source_file": row["source_file"],
                    "source_id": row.get("source_id"), "theme": row.get("theme"),
                    "stage": row.get("stage"), "elo_band": row.get("elo_band"),
                    "game_id": row.get("game_id"), "ply": row.get("ply"),
                    "model": model_path.parent.name, "model_path": str(model_path),
                    "elo_side": elo_side, "elo_opponent": elo_opponent,
                    "elo_setting": setting,
                    "elo_side_original": row.get("elo_side_original"),
                    "elo_opponent_original": row.get("elo_opponent_original"),
                    "side_to_move": "white" if white_turn else "black",
                    "value_expected_score_to_move": value,
                    "value_expected_score_white_signed": (2 * value - 1) * (1 if white_turn else -1),
                    "played_move_uci": observed, "result": row.get("result"),
                    "y_value_to_move": y,
                    "value_squared_error_to_move": None if y is None else (value - float(y)) ** 2,
                    "human_move_probability": human_prob,
                    "human_move_rank": human_rank,
                    "human_top1_agreement": human_agreement,
                    "elochess_top1_uci": moves[0]["uci"],
                    "sf_top1_uci": sf_moves[0]["uci"],
                    "sf_top1_cp_white": sf_moves[0]["cp_white"],
                    "sf_top1_mate_white": sf_moves[0]["mate_white"],
                    "top1_agreement": moves[0]["uci"] == sf_moves[0]["uci"],
                    "top1_in_sf_top_n": moves[0]["uci"] in sf_ranks,
                    "elochess_moves": moves,
                })

    tf.keras.backend.clear_session()
    return results, pairs, counts


def summarize(rows: list[dict], out: Path):
    path = out.with_name("summary.csv")
    result = {}
    for row in rows:
        key = (row["source"], row["elo_setting"])
        group = result.setdefault(key, {"n": 0, "top1": 0, "topn": 0,
                                        "human_n": 0, "human_top1": 0,
                                        "value_squared_error": 0.0,
                                        "human_prob": 0.0})
        group["n"] += 1
        group["top1"] += int(row["top1_agreement"])
        group["topn"] += int(row["top1_in_sf_top_n"])
        if row["human_top1_agreement"] is not None:
            group["human_n"] += 1
            group["human_top1"] += int(row["human_top1_agreement"])
            group["value_squared_error"] += row["value_squared_error_to_move"]
            group["human_prob"] += row["human_move_probability"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("source", "elo_setting", "predictions", "sf_top1_agreement",
                         "sf_top_n_coverage", "human_predictions", "human_top1_accuracy",
                         "mean_human_move_probability", "value_mse_to_move"))
        for (source, setting), item in sorted(result.items()):
            n, hn = item["n"], item["human_n"]
            writer.writerow((source, setting, n, item["top1"] / n,
                             item["topn"] / n, hn,
                             item["human_top1"] / hn if hn else "",
                             item["human_prob"] / hn if hn else "",
                             item["value_squared_error"] / hn if hn else ""))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, help="File benchmark .parquet")
    ap.add_argument("--stockfish", default=None, help="Parquet Stockfish (default: cartella del benchmark)")
    ap.add_argument("--model", required=True, help="Un solo checkpoint .keras")
    ap.add_argument("--output", default=None, help="Percorso Parquet output; default: cartella del modello")
    ap.add_argument("--elo_mode", choices=("hybrid", "grid"), default="hybrid")
    ap.add_argument("--elos", nargs="+", type=int, default=DEFAULT_ELOS)
    ap.add_argument("--top_n", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--positions_per_chunk", type=int, default=50)
    ap.add_argument("--max_positions", type=int, default=None, help="Limita solo per smoke test")
    ap.add_argument("--vocab", default=DEFAULTS["vocab"])
    args = ap.parse_args()

    if not args.elos or len(set(args.elos)) != len(args.elos) or any(e <= 0 for e in args.elos):
        ap.error("--elos richiede Elo positivi e distinti")
    if any(value < 1 for value in (args.top_n, args.batch_size, args.positions_per_chunk)):
        ap.error("--top_n, --batch_size e --positions_per_chunk devono essere positivi")
    if args.max_positions is not None and args.max_positions < 1:
        ap.error("--max_positions deve essere positivo")

    bench = resolve_path(args.benchmark)
    if not bench.is_file():
        ap.error(f"Benchmark non trovato: {bench}")
    model_path = resolve_path(args.model)
    if not model_path.is_file():
        ap.error(f"Modello non trovato: {model_path}")
    sf_path = resolve_path(args.stockfish) if args.stockfish else bench.parent / bench.stem / "stockfish.parquet"
    out = (resolve_path(args.output) if args.output else
           bench.parent / bench.stem / f"elochess_{model_path.parent.name}" / "results.parquet")
    metadata_path = out.with_suffix(".metadata.json")
    summary_path = out.with_name("summary.csv")
    if any(path.exists() for path in (out, metadata_path, summary_path)):
        ap.error(f"Output gia' presente nella cartella {out.parent}; scegli un nuovo --output o sposta i vecchi")

    all_rows = load_positions(bench)
    sf_all = validate_stockfish(bench, all_rows, sf_path, args.top_n)
    count = len(all_rows) if args.max_positions is None else min(args.max_positions, len(all_rows))
    rows = all_rows[:count]
    sf_rows = sf_all[:count]

    results, pairs, counts = analyze(args, rows, sf_rows, model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(results), out, compression="zstd")
    summary = summarize(results, out)
    meta = {
        "benchmark": str(bench), "benchmark_sha256": benchmark_digest(all_rows),
        "stockfish": str(sf_path), "model": str(model_path),
        "elo_mode": args.elo_mode, "elos_grid": sorted(args.elos),
        "elo_pairing_rule": "symmetric_max_ratio_2_with_next_available",
        "grid_pairs": len(pairs), "top_n": args.top_n,
        "positions_evaluated": len(rows), "n_predictions": len(results),
        "predictions_by_elo_setting": dict(counts),
        "stockfish_pov": "white", "elochess_value_pov": "side_to_move",
        "human_evaluation_only_on_original_elo": args.elo_mode == "hybrid",
    }
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Risultati: {out} ({len(results)} predizioni)")
    print(f"Riepilogo: {summary}; metadata: {metadata_path}")
    print(f"Origine Elo: {dict(counts)}; griglia: {len(pairs)} coppie")


if __name__ == "__main__":
    main()
