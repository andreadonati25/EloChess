#!/usr/bin/env python3
"""
parquet_to_npz.py

Converte i file Parquet prodotti da extract_pgn_to_parquet.py in file NPZ
pronti per il training. Ogni file Parquet produce un file NPZ con lo stesso
nome (uno shard tecnico, non un'unita' concettuale di apprendimento).

Rappresentazione di una posizione:
    board           uint8   (N, 8, 8, 12)   6 piani bianchi + 6 piani neri
    state           float32 (N, 71)         side_to_move (1), arrocchi (4),
                                            en passant (65), halfmove clock (1)
    elo_side        float32 (N,)            Elo di chi muove, normalizzato
    elo_opponent    float32 (N,)            Elo dell'avversario, normalizzato
    move            int32   (N,)            indice della mossa in move2idx_all.json
    legal_mask      uint8   (N, num_moves)  1 = mossa legale, 0 = illegale
    y_value         float32 (N,)            1.0 vittoria, 0.5 patta, 0.0 sconfitta
    game_id         int64   (N,)            serve per lo split a livello di partita
    ply             int16   (N,)

Orientamento della scacchiera: prospettiva fissa del Bianco, riga 0 = traversa 8,
colonna 0 = colonna a. Piani: P R N B Q K (bianchi), poi gli stessi per i neri.

Stato (71 feature):
    [0]      side_to_move (1.0 = bianco, 0.0 = nero)
    [1:5]    arrocco: bianco corto, bianco lungo, nero corto, nero lungo
    [5:70]   en passant: one-hot sulle 64 case (indice python-chess, a1 = 0,
             h8 = 63) + una posizione finale (64) per "nessun en passant"
    [70]     halfmove clock / 100 (limitato a 1.0)
Il fullmove number e' escluso nella V1.

Una partita in cui anche una sola riga e' incoerente (FEN non valido, turno
diverso da side_to_move, mossa non legale) viene scartata per intero dallo shard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import DEFAULTS  # noqa: E402
from src.config.utils import fill_board, fill_state, resolve_path  # noqa: E402

DEFAULT_VOCAB = DEFAULTS["vocab"]
DEFAULT_MAXFILE = DEFAULTS["max_files"]
STATE_SIZE = 71
EP_OFFSET = 5
EP_NONE = 64
HALFMOVE_MAX = 100.0
ELO_OFFSET = 1000.0
ELO_SCALE = 1000.0
REQUIRED_COLUMNS = [
    "game_id",
    "ply",
    "fen_before",
    "side_to_move",
    "elo_side",
    "elo_opponent",
    "move_uci",
    "y_value",
]

def load_vocab(path: str | Path) -> dict[str, int]:
    """Carica il vocabolario globale e verifica che gli indici siano 0..N-1."""
    with open(path, "r", encoding="utf-8") as f:
        move2idx = json.load(f)
    if sorted(move2idx.values()) != list(range(len(move2idx))):
        raise ValueError(f"Vocabolario non valido in {path}: indici non contigui")
    return move2idx

def convert_file(parquet_path: Path, out_path: Path, move2idx: dict[str, int], show_progress: bool = True) -> dict[str, int]:
    """Converte un file Parquet in un file NPZ. Restituisce le statistiche."""
    table = pq.read_table(parquet_path, columns=REQUIRED_COLUMNS)
    n = table.num_rows
    num_moves = len(move2idx)

    game_id = table.column("game_id").to_numpy()
    ply = table.column("ply").to_numpy()
    elo_side = table.column("elo_side").to_numpy().astype(np.float32)
    elo_opponent = table.column("elo_opponent").to_numpy().astype(np.float32)
    y_value = table.column("y_value").to_numpy().astype(np.float32)
    fens = table.column("fen_before").to_pylist()
    sides = table.column("side_to_move").to_pylist()
    ucis = table.column("move_uci").to_pylist()

    board_arr = np.zeros((n, 8, 8, 12), dtype=np.uint8)
    state_arr = np.zeros((n, STATE_SIZE), dtype=np.float32)
    move_arr = np.zeros(n, dtype=np.int32)
    legal_arr = np.zeros((n, num_moves), dtype=np.uint8)
    valid = np.ones(n, dtype=bool)
    bad_games: set[int] = set()

    for i in tqdm(range(n), desc=parquet_path.name, unit="pos", leave=False, disable=not show_progress):
        try:
            if sides[i] not in ("white", "black"):
                raise ValueError("side_to_move non valido")
            board = chess.Board(fens[i])
            if board.turn != (sides[i] == "white"):
                raise ValueError("side_to_move incoerente con il FEN")
            move = chess.Move.from_uci(ucis[i])
            if move not in board.legal_moves:
                raise ValueError("mossa non legale")
        except ValueError:
            valid[i] = False
            bad_games.add(int(game_id[i]))
            continue

        try:
            move_arr[i] = move2idx[ucis[i]]
            for legal in board.legal_moves:
                legal_arr[i, move2idx[legal.uci()]] = 1
        except KeyError as exc:
            raise RuntimeError(f"Mossa {exc} assente dal vocabolario: controllare move2idx_all.json") from exc

        fill_board(board, board_arr[i])
        fill_state(board, state_arr[i])

    # Scarto per intero delle partite con almeno una riga incoerente.
    if bad_games:
        valid &= ~np.isin(game_id, np.fromiter(bad_games, dtype=game_id.dtype))

    elo_side = (elo_side - ELO_OFFSET) / ELO_SCALE
    elo_opponent = (elo_opponent - ELO_OFFSET) / ELO_SCALE

    arrays = {
        "board": board_arr,
        "state": state_arr,
        "elo_side": elo_side,
        "elo_opponent": elo_opponent,
        "move": move_arr,
        "legal_mask": legal_arr,
        "y_value": y_value,
        "game_id": game_id,
        "ply": ply,
    }
    if not valid.all():
        arrays = {k: v[valid] for k, v in arrays.items()}

    kept = int(valid.sum())
    if kept > 0:
        # Scrittura su file temporaneo e rinomina, cosi' un'interruzione non
        # lascia NPZ incompleti che il resume scambierebbe per file validi.
        tmp_path = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, out_path)
    elif out_path.exists():
        # Shard senza posizioni valide: un vecchio NPZ (es. con --overwrite)
        # sarebbe obsoleto e non deve restare.
        out_path.unlink()

    return {
        "positions_read": n,
        "positions_written": kept,
        "games_discarded": len(bad_games),
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Converte i file Parquet delle posizioni in file NPZ per il training."
    )
    parser.add_argument("--parquet_dir", required=True, help="Cartella con i file .parquet")
    parser.add_argument("--out", "-o", required=True, help="Cartella di output per gli .npz")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Percorso di move2idx_all.json")
    parser.add_argument("--max_files", type=int, default=DEFAULT_MAXFILE, help="Numero massimo di file da convertire")
    parser.add_argument("--overwrite", action="store_true", help="Riconverte anche i file NPZ gia' esistenti")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parquet_dir = resolve_path(args.parquet_dir)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    move2idx = load_vocab(resolve_path(args.vocab))
    files = sorted(parquet_dir.glob("*.parquet"))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"Nessun file .parquet trovato in {parquet_dir}")
    total = {"positions_read": 0, "positions_written": 0, "games_discarded": 0}
    skipped = 0
    for path in tqdm(files, desc="Files", unit="file"):
        out_path = out_dir / f"{path.stem}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        stats = convert_file(path, out_path, move2idx)
        for key in total:
            total[key] += stats[key]
    print()
    print("Conversione completata.")
    print(f"File convertiti:     {len(files) - skipped}")
    print(f"File saltati:        {skipped}")
    print(f"Posizioni lette:     {total['positions_read']}")
    print(f"Posizioni scritte:   {total['positions_written']}")
    print(f"Partite scartate:    {total['games_discarded']}")


if __name__ == "__main__":
    main()
