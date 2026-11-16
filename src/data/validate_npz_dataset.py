#!/usr/bin/env python3
"""
validate_npz_dataset.py

Controlla un file NPZ prodotto da parquet_to_npz.py e ne mostra il contenuto.

Verifica:
    - presenza delle chiavi attese;
    - shape e dtype degli array;
    - range e valori di base;
    - struttura delle partite (ply consecutivi, turno, Elo e y_value coerenti);
    - su un campione di posizioni:
        - ricostruzione della posizione dalla board + state;
        - validita' della posizione ricostruita;
        - coerenza della mossa target con il vocabolario;
        - legalita' della mossa target;
        - coerenza completa della legal_mask con le mosse legali.

Esito: codice di uscita 0 se tutto e' corretto, 1 altrimenti.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.settings import DEFAULTS  # noqa: E402
from src.config.utils import resolve_path  # noqa: E402

EXPECTED_DTYPES = {
    "board": np.uint8,
    "state": np.float32,
    "elo_side": np.float32,
    "elo_opponent": np.float32,
    "move": np.int32,
    "legal_mask": np.uint8,
    "y_value": np.float32,
    "game_id": np.int64,
    "ply": np.int16,
}
EXPECTED_KEYS = set(EXPECTED_DTYPES)
PIECE_TO_TYPE = {
    0: chess.PAWN,
    1: chess.ROOK,
    2: chess.KNIGHT,
    3: chess.BISHOP,
    4: chess.QUEEN,
    5: chess.KING,
}
STATE_SIZE = 71
NUM_BOARD_PLANES = 12
EP_OFFSET = 5
EP_SIZE = 65
EP_NONE = 64
HALFMOVE_INDEX = 70
HALFMOVE_MAX = 100.0
# Devono coincidere con le costanti di parquet_to_npz.py.
ELO_OFFSET = 1000.0
ELO_SCALE = 1000.0

def load_vocab(path: str | Path) -> dict[str, int]:
    """Carica il vocabolario globale e verifica gli indici."""
    with open(path, "r", encoding="utf-8") as f:
        move2idx = json.load(f)
    if not move2idx:
        raise ValueError(f"Vocabolario vuoto: {path}")
    values = sorted(int(v) for v in move2idx.values())
    if values != list(range(len(values))):
        raise ValueError(f"Vocabolario non valido in {path}: indici non contigui da 0 a N-1")
    return {str(k): int(v) for k, v in move2idx.items()}

def load_npz_dataset(npz_path: str | Path) -> dict[str, np.ndarray]:
    """Carica l'NPZ e verifica la presenza delle chiavi richieste."""
    print(f"Carico dataset: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        missing = EXPECTED_KEYS - set(data.files)
        if missing:
            raise ValueError(f"Chiavi mancanti nell'NPZ: {sorted(missing)}")
        arrays = {key: data[key] for key in data.files}
    extra = set(arrays) - EXPECTED_KEYS
    if extra:
        print(f"Chiavi aggiuntive presenti: {sorted(extra)}")
    return arrays

def validate_shapes_and_dtypes(data: dict[str, np.ndarray]) -> int:
    """Controlla shape, dtype e dimensione comune degli array. Restituisce N."""
    board = data["board"]
    if board.ndim != 4 or board.shape[1:] != (8, 8, NUM_BOARD_PLANES):
        raise ValueError(f"Shape board non valida: {board.shape}, attesa (N, 8, 8, 12)")
    n = board.shape[0]
    if n == 0:
        raise ValueError("Il dataset e' vuoto (0 posizioni)")
    expected_shapes = {
        "state": (n, STATE_SIZE),
        "elo_side": (n,),
        "elo_opponent": (n,),
        "move": (n,),
        "y_value": (n,),
        "game_id": (n,),
        "ply": (n,),
    }
    for name, expected in expected_shapes.items():
        if data[name].shape != expected:
            raise ValueError(f"Shape {name} non valida: {data[name].shape}, attesa {expected}")
    legal_mask = data["legal_mask"]
    if legal_mask.ndim != 2 or legal_mask.shape[0] != n:
        raise ValueError(f"Shape legal_mask non valida: {legal_mask.shape}, attesa (N, num_moves)")
    for name, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = data[name].dtype
        if actual_dtype != expected_dtype:
            raise ValueError(f"dtype {name} non valido: {actual_dtype}, atteso {np.dtype(expected_dtype)}")
    print("\nShape e dtype:")
    for name in EXPECTED_DTYPES:
        print(f"  - {name:13s}: {str(data[name].shape):18s} {data[name].dtype}")
    return n

def validate_basic_ranges(data: dict[str, np.ndarray], num_moves: int) -> None:
    """Controlla range e valori fondamentali dell'intero dataset."""
    board = data["board"]
    state = data["state"]
    elo_side = data["elo_side"]
    elo_opponent = data["elo_opponent"]
    move = data["move"]
    legal_mask = data["legal_mask"]
    y_value = data["y_value"]
    game_id = data["game_id"]
    ply = data["ply"]

    # board e legal_mask sono uint8: basta controllare il massimo.
    if board.max() > 1:
        raise ValueError("board contiene valori diversi da 0 e 1")
    if board.sum(axis=3, dtype=np.int32).max() > 1:
        raise ValueError("board contiene caselle con piu' di un pezzo")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contiene valori non finiti")
    if not np.all(np.isfinite(elo_side)):
        raise ValueError("elo_side contiene valori non finiti")
    if not np.all(np.isfinite(elo_opponent)):
        raise ValueError("elo_opponent contiene valori non finiti")
    if not np.all(np.isin(state[:, 0], [0.0, 1.0])):
        raise ValueError("state[:, 0] contiene valori diversi da 0 e 1")
    if not np.all(np.isin(state[:, 1:5], [0.0, 1.0])):
        raise ValueError("castling rights contiene valori diversi da 0 e 1")
    ep_features = state[:, EP_OFFSET:EP_OFFSET + EP_SIZE]
    if not np.all(np.isin(ep_features, [0.0, 1.0])):
        raise ValueError("feature en passant contiene valori diversi da 0 e 1")
    if not np.all(ep_features.sum(axis=1) == 1):
        raise ValueError("Ogni posizione deve avere esattamente un feature en passant attivo")
    halfmove = state[:, HALFMOVE_INDEX]
    if np.any(halfmove < 0.0) or np.any(halfmove > 1.0):
        raise ValueError("halfmove_clock normalizzato fuori dal range [0, 1]")
    if np.any(move < 0) or np.any(move >= num_moves):
        raise ValueError(f"Esistono indici move fuori dal range [0, {num_moves})")
    if legal_mask.max() > 1:
        raise ValueError("legal_mask contiene valori diversi da 0 e 1")
    legal_counts = legal_mask.sum(axis=1, dtype=np.int32)
    if np.any(legal_counts == 0):
        raise ValueError("Esistono posizioni con legal_mask priva di mosse legali")
    if not np.all(np.isin(y_value, [0.0, 0.5, 1.0])):
        raise ValueError("y_value contiene valori diversi da 0.0, 0.5 e 1.0")
    if np.any(game_id < 0):
        raise ValueError("game_id contiene valori negativi")
    # ply parte da 0 (prima mossa della partita).
    if np.any(ply < 0):
        raise ValueError("ply contiene valori negativi")
    print("\nControlli base: OK")
    for label, values in (("Elo side", elo_side), ("Elo opponent", elo_opponent)):
        real = values * ELO_SCALE + ELO_OFFSET
        print(f"  - {label:13s}: min {real.min():.0f}, media {real.mean():.0f}, max {real.max():.0f}")
    print(f"  - mosse legali per posizione: min {legal_counts.min()}, media {legal_counts.mean():.1f}, max {legal_counts.max()}")
    counts = Counter(y_value.tolist())
    print("  - distribuzione y_value:")
    for value in (0.0, 0.5, 1.0):
        print(f"      {value:.1f}: {counts[value]}")

def validate_game_structure(data: dict[str, np.ndarray]) -> None:
    """
    Controlla che le partite siano intere e coerenti.

    Per ogni partita (righe contigue con lo stesso game_id):
        - ply parte da 0 e cresce di 1;
        - il turno alterna a partire dal Bianco (ply pari = Bianco);
        - elo_side/elo_opponent si scambiano a ogni mossa;
        - y_value di due mosse consecutive somma a 1.
    """
    game_id = data["game_id"]
    ply = data["ply"].astype(np.int32)
    state_turn = data["state"][:, 0]
    elo_side = data["elo_side"]
    elo_opponent = data["elo_opponent"]
    y_value = data["y_value"]
    n = len(game_id)
    new_game = np.ones(n, dtype=bool)
    new_game[1:] = game_id[1:] != game_id[:-1]
    starts = np.flatnonzero(new_game)
    n_games = len(starts)
    if len(np.unique(game_id)) != n_games:
        raise ValueError("Le righe di una stessa partita non sono contigue")
    if np.any(ply[starts] != 0):
        raise ValueError("Esistono partite che non iniziano da ply 0")
    same = ~new_game[1:]  # coppie di righe consecutive nella stessa partita
    if np.any(ply[1:][same] != ply[:-1][same] + 1):
        raise ValueError("Esistono partite con ply non consecutivi")
    if np.any((state_turn == 1.0) != (ply % 2 == 0)):
        raise ValueError("Il turno in state non e' coerente con la parita' di ply")
    if np.any(elo_side[1:][same] != elo_opponent[:-1][same]) or np.any(elo_opponent[1:][same] != elo_side[:-1][same]):
        raise ValueError("Elo non coerenti all'interno di una partita")
    if np.any(y_value[1:][same] + y_value[:-1][same] != 1.0):
        raise ValueError("y_value non coerente all'interno di una partita")
    lengths = np.diff(np.append(starts, n))
    print("\nStruttura delle partite: OK")
    print(f"  - posizioni: {n}")
    print(f"  - partite: {n_games}")
    print(f"  - posizioni per partita: min {lengths.min()}, media {lengths.mean():.1f}, max {lengths.max()}")

def reconstruct_board(planes: np.ndarray, state: np.ndarray) -> chess.Board:
    """Ricostruisce una chess.Board da board planes e state."""
    board = chess.Board.empty()
    for rank_index in range(8):
        for file_index in range(8):
            active = np.flatnonzero(planes[rank_index, file_index])
            if len(active) == 0:
                continue
            if len(active) != 1:
                raise ValueError(f"Casella ({rank_index}, {file_index}) non univoca")
            plane = int(active[0])
            piece_type = PIECE_TO_TYPE[plane % 6]
            color = chess.WHITE if plane < 6 else chess.BLACK
            square = chess.square(file_index, 7 - rank_index)
            board.set_piece_at(square, chess.Piece(piece_type, color))
    board.turn = chess.WHITE if state[0] == 1.0 else chess.BLACK
    castling = ""
    if state[1] == 1.0:
        castling += "K"
    if state[2] == 1.0:
        castling += "Q"
    if state[3] == 1.0:
        castling += "k"
    if state[4] == 1.0:
        castling += "q"
    board.set_castling_fen(castling if castling else "-")
    ep_index = int(np.argmax(state[EP_OFFSET:EP_OFFSET + EP_SIZE]))
    board.ep_square = None if ep_index == EP_NONE else ep_index
    board.halfmove_clock = min(
        int(round(float(state[HALFMOVE_INDEX]) * HALFMOVE_MAX)),
        int(HALFMOVE_MAX),
    )
    return board

def validate_sample(index: int, data: dict[str, np.ndarray], move2idx: dict[str, int], idx2move: dict[int, str]) -> tuple[bool, str | None]:
    """Valida completamente un singolo esempio."""
    try:
        board = reconstruct_board(data["board"][index], data["state"][index])
        if not board.is_valid():
            return False, "posizione ricostruita non valida"
        move_index = int(data["move"][index])
        true_uci = idx2move.get(move_index)
        if true_uci is None:
            return False, "indice move assente dal vocabolario"
        if chess.Move.from_uci(true_uci) not in board.legal_moves:
            return False, "mossa target non legale"
        mask = data["legal_mask"][index]
        if mask[move_index] != 1:
            return False, "mossa target marcata illegale nella legal_mask"
        expected = {move2idx[m.uci()] for m in board.legal_moves}
        actual = set(np.flatnonzero(mask).tolist())
        missing = expected - actual
        extra = actual - expected
        if missing and extra:
            return False, "legal_mask incoerente: mosse legali mancanti ed extra"
        if missing:
            return False, "legal_mask incoerente: mosse legali mancanti"
        if extra:
            return False, "legal_mask incoerente: mosse illegali marcate"
        return True, None
    except KeyError as exc:
        return False, f"mossa legale assente dal vocabolario: {exc}"
    except ValueError as exc:
        return False, str(exc)

def main() -> None:
    parser = argparse.ArgumentParser(description="Valida un file NPZ prodotto da parquet_to_npz.py.")
    parser.add_argument("npz", help="Percorso del file .npz da verificare.")
    parser.add_argument(
        "--vocab",
        default=DEFAULTS["vocab"],
        help="Percorso del vocabolario move2idx_all.json.",
    )
    parser.add_argument(
        "--check_n",
        type=int,
        default=2000,
        help="Numero massimo di esempi da controllare in dettaglio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Seed per il campionamento casuale.",
    )
    args = parser.parse_args()

    npz_path = resolve_path(args.npz)
    if not npz_path.exists():
        raise SystemExit(f"File NPZ non trovato: {npz_path}")

    move2idx = load_vocab(resolve_path(args.vocab))
    idx2move = {idx: uci for uci, idx in move2idx.items()}
    num_moves = len(move2idx)

    try:
        data = load_npz_dataset(npz_path)
        n = validate_shapes_and_dtypes(data)

        if data["legal_mask"].shape[1] != num_moves:
            raise ValueError(
                f"legal_mask ha {data['legal_mask'].shape[1]} colonne, "
                f"ma il vocabolario contiene {num_moves} mosse"
            )
        print(f"\nNumero classi mosse: {num_moves} (coincide con il vocabolario)")

        validate_basic_ranges(data, num_moves)
        validate_game_structure(data)
    except ValueError as exc:
        print("\n=== VALIDATION SUMMARY ===")
        print("Esito: FALLITO")
        print(f"  - {exc}")
        raise SystemExit(1)

    check_n = min(args.check_n, n)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(n, size=check_n, replace=False)

    sample_errors: Counter[str] = Counter()

    print(f"\nControllo dettagliato di {check_n} campioni (seed={args.seed})...")

    for position in trange(check_n, desc="validating samples", unit="sample"):
        ok, error = validate_sample(int(indices[position]), data, move2idx, idx2move)
        if not ok:
            sample_errors[error or "errore sconosciuto"] += 1

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Campioni controllati: {check_n} / {n}")

    if sample_errors:
        print("Esito: FALLITO")
        print("\nErrori trovati:")
        for error, count in sample_errors.items():
            print(f"  - {error}: {count}")
        raise SystemExit(1)

    print("Esito: OK")
    print("  - ricostruzione board: OK")
    print("  - validita' posizioni: OK")
    print("  - mossa target legale: OK")
    print("  - mossa target presente nella legal_mask: OK")
    print("  - legal_mask coerente con tutte le mosse legali: OK")


if __name__ == "__main__":
    main()
