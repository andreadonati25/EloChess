#!/usr/bin/env python3
"""
extract_pgn_to_parquet.py

Estrae da un archivio Lichess PGN.ZST le posizioni precedenti
a ogni mossa e le salva in Parquet.

Schema:
    game_id
    ply
    fen_before
    side_to_move
    elo_side
    elo_opponent
    move_uci
    result
    y_value
    termination

Una partita con Elo mancante/non valido, risultato non valido
o sequenza di mosse incoerente viene scartata interamente.
"""

from __future__ import annotations

import argparse
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import io
from pathlib import Path
from typing import TextIO, BinaryIO

import chess
import chess.pgn
import zstandard as zstd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.utils import resolve_path  # noqa: E402

SCHEMA = pa.schema(
    [("game_id", pa.int64()), ("ply", pa.int16()), ("fen_before", pa.string()), ("side_to_move", pa.string()),
    ("elo_side", pa.int16()), ("elo_opponent", pa.int16()), ("move_uci", pa.string()),
    ("result", pa.string()), ("y_value", pa.float32()), ("termination", pa.string())]
)
GAMES_PER_PART = 1500
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}

def open_pgn_stream(pgn_zst_path: str | Path) -> tuple[BinaryIO, TextIO]:
    """
    Apre un archivio PGN.ZST e restituisce:
        - il file binario sottostante;
        - il TextIO necessario a python-chess.

    Il file binario viene restituito per permettere al chiamante
    di chiudere correttamente tutte le risorse.
    """
    fh = open(pgn_zst_path, "rb")
    dctx = zstd.ZstdDecompressor()
    stream = dctx.stream_reader(fh)
    text_stream = io.TextIOWrapper(
        stream,
        encoding="utf-8",
        errors="replace",
    )
    return fh, text_stream

def parse_elo(value: str | None) -> int | None:
    """Converte un Elo in intero, oppure restituisce None se non valido."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        elo = int(value)
    except (TypeError, ValueError):
        return None
    if elo <= 0:
        return None
    return elo

def result_to_value(result: str, side_to_move: chess.Color) -> float:
    """
    Converte il risultato della partita nel valore dal punto di vista
    del giocatore che deve muovere.

        vittoria -> 1.0
        patta    -> 0.5
        sconfitta -> 0.0
    """
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        raise ValueError(f"Risultato non valido: {result!r}")
    return 1.0 if side_to_move == winner else 0.0

def validate_game_headers(game: chess.pgn.Game) -> tuple[int, int, str] | None:
    """
    Valida gli header necessari.

    Restituisce:
        (white_elo, black_elo, result)

    oppure None se la partita deve essere scartata.
    """
    white_elo = parse_elo(game.headers.get("WhiteElo"))
    black_elo = parse_elo(game.headers.get("BlackElo"))
    result = game.headers.get("Result", "").strip()
    if white_elo is None or black_elo is None:
        return None
    if result not in VALID_RESULTS:
        return None
    return white_elo, black_elo, result

def write_part(path: Path, rows: list[list]) -> None:
    """Converte le righe accumulate in una Table e le scrive in un file Parquet."""
    columns = list(zip(*rows))
    arrays = [
        pa.array(col, type=field.type)
        for col, field in zip(columns, SCHEMA)
    ]
    table = pa.Table.from_arrays(arrays, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd")

def extract_game_rows(game: chess.pgn.Game, game_id: int) -> list[list]:
    """
    Estrae tutte le righe di una singola partita.

    Se la partita è invalida, restituisce [].

    read_game() non solleva eccezioni sulle mosse illegali o sul SAN
    non valido: registra l'errore in game.errors e tronca la partita.
    Una partita troncata avrebbe un y_value fuorviante, quindi viene
    scartata per intero.
    """
    headers = validate_game_headers(game)
    if headers is None:
        return []
    if game.errors:
        return []
    white_elo, black_elo, result = headers
    termination = game.headers.get("Termination", "")
    board = game.board()
    rows = []
    for ply, move in enumerate(game.mainline_moves()):
        side_to_move = board.turn
        is_white = side_to_move == chess.WHITE
        rows.append(
            [
                game_id,
                ply,
                board.fen(),
                "white" if is_white else "black",
                white_elo if is_white else black_elo,
                black_elo if is_white else white_elo,
                move.uci(),
                result,
                result_to_value(result, side_to_move),
                termination,
            ]
        )
        board.push(move)
    return rows

def extract_games(pgn_zst_path: str | Path, output: str | Path, *, start_game_id: int = 0, max_games: int | None = None, show_progress: bool = True) -> dict[str, int]:
    """
    Parameters:
        pgn_zst_path:
            Archivio PGN.ZST.
        output:
            Cartella di destinazione dei file Parquet.
        start_game_id:
            ID globale da assegnare alla prima partita letta.
        max_games:
            Numero massimo di partite da leggere.
            None = fino alla fine del file.
        show_progress:
            Mostra una progress bar.

    Returns:
        dict con statistiche dell'estrazione.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = Path(pgn_zst_path).name.removesuffix(".pgn.zst")
    stats = {
        "games_read": 0,
        "games_written": 0,
        "games_discarded": 0,
        "positions_written": 0,
    }
    raw_fh, pgn_stream = open_pgn_stream(pgn_zst_path)
    buffer: list[list] = []
    games_in_part = 0
    part_idx = 0
    try:
        progress = tqdm(
            total=max_games,
            desc="Games",
            unit="game",
            disable=not show_progress,
        )
        try:
            while max_games is None or stats["games_read"] < max_games:
                try:
                    game = chess.pgn.read_game(pgn_stream)
                except Exception as exc:
                    print(f"Errore nella lettura PGN: {exc}")
                    break
                if game is None:
                    break
                current_game_id = start_game_id + stats["games_read"]
                stats["games_read"] += 1
                rows = extract_game_rows(game, current_game_id)
                if not rows:
                    stats["games_discarded"] += 1
                else:
                    buffer.extend(rows)
                    games_in_part += 1
                    stats["games_written"] += 1
                    stats["positions_written"] += len(rows)
                    if games_in_part >= GAMES_PER_PART:
                        write_part(output / f"{name}_part_{part_idx:04d}.parquet", buffer)
                        part_idx += 1
                        buffer = []
                        games_in_part = 0
                progress.update(1)
        finally:
            progress.close()
        if buffer:
            write_part(output / f"{name}_part_{part_idx:04d}.parquet", buffer)
    finally:
        pgn_stream.close()
        raw_fh.close()
    return stats

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estrae posizioni da un archivio Lichess PGN.ZST in Parquet."
    )
    parser.add_argument(
        "--pgn_zst",
        required=True,
        help="Percorso al file .pgn.zst",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Percorso di output",
    )
    parser.add_argument(
        "--start_game_id",
        type=int,
        default=0,
        help="Game ID globale della prima partita estratta",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=None,
        help="Numero massimo di partite da leggere",
    )
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    stats = extract_games(
        pgn_zst_path=resolve_path(args.pgn_zst),
        output=resolve_path(args.out),
        start_game_id=args.start_game_id,
        max_games=args.max_games,
    )

    print()
    print("Estrazione completata.")
    print(f"Partite lette:       {stats['games_read']}")
    print(f"Partite valide:      {stats['games_written']}")
    print(f"Partite scartate:    {stats['games_discarded']}")
    print(f"Posizioni estratte:  {stats['positions_written']}")


if __name__ == "__main__":
    main()