#!/usr/bin/env python3
"""Costruisce da zero il benchmark EloChess V1: 500 EPD + 4500 Lichess.

Non consulta l'archivio e non esegue analisi Stockfish/EloChess.
Le 500 posizioni EPD usano esattamente la procedura del generatore
originale (seed, stratificazione, deduplicazione e ordine di selezione).

Esempio dalla radice del repository:
    python src/evaluation/build_benchmark.py --month 2013-04 --seed 42

Richiede gli EPD locali in data/raw/benchmark_sources/ e il PGN.ZST del mese.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.utils import resolve_path  # noqa: E402

DEFAULT_SOURCES = ROOT / "data/raw/benchmark_sources"
DEFAULT_OUT = ROOT / "data/evaluation/benchmark_v1_5000.parquet"
STS_FILE = "STS1-STS15_LAN_v5.epd"
CHESS_FILES = (
    "benchmark.epd", "bratko-kopec.epd", "fortresses.epd", "kaufman.epd",
    "lct2.epd", "nolot.epd", "silent-but-deadly.epd", "rybka-forum.epd",
)
TARGETS = {"STS": 280, "Chess-EPDs": 180, "CCET": 40}
FEN_TRAILER = "0 1"
SUITE_TARGET = sum(TARGETS.values())
LICHESS_TARGET = 4500
STAGES = ("opening", "middlegame", "endgame")
COLORS = ("white", "black")
ELO_BANDS = ("low", "middle", "high")
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}

# Le prime dieci colonne coincidono con il precedente benchmark EPD.
# Campi non applicabili alle suite EPD rimangono NULL.
SCHEMA = pa.schema([
    ("position_id", pa.string()),
    ("fen", pa.string()),
    ("source", pa.string()),
    ("source_file", pa.string()),
    ("source_id", pa.string()),
    ("source_line", pa.int32()),
    ("theme", pa.string()),
    ("best_moves", pa.list_(pa.string())),
    ("original_epd", pa.string()),
    ("original_operations", pa.string()),
    ("game_id", pa.int64()),
    ("ply", pa.int32()),
    ("stage", pa.string()),
    ("elo_band", pa.string()),
    ("side_to_move", pa.string()),
    ("white_elo", pa.int32()),
    ("black_elo", pa.int32()),
    ("elo_side_original", pa.int32()),
    ("elo_opponent_original", pa.int32()),
    ("played_move_uci", pa.string()),
    ("result", pa.string()),
    ("y_value_to_move", pa.float32()),
    ("termination", pa.string()),
    ("game_url", pa.string()),
])


def fen_key(board: chess.Board) -> str:
    return board.epd(en_passant="fen")


def position_id(key: str) -> str:
    return "pos_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# --- 500 posizioni EPD: algoritmo originario, mantenuto invariato ---
def parse_line(raw: str, file: Path, line_no: int) -> dict | None:
    line = raw.strip().lstrip("\ufeff")
    if not line or line.startswith(("#", "%", "//")):
        return None
    parts = line.split(maxsplit=6)
    if len(parts) < 4:
        raise ValueError("Meno di quattro campi FEN")
    fen4 = " ".join(parts[:4])
    suffix = line[len(fen4):].strip()
    # Alcuni Chess-EPDs utilizzano 6 campi FEN + bm, senza punto e virgola.
    counters = re.match(r"^(\d+)\s+(\d+)(?:\s+|$)", suffix)
    if counters:
        suffix = suffix[counters.end():].strip()
    board = chess.Board(f"{fen4} {FEN_TRAILER}")
    if not board.is_valid() or board.is_game_over(claim_draw=False):
        raise ValueError("Posizione illegale o terminale")

    # EPD standard: estrae id, bm, c0, c7, c8, c9 ecc. quando possibile.
    # L'EPD di alcuni archivi non e' standard: in tal caso conserviamo il testo
    # grezzo e recuperiamo le mosse 'bm' con un fallback non distruttivo.
    operations = {}
    try:
        _, operations = chess.Board.from_epd(f"{fen4} {suffix}")
    except (ValueError, TypeError, AssertionError):
        pass
    source_id = str(operations.get("id", "")).strip() or f"{file.stem}:{line_no}"

    # Nei file senza terminatore ';', python-chess può non interpretare bm.
    bm_tokens = []
    if "bm" in operations:
        value = operations["bm"]
        if isinstance(value, (list, tuple)):
            bm_tokens = list(value)
        else:
            bm_tokens = [value]
    if not bm_tokens:
        match = re.search(r"(?:^|\s)bm\s+([^;]+)", suffix)
        if match:
            bm_tokens = match.group(1).split()

    best_moves = []
    for token in bm_tokens:
        try:
            if isinstance(token, chess.Move):
                move = token
            else:
                text = str(token).strip().rstrip(";,")
                try:
                    move = chess.Move.from_uci(text)
                    if move not in board.legal_moves:
                        move = board.parse_san(text)
                except ValueError:
                    move = board.parse_san(text)
            if move in board.legal_moves and move.uci() not in best_moves:
                best_moves.append(move.uci())
        except (ValueError, AssertionError):
            pass

    theme = ""
    if file.name == STS_FILE:
        match = re.search(r"STS\([^)]*\)\s+(.+?)\.\d+", source_id)
        if match:
            theme = match.group(1).strip()
        elif source_id:
            theme = re.sub(r"\.\d+$", "", source_id).strip()
    elif file.name == "CCET.epd":
        theme = "CCET difficult study"
    else:
        theme = file.stem.replace("-", " ")

    # Tutti i valori originali EPD (inclusi eventuali punteggi STS) sono
    # conservati come stringa raw, senza interpretazioni discutibili.
    ops_as_strings = {}
    for key, value in operations.items():
        if isinstance(value, list):
            ops_as_strings[key] = [x.uci() if isinstance(x, chess.Move) else str(x) for x in value]
        else:
            ops_as_strings[key] = str(value)

    return {
        "fen": board.fen(en_passant="fen"),
        "fen_key": board.epd(en_passant="fen"),
        "source_file": file.name,
        "source_id": source_id,
        "source_line": line_no,
        "theme": theme,
        "best_moves": best_moves,
        "original_epd": line,
        "original_operations": json.dumps(ops_as_strings, ensure_ascii=False, sort_keys=True),
    }


def read_candidates(path: Path, source: str) -> tuple[list[dict], Counter]:
    stats = Counter()
    results = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for idx, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith(("#", "%", "//")):
                continue
            stats["lines"] += 1
            try:
                rec = parse_line(line, path, idx)
            except (ValueError, AssertionError) as exc:
                stats["invalid"] += 1
                if stats["invalid"] <= 5:
                    print(f"  Avviso: {path.name}:{idx}: {exc}")
                continue
            if rec is not None:
                rec["source"] = source
                if source == "CCET":
                    # Gruppi ufficiali CCET in ordine di posizione della suite.
                    position_number = stats["lines"]
                    if position_number <= 46:
                        rec["theme"] = "CCET 01-46 previous suites"
                    elif position_number <= 80:
                        rec["theme"] = "CCET 47-80 published studies"
                    else:
                        rec["theme"] = "CCET 81-160 generated endgames"
                results.append(rec)
                stats["valid"] += 1
    return results, stats


def ordered_stratified(rows: list[dict], seed: int, stratify: str) -> list[dict]:
    """Mescola per gruppo e interlaccia i gruppi, senza preferire quelli grandi."""
    groups = defaultdict(list)
    for row in rows:
        label = row["theme"] if stratify == "theme" else row["source_file"]
        groups[label].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    while True:
        active = False
        for key in keys:
            if groups[key]:
                result.append(groups[key].pop())
                active = True
        if not active:
            break
    return result


def select(rows_by_source: dict[str, list[dict]], seed: int) -> list[dict]:
    ordered = {
        source: ordered_stratified(rows_by_source[source], seed + i * 19937,
                                  "source_file" if source == "Chess-EPDs" else "theme")
        for i, source in enumerate(TARGETS)
    }
    chosen = []
    seen = set()
    # Primo passaggio: tiene separate le quote. Gli eventuali duplicati tra
    # fonti vengono saltati, prendendo la successiva posizione disponibile.
    for source, target in TARGETS.items():
        accepted = 0
        for row in ordered[source]:
            if row["fen_key"] in seen:
                continue
            seen.add(row["fen_key"])
            chosen.append(row)
            accepted += 1
            if accepted == target:
                break
        if accepted != target:
            raise RuntimeError(
                f"{source}: richieste {target} posizioni distinte, trovate {accepted}. "
                "Controlla file locali e righe scartate; non riduco le quote in silenzio."
            )
    return chosen




def create_suite(sources_dir: Path, seed: int) -> tuple[list[dict], set[str], dict, list[Path]]:
    paths = {
        "STS": [sources_dir / "sts" / STS_FILE],
        "Chess-EPDs": [sources_dir / "chess_epds" / name for name in CHESS_FILES],
        "CCET": [sources_dir / "ccet" / "CCET.epd"],
    }
    missing = [p for group in paths.values() for p in group if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Sorgenti EPD non trovate (non si usa l'archivio):\n" +
            "\n".join(str(p) for p in missing)
        )
    candidates = {}
    stats_by_file = {}
    for source, source_paths in paths.items():
        candidates[source] = []
        for path in source_paths:
            rows, counts = read_candidates(path, source)
            candidates[source].extend(rows)
            stats_by_file[f"{source}/{path.name}"] = dict(counts)
            print(f"EPD {path.name}: {counts['valid']} valide, {counts['invalid']} scartate", flush=True)

    # La funzione select e' identica a quella del generatore iniziale:
    # seed=42 con gli stessi identici EPD produce le stesse 500 posizioni.
    selected = select(candidates, seed)
    suite = []
    seen = set()
    for row in selected:
        key = row["fen_key"]
        if key in seen:
            raise RuntimeError("FEN duplicato nelle suite")
        seen.add(key)
        suite.append({
            "position_id": position_id(key),
            "fen": row["fen"],
            "source": row["source"],
            "source_file": row["source_file"],
            "source_id": row["source_id"],
            "source_line": row["source_line"],
            "theme": row["theme"],
            "best_moves": row["best_moves"],
            "original_epd": row["original_epd"],
            "original_operations": row["original_operations"],
        })
    if len(suite) != SUITE_TARGET:
        raise RuntimeError(f"Attese {SUITE_TARGET} posizioni EPD, ottenute {len(suite)}")
    all_files = [p for group in paths.values() for p in group]
    return suite, seen, stats_by_file, all_files


# --- Campionamento PGN: 4500 posizioni, una per partita ---

def elo_band(elo: int, low_bound: int, high_bound: int) -> str:
    if elo < low_bound:
        return "low"
    if elo < high_bound:
        return "middle"
    return "high"


def game_stage(board: chess.Board, ply: int) -> str:
    """Fase approssimativa, non etichetta scacchistica certificata.

    Opening: prime 10 mosse intere; endgame: materiale alleggerito dopo
    almeno 15 mosse intere; tutte le altre posizioni sono middlegame.
    """
    if ply < 20:
        return "opening"
    if ply >= 30:
        non_pawns_kings = chess.popcount(board.occupied & ~(board.pawns | board.kings))
        if non_pawns_kings <= 6 or (not board.queens and non_pawns_kings <= 10):
            return "endgame"
    return "middlegame"


def outcome_to_move(result: str, color: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    return float((result == "1-0") == (color == chess.WHITE))


def game_elo(game: chess.pgn.Game) -> tuple[int, int] | None:
    try:
        white = int(game.headers.get("WhiteElo", ""))
        black = int(game.headers.get("BlackElo", ""))
    except (ValueError, TypeError):
        return None
    if white <= 0 or black <= 0:
        return None
    return white, black



class Reservoir:
    """Campione uniforme limitato in memoria di candidature da partite diverse."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.rng = rng
        self.total = 0
        self.items: list[dict] = []

    def add(self, candidate: dict) -> None:
        self.total += 1
        if len(self.items) < self.capacity:
            self.items.append(candidate)
        else:
            j = self.rng.randrange(self.total)
            if j < self.capacity:
                self.items[j] = candidate


def local_game_candidates(
    game: chess.pgn.Game,
    game_id: int,
    pgn_filename: str,
    rng: random.Random,
    elo_low: int,
    elo_high: int,
    min_ply: int,
    month: str,
) -> dict[tuple[str, str, str], dict]:
    """Per ogni partita propone al massimo una posizione per gruppo.

    La posizione e' scelta uniformemente tra quelle eleggibili della stessa
    partita e dello stesso gruppo; evita di privilegiare partite lunghe.
    """
    ratings = game_elo(game)
    result = game.headers.get("Result", "").strip()
    if ratings is None or result not in VALID_RESULTS or game.errors:
        return {}
    if game.headers.get("Variant", "Standard").lower() not in {"standard", "chess"}:
        return {}
    board = game.board()
    if board.fen() != chess.Board().fen():
        return {}
    white_elo, black_elo = ratings
    site = game.headers.get("Site", "")
    counts: Counter = Counter()
    chosen: dict[tuple[str, str, str], dict] = {}
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= min_ply:
            stage = game_stage(board, ply)
            white_turn = board.turn == chess.WHITE
            color = "white" if white_turn else "black"
            elo_side = white_elo if white_turn else black_elo
            elo_opponent = black_elo if white_turn else white_elo
            band = elo_band(elo_side, elo_low, elo_high)
            bucket = (stage, color, band)
            counts[bucket] += 1
            if rng.randrange(counts[bucket]) == 0:
                key = fen_key(board)
                chosen[bucket] = {
                    "position_id": position_id(key),
                    "fen": board.fen(en_passant="fen"),
                    "source": f"Lichess-{month}",
                    "source_file": pgn_filename,
                    "source_id": f"game_{game_id}:ply_{ply}",
                    "source_line": None,
                    "theme": f"Lichess {stage}",
                    "best_moves": [],
                    "original_epd": None,
                    "original_operations": None,
                    "game_id": game_id,
                    "ply": ply,
                    "stage": stage,
                    "elo_band": band,
                    "side_to_move": color,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "elo_side_original": elo_side,
                    "elo_opponent_original": elo_opponent,
                    "played_move_uci": move.uci(),
                    "result": result,
                    "y_value_to_move": outcome_to_move(result, board.turn),
                    "termination": game.headers.get("Termination", ""),
                    "game_url": site if site.startswith(("https://lichess.org/", "http://lichess.org/")) else None,
                }
        board.push(move)
    return chosen


def scan_pgn(
    pgn: Path,
    seed: int,
    quota: int,
    *,
    elo_low: int,
    elo_high: int,
    min_ply: int,
    max_games: int | None,
    reservoir_multiplier: int,
    month: str,
) -> tuple[dict[tuple[str, str, str], Reservoir], Counter]:
    buckets = [(stage, color, band) for stage in STAGES for color in COLORS for band in ELO_BANDS]
    rng = random.Random(seed)
    reservoirs = {bucket: Reservoir(quota * reservoir_multiplier, rng) for bucket in buckets}
    stats: Counter = Counter()
    with pgn.open("rb") as file_handle:
        with zstd.ZstdDecompressor().stream_reader(file_handle) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8", errors="replace") as text_stream:
                with tqdm(desc="Partite Lichess esaminate", unit="game", total=max_games) as progress:
                    while max_games is None or stats["games_read"] < max_games:
                        try:
                            game = chess.pgn.read_game(text_stream)
                        except Exception as exc:
                            raise RuntimeError(
                                f"Lettura PGN fallita alla partita {stats['games_read']}: {exc}"
                            ) from exc
                        if game is None:
                            break
                        game_id = stats["games_read"]  # stesso indice 0-based dell'estrattore PGN
                        stats["games_read"] += 1
                        local = local_game_candidates(
                            game, game_id, pgn.name, rng, elo_low, elo_high, min_ply, month
                        )
                        if not local:
                            stats["games_discarded_or_no_candidates"] += 1
                        else:
                            stats["games_with_candidates"] += 1
                            for bucket, candidate in local.items():
                                reservoirs[bucket].add(candidate)
                        progress.update(1)
    return reservoirs, stats


def choose_lichess(
    reservoirs: dict[tuple[str, str, str], Reservoir],
    already_seen: set[str],
    quota: int,
    seed: int,
) -> list[dict]:
    """Seleziona con quote esatte e senza riciclare partite o posizioni.

    Le fasce meno popolose sono servite per prime, cosi' collisioni tra
    gruppi non penalizzano i casi piu' rari.
    """
    rng = random.Random(seed + 911)
    prepared = {}
    for bucket, reservoir in reservoirs.items():
        items = reservoir.items.copy()
        rng.shuffle(items)
        prepared[bucket] = items
    order = sorted(prepared, key=lambda b: (len(prepared[b]), b))
    seen = set(already_seen)
    used_games = set()
    selected = []
    for bucket in order:
        accepted = 0
        for row in prepared[bucket]:
            key = fen_key(chess.Board(row["fen"]))
            if key in seen or row["game_id"] in used_games:
                continue
            selected.append(row)
            seen.add(key)
            used_games.add(row["game_id"])
            accepted += 1
            if accepted == quota:
                break
        if accepted != quota:
            label = "/".join(bucket)
            raise RuntimeError(
                f"Quota non raggiunta per {label}: {accepted}/{quota}. "
                f"Candidature del gruppo: {reservoir.total}, "
                f"nel reservoir: {len(prepared[bucket])}. "
                "Aumenta --reservoir_multiplier, controlla gli Elo del dump o "
                "modifica --elo_low/--elo_high. Nessuna quota viene ridotta in silenzio."
            )
    return selected



def write_results(rows: list[dict], output: Path, metadata: dict) -> None:
    meta_path = output.with_suffix(".metadata.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(
            f"Output esistente ({output} o {meta_path}). "
            "Non sovrascrivo un benchmark gia' generato."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = output.with_name(output.name + ".tmp")
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    try:
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, tmp_parquet, compression="zstd")
        tmp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_parquet.replace(output)
        tmp_meta.replace(meta_path)
    finally:
        tmp_parquet.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="2013-04", help="Mese Lichess, formato YYYY-MM")
    parser.add_argument("--sources_dir", default=str(DEFAULT_SOURCES), help="Cartella delle suite EPD locali")
    parser.add_argument("--pgn_zst", default=None, help="Override del PGN.ZST del mese")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--lichess_positions", type=int, default=LICHESS_TARGET)
    parser.add_argument("--elo_low", type=int, default=1200)
    parser.add_argument("--elo_high", type=int, default=1800)
    parser.add_argument("--min_ply", type=int, default=8)
    parser.add_argument("--reservoir_multiplier", type=int, default=8)
    parser.add_argument("--max_games", type=int, default=None, help="Solo smoke test")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", args.month):
        parser.error("--month deve essere nel formato YYYY-MM (es. 2013-04)")
    if not (0 < args.elo_low < args.elo_high):
        parser.error("--elo_low e --elo_high devono essere positivi e crescenti")
    if args.min_ply < 0 or args.reservoir_multiplier < 2:
        parser.error("--min_ply >= 0 e --reservoir_multiplier >= 2")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max_games deve essere positivo")
    n_buckets = len(STAGES) * len(COLORS) * len(ELO_BANDS)
    if args.lichess_positions <= 0 or args.lichess_positions % n_buckets:
        parser.error(f"--lichess_positions deve essere positivo e multiplo di {n_buckets}")
    quota = args.lichess_positions // n_buckets

    sources_dir = resolve_path(args.sources_dir)
    pgn_path = resolve_path(args.pgn_zst or f"data/raw/lichess_db_standard_rated_{args.month}.pgn.zst")
    output = resolve_path(args.output)
    if not pgn_path.is_file():
        parser.error(f"PGN.ZST del mese {args.month} mancante: {pgn_path}")
    if output.exists() or output.with_suffix(".metadata.json").exists():
        parser.error(f"Output gia' presente: {output}; scegli un altro --output o rimuovilo")

    # Si selezionano per prime le EPD; poi Lichess esclude tutte quelle posizioni.
    suite, suite_keys, suite_stats, suite_files = create_suite(sources_dir, args.seed)
    print(f"Suite ricostruite da zero: {len(suite)} posizioni", flush=True)
    reservoirs, stats = scan_pgn(
        pgn_path, args.seed, quota,
        elo_low=args.elo_low, elo_high=args.elo_high,
        min_ply=args.min_ply, max_games=args.max_games,
        reservoir_multiplier=args.reservoir_multiplier,
        month=args.month,
    )
    lichess = choose_lichess(reservoirs, suite_keys, quota, args.seed)
    rows = suite + lichess
    random.Random(args.seed + 1087).shuffle(rows)
    ids = [r["position_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("position_id duplicati nel benchmark finale")
    if len(rows) != SUITE_TARGET + args.lichess_positions:
        raise RuntimeError("Numero di posizioni non coerente")

    bucket_counts = Counter((r["stage"], r["side_to_move"], r["elo_band"]) for r in lichess)
    metadata = {
        "seed": args.seed,
        "month": args.month,
        "total": len(rows),
        "targets": {**TARGETS, f"Lichess-{args.month}": args.lichess_positions},
        "counts_by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        "epd_sources": [str(p.relative_to(sources_dir)) for p in suite_files],
        "epd_source_sha256": {
            str(p.relative_to(sources_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in suite_files
        },
        "epd_parse_counts": suite_stats,
        "pgn_zst": str(pgn_path),
        "pgn_size_bytes": pgn_path.stat().st_size,
        "pgn_games_read": stats["games_read"],
        "pgn_games_with_candidates": stats["games_with_candidates"],
        "sampling": {
            "suite_method": "original_STS_ChessEPDs_CCET_stratified_selection",
            "suite_seed": args.seed,
            "lichess_method": "stratified_reservoir_per_game_then_unique_game_and_epd",
            "lichess_positions": args.lichess_positions,
            "buckets": n_buckets,
            "quota_per_bucket": quota,
            "min_ply": args.min_ply,
            "elo_bands": {"low": f"elo < {args.elo_low}",
                          "middle": f"{args.elo_low} <= elo < {args.elo_high}",
                          "high": f"elo >= {args.elo_high}"},
            "stage_heuristic": "opening ply<20; endgame ply>=30 with reduced material; else middlegame",
            "quota_counts": {"/".join(k): bucket_counts[k] for k in sorted(bucket_counts)},
            "reservoir_multiplier": args.reservoir_multiplier,
            "one_position_per_lichess_game": True,
            "deduplication": "first_four_FEN_fields_with_explicit_en_passant",
        },
        "notes": [
            "Benchmark ricostruito esclusivamente da sorgenti EPD e PGN.ZST locali, senza archivio.",
            "Le 500 suite EPD corrispondono al metodo dell'esperimento preliminare con identici EPD e seed.",
            "Le posizioni suite EPD gia' esplorate non sono un holdout completamente nuovo.",
            "Le posizioni Lichess vanno escluse da training e dalla scelta ripetuta di iperparametri.",
            "Le fasi opening/middlegame/endgame sono classificazioni euristiche.",
            "La colonna played_move_uci non indica la mossa oggettivamente migliore.",
            "Permessi redistribuzione suite EPD da verificare prima della pubblicazione.",
        ],
    }
    write_results(rows, output, metadata)
    print(f"Benchmark scritto: {output} ({len(rows)} posizioni)")
    print("Suite:", dict(Counter(r["source"] for r in suite)))
    print(f"Lichess-{args.month}: {len(lichess)} posizioni da {len(set(r['game_id'] for r in lichess))} partite")
    print(f"Metadata: {output.with_suffix('.metadata.json')}")


if __name__ == "__main__":
    main()
