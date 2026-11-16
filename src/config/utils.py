import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Project root directory (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]

def utc_now() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def resolve_path(path: str | Path) -> Path:
    """Espande home e risolve i percorsi relativi rispetto alla radice del progetto."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path

def write_json_atomic(path: Path, content: dict) -> None:
    """Scrive su un file temporaneo e poi lo sostituisce al precedente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)

def load_num_moves(vocab_path: Path) -> int:
    """Carica il vocabolario e verifica che gli indici siano contigui."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        move2idx = json.load(f)
    values = sorted(int(value) for value in move2idx.values())
    if not values or values != list(range(len(values))):
        raise ValueError(f"Vocabolario non valido: {vocab_path}")
    return len(values)

def find_npz_files(data_dir: Path) -> list[Path]:
    """Trova gli shard NPZ in ordine deterministico."""
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"Nessun file NPZ trovato in {data_dir}")
    return files

# Codifica condivisa tra conversione NPZ e inferenza EloChess.
def fill_board(board, out) -> None:
    """Codifica la scacchiera nei 12 piani P,R,N,B,Q,K per colore."""
    import chess
    planes = {chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
              chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5}
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        plane = planes[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        out[row, col, plane] = 1

def fill_state(board, out) -> None:
    """Codifica le 71 feature: turno, arrocco, en passant legale, halfmove."""
    import chess
    out[0] = 1.0 if board.turn == chess.WHITE else 0.0
    out[1] = board.has_kingside_castling_rights(chess.WHITE)
    out[2] = board.has_queenside_castling_rights(chess.WHITE)
    out[3] = board.has_kingside_castling_rights(chess.BLACK)
    out[4] = board.has_queenside_castling_rights(chess.BLACK)
    ep = board.ep_square if board.has_legal_en_passant() else 64
    out[5 + ep] = 1.0
    out[70] = min(board.halfmove_clock, 100.0) / 100.0
