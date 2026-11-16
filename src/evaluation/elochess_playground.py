#!/usr/bin/env python3
"""EloChess Playground — scacchiera trascinabile + analisi interattiva.

Avvia dalla radice del repository:
    python -m streamlit run src/evaluation/elochess_playground.py

Richiede anche la cartella sibling elochess_board_component/ (frontend locale).
Stockfish e' opzionale: per l'analisi UCI serve un eseguibile locale.
Nessuna connessione Internet e' necessaria.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path

import chess
import chess.engine
import chess.svg
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from src.config.settings import DEFAULTS
from src.config.utils import resolve_path
from src.evaluation.manual_benchmark_catalog import load_manual_catalog

MANUAL_BENCHMARK_DIR = ROOT / "data/evaluation/manual_benchmark"
VOCAB_PATH = resolve_path(DEFAULTS["vocab"])
STOCKFISH_DEFAULT_PATH = "models/stockfish/stockfish-windows-x86-64-universal.exe"
MODELS_DIR = ROOT / "models"
ARCHIVE_DIR = ROOT / "archive"
FRONTEND_DIR = Path(__file__).resolve().with_name("elochess_board_component")

st.set_page_config(page_title="EloChess — Playground", page_icon="♟", layout="wide")

# Solo presentazione: non cambiamo calcoli, checkpoint o interazioni.
st.markdown("""
<style>
  .stMainBlockContainer, section[data-testid="stMain"] .block-container {
      max-width: 1800px; padding: 1.1rem 1.6rem 1.5rem;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
      padding-top: 0.9rem;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.45rem; }
  section[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.48rem; }
  section[data-testid="stMain"] [data-testid="stButton"] button {
      min-height: 2rem; padding: 0.2rem 0.45rem;
  }
  /* Tabelle: intestazioni e righe hanno contenitori propri.
     Non comprimere i blocchi di st.columns: in alcune versioni di
     Streamlit i bottoni finirebbero per sovrapporsi a intestazioni e testi. */
  .st-key-stockfish_moves_header,
  .st-key-elochess_moves_header {
      padding: .25rem 0 .05rem;
  }
  /* I separatori sono elementi autonomi nel flusso Streamlit, NON bordi dei
     contenitori: un bordo sul container può attraversare intestazioni o
     continuazioni a causa delle altezze interne gestite da Streamlit. */
  .elochess-table-separator {
      height: 1px;
      background-color: rgba(130, 140, 150, .24);
      margin: .23rem 0 .12rem;
  }
  .elochess-table-separator--header {
      background-color: rgba(130, 140, 150, .38);
      margin: .24rem 0 .18rem;
  }
  .st-key-stockfish_moves_header [data-testid="stMarkdown"] p,
  .st-key-elochess_moves_header [data-testid="stMarkdown"] p {
      text-align: center;
      font-weight: 700;
      font-size: .88rem;
      margin: 0;
  }
  /* Ogni riga comprende il bottone, i valori e la sua continuazione. */
  [class*="st-key-stockfish_move_row_"],
  [class*="st-key-elochess_move_row_"] {
      padding: .10rem 0 .10rem;
  }
  [class*="st-key-stockfish_move_row_"] [data-testid="stColumn"] [data-testid="stMarkdown"] p,
  [class*="st-key-elochess_move_row_"] [data-testid="stColumn"] [data-testid="stMarkdown"] p {
      text-align: center;
      font-variant-numeric: tabular-nums;
      margin: 0;
  }
  /* Solo lo spazio tra la riga principale e la continuazione; la
     spaziatura interna dei pulsanti rimane quella nativa di Streamlit. */
  [class*="st-key-stockfish_move_row_"] > [data-testid="stVerticalBlock"],
  [class*="st-key-elochess_move_row_"] > [data-testid="stVerticalBlock"] {
      gap: .16rem;
  }
  [class*="st-key-stockfish_move_row_"] [data-testid="stCaptionContainer"] p,
  [class*="st-key-elochess_move_row_"] [data-testid="stCaptionContainer"] p {
      line-height: 1.25;
      margin: 0;
  }
  /* Colonna della scacchiera a larghezza fissa, area delle analisi flessibile.
     L'ancora evita di cambiare le colonne interne (controlli e tabella mosse). */
  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(#elochess-board-column-anchor) {
      gap: 1rem;
  }
  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(#elochess-board-column-anchor) > [data-testid="stColumn"]:first-child {
      flex: 0 0 540px !important;
      width: 540px !important;
      max-width: 540px !important;
      min-width: 0 !important;
  }
  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(#elochess-board-column-anchor) > [data-testid="stColumn"]:nth-child(2) {
      flex: 1 1 0 !important;
      min-width: 0 !important;
      max-width: none !important;
  }
  /* Larghezze ridotte: le colonne diventano verticali senza comprimere l'analisi. */
  @media (max-width: 1100px) {
      section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(#elochess-board-column-anchor) {
          flex-direction: column !important;
      }
      section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(#elochess-board-column-anchor) > [data-testid="stColumn"] {
          flex: 0 1 auto !important;
          width: 100% !important;
          max-width: 100% !important;
      }
  }
  .elochess-value-heading {
      display: flex; align-items: baseline; flex-wrap: wrap;
      column-gap: 1rem; row-gap: .2rem; margin-bottom: .1rem;
  }
  .elochess-value-title { font-size: 1.15rem; font-weight: 650; }
  .elochess-value-legend { font-size: .84rem; opacity: .82; }
  .elochess-value-separator { margin: 0 .32rem; font-size: .65rem; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

if not (FRONTEND_DIR / "index.html").is_file():
    st.error(f"Componente scacchiera mancante: {FRONTEND_DIR / 'index.html'}")
    st.stop()

chessboard_component = components.declare_component("elochess_draggable_board", path=str(FRONTEND_DIR))


@st.cache_resource(show_spinner="Caricamento checkpoint...")
def load_checkpoint(model_path: str, mtime_ns: int):
    import tensorflow as tf
    from src.model.build_model import LegalMaskSoftmax  # noqa: F401 — registra il layer

    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data(show_spinner=False)
def load_vocabulary(mtime_ns: int) -> dict[str, int]:
    with VOCAB_PATH.open(encoding="utf-8") as handle:
        vocab = json.load(handle)
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Indici del vocabolario non contigui")
    return vocab


@st.cache_data(show_spinner=False)
def piece_assets() -> dict[str, str]:
    """Stessa grafica vettoriale usata da python-chess in review_benchmark.py."""
    return {
        symbol: "data:image/svg+xml;base64," + base64.b64encode(
            chess.svg.piece(chess.Piece.from_symbol(symbol), size=90).encode("utf-8")
        ).decode("ascii")
        for symbol in "KQRBNPkqrbnp"
    }


@st.cache_data(show_spinner=False, max_entries=1500)
def analyze(fen: str, elo_white: int, elo_black: int,
            model_path: str, model_mtime: int, vocab_mtime: int) -> dict:
    """Policy completa e Value; gli Elo sono di Bianco/Nero, non fissi al tratto."""
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=False):
        raise ValueError("Posizione terminale")

    vocab = load_vocabulary(vocab_mtime)
    elo_side = elo_white if board.turn == chess.WHITE else elo_black
    elo_opponent = elo_black if board.turn == chess.WHITE else elo_white
    inputs = encode_position(board, elo_side, elo_opponent, vocab)
    model = load_checkpoint(model_path, model_mtime)
    output = model(inputs, training=False)
    if not isinstance(output, dict) or set(output) != {"policy", "value"}:
        raise ValueError("Checkpoint incompatibile: output policy/value mancanti")

    probabilities = np.asarray(output["policy"], dtype=np.float64)
    if probabilities.shape != (1, len(vocab)) or not np.isfinite(probabilities).all():
        raise ValueError("Policy con shape o valori non validi")
    value_to_move = float(np.asarray(output["value"]).reshape(-1)[0])
    if not np.isfinite(value_to_move) or not (0.0 <= value_to_move <= 1.0):
        raise ValueError("Value non valido")
    expected_white = value_to_move if board.turn == chess.WHITE else 1.0 - value_to_move

    moves = []
    for move in board.legal_moves:
        uci = move.uci()
        idx = vocab.get(uci)
        if idx is None:
            raise ValueError(f"Mossa legale non presente nel vocabolario: {uci}")
        moves.append({"uci": uci, "san": board.san(move), "probability": float(probabilities[0, idx])})
    moves.sort(key=lambda move: (-move["probability"], vocab[move["uci"]]))

    return {"signed_white": 2.0 * expected_white - 1.0,
            "expected_white": expected_white,
            "expected_to_move": value_to_move,
            "moves": moves}


def suggested_stockfish_path() -> str:
    """Individua il motore UCI locale, senza download o accesso alla rete."""
    preferred = resolve_path(STOCKFISH_DEFAULT_PATH)
    if preferred.is_file():
        return STOCKFISH_DEFAULT_PATH
    directory = ROOT / "models/stockfish"
    if directory.is_dir():
        for candidate in sorted(directory.glob("stockfish*")):
            if candidate.is_file() and (
                candidate.suffix.lower() == ".exe" or os.access(candidate, os.X_OK)
            ):
                return str(candidate.relative_to(ROOT))
    for name in ("stockfish", "stockfish.exe"):
        executable = shutil.which(name)
        if executable:
            return executable
    return STOCKFISH_DEFAULT_PATH


def stockfish_executable(raw_path: str) -> Path:
    """Accetta percorsi assoluti, relativi al repository o comandi nel PATH."""
    cleaned = raw_path.strip().strip('"').strip("'").strip()
    if not cleaned:
        raise FileNotFoundError("Inserisci il percorso dell'eseguibile Stockfish.")
    candidate = resolve_path(cleaned)
    if candidate.is_file():
        return candidate.resolve()
    discovered = shutil.which(cleaned)
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError(
        f"Eseguibile Stockfish non trovato: {cleaned}. "
        "Controlla il percorso nella barra laterale."
    )


def stockfish_score_label(cp_white: int | None, mate_white: int | None) -> str:
    """Centipawn o matto dal punto di vista del Bianco, NON EloChess Value."""
    if mate_white is not None:
        return f"M{mate_white:+d}"  # M+3: matto del Bianco; M-2: matto del Nero
    if cp_white is not None:
        return f"{cp_white / 100:+.2f}"
    return "—"


@st.cache_data(show_spinner=False, max_entries=256)
def analyze_stockfish(fen: str, executable: str, executable_mtime: int,
                      limit_mode: str, limit_value: float, top_n: int,
                      threads: int, hash_mb: int) -> list[dict]:
    """Esegue MultiPV sulla FEN corrente. Cache separata da EloChess.

    Avvia il motore solo in caso di cache miss e lo chiude sempre. I punteggi
    sono da prospettiva del Bianco e non vengono convertiti nel Value di EloChess.
    """
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=False):
        return []
    executable_path = Path(executable)
    if not executable_path.is_file():
        raise FileNotFoundError(f"Stockfish non trovato: {executable_path}")
    if limit_mode == "Tempo":
        if limit_value <= 0:
            raise ValueError("Il tempo Stockfish deve essere positivo")
        limit = chess.engine.Limit(time=float(limit_value))
    elif limit_mode == "Profondità":
        if limit_value < 1:
            raise ValueError("La profondità Stockfish deve essere positiva")
        limit = chess.engine.Limit(depth=int(limit_value))
    else:
        raise ValueError("Modalità di analisi Stockfish non valida")

    with chess.engine.SimpleEngine.popen_uci(str(executable_path)) as engine:
        configuration = {}
        if "Threads" in engine.options:
            configuration["Threads"] = threads
        if "Hash" in engine.options:
            configuration["Hash"] = hash_mb
        if configuration:
            engine.configure(configuration)
        info_list = engine.analyse(
            board, limit, multipv=min(top_n, board.legal_moves.count()), game=object()
        )

    if isinstance(info_list, dict):
        info_list = [info_list]
    moves = []
    for info in info_list:
        pv = info.get("pv")
        score = info.get("score")
        if not pv or score is None or pv[0] not in board.legal_moves:
            continue
        white_score = score.white()
        cp_white = white_score.score()
        mate_white = white_score.mate()
        if cp_white is None and mate_white is None:
            continue
        first = pv[0]
        after_first = board.copy(stack=False)
        after_first.push(first)
        remainder = pv[1:12]
        try:
            continuation = after_first.variation_san(remainder) if remainder else ""
        except (ValueError, AssertionError):
            continuation = " ".join(move.uci() for move in remainder)
        moves.append({
            "uci": first.uci(),
            "san": board.san(first),
            "cp_white": cp_white,
            "mate_white": mate_white,
            "depth": info.get("depth"),
            "seldepth": info.get("seldepth"),
            "continuation": continuation,
        })
    if not moves:
        raise RuntimeError("Stockfish non ha restituito varianti valide.")
    return moves


@st.cache_data(show_spinner=False, max_entries=256)
def follow_lines(fen: str, elo_white: int, elo_black: int, model_path: str,
                 model_mtime: int, vocab_mtime: int, first_moves: tuple[str, ...],
                 replies: int = 4) -> dict[str, str]:
    """Varianti greedy: dopo ciascuna candidata, segue la Top-1 del nuovo giocatore al tratto."""
    result = {}
    for first_uci in first_moves:
        board = chess.Board(fen)
        first = chess.Move.from_uci(first_uci)
        if first not in board.legal_moves:
            continue
        line = [first]
        for _ in range(replies):
            board.push(line[-1])
            if board.is_game_over(claim_draw=False):
                break
            next_result = analyze(board.fen(), elo_white, elo_black,
                                  model_path, model_mtime, vocab_mtime)
            if not next_result["moves"]:
                break
            line.append(chess.Move.from_uci(next_result["moves"][0]["uci"]))
        # Non ripetere la mossa iniziale: e' gia' nel pulsante della candidata.
        # La numerazione resta corretta anche per linee che iniziano dal Nero.
        after_first = chess.Board(fen)
        after_first.push(first)
        continuation_moves = line[1:]
        try:
            result[first_uci] = after_first.variation_san(continuation_moves) if continuation_moves else ""
        except (ValueError, AssertionError):
            result[first_uci] = " ".join(move.uci() for move in continuation_moves)
    return result


# Lookahead opzionale. Ogni semimossa e' una mossa di un solo giocatore;
# alle ramificazioni interne si considerano le prime BEAM_WIDTH mosse della
# Policy del giocatore al tratto, con probabilita' rinormalizzate.
LOOKAHEAD_BEAM_WIDTH = 3
LOOKAHEAD_VALUE_BETA = 4.0
LOOKAHEAD_BATCH_SIZE = 64


@st.cache_data(show_spinner=False, max_entries=96)
def search_probabilities(fen: str, elo_white: int, elo_black: int,
                         model_path: str, model_mtime: int, vocab_mtime: int,
                         depth: int) -> dict[str, dict[str, float]]:
    """Ricalcola la preferenza Policy con lookahead *expectimax* limitato.

    Tutte le mosse legali alla radice vengono esaminate. Dalla seconda
    semimossa, le risposte sono le Top-K della Policy, pesate e rinormalizzate.
    Ai nodi foglia si usa il Value; ai nodi terminali l'esito effettivo.
    Il risultato e' una nuova distribuzione, NON una probabilita' di vittoria.
    """
    if not 1 <= depth <= 4:
        raise ValueError("La profondita' di ricerca deve essere tra 1 e 4 semimosse")
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=False):
        return {}
    vocab = load_vocabulary(vocab_mtime)
    model = load_checkpoint(model_path, model_mtime)
    original = analyze(fen, elo_white, elo_black, model_path, model_mtime, vocab_mtime)
    root_side = board.turn

    def terminal_score(position: chess.Board) -> float | None:
        outcome = position.outcome(claim_draw=False)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.5
        return 1.0 if outcome.winner == root_side else 0.0

    def infer(nodes: list[dict]) -> None:
        """Un'unica inferenza a batch per le FEN non terminali del livello."""
        active = [n for n in nodes if n['score'] is None]
        for offset in range(0, len(active), LOOKAHEAD_BATCH_SIZE):
            chunk = active[offset:offset + LOOKAHEAD_BATCH_SIZE]
            encoded = []
            for node in chunk:
                position = node['board']
                side = elo_white if position.turn == chess.WHITE else elo_black
                opponent = elo_black if position.turn == chess.WHITE else elo_white
                encoded.append(encode_position(position, side, opponent, vocab))
            inputs = {key: np.concatenate([item[key] for item in encoded], axis=0)
                      for key in encoded[0]}
            output = model(inputs, training=False)
            if not isinstance(output, dict) or set(output) != {'policy', 'value'}:
                raise ValueError('Checkpoint incompatibile: output policy/value mancanti')
            policies = np.asarray(output['policy'], dtype=np.float64)
            values = np.asarray(output['value'], dtype=np.float64).reshape(-1)
            if (policies.shape != (len(chunk), len(vocab)) or
                    values.shape != (len(chunk),) or
                    not np.isfinite(policies).all() or not np.isfinite(values).all() or
                    np.any(values < 0.0) or np.any(values > 1.0)):
                raise ValueError('Output Policy/Value non valido durante la ricerca')
            for i, node in enumerate(chunk):
                node['score'] = float(values[i] if node['board'].turn == root_side
                                      else 1.0 - values[i])
                node['policy'] = policies[i]

    # Ogni candidata alla radice ha il proprio albero; nessuna viene esclusa.
    roots: dict[str, dict] = {}
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        roots[move.uci()] = {'board': child, 'score': terminal_score(child),
                             'children': [], 'policy': None}
    layer = list(roots.values())
    for ply in range(1, depth + 1):
        infer(layer)
        if ply == depth:
            break
        next_layer = []
        for node in layer:
            if node['policy'] is None:  # posizione terminale
                continue
            legal = list(node['board'].legal_moves)
            legal.sort(key=lambda move: (-float(node['policy'][vocab[move.uci()]]),
                                         vocab[move.uci()]))
            selected = legal[:LOOKAHEAD_BEAM_WIDTH]
            weights = np.asarray([node['policy'][vocab[move.uci()]]
                                  for move in selected], dtype=np.float64)
            total = float(weights.sum())
            if not selected or total <= 0.0:
                raise ValueError('Policy senza mosse valide durante il lookahead')
            for move, weight in zip(selected, weights / total):
                child_board = node['board'].copy(stack=False)
                child_board.push(move)
                child = {'board': child_board,
                         'score': terminal_score(child_board), 'children': [], 'policy': None}
                node['children'].append((float(weight), child))
                next_layer.append(child)
        layer = next_layer
        if not layer:
            break

    def backed_up_score(node: dict) -> float:
        if node['children']:
            return sum(weight * backed_up_score(child)
                       for weight, child in node['children'])
        return float(node['score'])

    moves = original['moves']
    scores = np.asarray([backed_up_score(roots[move['uci']]) for move in moves],
                        dtype=np.float64)
    base = np.asarray([move['probability'] for move in moves], dtype=np.float64)
    # Soft reranking: P_new(a) proportional to P_policy(a) * exp(beta * Q(a)).
    # La traslazione con max(Q) evita overflow; non cambia il risultato.
    weights = base * np.exp(LOOKAHEAD_VALUE_BETA * (scores - scores.max()))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Impossibile normalizzare le probabilità della ricerca")
    revised = weights / total
    return {move['uci']: {'probability': float(revised[i]),
                          'expected_score': float(scores[i])}
            for i, move in enumerate(moves)}


def ensure_state() -> None:
    if "play_board" not in st.session_state:
        st.session_state.play_board = chess.Board()
    st.session_state.setdefault("play_error", "")
    st.session_state.setdefault("last_board_event", None)
    st.session_state.setdefault("move_san_history", [])
    st.session_state.setdefault("manual_initial_fen", "")
    st.session_state.setdefault("manual_position_label", "")


def play_uci(uci: str) -> None:
    board = st.session_state.play_board
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        st.session_state.play_error = "Formato mossa non valido"
        return
    if move not in board.legal_moves:
        st.session_state.play_error = "Mossa non legale nella posizione attuale"
        return
    st.session_state.move_san_history.append(board.san(move))
    board.push(move)
    st.session_state.play_error = ""


def apply_board_event(event: object) -> bool:
    """Gestisce una sola volta gli eventi JS, anche dopo i rerun di Streamlit."""
    if not isinstance(event, dict):
        return False
    nonce = event.get("id")
    uci = event.get("uci")
    fen = event.get("fen")
    if not nonce or not isinstance(uci, str) or not isinstance(fen, str):
        return False
    if st.session_state.last_board_event == nonce:
        return False
    if fen != st.session_state.play_board.fen():
        return False
    st.session_state.last_board_event = nonce
    previous = st.session_state.play_board.fen()
    play_uci(uci)
    return st.session_state.play_board.fen() != previous


def reset_board() -> None:
    st.session_state.play_board = chess.Board()
    st.session_state.move_san_history = []
    st.session_state.play_error = ""
    st.session_state.manual_initial_fen = ""
    st.session_state.manual_position_label = ""
    st.session_state.last_board_event = None


def load_manual_position(fen: str, label: str) -> None:
    """Carica la FEN del capitolo scelto, senza cambiare Elo o profondità."""
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("Posizione scacchisticamente non valida")
    except (ValueError, AssertionError) as exc:
        st.session_state.play_error = f"FEN dell'esercizio non valido: {exc}"
        return
    st.session_state.play_board = board
    st.session_state.move_san_history = []
    st.session_state.play_error = ""
    st.session_state.manual_initial_fen = fen
    st.session_state.manual_position_label = label
    st.session_state.last_board_event = None


def undo() -> None:
    board = st.session_state.play_board
    if board.move_stack:
        board.pop()
        if st.session_state.move_san_history:
            st.session_state.move_san_history.pop()
    st.session_state.play_error = ""


def load_fen() -> None:
    try:
        board = chess.Board(st.session_state.fen_entry.strip())
        if not board.is_valid():
            raise ValueError("La posizione non è valida scacchisticamente")
    except (ValueError, AssertionError) as exc:
        st.session_state.play_error = f"FEN non valido: {exc}"
        return
    st.session_state.play_board = board
    st.session_state.move_san_history = []
    st.session_state.play_error = ""
    st.session_state.manual_initial_fen = ""
    st.session_state.manual_position_label = ""
    st.session_state.last_board_event = None


def find_models() -> list[Path]:
    unique = set()
    directory = MODELS_DIR
    if directory.exists():
        unique.update(p.resolve() for p in directory.rglob("*.keras") if p.is_file())
    return sorted(unique, key=lambda p: str(p.relative_to(ROOT)))


def probability_label(p: float) -> str:
    """Percentuale di preferenza Policy/Ricerca, non probabilita' di vittoria."""
    return "0%" if p < 0.0001 else f"{p:.2%}"


def numbered_san(board: chess.Board, san: str) -> str:
    """Mossa con numero, inclusa la forma corretta per il Nero: 2... Rf6."""
    number = f"{board.fullmove_number}." if board.turn == chess.WHITE else f"{board.fullmove_number}..."
    return f"{number} {san}"


def white_signed_from_root_score(score: float, root_turn: chess.Color) -> float:
    """Q(a) di ricerca (score del giocatore al tratto) -> Value firmato per il Bianco."""
    expected_white = score if root_turn == chess.WHITE else 1.0 - score
    return 2.0 * expected_white - 1.0


SEARCH_HELP = """**Semimosse di ricerca (0–4)**

- **0:** mostra soltanto la Policy, cioè la distribuzione originale di probabilità sulle mosse legali.
- **1:** per ogni mossa legale, esamina la posizione risultante usando il Value del modello.
- **2–4:** esamina anche le risposte successive; a ogni nodo interno segue al massimo le 3 mosse con probabilità Policy maggiore, rinormalizzandone i pesi. È una ricerca selettiva, non minimax e non Stockfish.

**Come Policy e Value formano la Ricerca**

Per ogni mossa candidata `a`, la ricerca calcola `Q(a)`, uno **score atteso tra 0 e 1 dal punto di vista del giocatore che muove nella posizione iniziale**. Alle foglie usa il Value del modello; alle posizioni terminali usa il risultato effettivo. Le risposte esplorate sono combinate in base alle probabilità della Policy.

`P_ricerca(a) = [P_policy(a) × exp(4 × Q(a))] / Σ_b [P_policy(b) × exp(4 × Q(b))]`

Il parametro **β = 4** determina quanto `Q(a)` influisce sulla nuova preferenza. La colonna **Ricerca** mostra queste probabilità rinormalizzate, **non** probabilità di vittoria.

**Che cos'è la colonna Value**

Mostra **lo score `Q(a)` già calcolato per la singola mossa**, convertito linearmente nella scala **−1…+1 dalla prospettiva del Bianco**: −1 favorisce il Nero, +1 il Bianco. Se è il Bianco al tratto, il valore mostrato è `2 × Q(a) − 1`; se è il Nero, è `1 − 2 × Q(a)`. **Non si applicano né l'esponenziale né β = 4 a questa colonna**: quei passaggi servono soltanto a calcolare la preferenza Ricerca. Il Value sopra la tabella è invece la stima diretta della posizione corrente, senza ricerca.

**Calcola anche le prime 3 varianti (4 semimosse)**

È un'opzione **indipendente dalla profondità di ricerca**. Per le prime 3 mosse della classifica visualizzata simula altre 4 semimosse, scegliendo a ogni turno la Top-1 della Policy del giocatore al tratto. La continuazione non ripete la mossa iniziale già presente nella riga. Le linee sono illustrative: non modificano né Policy, né Ricerca, né Value.

Questa ricerca non usa Stockfish. Le profondità più alte richiedono più calcoli.

**Stockfish** viene eseguito separatamente, con i propri parametri. La sua
valutazione in pedoni (+1.00 = circa un pedone) o matto non e' direttamente
confrontabile con il Value di EloChess.
"""


def main() -> None:
    ensure_state()
    st.title("EloChess Playground", anchor=False)

    options = find_models()
    if not options:
        st.warning("Nessun checkpoint .keras nella cartella models/.")
        st.stop()
    # Percorsi mostrati relativi a models/, percorsi assoluti invariati internamente.
    names = [str(path.relative_to(MODELS_DIR)) for path in options]
    preferred = next((n for n in ("v1_6/best.keras", "v1_5/best.keras", "v1_4/best.keras")
                      if n in names), names[0])

    with st.sidebar:
        st.header("Manual Benchmark", anchor=False)
        try:
            categories = load_manual_catalog(MANUAL_BENCHMARK_DIR)
        except (OSError, ValueError) as exc:
            st.error(f"Errore nel catalogo del Manual Benchmark: {exc}")
            categories = []
        if not categories:
            st.info("Nessuna categoria trovata in data/evaluation/manual_benchmark/.")
        else:
            category_index = st.selectbox(
                "Categoria", range(len(categories)),
                format_func=lambda i: categories[i]["title"], key="manual_category_index")
            category = categories[category_index]
            studies = category["subcategories"]
            if not studies:
                st.caption("Categoria vuota: aggiungi un file .pgn per creare una sottocategoria.")
            else:
                study_index = st.selectbox(
                    "Sottocategoria", range(len(studies)),
                    format_func=lambda i: studies[i]["title"],
                    key=f"manual_study_{category['title']}")
                study = studies[study_index]
                chapters = study["chapters"]
                chapter_index = st.selectbox(
                    "Posizione", range(len(chapters)),
                    format_func=lambda i: chapters[i]["title"],
                    key=f"manual_chapter_{category['title']}_{study['title']}")
                chapter = chapters[chapter_index]
                if chapter["objective"]:
                    st.caption(f"Obiettivo originale: {chapter['objective']}")
                if chapter["description"]:
                    with st.expander("Descrizione dell'esercizio"):
                        st.write(chapter["description"])
                st.button(
                    "Carica posizione sulla scacchiera", type="primary",
                    use_container_width=True,
                    on_click=load_manual_position,
                    args=(chapter["fen"],
                          f"{category['title']} / {study['title']} / {chapter['title']}"),
                    key="load_manual_chapter")
                if chapter["url"].startswith("https://"):
                    st.link_button("Apri capitolo su Lichess", chapter["url"],
                                   use_container_width=True)

        st.divider()
        st.header("Modello e giocatori", anchor=False)
        model_name = st.selectbox("Checkpoint EloChess", names,
                                  index=names.index(preferred), key="chosen_model")
        elo_white = int(st.number_input("Elo del Bianco", min_value=100, max_value=4000,
                                        value=1600, step=50))
        elo_black = int(st.number_input("Elo del Nero", min_value=100, max_value=4000,
                                        value=1600, step=50))
        orientation = st.radio("Visuale della scacchiera", ["Bianco", "Nero"], horizontal=True)
        depth = st.slider("Semimosse di ricerca", min_value=0, max_value=4,
                          value=0, step=1, help=SEARCH_HELP)
        show_lines = st.toggle("Calcola anche le prime 3 varianti (4 semimosse)", value=False)

        st.divider()
        st.header("Stockfish", anchor=False)
        use_stockfish = st.toggle("Analizza anche con Stockfish", value=True)
        stockfish_path_text = ""
        stockfish_limit_mode = "Tempo"
        stockfish_limit_value = 1.0
        stockfish_top_n = 3
        stockfish_threads = 2
        stockfish_hash = 128
        if use_stockfish:
            stockfish_path_text = st.text_input(
                "Eseguibile Stockfish (UCI)", value=suggested_stockfish_path(),
                help="Percorso assoluto oppure relativo alla radice del repository. "
                     "Accetta anche 'stockfish' se il motore e' nel PATH.",
            )
            stockfish_limit_mode = st.radio(
                "Limite di analisi", ["Tempo", "Profondità"], horizontal=True
            )
            if stockfish_limit_mode == "Tempo":
                stockfish_limit_value = float(st.number_input(
                    "Secondi per posizione", min_value=0.5, max_value=30.0,
                    value=1.0, step=0.5, format="%.1f"
                ))
            else:
                stockfish_limit_value = float(st.number_input(
                    "Profondità (semimosse)", min_value=1, max_value=30,
                    value=16, step=1
                ))
            stockfish_top_n = int(st.slider("Varianti Stockfish (MultiPV)", 1, 7, 3))
            with st.expander("Prestazioni Stockfish"):
                stockfish_threads = int(st.number_input(
                    "Threads", min_value=1, max_value=32, value=2, step=1
                ))
                stockfish_hash = int(st.number_input(
                    "Hash (MB)", min_value=16, max_value=2048, value=128, step=16
                ))

    board = st.session_state.play_board
    model_path = MODELS_DIR / model_name
    if not VOCAB_PATH.is_file():
        st.error(f"Vocabolario non trovato: {VOCAB_PATH}")
        st.stop()
    model_mtime = model_path.stat().st_mtime_ns
    vocab_mtime = VOCAB_PATH.stat().st_mtime_ns

    left, right = st.columns([1, 1.5], gap="small")
    with left:
        st.markdown('<span id="elochess-board-column-anchor" style="display:none"></span>',
                    unsafe_allow_html=True)
        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1.4])
        ctrl1.button("↶ Annulla", on_click=undo, disabled=not bool(board.move_stack), use_container_width=True)
        ctrl2.button("↻ Nuova partita", on_click=reset_board, use_container_width=True)
        ctrl3.caption(f"Al tratto: **{'Bianco' if board.turn else 'Nero'}**")
        if st.session_state.manual_initial_fen:
            st.caption(f"Esercizio: **{st.session_state.manual_position_label}**")
            st.button("↺ Torna alla posizione iniziale dell'esercizio",
                      use_container_width=True, on_click=load_manual_position,
                      args=(st.session_state.manual_initial_fen,
                            st.session_state.manual_position_label))

        last_move = board.peek() if board.move_stack else None
        moves_legal = [m.uci() for m in board.legal_moves]
        event = chessboard_component(
            fen=board.fen(),
            pieces={chess.square_name(sq): piece.symbol() for sq, piece in board.piece_map().items()},
            legal=moves_legal,
            images=piece_assets(),
            orientation="white" if orientation == "Bianco" else "black",
            turn="white" if board.turn else "black",
            last_move=([chess.square_name(last_move.from_square), chess.square_name(last_move.to_square)]
                       if last_move else []),
            check=(chess.square_name(board.king(board.turn))
                   if board.is_check() and board.king(board.turn) is not None else ""),
            key="elochess_board",
            default=None,
        )
        if apply_board_event(event):
            st.rerun()

        if st.session_state.play_error:
            st.warning(st.session_state.play_error)

        # Stockfish direttamente sotto la scacchiera, prima di FEN e cronologia.
        # Valutazione e profondità restano nella sola tabella.
        if use_stockfish and not board.is_game_over(claim_draw=False):
            try:
                engine_path = stockfish_executable(stockfish_path_text)
                with st.spinner("Analisi Stockfish in corso..."):
                    stockfish_moves = analyze_stockfish(
                        board.fen(), str(engine_path), engine_path.stat().st_mtime_ns,
                        stockfish_limit_mode, stockfish_limit_value,
                        stockfish_top_n, stockfish_threads, stockfish_hash,
                    )
            except Exception as exc:
                st.warning(f"Analisi Stockfish non disponibile: {exc}")
            else:
                st.markdown(
                    '<div class="elochess-value-heading">'
                    '<span class="elochess-value-title">Stockfish</span>'
                    '</div>', unsafe_allow_html=True,
                )
                with st.container(border=True, key="stockfish_moves_table"):
                    sf_widths = [1.35, 1.35, 0.65]
                    with st.container(key="stockfish_moves_header"):
                        header = st.columns(sf_widths, gap="small", vertical_alignment="center")
                        for cell, title in zip(header, ("Mossa", "Valutazione", "Depth")):
                            cell.markdown(f"**{title}**")
                    st.markdown('<div class="elochess-table-separator elochess-table-separator--header"></div>',
                                unsafe_allow_html=True)
                    for rank, move in enumerate(stockfish_moves, start=1):
                        # Raggruppare i componenti evita sovrapposizioni tra
                        # pulsante, testo della variante e riga successiva.
                        with st.container(key=f"stockfish_move_row_{rank}"):
                            row = st.columns(sf_widths, gap="small", vertical_alignment="center")
                            row[0].button(
                                move["san"],
                                key=f"stockfish_candidate_{rank}_{move['uci']}",
                                use_container_width=True,
                                on_click=play_uci, args=(move["uci"],),
                            )
                            row[1].markdown(
                                stockfish_score_label(move["cp_white"], move["mate_white"])
                            )
                            row[2].markdown(
                                str(move["depth"]) if move["depth"] is not None else "—"
                            )
                            if move["continuation"]:
                                st.caption(f"↳ {move['continuation']}")
                        # Separatore DOPO l'intera riga, inclusa la continuazione.
                        if rank < len(stockfish_moves):
                            st.markdown('<div class="elochess-table-separator"></div>',
                                        unsafe_allow_html=True)

        with st.expander("FEN e cronologia", expanded=False):
            # Aggiorna il campo solo quando cambia davvero la posizione; non
            # sovrascrive un FEN che l'utente sta ancora digitando.
            if st.session_state.get("fen_entry_board") != board.fen():
                st.session_state.fen_entry = board.fen()
                st.session_state.fen_entry_board = board.fen()
            st.text_input("FEN (attuale o incolla una nuova posizione)", key="fen_entry")
            st.button("Carica FEN", on_click=load_fen)
            if board.move_stack:
                # Notazione standard: "1. e4 e5 2. Nf3 ...", anche da FEN personalizzati.
                st.write(board.root().variation_san(board.move_stack))

    with right:
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            result = 0.0 if outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
            st.metric("Risultato finale (Bianco)", f"{result:+.2f}")
            st.info(f"Partita conclusa: {outcome.result()}. Nessuna predizione in posizione terminale.")
            return
        try:
            with st.spinner("Analisi della posizione..."):
                analysis = analyze(board.fen(), elo_white, elo_black,
                                   str(model_path.resolve()), model_mtime, vocab_mtime)
        except Exception as exc:
            st.error(f"Impossibile valutare la posizione: {exc}")
            return

        st.markdown(
            '<div class="elochess-value-heading">'
            '<span class="elochess-value-title">Value</span>'
            '<span class="elochess-value-legend">'
            '+1 = vittoria Bianco <span class="elochess-value-separator">●</span> '
            '0 = patta <span class="elochess-value-separator">●</span> '
            '−1 = vittoria Nero'
            '</span></div>',
            unsafe_allow_html=True,
        )
        st.metric("Value", f"{analysis['signed_white']:+.2f}", label_visibility="collapsed")

        st.subheader(f"Mosse consigliate — {len(analysis['moves'])} legali", anchor=False)
        st.caption(
            "Policy = probabilità originale del modello. "
            "Ricerca = preferenza ricalcolata usando il Value."
        )
        moves_display = [dict(move) for move in analysis['moves']]
        has_search = False
        if depth:
            try:
                with st.spinner(f"Ricerca a {depth} semimosse..."):
                    revised = search_probabilities(
                        board.fen(), elo_white, elo_black,
                        str(model_path.resolve()), model_mtime, vocab_mtime, depth)
                for move in moves_display:
                    move['searched_probability'] = revised[move['uci']]['probability']
                    move['searched_score'] = revised[move['uci']]['expected_score']
                moves_display.sort(key=lambda m: -m['searched_probability'])
                has_search = True
            except Exception as exc:
                st.error(f"Ricerca non disponibile: {exc}. Mostro la Policy originale.")

        lines = {}
        if show_lines and moves_display:
            with st.spinner("Calcolo delle continuazioni EloChess..."):
                lines = follow_lines(board.fen(), elo_white, elo_black,
                                     str(model_path.resolve()), model_mtime, vocab_mtime,
                                     tuple(m["uci"] for m in moves_display[:3]), 4)

        widths = [1.55, 0.95, 0.95, 0.78] if has_search else [1.55, 1]
        headings = ["Mossa", "Policy", "Ricerca", "Value"] if has_search else ["Mossa", "Policy"]
        with st.container(height=550, border=True, key="elochess_moves_table"):
            with st.container(key="elochess_moves_header"):
                head_cells = st.columns(widths, gap="small", vertical_alignment="center")
                for cell, title in zip(head_cells, headings):
                    cell.markdown(f"**{title}**")
            st.markdown('<div class="elochess-table-separator elochess-table-separator--header"></div>',
                        unsafe_allow_html=True)
            for i, move in enumerate(moves_display, start=1):
                with st.container(key=f"elochess_move_row_{i}"):
                    row = st.columns(widths, gap="small", vertical_alignment="center")
                    row[0].button(numbered_san(board, move['san']),
                                  key=f"candidate_{i}_{move['uci']}",
                                  use_container_width=True,
                                  on_click=play_uci, args=(move["uci"],))
                    row[1].markdown(probability_label(move["probability"]))
                    if has_search:
                        row[2].markdown(probability_label(move["searched_probability"]))
                        value_white = white_signed_from_root_score(move['searched_score'], board.turn)
                        row[3].markdown(f"{value_white:+.2f}")
                    if move["uci"] in lines and lines[move["uci"]]:
                        st.caption(f"↳ {lines[move['uci']]}")
                # Separatore DOPO l'intera riga, inclusa la continuazione.
                if i < len(moves_display):
                    st.markdown('<div class="elochess-table-separator"></div>',
                                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
