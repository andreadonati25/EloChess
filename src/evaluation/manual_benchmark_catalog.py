#!/usr/bin/env python3
"""Lettura dei PGN del Manual Benchmark (categoria/cartella, studio/file, capitolo/FEN).

Il catalogo viene ricostruito dalle cartelle ad ogni chiamata: nuove categorie e
sottocategorie sono visibili al prossimo rerun del Playground, senza indici JSON.
I singoli PGN sono memorizzati in cache fino alla loro prossima modifica.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STANDARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
CATEGORY_ORDER = (
    "Checkmates", "Fundamental Tactics", "Advanced Tactics", "Pawn Endgames",
    "Rook Endgames", "Aperture e sviluppo", "Casi particolari Elo",
)
# Anche se sono conservati i vecchi file sorgenti qui sotto, non sono una categoria.
EXCLUDED_DIRS = {"pgn_originali", "__pycache__"}
HEADER_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]\s*$')
DRAWING_TAG_RE = re.compile(r'\[%[A-Za-z]+[^\]]*\]')
EVENT_SPLIT_RE = re.compile(r'(?=^\[Event\s+")', flags=re.MULTILINE)


def _unescape_header(value: str) -> str:
    return value.replace(r'\"', '"').replace(r'\\', '\\')


def _root_description(movetext: str) -> str:
    """Commenti PGN prima della prima mossa, senza marcatori grafici Lichess."""
    cursor = movetext.lstrip()
    comments: list[str] = []
    while cursor.startswith("{"):
        end = cursor.find("}")
        if end < 0:
            break
        comment = DRAWING_TAG_RE.sub("", cursor[1:end]).strip()
        if comment:
            comments.append(comment)
        cursor = cursor[end + 1:].lstrip()
    return "\n\n".join(comments)


@lru_cache(maxsize=512)
def _read_pgn_cached(path_str: str, mtime_ns: int, file_size: int) -> dict:
    """Legge i soli tag e commenti iniziali: non modifica le linee o i PGN."""
    path = Path(path_str)
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    chapters: list[dict] = []
    description = ""
    order: int | None = None

    for block in EVENT_SPLIT_RE.split(raw):
        if not block.strip():
            continue
        headers: dict[str, str] = {}
        lines = block.splitlines(keepends=True)
        start_movetext = 0
        for ix, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            match = HEADER_RE.fullmatch(stripped)
            if match is None:
                start_movetext = ix
                break
            headers[match.group(1)] = _unescape_header(match.group(2))
        else:
            start_movetext = len(lines)
        if not headers:
            raise ValueError(f"Intestazione PGN non riconosciuta in: {path.name}")
        if not headers.get("ChapterName"):
            raise ValueError(f"Manca [ChapterName] nel capitolo {len(chapters)+1} di {path.name}")
        fen = headers.get("FEN", STANDARD_FEN)
        if len(fen.split()) != 6:
            raise ValueError(f"FEN non valido per {headers['ChapterName']} in {path.name}")
        if headers.get("StudyDescription"):
            description = headers["StudyDescription"].replace(r"\n", "\n")
        if headers.get("StudyOrder"):
            try:
                order = int(headers["StudyOrder"])
            except ValueError as exc:
                raise ValueError(f"StudyOrder non numerico in {path.name}") from exc
        chapters.append({
            "title": headers["ChapterName"],
            "fen": fen,
            "description": _root_description("".join(lines[start_movetext:])),
            "url": headers.get("ChapterURL", ""),
            "objective": headers.get("Termination", ""),
        })
    if not chapters:
        raise ValueError(f"Nessuna posizione nel file: {path}")
    return {"title": path.stem, "description": description,
            "order": order, "chapters": chapters}


def load_manual_catalog(directory: Path) -> list[dict]:
    """Restituisce un catalogo gerarchico dinamico dai PGN sul filesystem.

    Un nuovo .pgn o una nuova cartella compare automaticamente al prossimo rerun.
    Le categorie vuote con un .gitkeep restano visibili, senza sottocategorie.
    """
    if not directory.is_dir():
        return []
    categories: list[dict] = []
    for folder in directory.iterdir():
        if not folder.is_dir() or folder.name.startswith(".") or folder.name.startswith("_"):
            continue
        if folder.name.lower() in EXCLUDED_DIRS:
            continue
        files = list(folder.glob("*.pgn"))
        if not files and not (folder / ".gitkeep").exists():
            continue
        subcategories: list[dict] = []
        for path in files:
            stat = path.stat()
            subcategories.append(_read_pgn_cached(str(path), stat.st_mtime_ns, stat.st_size))
        subcategories.sort(key=lambda s: (
            s["order"] is None,
            s["order"] if s["order"] is not None else 10**6,
            s["title"].casefold(),
        ))
        categories.append({"title": folder.name, "subcategories": subcategories})
    priority = {name: i for i, name in enumerate(CATEGORY_ORDER)}
    categories.sort(key=lambda c: (priority.get(c["title"], len(priority)), c["title"].casefold()))
    return categories


if __name__ == "__main__":
    from src.config.utils import resolve_path

    home = (resolve_path(sys.argv[1]) if len(sys.argv) > 1
            else ROOT / "data/evaluation/manual_benchmark")
    result = load_manual_catalog(home)
    for cat in result:
        print(f'{cat["title"]}: {len(cat["subcategories"])} sottocategorie, '
              f'{sum(len(s["chapters"]) for s in cat["subcategories"])} posizioni')
