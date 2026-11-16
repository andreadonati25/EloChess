#!/usr/bin/env python3
"""Analizza una volta tutte le posizioni del benchmark con Stockfish (punto di vista Bianco)."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import chess
import chess.engine
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.utils import resolve_path  # noqa: E402

def load_positions(path, max_positions):
    rows = pq.read_table(path, columns=['position_id', 'fen', 'source', 'source_file', 'theme']).to_pylist()
    if max_positions is not None:
        rows = rows[:max_positions]
    if not rows:
        raise ValueError('Benchmark vuoto')
    ids = [row['position_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('position_id duplicati nel benchmark')
    for row in rows:
        board = chess.Board(row['fen'])
        if not board.is_valid() or board.is_game_over(claim_draw=False):
            raise ValueError(f"Posizione non valida o terminale: {row['position_id']}")
    return rows



def benchmark_digest(rows):
    data = [(r['position_id'], r['fen']) for r in rows]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode('utf-8')).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--output', default=None)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--time', type=float, default=1.0)
    ap.add_argument('--depth', type=int, default=None)
    ap.add_argument('--top_n', type=int, default=7)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--hash_mb', type=int, default=64)
    args = ap.parse_args()
    if args.depth is None and args.time <= 0: ap.error('--time deve essere positivo')
    if args.depth is not None and args.depth <= 0: ap.error('--depth deve essere positivo')
    if min(args.top_n, args.threads, args.hash_mb) < 1: ap.error('top_n, threads e hash_mb devono essere positivi')
    bench = resolve_path(args.benchmark)
    rows = load_positions(bench, None)
    out = resolve_path(args.output) if args.output else bench.parent / bench.stem / "stockfish.parquet"
    if out.exists():
        ap.error(f'File già esistente: {out}. Specifica un nuovo --output oppure rimuovi manualmente il precedente.')
    engine_path = resolve_path(args.engine)
    if not engine_path.is_file(): ap.error(f'Stockfish non trovato: {engine_path}')
    analyses = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        settings = {}
        if 'Threads' in engine.options: settings['Threads'] = args.threads
        if 'Hash' in engine.options: settings['Hash'] = args.hash_mb
        if settings: engine.configure(settings)
        for record in tqdm(rows, desc='Stockfish', unit='pos'):
            board = chess.Board(record['fen'])
            limit = chess.engine.Limit(depth=args.depth) if args.depth else chess.engine.Limit(time=args.time)
            infos = engine.analyse(board, limit, multipv=min(args.top_n, board.legal_moves.count()), game=object())
            if isinstance(infos, dict): infos = [infos]
            moves=[]
            for rank, info in enumerate(infos, 1):
                if not info.get('pv') or 'score' not in info: continue
                mv = info['pv'][0]
                score = info['score'].white()
                moves.append({'rank':rank,'uci':mv.uci(),'san':board.san(mv),
                    'cp_white':score.score(),'mate_white':score.mate(),'depth':info.get('depth'),
                    'pv_uci':[v.uci() for v in info['pv'][:12]]})
            if not moves: raise RuntimeError(f"Nessuna analisi: {record['position_id']}")
            analyses.append({'position_id':record['position_id'],'fen':record['fen'],'moves':moves})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(analyses), out, compression='zstd')
    metadata={'benchmark':str(bench),'benchmark_sha256':benchmark_digest(rows),
        'engine':str(engine_path),'n_positions':len(rows),'top_n':args.top_n,
        'time_seconds':args.time if args.depth is None else None,'depth':args.depth,
        'threads':args.threads,'hash_mb':args.hash_mb,'evaluation_pov':'white',
        'note':'MultiPV delle prime N mosse, non analisi esaustiva di tutte le mosse legali.'}
    path = out.with_suffix('.metadata.json')
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Stockfish: {out}; metadata: {path}; posizioni: {len(analyses)}')

if __name__ == '__main__': main()
