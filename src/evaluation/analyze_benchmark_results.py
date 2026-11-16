#!/usr/bin/env python3
"""Analisi automatica di UN modello EloChess sul benchmark, senza rieseguire i modelli.

Legge i tre Parquet prodotti da build_benchmark, evaluate_benchmark_stockfish
ed evaluate_benchmark_elochess. Separa Elo originali Lichess e griglia Elo delle
suite. Produce Markdown, CSV e SVG in archive/manual_inspection/.

Esempio:
 python src/evaluation/analyze_benchmark_results.py \
   --benchmark data/evaluation/benchmark_v1_5000.parquet \
   --model models/v1/best.keras --focus_elo 1200 --max_cases 30
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import chess
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config.utils import resolve_path  # noqa: E402

EPS = 1e-12


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata mancante: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Metadata non validi: {path}")
    return obj


def digest(rows: list[dict]) -> str:
    content = [(r["position_id"], r["fen"]) for r in rows]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._") or "model"


def first_defined(*values):
    return next((x for x in values if x is not None), None)


def load_data(args):
    bench = resolve_path(args.benchmark)
    if not bench.is_file():
        raise FileNotFoundError(f"Benchmark non trovato: {bench}")
    base = bench.parent / bench.stem
    sf_path = resolve_path(args.stockfish) if args.stockfish else base / "stockfish.parquet"
    if args.results:
        result_path = resolve_path(args.results)
    elif args.model:
        result_path = base / f"elochess_{Path(args.model).parent.name}" / "results.parquet"
    else:
        found = sorted(base.glob("elochess_*/results.parquet"))
        if len(found) != 1:
            raise ValueError(
                f"Trovati {len(found)} risultati EloChess in {base}; "
                "indicare --model oppure --results per selezionarne uno."
            )
        result_path = found[0]
    for p in (sf_path, result_path):
        if not p.is_file():
            raise FileNotFoundError(f"File mancante: {p}")

    b = pq.read_table(bench).to_pylist()
    s = pq.read_table(sf_path).to_pylist()
    e = pq.read_table(result_path).to_pylist()
    if not b or not e or not s:
        raise ValueError("Uno dei Parquet e' vuoto")
    ids = [r["position_id"] for r in b]
    if len(set(ids)) != len(ids):
        raise ValueError("position_id ripetuti nel benchmark")
    bench_by_id = {r["position_id"]: r for r in b}
    sf_by_id = {r["position_id"]: r for r in s}
    if len(sf_by_id) != len(s) or set(sf_by_id) != set(bench_by_id):
        raise ValueError("Stockfish non contiene esattamente le posizioni del benchmark")
    for pid, row in bench_by_id.items():
        if sf_by_id[pid]["fen"] != row["fen"] or not sf_by_id[pid]["moves"]:
            raise ValueError(f"FEN o mosse Stockfish mancanti: {pid}")
    fingerprint = digest(b)
    sf_meta = read_json(sf_path.with_suffix(".metadata.json"))
    elo_meta = read_json(result_path.with_suffix(".metadata.json"))
    for label, meta in (("Stockfish", sf_meta), ("EloChess", elo_meta)):
        if meta.get("benchmark_sha256") != fingerprint:
            raise ValueError(f"SHA256 non corrispondente nel metadata {label}")
    if sf_meta.get("evaluation_pov") != "white" or elo_meta.get("stockfish_pov") != "white":
        raise ValueError("Prospettiva Stockfish diversa da white")
    if elo_meta.get("elochess_value_pov") != "side_to_move":
        raise ValueError("Prospettiva Value EloChess inattesa")
    if elo_meta.get("n_predictions") != len(e):
        raise ValueError("Numero di predizioni non coerente con i metadata EloChess")
    if elo_meta.get("positions_evaluated") != len({r["position_id"] for r in e}):
        raise ValueError("Numero di posizioni valutate non coerente")
    if elo_meta.get("top_n", 0) > sf_meta.get("top_n", 0):
        raise ValueError("Il modello richiede piu' varianti delle MultiPV disponibili")

    observed_models = {str(r["model"]) for r in e}
    if len(observed_models) != 1:
        raise ValueError("Il report accetta esattamente un modello per esecuzione")
    model_name = observed_models.pop()
    if args.model and model_name != Path(args.model).parent.name:
        raise ValueError(f"Il modello nel Parquet e' {model_name}, non {Path(args.model).parent.name}")
    if str(elo_meta.get("model", "")) and Path(elo_meta["model"]).parent.name != model_name:
        raise ValueError("Il modello dei metadata EloChess non corrisponde al Parquet")

    seen = set()
    pred_ids = set()
    for r in e:
        pid = r["position_id"]
        if pid not in bench_by_id or r["fen"] != bench_by_id[pid]["fen"]:
            raise ValueError(f"Posizione EloChess non corrispondente: {pid}")
        key = (pid, r["elo_setting"], int(r["elo_side"]), int(r["elo_opponent"]))
        if key in seen:
            raise ValueError(f"Duplicazione di posizione/Elo: {key}")
        seen.add(key)
        pred_ids.add(pid)
        if not r.get("elochess_moves"):
            raise ValueError(f"Lista delle mosse EloChess vuota: {pid}")
        if r["elochess_top1_uci"] != r["elochess_moves"][0]["uci"]:
            raise ValueError(f"Top-1 EloChess non coerente con le mosse: {pid}")
        if r["sf_top1_uci"] != sf_by_id[pid]["moves"][0]["uci"]:
            raise ValueError(f"Top-1 Stockfish non coerente: {pid}")
        if r["elo_setting"] not in ("original", "grid"):
            raise ValueError(f"Origine Elo non riconosciuta: {r['elo_setting']}")
        if r["elo_setting"] == "original" and bench_by_id[pid].get("played_move_uci") is None:
            raise ValueError(f"Elo originali senza mossa umana: {pid}")
    if len(pred_ids) < len(b):
        print(f"ATTENZIONE: analisi parziale: {len(pred_ids)} di {len(b)} posizioni.")

    output = resolve_path(args.output_dir) if args.output_dir else (
        ROOT / "archive/manual_inspection" / bench.stem / safe_name(model_name)
    )
    return b, s, e, sf_by_id, sf_meta, elo_meta, output, model_name, result_path


def enrich(predictions: list[dict], sf_by_id: dict, benchmark_by_id: dict) -> list[dict]:
    enriched = []
    for r in predictions:
        p = benchmark_by_id[r["position_id"]]
        sf = sf_by_id[r["position_id"]]
        sf_moves = sf["moves"]
        sf_by_uci = {m["uci"]: m for m in sf_moves}
        top = r["elochess_moves"][0]
        suggested = top["uci"]
        side_white = r["side_to_move"] == "white"
        sf_best = sf_moves[0]
        sf_chosen = sf_by_uci.get(suggested)
        cp_loss = None
        if sf_chosen is not None and sf_chosen["cp_white"] is not None and sf_best["cp_white"] is not None:
            delta = sf_best["cp_white"] - sf_chosen["cp_white"]
            cp_loss = float(delta if side_white else -delta)
        is_original = r["elo_setting"] == "original"
        actual = p.get("played_move_uci") if is_original else None
        actual_p = r.get("human_move_probability") if is_original else None
        y = p.get("y_value_to_move") if is_original else None
        value = float(r["value_expected_score_to_move"])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Value fuori [0,1]: {r['position_id']}")
        if actual_p is not None and not 0.0 <= float(actual_p) <= 1.0 + 1e-5:
            raise ValueError(f"Probabilita' umana non valida: {r['position_id']}")
        board = chess.Board(r["fen"])
        mv = chess.Move.from_uci(suggested)
        if mv not in board.legal_moves:
            raise ValueError(f"La Top-1 del modello non e' legale: {r['position_id']} / {suggested}")
        sf_move = chess.Move.from_uci(sf_best["uci"])
        sf_san = board.san(sf_move) if sf_move in board.legal_moves else "n/d"
        enriched.append({
            "position_id": r["position_id"], "fen": r["fen"],
            "source": str(r["source"]), "source_file": r.get("source_file"),
            "theme": r.get("theme"), "stage": r.get("stage"),
            "elo_band": r.get("elo_band"), "side_to_move": r["side_to_move"],
            "elo_side": int(r["elo_side"]), "elo_opponent": int(r["elo_opponent"]),
            "elo_setting": r["elo_setting"], "top1_uci": suggested,
            "top1_san": board.san(mv), "top1_probability": float(top["probability"]),
            "top_n_mass": sum(float(m["probability"]) for m in r["elochess_moves"]),
            "sf_top1_uci": sf_best["uci"], "sf_top1_san": sf_san,
            "sf_rank": None if sf_chosen is None else int(sf_chosen["rank"]),
            "sf_top1_match": suggested == sf_best["uci"],
            "sf_topn_match": sf_chosen is not None,
            "cp_loss_known": cp_loss,
            "value_to_move": value,
            "value_white_signed": (2 * value - 1) * (1 if side_white else -1),
            "played_move_uci": actual,
            "human_top1_match": None if actual is None else suggested == actual,
            "human_probability": None if actual_p is None else float(actual_p),
            "human_rank": r.get("human_move_rank") if is_original else None,
            "human_ce": None if actual_p is None else -math.log(max(float(actual_p), EPS)),
            "y_value_to_move": y,
            "value_squared_error": None if y is None else (value - float(y)) ** 2,
            "result": p.get("result") if is_original else None,
            "sf_topn_uci": " ".join(m["uci"] for m in sf_moves),
        })
        p_top = enriched[-1]["top1_probability"]
        mass = enriched[-1]["top_n_mass"]
        if not math.isfinite(p_top) or not 0 <= p_top <= 1.00001:
            raise ValueError(f"Probabilita' Top-1 non valida: {r['position_id']}")
        if not math.isfinite(mass) or mass < 0 or mass > 1.0001:
            raise ValueError(f"Massa di probabilita' Top-N non valida: {r['position_id']}")
    return enriched


def avg(rows: list[dict], name: str) -> float | None:
    values = [float(r[name]) for r in rows if r.get(name) is not None]
    return mean(values) if values else None


def pct_yes(rows: list[dict], name: str) -> float | None:
    values = [bool(r[name]) for r in rows if r.get(name) is not None]
    return sum(values) / len(values) if values else None


def pct_threshold(rows: list[dict], name: str, threshold: float) -> float | None:
    values = [r[name] for r in rows if r.get(name) is not None]
    return sum(float(p) >= threshold for p in values) / len(values) if values else None


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"predictions": 0, "positions": 0}
    original = [r for r in rows if r["elo_setting"] == "original"]
    known_cp = [r for r in rows if r["cp_loss_known"] is not None]
    return {
        "predictions": len(rows),
        "positions": len({r["position_id"] for r in rows}),
        "stockfish_top1_agreement": pct_yes(rows, "sf_top1_match"),
        "stockfish_topn_coverage": pct_yes(rows, "sf_topn_match"),
        "average_top1_probability": avg(rows, "top1_probability"),
        "top1_at_least_99_9_percent": pct_threshold(rows, "top1_probability", 0.999),
        "average_topn_probability_mass": avg(rows, "top_n_mass"),
        "mean_value_to_move": avg(rows, "value_to_move"),
        "cp_loss_conditional": avg(known_cp, "cp_loss_known"),
        "cp_loss_known_cases": len(known_cp),
        "human_cases": len(original),
        "human_top1_accuracy": pct_yes(original, "human_top1_match"),
        "human_move_top3_accuracy": sum(int(r["human_rank"] <= 3) for r in original if r["human_rank"] is not None) / len(original) if original else None,
        "human_move_top5_accuracy": sum(int(r["human_rank"] <= 5) for r in original if r["human_rank"] is not None) / len(original) if original else None,
        "human_mean_probability": avg(original, "human_probability"),
        "human_ce_capped_at_1e_12": avg(original, "human_ce"),
        "value_mse_to_move": avg(original, "value_squared_error"),
        "mean_observed_score_to_move": avg(original, "y_value_to_move"),
    }


def grouped(rows: list[dict], *keys: str) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple("n/d" if r.get(k) is None else str(r[k]) for k in keys)].append(r)
    out = []
    for key, items in sorted(groups.items()):
        out.append({**dict(zip(keys, key)), **metrics(items)})
    return out


def write_csv(path: Path, rows: list[dict]):
    # Anche un gruppo non disponibile produce un CSV leggibile e non un file assente.
    if not rows:
        path.write_text("note\nNessun dato disponibile per questo gruppo\n", encoding="utf-8-sig")
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sensitivity(rows: list[dict], fixed_col: str) -> list[dict]:
    grid = [r for r in rows if r["elo_setting"] == "grid"]
    groups = defaultdict(list)
    for r in grid:
        groups[(r["position_id"], r[fixed_col])].append(r)
    per_fixed = defaultdict(list)
    for (pid, fixed_elo), items in groups.items():
        if len(items) < 2:
            continue
        different = len({r["top1_uci"] for r in items})
        per_fixed[fixed_elo].append((different, len(items)))
    out = []
    for elo, data in sorted(per_fixed.items()):
        out.append({
            fixed_col: elo,
            "positions_with_two_or_more_comparisons": len(data),
            "positions_changing_top1": sum(n > 1 for n, _ in data),
            "fraction_changing_top1": mean(n > 1 for n, _ in data),
            "average_distinct_top1": mean(n for n, _ in data),
            "average_number_of_elo_comparisons": mean(n for _, n in data),
        })
    return out


def calibration(rows: list[dict], bins: int = 10) -> list[dict]:
    original = [r for r in rows if r["elo_setting"] == "original"]
    groups = defaultdict(list)
    for r in original:
        b = min(int(r["value_to_move"] * bins), bins - 1)
        groups[b].append(r)
    result = []
    for i in range(bins):
        items = groups.get(i, [])
        result.append({
            "predicted_score_interval": f"[{i/bins:.1f},{(i+1)/bins:.1f}{']' if i==bins-1 else ')'}",
            "positions": len(items),
            "mean_predicted_score_to_move": avg(items, "value_to_move"),
            "mean_observed_score_to_move": avg(items, "y_value_to_move"),
            "value_mse_to_move": avg(items, "value_squared_error"),
        })
    return result


def top_moves(rows: list[dict], max_rows: int = 35) -> list[dict]:
    result = []
    for setting in ("original", "grid"):
        records = [r for r in rows if r["elo_setting"] == setting]
        counts = Counter(r["top1_uci"] for r in records)
        pos = defaultdict(set)
        for r in records:
            pos[r["top1_uci"]].add(r["position_id"])
        for uci, count in counts.most_common(max_rows):
            result.append({"elo_setting": setting, "uci": uci,
                           "predictions": count, "distinct_positions": len(pos[uci]),
                           "share_of_group": count / len(records)})
    return result


def candidate_cases(rows: list[dict], n: int) -> list[dict]:
    """Casi distinti per posizione, diversificati per situazione e origine Elo."""
    if n <= 0:
        return []
    categories = {
        "high_confidence_outside_sf_topn": lambda r: not r["sf_topn_match"] and r["top1_probability"] >= 0.5,
        "human_disagreement": lambda r: r["elo_setting"] == "original" and r["human_top1_match"] is False,
        "large_value_error": lambda r: r["value_squared_error"] is not None and r["value_squared_error"] >= 0.25,
        "top1_matches_stockfish": lambda r: r["sf_top1_match"],
        "general_disagreement": lambda r: not r["sf_topn_match"],
    }
    def sort_value(cat, r):
        if cat == "large_value_error":
            return r["value_squared_error"] or 0
        return r["top1_probability"]
    used = set()
    cases = []
    # Favorire sia posizioni con Elo originali sia posizioni della griglia.
    for category in categories:
        for group in ("original", "grid"):
            target = max(1, math.ceil(n / (len(categories) * 2)))
            pool = sorted((r for r in rows if r["elo_setting"] == group and categories[category](r)),
                          key=lambda r: (-sort_value(category, r), r["position_id"], r["elo_side"], r["elo_opponent"]))
            added = 0
            for r in pool:
                if r["position_id"] in used:
                    continue
                cases.append({"reason": category, **r})
                used.add(r["position_id"])
                added += 1
                if added == target or len(cases) == n:
                    break
            if len(cases) == n:
                break
        if len(cases) == n:
            break
    if len(cases) < n:
        for r in sorted(rows, key=lambda r: (-r["top1_probability"], r["position_id"])):
            if r["position_id"] not in used:
                cases.append({"reason": "high_confidence", **r})
                used.add(r["position_id"])
            if len(cases) == n:
                break
    return cases


def detailed_grid(rows: list[dict], focus_elo: int):
    grid = [r for r in rows if r["elo_setting"] == "grid" and r["elo_side"] == focus_elo]
    groups = defaultdict(list)
    for r in grid:
        groups[r["position_id"]].append(r)
    result = []
    for pid, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: r["elo_opponent"])
        sample = ordered[0]
        result.append({
            "position_id": pid, "fen": sample["fen"], "source": sample["source"],
            "theme": sample["theme"], "elo_side": focus_elo,
            "opponent_elos_evaluated": len(ordered),
            "distinct_top1": len({r["top1_uci"] for r in ordered}),
            "top1_changes": len({r["top1_uci"] for r in ordered}) > 1,
            "continuation_by_opponent_elo": " | ".join(
                f"{r['elo_opponent']}: {r['top1_san']} {r['top1_probability']:.1%}" for r in ordered
            ),
        })
    return result


def number(v, digits=2):
    return "n/d" if v is None else f"{v:.{digits}f}"


def percent(v):
    return "n/d" if v is None else f"{v:.2%}"


def markdown_table(rows: list[dict], cols: list[tuple[str, str]], max_rows=20) -> str:
    if not rows:
        return "Nessun dato disponibile.\n"
    lines = ["| " + " | ".join(label for _, label in cols) + " |",
             "| " + " | ".join("---:" if i else "---" for i in range(len(cols))) + " |"]
    for r in rows[:max_rows]:
        parts = []
        for col, _ in cols:
            x = r.get(col)
            if isinstance(x, float):
                v = percent(x) if col.startswith("fraction_") or col in (
                    "stockfish_top1_agreement", "stockfish_topn_coverage", "human_top1_accuracy",
                    "top1_at_least_99_9_percent", "average_top1_probability", "human_mean_probability"
                ) else number(x, 3)
            else:
                v = "n/d" if x is None else str(x)
            parts.append(v.replace("|", "\\|"))
        lines.append("| " + " | ".join(parts) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n*Mostrate {max_rows} righe di {len(rows)}: dettagli nel CSV.*")
    return "\n".join(lines) + "\n"


def make_bar_svg(path: Path, title: str, rows: list[dict], labels: list[str], keys: list[str],
                 legend: list[str]):
    """Grafico SVG autonomo, senza Matplotlib e senza dipendenze aggiuntive."""
    if not rows or not labels:
        return
    colors = ["#277da1", "#43aa8b", "#f8961e"]
    width, margin_left, bar_h, gap, header = 870, 235, 17, 6, 95
    plot_w = 520
    height = header + len(rows) * (len(keys) * (bar_h + 3) + gap) + 35
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             f'<text x="22" y="30" font-family="Arial" font-size="18" font-weight="bold" fill="#23344b">{html.escape(title)}</text>']
    for j, name in enumerate(legend):
        x = 25 + j * 260
        lines.append(f'<rect x="{x}" y="48" width="13" height="13" fill="{colors[j]}"/>')
        lines.append(f'<text x="{x+19}" y="60" font-family="Arial" font-size="13">{html.escape(name)}</text>')
    for i, (r, label) in enumerate(zip(rows, labels)):
        group_y = header + i * (len(keys) * (bar_h + 3) + gap)
        lines.append(f'<text x="15" y="{group_y+18}" font-size="12" font-family="Arial" fill="#344154">{html.escape(str(label)[:33])}</text>')
        for j, k in enumerate(keys):
            val = r.get(k)
            if val is None:
                continue
            value = max(0, min(1, float(val)))
            y = group_y + j * (bar_h+3)
            lines.append(f'<rect x="{margin_left}" y="{y}" width="{plot_w}" height="{bar_h}" fill="#eef1f5"/>')
            lines.append(f'<rect x="{margin_left}" y="{y}" width="{value*plot_w:.1f}" height="{bar_h}" fill="{colors[j]}"/>')
            lines.append(f'<text x="{margin_left + plot_w + 9}" y="{y+13}" font-size="12" font-family="Arial">{value:.1%}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def report(path: Path, model: str, bench: str, result_file: str, sf_meta: dict, elo_meta: dict,
           groups: dict, sensitivity_side: list[dict], sensitivity_opp: list[dict], partial: bool,
           max_cases: int, focus_elo: int):
    original = groups["overall_original"][0] if groups["overall_original"] else None
    grid = groups["overall_grid"][0] if groups["overall_grid"] else None
    sf_n = sf_meta.get("top_n", "n/d")
    lines = [f"# EloChess — Analisi automatica: {model}", "",
             f"- Benchmark: `{bench}`", f"- Risultati EloChess: `{result_file}`",
             f"- Checkpoint: `{elo_meta.get('model', model)}`",
             f"- Modalità Elo: `{elo_meta.get('elo_mode', 'n/d')}`; MultiPV Stockfish: **{sf_n}**",
             f"- Stockfish: limite **{sf_meta.get('time_seconds') or sf_meta.get('depth')}** "
             f"({'secondi' if sf_meta.get('depth') is None else 'profondita'}) per posizione; "
             f"thread **{sf_meta.get('threads')}**, hash **{sf_meta.get('hash_mb')} MB**.",
             f"- Predizioni complessive: **{elo_meta['n_predictions']}**; "
             f"posizioni valutate: **{elo_meta['positions_evaluated']}**.", ""]
    if partial:
        lines += ["**ATTENZIONE: risultati parziali rispetto al benchmark completo.**", ""]
    lines += ["## 1. Risultati principali — due gruppi distinti", "",
              "In modalità `hybrid`, le partite con Elo registrati usano **gli Elo originali** "
              "e le suite senza Elo usano **una griglia controllata**; "
              "in modalità `grid`, tutti gli esempi usano la griglia. "
              "Le osservazioni della griglia riutilizzano più volte la stessa FEN; "
              "le medie dei due gruppi non vanno fuse per giudicare il modello.", "",
              markdown_table(groups["by_setting"], [("elo_setting", "Elo"),
                ("positions", "FEN"), ("predictions", "Predizioni"),
                ("stockfish_top1_agreement", "Stockfish Top-1"),
                ("stockfish_topn_coverage", f"Stockfish Top-{sf_n}"),
                ("average_top1_probability", "Prob. Top-1"),
                ("top1_at_least_99_9_percent", "Top-1 >=99,9%")]),
              "### Partite con Elo originali — confronto con il gioco umano", "",
              "Queste metriche sono calcolate **solo** dove `elo_setting=original` e "
              "si conoscono la mossa realmente giocata e il risultato della partita.", ""]
    if original:
        lines += [f"- Accuratezza Top-1 verso la mossa umana: **{percent(original['human_top1_accuracy'])}**.",
                  f"- Mossa umana nella Top-3 / Top-5 del modello: **{percent(original['human_move_top3_accuracy'])}** / **{percent(original['human_move_top5_accuracy'])}**.",
                  f"- Probabilità media assegnata alla mossa umana: **{percent(original['human_mean_probability'])}**.",
                  f"- CrossEntropy media della mossa umana: **{number(original['human_ce_capped_at_1e_12'], 4)}** "
                  "(per il log si tronca la probabilità minima a 1e-12).",
                  f"- MSE del Value rispetto all'esito della partita (0 / 0,5 / 1): **{number(original['value_mse_to_move'], 5)}**.",
                  f"- Risultato medio osservato, dalla prospettiva del giocatore al tratto: **{number(original['mean_observed_score_to_move'], 4)}**.", ""]
    else:
        lines += ["Nessuna posizione con Elo originali disponibile.", ""]
    lines += ["### Suite — sensibilità agli Elo", "",
              f"Griglia di analisi: {len([r for r in groups['by_setting'] if r['elo_setting']=='grid']) and 'attiva' or 'non presente'}. "
              "La variazione della Top-1 quando cambia l'Elo è un indicatore di sensibilità, "
              "non una prova che le scelte siano più efficaci.", "",
              "**Elo al tratto fisso, cambia l'Elo avversario:**", "",
              markdown_table(sensitivity_side, [("elo_side", "Elo al tratto"),
                 ("positions_with_two_or_more_comparisons", "FEN confrontate"),
                 ("fraction_changing_top1", "Top-1 variabile"),
                 ("average_distinct_top1", "Mosse distinte medie")], max_rows=20),
              "**Elo avversario fisso, cambia l'Elo al tratto:**", "",
              markdown_table(sensitivity_opp, [("elo_opponent", "Elo avversario"),
                 ("positions_with_two_or_more_comparisons", "FEN confrontate"),
                 ("fraction_changing_top1", "Top-1 variabile"),
                 ("average_distinct_top1", "Mosse distinte medie")], max_rows=20),
              f"Il file `grid_positions_focus_elo_{focus_elo}.csv` contiene una FEN per riga e "
              "le scelte per ciascun Elo avversario, a Elo del giocatore fissato.", "",
              "## 2. Differenze per fonte, fase di gioco e fascia Elo", "",
              "![Confronto per fonte](figures/stockfish_by_source.svg)", "",
              markdown_table(groups["by_source"], [("source", "Fonte"),
                  ("elo_setting", "Elo"), ("positions", "FEN"),
                  ("stockfish_top1_agreement", "Stockfish Top-1"),
                  ("stockfish_topn_coverage", "Stockfish Top-N"),
                  ("human_top1_accuracy", "Top-1 umana")], max_rows=25),
              "### Solo partite con Elo originali — fasi", "",
              markdown_table(groups["by_stage"], [("stage", "Fase"),
                  ("positions", "FEN"), ("human_top1_accuracy", "Top-1 umana"),
                  ("stockfish_topn_coverage", "Stockfish Top-N"),
                  ("value_mse_to_move", "Value MSE")]),
              "### Solo partite con Elo originali — fasce Elo al tratto", "",
              markdown_table(groups["by_band"], [("elo_band", "Fascia"),
                  ("positions", "FEN"), ("human_top1_accuracy", "Top-1 umana"),
                  ("stockfish_topn_coverage", "Stockfish Top-N"),
                  ("value_mse_to_move", "Value MSE")]),
              "I CSV `by_theme.csv`, `by_source_file.csv`, `by_elo_side.csv`, "
              "`by_elo_opponent.csv`, `by_side_to_move.csv` e `by_game_result.csv` "
              "forniscono le altre suddivisioni, mantenendo separata l'origine degli Elo.", "",
              "## 3. Value e affidabilità della Policy", "",
              "Il Value è l'**expected score del giocatore al tratto** (0 = sconfitta, "
              "0,5 = patta, 1 = vittoria), non la probabilità di vincere. "
              "Il CSV `value_calibration.csv` confronta la stima media con l'esito osservato "
              "in dieci intervalli di Value; l'MSE è sul risultato codificato 0/0,5/1.", "",
              "La probabilità Top-1 indica la **preferenza del modello** nella Policy, "
              "non la probabilità di vincere o l'accuratezza scacchistica. "
              "Con il solo elenco Top-N salvato non è possibile calcolare l'entropia esatta "
              "della distribuzione di tutte le mosse legali.", "",
              "## 4. Stockfish: cosa si può realmente dedurre", "",
              "La concordanza Top-1 significa che EloChess sceglie la prima mossa Stockfish; "
              "la copertura Top-N significa che la mossa EloChess è inclusa nelle varianti MultiPV. "
              "Una mossa fuori dalla MultiPV **non è automaticamente un errore**, perché "
              "manca la sua valutazione diretta.", "",
              "La colonna `cp_loss_conditional` nei CSV è riportata **solo** quando la "
              "mossa EloChess è nelle MultiPV e sia la prima mossa Stockfish sia quella "
              "selezionata hanno un punteggio numerico in cp (senza matto). "
              "Non è una perdita media in cp generale: i casi esclusi non sono casuali. "
              "Il punteggio Stockfish è dal punto di vista del Bianco; la differenza in cp "
              "viene invertita quando muove il Nero. Piccole differenze negative possono "
              "derivare anche da stime MultiPV non perfettamente coerenti.", "",
              "## 5. Posizioni selezionate per controllo manuale", "",
              f"Il file `selected_cases.csv` raccoglie fino a **{max_cases} FEN distinte**, "
              "diversificate per tipo di anomalia e origine Elo; include notazione SAN, "
              "FEN, valore firmato dal punto di vista del Bianco, Elo, mossa umana "
              "(quando presente) e migliori mosse Stockfish. "
              "Le posizioni sono selezionate per ispezione, **non** per stimare prestazioni medie.", "",
              "## 6. Limiti e interpretazione", "",
              "- Non viene eseguita alcuna nuova analisi Stockfish, neppure delle mosse fuori MultiPV.",
              "- Verificare nel metadata del benchmark il metodo di campionamento: "
              "un benchmark bilanciato per quote non rappresenta la frequenza naturale delle posizioni.",
              "- Le posizioni già utilizzate per scegliere modelli o alpha non sono "
              "un holdout completamente nuovo; le fonti vanno identificate nel metadata.",
              "- Il risultato finale di una partita non permette di attribuire causalmente "
              "il successo o fallimento a una singola mossa.",
              "- La concordanza con Stockfish e la capacità di imitare gli umani "
              "non equivalgono alla forza effettiva in una partita.",
              "- Il valore MSE non misura direttamente la probabilità di vittoria; "
              "è un errore sulla previsione dell'expected score.",
              "- Per confrontare più modelli occorrono lo stesso benchmark, "
              "gli stessi Elo, le stesse condizioni di analisi e, idealmente, "
              "confronti appaiati per posizione.", "",
              "## Provenienza dei dati", "",
              "| Misura | Origine nel progetto |", "| --- | --- |",
              "| Conteggi posizioni e origine dati | Parquet del benchmark indicato da `--benchmark` e relativo `.metadata.json` |",
              "| MultiPV, UCI, SAN e punteggi cp/matto | `stockfish.parquet` e `.metadata.json` |",
              "| Preferenze, probabilità Top-N, Value, Elo e accuratezza umana | `elochess_*/results.parquet` e `.metadata.json` |",
              "| Medie, raggruppamenti e casi | `analyze_benchmark_results.py` (questo script) |", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--stockfish", help="Parquet Stockfish; default nella cartella del benchmark")
    ap.add_argument("--model", help="Checkpoint per identificare il sottodirectory risultati, senza caricarlo")
    ap.add_argument("--results", help="Parquet EloChess esplicito, alternativa a --model")
    ap.add_argument("--output_dir", help="Destinazione report; default archive/manual_inspection/<benchmark>/<modello>")
    ap.add_argument("--focus_elo", type=int, default=1200)
    ap.add_argument("--max_cases", type=int, default=30)
    args = ap.parse_args()
    if args.max_cases < 1 or args.focus_elo <= 0:
        ap.error("--max_cases e --focus_elo devono essere positivi")
    if args.model and args.results:
        ap.error("Usa --model oppure --results, non entrambi")

    b, sf, e, sf_by_id, sf_meta, elo_meta, output, model, result_path = load_data(args)
    data = enrich(e, sf_by_id, {r["position_id"]: r for r in b})
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    original = [r for r in data if r["elo_setting"] == "original"]
    grid = [r for r in data if r["elo_setting"] == "grid"]
    groups = {
        "overall_original": [metrics(original)] if original else [],
        "overall_grid": [metrics(grid)] if grid else [],
        "by_setting": grouped(data, "elo_setting"),
        "by_source": grouped(data, "source", "elo_setting"),
        "by_source_file": grouped(data, "source", "source_file", "elo_setting"),
        "by_stage": grouped(original, "stage"),
        "by_band": grouped(original, "elo_band"),
        "by_theme": grouped(data, "source", "theme", "elo_setting"),
        "by_side": grouped(data, "side_to_move", "elo_setting"),
        "by_elo_side": grouped(data, "elo_setting", "elo_side"),
        "by_elo_opponent": grouped(data, "elo_setting", "elo_opponent"),
        "by_game_result": grouped(original, "result"),
    }
    outputs = {
        "overall_by_elo_setting.csv": groups["by_setting"],
        "by_source.csv": groups["by_source"],
        "by_source_file.csv": groups["by_source_file"],
        "by_stage.csv": groups["by_stage"],
        "by_elo_band.csv": groups["by_band"],
        "by_theme.csv": groups["by_theme"],
        "by_side_to_move.csv": groups["by_side"],
        "by_elo_side.csv": groups["by_elo_side"],
        "by_elo_opponent.csv": groups["by_elo_opponent"],
        "by_game_result.csv": groups["by_game_result"],
        "sensitivity_fixed_player_elo.csv": sensitivity(data, "elo_side"),
        "sensitivity_fixed_opponent_elo.csv": sensitivity(data, "elo_opponent"),
        "value_calibration.csv": calibration(data),
        "frequent_top1_moves.csv": top_moves(data),
        "selected_cases.csv": candidate_cases(data, args.max_cases),
        f"grid_positions_focus_elo_{args.focus_elo}.csv": detailed_grid(data, args.focus_elo),
    }
    for name, records in outputs.items():
        write_csv(output / name, records)
    source_groups = groups["by_source"]
    make_bar_svg(output / "figures/stockfish_by_source.svg", "Concordanza Stockfish per fonte",
                 source_groups, [f"{r['source']} ({r['elo_setting']})" for r in source_groups],
                 ["stockfish_top1_agreement", "stockfish_topn_coverage"],
                 ["Prima mossa Stockfish", "Nelle MultiPV Stockfish"])
    report(output / "REPORT.md", model, args.benchmark, str(result_path), sf_meta, elo_meta,
           groups, outputs["sensitivity_fixed_player_elo.csv"],
           outputs["sensitivity_fixed_opponent_elo.csv"],
           len({r["position_id"] for r in data}) < len(b), args.max_cases, args.focus_elo)
    print(f"Modello: {model}")
    print(f"Posizioni: {len({r['position_id'] for r in data})}; predizioni: {len(data)}")
    print(f"Elo originali: {len(original)}; griglia Elo: {len(grid)}")
    print(f"Report: {output / 'REPORT.md'}")
    print(f"Altri file CSV e SVG: {output}")


if __name__ == "__main__":
    main()
