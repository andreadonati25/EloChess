#!/usr/bin/env python3
"""
Launch_Command.py

Esegue in ordine una coda di comandi, pensata per lanciare in automatico
operazioni lunghe (anche di notte).

Come si usa:
    1. nella lista `cmd` qui sotto lascia decommentati solo i comandi che
       vuoi eseguire, nell'ordine in cui devono partire;
    2. controlla la coda con:   python scripts/Launch_Command.py --dry_run
    3. avviala con:             python scripts/Launch_Command.py

I comandi vengono eseguiti dalla radice del progetto, qualunque sia la cartella
da cui si lancia questo script, quindi i percorsi relativi (src/..., data/...)
sono sempre quelli della struttura del progetto.

Ogni esecuzione crea un proprio log in archive/, con data e ora di avvio nel
nome (es. Command_Log_2026-10-07_23-30-00.txt), quindi i log non si
sovrascrivono e non serve rinominarli. Per ogni comando vengono registrati ora
di inizio, durata ed esito. Se un comando fallisce la coda si ferma, per non
sprecare ore su passi che dipendono da quello fallito; con --keep_going prosegue
con i comandi successivi.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "archive/command_logs"
LOG_FILE: Path | None = None  # impostato in main() a ogni esecuzione

cmd: list[str] = []

# COSTRUZIONE FILE
DATE = "2015-05"
NUMBER = "0000"
CHECK_N = 50000

# --- Vocabolario globale delle mosse -> src/config/move2idx_all.json --------
#cmd.append("python src/data/all_uci_moves.py")
# --- PGN.ZST -> Parquet -----------------------------------------------------
#cmd.append(f"python src/data/extract_pgn_to_parquet.py --pgn_zst data/raw/lichess_db_standard_rated_{DATE}.pgn.zst --out data/interim/{DATE}")
# --- Parquet -> NPZ (una cartella intera) -----------------------------------
#cmd.append(f"python src/data/parquet_to_npz.py --parquet_dir data/interim/{DATE} --out data/processed/{DATE}")
# --- Validazione di uno shard NPZ -------------------------------------------
#cmd.append(f"python src/data/validate_npz_dataset.py data/processed/{DATE}/lichess_db_standard_rated_{DATE}_part_{NUMBER}.npz --check_n {CHECK_N}")

# TRAINING
TRAIN_MONTH_1 = "2013-07"
TRAIN_MONTH_2 = "2013-12"
TRAIN_MONTH_3 = "2014-07"
TRAIN_MONTH_4 = "2015-05"
TRAIN_ALPHA = 0.5
TRAIN_BATCH_SIZE = 32
TRAIN_LR_INITIAL = 0.001
TRAIN_LR_FINETUNE = 0.0003
TRAIN_SAVE_EVERY = 5000
DROPOUT = 0.20
LAMBDA_VALUE = 0.5
BOARD_HEAD = "flatten"

# V1.1 — Luglio 2013, da zero
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_1} --out_dir models/v1_1 --split_file data/validation/{TRAIN_MONTH_1}_game_split_seed42.json --epochs 2 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --learning_rate {TRAIN_LR_INITIAL} --dropout {DROPOUT} --board_head {BOARD_HEAD} --lambda_value {LAMBDA_VALUE} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')
# V1.2 — Dicembre 2013, dal best V1.1
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_2} --out_dir models/v1_2 --split_file data/validation/{TRAIN_MONTH_2}_game_split_seed42.json --init_model models/v1_1/best.keras --epochs 2 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --learning_rate {TRAIN_LR_FINETUNE} --lambda_value {LAMBDA_VALUE} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_2} --out_dir models/v1_2 --split_file data/validation/{TRAIN_MONTH_2}_game_split_seed42.json --resume --epochs 1 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')
# V1.3 — Ritorno a luglio 2013
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_1} --out_dir models/v1_3 --split_file data/validation/{TRAIN_MONTH_1}_game_split_seed42.json --init_model models/v1_2/best.keras --epochs 1 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --learning_rate {TRAIN_LR_FINETUNE} --lambda_value {LAMBDA_VALUE} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')
# V1.4 — Ritorno a dicembre 2013
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_2} --out_dir models/v1_4 --split_file data/validation/{TRAIN_MONTH_2}_game_split_seed42.json --init_model models/v1_3/best.keras --epochs 1 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --learning_rate {TRAIN_LR_FINETUNE} --lambda_value {LAMBDA_VALUE} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')
# V1.5 — Luglio 2014, dal best V1.4
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_3} --out_dir models/v1_5 --split_file data/validation/{TRAIN_MONTH_3}_game_split_seed42.json --init_model models/v1_4/best.keras --epochs 1 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --learning_rate {TRAIN_LR_FINETUNE} --lambda_value {LAMBDA_VALUE} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')
# V1.6 — Maggio 2015, dal best V1.5
#cmd.append(f'python src/training/train_model.py --data_dir data/processed/{TRAIN_MONTH_4} --out_dir models/v1_6 --split_file data/validation/{TRAIN_MONTH_4}_game_split_seed42.json --init_model models/v1_5/best.keras --epochs 1 --batch_size {TRAIN_BATCH_SIZE} --alpha {TRAIN_ALPHA} --learning_rate {TRAIN_LR_FINETUNE} --lambda_value {LAMBDA_VALUE} --monitor val_policy_loss --save_every_steps {TRAIN_SAVE_EVERY}')

# VALIDATION
VALIDATION_MODELS = "models/v1_1/best.keras models/v1_2/best.keras models/v1_3/best.keras models/v1_4/best.keras models/v1_5/best.keras models/v1_6/best.keras"
VALIDATION_MONTHS = f"{TRAIN_MONTH_1} {TRAIN_MONTH_2} {TRAIN_MONTH_3} {TRAIN_MONTH_4}"
VALIDATION_SAMPLES = 100000
# --- Valutazione incrociata dei checkpoint sulle validation -----------------
#cmd.append(f'python src/evaluation/evaluate_validation.py --models {VALIDATION_MODELS} --months {VALIDATION_MONTHS} --alpha {TRAIN_ALPHA} --lambda_value {LAMBDA_VALUE} --batch_size {TRAIN_BATCH_SIZE} --sample_positions {VALIDATION_SAMPLES} --seed 42')

# BENCHMARK
MESE_BENCHMARK = "2013-04"
SEED = 42
BENCHMARK = "data/evaluation/benchmark_v1_5000.parquet"
STOCKFISH = "models/stockfish/stockfish-windows-x86-64-universal.exe"
SF_TIME = 5
TOP_N = 7
SF_THREADS = 2
SF_HASH_MB = 256
MODEL = "models/v1_6/best.keras"
ELO_MODE = "hybrid"
#ELO_MODE = "grid"
ELOS = "500 800 1000 1200 1400 1600 1800 2000 2200 2400"
EVAL_BATCH_SIZE = 256
POSITIONS_PER_CHUNK = 50
FOCUS_ELO = 1200
MAX_CASES = 30

# --- Costruzione benchmark --------------------------------------------------
#cmd.append(f"python src/evaluation/build_benchmark.py --month {MESE_BENCHMARK} --seed {SEED}")
# --- Analisi Stockfish benchmark --------------------------------------------
#cmd.append(f'python src/evaluation/evaluate_benchmark_stockfish.py --benchmark "{BENCHMARK}" --engine "{STOCKFISH}" --time {SF_TIME} --top_n {TOP_N} --threads {SF_THREADS} --hash_mb {SF_HASH_MB}')
# --- Analisi EloChess benchmark ---------------------------------------------
#cmd.append(f'python src/evaluation/evaluate_benchmark_elochess.py --benchmark "{BENCHMARK}" --model "{MODEL}" --elo_mode {ELO_MODE} --elos {ELOS} --top_n {TOP_N} --batch_size {EVAL_BATCH_SIZE} --positions_per_chunk {POSITIONS_PER_CHUNK}')
# --- Confronto analisi benchmark --------------------------------------------
#cmd.append(f'python src/evaluation/analyze_benchmark_results.py --benchmark "{BENCHMARK}" --model "{MODEL}" --focus_elo {FOCUS_ELO} --max_cases {MAX_CASES}')

# PLAYGROUND
#cmd.append("python -m streamlit run src/evaluation/elochess_playground.py")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def log(text: str) -> None:
    """Scrive su schermo e nel file di log di questa esecuzione."""
    print(text, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{text}\n")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Esegue in ordine i comandi attivi nella lista `cmd`."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Mostra la coda senza eseguire nulla.",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="Se un comando fallisce, prosegue con i successivi.",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if not cmd:
        raise SystemExit("Nessun comando attivo: decommenta le righe cmd.append(...) che vuoi eseguire.")
    total = len(cmd)
    if args.dry_run:
        print(f"Coda ({total} comandi), cartella di lavoro: {ROOT}")
        for i, command in enumerate(cmd, start=1):
            print(f"  [{i}/{total}] {command}")
        return
    global LOG_FILE
    started = datetime.now()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f"Command_Log_{started:%Y-%m-%d_%H-%M-%S}.txt"
    failed: list[int] = []
    for i, command in enumerate(cmd, start=1):
        log(f"Time: {now()}, [Batch {i}/{total}] Running: {command}")
        command_start = datetime.now()
        try:
            result = subprocess.run(command, shell=True, cwd=ROOT)
        except KeyboardInterrupt:
            log(f"Time: {now()}, [Batch {i}/{total}] Interrotto dall'utente\n")
            raise SystemExit(130)
        duration = timedelta(seconds=round((datetime.now() - command_start).total_seconds()))
        status = "OK" if result.returncode == 0 else f"FALLITO (codice {result.returncode})"
        log(f"Time: {now()}, [Batch {i}/{total}] {status}, durata {duration}\n")
        if result.returncode != 0:
            failed.append(i)
            if not args.keep_going:
                break
    elapsed = timedelta(seconds=round((datetime.now() - started).total_seconds()))
    if failed:
        log(f"Coda terminata con errori nei comandi {failed}, durata totale {elapsed}\n")
        raise SystemExit(1)
    log(f"Coda completata, durata totale {elapsed}\n")


if __name__ == "__main__":
    main()
