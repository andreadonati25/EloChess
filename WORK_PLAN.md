# WORK PLAN — Learning_AI_EloChess

# 0. Obiettivo del progetto

Costruire **EloChess**, un modello che impari dalle partite umane a suggerire mosse **efficaci al livello Elo dei giocatori**, senza limitarsi a imitare la mossa più frequente o a riprodurre Stockfish.

`P(move | position, side_to_move, player_elo, opponent_elo)`

La stessa posizione può richiedere raccomandazioni differenti al variare del livello dei due giocatori. Il risultato delle partite fornisce un segnale di efficacia, non una prova che ogni singola mossa abbia causato la vittoria.

---

# 1. Struttura

Learning_AI_EloChess/
├── README.md
├── WORK_PLAN.md
├── VARIABLES.md
├── BENCHMARK_SOURCES.md
├── V2_IDEAS.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── config/       # settings.py, utils.py, move2idx_all.json
│   ├── data/         # estrazione PGN, conversione NPZ, controlli
│   ├── model/        # build_model.py
│   ├── training/     # train_model.py
│   └── evaluation/   # builder benchmark, Stockfish, EloChess, report e Playground
├── scripts/          # Launch_Command.py
├── data/             # raw/, interim/, processed/, validation/, evaluation/
├── models/           # checkpoint e risultati delle varie fasi
└── archive/          # log, esperimenti precedenti e report

---

# 2. Costruzione del dataset

Questa fase resta stabile anche quando vengono aggiunti nuovi mesi Lichess.

## 2A. PGN → Parquet

`src/data/extract_pgn_to_parquet.py`

* Leggere le partite dai PGN compressi e scartare le partite non valide.
* Salvare per ogni mossa il FEN **prima** della mossa, la mossa UCI, gli Elo dei due giocatori, il risultato dal punto di vista del giocatore al tratto e gli identificativi della partita.
* Suddividere l'estrazione in più shard Parquet senza spezzare una partita tra shard.

Mesi usati nella V1: **luglio 2013**, **dicembre 2013**, **luglio 2014** e **maggio 2015**. Per ciascun mese generare Parquet e NPZ in cartelle distinte: `data/interim/YYYY-MM/` e `data/processed/YYYY-MM/`. I conteggi dei dump non coincidono necessariamente con le partite accettate dalla pipeline. **Aprile 2013 è riservato al benchmark**, non al training.

## 2B. Parquet → NPZ

`src/data/parquet_to_npz.py`

* `board`: `8 × 8 × 12` (piani dei pezzi).
* `state`: `71` feature (turno, arrocco, en passant e halfmove clock).
* `elo_side`, `elo_opponent`: Elo separati e normalizzati.
* `move`: indice nel vocabolario UCI globale `src/config/move2idx_all.json`.
* `legal_mask`: mosse legali calcolate con `python-chess`.
* `y_value`, `game_id`, `ply`: target e riferimenti alla partita.

Controllare gli NPZ con `src/data/validate_npz_dataset.py`. **La codifica degli input e il vocabolario devono restare compatibili fra le versioni** per poter riutilizzare i pesi.

---

# 3. Costruzione del modello

`src/model/build_model.py`

* Ramo CNN per la scacchiera; rami densi per `state` e per i due Elo.
* Fusione delle rappresentazioni, con `board_head` configurabile (`flatten` iniziale oppure `gap`).
* **Policy**: probabilità sulle mosse, limitate tramite `legal_mask`.
* **Value**: expected score del giocatore al tratto, con output nell'intervallo `[0, 1]`.
* La Policy è l'obiettivo principale; il Value è ausiliario.

La costruzione e la compilazione sono separate. I parametri configurabili sono in `src/config/settings.py` e documentati in `VARIABLES.md`.

---

# 4. Training

`src/training/train_model.py`

* Usare tutte le posizioni delle partite valide, senza bilanciamenti artificiali per Elo, apertura o fase di gioco.
* Dividere **per partita** training e validation (inizialmente 90% / 10%), conservando uno split deterministico.
* Un'epoca corrisponde a una passata sui dati di training selezionati, attraversando tutti gli shard.
* Supportare shuffle, checkpoint `best.keras` e `last.keras`, cronologia, EarlyStopping e ReduceLROnPlateau.
* Distinguere `--resume` (prosecuzione da `last.keras` della stessa fase) da `--init_model` (nuova fase dai pesi di un checkpoint, con nuovo ottimizzatore e nuova cartella).
* `train_model.py` legge **una cartella mensile per esecuzione**: i mesi vengono usati in fasi successive con `--data_dir` e `--init_model`, non mescolando shard NPZ in un'unica cartella (i `game_id` possono ripetersi fra mesi).

La Policy usa l'obiettivo configurabile:

`L_policy = (1 - alpha) * (-log(p_move)) - alpha * (2*y_value - 1) * p_move`

* `alpha = 0`: imitazione delle mosse osservate.
* `alpha = 1`: premio per mosse associate a vittorie e penalizzazione per quelle associate a sconfitte.
* Valori intermedi: transizione graduale tra i due criteri, **senza statistiche preliminari sulle mosse**.

La loss complessiva comprende anche il Value: `L = L_policy + lambda_value * L_value`.

**V1 completata — `alpha = 0.5` in tutte le fasi:**

| Fase | Dataset | Epoche configurate | Learning rate iniziale | Inizializzazione |
|---|---|---:|---:|---|
| V1.1 | `2013-07` | 2 | `0.001` | Da zero |
| V1.2 | `2013-12` | 2 | `0.0003` | V1.1 (`best.keras`) |
| V1.3 | `2013-07` | 1 | `0.0003` | V1.2 |
| V1.4 | `2013-12` | 1 | `0.0003` | V1.3 |
| V1.5 | `2014-07` | 1 | `0.0003` | V1.4 |
| V1.6 | `2015-05` | 1 | `0.0003` | V1.5 |

Le configurazioni e le cronologie reali restano in `models/v1_*/`; V1.2 include una ripresa. I checkpoint delle diverse fasi vanno confrontati sugli **stessi dati** e non in base alla sola loss registrata nei rispettivi mesi.

---

# 5. Validazione

Tre livelli di controllo, con finalità diverse.

## 5A. Metriche automatiche

Durante tutte le fasi monitorare `val_policy_accuracy`, `val_policy_top3`, `val_policy_top5`, `val_policy_unweighted_ce` e `val_value_brier` su una validation coerente.

Con `alpha = 0.5` fisso, `val_policy_loss` conserva la stessa definizione, ma **non è direttamente confrontabile quando cambia il set di validation** fra luglio e dicembre. Confrontare metriche solo a parità di dati; il best per fase non equivale al miglior modello in assoluto. Le metriche di imitazione non misurano da sole l'efficacia delle mosse.

Per un confronto fra fasi diverse è disponibile anche `src/evaluation/evaluate_validation.py`, che usa gli stessi manifest di split su quattro mesi (`2013-07`, `2013-12`, `2014-07`, `2015-05`).

## 5B. Benchmark fisso e valutazione automatica

`src/evaluation/build_benchmark.py`, `evaluate_benchmark_stockfish.py`, `evaluate_benchmark_elochess.py`, `analyze_benchmark_results.py`

* **5.000 posizioni** generate da sorgenti locali con seed 42, senza usare l'archivio: 280 STS, 180 Chess-EPDs, 40 CCET e 4.500 da **Lichess aprile 2013**, una per partita. I 4.500 campioni sono stratificati su 18 gruppi: fase di gioco, colore al tratto e fascia Elo (250 per gruppo). Questo bilanciamento **non riproduce** le frequenze naturali delle partite Lichess.
* File fisso: `data/evaluation/benchmark_v1_5000.parquet` e `benchmark_v1_5000.metadata.json`. Aprile 2013 non deve entrare nel training o nella scelta ripetuta degli iperparametri. Le 500 posizioni delle suite sono già state esplorate nel test preliminare: non sono un holdout completamente nuovo.
* **Stockfish**: una valutazione per posizione, indipendente dal checkpoint, con `--time 5 --top_n 7 --threads 2 --hash_mb 256` nella configurazione V1. Output `data/evaluation/benchmark_v1_5000/stockfish.parquet` e `.metadata.json`. I centipedoni sono riferiti al **Bianco**.
* **EloChess**: un solo modello `.keras` per esecuzione. Modalità `hybrid`: Elo **originali** per le 4.500 posizioni Lichess, griglia controllata per le 500 suite (74 coppie con gli Elo attualmente configurati). Output separato per modello in `data/evaluation/benchmark_v1_5000/elochess_<nome_modello>/`.
* **Analisi**: lo script legge soltanto Parquet e metadata, senza rieseguire motori o training. Produce `REPORT.md`, CSV e SVG in `archive/manual_inspection/<benchmark>/<modello>/` e distingue le metriche umane dalle analisi della griglia.
* **Limite fondamentale:** concordanza Top-1 e copertura MultiPV Stockfish non misurano direttamente la bontà di tutte le altre mosse. Una differenza in centipedoni è calcolabile con i dati attuali solo per mosse valutate dalla MultiPV e con punteggio numerico, quindi il relativo campione è selezionato e **non rappresentativo di tutte le predizioni**. Il Value è expected score, non una valutazione in centipedoni.

## 5C. Valutazione manuale

`src/evaluation/elochess_playground.py` (con componente HTML della scacchiera)

* Giocare spostando i pezzi sulla scacchiera, caricare FEN, invertire la visuale e cambiare gli Elo dei due giocatori.
* Consultare tutte le mosse legali ordinate per probabilità Policy e visualizzare il Value dal punto di vista del Bianco, da −1 a +1; le probabilità sono **preferenze del modello**, non probabilità di vittoria.
* Esaminare continuazioni, posizioni libere e Manual Benchmark, confrontando EloChess con l’analisi indipendente di Stockfish (motore locale opzionale).

