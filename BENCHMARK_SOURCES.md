# EloChess — Benchmark: fonti e costruzione

## Obiettivo

`src/evaluation/build_benchmark.py` costruisce **offline e da zero** un benchmark fisso V1 di **5.000 posizioni scacchistiche**, senza usare EloChess, Stockfish o alcun file archiviato per selezionarle:

| Fonte | Posizioni | Criterio |
|---|---:|---|
| Strategic Test Suite (STS v5) | 280 | Campionamento riproducibile, bilanciato tra i temi STS |
| Chess-EPDs | 180 | Campionamento riproducibile, bilanciato fra gli otto file scelti |
| Challenging Chess Engine Test (CCET 2025) | 40 | Campionamento riproducibile sui tre gruppi originali |
| Lichess standard rated, **aprile 2013** | 4.500 | Una posizione per partita; stratificazione su fase, colore e fascia Elo |
| **Totale** | **5.000** | Deduplicazione per posizione (primi 4 campi del FEN, con en passant esplicito) |

Il seed predefinito è `42`. Per Lichess si utilizzano **18 gruppi** (apertura/mediogioco/finale × Bianco/Nero al tratto × Elo basso/medio/alto), da **250 posizioni ciascuno**, con fasce `<1200`, `1200–1799`, `>=1800`. La fase di gioco è classificata euristicamente (ply e materiale), non tramite annotazioni umane. La stratificazione bilancia il test ma **non** riproduce le frequenze reali delle posizioni nella popolazione Lichess.

I sorgenti vengono letti solo dal disco locale. File mancanti, quote non raggiungibili e output già esistenti producono un errore; il generatore non inserisce sostituzioni silenziose. Le **500 posizioni suite** conservano il metodo dell'esperimento preliminare con identici sorgenti e seed, pertanto **non costituiscono un holdout completamente inesplorato**.

## File locali utilizzati

data/raw/benchmark_sources/
├── sts/
│   └── STS1-STS15_LAN_v5.epd
├── chess_epds/
│   ├── benchmark.epd
│   ├── bratko-kopec.epd
│   ├── fortresses.epd
│   ├── kaufman.epd
│   ├── lct2.epd
│   ├── nolot.epd
│   ├── silent-but-deadly.epd
│   └── rybka-forum.epd
└── ccet/
    └── CCET.epd

In aggiunta ai file EPD, è necessario `data/raw/lichess_db_standard_rated_2013-04.pgn.zst` (157.871 partite nel dump originale). **Aprile 2013 è riservato alla valutazione e non deve essere usato per il training.**

## Provenienza e attribuzione

### STS-Rating — STS v5

- Repository: https://github.com/fsmosca/STS-Rating
- File: `epd/STS1-STS15_LAN_v5.epd`
- La suite ha 15 temi di strategia scacchistica. Il file contiene gli identificativi tematici e annotazioni relative a mosse e punteggi (ad esempio `bm`, `id`, `c0`, `c7`, `c8`, `c9`).

### Chess-EPDs

- Repository: https://github.com/ChrisWhittington/Chess-EPDs
- Otto file elencati sopra. Si tratta di test indipendenti, con annotazioni non uniformi e talvolta solo una mossa `bm`.

### CCET 2025

- Pagina ufficiale: https://sites.google.com/view/ccet2025
- File: `CCET.epd`, versione scaricata dalla sezione EPD.
- Suite di 160 posizioni con una singola mossa teoricamente corretta per posizione.

### Partite Lichess — aprile 2013

- Sorgente ufficiale: https://database.lichess.org/ (sezione *Standard rated games*); dump `lichess_db_standard_rated_2013-04.pgn.zst`.
- La posizione è quella **prima** della mossa umana osservata; vengono conservati identificativo della partita, Elo del Bianco e del Nero, giocatore al tratto, fase, mossa UCI osservata e risultato finale dal punto di vista del giocatore al tratto.
- Il risultato finale **non prova** che la mossa osservata sia buona o abbia determinato l'esito; costituisce un'etichetta osservazionale per analisi indipendenti.

## Come generare il benchmark

Dalla radice del progetto:

`python src/evaluation/build_benchmark.py --month 2013-04 --seed 42`

Il mese si imposta in `scripts/Launch_Command.py` con `MESE_BENCHMARK = "2013-04"`, indipendentemente dai mesi di training. L'esecuzione usa per default `data/raw/benchmark_sources/` e il PGN del mese selezionato; sono disponibili anche `--sources_dir`, `--pgn_zst` e `--output` per percorsi personalizzati.

File creati:

- `data/evaluation/benchmark_v1_5000.parquet` — 5.000 FEN univoci e dati originali.
- `data/evaluation/benchmark_v1_5000.metadata.json` — seed, mese, quote, conteggi, parametri di campionamento, provenienza e SHA-256 dei file EPD.

Il Parquet include i campi comuni `position_id`, `fen`, `source`, `source_file`, `source_id`, `theme`; per le suite anche `best_moves`, `original_epd`, `original_operations`; per Lichess anche `game_id`, `ply`, `stage`, `elo_band`, `side_to_move`, `white_elo`, `black_elo`, `elo_side_original`, `elo_opponent_original`, `played_move_uci`, `result` e `y_value_to_move`. I campi non applicabili a una fonte rimangono nulli. Le annotazioni EPD originali **non** sono valutazioni Stockfish uniformi.

### Valutazione successiva, distinta dalla costruzione

1. `evaluate_benchmark_stockfish.py` produce `data/evaluation/benchmark_v1_5000/stockfish.parquet` e metadata. Configurazione V1: **5 secondi**, **MultiPV 7**, **2 thread**, **256 MB hash**; valutazioni centipawn e matto dal punto di vista del **Bianco**. Non è previsto il salvataggio incrementale.
2. `evaluate_benchmark_elochess.py` usa il benchmark e il risultato Stockfish già salvato, valutando **un singolo `.keras` per esecuzione**. La modalità `hybrid` applica gli **Elo originali** alle 4.500 posizioni Lichess e una **griglia di 74 coppie Elo** alle 500 suite (con gli Elo attualmente configurati). Output in `data/evaluation/benchmark_v1_5000/elochess_<nome_modello>/`.
3. `analyze_benchmark_results.py` legge i Parquet delle due analisi e genera `REPORT.md`, CSV e SVG in `archive/manual_inspection/<benchmark>/<modello>/`, senza rieseguire Stockfish o TensorFlow.

La concordanza Top-1 / copertura MultiPV non misura la qualità delle mosse fuori MultiPV. Con l'output attuale non esiste un'analisi completa in centipedoni della mossa EloChess quando è fuori dalle varianti valutate da Stockfish.
