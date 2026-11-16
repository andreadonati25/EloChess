# VARIABLES.md — EloChess

Parametri **regolabili tra sessioni di addestramento**. I default sono in `src/config/settings.py`; un argomento passato a `train_model.py` prevale sul default. La **V1** è stata addestrata dalla V1.1 alla V1.6 con `alpha = 0.5` fisso, usando i comandi di `scripts/Launch_Command.py`.

## 1. Alpha: frequenza → efficacia

- `--alpha = 0.5` (**scelta e default attuale della V1**; dominio `[0, 1]`). Interpola **due termini della loss**, non due pesi fissi assegnati ai campioni.
- `--policy_loss = outcome_blend`: per ogni posizione, indicando con `p` la probabilità assegnata alla mossa realmente giocata, `y` il risultato dal punto di vista del giocatore al tratto e `r = 2y - 1`:

  `L_policy = (1 - alpha) * [-ln(p)] - alpha * r * p`

- `alpha = 0`: pura imitazione della mossa osservata (CrossEntropy); il risultato non entra nella Policy.
- `alpha = 1`: soltanto premio/penalità empirica `-r*p`; una vittoria spinge a **incrementare** `p`, una sconfitta a **ridurla**, una patta non dà gradiente Policy. Non significa stimare esattamente una percentuale di vittoria, né isolare causalmente l'effetto della mossa.
- `0 < alpha < 1`: transizione graduale. La componente di efficacia è **limitata tra −1 e +1** per esempio, anche quando penalizza una sconfitta. La CrossEntropy della parte imitativa è non negativa, ma non limitata superiormente. Non servono statistiche aggregate preliminari.

### Come leggere valori intermedi (esempio numerico)

Se il modello assegna **p = 0.20** alla mossa realmente giocata, la CrossEntropy vale `−ln(0.2) ≈ 1.609`. Nella tabella le tre colonne sono il contributo **alla loss da minimizzare** dello stesso esempio a seconda del risultato, non moltiplicatori indipendenti da `p`.

| alpha | Frazione CE `(1-alpha)` | Frazione premio/penalità `alpha` | Vittoria `L` | Patta `L` | Sconfitta `L` |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 100% | 0% | 1.609 | 1.609 | 1.609 |
| 0.2 | 80% | 20% | 1.247 | 1.288 | 1.328 |
| 0.3 | 70% | 30% | 1.066 | 1.127 | 1.187 |
| 0.5 | 50% | 50% | 0.705 | 0.805 | 0.905 |
| 0.7 | 30% | 70% | 0.343 | 0.483 | 0.623 |
| 0.9 | 10% | 90% | −0.019 | 0.161 | 0.341 |
| 1.0 | 0% | 100% | −0.200 | 0 | +0.200 |

**Attenzione:** 70%/30% o 50%/50% nella tabella indicano i *coefficienti della formula*, non la quota del gradiente effettivamente attribuibile all'imitazione o al risultato. I termini `−ln(p)` e `p` hanno scale e derivate diverse. `alpha = 0.5` è una decisione progettuale per la V1, non una stima ottimizzata della forza scacchistica.

Per un secondo esempio, se `p=0.8`, la CE è `0.223`: con `alpha=0.5` una vittoria vale `0.5*0.223 - 0.5*0.8 = −0.288`, una sconfitta `+0.512`. L'effetto del risultato è più forte perché il modello attribuisce già alta probabilità alla mossa.

**Lettura orientativa**: `0.0–0.2` enfatizza l'imitazione; `0.3–0.6` mantiene entrambi i termini; `0.8–1.0` riduce molto il contributo della CrossEntropy e può produrre Policy eccessivamente concentrate. Sono descrizioni qualitative, non percentuali quantitative di forza o di comportamento.

## 2. Altri parametri di apprendimento

- `--learning_rate`: `0.001` nella V1.1 e `0.0003` dalla V1.2 alla V1.6 (configurazione `Launch_Command.py`; default `0.001`). Il learning rate può anche essere ridotto dai callback durante una fase.
- `--batch_size = 32`: posizioni per aggiornamento (ultimo batch di uno shard anche più piccolo); modifica memoria, rumore e numero di step.
- `--epochs`: `2` nelle V1.1/V1.2, `1` dalla V1.3 alla V1.6 (default `1`). Sono passate **della singola esecuzione** sul relativo mese.
- `--dropout = 0.2`: regolarizzazione della rete, applicata solo in training; valore incorporato nel checkpoint: `--init_model` non lo cambia automaticamente.
- `--lambda_value = 0.5`: contributo Value alla loss totale `L_policy + lambda_value*L_value`; il Value rimane supervisionato da risultato `0 / 0.5 / 1`.
- `--value_loss = binary_crossentropy`: alternativa `mean_squared_error`.
- `--optimizer = adam`: alternativa `sgd`; `--init_model` ricrea l'ottimizzatore.
- `--policy_loss = outcome_blend`: nuova Policy con target `[move_index, y_value]`; per ora unica loss Policy supportata dagli script aggiornati.

## 3. Dataset e riproducibilità

- `--data_dir = data/processed/2013-07`: shard NPZ di input; cambiare dataset mantenendo schema, vocabolario e codifica.
- `--validation_fraction = 0.1`: frazione di **partite** in validation, non di righe.
- `--split_seed = 42`: scelta deterministica delle partite assegnate alla validation; `--split_file = None` usa un percorso automatico. La V1 passa esplicitamente `data/validation/YYYY-MM_game_split_seed42.json`: un manifest **distinto per mese**, da conservare.
- `--seed = 42`: inizializzazione e shuffle gestiti dallo script, non garanzia di determinismo identico fra hardware.
- `--max_files = None`: tutti gli shard; usare 1–3 soltanto negli smoke test.
- `--vocab = src/config/move2idx_all.json`: ordine e indici delle mosse **devono restare compatibili con i checkpoint**.
- `--board_head = flatten`: alternativa `gap` quando si costruisce da zero; cambia l'architettura, non riconfigura un checkpoint esistente.
- `--out_dir`: `models/v1_1` … `models/v1_6` per le sei fasi (`models/v1` è solo il default). Ogni fase usa una cartella nuova per checkpoint, metriche e configurazione.
- `--resume = false`: riparte da `last.keras` con stato ottimizzatore, per **lo stesso obiettivo e split**; non è il comando per cambiare alpha.
- `--init_model = None`: se specificato con un **percorso reale** `.keras`, carica i pesi, riparte con nuovo ottimizzatore/loss e nuova cartella. Per V1.1 va **omesso del tutto**; dalla V1.2 punta a `models/v1_1/best.keras`, e così via. Non si può combinare con `--resume`.

## 4. Monitor, callback e checkpoint

- `--monitor = val_policy_loss`: definisce `best.keras` secondo la loss Policy sulla **validation del mese corrente**. Alpha rimane `0.5` nella V1, ma **le validation di luglio e dicembre sono diverse**: il best di una fase non permette da solo di eleggere un vincitore fra fasi. Se cambia alpha, anche la funzione obiettivo cambia e la loss diventa ancor meno confrontabile.
- `val_policy_unweighted_ce`: CE della mossa giocata, diagnostica comparabile a parità di validation set; non misura da sola la capacità di vincere.
- `--patience = 6`: EarlyStopping dopo epoche senza miglioramento.
- `--reduce_lr_patience = 2`, `--reduce_lr_factor = 0.5`, `--min_learning_rate = 1e-6`: riduzione learning rate in plateau.
- `--save_every_steps = 5000`: checkpoint periodico, ma un'interruzione a metà epoch non riprende esattamente dal batch successivo.

## 5. V1: fasi eseguite e valutazione

La V1 è stata addestrata in sei fasi sequenziali, mantenendo `alpha = 0.5`. Le fasi successive alla V1.1 partono dal checkpoint `best.keras` precedente; V1.2 comprende anche una ripresa del training registrata nella sua cronologia.

| Fase | Dataset | Epoche configurate | LR iniziale | Partenza |
|---|---|---:|---:|---|
| V1.1 | `2013-07` | 2 | `0.001` | Da zero |
| V1.2 | `2013-12` | 2 | `0.0003` | V1.1 |
| V1.3 | `2013-07` | 1 | `0.0003` | V1.2 |
| V1.4 | `2013-12` | 1 | `0.0003` | V1.3 |
| V1.5 | `2014-07` | 1 | `0.0003` | V1.4 |
| V1.6 | `2015-05` | 1 | `0.0003` | V1.5 |

I dettagli effettivi di ogni esecuzione sono conservati localmente in `models/v1_*/config.json`, `history.json` e `best_metric.json` e nei log; il numero di epoche della tabella descrive i comandi configurati, non eventuali ripartenze.

### Valutazione

- **Validation incrociata:** `evaluate_validation.py` confronta checkpoint sulle stesse posizioni per mese (`2013-07`, `2013-12`, `2014-07`, `2015-05`). Le metriche tra mesi diversi non sono direttamente intercambiabili.
- **Benchmark fisso:** 5.000 posizioni (500 suite + 4.500 da Lichess `2013-04`), valutate separatamente da Stockfish ed EloChess; `analyze_benchmark_results.py` produce i report. La concordanza con Stockfish non misura, da sola, l'efficacia contro esseri umani.
- **Analisi manuale:** il Playground consente di variare gli Elo, esplorare Policy, Ricerca e Value, consultare Stockfish e provare posizioni libere o del Manual Benchmark.
