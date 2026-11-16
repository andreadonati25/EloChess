# Learning_AI_EloChess

## Italiano

**EloChess** nasce da una domanda: *quale mossa può essere efficace per un giocatore umano contro un avversario di un determinato livello Elo?*

La mossa migliore secondo un motore non è necessariamente quella più utile in una partita fra esseri umani. Allo stesso tempo, una mossa molto giocata non è automaticamente una buona scelta. EloChess prova a tenere insieme questi due aspetti: **come giocano le persone** e **quali risultati ottengono**, considerando l'Elo di entrambi i giocatori.

Il modello apprende da partite pubbliche di [Lichess](https://lichess.org) e, data una posizione, produce:

- **Policy:** una distribuzione di probabilità sulle mosse legali, influenzata anche dall'Elo dei giocatori.
- **Value:** una stima del risultato atteso dal punto di vista del giocatore al tratto.

L'obiettivo non è imitare Stockfish né sostenere che una mossa vincente in una partita sia stata la causa della vittoria. I risultati delle partite sono un **segnale statistico**, non una dimostrazione dell'efficacia di ogni singola scelta.

### Stato del progetto

La prima versione (**V1.1–V1.6**) è stata addestrata in sei fasi su partite Lichess del 2013, 2014 e 2015 e sottoposta a valutazioni automatiche. Il **Playground** permette di esplorare le mosse suggerite, variare gli Elo e confrontare l'analisi con Stockfish. L'analisi manuale servirà a individuare i punti di forza, i limiti e le idee per la V2.

Il codice e la documentazione sono pensati per essere condivisi; dataset, checkpoint e file di terzi non sono inclusi automaticamente nel repository.

Documenti: [`WORK_PLAN.md`](WORK_PLAN.md) (struttura e percorso del progetto), [`VARIABLES.md`](VARIABLES.md) (parametri), [`BENCHMARK_SOURCES.md`](BENCHMARK_SOURCES.md) (benchmark e fonti).

## English

**EloChess** starts with a question: *which move might work well for a human player against an opponent of a given Elo rating?*

The objectively best engine move is not always the most practical choice in human play. Conversely, a frequently played move is not necessarily a good one. EloChess explores both **how people play** and **what results they achieve**, taking both players' Elo ratings into account.

The model learns from public [Lichess](https://lichess.org) games and, given a position, produces:

- **Policy:** a probability distribution over legal moves, conditioned also on both players' ratings.
- **Value:** an estimate of the expected game score from the side-to-move's perspective.

The goal is not to imitate Stockfish or to claim that a move played in a won game caused the win. Game outcomes provide a **statistical signal**, not causal proof about individual moves.

### Project status

The first version (**V1.1–V1.6**) was trained in six stages using Lichess games from 2013, 2014 and 2015 and evaluated with automatic tests. The **Playground** allows users to inspect suggested moves, change Elo ratings and compare analyses with Stockfish. Manual inspection will inform the strengths, limitations and possible directions for V2.

The repository is intended to share code and documentation; datasets, checkpoints and third-party files are not included automatically.

Documents: [`WORK_PLAN.md`](WORK_PLAN.md) (project structure and development), [`VARIABLES.md`](VARIABLES.md) (parameters), [`BENCHMARK_SOURCES.md`](BENCHMARK_SOURCES.md) (benchmark sources).

---

Il progetto è sviluppato dall'autore del repository con assistenza di modelli linguistici nella scrittura, revisione e organizzazione di codice e documentazione. I dati di partenza provengono dai dump pubblici Lichess.

The repository author develops the project with assistance from language models for code and documentation. Source games come from public Lichess dumps.
