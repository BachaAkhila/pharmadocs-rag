# PharmaDocs RAG

A production-style Retrieval-Augmented Generation (RAG) pipeline for biomedical and pharmaceutical research document search, with the full MLOps stack wrapped around it: hybrid retrieval, a self-correcting LangGraph agent, RAGAS-style evaluation with regression gates, MLflow experiment tracking, drift monitoring, and a FastAPI serving layer.

This project is a scaled-down, fully runnable demonstration of the architecture patterns used in production GenAI systems -- document ingestion through to monitored deployment -- built to run end-to-end with **zero API keys and zero external services**.

---

## Why this project exists

Most "RAG demo" repos stop at "ask a question, get an answer." This one focuses on everything *around* that core loop that production systems actually need:

- How do you know your retrieval is any good? (`app/evaluation` -- RAGAS-style metrics)
- How do you know a prompt or chunking change made things *better*, not worse? (regression gates + MLflow)
- What happens when the model isn't confident in its answer? (LangGraph self-correcting agent)
- How do you know when the system is degrading in production? (drift detection with PSI / KS-test)

Every architectural decision here mirrors a real production RAG system: hybrid dense + sparse retrieval, confidence-gated multi-step agents, automated evaluation gates, and continuous monitoring.

---

## Architecture

```
data/raw/*.json
      |
      v
+----------------+     +-------------------+     +-----------------------+
|   Ingestion     | --> |    Embedding       | --> |    Vector Store        |
|  (chunking.py)  |     | (embeddings.py)    |     |     (FAISS)            |
+----------------+     +-------------------+     +-----------------------+
                                                              |
                                                              v
+----------------+     +-------------------+     +-----------------------+
|  BM25 Index     | --> | Hybrid Retriever   | <-- |  Reciprocal Rank       |
|  (sparse)       |     | (vectorstore.py)   |     |  Fusion                |
+----------------+     +-------------------+     +-----------------------+
                                  |
                                  v
                          +----------------+      +-----------------------+
                          |   RAG Chain     | <--- |   LLM Client            |
                          | (rag_chain.py)  |      | (Mock / OpenAI GPT-4)   |
                          +----------------+      +-----------------------+
                                  |
                                  v
                          +-----------------------+
                          |  LangGraph Agent        |
                          |  retrieve -> check       |
                          |  confidence ->            |
                          |  reformulate -> retry      |
                          +-----------------------+
                                  |
                  +---------------+----------------+
                  v                                  v
        +--------------------+              +-----------------------+
        |   FastAPI Service    |              |  RAGAS Evaluation       |
        |  /query              |              |  + Regression Gate      |
        |  /agent/query         |              |  (MLflow tracked)        |
        +--------------------+              +-----------------------+
                                                      |
                                                      v
                                            +-----------------------+
                                            |  Drift Monitoring       |
                                            |  (PSI + KS-test)          |
                                            |  -> Airflow DAG           |
                                            +-----------------------+
```

---

## Project structure

```
pharmadocs-rag/
├── app/
│   ├── ingestion/        document loading + chunking (semantic & fixed-size)
│   ├── retrieval/         embeddings, FAISS vector store, BM25, hybrid retrieval
│   ├── agents/             RAG chain, LangGraph self-correcting agent, LLM clients
│   ├── evaluation/         RAGAS-style metrics + regression gates
│   ├── monitoring/         MLflow tracking + drift detection (PSI/KS-test)
│   ├── api/                 FastAPI app
│   └── pipeline.py          single factory wiring everything together
├── dags/
│   └── drift_monitoring_dag.py   Airflow DAG (documents orchestration design)
├── data/
│   ├── raw/                  sample pharma/biotech abstracts (8 docs)
│   ├── eval/                  golden evaluation set + baseline scores
│   ├── processed/              generated chunks (gitignored)
│   └── monitoring/              query logs for drift detection (gitignored)
├── scripts/
│   ├── run_evaluation.py         CLI: evaluate + log to MLflow + regression gate
│   └── compare_chunking_strategies.py  A/B: semantic vs fixed-size chunking
├── tests/                          36 tests covering every module
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml         test -> evaluate -> docker build -> smoke test
```

---

## Quickstart

### 1. Local (Python)

```bash
git clone <your-repo-url>
cd pharmadocs-rag
pip install -r requirements-dev.txt

# Run the test suite (36 tests)
PYTHONPATH=. pytest -v

# Run evaluation and create the baseline
PYTHONPATH=. python scripts/run_evaluation.py --update-baseline

# Start the API
PYTHONPATH=. uvicorn app.api.main:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive API docs.

### 2. Docker

```bash
docker compose up --build
```

This starts:
- the FastAPI service on `http://localhost:8000`
- an MLflow UI on `http://localhost:5000` for browsing evaluation runs

---

## API examples

**Single-step RAG query:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What MAC reduction was achieved by the pruning framework on ResNet-56?"}'
```

```json
{
  "query": "What MAC reduction was achieved by the pruning framework on ResNet-56?",
  "answer": "This paper proposes a gradient-free pruning framework ... achieves a 5.1x reduction in multiply-accumulate operations, from 125.75 to 24.86 MMACs ...\n\n(Source: Structured Pruning of Convolutional Neural Networks for Edge Deployment)",
  "confidence": 0.5464,
  "model_name": "mock-extractive-v1",
  "latency_ms": 2.31,
  "retrieved_chunks": [ "..." ]
}
```

**Multi-step agent query (with confidence-gated retry):**

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about quantum computing in pharma manufacturing"}'
```

```json
{
  "query": "Tell me about quantum computing in pharma manufacturing",
  "answer": "Inventory optimization in biopharmaceutical manufacturing requires forecasting models ... (Source: Time-Series Forecasting Approaches for Inventory Optimization in Biopharmaceutical Manufacturing)",
  "confidence": 0.1841,
  "attempts": 3,
  "low_confidence": true
}
```

Note how the agent correctly flags `low_confidence: true` and reports it tried 3 times -- the query has no good match in the knowledge base, and the system says so rather than hallucinating confidently.

---

## Component deep-dive

### 1. Ingestion & chunking (`app/ingestion/chunking.py`)

Two chunking strategies are implemented and compared head-to-head:

- **Semantic chunking** -- groups sentences up to a target size, never splitting mid-sentence, with configurable sentence-level overlap for cross-chunk context.
- **Fixed-size chunking** -- naive character-window splitting with overlap (the common baseline).

Run the comparison:

```bash
PYTHONPATH=. python scripts/compare_chunking_strategies.py
```

Sample output on the included 8-document corpus:

```
  faithfulness       semantic=0.9315  fixed=0.9203  delta=+0.0112  (semantic)
  answer_relevancy   semantic=0.5918  fixed=0.5759  delta=+0.0159  (semantic)
  context_recall     semantic=0.9934  fixed=0.9934  delta=+0.0000  (tie)
  overall            semantic=0.8389  fixed=0.8299  delta=+0.0090  (semantic)
```

### 2. Embeddings & retrieval (`app/retrieval/`)

- **`embeddings.py`** -- pluggable `EmbeddingModel` interface. Default is a dependency-free hashed TF-IDF embedding (`TfidfEmbedding`), so the project runs without downloading model weights. A `SentenceTransformerEmbedding` (e.g. `BAAI/bge-small-en-v1.5`) implementation is included for production use -- swapping it in is a one-line change in `app/pipeline.py`.

- **`vectorstore.py`** -- FAISS `IndexFlatIP` for dense cosine-similarity search, a `BM25Okapi` index for sparse lexical search, and a `HybridRetriever` that merges both via **Reciprocal Rank Fusion (RRF)**. RRF is robust to the very different score scales of cosine similarity vs. BM25 scores.

### 3. RAG chain & agent (`app/agents/`)

- **`rag_chain.py`** -- single retrieve-then-generate step. Computes a **confidence score** from the top retrieval score and query/context lexical overlap.

- **`agent_graph.py`** -- a `LangGraphAgent` built on `langgraph.graph.StateGraph` with two nodes and a conditional edge:

  ```
  retrieve_and_generate --> [confidence >= threshold?] --> END
                          \-> [confidence < threshold] -> reformulate -> retrieve_and_generate
  ```

  If confidence is below threshold, the query is reformulated (stopword stripping / keyword extraction) and retried, up to `max_attempts`. This is the "self-correcting agent loop with reflection and query decomposition fallback" pattern.

- **`llm_client.py`** -- `MockLLMClient` (default, extractive, zero-dependency) and `OpenAIClient` (GPT-4, documents the production swap-in path).

### 4. Evaluation (`app/evaluation/ragas_eval.py`)

Implements three RAGAS-style metrics as **lexical-overlap heuristics** (no LLM judge required, so they run for free and deterministically):

| Metric | What it measures | Heuristic |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | Fraction of answer tokens present in retrieved context |
| **Answer Relevancy** | Does the answer address the query? | Token overlap between query/answer and reference answer |
| **Context Recall** | Does retrieved context contain what's needed? | Fraction of reference-answer tokens present in retrieved context |

Each function's docstring documents the exact `ragas` package metric it stands in for, so swapping to LLM-judged scoring in production is a contained change.

**`RegressionGate`** stores a baseline (`data/eval/baseline_scores.json`) and fails (exit code 1) if any metric drops more than `tolerance` (default 0.05) below baseline -- this is the CI/CD quality gate.

### 5. MLflow tracking (`app/monitoring/experiment_tracking.py`)

Every evaluation run is logged to MLflow (SQLite backend by default) with:
- **params**: chunking strategy, top_k, embedding model, LLM backend
- **metrics**: faithfulness, answer_relevancy, context_recall, overall, mean latency
- **artifacts**: full per-query scores as JSON

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### 6. Drift detection (`app/monitoring/drift_detection.py`)

Computes **Population Stability Index (PSI)** and the **Kolmogorov-Smirnov test** between a reference window and current window for two signals: retrieval confidence and response latency. A signal is flagged as drifted if `PSI >= 0.25` or `KS p-value < 0.05`.

```python
from app.monitoring.drift_detection import check_pipeline_drift

reports = check_pipeline_drift(reference_confidences, current_confidences,
                                  reference_latencies, current_latencies)
for r in reports:
    print(r.signal_name, r.drifted, r.psi)
```

### 7. Airflow DAG (`dags/drift_monitoring_dag.py`)

Documents the orchestration design for a daily drift-monitoring DAG:

```
collect_query_logs -> check_drift -> [drift detected]  -> run_regression_suite
                                    -> [no drift]        -> skip
```

If drift is detected, the RAGAS regression suite is re-run (`scripts/run_evaluation.py`); a failed regression gate fails the Airflow task and would trigger alerting (PagerDuty/Slack) in production. This file requires an Airflow installation to execute as a live DAG -- it is included to document the orchestration structure, gated behind an `AIRFLOW_AVAILABLE` import check so the rest of the project runs without Airflow installed.

### 8. FastAPI service (`app/api/main.py`)

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /query` | Single-step RAG query |
| `POST /agent/query` | Multi-step self-correcting agent query |
| `GET /metrics/latest` | Latest evaluation baseline scores |

The pipeline (index + retriever + agent) is built **once at startup** via FastAPI's `lifespan` context manager and shared across all requests.

---

## Running the tests

```bash
PYTHONPATH=. pytest -v
```

36 tests across chunking, embeddings/retrieval, the RAG chain and agent (including retry behavior), evaluation metrics and the regression gate, drift detection, and the API endpoints.

---

## Swapping in production components

This project is designed so every "mock" component has a documented, one-line swap to its production equivalent:

| Component | Default (this repo) | Production swap |
|---|---|---|
| LLM | `MockLLMClient` (extractive) | `OpenAIClient` (GPT-4) -- in `app/agents/llm_client.py` |
| Embeddings | `TfidfEmbedding` (hashed TF-IDF) | `SentenceTransformerEmbedding` (`BAAI/bge-small-en-v1.5`) -- in `app/retrieval/embeddings.py` |
| RAGAS metrics | Lexical-overlap heuristics | `ragas` package LLM-judged metrics -- same function signatures in `app/evaluation/ragas_eval.py` |
| MLflow backend | SQLite (`mlruns.db`) | Remote MLflow tracking server / managed MLflow |
| Orchestration | DAG file only (no scheduler) | Live Airflow deployment with `dags/drift_monitoring_dag.py` |
| Vector store | FAISS (local, in-memory) | FAISS on persistent volume, or managed (Pinecone) |

---

## License

MIT
