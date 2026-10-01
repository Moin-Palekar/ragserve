# ragserve: Design Spec

> Status: **Planned** · Owner: Moin Palekar · Targets: M3 (MLE resume-complete), M5 (ML Infra resume-complete) · Resume dates: Sep to Oct 2026
> Purpose of this doc: everything needed to build ragserve from zero without re-deriving decisions.

---

## 1. Problem

Retrieval-augmented generation (RAG) answers questions by first *finding* relevant passages in a corpus, then having a language model *write* an answer grounded in them, with citations. Demos are easy. Serving it well is not: every query runs three models (embedder, reranker, generator) plus a vector search, and each stage adds latency, cost, and a chance to lose accuracy.

ragserve is a production-style retrieval + generation service where the engineering focus is **model serving and inference optimization**: ONNX export and quantization, dynamic batching, ANN index tuning, per-stage latency budgets, and measured accuracy/latency trade-offs, deployed on Kubernetes.

**Why this project exists (portfolio):** closes the model serving / inference optimization gap on the MLE resume (M3) and the deployment, rollout, and autoscaling gap on the ML Infra resume (M5). The headline artifact is a table: *PyTorch vs ONNX FP32 vs ONNX INT8, latency and throughput vs retrieval quality retained.*

## 2. Goals and non-goals

### Goals
1. **Grounded answers with citations:** every answer cites the chunk IDs it used; streamed token by token.
2. **Measured optimization:** each inference optimization ships with a before/after number for latency, throughput, *and* quality.
3. **Retrieval quality tracked:** nDCG@10 and Recall@100 on a standard benchmark, re-run in CI on every model or index change.
4. **Latency budget:** p95 end-to-end time-to-first-token target set at M2 from the baseline, then held under load.
5. **Independent scaling:** the embedding/reranking service scales separately from the API.
6. **Observability:** per-stage latency histograms (embed, retrieve, rerank, generate TTFT, tokens/s).

### Non-goals
- Training or fine-tuning the LLM.
- Agentic multi-step tool use.
- Multi-tenant auth and billing.
- Beating SOTA retrieval; the point is the serving system around solid off-the-shelf models.

## 3. Architecture

```
                         ┌───────────────────────────────┐
  POST /v1/answer ──────▶│          ragserve-api         │  FastAPI, async
                         │  orchestrates the pipeline    │
                         └──┬─────────┬──────────┬───────┘
          1. embed query    │         │ 2. ANN   │ 4. generate (SSE stream)
                            ▼         ▼          ▼
               ┌──────────────────┐ ┌───────────────┐ ┌──────────────────────┐
               │ ragserve-infer   │ │  PostgreSQL   │ │ generator backend     │
               │ ONNX Runtime     │ │  + pgvector   │ │ OpenAI-compatible:    │
               │ embedder (INT8)  │ │ HNSW + tsvector│ │ vLLM / llama.cpp     │
               │ reranker (INT8)  │ └───────────────┘ └──────────────────────┘
               │ dynamic batcher  │        ▲
               └──────────────────┘        │
                 ▲  3. rerank top-k        │ ingest: chunk → embed → upsert
                 └─────────────────────────┘
 Prometheus scrapes api + infer → Grafana
```

### Components
| Service | Role |
|---|---|
| `ragserve-api` | FastAPI. `/search`, `/answer` (SSE), `/ingest`. Orchestrates stages, enforces timeouts, builds the prompt with citations. Stateless. |
| `ragserve-infer` | Model server for embedder + reranker on ONNX Runtime, with a dynamic micro-batcher. Separate deployment so it scales on its own (M3). Before M3 it runs in-process. |
| PostgreSQL + pgvector | Documents, chunks, embeddings (HNSW), full-text index for hybrid search. |
| Generator backend | Any OpenAI-compatible endpoint. Dev: llama.cpp server or a hosted API. M5: self-hosted vLLM on a GPU node. |
| `ragserve-bench` | Offline eval (BEIR metrics) + load generator (Locust). |

### Models
| Role | Model | Notes |
|---|---|---|
| Embedder | `BAAI/bge-small-en-v1.5` (384-d) | Small, strong on BEIR, exports cleanly to ONNX |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder over top-k candidates |
| Generator | `Qwen2.5-1.5B-Instruct` (default) | Swappable; anything behind an OpenAI-compatible API |

### Key design decisions (write each up as an ADR in `docs/adr/`)
| # | Decision | Why | Rejected alternative |
|---|---|---|---|
| ADR-1 | pgvector, not a dedicated vector DB | One system for metadata, filters, full-text, and vectors; transactional upserts; HNSW is good enough at this scale | Pinecone/Qdrant/Milvus: another system to run, no SQL joins |
| ADR-2 | ONNX Runtime for embed + rerank | Graph optimizations + INT8 dynamic quantization on CPU; framework-independent artifact | Serving raw PyTorch (slower, heavier image); TensorRT (GPU-only, revisit in M5) |
| ADR-3 | Dynamic micro-batching in `ragserve-infer` | Encoder throughput scales with batch size; collecting requests for a few ms trades tiny latency for large throughput | One forward pass per request |
| ADR-4 | Retrieve wide, rerank narrow | ANN top-50 for recall, cross-encoder to top-5 for precision | Feeding ANN top-5 straight to the LLM |
| ADR-5 | Hybrid search with Reciprocal Rank Fusion | Dense misses exact terms (IDs, acronyms); BM25 catches them; RRF needs no score calibration | Weighted score sum (needs per-corpus tuning) |
| ADR-6 | Generator behind OpenAI-compatible API | Swap llama.cpp / vLLM / hosted without code change | Embedding the LLM in the API process |
| ADR-7 | Separate infer deployment | Embedding is CPU-bound and bursty, API is I/O-bound; scale each on its own signal | Single monolith |

## 4. Data model

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source      TEXT NOT NULL,           -- 'beir/scifact', 'arxiv', 'upload'
  external_id TEXT,                    -- e.g. BEIR doc id, arXiv id
  title       TEXT,
  metadata    JSONB NOT NULL DEFAULT '{}',
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (source, external_id)
);

CREATE TABLE chunks (
  id            BIGSERIAL PRIMARY KEY,
  document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord           INT  NOT NULL,         -- position within document
  text          TEXT NOT NULL,
  token_count   INT  NOT NULL,
  embedding     VECTOR(384) NOT NULL,
  model_version TEXT NOT NULL,         -- e.g. 'bge-small-v1.5-onnx-int8'
  tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);      -- tuned in M2
CREATE INDEX chunks_tsv  ON chunks USING gin (tsv);
```

`model_version` on every chunk makes re-embedding after a model change explicit: a query only searches chunks whose version matches the active embedder.

### Chunking
Token-based, 256 tokens with 32 overlap, split on sentence boundaries where possible. Parameters recorded per ingest run so evals are reproducible.

## 5. API

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/v1/ingest` | `{source, documents: [{external_id, title, text, metadata}]}` | `202 {ingest_id, chunks}` |
| POST | `/v1/search` | `{query, k=5, mode: dense|hybrid, rerank: bool, filters?}` | ranked chunks with scores and per-stage timings |
| POST | `/v1/answer` | `{query, k=5, stream=true}` | SSE: `token` events, then a `citations` event `[{chunk_id, document_id, title}]`, then `done` with timings |
| GET | `/healthz`, `/readyz`, `/metrics` | | readyz checks PG + infer + generator |

Every response includes `timings_ms: {embed, retrieve, rerank, ttft, generate, total}`. That makes the latency story visible in a single curl.

**Prompt contract:** the system prompt instructs the model to answer only from the numbered context passages and cite them as `[n]`; if the passages don't contain the answer, say so. The API maps `[n]` back to chunk IDs.

## 6. Inference optimization plan (the core of the project)

Each row is a measured experiment; results go in `docs/bench/optimization.md`.

| # | Optimization | Measure | Guardrail |
|---|---|---|---|
| O1 | Baseline: sentence-transformers PyTorch FP32 | embed latency p50/p95, throughput (texts/s) at batch 1/8/32 | nDCG@10 baseline |
| O2 | ONNX export (`optimum-cli export onnx`) + graph optimization | same | nDCG@10 within 0.1 pt of O1 |
| O3 | INT8 dynamic quantization (`onnxruntime.quantization.quantize_dynamic`) | same + model size MB | ≥ 99% of O1 nDCG@10 |
| O4 | Dynamic micro-batching (`max_batch=32`, `max_wait=5ms`) | throughput and p95 under concurrent load | p95 increase bounded by max_wait |
| O5 | HNSW tuning: sweep `ef_search` ∈ {10, 20, 40, 80, 160} | retrieve latency vs Recall@100 curve | choose knee of the curve |
| O6 | Reranker ONNX INT8 + top-k sweep (k ∈ {20, 50, 100}) | rerank latency vs nDCG@10 gain | |
| O7 | Query embedding cache (LRU, M4) | hit rate, p50 on repeated queries | |
| O8 | Thread tuning (`intra_op_num_threads`) per pod CPU limit | throughput per core | |

### Dynamic batcher design
An `asyncio.Queue` of `(text, future)`. A single consumer loop takes the first item, then keeps draining until `max_batch` items or `max_wait` elapsed, runs one ONNX forward pass in a thread pool (`run_in_executor`, since ORT releases the GIL), and resolves each future. Exposes `infer_batch_size` histogram.

## 7. Evaluation

### Retrieval (offline, CI-gated)
- **Benchmark:** BEIR **SciFact** (~5k docs, 300 test queries; small enough to run in CI). FiQA as a second, larger set for the report.
- **Metrics:** nDCG@10, Recall@100, MRR@10.
- **Configs compared:** dense, dense + rerank, hybrid (RRF), hybrid + rerank.
- **CI rule:** a PR that changes models, chunking, or index params must not drop SciFact nDCG@10 by more than 0.5 pt.

### End-to-end answers (M4)
- 50-question hand-checked set over the demo corpus, plus LLM-as-judge for faithfulness (is every claim supported by a cited passage?) and answer relevance.
- Citation precision: fraction of cited passages that actually support the claim.

### Load
- Locust against `/search` and `/answer` on kind; report QPS at p95 budget, saturation point, and which stage saturates first.

Results fill the resume placeholders: XX% lower embedding latency, YY× throughput, ZZ% of baseline nDCG@10 retained, NN QPS at p95 < budget.

## 8. Observability

| Metric | Type | Labels |
|---|---|---|
| `ragserve_stage_seconds` | histogram | stage (embed/retrieve/rerank/ttft/generate) |
| `ragserve_requests_total` | counter | route, status |
| `ragserve_infer_batch_size` | histogram | model |
| `ragserve_infer_queue_wait_seconds` | histogram | model |
| `ragserve_generated_tokens_total` | counter | model |
| `ragserve_cache_hits_total` | counter | cache |

Grafana dashboard in `deploy/grafana/` with a stacked per-stage latency panel, the single most useful chart for explaining the system.

## 9. Repo layout
```
ragserve/
  api/            # FastAPI app, routes, prompt building, SSE
  infer/          # ONNX sessions, dynamic batcher, model loading
  retrieval/      # pgvector queries, hybrid RRF, filters
  ingest/         # loaders (BEIR, arXiv, files), chunker
  generate/       # OpenAI-compatible client, streaming, citation mapping
  common/         # config (pydantic-settings), metrics, logging
scripts/
  export_onnx.py  quantize.py  ingest_beir.py
bench/
  eval_beir.py    optimize_bench.py  locustfile.py
migrations/       # alembic
deploy/compose/   deploy/k8s/   deploy/grafana/
tests/            # pytest: unit + integration (testcontainers pgvector)
docs/SPEC.md  docs/adr/  docs/bench/
```

**Stack:** Python 3.11, FastAPI, uvicorn, `asyncpg`, pgvector, SQLAlchemy/Alembic (migrations only), ONNX Runtime, Hugging Face Optimum, sentence-transformers (baseline only), `httpx` (generator client), `prometheus-client`, pytest + testcontainers, Locust, Docker, kind, kustomize, GitHub Actions. M5 adds vLLM, KEDA, MinIO.

## 10. Milestones

| M | Scope | Done when |
|---|---|---|
| **M1: Working RAG** | Schema, BEIR SciFact ingest, PyTorch embedder, pgvector HNSW, `/search`, `/answer` with citations via external OpenAI-compatible endpoint, docker-compose, CI (lint, tests) | Answers with citations end to end; SciFact nDCG@10 baseline recorded. **Repo goes public, README v1.** |
| **M2: Optimized inference** | O1 to O5: ONNX export, INT8, dynamic batcher, HNSW sweep; `ragserve-bench` | `docs/bench/optimization.md` has the PyTorch vs ONNX vs INT8 table and the ef_search curve |
| **M3: Production pipeline** *(MLE resume target)* | Reranker (O6), hybrid RRF, SSE streaming, per-stage metrics + Grafana, `ragserve-infer` split out, K8s on kind, Locust load test, CI nDCG gate | README shows architecture, optimization table, retrieval table, load-test result |
| **M4: Quality + scale** | Query cache (O7), HPA on infer, end-to-end answer eval, demo corpus (arXiv cs.RO / cs.CV abstracts on autonomous driving), optional async ingestion through **taskq** | Answer-quality report; autoscaling demo |
| **M5: ML Infra** *(ML Infra resume target)* | Self-hosted vLLM generator on GPU node, versioned model artifacts in MinIO (model registry), canary rollout of a new embedder with shadow traffic + automatic re-embed job, KEDA autoscaling, SLO dashboard + alerts | Canary report comparing versions on latency and nDCG; SLO dashboard screenshot |

## 11. Resume mapping

Paste the MLE and ML Infra resume bullets for ragserve here. Each must point to a milestone and to the artifact that proves it.

| Bullet (paste) | Resume | Milestone | Evidence in repo |
|---|---|---|---|
| | MLE | | |
| | ML Infra | | |

Typical claim → evidence pairs:
- ONNX + INT8 latency/throughput gains with quality retained → `docs/bench/optimization.md`
- dynamic batching throughput → O4 row + batch-size histogram
- retrieval quality → SciFact/FiQA table, CI gate
- Kubernetes deployment, independent scaling → `deploy/k8s/`, HPA demo
- model rollout / registry (ML Infra) → M5 canary report

## 12. Interview talking points
- Why INT8 dynamic quantization barely hurts encoder quality, and how you proved it rather than assumed it.
- Batching: why throughput rises with batch size on encoders, and the latency cost bounded by `max_wait`.
- HNSW: what `m`, `ef_construction`, `ef_search` trade off; reading the recall/latency curve.
- Why retrieve wide and rerank narrow; bi-encoder vs cross-encoder cost.
- Why hybrid search: concrete query where dense retrieval failed and BM25 saved it.
- Re-embedding on model change: why `model_version` on each chunk, and how the canary avoids a stop-the-world migration.
- TTFT vs total latency, and why streaming changes perceived latency.

## 13. Open questions
- GPU for O-series benchmarks? Default: CPU numbers are the headline (cheaper, reproducible); add one GPU column if Sol access allows.
- Chunk size sweep (128/256/512) as an extra experiment in M2 or M4?
- Demo corpus: arXiv AV abstracts vs course notes; arXiv ties the project to the AV story.
