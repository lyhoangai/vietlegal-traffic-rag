# VietLegal Traffic RAG

![CI workflow](docs/assets/ci-badge.svg)

Scoped Vietnamese traffic-law RAG for the `LLM product / RAG engineer` lane. The app answers a narrow set of traffic-law questions, keeps short-term session memory, and can verify official sources when the user asks for a web-confirmed answer.

## Overview

- narrow scope instead of broad legal-AI claims
- active 2025 corpus managed through `data/manifest.json`
- reproducible benchmark artifacts and a public eval package
- local web demo with SSE streaming, session history, and Vietnamese TTS

## Evidence

![Chat UI](docs/assets/chat-ui.png)

![History Sidebar](docs/assets/history-sidebar.png)

### Benchmark Highlights

Local benchmark run on `2026-03-23` using the public eval package at [`datasets/vietlegal-traffic-eval-v1/README.md`](datasets/vietlegal-traffic-eval-v1/README.md):

| Mode | Cases | Pass Rate | Citation Rate | Avg Confidence |
| --- | ---: | ---: | ---: | ---: |
| `full` | 33 | 100.0% | 100.0% | 0.741 |
| `no_reranker` | 33 | 100.0% | 100.0% | 0.741 |
| `no_web_fallback` | 33 | 97.0% | 100.0% | 0.735 |

- Full artifacts: [`docs/benchmarks/latest_summary.md`](docs/benchmarks/latest_summary.md) and [`docs/benchmarks/latest_results.json`](docs/benchmarks/latest_results.json)
- Dataset package: [`datasets/vietlegal-traffic-eval-v1/README.md`](datasets/vietlegal-traffic-eval-v1/README.md)
- Published dataset: <https://huggingface.co/datasets/lyhoang0104ls/vietlegal-traffic-eval-v1>
- Demo flow: [`docs/demo-script.md`](docs/demo-script.md)

## Architecture

![Architecture](docs/assets/architecture.svg)

The public architecture view is intentionally compact. It shows only the real layers in this repo: web UI, FastAPI API, session memory, Chroma-backed retrieval, official-source web fallback, and benchmark artifacts.

- Longer walkthrough: [`docs/architecture.md`](docs/architecture.md)
- Scope statement: this is a traffic-law RAG demo, not a general legal chatbot platform

## Why This Is Trustworthy

- Scoped domain instead of "answer everything" behavior
- Explicit refusal for out-of-scope topics such as GPLX procedures or vehicle registration
- Active 2025 corpus policy managed via [`data/manifest.json`](data/manifest.json)
- Optional official-source web verification when the user explicitly asks for source confirmation
- Session memory for follow-up turns and sidebar-visible history recovery
- Benchmarkable pipeline with mode flags for `reranker` and `web fallback`

## Quick Start

Clone the public repo:

```powershell
git clone https://github.com/lyhoangai/vietlegal-traffic-rag.git
cd vietlegal-traffic-rag
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.env` from `.env.example` and set at least:

- `GROQ_API_KEY`
- `LLM_PROVIDER=groq`
- `EMBEDDING_PROVIDER=local`
- `MEMORY_DB_PATH=./chat_memory.db`

Optional but recommended for stronger web fallback:

- `SERPER_API_KEY`
- `TAVILY_API_KEY`

Build the vector database:

```powershell
.\.venv\Scripts\python.exe -m src.ingest.build_db
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open the web UI:

```text
http://127.0.0.1:8000/
```

## API / Endpoints

- `POST /chat`
- `POST /chat/stream`
- `GET /chat/history?session_id=...`
- `GET /chat/sessions`
- `DELETE /chat/sessions/{session_id}`
- `GET /tts/voices?locale=vi-VN`
- `POST /tts`
- `GET /health`
- `GET /eval/metrics`

## Tests

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Run the smoke benchmark used by CI:

```powershell
.\.venv\Scripts\python.exe -m src.eval.run_benchmark --smoke
```

Regression coverage includes routing, follow-up memory, official-source fallback, benchmark artifacts, manifest-driven ingestion, chat history endpoints, SSE streaming, and TTS endpoints.

## Known Limits

- not production-ready
- memory is intentionally short-term only
- retrieval quality still depends on chunking and corpus coverage
- the public benchmark is compact and portfolio-focused
