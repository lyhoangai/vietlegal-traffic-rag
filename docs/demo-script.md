# Demo Script

## Goal

Show the project in 2-3 minutes and prove five things:

1. the app answers a concrete traffic-law question
2. it remembers the previous turn
3. it restores history after refresh
4. it refuses out-of-scope questions
5. there is benchmark evidence behind the demo

## Before You Start

Open:

- `http://127.0.0.1:8000/`

Make sure:

- the session sidebar is visible
- a TTS voice is loaded
- `docs/benchmarks/latest_summary.md` is ready

## Suggested 2-3 Minute Talk Track

### 0:00 - 0:20

Say:

> "This is VietLegal Traffic RAG, a scoped Vietnamese traffic-law RAG demo. The goal is not to build general legal AI. The goal is to build something narrower, more trustworthy, and measurable."

### 0:20 - 0:50

Ask:

- `O to vuot den do phat bao nhieu?`

Point out:

- streaming response
- fine amount shown clearly
- source chips separated from the answer
- TTS is available

### 0:50 - 1:20

Ask:

- `The xe may thi sao?`

Point out:

- the app keeps the previous violation context
- only the vehicle type changes

Say:

> "The backend uses short-term session memory, so a follow-up question does not need to repeat the whole context."

### 1:20 - 1:40

Refresh the page.

Point out:

- the history comes back
- the sidebar still shows the session

Say:

> "Chat turns are stored in SQLite, so the demo survives a refresh without losing short-term context."

### 1:40 - 2:00

Ask:

- `Doi bang lai xe can giay to gi?`

Say:

> "The project has a clear boundary. Questions outside the supported scope are refused instead of guessed."

### 2:00 - 2:30

Close with:

> "What I wanted to show here is scoped product design, active-corpus management, enough memory for believable follow-up questions, and benchmark evidence behind the pipeline."

## Backup Demo Questions

- `Xe may khong doi mu bao hiem bi phat the nao?`
- `O to co nong do con thi bi xu phat the nao?`
- `Toc do khai thac toi da tren duong cao toc la bao nhieu?`

## If The Recruiter Asks For Proof

Show:

- the 33-case benchmark package under `datasets/vietlegal-traffic-eval-v1/`
- the benchmark summary at `docs/benchmarks/latest_summary.md`
- the UI assets in `docs/assets/`

## What Not To Say

- do not claim it is production-ready
- do not claim it answers all legal topics
- do not claim it is foundation-model research

Better framing:

- scoped
- grounded
- benchmarked
- portfolio-ready
