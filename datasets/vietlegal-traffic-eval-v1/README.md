# VietLegal Traffic Eval v1

Publish-ready benchmark dataset for the `vietlegal-ai-agent` portfolio repo.

## Scope

This dataset is intentionally narrow. It only covers the demo scope that the app is willing to answer:

- traffic violation penalties
- expressway speed rules
- follow-up turns that depend on short-term conversation memory
- scope refusals for out-of-scope questions

It is not a general legal QA dataset and it should not be marketed as one.

## Files

- `data.jsonl`: source-of-truth benchmark rows used by `python -m src.eval.run_benchmark`

## Required Fields

Each row contains these fields:

- `id`
- `category`
- `question`
- `reference_answer`
- `expected_behavior`
- `primary_source`

Some rows also include:

- `required_phrases`: phrases required for deterministic benchmark scoring
- `messages`: short prior turns for follow-up evaluation
- explicit official-source verification prompts so the public benchmark can measure `web fallback`

## Categories

- `penalty`
- `speed`
- `follow_up`
- `scope_refusal`

## Source Policy

- Active 2025 traffic-law corpus only
- No legacy 2019/2008 citations
- Out-of-scope refusals are grounded in the repo scope policy, not in old regulations

## Limitations

- Small benchmark focused on demo reliability, not broad legal coverage
- Phrase-based scoring is intentionally strict and portfolio-oriented
- Follow-up rows assume short-term conversation memory rather than long-horizon chat memory
- Official-source verification is made reproducible by manifest-pinned official URLs when live search APIs are unavailable

## Publish To Hugging Face

Recommended flow:

```powershell
cd D:\vietlegal-ai-agent
.\.venv\Scripts\python.exe -m pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create vietlegal-traffic-eval-v1 --type dataset
huggingface-cli upload <your-username>/vietlegal-traffic-eval-v1 datasets/vietlegal-traffic-eval-v1
```

Optional:

- enable manual gating if you want to collect downloader intent
- add a dataset card on Hugging Face using the contents of this README
- keep the local repo copy as the source of truth, then upload from this folder after every revision
