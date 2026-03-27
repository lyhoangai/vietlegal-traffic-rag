# VietLegal Traffic Eval v2

Curated ~300-case benchmark dataset for the `vietlegal-ai-agent` portfolio repo.

## Scope

This dataset keeps the app scope intentionally narrow. It only covers the demo behavior the project claims publicly:

- traffic violation penalties
- expressway speed rules
- short-term follow-up turns that depend on recent conversation context
- explicit refusals for out-of-scope questions

It is not a general legal QA dataset and it should not be marketed as one.

## Benchmark Shape

- ~300 curated rows
- hand-written question variants grouped by scenario
- deterministic citation-oriented scoring support
- category coverage across:
  - `penalty`
  - `speed`
  - `follow_up`
  - `scope_refusal`

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

Coverage metadata used for curation and analysis:

- `subtopic`
- `scenario_group`
- `variant`

Some rows also include:

- `required_phrases`: phrases required for deterministic benchmark scoring
- `messages`: short prior turns for follow-up evaluation

## Source Policy

- Active 2025 traffic-law corpus only
- No legacy 2019/2008 citations
- Public question patterns may inspire benchmark wording, but benchmark rows are manually rewritten and manually grounded
- Out-of-scope refusals are grounded in the repo scope policy, not in old regulations

## Limitations

- Curated benchmark focused on demo reliability, not broad legal coverage
- Phrase-based scoring remains intentionally strict and portfolio-oriented
- Follow-up rows assume short-term conversation memory rather than long-horizon chat memory
- The benchmark emphasizes reproducibility over open-ended legal interpretation

## Publish To Hugging Face

Recommended flow:

```powershell
cd D:\vietlegal-ai-agent
.\.venv\Scripts\python.exe -m pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create vietlegal-traffic-eval-v2 --type dataset
huggingface-cli upload <your-username>/vietlegal-traffic-eval-v2 datasets/vietlegal-traffic-eval-v2
```

Optional:

- keep `vietlegal-traffic-eval-v1` as the compact historical baseline
- publish this `v2` package as the stronger benchmark claim in the portfolio
- keep the local repo copy as the source of truth, then upload from this folder after every revision
