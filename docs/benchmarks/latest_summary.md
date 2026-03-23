# VietLegal Traffic RAG Benchmark Summary

- Generated: 2026-03-23T09:06:26.794481+00:00
- Dataset: `datasets/vietlegal-traffic-eval-v1/data.jsonl`
- Cases per mode: 33
- Smoke mode: no

## Overall

| Mode | Cases | Pass Rate | Citation Rate | Reference Match | Avg Confidence | Web Usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 33 | 100.0% | 100.0% | 100.0% | 0.741 | 48.5% |
| no_reranker | 33 | 100.0% | 100.0% | 100.0% | 0.741 | 48.5% |
| no_web_fallback | 33 | 97.0% | 100.0% | 99.0% | 0.735 | 0.0% |

## By Category (full)

| Category | Cases | Pass Rate | Citation Rate | Reference Match | Avg Confidence | Web Usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| follow_up | 8 | 100.0% | 100.0% | 100.0% | 0.918 | 75.0% |
| penalty | 9 | 100.0% | 100.0% | 100.0% | 0.923 | 100.0% |
| scope_refusal | 8 | 100.0% | 0.0% | 0.0% | 0.200 | 0.0% |
| speed | 8 | 100.0% | 100.0% | 100.0% | 0.900 | 12.5% |

## By Category (no_reranker)

| Category | Cases | Pass Rate | Citation Rate | Reference Match | Avg Confidence | Web Usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| follow_up | 8 | 100.0% | 100.0% | 100.0% | 0.918 | 75.0% |
| penalty | 9 | 100.0% | 100.0% | 100.0% | 0.923 | 100.0% |
| scope_refusal | 8 | 100.0% | 0.0% | 0.0% | 0.200 | 0.0% |
| speed | 8 | 100.0% | 100.0% | 100.0% | 0.900 | 12.5% |

## By Category (no_web_fallback)

| Category | Cases | Pass Rate | Citation Rate | Reference Match | Avg Confidence | Web Usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| follow_up | 8 | 100.0% | 100.0% | 100.0% | 0.908 | 0.0% |
| penalty | 9 | 88.9% | 100.0% | 97.2% | 0.911 | 0.0% |
| scope_refusal | 8 | 100.0% | 0.0% | 0.0% | 0.200 | 0.0% |
| speed | 8 | 100.0% | 100.0% | 100.0% | 0.900 | 0.0% |

## Scoring Notes

- `Pass Rate`: required phrases matched and expected behavior observed.
- `Citation Rate`: answer cases whose primary source appears in the answer or source chips.
- `Reference Match`: average required-phrase coverage for answer cases.
- `Web Usage`: cases that touched official-source web fallback.
