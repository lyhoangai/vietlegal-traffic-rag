# VietLegal Traffic RAG Benchmark Summary

- Generated: 2026-03-27T11:00:45.901030+00:00
- Dataset: `datasets/vietlegal-traffic-eval-v2/data.jsonl`
- Cases per mode: 300
- Smoke mode: no

## Overall

| Mode | Cases | Pass Rate | Errors | Citation Rate | Reference Match | Avg Confidence | Web Usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 300 | 98.3% | 0 | 100.0% | 99.6% | 0.823 | 70.0% |

## By Category (full)

| Category | Cases | Pass Rate | Errors | Citation Rate | Reference Match | Avg Confidence | Web Usage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| follow_up | 45 | 97.8% | 0 | 100.0% | 99.3% | 0.919 | 75.6% |
| penalty | 170 | 97.6% | 0 | 100.0% | 99.5% | 0.924 | 100.0% |
| scope_refusal | 40 | 100.0% | 0 | 0.0% | 0.0% | 0.200 | 0.0% |
| speed | 45 | 100.0% | 0 | 100.0% | 100.0% | 0.900 | 13.3% |

## Scoring Notes

- `Pass Rate`: required phrases matched and expected behavior observed.
- `Citation Rate`: answer cases whose primary source appears in the answer or source chips.
- `Reference Match`: average required-phrase coverage for answer cases.
- `Web Usage`: cases that touched official-source web fallback.
