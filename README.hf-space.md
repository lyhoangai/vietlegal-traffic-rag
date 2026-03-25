---
title: VietLegal Traffic RAG
emoji: "🚦"
colorFrom: amber
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Scoped Vietnamese traffic-law RAG with citations, session memory, and optional official-source verification.
---

# VietLegal Traffic RAG

Scoped Vietnamese traffic-law RAG for portfolio demos.

## What This Space Shows

- narrow legal scope instead of broad legal-AI claims
- retrieval over an active 2025 traffic-law corpus
- optional official-source web verification
- short-term chat memory
- Vietnamese TTS for local-style demos

## Notes

- Free Spaces can sleep after inactivity, so the first wake-up may be slow.
- This free setup uses non-persistent storage, so chat history and rebuilt Chroma state can reset on restart.
- Add `GROQ_API_KEY` in the Space settings before testing `/chat`.

## Source Repo

- GitHub: <https://github.com/lyhoangai/vietlegal-traffic-rag>
- Deploy guide: see `docs/deploy-huggingface-spaces.md` in the source repo
