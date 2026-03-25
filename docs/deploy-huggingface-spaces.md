# Deploy VietLegal Traffic RAG on Hugging Face Spaces

This guide is the cheapest path to get a public demo URL without adding a bank card to Render.

## Why This Works

- Hugging Face Spaces supports Docker-based apps.
- Free `CPU Basic` hardware gives you `2 vCPU`, `16 GB RAM`, and `50 GB` of non-persistent disk.
- This repo's active PDF corpus and local Chroma store are both small enough for a portfolio demo.

Official references:

- <https://huggingface.co/docs/hub/spaces-overview>
- <https://huggingface.co/docs/hub/spaces-sdks-docker>
- <https://huggingface.co/pricing>

## Before You Start

You need:

- a Hugging Face account
- a public Space with SDK set to `Docker`
- this repo checked out on the `render-deploy-vibe` branch or a later branch that includes the same deployment files

## Recommended Space Settings

Set these in the Space `Settings` page:

### Secrets

- `GROQ_API_KEY`
- `SERPER_API_KEY` (optional)
- `TAVILY_API_KEY` (optional)

### Variables

- `HOST=0.0.0.0`
- `PORT=7860`
- `LLM_PROVIDER=groq`
- `EMBEDDING_PROVIDER=local`
- `LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `CHROMA_DB_PATH=/tmp/chroma_db`
- `MEMORY_DB_PATH=/tmp/chat_memory.db`

## Create the Space

1. Go to <https://huggingface.co/new-space>
2. Choose:
   - Owner: your account
   - Space name: `vietlegal-traffic-rag`
   - License: `MIT`
   - SDK: `Docker`
   - Visibility: `Public`
3. Create the Space.

## Push the App Code

After the Space is created, clone the Space repository:

```powershell
git clone https://huggingface.co/spaces/<your-username>/vietlegal-traffic-rag
```

Copy these files from this repo into the Space repo:

- `src/`
- `data/`
- `datasets/`
- `docs/assets/chat-ui.png`
- `docs/assets/history-sidebar.png`
- `.dockerignore`
- `Dockerfile`
- `requirements.txt`
- `.env.example`

Then replace the Space repo `README.md` with the contents of [`../README.hf-space.md`](../README.hf-space.md).

Commit and push:

```powershell
git add .
git commit -m "Deploy VietLegal Traffic RAG Space"
git push
```

## What to Expect on Free Hardware

- The Space can sleep after inactivity.
- On a cold start, the container may need time to rebuild Chroma.
- Session memory is stored in `/tmp`, so it is not durable across rebuilds or restarts.
- This is acceptable for a free public portfolio demo, but not for production.

## First Checks

After the Space finishes building:

1. Open the Space URL.
2. Test a simple traffic-law penalty question.
3. Confirm the answer includes citations.
4. If chat fails immediately, check whether `GROQ_API_KEY` was added as a secret.
