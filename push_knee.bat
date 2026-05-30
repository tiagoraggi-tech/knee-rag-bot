@echo off
cd /d "C:\Users\raggi\OneDrive\Documentos\Claude\Projects\uriel_lorena\knee_rag_bot"

echo === Salvando alteracao em knee_retrieval_chain.py ===
git stash

echo === Sincronizando com origin/main ===
git fetch origin
git reset --hard origin/main

echo === Reaplicando alteracao ===
git stash pop

echo === Commitando ===
git add knee_retrieval_chain.py
git commit -m "Add multi-model fallback + reduce context tokens

- Fallback order: llama-3.3-70b-versatile -> llama-3.1-8b-instant -> gemma2-9b-it
- Each model has its own separate TPD quota on Groq free tier
- Reduced rerank_top_k from 5 to 3 (saves ~40pct tokens per call)
- Reduced doc excerpt from 1000 to 600 chars per document
- Automatic retry on 429 rate limit without silent failure"

echo === Push para GitHub ===
git push origin main
pause
