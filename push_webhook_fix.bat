@echo off
cd /d "C:\Users\raggi\OneDrive\Documentos\Claude\Projects\uriel_lorena\knee_rag_bot"

echo === Removendo lock se existir ===
if exist .git\index.lock del /f .git\index.lock

echo === Configurando identidade ===
git config user.email "tiagoraggi@gmail.com"
git config user.name "Tiago Raggi"

echo === Commitando bot_webhook.py ===
git add bot_webhook.py
git commit -m "fix: add MESSAGES_UPSERT route alias + uriel-bot as default instance

- Added route alias /webhook/messages-upsert/MESSAGES_UPSERT to handle
  Evolution API webhookByEvents=true (appends event name to URL path)
- Changed EVOLUTION_INSTANCE default from knee-bot to uriel-bot"

echo === Sincronizando com remote ===
git pull --rebase origin main

echo === Push para GitHub ===
git push origin main

echo.
echo === Pronto! Aguarde Railway fazer redeploy (~60s) ===
pause
