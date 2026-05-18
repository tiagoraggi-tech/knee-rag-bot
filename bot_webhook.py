"""
bot_webhook.py — Webhook Flask para integração Evolution API + KneeRAGChain
Endpoint: /webhook/messages-upsert
"""

import os
import hashlib
import logging
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from knee_retrieval_chain import KneeRAGChain, format_for_whatsapp

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot_webhook")

app = Flask(__name__)

# Inicializa chain 1x no startup (carrega modelos ~30s)
log.info("Inicializando KneeRAGChain...")
chain = KneeRAGChain(
    persist_dir=os.getenv("CHROMA_DIR", "/data/chroma_knee"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    audit_log_path=os.getenv("AUDIT_LOG", "/data/rag_audit.jsonl"),
)
log.info("Chain pronta.")

EVOLUTION_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "knee-bot")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
HANDOFF_NUMBER = os.getenv("HANDOFF_NUMBER", "5524988370406")


def send_whatsapp(phone: str, message: str) -> bool:
    """Envia mensagem via Evolution API."""
    url = f"{EVOLUTION_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {
        "Content-Type": "application/json",
        "apikey": EVOLUTION_API_KEY,
    }
    payload = {"number": phone, "text": message}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        log.info("Enviado para %s", phone[-4:])
        return True
    except Exception as e:
        log.error("Falha ao enviar: %s", e)
        return False


@app.route("/webhook/messages-upsert", methods=["POST"])
def messages_upsert():
    try:
        data = request.get_json(force=True)
        log.debug("Payload recebido: %s", str(data)[:300])

        # Estrutura padrão da Evolution API
        msg_data = data.get("data", {})
        key = msg_data.get("key", {})

        # Ignora mensagens do próprio bot
        if key.get("fromMe"):
            return jsonify({"status": "ignored_self"}), 200

        # Extrai texto e remetente
        phone = key.get("remoteJid", "").replace("@s.whatsapp.net", "")
        message_obj = msg_data.get("message", {})
        text = (
            message_obj.get("conversation")
            or message_obj.get("extendedTextMessage", {}).get("text")
            or ""
        ).strip()

        if not text or not phone:
            return jsonify({"status": "no_text"}), 200

        log.info("Mensagem de %s: %s", phone[-4:], text[:80])

        # Hash anônimo para auditoria (LGPD)
        phone_hash = hashlib.md5(phone.encode()).hexdigest()

        # Processa
        result = chain.ask(text, patient_id_hash=phone_hash)
        reply = format_for_whatsapp(result)

        # Se red flag, também notifica handoff
        if result["red_flag"]:
            handoff_msg = (
                f"🚨 RED FLAG detectada\n"
                f"De: {phone}\n"
                f"Msg: {text[:200]}\n"
                f"Hash: {phone_hash[:12]}"
            )
            send_whatsapp(HANDOFF_NUMBER, handoff_msg)

        # Envia resposta ao paciente
        send_whatsapp(phone, reply)

        return jsonify({"status": "ok", "red_flag": result["red_flag"]}), 200

    except Exception as e:
        log.exception("Erro no webhook: %s", e)
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/debug", methods=["GET"])
def debug():
    chroma_dir = os.getenv("CHROMA_DIR", "/data/chroma_knee")
    import os as _os
    return jsonify({
        "chroma_dir": chroma_dir,
        "chroma_exists": _os.path.exists(chroma_dir),
        "audit_log": os.getenv("AUDIT_LOG", "/data/rag_audit.jsonl"),
        "evolution_url": EVOLUTION_URL,
        "instance": EVOLUTION_INSTANCE,
    })


@app.route("/admin/ingest", methods=["POST"])
def admin_ingest():
    """Dispara ingestão KneeLoader em background. Protegido por GROQ_API_KEY como token."""
    token = request.headers.get("X-Admin-Token", "")
    if token != os.getenv("GROQ_API_KEY", ""):
        return jsonify({"status": "unauthorized"}), 403
    import threading
    from knee_loader import KneeKnowledgeLoader
    def run_ingest():
        try:
            log.info("=== INGESTÃO INICIADA ===")
            loader = KneeKnowledgeLoader(
                persist_dir=os.getenv("CHROMA_DIR", "/data/chroma_knee"),
                entrez_email=os.getenv("ENTREZ_EMAIL", "tiagoraggi@gmail.com"),
            )
            loader.ingest_pubmed(max_results=30)
            loader.ingest_websites()
            loader.build_vectorstore()
            log.info("=== INGESTÃO CONCLUÍDA ===")
        except Exception as e:
            log.exception("Erro na ingestão: %s", e)
    t = threading.Thread(target=run_ingest, daemon=True)
    t.start()
    return jsonify({"status": "ingest_started", "message": "Verifique os logs do Railway"}), 202


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
