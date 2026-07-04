"""
bot_webhook.py — API RAG (cerebro cientifico) consumida pelo uriel-lorena-bot

Este servico NAO atende WhatsApp. Toda a conversa (pacientes e supervisor)
acontece no uriel-lorena-bot, que consulta este servico via:

  POST /rag/ask       resposta educativa completa (guardrails CFM + fontes)
  POST /rag/retrieve  so o contexto de literatura + fontes

Autenticacao: header X-RAG-Token == RAG_BRIDGE_TOKEN (fail-closed).
Manutencao: /admin/ingest e /debug exigem X-Admin-Token == ADMIN_TOKEN.
"""
import os, hmac, logging, threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from knee_retrieval_chain import KneeRAGChain, format_for_whatsapp

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag_api")
app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma_knee")
AUDIT_LOG = os.getenv("AUDIT_LOG", "/data/rag_audit.jsonl")
# Token da ponte RAG (o uriel-lorena-bot envia em X-RAG-Token). Fail-closed.
RAG_BRIDGE_TOKEN = os.getenv("RAG_BRIDGE_TOKEN", "")
# Token dos endpoints de manutencao (/admin/ingest e /debug). Fail-closed.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

if not GROQ_API_KEY:
    log.warning("Config ausente: GROQ_API_KEY")
if not RAG_BRIDGE_TOKEN:
    log.warning("RAG_BRIDGE_TOKEN nao definido — a ponte RAG fica bloqueada (403).")
if not ADMIN_TOKEN:
    log.warning("ADMIN_TOKEN nao definido — /admin/ingest e /debug ficam bloqueados (403).")

log.info("Inicializando KneeRAGChain...")
chain = KneeRAGChain(
    persist_dir=CHROMA_DIR,
    groq_api_key=GROQ_API_KEY,
    audit_log_path=AUDIT_LOG,
)
log.info("Chain pronta.")


def _verify_token(req, header: str, expected: str) -> bool:
    if not expected:
        return False
    provided = req.headers.get(header, "")
    return bool(provided) and hmac.compare_digest(provided, expected)


@app.route("/rag/ask", methods=["POST"])
def rag_ask():
    """Resposta educativa completa (guardrails CFM) para uma pergunta de paciente."""
    if not _verify_token(request, "X-RAG-Token", RAG_BRIDGE_TOKEN):
        return jsonify({"status": "unauthorized"}), 403
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"status": "bad_request", "detail": "question obrigatoria"}), 400
    try:
        result = chain.ask(
            question,
            patient_id_hash=body.get("patient_hash"),
            protocol_context=body.get("protocol_context") or None,
        )
        return jsonify({
            "status": "ok",
            "answer": format_for_whatsapp(result),
            "red_flag": result["red_flag"],
            "retrieved_count": result["retrieved_count"],
        }), 200
    except Exception as e:
        log.exception("rag_ask: %s", e)
        return jsonify({"status": "error"}), 500


@app.route("/rag/retrieve", methods=["POST"])
def rag_retrieve():
    """So o contexto recuperado (literatura + fontes), para o chamador montar a resposta."""
    if not _verify_token(request, "X-RAG-Token", RAG_BRIDGE_TOKEN):
        return jsonify({"status": "unauthorized"}), 403
    body = request.get_json(force=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"status": "bad_request", "detail": "question obrigatoria"}), 400
    try:
        results = chain.retrieve(question)
        context, sources = chain._format_context(results) if results else ("", [])
        return jsonify({"status": "ok", "context": context, "sources": sources}), 200
    except Exception as e:
        log.exception("rag_retrieve: %s", e)
        return jsonify({"status": "error"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "rag-api"}), 200


@app.route("/debug", methods=["GET"])
def debug():
    if not _verify_token(request, "X-Admin-Token", ADMIN_TOKEN):
        return jsonify({"status": "unauthorized"}), 403
    return jsonify({
        "chroma_dir": CHROMA_DIR,
        "chroma_exists": os.path.exists(CHROMA_DIR),
        "audit_log": AUDIT_LOG,
    }), 200


@app.route("/admin/ingest", methods=["POST"])
def admin_ingest():
    if not _verify_token(request, "X-Admin-Token", ADMIN_TOKEN):
        return jsonify({"status": "unauthorized"}), 403
    from knee_loader import KneeKnowledgeLoader
    def run_ingest():
        try:
            log.info("=== INGESTAO INICIADA ===")
            loader = KneeKnowledgeLoader(
                persist_dir=CHROMA_DIR,
                entrez_email=os.getenv("ENTREZ_EMAIL", "tiagoraggi@gmail.com"),
            )
            loader.ingest_pubmed(max_results=30)
            loader.ingest_websites()
            loader.build_vectorstore()
            log.info("=== INGESTAO CONCLUIDA ===")
        except Exception as e:
            log.exception("Erro na ingestao: %s", e)
    threading.Thread(target=run_ingest, daemon=True).start()
    return jsonify({"status": "ingest_started", "message": "Verifique os logs do Railway"}), 202


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
