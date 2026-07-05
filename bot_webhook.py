"""
bot_webhook.py — API RAG (cerebro cientifico) consumida pelo uriel-lorena-bot

Este servico NAO atende WhatsApp. Toda a conversa (pacientes e supervisor)
acontece no uriel-lorena-bot, que consulta este servico via:

  POST /rag/ask       resposta educativa completa (guardrails CFM + fontes)
  POST /rag/retrieve  so o contexto de literatura + fontes

Autenticacao: header X-RAG-Token == RAG_BRIDGE_TOKEN (fail-closed).
Manutencao: /admin/ingest e /debug exigem X-Admin-Token == ADMIN_TOKEN.
"""
import os, time, hmac, logging, threading
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
# Auto-ingestao (opcao A): base se atualiza sozinha, sem comando manual.
AUTO_INGEST = os.getenv("AUTO_INGEST", "1") == "1"       # "0" desliga
INGEST_INTERVAL_DAYS = int(os.getenv("INGEST_INTERVAL_DAYS", "7"))
INGEST_PUBMED_MAX = int(os.getenv("INGEST_PUBMED_MAX", "30"))

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


_ingest_lock = threading.Lock()
_ingest_status = {"running": False, "last_ok": None, "last_error": None}


def _run_ingest():
    """Baixa PubMed + sites e reconstroi a base. Serializado por _ingest_lock."""
    if not _ingest_lock.acquire(blocking=False):
        log.info("Ingestao ja em andamento — ignorando novo pedido.")
        return
    try:
        _ingest_status["running"] = True
        from knee_loader import KneeKnowledgeLoader
        log.info("=== INGESTAO INICIADA ===")
        loader = KneeKnowledgeLoader(
            persist_dir=CHROMA_DIR,
            entrez_email=os.getenv("ENTREZ_EMAIL", "tiagoraggi@gmail.com"),
        )
        loader.ingest_pubmed(max_results=INGEST_PUBMED_MAX)
        loader.ingest_websites()
        loader.build_vectorstore()
        _ingest_status["last_ok"] = time.time()
        log.info("=== INGESTAO CONCLUIDA ===")
    except Exception as e:
        _ingest_status["last_error"] = str(e)
        log.exception("Erro na ingestao: %s", e)
    finally:
        _ingest_status["running"] = False
        _ingest_lock.release()


def _base_docs_count():
    try:
        return chain.vectorstore._collection.count()
    except Exception:
        return -1


def _auto_ingest_worker():
    """Opcao A: no boot popula a base se estiver vazia; depois reindexao semanal."""
    time.sleep(20)  # deixa o servico subir antes de puxar carga
    if _base_docs_count() == 0:
        log.info("Base vazia no boot — rodando ingestao inicial.")
        _run_ingest()
    intervalo = max(1, INGEST_INTERVAL_DAYS) * 86400
    while True:
        time.sleep(intervalo)
        log.info("Reindexacao periodica (a cada %d dia(s)).", INGEST_INTERVAL_DAYS)
        _run_ingest()


@app.route("/admin/ingest", methods=["POST"])
def admin_ingest():
    """Dispara ingestao manual (alem da automatica)."""
    if not _verify_token(request, "X-Admin-Token", ADMIN_TOKEN):
        return jsonify({"status": "unauthorized"}), 403
    threading.Thread(target=_run_ingest, daemon=True).start()
    return jsonify({"status": "ingest_started", "message": "Verifique os logs do Railway"}), 202


@app.route("/admin/ingest/status", methods=["GET"])
def admin_ingest_status():
    if not _verify_token(request, "X-Admin-Token", ADMIN_TOKEN):
        return jsonify({"status": "unauthorized"}), 403
    return jsonify({
        "running": _ingest_status["running"],
        "docs_na_base": _base_docs_count(),
        "last_ok": _ingest_status["last_ok"],
        "last_error": _ingest_status["last_error"],
        "auto_ingest": AUTO_INGEST,
        "intervalo_dias": INGEST_INTERVAL_DAYS,
    }), 200


if AUTO_INGEST:
    threading.Thread(target=_auto_ingest_worker, daemon=True, name="auto-ingest").start()
    log.info("Auto-ingestao ligada (intervalo=%d dia(s)).", INGEST_INTERVAL_DAYS)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
