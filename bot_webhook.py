"""
bot_webhook.py — Webhook Flask para integracao Evolution API + KneeRAGChain
"""
import os, re, json, time, hmac, hashlib, logging, threading, requests
from collections import OrderedDict
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from knee_retrieval_chain import KneeRAGChain, format_for_whatsapp
from knee_protocols import (
    save_protocol, list_protocols, assign_protocol,
    get_patient_protocols, format_protocols_as_context,
    list_patient_assignments, get_patient_detail, remove_patient_protocol,
    list_admin_help, get_protocol_by_number, get_protocol_by_index,
    delete_protocol_by_index,
)
from knee_prescriptions import (
    parse_prescription, add_prescription,
    get_due_prescriptions, advance_next_dose, deactivate_expired,
    cancel_prescription, cancel_patient_prescriptions,
    format_active_prescriptions, build_reminder_message, format_schedule,
    parse_receita_mode, save_template, list_templates,
    delete_template, apply_templates,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bot_webhook")
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuracao centralizada (falha cedo / avisa se faltar credencial critica)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EVOLUTION_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "uriel-bot")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
HANDOFF_NUMBER = os.getenv("HANDOFF_NUMBER", "5524988370406")
ADMIN_PHONE = os.getenv("ADMIN_PHONE", "5521999249903")
CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma_knee")
AUDIT_LOG = os.getenv("AUDIT_LOG", "/data/rag_audit.jsonl")
# Token que a Evolution API deve enviar no webhook (header X-Webhook-Token ou ?token=).
# Se vazio, o webhook NAO valida origem — defina em producao para bloquear payloads forjados.
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", "")
# Token dedicado para os endpoints de manutencao (/admin/ingest e /debug).
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

for _name in ("GROQ_API_KEY", "EVOLUTION_API_URL", "EVOLUTION_API_KEY"):
    if not os.getenv(_name):
        log.warning("Config ausente: %s (funcionalidade dependente vai falhar)", _name)
if not WEBHOOK_TOKEN:
    log.warning("WEBHOOK_TOKEN nao definido — o webhook aceita qualquer origem. "
                "Defina WEBHOOK_TOKEN e configure o header na Evolution API para proteger.")
if not ADMIN_TOKEN:
    log.warning("ADMIN_TOKEN nao definido — /admin/ingest e /debug ficam bloqueados (403).")

_seen_ids: OrderedDict = OrderedDict()
_seen_lock = threading.Lock()
_admin_sessions: dict = {}
_admin_sessions_lock = threading.Lock()
ADMIN_SESSION_TTL = 600
# IMPORTANTE: todo o estado de sessao vive em memoria e assume gunicorn --workers 1.
# Rodar com mais de 1 worker quebra sessoes de estudo/edicao e confirmacoes pendentes.
_protocol_sessions: dict = {}
_protocol_sessions_lock = threading.Lock()
PROTOCOL_SESSION_TTL = 600

_STUDY_SYSTEM = (
    "Voce e um assistente clinico de suporte ao Dr. Tiago Raggi, ortopedista.\n"
    "Seu interlocutor e o proprio medico, nao um paciente. Portanto:\n"
    "- Forneca dados clinicos completos: doses, posologia, protocolos cirurgicos, escalas, niveis de evidencia.\n"
    "- Cite artigos, guidelines e fontes da base de conhecimento disponivel.\n"
    "- Seja preciso e tecnico; use nomenclatura medica sem simplificacao.\n"
    "- Responda de forma conversacional — o Dr. Tiago esta estudando ou planejando conduta.\n"
    "- Apresente controversias, comparacoes entre tecnicas e limitacoes dos estudos quando relevante.\n"
    "- Nao adicione disclaimers de procure um medico — voce esta falando com o medico.\n"
    "- Se a base nao tiver informacao suficiente, diga claramente.\n"
    "- Portugues brasileiro, tom tecnico e collegial."
)

_STUDY_TEMPLATE = (
    "## LITERATURA RECUPERADA\n\n"
    "{context}\n\n"
    "## PERGUNTA / SOLICITACAO\n\n"
    "{question}\n\n"
    "Responda com profundidade clinica, citando as fontes recuperadas."
)


def _groq_chat(messages: list, model: str = "llama-3.3-70b-versatile",
               max_tokens: int = 800, temperature: float = 0.3, timeout: int = 30) -> str:
    """Chamada unica ao endpoint chat/completions do Groq. Levanta em erro HTTP."""
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _groq_study_turn(history: list, question: str, context: str) -> str:
    msgs = [{"role": "system", "content": _STUDY_SYSTEM}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": _STUDY_TEMPLATE.format(
        context=context or "Nenhum resultado relevante encontrado.", question=question)})
    try:
        return _groq_chat(msgs, model="llama-3.3-70b-versatile", max_tokens=1500, temperature=0.3, timeout=45)
    except Exception as e:
        log.error("Groq study error: %s", e)
        return "❌ Erro na consulta a IA. Tente novamente."


def _groq_query_protocol(title: str, content: str, question: str) -> str:
    sys_msg = "Voce e um assistente medico especializado. Responda a pergunta do Dr. Tiago com base no protocolo clinico fornecido."
    usr_msg = f"Protocolo: *{title}*\n\n{content}\n\nPergunta: {question}"
    try:
        return _groq_chat(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
            model="llama-3.3-70b-versatile", max_tokens=800, temperature=0.3, timeout=30)
    except Exception as e:
        log.error("Groq query error: %s", e)
        return "❌ Erro ao consultar IA. Tente novamente."


def _get_or_create_admin_session(phone: str) -> dict:
    with _admin_sessions_lock:
        sess = _admin_sessions.get(phone)
        if sess and (time.time() - sess["ts"]) > ADMIN_SESSION_TTL:
            del _admin_sessions[phone]; sess = None
        if not sess:
            _admin_sessions[phone] = {"history": [], "ts": time.time()}
        return _admin_sessions[phone]


def _reset_admin_session(phone: str):
    with _admin_sessions_lock:
        _admin_sessions.pop(phone, None)


def _purge_expired_protocol_sessions():
    """Remove sessoes de estudo/edicao/confirmacao inativas alem do TTL."""
    now = time.time()
    with _protocol_sessions_lock:
        for p in [p for p, s in _protocol_sessions.items()
                  if now - s.get("ts", now) > PROTOCOL_SESSION_TTL]:
            _protocol_sessions.pop(p, None)


def _normalize_admin_command(cmd: str) -> str:
    """Normaliza variacoes sem acento geradas pelo LLM para a grafia canonica."""
    _alias = {
        "/cancelar_prescricao:": "/cancelar_prescrição:",
        "/cancelar_receita:": "/cancelar_prescrição:",
    }
    lc = cmd.lower()
    for a, b in _alias.items():
        if lc.startswith(a):
            return b + cmd[len(a):]
    return cmd


def llm_interpret_admin_intent(text: str) -> dict:
    """
    Interpreta intenção do admin e normaliza para um comando executável.
    Retorna {"command": "/cmd...", "explanation": "desc"} ou {"command": None, "explanation": "msg"}
    """
    try:
        protos_ctx = list_protocols()
    except Exception:
        protos_ctx = "(indisponivel)"
    try:
        tmpls_ctx = list_templates()
    except Exception:
        tmpls_ctx = "(indisponivel)"
    try:
        pats_ctx = list_patient_assignments()
    except Exception:
        pats_ctx = "(indisponivel)"

    sys_prompt = (
        "Voce e o interpretador de comandos do Dr. Tiago Raggi (ortopedista).\n"
        "Dado o texto que ele enviou, determine o comando correto e normalize-o.\n"
        "Responda SOMENTE com JSON valido, sem texto fora do JSON.\n\n"
        "COMANDOS DISPONIVEIS:\n"
        "/instrucao {Titulo}: {conteudo}          salva protocolo clinico\n"
        "/{N}: {telefone}                          vincula protocolo #N ao paciente\n"
        "/ver_comandos                             lista todos os comandos\n"
        "/ver {N}                                  mostra protocolo #N\n"
        "/editar {N}                               edita protocolo #N\n"
        "/apagar {N}                               apaga protocolo #N\n"
        "/consultar {N}: {pergunta}                consulta protocolo #N com IA\n"
        "/ver_pacientes                            lista pacientes vinculados\n"
        "/ver_paciente {N}                         detalhe do paciente #N\n"
        "/remover_protocolo {N}: {Titulo}          desvincula protocolo do paciente\n"
        "/receita: {tel} {med} {posologia} {dur}   prescricao inline\n"
        "/receita: {texto sem tel}                 salva template de prescricao\n"
        "/receita: {tel} {N1} {N2} [por X dias]   aplica templates ao paciente\n"
        "/templates                                lista templates salvos\n"
        "/apagar receita {N}                       apaga template #N\n"
        "/cancelar_prescricao: {ID ou tel}         cancela prescricao (sem acento)\n"
        "/receitas                                 lista prescricoes ativas\n"
        "/estudo                                   ativa modo estudo clinico\n\n"
        "FORMATO JSON:\n"
        '{"command": "/comando completo normalizado", "explanation": "O que vai ser feito (pt-BR, 1 linha)"}\n'
        "Se nao conseguir mapear: {\"command\": null, \"explanation\": \"resposta ao Dr. Tiago\"}\n"
        "Se e uma pergunta clinica: {\"command\": \"/estudo\", \"explanation\": \"Ativar modo estudo\"}\n\n"
        f"DADOS ATUAIS:\nProtocolos: {protos_ctx}\nTemplates: {tmpls_ctx}\nPacientes: {pats_ctx}"
    )
    try:
        raw = _groq_chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": f'Dr. Tiago enviou: "{text}"'}],
            model="llama-3.1-8b-instant", max_tokens=250, temperature=0, timeout=10)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        log.warning("llm_interpret_admin_intent: %s", e)
    return {"command": None, "explanation": "Nao entendi. Use /ver_comandos para ver as opcoes."}


def _apoio_command_reference() -> str:
    """Tabela completa de comandos + dados ao vivo (protocolos/templates/pacientes)."""
    try:
        protos_ctx = list_protocols()
    except Exception:
        protos_ctx = "(indisponivel)"
    try:
        tmpls_ctx = list_templates()
    except Exception:
        tmpls_ctx = "(indisponivel)"
    try:
        pats_ctx = list_patient_assignments()
    except Exception:
        pats_ctx = "(indisponivel)"
    return (
        "TABELA COMPLETA DE COMANDOS:\n"
        "-- PROTOCOLOS --\n"
        "/instrucao {Titulo}: {conteudo}       salva novo protocolo clinico\n"
        "/{N}: {telefone}                       vincula protocolo #N ao paciente\n"
        "/ver {N}                               mostra conteudo do protocolo #N\n"
        "/editar {N}                            entra em modo edicao do protocolo #N\n"
        "/apagar {N}                            apaga protocolo #N permanentemente\n"
        "/consultar {N}: {pergunta}             pergunta ao LLM sobre o protocolo #N\n"
        "/ver_pacientes                         lista todos os pacientes vinculados\n"
        "/ver_paciente {N}                      detalhes do paciente na posicao #N\n"
        "/remover_protocolo {N}: {Titulo}       desvincula protocolo do paciente\n"
        "/estudo                                ativa modo estudo clinico (perguntas livres)\n\n"
        "-- PRESCRICOES --\n"
        "/receita: {tel} {med} {dose} {freq} por {X} dias   prescricao direta (inline)\n"
        "/receita: {texto sem telefone}         salva como template de prescricao\n"
        "/receita: {tel} {N1} {N2} ...          aplica templates #N1 #N2... ao paciente\n"
        "/receita: {tel} {N1} por {X} dias      aplica template com duracao customizada\n"
        "/templates                             lista todos os templates salvos\n"
        "/apagar receita {N}                    apaga template #N\n"
        "/cancelar_prescricao: {ID}             cancela prescricao pelo ID numerico\n"
        "/cancelar_prescricao: {telefone}       cancela todas as prescricoes ativas do paciente\n"
        "/receitas                              lista todas as prescricoes ativas\n\n"
        "-- OUTROS --\n"
        "/ver_comandos                          exibe este painel completo de comandos\n"
        "/apoio                                 abre este assistente de comandos\n\n"
        f"DADOS ATUAIS:\nProtocolos cadastrados: {protos_ctx}\n"
        f"Templates de prescricao: {tmpls_ctx}\n"
        f"Pacientes vinculados: {pats_ctx}"
    )


def llm_apoio_admin(query: str) -> str:
    """
    Atalho de uma pergunta (one-shot): explica em linguagem natural COMO usar o
    comando certo para o que o Dr. Tiago quer fazer, com o comando pronto para colar.
    """
    sys_prompt = (
        "Voce e o assistente de comandos do Dr. Tiago Raggi (ortopedista).\n"
        "Ele vai descrever o que quer fazer e voce deve:\n"
        "1. Identificar o(s) comando(s) necessario(s)\n"
        "2. Explicar brevemente o que cada um faz\n"
        "3. Fornecer o comando EXATO ja formatado e pronto para colar\n"
        "Responda em portugues, de forma direta e pratica. Use *negrito* para destacar comandos.\n\n"
        + _apoio_command_reference()
    )
    try:
        return _groq_chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query}],
            model="llama-3.3-70b-versatile", max_tokens=600, temperature=0.2, timeout=15)
    except Exception as e:
        log.warning("llm_apoio_admin: %s", e)
        return "❌ Não consegui consultar o assistente agora. Use /ver_comandos para ver a lista completa."


def llm_apoio_turn(history: list, user_msg: str) -> dict:
    """
    Turno conversacional do modo /apoio: mantem contexto, ajuda o Dr. Tiago a
    entender/montar o comando e, quando ha um comando unico pronto, devolve-o
    em `command` para oferecer execucao.
    Retorna {"reply": "<texto pt-BR>", "command": "<comando /... ou ''>"}.
    """
    sys_prompt = (
        "Voce e o assistente de comandos do Dr. Tiago Raggi (ortopedista), em modo conversa.\n"
        "Ajude-o a entender e montar o comando certo para o que ele quer. Mantenha o contexto.\n"
        "Responda SOMENTE com JSON valido, sem nenhum texto fora do JSON.\n"
        'FORMATO: {"reply": "<sua resposta em pt-BR, pode usar *negrito*>", '
        '"command": "<comando /... completo e pronto para executar, ou null>"}\n'
        "- Preencha 'command' APENAS quando houver UM unico comando concreto e pronto.\n"
        "- Se faltar informacao (telefone, dose, numero do protocolo...), pergunte no 'reply' e deixe command=null.\n"
        "- Nunca invente telefones ou numeros: use os dados atuais abaixo ou peca ao Dr. Tiago.\n\n"
        + _apoio_command_reference()
    )
    msgs = [{"role": "system", "content": sys_prompt}] + history + [{"role": "user", "content": user_msg}]
    try:
        raw = _groq_chat(msgs, model="llama-3.3-70b-versatile", max_tokens=700, temperature=0.2, timeout=20)
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {"reply": (data.get("reply") or "").strip(),
                    "command": (data.get("command") or "").strip()}
    except Exception as e:
        log.warning("llm_apoio_turn: %s", e)
    return {"reply": "❌ Não consegui processar agora. Reformule ou digite */sair*.", "command": ""}

MAX_SEEN = 2000
MAX_AGE_SECONDS = 300  # descarta mensagens com mais de 5 min (tolera deploys/reinicios curtos)

def _is_duplicate(msg_id: str) -> bool:
    with _seen_lock:
        if msg_id in _seen_ids: return True
        _seen_ids[msg_id] = True
        while len(_seen_ids) > MAX_SEEN: _seen_ids.popitem(last=False)
        return False

log.info("Inicializando KneeRAGChain...")
chain = KneeRAGChain(
    persist_dir=CHROMA_DIR,
    groq_api_key=GROQ_API_KEY,
    audit_log_path=AUDIT_LOG,
)
log.info("Chain pronta.")


def _verify_webhook_auth(req) -> bool:
    """Valida a origem do webhook. Se WEBHOOK_TOKEN nao estiver setado, nao bloqueia."""
    if not WEBHOOK_TOKEN:
        return True
    provided = req.headers.get("X-Webhook-Token") or req.args.get("token", "")
    return bool(provided) and hmac.compare_digest(provided, WEBHOOK_TOKEN)


def _verify_admin_token(req) -> bool:
    """Token dedicado para endpoints de manutencao. Fail-closed se nao configurado."""
    if not ADMIN_TOKEN:
        return False
    provided = req.headers.get("X-Admin-Token", "")
    return hmac.compare_digest(provided, ADMIN_TOKEN)


def send_whatsapp(phone: str, message: str) -> bool:
    url = f"{EVOLUTION_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {"Content-Type": "application/json", "apikey": EVOLUTION_API_KEY}
    payload = {"number": phone, "text": message}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        log.info("Enviado para %s", phone[-4:])
        return True
    except Exception as e:
        log.error("Falha ao enviar: %s", e); return False

@app.route("/webhook/messages-upsert", methods=["POST"])
@app.route("/webhook/messages-upsert/MESSAGES_UPSERT", methods=["POST"])
def messages_upsert():
    if not _verify_webhook_auth(request):
        log.warning("Webhook rejeitado: token invalido/ausente")
        return jsonify({"status": "unauthorized"}), 401
    try:
        data = request.get_json(force=True)
        msg_data = data.get("data", {})
        key = msg_data.get("key", {})
        if key.get("fromMe"): return jsonify({"status": "ignored_self"}), 200
        msg_id = key.get("id", "")
        if msg_id and _is_duplicate(msg_id): return jsonify({"status": "duplicate"}), 200
        msg_ts = msg_data.get("messageTimestamp", 0)
        if msg_ts and (time.time() - int(msg_ts)) > MAX_AGE_SECONDS: return jsonify({"status": "old_message"}), 200
        remote_jid = key.get("remoteJid", "")
        if "@g.us" in remote_jid: return jsonify({"status": "ignored_group"}), 200
        phone = remote_jid.replace("@s.whatsapp.net", "")
        message_obj = msg_data.get("message", {})
        text = (message_obj.get("conversation") or message_obj.get("extendedTextMessage", {}).get("text") or "").strip()
        if not text or not phone: return jsonify({"status": "no_text"}), 200
        log.info("Mensagem de %s: %s", phone[-4:], text[:80])
        is_admin = phone.lstrip("+") == ADMIN_PHONE.lstrip("+")
        text_lower = text.lower().strip()

        if is_admin:
            _purge_expired_protocol_sessions()

            # /estudo — ativa sessao de estudo clinico
            if text_lower == "/estudo":
                _protocol_sessions[phone] = {"mode": "study", "history": [], "ts": time.time()}
                reply = (
                    "🩺 *Modo estudo ativado.*\n\n"
                    "Faca suas perguntas clinicas — vou buscar na literatura do joelho\n"
                    "e responder com profundidade tecnica, sem restricoes de paciente.\n\n"
                    "A sessao mantem o contexto da conversa.\n"
                    "Digite */sair* para encerrar."
                )
                send_whatsapp(phone, reply)
                return jsonify({"status": "study_mode_started"}), 200

            # /apoio — abre sessao interativa do assistente de comandos
            if text_lower == "/apoio":
                _protocol_sessions[phone] = {"mode": "apoio", "history": [], "pending_command": None, "ts": time.time()}
                reply = (
                    "💡 *Modo apoio ativado.*\n\n"
                    "Me diga em portugues o que voce quer fazer e eu monto o comando certo pra voce.\n"
                    "Ex: _quero cancelar as receitas do paciente 5521999..._\n\n"
                    "Quando o comando estiver pronto eu ofereco pra executar (*sim* / *não*) — ou voce mesmo cola.\n"
                    "Digite */sair* para encerrar."
                )
                send_whatsapp(phone, reply)
                return jsonify({"status": "apoio_mode_started"}), 200

            # /sair — encerra sessao de estudo ou edicao
            if text_lower == "/sair":
                if phone in _protocol_sessions:
                    mode = _protocol_sessions.pop(phone).get("mode", "")
                    reply = f"✅ Sessao *{mode}* encerrada."
                else:
                    reply = "Nenhuma sessao de estudo/edicao ativa."
                send_whatsapp(phone, reply)
                return jsonify({"status": "session_closed"}), 200

            # Turno dentro da sessao de estudo
            psess = _protocol_sessions.get(phone)
            if psess and psess.get("mode") == "study":
                send_whatsapp(phone, "🔍 Buscando na literatura…")
                try:
                    results = chain.retrieve(text)
                    context_str, sources = chain._format_context(results) if results else ("", [])
                except Exception as e:
                    log.error("Retrieval error no /estudo: %s", e); context_str, sources = "", []
                answer = _groq_study_turn(psess["history"], text, context_str)
                if sources and "📚" not in answer:
                    fontes = "\n\n📚 *Fontes recuperadas:*\n"
                    for s in sources[:4]:
                        if s.get("url"): fontes += f"• {s['title'][:80]} — {s['url']}\n"
                    answer += fontes
                psess["history"].append({"role": "user", "content": text})
                psess["history"].append({"role": "assistant", "content": answer})
                if len(psess["history"]) > 20: psess["history"] = psess["history"][-20:]
                psess["ts"] = time.time()
                send_whatsapp(phone, answer)
                return jsonify({"status": "study_turn"}), 200

            # Modo edicao: aguardando novo conteudo
            if psess and psess.get("mode") == "editing":
                new_content = text.strip()
                if new_content.lower() == "/cancelar":
                    del _protocol_sessions[phone]
                    send_whatsapp(phone, "✏️ Edicao cancelada.")
                    return jsonify({"status": "edit_cancelled"}), 200
                reply = save_protocol(psess["title"], new_content)
                del _protocol_sessions[phone]
                send_whatsapp(phone, reply)
                return jsonify({"status": "protocol_edited"}), 200

            # Confirmacao pendente: admin responde "sim" ou "nao"
            psess2 = _protocol_sessions.get(phone)
            if psess2 and psess2.get("mode") == "pending_confirmation":
                answer = text_lower.strip().rstrip("!.")
                if answer in ("sim", "s", "confirmar", "confirma", "ok", "yes"):
                    pending_cmd = _normalize_admin_command(psess2["command"])
                    del _protocol_sessions[phone]
                    text = pending_cmd
                    text_lower = pending_cmd.lower().strip()
                    log.info("Admin confirmou comando: %s", pending_cmd[:60])
                    # Cai nos handlers abaixo com o comando normalizado
                elif answer in ("nao", "n", "cancelar", "cancel", "no", "nope"):
                    del _protocol_sessions[phone]
                    send_whatsapp(phone, "❌ Comando cancelado.")
                    return jsonify({"status": "cancelled"}), 200
                else:
                    # Trata como nova intenção: descarta confirmação anterior
                    del _protocol_sessions[phone]

            # Turno dentro da sessao /apoio (assistente de comandos conversacional)
            psess_apoio = _protocol_sessions.get(phone)
            if psess_apoio and psess_apoio.get("mode") == "apoio":
                _ans = text_lower.strip().rstrip("!.")
                _pending = psess_apoio.get("pending_command")
                if text.strip().startswith("/"):
                    # Comando real colado pelo Dr. Tiago: executa (cai nos handlers), mantem a sessao
                    psess_apoio["pending_command"] = None
                    psess_apoio["ts"] = time.time()
                elif _pending and _ans in ("sim", "s", "confirmar", "confirma", "ok", "yes", "pode", "manda"):
                    # Executa o comando que o assistente ofereceu, sem sair do modo apoio
                    text = _normalize_admin_command(_pending)
                    text_lower = text.lower().strip()
                    psess_apoio["pending_command"] = None
                    psess_apoio["ts"] = time.time()
                    log.info("Apoio: executando comando confirmado: %s", text[:60])
                    # Cai nos handlers abaixo com o comando normalizado
                elif _pending and _ans in ("nao", "n", "cancelar", "cancel", "no", "nope"):
                    psess_apoio["pending_command"] = None
                    psess_apoio["ts"] = time.time()
                    send_whatsapp(phone, "👍 Ok, não executei. Me diga o que ajustar ou */sair* para encerrar.")
                    return jsonify({"status": "apoio_declined"}), 200
                else:
                    # Turno de conversa: assistente ajuda a montar o comando
                    turn = llm_apoio_turn(psess_apoio["history"], text)
                    reply_text = turn["reply"] or "Pode me explicar de outro jeito o que voce quer fazer?"
                    cmd_ready = turn["command"]
                    psess_apoio["history"].append({"role": "user", "content": text})
                    psess_apoio["history"].append({"role": "assistant", "content": reply_text})
                    if len(psess_apoio["history"]) > 20:
                        psess_apoio["history"] = psess_apoio["history"][-20:]
                    if cmd_ready:
                        psess_apoio["pending_command"] = cmd_ready
                        reply_text += (f"\n\n➡️ Comando pronto:\n`{cmd_ready}`\n\n"
                                       "Quer que eu já execute? (*sim* / *não*) — ou cole voce mesmo.")
                    else:
                        psess_apoio["pending_command"] = None
                    psess_apoio["ts"] = time.time()
                    send_whatsapp(phone, reply_text)
                    return jsonify({"status": "apoio_turn"}), 200

            # /N: phone — vincula protocolo #N ao paciente
            _assign_m = re.match(r'^/(\d+):\s*(\d+)', text.strip())
            if _assign_m:
                _n = int(_assign_m.group(1)); _pnum = _assign_m.group(2).strip()
                _ptitle = get_protocol_by_number(_n)
                log.info("ASSIGN protocolo #%d ao paciente %s", _n, _pnum[-4:])
                reply = assign_protocol(_pnum, _ptitle, phone) if _ptitle else f"❌ Protocolo #{_n} nao encontrado."
                send_whatsapp(phone, reply)
                return jsonify({"status": "protocol_assigned"}), 200

            # /instrucao Titulo: conteudo
            if text_lower.startswith("/instrucao "):
                body = text[len("/instrucao "):].strip()
                if ":" in body:
                    title, cont = body.split(":", 1)
                    reply = save_protocol(title.strip(), cont.strip())
                else:
                    reply = "❌ Formato: /instrucao Titulo: conteudo"
                send_whatsapp(phone, reply)
                return jsonify({"status": "protocol_saved"}), 200

            # ver comandos
            if text_lower in ("/ver_comandos", "ver comandos"):
                send_whatsapp(phone, list_admin_help())
                return jsonify({"status": "commands_listed"}), 200

            # /ver N — conteudo completo do protocolo #N
            if text_lower.startswith("/ver "):
                raw = text[5:].strip()
                if raw.isdigit():
                    proto = get_protocol_by_index(int(raw))
                    reply = (f"📄 *{proto[0]}*\n\n{proto[1]}") if proto else f"❌ Protocolo #{raw} nao encontrado."
                else:
                    reply = "❌ Use: /ver N  (ex: /ver 2)"
                send_whatsapp(phone, reply)
                return jsonify({"status": "protocol_viewed"}), 200

            # /editar N — inicia edicao de protocolo
            if text_lower.startswith("/editar "):
                raw = text[8:].strip()
                if raw.isdigit():
                    proto = get_protocol_by_index(int(raw))
                    if proto:
                        _protocol_sessions[phone] = {"mode": "editing", "index": int(raw), "title": proto[0], "ts": time.time()}
                        reply = (f"✏️ *Editando: {proto[0]}*\n\n"
                                 f"Conteudo atual:\n{proto[1]}\n\n"
                                 "Envie o *novo conteudo completo* na proxima mensagem, ou */cancelar* para abortar.")
                    else:
                        reply = f"❌ Protocolo #{raw} nao encontrado."
                else:
                    reply = "❌ Use: /editar N  (ex: /editar 2)"
                send_whatsapp(phone, reply)
                return jsonify({"status": "edit_mode_started"}), 200

            # /apagar receita N — apaga template; /apagar N — remove protocolo
            if text_lower.startswith("/apagar "):
                raw = text[8:].strip()
                if raw.lower().startswith("receita "):
                    n_str = raw[8:].strip()
                    reply = delete_template(int(n_str)) if n_str.isdigit() else "❌ Use: /apagar receita N"
                else:
                    reply = delete_protocol_by_index(int(raw)) if raw.isdigit() else "❌ Use: /apagar N  (ex: /apagar 2)"
                send_whatsapp(phone, reply)
                return jsonify({"status": "deleted"}), 200

            # /consultar N: pergunta
            if text_lower.startswith("/consultar "):
                body = text[11:].strip()
                if ":" in body:
                    idx_str, question = body.split(":", 1)
                    if idx_str.strip().isdigit():
                        proto = get_protocol_by_index(int(idx_str.strip()))
                        if proto:
                            send_whatsapp(phone, f"🔍 Consultando protocolo *{proto[0]}*…")
                            reply = _groq_query_protocol(proto[0], proto[1], question.strip())
                        else:
                            reply = f"❌ Protocolo #{idx_str.strip()} nao encontrado."
                    else:
                        reply = "❌ Use: /consultar N: sua pergunta"
                else:
                    reply = "❌ Use: /consultar N: pergunta  (ex: /consultar 2: resumir em 3 pontos)"
                send_whatsapp(phone, reply)
                return jsonify({"status": "protocol_consulted"}), 200

            # /ver_pacientes
            if text_lower == "/ver_pacientes":
                send_whatsapp(phone, list_patient_assignments())
                return jsonify({"status": "patients_listed"}), 200

            # /ver_paciente N
            if text_lower.startswith("/ver_paciente "):
                arg = text[len("/ver_paciente "):].strip()
                try:
                    reply = get_patient_detail(int(arg))
                except ValueError:
                    reply = "❌ Use /ver_paciente N onde N e o numero do paciente em /ver_pacientes"
                send_whatsapp(phone, reply)
                return jsonify({"status": "patient_detail"}), 200

            # /remover_protocolo N: Titulo
            if text_lower.startswith("/remover_protocolo "):
                body = text[len("/remover_protocolo "):].strip()
                if ":" in body:
                    n_str, titulo = body.split(":", 1)
                    try:
                        reply = remove_patient_protocol(int(n_str.strip()), titulo.strip())
                    except ValueError:
                        reply = "❌ Use /remover_protocolo N: Titulo"
                else:
                    reply = "❌ Use /remover_protocolo N: Titulo"
                send_whatsapp(phone, reply)
                return jsonify({"status": "protocol_removed"}), 200

            # /receita: — salvar template, aplicar templates ou prescrever inline
            if text_lower.startswith("/receita:"):
                body = text[len("/receita:"):].strip()
                if not body:
                    reply = ("❌ Uso:\n"
                             "Salvar template: /receita: tramadol 50mg 1 comp de 8/8h por 5 dias\n"
                             "Aplicar:         /receita: 55249XXXXXXX 1 2 3\n"
                             "Inline:          /receita: 55249XXXXXXX tramadol 50mg 1 comp de 8/8h")
                    send_whatsapp(phone, reply)
                    return jsonify({"status": "receita_help"}), 200
                mode_result = parse_receita_mode(body)
                if mode_result[0] == 'save_template':
                    reply = save_template(mode_result[1])
                elif mode_result[0] == 'apply_templates':
                    _, pat_phone, tids, overrides_text = mode_result
                    results = apply_templates(pat_phone, tids, overrides_text)
                    lines = [f"📱 Paciente: *...{pat_phone[-4:]}*\n"]
                    for pid, msg in results:
                        lines.append(f"✅ Prescrição #{pid} — {msg}" if pid else f"❌ {msg}")
                    reply = "\n".join(lines)
                else:  # inline
                    _, pat_phone, rest_text = mode_result
                    parsed = parse_prescription(f"{pat_phone} {rest_text}")
                    if not parsed:
                        reply = ("❌ Formato:\n"
                                 "/receita: 55249XXXXXXX tramadol 50mg 1 comp de 8/8h por 5 dias\n"
                                 "/receita: 55249XXXXXXX pregabalina 75mg 1 comp às 21h por 30 dias")
                    else:
                        pid = add_prescription(
                            patient_phone=parsed["patient_phone"],
                            med_text=parsed["med_text"],
                            condition_text=parsed["condition_text"],
                            interval_hours=parsed["interval_hours"],
                            specific_hour=parsed["specific_hour"],
                            duration_days=parsed["duration_days"],
                        )
                        cond_str = f" {parsed['condition_text']}" if parsed["condition_text"] else ""
                        sched_str = format_schedule(parsed['specific_hour'], parsed['interval_hours'])
                        reply = (f"✅ Prescrição #{pid} criada\n"
                                 f"📱 Paciente: *...{parsed['patient_phone'][-4:]}*\n"
                                 f"💊 {parsed['med_text']}{cond_str}\n"
                                 f"⏱ {sched_str} por {parsed['duration_days']} dias")
                send_whatsapp(phone, reply)
                return jsonify({"status": "prescription_handled"}), 200

            # /receitas — lista prescrições ativas
            if text_lower in ("/receitas", "/ver_receitas"):
                send_whatsapp(phone, format_active_prescriptions())
                return jsonify({"status": "prescriptions_listed"}), 200

            # /apoio: — assistente de uso dos comandos
            if text_lower.startswith("/apoio:"):
                query = text[len("/apoio:"):].strip()
                if not query:
                    send_whatsapp(phone, "💡 Use: */apoio:* o que quero fazer\nEx: /apoio: quero cancelar as prescrições do paciente 5521999...\n\nOu digite só */apoio* para abrir uma conversa passo a passo.")
                    return jsonify({"status": "apoio_hint"}), 200
                resposta = llm_apoio_admin(query)
                send_whatsapp(phone, resposta)
                return jsonify({"status": "apoio_ok"}), 200

            # /templates — lista templates de prescrição
            if text_lower == "/templates":
                send_whatsapp(phone, list_templates())
                return jsonify({"status": "templates_listed"}), 200

            # /cancelar_prescricao: ou /cancelar_prescrição: (ambas grafias aceitas)
            _cancel_rx = None
            if text_lower.startswith("/cancelar_prescrição:"):
                _cancel_rx = text[len("/cancelar_prescrição:"):].strip()
            elif text_lower.startswith("/cancelar_prescricao:"):
                _cancel_rx = text[len("/cancelar_prescricao:"):].strip()
            if _cancel_rx is not None:
                arg = _cancel_rx
                # Número com >= 10 dígitos é sempre TELEFONE, não ID de prescrição.
                # IDs são pequenos inteiros sequenciais (1, 2, 3...).
                if arg.isdigit() and len(arg) < 10:
                    ok = cancel_prescription(int(arg))
                    reply = f"✅ Prescrição #{arg} cancelada." if ok else f"❌ Prescrição #{arg} não encontrada."
                elif len(arg) >= 10:
                    # Telefone (com ou sem dígitos apenas) — cancela todas as prescrições ativas do paciente
                    count = cancel_patient_prescriptions(arg)
                    reply = (f"✅ {count} prescrição(ões) cancelada(s) para *...{arg[-4:]}*."
                             if count else f"❌ Nenhuma prescrição ativa para *...{arg[-4:]}*.")
                else:
                    reply = "❌ Use: /cancelar_prescrição: ID  ou  /cancelar_prescrição: 55249XXXXXXX"
                send_whatsapp(phone, reply)
                return jsonify({"status": "prescription_cancelled"}), 200

            # Interpretador LLM unificado: qualquer mensagem que chegou ate aqui
            # nao foi reconhecida como comando exato — LLM interpreta a intencao
            intent = llm_interpret_admin_intent(text)
            cmd = (intent.get("command") or "").strip()
            explanation = intent.get("explanation", "")

            # Comando especial: /estudo direto (nao precisa de confirmacao)
            if cmd == "/estudo":
                _protocol_sessions[phone] = {"mode": "study", "history": [], "ts": time.time()}
                reply = (
                    "🩺 *Modo estudo ativado.*\n\n"
                    "Faca suas perguntas clinicas — vou buscar na literatura do joelho\n"
                    "e responder com profundidade tecnica.\n"
                    "Digite */sair* para encerrar."
                )
                send_whatsapp(phone, reply)
                return jsonify({"status": "study_mode_started_llm"}), 200

            if cmd:
                _protocol_sessions[phone] = {"mode": "pending_confirmation", "command": cmd, "ts": time.time()}
                reply = (f"🤖 Interpretei como:\n"
                         f"`{cmd}`\n\n"
                         f"_{explanation}_\n\n"
                         "Confirma? (*sim* / *não*)")
                send_whatsapp(phone, reply)
                return jsonify({"status": "intent_pending"}), 200

            # LLM nao conseguiu mapear
            send_whatsapp(phone, explanation or "Nao entendi. Use /ver_comandos para ver as opcoes.")
            return jsonify({"status": "unknown_command"}), 200

        # Resposta ao paciente
        phone_hash = hashlib.sha256(phone.encode()).hexdigest()
        patient_protos = get_patient_protocols(phone)
        protocol_context = format_protocols_as_context(patient_protos) if patient_protos else ""
        result = chain.ask(text, patient_id_hash=phone_hash, protocol_context=protocol_context)
        reply = format_for_whatsapp(result)
        if result["red_flag"]:
            handoff_msg = f"🚨 RED FLAG detectada\nDe: {phone}\nMsg: {text[:200]}\nHash: {phone_hash[:12]}"
            send_whatsapp(HANDOFF_NUMBER, handoff_msg)
        send_whatsapp(phone, reply)
        return jsonify({"status": "ok", "red_flag": result["red_flag"]}), 200

    except Exception as e:
        log.exception("Erro no webhook: %s", e)
        return jsonify({"status": "error"}), 500

@app.route("/health", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/debug", methods=["GET"])
def debug():
    if not _verify_admin_token(request):
        return jsonify({"status": "unauthorized"}), 403
    return jsonify({"chroma_dir": CHROMA_DIR, "chroma_exists": os.path.exists(CHROMA_DIR),
        "audit_log": AUDIT_LOG,
        "evolution_url": EVOLUTION_URL, "instance": EVOLUTION_INSTANCE})

@app.route("/admin/ingest", methods=["POST"])
def admin_ingest():
    if not _verify_admin_token(request):
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
        except Exception as e: log.exception("Erro na ingestao: %s", e)
    threading.Thread(target=run_ingest, daemon=True).start()
    return jsonify({"status": "ingest_started", "message": "Verifique os logs do Railway"}), 202

def _prescription_reminder_worker():
    """Thread de fundo: envia lembretes de medicacao a cada 5 minutos."""
    while True:
        try:
            deactivate_expired()
            for row in get_due_prescriptions():
                msg = build_reminder_message(row["med_text"], row["condition_text"])
                if send_whatsapp(row["patient_phone"], msg):
                    advance_next_dose(row["id"], row["interval_hours"], row["specific_hour"])
                    log.info("Lembrete enviado para *%s: %s",
                             row["patient_phone"][-4:], row["med_text"][:30])
        except Exception as e:
            log.warning("prescription_reminder_worker: %s", e)
        time.sleep(300)  # 5 minutos

threading.Thread(
    target=_prescription_reminder_worker,
    daemon=True,
    name="prescription-reminders"
).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
