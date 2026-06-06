"""
bot_webhook.py — Webhook Flask para integracao Evolution API + KneeRAGChain
"""
import os, time, hashlib, logging, threading, requests
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

_seen_ids: OrderedDict = OrderedDict()
_seen_lock = threading.Lock()
_admin_sessions: dict = {}
_admin_sessions_lock = threading.Lock()
ADMIN_SESSION_TTL = 600
_protocol_sessions: dict = {}

_STUDY_SYSTEM = (
    "Voce e um assistente clinico de suporte ao Dr. Tiago Raggi, ortopedista." + chr(10) +
    "Seu interlocutor e o proprio medico, nao um paciente. Portanto:" + chr(10) +
    "- Forneca dados clinicos completos: doses, posologia, protocolos cirurgicos, escalas, niveis de evidencia." + chr(10) +
    "- Cite artigos, guidelines e fontes da base de conhecimento disponivel." + chr(10) +
    "- Seja preciso e tecnico; use nomenclatura medica sem simplificacao." + chr(10) +
    "- Responda de forma conversacional — o Dr. Tiago esta estudando ou planejando conduta." + chr(10) +
    "- Apresente controversias, comparacoes entre tecnicas e limitacoes dos estudos quando relevante." + chr(10) +
    "- Nao adicione disclaimers de procure um medico — voce esta falando com o medico." + chr(10) +
    "- Se a base nao tiver informacao suficiente, diga claramente." + chr(10) +
    "- Portugues brasileiro, tom tecnico e collegial."
)

_STUDY_TEMPLATE = (
    "## LITERATURA RECUPERADA" + chr(10) + chr(10) +
    "{context}" + chr(10) + chr(10) +
    "## PERGUNTA / SOLICITACAO" + chr(10) + chr(10) +
    "{question}" + chr(10) + chr(10) +
    "Responda com profundidade clinica, citando as fontes recuperadas."
)

def _groq_study_turn(history: list, question: str, context: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    hdrs = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY','')}", "Content-Type": "application/json"}
    msgs = [{"role": "system", "content": _STUDY_SYSTEM}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": _STUDY_TEMPLATE.format(
        context=context or "Nenhum resultado relevante encontrado.", question=question)})
    payload = {"model": "llama-3.3-70b-versatile", "messages": msgs, "max_tokens": 1500, "temperature": 0.3}
    try:
        r = requests.post(url, headers=hdrs, json=payload, timeout=45)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("Groq study error: %s", e)
        return "❌ Erro na consulta a IA. Tente novamente."

def _groq_query_protocol(title: str, content: str, question: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    hdrs = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY','')}", "Content-Type": "application/json"}
    sys_msg = "Voce e um assistente medico especializado. Responda a pergunta do Dr. Tiago com base no protocolo clinico fornecido."
    usr_msg = f"Protocolo: *{title}*" + chr(10) + chr(10) + content + chr(10) + chr(10) + f"Pergunta: {question}"
    payload = {"model": "llama-3.3-70b-versatile", "messages": [
        {"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}],
        "max_tokens": 800, "temperature": 0.3}
    try:
        r = requests.post(url, headers=hdrs, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
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

def handle_admin_patient_session(phone: str, text: str) -> str | None:
    import re as _re3, json as _json3, requests as _req3
    PATIENT_KW = ["paciente","ver_paciente","ver_pacientes","remov","desvincul","vinculado","quem tem","listar pacient"]
    CREATION_KW = ["cadastr","cri","novo protocolo","adicionar protocolo","instrucao"]
    if any(k in text.lower() for k in CREATION_KW): return None
    sess = _get_or_create_admin_session(phone)
    if not sess["history"] and not any(k in text.lower() for k in PATIENT_KW): return None
    from knee_protocols import list_patient_assignments as _lpa, get_patient_detail as _gpd, remove_patient_protocol as _rpp, list_protocols as _lp
    sys_prompt = (
        "Voce e o assistente de gestao de pacientes do Dr. Tiago Raggi." + chr(10) +
        "Interprete o que ele quer fazer e execute ou peca esclarecimento." + chr(10) + chr(10) +
        f"DADOS ATUAIS:{chr(10)}{_lpa()}{chr(10)}{chr(10)}{_lp()}{chr(10)}{chr(10)}" +
        "REGRAS:" + chr(10) +
        "- Responda SOMENTE com JSON valido, sem texto fora do JSON" + chr(10) +
        '- Formato: {"action":"reply"|"get_patient"|"remove_protocol","patient_n":<int ou null>,"protocol":"<titulo ou null>","message":"<resposta pt-BR>"}' + chr(10) +
        "- reply: apenas responde/pergunta (sem executar)" + chr(10) +
        "- get_patient: mostra protocolos do paciente N (patient_n obrigatorio)" + chr(10) +
        "- remove_protocol: remove protocolo do paciente (patient_n + protocol obrigatorios)" + chr(10) +
        "- Antes de remover SEMPRE peca confirmacao com reply" + chr(10) +
        "- NUNCA finja cadastrar protocolos — use /instrucao Titulo: conteudo"
    )
    history = sess["history"] + [{"role": "user", "content": text}]
    msgs_llm = [{"role": "system", "content": sys_prompt}] + history
    try:
        resp = _req3.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant", "messages": msgs_llm, "max_tokens": 300, "temperature": 0},
            timeout=10)
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        m = _re3.search(r'{.*}', raw, _re3.DOTALL)
        parsed = _json3.loads(m.group()) if m else None
    except Exception as e:
        log.warning("Admin patient LLM error: %s", e); _reset_admin_session(phone); return None
    if not parsed: _reset_admin_session(phone); return None
    action = parsed.get("action", "reply")
    patient_n = parsed.get("patient_n")
    protocol = parsed.get("protocol")
    message = parsed.get("message", "")
    if action == "get_patient" and patient_n:
        reply = _gpd(int(patient_n)); _reset_admin_session(phone)
    elif action == "remove_protocol" and patient_n and protocol:
        reply = _rpp(int(patient_n), str(protocol)); _reset_admin_session(phone)
    else:
        reply = message
        sess["history"].append({"role": "user", "content": text})
        sess["history"].append({"role": "assistant", "content": raw})
        sess["ts"] = time.time()
        if len(sess["history"]) > 10: sess["history"] = sess["history"][-10:]
    return reply or message


def llm_interpret_admin_intent(text: str) -> dict:
    """
    Interpreta intenção do admin e normaliza para um comando executável.
    Retorna {"command": "/cmd...", "explanation": "desc"} ou {"command": None, "explanation": "msg"}
    """
    import re as _re_i, json as _json_i
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
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY','')}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "system", "content": sys_prompt},
                                {"role": "user", "content": f'Dr. Tiago enviou: "{text}"'}],
                  "max_tokens": 250, "temperature": 0},
            timeout=10
        )
        raw = r.json()["choices"][0]["message"]["content"].strip()
        m = _re_i.search(r'\{.*\}', raw, _re_i.DOTALL)
        if m:
            return _json_i.loads(m.group())
    except Exception as e:
        log.warning("llm_interpret_admin_intent: %s", e)
    return {"command": None, "explanation": "Nao entendi. Use /ver_comandos para ver as opcoes."}

MAX_SEEN = 2000
MAX_AGE_SECONDS = 60

def _is_duplicate(msg_id: str) -> bool:
    with _seen_lock:
        if msg_id in _seen_ids: return True
        _seen_ids[msg_id] = True
        while len(_seen_ids) > MAX_SEEN: _seen_ids.popitem(last=False)
        return False

log.info("Inicializando KneeRAGChain...")
chain = KneeRAGChain(
    persist_dir=os.getenv("CHROMA_DIR", "/data/chroma_knee"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    audit_log_path=os.getenv("AUDIT_LOG", "/data/rag_audit.jsonl"),
)
log.info("Chain pronta.")

EVOLUTION_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "uriel-bot")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY")
HANDOFF_NUMBER = os.getenv("HANDOFF_NUMBER", "5524988370406")
ADMIN_PHONE = os.getenv("ADMIN_PHONE", "5521999249903")

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
            import re as _re_admin

            # /estudo — ativa sessao de estudo clinico
            if text_lower == "/estudo":
                _protocol_sessions[phone] = {"mode": "study", "history": []}
                reply = (
                    "🩺 *Modo estudo ativado.*" + chr(10) + chr(10) +
                    "Faca suas perguntas clinicas — vou buscar na literatura do joelho" + chr(10) +
                    "e responder com profundidade tecnica, sem restricoes de paciente." + chr(10) + chr(10) +
                    "A sessao mantem o contexto da conversa." + chr(10) +
                    "Digite */sair* para encerrar."
                )
                send_whatsapp(phone, reply)
                return jsonify({"status": "study_mode_started"}), 200

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
                    fontes = chr(10) + chr(10) + "📚 *Fontes recuperadas:*" + chr(10)
                    for s in sources[:4]:
                        if s.get("url"): fontes += f"• {s['title'][:80]} — {s['url']}" + chr(10)
                    answer += fontes
                psess["history"].append({"role": "user", "content": text})
                psess["history"].append({"role": "assistant", "content": answer})
                if len(psess["history"]) > 20: psess["history"] = psess["history"][-20:]
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
                    pending_cmd = psess2["command"]
                    del _protocol_sessions[phone]
                    # Normaliza variações sem acento geradas pelo LLM
                    _alias = {
                        "/cancelar_prescricao:": "/cancelar_prescrição:",
                        "/cancelar_receita:": "/cancelar_prescrição:",
                    }
                    _lc = pending_cmd.lower()
                    for _a, _b in _alias.items():
                        if _lc.startswith(_a):
                            pending_cmd = _b + pending_cmd[len(_a):]
                            break
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

            # /N: phone — vincula protocolo #N ao paciente
            _assign_m = _re_admin.match(r'^/(\d+):\s*(\d+)', text.strip())
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
                    reply = (f"📄 *{proto[0]}*" + chr(10) + chr(10) + proto[1]) if proto else f"❌ Protocolo #{raw} nao encontrado."
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
                        _protocol_sessions[phone] = {"mode": "editing", "index": int(raw), "title": proto[0]}
                        reply = (f"✏️ *Editando: {proto[0]}*" + chr(10) + chr(10) +
                                 f"Conteudo atual:" + chr(10) + proto[1] + chr(10) + chr(10) +
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

            # /templates — lista templates de prescrição
            if text_lower == "/templates":
                send_whatsapp(phone, list_templates())
                return jsonify({"status": "templates_listed"}), 200

            # /cancelar_prescrição: ID ou telefone
            if text_lower.startswith("/cancelar_prescrição:"):
                arg = text[len("/cancelar_prescrição:"):].strip()
                if arg.isdigit():
                    ok = cancel_prescription(int(arg))
                    reply = f"✅ Prescrição #{arg} cancelada." if ok else f"❌ Prescrição #{arg} não encontrada."
                elif len(arg) >= 10:
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
                _protocol_sessions[phone] = {"mode": "study", "history": []}
                reply = (
                    "🩺 *Modo estudo ativado.*" + chr(10) + chr(10) +
                    "Faca suas perguntas clinicas — vou buscar na literatura do joelho" + chr(10) +
                    "e responder com profundidade tecnica." + chr(10) +
                    "Digite */sair* para encerrar."
                )
                send_whatsapp(phone, reply)
                return jsonify({"status": "study_mode_started_llm"}), 200

            if cmd:
                _protocol_sessions[phone] = {"mode": "pending_confirmation", "command": cmd}
                reply = (f"🤖 Interpretei como:" + chr(10) +
                         f"`{cmd}`" + chr(10) + chr(10) +
                         f"_{explanation}_" + chr(10) + chr(10) +
                         "Confirma? (*sim* / *não*)")
                send_whatsapp(phone, reply)
                return jsonify({"status": "intent_pending"}), 200

            # LLM nao conseguiu mapear
            send_whatsapp(phone, explanation or "Nao entendi. Use /ver_comandos para ver as opcoes.")
            return jsonify({"status": "unknown_command"}), 200

        # Resposta ao paciente
        phone_hash = hashlib.md5(phone.encode()).hexdigest()
        patient_protos = get_patient_protocols(phone)
        protocol_context = format_protocols_as_context(patient_protos) if patient_protos else ""
        result = chain.ask(text, patient_id_hash=phone_hash, protocol_context=protocol_context)
        reply = format_for_whatsapp(result)
        if result["red_flag"]:
            handoff_msg = "🚨 RED FLAG detectada" + chr(10) + f"De: {phone}" + chr(10) + f"Msg: {text[:200]}" + chr(10) + f"Hash: {phone_hash[:12]}"
            send_whatsapp(HANDOFF_NUMBER, handoff_msg)
        send_whatsapp(phone, reply)
        return jsonify({"status": "ok", "red_flag": result["red_flag"]}), 200

    except Exception as e:
        log.exception("Erro no webhook: %s", e)
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/health", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/debug", methods=["GET"])
def debug():
    chroma_dir = os.getenv("CHROMA_DIR", "/data/chroma_knee")
    import os as _os
    return jsonify({"chroma_dir": chroma_dir, "chroma_exists": _os.path.exists(chroma_dir),
        "audit_log": os.getenv("AUDIT_LOG", "/data/rag_audit.jsonl"),
        "evolution_url": EVOLUTION_URL, "instance": EVOLUTION_INSTANCE})

@app.route("/admin/ingest", methods=["POST"])
def admin_ingest():
    token = request.headers.get("X-Admin-Token", "")
    if token != os.getenv("GROQ_API_KEY", ""): return jsonify({"status": "unauthorized"}), 403
    from knee_loader import KneeKnowledgeLoader
    def run_ingest():
        try:
            log.info("=== INGESTAO INICIADA ===")
            loader = KneeKnowledgeLoader(
                persist_dir=os.getenv("CHROMA_DIR", "/data/chroma_knee"),
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
    import time as _time
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
        _time.sleep(300)  # 5 minutos

threading.Thread(
    target=_prescription_reminder_worker,
    daemon=True,
    name="prescription-reminders"
).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
