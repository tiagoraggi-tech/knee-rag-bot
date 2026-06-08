"""
knee_protocols.py - Gerenciamento de protocolos clinicos do Dr. Tiago
Salva protocolos e vinculos paciente->protocolo em SQLite.
"""
import sqlite3
import os
import logging

log = logging.getLogger("knee_protocols")

DB_PATH = os.getenv("PROTOCOLS_DB", "/data/knee_protocols.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS protocols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS patient_protocols (
                patient_phone TEXT NOT NULL,
                protocol_title TEXT NOT NULL,
                assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                assigned_by TEXT,
                PRIMARY KEY (patient_phone, protocol_title)
            );
        """)


def save_protocol(title, content):
    init_db()
    title = title.strip()
    content = content.strip()
    with _conn() as c:
        c.execute("""
            INSERT INTO protocols (title, content) VALUES (?, ?)
            ON CONFLICT(title) DO UPDATE SET content=excluded.content, created_at=CURRENT_TIMESTAMP
        """, (title, content))
    log.info("Protocolo salvo: %s", title)
    return f"Protocolo *{title}* salvo com sucesso."


def list_protocols():
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title, content, created_at FROM protocols ORDER BY created_at DESC").fetchall()
    if not rows:
        return "Nenhum protocolo cadastrado ainda."
    lines = ["*Protocolos cadastrados:*\n"]
    for title, content, created_at in rows:
        lines.append(f"*{title}*\n{content}\n_(salvo em {created_at[:10]})_\n")
    return "\n".join(lines)


def assign_protocol(patient_phone, protocol_title, assigned_by):
    init_db()
    patient_phone = patient_phone.strip().lstrip("+")
    protocol_title = protocol_title.strip()
    with _conn() as c:
        row = c.execute("SELECT title FROM protocols WHERE title = ?", (protocol_title,)).fetchone()
        if not row:
            row = c.execute("SELECT title FROM protocols WHERE LOWER(title) = LOWER(?)", (protocol_title,)).fetchone()
        if not row:
            return f"Protocolo *{protocol_title}* nao encontrado."
        real_title = row[0]
        c.execute("""
            INSERT INTO patient_protocols (patient_phone, protocol_title, assigned_by)
            VALUES (?, ?, ?)
            ON CONFLICT(patient_phone, protocol_title) DO UPDATE SET assigned_at=CURRENT_TIMESTAMP
        """, (patient_phone, real_title, assigned_by))
    log.info("Protocolo '%s' vinculado ao paciente %s", real_title, patient_phone[-4:])
    return f"Protocolo *{real_title}* vinculado ao paciente ...{patient_phone[-4:]}."


def get_patient_protocols(patient_phone):
    init_db()
    patient_phone = patient_phone.strip().lstrip("+")
    with _conn() as c:
        rows = c.execute("""
            SELECT p.title, p.content FROM patient_protocols pp
            JOIN protocols p ON pp.protocol_title = p.title
            WHERE pp.patient_phone = ? ORDER BY pp.assigned_at DESC
        """, (patient_phone,)).fetchall()
    return [{"title": t, "content": ct} for t, ct in rows]


def list_protocols_numbered():
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title, content FROM protocols ORDER BY id ASC").fetchall()
    if not rows:
        return "Nenhum protocolo cadastrado ainda."
    lines = ["*Protocolos cadastrados:*\n"]
    for i, (title, content) in enumerate(rows, 1):
        preview = content[:60].replace("\n", " ")
        if len(content) > 60:
            preview += "..."
        lines.append(f"{i}. *{title}*\n   _{preview}_")
    lines.append("\n*Comandos disponiveis:*")
    lines.append("- */ver N* - conteudo completo do protocolo #N")
    lines.append("- */editar N* - editar protocolo #N")
    lines.append("- */apagar N* - remover protocolo #N")
    lines.append("- */consultar N: pergunta* - IA responde sobre protocolo #N")
    lines.append("- */estudo* - sessao de estudo clinico com literatura")
    lines.append("- */sair* - encerra sessao ativa")
    lines.append("- */instrucao Titulo: conteudo* - criar/atualizar protocolo")
    lines.append("- */receita: tel med de X/Xh por Y dias* - prescrever lembretes")
    lines.append("- */receitas* - listar prescricoes ativas")
    lines.append("- */cancelar_receita: ID_ou_tel* - cancelar prescricao")
    return "\n".join(lines)


def get_protocol_by_index(n):
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title, content FROM protocols ORDER BY id ASC").fetchall()
    if 1 <= n <= len(rows):
        return rows[n - 1]
    return None


def delete_protocol(n):
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title FROM protocols ORDER BY id ASC").fetchall()
        if not (1 <= n <= len(rows)):
            return f"Protocolo #{n} nao existe."
        title = rows[n - 1][0]
        c.execute("DELETE FROM protocols WHERE title = ?", (title,))
        c.execute("DELETE FROM patient_protocols WHERE protocol_title = ?", (title,))
    log.info("Protocolo apagado: %s", title)
    return f"Protocolo *{title}* apagado."


def list_patient_assignments():
    init_db()
    with _conn() as c:
        rows = c.execute("""
            SELECT patient_phone, GROUP_CONCAT(protocol_title, ', ') as protos
            FROM patient_protocols GROUP BY patient_phone
            ORDER BY MAX(assigned_at) DESC
        """).fetchall()
    if not rows:
        return "Nenhum paciente com protocolo vinculado."
    lines = ["*Pacientes com protocolos:*\n"]
    for i, (phone, protos) in enumerate(rows, 1):
        lines.append(f"{i}. *...{phone[-4:]}* ({phone})\n   {protos}")
    return "\n".join(lines)


def get_patient_detail(n):
    init_db()
    with _conn() as c:
        rows = c.execute("""
            SELECT patient_phone, GROUP_CONCAT(protocol_title, ', ') as protos
            FROM patient_protocols GROUP BY patient_phone
            ORDER BY MAX(assigned_at) DESC
        """).fetchall()
    if not (1 <= n <= len(rows)):
        return f"Paciente #{n} nao encontrado. Use /ver_pacientes."
    phone, protos = rows[n - 1]
    proto_list = protos.split(", ")
    lines = [f"*Paciente #{n}*", f"Tel: {phone}", "", "*Protocolos vinculados:*"]
    for p in proto_list:
        lines.append(f"- {p}")
    return "\n".join(lines)


def remove_patient_protocol(n, protocol_title):
    init_db()
    with _conn() as c:
        rows = c.execute("""
            SELECT patient_phone FROM patient_protocols
            GROUP BY patient_phone ORDER BY MAX(assigned_at) DESC
        """).fetchall()
    if not (1 <= n <= len(rows)):
        return f"Paciente #{n} nao encontrado."
    phone = rows[n - 1][0]
    with _conn() as c:
        result = c.execute(
            "DELETE FROM patient_protocols WHERE patient_phone=? AND LOWER(protocol_title)=LOWER(?)",
            (phone, protocol_title)
        )
        if result.rowcount == 0:
            return f"Protocolo *{protocol_title}* nao encontrado para paciente ...{phone[-4:]}."
    log.info("Protocolo '%s' removido do paciente %s", protocol_title, phone[-4:])
    return f"Protocolo *{protocol_title}* removido do paciente ...{phone[-4:]}."


def list_admin_help():
    protos = list_protocols_numbered()
    cmds = """*COMANDOS ADMIN — URIEL BOT*

*PROTOCOLOS CLINICOS*
/instrucao Titulo: conteudo  — salvar protocolo
/ver N                        — ver protocolo #N
/editar N                     — editar protocolo #N
/apagar N                     — apagar protocolo #N
/consultar N: pergunta        — consultar protocolo com IA
/N: 55249XXXXXXX              — vincular protocolo #N ao paciente
/ver_pacientes                — listar pacientes vinculados
/ver_paciente N               — detalhe do paciente #N
/remover_protocolo N: Titulo  — desvincular protocolo do paciente

*PRESCRICOES*
/receita: 55249XXX med posol  — prescrever (inline)
/receita: med posologia dur   — salvar template (sem tel)
/receita: 55249XXX 1 2 3      — aplicar templates ao paciente
/receita: 55249XXX 1 por Xd   — aplicar template com duracao
/templates                    — listar templates salvos
/apagar receita N             — apagar template #N
/receitas                     — listar prescricoes ativas
/cancelar_prescricao: ID|tel  — cancelar prescricao

*OUTROS*
/estudo                       — modo estudo clinico (RAG)
/ver_comandos                 — este painel

"""
    return cmds + protos


def get_protocol_by_number(n):
    proto = get_protocol_by_index(n)
    return proto[0] if proto else None


def delete_protocol_by_index(n):
    return delete_protocol(n)


def format_protocols_as_context(protocols):
    if not protocols:
        return ""
    lines = ["## PROTOCOLOS CLINICOS DO DR. TIAGO (PRIORIDADE MAXIMA)\n",
             "Siga EXATAMENTE estas instrucoes ao responder este paciente:\n"]
    for p in protocols:
        lines.append(f"### {p['title']}\n{p['content']}\n")
    return "\n".join(lines)
