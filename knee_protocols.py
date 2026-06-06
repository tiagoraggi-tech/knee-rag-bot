"""
knee_protocols.py — Gerenciamento de protocolos clínicos do Dr. Tiago
Salva protocolos e vínculos paciente→protocolo em SQLite.
"""
import sqlite3
import os
import logging
from datetime import datetime

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


def save_protocol(title: str, content: str) -> str:
    """Salva ou atualiza um protocolo. Retorna mensagem de confirmação."""
    init_db()
    title = title.strip()
    content = content.strip()
    with _conn() as c:
        c.execute("""
            INSERT INTO protocols (title, content)
            VALUES (?, ?)
            ON CONFLICT(title) DO UPDATE SET content=excluded.content, created_at=CURRENT_TIMESTAMP
        """, (title, content))
    log.info("Protocolo salvo: %s", title)
    return f"✅ Protocolo *{title}* salvo com sucesso."


def list_protocols() -> str:
    """Retorna string com todos os protocolos cadastrados."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title, content, created_at FROM protocols ORDER BY created_at DESC").fetchall()
    if not rows:
        return "Nenhum protocolo cadastrado ainda."
    lines = ["📋 *Protocolos cadastrados:*\n"]
    for title, content, created_at in rows:
        lines.append(f"*{title}*\n{content}\n_(salvo em {created_at[:10]})_\n")
    return "\n".join(lines)


def assign_protocol(patient_phone: str, protocol_title: str, assigned_by: str) -> str:
    """Vincula um protocolo a um número de paciente."""
    init_db()
    patient_phone = patient_phone.strip().lstrip("+")
    protocol_title = protocol_title.strip()

    # Verifica se o protocolo existe
    with _conn() as c:
        row = c.execute("SELECT title FROM protocols WHERE title = ?", (protocol_title,)).fetchone()
        if not row:
            # Tenta busca case-insensitive
            row = c.execute(
                "SELECT title FROM protocols WHERE LOWER(title) = LOWER(?)", (protocol_title,)
            ).fetchone()
        if not row:
            return f"❌ Protocolo *{protocol_title}* não encontrado. Use /ver_comandos para ver os disponíveis."
        real_title = row[0]
        c.execute("""
            INSERT INTO patient_protocols (patient_phone, protocol_title, assigned_by)
            VALUES (?, ?, ?)
            ON CONFLICT(patient_phone, protocol_title) DO UPDATE SET assigned_at=CURRENT_TIMESTAMP
        """, (patient_phone, real_title, assigned_by))
    log.info("Protocolo '%s' vinculado ao paciente %s", real_title, patient_phone[-4:])
    return f"✅ Protocolo *{real_title}* vinculado ao paciente {patient_phone[-4:]}."


def get_patient_protocols(patient_phone: str) -> list[dict]:
    """Retorna lista de protocolos vinculados a um paciente."""
    init_db()
    patient_phone = patient_phone.strip().lstrip("+")
    with _conn() as c:
        rows = c.execute("""
            SELECT p.title, p.content
            FROM patient_protocols pp
            JOIN protocols p ON pp.protocol_title = p.title
            WHERE pp.patient_phone = ?
            ORDER BY pp.assigned_at DESC
        """, (patient_phone,)).fetchall()
    return [{"title": t, "content": ct} for t, ct in rows]


def list_protocols_numbered() -> str:
    """Retorna lista numerada de protocolos (título + prévia de 60 chars)."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title, content FROM protocols ORDER BY id ASC").fetchall()
    if not rows:
        return "Nenhum protocolo cadastrado ainda."
    lines = ["📋 *Protocolos cadastrados:*\n"]
    for i, (title, content) in enumerate(rows, 1):
        preview = content[:60].replace("\n", " ")
        if len(content) > 60:
            preview += "…"
        lines.append(f"{i}. *{title}*\n   _{preview}_")
    lines.append("\n*Comandos disponíveis:*")
    lines.append("• */ver N* — conteúdo completo do protocolo #N")
    lines.append("• */editar N* — editar protocolo #N")
    lines.append("• */apagar N* — remover protocolo #N")
    lines.append("• */consultar N: pergunta* — IA responde sobre protocolo #N")
    lines.append("• */estudo* — sessão de estudo clínico com literatura (sem restrições)")
    lines.append("• */sair* — encerra sessão ativa")
    lines.append("• */instrucao Titulo: conteudo* — criar/atualizar protocolo")
    lines.append("• */Titulo: número_paciente* — vincular protocolo a paciente")
    return "\n".join(lines)


def get_protocol_by_index(n: int) -> tuple[str, str] | None:
    """Retorna (title, content) do protocolo na posição n (1-based), ou None."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title, content FROM protocols ORDER BY id ASC").fetchall()
    if 1 <= n <= len(rows):
        return rows[n - 1]
    return None


def delete_protocol(n: int) -> str:
    """Remove o protocolo na posição n (1-based)."""
    init_db()
    with _conn() as c:
        rows = c.execute("SELECT title FROM protocols ORDER BY id ASC").fetchall()
        if not (1 <= n <= len(rows)):
            return f"❌ Protocolo #{n} não existe."
        title = rows[n - 1][0]
        c.execute("DELETE FROM protocols WHERE title = ?", (title,))
        c.execute("DELETE FROM patient_protocols WHERE protocol_title = ?", (title,))
    log.info("Protocolo apagado: %s", title)
    return f"🗑️ Protocolo *{title}* apagado."


def format_protocols_as_context(protocols: list[dict]) -> str:
    """Formata protocolos como bloco de contexto prioritario para o LLM."""
    if not protocols:
        return ""
    lines = ["## PROTOCOLOS CLINICOS DO DR. TIAGO (PRIORIDADE MAXIMA)\n",
             "Siga EXATAMENTE estas instrucoes ao responder este paciente:\n"]
    for p in protocols:
        lines.append(f"### {p['title']}\n{p['content']}\n")
    return "\n".join(lines)
