"""
knee_prescriptions.py -- Gerenciamento de prescricoes e lembretes de medicacao via WhatsApp.

Comando: /receita: 55249XXXXXXX tramadol 50mg 1 comp VO de 8/8h em caso de dor por 5 dias
         /receita: 55249XXXXXXX pregabalina 75mg 1 comp VO as 21h por 30 dias

Regras:
- Lembretes somente entre 06:00 e 00:00
- Horario especifico ("as Xh") tem prioridade sobre intervalo
- Duracao padrao: 7 dias se nao informada
- "em caso de X" entra na mensagem do lembrete quando presente
"""

import os, re, logging, sqlite3
from datetime import datetime, timedelta

log = logging.getLogger("knee_prescriptions")

DB_PATH = os.getenv("PRESCRIPTIONS_DB", "/data/prescriptions.db")

HOUR_MIN = 6    # 06:00
HOUR_MAX = 24   # 00:00 (meia-noite)


# ----------------------------------------------------------------- banco

def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_phone   TEXT    NOT NULL,
            med_text        TEXT    NOT NULL,
            condition_text  TEXT    DEFAULT '',
            interval_hours  REAL    DEFAULT NULL,
            specific_hour   INTEGER DEFAULT NULL,
            duration_days   INTEGER DEFAULT 7,
            start_at        TEXT    NOT NULL,
            end_at          TEXT    NOT NULL,
            next_dose_at    TEXT    NOT NULL,
            active          INTEGER DEFAULT 1,
            created_at      TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ----------------------------------------------------------------- parsing

def _next_dose_in_window(base_dt, interval_hours, specific_hour):
    """
    Calcula a primeira dose a partir de base_dt que cai dentro da janela 06h-00h.
    - specific_hour: dose diaria nessa hora exata
    - interval_hours: proxima dose = base_dt + interval, deslocada se fora da janela
    """
    if specific_hour is not None:
        candidate = base_dt.replace(hour=specific_hour, minute=0, second=0, microsecond=0)
        if candidate <= base_dt:
            candidate += timedelta(days=1)
        return candidate

    candidate = base_dt
    for _ in range(48):
        if HOUR_MIN <= candidate.hour < HOUR_MAX:
            return candidate
        candidate = (candidate + timedelta(days=1)).replace(
            hour=HOUR_MIN, minute=0, second=0, microsecond=0
        )
    return candidate


def parse_prescription(text):
    """
    Extrai campos de uma prescricao a partir do texto apos '/receita:'.
    Retorna dict ou None se invalido.

    Exemplos:
      "55249XXXXXXX tramadol 50mg 1 comp VO de 8/8h em caso de dor por 5 dias"
      "55249XXXXXXX pregabalina 75mg 1 comp VO as 21h por 30 dias"
      "55249XXXXXXX dipirona 500mg 1 comp VO de 6/6h"
    """
    t = text.strip()

    # 1. Extrai telefone (primeiro token numerico >= 10 digitos)
    phone_match = re.match(r'^(\d{10,15})\s+', t)
    if not phone_match:
        return None
    patient_phone = phone_match.group(1)
    rest = t[phone_match.end():].strip()

    # 2. Duracao ("por X dias")
    duration_days = 7
    dur_match = re.search(r'\bpor\s+(\d+)\s*dias?\b', rest, re.IGNORECASE)
    if dur_match:
        duration_days = int(dur_match.group(1))
        rest = (rest[:dur_match.start()] + rest[dur_match.end():]).strip()

    # 3. Condicao ("em caso de ...")
    condition_text = ""
    cond_match = re.search(r'\bem\s+caso\s+de\s+(.+?)(?=\bpor\b|\bde\s+\d|\bas\b|$)',
                           rest, re.IGNORECASE)
    if cond_match:
        condition_text = "em caso de " + cond_match.group(1).strip().rstrip(",. ")
        rest = (rest[:cond_match.start()] + rest[cond_match.end():]).strip()

    # 4. Horario especifico -- SOMENTE quando "as/as" esta presente ou formato HH:MM
    #    Aceita: "as 21h", "as 21 horas", "as 21", "21:00"
    #    NAO aceita "8h" sozinho (isso eh intervalo)
    specific_hour = None
    hour_match = re.search(
        r'\b[aà]s\s+(\d{1,2})(?:\s*(?:h(?:oras?)?|:\d{2}))?\b'  # as X / as Xh / as X horas
        r'|(?<!\d)(\d{1,2}):\d{2}\b',                         # 21:00
        rest, re.IGNORECASE
    )
    if hour_match:
        h = int(hour_match.group(1) if hour_match.group(1) is not None else hour_match.group(2))
        if 0 <= h <= 23:
            specific_hour = h
            rest = (rest[:hour_match.start()] + rest[hour_match.end():]).strip()

    # 5. Intervalo ("de 8/8h", "de 6/6 horas", "8/8h", "6h")
    interval_hours = None
    if specific_hour is None:
        int_match = re.search(
            r'(?:de\s+)?(\d+)\s*/\s*(\d+)\s*h(?:oras?)?'   # de 8/8h ou 8/8h
            r'|(?:de\s+)?(\d+)\s*h(?:oras?)?'               # de 6h ou 6h ou 6horas
            r'|(?:de\s+)?(\d+)\s+h(?:oras?)',                # 6 horas (com espaco)
            rest, re.IGNORECASE
        )
        if int_match:
            if int_match.group(1):     # X/Xh
                interval_hours = float(int_match.group(1))
            elif int_match.group(3):   # Xh
                interval_hours = float(int_match.group(3))
            elif int_match.group(4):   # X horas
                interval_hours = float(int_match.group(4))
            rest = (rest[:int_match.start()] + rest[int_match.end():]).strip()

    # 6. Texto do medicamento = o que sobrar limpo
    med_text = re.sub(r'\s{2,}', ' ', rest).strip().rstrip(",.")
    if not med_text:
        return None

    return {
        "patient_phone":  patient_phone,
        "med_text":       med_text,
        "condition_text": condition_text,
        "interval_hours": interval_hours,
        "specific_hour":  specific_hour,
        "duration_days":  duration_days,
    }


# ----------------------------------------------------------------- CRUD

def add_prescription(patient_phone, med_text, condition_text,
                     interval_hours, specific_hour, duration_days):
    """Salva prescricao e retorna o ID."""
    init_db()
    now = datetime.utcnow()
    now_brt = now - timedelta(hours=3)
    end_at = now_brt + timedelta(days=duration_days)

    next_dose = _next_dose_in_window(
        now_brt + (timedelta(minutes=5) if interval_hours else timedelta()),
        interval_hours, specific_hour
    )
    next_dose_utc = next_dose + timedelta(hours=3)
    end_utc = end_at + timedelta(hours=3)

    conn = _conn()
    cur = conn.execute("""
        INSERT INTO prescriptions
          (patient_phone, med_text, condition_text, interval_hours, specific_hour,
           duration_days, start_at, end_at, next_dose_at, active, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,1,?)
    """, (patient_phone, med_text, condition_text,
          interval_hours, specific_hour, duration_days,
          now.isoformat(), end_utc.isoformat(),
          next_dose_utc.isoformat(), now.isoformat()))
    pid = cur.lastrowid
    conn.commit()
    conn.close()
    log.info("Prescricao #%d criada: %s para *%s", pid, med_text[:40], patient_phone[-4:])
    return pid


def get_due_prescriptions():
    """Retorna prescricoes ativas com next_dose_at <= agora."""
    try:
        init_db()
        conn = _conn()
        now_iso = datetime.utcnow().isoformat()
        rows = conn.execute("""
            SELECT * FROM prescriptions
            WHERE active=1 AND next_dose_at <= ? AND end_at >= ?
        """, (now_iso, now_iso)).fetchall()
        conn.close()
        return rows
    except Exception as e:
        log.warning("get_due_prescriptions: %s", e)
        return []


def advance_next_dose(prescription_id, interval_hours, specific_hour):
    """Atualiza next_dose_at para a proxima dose."""
    try:
        init_db()
        conn = _conn()
        row = conn.execute("SELECT next_dose_at FROM prescriptions WHERE id=?",
                           (prescription_id,)).fetchone()
        if not row:
            conn.close()
            return
        last_utc = datetime.fromisoformat(row["next_dose_at"])
        last_brt = last_utc - timedelta(hours=3)

        if specific_hour is not None:
            next_brt = last_brt + timedelta(days=1)
            next_brt = next_brt.replace(hour=specific_hour, minute=0, second=0, microsecond=0)
        else:
            next_brt = last_brt + timedelta(hours=interval_hours)
            if not (HOUR_MIN <= next_brt.hour < HOUR_MAX):
                next_brt = (next_brt + timedelta(days=1)).replace(
                    hour=HOUR_MIN, minute=0, second=0, microsecond=0
                )
        next_utc = next_brt + timedelta(hours=3)
        conn.execute("UPDATE prescriptions SET next_dose_at=? WHERE id=?",
                     (next_utc.isoformat(), prescription_id))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("advance_next_dose #%d: %s", prescription_id, e)


def deactivate_expired():
    """Desativa prescricoes vencidas."""
    try:
        init_db()
        conn = _conn()
        conn.execute("UPDATE prescriptions SET active=0 WHERE active=1 AND end_at < ?",
                     (datetime.utcnow().isoformat(),))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("deactivate_expired: %s", e)


def cancel_prescription(prescription_id):
    """Cancela prescricao pelo ID."""
    try:
        init_db()
        conn = _conn()
        conn.execute("UPDATE prescriptions SET active=0 WHERE id=?", (prescription_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def cancel_patient_prescriptions(patient_phone):
    """Cancela todas as prescricoes ativas de um paciente. Retorna quantidade."""
    try:
        init_db()
        conn = _conn()
        cur = conn.execute(
            "UPDATE prescriptions SET active=0 WHERE active=1 AND patient_phone=?",
            (patient_phone,)
        )
        count = cur.rowcount
        conn.commit()
        conn.close()
        return count
    except Exception:
        return 0


def list_active_prescriptions():
    """Lista todas as prescricoes ativas."""
    try:
        init_db()
        conn = _conn()
        rows = conn.execute("""
            SELECT * FROM prescriptions WHERE active=1
            ORDER BY patient_phone, next_dose_at
        """).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def format_active_prescriptions():
    """Formata lista de prescricoes ativas para exibicao no WhatsApp."""
    rows = list_active_prescriptions()
    if not rows:
        return "Nenhuma prescricao ativa no momento."
    lines = ["*Prescricoes ativas:*\n"]
    current_phone = None
    for r in rows:
        if r["patient_phone"] != current_phone:
            current_phone = r["patient_phone"]
            lines.append(f"*...{current_phone[-4:]}*")
        med = r["med_text"]
        cond = f" {r['condition_text']}" if r["condition_text"] else ""
        if r["specific_hour"] is not None:
            schedule = f"as {r['specific_hour']:02d}h diariamente"
        else:
            h = r["interval_hours"]
            if h and h == int(h):
                schedule = f"de {int(h)}/{int(h)}h"
            elif h:
                schedule = f"a cada {h}h"
            else:
                schedule = "horario nao detectado"
        end_brt = (datetime.fromisoformat(r["end_at"]) - timedelta(hours=3)).strftime("%d/%m")
        lines.append(f"  #{r['id']} {med}{cond} -- {schedule} ate {end_brt}")
    return "\n".join(lines)


def build_reminder_message(med_text, condition_text):
    """Monta a mensagem de lembrete para o paciente."""
    cond = f" {condition_text}" if condition_text else ""
    return f"E hora de tomar {med_text}{cond}."
