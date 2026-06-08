"""
knee_prescriptions.py -- Gerenciamento de prescricoes e lembretes de medicacao via WhatsApp.
"""

import os, re, logging, sqlite3
from datetime import datetime, timedelta

log = logging.getLogger("knee_prescriptions")

DB_PATH = os.getenv("PRESCRIPTIONS_DB", "/data/prescriptions.db")

HOUR_MIN = 6
HOUR_MAX = 24

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prescription_templates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            text            TEXT    NOT NULL,
            med_text        TEXT    NOT NULL,
            condition_text  TEXT    DEFAULT '',
            interval_hours  REAL    DEFAULT NULL,
            specific_hour   INTEGER DEFAULT NULL,
            duration_days   INTEGER DEFAULT 7,
            created_at      TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ----------------------------------------------------------------- helpers

def format_schedule(specific_hour, interval_hours):
    """Formata horario/intervalo para exibicao."""
    if specific_hour is not None:
        return f"as {specific_hour:02d}h"
    h = interval_hours
    if h and h == int(h):
        return f"de {int(h)}/{int(h)}h"
    elif h:
        return f"a cada {h}h"
    return "horario nao definido"


# ----------------------------------------------------------------- parsing

def _next_dose_in_window(base_dt, interval_hours, specific_hour):
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
    """
    t = text.strip()

    # 1. Telefone
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
    cond_match = re.search(r'\bem\s+caso\s+de\s+(.+?)(?=\bpor\b|\bde\s+\d|\b[aà]s\b|$)',
                           rest, re.IGNORECASE)
    if cond_match:
        condition_text = "em caso de " + cond_match.group(1).strip().rstrip(",. ")
        rest = (rest[:cond_match.start()] + rest[cond_match.end():]).strip()

    # 4. Horario especifico (somente quando "as/às" presente ou HH:MM)
    specific_hour = None
    hour_match = re.search(
        r'\b[aà]s\s+(\d{1,2})(?:\s*(?:h(?:oras?)?|:\d{2}))?\b'
        r'|(?<!\d)(\d{1,2}):\d{2}\b',
        rest, re.IGNORECASE
    )
    if hour_match:
        h = int(hour_match.group(1) if hour_match.group(1) is not None else hour_match.group(2))
        if 0 <= h <= 23:
            specific_hour = h
            rest = (rest[:hour_match.start()] + rest[hour_match.end():]).strip()

    # 5. Intervalo ("de 8/8h", "6h", "6 horas")
    interval_hours = None
    if specific_hour is None:
        int_match = re.search(
            r'(?:de\s+)?(\d+)\s*/\s*(\d+)\s*h(?:oras?)?'
            r'|(?:de\s+)?(\d+)\s*h(?:oras?)?'
            r'|(?:de\s+)?(\d+)\s+h(?:oras?)',
            rest, re.IGNORECASE
        )
        if int_match:
            if int_match.group(1):
                interval_hours = float(int_match.group(1))
            elif int_match.group(3):
                interval_hours = float(int_match.group(3))
            elif int_match.group(4):
                interval_hours = float(int_match.group(4))
            rest = (rest[:int_match.start()] + rest[int_match.end():]).strip()

    # 6. Texto do medicamento
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


def parse_receita_mode(text):
    """
    Determina o modo do comando /receita:
    Retorna:
      ('save_template', template_text)
      ('apply_templates', phone, [ids], overrides_text)
      ('inline', phone, rest_text)
    """
    t = text.strip()

    # Sem telefone → salvar template
    phone_match = re.match(r'^(\d{10,15})\s+(.*)', t)
    if not phone_match:
        return ('save_template', t)

    phone = phone_match.group(1)
    rest = phone_match.group(2).strip()

    # Verifica se rest comeca com IDs de template (numeros 1-3 digitos)
    OVERRIDE_KW = {'por', 'as', 'de', 'em', 'ate', 'a'}
    tokens = rest.split()
    numeric_prefix = []
    for token in tokens:
        if re.match(r'^\d{1,3}$', token):
            numeric_prefix.append(int(token))
        else:
            break

    if numeric_prefix:
        remaining = tokens[len(numeric_prefix):]
        first_remaining = remaining[0].lower().rstrip('.,') if remaining else ''
        # Tambem aceita 'às' com acento
        is_override = (not remaining or
                       first_remaining in OVERRIDE_KW or
                       first_remaining.startswith('\xe0'))  # 'à'
        if is_override:
            overrides = ' '.join(remaining)
            return ('apply_templates', phone, numeric_prefix, overrides)

    # Inline (comportamento existente)
    return ('inline', phone, rest)


# ----------------------------------------------------------------- CRUD prescricoes

def add_prescription(patient_phone, med_text, condition_text,
                     interval_hours, specific_hour, duration_days):
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
    try:
        init_db()
        conn = _conn()
        row = conn.execute("SELECT next_dose_at FROM prescriptions WHERE id=?",
                           (prescription_id,)).fetchone()
        if not row:
            conn.close()
            return
        # Usa o MAIOR entre o horário agendado e agora para evitar loop de reenvio
        scheduled_utc = datetime.fromisoformat(row["next_dose_at"])
        now_utc = datetime.utcnow()
        base_utc = max(scheduled_utc, now_utc)
        base_brt = base_utc - timedelta(hours=3)

        # Garante valores válidos — se ambos None, assume intervalo diário
        _interval = interval_hours if interval_hours is not None else 24
        _spec = int(specific_hour) if specific_hour is not None else None

        if _spec is not None:
            next_brt = base_brt + timedelta(days=1)
            next_brt = next_brt.replace(hour=_spec, minute=0, second=0, microsecond=0)
        else:
            next_brt = base_brt + timedelta(hours=_interval)
            if not (HOUR_MIN <= next_brt.hour < HOUR_MAX):
                next_brt = (next_brt + timedelta(days=1)).replace(
                    hour=HOUR_MIN, minute=0, second=0, microsecond=0
                )

        # Garantia extra: next_dose nunca fica no passado
        next_utc = next_brt + timedelta(hours=3)
        while next_utc <= now_utc:
            if _spec is not None:
                next_utc += timedelta(days=1)
            else:
                next_utc += timedelta(hours=max(_interval, 1))

        conn.execute("UPDATE prescriptions SET next_dose_at=? WHERE id=?",
                     (next_utc.isoformat(), prescription_id))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("advance_next_dose #%d: %s", prescription_id, e)


def deactivate_expired():
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
        sched = format_schedule(r["specific_hour"], r["interval_hours"])
        end_brt = (datetime.fromisoformat(r["end_at"]) - timedelta(hours=3)).strftime("%d/%m")
        lines.append(f"  #{r['id']} {med}{cond} -- {sched} ate {end_brt}")
    return "\n".join(lines)


def build_reminder_message(med_text, condition_text):
    cond = f" {condition_text}" if condition_text else ""
    return f"E hora de tomar {med_text}{cond}."


# ----------------------------------------------------------------- templates

def save_template(text):
    """Parse texto sem telefone e salva como template numerado."""
    dummy = f"5500000000000 {text.strip()}"
    parsed = parse_prescription(dummy)
    if not parsed:
        return "Nao consegui interpretar. Formato: /receita: [med] [posologia] [duracao]"
    init_db()
    now = datetime.utcnow().isoformat()
    conn = _conn()
    cur = conn.execute("""
        INSERT INTO prescription_templates
          (text, med_text, condition_text, interval_hours, specific_hour, duration_days, created_at)
        VALUES (?,?,?,?,?,?,?)
    """, (text.strip(), parsed['med_text'], parsed['condition_text'],
          parsed['interval_hours'], parsed['specific_hour'], parsed['duration_days'], now))
    rowid = cur.lastrowid
    conn.commit()
    rows = conn.execute("SELECT id FROM prescription_templates ORDER BY id ASC").fetchall()
    conn.close()
    idx = next((i+1 for i, (r,) in enumerate(rows) if r == rowid), rowid)
    sched = format_schedule(parsed['specific_hour'], parsed['interval_hours'])
    cond = f" {parsed['condition_text']}" if parsed['condition_text'] else ""
    return (f"Template *#{idx}* salvo:\n"
            f"{parsed['med_text']}{cond} -- {sched} por {parsed['duration_days']}d\n\n"
            f"Para prescrever: /receita: [tel] {idx}")


def list_templates():
    """Lista numerada de templates salvos."""
    try:
        init_db()
        conn = _conn()
        rows = conn.execute("""
            SELECT id, med_text, condition_text, interval_hours, specific_hour, duration_days
            FROM prescription_templates ORDER BY id ASC
        """).fetchall()
        conn.close()
    except Exception:
        return "Nenhum template cadastrado ainda."
    if not rows:
        return "Nenhum template cadastrado ainda.\nUse */receita: [texto sem numero]* para salvar."
    lines = ["*Templates de prescricao:*\n"]
    for i, row in enumerate(rows, 1):
        rid, med, cond, interval, hour, dur = row
        cond_str = f" {cond}" if cond else ""
        sched = format_schedule(hour, interval)
        lines.append(f"{i}. {med}{cond_str} -- {sched} por {dur}d")
    lines.append("\n*Aplicar:* /receita: [tel] 1 2 3")
    lines.append("*Sobrescrever:* /receita: [tel] 1 por 3 dias")
    lines.append("*Apagar:* /apagar receita N")
    return "\n".join(lines)


def delete_template(n):
    """Apaga template #n (base 1). Retorna mensagem de status."""
    try:
        init_db()
        conn = _conn()
        rows = conn.execute(
            "SELECT id, med_text FROM prescription_templates ORDER BY id ASC"
        ).fetchall()
        if not (1 <= n <= len(rows)):
            conn.close()
            return f"Template #{n} nao encontrado. Use /templates para ver a lista."
        rid, med = rows[n-1]
        conn.execute("DELETE FROM prescription_templates WHERE id=?", (rid,))
        conn.commit()
        conn.close()
        return f"Template *#{n}* ({med}) apagado."
    except Exception as e:
        return f"Erro ao apagar template: {e}"


def _parse_overrides(text):
    """Extrai campos de sobrescrita (duracao, horario, intervalo) do texto."""
    if not text.strip():
        return {}
    dummy = f"5500000000000 placeholder {text}"
    parsed = parse_prescription(dummy)
    if not parsed:
        return {}
    overrides = {}
    if re.search(r'\bpor\s+\d+\s*dias?\b', text, re.IGNORECASE):
        overrides['duration_days'] = parsed['duration_days']
    if parsed.get('specific_hour') is not None:
        overrides['specific_hour'] = parsed['specific_hour']
        overrides['interval_hours'] = None
    elif parsed.get('interval_hours') is not None:
        overrides['interval_hours'] = parsed['interval_hours']
        overrides['specific_hour'] = None
    return overrides


def apply_templates(patient_phone, template_ids, overrides_text):
    """
    Aplica templates ao paciente com sobrescritas opcionais.
    Retorna lista de (pid_ou_None, mensagem).
    """
    try:
        init_db()
        conn = _conn()
        rows = conn.execute("""
            SELECT id, med_text, condition_text, interval_hours, specific_hour, duration_days
            FROM prescription_templates ORDER BY id ASC
        """).fetchall()
        conn.close()
    except Exception:
        rows = []

    overrides = _parse_overrides(overrides_text)
    results = []

    for tid in template_ids:
        if not (1 <= tid <= len(rows)):
            results.append((None, f"Template #{tid} nao encontrado."))
            continue
        rid, med, cond, interval, hour, dur = rows[tid-1]

        final_dur = overrides.get('duration_days', dur)
        if 'specific_hour' in overrides:
            final_hour = overrides['specific_hour']
            final_interval = None
        elif 'interval_hours' in overrides:
            final_hour = None
            final_interval = overrides['interval_hours']
        else:
            final_hour = hour
            final_interval = interval

        pid = add_prescription(patient_phone, med, cond, final_interval, final_hour, final_dur)
        sched = format_schedule(final_hour, final_interval)
        results.append((pid, f"#{tid} {med} -- {sched} por {final_dur}d"))

    return results
