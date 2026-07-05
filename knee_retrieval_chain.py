"""
knee_retrieval_chain.py — Retrieval chain com guardrails CFM 2.314/2022 + LGPD

Plug direto no Chroma criado por knee_loader.py.

Arquitetura:
  Query → Retrieval (Chroma, k=20) → Reranker (cross-encoder) → Top-5
        → Prompt com guardrails → Groq llama-3.3-70b → Resposta + citações
        → Pós-processamento (disclaimer, auditoria)
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("knee_chain")


# ===== PROMPTS =====

SYSTEM_PROMPT = """Você é um assistente de educação em saúde do consultório do Dr. Tiago Raggi (ortopedista, CRM Brasil). Sua função é fornecer informações educativas sobre saúde do joelho a pacientes via WhatsApp.

## REGRAS OBRIGATÓRIAS — CFM Resolução 2.314/2022

1. **NUNCA emita diagnóstico definitivo.** Você pode descrever condições e sintomas em termos educativos, mas sempre indicando que apenas avaliação presencial com médico permite diagnóstico.

2. **NUNCA prescreva medicamentos, doses ou condutas terapêuticas individualizadas.** Pode mencionar classes terapêuticas comumente usadas (ex: "anti-inflamatórios são frequentemente prescritos") sem indicar marca, dose ou posologia específica.

3. **NUNCA substitua consulta presencial.** Toda resposta deve reforçar que dúvidas clínicas devem ser tratadas em consulta com o Dr. Tiago ou outro médico.

4. **Em sinais de gravidade** (dor intensa, edema súbito, incapacidade de apoiar peso, febre, deformidade, sinais neurológicos), oriente busca por pronto-atendimento IMEDIATAMENTE.

5. **Não solicite nem armazene dados pessoais sensíveis** (CPF, exames, prontuário). Se o paciente compartilhar, oriente que esses dados devem ser apresentados em consulta.

## REGRAS DE CONTEÚDO

6. **Use APENAS o CONTEXTO fornecido abaixo.** Se a informação não estiver no contexto, diga "Essa informação específica não está na minha base — recomendo conversar com o Dr. Tiago em consulta."

7. **CITE as fontes** ao final da resposta, no formato:
   ```
   📚 Fontes:
   • [Título curto] — [URL]
   ```

8. **Linguagem acessível** ao paciente leigo: explique termos técnicos (ex: "gonartrose, que é o desgaste da cartilagem do joelho").

9. **Português brasileiro**, tom acolhedor e profissional. Sem emojis excessivos (no máximo 1-2 quando agregar clareza).

10. **Resposta curta** adequada ao WhatsApp: idealmente 3-6 parágrafos curtos. Use listas quando facilitar.

## FORMATO DE RESPOSTA

[Resposta educativa baseada no CONTEXTO]

⚠️ *Esta informação é educativa e não substitui consulta médica. Para avaliação do seu caso específico, agende com o Dr. Tiago.*

📚 Fontes:
• [fonte 1] — [url]
• [fonte 2] — [url]
"""

USER_TEMPLATE = """## CONTEXTO RECUPERADO

{context}

## PERGUNTA DO PACIENTE

{question}

Responda seguindo TODAS as regras do system prompt."""


# ===== RED FLAGS =====

RED_FLAG_PATTERNS = [
    r"\b(n[ãa]o consigo (andar|apoiar|levantar))\b",
    r"\b(dor (insuport[áa]vel|intensa|muito forte))\b",
    r"\b(joelho (deformado|torto|deslocado))\b",
    r"\b(estourou|estalou (muito |forte))\b",
    r"\b(inchou (muito |de repente|subitamente))\b",
    r"\b(febre|calafrio).{0,30}(joelho|articula)",
    r"\b(formigamento|dorm[êe]ncia|perdi (a )?sensibilidade)\b",
    r"\b(perna roxa|p[ée] roxo|cianose)\b",
    r"\b(acidente|trauma|queda).{0,40}(agora|hoje|h[áa] pouco)",
]

EMERGENCY_RESPONSE = """⚠️ **Os sintomas que você descreveu podem indicar uma situação que precisa de avaliação médica URGENTE.**

Por favor, procure atendimento agora:
• **Pronto-socorro ortopédico** mais próximo, ou
• **SAMU 192** se houver dificuldade de locomoção

Não espere para agendar consulta de rotina. Após o atendimento de urgência, entre em contato para acompanhamento com o Dr. Tiago.

⚠️ *Esta orientação é automática e baseada nos sintomas descritos. Em qualquer dúvida sobre a gravidade, sempre opte por buscar atendimento.*"""


def has_red_flags(text: str) -> bool:
    text_low = text.lower()
    return any(re.search(p, text_low) for p in RED_FLAG_PATTERNS)


# ===== CHAIN =====

class KneeRAGChain:
    def __init__(
        self,
        persist_dir: str = "./chroma_knee",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        retrieval_k: int = 20,
        rerank_top_k: int = 5,
        temperature: float = 0.2,
        audit_log_path: Optional[str] = "./rag_audit.jsonl",
    ):
        self.retrieval_k = retrieval_k
        self.rerank_top_k = rerank_top_k
        self.audit_log_path = audit_log_path

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            collection_name="knee_orthopedics",
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        log.info("Loading reranker %s...", reranker_model)
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY não fornecida.")

        self.llm = ChatGroq(
            api_key=api_key,
            model=groq_model,
            temperature=temperature,
            max_tokens=1024,
        )

        log.info("KneeRAGChain ready | model=%s, k=%d→%d", groq_model, retrieval_k, rerank_top_k)

    def retrieve(self, query: str, scope_filter: Optional[str] = None) -> List[Tuple[Document, float]]:
        filter_dict = None
        if scope_filter and scope_filter in ("surgical", "conservative", "mixed"):
            filter_dict = {"scope": {"$in": [scope_filter, "mixed"]}}

        candidates = self.vectorstore.similarity_search_with_score(
            query, k=self.retrieval_k, filter=filter_dict
        )
        if not candidates:
            return []

        pairs = [(query, doc.page_content[:1000]) for doc, _ in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        reranked = [(doc, float(score)) for (doc, _), score in zip(candidates, rerank_scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.rerank_top_k]

    def _format_context(self, results: List[Tuple[Document, float]]) -> Tuple[str, List[Dict]]:
        context_blocks = []
        sources = []
        for i, (doc, score) in enumerate(results, 1):
            md = doc.metadata
            title = md.get("title", "Sem título")[:150]
            url = md.get("url", "")
            source_type = md.get("source", "")
            year = md.get("year", "")

            header = f"[FONTE {i}] {title}"
            if year:
                header += f" ({year})"
            if source_type:
                header += f" — {source_type}"

            context_blocks.append(f"{header}\n{doc.page_content}\n")
            sources.append({
                "index": i, "title": title, "url": url,
                "source_type": source_type, "rerank_score": round(score, 3),
            })
        return "\n---\n".join(context_blocks), sources

    def _audit(self, entry: Dict[str, Any]) -> None:
        if not self.audit_log_path:
            return
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning("Falha em audit log: %s", e)

    def ask(
        self,
        question: str,
        scope_filter: Optional[str] = None,
        patient_id_hash: Optional[str] = None,
        protocol_context: Optional[str] = None,
        safety_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        # safety_context: politica clinica + perfil de seguranca do paciente.
        # Injetado no system SEM desligar o RAG (diferente de protocol_context).
        timestamp = datetime.utcnow().isoformat()

        if has_red_flags(question):
            log.warning("Red flag detectada")
            self._audit({
                "ts": timestamp, "patient_hash": patient_id_hash,
                "query": question, "red_flag": True, "answer_type": "emergency_bypass",
            })
            return {
                "answer": EMERGENCY_RESPONSE, "sources": [],
                "red_flag": True, "retrieved_count": 0,
            }

        # Se há protocolo clínico vinculado, não consulta RAG de joelho —
        # o protocolo do Dr. Tiago tem prioridade total sobre a literatura genérica.
        if protocol_context:
            results = []
        else:
            results = self.retrieve(question, scope_filter=scope_filter)

        # Se há protocolo vinculado, usa mesmo sem resultados do RAG
        if not results and not protocol_context:
            answer = (
                "Essa informação específica não está na minha base de conhecimento. "
                "Recomendo conversar com o Dr. Tiago em consulta para uma orientação adequada ao seu caso.\n\n"
                "⚠️ *Esta resposta é educativa e não substitui consulta médica.*"
            )
            self._audit({
                "ts": timestamp, "patient_hash": patient_id_hash,
                "query": question, "red_flag": False,
                "answer_type": "no_results", "retrieved_count": 0,
            })
            return {"answer": answer, "sources": [], "red_flag": False, "retrieved_count": 0}

        context, sources = self._format_context(results) if results else ("", [])

        # Protocolos clínicos têm prioridade máxima — vêm antes do contexto RAG
        if protocol_context:
            full_context = protocol_context + ("\n\n" + context if context else "")
        else:
            full_context = context

        # Quando há protocolo, informa o LLM que não deve limitar ao joelho
        if protocol_context:
            system_content = SYSTEM_PROMPT + (
                "\n\n## ATENÇÃO — PACIENTE COM PROTOCOLO ESPECÍFICO\n"
                "Este paciente possui um PROTOCOLO CLÍNICO específico cadastrado pelo Dr. Tiago (seção ## PROTOCOLOS CLÍNICOS acima). "
                "Responda EXCLUSIVAMENTE com base neste protocolo. "
                "Não limite sua resposta a condições do joelho — o protocolo pode ser para qualquer região ou condição tratada pelo Dr. Tiago. "
                "NÃO mencione joelho, neuroproloterapia do joelho ou condições não relacionadas ao protocolo do paciente."
            )
        else:
            system_content = SYSTEM_PROMPT

        # Politica clinica / perfil de seguranca do paciente — reforca sem desligar o RAG
        if safety_context:
            system_content = system_content + "\n\n" + safety_context

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=USER_TEMPLATE.format(context=full_context, question=question)),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            log.error("LLM failure: %s", e)
            answer = (
                "Tive um problema técnico ao gerar a resposta. "
                "Por favor, tente novamente em instantes ou entre em contato com o consultório."
            )

        if "não substitui consulta" not in answer.lower() and "consulta médica" not in answer.lower():
            answer += "\n\n⚠️ *Esta informação é educativa e não substitui consulta médica.*"

        self._audit({
            "ts": timestamp, "patient_hash": patient_id_hash,
            "query": question, "red_flag": False, "answer_type": "rag",
            "retrieved_count": len(results),
            "sources_used": [s["url"] for s in sources if s["url"]],
            "scope_filter": scope_filter,
        })

        return {
            "answer": answer, "sources": sources,
            "red_flag": False, "retrieved_count": len(results),
        }


def format_for_whatsapp(result: Dict[str, Any]) -> str:
    answer = result["answer"]
    if result["sources"] and "📚 Fontes" not in answer and "fontes" not in answer.lower():
        sources_block = "\n\n📚 Fontes:\n"
        for s in result["sources"][:3]:
            if s.get("url"):
                sources_block += f"• {s['title'][:80]} — {s['url']}\n"
        answer += sources_block
    return answer


if __name__ == "__main__":
    import hashlib
    from dotenv import load_dotenv
    load_dotenv()

    chain = KneeRAGChain(
        persist_dir=os.getenv("CHROMA_DIR", "./chroma_knee"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        audit_log_path=os.getenv("AUDIT_LOG", "./rag_audit.jsonl"),
    )

    print("\n" + "=" * 60)
    print("CASO 1: Pergunta educativa")
    print("=" * 60)
    r1 = chain.ask(
        "O que é artrose de joelho e quais os tratamentos sem cirurgia?",
        scope_filter="conservative",
        patient_id_hash=hashlib.md5(b"+5524999999999").hexdigest(),
    )
    print(format_for_whatsapp(r1))

    print("\n" + "=" * 60)
    print("CASO 2: Red flag")
    print("=" * 60)
    r2 = chain.ask("Doutor, caí da escada agora, meu joelho está deformado e não consigo apoiar")
    print(format_for_whatsapp(r2))

    print("\n" + "=" * 60)
    print("CASO 3: Cirúrgico")
    print("=" * 60)
    r3 = chain.ask("Como é a recuperação após reconstrução de LCA?", scope_filter="surgical")
    print(format_for_whatsapp(r3))
