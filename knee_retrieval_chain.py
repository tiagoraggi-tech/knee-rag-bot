"""
knee_retrieval_chain.py â Retrieval chain com guardrails CFM 2.314/2022 + LGPD

Plug direto no Chroma criado por knee_loader.py.

Arquitetura:
  Query â Retrieval (Chroma, k=20) â Reranker (cross-encoder) â Top-3
        â Prompt com guardrails â Groq (fallback: llama-3.3-70b â llama-3.1-8b â gemma2-9b)
        â Resposta + citaÃ§Ãµes â PÃ³s-processamento (disclaimer, auditoria)
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


# ===== MODELOS FALLBACK =====

GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",   # primario: 100k TPD
    "llama-3.1-8b-instant",      # fallback: quota separada (~500k TPD)
    "gemma2-9b-it",              # ultimo recurso
]


# ===== PROMPTS =====

SYSTEM_PROMPT = """VocÃª Ã© um assistente de educaÃ§Ã£o em saÃºde do consultÃ³rio do Dr. Tiago Raggi (ortopedista, CRM Brasil). Sua funÃ§Ã£o Ã© fornecer informaÃ§Ãµes educativas sobre saÃºde do joelho a pacientes via WhatsApp.

## REGRAS OBRIGATÃRIAS â CFM ResoluÃ§Ã£o 2.314/2022

1. **NUNCA emita diagnÃ³stico definitivo.** VocÃª pode descrever condiÃ§Ãµes e sintomas em termos educativos, mas sempre indicando que apenas avaliaÃ§Ã£o presencial com mÃ©dico permite diagnÃ³stico.

2. **NUNCA prescreva medicamentos, doses ou condutas terapÃªuticas individualizadas.** Pode mencionar classes terapÃªuticas comumente usadas (ex: "anti-inflamatÃ³rios sÃ£o frequentemente prescritos") sem indicar marca, dose ou posologia especÃ­fica.

3. **NUNCA substitua consulta presencial.** Toda resposta deve reforÃ§ar que dÃºvidas clÃ­nicas devem ser tratadas em consulta com o Dr. Tiago ou outro mÃ©dico.

4. **Em sinais de gravidade** (dor intensa, edema sÃºbito, incapacidade de apoiar peso, febre, deformidade, sinais neurolÃ³gicos), oriente busca por pronto-atendimento IMEDIATAMENTE.

5. **NÃ£o solicite nem armazene dados pessoais sensÃ­veis** (CPF, exames, prontuÃ¡rio). Se o paciente compartilhar, oriente que esses dados devem ser apresentados em consulta.

## REGRAS DE CONTEÃDO

6. **Use APENAS o CONTEXTO fornecido abaixo.** Se a informaÃ§Ã£o nÃ£o estiver no contexto, diga "Essa informaÃ§Ã£o especÃ­fica nÃ£o estÃ¡ na minha base â recomendo conversar com o Dr. Tiago em consulta."

7. **CITE as fontes** ao final da resposta, no formato:
   ```
   ð Fontes:
   â¢ [TÃ­tulo curto] â [URL]
   ```

8. **Linguagem acessÃ­vel** ao paciente leigo: explique termos tÃ©cnicos (ex: "gonartrose, que Ã© o desgaste da cartilagem do joelho").

9. **PortuguÃªs brasileiro**, tom acolhedor e profissional. Sem emojis excessivos (no mÃ¡ximo 1-2 quando agregar clareza).

10. **Resposta curta** adequada ao WhatsApp: idealmente 3-6 parÃ¡grafos curtos. Use listas quando facilitar.

## FORMATO DE RESPOSTA

[Resposta educativa baseada no CONTEXTO]

â ï¸ *Esta informaÃ§Ã£o Ã© educativa e nÃ£o substitui consulta mÃ©dica. Para avaliaÃ§Ã£o do seu caso especÃ­fico, agende com o Dr. Tiago.*

ð Fontes:
â¢ [fonte 1] â [url]
â¢ [fonte 2] â [url]
"""

USER_TEMPLATE = """## CONTEXTO RECUPERADO

{context}

## PERGUNTA DO PACIENTE

{question}

Responda seguindo TODAS as regras do system prompt."""


# ===== RED FLAGS =====

RED_FLAG_PATTERNS = [
    r"\b(n[Ã£a]o consigo (andar|apoiar|levantar))\b",
    r"\b(dor (insuport[Ã¡a]vel|intensa|muito forte))\b",
    r"\b(joelho (deformado|torto|deslocado))\b",
    r"\b(estourou|estalou (muito |forte))\b",
    r"\b(inchou (muito |de repente|subitamente))\b",
    r"\b(febre|calafrio).{0,30}(joelho|articula)",
    r"\b(formigamento|dorm[Ãªe]ncia|perdi (a )?sensibilidade)\b",
    r"\b(perna roxa|p[Ã©e] roxo|cianose)\b",
    r"\b(acidente|trauma|queda).{0,40}(agora|hoje|h[Ã¡a] pouco)",
]

EMERGENCY_RESPONSE = """â ï¸ **Os sintomas que vocÃª descreveu podem indicar uma situaÃ§Ã£o que precisa de avaliaÃ§Ã£o mÃ©dica URGENTE.**

Por favor, procure atendimento agora:
â¢ **Pronto-socorro ortopÃ©dico** mais prÃ³ximo, ou
â¢ **SAMU 192** se houver dificuldade de locomoÃ§Ã£o

NÃ£o espere para agendar consulta de rotina. ApÃ³s o atendimento de urgÃªncia, entre em contato para acompanhamento com o Dr. Tiago.

â ï¸ *Esta orientaÃ§Ã£o Ã© automÃ¡tica e baseada nos sintomas descritos. Em qualquer dÃºvida sobre a gravidade, sempre opte por buscar atendimento.*"""


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
        rerank_top_k: int = 3,
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
            raise ValueError("GROQ_API_KEY nÃ£o fornecida.")

        models_to_try = [groq_model] + [m for m in GROQ_FALLBACK_MODELS if m != groq_model]
        self.llms: List[ChatGroq] = []
        for model_name in models_to_try:
            self.llms.append(ChatGroq(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=1024,
            ))

        log.info(
            "KneeRAGChain ready | modelos=%d, k=%dâ%d",
            len(self.llms), retrieval_k, rerank_top_k,
        )

    def _invoke_with_fallback(self, messages: list) -> Optional[str]:
        """Tenta cada modelo em ordem; fallback automatico em rate limit (429)."""
        last_error = None
        for llm in self.llms:
            name = getattr(llm, "model_name", str(llm))
            try:
                response = llm.invoke(messages)
                log.info("Modelo usado: %s", name)
                return response.content
            except Exception as e:
                s = str(e)
                if "429" in s or "rate_limit" in s.lower() or "Rate limit" in s:
                    log.warning("Rate limit em %s, tentando proximo...", name)
                else:
                    log.error("Erro em %s: %s", name, e)
                last_error = e
        log.error("Todos os modelos falharam: %s", last_error)
        return None

    def retrieve(self, query: str, scope_filter: Optional[str] = None) -> List[Tuple[Document, float]]:
        filter_dict = None
        if scope_filter and scope_filter in ("surgical", "conservative", "mixed"):
            filter_dict = {"scope": {"$in": [scope_filter, "mixed"]}}

        candidates = self.vectorstore.similarity_search_with_score(
            query, k=self.retrieval_k, filter=filter_dict
        )
        if not candidates:
            return []

        pairs = [(query, doc.page_content[:600]) for doc, _ in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        reranked = [(doc, float(score)) for (doc, _), score in zip(candidates, rerank_scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.rerank_top_k]

    def _format_context(self, results: List[Tuple[Document, float]]) -> Tuple[str, List[Dict]]:
        context_blocks = []
        sources = []
        for i, (doc, score) in enumerate(results, 1):
            md = doc.metadata
            title = md.get("title", "Sem tÃ­tulo")[:150]
            url = md.get("url", "")
            source_type = md.get("source", "")
            year = md.get("year", "")

            header = f"[FONTE {i}] {title}"
            if year:
                header += f" ({year})"
            if source_type:
                header += f" â {source_type}"

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
    ) -> Dict[str, Any]:
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

        results = self.retrieve(question, scope_filter=scope_filter)
        if not results:
            answer = (
                "Essa informaÃ§Ã£o especÃ­fica nÃ£o estÃ¡ na minha base de conhecimento. "
                "Recomendo conversar com o Dr. Tiago em consulta para uma orientaÃ§Ã£o adequada ao seu caso.\n\n"
                "â ï¸ *Esta resposta Ã© educativa e nÃ£o substitui consulta mÃ©dica.*"
            )
            self._audit({
                "ts": timestamp, "patient_hash": patient_id_hash,
                "query": question, "red_flag": False,
                "answer_type": "no_results", "retrieved_count": 0,
            })
            return {"answer": answer, "sources": [], "red_flag": False, "retrieved_count": 0}

        context, sources = self._format_context(results)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_TEMPLATE.format(context=context, question=question)),
        ]

        answer = self._invoke_with_fallback(messages)
        if answer is None:
            answer = (
                "Tive um problema tÃ©cnico ao gerar a resposta. "
                "Por favor, tente novamente em instantes ou entre em contato com o consultÃ³rio."
            )

        if "nÃ£o substitui consulta" not in answer.lower() and "consulta mÃ©dica" not in answer.lower():
            answer += "\n\nâ ï¸ *Esta informaÃ§Ã£o Ã© educativa e nÃ£o substitui consulta mÃ©dica.*"

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
    if result["sources"] and "ð Fontes" not in answer and "fontes" not in answer.lower():
        sources_block = "\n\nð Fontes:\n"
        for s in result["sources"][:3]:
            if s.get("url"):
                sources_block += f"â¢ {s['title'][:80]} â {s['url']}\n"
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
        "O que Ã© artrose de joelho e quais os tratamentos sem cirurgia?",
        scope_filter="conservative",
        patient_id_hash=hashlib.md5(b"+5524999999999").hexdigest(),
    )
    print(format_for_whatsapp(r1))

    print("\n" + "=" * 60)
    print("CASO 2: Red flag")
    print("=" * 60)
    r2 = chain.ask("Doutor, caÃ­ da escada agora, meu joelho estÃ¡ deformado e nÃ£o consigo apoiar")
    print(format_for_whatsapp(r2))

    print("\n" + "=" * 60)
    print("CASO 3: CirÃºrgico")
    print("=" * 60)
    r3 = chain.ask("Como Ã© a recuperaÃ§Ã£o apÃ³s reconstruÃ§Ã£o de LCA?", scope_filter="surgical")
    print(format_for_whatsapp(r3))
