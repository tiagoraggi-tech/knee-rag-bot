"""
knee_loader.py — RAG loader para ortopedia de joelho (cirúrgico + conservador)

Arquitetura adaptada do paper JMIR AI (Oct/2025) — chatbot RAG ortopédico alemão:
  - unstructured.partition.pdf + PyMuPDF fallback
  - metadados normalizados (source, category, page, url, timestamp, doi, scope)
  - chunking semântico com RecursiveCharacterTextSplitter
  - embeddings multilíngue (paraphrase-multilingual-mpnet-base-v2) p/ PT-BR + EN

Fontes:
  1. PubMed (Entrez via Biopython)
  2. PDFs locais (diretrizes SBOT/AAOS/SBCJ)
  3. Google Scholar (scholarly)
  4. Web scraping (sites de sociedades — SBOT, SBCJ, AAOS OrthoInfo)

Vector store: ChromaDB persistente.

Uso:
  loader = KneeKnowledgeLoader(persist_dir="./chroma_knee")
  loader.ingest_pubmed(max_results=50)
  loader.ingest_pdfs("./diretrizes_pdf/")
  loader.ingest_scholar(max_results=20)
  loader.ingest_websites()
  loader.build_vectorstore()
"""

import os
import re
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from Bio import Entrez
from scholarly import scholarly
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

try:
    from unstructured.partition.pdf import partition_pdf
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("knee_loader")


# ===== QUERIES CLÍNICAS — escopo misto =====

KNEE_QUERIES_PUBMED = {
    # Conservador
    "artrose_joelho": '("knee osteoarthritis"[Title/Abstract]) AND ("conservative treatment"[Title/Abstract] OR "non-surgical"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "tendinopatia_patelar": '("patellar tendinopathy"[Title/Abstract] OR "jumper\'s knee"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "condromalacia": '("chondromalacia patellae"[Title/Abstract] OR "patellofemoral pain"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "infiltracao": '("knee"[Title/Abstract]) AND ("hyaluronic acid"[Title/Abstract] OR "corticosteroid injection"[Title/Abstract] OR "PRP"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',

    # Cirúrgico
    "lca_reconstrucao": '("anterior cruciate ligament reconstruction"[Title/Abstract] OR "ACL reconstruction"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "menisco": '("meniscal tear"[Title/Abstract] OR "meniscectomy"[Title/Abstract] OR "meniscus repair"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "artroplastia_total": '("total knee arthroplasty"[Title/Abstract] OR "total knee replacement"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "artroplastia_unicompartimental": '("unicompartmental knee arthroplasty"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',
    "osteotomia": '("high tibial osteotomy"[Title/Abstract] OR "knee osteotomy"[Title/Abstract]) AND ("2020"[PDAT]:"2026"[PDAT])',

    # Proloterapia / neuroproloterapia (procedimentos do Dr. Tiago) — janela
    # temporal maior: os ensaios de referencia sao de 2005 em diante
    "proloterapia_joelho": '("prolotherapy"[Title/Abstract] OR "dextrose injection"[Title/Abstract]) AND ("knee"[Title/Abstract]) AND ("2005"[PDAT]:"2026"[PDAT])',
    "proloterapia_sacroiliaca_lombar": '("prolotherapy"[Title/Abstract]) AND ("sacroiliac"[Title/Abstract] OR "low back pain"[Title/Abstract] OR "lumbar"[Title/Abstract]) AND ("2005"[PDAT]:"2026"[PDAT])',
    "proloterapia_dextrose_geral": '("dextrose prolotherapy"[Title/Abstract]) AND ("mechanism"[Title/Abstract] OR "adverse"[Title/Abstract] OR "systematic review"[Title/Abstract] OR "randomized"[Title/Abstract]) AND ("2005"[PDAT]:"2026"[PDAT])',
    "neuroproloterapia": '("perineural injection"[Title/Abstract] OR "neural prolotherapy"[Title/Abstract] OR "subcutaneous prolotherapy"[Title/Abstract]) AND ("2005"[PDAT]:"2026"[PDAT])',
    "viscossuplementacao": '("viscosupplementation"[Title/Abstract]) AND ("knee"[Title/Abstract]) AND ("2015"[PDAT]:"2026"[PDAT])',

    # ── ESCOPO AMPLIADO — dor musculoesqueletica e terapias regenerativas ──
    # (pratica do Dr. Tiago vai alem do joelho; o PubMed ao vivo ja cobre
    #  qualquer tema, mas estas queries enriquecem a base curada/reranqueada)
    "ombro": '("rotator cuff"[Title/Abstract] OR "shoulder pain"[Title/Abstract] OR "adhesive capsulitis"[Title/Abstract] OR "shoulder tendinopathy"[Title/Abstract]) AND ("conservative"[Title/Abstract] OR "injection"[Title/Abstract] OR "rehabilitation"[Title/Abstract]) AND ("2018"[PDAT]:"2026"[PDAT])',
    "quadril": '("hip osteoarthritis"[Title/Abstract] OR "greater trochanteric pain"[Title/Abstract] OR "gluteal tendinopathy"[Title/Abstract]) AND ("2018"[PDAT]:"2026"[PDAT])',
    "coluna_lombar": '("chronic low back pain"[Title/Abstract] OR "lumbar degenerative disc"[Title/Abstract] OR "facet joint"[Title/Abstract]) AND ("conservative"[Title/Abstract] OR "injection"[Title/Abstract] OR "prolotherapy"[Title/Abstract]) AND ("2015"[PDAT]:"2026"[PDAT])',
    "coluna_cervical": '("chronic neck pain"[Title/Abstract] OR "cervical radiculopathy"[Title/Abstract]) AND ("conservative"[Title/Abstract] OR "injection"[Title/Abstract] OR "physical therapy"[Title/Abstract]) AND ("2018"[PDAT]:"2026"[PDAT])',
    "cotovelo": '("lateral epicondylitis"[Title/Abstract] OR "tennis elbow"[Title/Abstract] OR "medial epicondylitis"[Title/Abstract]) AND ("2015"[PDAT]:"2026"[PDAT])',
    "pe_tornozelo": '("plantar fasciitis"[Title/Abstract] OR "achilles tendinopathy"[Title/Abstract] OR "chronic ankle instability"[Title/Abstract]) AND ("injection"[Title/Abstract] OR "prolotherapy"[Title/Abstract] OR "conservative"[Title/Abstract]) AND ("2015"[PDAT]:"2026"[PDAT])',
    "proloterapia_regioes": '("prolotherapy"[Title/Abstract] OR "perineural injection"[Title/Abstract]) AND ("shoulder"[Title/Abstract] OR "elbow"[Title/Abstract] OR "hip"[Title/Abstract] OR "ankle"[Title/Abstract] OR "tendinopathy"[Title/Abstract]) AND ("2005"[PDAT]:"2026"[PDAT])',
    "regenerativa_prp": '("platelet-rich plasma"[Title/Abstract]) AND ("musculoskeletal"[Title/Abstract] OR "tendinopathy"[Title/Abstract] OR "osteoarthritis"[Title/Abstract]) AND ("2019"[PDAT]:"2026"[PDAT])',
    "dor_miofascial": '("myofascial pain"[Title/Abstract] OR "trigger point"[Title/Abstract]) AND ("injection"[Title/Abstract] OR "dry needling"[Title/Abstract]) AND ("2015"[PDAT]:"2026"[PDAT])',
}

KNEE_QUERIES_SCHOLAR = [
    "knee osteoarthritis conservative management guidelines 2024",
    "ACL reconstruction outcomes meta-analysis",
    "total knee arthroplasty enhanced recovery protocol",
    "meniscus repair vs meniscectomy long-term",
    "patellofemoral pain syndrome physical therapy",
    "high tibial osteotomy indications systematic review",
]

SOCIETY_URLS = {
    "AAOS_OrthoInfo_Knee": "https://orthoinfo.aaos.org/en/diseases--conditions/?topic=Knee",
    "SBOT_Joelho": "https://sbot.org.br/",
    "SBCJ": "https://www.sbcj.org.br/",
    # Proloterapia — fontes usadas no protocolo pos-procedimento (04/07/2026)
    "Rabago_Prolotherapy_PrimaryCare": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2831229/",
    "MDPI_Prolotherapy_LowBack_Review": "https://www.mdpi.com/1648-9144/61/9/1588",
    "FasciaInstitute_PostInjection": "https://fasciainstitute.org/prolotherapy-post-injection-instructions/",
    "AAPMR_Dextrose_Injection": "https://now.aapmr.org/therapeutic-injection-of-dextrose-prolotherapy-perineural-injection-therapy-and-hydrodissection/",
}


# ===== NORMALIZAÇÃO =====

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    return text.strip()


def classify_scope(text: str) -> str:
    surgical_terms = [
        "arthroscopy", "artroscopia", "reconstruction", "reconstrução",
        "arthroplasty", "artroplastia", "osteotomy", "osteotomia",
        "meniscectomy", "meniscectomia", "surgical", "cirúrgico",
    ]
    conservative_terms = [
        "physical therapy", "fisioterapia", "conservative", "conservador",
        "injection", "infiltração", "exercise", "exercício",
        "non-surgical", "não-cirúrgico", "rehabilitation", "reabilitação",
    ]
    text_low = text.lower()
    s = sum(t in text_low for t in surgical_terms)
    c = sum(t in text_low for t in conservative_terms)
    if s > c * 1.5:
        return "surgical"
    if c > s * 1.5:
        return "conservative"
    return "mixed"


def make_doc_id(source: str, identifier: str) -> str:
    raw = f"{source}::{identifier}"
    return hashlib.md5(raw.encode()).hexdigest()


# ===== LOADER PRINCIPAL =====

class KneeKnowledgeLoader:
    def __init__(
        self,
        persist_dir: str = "./chroma_knee",
        entrez_email: str = "ortopediaraggi@gmail.com",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        self.entrez_email = entrez_email
        Entrez.email = entrez_email

        self.documents: List[Document] = []
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("Loader initialized | persist_dir=%s", persist_dir)

    def ingest_pubmed(self, max_results: int = 30) -> int:
        log.info("=== Ingesting PubMed ===")
        count = 0
        for topic, query in KNEE_QUERIES_PUBMED.items():
            try:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
                results = Entrez.read(handle)
                handle.close()
                ids = results.get("IdList", [])
                if not ids:
                    continue

                handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
                records = Entrez.read(handle)
                handle.close()

                for article in records.get("PubmedArticle", []):
                    try:
                        med = article["MedlineCitation"]
                        art = med["Article"]
                        pmid = str(med["PMID"])
                        title = str(art.get("ArticleTitle", ""))

                        abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
                        abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else ""

                        if not abstract or len(abstract) < 100:
                            continue

                        doi = ""
                        for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
                            if aid.attributes.get("IdType") == "doi":
                                doi = str(aid)
                                break

                        pubdate = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                        year = str(pubdate.get("Year", ""))

                        content = normalize_text(f"# {title}\n\n{abstract}")

                        doc = Document(
                            page_content=content,
                            metadata={
                                "doc_id": make_doc_id("pubmed", pmid),
                                "source": "pubmed",
                                "category": topic,
                                "pmid": pmid,
                                "doi": doi,
                                "title": title[:300],
                                "year": year,
                                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                                "scope": classify_scope(content),
                                "timestamp": datetime.utcnow().isoformat(),
                                "lang": "en",
                            },
                        )
                        self.documents.append(doc)
                        count += 1
                    except Exception as e:
                        log.warning("PubMed parse error (topic=%s): %s", topic, e)

                log.info("  %s → %d artigos", topic, len(ids))
            except Exception as e:
                log.error("PubMed query failed (%s): %s", topic, e)

        log.info("PubMed ingest done: %d documents", count)
        return count

    def ingest_pdfs(self, pdf_dir: str) -> int:
        log.info("=== Ingesting PDFs from %s ===", pdf_dir)
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            log.warning("PDF dir não existe: %s", pdf_dir)
            return 0

        count = 0
        for pdf_file in pdf_path.glob("*.pdf"):
            text = self._extract_pdf(pdf_file)
            if not text or len(text) < 200:
                log.warning("PDF sem texto útil: %s", pdf_file.name)
                continue

            text = normalize_text(text)
            fname_low = pdf_file.name.lower()
            if "sbot" in fname_low:
                society = "SBOT"
            elif "aaos" in fname_low:
                society = "AAOS"
            elif "sbcj" in fname_low:
                society = "SBCJ"
            else:
                society = "unknown"

            doc = Document(
                page_content=text,
                metadata={
                    "doc_id": make_doc_id("pdf", pdf_file.name),
                    "source": "pdf_guideline",
                    "category": "diretriz",
                    "society": society,
                    "filename": pdf_file.name,
                    "title": pdf_file.stem,
                    "url": "",
                    "scope": classify_scope(text),
                    "timestamp": datetime.utcnow().isoformat(),
                    "lang": "pt" if society in ("SBOT", "SBCJ") else "en",
                },
            )
            self.documents.append(doc)
            count += 1
            log.info("  %s [%s, %d chars]", pdf_file.name, society, len(text))

        log.info("PDF ingest done: %d documents", count)
        return count

    def _extract_pdf(self, pdf_file: Path) -> str:
        if HAS_UNSTRUCTURED:
            try:
                elements = partition_pdf(str(pdf_file), strategy="fast")
                text = "\n\n".join(str(el) for el in elements if str(el).strip())
                if len(text) > 200:
                    return text
            except Exception as e:
                log.warning("unstructured falhou em %s: %s — tentando PyMuPDF", pdf_file.name, e)

        try:
            doc = fitz.open(str(pdf_file))
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            log.error("PyMuPDF falhou em %s: %s", pdf_file.name, e)
            return ""

    def ingest_scholar(self, max_results: int = 10) -> int:
        log.info("=== Ingesting Google Scholar ===")
        count = 0
        for query in KNEE_QUERIES_SCHOLAR:
            try:
                search = scholarly.search_pubs(query)
                for i in range(max_results):
                    try:
                        pub = next(search)
                    except StopIteration:
                        break
                    except Exception as e:
                        log.warning("Scholar iter error: %s", e)
                        break

                    bib = pub.get("bib", {})
                    title = bib.get("title", "")
                    abstract = bib.get("abstract", "")
                    if not abstract or len(abstract) < 100:
                        continue

                    content = normalize_text(f"# {title}\n\n{abstract}")
                    identifier = bib.get("pub_url", "") or title

                    doc = Document(
                        page_content=content,
                        metadata={
                            "doc_id": make_doc_id("scholar", identifier),
                            "source": "google_scholar",
                            "category": "literature",
                            "title": title[:300],
                            "year": str(bib.get("pub_year", "")),
                            "author": ", ".join(bib.get("author", []))[:200] if isinstance(bib.get("author"), list) else str(bib.get("author", ""))[:200],
                            "venue": bib.get("venue", ""),
                            "url": bib.get("pub_url", ""),
                            "scope": classify_scope(content),
                            "timestamp": datetime.utcnow().isoformat(),
                            "lang": "en",
                        },
                    )
                    self.documents.append(doc)
                    count += 1

                log.info("  '%s' OK", query[:60])
            except Exception as e:
                log.error("Scholar query failed (%s): %s", query[:60], e)

        log.info("Scholar ingest done: %d documents", count)
        return count

    def ingest_websites(self, custom_urls: Optional[Dict[str, str]] = None) -> int:
        log.info("=== Ingesting society websites ===")
        urls = custom_urls or SOCIETY_URLS
        count = 0
        headers = {"User-Agent": "KneeRAG-Educational/1.0 (research; contact: ortopediaraggi@gmail.com)"}

        for name, url in urls.items():
            try:
                r = requests.get(url, headers=headers, timeout=20)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "lxml")

                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = normalize_text(soup.get_text(separator="\n"))
                if len(text) < 300:
                    log.warning("Conteúdo curto em %s", url)
                    continue

                domain = urlparse(url).netloc

                doc = Document(
                    page_content=text,
                    metadata={
                        "doc_id": make_doc_id("web", url),
                        "source": "society_website",
                        "category": "patient_education",
                        "society": name,
                        "title": soup.title.string if soup.title else name,
                        "url": url,
                        "domain": domain,
                        "scope": classify_scope(text),
                        "timestamp": datetime.utcnow().isoformat(),
                        "lang": "pt" if ".br" in domain else "en",
                    },
                )
                self.documents.append(doc)
                count += 1
                log.info("  %s [%d chars]", name, len(text))
            except Exception as e:
                log.error("Web scrape failed (%s): %s", name, e)

        log.info("Web ingest done: %d documents", count)
        return count

    def build_vectorstore(self, batch_size: int = 100) -> Chroma:
        log.info("=== Building vectorstore ===")
        log.info("Total raw docs: %d", len(self.documents))

        if not self.documents:
            raise RuntimeError("Nenhum documento ingerido. Rode ingest_* antes.")

        chunks = self.splitter.split_documents(self.documents)
        log.info("Total chunks após split: %d", len(chunks))

        ids = []
        seen = set()
        for i, ch in enumerate(chunks):
            base = ch.metadata.get("doc_id", "nodoc")
            cid = f"{base}_chunk{i}"
            while cid in seen:
                cid += "_x"
            seen.add(cid)
            ids.append(cid)
            ch.metadata["chunk_id"] = cid

        vectorstore = Chroma(
            collection_name="knee_orthopedics",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]
            vectorstore.add_documents(documents=batch, ids=batch_ids)
            log.info("  batch %d-%d enviado", i, i + len(batch))

        vectorstore.persist()
        log.info("Vectorstore persistido em %s", self.persist_dir)
        return vectorstore

    def run_full_pipeline(
        self,
        pdf_dir: Optional[str] = None,
        pubmed_max: int = 30,
        scholar_max: int = 10,
    ) -> Chroma:
        self.ingest_pubmed(max_results=pubmed_max)
        if pdf_dir:
            self.ingest_pdfs(pdf_dir)
        self.ingest_scholar(max_results=scholar_max)
        self.ingest_websites()
        return self.build_vectorstore()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    loader = KneeKnowledgeLoader(
        persist_dir=os.getenv("CHROMA_DIR", "./chroma_knee"),
        entrez_email=os.getenv("ENTREZ_EMAIL", "ortopediaraggi@gmail.com"),
    )

    vs = loader.run_full_pipeline(
        pdf_dir="./diretrizes_pdf/",
        pubmed_max=30,
        scholar_max=10,
    )

    # Teste rápido
    print("\n=== Teste retrieval ===")
    results = vs.similarity_search_with_score(
        "Qual o tratamento conservador para artrose de joelho grau 2?",
        k=3,
        filter={"scope": {"$in": ["conservative", "mixed"]}},
    )
    for doc, score in results:
        print(f"\n[score={score:.3f}] {doc.metadata.get('source')} | {doc.metadata.get('title', '')[:80]}")
        print(doc.page_content[:300], "...")
