import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import re

# ============================================================
# Utility: Sentence splitting (NLTK-free, SSL-safe)
# ============================================================

def sentence_split(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

# ---------------------------
# Environment Setup
# ---------------------------
load_dotenv()

DEBUG = True  # Toggle verbose debugging

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()


TOP_10_CATEGORIES = {
    "Computer Science": "cs",
    "Mathematics": "math",
    "Condensed Matter Physics": "cond-mat",
    "Astrophysics": "astro-ph",
    "Physics": "physics",
    "High Energy Physics – Phenomenology": "hep-ph",
    "Quantum Physics": "quant-ph",
    "High Energy Physics – Theory": "hep-th",
    "General Relativity & Quantum Cosmology": "gr-qc",
    "Electrical Engineering & Systems Science": "eess"
}

DOMAIN_CODE_TO_NAME = {
    v: k for k, v in TOP_10_CATEGORIES.items()
}


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

# ---------------------------
# NLP Models (Query-time only)
# ---------------------------
bi_encoder = SentenceTransformer("all-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

ARXIV_JSON_PATH = "arxiv-metadata-oai-snapshot.json"

# ============================================================
# 1. Stream arXiv candidates (NO full load)
# ============================================================

def stream_arxiv_candidates(
    domain_code: str,
    query_terms: List[str],
    max_docs: int = 200
) -> List[Dict]:

    candidates = []

    with open(ARXIV_JSON_PATH, "r") as f:
        for line in f:
            paper = json.loads(line)

            if not paper.get("categories", "").startswith(domain_code):
                continue

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text = f"{title} {abstract}".lower()

            if any(term in text for term in query_terms):
                candidates.append({
                    "id": paper.get("id"),
                    "title": title,
                    "abstract": abstract
                })

            if len(candidates) >= max_docs:
                break

    if DEBUG:
        print(f"[DEBUG] Candidates retrieved: {len(candidates)}")
        for c in candidates[:5]:
            print({"id": c["id"], "title": c["title"][:120]})

    return candidates

# ============================================================
# 2. Bi-Encoder Semantic Re-ranking (Fast)
# ============================================================

def rerank_with_bert(query: str, candidates: List[Dict], top_k: int = 10):

    if not candidates:
        return []

    query_emb = bi_encoder.encode(query, convert_to_tensor=True)
    doc_embs = bi_encoder.encode(
        [c["abstract"] for c in candidates],
        convert_to_tensor=True
    )

    scores = util.cos_sim(query_emb, doc_embs)[0]

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    if DEBUG:
        print("[DEBUG] Top bi-encoder results:")
        for doc, score in ranked[:5]:
            print({"score": float(score), "title": doc["title"][:120]})

    return ranked[:top_k]

# ============================================================
# 3. Cross-Encoder Deep Re-ranking (High precision)
# ============================================================

def cross_encode_rerank(query: str, ranked_docs, top_k: int = 5):

    if not ranked_docs:
        return []

    pairs = [(query, d[0]["abstract"]) for d in ranked_docs]
    scores = cross_encoder.predict(pairs)

    final = sorted(
        zip(ranked_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    if DEBUG:
        print("[DEBUG] Cross-encoder final ranking:")
        for ((doc, bert_score), ce_score) in final:
            print({
                "cross_score": float(ce_score),
                "bert_score": float(bert_score),
                "title": doc["title"][:120]
            })

    return final[:top_k]

# ============================================================
# 4. Sentence-Level Information Extraction
# ============================================================

def extract_relevant_sentences(query: str, abstract: str, top_n: int = 3):

    if not abstract:
        return []

    sentences = sentence_split(abstract)
    if not sentences:
        return []

    sent_embs = bi_encoder.encode(sentences, convert_to_tensor=True)
    query_emb = bi_encoder.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, sent_embs)[0]
    top_idx = scores.argsort(descending=True)[:top_n]

    selected = [sentences[i] for i in top_idx]

    if DEBUG:
        print("[DEBUG] Extracted sentences:")
        for s in selected:
            print(f"- {s}")

    return selected

# ============================================================
# 5. Build High-Signal LLM Context
# ============================================================

def build_llm_context(query: str, top_papers) -> str:

    context_blocks = []

    for (paper, _bert_score), _ce_score in top_papers:
        key_sents = extract_relevant_sentences(query, paper["abstract"])

        block = f"""
            Paper: {paper['title']}
            Key Points:
            - {' '.join(key_sents)}
            Reference: https://arxiv.org/abs/{paper['id']}
        """
        context_blocks.append(block.strip())

    return "\n\n".join(context_blocks)

# ============================================================
# 6. End-to-End Query → Context → LLM
# ============================================================

def answer_query(query: str, domain_code: str) -> str:

    query_terms = query.lower().split()

    candidates = stream_arxiv_candidates(domain_code, query_terms)
    bert_ranked = rerank_with_bert(query, candidates)
    final_ranked = cross_encode_rerank(query, bert_ranked)

    if not final_ranked:
        return "I don't know."

    context = build_llm_context(query, final_ranked)

    if DEBUG:
        print("\n### FINAL CONTEXT SENT TO LLM ###\n")
        print(context)

    prompt = PromptTemplate(
        input_variables=["context", "question", "domain"],
        template="""
            You are an expert scientific research assistant specializing in **{domain}**,
            with deep expertise in physics, computational mechanics, numerical methods,
            and domain-specific theoretical frameworks.

            INSTRUCTIONS:
            1. Use ONLY the information provided in the Context below.
            2. Every major claim, explanation, or conclusion MUST be followed by a citation in the format: [arXiv:XXXX.XXXXX].
            3. Citations MUST correspond to references explicitly present in the Context. 
            4. Do NOT invent equations, assumptions, results, DOIs or references.
            5. Treat the domain ("{domain}") as authoritative for terminology,
            assumptions, and methodological standards.
            6. If the context is insufficient to fully answer the question, explicitly say:
            "The provided context does not contain sufficient information to answer this question."

            RESPONSE STRUCTURE:
            - Begin with a **high-level summary (with citations)** (2–3 sentences) tailored to the domain.
            - Follow with a **technical explanation (each paragraph must include citations)**, including:
            • governing physical or theoretical principles relevant to {domain}
            • numerical, algorithmic, or analytical methods used
            • assumptions, constraints, and approximations
            • why the method works and what domain-specific problem it solves
            - Use precise scientific language while explaining complex concepts clearly.

            Context:
            {context}

            Question:
            {question}
        """
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": query,
        "domain": DOMAIN_CODE_TO_NAME.get(domain_code)
    })

# ============================================================
# Chain Wrapper (Streamlit-friendly)
# ============================================================

def get_query_chain(domain_code: str):
    return RunnableLambda(lambda query: answer_query(query, domain_code))


# chain = get_query_chain("cond-mat")
# response = chain.invoke("Explain how the numerical coupling between shock and ramp compression is formulated at a material interface. Derive the jump conditions and continuity constraints for stress, particle velocity, and internal energy at a planar interface between two dissimilar materials, and show how the algorithm ensures conservation of mass, momentum, and energy across composite paths. What are the implications of interface impedance mismatch on numerical convergence and accuracy?")

# print(response)