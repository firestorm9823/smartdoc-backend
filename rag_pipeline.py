"""
RAG Pipeline — the brain of SmartDoc Analyzer

Flow:
  PDF → parse → chunk → embed → FAISS index
  Question → embed → similarity search → top-k chunks → LLM → answer
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq  # Free LLM via Groq (Llama 3)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Config ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast, free, good quality
CHUNK_SIZE = 500                         # Characters per chunk
CHUNK_OVERLAP = 100                      # Overlap to preserve context
TOP_K = 4                                # How many chunks to retrieve
GROQ_MODEL = "llama-3.1-8b-instant"          # Free on Groq


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks: list[str] = []
        self.num_chunks = 0
        self.session_id = str(uuid.uuid4())

    # ── Step 1: Parse PDF ──────────────────────────────────────────
    def _parse_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    # ── Step 2: Chunk text ─────────────────────────────────────────
    def _chunk_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    # ── Step 3: Embed chunks → FAISS ──────────────────────────────
    def _build_index(self, chunks: list[str]):
        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        self.index.add(embeddings)

    # ── Ingest (Steps 1-3 combined) ────────────────────────────────
    def ingest(self, pdf_path: str, filename: str = "") -> str:
        text = self._parse_pdf(pdf_path)
        if not text.strip():
            raise ValueError("Could not extract text from PDF. It may be a scanned image.")

        self.chunks = self._chunk_text(text)
        self.num_chunks = len(self.chunks)
        self._build_index(self.chunks)
        return self.session_id

    # ── Step 4: Retrieve relevant chunks ──────────────────────────
    def _retrieve(self, question: str) -> list[str]:
        q_emb = self.embedder.encode([question])
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, TOP_K)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    # ── Step 5: Generate answer with LLM ──────────────────────────
    def _generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)

        system_prompt = """You are a precise document analysis assistant.
Your job is to answer questions STRICTLY based on the provided document context.
Rules:
- Only use information from the context below.
- If the answer is not in the context, say "I couldn't find this information in the document."
- Be concise but complete.
- Do not hallucinate or add outside knowledge."""

        user_prompt = f"""Context from the document:
{context}

Question: {question}

Answer:"""

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.1,  # Low temp = more factual
        )
        return response.choices[0].message.content

    # ── Query (Steps 4-5 combined) ─────────────────────────────────
    def query(self, question: str) -> tuple[str, list[str]]:
        if self.index is None:
            raise ValueError("No document ingested yet.")

        relevant_chunks = self._retrieve(question)
        answer = self._generate(question, relevant_chunks)

        # Return answer + short source snippets for UI display
        sources = [chunk[:200] + "..." for chunk in relevant_chunks]
        return answer, sources