"""
RAG Pipeline — lightweight version for free tier deployment
Uses numpy for embeddings via a simple TF-IDF approach + Groq for generation
No torch/sentence-transformers needed = fits in 512MB RAM
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import fitz  # PyMuPDF
import numpy as np
import faiss
import math
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Config ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
GROQ_MODEL = "llama-3.1-8b-instant"


class TFIDFEmbedder:
    """Lightweight TF-IDF based embedder — no torch needed, fits in 512MB RAM"""

    def __init__(self, dim=256):
        self.dim = dim
        self.vocab = {}
        self.fitted = False

    def _tokenize(self, text):
        return text.lower().split()

    def _build_vocab(self, texts):
        word_counts = {}
        for text in texts:
            for word in set(self._tokenize(text)):
                word_counts[word] = word_counts.get(word, 0) + 1
        # Keep top-dim words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.vocab = {word: i % self.dim for word, (i, _) in enumerate(sorted_words[:self.dim * 10])}
        self.fitted = True

    def _embed_one(self, text):
        tokens = self._tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        total = len(tokens)
        if total == 0:
            return vec
        for word in tokens:
            if word in self.vocab:
                idx = self.vocab[word]
                # TF component
                tf = tokens.count(word) / total
                vec[idx] += tf
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts, show_progress_bar=False):
        if not self.fitted:
            self._build_vocab(texts)
        return np.array([self._embed_one(t) for t in texts], dtype=np.float32)


class RAGPipeline:
    def __init__(self):
        self.embedder = TFIDFEmbedder(dim=512)
        self.index = None
        self.chunks = []
        self.num_chunks = 0
        self.session_id = str(uuid.uuid4())

    def _parse_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def _chunk_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    def _build_index(self, chunks: list[str]):
        embeddings = self.embedder.encode(chunks)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def ingest(self, pdf_path: str, filename: str = "") -> str:
        text = self._parse_pdf(pdf_path)
        if not text.strip():
            raise ValueError("Could not extract text from PDF. It may be a scanned image.")
        self.chunks = self._chunk_text(text)
        self.num_chunks = len(self.chunks)
        self._build_index(self.chunks)
        return self.session_id

    def _retrieve(self, question: str) -> list[str]:
        q_emb = self.embedder.encode([question])
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, TOP_K)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _generate(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(context_chunks)

        system_prompt = """You are a precise document analysis assistant.
Answer questions STRICTLY based on the provided document context.
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
            temperature=0.1,
        )
        return response.choices[0].message.content

    def query(self, question: str) -> tuple[str, list[str]]:
        if self.index is None:
            raise ValueError("No document ingested yet.")
        relevant_chunks = self._retrieve(question)
        answer = self._generate(question, relevant_chunks)
        sources = [chunk[:200] + "..." for chunk in relevant_chunks]
        return answer, sources