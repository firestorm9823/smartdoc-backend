from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from rag_pipeline import RAGPipeline
import os
import tempfile

app = FastAPI(title="SmartDoc Analyzer API")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, set to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: session_id -> RAGPipeline
sessions: dict[str, RAGPipeline] = {}


class QueryRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, chunk it, embed it, and return a session_id."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = RAGPipeline()
        session_id = pipeline.ingest(tmp_path, file.filename)
        sessions[session_id] = pipeline
        return {
            "session_id": session_id,
            "filename": file.filename,
            "chunks": pipeline.num_chunks,
            "message": "Document processed successfully!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    """Ask a question about the uploaded document."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    pipeline = sessions[req.session_id]
    answer, sources = pipeline.query(req.question)
    return QueryResponse(answer=answer, sources=sources, session_id=req.session_id)


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session deleted."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)