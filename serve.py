# serve.py — minimal API for your existing Cassandra vector store (with inline keys)
from langchain.prompts import PromptTemplate
import os, pathlib, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain_community.vectorstores import Cassandra
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# === Config  ===
KEYSPACE      = (os.getenv("ASTRA_DB_KEYSPACE") or "default_keyspace").strip()
TABLE_NAME    = (os.getenv("CASSANDRA_TABLE") or "ml_supervised").strip()
BUNDLE_PATH   = (os.getenv("ASTRA_SECURE_BUNDLE") or "secure-connect-ml-db.zip").strip()
OPENAI_KEY    = (os.getenv("OPENAI_API_KEY") or "").strip()
ASTRA_TOKEN   = (os.getenv("ASTRA_DB_APPLICATION_TOKEN") or "").strip()
ASTRA_DB_ID   = (os.getenv("ASTRA_DB_ID") or "").strip()
MODEL_NAME    = (os.getenv("OPENAI_MODEL") or "gpt-4.1-nano").strip()
ALLOWED_ORIGS = (os.getenv("ALLOWED_ORIGINS") or "*").strip()

ALLOWED_ORIGS = "*"

assert OPENAI_KEY.startswith("sk-"), "OPENAI_API_KEY missing/invalid"
assert ASTRA_TOKEN.startswith("AstraCS:"), "ASTRA token missing/invalid"
assert len(ASTRA_DB_ID) > 10, "ASTRA DB ID missing/invalid"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading, time

app = FastAPI(title="RAG ML Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if (os.getenv("ALLOWED_ORIGINS") or "*") == "*" else [
        o.strip() for o in (os.getenv("ALLOWED_ORIGINS") or "").split(",")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LAZY STATE ----------
_state = {
    "ready": False,
    "error": None,
    "vs": None,
    "idx": None,
    "llm": None,
}
_lock = threading.Lock()

def _init_once():
    """Idempotent init that connects to Astra and builds the vector store/LLM."""
    if _state["ready"] or _state["error"]:
        return
    with _lock:
        if _state["ready"] or _state["error"]:
            return
        try:
            # Ensure bundle exists
            import pathlib, requests
            BUNDLE_PATH = (os.getenv("ASTRA_SECURE_BUNDLE") or "secure-connect-ml-db.zip").strip()
            ASTRA_DB_ID = (os.getenv("ASTRA_DB_ID") or "").strip()
            ASTRA_TOKEN = (os.getenv("ASTRA_DB_APPLICATION_TOKEN") or "").strip()
            if not pathlib.Path(BUNDLE_PATH).exists():
                url = f"https://api.astra.datastax.com/v2/databases/{ASTRA_DB_ID}/secureBundleURL"
                r = requests.post(url, headers={
                    "Authorization": f"Bearer {ASTRA_TOKEN}",
                    "Content-Type": "application/json"
                }, json={})
                r.raise_for_status()
                dl = r.json().get("downloadURL")
                if not dl:
                    raise RuntimeError("Could not get secure bundle URL from Astra.")
                with requests.get(dl, stream=True) as resp:
                    resp.raise_for_status()
                    with open(BUNDLE_PATH, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)

            # Connect to Astra
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            KEYSPACE = (os.getenv("ASTRA_DB_KEYSPACE") or "default_keyspace").strip()
            cloud_config = {"secure_connect_bundle": BUNDLE_PATH}
            auth_provider = PlainTextAuthProvider("token", ASTRA_TOKEN)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect(KEYSPACE)

            # Build vector store + LLM
            from langchain_community.vectorstores import Cassandra
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from langchain.indexes.vectorstore import VectorStoreIndexWrapper

            OPENAI_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
            TABLE_NAME = (os.getenv("CASSANDRA_TABLE") or "ml_supervised").strip()
            MODEL_NAME = (os.getenv("OPENAI_MODEL") or "gpt-4.1-nano").strip()

            emb = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
            vs = Cassandra(embedding=emb, table_name=TABLE_NAME, session=session, keyspace=KEYSPACE)
            idx = VectorStoreIndexWrapper(vectorstore=vs)
            llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name=MODEL_NAME, temperature=0.2)

            _state["vs"] = vs
            _state["idx"] = idx
            _state["llm"] = llm
            _state["ready"] = True
        except Exception as e:
            _state["error"] = str(e)

# Background warmup after the server starts
def _background_warm():
    # small delay so Uvicorn is fully up before we do work
    time.sleep(1.0)
    try:
        _init_once()
    except Exception:
        pass

@app.on_event("startup")
def _kickoff_warm():
    t = threading.Thread(target=_background_warm, daemon=True)
    t.start()

# ---------- ROUTES ----------
class AskRequest(BaseModel):
    question: str
    top_k: int = 4

class AskResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {
        "status": "ok" if _state["ready"] and not _state["error"] else "warming" if not _state["error"] else "error",
        "error": _state["error"],
    }

# optional nicer root so / doesn’t 404
@app.get("/")
def index():
    return {
        "name": "RAG ML Assistant API",
        "endpoints": {"health": "/health", "ask": "/ask (POST)"},
        "status": "ok" if _state["ready"] and not _state["error"] else "warming",
    }

from langchain.prompts import PromptTemplate
CONTEXT_PROMPT = PromptTemplate.from_template("""
You are an ML assistant. Answer ONLY using the context below. If the answer is not present, say:
"I don't know based on the indexed PDFs."

Question: {question}

Context:
{context}

Give a concise answer (3–6 sentences).
""")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not _state["ready"]:
        # try to init on-demand (covers cold start)
        _init_once()
        if not _state["ready"]:
            return AskResponse(
                answer="Backend is waking up on Render… please try again in a few seconds.",
                sources=[],
            )
    if _state["error"]:
        return AskResponse(
            answer=f"Backend startup error: {_state['error']}",
            sources=[],
        )

    q = (req.question or "").strip()
    if not q:
        return AskResponse(answer="Please provide a question.", sources=[])

    vs = _state["vs"]; idx = _state["idx"]; llm = _state["llm"]
    # Stronger retrieval if available
    if hasattr(vs, "max_marginal_relevance_search_with_score"):
        docs_scores = vs.max_marginal_relevance_search_with_score(q, k=max(6, req.top_k), fetch_k=24)
    else:
        docs_scores = vs.similarity_search_with_score(q, k=max(8, req.top_k))

    context_chunks, sources = [], []
    for doc, score in docs_scores:
        txt = (doc.page_content or "").strip().replace("\n", " ")
        if txt:
            context_chunks.append(txt[:1200])
        meta = doc.metadata or {}
        sources.append({
            "score": float(score),
            "snippet": (doc.page_content or "")[:220],
            "source_pdf": meta.get("source_pdf"),
            "page": meta.get("page"),
        })
    prompt = CONTEXT_PROMPT.format(question=q, context="\n\n".join(context_chunks))
    ans = _state["llm"].invoke(prompt).content.strip()
    return AskResponse(answer=ans, sources=sources)
